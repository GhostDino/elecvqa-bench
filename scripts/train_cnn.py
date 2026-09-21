#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn.py - T30 CNN 基线训练 (使用 torchvision 预训练模型)

训练 ResNet-50 / Swin-T / ConvNeXt-T 在 InsPLAD 缺陷分类任务上
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


LABEL_CANON = {
    "corrosao": "rust", "corrosion": "rust", "rust": "rust",
    "normal": "good", "good": "good",
    "missing-cap": "missing-cap", "missingcap": "missing-cap",
    "bird-nest": "nest", "nest": "nest",
    "torned-up": "torned-up", "peeling-paint": "peeling-paint",
}

def canon(label):
    return LABEL_CANON.get(label.strip().lower(), label.strip().lower())


def get_model(name, num_classes):
    """获取 torchvision 预训练模型"""
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, 224
    elif name == "swin_tiny":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m, 224
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m, 224
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, 300
    else:
        raise ValueError(f"Unknown model: {name}")


class InsPLADDataset(Dataset):
    def __init__(self, records, data_root, img_size=224, is_train=True, label2idx=None):
        self.records = [r for r in records if r["source_subset"] in ("supervised", "unsupervised")]
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.is_train = is_train
        
        if label2idx is not None:
            self.label2idx = label2idx
        else:
            all_labels = set()
            for r in self.records:
                all_labels.add(canon(r["label"]))
            self.label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.data_root / rec["path"]
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)
        label = self.label2idx[canon(rec["label"])]
        return img, label


def load_splits(splits_dir):
    splits = {}
    for name in ["train", "val", "test"]:
        path = Path(splits_dir) / f"{name}.jsonl"
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        splits[name] = recs
    return splits


def train_model(model_name, splits, data_root, out_dir, epochs, batch_size, lr, seed):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    
    # 从所有 split 的标签中构建映射 (包括 test-only 的稀有类)
    all_labels = set()
    for split_name in ["train", "val", "test"]:
        for r in splits[split_name]:
            if r["source_subset"] in ("supervised", "unsupervised"):
                all_labels.add(canon(r["label"]))
    label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
    n_classes = len(label2idx)
    
    train_ds = InsPLADDataset(splits["train"], data_root, 224, is_train=True, label2idx=label2idx)
    val_ds = InsPLADDataset(splits["val"], data_root, 224, is_train=False, label2idx=label2idx)
    test_ds = InsPLADDataset(splits["test"], data_root, 224, is_train=False, label2idx=label2idx)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Classes: {n_classes} -> {label2idx}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda:0")
    model, img_size = get_model(model_name, n_classes)
    model = model.to(device)
    
    # 类别权重
    label_counts = Counter()
    for r in splits["train"]:
        if r["source_subset"] in ("supervised", "unsupervised"):
            label_counts[canon(r["label"])] += 1
    class_weights = torch.ones(n_classes).to(device)
    for label, idx in label2idx.items():
        count = label_counts.get(label, 1)
        class_weights[idx] = 1.0 / max(count, 1)
    class_weights = class_weights / class_weights.mean()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        train_time = time.time() - t0
        
        # 验证
        model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_gts.extend(labels.cpu().numpy())
        
        val_acc = balanced_accuracy_score(all_gts, all_preds)
        val_f1 = f1_score(all_gts, all_preds, average="macro", zero_division=0)
        
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val BalAcc: {val_acc:.4f} | F1: {val_f1:.4f} | Time: {train_time:.0f}s | Mem: {mem_gb:.1f}GB")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        torch.cuda.reset_peak_memory_stats()
    
    # 测试
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
    
    test_acc = balanced_accuracy_score(all_gts, all_preds)
    test_f1 = f1_score(all_gts, all_preds, average="macro", zero_division=0)
    
    labels_sorted = sorted(set(all_gts + all_preds))
    cm = confusion_matrix(all_gts, all_preds, labels=labels_sorted)
    label_names = [train_ds.idx2label[l] for l in labels_sorted]
    
    print(f"\n  Test BalAcc: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    # 保存
    model_out = Path(out_dir) / f"cnn_{model_name}"
    model_out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_out / "model.pt")
    
    metrics = {
        "model": model_name,
        "test_balanced_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "best_val_balanced_accuracy": round(best_val_acc, 4),
        "n_classes": n_classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "label2idx": label2idx,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": label_names,
    }
    with open(model_out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved: {model_out}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="work/splits")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--models", nargs="+", default=["resnet50", "swin_tiny", "convnext_tiny"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 60)
    print("T30 CNN Baseline Training (torchvision)")
    print(f"  Models: {args.models}")
    print(f"  Epochs: {args.epochs}")
    print("=" * 60)
    
    splits = load_splits(args.splits)
    
    all_metrics = []
    for model_name in args.models:
        try:
            metrics = train_model(
                model_name, splits, args.data_root, args.out,
                args.epochs, args.batch_size, args.lr, args.seed
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<25} {'BalAcc':>8} {'F1':>8}")
    print("-" * 45)
    for m in all_metrics:
        print(f"{m['model']:<25} {m['test_balanced_accuracy']:>8.4f} {m['test_macro_f1']:>8.4f}")
    
    with open(Path(args.out) / "cnn_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
