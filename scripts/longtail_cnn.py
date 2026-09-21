#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
longtail_cnn.py - C3 长尾交叉点实验: CNN 部分

每类样本量 n in {4, 8, 16, 32, 64, 128, all} x 3 seed
路线: ResNet-50, Swin-T
测试集: 固定不变 (full test)

核心产出: 交叉点曲线数据 (log(n) vs BalAcc)
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score, f1_score


LABEL_CANON = {
    "corrosao": "rust", "corrosion": "rust", "rust": "rust",
    "normal": "good", "good": "good",
    "missing-cap": "missing-cap", "missingcap": "missing-cap",
    "bird-nest": "nest", "nest": "nest",
    "torned-up": "torned-up", "peeling-paint": "peeling-paint",
}

def canon(label):
    return LABEL_CANON.get(label.strip().lower(), label.strip().lower())


class InsPLADDataset(Dataset):
    def __init__(self, records, data_root, img_size=224, is_train=True, label2idx=None):
        self.records = records
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.label2idx = label2idx
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
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


def sample_by_n(train_records, n_per_class, seed, label2idx):
    """每类采样 n 个训练样本"""
    random.seed(seed)
    by_class = defaultdict(list)
    for r in train_records:
        if r["source_subset"] in ("supervised", "unsupervised"):
            label = canon(r["label"])
            by_class[label].append(r)
    
    sampled = []
    for label, recs in by_class.items():
        if n_per_class == 0:  # all
            sampled.extend(recs)
        else:
            random.shuffle(recs)
            sampled.extend(recs[:n_per_class])
    
    random.shuffle(sampled)
    return sampled


def get_model(name, num_classes):
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
    raise ValueError(name)


def train_and_eval(model_name, train_records, test_records, data_root, label2idx, epochs, batch_size, lr, seed):
    n_classes = len(label2idx)
    model, img_size = get_model(model_name, n_classes)
    
    train_ds = InsPLADDataset(train_records, data_root, img_size, is_train=True, label2idx=label2idx)
    test_ds = InsPLADDataset(test_records, data_root, img_size, is_train=False, label2idx=label2idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda:0")
    model = model.to(device)
    
    # class weights
    label_counts = Counter()
    for r in train_records:
        label_counts[canon(r["label"])] += 1
    class_weights = torch.ones(n_classes).to(device)
    for label, idx in label2idx.items():
        count = label_counts.get(label, 1)
        class_weights[idx] = 1.0 / max(count, 1)
    class_weights = class_weights / class_weights.mean()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # test
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
    
    bal_acc = balanced_accuracy_score(all_gts, all_preds)
    macro_f1 = f1_score(all_gts, all_preds, average="macro", zero_division=0)
    
    # per-class accuracy
    per_class = {}
    for label, idx in label2idx.items():
        cls_idx = [i for i, g in enumerate(all_gts) if g == idx]
        if cls_idx:
            cls_acc = sum(1 for i in cls_idx if all_preds[i] == idx) / len(cls_idx)
        else:
            cls_acc = 0.0
        per_class[label] = round(cls_acc, 4)
    
    del model
    torch.cuda.empty_cache()
    
    return {"bal_acc": round(bal_acc, 4), "macro_f1": round(macro_f1, 4), "per_class": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="work/splits")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--models", nargs="+", default=["resnet50", "swin_tiny"])
    ap.add_argument("--n-values", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128, 0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/longtail_cnn")
    args = ap.parse_args()
    
    print("=" * 60)
    print("C3 Long-tail Crossover: CNN")
    print(f"  Models: {args.models}")
    print(f"  N values: {args.n_values} (0=all)")
    print(f"  Seeds: {args.seeds}")
    print(f"  Epochs: {args.epochs}")
    print("=" * 60)
    
    splits = load_splits(args.splits)
    
    # build label2idx from all splits
    all_labels = set()
    for split_name in ["train", "val", "test"]:
        for r in splits[split_name]:
            if r["source_subset"] in ("supervised", "unsupervised"):
                all_labels.add(canon(r["label"]))
    label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
    print(f"  Classes: {len(label2idx)} -> {label2idx}")
    
    # test set (fixed)
    test_records = [r for r in splits["test"] if r["source_subset"] in ("supervised", "unsupervised")]
    train_records = [r for r in splits["train"] if r["source_subset"] in ("supervised", "unsupervised")]
    print(f"  Train pool: {len(train_records)}, Test: {len(test_records)}")
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for model_name in args.models:
        for n in args.n_values:
            n_label = "all" if n == 0 else str(n)
            for seed in args.seeds:
                run_id = f"{model_name}_n{n_label}_seed{seed}"
                print(f"\n--- {run_id} ---")
                
                # check if already done
                result_file = out_dir / f"{run_id}.json"
                if result_file.exists():
                    print(f"  Already done, skipping")
                    with open(result_file) as f:
                        results.append(json.load(f))
                    continue
                
                # sample
                sampled_train = sample_by_n(train_records, n, seed, label2idx)
                print(f"  Train samples: {len(sampled_train)}")
                
                # train and eval
                t0 = time.time()
                metrics = train_and_eval(
                    model_name, sampled_train, test_records, args.data_root,
                    label2idx, args.epochs, args.batch_size, args.lr, seed
                )
                elapsed = time.time() - t0
                
                result = {
                    "run_id": run_id,
                    "model": model_name,
                    "n_per_class": n,
                    "seed": seed,
                    "n_train": len(sampled_train),
                    "elapsed_s": round(elapsed, 1),
                    **metrics,
                }
                results.append(result)
                
                with open(result_file, "w") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"  BalAcc: {metrics['bal_acc']:.4f}, F1: {metrics['macro_f1']:.4f}, Time: {elapsed:.0f}s")
                print(f"  Per-class: {metrics['per_class']}")
    
    # summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'N':>5} {'BalAcc':>8} {'F1':>8} {'seed':>5}")
    print("-" * 45)
    
    # aggregate by model + n
    agg = defaultdict(list)
    for r in results:
        key = (r["model"], r["n_per_class"])
        agg[key].append(r)
    
    for (model, n), runs in sorted(agg.items()):
        balaccs = [r["bal_acc"] for r in runs]
        f1s = [r["macro_f1"] for r in runs]
        n_label = "all" if n == 0 else str(n)
        mean_acc = sum(balaccs) / len(balaccs)
        std_acc = (sum((x - mean_acc) ** 2 for x in balaccs) / len(balaccs)) ** 0.5
        mean_f1 = sum(f1s) / len(f1s)
        print(f"{model:<15} {n_label:>5} {mean_acc:>8.4f} {mean_f1:>8.4f} {len(runs):>5} (std={std_acc:.4f})")
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
