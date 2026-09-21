#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_sft.py - 评测 SFT 后的 InternVL3.5-8B 在 test 集上的表现

使用 swift infer 加载 LoRA adapter，对 T2/T3 test 集做推理
然后计算 BalAcc / Macro-F1 / 漏检率 / 幻觉率
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import torch
from PIL import Image


def load_vqa_test(vqa_dir, task):
    """加载 test 集 VQA 数据"""
    task_files = {"T2": "T2_binary.jsonl", "T3": "T3_multiclass.jsonl"}
    path = Path(vqa_dir) / task_files.get(task, f"{task}.jsonl")
    if not path.exists():
        return []
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("split") == "test":
                    recs.append(rec)
    return recs


def load_model_with_lora(model_path, adapter_path):
    """加载模型 + LoRA adapter"""
    from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
    from peft import PeftModel
    
    print(f"  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"  Loading image processor ...")
    image_processor = CLIPImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"  Loading base model ...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print(f"  Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    load_time = time.time() - t0
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Loaded in {load_time:.1f}s, GPU: {mem:.1f} GB")
    
    return model, tokenizer, image_processor


def internvl_infer(model, tokenizer, image_processor, image_path, question, device):
    """InternVL3.5 推理"""
    from qwen_vl_utils import process_vision_info
    
    # 加载图片
    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if w * h > 589824:  # 768x768
            scale = (589824 / (w * h)) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    except Exception:
        return "[IMAGE_ERROR]"
    
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
    
    # 构造对话
    img_token = "<image>"
    text = f"<|im_start|>user\n{img_token}\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # 获取 image token id
    try:
        image_token_id = tokenizer.convert_tokens_to_ids(img_token)
    except:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    # 替换第一个 token 为 image token
    if image_token_id and image_token_id != tokenizer.unk_token_id:
        inputs["input_ids"][0, 0] = image_token_id
    
    inputs["pixel_values"] = pixel_values
    
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        # 生成
        try:
            output_ids = model.generate(
                **batch,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            # 只取新生成的 token
            input_len = batch["input_ids"].shape[1]
            response = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
            return response
        except Exception as e:
            return f"[GEN_ERROR] {str(e)[:100]}"


def parse_response(response, task):
    """解析模型响应"""
    response = response.strip()
    for ch in response:
        if ch in "ABCDE":
            return ch, True
    if "正常" in response and "缺陷" not in response:
        return "A", True
    if "缺陷" in response or "锈" in response:
        return "B", True
    return None, False


def compute_metrics(predictions, ground_truths, task):
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    
    valid_preds = []
    valid_gts = []
    n_format_fail = 0
    
    for pred, gt in zip(predictions, ground_truths):
        if pred is None:
            n_format_fail += 1
            continue
        valid_preds.append(pred)
        valid_gts.append(gt)
    
    n_total = len(predictions)
    n_valid = len(valid_preds)
    fmt_fail_rate = n_format_fail / n_total if n_total > 0 else 0
    
    if n_valid == 0:
        return {"n_total": n_total, "n_valid": 0, "format_failure_rate": 1.0,
                "balanced_accuracy": 0.0, "macro_f1": 0.0}
    
    bal_acc = balanced_accuracy_score(valid_gts, valid_preds)
    macro_f1 = f1_score(valid_gts, valid_preds, average="macro", zero_division=0)
    
    labels = sorted(set(valid_gts + valid_preds))
    cm = confusion_matrix(valid_gts, valid_preds, labels=labels)
    
    miss_rate = 0.0
    hallucination_rate = 0.0
    if task == "T2":
        defect_idx = [i for i, gt in enumerate(valid_gts) if gt == "B"]
        if defect_idx:
            miss_rate = sum(1 for i in defect_idx if valid_preds[i] == "A") / len(defect_idx)
        good_idx = [i for i, gt in enumerate(valid_gts) if gt == "A"]
        if good_idx:
            hallucination_rate = sum(1 for i in good_idx if valid_preds[i] == "B") / len(good_idx)
    
    return {
        "n_total": n_total, "n_valid": n_valid,
        "format_failure_rate": round(fmt_fail_rate, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "miss_rate": round(miss_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "pred_distribution": dict(Counter(valid_preds)),
        "gt_distribution": dict(Counter(valid_gts)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/models/models/OpenGVLab--InternVL3_5-8B/snapshots/master")
    ap.add_argument("--adapter-path", default="runs/sft_internvl3_8b_main/v0-20260824-155006/checkpoint-4052")
    ap.add_argument("--vqa-dir", default="work/vqa")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--tasks", nargs="+", default=["T2", "T3"])
    ap.add_argument("--n-samples", type=int, default=0, help="0=all, N=sample N")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    
    print("=" * 60)
    print("T52: Evaluate SFT InternVL3.5-8B on Test Set")
    print(f"  Adapter: {args.adapter_path}")
    print(f"  Tasks: {args.tasks}")
    print("=" * 60)
    
    device = torch.device("cuda:0")
    model, tokenizer, image_processor = load_model_with_lora(args.model_path, args.adapter_path)
    
    all_results = {}
    
    for task in args.tasks:
        test_data = load_vqa_test(args.vqa_dir, task)
        if args.n_samples > 0 and args.n_samples < len(test_data):
            import random
            random.seed(42)
            test_data = random.sample(test_data, args.n_samples)
        
        print(f"\n--- {task}: {len(test_data)} test samples ---")
        
        predictions = []
        ground_truths = []
        raw_responses = []
        
        for i, rec in enumerate(test_data):
            img_path = Path(args.data_root) / rec["image_path"]
            response = internvl_infer(model, tokenizer, image_processor, str(img_path), rec["question"], device)
            pred, fmt_ok = parse_response(response, task)
            
            predictions.append(pred)
            ground_truths.append(rec["answer"])
            raw_responses.append(response)
            
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(test_data)} ...")
        
        metrics = compute_metrics(predictions, ground_truths, task)
        metrics["model"] = "InternVL3.5-8B-SFT"
        metrics["task"] = task
        metrics["adapter"] = args.adapter_path
        
        print(f"\n  Results:")
        print(f"    N: {metrics['n_total']}, Valid: {metrics['n_valid']}")
        print(f"    Format failure: {metrics['format_failure_rate']:.2%}")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    Macro-F1: {metrics['macro_f1']:.4f}")
        if task == "T2":
            print(f"    Miss rate: {metrics.get('miss_rate', 0):.4f}")
            print(f"    Hallucination rate: {metrics.get('hallucination_rate', 0):.4f}")
        
        # Save
        run_id = f"InternVL3.5-8B-SFT_{task}_P3_test"
        if args.n_samples > 0:
            run_id += f"_sample{args.n_samples}"
        run_dir = Path(args.out) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for i, rec in enumerate(test_data):
                f.write(json.dumps({
                    "crop_id": rec.get("crop_id", ""),
                    "gt": rec["answer"],
                    "pred": predictions[i],
                    "raw_response": raw_responses[i],
                    "asset": rec.get("asset", ""),
                    "label": rec.get("label", ""),
                }, ensure_ascii=False) + "\n")
        
        all_results[run_id] = metrics
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary: SFT vs Zero-shot")
    print("=" * 60)
    print(f"{'Model':<35} {'Task':<5} {'BalAcc':>8} {'F1':>8} {'Miss':>8} {'Hall':>8}")
    print("-" * 75)
    for run_id, m in sorted(all_results.items()):
        miss = m.get("miss_rate", 0)
        hall = m.get("hallucination_rate", 0)
        print(f"{'InternVL3.5-8B-SFT':<35} {m['task']:<5} {m['balanced_accuracy']:>8.4f} {m['macro_f1']:>8.4f} {miss:>8.4f} {hall:>8.4f}")
    
    # Compare with zero-shot
    print(f"\n--- Zero-shot comparison ---")
    print(f"{'Qwen3.5-122B (zero-shot)':<35} T2   0.7103   0.4940   0.4504   0.0414")
    print(f"{'Qwen3.8-27B (zero-shot)':<35} T2   0.6650   0.4461   0.6449   0.0096")
    print(f"{'Qwen3.5-122B (zero-shot)':<35} T3   0.6555   0.3195   -        -")
    print(f"{'Qwen3.8-27B (zero-shot)':<35} T3   0.6415   0.5897   -        -")
    
    with open(Path(args.out) / "sft_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.out}/sft_eval_summary.json")


if __name__ == "__main__":
    main()
