#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sft_data.py - 将 VQA 数据转换为 Qwen3-VL SFT 训练格式

从 T2/T3 VQA 数据中抽取少量样本，转换为 Qwen3-VL chat 格式的 JSONL
用于 SFT 冒烟测试

用法:
  python scripts/prepare_sft_data.py --vqa-dir work/vqa --out work/sft \
      --n-train 100 --n-val 20 --tasks T2 T3
"""

import argparse
import io
import json
import os
import random
from pathlib import Path

from PIL import Image


def load_vqa(vqa_dir, task):
    task_files = {
        "T1": "T1_grounding.jsonl",
        "T2": "T2_binary.jsonl",
        "T3": "T3_multiclass.jsonl",
    }
    path = Path(vqa_dir) / task_files.get(task, f"{task}.jsonl")
    if not path.exists():
        return []
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def vqa_to_sft(rec, data_root, answer_map=None):
    """将 VQA 记录转换为 Qwen3-VL SFT 格式"""
    img_path = Path(data_root) / rec["image_path"]
    if not img_path.exists():
        return None
    
    # 构造 answer 文本
    if answer_map and rec["answer"] in answer_map:
        answer_text = answer_map[rec["answer"]]
    else:
        answer_text = rec["answer"]
    
    # Qwen3-VL chat 格式
    sft_rec = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": rec["question"]},
                ]
            },
            {
                "role": "assistant",
                "content": answer_text,
            }
        ],
        "metadata": {
            "task": rec.get("task", ""),
            "split": rec.get("split", ""),
            "crop_id": rec.get("crop_id", ""),
            "asset": rec.get("asset", ""),
            "label": rec.get("label", ""),
            "gt_answer": rec["answer"],
        }
    }
    return sft_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-dir", default="work/vqa")
    ap.add_argument("--out", default="work/sft")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--tasks", nargs="+", default=["T2", "T3"])
    args = ap.parse_args()
    
    print("=" * 60)
    print("准备 SFT 训练数据")
    print(f"  任务: {args.tasks}")
    print(f"  训练样本: {args.n_train}")
    print(f"  验证样本: {args.n_val}")
    print("=" * 60)
    
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # T2 answer 映射
    t2_answer_map = {"A": "A", "B": "B", "C": "C"}
    
    all_train = []
    all_val = []
    
    for task in args.tasks:
        recs = load_vqa(args.vqa_dir, task)
        train_recs = [r for r in recs if r.get("split") == "train"]
        val_recs = [r for r in recs if r.get("split") == "val"]
        
        random.seed(42)
        random.shuffle(train_recs)
        random.shuffle(val_recs)
        
        n_train_per_task = args.n_train // len(args.tasks)
        n_val_per_task = args.n_val // len(args.tasks)
        
        for rec in train_recs[:n_train_per_task]:
            sft = vqa_to_sft(rec, data_root, t2_answer_map if task == "T2" else None)
            if sft:
                all_train.append(sft)
        
        for rec in val_recs[:n_val_per_task]:
            sft = vqa_to_sft(rec, data_root, t2_answer_map if task == "T2" else None)
            if sft:
                all_val.append(sft)
        
        print(f"  {task}: train={n_train_per_task}, val={n_val_per_task}")
    
    # 写出
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in all_train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for rec in all_val:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\n产出:")
    print(f"  {train_path}: {len(all_train)} 条")
    print(f"  {val_path}: {len(all_val)} 条")
    
    # 打印示例
    if all_train:
        print(f"\n示例 (train[0]):")
        print(json.dumps(all_train[0], ensure_ascii=False, indent=2)[:500])
    
    # 写统计
    stats = {
        "n_train": len(all_train),
        "n_val": len(all_val),
        "tasks": args.tasks,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSFT 数据准备完成。")


if __name__ == "__main__":
    main()
