#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_full_sft.py - 将全量 T2+T3 VQA 数据转换为 swift SFT 格式

输出 swift 兼容的 JSONL，每行一个 messages 格式的对话
"""
import json
import os
import random
from pathlib import Path
from PIL import Image

def load_vqa(vqa_dir, task):
    task_files = {"T2": "T2_binary.jsonl", "T3": "T3_multiclass.jsonl"}
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

def vqa_to_sft(rec, data_root):
    img_path = Path(data_root) / rec["image_path"]
    if not img_path.exists():
        return None
    
    sft_rec = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": rec["question"]},
            ]},
            {"role": "assistant", "content": rec["answer"]},
        ]
    }
    return sft_rec

def main():
    data_root = Path("data/raw")
    vqa_dir = Path("work/vqa")
    out_dir = Path("work/sft_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_train = []
    all_val = []
    
    for task in ["T2", "T3"]:
        recs = load_vqa(vqa_dir, task)
        train_recs = [r for r in recs if r.get("split") == "train"]
        val_recs = [r for r in recs if r.get("split") == "val"]
        
        n_train_ok = 0
        n_train_skip = 0
        for rec in train_recs:
            sft = vqa_to_sft(rec, data_root)
            if sft:
                all_train.append(sft)
                n_train_ok += 1
            else:
                n_train_skip += 1
        
        n_val_ok = 0
        for rec in val_recs:
            sft = vqa_to_sft(rec, data_root)
            if sft:
                all_val.append(sft)
                n_val_ok += 1
        
        print(f"{task}: train={n_train_ok} (skip {n_train_skip}), val={n_val_ok}")
    
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in all_train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for rec in all_val:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\nTotal: train={len(all_train)}, val={len(all_val)}")
    print(f"Output: {train_path}, {val_path}")

if __name__ == "__main__":
    main()
