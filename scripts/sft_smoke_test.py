#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_smoke_test.py - Qwen3-VL-8B LoRA SFT 冒烟测试 (简化版)

验证:
  1. 模型可加载
  2. LoRA 可注入
  3. 图文数据可前向传播
  4. 反向传播可更新参数
  5. adapter 可保存
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",
                     default="/root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master")
    ap.add_argument("--train-data", default="work/sft/train.jsonl")
    ap.add_argument("--output-dir", default="runs/sft_smoke")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-pixels", type=int, default=589824)
    args = ap.parse_args()

    print("=" * 60)
    print("Qwen3-VL-8B LoRA SFT Smoke Test")
    print(f"  model: {args.model_path}")
    print(f"  samples: {args.n_samples}")
    print(f"  steps: {args.n_steps}")
    print(f"  lora_rank: {args.lora_rank}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    print("\n[1/6] Load training data ...")
    train_recs = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_recs.append(json.loads(line))
    print(f"  Loaded {len(train_recs)} records, using {min(args.n_samples, len(train_recs))}")

    # ------------------------------------------------------------------
    # 2. 加载模型
    # ------------------------------------------------------------------
    print("\n[2/6] Load model ...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model loaded. GPU mem: {mem_gb:.1f} GB")

    # ------------------------------------------------------------------
    # 3. 注入 LoRA
    # ------------------------------------------------------------------
    print("\n[3/6] Inject LoRA ...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. 准备数据
    # ------------------------------------------------------------------
    print("\n[4/6] Prepare batch data ...")
    from qwen_vl_utils import process_vision_info

    def prepare_sample(rec, processor, max_pixels):
        """处理单条样本，返回 input_ids, attention_mask, pixel_values, labels"""
        messages = rec["messages"]

        # 提取图片
        image_inputs, video_inputs = process_vision_info(messages)

        # 应用 chat template (add_generation_prompt=False, 因为有 assistant 回复)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        # 处理为模型输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 创建 labels: 对 input_ids 做 copy，mask 掉 user 部分
        # 简化处理：整个序列都参与训练（实际应只训练 assistant 部分）
        labels = inputs["input_ids"].clone()
        # mask padding
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        return inputs

    # 处理前 n_samples 条
    batches = []
    for i, rec in enumerate(train_recs[:args.n_samples]):
        try:
            inputs = prepare_sample(rec, processor, args.max_pixels)
            batches.append(inputs)
            print(f"  Sample {i+1}: input_ids shape={inputs['input_ids'].shape}")
        except Exception as e:
            print(f"  Sample {i+1}: SKIP - {e}")

    if not batches:
        print("  [FATAL] No valid samples!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. 训练
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training ({args.n_steps} steps) ...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    model.train()
    total_loss = 0
    step = 0

    for epoch in range(10):  # 最多 10 个 epoch
        for inputs in batches:
            if step >= args.n_steps:
                break

            # 移到 GPU
            batch = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    batch[k] = [x.to(model.device) for x in v]
                else:
                    batch[k] = v

            try:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                step += 1

                mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  Step {step}/{args.n_steps} | Loss: {loss.item():.4f} | "
                      f"Avg: {total_loss/step:.4f} | GPU Peak: {mem_gb:.1f} GB")

            except torch.cuda.OutOfMemoryError:
                print(f"  Step {step}: OOM! Clearing cache ...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"  Step {step}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                continue

        if step >= args.n_steps:
            break

    print(f"\n  Training done: {step} steps, avg loss: {total_loss/max(step,1):.4f}")
    print(f"  GPU peak mem: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    # ------------------------------------------------------------------
    # 6. 保存
    # ------------------------------------------------------------------
    print("\n[6/6] Save LoRA adapter ...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir / "lora_adapter")
    processor.save_pretrained(output_dir / "lora_adapter")

    log = {
        "model": args.model_path,
        "n_samples": len(batches),
        "n_steps": step,
        "avg_loss": total_loss / max(step, 1),
        "gpu_peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "lora_rank": args.lora_rank,
        "lr": args.lr,
        "status": "success" if step > 0 else "failed",
    }
    (output_dir / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Adapter saved: {output_dir / 'lora_adapter'}")
    print(f"  Log: {output_dir / 'train_log.json'}")
    print("\n=== SFT Smoke Test Complete ===")


if __name__ == "__main__":
    main()
