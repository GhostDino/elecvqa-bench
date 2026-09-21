#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_smoke_internvl.py - InternVL3.5 LoRA SFT 冒烟测试

InternVL3.5 使用自定义 modeling 代码，需要特殊处理 processor 和数据格式
"""
import json
import os
import sys
import time
import traceback

import torch
from PIL import Image


def test_internvl(model_id, model_path):
    """测试 InternVL3.5 模型"""
    result = {"id": model_id, "path": model_path, "status": "unknown"}
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    model = None
    try:
        # Step 1: 加载 tokenizer (不是 processor)
        print("[1/4] Load tokenizer ...")
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  Tokenizer: {type(tokenizer).__name__}")

        # Step 2: 加载模型
        print("[2/4] Load model ...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  model_type: {config.model_type}")

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        load_time = time.time() - t0
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"  Loaded in {load_time:.1f}s, GPU mem: {mem_gb:.1f} GB")
        result["load_time_s"] = round(load_time, 1)
        result["load_mem_gb"] = round(mem_gb, 1)

        # 检查模型是否有 vision_tower
        has_vision = hasattr(model, "vision_tower") or hasattr(model, "model") and hasattr(model.model, "vision_tower")
        print(f"  Has vision_tower: {has_vision}")

        # Step 3: LoRA
        print("[3/4] Inject LoRA ...")
        from peft import LoraConfig, get_peft_model

        # InternVL 的 LLM 部分是 Qwen2/Qwen3，LoRA 目标在 language_model 中
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = trainable / total * 100 if total > 0 else 0
        print(f"  Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
        result["trainable_params"] = trainable
        result["total_params"] = total
        result["trainable_pct"] = round(pct, 4)

        # Step 4: 简单文本前向+反向 (不加载图片，只验证训练流程)
        print("[4/4] Forward + backward (text only) ...")
        model.train()

        # 加载训练样本获取 question 和 answer
        with open("work/sft/train.jsonl", "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        messages = rec["messages"]
        question = ""
        answer = ""
        for content in messages[0]["content"]:
            if content["type"] == "text":
                question = content["text"]
        if len(messages) > 1:
            answer = messages[1]["content"]

        # InternVL 的对话格式
        # 使用简单文本编码
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = inputs["input_ids"].clone()
        inputs["labels"] = labels

        # to GPU
        batch = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Loss: {loss.item():.4f}")
        print(f"  GPU peak: {peak_mem:.1f} GB")
        result["loss"] = round(loss.item(), 4)
        result["gpu_peak_gb"] = round(peak_mem, 1)
        result["status"] = "success"
        result["note"] = "text-only forward (image pipeline needs InternVL custom code)"

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)[:500]
    finally:
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


def main():
    print("=" * 60)
    print("InternVL3.5 SFT Smoke Test")
    print("=" * 60)

    models = [
        ("InternVL3_5-2B", "/root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master"),
        ("InternVL3_5-8B", "/root/models/models/OpenGVLab--InternVL3_5-8B/snapshots/master"),
    ]

    results = []
    for model_id, path in models:
        r = test_internvl(model_id, path)
        results.append(r)
        with open("runs/sft_smoke_internvl.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        s = r["status"]
        ls = r.get("load_time_s", "-")
        m = r.get("load_mem_gb", "-")
        tp = f"{r.get('trainable_pct', 0):.2f}%" if "trainable_pct" in r else "-"
        l = r.get("loss", "-")
        p = r.get("gpu_peak_gb", "-")
        print(f"{r['id']:<20} {s:<8} load={ls}s  mem={m}GB  train={tp}  loss={l}  peak={p}GB")

    n = sum(1 for r in results if r["status"] == "success")
    print(f"\n{n}/{len(results)} InternVL models passed.")


if __name__ == "__main__":
    main()
