#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_smoke_all_v2.py - 对所有模型进行 LoRA SFT 冒烟测试 (v2)

修复:
  - 使用 AutoModelForCausalLM 替代 AutoModel
  - 对 VLM 模型不设置 task_type (让 peft 自动推断)
  - 对 InternVL 使用 trust_remote_code
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image


MODELS = [
    {
        "id": "Qwen3-VL-8B-Instruct",
        "path": "/root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master",
        "model_class": "Qwen3VLForConditionalGeneration",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },
    {
        "id": "Qwen3-VL-2B-Instruct",
        "path": "/root/models/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master",
        "model_class": "Qwen3VLForConditionalGeneration",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },
    {
        "id": "InternVL3_5-8B",
        "path": "/root/models/models/OpenGVLab--InternVL3_5-8B/snapshots/master",
        "model_class": "AutoModelForCausalLM",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },
    {
        "id": "InternVL3_5-2B",
        "path": "/root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master",
        "model_class": "AutoModelForCausalLM",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },
    {
        "id": "Qwen2.5-VL-7B-Instruct",
        "path": "/root/models/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master",
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },
]


def load_train_sample():
    with open("work/sft/train.jsonl", "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def get_model_class(class_name):
    """动态获取模型类"""
    import transformers
    if class_name == "AutoModelForCausalLM":
        return transformers.AutoModelForCausalLM
    # 尝试从 transformers 直接导入
    if hasattr(transformers, class_name):
        return getattr(transformers, class_name)
    # 尝试从子模块导入
    try:
        # Qwen3VL
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    except ImportError:
        pass
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        pass
    # fallback
    return transformers.AutoModelForCausalLM


def test_model(model_cfg, train_sample):
    """测试单个模型"""
    model_id = model_cfg["id"]
    model_path = model_cfg["path"]
    result = {"id": model_id, "path": model_path, "status": "unknown"}

    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"  Path: {model_path}")
    print(f"  Class: {model_cfg['model_class']}")
    print(f"{'='*60}")

    model = None
    try:
        # Step 1: processor
        print("[1/4] Load processor ...")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        result["processor"] = "OK"

        # Step 2: model
        print("[2/4] Load model ...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        ModelClass = get_model_class(model_cfg["model_class"])
        print(f"  Using: {ModelClass.__name__}")

        model = ModelClass.from_pretrained(
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

        # Step 3: LoRA
        print("[3/4] Inject LoRA ...")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=model_cfg["lora_targets"],
            bias="none",
            # 不设置 task_type, 让 peft 自动处理
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = trainable / total * 100 if total > 0 else 0
        print(f"  Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
        result["trainable_params"] = trainable
        result["total_params"] = total
        result["trainable_pct"] = round(pct, 4)

        # Step 4: forward + backward
        print("[4/4] Forward + backward ...")
        model.train()

        messages = train_sample["messages"]

        # 尝试 qwen_vl_utils
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        except Exception as e1:
            print(f"  qwen_vl_utils failed ({e1}), trying manual ...")
            # manual processing
            image_path = None
            question = ""
            answer = ""
            for content in messages[0]["content"]:
                if content["type"] == "image":
                    image_path = content["image"]
                elif content["type"] == "text":
                    question = content["text"]
            if len(messages) > 1:
                answer = messages[1]["content"]

            image = None
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                w, h = image.size
                if w * h > 589824:
                    scale = (589824 / (w * h)) ** 0.5
                    image = image.resize(
                        (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(
                text=[text], images=[image] if image else None,
                padding=True, return_tensors="pt",
            )

        # labels
        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels

        # to GPU
        batch = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                batch[k] = [x.to(model.device) for x in v]
            else:
                batch[k] = v

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Loss: {loss.item():.4f}")
        print(f"  GPU peak: {peak_mem:.1f} GB")
        result["loss"] = round(loss.item(), 4)
        result["gpu_peak_gb"] = round(peak_mem, 1)
        result["status"] = "success"

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
    print("SFT Smoke Test for All Models (v2)")
    print(f"Models: {len(MODELS)}")
    print("=" * 60)

    train_sample = load_train_sample()
    print(f"Train sample: {train_sample['metadata']['task']}")

    results = []
    for cfg in MODELS:
        result = test_model(cfg, train_sample)
        results.append(result)
        with open("runs/sft_smoke_all.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<30} {'Status':<8} {'Load(s)':>8} {'Mem(GB)':>8} {'Train%':>8} {'Loss':>8} {'Peak(GB)':>8}")
    print("-" * 85)
    for r in results:
        s = r["status"]
        ls = str(r.get("load_time_s", "-"))
        m = str(r.get("load_mem_gb", "-"))
        tp = f"{r.get('trainable_pct', 0):.2f}" if "trainable_pct" in r else "-"
        l = str(r.get("loss", "-"))
        p = str(r.get("gpu_peak_gb", "-"))
        print(f"{r['id']:<30} {s:<8} {ls:>8} {m:>8} {tp:>8} {l:>8} {p:>8}")

    n = sum(1 for r in results if r["status"] == "success")
    print(f"\n{n}/{len(results)} models passed.")

    with open("runs/sft_smoke_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
