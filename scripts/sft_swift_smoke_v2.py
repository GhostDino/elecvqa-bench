#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_swift_smoke_v2.py - 用 ms-swift 4.5.2 API 做 SFT 冒烟测试

直接调用 swift sft_main 函数进行最小化训练
"""
import json
import os
import sys
import time
import traceback

import torch


MODELS = [
    ("Qwen3-VL-8B", "/root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master"),
    ("Qwen3-VL-2B", "/root/models/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master"),
    ("InternVL3_5-2B", "/root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master"),
    ("InternVL3_5-8B", "/root/models/models/OpenGVLab--InternVL3_5-8B/snapshots/master"),
    ("Qwen2.5-VL-7B", "/root/models/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master"),
]


def test_swift_sft(model_id, model_path):
    """用 swift sft_main 做最小化训练"""
    result = {"id": model_id, "path": model_path, "status": "unknown"}
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    try:
        from swift import sft_main, SftArguments

        output_dir = f"runs/swift_smoke/{model_id}"
        os.makedirs(output_dir, exist_ok=True)

        # 构造参数 - swift 4.5.2 使用 tuner_type 而非 sft_type
        args = SftArguments(
            model=model_path,
            output_dir=output_dir,
            tuner_type="lora",
            lora_rank=8,
            lora_alpha=16,
            target_modules="ALL",
            per_device_train_batch_size=1,
            max_steps=3,
            learning_rate=1e-4,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            dataset=["#custom"],
            custom_train_dataset_path="work/sft/train.jsonl",
            custom_val_dataset_path="work/sft/val.jsonl",
            max_length=1024,
        )

        print("[1/2] Starting swift sft ...")
        t0 = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sft_main(args)

        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Training completed in {elapsed:.1f}s")
        print(f"  GPU peak: {peak_mem:.1f} GB")
        result["time_s"] = round(elapsed, 1)
        result["gpu_peak_gb"] = round(peak_mem, 1)
        result["status"] = "success"

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)[:500]
    finally:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


def main():
    print("=" * 60)
    print("Swift SFT Smoke Test v2")
    print("=" * 60)

    results = []
    for model_id, path in MODELS:
        r = test_swift_sft(model_id, path)
        results.append(r)
        with open("runs/swift_smoke_v2.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        s = r["status"]
        t = r.get("time_s", "-")
        m = r.get("gpu_peak_gb", "-")
        print(f"  {r['id']:<25} {s:<8} time={t}s  peak={m}GB")

    n = sum(1 for r in results if r["status"] == "success")
    print(f"\n{n}/{len(results)} models passed swift SFT smoke test.")


if __name__ == "__main__":
    main()
