#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_swift_final.py - 用 swift sft_main 做正确参数的冒烟测试

swift 4.5.2 正确参数:
  - dataset: list of dataset names or paths
  - val_dataset: list of val dataset names or paths  
  - tuner_type: "lora" (not sft_type)
"""
import json
import os
import sys
import time
import traceback

import torch


MODELS = [
    ("Qwen3-VL-8B", "/root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master"),
    ("InternVL3_5-2B", "/root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master"),
]


def test_swift_sft(model_id, model_path):
    result = {"id": model_id, "path": model_path, "status": "unknown"}
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")

    try:
        from swift import sft_main, SftArguments

        output_dir = f"runs/swift_smoke/{model_id}"
        os.makedirs(output_dir, exist_ok=True)

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
            dataset=["work/sft/train.jsonl"],
            val_dataset=["work/sft/val.jsonl"],
            max_length=1024,
        )

        print("[1/1] Running swift sft ...")
        t0 = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sft_main(args)

        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Done in {elapsed:.1f}s, GPU peak: {peak_mem:.1f} GB")
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
    print("Swift SFT Final Smoke Test")
    print("=" * 60)

    results = []
    for model_id, path in MODELS:
        r = test_swift_sft(model_id, path)
        results.append(r)
        with open("runs/swift_smoke_final.json", "w", encoding="utf-8") as f:
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
    print(f"\n{n}/{len(results)} passed.")


if __name__ == "__main__":
    main()
