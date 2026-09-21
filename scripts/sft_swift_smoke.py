#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_swift_smoke.py - 用 ms-swift API 做 SFT 冒烟测试

测试所有模型是否可以通过 swift 加载和微调
"""
import json
import os
import sys
import time
import traceback

import torch


MODELS = [
    ("Qwen3-VL-8B-Instruct", "/root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master"),
    ("Qwen3-VL-2B-Instruct", "/root/models/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master"),
    ("Qwen2.5-VL-7B-Instruct", "/root/models/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master"),
    ("InternVL3_5-2B", "/root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master"),
    ("InternVL3_5-8B", "/root/models/models/OpenGVLab--InternVL3_5-8B/snapshots/master"),
]


def test_swift_load(model_id, model_path):
    """用 swift API 测试模型加载"""
    result = {"id": model_id, "path": model_path, "status": "unknown"}
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")

    try:
        from swift import SftArguments, get_model_processor, get_template
        from swift.utils import get_logger

        logger = get_logger()

        # 构造 SftArguments
        # swift 4.5.2 使用 SftArguments 来配置
        args = SftArguments(
            model=model_path,
            model_type=None,  # 自动推断
            torch_dtype=torch.bfloat16,
            sft_type="lora",
            lora_rank=8,
            lora_alpha=16,
            target_modules="ALL",
            output_dir="runs/swift_smoke",
        )

        print("[1/3] Loading model via swift ...")
        t0 = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model, processor = get_model_processor(args.model, args.model_type, 
                                                 torch_dtype=args.torch_dtype,
                                                 model_cache_dir=args.model_cache_dir)
        
        load_time = time.time() - t0
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  Loaded in {load_time:.1f}s, GPU: {mem:.1f} GB")
        result["load_time_s"] = round(load_time, 1)
        result["load_mem_gb"] = round(mem, 1)

        # 检查参数
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")
        result["total_params"] = n_params

        print("[2/3] Model type: " + str(type(model).__name__))
        result["model_class"] = type(model).__name__

        print("[3/3] Cleanup ...")
        del model
        torch.cuda.empty_cache()
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
    print("Swift SFT Smoke Test")
    print("=" * 60)

    results = []
    for model_id, path in MODELS:
        r = test_swift_load(model_id, path)
        results.append(r)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        s = r["status"]
        ls = r.get("load_time_s", "-")
        m = r.get("load_mem_gb", "-")
        mc = r.get("model_class", "-")
        print(f"  {r['id']:<30} {s:<8} load={ls}s  mem={m}GB  class={mc}")

    n = sum(1 for r in results if r["status"] == "success")
    print(f"\n{n}/{len(results)} models loaded successfully via swift.")

    with open("runs/swift_smoke.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
