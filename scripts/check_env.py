#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T01_check_env: Gate-0 环境验收脚本

检查项:
  1. CUDA compute capability == (12, 0)  [sm_120 Blackwell]
  2. GPU 显存 >= 95 GB
  3. bf16 支持
  4. 关键 Python 包已安装
  5. 磁盘剩余空间 >= 1.5 TB
  6. nvidia-smi driver >= 580

用法:
  python scripts/check_env.py --out work/audit/env_report.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def check_gpu():
    """检查 GPU 硬件"""
    results = {}
    try:
        import torch
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            results["compute_capability"] = list(cap)
            results["gpu_name"] = props.name
            results["total_memory_gb"] = round(props.total_memory / 1024**3, 1)
            results["bf16_supported"] = torch.cuda.is_bf16_supported()
            results["cap_ok"] = cap == (12, 0)
            results["mem_ok"] = props.total_memory >= 95 * 1024**3
            results["bf16_ok"] = results["bf16_supported"]
        else:
            results["cap_ok"] = False
            results["mem_ok"] = False
            results["bf16_ok"] = False
    except Exception as e:
        results["error"] = str(e)
        results["cap_ok"] = False
        results["mem_ok"] = False
        results["bf16_ok"] = False
    return results


def check_packages():
    """检查关键包"""
    required = {
        "must_have": ["torch", "transformers", "accelerate", "peft", "PIL",
                       "datasets", "pyyaml", "tqdm", "pandas", "numpy",
                       "sklearn", "scipy"],
        "for_inference": ["vllm"],
        "for_training": ["trl", "swift"],
        "for_data": ["pycocotools", "cv2"],
        "for_quant": ["bitsandbytes"],
    }
    results = {}
    for category, pkgs in required.items():
        results[category] = {}
        for pkg in pkgs:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                results[category][pkg] = {"installed": True, "version": ver}
            except ImportError:
                results[category][pkg] = {"installed": False}
    return results


def check_disk(path="/workspace"):
    """检查磁盘空间"""
    results = {}
    try:
        stat = os.statvfs(path)
        free_gb = stat.f_bavail * stat.f_frsize / 1024**3
        total_gb = stat.f_blocks * stat.f_frsize / 1024**3
        results["path"] = path
        results["free_gb"] = round(free_gb, 1)
        results["total_gb"] = round(total_gb, 1)
        results["free_ok"] = free_gb >= 1500  # 1.5 TB
    except Exception as e:
        results["error"] = str(e)
        results["free_ok"] = False
    return results


def check_driver():
    """检查 NVIDIA 驱动版本"""
    results = {}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                        "--format=csv,noheader"], timeout=10)
        ver_str = out.decode().strip()
        results["driver_version"] = ver_str
        major = int(ver_str.split(".")[0])
        results["driver_ok"] = major >= 580
    except Exception as e:
        results["error"] = str(e)
        results["driver_ok"] = False
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/audit/env_report.json")
    args = ap.parse_args()

    print("=" * 60)
    print("Gate-0: 环境验收")
    print("=" * 60)

    report = {"meta": {"script": "check_env.py", "version": "1.0"}}

    print("\n[1/4] GPU 硬件检查 ...")
    report["gpu"] = check_gpu()
    g = report["gpu"]
    print(f"  GPU: {g.get('gpu_name', 'N/A')}")
    print(f"  Compute Capability: {g.get('compute_capability', 'N/A')} "
          f"{'[OK]' if g.get('cap_ok') else '[FAIL]'} (期望 (12, 0))")
    print(f"  显存: {g.get('total_memory_gb', 'N/A')} GB "
          f"{'[OK]' if g.get('mem_ok') else '[FAIL]'} (期望 >= 95 GB)")
    print(f"  BF16: {g.get('bf16_supported', 'N/A')} "
          f"{'[OK]' if g.get('bf16_ok') else '[FAIL]'}")

    print("\n[2/4] Python 包检查 ...")
    report["packages"] = check_packages()
    for cat, pkgs in report["packages"].items():
        missing = [p for p, v in pkgs.items() if not v["installed"]]
        if missing:
            print(f"  {cat}: 缺失 {missing}")
        else:
            print(f"  {cat}: 全部已安装 [OK]")

    print("\n[3/4] 磁盘空间检查 ...")
    report["disk"] = check_disk("/workspace")
    d = report["disk"]
    print(f"  路径: {d.get('path')} / 剩余: {d.get('free_gb')} GB "
          f"{'[OK]' if d.get('free_ok') else '[WARN]'} (期望 >= 1500 GB)")

    print("\n[4/4] NVIDIA 驱动检查 ...")
    report["driver"] = check_driver()
    dr = report["driver"]
    print(f"  驱动版本: {dr.get('driver_version', 'N/A')} "
          f"{'[OK]' if dr.get('driver_ok') else '[FAIL]'} (期望 >= 580)")

    # Gate 判定
    gate_pass = (
        g.get("cap_ok", False) and
        g.get("mem_ok", False) and
        g.get("bf16_ok", False) and
        dr.get("driver_ok", False)
    )
    report["gate_pass"] = gate_pass

    # 缺失包汇总
    all_missing = []
    for cat, pkgs in report["packages"].items():
        for p, v in pkgs.items():
            if not v["installed"]:
                all_missing.append(f"{cat}/{p}")
    report["missing_packages"] = all_missing

    print("\n" + "=" * 60)
    if gate_pass:
        print("Gate-0: [PASS] 核心环境验收通过")
    else:
        print("Gate-0: [FAIL] 核心环境验收未通过")
    if all_missing:
        print(f"  缺失包（非阻塞，按需安装）: {all_missing}")
    print("=" * 60)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")

    # exit 0 if core gate passes, 2 if missing non-core packages
    sys.exit(0 if gate_pass else 2)


if __name__ == "__main__":
    main()
