#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
longtail_vlm.py - C3 VLM long-tail crossover experiment

For each n in {4, 8, 16, 32, 64, 128}:
  1. Subsample SFT training data to n samples per class
  2. Run swift SFT with subsampled data (LoRA, reduced epochs)
  3. Evaluate on fixed test set using swift infer
  4. Record BalAcc / Macro-F1 / per-class accuracy

The "all" data point comes from the main T60/T70 training runs.

Usage:
  python longtail_vlm.py --model internvl2b --n-values 4 8 16 32 64 128 --seeds 42
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

MODEL_CONFIGS = {
    "internvl2b": {
        "model_path": "/root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master",
        "model_name": "InternVL3.5-2B",
    },
    "internvl8b": {
        "model_path": "/root/models/models/OpenGVLab--InternVL3_5-8B/snapshots/master",
        "model_name": "InternVL3.5-8B",
    },
    "qwen3vl8b": {
        "model_path": "/root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master",
        "model_name": "Qwen3-VL-8B",
    },
}

LABEL_CANON = {
    "corrosao": "rust", "corrosion": "rust", "rust": "rust",
    "normal": "good", "good": "good",
    "missing-cap": "missing-cap", "missingcap": "missing-cap",
    "bird-nest": "nest", "nest": "nest",
    "torned-up": "torned-up", "peeling-paint": "peeling-paint",
}

def canon(label):
    return LABEL_CANON.get(label.strip().lower(), label.strip().lower())


def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_splits(splits_dir):
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = load_jsonl(Path(splits_dir) / f"{name}.jsonl")
    return splits


def match_sft_to_label(sft_recs, splits):
    """Match SFT records to labels using image_path -> label mapping"""
    path_to_label = {}
    for split_name in ["train", "val", "test"]:
        for r in splits[split_name]:
            if r.get("source_subset") in ("supervised", "unsupervised"):
                p = r.get("path", "")
                path_to_label[p] = canon(r["label"])
                path_to_label[Path(p).name] = canon(r["label"])

    sft_with_label = []
    for sft_rec in sft_recs:
        messages = sft_rec.get("messages", [])
        if not messages:
            continue
        for item in messages[0].get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                img_path = item.get("image", "")
                label = path_to_label.get(img_path) or path_to_label.get(Path(img_path).name)
                if label:
                    sft_with_label.append({**sft_rec, "_label": label})
                    break
    return sft_with_label


def sample_sft_by_n(sft_with_label, n_per_class, seed):
    random.seed(seed)
    by_class = defaultdict(list)
    for rec in sft_with_label:
        by_class[rec["_label"]].append(rec)
    sampled = []
    for label, recs in by_class.items():
        if n_per_class == 0:
            sampled.extend(recs)
        else:
            random.shuffle(recs)
            sampled.extend(recs[:n_per_class])
    random.shuffle(sampled)
    return [{k: v for k, v in rec.items() if not k.startswith("_")} for rec in sampled]


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_swift_sft(model_path, train_jsonl, val_jsonl, output_dir, max_steps=0,
                  num_epochs=3, seed=42):
    cmd = [
        "/root/miniconda3/envs/py3.11/bin/swift", "sft",
        "--model", model_path,
        "--tuner_type", "lora",
        "--lora_rank", "32", "--lora_alpha", "64",
        "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "1e-4",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--weight_decay", "0.01",
        "--bf16", "true",
        "--gradient_checkpointing", "true",
        "--logging_steps", "10",
        "--save_strategy", "epoch",
        "--save_total_limit", "1",
        "--report_to", "none",
        "--dataset", str(train_jsonl),
        "--val_dataset", str(val_jsonl),
        "--max_length", "2048",
        "--truncation_strategy", "delete",
        "--output_dir", str(output_dir),
        "--seed", str(seed),
    ]
    if max_steps > 0:
        cmd.extend(["--max_steps", str(max_steps)])
    else:
        cmd.extend(["--num_train_epochs", str(num_epochs)])

    print(f"  Running swift sft ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] swift sft failed (rc={result.returncode})")
        print(f"  stderr (last 500): {result.stderr[-500:]}")
        return None, elapsed
    print(f"  swift sft done in {elapsed:.0f}s")
    return output_dir, elapsed


def find_latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    version_dirs = sorted(output_dir.glob("v0-*"))
    if not version_dirs:
        return None
    latest = version_dirs[-1]
    ckpt_dirs = sorted(latest.glob("checkpoint-*"))
    if ckpt_dirs:
        return str(ckpt_dirs[-1])
    return str(latest)


def run_swift_infer_eval(model_path, adapter_path, test_jsonl, result_path):
    """Run swift infer and return parsed metrics"""
    cmd = [
        "/root/miniconda3/envs/py3.11/bin/swift", "infer",
        "--model", model_path,
        "--val_dataset", str(test_jsonl),
        "--infer_backend", "pt",
        "--max_new_tokens", "32",
        "--torch_dtype", "bfloat16",
        "--device_map", "auto",
        "--result_path", str(result_path),
    ]
    if adapter_path and Path(adapter_path).exists():
        cmd.extend(["--adapters", str(adapter_path)])

    print(f"  Running swift infer ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] swift infer failed (rc={result.returncode})")
        print(f"  stderr (last 500): {result.stderr[-500:]}")
        return None, elapsed

    # Parse results
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    results = load_jsonl(result_path)
    preds_raw = [r.get("response", "").strip() for r in results]
    gts = [r.get("labels", "").strip() for r in results]

    parsed_preds = []
    n_fail = 0
    for p in preds_raw:
        found = None
        for ch in p:
            if ch in "ABCDE":
                found = ch
                break
        if found is None:
            if "正常" in p and "缺陷" not in p:
                found = "A"
            elif "缺陷" in p or "锈" in p:
                found = "B"
        if found is None:
            n_fail += 1
        parsed_preds.append(found)

    valid_preds = [p for p in parsed_preds if p is not None]
    valid_gts = [g for p, g in zip(parsed_preds, gts) if p is not None]

    if not valid_preds:
        return {"balanced_accuracy": 0, "macro_f1": 0, "format_failure_rate": 1.0,
                "n_total": len(preds_raw), "n_valid": 0}, elapsed

    labels = sorted(set(valid_gts) | set(valid_preds))
    bal_acc = balanced_accuracy_score(valid_gts, valid_preds)
    macro_f1 = f1_score(valid_gts, valid_preds, average="macro", labels=labels, zero_division=0)

    return {
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "format_failure_rate": round(n_fail / len(preds_raw), 4),
        "n_total": len(preds_raw),
        "n_valid": len(valid_preds),
        "pred_distribution": dict(Counter(valid_preds)),
        "gt_distribution": dict(Counter(valid_gts)),
    }, elapsed


def prepare_test_data(vqa_dir, data_root, out_path, task, n_samples=200):
    """Prepare test data in swift format"""
    task_files = {"T2": "T2_binary.jsonl", "T3": "T3_multiclass.jsonl"}
    recs = load_jsonl(Path(vqa_dir) / task_files.get(task, f"{task}.jsonl"))
    test_recs = [r for r in recs if r.get("split") == "test"]
    if n_samples > 0 and n_samples < len(test_recs):
        random.seed(42)
        test_recs = random.sample(test_recs, n_samples)

    swift_data = []
    for rec in test_recs:
        img_path = str(Path(data_root) / rec["image_path"])
        swift_data.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": rec["question"]},
                ]},
                {"role": "assistant", "content": rec["answer"]},
            ],
        })
    write_jsonl(swift_data, out_path)
    return len(swift_data)


def main():
    ap = argparse.ArgumentParser(description="C3 VLM long-tail crossover experiment")
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--n-values", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--sft-dir", default="work/sft_full")
    ap.add_argument("--splits-dir", default="work/splits")
    ap.add_argument("--vqa-dir", default="work/vqa")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--num-epochs", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--out", default="runs/longtail_vlm")
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    base_out = Path(args.out) / args.model
    base_out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"C3 VLM Long-tail: {cfg['model_name']}")
    print(f"  N values: {args.n_values}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Max steps: {args.max_steps if args.max_steps > 0 else f'{args.num_epochs} epochs'}")
    print(f"  Test samples per task: {args.n_test}")
    print("=" * 60)

    # Load and prepare data
    splits = load_splits(args.splits_dir)
    sft_train = load_jsonl(Path(args.sft_dir) / "train.jsonl")
    sft_val = load_jsonl(Path(args.sft_dir) / "val.jsonl")
    sft_with_label = match_sft_to_label(sft_train, splits)
    print(f"  SFT train: {len(sft_train)}, with label: {len(sft_with_label)}")
    label_counts = Counter(r["_label"] for r in sft_with_label)
    print(f"  Label distribution: {dict(label_counts)}")

    # Prepare fixed test data
    test_files = {}
    for task in ["T2", "T3"]:
        test_path = base_out / f"test_{task}.jsonl"
        n = prepare_test_data(args.vqa_dir, args.data_root, test_path, task, args.n_test)
        test_files[task] = test_path
        print(f"  Test {task}: {n} samples")

    # Use small val set
    if len(sft_val) > 200:
        random.seed(42)
        sft_val_small = random.sample(sft_val, 200)
    else:
        sft_val_small = sft_val
    val_jsonl = base_out / "val_small.jsonl"
    write_jsonl(sft_val_small, val_jsonl)

    results = []
    swift_bin = "/root/miniconda3/envs/py3.11/bin/swift"
    py_bin = "/root/miniconda3/envs/py3.11/bin/python"

    for n in args.n_values:
        n_label = "all" if n == 0 else str(n)
        for seed in args.seeds:
            run_id = f"{cfg['model_name']}_n{n_label}_seed{seed}"
            result_file = base_out / f"{run_id}.json"

            if result_file.exists():
                print(f"\n--- {run_id}: already done, skipping ---")
                with open(result_file) as f:
                    results.append(json.load(f))
                continue

            print(f"\n{'='*60}")
            print(f"--- {run_id} ---")
            print(f"{'='*60}")

            # Subsample
            sampled = sample_sft_by_n(sft_with_label, n, seed)
            print(f"  Sampled train: {len(sampled)}")
            train_jsonl = base_out / "data" / f"train_n{n_label}_seed{seed}.jsonl"
            write_jsonl(sampled, train_jsonl)

            # SFT training
            sft_out = base_out / "sft" / run_id
            ckpt, train_time = run_swift_sft(
                cfg["model_path"], str(train_jsonl), str(val_jsonl),
                str(sft_out), args.max_steps, args.num_epochs, seed
            )
            if ckpt is None:
                continue
            ckpt_path = find_latest_checkpoint(ckpt)
            print(f"  Checkpoint: {ckpt_path}")

            # Evaluation
            task_metrics = {}
            for task, test_path in test_files.items():
                result_path = base_out / f"result_{run_id}_{task}.jsonl"
                metrics, eval_time = run_swift_infer_eval(
                    cfg["model_path"], ckpt_path, test_path, result_path
                )
                if metrics:
                    task_metrics[task] = metrics
                    print(f"  {task}: BalAcc={metrics['balanced_accuracy']:.4f}, "
                          f"F1={metrics['macro_f1']:.4f}")

            result = {
                "run_id": run_id,
                "model": cfg["model_name"],
                "n_per_class": n,
                "seed": seed,
                "n_train": len(sampled),
                "checkpoint": ckpt_path,
                "train_time_s": round(train_time, 1),
                "metrics": task_metrics,
            }
            results.append(result)
            with open(result_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {cfg['model_name']}")
    print(f"{'='*60}")
    print(f"{'N':>5} {'T2_BalAcc':>10} {'T2_F1':>8} {'T3_BalAcc':>10} {'T3_F1':>8}")
    print("-" * 45)
    for r in results:
        n_label = "all" if r["n_per_class"] == 0 else str(r["n_per_class"])
        m = r.get("metrics", {})
        t2 = m.get("T2", {})
        t3 = m.get("T3", {})
        print(f"{n_label:>5} {t2.get('balanced_accuracy', 0):>10.4f} "
              f"{t2.get('macro_f1', 0):>8.4f} {t3.get('balanced_accuracy', 0):>10.4f} "
              f"{t3.get('macro_f1', 0):>8.4f}")

    with open(base_out / "summary.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {base_out / 'summary.json'}")


if __name__ == "__main__":
    main()
