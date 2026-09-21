#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_sft_swift.py - Evaluate SFT models using swift infer engine

Uses swift's own inference pipeline (handles chat templates, image processing
correctly for both InternVL3.5 and Qwen3-VL).

Steps:
  1. Convert VQA test data to swift messages format
  2. Run swift infer CLI with --adapters
  3. Parse results and compute metrics
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter


def load_vqa_test(vqa_dir, task):
    task_files = {"T2": "T2_binary.jsonl", "T3": "T3_multiclass.jsonl"}
    path = Path(vqa_dir) / task_files.get(task, f"{task}.jsonl")
    if not path.exists():
        return []
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("split") == "test":
                    recs.append(rec)
    return recs


def vqa_to_swift_infer(rec, data_root):
    """Convert VQA record to swift messages format (with GT answer for metric)"""
    img_path = Path(data_root) / rec["image_path"]
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": rec["question"]},
            ]},
            {"role": "assistant", "content": rec["answer"]},
        ],
        # Extra metadata for analysis
        "crop_id": rec.get("crop_id", ""),
        "asset": rec.get("asset", ""),
        "label": rec.get("label", ""),
        "gt_answer": rec["answer"],
    }


def parse_response(response, task):
    response = response.strip()
    for ch in response:
        if ch in "ABCDE":
            return ch, True
    if "正常" in response and "缺陷" not in response:
        return "A", True
    if "缺陷" in response or "锈" in response:
        return "B", True
    return None, False


def compute_metrics(predictions, ground_truths, task):
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    valid_preds, valid_gts = [], []
    n_format_fail = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred is None:
            n_format_fail += 1
            continue
        valid_preds.append(pred)
        valid_gts.append(gt)
    n_total = len(predictions)
    n_valid = len(valid_preds)
    fmt_fail_rate = n_format_fail / n_total if n_total > 0 else 0
    if n_valid == 0:
        return {"n_total": n_total, "n_valid": 0, "format_failure_rate": 1.0,
                "balanced_accuracy": 0.0, "macro_f1": 0.0}
    labels = sorted(set(valid_gts) | set(valid_preds))
    bal_acc = balanced_accuracy_score(valid_gts, valid_preds)
    macro_f1 = f1_score(valid_gts, valid_preds, average="macro", labels=labels, zero_division=0)
    cm = confusion_matrix(valid_gts, valid_preds, labels=labels)
    metrics = {
        "n_total": n_total, "n_valid": n_valid,
        "format_failure_rate": round(fmt_fail_rate, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "pred_distribution": dict(Counter(valid_preds)),
        "gt_distribution": dict(Counter(valid_gts)),
    }
    if task == "T2":
        miss, halluc = 0, 0
        n_defect_gt, n_normal_gt = 0, 0
        for pred, gt in zip(valid_preds, valid_gts):
            if gt == "B":
                n_defect_gt += 1
                if pred == "A":
                    miss += 1
            elif gt == "A":
                n_normal_gt += 1
                if pred == "B":
                    halluc += 1
        metrics["miss_rate"] = round(miss / n_defect_gt, 4) if n_defect_gt > 0 else 0.0
        metrics["hallucination_rate"] = round(halluc / n_normal_gt, 4) if n_normal_gt > 0 else 0.0
    return metrics


def run_swift_infer(model_path, adapter_path, test_jsonl, result_path,
                    max_new_tokens=32, infer_backend="pt"):
    """Run swift infer CLI command"""
    cmd = [
        "/root/miniconda3/envs/py3.11/bin/swift", "infer",
        "--model", model_path,
        "--val_dataset", str(test_jsonl),
        "--infer_backend", infer_backend,
        "--max_new_tokens", str(max_new_tokens),
        "--result_path", str(result_path),
        "--torch_dtype", "bfloat16",
        "--device_map", "auto",
    ]
    if adapter_path and Path(adapter_path).exists():
        cmd.extend(["--adapters", str(adapter_path)])

    print(f"  Running swift infer ...")
    print(f"  Model: {model_path}")
    print(f"  Adapter: {adapter_path or '(none)'}")
    print(f"  Test data: {test_jsonl}")
    print(f"  Result: {result_path}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] swift infer failed (rc={result.returncode})")
        print(f"  stdout (last 1000): {result.stdout[-1000:]}")
        print(f"  stderr (last 1000): {result.stderr[-1000:]}")
        return None, elapsed

    print(f"  swift infer done in {elapsed:.0f}s")
    return result_path, elapsed


def parse_swift_results(result_path, test_data):
    """Parse swift infer result JSONL and match with ground truth"""
    results = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    predictions = []
    ground_truths = []
    raw_responses = []
    matched = 0

    for i, rec in enumerate(test_data):
        if i < len(results):
            r = results[i]
            # swift infer result format: {"response": "...", "label": "...", ...}
            response = r.get("response", r.get("prediction", r.get("output", "")))
            if isinstance(response, list):
                response = " ".join(str(x) for x in response)
            raw_responses.append(response)
            pred, _ = parse_response(response, rec.get("task", "T2"))
            predictions.append(pred)
            ground_truths.append(rec["answer"])
            matched += 1
        else:
            predictions.append(None)
            ground_truths.append(rec["answer"])
            raw_responses.append("[MISSING]")

    print(f"  Matched {matched}/{len(test_data)} results")
    return predictions, ground_truths, raw_responses


def main():
    ap = argparse.ArgumentParser(description="Evaluate SFT models using swift infer")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default="")
    ap.add_argument("--vqa-dir", default="work/vqa")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--tasks", nargs="+", default=["T2", "T3"])
    ap.add_argument("--n-samples", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--infer-backend", default="pt")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    model_name = args.run_name or Path(args.model_path).name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SFT Evaluation via Swift Infer")
    print(f"  Model: {args.model_path}")
    print(f"  Adapter: {args.adapter_path or '(none)'}")
    print(f"  Run name: {model_name}")
    print("=" * 60)

    all_results = {}

    for task in args.tasks:
        test_data = load_vqa_test(args.vqa_dir, task)
        if args.n_samples > 0 and args.n_samples < len(test_data):
            random.seed(42)
            test_data = random.sample(test_data, args.n_samples)

        print(f"\n{'='*60}")
        print(f"Task {task}: {len(test_data)} test samples")
        print(f"{'='*60}")

        # Convert to swift format
        swift_data = [vqa_to_swift_infer(rec, args.data_root) for rec in test_data]
        test_jsonl = out_dir / f"_test_{task}_{model_name}.jsonl"
        with open(test_jsonl, "w", encoding="utf-8") as f:
            for rec in swift_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(swift_data)} test records to {test_jsonl}")

        # Run swift infer
        result_path = out_dir / f"_result_{task}_{model_name}.jsonl"
        result_file, infer_time = run_swift_infer(
            args.model_path, args.adapter_path, test_jsonl, result_path,
            args.max_new_tokens, args.infer_backend
        )

        if result_file is None:
            print(f"  [SKIP] Inference failed for {task}")
            continue

        # Parse results
        predictions, ground_truths, raw_responses = parse_swift_results(result_file, test_data)

        # Compute metrics
        metrics = compute_metrics(predictions, ground_truths, task)
        metrics["model"] = model_name
        metrics["task"] = task
        metrics["adapter"] = args.adapter_path or "(none)"
        metrics["infer_time_s"] = round(infer_time, 1)

        print(f"\n  Results ({task}):")
        print(f"    N: {metrics['n_total']}, Valid: {metrics['n_valid']}")
        print(f"    Format failure: {metrics['format_failure_rate']:.2%}")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    Macro-F1: {metrics['macro_f1']:.4f}")
        if task == "T2":
            print(f"    Miss rate: {metrics.get('miss_rate', 0):.4f}")
            print(f"    Hallucination rate: {metrics.get('hallucination_rate', 0):.4f}")

        # Save metrics
        run_id = f"{model_name}_{task}_P3_test"
        if args.n_samples > 0:
            run_id += f"_sample{args.n_samples}"
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for i, rec in enumerate(test_data):
                f.write(json.dumps({
                    "crop_id": rec.get("crop_id", ""),
                    "gt": rec["answer"],
                    "pred": predictions[i],
                    "raw_response": raw_responses[i],
                    "asset": rec.get("asset", ""),
                    "label": rec.get("label", ""),
                }, ensure_ascii=False) + "\n")

        all_results[run_id] = metrics

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {model_name}")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Task':<5} {'BalAcc':>8} {'F1':>8} {'Miss':>8} {'Hall':>8}")
    print("-" * 70)
    for run_id, m in sorted(all_results.items()):
        miss = m.get("miss_rate", 0)
        hall = m.get("hallucination_rate", 0)
        print(f"{model_name:<30} {m['task']:<5} {m['balanced_accuracy']:>8.4f} {m['macro_f1']:>8.4f} {miss:>8.4f} {hall:>8.4f}")

    summary_path = out_dir / f"sft_eval_{model_name.replace('-','_').replace('.','_')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {summary_path}")

    # Cleanup temp files
    for task in args.tasks:
        for suffix in ["test", "result"]:
            tmp = out_dir / f"_{suffix}_{task}_{model_name}.jsonl"
            if tmp.exists():
                tmp.unlink()


if __name__ == "__main__":
    main()
