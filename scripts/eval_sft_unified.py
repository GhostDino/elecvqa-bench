#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_sft_unified.py - Unified SFT evaluation for InternVL3.5 + Qwen3-VL

Supports:
  - InternVL3.5-2B/8B (CLIPImageProcessor + manual chat template)
  - Qwen3-VL-8B (AutoProcessor + qwen_vl_utils)

Metrics: BalAcc / Macro-F1 / miss_rate / hallucination_rate / format_failure_rate
"""
import argparse, json, os, sys, time, random
from pathlib import Path
from collections import Counter
import torch
from PIL import Image


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


def detect_model_type(model_path):
    path_lower = str(model_path).lower()
    if "internvl" in path_lower:
        return "internvl"
    if "qwen3-vl" in path_lower or "qwen3vl" in path_lower:
        return "qwen3vl"
    if "qwen2.5-vl" in path_lower or "qwen2.5vl" in path_lower:
        return "qwen25vl"
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", [])
        if arch and "InternVL" in arch[0]:
            return "internvl"
        if arch and "Qwen" in arch[0]:
            return "qwen3vl"
    return "internvl"


def load_internvl(model_path, adapter_path):
    from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
    from peft import PeftModel
    print("  [InternVL] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("  [InternVL] Loading image processor ...")
    image_processor = CLIPImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("  [InternVL] Loading base model ...")
    torch.cuda.empty_cache()
    t0 = time.time()
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    if adapter_path and Path(adapter_path).exists():
        print(f"  [InternVL] Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    load_time = time.time() - t0
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  [InternVL] Loaded in {load_time:.1f}s, GPU: {mem:.1f} GB")
    return model, tokenizer, image_processor


def internvl_infer(model, tokenizer, image_processor, image_path, question, device):
    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if w * h > 589824:
            scale = (589824 / (w * h)) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    except Exception:
        return "[IMAGE_ERROR]"
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
    img_token = "<image>"
    text = f"<|im_start|>user\n{img_token}\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    try:
        image_token_id = tokenizer.convert_tokens_to_ids(img_token)
    except Exception:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id and image_token_id != tokenizer.unk_token_id:
        inputs["input_ids"][0, 0] = image_token_id
    inputs["pixel_values"] = pixel_values
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        try:
            output_ids = model.generate(
                **batch, max_new_tokens=32, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            input_len = batch["input_ids"].shape[1]
            response = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
            return response
        except Exception as e:
            return f"[GEN_ERROR] {str(e)[:100]}"


def load_qwen3vl(model_path, adapter_path):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    print("  [Qwen3-VL] Loading processor ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("  [Qwen3-VL] Loading base model ...")
    torch.cuda.empty_cache()
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    if adapter_path and Path(adapter_path).exists():
        print(f"  [Qwen3-VL] Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    load_time = time.time() - t0
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  [Qwen3-VL] Loaded in {load_time:.1f}s, GPU: {mem:.1f} GB")
    return model, processor, None


def qwen3vl_infer(model, processor, _, image_path, question, device):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": question},
    ]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        response = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        return response
    except Exception as e:
        return f"[GEN_ERROR] {str(e)[:100]}"


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


def main():
    ap = argparse.ArgumentParser(description="Unified SFT evaluation for InternVL3.5 / Qwen3-VL")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default="")
    ap.add_argument("--model-type", default="auto", choices=["auto", "internvl", "qwen3vl"])
    ap.add_argument("--vqa-dir", default="work/vqa")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--tasks", nargs="+", default=["T2", "T3"])
    ap.add_argument("--n-samples", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    model_type = args.model_type
    if model_type == "auto":
        model_type = detect_model_type(args.model_path)
    print(f"Model type: {model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Adapter: {args.adapter_path or '(none)'}")

    device = torch.device("cuda:0")
    if model_type == "internvl":
        model, tok_proc, img_proc = load_internvl(args.model_path, args.adapter_path)
        infer_fn = internvl_infer
    elif model_type == "qwen3vl":
        model, tok_proc, img_proc = load_qwen3vl(args.model_path, args.adapter_path)
        infer_fn = qwen3vl_infer
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_name = args.run_name if args.run_name else (
        "InternVL3.5-8B-SFT" if "8b" in str(args.model_path).lower() and "internvl" in model_type else
        "InternVL3.5-2B-SFT" if "2b" in str(args.model_path).lower() and "internvl" in model_type else
        "Qwen3-VL-8B-SFT" if "qwen3vl" in model_type else "VLM-SFT"
    )

    all_results = {}
    for task in args.tasks:
        test_data = load_vqa_test(args.vqa_dir, task)
        if args.n_samples > 0 and args.n_samples < len(test_data):
            random.seed(42)
            test_data = random.sample(test_data, args.n_samples)
        print(f"\n{'='*60}")
        print(f"Task {task}: {len(test_data)} test samples")
        print(f"{'='*60}")
        predictions, ground_truths, raw_responses = [], [], []
        t_start = time.time()
        for i, rec in enumerate(test_data):
            img_path = Path(args.data_root) / rec["image_path"]
            response = infer_fn(model, tok_proc, img_proc, str(img_path), rec["question"], device)
            pred, fmt_ok = parse_response(response, task)
            predictions.append(pred)
            ground_truths.append(rec["answer"])
            raw_responses.append(response)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                speed = (i + 1) / elapsed
                eta = (len(test_data) - i - 1) / speed
                print(f"  {i+1}/{len(test_data)} ({speed:.1f} it/s, ETA {eta:.0f}s)")
        metrics = compute_metrics(predictions, ground_truths, task)
        metrics["model"] = model_name
        metrics["task"] = task
        metrics["adapter"] = args.adapter_path or "(none)"
        metrics["eval_time_s"] = round(time.time() - t_start, 1)
        print(f"\n  Results ({task}):")
        print(f"    N: {metrics['n_total']}, Valid: {metrics['n_valid']}")
        print(f"    Format failure: {metrics['format_failure_rate']:.2%}")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    Macro-F1: {metrics['macro_f1']:.4f}")
        if task == "T2":
            print(f"    Miss rate: {metrics.get('miss_rate', 0):.4f}")
            print(f"    Hallucination rate: {metrics.get('hallucination_rate', 0):.4f}")
        run_id = f"{model_name}_{task}_P3_test"
        if args.n_samples > 0:
            run_id += f"_sample{args.n_samples}"
        run_dir = Path(args.out) / run_id
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

    print(f"\n{'='*60}")
    print(f"Summary: {model_name}")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Task':<5} {'BalAcc':>8} {'F1':>8} {'Miss':>8} {'Hall':>8}")
    print("-" * 70)
    for run_id, m in sorted(all_results.items()):
        miss = m.get("miss_rate", 0)
        hall = m.get("hallucination_rate", 0)
        print(f"{model_name:<30} {m['task']:<5} {m['balanced_accuracy']:>8.4f} {m['macro_f1']:>8.4f} {miss:>8.4f} {hall:>8.4f}")
    summary_path = Path(args.out) / f"sft_eval_{model_name.replace('-','_').replace('.','_')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()

