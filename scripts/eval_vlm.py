#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_vlm.py - VLM 评测 runner

支持:
  - 多模型评测 (从 models.yaml 加载配置)
  - T2 (二分类) / T3 (多分类) 任务
  - P1-P4 四档 prompt
  - 图像 base64 编码发送
  - 格式失败率统计
  - Balanced Accuracy / Macro-F1 指标
  - smoke test 模式 (--smoke N 先跑 N 条)

用法:
  # smoke test (20 条)
  python scripts/eval_vlm.py --models-config configs/models.yaml --group opensource \
      --tasks T2 T3 --prompts P3 --smoke 20 --out runs

  # 全量评测
  python scripts/eval_vlm.py --models-config configs/models.yaml --group opensource \
      --tasks T2 T3 --prompts P1 P2 P3 P4 --out runs
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def encode_image_b64(path, max_size=768):
    """读取图片并编码为 base64，自动缩放到 max_size 以内"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_models_config(config_path, group="opensource"):
    """加载模型配置"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get(group, [])


def load_vqa_data(vqa_dir, task, split="test"):
    """加载 VQA 数据"""
    task_files = {
        "T1": "T1_grounding.jsonl",
        "T2": "T2_binary.jsonl",
        "T3": "T3_multiclass.jsonl",
    }
    path = Path(vqa_dir) / task_files.get(task, f"{task}.jsonl")
    if not path.exists():
        print(f"[WARN] {path} not found")
        return []
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("split") == split:
                    recs.append(rec)
    return recs


# ----------------------------------------------------------------------
# API 调用
# ----------------------------------------------------------------------

def call_vlm_api(model_cfg, image_b64, question, max_retries=2):
    """调用 VLM API (OpenAI 兼容格式)"""
    import urllib.request
    import urllib.error

    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"
    
    content = []
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}",
                "detail": "low",
            }
        })
    content.append({"type": "text", "text": question})

    payload = {
        "model": model_cfg["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": model_cfg.get("max_tokens", 512),
        "temperature": model_cfg.get("temperature", 0.0),
    }
    
    # 添加 chat_template_kwargs (如果配置中存在)
    ctk = model_cfg.get("chat_template_kwargs")
    if ctk:
        payload["chat_template_kwargs"] = ctk

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_cfg.get('api_key', '')}",
    }

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
            else:
                return f"[API_ERROR] {str(e)}"


# ----------------------------------------------------------------------
# 响应解析
# ----------------------------------------------------------------------

def parse_response(response, task):
    """解析模型响应，返回 (predicted_label, format_ok)"""
    response = response.strip()
    
    if task in ("T2", "T3"):
        # 提取选项字母 A/B/C/D/E
        for ch in response:
            if ch in "ABCDE":
                return ch, True
        # 尝试匹配中文文本
        if "正常" in response and "缺陷" not in response:
            return "A", True
        if "缺陷" in response or "锈蚀" in response or "锈" in response:
            return "B", True
        return None, False
    
    return response, True


# ----------------------------------------------------------------------
# 指标计算
# ----------------------------------------------------------------------

def compute_metrics(predictions, ground_truths, task):
    """计算评测指标"""
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    
    valid_preds = []
    valid_gts = []
    n_format_fail = 0
    
    for pred, gt in zip(predictions, ground_truths):
        if pred is None:
            n_format_fail += 1
            continue
        valid_preds.append(pred)
        valid_gts.append(gt)
    
    n_total = len(predictions)
    n_valid = len(valid_preds)
    format_failure_rate = n_format_fail / n_total if n_total > 0 else 0
    
    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": 0,
            "format_failure_rate": 1.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
        }
    
    bal_acc = balanced_accuracy_score(valid_gts, valid_preds)
    macro_f1 = f1_score(valid_gts, valid_preds, average="macro", zero_division=0)
    
    # 混淆矩阵
    labels = sorted(set(valid_gts + valid_preds))
    cm = confusion_matrix(valid_gts, valid_preds, labels=labels)
    
    # 漏检率 (缺陷样本被判为正常)
    if task == "T2":
        # B=缺陷, A=正常
        defect_indices = [i for i, gt in enumerate(valid_gts) if gt == "B"]
        if defect_indices:
            miss_rate = sum(1 for i in defect_indices if valid_preds[i] == "A") / len(defect_indices)
        else:
            miss_rate = 0.0
        # 幻觉率 (正常样本被判为缺陷)
        good_indices = [i for i, gt in enumerate(valid_gts) if gt == "A"]
        if good_indices:
            hallucination_rate = sum(1 for i in good_indices if valid_preds[i] == "B") / len(good_indices)
        else:
            hallucination_rate = 0.0
    else:
        miss_rate = 0.0
        hallucination_rate = 0.0
    
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "format_failure_rate": round(format_failure_rate, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "miss_rate": round(miss_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "pred_distribution": dict(Counter(valid_preds)),
        "gt_distribution": dict(Counter(valid_gts)),
    }


# ----------------------------------------------------------------------
# 评测主逻辑
# ----------------------------------------------------------------------

def evaluate_model(model_cfg, vqa_dir, tasks, prompts, split, smoke_n, out_dir, data_root):
    """评测单个模型"""
    model_id = model_cfg["id"]
    print(f"\n{'='*60}")
    print(f"评测模型: {model_id}")
    print(f"  URL: {model_cfg['base_url']}")
    print(f"{'='*60}")
    
    results = {}
    
    for task in tasks:
        for prompt_level in prompts:
            run_id = f"{model_id}_{task}_{prompt_level}_{split}"
            if smoke_n:
                run_id += f"_smoke{smoke_n}"
            
            print(f"\n--- {run_id} ---")
            
            # 加载数据
            vqa_data = load_vqa_data(vqa_dir, task, split)
            if smoke_n and smoke_n < len(vqa_data):
                random.seed(42)
                vqa_data = random.sample(vqa_data, smoke_n)
            
            print(f"  样本数: {len(vqa_data)}")
            
            predictions = []
            ground_truths = []
            raw_responses = []
            
            for i, rec in enumerate(vqa_data):
                # 加载图片
                img_path = data_root / rec["image_path"]
                if not img_path.exists():
                    predictions.append(None)
                    ground_truths.append(rec["answer"])
                    raw_responses.append("[IMAGE_NOT_FOUND]")
                    continue
                
                # 编码图片
                try:
                    img_b64 = encode_image_b64(img_path)
                except Exception as e:
                    predictions.append(None)
                    ground_truths.append(rec["answer"])
                    raw_responses.append(f"[ENCODE_ERROR] {str(e)}")
                    continue
                
                # 构造 prompt (使用已有的 question 字段)
                question = rec["question"]
                
                # 调用 API
                response = call_vlm_api(model_cfg, img_b64, question)
                raw_responses.append(response)
                
                # 解析响应
                pred, fmt_ok = parse_response(response, task)
                predictions.append(pred)
                ground_truths.append(rec["answer"])
                
                if (i + 1) % 20 == 0:
                    print(f"    {i+1}/{len(vqa_data)} ...")
            
            # 计算指标
            metrics = compute_metrics(predictions, ground_truths, task)
            metrics["model"] = model_id
            metrics["task"] = task
            metrics["prompt"] = prompt_level
            metrics["split"] = split
            metrics["smoke"] = smoke_n
            
            print(f"\n  结果:")
            print(f"    样本数: {metrics['n_total']}")
            print(f"    格式失败率: {metrics['format_failure_rate']:.2%}")
            print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"    Macro-F1: {metrics['macro_f1']:.4f}")
            if task == "T2":
                print(f"    漏检率: {metrics.get('miss_rate', 0):.4f}")
                print(f"    幻觉率: {metrics.get('hallucination_rate', 0):.4f}")
            
            # 保存结果
            run_dir = out_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            # 保存详细预测
            with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
                for i, rec in enumerate(vqa_data):
                    f.write(json.dumps({
                        "crop_id": rec["crop_id"],
                        "gt": rec["answer"],
                        "pred": predictions[i],
                        "raw_response": raw_responses[i],
                        "asset": rec.get("asset", ""),
                        "label": rec.get("label", ""),
                    }, ensure_ascii=False) + "\n")
            
            results[run_id] = metrics
    
    return results


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------

def main():
    
    ap = argparse.ArgumentParser(description="VLM 评测 runner")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--group", default="opensource", choices=["opensource", "api"])
    ap.add_argument("--tasks", nargs="+", default=["T2", "T3"])
    ap.add_argument("--prompts", nargs="+", default=["P3"])
    ap.add_argument("--vqa-dir", default="work/vqa")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--smoke", type=int, default=0, help="先跑 N 条 (0=全量)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--full-metrics", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0)
    
    args = ap.parse_args()
    
    print("=" * 60)
    print("VLM 评测")
    print(f"  模型组: {args.group}")
    print(f"  任务: {args.tasks}")
    print(f"  Prompts: {args.prompts}")
    print(f"  划分: {args.split}")
    if args.smoke:
        print(f"  Smoke: {args.smoke} 条")
    print("=" * 60)
    
    # 加载模型配置
    models = load_models_config(args.models_config, args.group)
    print(f"\n模型数: {len(models)}")
    for m in models:
        print(f"  - {m['id']} @ {m['base_url']}")
    
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_cfg in models:
        results = evaluate_model(
            model_cfg, args.vqa_dir, args.tasks, args.prompts,
            args.split, args.smoke, out_dir, data_root
        )
        all_results.update(results)
    
    # 汇总表
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"{'Model':<30} {'Task':<5} {'Prompt':<5} {'N':>5} {'BalAcc':>8} {'F1':>8} {'FmtFail':>8}")
    print("-" * 75)
    for run_id, m in sorted(all_results.items()):
        print(f"{m['model']:<30} {m['task']:<5} {m['prompt']:<5} {m['n_total']:>5} "
              f"{m['balanced_accuracy']:>8.4f} {m['macro_f1']:>8.4f} {m['format_failure_rate']:>7.2%}")
    
    # Gate-C 判定
    print("\n--- Gate-C 判定 ---")
    for run_id, m in sorted(all_results.items()):
        if m["task"] == "T2":
            balacc = m["balanced_accuracy"]
            fmtfail = m["format_failure_rate"]
            if fmtfail > 0.10:
                print(f"  [WARN] {m['model']} T2 {m['prompt']}: 格式失败率 {fmtfail:.2%} > 10%, 需修 prompt")
            elif balacc >= 0.85:
                print(f"  [WARN] {m['model']} T2 {m['prompt']}: BalAcc {balacc:.4f} >= 85%, 任务无区分度")
            elif balacc <= 0.60:
                print(f"  [WARN] {m['model']} T2 {m['prompt']}: BalAcc {balacc:.4f} <= 60%, 排查解析问题")
            else:
                print(f"  [OK] {m['model']} T2 {m['prompt']}: BalAcc {balacc:.4f}, 理想范围 (60-85%)")
    
    # 保存汇总
    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
