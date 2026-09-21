#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_vqa.py - ElecVQA-Bench 数据构造工具

子命令:
  crops  - 从 det COCO 标注生成 ROI crop，整理 supervised/unsupervised crop 引用
  qa     - 构造 T1(定位)/T2(二分类)/T3(细分) VQA 问答对
  t4     - (预留) 调用 API 生成研判依据
  gold   - (预留) 准备 gold test set 标注包

用法:
  python scripts/build_vqa.py crops --splits work/splits --out work/crops --expand 1.0 1.5 2.0 3.0
  python scripts/build_vqa.py qa --crops work/crops/expand_2.0 --splits work/splits \
      --tasks T1 T2 T3 --hard-negative same_asset --prompt-version v1 --out work/vqa
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

LABEL_CANON = {
    "corrosao": "rust", "corrosion": "rust", "rust": "rust",
    "normal": "good", "good": "good",
    "missing-cap": "missing-cap", "missingcap": "missing-cap",
    "bird-nest": "nest", "nest": "nest",
    "torned-up": "torned-up", "peeling-paint": "peeling-paint",
}

# 资产中文名
ASSET_CN = {
    "glass-insulator": "玻璃绝缘子",
    "lightning-rod-suspension": "避雷线悬垂",
    "lightning-rod-shackle": "避雷线球头",
    "polymer-insulator": "复合绝缘子",
    "polymer-insulator-upper-shackle": "复合绝缘子上球头",
    "polymer-insulator-lower-shackle": "复合绝缘子下球头",
    "polymer-insulator-tower-shackle": "复合绝缘子塔球头",
    "glass-insulator-big-shackle": "玻璃绝缘子大球头",
    "glass-insulator-small-shackle": "玻璃绝缘子小球头",
    "glass-insulator-tower-shackle": "玻璃绝缘子塔球头",
    "vari-grip": "防震锤",
    "yoke": "联塔件",
    "yoke-suspension": "悬垂联塔件",
    "spacer": "间隔棒",
    "damper-preformed": "预绞丝防震锤",
    "damper-stockbridge": "斯托克布里奇防震锤",
    "plate": "塔号牌",
    "sphere": "球",
    "stockbridge damper": "斯托克布里奇防震锤",
    "lightning rod shackle": "避雷线球头",
    "lightning rod suspension": "避雷线悬垂",
    "polymer insulator": "复合绝缘子",
    "glass insulator": "玻璃绝缘子",
    "tower id plate": "塔号牌",
    "yoke suspension": "悬垂联塔件",
    "yoke": "联塔件",
    "spiral damper": "螺旋防震锤",
}

# 缺陷中文名
DEFECT_CN = {
    "rust": "锈蚀",
    "missing-cap": "盖帽缺失",
    "nest": "鸟巢",
    "torned-up": "撕裂",
    "peeling-paint": "油漆剥落",
}

# det COCO 类别名 -> 统一名
def canon(label):
    return LABEL_CANON.get(label.strip().lower(), label.strip().lower())


# ----------------------------------------------------------------------
# 划分加载
# ----------------------------------------------------------------------

def load_splits(splits_dir):
    """加载 train/val/test.jsonl"""
    splits = {}
    for name in ["train", "val", "test"]:
        path = Path(splits_dir) / f"{name}.jsonl"
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        splits[name] = recs
        print(f"  {name}: {len(recs)} records")
    return splits


# ----------------------------------------------------------------------
# crops 子命令
# ----------------------------------------------------------------------

def load_coco_annotations(det_dir):
    """加载 COCO 标注，返回 {image_file_name: [ann_dict, ...]}"""
    ann_by_img = {}
    for split in ["train", "val"]:
        ann_path = det_dir / "annotations" / f"instances_{split}.json"
        if not ann_path.exists():
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        img_map = {im["id"]: im for im in coco["images"]}

        for ann in coco["annotations"]:
            img = img_map.get(ann["image_id"])
            if img is None:
                continue
            fname = img["file_name"]
            if fname not in ann_by_img:
                ann_by_img[fname] = {
                    "width": img["width"],
                    "height": img["height"],
                    "annotations": []
                }
            ann_by_img[fname]["annotations"].append({
                "category_id": ann["category_id"],
                "category_name": cats.get(ann["category_id"], "unknown"),
                "bbox": ann["bbox"],  # [x, y, w, h]
                "area": ann.get("area", 0),
            })
    return ann_by_img


def crop_with_expand(img, bbox, expand_ratio, min_short_side=64):
    """按外扩比例裁剪 ROI
    
    bbox: [x, y, w, h] in original image coordinates
    expand_ratio: 宽高各乘 r
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    new_w = w * expand_ratio
    new_h = h * expand_ratio
    
    # clip 到图像边界
    img_w, img_h = img.size
    x1 = max(0, int(cx - new_w / 2))
    y1 = max(0, int(cy - new_h / 2))
    x2 = min(img_w, int(cx + new_w / 2))
    y2 = min(img_h, int(cy + new_h / 2))
    
    # 若短边 < min_short_side，按短边反推
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < min_short_side:
        pad = (min_short_side - crop_w) // 2
        x1 = max(0, x1 - pad)
        x2 = min(img_w, x1 + min_short_side)
    if crop_h < min_short_side:
        pad = (min_short_side - crop_h) // 2
        y1 = max(0, y1 - pad)
        y2 = min(img_h, y1 + min_short_side)
    
    return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def cmd_crops(args):
    """生成 ROI crop 缓存"""
    print("=" * 60)
    print("T20_build_crops: 生成 ROI crop 缓存")
    print(f"  外扩比例: {args.expand}")
    print(f"  输出目录: {args.out}")
    print("=" * 60)
    
    splits = load_splits(args.splits)
    data_root = Path("data/raw")
    det_dir = data_root / "InsPLAD-det"
    
    # 加载 COCO 标注
    print("\n[1/3] 加载 COCO 标注 ...")
    coco_anns = load_coco_annotations(det_dir)
    print(f"  {len(coco_anns)} 张图片有标注")
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    stats = {ratio: {"det_crops": 0, "sup_refs": 0, "unsup_refs": 0} for ratio in args.expand}
    
    for ratio in args.expand:
        ratio_dir = out_dir / f"expand_{ratio}"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[2/3] 外扩比例 {ratio} ...")
        
        # manifest 文件
        manifest_path = ratio_dir / "manifest.jsonl"
        manifest_file = open(manifest_path, "w", encoding="utf-8")
        
        for split_name, recs in splits.items():
            split_dir = ratio_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for i, rec in enumerate(recs):
                subset = rec["source_subset"]
                
                if subset == "det":
                    # 从完整图像裁剪每个标注对象
                    img_path = data_root / rec["path"]
                    if not img_path.exists():
                        continue
                    
                    fname = img_path.name
                    ann_info = coco_anns.get(fname)
                    if not ann_info:
                        continue
                    
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue
                    
                    for j, ann in enumerate(ann_info["annotations"]):
                        crop_id = f"det_{Path(fname).stem}_{j}"
                        crop_img, (x1, y1, x2, y2) = crop_with_expand(
                            img, ann["bbox"], ratio)
                        
                        crop_fname = f"{crop_id}.jpg"
                        crop_path = split_dir / crop_fname
                        crop_img.save(crop_path, "JPEG", quality=95)
                        stats[ratio]["det_crops"] += 1
                        
                        manifest_file.write(json.dumps({
                            "crop_id": crop_id,
                            "split": split_name,
                            "source_subset": "det",
                            "parent_id": rec["parent_id"],
                            "asset": ann["category_name"],
                            "label": "det_annotation",
                            "path": str(crop_path.relative_to(out_dir.parent)),
                            "crop_bbox": [x1, y1, x2, y2],
                            "orig_bbox": ann["bbox"],
                            "expand_ratio": ratio,
                        }, ensure_ascii=False) + "\n")
                
                elif subset in ("supervised", "unsupervised"):
                    # 已是 crop，直接引用
                    img_path = data_root / rec["path"]
                    if not img_path.exists():
                        continue
                    
                    # 创建符号链接或复制
                    link_path = split_dir / img_path.name
                    if not link_path.exists():
                        try:
                            os.symlink(img_path.resolve(), link_path)
                        except OSError:
                            # symlink 失败则跳过（文件已在原位）
                            pass
                    
                    stats[ratio]["sup_refs" if subset == "supervised" else "unsup_refs"] += 1
                    
                    manifest_file.write(json.dumps({
                        "crop_id": rec["crop_id"],
                        "split": split_name,
                        "source_subset": subset,
                        "parent_id": rec["parent_id"],
                        "asset": rec["asset"],
                        "label": rec["label"],
                        "path": rec["path"],
                        "expand_ratio": "original",
                    }, ensure_ascii=False) + "\n")
                
                if (i + 1) % 5000 == 0:
                    print(f"    {split_name}: {i+1}/{len(recs)} ...")
        
        manifest_file.close()
        
        # 统计 manifest 行数
        with open(manifest_path, "r") as f:
            n_lines = sum(1 for _ in f)
        print(f"  expand_{ratio}: {n_lines} 条 manifest 记录")
    
    print(f"\n[3/3] 统计汇总:")
    for ratio, s in stats.items():
        print(f"  expand_{ratio}: det_crops={s['det_crops']}, sup_refs={s['sup_refs']}, unsup_refs={s['unsup_refs']}")
    
    # 写哨兵
    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    summary = {"task": "T20_build_crops", "status": "done", "expand_ratios": args.expand, "stats": stats}
    (state_dir / "T20_build_crops.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (state_dir / "T20_build_crops.done").write_text("done", encoding="utf-8")
    print("\nT20_build_crops 完成。")


# ----------------------------------------------------------------------
# qa 子命令
# ----------------------------------------------------------------------

# Prompt 模板 (v1)
# prompt_version "v1" 对应此版本模板, 默认使用 P3 级别 (训练用)
PROMPT_LEVEL_MAP = {"v1": "P3"}
PROMPTS = {
    "P1": {
        "T2": "请判断图中{asset_cn}的状态。\n选项：A. 状态正常  B. 存在缺陷  C. 图像质量不足以判断\n请只输出选项字母。",
        "T3_template": "请判断图中{asset_cn}的缺陷类型。\n选项：{options}\n请只输出选项字母。",
    },
    "P2": {
        "T2": "图中所示为电力线路巡检中的{asset_cn}。\n{asset_cn}是输电线路上的重要部件。\n请判断其状态。\n选项：A. 状态正常  B. 存在缺陷  C. 图像质量不足以判断\n请只输出选项字母。",
        "T3_template": "图中所示为电力线路巡检中的{asset_cn}。\n{asset_cn}是输电线路上的重要部件。\n常见缺陷类型包括：{defect_defs}\n请判断图中{asset_cn}的缺陷类型。\n选项：{options}\n请只输出选项字母。",
    },
    "P3": {
        "T2": "图中所示为电力线路巡检中的{asset_cn}。\n{asset_cn}是输电线路上的重要部件。\n判定要点：\n- 检查表面是否有锈蚀、变色\n- 检查结构是否完整（无缺失、无破损）\n- 检查是否有异物附着（如鸟巢）\n- 若图像模糊或遮挡严重，选择「图像质量不足以判断」\n请判断其状态。\n选项：A. 状态正常  B. 存在缺陷  C. 图像质量不足以判断\n请只输出选项字母。",
        "T3_template": "图中所示为电力线路巡检中的{asset_cn}。\n{asset_cn}是输电线路上的重要部件。\n判定要点：\n- 锈蚀：表面出现红褐色或橙色腐蚀产物\n- 盖帽缺失：绝缘子顶端金属盖帽不存在\n- 鸟巢：有鸟类筑巢痕迹\n- 撕裂：部件出现物理断裂或撕裂\n- 油漆剥落：表面油漆层脱落\n- 若图像模糊或遮挡严重，选择「无法判断」\n请判断图中{asset_cn}的缺陷类型。\n选项：{options}\n请只输出选项字母。",
    },
    "P4": {
        "T2": "（P4 = P3 + ICL 参考图，训练时不使用，推理时使用）\n图中所示为电力线路巡检中的{asset_cn}。\n请判断其状态。\n选项：A. 状态正常  B. 存在缺陷  C. 图像质量不足以判断\n请只输出选项字母。",
        "T3_template": "（P4 = P3 + ICL 参考图，训练时不使用，推理时使用）\n图中所示为电力线路巡检中的{asset_cn}。\n请判断图中{asset_cn}的缺陷类型。\n选项：{options}\n请只输出选项字母。",
    },
}


def get_asset_cn(asset):
    return ASSET_CN.get(asset, asset)


def build_t1_qa(splits, crops_dir, prompt_version, out_dir):
    """T1: 部件识别与定位 (from det)"""
    print("\n[T1] 构造部件识别与定位 QA ...")
    data_root = Path("data/raw")
    det_dir = data_root / "InsPLAD-det"
    
    # 加载 COCO 标注
    coco_anns = load_coco_annotations(det_dir)
    
    qa_list = []
    for split_name in ["train", "val", "test"]:
        for rec in splits[split_name]:
            if rec["source_subset"] != "det":
                continue
            
            fname = Path(rec["path"]).name
            ann_info = coco_anns.get(fname)
            if not ann_info:
                continue
            
            # 构造标注答案
            objects = []
            for ann in ann_info["annotations"]:
                x, y, w, h = ann["bbox"]
                # 归一化
                W, H = ann_info["width"], ann_info["height"]
                objects.append({
                    "category": ann["category_name"],
                    "bbox_norm": [round(x/W, 4), round(y/H, 4),
                                  round(w/W, 4), round(h/H, 4)],
                })
            
            # P3 prompt for T1
            question = (f"请识别并定位图中所有的电力线路部件。\n"
                        f"输出 JSON 格式：[{{\"category\": \"类别名\", "
                        f"\"bbox\": [x, y, w, h]}}]，坐标为归一化值 [0,1]。\n"
                        f"已知图中可能包含的部件类型：{', '.join(sorted(set(o['category'] for o in objects)))}。")
            
            answer = json.dumps(objects, ensure_ascii=False)
            
            qa_list.append({
                "task": "T1",
                "split": split_name,
                "crop_id": rec["crop_id"],
                "parent_id": rec["parent_id"],
                "image_path": rec["path"],
                "prompt_version": prompt_version,
                "question": question,
                "answer": answer,
                "gt_objects": objects,
            })
    
    out_path = out_dir / "T1_grounding.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for qa in qa_list:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"  T1: {len(qa_list)} 条 -> {out_path}")
    return qa_list


def build_t2_qa(splits, crops_dir, prompt_version, hard_negative, out_dir):
    """T2: 缺陷判别（二分类）with hard negatives"""
    print("\n[T2] 构造缺陷判别 QA (二分类) ...")
    
    # 收集所有 supervised + unsupervised 的缺陷/正常样本
    defect_samples = defaultdict(list)  # asset -> [recs with defect]
    good_samples = defaultdict(list)    # asset -> [recs with good]
    
    for split_name in ["train", "val", "test"]:
        for rec in splits[split_name]:
            if rec["source_subset"] not in ("supervised", "unsupervised"):
                continue
            asset = rec["asset"]
            rec["split"] = split_name
            if rec["label"] == "good":
                good_samples[asset].append(rec)
            else:
                defect_samples[asset].append(rec)
    
    qa_list = []
    rng = random.Random(42)
    
    for asset in sorted(set(list(defect_samples.keys()) + list(good_samples.keys()))):
        asset_cn = get_asset_cn(asset)
        
        # 缺陷样本 -> answer = "B"
        for rec in defect_samples[asset]:
            prompt = PROMPTS[PROMPT_LEVEL_MAP.get(prompt_version, "P3")]["T2"].format(asset_cn=asset_cn)
            qa_list.append({
                "task": "T2",
                "split": rec["split"],
                "crop_id": rec["crop_id"],
                "parent_id": rec["parent_id"],
                "image_path": rec["path"],
                "asset": asset,
                "label": rec["label"],
                "prompt_version": prompt_version,
                "question": prompt,
                "answer": "B",  # 存在缺陷
                "gt_label": "defect",
            })
        
        # 正常样本 -> answer = "A"，按 1:1 采样
        good_pool = good_samples.get(asset, [])
        n_defect = len(defect_samples.get(asset, []))
        n_good = min(n_defect, len(good_pool)) if n_defect > 0 else len(good_pool)
        
        if hard_negative == "same_asset":
            sampled_good = rng.sample(good_pool, n_good) if good_pool else []
        elif hard_negative == "same_parent":
            # 从同一父图的 good 中采样
            defect_pids = set(r["parent_id"] for r in defect_samples.get(asset, []))
            same_parent_good = [r for r in good_pool if r["parent_id"] in defect_pids]
            other_good = [r for r in good_pool if r["parent_id"] not in defect_pids]
            sampled_good = []
            for r in same_parent_good:
                if len(sampled_good) < n_good:
                    sampled_good.append(r)
            if len(sampled_good) < n_good:
                sampled_good.extend(rng.sample(other_good, min(n_good - len(sampled_good), len(other_good))))
        else:  # random
            sampled_good = rng.sample(good_pool, n_good) if good_pool else []
        
        for rec in sampled_good:
            prompt = PROMPTS[PROMPT_LEVEL_MAP.get(prompt_version, "P3")]["T2"].format(asset_cn=asset_cn)
            qa_list.append({
                "task": "T2",
                "split": rec["split"],
                "crop_id": rec["crop_id"],
                "parent_id": rec["parent_id"],
                "image_path": rec["path"],
                "asset": asset,
                "label": rec["label"],
                "prompt_version": prompt_version,
                "question": prompt,
                "answer": "A",  # 状态正常
                "gt_label": "good",
                "hard_negative": hard_negative,
            })
    
    out_path = out_dir / "T2_binary.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for qa in qa_list:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    # 统计
    by_split = Counter(qa["split"] for qa in qa_list)
    by_answer = Counter(qa["answer"] for qa in qa_list)
    print(f"  T2: {len(qa_list)} 条 -> {out_path}")
    print(f"    按划分: {dict(by_split)}")
    print(f"    按答案: A(正常)={by_answer.get('A',0)}, B(缺陷)={by_answer.get('B',0)}")
    return qa_list


def build_t3_qa(splits, crops_dir, prompt_version, out_dir):
    """T3: 缺陷类型细分 (多分类)"""
    print("\n[T3] 构造缺陷类型细分 QA (多分类) ...")
    
    # 收集所有缺陷类型
    defect_types_by_asset = defaultdict(set)
    all_samples = defaultdict(list)  # (asset, label) -> [recs]
    
    for split_name in ["train", "val", "test"]:
        for rec in splits[split_name]:
            if rec["source_subset"] not in ("supervised", "unsupervised"):
                continue
            rec["split"] = split_name
            asset = rec["asset"]
            label = rec["label"]
            all_samples[(asset, label)].append(rec)
            if label != "good":
                defect_types_by_asset[asset].add(label)
    
    qa_list = []
    
    for asset in sorted(defect_types_by_asset.keys()):
        asset_cn = get_asset_cn(asset)
        defect_types = sorted(defect_types_by_asset[asset])
        
        # 构造选项
        options = ["A. 状态正常"]
        for i, dt in enumerate(defect_types):
            dt_cn = DEFECT_CN.get(dt, dt)
            options.append(f"{chr(66+i)}. {dt_cn}({dt})")
        options.append(f"{chr(66+len(defect_types))}. 无法判断")
        options_str = "  ".join(options)
        
        # 为该资产的所有样本生成 QA
        for label in ["good"] + defect_types:
            recs = all_samples.get((asset, label), [])
            for rec in recs:
                if label == "good":
                    answer = "A"
                else:
                    answer = chr(66 + defect_types.index(label))
                
                prompt = PROMPTS[PROMPT_LEVEL_MAP.get(prompt_version, "P3")]["T3_template"].format(
                    asset_cn=asset_cn, options=options_str,
                    defect_defs="  ".join(f"{DEFECT_CN.get(dt, dt)}({dt})" for dt in defect_types))
                
                qa_list.append({
                    "task": "T3",
                    "split": rec["split"],
                    "crop_id": rec["crop_id"],
                    "parent_id": rec["parent_id"],
                    "image_path": rec["path"],
                    "asset": asset,
                    "label": label,
                    "prompt_version": prompt_version,
                    "question": prompt,
                    "answer": answer,
                    "gt_label": label,
                    "options": options,
                })
    
    out_path = out_dir / "T3_multiclass.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for qa in qa_list:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    # 统计
    by_split = Counter(qa["split"] for qa in qa_list)
    by_asset = Counter(qa["asset"] for qa in qa_list)
    print(f"  T3: {len(qa_list)} 条 -> {out_path}")
    print(f"    按划分: {dict(by_split)}")
    print(f"    按资产: {dict(by_asset)}")
    return qa_list


def cmd_qa(args):
    """构造 VQA 问答对"""
    print("=" * 60)
    print("T21_build_vqa: 构造 VQA 数据")
    print(f"  任务: {args.tasks}")
    print(f"  难负样本: {args.hard_negative}")
    print(f"  Prompt 版本: {args.prompt_version}")
    print("=" * 60)
    
    splits = load_splits(args.splits)
    crops_dir = Path(args.crops)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_qa = {}
    
    if "T1" in args.tasks:
        all_qa["T1"] = build_t1_qa(splits, crops_dir, args.prompt_version, out_dir)
    if "T2" in args.tasks:
        all_qa["T2"] = build_t2_qa(splits, crops_dir, args.prompt_version, args.hard_negative, out_dir)
    if "T3" in args.tasks:
        all_qa["T3"] = build_t3_qa(splits, crops_dir, args.prompt_version, out_dir)
    
    # 汇总
    print("\n" + "=" * 60)
    print("T21_build_vqa 汇总:")
    total = 0
    for task, qa_list in all_qa.items():
        by_split = Counter(qa["split"] for qa in qa_list)
        print(f"  {task}: {len(qa_list)} 条  {dict(by_split)}")
        total += len(qa_list)
    print(f"  总计: {total} 条")
    print("=" * 60)
    
    # 写汇总文件
    summary = {
        "task": "T21_build_vqa",
        "status": "done",
        "prompt_version": args.prompt_version,
        "hard_negative": args.hard_negative,
        "tasks": {},
    }
    for task, qa_list in all_qa.items():
        summary["tasks"][task] = {
            "total": len(qa_list),
            "by_split": dict(Counter(qa["split"] for qa in qa_list)),
        }
    
    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    (state_dir / "T21_build_vqa.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (state_dir / "T21_build_vqa.done").write_text("done", encoding="utf-8")
    
    print("\nT21_build_vqa 完成。")


# ----------------------------------------------------------------------
# t4 子命令 (预留)
# ----------------------------------------------------------------------

def cmd_t4(args):
    """T4: 研判依据生成 (调用 API)"""
    print("T4 研判依据生成 - 待实现")
    print(f"  Provider: {args.provider}")
    print(f"  N: {args.n}")
    print("  使用 AGENTS.md 中的 Qwen3.5-122B-A10B-FP8 API 生成推理链")


# ----------------------------------------------------------------------
# gold 子命令 (预留)
# ----------------------------------------------------------------------

def cmd_gold(args):
    """Gold test set 准备"""
    print("Gold test set 准备 - 待实现")
    print(f"  N: {args.n}")


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ElecVQA-Bench 数据构造工具")
    sub = ap.add_subparsers(dest="command", required=True)
    
    # crops 子命令
    p_crops = sub.add_parser("crops", help="生成 ROI crop 缓存")
    p_crops.add_argument("--splits", default="work/splits")
    p_crops.add_argument("--out", default="work/crops")
    p_crops.add_argument("--expand", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    p_crops.set_defaults(func=cmd_crops)
    
    # qa 子命令
    p_qa = sub.add_parser("qa", help="构造 VQA 问答对")
    p_qa.add_argument("--crops", default="work/crops/expand_2.0")
    p_qa.add_argument("--splits", default="work/splits")
    p_qa.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    p_qa.add_argument("--hard-negative", default="same_asset",
                       choices=["random", "same_asset", "same_parent"])
    p_qa.add_argument("--prompt-version", default="v1")
    p_qa.add_argument("--out", default="work/vqa")
    p_qa.set_defaults(func=cmd_qa)
    
    # t4 子命令
    p_t4 = sub.add_parser("t4", help="研判依据生成")
    p_t4.add_argument("--crops", default="work/crops/expand_2.0")
    p_t4.add_argument("--n", type=int, default=5000)
    p_t4.add_argument("--provider", default="qwen3.5-122b")
    p_t4.add_argument("--out", default="work/vqa")
    p_t4.set_defaults(func=cmd_t4)
    
    # gold 子命令
    p_gold = sub.add_parser("gold", help="Gold test set 准备")
    p_gold.add_argument("--n", type=int, default=600)
    p_gold.add_argument("--out", default="work/vqa/gold")
    p_gold.set_defaults(func=cmd_gold)
    
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
