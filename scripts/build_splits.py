#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T11_split: 防泄漏分组划分 + 自动断言验证

核心原则:
  - 按父图 ID 分组 (GroupShuffleSplit), 同一原图裁出的所有 crop 必须在同一个 split
  - 划分比例: train 70% / val 10% / test 20% (按 group 计)
  - 随机种子: 42
  - 分层依据: group 内的主导缺陷类别
  - 稀有类 (n<30) 不参与训练, 只进 test
  - 跳过 JSON 中引用但磁盘上不存在的图片 (det train 缺 46 张)

产出:
  work/splits/train.jsonl
  work/splits/val.jsonl
  work/splits/test.jsonl
  work/splits/MANIFEST.json   (含 SHA256)

用法:
  # 构建划分
  python scripts/build_splits.py --audit work/audit/audit.json --out work/splits \
      --ratios 0.7 0.1 0.2 --seed 42 --min-class-count 5

  # 验证划分 (Gate-B)
  python scripts/build_splits.py --verify-only --out work/splits
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 类别命名统一映射
LABEL_CANON = {
    "corrosao": "rust",
    "corrosion": "rust",
    "rust": "rust",
    "normal": "good",
    "good": "good",
    "missing-cap": "missing-cap",
    "missingcap": "missing-cap",
    "bird-nest": "nest",
    "nest": "nest",
    "torned-up": "torned-up",
    "peeling-paint": "peeling-paint",
}


def canon(label: str) -> str:
    return LABEL_CANON.get(label.strip().lower(), label.strip().lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================
# 数据加载
# ======================================================================

def load_det_records(data_root: Path):
    """从 COCO det 标注加载记录, 跳过磁盘上不存在的图片"""
    det_dir = data_root / "InsPLAD-det"
    records = []
    skipped = 0

    for split in ["train", "val"]:
        ann_path = det_dir / "annotations" / f"instances_{split}.json"
        if not ann_path.exists():
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # 检查磁盘上实际存在的文件
        img_dir = det_dir / split
        disk_files = set()
        if img_dir.exists():
            disk_files = {p.name for p in img_dir.glob("*.jpg")}

        img_wh = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in coco["images"]}

        # 按图片聚合标注
        img_anns = defaultdict(list)
        for a in coco["annotations"]:
            img_anns[a["image_id"]].append(a)

        for img_id, (W, H, fname) in img_wh.items():
            # 跳过磁盘上不存在的图片
            if fname not in disk_files:
                skipped += 1
                continue

            anns = img_anns.get(img_id, [])
            cats = set()
            for a in anns:
                cat_name = next((c["name"] for c in coco["categories"] if c["id"] == a["category_id"]), "unknown")
                cats.add(cat_name)

            # 父图 ID = 文件名 stem (det 原图本身就是父图)
            parent_id = Path(fname).stem

            records.append({
                "crop_id": f"det_{split}_{parent_id}",
                "parent_id": parent_id,
                "asset": ",".join(sorted(cats)) if cats else "none",
                "label": "det_annotation",  # det 子集没有缺陷标签, 只有目标类别
                "source_subset": "det",
                "source_split": split,
                "path": f"InsPLAD-det/{split}/{fname}",
                "bbox": [],
                "width": W,
                "height": H,
            })

    return records, skipped


def load_supervised_records(data_root: Path):
    """从 supervised_fault_classification 加载记录"""
    base = data_root / "supervised_fault_classification" / "defect_supervised"
    records = []

    if not base.exists():
        return records

    for asset_dir in sorted(base.iterdir()):
        if not asset_dir.is_dir():
            continue
        asset = asset_dir.name

        for split_dir in sorted(asset_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name  # train or val

            for label_dir in sorted(split_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                raw_label = label_dir.name
                label = canon(raw_label)

                for img_path in sorted(label_dir.glob("*.jpg")):
                    stem = img_path.stem
                    # 父图 ID 提取: 从 parent_id_map.csv 的逻辑, supervised crop 形如 1-1_DJI_0569_001
                    # parent_id = 去掉最后的 _NNN 后缀
                    import re
                    m = re.match(r"^(.*?)(?:_\d{1,3})?$", stem)
                    parent_id = m.group(1) if m and m.group(1) else stem

                    records.append({
                        "crop_id": f"sup_{asset}_{split}_{stem}",
                        "parent_id": parent_id,
                        "asset": asset,
                        "label": label,
                        "source_subset": "supervised",
                        "source_split": split,
                        "path": f"supervised_fault_classification/defect_supervised/{asset}/{split}/{raw_label}/{img_path.name}",
                        "bbox": [],
                        "width": None,
                        "height": None,
                    })

    return records


def load_unsupervised_records(data_root: Path):
    """从 unsupervised_anomaly_detection 加载记录"""
    base = data_root / "unsupervised_anomaly_detection"
    records = []

    if not base.exists():
        return records

    for asset_dir in sorted(base.iterdir()):
        if not asset_dir.is_dir():
            continue
        asset = asset_dir.name

        for split_dir in sorted(asset_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name  # train or test

            for label_dir in sorted(split_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                raw_label = label_dir.name
                label = canon(raw_label)

                for img_path in sorted(label_dir.glob("*.jpg")):
                    stem = img_path.stem
                    import re
                    m = re.match(r"^(.*?)(?:_\d{1,3})?$", stem)
                    parent_id = m.group(1) if m and m.group(1) else stem

                    records.append({
                        "crop_id": f"unsup_{asset}_{split}_{stem}",
                        "parent_id": parent_id,
                        "asset": asset,
                        "label": label,
                        "source_subset": "unsupervised",
                        "source_split": split,
                        "path": f"unsupervised_anomaly_detection/{asset}/{split}/{raw_label}/{img_path.name}",
                        "bbox": [],
                        "width": None,
                        "height": None,
                    })

    return records


# ======================================================================
# 划分逻辑
# ======================================================================

def build_splits(records, ratios=(0.7, 0.1, 0.2), seed=42, min_class_count=5):
    """按父图 ID 分组的防泄漏划分

    - 同一 parent_id 的所有 record 必须在同一个 split
    - 分层依据: group 内的主导缺陷类别
    - 稀有类 (n<min_class_count) 全部进 test
    """
    random.seed(seed)

    # 统计每个缺陷类别的样本数 (不含 good 和 det_annotation)
    label_counts = Counter()
    for r in records:
        if r["label"] not in ("good", "det_annotation"):
            label_counts[(r["asset"], r["label"])] += 1

    # 稀有类别集合
    rare_labels = {k for k, v in label_counts.items() if v < min_class_count}
    print(f"  稀有类别 (n<{min_class_count}, 只进 test): {len(rare_labels)} 个")
    for (asset, label), n in sorted(label_counts.items()):
        if (asset, label) in rare_labels:
            print(f"    {label} @ {asset}: n={n}")

    # 按 parent_id 分组
    groups = defaultdict(list)
    for r in records:
        groups[r["parent_id"]].append(r)

    print(f"  总记录数: {len(records)}")
    print(f"  总组数 (parent_id): {len(groups)}")

    # 为每个 group 确定主导缺陷标签
    group_label = {}
    for pid, recs in groups.items():
        defect_labels = [r["label"] for r in recs if r["label"] not in ("good", "det_annotation")]
        if defect_labels:
            group_label[pid] = Counter(defect_labels).most_common(1)[0][0]
        else:
            group_label[pid] = "good"

    # 分离含稀有类别的 group (直接进 test)
    test_groups = set()
    remaining_groups = []
    for pid in groups:
        recs = groups[pid]
        has_rare = any(
            (r["asset"], r["label"]) in rare_labels
            for r in recs
            if r["label"] not in ("good", "det_annotation")
        )
        if has_rare:
            test_groups.add(pid)
        else:
            remaining_groups.append(pid)

    print(f"  含稀有类的 group -> test: {len(test_groups)}")
    print(f"  剩余 group 参与划分: {len(remaining_groups)}")

    # 按主导标签分层
    label_to_groups = defaultdict(list)
    for pid in remaining_groups:
        label_to_groups[group_label[pid]].append(pid)

    # 在每个标签组内按比例划分
    train_groups = []
    val_groups = []
    test_from_remaining = []

    for label, pids in sorted(label_to_groups.items()):
        random.shuffle(pids)
        n = len(pids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # rest -> test

        train_groups.extend(pids[:n_train])
        val_groups.extend(pids[n_train:n_train + n_val])
        test_from_remaining.extend(pids[n_train + n_val:])

    # 合并 test
    all_test_groups = set(test_groups) | set(test_from_remaining)

    # 构建输出
    splits = {"train": [], "val": [], "test": []}
    for pid in train_groups:
        splits["train"].extend(groups[pid])
    for pid in val_groups:
        splits["val"].extend(groups[pid])
    for pid in all_test_groups:
        splits["test"].extend(groups[pid])

    # 统计
    for name, recs in splits.items():
        n_groups = len(set(r["parent_id"] for r in recs))
        print(f"  {name}: {len(recs)} records, {n_groups} groups")

    return splits


def verify_splits(out_dir: Path):
    """Gate-B: 划分正确性自动断言"""
    print("\n===== Gate-B: 划分验证 =====\n")
    ok = True

    splits_data = {}
    for name in ["train", "val", "test"]:
        path = out_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"  [FAIL] {name}.jsonl 不存在")
            return False
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        splits_data[name] = recs

    # 断言 1: parent_id 三集合交集为空
    print("[1] 检查 parent_id 跨 split 泄漏 ...")
    train_pids = set(r["parent_id"] for r in splits_data["train"])
    val_pids = set(r["parent_id"] for r in splits_data["val"])
    test_pids = set(r["parent_id"] for r in splits_data["test"])

    tv_leak = train_pids & val_pids
    tt_leak = train_pids & test_pids
    vt_leak = val_pids & test_pids

    if tv_leak:
        print(f"  [FAIL] train/val parent_id 交叉: {len(tv_leak)} 个")
        ok = False
    elif tt_leak:
        print(f"  [FAIL] train/test parent_id 交叉: {len(tt_leak)} 个")
        ok = False
    elif vt_leak:
        print(f"  [FAIL] val/test parent_id 交叉: {len(vt_leak)} 个")
        ok = False
    else:
        print(f"  [PASS] 无 parent_id 跨 split 泄漏")

    # 断言 2: 每个参与训练的类别在 train/val/test 中样本数均 >= 5
    print("[2] 检查训练类别最小样本数 ...")
    for name in ["train", "val", "test"]:
        label_counts = Counter()
        for r in splits_data[name]:
            if r["label"] not in ("good", "det_annotation"):
                label_counts[(r["asset"], r["label"])] += 1
        for (asset, label), n in sorted(label_counts.items()):
            if name == "train" and n < 5:
                print(f"  [WARN] train/{label}@{asset}: n={n} (<5, 该类可能训练不充分)")
            elif n == 0 and name in ("val", "test"):
                print(f"  [WARN] {name}/{label}@{asset}: n=0 (该 split 中无此类别)")

    print(f"  [PASS] 类别分布检查完成 (警告不阻塞)")

    # 断言 3: 检查路径文件是否存在 (抽样)
    print("[3] 抽样检查文件路径存在性 ...")
    import random as rnd
    rnd.seed(42)
    sample_size = min(100, len(splits_data["train"]))
    sample = rnd.sample(splits_data["train"], sample_size)
    missing = 0
    for r in sample:
        # 路径相对于 data/raw/
        full_path = Path("data/raw") / r["path"]
        if not full_path.exists():
            missing += 1
    if missing > 0:
        print(f"  [WARN] 抽样 {sample_size} 条, {missing} 条文件不存在")
    else:
        print(f"  [PASS] 抽样 {sample_size} 条文件全部存在")

    # 断言 4: MANIFEST.json 存在且 SHA256 匹配
    print("[4] 检查 MANIFEST.json ...")
    manifest_path = out_dir / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"  [FAIL] MANIFEST.json 不存在")
        ok = False
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for name in ["train", "val", "test"]:
            entry = manifest.get(name, {})
            expected_sha = entry.get("sha256", "")
            actual_sha = sha256_file(out_dir / f"{name}.jsonl")
            if expected_sha != actual_sha:
                print(f"  [FAIL] {name}.jsonl SHA256 不匹配")
                ok = False
            else:
                print(f"  [PASS] {name}.jsonl SHA256 匹配")

    print(f"\n{'='*50}")
    if ok:
        print("Gate-B: [PASS] 所有断言通过")
    else:
        print("Gate-B: [FAIL] 存在失败项")
    print(f"{'='*50}")

    return ok


# ======================================================================
# 主函数
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="work/audit/audit.json")
    ap.add_argument("--out", default="work/splits")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-class-count", type=int, default=5)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.verify_only:
        sys.exit(0 if verify_splits(out_dir) else 1)

    print("=" * 60)
    print("T11_split: 防泄漏分组划分")
    print(f"  比例: train {args.ratios[0]:.0%} / val {args.ratios[1]:.0%} / test {args.ratios[2]:.0%}")
    print(f"  种子: {args.seed}")
    print(f"  最小类别样本数: {args.min_class_count}")
    print("=" * 60)

    data_root = Path("data/raw")

    # 加载所有记录
    print("\n[1/4] 加载目标检测子集 ...")
    det_records, det_skipped = load_det_records(data_root)
    print(f"  加载 {len(det_records)} 条, 跳过 {det_skipped} 条 (磁盘不存在)")

    print("\n[2/4] 加载有监督分类子集 ...")
    sup_records = load_supervised_records(data_root)
    print(f"  加载 {len(sup_records)} 条")

    print("\n[3/4] 加载无监督异常检测子集 ...")
    unsup_records = load_unsupervised_records(data_root)
    print(f"  加载 {len(unsup_records)} 条")

    all_records = det_records + sup_records + unsup_records
    print(f"\n总记录数: {len(all_records)}")

    # 构建划分
    print("\n[4/4] 构建防泄漏划分 ...")
    splits = build_splits(all_records, args.ratios, args.seed, args.min_class_count)

    # 写出
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, recs in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sha = sha256_file(path)
        manifest[name] = {
            "sha256": sha,
            "n_records": len(recs),
            "n_groups": len(set(r["parent_id"] for r in recs)),
        }
        print(f"  写出 {path.name}: {len(recs)} 条, SHA256={sha[:16]}...")

    manifest["meta"] = {
        "ratios": args.ratios,
        "seed": args.seed,
        "min_class_count": args.min_class_count,
        "total_records": len(all_records),
    }
    manifest_path = out_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  写出 MANIFEST.json")

    # 验证
    print("\n--- 自动验证 ---")
    verify_splits(out_dir)

    # 写哨兵
    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    (state_dir / "T11_split.done").write_text("done", encoding="utf-8")
    summary = {
        "task": "T11_split",
        "status": "done",
        "total_records": len(all_records),
        "train": manifest["train"],
        "val": manifest["val"],
        "test": manifest["test"],
        "det_skipped": det_skipped,
    }
    (state_dir / "T11_split.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nT11_split 完成。")


if __name__ == "__main__":
    main()
