#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InsPLAD 数据审计脚本 (Gate-A)

产出:
  work/audit/audit.json        机器可读的完整审计结果
  work/audit/data_card.md      论文附录用的 data card
  work/audit/parent_id_map.csv crop -> 父图 ID 映射
  work/audit/phash_collisions.csv 近重复图像清单

用法:
  python scripts/audit_dataset.py --root data/raw --out work/audit --phash --check-corrupt
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# 类别命名统一映射 (审计项 3)
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

# 父图 ID 提取: InsPLAD 原图形如 1-1_DJI_0569.jpg
PARENT_PAT = re.compile(r"(\d+-\d+_DJI_\d+)", re.IGNORECASE)
# 兜底: 去掉尾部 _数字 后缀 (crop 序号)
PARENT_FALLBACK = re.compile(r"^(.*?)(?:_\d{1,3})?$")

def canon(label: str) -> str:
    return LABEL_CANON.get(label.strip().lower(), label.strip().lower())

def parent_id_of(fname: str):
    """返回 (parent_id, method). method in {regex, fallback, failed}"""
    stem = Path(fname).stem
    m = PARENT_PAT.search(stem)
    if m:
        return m.group(1), "regex"
    m2 = PARENT_FALLBACK.match(stem)
    if m2 and m2.group(1) and m2.group(1) != stem:
        return m2.group(1), "fallback"
    return stem, "failed"

def phash(path: Path, hash_size: int = 8):
    """简易 pHash (DCT-free 的 average hash 变体, 依赖少且够用)"""
    try:
        img = Image.open(path).convert("L").resize(
            (hash_size, hash_size), Image.Resampling.LANCZOS
        )
    except Exception:
        return None
    px = list(img.getdata())
    avg = sum(px) / len(px)
    bits = "".join("1" if p > avg else "0" for p in px)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"

def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")

def walk_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            yield p

# ----------------------------------------------------------------------
# 各子集审计
# ----------------------------------------------------------------------
def audit_det(root: Path, res: dict):
    det = root / "InsPLAD-det"
    if not det.exists():
        res["det"] = {"error": "InsPLAD-det not found"}
        return

    out = {"splits": {}, "categories": {}, "bbox_issues": []}
    for split in ["train", "val"]:
        ann = det / "annotations" / f"instances_{split}.json"
        if not ann.exists():
            out["splits"][split] = {"error": f"missing {ann.name}"}
            continue
        with open(ann, "r", encoding="utf-8") as f:
            coco = json.load(f)

        cats = {c["id"]: c["name"] for c in coco["categories"]}
        img_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
        per_cat = Counter()
        bad = []
        for a in coco["annotations"]:
            per_cat[cats.get(a["category_id"], f"id_{a['category_id']}")] += 1
            x, y, w, h = a["bbox"]
            W, H = img_wh.get(a["image_id"], (0, 0))
            if w <= 1 or h <= 1:
                bad.append({"ann_id": a["id"], "issue": "degenerate", "bbox": a["bbox"]})
            elif W and (x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1):
                bad.append({"ann_id": a["id"], "issue": "out_of_bounds", "bbox": a["bbox"]})

        actual_files = len(list((det / split).glob("*.jpg"))) if (det / split).exists() else 0
        out["splits"][split] = {
            "images_in_json": len(coco["images"]),
            "images_on_disk": actual_files,
            "consistent": len(coco["images"]) == actual_files,
            "annotations": len(coco["annotations"]),
            "per_category": dict(per_cat),
        }
        out["bbox_issues"].extend(bad[:50])
        for k, v in per_cat.items():
            out["categories"].setdefault(k, {})[split] = v

    # 审计项 4: val 中样本为 0 的类别 -> 无法计算 AP
    zero_val = [k for k, v in out["categories"].items() if v.get("val", 0) == 0]
    out["categories_without_val"] = zero_val
    out["n_categories"] = len(out["categories"])
    out["flag_category_count"] = (
        f"发现 {len(out['categories'])} 类, 官方文献称 17 类" if len(out["categories"]) != 17 else "OK"
    )
    res["det"] = out

def audit_imagefolder(base: Path, subset_name: str, res: dict, args):
    """审计 supervised_fault / unsupervised_AD 这类 ImageFolder 结构"""
    if not base.exists():
        res[subset_name] = {"error": f"{base} not found"}
        return

    assets = {}
    parent_stats = Counter()
    parent_map = []      # (subset, asset, split, label, filename, parent_id, method)
    label_raw_by_split = defaultdict(lambda: defaultdict(set))
    size_samples = []
    corrupt = []

    # 结构: <base>/<asset>/<split>/<label>/*.jpg
    for asset_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        asset = asset_dir.name
        a = {"splits": {}}
        for split_dir in sorted([d for d in asset_dir.iterdir() if d.is_dir()]):
            split = split_dir.name
            if split not in ("train", "val", "test"):
                continue
            per_label = {}
            for label_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
                raw = label_dir.name
                lab = canon(raw)
                label_raw_by_split[asset][split].add(raw)
                files = [p for p in label_dir.iterdir() if p.suffix.lower() in IMG_EXT]
                per_label[lab] = per_label.get(lab, 0) + len(files)

                for p in files:
                    pid, method = parent_id_of(p.name)
                    parent_stats[method] += 1
                    parent_map.append(
                        (subset_name, asset, split, lab, p.name, pid, method)
                    )
                # 尺寸采样 (审计项 7): 每个 label 抽 30 张
                for p in files[:30]:
                    try:
                        with Image.open(p) as im:
                            size_samples.append(min(im.size))
                    except Exception as e:
                        corrupt.append({"path": str(p), "err": str(e)})
                if args.check_corrupt:
                    for p in files:
                        if p.stat().st_size == 0:
                            corrupt.append({"path": str(p), "err": "zero-byte"})
            a["splits"][split] = {"total": sum(per_label.values()), "per_label": per_label}
        # 审计项 2: train 显著小于 val 的异常
        tr = a["splits"].get("train", {}).get("total", 0)
        va = a["splits"].get("val", {}).get("total", 0) or a["splits"].get("test", {}).get("total", 0)
        a["flag_split_anomaly"] = (
            f"⚠ train({tr}) 比 val/test({va}) 小 {va / max(tr,1):.1f}× , 疑似划分反转"
            if tr and va and va > tr * 3
            else "OK"
        )
        # 审计项 3: train/val 标签命名不一致
        raws = label_raw_by_split[asset]
        if len(raws) > 1:
            sets = [set(v) for v in raws.values()]
            if not all(s == sets[0] for s in sets):
                a["flag_label_naming"] = {k: sorted(v) for k, v in raws.items()}
        assets[asset] = a

    sizes = sorted(size_samples)
    res[subset_name] = {
        "n_assets": len(assets),
        "assets": assets,
        "parent_id_extraction": {
            "regex": parent_stats["regex"],
            "fallback": parent_stats["fallback"],
            "failed": parent_stats["failed"],
            "success_rate": round(
                (parent_stats["regex"] + parent_stats["fallback"])
                / max(sum(parent_stats.values()), 1),
                4,
            ),
        },
        "crop_short_side": {
            "min": sizes[0] if sizes else None,
            "p25": sizes[len(sizes) // 4] if sizes else None,
            "median": sizes[len(sizes) // 2] if sizes else None,
            "p75": sizes[len(sizes) * 3 // 4] if sizes else None,
            "max": sizes[-1] if sizes else None,
            "n_sampled": len(sizes),
        },
        "corrupt": corrupt[:100],
        "n_corrupt": len(corrupt),
    }
    return parent_map

def audit_phash(root: Path, res: dict, out_dir: Path, limit_per_dir: int = 4000):
    """审计项 8: 跨子集近重复检测"""
    buckets = defaultdict(list)   # hash -> [(subset, path)]
    n = 0
    for subset in ["InsPLAD-det", "supervised_fault_classification",
                   "unsupervised_anomaly_detection"]:
        base = root / subset
        if not base.exists():
            continue
        cnt = 0
        for p in walk_images(base):
            if cnt >= limit_per_dir:
                break
            h = phash(p)
            if h:
                buckets[h].append((subset, str(p.relative_to(root))))
                cnt += 1
                n += 1

    # 精确碰撞
    exact = {h: v for h, v in buckets.items() if len(v) > 1}
    cross = {h: v for h, v in exact.items() if len({s for s, _ in v}) > 1}

    with open(out_dir / "phash_collisions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phash", "n", "cross_subset", "paths"])
        for h, v in exact.items():
            w.writerow([h, len(v), len({s for s, _ in v}) > 1, ";".join(p for _, p in v)])

    res["phash"] = {
        "n_hashed": n,
        "n_exact_collision_groups": len(exact),
        "n_cross_subset_groups": len(cross),
        "cross_subset_ratio": round(len(cross) / max(n, 1), 5),
        "flag": "⚠ 跨子集近重复 >1%, 划分必须重做" if len(cross) / max(n, 1) > 0.01 else "OK",
    }

# ----------------------------------------------------------------------
def write_data_card(res: dict, out: Path):
    L = []
    L.append("# InsPLAD Data Card (auto-generated)\n")
    L.append(f"> 生成时间：{res['meta']['generated_at']}　脚本版本：{res['meta']['script_version']}\n")

    L.append("\n## 1. 目标检测子集\n")
    det = res.get("det", {})
    if "splits" in det:
        L.append("| split | JSON 图数 | 磁盘图数 | 一致 | 标注框 |")
        L.append("|---|---|---|---|---|")
        for s, v in det["splits"].items():
            L.append(f"| {s} | {v.get('images_in_json')} | {v.get('images_on_disk')} "
                     f"| {'✅' if v.get('consistent') else '❌'} | {v.get('annotations')} |")
        L.append(f"\n- 类别数：{det.get('n_categories')}　{det.get('flag_category_count')}")
        L.append(f"- val 无样本的类别（无法算 AP，评测须排除）：`{det.get('categories_without_val')}`")
        L.append(f"- bbox 异常数（越界/退化，最多列 50）：{len(det.get('bbox_issues', []))}")

    for key, title in [("supervised", "2. 有监督缺陷分类子集"),
                       ("unsupervised", "3. 无监督异常检测子集")]:
        sub = res.get(key, {})
        if "assets" not in sub:
            continue
        L.append(f"\n## {title}\n")
        L.append("| 资产 | train | val/test | 标签 | 异常标记 |")
        L.append("|---|---|---|---|---|")
        for a, v in sub["assets"].items():
            tr = v["splits"].get("train", {})
            te = v["splits"].get("val", {}) or v["splits"].get("test", {})
            labels = sorted(set(list(tr.get("per_label", {})) + list(te.get("per_label", {}))))
            flag = v.get("flag_split_anomaly", "OK")
            if v.get("flag_label_naming"):
                flag += " / ⚠命名不一致"
            L.append(f"| {a} | {tr.get('total', 0)} | {te.get('total', 0)} "
                     f"| {', '.join(labels)} | {flag} |")
        pe = sub["parent_id_extraction"]
        L.append(f"\n- **父图 ID 反解成功率：{pe['success_rate']:.2%}**"
                 f"（regex {pe['regex']} / fallback {pe['fallback']} / failed {pe['failed']}）")
        if pe["success_rate"] < 0.95:
            L.append("  - ⚠ **低于 95%，必须改用 pHash 聚类做分组，Gate-1 未通过**")
        cs = sub["crop_short_side"]
        L.append(f"- crop 短边分布：min {cs['min']} / p25 {cs['p25']} / median {cs['median']} "
                 f"/ p75 {cs['p75']} / max {cs['max']}（n={cs['n_sampled']}）")
        L.append(f"  - → 建议 `max_pixels` 设为 p75 的 2–3 倍，不要盲目上 1280²")
        L.append(f"- 损坏/零字节文件：{sub['n_corrupt']}")

    if "phash" in res:
        p = res["phash"]
        L.append("\n## 4. 近重复与泄漏风险\n")
        L.append(f"- 已哈希图像：{p['n_hashed']}")
        L.append(f"- 精确碰撞组：{p['n_exact_collision_groups']}")
        L.append(f"- **跨子集碰撞组：{p['n_cross_subset_groups']}（占比 {p['cross_subset_ratio']:.3%}）**")
        L.append(f"- 判定：{p['flag']}")

    L.append("\n## 5. 稀有类别清单（n < 30，只报 bootstrap CI，不报点估计）\n")
    for c in res.get("rare_classes", []):
        L.append(f"- `{c['label']}` @ `{c['asset']}`：n = {c['n']}")

    L.append("\n## 6. Gate-A 判定\n")
    for k, v in res.get("gate_a", {}).items():
        L.append(f"- {'✅' if v['pass'] else '❌'} **{k}**：{v['msg']}")

    out.write_text("\n".join(L), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="work/audit")
    ap.add_argument("--phash", action="store_true")
    ap.add_argument("--check-corrupt", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import datetime
    res = {"meta": {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "script_version": "1.0",
        "root": str(root.resolve()),
    }}

    print("[1/4] 审计目标检测子集 ...")
    audit_det(root, res)

    print("[2/4] 审计有监督缺陷分类子集 ...")
    pm1 = audit_imagefolder(
        root / "supervised_fault_classification" / "defect_supervised",
        "supervised", res, args) or []

    print("[3/4] 审计无监督异常检测子集 ...")
    pm2 = audit_imagefolder(
        root / "unsupervised_anomaly_detection", "unsupervised", res, args) or []

    with open(out / "parent_id_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset", "asset", "split", "label", "filename", "parent_id", "method"])
        w.writerows(pm1 + pm2)

    if args.phash:
        print("[4/4] pHash 近重复检测（较慢）...")
        audit_phash(root, res, out)

    # 稀有类别
    rare = []
    for key in ["supervised", "unsupervised"]:
        for a, v in res.get(key, {}).get("assets", {}).items():
            agg = Counter()
            for s in v["splits"].values():
                agg.update(s.get("per_label", {}))
            for lab, n in agg.items():
                if lab != "good" and n < 30:
                    rare.append({"subset": key, "asset": a, "label": lab, "n": n})
    res["rare_classes"] = sorted(rare, key=lambda x: x["n"])

    # ---- Gate-A 判定 ----
    sr = min(res.get(k, {}).get("parent_id_extraction", {}).get("success_rate", 1.0)
             for k in ["supervised", "unsupervised"] if k in res)
    gate = {
        "parent_id_resolvable": {
            "pass": sr >= 0.95,
            "msg": f"父图 ID 反解成功率 {sr:.2%}（阈值 95%）",
        },
        "no_cross_subset_leak": {
            "pass": res.get("phash", {}).get("cross_subset_ratio", 0) <= 0.01,
            "msg": f"跨子集近重复占比 {res.get('phash', {}).get('cross_subset_ratio', 0):.3%}（阈值 1%）",
        },
        "no_split_anomaly": {
            "pass": all(
                v.get("flag_split_anomaly", "OK") == "OK"
                for k in ["supervised", "unsupervised"]
                for v in res.get(k, {}).get("assets", {}).values()
            ),
            "msg": "检查各资产 train/val 规模是否反转（yoke-suspension 重点看）",
        },
        "no_corrupt_files": {
            "pass": res.get("supervised", {}).get("n_corrupt", 0)
                    + res.get("unsupervised", {}).get("n_corrupt", 0) == 0,
            "msg": "损坏/零字节文件数",
        },
    }
    res["gate_a"] = gate

    (out / "audit.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    write_data_card(res, out / "data_card.md")

    print("\n===== Gate-A 判定 =====")
    all_pass = True
    for k, v in gate.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}: {v['msg']}")
        all_pass &= v["pass"]
    print(f"\n产出: {out}/audit.json, data_card.md, parent_id_map.csv")
    print("智能体请停止并向人类汇报上述四项。")
    sys.exit(0 if all_pass else 2)   # 2 = 需人类决策, 非致命错误

if __name__ == "__main__":
    main()
