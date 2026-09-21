from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "data" / "benchmark" / "tasks"
SPLIT_DIR = ROOT / "data" / "benchmark" / "splits"
CROP_MANIFEST = ROOT / "data" / "crops" / "expand_2.0_manifest.jsonl"
METADATA_DIR = ROOT / "data" / "metadata"
CONFIG_DIR = ROOT / "configs"

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

LABEL_INFO = {
    "good": ("normal / no defect", "状态正常"),
    "normal": ("binary normal label", "状态正常"),
    "defective": ("binary defect label", "存在缺陷"),
    "rust": ("localized oxidation on metal fittings", "锈蚀"),
    "corrosão": ("generalized corrosion", "腐蚀（广义）"),
    "missing-cap": ("missing cap", "盖帽缺失"),
    "nest": ("bird nest", "鸟巢"),
    "torned-up": ("torn or physically damaged component", "撕裂"),
    "peeling-paint": ("peeling paint", "油漆剥落"),
    "cannot_judge": ("explicit abstention option", "无法判断"),
    "det_annotation": ("source detection annotation only; not a T3 defect label", "检测标注"),
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = {
        "T1": list(read_jsonl(TASK_DIR / "T1_grounding.jsonl")),
        "T2": list(read_jsonl(TASK_DIR / "T2_binary.jsonl")),
        "T3": list(read_jsonl(TASK_DIR / "T3_multiclass.jsonl")),
    }

    # Prompt release.
    write_json(CONFIG_DIR / "prompts.json", {
        "prompt_version": "v1",
        "default_level": "P3",
        "prompt_level_map": {"v1": "P3"},
        "templates": PROMPTS,
        "notes": [
            "P4 requires in-context reference images and is not used for training in the released pipeline.",
            "All prompts include an explicit abstention/cannot-judge option.",
        ],
    })

    # Asset metadata and option lists.
    asset_meta = {}
    asset_cn_pattern = re.compile(r"图中所示为电力线路巡检中的(.+?)。")
    for row in tasks["T3"]:
        asset = row["asset"]
        if asset not in asset_meta:
            match = asset_cn_pattern.search(row["question"])
            asset_meta[asset] = {
                "asset_cn": match.group(1) if match else asset,
                "t3_options": row["options"],
                "answer_by_label": {},
                "labels": set(),
                "n_records": 0,
            }
        asset_meta[asset]["answer_by_label"][row["gt_label"]] = row["answer"]
        asset_meta[asset]["labels"].add(row["gt_label"])
        asset_meta[asset]["n_records"] += 1
    for meta in asset_meta.values():
        meta["labels"] = sorted(meta["labels"])
    write_json(CONFIG_DIR / "asset_metadata.json", asset_meta)

    # Label crosswalk.
    observed_labels = set()
    for task_rows in tasks.values():
        for row in task_rows:
            observed_labels.update([row.get("gt_label"), row.get("label")])
    observed_labels.discard(None)
    observed_labels.update(LABEL_INFO)
    with (CONFIG_DIR / "label_crosswalk.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark_label", "meaning_en", "meaning_zh", "used_in"])
        writer.writeheader()
        for label in sorted(observed_labels):
            info = LABEL_INFO.get(label, ("source label; see study documentation", label))
            used = []
            if label in {r.get("gt_label") for r in tasks["T2"]}:
                used.append("T2")
            if label in {r.get("gt_label") for r in tasks["T3"]}:
                used.append("T3")
            if label == "det_annotation":
                used.append("source/T1")
            writer.writerow({
                "benchmark_label": label,
                "meaning_en": info[0],
                "meaning_zh": info[1],
                "used_in": ",".join(used) or "source metadata",
            })

    # Per-class and per-asset support.
    support_rows = []
    for task, rows in tasks.items():
        counts = Counter()
        for row in rows:
            asset = row.get("asset", "all")
            label = row.get("gt_label") or row.get("label") or "grounding"
            counts[(row["split"], asset, label)] += 1
        for (split, asset, label), count in sorted(counts.items()):
            support_rows.append({"task": task, "split": split, "asset": asset, "label": label, "count": count})
    with (METADATA_DIR / "per_class_support.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "split", "asset", "label", "count"])
        writer.writeheader()
        writer.writerows(support_rows)

    # Task summary.
    task_summary = {}
    for task, rows in tasks.items():
        task_summary[task] = {
            "total": len(rows),
            "by_split": dict(Counter(r["split"] for r in rows)),
            "by_asset": dict(Counter(r.get("asset", "all") for r in rows)),
            "by_label": dict(Counter(r.get("gt_label") or r.get("label") or "grounding" for r in rows)),
        }
    write_json(METADATA_DIR / "task_summary.json", task_summary)

    # Construction audit from actual candidate and final T3 records.
    final_ids = {r["crop_id"] for r in tasks["T3"]}
    candidates = [r for r in read_jsonl(CROP_MANIFEST) if r.get("source_subset") in {"supervised", "unsupervised"}]
    audit_rows = []
    reason_counts = Counter()
    included_counts = Counter()
    for row in candidates:
        included = row["crop_id"] in final_ids
        reason = "included" if included else "asset_has_no_observed_defect_class"
        reason_counts[reason] += 1
        included_counts[row.get("source_subset", "unknown")] += int(included)
        audit_rows.append({
            "crop_id": row.get("crop_id", ""),
            "parent_id": row.get("parent_id", ""),
            "source_subset": row.get("source_subset", ""),
            "source_split": row.get("source_split", row.get("split", "")),
            "assigned_split": row.get("split", ""),
            "asset": row.get("asset", ""),
            "label": row.get("label", ""),
            "candidate_path": row.get("path", ""),
            "included_in_t3": included,
            "exclusion_reason": reason,
        })
    with (METADATA_DIR / "construction_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    write_json(METADATA_DIR / "construction_audit_summary.json", {
        "candidate_scope": "supervised_fault_classification + unsupervised_anomaly_detection records in the released split pipeline",
        "n_candidates": len(candidates),
        "n_included_in_t3": sum(r["included_in_t3"] for r in audit_rows),
        "n_excluded_from_t3": sum(not r["included_in_t3"] for r in audit_rows),
        "reason_counts": dict(reason_counts),
        "included_by_source_subset": dict(included_counts),
        "reproducibility_note": "This audit is generated from the released crop manifest and final T3 task file. The current study states 38,350 candidates and 1,980 exclusions; the released files contain 39,145 candidates and 2,775 exclusions under the actual build script. This discrepancy should be reconciled before submission.",
    })

    # Verify base split manifest.
    manifest = json.loads((SPLIT_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    verification = {}
    for split in ["train", "val", "test"]:
        path = SPLIT_DIR / f"{split}.jsonl"
        actual = sha256(path)
        expected = manifest[split]["sha256"]
        verification[split] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
            "n_records": manifest[split]["n_records"],
        }
    write_json(METADATA_DIR / "split_manifest_verification.json", {
        "all_match": all(v["match"] for v in verification.values()),
        "splits": verification,
    })

    print(f"Wrote release metadata to {METADATA_DIR}")
    print(f"T3 construction audit: {len(candidates)} candidates, {sum(not r['included_in_t3'] for r in audit_rows)} excluded")


if __name__ == "__main__":
    main()

