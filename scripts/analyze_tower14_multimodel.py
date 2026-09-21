#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OPTION_MAP = {
    'A. 状态正常  B. 锈蚀(rust)  C. 无法判断': {'A':'good','B':'rust','C':'unknown'},
    'A. 状态正常  B. 盖帽缺失(missing-cap)  C. 无法判断': {'A':'good','B':'missing-cap','C':'unknown'},
    'A. 状态正常  B. 撕裂(torned-up)  C. 无法判断': {'A':'good','B':'torned-up','C':'unknown'},
    'A. 状态正常  B. corrosão(corrosão)  C. 锈蚀(rust)  D. 无法判断': {'A':'good','B':'corrosão','C':'rust','D':'unknown'},
    'A. 状态正常  B. 鸟巢(nest)  C. 锈蚀(rust)  D. 无法判断': {'A':'good','B':'nest','C':'rust','D':'unknown'},
    'A. 状态正常  B. 鸟巢(nest)  C. 无法判断': {'A':'good','B':'nest','C':'unknown'},
    'A. 状态正常  B. 油漆剥落(peeling-paint)  C. 无法判断': {'A':'good','B':'peeling-paint','C':'unknown'},
}

def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_cnn(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return [d['labels'][x] for x in d['preds']]

def parse_qwen(test, pred):
    gt, out = [], []
    for t, p in zip(test, pred):
        text = t['messages'][0]['content'][1]['text']
        m = re.search(r'选项：(.+)', text)
        mapping = OPTION_MAP[m.group(1).strip()]
        gt.append(t.get('gt_label', mapping.get(t['messages'][1]['content'])))
        out.append(mapping.get(p['response'], 'unknown'))
    return gt, out

def recalls(gt, pred, indices, classes):
    result = {}
    for c in classes:
        idx = [i for i in indices if gt[i] == c]
        if not idx:
            return None
        result[c] = sum(1 for i in idx if pred[i] == c) / len(idx)
    return result

def comparison(gt, a, b, clusters, classes, n_bootstrap, n_permutation, seed):
    cluster_ids = sorted(clusters)
    indices = list(range(len(gt)))
    ra = recalls(gt, a, indices, classes)
    rb = recalls(gt, b, indices, classes)
    macro_a = float(np.mean([ra[c] for c in classes]))
    macro_b = float(np.mean([rb[c] for c in classes]))
    observed = macro_a - macro_b

    rng = np.random.default_rng(seed)
    macro_samples = []
    valid = invalid = 0
    for _ in range(n_bootstrap):
        chosen = rng.choice(len(cluster_ids), size=len(cluster_ids), replace=True)
        sample_indices = []
        for j in chosen:
            sample_indices.extend(clusters[cluster_ids[j]])
        sa = recalls(gt, a, sample_indices, classes)
        sb = recalls(gt, b, sample_indices, classes)
        if sa is None or sb is None:
            invalid += 1
            continue
        valid += 1
        macro_samples.append(float(np.mean([sa[c] - sb[c] for c in classes])))

    perm_diffs = []
    a_list, b_list = list(a), list(b)
    for _ in range(n_permutation):
        aa, bb = list(a_list), list(b_list)
        for cluster_indices in clusters.values():
            mask = rng.random(len(cluster_indices)) < 0.5
            for flag, i in zip(mask, cluster_indices):
                if flag:
                    aa[i], bb[i] = bb[i], aa[i]
        pa = recalls(gt, aa, indices, classes)
        pb = recalls(gt, bb, indices, classes)
        if pa is not None and pb is not None:
            perm_diffs.append(float(np.mean([pa[c] - pb[c] for c in classes])))
    perm_diffs = np.asarray(perm_diffs)
    p = 2 * min(float(np.mean(perm_diffs >= observed)), float(np.mean(perm_diffs <= observed)))
    return {
        'classes': classes,
        'model_a': {'per_class_recall': ra, 'macro_recall': macro_a},
        'model_b': {'per_class_recall': rb, 'macro_recall': macro_b},
        'difference': observed,
        'cluster_bootstrap': {
            'ci95': [float(np.percentile(macro_samples, 2.5)), float(np.percentile(macro_samples, 97.5))],
            'valid_replicates': valid,
            'invalid_replicates': invalid,
            'seed': seed,
        },
        'cluster_permutation_raw_p': float(min(1.0, p)),
        'n_permutation_valid': int(len(perm_diffs)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--qwen-pred', required=True)
    ap.add_argument('--resnet-pred', required=True)
    ap.add_argument('--swin-pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-bootstrap', type=int, default=10000)
    ap.add_argument('--n-permutation', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    test = load_jsonl(args.test)
    qpred = load_jsonl(args.qwen_pred)
    gt, qp = parse_qwen(test, qpred)
    rp = load_cnn(args.resnet_pred)
    sp = load_cnn(args.swin_pred)
    if not (len(gt) == len(qp) == len(rp) == len(sp)):
        raise RuntimeError('Prediction length mismatch')

    clusters = defaultdict(list)
    for i, rec in enumerate(test):
        clusters[rec['_tower']].append(i)
    all_classes = sorted(set(gt))
    common_six = [c for c in all_classes if c != 'peeling-paint']

    per_class = {
        'qwen3_vl_8b': recalls(gt, qp, range(len(gt)), all_classes),
        'resnet50_448': recalls(gt, rp, range(len(gt)), all_classes),
        'swin_t_448': recalls(gt, sp, range(len(gt)), all_classes),
    }
    result = {
        'n': len(gt),
        'cluster_unit': 'tower',
        'n_clusters': len(clusters),
        'classes': all_classes,
        'common_six_classes': common_six,
        'per_class_recall': per_class,
        'macro_recall_full_seven': {
            name: float(np.mean([per_class[name][c] for c in all_classes]))
            for name in per_class
        },
        'macro_recall_common_six': {
            name: float(np.mean([per_class[name][c] for c in common_six]))
            for name in per_class
        },
        'comparisons': {
            'qwen_vs_resnet_common_six': comparison(gt, qp, rp, clusters, common_six, args.n_bootstrap, args.n_permutation, args.seed),
            'qwen_vs_swin_common_six': comparison(gt, qp, sp, clusters, common_six, args.n_bootstrap, args.n_permutation, args.seed + 1),
            'resnet_vs_swin_common_six': comparison(gt, rp, sp, clusters, common_six, args.n_bootstrap, args.n_permutation, args.seed + 2),
            'qwen_vs_swin_full_seven': comparison(gt, qp, sp, clusters, all_classes, args.n_bootstrap, args.n_permutation, args.seed + 3),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

