#!/usr/bin/env python3
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

COMMON_SIX = ['corrosão', 'good', 'missing-cap', 'nest', 'rust', 'torned-up']
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

def parse_qwen(test, pred):
    gt, out = [], []
    for t, p in zip(test, pred):
        txt = t['messages'][0]['content'][1]['text']
        m = re.search(r'选项：(.+)', txt)
        mp = OPTION_MAP[m.group(1).strip()]
        gt.append(t.get('gt_label', mp.get(t['messages'][1]['content'])))
        out.append(mp.get(p['response'], 'unknown'))
    return gt, out

def load_cnn(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return [d['labels'][x] for x in d['preds']]

def recalls(gt, a, b, indices, classes):
    ra, rb = {}, {}
    for c in classes:
        idx = [i for i in indices if gt[i] == c]
        if not idx:
            return None, None
        ra[c] = sum(1 for i in idx if a[i] == c) / len(idx)
        rb[c] = sum(1 for i in idx if b[i] == c) / len(idx)
    return ra, rb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--qwen-pred', required=True)
    ap.add_argument('--cnn-pred', required=True)
    ap.add_argument('--cluster-key', required=True, choices=['_tower', '_parent_id'])
    ap.add_argument('--classes', nargs='+', default=COMMON_SIX)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    test = load_jsonl(args.test)
    qp = load_cnn(args.qwen_pred)
    cp = load_cnn(args.cnn_pred)
    gt = []
    for rec in test:
        txt = rec['messages'][0]['content'][1]['text']
        m = re.search(r'选项：(.+)', txt)
        mp = OPTION_MAP[m.group(1).strip()]
        gt.append(rec.get('gt_label', mp.get(rec['messages'][1]['content'])))
    if not (len(gt) == len(qp) == len(cp)):
        raise RuntimeError('Prediction length mismatch')

    clusters = defaultdict(list)
    for i, rec in enumerate(test):
        clusters[rec[args.cluster_key]].append(i)
    cluster_ids = sorted(clusters)
    indices = list(range(len(gt)))
    classes = args.classes

    obs_ra, obs_rb = recalls(gt, qp, cp, indices, classes)
    obs_macro = sum(obs_ra[c] - obs_rb[c] for c in classes) / len(classes)
    obs_contrib = {c: (obs_ra[c] - obs_rb[c]) / len(classes) for c in classes}

    rng = np.random.default_rng(args.seed)
    null_macro = []
    null_contrib = {c: [] for c in classes}
    for _ in range(args.n):
        aa, bb = list(qp), list(cp)
        for cid in cluster_ids:
            if rng.random() < 0.5:
                for i in clusters[cid]:
                    aa[i], bb[i] = bb[i], aa[i]
        ra, rb = recalls(gt, aa, bb, indices, classes)
        if ra is None or rb is None:
            raise RuntimeError('Sign-flip unexpectedly removed class support')
        null_macro.append(sum(ra[c] - rb[c] for c in classes) / len(classes))
        for c in classes:
            null_contrib[c].append((ra[c] - rb[c]) / len(classes))

    null_macro = np.asarray(null_macro)
    extreme = int(np.sum(np.abs(null_macro) >= abs(obs_macro)))
    p = (extreme + 1) / (args.n + 1)

    shifted_macro = null_macro + obs_macro
    result = {
        'method': 'cluster-level sign-flip randomization; shifted null distribution for CI',
        'n': len(gt),
        'n_clusters': len(cluster_ids),
        'cluster_key': args.cluster_key,
        'classes': classes,
        'n_randomizations': args.n,
        'seed': args.seed,
        'qwen_recall': obs_ra,
        'cnn_recall': obs_rb,
        'macro_difference': obs_macro,
        'macro_difference_ci95': [float(np.percentile(shifted_macro, 2.5)), float(np.percentile(shifted_macro, 97.5))],
        'p_value': float(p),
        'per_class_contribution': obs_contrib,
        'per_class_contribution_ci95': {
            c: [float(np.percentile(np.asarray(null_contrib[c]) + obs_contrib[c], 2.5)),
                float(np.percentile(np.asarray(null_contrib[c]) + obs_contrib[c], 97.5))]
            for c in classes
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

