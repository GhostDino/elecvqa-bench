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

def recalls(gt, pred, indices, classes):
    out = {}
    for c in classes:
        idx = [i for i in indices if gt[i] == c]
        if not idx:
            return None
        out[c] = sum(1 for i in idx if pred[i] == c) / len(idx)
    return out

def macro_diff(gt, a, b, indices, classes):
    ra = recalls(gt, a, indices, classes)
    rb = recalls(gt, b, indices, classes)
    if ra is None or rb is None:
        return None, None, None
    diff = sum(ra[c] - rb[c] for c in classes) / len(classes)
    return ra, rb, diff

def cluster_bootstrap(gt, a, b, clusters, classes, n, seed):
    rng = np.random.default_rng(seed)
    cluster_ids = sorted(clusters)
    macro_samples = []
    contrib_samples = {c: [] for c in classes}
    valid = invalid = 0
    for _ in range(n):
        chosen = rng.choice(len(cluster_ids), size=len(cluster_ids), replace=True)
        indices = []
        for j in chosen:
            indices.extend(clusters[cluster_ids[j]])
        ra, rb, diff = macro_diff(gt, a, b, indices, classes)
        if ra is None or rb is None:
            invalid += 1
            continue
        valid += 1
        macro_samples.append(diff)
        for c in classes:
            contrib_samples[c].append((ra[c] - rb[c]) / len(classes))
    macro_samples = np.asarray(macro_samples)
    result = {
        'n_requested': n,
        'valid_replicates': valid,
        'invalid_replicates': invalid,
        'seed': seed,
        'macro_difference_ci95': [float(np.percentile(macro_samples, 2.5)), float(np.percentile(macro_samples, 97.5))],
        'per_class_contribution_ci95': {
            c: [float(np.percentile(contrib_samples[c], 2.5)), float(np.percentile(contrib_samples[c], 97.5))]
            for c in classes
        },
    }
    return result, macro_samples

def permutation_p(gt, a, b, clusters, classes, n, seed, mode):
    rng = np.random.default_rng(seed)
    cluster_ids = sorted(clusters)
    indices = list(range(len(gt)))
    _, _, obs = macro_diff(gt, a, b, indices, classes)
    extreme = 0
    diffs = []
    a0, b0 = list(a), list(b)
    for _ in range(n):
        aa, bb = list(a0), list(b0)
        if mode == 'within':
            for cid in cluster_ids:
                idx = clusters[cid]
                mask = rng.random(len(idx)) < 0.5
                for flag, i in zip(mask, idx):
                    if flag:
                        aa[i], bb[i] = bb[i], aa[i]
        elif mode == 'cluster':
            for cid in cluster_ids:
                idx = clusters[cid]
                if rng.random() < 0.5:
                    for i in idx:
                        aa[i], bb[i] = bb[i], aa[i]
        else:
            raise ValueError(mode)
        _, _, d = macro_diff(gt, aa, bb, indices, classes)
        diffs.append(d)
        if abs(d) >= abs(obs):
            extreme += 1
    p = (extreme + 1) / (n + 1)
    return {
        'observed': obs,
        'p_value': float(p),
        'n_permutations': n,
        'seed': seed,
        'mode': mode,
        'permutation_mean': float(np.mean(diffs)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--qwen-pred', required=True)
    ap.add_argument('--cnn-pred', required=True)
    ap.add_argument('--cluster-key', required=True, choices=['_tower', '_parent_id'])
    ap.add_argument('--classes', nargs='+', default=COMMON_SIX)
    ap.add_argument('--n-bootstrap', type=int, default=10000)
    ap.add_argument('--n-permutation', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    test = load_jsonl(args.test)
    qpred = load_jsonl(args.qwen_pred)
    gt, qp = parse_qwen(test, qpred)
    cp = load_cnn(args.cnn_pred)
    if not (len(gt) == len(qp) == len(cp)):
        raise RuntimeError('Prediction length mismatch')

    clusters = defaultdict(list)
    for i, rec in enumerate(test):
        clusters[rec[args.cluster_key]].append(i)
    cluster_ids = sorted(clusters)
    indices = list(range(len(gt)))
    classes = args.classes

    ra, rb, obs = macro_diff(gt, qp, cp, indices, classes)
    bootstrap, macro_samples = cluster_bootstrap(gt, qp, cp, clusters, classes, args.n_bootstrap, args.seed)
    # Bootstrap p-value: two-sided tail probability around the observed bootstrap distribution.
    left = float(np.mean(macro_samples <= obs))
    right = float(np.mean(macro_samples >= obs))
    bootstrap_p = 2 * min(left, right)
    bootstrap_p = min(1.0, bootstrap_p)
    within = permutation_p(gt, qp, cp, clusters, classes, args.n_permutation, args.seed, 'within')
    cluster = permutation_p(gt, qp, cp, clusters, classes, args.n_permutation, args.seed, 'cluster')

    result = {
        'n': len(gt),
        'n_clusters': len(cluster_ids),
        'cluster_key': args.cluster_key,
        'classes': classes,
        'qwen_recall': ra,
        'cnn_recall': rb,
        'macro_difference': obs,
        'per_class_contribution': {c: (ra[c] - rb[c]) / len(classes) for c in classes},
        'bootstrap': bootstrap,
        'bootstrap_p_value': bootstrap_p,
        'permutation_within_cluster': within,
        'permutation_cluster_level': cluster,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
