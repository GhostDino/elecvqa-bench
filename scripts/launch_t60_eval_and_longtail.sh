#!/bin/bash
# launch_t60_eval_and_longtail.sh
# Run after T60 (InternVL3.5-2B) training completes
# 1. Evaluate T60 final checkpoint on T2/T3 (200 samples)
# 2. Start C3 VLM long-tail experiment

set -e
cd /workspace/elecvqa
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py3.11

PY=/root/miniconda3/envs/py3.11/bin/python
SWIFT=/root/miniconda3/envs/py3.11/bin/swift

echo "========================================"
echo "Step 1: Evaluate T60 (InternVL3.5-2B-SFT)"
echo "========================================"

# Find T60 final checkpoint
T60_CKPT=$(ls -d runs/sft_internvl3_2b/v0-*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "T60 checkpoint: $T60_CKPT"

if [ -z "$T60_CKPT" ]; then
    echo "ERROR: No T60 checkpoint found!"
    exit 1
fi

# Prepare test data (if not already done)
for task in T2 T3; do
    TEST_FILE="work/vqa/_test_${task}_200.jsonl"
    if [ ! -f "$TEST_FILE" ]; then
        echo "Preparing $task test data..."
        $PY -c "
import json, random
from pathlib import Path
task_files = {'T2': 'T2_binary.jsonl', 'T3': 'T3_multiclass.jsonl'}
recs = []
with open(f'work/vqa/{task_files[\"$task\"]}') as f:
    for line in f:
        r = json.loads(line.strip())
        if r.get('split') == 'test':
            recs.append(r)
random.seed(42)
sample = random.sample(recs, min(200, len(recs)))
with open('$TEST_FILE', 'w') as f:
    for r in sample:
        img_path = str(Path('data/raw') / r['image_path'])
        rec = {'messages': [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': img_path},
                {'type': 'text', 'text': r['question']}
            ]},
            {'role': 'assistant', 'content': r['answer']}
        ]}
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
print(f'Wrote {len(sample)} samples')
"
    fi
done

# Run evaluation
for task in T2 T3; do
    echo "Evaluating $task..."
    $SWIFT infer \
        --model /root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master \
        --adapters "$T60_CKPT" \
        --val_dataset "work/vqa/_test_${task}_200.jsonl" \
        --infer_backend pt \
        --max_new_tokens 32 \
        --torch_dtype bfloat16 \
        --device_map auto \
        --result_path "runs/_eval_T60_${task}_200.jsonl" 2>&1 | tail -5
    echo "$task done"
done

# Compute metrics
$PY << 'PYMETRICS'
import json
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score

for task in ["T2", "T3"]:
    results = []
    with open(f"runs/_eval_T60_{task}_200.jsonl") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    preds_raw = [r.get("response", "").strip() for r in results]
    gts = [r.get("labels", "").strip() for r in results]
    
    parsed = []
    n_fail = 0
    for p in preds_raw:
        found = None
        for ch in p:
            if ch in "ABCDE":
                found = ch
                break
        if found is None:
            n_fail += 1
        parsed.append(found)
    
    valid_p = [p for p in parsed if p is not None]
    valid_g = [g for p, g in zip(parsed, gts) if p is not None]
    
    bal = balanced_accuracy_score(valid_g, valid_p)
    f1 = f1_score(valid_g, valid_p, average="macro", zero_division=0)
    
    print(f"T60 {task}: BalAcc={bal:.4f}, F1={f1:.4f}, N={len(valid_p)}, Fail={n_fail}")
    
    if task == "T2":
        miss = sum(1 for p, g in zip(valid_p, valid_g) if g == "B" and p == "A")
        halluc = sum(1 for p, g in zip(valid_p, valid_g) if g == "A" and p == "B")
        n_d = sum(1 for g in valid_g if g == "B")
        n_n = sum(1 for g in valid_g if g == "A")
        print(f"  Miss={miss}/{n_d}={miss/n_d:.4f}, Halluc={halluc}/{n_n}={halluc/n_n:.4f}")
PYMETRICS

echo ""
echo "========================================"
echo "Step 2: Start C3 VLM Long-tail Experiment"
echo "========================================"

# Run long-tail experiment with InternVL3.5-2B
# Start with small n values first, use reduced steps for speed
$PY scripts/longtail_vlm.py \
    --model internvl2b \
    --n-values 4 8 16 32 64 128 \
    --seeds 42 \
    --max-steps 500 \
    --n-test 200 \
    --out runs/longtail_vlm 2>&1 | tee logs/longtail_vlm_internvl2b.log

echo ""
echo "========================================"
echo "C3 VLM Long-tail experiment complete!"
echo "========================================"
echo "True" > state/longtail_vlm_internvl2b.rc
