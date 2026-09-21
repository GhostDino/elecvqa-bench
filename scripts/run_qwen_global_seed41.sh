#!/bin/bash
set -euo pipefail
cd /workspace/elecvqa
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py3.11
PYTHON=/root/miniconda3/envs/py3.11/bin/python
SWIFT=/root/miniconda3/envs/py3.11/lib/python3.11/site-packages/swift/cli/sft.py
OUT=runs/multiseed/Q_8B_s41
EVAL_OUT=runs/multiseed/Q_8B_s41_eval
mkdir -p runs logs
if [[ -f "$EVAL_OUT/sft_eval_Q_8B_s41.json" ]]; then
  echo 'Qwen global seed-41 evaluation already complete.'
  exit 0
fi
if [[ ! -d "$OUT" ]] || [[ -z "$(find "$OUT" -type d -name 'checkpoint-*' | sort -V | tail -1)" ]]; then
  echo '=== Qwen3-VL-8B global seed-41 training ==='
  "$PYTHON" "$SWIFT" \
    --model /root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master \
    --dataset work/sft_full/train.jsonl \
    --val_dataset work/sft_full/val.jsonl \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --report_to none \
    --max_length 2048 \
    --truncation_strategy delete \
    --output_dir "$OUT" \
    --seed 41 2>&1 | tee logs/qwen_global_seed41_train.log
fi
CKPT=$(find "$OUT" -type d -name 'checkpoint-*' | sort -V | tail -1)
if [[ -z "$CKPT" ]]; then
  echo 'ERROR: no Qwen seed-41 checkpoint found.' >&2
  exit 1
fi
echo "=== Qwen3-VL-8B global seed-41 evaluation using $CKPT ==="
"$PYTHON" scripts/eval_sft_unified.py \
  --model-path /root/data-tmp/modelscope_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master \
  --adapter-path "$CKPT" \
  --model-type qwen3vl \
  --vqa-dir work/vqa \
  --data-root data/raw \
  --tasks T2 T3 \
  --out "$EVAL_OUT" \
  --run-name Q_8B_s41 2>&1 | tee logs/qwen_global_seed41_eval.log
touch runs/qwen_global_seed41.done
echo 'QWEN_GLOBAL_SEED41_DONE'
