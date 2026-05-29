#!/usr/bin/env bash
# Quick template train + preview + numeric validation (run per plugin before full train).
# Usage: bash scripts/train_plugin_template.sh B
set -euo pipefail
cd "$(dirname "$0")/.."

PLUGIN="${1:?Usage: $0 <A|B|C|D>}"
PLUGIN=$(echo "$PLUGIN" | tr '[:lower:]' '[:upper:]')

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,6,7}"

# --- template hyperparams (edit here) ---
MAX_SAMPLES=512
EPOCHS=3
BATCH_SIZE=64
SIZE=128
LR=5e-4
SAVE_DIR=./checkpoints/rwkv_plugin_template
RUN_NAME=template
OUT_PREVIEW=./outputs/plugin_template_preview

case "$PLUGIN" in
  A) W_ANCHOR=0.3 ;;
  B) W_ANCHOR=0.5 ;;
  C) W_ANCHOR=0.6 ;;
  D) W_ANCHOR=0.6 ;;
  *) echo "Unknown plugin $PLUGIN"; exit 1 ;;
esac

LAST_EPOCH=$((EPOCHS - 1))
LOG="logs/template_${PLUGIN}.log"

{
  echo "=== $(date) template train Plugin ${PLUGIN} ==="
  python clean_train_plugin.py \
    --plugin "$PLUGIN" \
    --max_samples "$MAX_SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --size "$SIZE" \
    --lr "$LR" \
    --w_anchor "$W_ANCHOR" \
    --save_dir "$SAVE_DIR" \
    --run_name "$RUN_NAME"

  python scripts/preview_plugins.py \
    --plugin_dir "$SAVE_DIR" \
    --run_name "$RUN_NAME" \
    --only "$PLUGIN" \
    --epoch "$LAST_EPOCH" \
    --size "$SIZE" \
    --out_dir "${OUT_PREVIEW}/${PLUGIN}"

  python scripts/validate_plugin_template.py \
    --plugin "$PLUGIN" \
    --epoch "$LAST_EPOCH" \
    --size "$SIZE" \
    --ckpt_dir "$SAVE_DIR" \
    --run_name "$RUN_NAME"

  echo "=== $(date) template done ${PLUGIN} ==="
  echo "Preview: ${OUT_PREVIEW}/${PLUGIN}/"
  echo "If PASS above, run: bash scripts/train_plugin_full.sh ${PLUGIN}"
} 2>&1 | tee -a "$LOG"
