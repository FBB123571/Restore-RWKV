#!/usr/bin/env bash
# Full-data training — only after template PASS.
# Usage: bash scripts/train_plugin_full.sh B
set -euo pipefail
cd "$(dirname "$0")/.."

PLUGIN="${1:?Usage: $0 <A|B|C|D>}"
PLUGIN=$(echo "$PLUGIN" | tr '[:lower:]' '[:upper:]')

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,6,7}"

# Require template validation passed
if ! python scripts/validate_plugin_template.py \
  --plugin "$PLUGIN" --epoch 2 --ckpt_dir ./checkpoints/rwkv_plugin_template --run_name template; then
  echo "Template validation failed. Run: bash scripts/train_plugin_template.sh ${PLUGIN}"
  exit 1
fi

case "$PLUGIN" in
  A) W_ANCHOR=0.3 ;;
  B) W_ANCHOR=0.5 ;;
  C) W_ANCHOR=0.6 ;;
  D) W_ANCHOR=0.6 ;;
esac

EPOCHS=15
SAVE_DIR=./checkpoints/rwkv_plugin_v2
LOG="logs/full_${PLUGIN}.log"

{
  echo "=== $(date) FULL train Plugin ${PLUGIN} ==="
  python clean_train_plugin.py \
    --plugin "$PLUGIN" \
    --epochs "$EPOCHS" \
    --batch_size 128 \
    --size 128 \
    --lr 5e-4 \
    --w_anchor "$W_ANCHOR" \
    --save_dir "$SAVE_DIR" \
    --run_name full

  python scripts/preview_plugins.py \
    --plugin_dir "$SAVE_DIR" \
    --run_name full \
    --only "$PLUGIN" \
    --epoch $((EPOCHS - 1)) \
    --size 128 \
    --out_dir "./outputs/preview_plugins_v2/${PLUGIN}"

  if ! python scripts/validate_plugin_template.py \
    --plugin "$PLUGIN" \
    --epoch $((EPOCHS - 1)) \
    --ckpt_dir "$SAVE_DIR" \
    --run_name full \
    --max_diff_v1 0.15; then
    echo "WARN: post-train validation failed for ${PLUGIN}; checkpoint still saved."
  fi

  echo "=== $(date) FULL done ${PLUGIN} ==="
} 2>&1 | tee -a "$LOG"
