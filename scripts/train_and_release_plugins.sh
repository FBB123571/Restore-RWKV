#!/usr/bin/env bash
# Train all plugins, copy final weights to checkpoints/rwkv_plugin_release/, then upload.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,6,7}"

LOG=logs/package_all_plugins.log
RELEASE_DIR=checkpoints/rwkv_plugin_release
mkdir -p "$RELEASE_DIR"

{
  echo "=== $(date) train all plugins for release ==="
  for P in A B C D; do
    bash scripts/train_plugin_template.sh "$P"
    bash scripts/train_plugin_full.sh "$P"
    SRC="checkpoints/rwkv_plugin_v2/plugin_${P}_full/plugin_${P}_epoch_14.pth"
    cp "$SRC" "$RELEASE_DIR/plugin_${P}_epoch_14.pth"
    echo "copied $SRC"
  done
  # A 额外保留 ep0 风格（全量第 0 epoch 若存在）
  if [ -f checkpoints/rwkv_plugin_v2/plugin_A_full/plugin_A_epoch_0.pth ]; then
    cp checkpoints/rwkv_plugin_v2/plugin_A_full/plugin_A_epoch_0.pth "$RELEASE_DIR/plugin_A_epoch_0.pth"
  fi
  bash scripts/package_plugin_release.sh
  echo "=== $(date) done ==="
} 2>&1 | tee -a "$LOG"
