#!/usr/bin/env bash
# Zip plugin weights + manifest for GitHub Release.
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE_DIR=checkpoints/rwkv_plugin_release
OUT=checkpoints/rwkv_plugin_weights.zip
mkdir -p "$RELEASE_DIR"

cat > "$RELEASE_DIR/README.txt" <<'EOF'
Restore-RWKV Plugin Weights (frozen V1 backbone)

Load with model_rwkv_plugin.DehazeRWKV_V1_Plugin + load_plugin_weights().

Files:
  plugin_A_epoch_0.pth   - Direction A (Fourier), early / near V1
  plugin_A_epoch_14.pth  - Direction A, full train
  plugin_B_epoch_14.pth  - Direction B (dual-domain high-freq)
  plugin_C_epoch_14.pth  - Direction C (depth gate)
  plugin_D_epoch_14.pth  - Direction D (spatial refine)

V1 backbone (required): checkpoints/rwkv/rwkv_dehaze_epoch_50.pth
EOF

cd checkpoints
zip -r rwkv_plugin_weights.zip rwkv_plugin_release/
echo "Created checkpoints/rwkv_plugin_weights.zip"
ls -lh rwkv_plugin_weights.zip
