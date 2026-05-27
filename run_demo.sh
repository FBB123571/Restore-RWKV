#!/usr/bin/env bash
# 在 Cursor / 非 login 终端里 conda、python 常不在 PATH，用本脚本运行推理。
set -e
cd "$(dirname "$0")"

PYTHON=""

# 1) 尝试初始化 conda 并激活 restore
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx restore; then
    conda activate restore
    PYTHON=python
  fi
fi

# 2) restore 不存在时，用本机已有的 snake1（已验证可跑通）
if [ -z "${PYTHON}" ]; then
  if [ -x "${HOME}/miniconda3/envs/snake1/bin/python" ]; then
    PYTHON="${HOME}/miniconda3/envs/snake1/bin/python"
    echo "ℹ️ 未找到 restore 环境，使用: ${PYTHON}"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
    echo "ℹ️ 使用系统 python3（需已安装 torch）"
  else
    echo "❌ 找不到 Python。请先安装环境，例如："
    echo "   conda create -n restore python=3.10 pytorch torchvision -c pytorch"
    exit 1
  fi
fi

echo "========== RWKV 去雾 =========="
"${PYTHON}" test_demo.py
echo ""
echo "========== 纯卷积基线 =========="
"${PYTHON}" test_demo_cnn.py
echo ""
echo "✅ 完成。结果见 outputs/rwkv/ 与 outputs/cnn_ablation/"
