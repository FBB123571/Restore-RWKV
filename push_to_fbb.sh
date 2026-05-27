#!/usr/bin/env bash
# 在已用 gh 登录 FBB123571 后执行： bash push_to_fbb.sh
set -e
cd "$(dirname "$0")"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY='*'

echo "当前 gh 账号："
gh api user -q '.login' 2>/dev/null || { echo "请先: gh auth login"; exit 1; }

LOGIN=$(gh api user -q '.login')
if [ "$LOGIN" != "FBB123571" ]; then
  echo "⚠️ 当前登录的是 $LOGIN，不是 FBB123571"
  echo "请运行: gh auth login -h github.com  并选择 FBB123571"
  exit 1
fi

git remote set-url origin https://github.com/FBB123571/Restore-RWKV.git

if gh repo view FBB123571/Restore-RWKV >/dev/null 2>&1; then
  echo "仓库已存在，直接推送..."
  git -c http.proxy= -c https.proxy= push -u origin main
else
  echo "创建仓库并推送..."
  gh repo create Restore-RWKV --public --description "Vision-RWKV image dehazing with CNN ablation baseline" --source=. --remote=origin --push
fi

echo "✅ 完成: https://github.com/FBB123571/Restore-RWKV"
