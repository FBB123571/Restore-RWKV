# 上传到 FBB123571 的 GitHub

本机当前 **未** 用 [FBB123571](https://github.com/FBB123571) 登录 Git，需要你先完成一次授权（约 1 分钟）。

## 步骤 1：登录你的 GitHub（FBB123571）

在终端执行（关闭代理更稳）：

```bash
env -u http_proxy -u https_proxy -u ALL_PROXY NO_PROXY='*' gh auth login -h github.com
```

- 选 **GitHub.com**
- 选 **HTTPS**
- 选 **Login with a web browser**，用 **FBB123571** 账号完成授权
- 完成后执行：`gh auth setup-git`

## 步骤 2：创建仓库并推送

```bash
cd /mnt/sdb1/leijh/EnergySnake1/robot/Restore-RWKV

# 在 FBB123571 下创建空仓库并推送（若网页已建好可跳过 create，直接 push）
env -u http_proxy -u https_proxy -u ALL_PROXY NO_PROXY='*' \
  gh repo create Restore-RWKV --public --source=. --remote=origin --push
```

若提示仓库已存在，只推送：

```bash
env -u http_proxy -u https_proxy -u ALL_PROXY NO_PROXY='*' \
  git -c http.proxy= -c https.proxy= push -u origin main
```

## 完成后

仓库地址：**https://github.com/FBB123571/Restore-RWKV**

---

> 说明：之前误用其他账号在 `ely2665253325-code/Restore-RWKV` 建过空仓库，不需要可在 GitHub 设置里删除。
