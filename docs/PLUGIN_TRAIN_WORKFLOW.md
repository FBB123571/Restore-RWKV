# 插件训练流程（先模板，再全量）

避免一次性全量训错，**每个方向 B/C/D 单独走模板 → 你看图 → 再全量**。

## 1. 模板试训（约 3–5 分钟/个）

```bash
bash scripts/template_B.sh   # 或 template_C.sh / template_D.sh
```

默认配置（在 `scripts/train_plugin_template.sh` 顶部可改）：

| 参数 | 值 |
|------|-----|
| 样本数 | 512 |
| epoch | 3 |
| 分辨率 | 128 |
| batch | 64 |

产出：

- 权重：`checkpoints/rwkv_plugin_template/plugin_{B,C,D}_template/`
- 预览：`outputs/plugin_template_preview/{B,C,D}/`
- 日志：`logs/template_{B,C,D}.log`
- 终端自动跑 **数值校验**（与 V1 差异、对比度）

**只有终端出现 `PASS — safe to run full training` 才进入下一步。**

## 2. 人工看图

打开 `outputs/plugin_template_preview/B/`（或 C/D），确认没有：

- 发灰发白（对比度塌了）
- 竖条纹 / 严重糊块
- 明显比 V1 更差

A 你已认可，可继续用 `outputs/preview_plugins_best128/` 的 **epoch 0**。

## 3. 全量训练（约 15–18 分钟/个）

```bash
bash scripts/train_plugin_full.sh B   # 需模板 PASS 才会开训
```

- 全量 ~14k 图，15 epoch，128×128
- 权重：`checkpoints/rwkv_plugin_v2/plugin_{X}_full/`
- 预览：`outputs/preview_plugins_v2/{X}/`

## 4. 不要用的旧脚本

- `scripts/retrain_plugins_fixed.sh` — 已停用（一次性连训易出错）
- 旧权重 `checkpoints/rwkv_plugin/` epoch 29 — 已证实过拟合/崩溃

## 推荐顺序

1. `template_B.sh` → 你看图 → `train_plugin_full.sh B`
2. `template_C.sh` → …
3. `template_D.sh` → …
