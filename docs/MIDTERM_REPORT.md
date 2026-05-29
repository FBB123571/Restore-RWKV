# Restore-RWKV 去雾中期实验报告

**项目**：在 Vision-RWKV 去雾基线（V1）上，探索四个文献方向（A/B/C/D）的轻量插件式改进  
**日期**：2026-05-28  
**数据**：`data/dehaze_data/archive(1)`，约 13,990 对 hazy/clear（128×128 训练）  
**基线权重**：`checkpoints/rwkv/rwkv_dehaze_epoch_50.pth`（`hidden_dim=32`, `num_blocks=2`）

---

## 1. 实验目标

在 **不修改** `model_rwkv.py` 与 V1 预训练权重的前提下，验证四条改进路线是否能在去雾任务上带来稳定增益：

| 方向 | 文献思路 | 插件实现 |
|------|----------|----------|
| **A** | Fourier-RWKV 频域混合 | `FourierMix2D` 残差 |
| **B** | 双域（空域+频域） | 高频分支 + depthwise 精炼 |
| **C** | 深度先验（UDPNet 风格） | 轻量深度门控 |
| **D** | 组合损失 / 特征精炼 | 轻量空间残差头（同 B 思路） |

---

## 2. 基线 V1

- **训练**：128×128，MSE，全量数据，50 epoch  
- **结构**：`in_conv → RWKV blocks → out_conv + x → sigmoid`  
- **表现**：5 张测试图主观效果良好；hold-out 500 张 **PSNR 20.66 / SSIM 0.827**

---

## 3. 阶段一：V2 全网络重训（已放弃）

### 3.1 做法

- 新建 `model_rwkv_v2.py`，在 256×256 上从零训练四个方向（含组合损失 D）  
- 快速消融（3k 样本）与公平重训（全量 50 epoch）均已完成  

### 3.2 结果与问题

| 现象 | 原因分析 |
|------|----------|
| 输出发灰、对比度塌缩（std≈0.05） | 与 V1 训练配方不一致（256 vs 128）、残差学习未复现 V1 行为 |
| A/B 条纹或 NaN | 频域模块 FP16 数值问题；插件 scale 无约束导致过强 |
| 无法公平对比 V1 | 整网重训未继承 V1 强先验 |

**结论**：V2 路线不适合作为中期主结果，改为 **冻结 V1 + 可插拔插件**。

---

## 4. 阶段二：V1 冻结 + 插件消融（当前主方案）

### 4.1 设计原则

1. **骨干冻结**：仅训练插件参数（数百～数千）  
2. **有界残差**：`scale` 经 sigmoid 限制（A: 0.12，B/C/D: 0.04～0.06）  
3. **零初始化**：B/C/D 最后一层零初始化，初始输出 ≡ V1  
4. **V1 锚定损失**：`L = MSE(out, clear) + w_anchor × MSE(out, V1)`  
5. **先模板后全量**：512 样本 × 3 epoch 试训 → 人工看图 → 全量 15 epoch  

### 4.2 训练流程

```bash
# 模板试训（约 3–5 分钟/方向）
bash scripts/template_B.sh

# 全量（约 15–18 分钟/方向，需模板 PASS）
bash scripts/train_plugin_full.sh B

# 指标评测
python scripts/eval_plugins_metrics.py \
  --plugin_dir checkpoints/rwkv_plugin_v2 --run_name full --epoch 14
```

详见 [`docs/PLUGIN_TRAIN_WORKFLOW.md`](PLUGIN_TRAIN_WORKFLOW.md)。

### 4.3 主观结果（5 张 test_images，128px）

| 模型 | 主观评价 |
|------|----------|
| **V1** | 清晰、对比度正常 |
| **Plg-A (ep0)** | 与 V1 几乎一致，**可接受** |
| **Plg-B (full ep14)** | 略偏柔和，**可接受** |
| **Plg-C (full ep14)** | 与 V1 接近，**可接受** |
| **Plg-D (full ep14)** | 部分图略灰，整体 **可接受** |

预览图：

- A：`outputs/preview_plugins_best128/`
- B/C/D：`outputs/preview_plugins_v2/{B,C,D}/`

### 4.4 客观指标（500 张 hold-out，offset=12000，128px）

| Model | PSNR ↑ | SSIM ↑ | ΔPSNR vs V1 |
|-------|--------|--------|-------------|
| Hazy | 13.88 | 0.702 | — |
| **V1** | **20.66** | **0.827** | — |
| Plg-A (ep0) | 20.62 | 0.826 | −0.04 |
| Plg-B (full ep14) | 16.68 | 0.697 | −3.98 |
| Plg-C (full ep14) | 20.14 | 0.817 | −0.52 |
| Plg-D (full ep14) | 15.73 | 0.728 | −4.92 |

完整 CSV：[`data/result/plugin_metrics_full.csv`](../data/result/plugin_metrics_full.csv)

**解读**：

- **A / C** 与 V1 数值接近，符合「保守插件 + 强锚定」设计。  
- **B** 在 5 张 test 图上主观可接受，但 hold-out PSNR 偏低，可能与 epoch 14 在部分难例上过修有关；后续可扫 ep5/ep10 选优。  
- **D** 全量后 hold-out 掉分较多，模板阶段正常、全量后漂移，需更强约束或早停。  
- **肉眼与 PSNR 不完全一致**：小样本主观 + 强正则下，应用 **多指标 + 难例集** 联合判断。

---

## 5. 核心结论（中期）

1. **V1 已是强基线**；在冻结骨干 + 轻量插件设定下，四方向 **未显著超越 V1**，但实现了 **机制可插拔、训练稳定、不破坏 V1**。  
2. **失败经验**：整网 V2 重训、无界 scale、过重 C/D 模块会导致灰图与失真。  
3. **可交付成果**：  
   - 代码：`model_rwkv_plugin.py`、`clean_train_plugin.py`、模板/全量/评测脚本  
   - 流程：模板 → 验证 → 全量  
   - 结果图与指标表（见上）  
4. **推荐对外展示**：V1 + Plg-A（ep0）或 Plg-C（full ep14）作为「稳定版」；B/D 作为消融条目说明机制差异。

---

## 6. 后续工作

- [ ] 对 B/D 做 **epoch 扫描**（ep0/5/10/14），按 val PSNR 选 checkpoint  
- [ ] 构建 **难例子集**（浓雾、低对比）单独评测  
- [ ] 适度降低 `w_anchor` 或单方向放宽 `scale`，观察是否有可接受的增益  
- [ ] 整理论文用对比图：Hazy | V1 | A | B | C | D 六列 montage  
- [ ] 插件权重打包发布（当前 `.gitignore` 仅跟踪 V1/CNN 基线 pth）

---

## 7. 仓库结构（中期）

```
Restore-RWKV/
├── model_rwkv.py              # V1 网络（不改动）
├── model_rwkv_plugin.py       # 冻结 V1 + 插件 A/B/C/D
├── clean_train.py             # V1 训练
├── clean_train_plugin.py      # 插件训练
├── scripts/
│   ├── train_plugin_template.sh
│   ├── train_plugin_full.sh
│   ├── preview_plugins.py
│   ├── validate_plugin_template.py
│   └── eval_plugins_metrics.py
├── docs/
│   ├── MIDTERM_REPORT.md      # 本报告
│   └── PLUGIN_TRAIN_WORKFLOW.md
├── data/result/               # 指标 CSV
└── outputs/
    ├── preview_plugins_best128/   # A
    └── preview_plugins_v2/        # B/C/D
```

---

## 8. 参考文献（方向来源）

- Restore-RWKV: [arXiv:2407.11087](https://arxiv.org/abs/2407.11087)  
- Fourier-RWKV 等方向见项目内 V2 设计笔记与 `model_rwkv_v2.py`（已弃用，仅作记录）
