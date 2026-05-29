#!/usr/bin/env python3
"""
Premium midterm report PDF: Origin-style flowcharts, comparison montages, metrics charts.
Output: docs/Restore-RWKV_中期实验报告.pdf
"""
import csv
import os
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
FIG_DIR = os.path.join(DOCS, "figures")
OUT_PDF = os.path.join(DOCS, "Restore-RWKV_中期实验报告.pdf")
METRICS_CSV = os.path.join(ROOT, "data/result/plugin_metrics_full.csv")

# Origin-style palette
C_NAVY = "#1B3A5C"
C_BLUE = "#2E6DA4"
C_LIGHT = "#E8F0F8"
C_BOX = "#F4F7FB"
C_GRID = "#CBD5E0"
C_ACCENT = "#C0392B"
C_GREEN = "#1E7A46"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
if os.path.isfile(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
FP = font_manager.FontProperties(fname=FONT_PATH)
FP_BOLD = font_manager.FontProperties(fname=FONT_PATH, weight="bold")
plt.rcParams["axes.unicode_minus"] = False

PAGE = (8.27, 11.69)  # A4 inch
MARGIN = 0.55


def txt(ax, x, y, s, size=10, color=C_NAVY, ha="left", va="top", bold=False):
    fp = FP_BOLD if bold else FP
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontproperties=fp, zorder=5)


def header_footer(ax, title, page_no):
    ax.axhline(0.94, color=C_BLUE, linewidth=2.2, xmin=0.06, xmax=0.94)
    txt(ax, 0.06, 0.97, title, size=11, bold=True, color=C_BLUE)
    txt(ax, 0.94, 0.97, f"— {page_no} —", size=8, ha="right", color="#718096")


def new_page(pdf, title, page_no):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    header_footer(ax, title, page_no)
    return fig, ax


def draw_box(ax, x, y, w, h, label, fc=C_BOX, ec=C_BLUE, fontsize=8.5, sub=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    cy = y + h * 0.58 if sub else y + h / 2
    txt(ax, x + w / 2, cy, label, size=fontsize, ha="center", va="center", bold=True)
    if sub:
        txt(ax, x + w / 2, y + h * 0.28, sub, size=7, ha="center", va="center", color="#4A5568")


def arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=C_NAVY, zorder=3,
    )
    ax.add_patch(arr)


def page_cover(pdf):
    fig, ax = new_page(pdf, "", 1)
    ax.add_patch(mpatches.Rectangle((0, 0.72), 1, 0.28, color=C_NAVY, zorder=0))
    txt(ax, 0.5, 0.84, "Restore-RWKV 图像去雾", size=22, ha="center", color="white", bold=True)
    txt(ax, 0.5, 0.77, "中期实验报告", size=16, ha="center", color="#BEE3F8", bold=True)
    txt(ax, 0.5, 0.58, "V1 冻结骨干 + 轻量插件消融（方向 A / B / C / D）", size=12, ha="center", bold=True)
    lines = [
        f"报告日期：{date.today().isoformat()}",
        "任务：医学影像启发式 Vision-RWKV 去雾扩展",
        "数据：~13,990 对 hazy/clear · 训练分辨率 128×128",
        "基线：rwkv_dehaze_epoch_50（hidden_dim=32, num_blocks=2）",
        "仓库：github.com/FBB123571/Restore-RWKV",
    ]
    y = 0.48
    for line in lines:
        txt(ax, 0.5, y, line, size=10, ha="center", color="#2D3748")
        y -= 0.05
    ax.add_patch(FancyBboxPatch((0.12, 0.12), 0.76, 0.22, boxstyle="round,pad=0.02",
                                linewidth=1, edgecolor=C_BLUE, facecolor=C_LIGHT, zorder=1))
    txt(ax, 0.5, 0.28, "摘要", size=11, ha="center", bold=True, color=C_BLUE)
    txt(ax, 0.5, 0.20,
        "在冻结 V1 的前提下，验证频域/双域/深度/空间四条插件路线。\n"
        "V2 整网重训失败；插件方案训练稳定，主观可接受，数值与 V1 接近。",
        size=9, ha="center", va="center", color="#2D3748")
    pdf.savefig(fig)
    plt.close(fig)


def page_summary(pdf, metrics):
    fig, ax = new_page(pdf, "1  实验摘要与核心指标", 2)
    txt(ax, MARGIN, 0.88, "1.1  研究问题", size=11, bold=True)
    txt(ax, MARGIN, 0.83,
        "在不修改 model_rwkv.py 的前提下，四条文献方向能否作为可插拔模块，稳定提升去雾效果？",
        size=9.5)
    txt(ax, MARGIN, 0.76, "1.2  主要结论", size=11, bold=True)
    bullets = [
        "V1 为强基线；轻量插件 + V1 锚定损失可保证训练不崩溃。",
        "A/C 与 V1 数值最接近；B/D 主观可接受，部分 hold-out 指标偏低。",
        "推荐展示：V1 + Plg-A（ep0）或 Plg-C（full ep14）。",
    ]
    y = 0.71
    for b in bullets:
        txt(ax, MARGIN + 0.02, y, "•  " + b, size=9)
        y -= 0.045

    txt(ax, MARGIN, 0.54, "1.3  客观指标（500 张 hold-out，128px）", size=11, bold=True)
    # table
    cols = ["模型", "PSNR↑", "SSIM↑", "ΔPSNR"]
    rows = []
    v1_psnr = None
    for m in metrics:
        if m["model"] == "V1":
            v1_psnr = float(m["PSNR_mean"])
    for m in metrics:
        if m["model"] == "Hazy (input)":
            continue
        psnr = float(m["PSNR_mean"])
        ssim = float(m["SSIM_mean"])
        delta = psnr - v1_psnr if v1_psnr and m["model"] != "V1" else 0
        d_str = "—" if m["model"] == "V1" else f"{delta:+.2f}"
        rows.append([m["model"], f"{psnr:.2f}", f"{ssim:.3f}", d_str])

    tx, ty, tw = MARGIN, 0.48, 0.88
    rh, n = 0.032, len(rows) + 1
    ax.add_patch(mpatches.Rectangle((tx, ty - rh * n), tw, rh, color=C_BLUE, zorder=1))
    for i, c in enumerate(cols):
        txt(ax, tx + 0.02 + i * 0.22, ty - rh * 0.65, c, size=8.5, color="white", bold=True)
    for ri, row in enumerate(rows):
        yy = ty - rh * (ri + 2) + 0.01
        bg = C_LIGHT if ri % 2 == 0 else "white"
        ax.add_patch(mpatches.Rectangle((tx, yy), tw, rh, color=bg, ec=C_GRID, linewidth=0.5, zorder=1))
        for ci, cell in enumerate(row):
            txt(ax, tx + 0.02 + ci * 0.22, yy + rh * 0.65, cell, size=8.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_arch_flow(pdf):
    fig, ax = new_page(pdf, "2  系统架构（Origin 风格流程图）", 3)
    txt(ax, MARGIN, 0.88, "2.1  V1 冻结 + 插件推理路径", size=11, bold=True)
    draw_box(ax, 0.08, 0.72, 0.14, 0.08, "有雾图 x", fc="#FFF5F5", ec=C_ACCENT)
    draw_box(ax, 0.28, 0.70, 0.18, 0.12, "V1 Backbone", sub="in_conv + RWKV×2", fc=C_LIGHT)
    draw_box(ax, 0.52, 0.70, 0.16, 0.12, "Plugin", sub="A/B/C/D", fc="#FFFAF0", ec="#D69E2E")
    draw_box(ax, 0.74, 0.70, 0.18, 0.12, "out_conv+x", sub="sigmoid", fc=C_LIGHT)
    arrow(ax, 0.22, 0.76, 0.28, 0.76)
    arrow(ax, 0.46, 0.76, 0.52, 0.76)
    arrow(ax, 0.68, 0.76, 0.74, 0.76)
    txt(ax, 0.50, 0.64, "骨干参数冻结 · 仅训练插件（≤2万参数）", size=8.5, ha="center", color="#4A5568")

    txt(ax, MARGIN, 0.56, "2.2  四方向插件机制", size=11, bold=True)
    items = [
        ("A  Fourier", "FourierMix2D 频域门控残差", 0.08, 0.42),
        ("B  Dual", "高频 FFT 分支 + depthwise 精炼", 0.08, 0.30),
        ("C  Depth", "伪深度图 → 轻量门控", 0.08, 0.18),
        ("D  Spatial", "1×1 + DWConv 空间残差", 0.52, 0.36),
    ]
    for title, desc, x, y in items:
        draw_box(ax, x, y, 0.38, 0.09, title, sub=desc, fontsize=9)

    txt(ax, MARGIN, 0.08, "图 1  推理架构与四方向插件（矢量绘制，Origin 风格配色）", size=8, color="#718096")
    pdf.savefig(fig)
    plt.savefig(os.path.join(FIG_DIR, "flow_architecture.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def page_train_flow(pdf):
    fig, ax = new_page(pdf, "3  训练与验证流程", 4)
    txt(ax, MARGIN, 0.88, "3.1  模板 → 全量 两阶段流程", size=11, bold=True)
    steps = [
        (0.06, 0.74, "模板试训\n512张×3ep"),
        (0.26, 0.74, "数值校验\nPSNR/SSIM"),
        (0.46, 0.74, "人工看图\n5张 test"),
        (0.66, 0.74, "全量训练\n14k×15ep"),
        (0.82, 0.74, "发布权重\n+评测"),
    ]
    for i, (x, y, lab) in enumerate(steps):
        draw_box(ax, x, y, 0.16, 0.11, lab.replace("\n", " "), fontsize=7.5)
        if i < len(steps) - 1:
            arrow(ax, x + 0.16, y + 0.055, x + 0.20, y + 0.055)

    txt(ax, MARGIN, 0.62, "3.2  损失函数与约束", size=11, bold=True)
    draw_box(ax, 0.08, 0.48, 0.84, 0.10, "L = MSE(out, clear) + w_anchor · MSE(out, V1)", fontsize=10, fc=C_LIGHT)
    txt(ax, MARGIN, 0.44, "scale 有界（sigmoid）· 零初始化 · B/C/D: w_anchor=0.5~0.6", size=8.5, color="#4A5568")

    txt(ax, MARGIN, 0.36, "3.3  实验路线演进", size=11, bold=True)
    draw_box(ax, 0.06, 0.18, 0.26, 0.14, "阶段一 V2", sub="256整网重训\n失败:灰图", fc="#FFF5F5", ec=C_ACCENT)
    arrow(ax, 0.32, 0.25, 0.38, 0.25)
    draw_box(ax, 0.38, 0.18, 0.30, 0.14, "阶段二 插件", sub="128冻结V1\n成功:稳定", fc="#F0FFF4", ec=C_GREEN)
    arrow(ax, 0.68, 0.25, 0.74, 0.25)
    draw_box(ax, 0.74, 0.18, 0.20, 0.14, "中期结论", sub="机制可插拔\n未显著超V1", fc=C_LIGHT)

    txt(ax, MARGIN, 0.08, "图 2  训练验证流程与实验演进", size=8, color="#718096")
    pdf.savefig(fig)
    plt.savefig(os.path.join(FIG_DIR, "flow_training.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def build_montage(name):
    """6 columns: Hazy | V1 | A | B | C | D"""
    paths = {
        "hazy": os.path.join(ROOT, "test_images", name),
        "v1": os.path.join(ROOT, "outputs/rwkv/result", name),
        "A": os.path.join(ROOT, "outputs/preview_plugins_best128", name),
        "B": os.path.join(ROOT, "outputs/preview_plugins_v2/B", name),
        "C": os.path.join(ROOT, "outputs/preview_plugins_v2/C", name),
        "D": os.path.join(ROOT, "outputs/preview_plugins_v2/D", name),
    }
    # fallback v1 from compare split
    if not os.path.isfile(paths["v1"]):
        cmp_path = os.path.join(ROOT, "outputs/rwkv/compare", name)
        if os.path.isfile(cmp_path):
            cmp = Image.open(cmp_path).convert("RGB")
            w = cmp.width // 2
            paths["v1"] = None
            v1_crop = cmp.crop((w, 0, cmp.width, cmp.height))
        else:
            v1_crop = None
    else:
        v1_crop = Image.open(paths["v1"]).convert("RGB")

    tile = 128
    cols_img = []
    labels = ["Hazy", "V1", "Plg-A", "Plg-B", "Plg-C", "Plg-D"]
    sources = ["hazy", "v1", "A", "B", "C", "D"]
    for lab, key in zip(labels, sources):
        if key == "v1" and v1_crop is not None:
            im = v1_crop.resize((tile, tile))
        elif key in ("A", "B", "C", "D"):
            p = paths[key]
            full = Image.open(p).convert("RGB")
            im = full.crop((full.width - tile, 0, full.width, tile))
        else:
            im = Image.open(paths["hazy"]).convert("RGB").resize((tile, tile))
        cols_img.append(im)

    montage = Image.new("RGB", (tile * 6, tile))
    for i, im in enumerate(cols_img):
        montage.paste(im, (i * tile, 0))
    out = os.path.join(FIG_DIR, f"montage_{name}")
    montage.save(out)
    return out, labels


def page_comparisons(pdf, page_no, names, sec_title):
    fig, ax = new_page(pdf, sec_title, page_no)
    y0 = 0.82
    for name in names:
        mpath, labels = build_montage(name)
        img = plt.imread(mpath)
        h_frac = 0.22
        ax_img = fig.add_axes([MARGIN, y0 - h_frac, 0.88, h_frac])
        ax_img.imshow(img)
        ax_img.axis("off")
        tw = 0.88 / 6
        for i, lab in enumerate(labels):
            ax_img.text(
                i / 6 + 1 / 12, 1.06, lab, transform=ax_img.transAxes,
                ha="center", va="bottom", fontsize=7, color=C_NAVY, fontproperties=FP,
            )
        txt(ax, MARGIN, y0 + 0.01, name.replace(".png", ""), size=7.5, color="#718096")
        y0 -= h_frac + 0.04
    txt(ax, MARGIN, 0.06, "图 3  去雾效果对比（128px）：Hazy | V1 | Plg-A | Plg-B | Plg-C | Plg-D", size=8, color="#718096")
    pdf.savefig(fig)
    plt.close(fig)


def page_metrics_chart(pdf, metrics, page_no):
    fig, ax = new_page(pdf, "5  指标可视化", page_no)
    models, psnr, ssim = [], [], []
    v1p = 20.66
    for m in metrics:
        if m["model"] in ("Hazy (input)",):
            continue
        models.append(m["model"].replace("Plg-", "P-").replace(" (full ep14)", "").replace(" (ep0)", ""))
        psnr.append(float(m["PSNR_mean"]))
        ssim.append(float(m["SSIM_mean"]))

    ax_bar = fig.add_axes([0.10, 0.48, 0.82, 0.38])
    x = np.arange(len(models))
    colors = [C_BLUE if "V1" in models[i] else "#63B3ED" for i in range(len(models))]
    bars = ax_bar.bar(x, psnr, color=colors, edgecolor=C_NAVY, linewidth=0.6, width=0.62)
    ax_bar.axhline(v1p, color=C_ACCENT, linestyle="--", linewidth=1, label=f"V1 PSNR={v1p:.2f}")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, fontproperties=FP, fontsize=7, rotation=15, ha="right")
    ax_bar.set_ylabel("PSNR (dB)", fontproperties=FP, fontsize=9)
    ax_bar.set_title("PSNR 对比（Origin 风格柱状图）", fontproperties=FP_BOLD, fontsize=10, color=C_NAVY)
    ax_bar.grid(axis="y", color=C_GRID, linestyle="-", linewidth=0.5, alpha=0.8)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(prop=FP, fontsize=7)
    for b, v in zip(bars, psnr):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=7, fontproperties=FP)

    ax2 = fig.add_axes([0.10, 0.10, 0.82, 0.30])
    ax2.plot(x, ssim, "o-", color=C_GREEN, linewidth=1.8, markersize=6, markerfacecolor="white", markeredgewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontproperties=FP, fontsize=7, rotation=15, ha="right")
    ax2.set_ylabel("SSIM", fontproperties=FP, fontsize=9)
    ax2.set_title("SSIM 曲线", fontproperties=FP_BOLD, fontsize=10, color=C_NAVY)
    ax2.grid(color=C_GRID, linestyle="-", linewidth=0.5, alpha=0.8)
    ax2.set_ylim(0.65, 0.85)

    txt(ax, MARGIN, 0.06, "图 4  客观指标（500 张 hold-out）", size=8, color="#718096")
    pdf.savefig(fig)
    plt.savefig(os.path.join(FIG_DIR, "chart_metrics.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def page_conclusion(pdf, page_no):
    fig, ax = new_page(pdf, "6  结论与后续工作", page_no)
    txt(ax, MARGIN, 0.88, "6.1  中期结论", size=11, bold=True)
    items = [
        ("可交付", "插件式消融代码、训练流程、对比图与指标表已齐备。"),
        ("稳定性", "相对 V2 整网重训，插件方案不破坏 V1 视觉质量。"),
        ("性能", "四方向未显著超越 V1；A/C 数值最接近，适合作为「安全增强」。"),
        ("论文表述", "强调机制消融与训练范式，而非绝对 SOTA。"),
    ]
    y = 0.82
    for title, body in items:
        draw_box(ax, MARGIN, y - 0.09, 0.88, 0.09, title, sub=body, fontsize=9)
        y -= 0.11

    txt(ax, MARGIN, 0.34, "6.2  后续计划", size=11, bold=True)
    nexts = ["B/D epoch 扫描与难例子集评测", "适度降低 w_anchor 探索可接受增益", "六列 montage 论文图与权重 Release"]
    y = 0.29
    for n in nexts:
        txt(ax, MARGIN + 0.02, y, f"□  {n}", size=9)
        y -= 0.04

    pdf.savefig(fig)
    plt.close(fig)


def load_metrics():
    if not os.path.isfile(METRICS_CSV):
        return []
    with open(METRICS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    metrics = load_metrics()
    test_names = sorted(
        f for f in os.listdir(os.path.join(ROOT, "test_images"))
        if f.lower().endswith((".png", ".jpg"))
    )

    with PdfPages(OUT_PDF) as pdf:
        d = pdf.infodict()
        d["Title"] = "Restore-RWKV 中期实验报告"
        d["Author"] = "Restore-RWKV Project"
        d["Subject"] = "Dehazing Plugin Ablation Midterm Report"

        page_cover(pdf)
        if metrics:
            page_summary(pdf, metrics)
        page_arch_flow(pdf)
        page_train_flow(pdf)
        if len(test_names) >= 3:
            page_comparisons(pdf, 5, test_names[:3], "4  主观效果对比（一）")
            page_comparisons(pdf, 6, test_names[3:], "4  主观效果对比（二）")
        if metrics:
            page_metrics_chart(pdf, metrics, 7)
        page_conclusion(pdf, 8)

    print(OUT_PDF)


if __name__ == "__main__":
    main()
