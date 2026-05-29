#!/usr/bin/env python3
"""
大创中期报告 PDF（PIL 渲染中文，无乱码）+ 对比图 + 流程图。
输出: docs/Restore-RWKV_大创中期报告.pdf
"""
from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
OUT = DOCS / "Restore-RWKV_大创中期报告.pdf"
DESKTOP = Path.home() / "桌面"
if not DESKTOP.is_dir():
    DESKTOP = Path.home() / "Desktop"
METRICS = ROOT / "data/result/plugin_metrics_full.csv"

W, H = 1654, 2339  # A4 @200dpi
M = 90
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

C_TITLE = "#0D47A1"
C_H1 = "#1565C0"
C_H2 = "#1976D2"
C_BODY = "#212121"
C_MUTED = "#616161"
C_LINE = "#B0BEC5"
C_BG_HL = "#E3F2FD"
C_BOX = "#F5F9FC"
C_GOLD = "#C9A227"
C_NAVY = "#0A2463"
C_NAVY2 = "#1E3A8A"

PROJECT_META = {
    "school": "中山大学",
    "college": "智能工程学院",
    "title": "基于 Vision-RWKV 的轻量化图像去雾方法研究与消融验证",
    "leader": "刘小凡",
    "members": "刘小凡、白冉",
    "supervisor": "彭键清",
    "supervisor_title": "副教授",
    "project_type": "大学生创新训练计划（创新类）",
    "period": "2025—2026 学年",
    "date": "2026 年 5 月",
    "repo": "https://github.com/FBB123571/Restore-RWKV",
    "abstract": (
        "本课题在 Restore-RWKV 去雾基线（V1）上，围绕频域混合、双域高频、深度门控、"
        "空间精炼四条方向开展可插拔消融。中期已完成基线复现、插件框架、模板—全量训练流程、"
        "主观对比图与 500 张 hold-out 客观评测。结果表明：轻量插件在强锚定约束下可稳定训练，"
        "A/C 与 V1 最接近；尚未显著超越 V1，后续将针对难例与 epoch 选择继续优化。"
    ),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_PATH
    return ImageFont.truetype(path, size, index=0)


class Page:
    def __init__(self):
        self.im = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.im)
        self.y = M

    def fit(self, need: int) -> bool:
        if self.y + need > H - M:
            return False
        return True

    def line(self, dy: int = 8):
        self.y += dy

    def text(self, s: str, size: int = 20, color: str = C_BODY, bold: bool = False, indent: int = 0):
        f = font(size, bold)
        max_w = W - 2 * M - indent
        avg = max(1, int(max_w / (size * 0.95)))
        for para in s.split("\n"):
            for ln in textwrap.wrap(para, width=avg) if para.strip() else [""]:
                if not self.fit(size + 12):
                    return False
                self.d.text((M + indent, self.y), ln, fill=color, font=f)
                self.y += int(size * 1.55)
        return True

    def h1(self, s: str):
        if not self.fit(50):
            return False
        self.d.line([(M, self.y), (W - M, self.y)], fill=C_LINE, width=1)
        self.y += 14
        self.text(s, 28, C_H1, bold=True)
        self.y += 6
        return True

    def h2(self, s: str):
        self.y += 8
        return self.text(s, 22, C_H2, bold=True)

    def table(self, headers: list[str], rows: list[list[str]], col_w: list[int] | None = None):
        fs = 16
        f = font(fs)
        fh = int(fs * 1.8)
        cols = len(headers)
        if col_w is None:
            tw = W - 2 * M
            col_w = [tw // cols] * cols
        if not self.fit(fh * (len(rows) + 2)):
            return False
        x0 = M
        # header
        self.d.rectangle([x0, self.y, x0 + sum(col_w), self.y + fh], fill=C_BG_HL, outline=C_LINE)
        x = x0
        for i, h in enumerate(headers):
            self.d.text((x + 6, self.y + 4), h, fill=C_TITLE, font=font(fs, True))
            x += col_w[i]
        self.y += fh
        for ri, row in enumerate(rows):
            bg = "#FAFAFA" if ri % 2 else "white"
            self.d.rectangle([x0, self.y, x0 + sum(col_w), self.y + fh], fill=bg, outline=C_LINE)
            x = x0
            for i, cell in enumerate(row):
                self.d.text((x + 6, self.y + 4), str(cell)[:28], fill=C_BODY, font=f)
                x += col_w[i]
            self.y += fh
        self.y += 10
        return True

    def image(self, path: Path, caption: str = "", height: int = 420):
        if not path.is_file():
            return True
        img = Image.open(path).convert("RGB")
        ratio = (W - 2 * M) / img.width
        nh = min(height, int(img.height * ratio))
        nw = int(nh / img.height * img.width)
        img = img.resize((nw, nh), Image.LANCZOS)
        if not self.fit(nh + 40):
            return False
        x = M + (W - 2 * M - nw) // 2
        self.im.paste(img, (x, self.y))
        self.y += nh + 8
        if caption:
            self.text(caption, 14, C_MUTED)
        return True

    def flowchart(self):
        if not self.fit(520):
            return False
        self.h2("技术路线流程图")
        boxes = [
            (M, self.y, "有雾数据集\n~13990对"),
            (M + 280, self.y, "V1 基线训练\n128 MSE 50ep"),
            (M + 560, self.y, "冻结骨干\n+插件 A/B/C/D"),
            (M + 140, self.y + 130, "模板试训\n512×3ep"),
            (M + 420, self.y + 130, "全量训练\n15ep"),
            (M + 700, self.y + 130, "评测发布\nPSNR/SSIM"),
        ]
        for x, y, lab in boxes:
            self.d.rounded_rectangle([x, y, x + 240, y + 88], radius=12, outline=C_H1, width=2, fill=C_BOX)
            self.text_at(x + 12, y + 12, lab, 16, bold=True)
        # arrows
        arr = [(M + 240, self.y + 44), (M + 280, self.y + 44), (M + 520, self.y + 44), (M + 560, self.y + 44),
               (M + 380, self.y + 88), (M + 380, self.y + 130), (M + 540, self.y + 174), (M + 700, self.y + 174)]
        for i in range(0, len(arr) - 1, 2):
            self.d.line([arr[i], arr[i + 1]], fill=C_TITLE, width=2)
        self.y += 260
        return True

    def text_at(self, x, y, s, size, bold=False, color=C_BODY):
        f = font(size, bold)
        for i, ln in enumerate(s.split("\n")):
            self.d.text((x, y + i * int(size * 1.4)), ln, fill=color, font=f)


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _draw_gradient_rect(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], c1: str, c2: str):
    x0, y0, x1, y1 = box
    r1, g1, b1 = _hex_rgb(c1)
    r2, g2, b2 = _hex_rgb(c2)
    h = max(1, y1 - y0)
    for i in range(h):
        t = i / h
        d.line([(x0, y0 + i), (x1, y0 + i)], fill=(_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t)))


def _wrap_lines(text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        buf = ""
        for ch in para:
            trial = buf + ch
            if f.getlength(trial) <= max_w:
                buf = trial
            else:
                if buf:
                    lines.append(buf)
                buf = ch
        if buf:
            lines.append(buf)
    return lines or [""]


def render_cover() -> Image.Image:
    """精美封面：渐变顶栏 + 信息卡片 + 摘要区。"""
    im = Image.new("RGB", (W, H), "#F7FAFD")
    d = ImageDraw.Draw(im)

    # 顶部渐变横幅
    banner_h = 520
    _draw_gradient_rect(d, (0, 0, W, banner_h), C_NAVY, C_NAVY2)

    # 装饰光晕与斜线
    overlay = Image.new("RGBA", (W, banner_h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i in range(-banner_h, W, 90):
        od.line([(i, 0), (i + banner_h, banner_h)], fill=(255, 255, 255, 18), width=2)
    od.ellipse([W - 340, -80, W + 60, 320], fill=(255, 255, 255, 22))
    od.ellipse([-120, 260, 220, 600], fill=(201, 162, 39, 28))
    im.paste(overlay, (0, 0), overlay)

    d = ImageDraw.Draw(im)
    d.rectangle([M, 36, W - M, 38], fill=C_GOLD)
    d.text((M, 58), PROJECT_META["school"], fill="#E8EEF9", font=font(26))
    d.text((M, 96), PROJECT_META["college"], fill="#B8C9E8", font=font(20))

    cx = W // 2
    badge_y = 150
    d.ellipse([cx - 54, badge_y, cx + 54, badge_y + 108], outline=C_GOLD, width=3, fill="#163A7A")
    d.text((cx - 36, badge_y + 32), "大创", fill=C_GOLD, font=font(36, True))

    title1 = "大学生创新训练计划"
    title2 = "项目中期检查报告"
    f1, f2 = font(30, True), font(46, True)
    d.text((cx - f1.getlength(title1) / 2, 290), title1, fill="#E3ECFA", font=f1)
    d.text((cx - f2.getlength(title2) / 2, 340), title2, fill="white", font=f2)
    d.line([(M + 40, 430), (W - M - 40, 430)], fill=C_GOLD, width=2)

    # 项目名称条
    card_x0, card_y0 = M - 10, 470
    card_x1, card_y1 = W - M + 10, 560
    d.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=14, fill="white", outline=C_GOLD, width=2)
    proj = PROJECT_META["title"]
    fp = font(24, True)
    for i, ln in enumerate(_wrap_lines(proj, fp, card_x1 - card_x0 - 48)):
        d.text((card_x0 + 24, card_y0 + 22 + i * 34), ln, fill=C_NAVY, font=fp)

    # 信息表
    info_y = 600
    info_h = 360
    d.rounded_rectangle([M, info_y, W - M, info_y + info_h], radius=18, fill="white", outline=C_LINE, width=2)
    d.rectangle([M, info_y, M + 8, info_y + info_h], fill=C_H1)

    rows = [
        ("项目类型", PROJECT_META["project_type"]),
        ("项目负责人", PROJECT_META["leader"]),
        ("项目成员", PROJECT_META["members"]),
        ("指导教师", f"{PROJECT_META['supervisor']}（{PROJECT_META['supervisor_title']}）"),
        ("所在学院", f"{PROJECT_META['school']} · {PROJECT_META['college']}"),
        ("项目周期", PROJECT_META["period"]),
        ("报告日期", PROJECT_META["date"]),
        ("代码仓库", PROJECT_META["repo"]),
    ]
    label_w = 200
    row_h = 42
    fy, fv = font(18, True), font(17)
    for i, (lab, val) in enumerate(rows):
        y = info_y + 18 + i * row_h
        if i % 2 == 0:
            d.rectangle([M + 8, y - 2, W - M, y + row_h - 6], fill="#F8FBFF")
        d.text((M + 28, y + 6), lab, fill=C_H1, font=fy)
        for j, ln in enumerate(_wrap_lines(val, fv, W - M - M - label_w - 40)):
            d.text((M + 28 + label_w, y + 6 + j * 22), ln, fill=C_BODY, font=fv)

    # 摘要
    abs_y = info_y + info_h + 36
    abs_h = 300
    d.rounded_rectangle([M, abs_y, W - M, abs_y + abs_h], radius=16, fill=C_BOX, outline=C_H1, width=2)
    d.rectangle([M, abs_y, M + 6, abs_y + abs_h], fill=C_GOLD)
    d.text((M + 28, abs_y + 20), "摘　要", fill=C_H1, font=font(26, True))
    fa = font(18)
    ay = abs_y + 68
    for ln in _wrap_lines(PROJECT_META["abstract"], fa, W - 2 * M - 56):
        d.text((M + 28, ay), ln, fill=C_BODY, font=fa)
        ay += 30

    # 页脚装饰
    d.line([(M, H - 70), (W - M, H - 70)], fill=C_LINE, width=1)
    foot = f"{PROJECT_META['school']} {PROJECT_META['college']} · Restore-RWKV 去雾消融研究"
    d.text((M, H - 52), foot, fill=C_MUTED, font=font(15))
    return im


def build_montage(name: str) -> Path | None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"montage_{name}"
    if out.is_file():
        return out
    tile = 128
    parts = []
    labels = []
    hazy = ROOT / "test_images" / name
    if not hazy.is_file():
        return None
    parts.append(Image.open(hazy).convert("RGB").resize((tile, tile)))
    labels.append("Hazy")
    v1p = ROOT / "outputs/rwkv/result" / name
    if v1p.is_file():
        parts.append(Image.open(v1p).convert("RGB").resize((tile, tile)))
    else:
        cmp = Image.open(ROOT / "outputs/rwkv/compare" / name)
        parts.append(cmp.crop((cmp.width // 2, 0, cmp.width, cmp.height)).resize((tile, tile)))
    labels.append("V1")
    for key, sub in [("A", "preview_plugins_best128"), ("B", "preview_plugins_v2/B"), ("C", "preview_plugins_v2/C"), ("D", "preview_plugins_v2/D")]:
        p = ROOT / "outputs" / sub / name if sub != "preview_plugins_best128" else ROOT / "outputs" / sub / name
        if p.is_file():
            im = Image.open(p).convert("RGB")
            parts.append(im.crop((im.width - tile, 0, im.width, tile)))
            labels.append(f"Plg-{key}")
    if len(parts) < 3:
        return None
    mont = Image.new("RGB", (tile * len(parts), tile + 36), "white")
    dr = ImageDraw.Draw(mont)
    f = font(18, True)
    for i, (im, lab) in enumerate(zip(parts, labels)):
        mont.paste(im, (i * tile, 36))
        dr.text((i * tile + 8, 6), lab, fill=C_TITLE, font=f)
    mont.save(out)
    return out


def load_metrics():
    if not METRICS.is_file():
        return []
    with open(METRICS, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_pages() -> list[Image.Image]:
    pages: list[Page] = []

    def flush(pg: Page | None) -> Page:
        nonlocal pages
        if pg is not None:
            pages.append(pg.im)
        return Page()

    pages.append(render_cover())
    p = Page()

    sections = [
        ("一、研究背景与意义", (
            "我国智能感知与自动驾驶、安防监控等行业对实时去雾提出更高要求。"
            "传统物理模型方法在浓雾、非均匀雾场景下易失效；深度学习方法效果好但计算开销大。"
            "Restore-RWKV 将线性复杂度的 RWKV 引入图像复原，为实时去雾提供新路径。"
            "本课题在开源去雾基线上，验证 2025 年以来文献方向的可插拔改进，具有明确工程与双创价值。"
        )),
        ("二、研究目标与任务书对照", (
            "对照任务书，中期需完成：①基线训练与验证；②四方向文献拆解；③消融框架与训练流程；"
            "④对比图与指标表；⑤中期报告。除「显著超越 V1」尚在进行外，其余条目均已落地。"
        )),
    ]
    for title, body in sections:
        if not p.h1(title):
            p = flush(p)
            p.h1(title)
        p.text(body, 19)

    if not p.flowchart():
        p = flush(p)
        p.flowchart()

    p.h1("三、阶段性研究进展")
    for sub, body in [
        ("3.1 基线 V1", "完成 128×128 MSE 训练，权重 epoch_50；hold-out PSNR 20.66 / SSIM 0.827。"),
        ("3.2 V2 探索（终止）", "256 整网重训出现灰图与对比度塌缩，不作为主结果。"),
        ("3.3 插件主线", "完成 A/B/C/D 模板试训与 B/C/D 全量；建立 validate + preview + metrics 工具链。"),
        ("3.4 开源", "代码与文档已推送 GitHub，含训练脚本、对比图与 CSV 指标。"),
    ]:
        p.h2(sub)
        p.text(body, 18)

    metrics = load_metrics()
    if metrics:
        if not p.h1("四、实验结果与分析"):
            p = flush(p)
            p.h1("四、实验结果与分析")
        rows = []
        v1p = float(next(m["PSNR_mean"] for m in metrics if m["model"] == "V1"))
        for m in metrics:
            if m["model"] == "Hazy (input)":
                continue
            d = "—" if m["model"] == "V1" else f"{float(m['PSNR_mean']) - v1p:+.2f}"
            rows.append([m["model"], f"{float(m['PSNR_mean']):.2f}", f"{float(m['SSIM_mean']):.3f}", d])
        p.table(["模型", "PSNR", "SSIM", "ΔPSNR"], rows, [280, 120, 120, 120])

    p.h1("五、主观效果对比（128px）")
    p.text("下图从左至右：Hazy | V1 | Plg-A | Plg-B | Plg-C | Plg-D。", 17, C_MUTED)
    names = sorted(x.name for x in (ROOT / "test_images").glob("*.png"))
    for name in names[:3]:
        mp = build_montage(name)
        if mp and not p.image(mp, name, 300):
            p = flush(p)
            p.image(mp, name, 300)
    p = flush(p)
    p.h1("五、主观效果对比（续）")
    for name in names[3:]:
        mp = build_montage(name)
        if mp:
            p.image(mp, name, 300)

    p.h1("六、创新点与特色")
    for t in [
        "冻结 V1 骨干的可插拔消融，保证公平对比与可回退；",
        "模板—全量两阶段训练，降低消融试错成本；",
        "统一 hold-out 评测脚本，支撑报告/论文表格自动生成；",
        "RWKV 线性复杂度主干，具备实时部署潜力。",
    ]:
        p.text("• " + t, 18)

    p.h1("七、存在问题与下一步计划")
    p.table(
        ["问题", "改进"],
        [
            ["插件未显著超 V1", "单方向放宽 scale + val 早停"],
            ["B/D hold-out 偏低", "epoch 扫描 ep5/10/14"],
            ["难例不足", "构建浓雾子集"],
        ],
        [520, 520],
    )
    p.text("下一步：2026.06 完成最优 checkpoint 评选；07 月整理论文级对比图；09 月提交大创结题材料。", 18)

    p.h1("八、参考文献")
    refs = [
        "[1] Yang Z. et al. Restore-RWKV. IEEE JBHI, 2026.",
        "[2] Fourier-RWKV. arXiv:2512.08161, 2025.",
        "[3] He K. et al. Dark Channel Prior. TPAMI, 2011.",
    ]
    for r in refs:
        p.text(r, 17)

    p.line(30)
    p.text("项目负责人签字：____________    指导教师签字：____________    日期：2026年5月", 18)

    pages.append(p.im)
    return pages


def main():
    imgs = render_pages()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(OUT, "PDF", resolution=200, save_all=True, append_images=imgs[1:])
    if DESKTOP.is_dir():
        desk = DESKTOP / OUT.name
        imgs[0].save(desk, "PDF", resolution=200, save_all=True, append_images=imgs[1:])
        print(desk)
    print(OUT)


if __name__ == "__main__":
    main()
