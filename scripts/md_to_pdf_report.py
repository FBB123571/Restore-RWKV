#!/usr/bin/env python3
"""Convert docs/MIDTERM_REPORT.md to PDF on desktop (matplotlib)."""
import os
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(ROOT, "docs", "MIDTERM_REPORT.md")
DESKTOP = os.path.join(os.path.expanduser("~"), "桌面")
if not os.path.isdir(DESKTOP):
    DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT_PDF = os.path.join(DESKTOP, "Restore-RWKV_中期实验报告.pdf")

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
if os.path.isfile(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH) if os.path.isfile(FONT_PATH) else None
if FONT_PROP:
    plt.rcParams["font.family"] = FONT_PROP.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def parse_md(path):
    blocks = []
    in_code = False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                blocks.append(("code", line))
                continue
            if line.startswith("# "):
                blocks.append(("h1", line[2:].strip()))
            elif line.startswith("## "):
                blocks.append(("h2", line[3:].strip()))
            elif line.startswith("### "):
                blocks.append(("h3", line[4:].strip()))
            elif line.strip() == "---":
                blocks.append(("hr", ""))
            elif line.startswith("|"):
                blocks.append(("table", line.strip()))
            elif line.startswith("- ") or re.match(r"^\d+\.\s", line):
                blocks.append(("li", line.lstrip("- ").strip()))
            elif line.strip():
                blocks.append(("p", line.strip()))
            else:
                blocks.append(("blank", ""))
    return blocks


def wrap_line(text, width):
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return textwrap.wrap(text, width=width) or [""]


def render_pdf(blocks, out_path):
    page_w, page_h = 8.27, 11.69  # A4 inch
    margin_x, margin_top, margin_bottom = 0.75, 0.75, 0.75
    usable_w = page_w - 2 * margin_x
    line_h = { "h1": 0.38, "h2": 0.30, "h3": 0.26, "p": 0.22, "li": 0.22, "table": 0.20, "code": 0.18, "blank": 0.12, "hr": 0.15 }
    sizes = { "h1": 16, "h2": 13, "h3": 11, "p": 9.5, "li": 9.5, "table": 8.5, "code": 8 }

    pages = []
    y = page_h - margin_top
    lines_buf = []

    def flush_page():
        nonlocal y, lines_buf
        if lines_buf:
            pages.append(list(lines_buf))
            lines_buf = []
        y = page_h - margin_top

    def need(h):
        nonlocal y
        if y - h < margin_bottom:
            flush_page()

    def add(kind, text):
        nonlocal y
        if kind == "hr":
            need(line_h["hr"] + 0.05)
            lines_buf.append(("hr", "", 9))
            y -= line_h["hr"] + 0.05
            return
        if kind == "blank":
            y -= line_h["blank"]
            return
        width = 52 if kind != "code" else 70
        prefix = "  • " if kind == "li" else ""
        for part in wrap_line(prefix + text, width):
            need(line_h.get(kind, 0.22))
            lines_buf.append((kind, part, sizes.get(kind, 9.5)))
            y -= line_h.get(kind, 0.22)

    for kind, text in blocks:
        add(kind, text)

    flush_page()

    with PdfPages(out_path) as pdf:
        for page_lines in pages:
            fig, ax = plt.subplots(figsize=(page_w, page_h))
            ax.set_xlim(0, page_w)
            ax.set_ylim(0, page_h)
            ax.axis("off")
            y = page_h - margin_top
            for kind, text, size in page_lines:
                if kind == "hr":
                    ax.plot([margin_x, page_w - margin_x], [y, y], color="#999999", linewidth=0.6)
                    y -= line_h["hr"]
                    continue
                weight = "bold" if kind in ("h1", "h2", "h3") else "normal"
                if kind == "code":
                    ax.text(margin_x, y, text, fontsize=size, fontfamily="monospace", va="top")
                elif FONT_PROP:
                    fs = size + (2 if kind == "h1" else 1 if kind in ("h2", "h3") else 0)
                    ax.text(margin_x, y, text, fontsize=fs, fontproperties=FONT_PROP, va="top")
                y -= line_h.get(kind, 0.22)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    os.makedirs(DESKTOP, exist_ok=True)
    blocks = parse_md(MD_PATH)
    render_pdf(blocks, OUT_PDF)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
