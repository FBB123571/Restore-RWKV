#!/usr/bin/env python3
"""PSNR / SSIM: V1 vs plugins A/B/C/D on held-out pairs (hazy, clear)."""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from losses import ssim_loss
from model_rwkv import DehazeRWKV_Real
from model_rwkv_plugin import DehazeRWKV_V1_Plugin

V1_CKPT = os.path.join(ROOT, "checkpoints/rwkv/rwkv_dehaze_epoch_50.pth")
DATA_ROOT = os.path.join(ROOT, "data/dehaze_data")


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    return 1.0 - ssim_loss(pred, target).item()


def resolve_clear_path(hazy_path: str, clear_dir: str, clear_paths: list, idx: int) -> str:
    name = os.path.basename(hazy_path)
    cp = os.path.join(clear_dir, name)
    if os.path.isfile(cp):
        return cp
    stem = name.split("_")[0].split(".")[0]
    for ext in (".png", ".jpg"):
        cp2 = os.path.join(clear_dir, stem + ext)
        if os.path.isfile(cp2):
            return cp2
    return clear_paths[idx % len(clear_paths)]


def list_pairs(data_root: str, max_samples: int, offset: int):
    hazy_dir = os.path.join(data_root, "archive(1)", "hazy")
    clear_dir = os.path.join(data_root, "archive(1)", "clear")
    hazy_paths = sorted(glob.glob(os.path.join(hazy_dir, "*.jpg"))) + sorted(
        glob.glob(os.path.join(hazy_dir, "*.png"))
    )
    clear_paths = sorted(glob.glob(os.path.join(clear_dir, "*.jpg"))) + sorted(
        glob.glob(os.path.join(clear_dir, "*.png"))
    )
    pairs = []
    for i, hp in enumerate(hazy_paths[offset:], start=offset):
        cp = resolve_clear_path(hp, clear_dir, clear_paths, i)
        if os.path.isfile(cp):
            pairs.append((hp, cp))
        if len(pairs) >= max_samples:
            break
    return pairs


def load_pair(hazy_path, clear_path, size, device):
    hazy = Image.open(hazy_path).convert("RGB").resize((size, size))
    clear = Image.open(clear_path).convert("RGB").resize((size, size))
    h = torch.tensor(np.array(hazy).transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    c = torch.tensor(np.array(clear).transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    return h, c


def load_v1(device):
    m = DehazeRWKV_Real(32, 2).to(device)
    sd = torch.load(V1_CKPT, map_location=device)
    if list(sd.keys())[0].startswith("module."):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.eval()


def load_plugin(name, ckpt_path, device):
    m = DehazeRWKV_V1_Plugin(name).to(device)
    m.load_v1_backbone(V1_CKPT, device)
    m.load_plugin_weights(ckpt_path, device)
    return m.eval()


def eval_model(model, pairs, size, device, needs_plugin=False):
    psnrs, ssims = [], []
    with torch.no_grad():
        for hp, cp in pairs:
            h, c = load_pair(hp, cp, size, device)
            if needs_plugin:
                out = model(h)
            else:
                out = model(h)
            out = out.clamp(0, 1)
            psnrs.append(psnr(out, c))
            ssims.append(ssim_metric(out, c))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.std(psnrs)), float(np.std(ssims))


def resolve_ckpt(plugin_dir, name, run_name, epoch):
    tag = f"_{run_name}" if run_name else ""
    p = os.path.join(plugin_dir, f"plugin_{name}{tag}", f"plugin_{name}_epoch_{epoch}.pth")
    return p if os.path.isfile(p) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--offset", type=int, default=12000, help="Hold-out tail of sorted list")
    p.add_argument("--plugin_dir", default=os.path.join(ROOT, "checkpoints/rwkv_plugin_template"))
    p.add_argument("--run_name", default="template")
    p.add_argument("--epoch", type=int, default=2)
    p.add_argument("--a_ckpt", default=os.path.join(ROOT, "checkpoints/rwkv_plugin/plugin_A/plugin_A_epoch_0.pth"))
    p.add_argument("--out_csv", default=os.path.join(ROOT, "data/result/plugin_metrics.csv"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = list_pairs(DATA_ROOT, args.max_samples, args.offset)
    if not pairs:
        print("No hazy/clear pairs found.")
        sys.exit(1)

    print(f"Eval on {len(pairs)} pairs @ {args.size}px (offset={args.offset})")

    rows = []
    v1 = load_v1(device)
    m_p, m_s, s_p, s_s = eval_model(v1, pairs, args.size, device)
    rows.append(("V1", m_p, m_s, s_p, s_s, "-"))

    if os.path.isfile(args.a_ckpt):
        pa = load_plugin("A", args.a_ckpt, device)
        m_p, m_s, s_p, s_s = eval_model(pa, pairs, args.size, device, True)
        rows.append(("Plg-A (ep0)", m_p, m_s, s_p, s_s, args.a_ckpt))

    for name in ["B", "C", "D"]:
        ck = resolve_ckpt(args.plugin_dir, name, args.run_name, args.epoch)
        if ck is None:
            print(f"skip {name}: no ckpt")
            continue
        pm = load_plugin(name, ck, device)
        m_p, m_s, s_p, s_s = eval_model(pm, pairs, args.size, device, True)
        rows.append((f"Plg-{name} ({args.run_name} ep{args.epoch})", m_p, m_s, s_p, s_s, ck))

    h_psnr, h_ssim = [], []
    with torch.no_grad():
        for hp, cp in pairs:
            h, c = load_pair(hp, cp, args.size, device)
            h_psnr.append(psnr(h, c))
            h_ssim.append(ssim_metric(h, c))
    rows.append(("Hazy (input)", float(np.mean(h_psnr)), float(np.mean(h_ssim)), 0, 0, "-"))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "PSNR_mean", "SSIM_mean", "PSNR_std", "SSIM_std", "checkpoint"])
        w.writerows(rows)

    print("\n| Model | PSNR ↑ | SSIM ↑ | ΔPSNR vs V1 |")
    print("|-------|--------|--------|-------------|")
    v1_psnr = rows[0][1]
    for r in rows:
        delta = r[1] - v1_psnr if r[0] != "Hazy (input)" else float("nan")
        d_str = f"{delta:+.3f}" if r[0] not in ("V1", "Hazy (input)") else "-"
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.4f} | {d_str} |")
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
