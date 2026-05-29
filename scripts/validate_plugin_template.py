#!/usr/bin/env python3
"""Numeric smoke test after template train — pass before full-data training."""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_rwkv import DehazeRWKV_Real
from model_rwkv_plugin import DehazeRWKV_V1_Plugin

V1_CKPT = os.path.join(ROOT, "checkpoints/rwkv/rwkv_dehaze_epoch_50.pth")


def load_img(path, size, device):
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plugin", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--epoch", type=int, default=2)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--ckpt_dir", default=os.path.join(ROOT, "checkpoints/rwkv_plugin_template"))
    p.add_argument("--run_name", default="template")
    p.add_argument("--max_diff_v1", type=float, default=None)
    p.add_argument("--min_std_ratio", type=float, default=0.85, help="out_std >= v1_std * ratio")
    args = p.parse_args()
    if args.max_diff_v1 is None:
        args.max_diff_v1 = 0.05 if args.plugin in ("C", "D") else 0.12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"_{args.run_name}" if args.run_name else ""
    ckpt = os.path.join(args.ckpt_dir, f"plugin_{args.plugin}{tag}", f"plugin_{args.plugin}_epoch_{args.epoch}.pth")
    if not os.path.isfile(ckpt):
        print(f"FAIL: missing checkpoint {ckpt}")
        sys.exit(1)

    v1 = DehazeRWKV_Real(32, 2).to(device)
    sd = torch.load(V1_CKPT, map_location=device)
    v1.load_state_dict(sd)
    v1.eval()

    m = DehazeRWKV_V1_Plugin(args.plugin).to(device)
    m.load_v1_backbone(V1_CKPT, device)
    m.load_plugin_weights(ckpt, device)
    m.eval()

    paths = sorted(
        x for x in glob.glob(os.path.join(ROOT, "test_images", "*.*"))
        if x.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    ok = True
    print(f"=== validate Plugin {args.plugin} epoch {args.epoch} @ {args.size}px ===")
    with torch.no_grad():
        for path in paths:
            x = load_img(path, args.size, device)
            o_v1 = v1(x)
            o = m(x)
            if not torch.isfinite(o).all():
                print(f"FAIL {os.path.basename(path)}: NaN/Inf")
                ok = False
                continue
            std_v1, std_o = o_v1.std().item(), o.std().item()
            diff = (o - o_v1).abs().mean().item()
            name = os.path.basename(path)
            flag = "OK" if diff <= args.max_diff_v1 and std_o >= std_v1 * args.min_std_ratio else "WARN"
            if flag == "WARN":
                ok = False
            print(f"  [{flag}] {name} std_v1={std_v1:.3f} std_out={std_o:.3f} diff_v1={diff:.4f}")

    if ok:
        print("PASS — safe to run full training.")
        sys.exit(0)
    print("FAIL — do NOT run full training; fix template/hyperparams first.")
    sys.exit(1)


if __name__ == "__main__":
    main()
