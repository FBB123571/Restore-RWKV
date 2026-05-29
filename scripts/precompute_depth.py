#!/usr/bin/env python3
"""Precompute depth maps for training (Direction C, offline cache)."""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from depth_utils import depth_cache_path, estimate_depth, save_depth_npy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=os.path.join(ROOT, "data/dehaze_data/archive(1)"),
        help="Folder containing hazy/ and clear/",
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(ROOT, "data/dehaze_data/depth_cache"),
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--split", choices=["hazy", "clear", "both"], default="hazy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = ["hazy", "clear"] if args.split == "both" else [args.split]

    paths = []
    for sp in splits:
        d = os.path.join(args.data_root, sp)
        paths.extend(sorted(glob.glob(os.path.join(d, "*.jpg"))))
        paths.extend(sorted(glob.glob(os.path.join(d, "*.png"))))

    print(f"Precomputing depth for {len(paths)} images -> {args.cache_dir}")
    os.makedirs(args.cache_dir, exist_ok=True)

    for p in tqdm(paths):
        out = depth_cache_path(args.cache_dir, p)
        if os.path.isfile(out):
            continue
        img = Image.open(p).convert("RGB").resize((args.size, args.size))
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        depth = estimate_depth(t, device)
        save_depth_npy(depth, out)

    print("Done.")


if __name__ == "__main__":
    main()
