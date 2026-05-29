#!/usr/bin/env python3
"""Compare Hazy | V1 | Plugin-A | Plugin-B | Plugin-C (frozen V1 backbone)."""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_rwkv import DehazeRWKV_Real
from model_rwkv_plugin import DehazeRWKV_V1_Plugin

V1_CKPT = os.path.join(ROOT, "checkpoints/rwkv/rwkv_dehaze_epoch_50.pth")
PLUGIN_DIR = os.path.join(ROOT, "checkpoints/rwkv_plugin_v2")
LEGACY_DIR = os.path.join(ROOT, "checkpoints/rwkv_plugin")
# Best epochs from 128px sweep (epoch 29 overfits / collapses for A,B,C,D v1)
DEFAULT_EPOCHS = {"A": 0, "B": 14, "C": 14, "D": 14}


def load_v1(device):
    m = DehazeRWKV_Real(32, 2).to(device)
    sd = torch.load(V1_CKPT, map_location=device)
    if list(sd.keys())[0].startswith("module."):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.eval()


def load_plugin(name, epoch, device, plugin_dir, run_name=""):
    m = DehazeRWKV_V1_Plugin(plugin=name).to(device)
    m.load_v1_backbone(V1_CKPT, device)
    tag = f"_{run_name}" if run_name else ""
    folder = f"plugin_{name}{tag}"
    ckpt = os.path.join(plugin_dir, folder, f"plugin_{name}_epoch_{epoch}.pth")
    if not os.path.isfile(ckpt):
        return None
    m.load_plugin_weights(ckpt, device)
    return m.eval()


def label_row(row, labels, tile_w):
    arr = (row.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for i, lab in enumerate(labels):
        x0 = i * tile_w + 4
        draw.rectangle([x0, 2, x0 + len(lab) * 7 + 8, 20], fill=(0, 0, 0))
        draw.text((x0 + 4, 3), lab, fill=(255, 255, 255), font=font)
    return torch.tensor(np.array(pil).transpose(2, 0, 1) / 255.0).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, default=None, help="Same epoch for all plugins")
    p.add_argument("--plugin_dir", default=None)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--only", default="", help="Comma-separated plugins to show, e.g. A,B")
    p.add_argument("--run_name", default="", help="Checkpoint subdir tag, e.g. template or full")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "outputs/preview_plugins"))
    args = p.parse_args()
    plugin_dir = args.plugin_dir or PLUGIN_DIR
    if not os.path.isdir(plugin_dir):
        plugin_dir = LEGACY_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    v1 = load_v1(device)
    plugins = {}
    only = [x.strip().upper() for x in args.only.split(",") if x.strip()]
    names = only if only else ["A", "B", "C", "D"]
    for name in names:
        ep = args.epoch if args.epoch is not None else DEFAULT_EPOCHS.get(name, 0)
        m = load_plugin(name, ep, device, plugin_dir, args.run_name)
        if m is not None:
            plugins[name] = m
            print(f"loaded plugin {name} epoch {ep} from {plugin_dir}")
        else:
            print(f"skip plugin {name} (no ckpt)")

    paths = sorted(
        x for x in glob.glob(os.path.join(ROOT, "test_images", "*.*"))
        if x.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    with torch.no_grad():
        for path in paths:
            name = os.path.basename(path)
            arr = np.array(Image.open(path).convert("RGB").resize((args.size, args.size)), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            tiles = [t, torch.clamp(v1(t), 0, 1)]
            labs = ["Hazy", "V1"]
            for key, lab in [("A", "Plg-A"), ("B", "Plg-B"), ("C", "Plg-C"), ("D", "Plg-D")]:
                if key in plugins:
                    tiles.append(torch.clamp(plugins[key](t), 0, 1))
                    labs.append(lab)

            row = label_row(torch.cat(tiles, dim=3), labs, args.size)
            save_image(row, os.path.join(args.out_dir, name))
            print("saved", name)

    print("Done ->", args.out_dir)


if __name__ == "__main__":
    main()
