"""Train plugins on frozen V1 backbone (128x128, MSE — same recipe as V1)."""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from model_rwkv import DehazeRWKV_Real
from model_rwkv_plugin import DehazeRWKV_V1_Plugin


class DehazeDataset128(Dataset):
    def __init__(self, data_root, size=128, max_samples=None):
        self.size = size
        self.hazy_dir = os.path.join(data_root, "archive(1)", "hazy")
        self.clear_dir = os.path.join(data_root, "archive(1)", "clear")
        self.hazy_paths = sorted(glob.glob(os.path.join(self.hazy_dir, "*.jpg"))) + sorted(
            glob.glob(os.path.join(self.hazy_dir, "*.png"))
        )
        self.clear_paths = sorted(glob.glob(os.path.join(self.clear_dir, "*.jpg"))) + sorted(
            glob.glob(os.path.join(self.clear_dir, "*.png"))
        )
        if max_samples is not None and max_samples > 0:
            self.hazy_paths = self.hazy_paths[:max_samples]

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        hazy_path = self.hazy_paths[idx]
        filename = os.path.basename(hazy_path)
        clear_path = os.path.join(self.clear_dir, filename)
        if not os.path.exists(clear_path):
            clear_path = self.clear_paths[idx % len(self.clear_paths)]
        hazy = Image.open(hazy_path).convert("RGB").resize((self.size, self.size))
        clear = Image.open(clear_path).convert("RGB").resize((self.size, self.size))
        h = torch.tensor(np.array(hazy).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        c = torch.tensor(np.array(clear).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        return h, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plugin", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--w_anchor", type=float, default=None, help="Keep output close to frozen V1")
    p.add_argument("--v1_ckpt", default="./checkpoints/rwkv/rwkv_dehaze_epoch_50.pth")
    p.add_argument("--data_root", default="./data/dehaze_data")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save_dir", default="./checkpoints/rwkv_plugin_v2")
    p.add_argument("--max_samples", type=int, default=None, help="Quick template: e.g. 512")
    p.add_argument("--run_name", default="", help="Subdir tag, e.g. template or full")
    args = p.parse_args()
    if args.w_anchor is None:
        args.w_anchor = 0.5 if args.plugin == "B" else 0.3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))

    model = DehazeRWKV_V1_Plugin(plugin=args.plugin)
    model.load_v1_backbone(args.v1_ckpt, device)
    model = model.to(device)

    v1_ref = DehazeRWKV_Real(32, 2).to(device)
    sd = torch.load(args.v1_ckpt, map_location=device)
    if list(sd.keys())[0].startswith("module."):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    v1_ref.load_state_dict(sd)
    v1_ref.eval()
    for p in v1_ref.parameters():
        p.requires_grad = False

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Plugin {args.plugin} | trainable params: {sum(x.numel() for x in trainable)} | w_anchor={args.w_anchor}")

    loader = DataLoader(
        DehazeDataset128(args.data_root, args.size, max_samples=args.max_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    optimizer = AdamW(trainable, lr=args.lr)
    criterion = nn.MSELoss()

    tag = f"_{args.run_name}" if args.run_name else ""
    save_dir = os.path.join(args.save_dir, f"plugin_{args.plugin}{tag}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"save_dir={save_dir} | samples={len(loader.dataset)} | epochs={args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        for i, (hazy, clear) in enumerate(loader):
            hazy, clear = hazy.to(device, non_blocking=True), clear.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(hazy)
            with torch.no_grad():
                v1_out = v1_ref(hazy)
            loss = criterion(out, clear) + args.w_anchor * criterion(out, v1_out)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Epoch {epoch} Step {i} | loss: {loss.item():.4f}")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            prefix = "plugin."
            plugin_state = {
                k.replace("module.", ""): v
                for k, v in state.items()
                if k.replace("module.", "").startswith(prefix)
            }
            path = os.path.join(save_dir, f"plugin_{args.plugin}_epoch_{epoch}.pth")
            torch.save(plugin_state, path)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
