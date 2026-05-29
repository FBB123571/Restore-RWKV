"""Depth prior utilities for Direction C (UDPNet-style, offline-first)."""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


_depth_model = None


def pseudo_depth_from_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Fallback depth in [0,1]: darker regions treated as nearer (DCP-inspired)."""
    gray = rgb.mean(dim=1, keepdim=True)
    dark = 1.0 - gray
    return dark / (dark.amax(dim=(2, 3), keepdim=True) + 1e-6)


def load_depth_anything(device: torch.device):
    """Try loading Depth Anything V2 Small via torch.hub (optional)."""
    global _depth_model
    if _depth_model is not None:
        return _depth_model
    try:
        model = torch.hub.load("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        _depth_model = model
        return model
    except Exception:
        return None


@torch.no_grad()
def estimate_depth(rgb: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    rgb: [B,3,H,W] in [0,1]
    returns depth: [B,1,H,W] in [0,1]
    """
    if device is None:
        device = rgb.device
    model = load_depth_anything(device)
    if model is None:
        return pseudo_depth_from_rgb(rgb)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (rgb - mean) / std
    h, w = x.shape[-2:]
    if max(h, w) > 518:
        scale = 518 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        x_in = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    else:
        x_in = x
    try:
        depth = model(x_in)
        if isinstance(depth, (list, tuple)):
            depth = depth[-1]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        depth = (depth - depth.amin(dim=(2, 3), keepdim=True)) / (
            depth.amax(dim=(2, 3), keepdim=True) - depth.amin(dim=(2, 3), keepdim=True) + 1e-6
        )
        return depth
    except Exception:
        return pseudo_depth_from_rgb(rgb)


def depth_cache_path(cache_dir: str, image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(cache_dir, f"{base}.npy")


def load_cached_depth(
    cache_dir: str, image_path: str, size: int, device: torch.device
) -> torch.Tensor:
    path = depth_cache_path(cache_dir, image_path)
    if os.path.isfile(path):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        t = torch.from_numpy(arr).unsqueeze(0).to(device)
        if t.shape[-1] != size or t.shape[-2] != size:
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return t.clamp(0, 1)
    return None


class DepthGuidedAttention(nn.Module):
    """Lightweight DGAM (UDPNet-inspired channel gate)."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if depth.shape[-2:] != feat.shape[-2:]:
            depth = F.interpolate(depth, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return feat * self.gate(torch.cat([feat, depth], dim=1)) + feat


class DepthPriorFusion(nn.Module):
    """Lightweight DPFM: fuse multi-scale depth cues into features."""

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if depth.shape[-2:] != feat.shape[-2:]:
            depth = F.interpolate(depth, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([feat, depth], dim=1)) + feat


def save_depth_npy(depth: torch.Tensor, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    d = depth.squeeze().detach().cpu().numpy()
    np.save(out_path, d)
