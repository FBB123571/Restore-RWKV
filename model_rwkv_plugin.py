"""
Frozen V1 backbone + lightweight plugins (Directions A/B/C).

Does NOT modify model_rwkv.py. Load V1 weights into backbone; only plugins are trained.
"""
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_utils import pseudo_depth_from_rgb
from model_rwkv import DehazeRWKV_Real

PluginName = Literal["A", "B", "C", "D"]


class FourierMix2D(nn.Module):
    """Frequency-domain amplitude gating (Direction A)."""

    def __init__(self, channels: int):
        super().__init__()
        self.amp_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dtype = x.dtype
        x32 = x.float()
        xf = torch.fft.rfft2(x32, norm="ortho")
        gate = self.amp_gate(
            F.interpolate(x32, size=xf.shape[-2:], mode="bilinear", align_corners=False)
        )
        xf = xf * gate.to(xf.dtype)
        out = torch.fft.irfft2(xf, s=(H, W), norm="ortho")
        return out.to(dtype)


# Cap plugin contribution so training cannot blow past V1 (scale was growing to ~1.75).
SCALE_MAX = 0.12
B_SCALE_MAX = 0.06  # high-freq branch is more sensitive; keep smaller than A
C_SCALE_MAX = 0.04  # depth gate must stay very small
D_SCALE_MAX = 0.04


def _bounded_scale(logit: torch.Tensor, cap: float = SCALE_MAX) -> torch.Tensor:
    return cap * torch.sigmoid(logit)


def split_freq(x: torch.Tensor):
    B, C, H, W = x.shape
    dtype = x.dtype
    x32 = x.float()
    xf = torch.fft.rfft2(x32, norm="ortho")
    mask = torch.zeros_like(xf)
    rh, rw = max(1, xf.shape[-2] // 8), max(1, xf.shape[-1] // 8)
    mask[..., :rh, :rw] = 1.0
    low = torch.fft.irfft2(xf * mask, s=(H, W), norm="ortho").to(dtype)
    high = torch.fft.irfft2(xf * (1.0 - mask), s=(H, W), norm="ortho").to(dtype)
    return low, high


class PluginA(nn.Module):
    """Direction A: Fourier mixing on backbone features (residual)."""

    def __init__(self, channels: int):
        super().__init__()
        self.fourier = FourierMix2D(channels)
        self.scale_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return feat + _bounded_scale(self.scale_logit) * self.fourier(feat)


class PluginB(nn.Module):
    """Direction B: mild high-frequency refine (zero-init last layer, small cap)."""

    def __init__(self, channels: int):
        super().__init__()
        self.high_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        last = self.high_refine[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # Start near identity (effective scale ~0.01)
        self.scale_logit = nn.Parameter(torch.tensor(-2.5))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        _, high = split_freq(feat)
        delta = self.high_refine(high)
        return feat + _bounded_scale(self.scale_logit, B_SCALE_MAX) * delta


class PluginC(nn.Module):
    """Direction C: tiny depth gate (zero-init, no heavy DGAM/DPFM stack)."""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 8, 4)
        self.depth_gate = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1),
        )
        last = self.depth_gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.scale_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if depth.shape[-2:] != feat.shape[-2:]:
            depth = F.interpolate(depth, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.depth_gate(depth)
        return feat + _bounded_scale(self.scale_logit, C_SCALE_MAX) * delta


class PluginD(nn.Module):
    """Direction D: minimal spatial residual (same spirit as B, zero-init)."""

    def __init__(self, channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        last = self.refine[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.scale_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return feat + _bounded_scale(self.scale_logit, D_SCALE_MAX) * self.refine(feat)


class DehazeRWKV_V1_Plugin(nn.Module):
    """
    Frozen DehazeRWKV_Real + optional single plugin before out_conv.
    With plugin=None and V1 weights, output matches V1 (plugin has no params).
    """

    def __init__(self, plugin: Optional[PluginName] = None, hidden_dim: int = 32, num_blocks: int = 2):
        super().__init__()
        self.plugin_name = plugin
        self.backbone = DehazeRWKV_Real(hidden_dim=hidden_dim, num_blocks=num_blocks)

        for p in self.backbone.parameters():
            p.requires_grad = False

        c = hidden_dim
        if plugin == "A":
            self.plugin = PluginA(c)
        elif plugin == "B":
            self.plugin = PluginB(c)
        elif plugin == "C":
            self.plugin = PluginC(c)
        elif plugin == "D":
            self.plugin = PluginD(c)
        else:
            self.plugin = None

    def trainable_parameters(self):
        if self.plugin is None:
            return []
        return list(self.plugin.parameters())

    def load_v1_backbone(self, ckpt_path: str, device: torch.device):
        state = torch.load(ckpt_path, map_location=device)
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        self.backbone.load_state_dict(state, strict=True)

    def load_plugin_weights(self, ckpt_path: str, device: torch.device):
        state = torch.load(ckpt_path, map_location=device)
        cleaned = {}
        for k, v in state.items():
            k = k.replace("module.", "")
            if k.startswith("plugin."):
                k = k[len("plugin.") :]
            if k == "scale":
                val = float(v.reshape(-1)[0].item())
                if val > SCALE_MAX:
                    cleaned["scale_logit"] = torch.tensor(0.0)
                else:
                    val = max(1e-4, min(val, SCALE_MAX - 1e-4))
                    cleaned["scale_logit"] = torch.log(torch.tensor(val / (SCALE_MAX - val)))
                continue
            cleaned[k] = v
        if self.plugin is None:
            raise ValueError("No plugin module to load")
        self.plugin.load_state_dict(cleaned, strict=False)

    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.backbone.in_conv(x)
        for block in self.backbone.blocks:
            feat = block(feat)

        if self.plugin is not None:
            if self.plugin_name == "C":
                if depth is None:
                    depth = pseudo_depth_from_rgb(x)
                feat = self.plugin(feat, depth)
            else:
                feat = self.plugin(feat)

        return self.backbone.sigmoid(self.backbone.out_conv(feat) + x)
