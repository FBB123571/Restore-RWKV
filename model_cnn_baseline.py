"""纯卷积去雾基线，用于与 Vision-RWKV 做消融对比。"""
import torch
import torch.nn as nn


class DehazeCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(),
    )
    self.decoder = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 3, kernel_size=3, padding=1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return self.decoder(self.encoder(x))
