"""Policy network: small CNN on 64x64 RGB -> 3-dim delta action.

Same architecture for both heads; only the output-frame convention differs:
- 'base': output is a base-frame delta, applied directly.
- 'cam':  output is a camera-frame delta, synthesized into base frame at execution
          with the (estimated) extrinsics rotation.
"""
from __future__ import annotations

import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
