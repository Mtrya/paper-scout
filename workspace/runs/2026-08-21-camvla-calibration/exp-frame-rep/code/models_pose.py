"""Policy network with optional auxiliary conditioning input (camera pose / embodiment id).

Identical conv stack and MLP width as `models.Policy`; when `extra_dim > 0` the flattened
conv features (64*8*8) are concatenated with the conditioning vector before the first MLP
layer. For `extra_dim == 0` the layer construction order is identical to `models.Policy`,
so under the same `torch.manual_seed` the weights are bit-identical -- the 'base' head here
is the same network as experiment B.
"""
from __future__ import annotations

import torch
from torch import nn


class PolicyX(nn.Module):
    def __init__(self, out_dim: int = 3, extra_dim: int = 0):
        super().__init__()
        self.extra_dim = extra_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8 + extra_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv(x)
        if self.extra_dim:
            assert extra is not None and extra.shape[1] == self.extra_dim
            x = torch.cat([x, extra], dim=1)
        return self.head(x)
