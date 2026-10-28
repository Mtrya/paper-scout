"""Visual flow-matching policy with a JAM-style foresight branch.

Backbone: CNN -> 6x6x64 spatial tokens -> 2 pre-LN transformer blocks.
Action expert: 2 joint transformer blocks over [visual tokens + action tokens];
visual tokens cannot attend to action tokens. x-prediction parameterization,
v = (a_hat - x)/(1-tau) at sampling, matching lowrank-flow.
Foresight head: 1 transformer block + linear readout per target kind.
Latent-role arms retarget the same action interface to a future global vector.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x, mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


class VisualPolicy(nn.Module):
    def __init__(self, dim=64, heads=4, chunk=4, act_dim=2, img_tokens=36):
        super().__init__()
        self.dim = dim
        self.chunk = chunk
        self.act_dim = act_dim
        self.img_tokens = img_tokens
        # CNN encoder: 48 -> 24 -> 12 -> 6
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, dim, 3, stride=2, padding=1), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, img_tokens, dim) * 0.02)
        self.backbone = nn.ModuleList([Block(dim, heads) for _ in range(2)])
        self.expert = nn.ModuleList([Block(dim, heads) for _ in range(2)])
        self.time_mlp = nn.Sequential(nn.Linear(64, dim), nn.GELU(), nn.Linear(dim, dim))
        self.act_in = nn.Linear(act_dim, dim)
        self.act_pos = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.head = nn.Linear(dim, act_dim)
        self.z_head = None  # built on demand for latent-role arms

    def encode(self, img):
        """img [B,3,48,48] -> tokens [B,36,dim]"""
        f = self.cnn(img)
        f = f.flatten(2).transpose(1, 2)
        x = f + self.pos
        for b in self.backbone:
            x = b(x)
        return x

    def forward(self, img, x_tau, tau):
        """x_tau [B,chunk,act_dim], tau [B] -> a_hat [B,chunk,act_dim]"""
        B = img.shape[0]
        v = self.encode(img)
        te = self.time_mlp(timestep_embedding(tau, 64))[:, None, :]
        a = self.act_in(x_tau) + self.act_pos + te
        x = torch.cat([v, a], dim=1)
        nv = v.shape[1]
        # visual tokens may not attend to action tokens
        mask = torch.zeros(nv + self.chunk, nv + self.chunk, dtype=torch.bool, device=img.device)
        mask[:nv, nv:] = True
        for b in self.expert:
            x = b(x, mask=mask)
        return self.head(x[:, nv:])

    def foresight_features(self, img):
        return self.encode(img)


class ForesightHead(nn.Module):
    """Small transformer trunk + linear readout to target dim."""

    def __init__(self, dim=64, target_dim=64, heads=4):
        super().__init__()
        self.block = Block(dim, heads)
        self.readout = nn.Linear(dim, target_dim)

    def forward(self, tokens):
        h = self.block(tokens)
        return self.readout(h), h


class RandomAnchor(nn.Module):
    """Frozen randomly-initialized CNN+transformer, JAM's random future anchor."""

    def __init__(self, dim=64, heads=4, seed=1234):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, dim, 3, stride=2, padding=1), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, 36, dim, generator=g) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(2)])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, img):
        f = self.cnn(img).flatten(2).transpose(1, 2)
        x = f + self.pos
        for b in self.blocks:
            x = b(x)
        return x  # [B,36,dim]


def pixel_tokens(img, patch=8):
    """VAE-direct analog: raw 8x8 RGB patches as tokens -> [B,36,192]"""
    B, C, H, W = img.shape
    p = F.unfold(img, kernel_size=patch, stride=patch)  # [B, C*patch*patch, 36]
    return p.transpose(1, 2)
