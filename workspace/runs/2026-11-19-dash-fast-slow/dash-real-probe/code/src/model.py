"""Lightweight flow-matching action head: cross-only blocks over stale VLM states [+ fresh ResNet tokens].

Mirrors the DASH action head in structure (cross-attention + FFN per block, no self-attention, flow matching) at a
size that fits a single 4090, but the VLM is frozen Qwen3-VL-2B-Instruct and the fresh encoder is a from-scratch
ResNet-18 (not ResNet-152).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        r = x if self.down is None else self.down(x)
        return torch.relu(y + r)


class ResNet18Small(nn.Module):
    """From-scratch ResNet-18 trunk for 128x128 input; final map is 4x4."""

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.l1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.l2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.l3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.l4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.proj = nn.Conv2d(512, out_dim, 1)

    def forward(self, x):
        h = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        return self.proj(h)  # (B, out_dim, 4, 4)


class CrossBlock(nn.Module):
    def __init__(self, D: int, n_heads: int, ffn: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(D)
        self.ln_kv = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, ffn), nn.GELU(), nn.Linear(ffn, D))

    def forward(self, q, kv):
        q2, kv2 = self.ln_q(q), self.ln_kv(kv)
        q = q + self.attn(q2, kv2, kv2, need_weights=False)[0]
        return q + self.ff(self.ln_ff(q))


def sinusoidal(x: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    ang = x[:, None] * freqs[None] * (2 * math.pi)
    return torch.cat([ang.sin(), ang.cos()], -1)


class DashHead(nn.Module):
    def __init__(self, D=512, n_blocks=4, n_heads=8, horizon=8, act_dim=13, n_layers=4, n_vis=32,
                 vlm_dim=2048, state_dim=11, use_fresh=True, steps=4):
        super().__init__()
        assert n_blocks % n_layers == 0 or n_layers % n_blocks == 0
        self.D, self.H, self.A = D, horizon, act_dim
        self.n_blocks, self.n_layers, self.use_fresh, self.steps = n_blocks, n_layers, use_fresh, steps
        self.state_enc = nn.Linear(state_dim, D)
        self.act_enc = nn.Linear(act_dim, D)
        self.pos = nn.Parameter(torch.zeros(horizon, D))
        nn.init.normal_(self.pos, std=0.02)
        self.tau_mlp = nn.Sequential(nn.Linear(64, D), nn.SiLU(), nn.Linear(D, D))
        self.w_vis = nn.ModuleList([nn.Linear(vlm_dim, D, bias=False) for _ in range(n_layers)])
        self.w_txt = nn.ModuleList([nn.Linear(vlm_dim, D, bias=False) for _ in range(n_layers)])
        self.blocks = nn.ModuleList([CrossBlock(D, n_heads, 2 * D) for _ in range(n_blocks)])
        self.out = nn.Linear(D, act_dim)
        if use_fresh:
            self.resnet = ResNet18Small(D)
            self.w_fresh = nn.Linear(D, D, bias=False)
            self.cam_emb = nn.Parameter(torch.zeros(2, D))

    def layer_of_block(self, d: int) -> int:
        return min(self.n_layers - 1, int(d * self.n_layers // self.n_blocks))

    def encode(self, vis, txt, img):
        """vis (B,L,n_vis,2048) fp, txt (B,L,2048), img (B,2,3,128,128) or None -> list of KV per block."""
        B = vis.shape[0]
        fresh = None
        if self.use_fresh and img is not None:
            f = self.resnet(img.reshape(B * 2, 3, 128, 128)).flatten(2).transpose(1, 2)  # (2B, 16, D)
            f = f.reshape(B, 2, -1, self.D) + self.cam_emb[None, :, None, :]
            fresh = self.w_fresh(f.reshape(B, -1, self.D))
        kvs = []
        for d in range(self.n_blocks):
            l = self.layer_of_block(d)
            parts = [self.w_vis[l](vis[:, l]), self.w_txt[l](txt[:, l])[:, None, :]]
            if fresh is not None:
                parts.append(fresh)
            kvs.append(torch.cat(parts, 1))
        return kvs

    def velocity(self, kvs, state, a_tau, tau):
        q = torch.cat([self.state_enc(state)[:, None, :], self.act_enc(a_tau) + self.pos[None]], 1)
        q = q + self.tau_mlp(sinusoidal(tau, 64))[:, None, :]
        for blk, kv in zip(self.blocks, kvs):
            q = blk(q, kv)
        return self.out(q[:, 1:])

    def loss(self, kvs, state, act, rng=None):
        B = act.shape[0]
        eps = torch.randn_like(act)
        tau = torch.rand(B, 1, device=act.device)
        a_tau = tau[:, :, None] * act + (1 - tau[:, :, None]) * eps
        v = self.velocity(kvs, state, a_tau, tau[:, 0])
        return ((v - (act - eps)) ** 2).mean()

    @torch.no_grad()
    def sample(self, kvs, state, noise, steps=None):
        steps = steps or self.steps
        a = noise.clone()
        for i in range(steps):
            tau = torch.full((a.shape[0],), i / steps, device=a.device)
            a = a + self.velocity(kvs, state, a, tau) / steps
        return a
