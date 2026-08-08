"""Dense-grid carrier: multi-channel 48x48 encoding + independent-head CNN (~3M).

Per-cell softmax over 6 classes (empty, food, s0 head, s0 body, s1 head,
s1 body) decoded independently per cell; global heads predict per-snake
heading (4-way) and survival (2-way). The recurrent object is the raw argmax
class grid — no cross-cell constraint exists at decode time.

Input channels (27):
  0 food | 1-4 s0/s1 body/head | 5-8 s0 heading@head | 9-12 s1 heading@head |
  13-14 alive broadcast | 15-19 s0 action one-hot | 20-24 s1 action one-hot |
  25 spawn cells | 26 tick/512 broadcast
"""
import numpy as np
import torch
import torch.nn as nn

from codec import MAX_SLOTS
from snake_engine import ARENA, MAX_BODY

N_CLASS = 6
EMPTY, FOOD, S0_HEAD, S0_BODY, S1_HEAD, S1_BODY = range(6)
IN_CH = 27


def fields_to_grid(fields, actions, spawns):
    """Canonical fields + inputs -> float32[IN_CH,48,48]."""
    g = np.zeros((IN_CH, ARENA, ARENA), dtype=np.float32)
    for c in fields["food"]:
        if c >= 0:
            g[FOOD].flat[c] = 1.0
    for i in range(2):
        sl = fields["slots"][i]
        if sl["inactive"]:
            continue
        body = [c for c in sl["body"][:sl["length"]] if c >= 0]
        if body:
            g[S0_HEAD + 2 * i].flat[body[0]] = 1.0
            for c in body[1:]:
                g[S0_BODY + 2 * i].flat[c] = 1.0
            if sl["heading"] >= 0:
                g[5 + 4 * i + sl["heading"]].flat[body[0]] = 1.0
        g[13 + i] = float(sl["alive"])
        a = int(actions[i]) if i < len(actions) else 0
        g[15 + 5 * i + a] = 1.0
    for c in spawns:
        if c >= 0:
            g[25].flat[c] = 1.0
    g[26] = min(fields["tick"], 4096) / 512.0
    return g


def grid_to_carry(cls, headings, alives):
    """Recurrent carry for the dense model: class grid + per-snake attrs."""
    return dict(cls=cls, headings=headings, alives=alives)


def carry_to_grid(carry, actions, spawns, tick):
    """Re-encode a predicted carry (argmax grid + attrs) as model input."""
    cls, headings, alives = carry["cls"], carry["headings"], carry["alives"]
    g = np.zeros((IN_CH, ARENA, ARENA), dtype=np.float32)
    g[FOOD][cls == FOOD] = 1.0
    for i in range(2):
        head_ch, body_ch = S0_HEAD + 2 * i, S0_BODY + 2 * i
        g[head_ch][cls == head_ch] = 1.0
        g[body_ch][cls == body_ch] = 1.0
        heads = np.argwhere(cls == head_ch)
        if len(heads):
            y, x = heads[0]
            g[5 + 4 * i + int(headings[i])][y, x] = 1.0
        g[13 + i] = float(alives[i])
        a = int(actions[i]) if i < len(actions) else 0
        g[15 + 5 * i + a] = 1.0
    for c in spawns:
        if c >= 0:
            g[25].flat[c] = 1.0
    g[26] = min(tick, 4096) / 512.0
    return g


def fields_to_target(fields):
    """Ground-truth per-cell class grid + attrs for training/eval."""
    cls = np.zeros((ARENA, ARENA), dtype=np.int64)
    for c in fields["food"]:
        if c >= 0:
            cls.flat[c] = FOOD
    headings = np.zeros(2, dtype=np.int64)
    alives = np.zeros(2, dtype=np.int64)
    for i in range(2):
        sl = fields["slots"][i]
        headings[i] = max(sl["heading"], 0)
        alives[i] = sl["alive"]
        if sl["inactive"]:
            continue
        body = [c for c in sl["body"][:sl["length"]] if c >= 0]
        if body:
            cls.flat[body[0]] = S0_HEAD + 2 * i
            for c in body[1:]:
                cls.flat[c] = S0_BODY + 2 * i
    return cls, headings, alives


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = nn.GroupNorm(8, c)
        self.n2 = nn.GroupNorm(8, c)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        return x + self.n2(self.c2(h))


class DenseCNN(nn.Module):
    def __init__(self, width=160, blocks=6):
        super().__init__()
        self.stem = nn.Conv2d(IN_CH, width, 3, padding=1)
        self.blocks = nn.ModuleList(ResBlock(width) for _ in range(blocks))
        self.cell_head = nn.Conv2d(width, N_CLASS, 1)
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(width, 256), nn.GELU())
        self.head_head = nn.Linear(256, 2 * 4)   # per-snake heading
        self.alive_head = nn.Linear(256, 2 * 2)  # per-snake survival

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        cell = self.cell_head(h)
        g = self.glob(h)
        return cell, self.head_head(g).view(-1, 2, 4), self.alive_head(g).view(-1, 2, 2)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = DenseCNN()
    print(f"params: {count_params(m)/1e6:.2f}M")
