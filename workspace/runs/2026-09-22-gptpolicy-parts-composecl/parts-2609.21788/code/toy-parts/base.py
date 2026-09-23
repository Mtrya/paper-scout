"""Base 策略:2 层 MLP(隐藏 128),输入 obs14,输出动作 chunk(C=10 × 3)。
BC 训练,冻结后供所有方法臂共用。推理时每 REPLAN=5 步重规划一次,
每次预测 C=10 步 chunk、执行前 5 步(π0.5 式 chunking:E ≤ C)。
"""

import numpy as np
import torch
import torch.nn as nn

import expert
import env as E

CHUNK = expert.CHUNK
REPLAN = 5
OBS_DIM = E.N_OBS            # 14
ACT_DIM = E.N_ACTION * CHUNK  # 30


class MLPChunkPolicy(nn.Module):
    def __init__(self, in_dim=OBS_DIM, out_dim=ACT_DIM, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim), nn.Tanh(),   # 动作 ∈ [-1,1]
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def chunk(self, obs14):
        x = torch.as_tensor(obs14, dtype=torch.float32).unsqueeze(0)
        return self.forward(x).squeeze(0).numpy().reshape(CHUNK, E.N_ACTION)


def train_base(X, Y, W=None, seed=0, steps=3000, lr=1e-3, batch=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pol = MLPChunkPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y)
    Wt = torch.as_tensor(W) if W is not None else torch.ones(len(Xt))
    n = len(Xt)
    for it in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, size=batch))
        r = (pol(Xt[idx]) - Yt[idx]) ** 2
        loss = (r.mean(dim=1) * Wt[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pol


def load_or_train_base(seed=0, cache_dir="cache"):
    """单一共享 base:所有方法臂、所有 RL 种子共用(只训一次,seed=0),
    消除 base 重训带来的系统性偏差抖动;种子只控制环境采样与 RL 随机性。"""
    import os
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "base_shared.pt")
    if os.path.exists(path):
        pol = MLPChunkPolicy()
        pol.load_state_dict(torch.load(path, weights_only=True))
        return pol
    X, Y, W, info = expert.gen_bc_dataset(seed=0)
    pol = train_base(X, Y, W, seed=0)
    torch.save(pol.state_dict(), path)
    return pol
