"""诊断 beta=0 的 msign 随机游走: 分窗平均 ||g||_nuc 随时间的剖面。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from muon_floor_probe import make_problem, batch_grad, msign

A, Y, W_star, L = make_problem(seed=11)
eta = 0.02 / L

for T in [4000, 20000]:
    for seed in range(5):
        rng = np.random.default_rng(seed + 7717)
        W = np.zeros_like(W_star); m = np.zeros_like(W_star)
        gs = []
        for t in range(1, T + 1):
            g = batch_grad(A, Y, W, 1, rng)
            gs.append(float(np.linalg.norm(g, "nuc")))
            m = g  # beta=0
            W = W - eta * msign(m)
        gs = np.array(gs)
        win = T // 10
        profile = [round(float(gs[i*win:(i+1)*win].mean()), 3) for i in range(10)]
        dist = float(np.linalg.norm(W - W_star, "fro"))
        print(f"T={T:<6} seed={seed} windows={profile} final||W-W*||={dist:.3f}")
    print()
