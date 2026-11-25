"""Muon LMO 渐近噪声地板探针 v2。

v1 教训: 从 W=0 出发时 beta=0/B=1 的 msign 收敛极慢(方向被 B=1 噪声支配),
尾部均值是收敛瞬态伪影。v2 从 W* 附近出发, 直接测平稳噪声地板:
  floor(beta, B) ~ 2*rho*sigma*sqrt(1-beta)/sqrt(B) + 2*beta*L*eta/(1-beta)
右支(大 beta)漂移项应被定量复现; 左支随 sqrt(1-beta)/sqrt(B) 下降。
附带测收敛瞬态(W 从 0 出发)作为独立观察记录。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from muon_floor_probe import make_problem, batch_grad, msign


def run(A, Y, W_star, L, *, beta, B, eta, T, seed, W_init_noise=0.05):
    rng = np.random.default_rng(seed + 7717)
    W = W_star + W_init_noise * rng.standard_normal(W_star.shape)
    m = np.zeros_like(W_star)
    gs, dists = [], []
    for t in range(1, T + 1):
        g = batch_grad(A, Y, W, B, rng)
        gs.append(float(np.linalg.norm(g, "nuc")))
        dists.append(float(np.linalg.norm(W - W_star, "fro")))
        m = beta * m + (1.0 - beta) * g
        W = W - eta * msign(m)
    gs = np.array(gs)
    tail = gs[-T // 5:]
    return {"beta": beta, "B": B, "eta": eta, "T": T,
            "tail_avg_gnuc": float(tail.mean()),
            "tail_std_gnuc": float(tail.std()),
            "tail_avg_dist": float(np.mean(dists[-T // 5:])),
            "first_tenth_avg_gnuc": float(gs[:T // 10].mean())}


def sigma_l2(A, Y, W_star, rng, n_samples=1024):
    """E||g_tilde - g||_2 @ W* (B=1), 作为 sigma 的 l2 口径。"""
    n = A.shape[0]
    g = A.T @ (A @ W_star - Y) / n
    tot = 0.0
    for _ in range(n_samples):
        i = rng.integers(0, n)
        gi = A[i:i + 1].T @ (A[i:i + 1] @ W_star - Y[i:i + 1])
        tot += float(np.linalg.norm(gi - g, 2))
    return tot / n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    T = 6000 if args.quick else 15000
    n_seeds = 3 if args.quick else 5
    A, Y, W_star, L = make_problem(seed=11)
    eta = 0.02 / L
    sig = sigma_l2(A, Y, W_star, np.random.default_rng(99))
    print(f"L={L:.3f} eta={eta:.4f} sigma_l2(B=1)={sig:.4f}")

    betas = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995]
    results = []
    for beta in betas:
        for B in ([1, 4, 16, 64] if beta in (0.0, 0.9) else [1]):
            runs = [run(A, Y, W_star, L, beta=beta, B=B, eta=eta, T=T, seed=s)
                    for s in range(n_seeds)]
            mean = float(np.mean([r["tail_avg_gnuc"] for r in runs]))
            sem = float(np.std([r["tail_avg_gnuc"] for r in runs]) / np.sqrt(n_seeds))
            dmean = float(np.mean([r["tail_avg_dist"] for r in runs]))
            drift = 2 * beta * L * eta / (1 - beta)
            results.append({"beta": beta, "B": B, "tail_avg_gnuc": mean,
                            "sem": sem, "tail_avg_dist": dmean,
                            "drift_term": drift})
            print(f"beta={beta:<6} B={B:<3} floor={mean:.4f}±{sem:.4f} "
                  f"dist={dmean:.4f} drift_pred={drift:.4f}")

    # 左支预言: floor(beta,B=1) - floor(0,B=1) 应 ~ sqrt(1-beta) 比例;
    # B 标度: floor(0,B)/floor(0,1) 应 ~ 1/sqrt(B)
    f0 = next(r["tail_avg_gnuc"] for r in results if r["beta"] == 0.0 and r["B"] == 1)
    out = {"config": {"T": T, "n_seeds": n_seeds, "eta": eta, "L": L,
                      "sigma_l2_B1": sig, "betas": betas},
           "results": results,
           "sqrt_1mbeta_pred": {str(b): f0 * np.sqrt(1 - b) for b in betas}}
    p = Path(__file__).with_name("results_muon_floor_v2.json")
    p.write_text(json.dumps(out, indent=1))
    print("written", p)


if __name__ == "__main__":
    main()
