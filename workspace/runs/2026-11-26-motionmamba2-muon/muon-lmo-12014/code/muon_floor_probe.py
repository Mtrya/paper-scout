"""Muon LMO 渐近梯度地板探针。

检验 2503.12645 / 苏剑林《Muon背后的优化理论(一)》动量上界的三个可分离预言:
  E[min_k ||g_k||_*] 的渐近地板
    ~ 2*rho*sigma*sqrt(1-beta)/sqrt(B)  (噪声项, U 形左支随 beta 下降)
    + 2*beta*L*eta/(1-beta)            (漂移项, U 形右支随 beta 上升)
  以及初值瞬态按 ~1/((1-beta)T) 冲刷。

设置: 随机二次型 f(W) = 0.5*||A W - Y||_F^2 / n,Lipschitz 常数可算;
Muon 更新: m <- beta*m + (1-beta)*g_tilde;  W <- W - eta * msign(m)。
msign = U V^T (SVD 极式符号), 谱范数下的对偶梯度。

用法: python3 muon_floor_probe.py [--quick]
输出: results_muon_floor.json + 终端表格
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def make_problem(n=64, d=32, cond=10.0, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.logspace(0.0, math.log10(cond), d)
    A = (U[:, :d] * s) @ V.T
    W_star = rng.standard_normal((d, d)) * 0.5
    Y = A @ W_star
    # f(W) = 0.5/n ||A W - Y||^2;  L = ||A^T A||_op / n
    L = float(np.linalg.norm(A.T @ A, 2) / n)
    return A, Y, W_star, L


def batch_grad(A, Y, W, B, rng):
    n = A.shape[0]
    idx = rng.integers(0, n, size=B)
    G = A[idx].T @ (A[idx] @ W - Y[idx]) / B
    return G


def msign(M, eps=1e-12):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U * (s > eps)) @ Vt


def run_momentum(A, Y, W_star, L, *, beta, B, eta, T, seed):
    rng = np.random.default_rng(seed + 7717)
    W = np.zeros_like(W_star)
    m = np.zeros_like(W_star)
    sigma = per_sample_noise_scale(A, Y, W_star, rng)
    min_gnuc = np.inf
    sum_gnuc = 0.0
    gs = []
    for t in range(1, T + 1):
        g = batch_grad(A, Y, W, B, rng)
        gnuc = float(np.linalg.norm(g, "nuc"))
        min_gnuc = min(min_gnuc, gnuc)
        sum_gnuc += gnuc
        gs.append(gnuc)
        m = beta * m + (1.0 - beta) * g
        W = W - eta * msign(m)
    return {
        "beta": beta, "B": B, "eta": eta, "T": T,
        "avg_gnuc": sum_gnuc / T,
        "min_gnuc": min_gnuc,
        "tail_avg_gnuc": float(np.mean(gs[-T // 10:])),
        "sigma": sigma,
        "L": L,
    }


def per_sample_noise_scale(A, Y, W_star, rng, n_samples=512):
    """估计 E||g_tilde - g||_2 在 B=1 处的量级 (rho=1, l2->l2)。"""
    n = A.shape[0]
    W = W_star
    g = A.T @ (A @ W - Y) / n
    tot = 0.0
    for _ in range(n_samples):
        i = rng.integers(0, n)
        gi = A[i:i + 1].T @ (A[i:i + 1] @ W - Y[i:i + 1])
        tot += float(np.linalg.norm(gi - g, 2))
    return tot / n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    T = 4000 if args.quick else 20000
    n_seeds = 3 if args.quick else 5
    A, Y, W_star, L = make_problem(seed=11)

    # 固定 eta 于理论平衡值量级: L*eta ~ 0.02, 让漂移项可测但不淹没噪声
    eta = 0.02 / L

    betas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
    batches = [1, 4, 16, 64]

    results = []
    for beta in betas:
        for B in ([1] if beta != 0.9 else batches):
            runs = [run_momentum(A, Y, W_star, L, beta=beta, B=B, eta=eta, T=T, seed=s)
                    for s in range(n_seeds)]
            avg = float(np.mean([r["tail_avg_gnuc"] for r in runs]))
            std = float(np.std([r["tail_avg_gnuc"] for r in runs]))
            results.append({"beta": beta, "B": B, "eta": eta, "T": T,
                            "tail_avg_gnuc_mean": avg, "tail_avg_gnuc_std": std,
                            "n_seeds": n_seeds, "L": L,
                            "sigma1": runs[0]["sigma"]})
            print(f"beta={beta:<6} B={B:<3} tail_avg||g||_*={avg:.4f} ± {std:.4f}")

    out = {"config": {"T": T, "n_seeds": n_seeds, "eta": eta, "L": L,
                      "betas": betas, "batches": batches},
           "results": results}
    # 初值冲刷: beta=0.95 vs 0.5, 记录前 1/4 段平均
    for beta in [0.5, 0.95]:
        r = run_momentum(A, Y, W_star, L, beta=beta, B=1, eta=eta, T=T, seed=3)
        quarter = T // 4
        early = float(np.mean([r["tail_avg_gnuc"]]))  # placeholder, rerun below
        # 重新记录整条轨迹的前段
        rng = np.random.default_rng(3 + 7717)
        W = np.zeros_like(W_star); m = np.zeros_like(W_star); gs = []
        for t in range(1, T + 1):
            g = batch_grad(A, Y, W, 1, rng)
            gs.append(float(np.linalg.norm(g, "nuc")))
            m = beta * m + (1.0 - beta) * g
            W = W - eta * msign(m)
        out.setdefault("washout", {})[str(beta)] = {
            "first_quarter_avg": float(np.mean(gs[:quarter])),
            "last_quarter_avg": float(np.mean(gs[-quarter:])),
        }
    p = Path(__file__).with_name("results_muon_floor.json")
    p.write_text(json.dumps(out, indent=1))
    print("written", p)


if __name__ == "__main__":
    main()
