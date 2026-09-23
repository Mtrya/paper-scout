"""绘制结果图(读取 results/*.json):
  fig_training.png : 训练曲线(评估快照)— 全任务成功率 + 瓶颈段 σ3 成功率,4 方法,均值±std(3 种子)
  fig_noise.png    : verifier 噪声扫描 — 左:最终成功率 vs ε;右:各 ε 下 σ3 训练曲线
用法: python plot.py [--results results]
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = np.arange(0, 60001, 10000)
NAMES = {"A": "A: base-only", "B": "B: full-task RL (sparse R)",
         "C": "C: PARTS (retrain)", "D": "D: PARTS no-retrain"}
COLORS = {"A": "#888888", "B": "#d62728", "C": "#1f77b4", "D": "#2ca02c"}


def load(results, method, eps, seeds=(0, 1, 2)):
    out = []
    for s in seeds:
        path = os.path.join(results, f"{method}_s{s}_e{eps}.json")
        if os.path.exists(path):
            out.append(json.load(open(path)))
    return out


def curves(runs, key):
    """把各 seed 的 (step, value) 快照插值到公共网格,返回 mean, std。"""
    ys = []
    for r in runs:
        xs = np.array([s["step"] for s in r["snapshots"]], dtype=float)
        vs = np.array([s[key] for s in r["snapshots"]], dtype=float)
        if len(xs) == 1:                       # A:单点 → 扁平
            ys.append(np.full_like(GRID, vs[0], dtype=float))
        else:
            ys.append(np.interp(GRID, np.clip(xs, 0, 60000), vs))
    ys = np.array(ys)
    return ys.mean(axis=0), ys.std(axis=0)


def plot_training(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, key, title in zip(axes, ("full", "s3"),
                              ("full-task success", "bottleneck subtask σ3 success")):
        for m in ("A", "B", "C", "D"):
            runs = load(results, m, 0.0)
            if not runs:
                continue
            mu, sd = curves(runs, key)
            ax.plot(GRID / 1000, mu, color=COLORS[m], label=NAMES[m], lw=2)
            ax.fill_between(GRID / 1000, mu - sd, mu + sd, color=COLORS[m], alpha=0.18)
        ax.set_xlabel("training env steps (×1000)")
        ax.set_ylabel("eval success rate")
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("PARTS toy reproduction — training curves (mean ± std, 3 seeds, eval 100 eps)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_training.png"), dpi=150)
    plt.close(fig)


def plot_noise(results, out_dir, eps_list=(0.0, 0.05, 0.15, 0.3, 0.5)):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # 左:最终成功率 vs ε
    for key, marker, label in (("s3", "o", "bottleneck σ3"),
                               ("full", "s", "full task")):
        mus, sds = [], []
        for e in eps_list:
            runs = load(results, "C", e)
            vals = [r["snapshots"][-1][key] for r in runs]
            mus.append(np.mean(vals)); sds.append(np.std(vals))
        axes[0].errorbar(eps_list, mus, yerr=sds, marker=marker, capsize=4,
                         lw=2, label=label)
    axes[0].set_xlabel("verifier flip probability ε")
    axes[0].set_ylabel("final eval success rate (60k steps)")
    axes[0].set_title("verifier noise robustness (method C)")
    axes[0].set_ylim(-0.03, 1.03); axes[0].grid(alpha=0.3); axes[0].legend()
    # 右:各 ε 的 σ3 训练曲线
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(eps_list):
        runs = load(results, "C", e)
        if not runs:
            continue
        mu, sd = curves(runs, "s3")
        c = cmap(i / max(1, len(eps_list) - 1))
        axes[1].plot(GRID / 1000, mu, color=c, lw=2, label=f"ε={e}")
        axes[1].fill_between(GRID / 1000, mu - sd, mu + sd, color=c, alpha=0.15)
    axes[1].set_xlabel("training env steps (×1000)")
    axes[1].set_ylabel("eval σ3 success")
    axes[1].set_title("σ3 learning under verifier noise")
    axes[1].set_ylim(-0.03, 1.03); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle("PARTS under noisy success verifiers (mean ± std, 3 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_noise.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    plot_training(args.results, args.results)
    plot_noise(args.results, args.results)
    print("wrote", os.path.join(args.results, "fig_training.png"),
          "and fig_noise.png")


if __name__ == "__main__":
    main()
