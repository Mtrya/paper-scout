#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def baselines(rewards, history_mean, group_size=8, pseudo_count=2.0, ema=0.25):
    rewards = np.asarray(rewards, dtype=np.float64)
    assert rewards.shape == (group_size,)
    total = rewards.sum()
    others = total - rewards
    rloo = others / (group_size - 1)
    if history_mean is None:
        remembered = rloo.copy()
        new_mean = float(rewards.mean())
    else:
        remembered = (pseudo_count * history_mean + others) / (pseudo_count + group_size - 1)
        new_mean = ema * float(rewards.mean()) + (1 - ema) * history_mean
    return rewards - rloo, rewards - remembered, new_mean


def stationary_experiment(seed, chains, visits, group_size):
    rng = np.random.default_rng(seed)
    rows = []
    probabilities = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]
    for p in probabilities:
        for chain in range(chains):
            mean = None
            for visit in range(visits):
                rewards = (rng.random(group_size) < p).astype(float)
                rloo, remembered, mean = baselines(rewards, mean, group_size)
                uniform = bool(np.all(rewards == rewards[0]))
                rows.append({
                    "p": p,
                    "chain": chain,
                    "visit": visit,
                    "uniform": int(uniform),
                    "rloo_abs_mean": float(np.abs(rloo).mean()),
                    "memory_abs_mean": float(np.abs(remembered).mean()),
                    "rloo_nonzero": int(np.any(np.abs(rloo) > 1e-12)),
                    "memory_nonzero": int(np.any(np.abs(remembered) > 1e-12)),
                    "memory_signed_mean": float(remembered.mean()),
                    "history_mean": mean,
                })
    return rows


def abrupt_experiment(group_size):
    rows = []
    for name, before, after in [("improvement", 0.1, 1.0), ("degradation", 0.9, 0.0)]:
        mean = before
        for visit in range(17):
            rewards = np.full(group_size, float(round(before)) if visit < 4 else after)
            rloo, remembered, new_mean = baselines(rewards, mean, group_size)
            rows.append({
                "trajectory": name,
                "visit": visit - 4,
                "uniform_return": float(rewards[0]),
                "history_before": mean,
                "history_after": new_mean,
                "rloo_advantage": float(rloo[0]),
                "memory_advantage": float(remembered[0]),
            })
            mean = new_mean
    return rows


def summarize_stationary(rows):
    summary = []
    for p in sorted({r["p"] for r in rows}):
        selected = [r for r in rows if r["p"] == p and r["visit"] >= 20]
        uniform = [r for r in selected if r["uniform"]]
        for method in ["rloo", "memory"]:
            summary.append({
                "p": p,
                "method": method,
                "groups": len(selected),
                "uniform_groups": len(uniform),
                "uniform_fraction": len(uniform) / len(selected),
                "uniform_nonzero_fraction": sum(r[f"{method}_nonzero"] for r in uniform) / len(uniform),
                "rms_advantage": float(np.sqrt(np.mean([r[f"{method}_abs_mean"] ** 2 for r in selected]))),
                "mean_abs_advantage": float(np.mean([r[f"{method}_abs_mean"] for r in selected])),
            })
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, trajectory, path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    colors = {"rloo": "#737373", "memory": "#C44E52"}
    labels = {"rloo": "Current group only", "memory": "With context memory"}
    for method in ["rloo", "memory"]:
        rows = [r for r in summary if r["method"] == method]
        p = [r["p"] for r in rows]
        axes[0].plot(p, [100 * r["uniform_nonzero_fraction"] for r in rows], "o-", color=colors[method], label=labels[method])
        axes[1].plot(p, [r["rms_advantage"] for r in rows], "o-", color=colors[method], label=labels[method])
    axes[0].set(xlabel="True success probability", ylabel="Uniform groups with nonzero advantage (%)", ylim=(-3, 105))
    axes[1].set(xlabel="True success probability", ylabel="RMS advantage per update")
    axes[0].legend(frameon=False, loc="center")
    for name, color in [("improvement", "#4C72B0"), ("degradation", "#DD8452")]:
        rows = [r for r in trajectory if r["trajectory"] == name]
        axes[2].plot([r["visit"] for r in rows], [r["memory_advantage"] for r in rows], "o-", color=color, label=name)
    axes[2].axhline(0, color="#333333", lw=0.8)
    axes[2].axvline(0, color="#999999", lw=0.8, ls="--")
    axes[2].set(xlabel="Visits after an abrupt change", ylabel="Advantage in a uniform group")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--visits", type=int, default=160)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = stationary_experiment(args.seed, args.chains, args.visits, args.group_size)
    summary = summarize_stationary(rows)
    trajectory = abrupt_experiment(args.group_size)
    write_csv(args.output / "stationary_summary.csv", summary)
    write_csv(args.output / "abrupt_trajectory.csv", trajectory)
    plot(summary, trajectory, args.output / "grpo_context_memory.png")
    result = {
        "seed": args.seed,
        "group_size": args.group_size,
        "pseudo_count": 2,
        "ema": 0.25,
        "chains_per_probability": args.chains,
        "visits_per_chain": args.visits,
        "warmup_visits_excluded": 20,
        "rloo_uniform_nonzero_max": max(r["uniform_nonzero_fraction"] for r in summary if r["method"] == "rloo"),
        "memory_uniform_nonzero_min": min(r["uniform_nonzero_fraction"] for r in summary if r["method"] == "memory"),
        "memory_uniform_nonzero_max": max(r["uniform_nonzero_fraction"] for r in summary if r["method"] == "memory"),
        "trajectory": trajectory,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
