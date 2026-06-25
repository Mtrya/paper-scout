#!/usr/bin/env python3
"""Parse the artifact repo trial summaries and generate tables/plots.

Run from the thread code/ directory:
    /path/to/venv/bin/python parse_and_plot.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def breakout_milestones(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    wanted = {
        "baseline_v0": "initial RAM intercept",
        "tunnel0_v1": "return the ball, no loop breaker",
        "stuck12_t1024_s256_v1": "add stuck-loop perturbation",
        "fastlead3_v1": "handle fast low balls",
        "secondwall_stucktaper8_lag2_v1": "late-game stuck taper + lag comp",
        "breakout_default_864_eval3_v1": "default 864 3-episode validation",
        "breakout_vision_min0_h8_eval3_v1": "pure-vision RGB controller 864",
    }
    by_name = {r["trial_name"]: r for r in rows}
    out = []
    for name, note in wanted.items():
        r = by_name.get(name)
        if r is None:
            continue
        score = to_float(r.get("score_mean", ""))
        steps = to_float(r.get("cumulative_env_steps", ""))
        if score is None or steps is None:
            continue
        out.append(
            {
                "trial_name": name,
                "score": score,
                "cumulative_env_steps": int(steps),
                "note": note,
            }
        )
    return out


def ant_milestones(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    wanted = {
        "ant_lr_cpgpd_v1": "left/right anti-phase CPG + PD",
        "ant_yawaxis_grid_v2": "add yaw feedback + retune",
        "ant_h3_428_v1": "add 2nd/3rd harmonics",
        "ant_mpc_residual_v1_ep1": "residual MPC baseline (H=6, C=32)",
        "ant_mpc_residual_warm02_eval5_v1": "warm-start residual plan",
        "ant_mpc_default_adaptive_ep1_v1": "speed-adaptive phase + stance",
    }
    by_name = {r["trial_name"]: r for r in rows}
    out = []
    for name, note in wanted.items():
        r = by_name.get(name)
        if r is None:
            continue
        score = to_float(r.get("score_mean", ""))
        steps = to_float(r.get("cumulative_env_steps", ""))
        if score is None or steps is None:
            continue
        out.append(
            {
                "trial_name": name,
                "score": score,
                "cumulative_env_steps": int(steps),
                "note": note,
            }
        )
    return out


def plot_progression(
    rows: list[dict[str, str]],
    milestones: list[dict],
    title: str,
    ylabel: str,
    out_path: Path,
    kind: str = "eval",
) -> None:
    xs, ys = [], []
    running_best = []
    best = -math.inf
    for r in rows:
        if r.get("kind", "") != kind:
            continue
        score = to_float(r.get("score_mean", ""))
        steps = to_float(r.get("cumulative_env_steps", ""))
        if score is None or steps is None:
            continue
        xs.append(steps)
        ys.append(score)
        best = max(best, score)
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(xs, ys, s=15, alpha=0.4, label="individual eval", color="steelblue")
    ax.plot(xs, running_best, lw=2, label="running best", color="darkorange")
    for m in milestones:
        ax.axvline(m["cumulative_env_steps"], color="gray", ls="--", alpha=0.3)
        ax.scatter(
            [m["cumulative_env_steps"]],
            [m["score"]],
            s=80,
            color="red",
            zorder=5,
        )
        ax.annotate(
            f"{m['score']:.0f}",
            (m["cumulative_env_steps"], m["score"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("cumulative environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def atari57_plot(path: Path, out_path: Path) -> None:
    rows = read_csv(path)
    modes = {}
    for r in rows:
        mode = r.get("obs_mode", "").strip()
        x = to_float(r.get("x", ""))
        y = to_float(r.get("median_hns", ""))
        if x is None or y is None:
            continue
        modes.setdefault(mode, []).append((x, y))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, pts in sorted(modes.items()):
        pts = sorted(pts)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=2, label=f"Codex {mode}")
    # Reference numbers from the blog text for PPO baselines at ~10M steps.
    ax.axhline(0.88, color="gray", ls="--", alpha=0.5, label="OpenRL PPO2 ~0.88")
    ax.axhline(0.92, color="black", ls="--", alpha=0.5, label="CleanRL EnvPool PPO ~0.92")
    ax.set_xlabel("cumulative environment steps")
    ax.set_ylabel("median human-normalized score")
    ax.set_title("Atari57 unattended Codex heuristic search vs PPO baselines")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def per_game_table(path: Path) -> list[dict[str, float | str]]:
    rows = read_csv(path)
    comps = []
    for r in rows:
        game = r.get("game", "")
        ppo2 = to_float(r.get("OpenAI Baselines PPO2", ""))
        cleanrl = to_float(r.get("CleanRL EnvPool PPO", ""))
        codex = to_float(r.get("Codex best input mean", ""))
        codex_single = to_float(r.get("Codex best single run", ""))
        if ppo2 is None or cleanrl is None or codex is None:
            continue
        comps.append(
            {
                "game": game,
                "ppo2": ppo2,
                "cleanrl": cleanrl,
                "codex_mean": codex,
                "codex_single": codex_single,
                "codex_minus_ppo": codex - ppo2,
            }
        )
    return comps


def main() -> None:
    # Breakout
    breakout_rows = read_csv(DATA / "heuristic_breakout_trials_summary.csv")
    breakout_ms = breakout_milestones(breakout_rows)
    print("=== Breakout milestones ===")
    for m in breakout_ms:
        print(
            f"{m['trial_name']:45s} score={m['score']:6.0f} steps={m['cumulative_env_steps']:>10}  {m['note']}"
        )
    plot_progression(
        breakout_rows,
        breakout_ms,
        "Breakout heuristic progression (RAM policy)",
        "episode score",
        FIGS / "breakout_progression.png",
    )

    # Ant
    ant_rows = read_csv(DATA / "heuristic_ant_trials_summary.csv")
    ant_ms = ant_milestones(ant_rows)
    print("\n=== Ant milestones ===")
    for m in ant_ms:
        print(
            f"{m['trial_name']:45s} score={m['score']:7.1f} steps={m['cumulative_env_steps']:>10}  {m['note']}"
        )
    plot_progression(
        ant_rows,
        ant_ms,
        "Ant heuristic progression (CPG + residual MPC)",
        "5-episode mean return",
        FIGS / "ant_progression.png",
    )

    # Atari57
    atari57_plot(DATA / "atari57_aggregate_curve_steps_median.csv", FIGS / "atari57_median_hns.png")
    comps = per_game_table(DATA / "openrl_atari57_per_game_hns_comparison.csv")
    comps_sorted = sorted(comps, key=lambda d: d["codex_minus_ppo"], reverse=True)
    print("\n=== Atari57 per-game HNS: Codex mean vs PPO2 ===")
    print(f"{'game':<20} {'ppo2':>8} {'cleanrl':>8} {'codex_mean':>10} {'codex_single':>12}")
    for c in comps_sorted[:15]:
        print(
            f"{c['game']:<20} {c['ppo2']:8.2f} {c['cleanrl']:8.2f} {c['codex_mean']:10.2f} {c['codex_single'] or 0:12.2f}"
        )
    print("...")
    for c in comps_sorted[-5:]:
        print(
            f"{c['game']:<20} {c['ppo2']:8.2f} {c['cleanrl']:8.2f} {c['codex_mean']:10.2f} {c['codex_single'] or 0:12.2f}"
        )

    # Save JSON artifact for the report
    summary = {
        "breakout_milestones": breakout_ms,
        "ant_milestones": ant_ms,
        "atari57_best_codex_median_hns": {
            "native_obs": float(
                max(
                    (r for r in read_csv(DATA / "atari57_aggregate_curve_steps_median.csv") if r["obs_mode"] == "native_obs"),
                    key=lambda r: to_float(r["median_hns"]) or -1,
                )["median_hns"]
            ),
            "ram": float(
                max(
                    (r for r in read_csv(DATA / "atari57_aggregate_curve_steps_median.csv") if r["obs_mode"] == "ram"),
                    key=lambda r: to_float(r["median_hns"]) or -1,
                )["median_hns"]
            ),
        },
        "atari57_per_game_top_advantage": [
            {"game": c["game"], "advantage_over_ppo2": round(c["codex_minus_ppo"], 3)}
            for c in comps_sorted[:10]
        ],
    }
    with (Path(__file__).resolve().parent / "parsed_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved figures to", FIGS)


if __name__ == "__main__":
    main()
