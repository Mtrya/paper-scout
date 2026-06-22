#!/usr/bin/env python3
"""
Probe: reconstruct and sanity-check the egocentric-pretraining scaling law from
HumanScale (arXiv:2606.20521) using the reported best post-training action losses.

What it does
------------
- Fits the paper's claimed log-linear law  L = a - b * ln(D)  to the reported
  100-hour and 5,000-hour losses for both Seen and Unseen evaluation splits.
- Reconstructs the implied 1,000-hour loss under perfect log-linearity (the paper
  only gives R^2 values, not the raw 1k point).
- Compares egocentric pretraining with the matched-scale real-robot pretraining
  numbers reported at 5,000 hours.
- Extrapolates the fitted laws to 20,000 and 100,000 hours to show where the
  trend predicts saturation would begin.
- Writes a CSV table and, if matplotlib is available, a small diagnostic plot.

Reported source numbers (Section 4.1 and 4.2 of the paper):
- Ego Seen:    0.0080 @ 100 h,  0.0067 @ 5,000 h
- Ego Unseen:  0.0234 @ 100 h,  0.0204 @ 5,000 h
- Robot Seen:  0.0071 @ 5,000 h
- Robot Unseen: 0.0254 @ 5,000 h
- Wan2.2 no-pretrain baseline implied from the paper's percentage reductions
  ("35% lower" on Seen, "24% lower" on Unseen).

Usage
-----
    python ego_scaling_probe.py

Outputs (in the same directory as this script):
    ego_scaling_table.csv
    ego_scaling.png

Dependencies
------------
    python >= 3.8, numpy, matplotlib (optional, for the plot)
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def fit_log_linear(hours: np.ndarray, losses: np.ndarray) -> Tuple[float, float]:
    """Fit L = a - b * ln(hours) and return (a, b)."""
    x = np.log(hours)
    # solve least squares for L = a + c*x where c = -b
    A = np.vstack([np.ones_like(x), x]).T
    a, c = np.linalg.lstsq(A, losses, rcond=None)[0]
    return float(a), float(-c)


def predict(a: float, b: float, hours: float) -> float:
    return a - b * math.log(hours)


@dataclass
class SplitResult:
    name: str
    loss_100: float
    loss_5000: float
    robot_5000: float
    baseline: float
    a: float
    b: float
    loss_1000_reconstructed: float
    improvement_over_robot_pct: float
    improvement_over_baseline_pct: float


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(script_dir, "ego_scaling_table.csv")
    out_png = os.path.join(script_dir, "ego_scaling.png")

    # ------------------------------------------------------------------
    # Source numbers from the paper (Section 4.1 / 4.2)
    # ------------------------------------------------------------------
    ego_seen_100 = 0.0080
    ego_seen_5000 = 0.0067
    ego_unseen_100 = 0.0234
    ego_unseen_5000 = 0.0204

    robot_seen_5000 = 0.0071
    robot_unseen_5000 = 0.0254

    # Baseline inferred from "35% lower" (Seen) and "24% lower" (Unseen).
    baseline_seen = ego_seen_5000 / (1 - 0.35)
    baseline_unseen = ego_unseen_5000 / (1 - 0.24)

    # ------------------------------------------------------------------
    # Fit log-linear law L = a - b * ln(D) using the two endpoints.
    # ------------------------------------------------------------------
    hours = np.array([100.0, 5000.0])
    a_seen, b_seen = fit_log_linear(hours, np.array([ego_seen_100, ego_seen_5000]))
    a_unseen, b_unseen = fit_log_linear(
        hours, np.array([ego_unseen_100, ego_unseen_5000])
    )

    # Reconstruct the implied 1,000-hour point under perfect log-linearity.
    loss_1000_seen = predict(a_seen, b_seen, 1000.0)
    loss_1000_unseen = predict(a_unseen, b_unseen, 1000.0)

    results: List[SplitResult] = [
        SplitResult(
            name="Seen",
            loss_100=ego_seen_100,
            loss_5000=ego_seen_5000,
            robot_5000=robot_seen_5000,
            baseline=baseline_seen,
            a=a_seen,
            b=b_seen,
            loss_1000_reconstructed=loss_1000_seen,
            improvement_over_robot_pct=(robot_seen_5000 - ego_seen_5000)
            / robot_seen_5000
            * 100.0,
            improvement_over_baseline_pct=(baseline_seen - ego_seen_5000)
            / baseline_seen
            * 100.0,
        ),
        SplitResult(
            name="Unseen",
            loss_100=ego_unseen_100,
            loss_5000=ego_unseen_5000,
            robot_5000=robot_unseen_5000,
            baseline=baseline_unseen,
            a=a_unseen,
            b=b_unseen,
            loss_1000_reconstructed=loss_1000_unseen,
            improvement_over_robot_pct=(robot_unseen_5000 - ego_unseen_5000)
            / robot_unseen_5000
            * 100.0,
            improvement_over_baseline_pct=(baseline_unseen - ego_unseen_5000)
            / baseline_unseen
            * 100.0,
        ),
    ]

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("HumanScale egocentric-pretraining scaling probe")
    print("=" * 60)
    for r in results:
        print(f"\nSplit: {r.name}")
        print(f"  Fitted law: L = {r.a:.6f} - {r.b:.6f} * ln(hours)")
        print(f"  Slope interpretation: -{r.b:.6f} loss per ln(hour)")
        print(
            f"  Reconstructed 1,000 h loss (perfect log-linearity): "
            f"{r.loss_1000_reconstructed:.4f}"
        )
        print(
            f"  Ego vs Robot @ 5,000 h: "
            f"{r.improvement_over_robot_pct:.1f}% lower ({r.loss_5000:.4f} vs {r.robot_5000:.4f})"
        )
        print(
            f"  Ego vs Wan2.2 baseline: "
            f"{r.improvement_over_baseline_pct:.1f}% lower "
            f"(inferred baseline {r.baseline:.4f})"
        )
        for h in [20_000, 100_000]:
            print(
                f"  Extrapolated {h:,} h loss: {predict(r.a, r.b, h):.4f}"
            )

    # ------------------------------------------------------------------
    # Write CSV table
    # ------------------------------------------------------------------
    extrapolation_hours = [100, 1000, 5000, 20_000, 100_000]
    rows: List[Dict[str, float]] = []
    for r in results:
        row: Dict[str, float] = {"split": r.name}
        for h in extrapolation_hours:
            row[f"loss_{h}h"] = round(predict(r.a, r.b, h), 4)
        row["robot_5000h"] = round(r.robot_5000, 4)
        row["baseline_inferred"] = round(r.baseline, 4)
        row["a"] = round(r.a, 6)
        row["b"] = round(r.b, 6)
        rows.append(row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote CSV: {out_csv}")

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib not available, skipping plot ({exc})")
        return 0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_hours = np.logspace(2, 5.2, 200)  # 100 h to ~160 kh
    colors = {"Seen": "#1f77b4", "Unseen": "#ff7f0e"}
    markers = {"Seen": "o", "Unseen": "s"}

    for r in results:
        pred = np.array([predict(r.a, r.b, h) for h in plot_hours])
        ax.plot(
            plot_hours,
            pred,
            label=f"Ego {r.name} fit",
            color=colors[r.name],
            linestyle="--",
            linewidth=1.5,
        )
        ax.scatter(
            [100, 1000, 5000],
            [r.loss_100, r.loss_1000_reconstructed, r.loss_5000],
            color=colors[r.name],
            marker=markers[r.name],
            s=60,
            zorder=5,
            label=f"Ego {r.name} (100/1k/5k h)",
        )
        # robot at 5k
        ax.scatter(
            [5000],
            [r.robot_5000],
            color=colors[r.name],
            marker="x",
            s=100,
            linewidths=2,
            zorder=5,
            label=f"Robot {r.name} @ 5k h" if r.name == "Seen" else "",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Pretraining hours (log scale)")
    ax.set_ylabel("Best post-training action loss")
    ax.set_title("HumanScale egocentric pretraining scaling reconstruction")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Wrote plot: {out_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
