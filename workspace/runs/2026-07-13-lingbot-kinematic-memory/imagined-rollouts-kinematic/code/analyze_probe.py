#!/usr/bin/env python3
"""Reproduce the headline friction-sweep iKCE comparison from the probe CSVs.

Run from the thread directory:

    python code/analyze_probe.py

Expects assets/ikce_wm_full.csv and assets/ikce_physics_full.csv produced by
the probe configs in this thread.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    magnitudes, means = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            magnitudes.append(float(row["magnitude"]))
            means.append(float(row["mean"]))
    return np.array(magnitudes), np.array(means)


def main() -> None:
    assets = Path(__file__).parent.parent / "assets"
    mu_wm, y_wm = load(assets / "ikce_wm_full.csv")
    mu_ph, y_ph = load(assets / "ikce_physics_full.csv")

    slope_wm, _, _, p_wm, _ = linregress(np.log(mu_wm), np.log(y_wm))
    slope_ph, _, _, p_ph, _ = linregress(np.log(mu_ph), np.log(y_ph))

    print(f"WM    log-log slope: {slope_wm:.4f} (p={p_wm:.4f})")
    print(f"Phys  log-log slope: {slope_ph:.4f} (p={p_ph:.4e})")
    print(f"WM/Phys ratio at mu=1.0: {y_wm[mu_wm == 1.0].item() / y_ph[mu_ph == 1.0].item():.1f}x")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mu_ph, y_ph, "o-", label="physics + policy")
    ax.plot(mu_wm, y_wm, "s-", label="WM (imagined)")
    ax.set_xlabel("friction multiplier μ")
    ax.set_ylabel("iKCE")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.5)
    out = assets / "ikce_friction_probe.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
