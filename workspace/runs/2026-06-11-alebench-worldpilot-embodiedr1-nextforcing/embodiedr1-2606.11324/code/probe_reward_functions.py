"""
Probe: Reconstruct the five reward families from embodied_reward.py
and visualize their piecewise-linear decay curves.

The paper describes five families (§4.2.2):
  1) exact-match       → binary {0,1}
  2) IoU               → continuous [0,1]
  3) point-distance    → φ(d; 40, 150)   (but code uses 40, 150 in comments
                        and 40/150 in the strict file; embodied_reward.py
                        uses perfect_threshold=40.0, zero_threshold=150.0)
  4) trajectory-RMSE   → φ(RMSE; 50, 120) for 2D, depth MAE φ(0.1, 0.4)
  5) semantic-sim      → sigmoid-normalized RM score or BLEU fallback

This script plots the continuous decay curves so the thresholds are直观.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def phi(d, tau_p, tau_z):
    """Piecewise-linear decay from the paper, Eq. (1)."""
    if d < tau_p:
        return 1.0
    elif d >= tau_z:
        return 0.0
    else:
        return (tau_z - d) / (tau_z - tau_p)

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Point distance
    d = np.linspace(0, 200, 500)
    r_point = [phi(x, 40, 150) for x in d]
    ax.plot(d, r_point, label="point (τp=40, τz=150)", lw=2)

    # Trajectory 2D RMSE
    r_trace = [phi(x, 50, 120) for x in d]
    ax.plot(d, r_trace, label="trace 2D (τp=50, τz=120)", lw=2, linestyle='--')

    # Trajectory 3D depth MAE (scaled to same x-axis for comparison)
    d_depth = np.linspace(0, 0.5, 500)
    r_depth = [phi(x, 0.1, 0.4) for x in d_depth]
    ax2 = ax.twiny()
    ax2.plot(d_depth * 1000, r_depth, label="trace depth (τp=0.1m, τz=0.4m)",
             lw=2, linestyle='-.', color='green')
    ax2.set_xlim(0, 500)
    ax2.set_xlabel("Depth MAE (mm, top axis)")

    ax.axvline(40, color='C0', alpha=0.3, linestyle=':')
    ax.axvline(150, color='C0', alpha=0.3, linestyle=':')
    ax.axvline(50, color='C1', alpha=0.3, linestyle=':')
    ax.axvline(120, color='C1', alpha=0.3, linestyle=':')

    ax.set_xlabel("Pixel distance / RMSE")
    ax.set_ylabel("Reward")
    ax.set_title("Piecewise-linear decay rewards (Embodied-R1.5)")
    ax.legend(loc='upper right')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    out = "reward_decay_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
