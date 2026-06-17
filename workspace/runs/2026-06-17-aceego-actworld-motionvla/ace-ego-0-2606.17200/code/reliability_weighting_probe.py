#!/usr/bin/env python3
"""
ACE-Ego-0 reliability-aware human auxiliary loss probe.

This script concretizes the non-obvious mechanism in ACE-Ego-0 Sec. 3.2 / Appendix A.5:
how noisy pseudo-action labels from egocentric human video are down-weighted by a
spatiotemporal reliability factor W_{t,j} = rho_j * w_data(d,h) * w_step(t,h)
so that the auxiliary loss concentrates on the highly-reliable position channels
and on smooth time steps.

The probe uses purely synthetic data so it runs without any data download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Action layout: unified 22-D bimanual camera-space action (Eq. 14-15)
# ---------------------------------------------------------------------------
#  [left_xyz(3), left_rot6d(6), left_gripper(1), left_active(1),
#   right_xyz(3), right_rot6d(6), right_gripper(1), right_active(1)]
DIM = 22
POS_L = slice(0, 3)
ROT_L = slice(3, 9)
GRP_L = 9
ACT_L = 10
POS_R = slice(11, 14)
ROT_R = slice(14, 20)
GRP_R = 20
ACT_R = 21

POS_DIMS = list(range(POS_L.start, POS_L.stop)) + list(range(POS_R.start, POS_R.stop))
ROT_DIMS = list(range(ROT_L.start, ROT_L.stop)) + list(range(ROT_R.start, ROT_R.stop))
GRP_DIMS = [GRP_L, GRP_R]
ACT_DIMS = [ACT_L, ACT_R]

CHANNEL_NAMES = [f"L.pos.{i}" for i in range(3)] + [f"L.rot.{i}" for i in range(6)] + ["L.grip", "L.act"] + \
                [f"R.pos.{i}" for i in range(3)] + [f"R.rot.{i}" for i in range(6)] + ["R.grip", "R.act"]

HAND_OF_DIM = {}
for d in range(DIM):
    HAND_OF_DIM[d] = "left" if d < 11 else "right"


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def make_clean_robot_chunk(steps: int = 64, seed: int = 0) -> np.ndarray:
    """A smooth bimanual reaching trajectory in camera space."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    a = np.zeros((steps, DIM), dtype=np.float32)

    # left arm: reach forward and slightly right
    a[:, POS_L] = np.stack([
        -0.10 + 0.05 * t + 0.005 * rng.normal(size=steps),
        -0.20 + 0.15 * t + 0.005 * rng.normal(size=steps),
         0.40 - 0.10 * t + 0.005 * rng.normal(size=steps),
    ], axis=1)
    # right arm: reach forward and slightly left
    a[:, POS_R] = np.stack([
         0.10 - 0.05 * t + 0.005 * rng.normal(size=steps),
        -0.20 + 0.15 * t + 0.005 * rng.normal(size=steps),
         0.40 - 0.10 * t + 0.005 * rng.normal(size=steps),
    ], axis=1)

    # continuous 6D orientation: small rotation around vertical axis
    for pos, rot in [(POS_L, ROT_L), (POS_R, ROT_R)]:
        theta = 0.3 * np.sin(2 * np.pi * t) + 0.05 * rng.normal(size=steps)
        # rot6d = [cos theta, 0, sin theta, 0, 1, 0] approximated
        a[:, rot] = np.stack([
            np.cos(theta), np.zeros_like(theta), np.sin(theta),
            np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)
        ], axis=1)

    # gripper: open -> close for left, close -> open for right
    a[:, GRP_L] = 0.08 - 0.06 * np.sin(np.pi * t)
    a[:, GRP_R] = 0.04 + 0.06 * np.sin(np.pi * t)

    a[:, ACT_L] = 1.0
    a[:, ACT_R] = 1.0
    return a


def corrupt_human_chunk(clean: np.ndarray, seed: int = 1) -> np.ndarray:
    """
    Inject pseudo-action estimation noise typical of a hand-reconstruction pipeline:
    - low-amplitude position jitter (clean-ish)
    - intermittent rotation spikes from occlusion
    - gripper noise from thumb-to-palm distance estimation
    """
    rng = np.random.default_rng(seed)
    noisy = clean.copy()
    steps = clean.shape[0]

    # position: Gaussian jitter with std growing at occluded frames
    pos_std = 0.004 + 0.008 * rng.random(steps)
    for d in POS_DIMS:
        noisy[:, d] += rng.normal(0, pos_std, size=steps)

    # rotation: sparse large spikes (e.g., frame-wise MANO yaw ambiguity / occlusion)
    spike_mask = rng.random(steps) < 0.12  # 12% of frames corrupted
    for d in ROT_DIMS:
        noisy[:, d] += spike_mask * rng.normal(0, 0.60, size=steps)

    # gripper: strong bias + noise from thumb-to-palm estimation
    for d in GRP_DIMS:
        noisy[:, d] += 0.015 * rng.normal(size=steps) + 0.02 * (rng.random(steps) - 0.5)
        noisy[:, d] = np.clip(noisy[:, d], 0.04, 0.10)

    return noisy


# ---------------------------------------------------------------------------
# Reliability-aware weighting (Eq. 6, 20-24 and Table 6)
# ---------------------------------------------------------------------------
@dataclass
class ReliabilityConfig:
    rho_pos: float = 1.0
    rho_rot_grip: float = 0.001
    w_min: float = 0.2
    alpha: float = 1.5
    smoothing_window: int = 3
    dataset_prior_left: float = 0.85
    dataset_prior_right: float = 0.70


def channel_prior(dim: int, cfg: ReliabilityConfig) -> float:
    if dim in POS_DIMS:
        return cfg.rho_pos
    return cfg.rho_rot_grip


def temporal_smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Simple causal+anti-causal moving average (paper uses window=3)."""
    if window <= 1:
        return a
    pad = window // 2
    out = np.empty_like(a)
    for d in range(a.shape[1]):
        s = np.pad(a[:, d], pad, mode="edge")
        out[:, d] = np.convolve(s, np.ones(window) / window, mode="valid")
    return out


def compute_step_weights(positions: np.ndarray, cfg: ReliabilityConfig) -> dict[str, np.ndarray]:
    """
    positions: (T, 6) -> left xyz then right xyz.
    Returns per-hand step weights w_step(t, h) in {left, right}.
    Thresholds are set to the 95th percentile of clean robot data as a reference.
    """
    hands = {
        "left": positions[:, :3],
        "right": positions[:, 3:],
    }
    # Use the clean position statistics as robot reference (paper: per-dataset/hand 95th percentile).
    ref_jump = {h: np.percentile(np.linalg.norm(np.diff(hands[h], axis=0, prepend=hands[h][:1]), axis=1), 95)
                for h in hands}
    ref_jerk = {h: np.percentile(
        np.linalg.norm(hands[h][2:] - 2 * hands[h][1:-1] + hands[h][:-2], axis=1),
        95) for h in hands}
    # avoid zeros
    ref_jump = {h: max(v, 1e-6) for h, v in ref_jump.items()}
    ref_jerk = {h: max(v, 1e-6) for h, v in ref_jerk.items()}

    weights = {}
    for h, pts in hands.items():
        dp = np.linalg.norm(np.diff(pts, axis=0, prepend=pts[:1]), axis=1)
        d2p = np.linalg.norm(pts[2:] - 2 * pts[1:-1] + pts[:-2], axis=1)
        d2p = np.concatenate([d2p[:1], d2p, d2p[-1:]])
        q = np.maximum(dp / ref_jump[h], d2p / ref_jerk[h])
        w = np.where(q <= 1.0, 1.0, np.maximum(cfg.w_min, np.exp(-cfg.alpha * (q - 1.0))))
        weights[h] = w.astype(np.float32)
    return weights


def reliability_weights(human_noisy: np.ndarray, cfg: ReliabilityConfig) -> np.ndarray:
    """Full W_{t,j}."""
    W = np.zeros((human_noisy.shape[0], DIM), dtype=np.float32)
    step_w = compute_step_weights(human_noisy[:, POS_DIMS], cfg)
    data_prior = {"left": cfg.dataset_prior_left, "right": cfg.dataset_prior_right}
    for d in range(DIM):
        rho = channel_prior(d, cfg)
        hand = HAND_OF_DIM[d]
        W[:, d] = rho * data_prior[hand] * step_w[hand]
    return W


def huber(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.where(np.abs(x) <= beta, 0.5 * x**2, beta * (np.abs(x) - 0.5 * beta))


def human_auxiliary_loss(pred_vel: np.ndarray, target_vel: np.ndarray, W: np.ndarray,
                         mask: np.ndarray, beta: float = 1.0) -> float:
    """Eq. 8: normalized Huber loss weighted by W."""
    err = huber(pred_vel - target_vel, beta)
    num = np.sum(mask * W * err)
    den = np.sum(mask * W)
    return float(num / (den + 1e-8))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    cfg = ReliabilityConfig()
    steps = 64

    robot_clean = make_clean_robot_chunk(steps, seed=0)
    human_noisy = corrupt_human_chunk(robot_clean, seed=1)
    human_smooth = temporal_smooth(human_noisy, cfg.smoothing_window)

    # A simple velocity target (tilde{a} - epsilon with epsilon=0 for visualization)
    target_vel = human_smooth
    # Pretend model prediction: small errors on clean position channels,
    # larger errors on noisy rotation/gripper channels (simulating corruption
    # when naive joint training treats pseudo-actions equally).
    pred_vel = target_vel.copy()
    pred_vel[:, POS_DIMS] += 0.015 * np.random.default_rng(7).normal(size=(steps, len(POS_DIMS)))
    pred_vel[:, ROT_DIMS] += 0.35 * np.random.default_rng(8).normal(size=(steps, len(ROT_DIMS)))
    pred_vel[:, GRP_DIMS] += 0.10 * np.random.default_rng(9).normal(size=(steps, len(GRP_DIMS)))

    # validity mask: all valid except we simulate a few masked-out projection failures
    mask = np.ones((steps, DIM), dtype=np.float32)
    mask[20:22, ROT_R] = 0.0
    mask[50:52, POS_L] = 0.0

    W = reliability_weights(human_noisy, cfg)

    loss_reliable = human_auxiliary_loss(pred_vel, target_vel, W, mask, beta=1.0)
    loss_naive = human_auxiliary_loss(pred_vel, target_vel, np.ones_like(W), mask, beta=1.0)

    # Effective supervision mass per channel group
    eff_pos = np.sum(mask[:, POS_DIMS] * W[:, POS_DIMS])
    eff_rot = np.sum(mask[:, ROT_DIMS] * W[:, ROT_DIMS])
    eff_grp = np.sum(mask[:, GRP_DIMS] * W[:, GRP_DIMS])
    total = eff_pos + eff_rot + eff_grp + 1e-8

    print("=" * 60)
    print("ACE-Ego-0 reliability-aware human auxiliary loss probe")
    print("=" * 60)
    print(f"Synthetic episode length     : {steps} steps")
    print(f"Position channel prior rho   : {cfg.rho_pos}")
    print(f"Rotation/gripper prior rho   : {cfg.rho_rot_grip}")
    print(f"Dataset prior left/right     : {cfg.dataset_prior_left}/{cfg.dataset_prior_right}")
    print(f"Naive auxiliary loss         : {loss_naive:.4f}")
    print(f"Reliability-aware loss       : {loss_reliable:.4f}")
    print(f"Effective supervision mass   : pos={eff_pos:.1f} ({eff_pos/total*100:.1f}%), "
          f"rot={eff_rot:.1f} ({eff_rot/total*100:.1f}%), "
          f"grip={eff_grp:.1f} ({eff_grp/total*100:.1f}%)")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    t = np.arange(steps)

    # Panel 1: left/right wrist position trajectories colored by step reliability
    ax = axes[0]
    pos = human_noisy[:, POS_DIMS]
    for hand_idx, (label, s_start) in enumerate([("Left", 0), ("Right", 3)]):
        x = pos[:, s_start]
        y = pos[:, s_start + 1]
        z = pos[:, s_start + 2]
        hand = "left" if hand_idx == 0 else "right"
        step_w = compute_step_weights(pos, cfg)[hand]
        scatter = ax.scatter(x, y, c=step_w,
                             cmap="RdYlGn", s=20, vmin=0, vmax=1,
                             label=f"{hand} hand XY")
        ax.plot(x, y, "k-", alpha=0.2, lw=0.5)
    ax.set_xlabel("Camera X (m)")
    ax.set_ylabel("Camera Y (m)")
    ax.set_title("Horizontal wrist trajectory colored by step reliability w_step")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("w_step")

    # Panel 2: per-dimension mean reliability
    ax = axes[1]
    mean_W = np.mean(W * mask, axis=0)
    colors = ["#2ca02c" if d in POS_DIMS else "#d62728" if d in ROT_DIMS else "#9467bd"
              for d in range(DIM)]
    bars = ax.bar(range(DIM), mean_W, color=colors)
    ax.set_xticks(range(DIM))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
    ax.set_ylabel("Mean W_{t,j}")
    ax.set_title("Mean spatiotemporal reliability per action dimension")
    ax.set_ylim(0, 1.05)
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#2ca02c", label="Position (rho=1.0)"),
              Patch(facecolor="#d62728", label="Rotation (rho=0.001)"),
              Patch(facecolor="#9467bd", label="Gripper (rho=0.001)")]
    ax.legend(handles=legend, loc="upper right")

    # Panel 3: loss decomposition by channel group
    ax = axes[2]
    group_err = {"Position": huber(pred_vel[:, POS_DIMS] - target_vel[:, POS_DIMS]),
                 "Rotation": huber(pred_vel[:, ROT_DIMS] - target_vel[:, ROT_DIMS]),
                 "Gripper": huber(pred_vel[:, GRP_DIMS] - target_vel[:, GRP_DIMS])}
    x = np.arange(steps)
    bottoms = np.zeros(steps)
    palette = {"Position": "#2ca02c", "Rotation": "#d62728", "Gripper": "#9467bd"}
    for name, err in group_err.items():
        weighted = err * W[:, [POS_DIMS, ROT_DIMS, GRP_DIMS][list(group_err.keys()).index(name)]]
        # sum over channels in group per timestep
        idx = [POS_DIMS, ROT_DIMS, GRP_DIMS][list(group_err.keys()).index(name)]
        weighted_t = np.sum(weighted * mask[:, idx], axis=1)
        ax.bar(x, weighted_t, bottom=bottoms, label=name, color=palette[name], width=1.0)
        bottoms += weighted_t
    ax.set_xlabel("Time step")
    ax.set_ylabel("Weighted Huber contribution")
    ax.set_title("Per-timestep reliability-weighted auxiliary loss decomposition")
    ax.legend(loc="upper right")
    ax.set_xlim(0, steps)

    # Panel 4: naive vs reliability-aware loss
    ax = axes[3]
    bars = ax.bar(["Naive (W=1)", "Reliability-aware"], [loss_naive, loss_reliable],
                  color=["#d62728", "#2ca02c"])
    ax.set_ylabel("Auxiliary loss")
    ax.set_title("Naive pseudo-action supervision vs reliability-aware supervision")
    ax.set_yscale("log")
    for bar, val in zip(bars, [loss_naive, loss_reliable]):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.3, f"{val:.2e}",
                ha="center", va="bottom", fontsize=11)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ace-ego-0-reliability-probe.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
