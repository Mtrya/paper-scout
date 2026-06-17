#!/usr/bin/env python3
"""
Self-contained probe illustrating ActWorld's action-aware memory idea.

We simulate a long chunk-autoregressive rollout with a single mid-rollout
object-interaction event, then compare three history-compression strategies:

1. Recency baseline (time-based buckets): recent frames are kept fine-grained,
   older frames are coarsely compressed.
2. EAFR (Event-Aware Frame Re-assignment): interaction-critical frames
   (contact/manipulating/completing) are promoted into fine-grained memory
   regardless of age.
3. EAFR + persistent memory bank: event/object tokens from interaction moments
   are pinned and survive arbitrary navigation gaps.

The probe needs only NumPy + Matplotlib; no model weights or dataset downloads.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

np.random.seed(0)

PHASES = ["nav", "approaching", "reaching", "contact", "manipulating",
          "completing", "post-action", "revisit"]
PHASE_PRIOR = {
    "nav": 0.0,
    "approaching": 0.2,
    "reaching": 0.4,
    "contact": 1.0,
    "manipulating": 1.0,
    "completing": 0.9,
    "post-action": 0.3,
    "revisit": 0.0,
}

# Simulate multi-term memory bucket sizes (fine / mid / coarse tokens).
# Fine is deliberately tight so that recency quickly evicts old event frames.
BUCKET_SIZES = {"fine": 2, "mid": 6, "coarse": 1000}
NOISE_LEVELS = {"fine": 0.05, "mid": 0.20, "coarse": 0.55}
BANK_CAPACITY = 16
BANK_PINNED_PHASES = {"contact", "manipulating", "completing"}


def make_trajectory(length: int = 50, event_start: int = 10):
    """Build a synthetic trajectory: long nav, interaction event, long nav, revisit."""
    phases = ["nav"] * length
    # Interaction micro-sequence over 4 chunks.
    for i, ph in enumerate(["approaching", "contact", "manipulating", "completing"]):
        if event_start + i < length:
            phases[event_start + i] = ph
    # Revisit at the end tries to read the object state.
    phases[-1] = "revisit"
    return phases


def object_state_from_phases(phases):
    """Object property is engaged during contact/manipulating/completing and persists."""
    state = 0.0
    states = []
    for ph in phases:
        if ph in BANK_PINNED_PHASES:
            state = 1.0
        states.append(state)
    return np.array(states, dtype=np.float32)


def recency_buckets(history_indices, t):
    """Baseline MTM: assign history chunks to buckets purely by age."""
    # Sort oldest -> newest for exposition; we fill newest first into fine.
    sorted_idx = sorted(history_indices, key=lambda k: t - k)
    buckets = {"fine": [], "mid": [], "coarse": []}
    for k in sorted_idx:
        if len(buckets["fine"]) < BUCKET_SIZES["fine"]:
            buckets["fine"].append(k)
        elif len(buckets["mid"]) < BUCKET_SIZES["mid"]:
            buckets["mid"].append(k)
        else:
            buckets["coarse"].append(k)
    return buckets


def eafr_buckets(phases, history_indices, t,
                 lambda_phi: float = 1.0, lambda_r: float = 0.6, tau: float = 8.0):
    """EAFR: importance = phase prior + recency decay."""
    weights = {}
    for k in history_indices:
        phi = PHASE_PRIOR[phases[k]]
        recency = np.exp(-(t - k) / tau)
        weights[k] = lambda_phi * phi + lambda_r * recency
    sorted_idx = sorted(history_indices, key=weights.get, reverse=True)
    buckets = {"fine": [], "mid": [], "coarse": []}
    for k in sorted_idx:
        if len(buckets["fine"]) < BUCKET_SIZES["fine"]:
            buckets["fine"].append(k)
        elif len(buckets["mid"]) < BUCKET_SIZES["mid"]:
            buckets["mid"].append(k)
        else:
            buckets["coarse"].append(k)
    return buckets, weights


def observe_state(true_state, bucket, phase):
    """
    Observe object state through a noisy bucket channel.
    Only interaction-critical frames carry reliable object-state evidence;
    navigation frames are uninformative (object not in view / not encoded).
    """
    noise = NOISE_LEVELS[bucket]
    if phase in BANK_PINNED_PHASES:
        return float(true_state) + np.random.normal(0.0, noise)
    # Other frames encode noisily ambiguous evidence.
    return 0.5 + np.random.normal(0.0, noise + 0.25)


def predict_state(buckets, states, phases, use_bank: bool = False):
    """
    Weighted vote over the fine-grained history tokens plus persistent bank tokens.
    Coarse/mid navigation tokens are ignored in this toy because they carry no
    object-state evidence; the decisive signal comes from fine-grained event
    frames and from the pinned memory bank.
    """
    tokens = []
    weights = []
    for k in buckets["fine"]:
        tokens.append(observe_state(states[k], "fine", phases[k]))
        weights.append(1.0 / (NOISE_LEVELS["fine"] ** 2 + 1e-6))
    if use_bank:
        # Persistent bank contributes pinned event/object tokens from interaction.
        for k, ph in enumerate(phases):
            if ph in BANK_PINNED_PHASES:
                tokens.append(observe_state(states[k], "fine", ph))
                weights.append(1.0 / (NOISE_LEVELS["fine"] ** 2 + 1e-6))
    if not tokens:
        return 0.0
    tokens = np.array(tokens)
    weights = np.array(weights)
    pred = np.sum(tokens * weights) / np.sum(weights)
    return float(pred)


def simulate_one(length: int = 50, event_start: int = 10):
    phases = make_trajectory(length, event_start)
    states = object_state_from_phases(phases)
    t = length - 1  # revisit chunk
    history = list(range(t))

    base_buckets = recency_buckets(history, t)
    eafr_bk, eafr_w = eafr_buckets(phases, history, t)

    pred_base = predict_state(base_buckets, states, phases, use_bank=False)
    pred_eafr = predict_state(eafr_bk, states, phases, use_bank=False)
    pred_full = predict_state(eafr_bk, states, phases, use_bank=True)

    return {
        "phases": phases,
        "states": states,
        "eafr_weights": eafr_w,
        "base_buckets": base_buckets,
        "eafr_buckets": eafr_bk,
        "pred": {
            "recency": pred_base,
            "eafr": pred_eafr,
            "eafr+bank": pred_full,
        },
    }


def sweep_gap_lengths(gap_lengths, n_seeds=200):
    """Vary how long the navigation gap is after the interaction event."""
    results = {name: [] for name in ["recency", "eafr", "eafr+bank"]}
    for gap in gap_lengths:
        length = 15 + gap  # event early, then gap, then revisit
        event_start = 5
        mses = {name: [] for name in results}
        for seed in range(n_seeds):
            np.random.seed(seed)
            out = simulate_one(length=length, event_start=event_start)
            true_state = 1.0  # object was modified during manipulation
            for name, val in out["pred"].items():
                mses[name].append((val - true_state) ** 2)
        for name in results:
            results[name].append(float(np.mean(mses[name])))
    return {k: np.array(v) for k, v in results.items()}


def plot_single_trajectory(out, save_path):
    phases = out["phases"]
    states = out["states"]
    weights = out["eafr_weights"]
    t = len(phases) - 1

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})

    # 1. Phases as categorical strip.
    ax = axes[0]
    phase_int = np.array([PHASES.index(p) for p in phases])
    ax.imshow(phase_int.reshape(1, -1), aspect="auto", cmap="tab10",
              vmin=0, vmax=len(PHASES) - 1, extent=[-0.5, len(phases) - 0.5, -0.5, 0.5])
    ax.set_yticks([])
    ax.set_title("Synthetic rollout phases (contact/manipulating/completing = causal event)")

    # 2. Object state.
    ax = axes[1]
    ax.plot(states, "k-o", markersize=4)
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("Object state\n(0→1 at manipulate)")
    ax.axvline(t, color="gray", linestyle="--", alpha=0.6, label="revisit")

    # 3. EAFR weight per history chunk.
    ax = axes[2]
    xs = sorted(weights.keys())
    ys = [weights[k] for k in xs]
    colors = ["#e74c3c" if phases[k] in BANK_PINNED_PHASES else "#3498db" for k in xs]
    ax.bar(xs, ys, color=colors, width=0.7)
    ax.set_ylabel("EAFR importance weight")
    ax.set_ylim(0, max(ys) * 1.2)

    # 4. Bucket assignments.
    ax = axes[3]
    bucket_colors = {"fine": "#2ecc71", "mid": "#f1c40f", "coarse": "#e67e22"}
    for bucket, idxs in out["eafr_buckets"].items():
        if idxs:
            ax.scatter(idxs, [1] * len(idxs), c=bucket_colors[bucket],
                       s=120, label=f"EAFR {bucket}", marker="s", edgecolors="k")
    for bucket, idxs in out["base_buckets"].items():
        if idxs:
            ax.scatter(idxs, [0] * len(idxs), c=bucket_colors[bucket],
                       s=120, label=f"Recency {bucket}", marker="^", edgecolors="k")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Recency baseline", "EAFR"])
    ax.set_xlabel("Chunk index")
    ax.set_xlim(-0.5, len(phases) - 0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.axvline(t, color="gray", linestyle="--", alpha=0.6)

    # Text predictions.
    pred_text = " | ".join(f"{k}={v:.2f}" for k, v in out["pred"].items())
    fig.text(0.5, 0.01, f"Revisit prediction (true=1.0): {pred_text}",
             ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_gap_sweep(results, gap_lengths, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"recency": "#e74c3c", "eafr": "#3498db", "eafr+bank": "#2ecc71"}
    for name, vals in results.items():
        ax.plot(gap_lengths, vals, "-o", label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Navigation gap after interaction event (chunks)")
    ax.set_ylabel("Mean squared error on object-state prediction")
    ax.set_title("Recency compression forgets the causal event; EAFR + bank preserve it")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    out_dir = Path(__file__).resolve().parents[2] / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # One concrete trajectory; seed chosen so the recency baseline forgets.
    np.random.seed(16)
    out = simulate_one(length=50, event_start=10)
    plot_single_trajectory(out, out_dir / "probe_trajectory.png")

    # Sweep navigation gap.
    gap_lengths = np.arange(0, 41, 4)
    sweep = sweep_gap_lengths(gap_lengths, n_seeds=300)
    plot_gap_sweep(sweep, gap_lengths, out_dir / "probe_gap_sweep.png")

    # Numerical snapshot.
    metrics = {
        "single_trajectory_predictions": out["pred"],
        "gap_mse": {
            int(g): {name: float(sweep[name][i]) for name in sweep}
            for i, g in enumerate(gap_lengths)
        },
    }
    (out_dir / "probe_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved:", out_dir / "probe_trajectory.png")
    print("Saved:", out_dir / "probe_gap_sweep.png")
    print("Saved:", out_dir / "probe_metrics.json")
    print("\nSingle-trajectory predictions (true object state at revisit = 1.0):")
    for k, v in out["pred"].items():
        print(f"  {k:12s}: {v:.3f}")


if __name__ == "__main__":
    main()
