"""
DenseReward synthetic reward-curve probe.

Reconstructs the per-timestep dense reward curves implied by the paper's
five-phase, six-failure-mode formulation.  The paper does not give an explicit
closed-form reward function, so this script encodes the qualitative rules
stated in Section 3.1-3.2:

  Phases: Reach -> Grasp -> Lift -> Move -> Place
  Modes : Success, Collision, Miss, Fall, Smooth, Recover

The probe is intentionally schematic: it shows the *shape* the paper claims
(success monotonic rise, collision/miss/fall mountain-shaped then decay,
smooth scaled/penalised, recover dip-then-resume) rather than claiming to
reproduce the exact dataset labels.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Phase decomposition (Section 3.1)
# ---------------------------------------------------------------------------
PHASES = ["Reach", "Grasp", "Lift", "Move", "Place"]
PHASE_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # reward interval per phase


def phase_of(t, T):
    """Map timestep t in [0, T-1] to one of the five phases."""
    p = int(np.clip(t / T, 0, 0.9999) * 5)
    return p


def base_progress(t, T):
    """Linear phase-progress reward for a nominal successful trajectory."""
    return t / T


# ---------------------------------------------------------------------------
# Failure-mode reward builders (Section 3.2)
# ---------------------------------------------------------------------------
def success_curve(T):
    """Mode 1: unperturbed execution, reward rises monotonically to 1."""
    return np.linspace(0.0, 1.0, T)


def collision_curve(T, collision_frac=0.55, decay_rate=6.0):
    """
    Mode 2: motion planned with collision avoidance disabled.
    Reward rises with phase progress until the collision event, then decays.
    """
    t = np.linspace(0, 1, T)
    peak = base_progress(int(collision_frac * T), T)
    reward = np.where(
        t <= collision_frac,
        peak * (t / collision_frac),                 # rise to collision
        peak * np.exp(-decay_rate * (t - collision_frac))  # decay after
    )
    return reward


def miss_curve(T, miss_frac=0.35, decay_rate=5.0):
    """
    Mode 3: grasp pose offset -> gripper closes in air.
    Reward rises until the failed grasp attempt, then decays.
    """
    t = np.linspace(0, 1, T)
    peak = base_progress(int(miss_frac * T), T)
    reward = np.where(
        t <= miss_frac,
        peak * (t / miss_frac),
        peak * np.exp(-decay_rate * (t - miss_frac))
    )
    return reward


def fall_curve(T, lift_frac=0.42, drop_frac=0.68, decay_rate=5.0):
    """
    Mode 4: rotation perturbations during Move cause object to fall.
    Reward rises through Reach/Grasp/Lift, peaks at the drop event,
    then decays (partial progress ultimately unsuccessful).
    """
    t = np.linspace(0, 1, T)
    peak = base_progress(int(drop_frac * T), T)
    # Pre-drop: normal progress up to the drop point.
    reward = np.where(
        t <= drop_frac,
        peak * (t / drop_frac),
        peak * np.exp(-decay_rate * (t - drop_frac))
    )
    return reward


def smooth_curve(T, penalty=0.25, noise_scale=0.03):
    """
    Mode 5: small Gaussian joint noise at every timestep -> jittery trajectory.
    Task completes but reward is a penalised, scaled version of success.
    """
    rng = np.random.default_rng(seed=5)
    r = np.linspace(0.0, 1.0 - penalty, T) + rng.normal(0, noise_scale, T)
    return np.clip(r, 0.0, 1.0)


def recover_curve(T, collision_start=0.35, collision_end=0.50):
    """
    Mode 6: initial collision, then replanned clear path to completion.
    Reward drops during the collision window, then resumes climbing.
    """
    t = np.linspace(0, 1, T)
    r = np.zeros_like(t)
    for i, x in enumerate(t):
        if x < collision_start:
            # Normal progress up to collision.
            r[i] = base_progress(int(x * T), T)
        elif x < collision_end:
            # Collision window: drop to a low value.
            r[i] = 0.15
        else:
            # Recovery: resume climbing from the low value to 1.0.
            frac = (x - collision_end) / (1.0 - collision_end)
            r[i] = 0.15 + 0.85 * frac
    return r


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(T=100, out_path="dense_reward_curves.png"):
    curves = {
        "Success": success_curve(T),
        "Collision": collision_curve(T),
        "Miss": miss_curve(T),
        "Fall": fall_curve(T),
        "Smooth": smooth_curve(T),
        "Recover": recover_curve(T),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))

    for ax, (name, reward), c in zip(axes, curves.items(), colors):
        t = np.arange(T)
        ax.plot(t, reward, color=c, lw=2.2)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, T - 1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("timestep")
        ax.set_ylabel("dense reward r_t")

        # Mark phase boundaries as vertical reference lines.
        for b in [0.2, 0.4, 0.6, 0.8]:
            ax.axvline(b * T, color="gray", ls="--", alpha=0.4, lw=0.8)

    fig.suptitle(
        "DenseReward: synthetic per-timestep reward curves by failure mode\n"
        "(five phases: Reach | Grasp | Lift | Move | Place)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import os
    out = os.path.join(
        os.path.dirname(__file__),
        "dense_reward_curves.png",
    )
    plot_all(T=100, out_path=out)
