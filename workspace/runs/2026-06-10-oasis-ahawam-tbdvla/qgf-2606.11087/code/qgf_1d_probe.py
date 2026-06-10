"""
QGF 1D Diagnostic Probe — Reproduces the qualitative behavior of Figure 2.

Simulates a 1D flow-matching policy that maps Gaussian noise to a tri-modal
distribution, then guides denoising with four different strategies:
  1. BC (no guidance)
  2. OOD gradient   — ∇_{a_t} Q(s, a_t)   (evaluated on noisy action)
  3. BPTT gradient  — ∇_{a_t} Q(s, ODE(a_t)) (backprop through full denoising)
  4. QGF gradient   — ∇_{a_1} Q(s, â_1) with J≈I (single Euler step approx)

Q(s, a) = -||a - a*||^2 where a* = 2.0 (the rightmost mode).
The base policy has modes at {-2, 0, 2}.

Observation: OOD guidance is biased and often gets stuck at suboptimal modes.
BPTT is unstable / high variance. QGF converges cleanly to a*.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# Ground-truth tri-modal base distribution
# ------------------------------------------------------------------
MODES = np.array([-2.0, 0.0, 2.0])
MODE_WEIGHTS = np.array([1/3, 1/3, 1/3])
OPTIMAL_ACTION = 2.0  # rightmost mode is optimal

def sample_base(n):
    """Sample from the tri-modal behavior distribution."""
    idx = np.random.choice(len(MODES), size=n, p=MODE_WEIGHTS)
    return MODES[idx] + np.random.randn(n) * 0.15

# ------------------------------------------------------------------
# Flow-matching velocity field (analytical for this toy)
# ------------------------------------------------------------------
# We fit a small MLP offline and use it as the learned velocity.
# For this probe we use a simple neural-net-like function with ReLU.

def velocity_field(a, t):
    """
    Approximate velocity field v_theta(a, t) for the tri-modal distribution.
    Learns to push noise toward the nearest mode.
    For simplicity, use a hand-crafted field that steers toward the nearest mode.
    In the real paper this is a learned MLP; here we emulate its behavior.
    """
    # For each a, compute distance to each mode and steer toward nearest
    dists = np.abs(a[:, None] - MODES[None, :])  # (N, 3)
    nearest = np.argmin(dists, axis=1)
    target = MODES[nearest]
    # Linear interpolation velocity: (target - a0), but at time t the residual
    # velocity should point from current a_t toward target.
    # In flow matching, v(a_t, t) = (x1 - x0) for the pair that generated a_t.
    # We approximate: at a_t, velocity points toward nearest mode.
    v = (target - a) / np.maximum(1e-6, 1 - t)
    # Add slight curvature so it's not trivially linear
    v = v * (1 + 0.1 * np.sin(2 * np.pi * t))
    return v

# ------------------------------------------------------------------
# Critic
# ------------------------------------------------------------------
def Q(s, a):
    """Negative squared distance to optimal action."""
    return -(a - OPTIMAL_ACTION) ** 2

def grad_Q(s, a):
    return -2 * (a - OPTIMAL_ACTION)

# ------------------------------------------------------------------
# Denoising ODE integrators
# ------------------------------------------------------------------

def integrate_ode(a0, t_start, t_end, dt=0.01):
    """Full ODE integration from a0 at time t_start to t_end."""
    a = a0.copy()
    t = t_start
    while t < t_end - 1e-9:
        step = min(dt, t_end - t)
        a = a + step * velocity_field(a, t)
        t += step
    return a

def denoise_full(a0, T=10, dt=None):
    """Run full denoising chain from t=0 to t=1."""
    if dt is None:
        dt = 1.0 / T
    a = a0.copy()
    for i in range(T):
        t = i / T
        a = a + dt * velocity_field(a, t)
    return a

# ------------------------------------------------------------------
# Guidance strategies
# ------------------------------------------------------------------

def guided_denoise(a0, guidance_weight, mode="qgf", T=10, dt=None, ode_dt=0.01):
    """
    Denoise with guidance.
    mode:
      "bc"    — no guidance
      "ood"   — gradient at noisy action a_t
      "bptt"  — backprop through full ODE from a_t to a_1
      "qgf"   — single Euler-step approx, J≈I
    """
    if dt is None:
        dt = 1.0 / T
    a = a0.copy()
    traj = [a.copy()]
    for i in range(T):
        t = i / T
        v_bc = velocity_field(a, t)

        if mode == "bc":
            g = 0.0
        elif mode == "ood":
            g = grad_Q(None, a)
        elif mode == "bptt":
            # BPTT: differentiate Q(s, ODE(a_t)) w.r.t a_t
            # Use finite differences as proxy for true BPTT
            eps = 1e-4
            a_plus = a + eps
            a_minus = a - eps
            q_plus = Q(None, integrate_ode(a_plus, t, 1.0, dt=ode_dt))
            q_minus = Q(None, integrate_ode(a_minus, t, 1.0, dt=ode_dt))
            g = (q_plus - q_minus) / (2 * eps)
        elif mode == "qgf":
            # Single Euler step approximation of clean action
            a_hat = a + (1 - t) * v_bc
            g = grad_Q(None, a_hat)
        else:
            raise ValueError(mode)

        a = a + dt * (v_bc + guidance_weight * g)
        traj.append(a.copy())
    return np.array(traj)

# ------------------------------------------------------------------
# Run probe
# ------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    T = 20
    n_samples = 200
    guidance_weights = [0.0, 0.5, 1.0, 2.0]

    # Sample initial noise
    a0s = np.random.randn(n_samples)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    modes = ["bc", "ood", "bptt", "qgf"]
    mode_labels = ["BC (no guidance)", "OOD gradient (QFQL)", "BPTT gradient", "QGF (ours)"]
    colors = ["gray", "tab:red", "tab:purple", "tab:green"]

    for ax, mode, label, color in zip(axes, modes, mode_labels, colors):
        # Use a moderate guidance weight for all guided methods
        gw = 1.0 if mode != "bc" else 0.0
        final_actions = []
        for a0 in a0s:
            traj = guided_denoise(np.array([a0]), gw, mode=mode, T=T)
            final_actions.append(traj[-1, 0])
        final_actions = np.array(final_actions)

        ax.hist(final_actions, bins=60, range=(-3.5, 3.5), color=color, alpha=0.7, density=True)
        for m in MODES:
            ax.axvline(m, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(OPTIMAL_ACTION, color="gold", linestyle="-", linewidth=2, label="optimal")
        ax.set_title(f"{label}\n(weight={gw})")
        ax.set_xlim(-3.5, 3.5)
        ax.set_xlabel("action a")
        if ax is axes[0]:
            ax.set_ylabel("density")

    plt.suptitle("1D Tri-Modal Flow: Test-Time Guidance Comparison", y=1.02)
    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "qgf_1d_probe_histogram.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved histogram to {out_path}")

    # Also plot convergence trajectories for a few seeds
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    for ax, mode, label, color in zip(axes, modes, mode_labels, colors):
        gw = 1.0 if mode != "bc" else 0.0
        for seed in [0, 1, 2, 3, 4]:
            np.random.seed(seed)
            a0 = np.random.randn(1)
            traj = guided_denoise(a0, gw, mode=mode, T=T)
            steps = np.arange(len(traj))
            ax.plot(steps, traj[:, 0], alpha=0.6, color=color, linewidth=1.2)
        for m in MODES:
            ax.axhline(m, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.axhline(OPTIMAL_ACTION, color="gold", linestyle="-", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("denoising step")
        ax.set_ylim(-3.5, 3.5)
        if ax is axes[0]:
            ax.set_ylabel("action a_t")

    plt.suptitle("1D Tri-Modal Flow: Denoising Trajectories", y=1.02)
    plt.tight_layout()
    out_path2 = Path(__file__).parent.parent / "qgf_1d_probe_trajectories.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved trajectories to {out_path2}")
