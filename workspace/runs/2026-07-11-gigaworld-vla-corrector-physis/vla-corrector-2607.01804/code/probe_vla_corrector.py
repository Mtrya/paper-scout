#!/usr/bin/env python3
"""
Synthetic probe of VLA-Corrector's detect-and-correct loop.

This is a humble reconstruction, not the official repo.  It shows:
  1. A fixed-horizon action-chunked baseline accumulating open-loop error after a perturbation.
  2. A lightweight latent-dynamics corrector trained on normal trajectories.
  3. An LVM-style online inconsistency score and event-triggered truncation.
  4. OGG-style gradient guidance on the recovery replan.
  5. The corrector is tiny compared with the (simulated) VLA backbone.

Run with:
    python probe_vla_corrector.py
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Tiny NumPy MLP with forward + backward.
# -----------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


class MLP:
    """Fully-connected MLP with ReLU activations."""

    def __init__(self, layer_dims: list[int], seed: int | None = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        self.dims = layer_dims
        self.Ws = []
        self.bs = []
        for i in range(len(layer_dims) - 1):
            # He-ish init
            w = rng.normal(0.0, math.sqrt(2.0 / layer_dims[i]), (layer_dims[i], layer_dims[i + 1]))
            b = np.zeros(layer_dims[i + 1], dtype=np.float64)
            self.Ws.append(w.astype(np.float64))
            self.bs.append(b)
        self._cache_z = []  # pre-activation
        self._cache_a = []  # post-activation

    def n_params(self) -> int:
        return sum(int(w.size + b.size) for w, b in zip(self.Ws, self.bs))

    def flops_forward(self, batch_size: int = 1) -> int:
        """Approximate multiply-add FLOPs for one forward pass."""
        total = 0
        for i in range(len(self.Ws)):
            total += 2 * batch_size * self.dims[i] * self.dims[i + 1]
        return total

    def forward(self, x: np.ndarray, store: bool = False) -> np.ndarray:
        a = x
        if store:
            self._cache_z.clear()
            self._cache_a.clear()
            self._cache_a.append(a)
        for W, b in zip(self.Ws, self.bs):
            z = a @ W + b
            if store:
                self._cache_z.append(z)
            a = _relu(z)
            if store:
                self._cache_a.append(a)
        return a

    def backward_to_input(self, loss_grad_output: np.ndarray) -> np.ndarray:
        """Back-propagate a gradient at the output back to the network input.

        Must have called forward(..., store=True) first.
        """
        g = loss_grad_output
        for i in reversed(range(len(self.Ws))):
            g = g * _relu_grad(self._cache_z[i])
            g = g @ self.Ws[i].T
        return g

    def sgd_step(self, x: np.ndarray, y: np.ndarray, lr: float, beta_cos: float = 0.0) -> float:
        """One SGD step on MSE + beta_cos * (1 - cosine_similarity)."""
        pred = self.forward(x, store=True)
        diff = pred - y
        mse_grad = 2.0 * diff / x.shape[0]

        if beta_cos > 0.0:
            # Gradient of 1 - cosine_similarity(pred, y) w.r.t. pred.
            pred_n = np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8
            y_n = np.linalg.norm(y, axis=1, keepdims=True) + 1e-8
            cos_grad = (pred / (pred_n * y_n)) - (y * (pred_n / (y_n * pred_n ** 2 + 1e-8)))
            cos_grad = cos_grad / x.shape[0]
            g_out = mse_grad + beta_cos * cos_grad
        else:
            g_out = mse_grad

        # Backprop through layers and update weights.
        g = g_out
        for i in reversed(range(len(self.Ws))):
            a_prev = self._cache_a[i]
            z = self._cache_z[i]
            g = g * _relu_grad(z)
            grad_W = a_prev.T @ g / x.shape[0]
            grad_b = g.mean(axis=0)
            # SGD update
            self.Ws[i] -= lr * grad_W
            self.bs[i] -= lr * grad_b
            g = g @ self.Ws[i].T

        loss = float(np.mean(diff ** 2))
        if beta_cos > 0.0:
            cos = np.sum(pred * y, axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(y, axis=1) + 1e-8)
            loss += beta_cos * float(np.mean(1.0 - cos))
        return loss


# -----------------------------------------------------------------------------
# Synthetic environment and VLA-style policy.
# -----------------------------------------------------------------------------

STATE_DIM = 2
ACTION_DIM = 2
TOKEN_DIM = 32
MAX_SPEED = 0.4
KP = 1.0
K_STEP = 5

# Fixed visual encoder: maps state -> latent token.
ENCODER = MLP([STATE_DIM, 32, 32, TOKEN_DIM], seed=0)


def encode_state(s: np.ndarray) -> np.ndarray:
    return ENCODER.forward(s)


def clip_action(a: np.ndarray, max_speed: float = MAX_SPEED) -> np.ndarray:
    norm = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
    scale = np.minimum(1.0, max_speed / norm)
    return a * scale


def policy_action_chunk(state: np.ndarray, target: np.ndarray, horizon: int) -> np.ndarray:
    """Deterministic VLA-style controller: each action points toward the current target."""
    a = clip_action(KP * (target - state))
    return np.tile(a, (horizon, 1))


def step_state(state: np.ndarray, action: np.ndarray, noise_scale: float = 0.02) -> np.ndarray:
    return state + clip_action(action) + np.random.normal(0.0, noise_scale, size=state.shape)


def generate_normal_trajectory(length: int = 80) -> tuple[list, list, list]:
    """Generate one demonstration with no perturbation."""
    target = np.random.uniform(-2.0, 2.0, size=STATE_DIM)
    state = target + np.random.uniform(-1.2, 1.2, size=STATE_DIM)
    states, actions, latents = [state.copy()], [], []
    for _ in range(length):
        a = policy_action_chunk(state, target, 1)[0]
        actions.append(a.copy())
        state = step_state(state, a)
        states.append(state.copy())
        latents.append(encode_state(state.reshape(1, -1))[0])
    return states, actions, latents


# -----------------------------------------------------------------------------
# Corrector training.
# -----------------------------------------------------------------------------


def train_corrector(n_demo: int = 400, epochs: int = 30, batch_size: int = 256) -> MLP:
    """Train a latent-dynamics corrector on normal trajectories."""
    corrector = MLP([TOKEN_DIM + ACTION_DIM, 64, 64, 64, TOKEN_DIM], seed=1)

    # Collect transition dataset (z_t, a_t, z_{t+K_STEP}).
    xs, ys = [], []
    for _ in range(n_demo):
        states, actions, latents = generate_normal_trajectory(length=120)
        for t in range(len(latents) - K_STEP):
            zt = latents[t]
            at = actions[t]
            zt_k = latents[t + K_STEP]
            xs.append(np.concatenate([zt, at]))
            ys.append(zt_k - zt)
    xs = np.stack(xs)
    ys = np.stack(ys)

    # Normalize targets to keep training stable.
    y_mean = ys.mean(axis=0)
    y_std = ys.std(axis=0) + 1e-6
    ys_norm = (ys - y_mean) / y_std

    n = xs.shape[0]
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            loss = corrector.sgd_step(xs[batch_idx], ys_norm[batch_idx], lr=5e-3, beta_cos=0.2)
            losses.append(loss)
        if (ep + 1) % 10 == 0:
            print(f"  corrector epoch {ep + 1:3d}  loss={np.mean(losses):.4f}")

    # Wrap corrector so predictions are in original residual scale.
    class _Normed:
        def __init__(self, net: MLP, mean: np.ndarray, std: np.ndarray):
            self.net = net
            self.mean = mean
            self.std = std

        def predict(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
            x = np.concatenate([z.reshape(-1, TOKEN_DIM), a.reshape(-1, ACTION_DIM)], axis=1)
            return self.net.forward(x) * self.std + self.mean

    return _Normed(corrector, y_mean, y_std)


# -----------------------------------------------------------------------------
# Online monitor and OGG guidance.
# -----------------------------------------------------------------------------


def cosine_error(pred: np.ndarray, real: np.ndarray) -> float:
    pred = pred.flatten()
    real = real.flatten()
    denom = np.linalg.norm(pred) * np.linalg.norm(real) + 1e-8
    return float(1.0 - np.dot(pred, real) / denom)


class RobustMonitor:
    """Simplified LVM with median+MAD thresholds and persistence."""

    def __init__(
        self,
        window_size: int = 15,
        k_on: float = 3.0,
        k_off: float = 2.0,
        persistence: int = 3,
        cooldown: int = 5,
    ):
        self.window = deque(maxlen=window_size)
        self.k_on = k_on
        self.k_off = k_off
        self.persistence = persistence
        self.cooldown = cooldown
        self.over_count = 0
        self.cooldown_count = 0

    def update(self, error: float) -> tuple[float, bool]:
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return float("nan"), False

        self.window.append(error)
        if len(self.window) < 10:
            return float("nan"), False

        arr = np.asarray(self.window)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-8
        ton = med + self.k_on * mad
        toff = med + self.k_off * mad

        if error > ton:
            self.over_count += 1
        else:
            self.over_count = 0

        if error < toff:
            self.over_count = 0

        if self.over_count >= self.persistence:
            self.cooldown_count = self.cooldown
            self.over_count = 0
            return ton, True
        return ton, False


def ogg_correct_first_action(
    corrector_net: MLP,
    z_current: np.ndarray,
    action0: np.ndarray,
    target_delta: np.ndarray,
    eta: float = 0.5,
    steps: int = 5,
) -> np.ndarray:
    """Apply a small number of gradient steps to align the predicted action effect
    with the corrective latent direction.  Returns the refined first action."""
    a = action0.copy()
    for _ in range(steps):
        x = np.concatenate([z_current.reshape(1, -1), a.reshape(1, -1)], axis=1)
        pred = corrector_net.forward(x, store=True)
        pred_flat = pred.reshape(1, -1)
        tgt_flat = target_delta.reshape(1, -1)
        pn = np.linalg.norm(pred_flat, axis=1, keepdims=True) + 1e-8
        tn = np.linalg.norm(tgt_flat, axis=1, keepdims=True) + 1e-8
        # Loss = 1 - cosine_similarity(pred, target)
        cos = np.sum(pred_flat * tgt_flat, axis=1, keepdims=True) / (pn * tn)
        # dL/dpred = pred/(pn*tn) - target * pn/(tn * pn^2)
        grad_pred = (pred_flat / (pn * tn)) - (tgt_flat * (pn / (tn * pn ** 2)))
        grad_a = corrector_net.backward_to_input(grad_pred)
        # Gradient w.r.t. the action part of the input.
        a -= eta * grad_a[0, TOKEN_DIM:]
        a = clip_action(a, MAX_SPEED)
    return a


# -----------------------------------------------------------------------------
# Rollout variants.
# -----------------------------------------------------------------------------


def rollout_baseline(
    horizon: int,
    length: int = 80,
    perturb_t: int = 25,
    perturb_radius: float = 1.5,
) -> dict:
    target = np.random.uniform(-2.0, 2.0, size=STATE_DIM)
    state = target + np.random.uniform(-1.2, 1.2, size=STATE_DIM)
    queue = deque()
    policy_calls = 0
    states = [state.copy()]
    targets = [target.copy()]

    for t in range(length):
        if t == perturb_t:
            angle = np.random.uniform(0.0, 2 * math.pi)
            target = target + perturb_radius * np.array([math.cos(angle), math.sin(angle)])

        if not queue:
            chunk = policy_action_chunk(state, target, horizon)
            queue.extend(chunk)
            policy_calls += 1

        a = queue.popleft()
        state = step_state(state, a)
        states.append(state.copy())
        targets.append(target.copy())

    return {
        "states": np.stack(states),
        "targets": np.stack(targets),
        "policy_calls": policy_calls,
        "truncations": 0,
    }


def rollout_corrector(
    corrector_norm,
    corrector_net: MLP,
    horizon: int,
    use_ogg: bool = True,
    length: int = 80,
    perturb_t: int = 25,
    perturb_radius: float = 1.5,
) -> dict:
    target = np.random.uniform(-2.0, 2.0, size=STATE_DIM)
    state = target + np.random.uniform(-1.2, 1.2, size=STATE_DIM)
    queue = deque()
    policy_calls = 0
    truncations = 0
    monitor = RobustMonitor()
    z_history: deque = deque(maxlen=K_STEP + 1)
    pred_queue: deque = deque()
    states = [state.copy()]
    targets = [target.copy()]

    for t in range(length):
        if t == perturb_t:
            angle = np.random.uniform(0.0, 2 * math.pi)
            target = target + perturb_radius * np.array([math.cos(angle), math.sin(angle)])

        z_t = encode_state(state.reshape(1, -1))[0]
        z_history.append(z_t.copy())

        # Schedule expected residual for k-step lookahead monitor.
        if queue and t + K_STEP <= length:
            a_exec = queue[0]
            pred_delta = corrector_norm.predict(z_t, a_exec)[0]
            pred_queue.append((t + K_STEP, pred_delta))

        # LVM check.
        triggered = False
        if len(z_history) == K_STEP + 1 and pred_queue:
            target_t, delta_pred = pred_queue[0]
            if target_t == t:
                z_t_minus_k = z_history[0]
                delta_real = z_t - z_t_minus_k
                error = cosine_error(delta_pred, delta_real)
                _, triggered = monitor.update(error)
                pred_queue.popleft()

        if triggered:
            queue.clear()
            pred_queue.clear()
            truncations += 1

        if not queue:
            chunk = policy_action_chunk(state, target, horizon)
            # OGG refinement on the recovery replan.
            if use_ogg and triggered and len(z_history) == K_STEP + 1:
                z_current = z_t.reshape(1, -1)
                z_minus_k = z_history[0].reshape(1, -1)
                delta_dev = (z_current - z_minus_k).reshape(1, -1)
                # Expected residual from step t-k was already predicted, but recompute for clarity.
                a_at_minus_k = chunk[0].reshape(1, -1)  # proxy for the action that caused the deviation
                delta_exp = corrector_norm.predict(z_minus_k, a_at_minus_k)
                corrective = delta_exp - delta_dev
                chunk[0] = ogg_correct_first_action(
                    corrector_net, z_current, chunk[0], corrective[0], eta=0.5, steps=5
                )
            queue.extend(chunk)
            policy_calls += 1

        a = queue.popleft()
        state = step_state(state, a)
        states.append(state.copy())
        targets.append(target.copy())

    return {
        "states": np.stack(states),
        "targets": np.stack(targets),
        "policy_calls": policy_calls,
        "truncations": truncations,
    }


# -----------------------------------------------------------------------------
# Experiment harness.
# -----------------------------------------------------------------------------


def final_distance(states: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(states[-1] - target))


def run_experiments(n_episodes: int = 200):
    print("Training latent-dynamics corrector...")
    corrector_norm = train_corrector(n_demo=400, epochs=30, batch_size=256)
    corrector_net = corrector_norm.net
    print(f"Corrector parameters: {corrector_net.n_params():,}")
    print(f"Base encoder parameters: {ENCODER.n_params():,}")
    print(
        f"Corrector forward FLOPs: {corrector_net.flops_forward():,}  "
        f"vs encoder forward FLOPs: {ENCODER.flops_forward():,}"
    )

    horizons = [5, 10, 20, 50]
    results = {h: [] for h in horizons}
    results["corr_trunc"] = []
    results["corr_ogg"] = []

    for ep in range(n_episodes):
        if (ep + 1) % 50 == 0:
            print(f"  episode {ep + 1}/{n_episodes}")
        seed = ep + 100
        np.random.seed(seed)
        random.seed(seed)
        perturb_t = np.random.randint(15, 45)
        perturb_radius = np.random.uniform(0.6, 1.0)

        for h in horizons:
            r = rollout_baseline(h, perturb_t=perturb_t, perturb_radius=perturb_radius)
            results[h].append(r)

        r_t = rollout_corrector(
            corrector_norm,
            corrector_net,
            horizon=20,
            use_ogg=False,
            perturb_t=perturb_t,
            perturb_radius=perturb_radius,
        )
        results["corr_trunc"].append(r_t)

        r_o = rollout_corrector(
            corrector_norm,
            corrector_net,
            horizon=20,
            use_ogg=True,
            perturb_t=perturb_t,
            perturb_radius=perturb_radius,
        )
        results["corr_ogg"].append(r_o)

    return results


def summarize(label: str, rollouts: list, threshold: float = 0.8) -> dict:
    dists = [final_distance(r["states"], r["targets"][-1]) for r in rollouts]
    calls = [r["policy_calls"] for r in rollouts]
    truncs = [r["truncations"] for r in rollouts]
    succ = [d < threshold for d in dists]
    return {
        "label": label,
        "mean_final_dist": float(np.mean(dists)),
        "success_rate": float(np.mean(succ)),
        "mean_policy_calls": float(np.mean(calls)),
        "success_per_call": float(np.mean(succ) / (np.mean(calls) + 1e-8)),
        "mean_truncations": float(np.mean(truncs)),
    }


def plot_trajectories(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ["Baseline H=20", "+ Truncation", "+ Truncation + OGG"]
    keys = [20, "corr_trunc", "corr_ogg"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    # Shared axis limits across all three panels.
    all_states = np.concatenate([results[k][0]["states"] for k in keys], axis=0)
    all_targets = np.concatenate([results[k][0]["targets"] for k in keys], axis=0)
    margin = 0.5
    xmin, xmax = all_states[:, 0].min() - margin, all_states[:, 0].max() + margin
    ymin, ymax = all_states[:, 1].min() - margin, all_states[:, 1].max() + margin

    for ax, key, label, color in zip(axes, keys, labels, colors):
        r = results[key][0]
        states = r["states"]
        targets = r["targets"]
        ax.plot(states[:, 0], states[:, 1], color=color, label="trajectory")
        ax.scatter(targets[0, 0], targets[0, 1], c="black", marker="x", s=80, label="initial target")
        ax.scatter(targets[-1, 0], targets[-1, 1], c="green", marker="*", s=120, label="perturbed target")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=7)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "probe_trajectories.png", dpi=150)
    plt.close(fig)


def plot_metrics(summary_rows: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(x, [row["success_rate"] for row in summary_rows], color="steelblue")
    axes[0].set_ylabel("Success rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim([0, 1])
    axes[0].set_title("Task success")

    axes[1].bar(x, [row["mean_policy_calls"] for row in summary_rows], color="coral")
    axes[1].set_ylabel("Avg. policy calls")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_title("Policy-call frequency")

    axes[2].bar(x, [row["success_per_call"] for row in summary_rows], color="seagreen")
    axes[2].set_ylabel("Success / policy call")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_title("Success-per-call efficiency")

    fig.tight_layout()
    fig.savefig(out_dir / "probe_metrics.png", dpi=150)
    plt.close(fig)


def main():
    np.random.seed(7)
    random.seed(7)

    results = run_experiments(n_episodes=200)
    summary_rows = []
    for h in [5, 10, 20, 50]:
        summary_rows.append(summarize(f"Baseline H={h}", results[h]))
    summary_rows.append(summarize("+ Truncation (H=20)", results["corr_trunc"]))
    summary_rows.append(summarize("+ Truncation + OGG (H=20)", results["corr_ogg"]))

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"{row['label']:28s}  dist={row['mean_final_dist']:.3f}  "
            f"succ={row['success_rate']:.2%}  calls={row['mean_policy_calls']:.2f}  "
            f"succ/call={row['success_per_call']:.3f}  truncs={row['mean_truncations']:.2f}"
        )

    out_dir = Path(__file__).parent / "probe_outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    plot_trajectories(results, out_dir)
    plot_metrics(summary_rows, out_dir)
    print(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
