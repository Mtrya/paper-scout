"""Minimal reconstruction of LabVLA's DiT flow-matching action head.

The real head is in `src/policies/LabVLA/dit_action_head.py` and is consumed by
`src/policies/LabVLA/modeling_labvla.py`. It cross-attends to VLM prefix
features, conditions on a flow timestep, and predicts the velocity field that
maps Gaussian noise to a clean action chunk.

This script implements the *mechanism* in NumPy with tiny random MLPs:
  - prefix features from a dummy VLM,
  - a small MLP that conditions on timestep and predicts velocity,
  - flow matching training loss (MSE between predicted and target velocity),
  - 10-step Euler sampling at inference.

PyTorch is not required.

Run: python flow_matching_action_head.py
"""
from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


class MLPVelocityModel:
    """Tiny velocity network: v_theta(x_t, state, prefix, t)."""

    def __init__(self, action_dim: int, state_dim: int, prefix_dim: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.action_w = rng.normal(0, 0.1, (action_dim, hidden))
        self.state_w = rng.normal(0, 0.1, (state_dim, hidden))
        self.prefix_w = rng.normal(0, 0.1, (prefix_dim, hidden))
        self.time_w = rng.normal(0, 0.1, (1, hidden))
        self.h1_w = rng.normal(0, 0.1, (hidden, hidden))
        self.h1_b = np.zeros(hidden)
        self.out_w = rng.normal(0, 0.1, (hidden, action_dim))
        self.out_b = np.zeros(action_dim)

    def __call__(
        self,
        x_t: np.ndarray,    # (B, K, D)
        state: np.ndarray,  # (B, S)
        prefix: np.ndarray, # (B, L, P)
        t: np.ndarray,      # (B,)
    ) -> np.ndarray:
        # Aggregate prefix via mean pooling.
        prefix_pooled = prefix.mean(axis=1)  # (B, P)
        # Per-timestep conditioning.
        B, K, D = x_t.shape
        h = np.zeros((B, K, self.h1_w.shape[0]))
        h += x_t @ self.action_w
        h += state[:, None, :] @ self.state_w
        h += prefix_pooled[:, None, :] @ self.prefix_w
        h += t[:, None, None] @ self.time_w
        h = relu(h @ self.h1_w + self.h1_b)
        return h @ self.out_w + self.out_b


def sample_time(b: int, rng: np.random.Generator) -> np.ndarray:
    """Beta(1.0, 1.5) biased toward noise end, scaled to [0, 0.999]."""
    return rng.beta(1.0, 1.5, size=b) * 0.999


def flow_matching_loss(
    model: MLPVelocityModel,
    prefix: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    rng: np.random.Generator,
) -> float:
    B, K, D = action.shape
    noise = rng.standard_normal(action.shape)
    t = sample_time(B, rng)
    t_exp = t[:, None, None]
    x_t = t_exp * action + (1 - t_exp) * noise
    u_t = action - noise
    v_theta = model(x_t, state, prefix, t)
    return float(np.mean((v_theta - u_t) ** 2))


def sample_action(
    model: MLPVelocityModel,
    prefix: np.ndarray,
    state: np.ndarray,
    chunk_size: int,
    action_dim: int,
    num_steps: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    B = state.shape[0]
    x_t = rng.standard_normal((B, chunk_size, action_dim))
    dt = 1.0 / num_steps
    time_val = 0.0
    while time_val < 1.0 - dt / 2:
        t = np.full(B, time_val)
        v = model(x_t, state, prefix, t)
        x_t = x_t + dt * v
        time_val += dt
    return x_t


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    B, K, D = 4, 8, 7
    L, P = 16, 32
    S = 14

    model = MLPVelocityModel(action_dim=D, state_dim=S, prefix_dim=P, seed=0)
    prefix = rng.standard_normal((B, L, P))
    state = rng.standard_normal((B, S))
    action = rng.standard_normal((B, K, D))

    loss = flow_matching_loss(model, prefix, state, action, rng)
    print(f"Training flow-matching loss: {loss:.4f}")

    sampled = sample_action(model, prefix, state, chunk_size=K, action_dim=D, num_steps=10, rng=rng)
    print(f"Sampled action chunk shape: {sampled.shape}")
    print(f"Sample mean/std: {sampled.mean():.3f} / {sampled.std():.3f}")

    # Sanity check: sampling from a perfect velocity field should recover the
    # action distribution used during training (standard normal here).
    print(f"Target distribution (N(0,1)) mean/std: 0.000 / 1.000")
