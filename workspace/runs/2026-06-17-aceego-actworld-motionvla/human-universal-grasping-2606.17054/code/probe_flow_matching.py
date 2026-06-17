"""
HUG mechanism probe: rectified-flow matching ODE for MANO grasp generation.

HUG's grasp state x = [translation (3), wrist rotation 6D (6), finger joints 6D
(90)] is 99-dimensional. Training uses the rectified-flow objective:

    x_t = (1 - t) * x_0 + t * eps,      v_target = eps - x_0

and inference integrates the learned velocity field from t=1 (noise) to t=0
(data) with Euler steps.

This probe demonstrates the ODE on a toy 2D grasp subspace (e.g. wrist y and
finger flexion) using a tiny MLP implemented in NumPy. No PyTorch, MANO, or
real data is required.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class TinyVelocityMLP:
    """Minimal velocity net v(x, t) for the 2D toy problem."""

    def __init__(self, dim: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.W1 = rng.normal(0, 0.1, size=(dim + 1, 64))
        self.b1 = np.zeros(64)
        self.W2 = rng.normal(0, 0.1, size=(64, 64))
        self.b2 = np.zeros(64)
        self.W3 = rng.normal(0, 0.1, size=(64, dim))
        self.b3 = np.zeros(dim)

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        # x: (B, dim); t: (B,)
        inp = np.concatenate([x, t[:, None]], axis=-1)
        h = silu(inp @ self.W1 + self.b1)
        h = silu(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


def sample_targets(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample synthetic target grasps clustered around the object surface."""
    mode = rng.integers(0, 2, size=n)
    x = np.zeros((n, 2), dtype=np.float32)
    x[:, 0] = rng.normal(0.5, 0.05, size=n)
    flex_mean = np.where(mode == 0, 0.75, 0.35).astype(np.float32)
    x[:, 1] = flex_mean + rng.normal(0.0, 0.06, size=n).astype(np.float32)
    return x


def train_velocity_model(
    model: TinyVelocityMLP,
    n_steps: int = 5000,
    batch_size: int = 256,
    lr: float = 5e-3,
) -> list[float]:
    rng = np.random.default_rng(7)
    losses = []
    for step in range(n_steps):
        x0 = sample_targets(batch_size, rng)
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        t = rng.random(batch_size).astype(np.float32)
        tt = t[:, None]
        xt = (1 - tt) * x0 + tt * eps
        v_target = eps - x0
        v_pred = model.forward(xt, t)

        # Backprop through the network manually.
        loss = float(np.mean((v_pred - v_target) ** 2))
        losses.append(loss)

        grad_out = (2.0 / (batch_size * model.dim)) * (v_pred - v_target)
        # Layer 3
        inp = np.concatenate([xt, t[:, None]], axis=-1)
        h1 = silu(inp @ model.W1 + model.b1)
        h2 = silu(h1 @ model.W2 + model.b2)
        dW3 = h2.T @ grad_out
        db3 = grad_out.sum(axis=0)
        grad_h2 = grad_out @ model.W3.T
        grad_h2 *= h2 * (1 - np.exp(-h2) / (1 + np.exp(-h2)) ** 2)  # silu derivative
        # Layer 2
        dW2 = h1.T @ grad_h2
        db2 = grad_h2.sum(axis=0)
        grad_h1 = grad_h2 @ model.W2.T
        sig = 1 / (1 + np.exp(-h1))
        grad_h1 *= sig * (1 + h1 * (1 - sig))  # silu derivative
        # Layer 1
        dW1 = inp.T @ grad_h1
        db1 = grad_h1.sum(axis=0)

        for dW, W in zip([dW1, db1, dW2, db2, dW3, db3], model.parameters()):
            W -= lr * dW

        if step % 1000 == 0:
            print(f"step {step:5d}  loss {loss:.4f}")
    return losses


def sample_ode(
    model: TinyVelocityMLP,
    n_samples: int,
    n_steps: int = 50,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Euler integration from t=1 to t=0."""
    rng = np.random.default_rng(99)
    x = rng.standard_normal((n_samples, model.dim)).astype(np.float32)
    dt = 1.0 / n_steps
    trajectory = [x.copy()]
    for i in reversed(range(n_steps)):
        t = np.full((n_samples,), (i + 1) * dt, dtype=np.float32)
        v = model.forward(x, t)
        x = x - v * dt
        trajectory.append(x.copy())
    return x, trajectory


def main():
    print("Training toy flow-matching model (NumPy)...")
    model = TinyVelocityMLP(dim=2, seed=0)
    losses = train_velocity_model(model)

    print("\nSampling trajectories...")
    _, traj = sample_ode(model, n_samples=12, n_steps=50)
    traj = np.stack(traj, axis=0)  # (T+1, B, 2)

    x_large, _ = sample_ode(model, n_samples=1000, n_steps=50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training loss.
    ax = axes[0]
    ax.plot(losses, lw=1)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Flow-matching MSE")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)

    # Trajectories in state space.
    ax = axes[1]
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.5, lw=1)
        ax.scatter(traj[0, i, 0], traj[0, i, 1], c="blue", s=20, zorder=3)
        ax.scatter(traj[-1, i, 0], traj[-1, i, 1], c="red", s=40, zorder=3)
    ax.axhspan(0.25, 0.85, xmin=0.3, xmax=0.7, color="green", alpha=0.1)
    ax.text(0.5, 0.55, "grasp region", ha="center", va="center", color="green")
    ax.set_xlabel("wrist y (m)")
    ax.set_ylabel("finger flexion")
    ax.set_title("Flow ODE trajectories (noise → grasp)")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)

    # Marginal distribution of final samples.
    ax = axes[2]
    ax.hist(x_large[:, 0], bins=30, alpha=0.5, label="wrist y")
    ax.hist(x_large[:, 1], bins=30, alpha=0.5, label="finger flexion")
    ax.axvline(0.5, color="green", linestyle="--", label="target")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title("Final sample marginals (n=1000)")
    ax.legend()

    fig.tight_layout()
    out_dir = Path("runs/2026-06-17-aceego-actworld-motionvla/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hug_flow_matching_probe.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization: {out_path.resolve()}")


if __name__ == "__main__":
    main()
