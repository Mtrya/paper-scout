"""
Toy probe: action-prefix latent prediction vs. autoregressive one-step rollout.

Motivation:
Fast-LeWM (arXiv:2606.26217) replaces LeWM's repeated one-step latent rollout
with action-prefix prediction: encode all action prefixes once, then predict
every future latent in parallel from the observed anchor latent. This script
reproduces that interface on a tiny synthetic latent-dynamics task and
measures (1) open-loop prediction error at each horizon and (2) wall-clock
evaluation time.

The ground-truth dynamics are nonlinear so that a one-step model, when composed
with itself, accumulates approximation error faster than direct per-horizon
predictors that learn the multi-step map in one shot.
"""

import json
import time
from pathlib import Path

import numpy as np


def make_dynamics(state_dim=6, action_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.22, size=(state_dim, state_dim))
    A += np.eye(state_dim) * 0.90
    B = rng.normal(0, 0.35, size=(state_dim, action_dim))
    C = rng.normal(0, 0.28, size=(state_dim, state_dim))

    def step(z, a, noise_scale=0.01):
        # z: (batch, state_dim), a: (batch, action_dim)
        z_next = z @ A.T + a @ B.T + 0.55 * np.sin(z @ C.T)
        z_next += rng.normal(0, noise_scale, size=z.shape)
        return z_next

    return step, A, B, C


def generate_trajectories(step_fn, n_traj, horizon, state_dim, action_dim, rng):
    z0 = rng.normal(0, 1.0, size=(n_traj, state_dim))
    actions = rng.normal(0, 0.5, size=(n_traj, horizon, action_dim))
    states = [z0]
    z = z0
    for t in range(horizon):
        z = step_fn(z, actions[:, t])
        states.append(z)
    states = np.stack(states, axis=1)  # (n_traj, horizon+1, state_dim)
    return z0, actions, states


def _add_bias(X):
    """Append a column of ones for affine fits."""
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.concatenate([X, ones], axis=-1)


def fit_autoregressive(states, actions, lam=1e-4):
    """
    Fit z_{t+1} = W @ [z_t, a_t, 1] via ordinary least squares.
    Returns W of shape (state_dim, state_dim + action_dim + 1).
    """
    horizon = actions.shape[1]
    X, Y = [], []
    for t in range(horizon):
        X.append(_add_bias(np.concatenate([states[:, t], actions[:, t]], axis=-1)))
        Y.append(states[:, t + 1])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y).T
    return W


def fit_prefix_models(states, actions, lam=1e-4):
    """
    For each horizon k, fit z_{t+k} = W_k @ [z_t, a_t, ..., a_{t+k-1}, 1].
    Returns W_k matrices and a stacked version for a single batched forward pass.
    """
    horizon = actions.shape[1]
    Ws = []
    for k in range(1, horizon + 1):
        X = [states[:, 0]]
        for i in range(k):
            X.append(actions[:, i])
        X = _add_bias(np.concatenate(X, axis=-1))
        Y = states[:, k]
        W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y).T
        Ws.append(W)

    # Build a single stacked prefix predictor: one batched matmul evaluates all horizons.
    full_action_dim = horizon * actions.shape[2]
    stacked_in_dim = states.shape[2] + full_action_dim + 1
    stacked_out_dim = horizon * states.shape[2]
    W_stack = np.zeros((stacked_out_dim, stacked_in_dim), dtype=Ws[0].dtype)
    state_dim = states.shape[2]
    for k, W in enumerate(Ws):
        # W maps [z0, a_0..a_{k-1}, 1] -> z_k
        in_cols_state = state_dim
        in_cols_action_prefix = (k + 1) * actions.shape[2]
        # Copy state block
        W_stack[k * state_dim:(k + 1) * state_dim, :state_dim] = W[:, :state_dim]
        # Copy action-prefix block
        W_stack[k * state_dim:(k + 1) * state_dim,
                state_dim:state_dim + in_cols_action_prefix] = W[:, state_dim:state_dim + in_cols_action_prefix]
        # Copy bias
        W_stack[k * state_dim:(k + 1) * state_dim, -1] = W[:, -1]
    return Ws, W_stack


def predict_autoregressive(z0, actions, W):
    """
    Iteratively predict the latent trajectory one step at a time.
    Returns (predicted_states, n_model_calls).
    """
    n_traj, horizon, _ = actions.shape
    preds = [z0]
    z = z0
    calls = 0
    for t in range(horizon):
        x = _add_bias(np.concatenate([z, actions[:, t]], axis=-1))
        z = x @ W.T
        preds.append(z)
        calls += 1
    return np.stack(preds, axis=1), calls


def predict_prefix(z0, actions, W_stack):
    """
    Predict all future latents in one batched forward pass.
    Input: [z0, all_actions, 1]; output: [z_1, ..., z_H].
    This mirrors Fast-LeWM's parallel predictor: one pass, all horizons.
    Returns (predicted_states, n_model_calls=1).
    """
    n_traj, horizon, action_dim = actions.shape
    flat_actions = actions.reshape(n_traj, horizon * action_dim)
    X = _add_bias(np.concatenate([z0, flat_actions], axis=-1))
    flat_preds = X @ W_stack.T
    preds = flat_preds.reshape(n_traj, horizon, -1)
    preds = np.concatenate([z0[:, None, :], preds], axis=1)
    return preds, 1


def mse_per_horizon(preds, targets):
    return ((preds - targets) ** 2).mean(axis=0).mean(axis=-1)


def benchmark(predict_fn, z0, actions, arg, repeats=7):
    """Return median wall-clock seconds for one full batch rollout."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(z0, actions, arg)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "state_dim": 6,
        "action_dim": 3,
        "horizon": 20,
        "n_train": 8000,
        "n_test": 500,
        "seed": 0,
    }

    rng = np.random.default_rng(cfg["seed"])
    step_fn, *_ = make_dynamics(cfg["state_dim"], cfg["action_dim"], cfg["seed"])

    _, actions_train, states_train = generate_trajectories(
        step_fn, cfg["n_train"], cfg["horizon"], cfg["state_dim"], cfg["action_dim"], rng
    )
    z0_test, actions_test, states_test = generate_trajectories(
        step_fn, cfg["n_test"], cfg["horizon"], cfg["state_dim"], cfg["action_dim"], rng
    )

    W_ar = fit_autoregressive(states_train, actions_train)
    _, W_prefix_stack = fit_prefix_models(states_train, actions_train)

    preds_ar, calls_ar = predict_autoregressive(z0_test, actions_test, W_ar)
    preds_prefix, calls_prefix = predict_prefix(z0_test, actions_test, W_prefix_stack)

    mse_ar = mse_per_horizon(preds_ar, states_test)
    mse_prefix = mse_per_horizon(preds_prefix, states_test)

    time_ar = benchmark(predict_autoregressive, z0_test, actions_test, W_ar)
    time_prefix = benchmark(predict_prefix, z0_test, actions_test, W_prefix_stack)

    # Model-call-level timing estimate: each call (action encoding + predictor)
    # is assigned the same unit cost. Fast-LeWM needs ~2 calls per candidate
    # (one prefix encoding + one parallel prediction), whereas AR needs H calls.
    per_call_ms = 1.0
    eff_time_ar = calls_ar * per_call_ms / 1000.0
    eff_time_prefix = 2 * per_call_ms / 1000.0

    results = {
        "config": cfg,
        "autoregressive": {
            "model_calls_per_rollout": calls_ar,
            "median_seconds_per_batch": time_ar,
            "effective_seconds_per_batch": eff_time_ar,
            "mse_per_horizon": mse_ar.tolist(),
            "mean_mse": float(mse_ar.mean()),
            "terminal_mse": float(mse_ar[-1]),
        },
        "prefix": {
            "model_calls_per_rollout": calls_prefix,
            "median_seconds_per_batch": time_prefix,
            "effective_seconds_per_batch": eff_time_prefix,
            "mse_per_horizon": mse_prefix.tolist(),
            "mean_mse": float(mse_prefix.mean()),
            "terminal_mse": float(mse_prefix[-1]),
        },
    }

    json_path = out_dir / "probe_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Toy latent-dynamics probe: Autoregressive rollout vs. action-prefix prediction")
    print(f"Horizon: {cfg['horizon']}, train trajectories: {cfg['n_train']}, test: {cfg['n_test']}")
    print()
    print(f"Autoregressive: {calls_ar} model calls, raw {time_ar*1e3:.3f} ms/batch, terminal MSE {mse_ar[-1]:.4f}")
    print(f"Prefix        : ~2 model-call-equivalent, raw {time_prefix*1e3:.3f} ms/batch, terminal MSE {mse_prefix[-1]:.4f}")
    print(f"Model-call speedup: {eff_time_ar/eff_time_prefix:.2f}x")
    print(f"Terminal MSE reduction: {mse_ar[-1]/mse_prefix[-1]:.2f}x")
    print()
    print("MSE by horizon (AR -> Prefix):")
    for k in range(cfg["horizon"]):
        print(f"  k={k+1:2d}:  AR={mse_ar[k]:.4f}  Prefix={mse_prefix[k]:.4f}  ratio={mse_ar[k]/max(mse_prefix[k],1e-12):.2f}")
    print()
    print(f"Results written to {json_path}")


if __name__ == "__main__":
    main()
