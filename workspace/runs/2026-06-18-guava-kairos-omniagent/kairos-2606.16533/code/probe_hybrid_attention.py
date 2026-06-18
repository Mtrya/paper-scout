#!/usr/bin/env python3
"""
Kairos Hybrid Linear Temporal Attention — minimal NumPy probe.

Demonstrates the core idea of Kairos's hybrid temporal memory on a synthetic
1-D sequence task: a single informative "cue" token at the start of a long
sequence must be recovered from the final "query" token.  The probe compares:

* Pure sliding-window attention (SWA) baseline
* A global gated-linear-attention (GLA) path
* A Hybrid model that concatenates SWA local features with the GLA state

It also visualises the contractive error bound from the paper's theory.

Run inside the bundled venv (or any env with numpy+matplotlib):

    ../.venv/bin/python probe_hybrid_attention.py

Output:
    probe_result.png  — accuracy vs. horizon + contractive error bound demo
    console table of final accuracies
"""

import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:
    HAS_MPL = False
    print(f"[warn] matplotlib not available ({e}); only console results will be printed")


def make_data(n, T, num_classes=4, d_model=32, W_emb=None, seed=None):
    """
    Generate sequences where the first token encodes a class and the final
    token is a fixed query.  Everything in between is random distractor noise.
    If W_emb is not provided it is created from the rng; pass the same W_emb
    to train and test splits to make the task learnable.
    """
    rng = np.random.default_rng(seed)
    vocab_size = num_classes + 2
    if W_emb is None:
        W_emb = rng.normal(0, 0.5, (vocab_size, d_model)).astype(np.float32)

    labels = rng.integers(0, num_classes, size=n)
    tokens = np.full((n, T), num_classes, dtype=np.int64)  # distractor token id
    tokens[:, 0] = labels
    tokens[:, -1] = num_classes + 1  # query token id

    X = W_emb[tokens]  # (n, T, d)
    return X, labels, W_emb


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=-1) == y)


def swa_features(X, window):
    """Mean-pool the last `window` tokens (pure sliding-window memory)."""
    n, T, d = X.shape
    w = min(window, T)
    return X[:, -w:, :].mean(axis=1)  # (n, d)


def gla_features(X, alpha=0.99):
    """
    Simplified Gated Linear Attention: recurrent state
        h_t = alpha * h_{t-1} + beta_t * x_t
    with an *idealised* write gate: beta_0 = 0.99, beta_{t>0} = 0.01.
    This isolates the global-memory mechanism from the harder gate-learning
    problem, showing that once the gate knows what to remember, the recurrent
    state propagates the cue across arbitrary horizons.
    """
    n, T, d = X.shape
    beta = np.full(T, 0.01, dtype=np.float32)
    beta[0] = 0.99
    h = np.zeros((n, d), dtype=np.float32)
    for t in range(T):
        h = alpha * h + beta[t] * X[:, t]
    return h


def train_and_eval(feature_mode, X_train, y_train, X_test, y_test,
                   num_classes, window=8, alpha=0.99,
                   lr=0.1, steps=2000, batch=128, seed=42):
    """
    Train a linear readout on the chosen features.
    feature_mode: 'swa' | 'gla' | 'hybrid'.
    """
    rng = np.random.default_rng(seed)
    n, T, d = X_train.shape
    feat_dim = d if feature_mode != "hybrid" else 2 * d
    params = {
        "W": rng.normal(0, 0.05, (feat_dim, num_classes)).astype(np.float32),
        "b": np.zeros(num_classes, dtype=np.float32),
    }

    n_train = X_train.shape[0]
    for step in range(steps):
        idx = rng.integers(0, n_train, size=batch)
        Xb, yb = X_train[idx], y_train[idx]

        if feature_mode == "swa":
            F = swa_features(Xb, window)
        elif feature_mode == "gla":
            F = gla_features(Xb, alpha)
        else:  # hybrid
            F = np.concatenate([
                swa_features(Xb, window),
                gla_features(Xb, alpha)
            ], axis=1)

        logits = F @ params["W"] + params["b"]
        probs = softmax(logits)
        grad_logits = probs.copy()
        grad_logits[np.arange(batch), yb] -= 1.0
        grad_logits /= batch

        params["W"] -= lr * (F.T @ grad_logits)
        params["b"] -= lr * grad_logits.sum(axis=0)

    # Evaluate
    if feature_mode == "swa":
        F_test = swa_features(X_test, window)
    elif feature_mode == "gla":
        F_test = gla_features(X_test, alpha)
    else:
        F_test = np.concatenate([
            swa_features(X_test, window),
            gla_features(X_test, alpha)
        ], axis=1)
    logits_test = F_test @ params["W"] + params["b"]
    return accuracy(logits_test, y_test), params


def contractive_error_bound_demo(rho=0.95, T=200, seed=7):
    rng = np.random.default_rng(seed)
    e0 = 0.5
    xi = rng.normal(0, 0.05, size=T)
    e = np.zeros(T)
    e[0] = rho * e0 + xi[0]
    for t in range(1, T):
        e[t] = rho * e[t - 1] + xi[t]
    tidx = np.arange(1, T + 1)
    bound = rho ** tidx * e0 + (1 - rho ** tidx) / (1 - rho) * np.max(np.abs(xi))
    return e, bound


def main():
    num_classes = 4
    d_model = 32
    window = 8
    horizons = [10, 16, 24, 40, 80, 120]
    n_train, n_test = 3000, 800

    results = {"SWA": [], "GLA": [], "Hybrid": []}
    print("\nSynthetic cue-recovery task: first token = class, last token = query")
    print("=" * 70)
    for T in horizons:
        X_train, y_train, W_emb = make_data(n_train, T, num_classes, d_model, seed=1000 + T)
        X_test, y_test, _ = make_data(n_test, T, num_classes, d_model, W_emb=W_emb, seed=2000 + T)

        acc_swa, _ = train_and_eval("swa", X_train, y_train, X_test, y_test, num_classes, window=window)
        acc_gla, _ = train_and_eval("gla", X_train, y_train, X_test, y_test, num_classes)
        acc_hyb, _ = train_and_eval("hybrid", X_train, y_train, X_test, y_test, num_classes, window=window)

        results["SWA"].append(acc_swa)
        results["GLA"].append(acc_gla)
        results["Hybrid"].append(acc_hyb)
        print(f"T={T:3d}  SWA={acc_swa:.3f}  GLA={acc_gla:.3f}  Hybrid={acc_hyb:.3f}")
    print("=" * 70)

    e, bound = contractive_error_bound_demo(rho=0.95, T=200)

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for name, vals in results.items():
            marker = {"SWA": "o", "GLA": "s", "Hybrid": "^"}[name]
            ax1.plot(horizons, vals, marker=marker, label=name, linewidth=2)
        ax1.axvline(window, color="gray", linestyle="--", alpha=0.6, label=f"window={window}")
        ax1.set_xlabel("Sequence length T")
        ax1.set_ylabel("Test accuracy")
        ax1.set_title("Cue recovery across long horizons")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        tidx = np.arange(1, len(e) + 1)
        ax2.plot(tidx, np.abs(e), label="|e_t|", alpha=0.8)
        ax2.plot(tidx, bound, label=r"$\rho^t e_0 + \frac{1-\rho^t}{1-\rho}\bar{\xi}$", linestyle="--")
        ax2.set_xlabel("Time step t")
        ax2.set_ylabel("Memory perturbation |e_t|")
        ax2.set_title("Contractive global-memory error bound")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        out_path = os.path.join(os.path.dirname(__file__), "probe_result.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved figure to {out_path}")
    else:
        print("\nAccuracies:")
        print("T     ", "  ".join(f"{t:3d}" for t in horizons))
        for name, vals in results.items():
            print(name.ljust(6), "  ".join(f"{v:.3f}" for v in vals))


if __name__ == "__main__":
    main()
