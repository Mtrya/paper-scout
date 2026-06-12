"""Sketch of FAST-style action tokenization.

The real FAST tokenizer (Pertsch et al., 2025) shipped by Physical Intelligence
uses DCT + learned BPE. LabVLA calls it via `FastTokenizerWrapper` in
`src/policies/LabVLA/ki/fast_tokenizer.py` and uses the resulting discrete tokens
as CE targets during VLM pretraining and KI posttraining.

This script demonstrates the *idea* with a minimal numpy implementation:
  1. Normalize a continuous action chunk to [-1, 1].
  2. Apply 1-D DCT per action dimension to compress temporal structure.
  3. Vector-quantize DCT coefficients with a small k-means codebook.
  4. Emit a token sequence that can be fed to a language-model CE loss.

Run: python fast_tokenizer_sketch.py
"""
from __future__ import annotations

import numpy as np


def dct1d(x: np.ndarray) -> np.ndarray:
    """Type-II DCT along the time axis (T, D) -> (T, D), numpy only."""
    T = x.shape[0]
    n = np.arange(T)
    k = np.arange(T)[:, None]
    basis = np.cos(np.pi * k * (n + 0.5) / T)
    return basis @ x * np.sqrt(2.0 / T)


def idct1d(x: np.ndarray) -> np.ndarray:
    """Type-II IDCT along the time axis (T, D) -> (T, D), numpy only."""
    T = x.shape[0]
    n = np.arange(T)
    k = np.arange(T)[:, None]
    basis = np.cos(np.pi * k * (n + 0.5) / T)
    scale = np.sqrt(2.0 / T)
    return (basis.T @ x) * scale


def kmeans_simple(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 30) -> np.ndarray:
    """Minimal k-means returning cluster centers (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None], axis=-1)
        labels = dists.argmin(axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def make_codebook(actions: list[np.ndarray], vocab_size: int = 64, seed: int = 0) -> np.ndarray:
    """Fit k-means on flattened DCT coefficient vectors."""
    feats = []
    for a in actions:
        c = dct1d(a)
        feats.append(c.flatten())
    X = np.stack(feats)
    return kmeans_simple(X, vocab_size, seed=seed)


def encode(action: np.ndarray, codebook: np.ndarray) -> list[int]:
    """Return a single token id for the whole action chunk (simplification).

    Real FAST tokenizes per-dimension coefficient chunks via BPE, producing a
    variable-length token sequence. Here we collapse to one token to keep the
    sketch small while preserving the DCT+quantization mechanism.
    """
    c = dct1d(action).flatten()
    dists = np.linalg.norm(codebook - c, axis=-1)
    return [int(dists.argmin())]


def decode(token_ids: list[int], codebook: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-codebook reconstruction."""
    cid = token_ids[0]
    flat = codebook[cid]
    coeffs = flat.reshape(target_shape)
    return idct1d(coeffs)


def normalize_action(a: np.ndarray, q01: float, q99: float) -> np.ndarray:
    """Per-dataset q01/q99 normalization to [-1, 1] (used by LabVLA stats)."""
    return np.clip((a - q01) / (q99 - q01) * 2.0 - 1.0, -1.0, 1.0)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Generate toy action chunks: (T=16, D=7 arm joints + 1 gripper)
    n_train = 200
    T, D = 16, 8
    train_actions = []
    for _ in range(n_train):
        t = np.linspace(0, 1, T)
        a = np.zeros((T, D))
        for d in range(D - 1):
            a[:, d] = np.sin(2 * np.pi * (d + 1) * t + rng.random()) + rng.normal(0, 0.1, T)
        # gripper as binary open/close
        a[:, -1] = rng.choice([0.0, 1.0])
        train_actions.append(a)

    q01, q99 = -1.2, 1.2
    norm_actions = [normalize_action(a, q01, q99) for a in train_actions]

    codebook = make_codebook(norm_actions, vocab_size=64, seed=0)

    # Test reconstruction
    test_a = norm_actions[0]
    tokens = encode(test_a, codebook)
    recon = decode(tokens, codebook, test_a.shape)
    mse = float(np.mean((test_a - recon) ** 2))

    print(f"Action chunk shape: {test_a.shape}")
    print(f"Discrete token ids: {tokens}")
    print(f"Reconstruction MSE vs normalized action: {mse:.4f}")
    print(f"Codebook vocab size: {codebook.shape[0]}")

    # Show that temporal DCT compresses structure
    coeffs = dct1d(test_a)
    print(f"DCT coefficient energy (first 5 timesteps): {np.sum(coeffs[:5]**2):.3f}")
    print(f"DCT coefficient energy (remaining):          {np.sum(coeffs[5:]**2):.3f}")
