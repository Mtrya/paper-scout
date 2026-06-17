#!/usr/bin/env python3
"""
Self-contained probe for MotionVLA's DSFT tokenizer idea.

Demonstrates the core claim on synthetic 1D/2D motion:
  - motion is not spectrally homogeneous: a "Base" position stream is
    low-frequency, while a "Phys" velocity stream is high-frequency;
  - a single shared DCT+BPE codebook under-represents the Phys stream;
  - a Dual-Stream Frequency-domain Tokenizer (DSFT) gives better
    reconstruction of physical dynamics with fewer tokens.

No external motion dataset is required.  The script generates synthetic
walk-like sequences, fits tiny BPE tokenizers, and produces a figure +
printed table.

Run:
    python3 probe_dsft.py
Output:
    runs/2026-06-17-aceego-actworld-motionvla/assets/motionvla_probe.png
"""

import json
import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets"
)
OUT_DIR = os.path.normpath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
FIG_PATH = os.path.join(OUT_DIR, "motionvla_probe.png")
JSON_PATH = os.path.join(OUT_DIR, "motionvla_probe.json")


# ------------------------------------------------------------------
# synthetic motion generator
# ------------------------------------------------------------------
def make_synthetic_motion(T: int = 120, seed: int = None):
    """Generate one [T, 2] motion: dim0 = base position, dim1 = phys velocity."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / T

    # Base: low-frequency pose trajectory (slow sinusoid + drift)
    # Base: smooth, low-frequency pose trajectory (small noise)
    base = (
        0.6 * np.sin(2 * np.pi * 1.5 * t)
        + 0.2 * np.sin(2 * np.pi * 0.4 * t)
        + 0.01 * rng.standard_normal(T)
    )

    # Phys: velocity = derivative of base + high-frequency contact/impact noise
    velocity = np.gradient(base)
    # add sharp high-frequency impulses (foot-strike-like)
    for _ in range(4):
        idx = rng.integers(10, T - 10)
        velocity[idx:idx + 3] += rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
    velocity += 0.05 * rng.standard_normal(T)

    motion = np.stack([base, velocity], axis=1).astype(np.float32)
    return motion


def make_dataset(n_seq: int = 500, T: int = 120):
    return [make_synthetic_motion(T, seed=i) for i in range(n_seq)]


# ------------------------------------------------------------------
# DCT helpers
# ------------------------------------------------------------------
def dct_truncate(motion: np.ndarray, K: int):
    """DCT along time axis and keep first K coefficients."""
    K_eff = min(K, motion.shape[0])
    freq = dct(motion, axis=0, norm="ortho")
    return freq[:K_eff, :]


def idct_from_truncate(freq_k: np.ndarray, T: int):
    """Zero-pad truncated DCT coefficients to length T and inverse-DCT."""
    K_eff, D = freq_k.shape
    freq_full = np.zeros((T, D), dtype=np.float32)
    freq_full[:K_eff, :] = freq_k
    return idct(freq_full, axis=0, norm="ortho")


def cumulative_energy(motions: List[np.ndarray]):
    """Return mean cumulative DCT energy ratio per dimension over dataset."""
    energy = []
    for m in motions:
        c = dct(m, axis=0, norm="ortho")
        e = c ** 2
        energy.append(e / (e.sum(axis=0, keepdims=True) + 1e-10))
    energy = np.stack(energy, axis=0).mean(axis=0)
    return np.cumsum(energy.mean(axis=1))


# ------------------------------------------------------------------
# tiny FAST-style tokenizer (DCT + BPE)
# ------------------------------------------------------------------
class FastTokenizer:
    """DCT truncation + integer quantization + BPE on a single stream."""

    def __init__(self, bpe: PreTrainedTokenizerFast, scale: float,
                 min_token: int, max_token: int, K: int, action_dim: int):
        self.bpe = bpe
        self.scale = scale
        self.min_token = min_token
        self.max_token = max_token
        self.token_range = max_token - min_token
        self.K = K
        self.action_dim = action_dim

    def encode(self, motion: np.ndarray) -> List[int]:
        T = motion.shape[0]
        K_eff = min(self.K, T)
        freq = dct(motion, axis=0, norm="ortho")[:K_eff, :]
        vals = np.around(freq.flatten() * self.scale).astype(int)
        # Clamp to the training range so every shifted value maps to a valid char.
        vals_shifted = np.clip(vals - self.min_token, 0, self.token_range)
        token_str = "".join(map(chr, vals_shifted))
        return self.bpe(token_str)["input_ids"]

    def decode(self, token_ids: List[int], T: int) -> np.ndarray:
        K_eff = min(self.K, T)
        expected_len = K_eff * self.action_dim
        decoded_str = self.bpe.decode(token_ids)
        vals_shifted = np.array(list(map(ord, decoded_str)))
        if len(vals_shifted) >= expected_len:
            vals_shifted = vals_shifted[:expected_len]
        else:
            vals_shifted = np.pad(vals_shifted,
                                  (0, expected_len - len(vals_shifted)))
        vals = (vals_shifted + self.min_token).reshape(K_eff, self.action_dim)
        return idct_from_truncate(vals / self.scale, T)

    @classmethod
    def fit(cls, data: List[np.ndarray], K: int, scale: float,
            vocab_size: int, action_dim: int):
        all_vals = []
        for motion in data:
            T = motion.shape[0]
            if T < 2:
                continue
            K_eff = min(K, T)
            freq = dct(motion, axis=0, norm="ortho")[:K_eff, :]
            all_vals.extend(np.around(freq.flatten() * scale).astype(int).tolist())

        arr = np.array(all_vals)
        min_token = int(arr.min())
        max_token = int(arr.max())
        token_range = max_token - min_token
        assert vocab_size > token_range, (
            f"vocab_size={vocab_size} <= token_range={token_range}; "
            "increase vocab_size or decrease scale."
        )

        def _iter():
            for motion in data:
                T = motion.shape[0]
                if T < 2:
                    continue
                K_eff = min(K, T)
                freq = dct(motion, axis=0, norm="ortho")[:K_eff, :]
                vals = np.around(freq.flatten() * scale).astype(int) - min_token
                yield "".join(map(chr, vals))

        alphabet = [chr(i) for i in range(token_range + 1)]
        bpe_tok = ByteLevelBPETokenizer()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            show_progress=False,
            special_tokens=[],
            initial_alphabet=alphabet,
            max_token_length=64,
        )
        bpe_tok._tokenizer.train_from_iterator(_iter(), trainer=trainer)
        fast = PreTrainedTokenizerFast(
            tokenizer_object=bpe_tok,
            clean_up_tokenization_spaces=False,
        )
        return cls(fast, scale, min_token, max_token, K, action_dim)


# ------------------------------------------------------------------
# main experiment
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MotionVLA / DSFT self-contained probe")
    print("=" * 60)

    T = 120
    n_train = 500
    n_test = 100
    scale = 50.0

    # Hyperparameters mirroring the paper's insight: Base is compact,
    # Phys needs a much larger frequency budget.
    K_base = 5
    K_phys = 20
    # Single-stream baseline with roughly the same total number of DCT coeffs.
    K_single = (K_base + K_phys) // 2

    print(f"\nGenerating {n_train} train + {n_test} test synthetic sequences (T={T})")
    train = make_dataset(n_train, T)
    test = make_dataset(n_test, T)

    # split into Base / Phys streams
    base_train = [m[:, 0:1] for m in train]
    phys_train = [m[:, 1:2] for m in train]

    # ------------------------------------------------------------------
    # 1. Spectral analysis: low-frequency energy coverage
    # ------------------------------------------------------------------
    base_energy = cumulative_energy(base_train)
    phys_energy = cumulative_energy(phys_train)
    base_k5 = base_energy[4]
    phys_k5 = phys_energy[4]

    print(f"\nSpectral analysis (first 5 DCT coefficients):")
    print(f"  Base stream energy coverage: {base_k5*100:.1f}%")
    print(f"  Phys stream energy coverage: {phys_k5*100:.1f}%")

    # ------------------------------------------------------------------
    # 2. Train tokenizers
    # ------------------------------------------------------------------
    print("\nTraining tokenizers...")
    single_tok = FastTokenizer.fit(
        train, K=K_single, scale=scale,
        vocab_size=2048, action_dim=2
    )
    base_tok = FastTokenizer.fit(
        base_train, K=K_base, scale=scale,
        vocab_size=512, action_dim=1
    )
    phys_tok = FastTokenizer.fit(
        phys_train, K=K_phys, scale=scale,
        vocab_size=1024, action_dim=1
    )

    # ------------------------------------------------------------------
    # 3. Encode/decode test set and measure reconstruction
    # ------------------------------------------------------------------
    def mse(a, b):
        return float(((a - b) ** 2).mean())

    single_mse, dual_mse = [], []
    single_base_mse, dual_base_mse = [], []
    single_phys_mse, dual_phys_mse = [], []
    single_tokpf, dual_base_tokpf, dual_phys_tokpf = [], [], []

    for m in test:
        # single-stream baseline
        ids = single_tok.encode(m)
        rec = single_tok.decode(ids, T=m.shape[0])
        single_mse.append(mse(m, rec))
        single_base_mse.append(mse(m[:, 0], rec[:, 0]))
        single_phys_mse.append(mse(m[:, 1], rec[:, 1]))
        single_tokpf.append(len(ids) / m.shape[0])

        # dual-stream
        b_ids = base_tok.encode(m[:, 0:1])
        p_ids = phys_tok.encode(m[:, 1:2])
        b_rec = base_tok.decode(b_ids, T=m.shape[0])
        p_rec = phys_tok.decode(p_ids, T=m.shape[0])
        rec_dual = np.concatenate([b_rec, p_rec], axis=1)
        dual_mse.append(mse(m, rec_dual))
        dual_base_mse.append(mse(m[:, 0], b_rec[:, 0]))
        dual_phys_mse.append(mse(m[:, 1], p_rec[:, 0]))
        dual_base_tokpf.append(len(b_ids) / m.shape[0])
        dual_phys_tokpf.append(len(p_ids) / m.shape[0])

    stats = {
        "single_stream": {
            "mse_full": float(np.mean(single_mse)),
            "mse_base": float(np.mean(single_base_mse)),
            "mse_phys": float(np.mean(single_phys_mse)),
            "tokens_per_frame": float(np.mean(single_tokpf)),
        },
        "dual_stream": {
            "mse_full": float(np.mean(dual_mse)),
            "mse_base": float(np.mean(dual_base_mse)),
            "mse_phys": float(np.mean(dual_phys_mse)),
            "base_tokens_per_frame": float(np.mean(dual_base_tokpf)),
            "phys_tokens_per_frame": float(np.mean(dual_phys_tokpf)),
            "total_tokens_per_frame": float(np.mean(dual_base_tokpf) +
                                            np.mean(dual_phys_tokpf)),
        },
        "energy_coverage": {
            "base_k5": float(base_k5),
            "phys_k5": float(phys_k5),
        },
    }

    print("\n" + "-" * 60)
    print(f"{'Metric':<35} {'Single-Stream':>14} {'DSFT':>14}")
    print("-" * 60)
    print(f"{'Full-motion MSE':<35} {stats['single_stream']['mse_full']:>14.5f} "
          f"{stats['dual_stream']['mse_full']:>14.5f}")
    print(f"{'Base-stream MSE':<35} {stats['single_stream']['mse_base']:>14.5f} "
          f"{stats['dual_stream']['mse_base']:>14.5f}")
    print(f"{'Phys-stream MSE':<35} {stats['single_stream']['mse_phys']:>14.5f} "
          f"{stats['dual_stream']['mse_phys']:>14.5f}")
    print(f"{'Tokens / frame':<35} {stats['single_stream']['tokens_per_frame']:>14.2f} "
          f"{stats['dual_stream']['total_tokens_per_frame']:>14.2f}")
    print("-" * 60)
    print(f"DSFT uses K_base={K_base}, K_phys={K_phys}; "
          f"single-stream uses K={K_single} (same total coefficients).")

    # ------------------------------------------------------------------
    # 4. Figure
    # ------------------------------------------------------------------
    example = test[0]
    ids_single = single_tok.encode(example)
    rec_single = single_tok.decode(ids_single, T=example.shape[0])
    b_ids = base_tok.encode(example[:, 0:1])
    p_ids = phys_tok.encode(example[:, 1:2])
    rec_dual = np.concatenate([
        base_tok.decode(b_ids, T=example.shape[0]),
        phys_tok.decode(p_ids, T=example.shape[0]),
    ], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        "MotionVLA / DSFT probe on synthetic motion\n"
        "Base = low-frequency position, Phys = high-frequency velocity",
        fontsize=11, fontweight="bold"
    )

    # (a) example Base stream
    ax = axes[0, 0]
    ax.plot(example[:, 0], label="Ground-truth Base", color="#2196F3", lw=2)
    ax.plot(rec_single[:, 0], label="Single-stream recon", color="#9E9E9E", lw=1.5, ls="--")
    ax.plot(rec_dual[:, 0], label="DSFT Base recon", color="#1565C0", lw=1.5, ls=":")
    ax.set_title("Base (low-frequency position)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (b) example Phys stream
    ax = axes[0, 1]
    ax.plot(example[:, 1], label="Ground-truth Phys", color="#FF5722", lw=2)
    ax.plot(rec_single[:, 1], label="Single-stream recon", color="#9E9E9E", lw=1.5, ls="--")
    ax.plot(rec_dual[:, 1], label="DSFT Phys recon", color="#BF360C", lw=1.5, ls=":")
    ax.set_title("Phys (high-frequency velocity)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (c) cumulative DCT energy
    ax = axes[1, 0]
    ks = np.arange(1, len(base_energy) + 1)
    ax.plot(ks, base_energy * 100, color="#2196F3", lw=2.2,
            label=f"Base: {base_k5*100:.0f}% @ K=5")
    ax.plot(ks, phys_energy * 100, color="#FF5722", lw=2.2,
            label=f"Phys: {phys_k5*100:.0f}% @ K=5")
    ax.axvline(5, color="gray", ls=":", lw=1)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 105)
    ax.set_xlabel("DCT coefficients retained")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Spectral heterogeneity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # (d) reconstruction quality bar chart
    ax = axes[1, 1]
    labels = ["Full motion\nMSE", "Phys stream\nMSE", "Tokens /\nframe"]
    single_vals = [
        stats["single_stream"]["mse_full"],
        stats["single_stream"]["mse_phys"],
        stats["single_stream"]["tokens_per_frame"],
    ]
    dual_vals = [
        stats["dual_stream"]["mse_full"],
        stats["dual_stream"]["mse_phys"],
        stats["dual_stream"]["total_tokens_per_frame"],
    ]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, single_vals, width, label="Single-stream", color="#9E9E9E")
    ax.bar(x + width / 2, dual_vals, width, label="DSFT", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Value (linear scale)")
    ax.set_title("Single-stream vs. DSFT")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {FIG_PATH}")

    with open(JSON_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved:  {JSON_PATH}")


if __name__ == "__main__":
    main()
