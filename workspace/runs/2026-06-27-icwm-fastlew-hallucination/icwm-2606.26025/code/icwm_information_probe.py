"""
icwm_information_probe.py

Synthetic diagnostic for Proposition 1 of
"In-Context World Modeling for Robotic Control" (arXiv:2606.26025):

    Under partial observability (A1) and information-preserving transitions (A2),
    the interaction context T = (o_0:t, a_1:t) carries strictly more information
    about the time-invariant system configuration psi than any single observation:

        I(psi ; o_0:t, a_1:t) > I(psi ; o_0)

The paper's motivating example is a camera viewpoint psi (azimuth angle). A
single image measurement is noisy and therefore ambiguous, but a short sequence
of known probing actions produces multiple image transitions that constrain psi.

This script builds a minimal 2-D world where:
    - the robot end-effector is a point whose state is approximately known
    - the camera observes a scalar projection p = x * cos(psi) + y * sin(psi)
      (the horizontal image coordinate under viewpoint psi), plus noise
    - a probing action moves the point by a known delta (dx, dy); each new
      observation adds an independent noisy constraint on psi

We quantify ambiguity via the posterior entropy over psi after seeing either a
single observation or an interaction sequence. Lower entropy = more information.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Candidate camera viewpoints, covering the LIBERO ID/OOD angles from the paper.
VIEWPOINTS_DEG = np.array([30, 60, 90, 120, 240, 270, 300, 330,
                           45, 135, 225, 255, 285, 315])
VIEWPOINTS_RAD = np.deg2rad(VIEWPOINTS_DEG)


def project(state: np.ndarray, psi: float) -> float:
    """Scalar camera projection under viewpoint psi."""
    x, y = state
    return x * np.cos(psi) + y * np.sin(psi)


def observe(state: np.ndarray, psi: float, noise: float = 0.03) -> float:
    """Noisy scalar projection."""
    return project(state, psi) + np.random.normal(scale=noise)


def posterior_from_likelihoods(likes: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize likelihoods and compute posterior entropy (bits)."""
    probs = likes / (likes.sum() + 1e-12)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    return probs, entropy


def main() -> None:
    np.random.seed(7)

    # Ground-truth configuration: an OOD viewpoint (45 deg, used in LIBERO eval).
    true_psi = np.deg2rad(45)

    # Approximate initial end-effector state (e.g. from proprioception).
    true_state = np.array([0.25, -0.05])
    state_prior_std = 0.02

    # --- Single observation ----------------------------------------------
    o0 = observe(true_state, true_psi)
    single_likes = np.array([
        np.exp(-0.5 * ((project(true_state, psi) - o0) / 0.03) ** 2)
        for psi in VIEWPOINTS_RAD
    ])
    _, H_single = posterior_from_likelihoods(single_likes)

    # --- Interaction context (T=5 probing moves) -------------------------
    T = 5
    actions = np.array([
        [0.10, 0.00],
        [-0.05, 0.10],
        [0.00, -0.10],
        [-0.08, -0.05],
        [0.05, 0.08],
    ])
    states = [true_state]
    for a in actions:
        states.append(states[-1] + a)
    obs_seq = np.array([observe(s, true_psi) for s in states])

    # Marginalize over a small Gaussian state uncertainty for the initial state.
    n_samples = 2000
    state_samples = np.random.normal(true_state, state_prior_std, size=(n_samples, 2))
    seq_likes = np.zeros(len(VIEWPOINTS_RAD))
    for psi_idx, psi in enumerate(VIEWPOINTS_RAD):
        log_likes = np.zeros(n_samples)
        samples = state_samples.copy()
        for t, a in enumerate(actions):
            pred = samples[:, 0] * np.cos(psi) + samples[:, 1] * np.sin(psi)
            log_likes += -0.5 * ((pred - obs_seq[t]) / 0.03) ** 2
            samples = samples + a
        pred = samples[:, 0] * np.cos(psi) + samples[:, 1] * np.sin(psi)
        log_likes += -0.5 * ((pred - obs_seq[-1]) / 0.03) ** 2
        seq_likes[psi_idx] = np.mean(np.exp(log_likes - np.max(log_likes)))

    _, H_seq = posterior_from_likelihoods(seq_likes)

    prior_entropy = np.log2(len(VIEWPOINTS_RAD))
    information_gain_single = prior_entropy - H_single
    information_gain_seq = prior_entropy - H_seq

    print("=" * 70)
    print("Proposition 1 synthetic diagnostic: viewpoint disambiguation")
    print("=" * 70)
    print(f"Prior entropy over {len(VIEWPOINTS_RAD)} viewpoints: {prior_entropy:.3f} bits")
    print(f"Posterior entropy after single observation: {H_single:.3f} bits")
    print(f"Posterior entropy after T={T} interaction steps: {H_seq:.3f} bits")
    print(f"Information gain (single obs): {information_gain_single:.3f} bits")
    print(f"Information gain (interaction): {information_gain_seq:.3f} bits")
    print(
        f"Interaction context carries {information_gain_seq - information_gain_single:.3f} "
        "extra bits about psi"
    )

    # Identify top hypotheses.
    top_single = VIEWPOINTS_DEG[np.argsort(single_likes)[-3:]].tolist()[::-1]
    top_seq = VIEWPOINTS_DEG[np.argsort(seq_likes)[-3:]].tolist()[::-1]
    print(f"Top viewpoints after single obs: {top_single}")
    print(f"Top viewpoints after interaction: {top_seq}")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    result = {
        "true_psi_deg": 45,
        "n_viewpoints": len(VIEWPOINTS_DEG),
        "T": T,
        "prior_entropy_bits": float(prior_entropy),
        "single_obs": {
            "posterior_entropy_bits": float(H_single),
            "information_gain_bits": float(information_gain_single),
            "top_viewpoints_deg": top_single,
        },
        "interaction_context": {
            "posterior_entropy_bits": float(H_seq),
            "information_gain_bits": float(information_gain_seq),
            "top_viewpoints_deg": top_seq,
        },
    }
    (out_dir / "information_probe.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved results to {out_dir / 'information_probe.json'}")


if __name__ == "__main__":
    main()
