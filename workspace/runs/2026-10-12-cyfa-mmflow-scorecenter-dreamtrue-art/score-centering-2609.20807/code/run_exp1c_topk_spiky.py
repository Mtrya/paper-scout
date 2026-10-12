"""EXPERIMENT 1c -- top-k residual drift as a function of how peaked the
next-token distribution is (the regime the paper works in: k=128 covering
>99.9% of the sampler's mass).

Same certified formula as run_exp1_official.topk_residual: with a constant
reward the residual drift of top-k centering is
    || E_q[s] - sum_{v in H} (q_v - rho p_v)(e_v - p) || / ||q - p||.
Usage: uv run --with jax --with optax --with numpy python3 run_exp1c_topk_spiky.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
from run_exp1_official import topk_residual  # noqa: E402

FIGDIR = os.environ.get("SC_FIGDIR", "figs")


def main():
    rng = np.random.default_rng(7)
    V = 4096
    rows = []
    for label, spike, sd in (("flat  sd=3.0", 0.0, 3.0),
                             ("mid   sd=3.0+spike 12", 12.0, 3.0),
                             ("peaky sd=1.5+spike 15", 15.0, 1.5),
                             ("wordlike sd=1.0+spike 25", 25.0, 1.0)):
        for trial in range(3):
            logits = sd * rng.standard_normal(V)
            if spike:
                idx = rng.permutation(V)[:8]
                logits[idx] += spike * np.linspace(1.0, 0.3, 8)
            p_logits = logits + 0.05 * rng.standard_normal(V)
            for k in (32, 128, 512, 2048):
                rec = topk_residual(logits, p_logits, k)
                rec.update(lane=label, trial=trial, V=V)
                rows.append(rec)
    print(f"{'lane':24s} {'k':>5s} {'head mass':>10s} {'residual/vanilla':>17s}")
    for label in sorted({r['lane'] for r in rows}):
        for k in (32, 128, 512, 2048):
            sub = [r for r in rows if r["lane"] == label and r["k"] == k]
            print(f"{label:24s} {k:5d} {np.mean([r['head_mass'] for r in sub]):10.5f}"
                  f" {np.mean([r['residual_frac'] for r in sub]):17.3e}")
    with open(os.path.join(FIGDIR, "exp1c_topk_spiky.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {FIGDIR}/exp1c_topk_spiky.json")


if __name__ == "__main__":
    main()
