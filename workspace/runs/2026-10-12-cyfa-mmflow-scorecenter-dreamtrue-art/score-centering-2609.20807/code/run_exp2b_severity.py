"""EXPERIMENT 2b -- constant-reward drift, amplified (larger lr): the severity
axis of the paper's Figure 3.

run_exp2_drift.py uses the faithful objective scale (a prefix's contribution is
weighted by the probability of visiting it, and by 1/length), which makes the
256 depth-2 rows move 64x slower than the 4 root rows and hides the collapse in
an average over all rows.  This script raises lr by 10x and reports root-row
diagnostics separately, so the "mild mismatch: no separation / severe mismatch:
vanilla collapses, SC frozen" axis becomes visible.  Same rules, same TIMs.

Usage: uv run --with numpy python3 run_exp2b_severity.py
"""

import json
import os

import numpy as np

import toy_lib as T

FIGDIR = os.environ.get("SC_FIGDIR", "figs")
STEPS = 600
LR = 1.0
METHODS = ["vanilla", "SC", "TIS", "MIS"]
LANES = [("noise 0.6", dict(tim="noise", sigma=0.6)),
         ("noise 2.0", dict(tim="noise", sigma=2.0)),
         ("quant 3b", dict(tim="quant", bits=3)),
         ("stale k=16", dict(tim="stale", stale_k=16))]


def root_metrics(toy):
    """Root rows only (4 of 260), where the drift is not visit-weighted."""
    t1, t2 = toy.params.logits()
    p1 = T.softmax(t1)
    p1_0 = T.softmax(toy.logits_0[0])
    s1, s2 = toy.tim.sampler_logits(t1, t2)
    q1 = T.softmax(s1)
    return {
        "root_tv_p0": float(np.mean(T.tv(p1, p1_0))),
        "root_tv_pq": float(np.mean(T.tv(p1, q1))),
        "root_entropy": float(np.mean(
            -(p1 * np.log(np.maximum(p1, 1e-300))).sum(-1))),
    }


def main():
    out = {}
    for label, lane in LANES:
        out[label] = {}
        for m in METHODS:
            cfg = dict(V=64, C=4, sigma_r=0.35, seed=0, init_scale=0.5, lr=LR,
                       method=m, reward_mode="const", track_split=True, **lane)
            toy, hist = T.run(cfg, steps=STEPS)
            # re-run with root diagnostics at the same seeds/config
            toy2 = T.Toy(cfg)
            rl = []
            for _ in range(STEPS):
                toy2.step()
                rl.append(root_metrics(toy2))
            out[label][m] = {
                "entropy": [h["entropy"] for h in hist],
                "tv_p0": [h["tv_p0"] for h in hist],
                "tv_pq": [h["tv_pq"] for h in hist],
                "update_norm": [h.get("update_norm", float("nan")) for h in hist],
                "root_tv_p0": [r["root_tv_p0"] for r in rl],
                "root_tv_pq": [r["root_tv_pq"] for r in rl],
                "root_entropy": [r["root_entropy"] for r in rl],
            }
        print(f"-- {label}")
        for m in METHODS:
            h = out[label][m]
            print(f"   {m:8s} ent 4.04->{h['entropy'][-1]:.3f}  all-row tv_p0={h['tv_p0'][-1]:.4f}"
                  f"  root tv_p0={h['root_tv_p0'][-1]:.4f}  root ent={h['root_entropy'][-1]:.3f}"
                  f"  root tv_pq={h['root_tv_pq'][-1]:.4f}")
    with open(os.path.join(FIGDIR, "exp2b_severity.json"), "w") as f:
        json.dump({"config": {"steps": STEPS, "lr": LR, "V": 64, "C": 4},
                   "lanes": out}, f)
    print(f"wrote {FIGDIR}/exp2b_severity.json")


if __name__ == "__main__":
    main()
