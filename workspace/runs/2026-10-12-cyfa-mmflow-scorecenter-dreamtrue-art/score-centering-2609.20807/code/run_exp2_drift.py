"""EXPERIMENT 2 -- constant reward (+1): pure drift, no signal.

Protocol: reward is exactly +1 for every rollout (the paper's "+1/0" reward mode,
the least stable online setting in their Figure 2), so the expected policy
gradient is E_q[R s] = s_bar = q - p and the signal term Cov_q(R, s) is exactly
zero. Everything is an exact expectation under the sampler; no sampling.

For each TIM model we run vanilla PG and the correction rules and record the
drift-resolved trajectory.

Usage (from a directory that contains toy_lib.py):
    uv run --with numpy --with matplotlib python3 run_exp2_drift.py
"""

import json
import os

import numpy as np

import toy_lib as T

FIGDIR = os.environ.get("SC_FIGDIR", "figs")
STEPS = 1500
LR = 0.1
METHODS = ["vanilla", "SC", "TIS", "MIS", "SC+TIS"]

LANES = [
    ("noise s=0.3", dict(tim="noise", sigma=0.3)),
    ("noise s=0.6", dict(tim="noise", sigma=0.6)),
    ("noise s=1.0", dict(tim="noise", sigma=1.0)),
    ("quant 4b", dict(tim="quant", bits=4)),
    ("quant 3b", dict(tim="quant", bits=3)),
    ("stale k=16", dict(tim="stale", stale_k=16)),
]


def run_lane(label, lane, steps=STEPS, lr=LR, methods=METHODS, reward_mode="const"):
    out = {}
    for m in methods:
        cfg = dict(V=64, C=4, sigma_r=0.35, seed=0, init_scale=0.5, method=m,
                   reward_mode=reward_mode, lr=lr, track_split=True)
        cfg.update(lane)
        toy, hist = T.run(cfg, steps=steps)
        out[m] = {k: [h[k] for h in hist] for k in
                  ["step", "tv_pq", "kl_qp", "tv_p0", "entropy", "logit_norm",
                   "reward_trainer", "update_norm", "drift_norm", "signal_norm",
                   "cos_update_drift"]}
    return out


def collapse_step(hist, thresh=1.0):
    """First step at which the policy entropy drops below thresh (nats)."""
    ent = hist["entropy"]
    for i, e in enumerate(ent):
        if e < thresh:
            return i + 1
    return None


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    results = {}
    for label, lane in LANES:
        res = run_lane(label, lane)
        results[label] = res
        print(f"-- {label}")
        for m, h in res.items():
            print(f"   {m:8s} tv_pq={h['tv_pq'][-1]:.4f} tv_p0={h['tv_p0'][-1]:.4f} "
                  f"entropy={h['entropy'][-1]:.4f} logit_norm={h['logit_norm'][-1]:.1f} "
                  f"update_norm={h['update_norm'][-1]:.4f} "
                  f"drift_ref={h['drift_norm'][-1]:.4f}")

    # ---- collapse-time sweep over the noise scale (paper Figure 3 analogue)
    sweep = {}
    for sigma in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        for m in ["vanilla", "SC", "TIS"]:
            cfg = dict(V=64, C=4, sigma_r=0.35, seed=0, init_scale=0.5, method=m,
                       reward_mode="const", lr=LR, tim="noise", sigma=sigma,
                       track_split=True)
            _, hist = T.run(cfg, steps=STEPS)
            sweep[f"{sigma}|{m}"] = {
                "sigma": sigma, "method": m,
                "collapse_step": collapse_step({k: [h[k] for h in hist] for k in hist[0]}),
                "entropy_end": hist[-1]["entropy"],
                "tv_p0_end": hist[-1]["tv_p0"],
                "tv_pq_end": hist[-1]["tv_pq"],
            }
    print("-- noise-scale collapse sweep (entropy < 1 nat)")
    for sigma in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        row = [sweep[f"{sigma}|{m}"] for m in ["vanilla", "SC", "TIS"]]
        print(f"   sigma={sigma:<4} " + "  ".join(
            f"{m}: step={r['collapse_step']} ent_end={r['entropy_end']:.2f}"
            for m, r in zip(["vanilla", "SC", "TIS"], row)))

    with open(os.path.join(FIGDIR, "exp2_drift.json"), "w") as f:
        json.dump({"lanes": results, "noise_sweep": sweep,
                   "config": {"steps": STEPS, "lr": LR, "V": 64, "C": 4,
                              "init_scale": 0.5, "seed": 0}}, f, indent=1)
    print(f"wrote {FIGDIR}/exp2_drift.json")


if __name__ == "__main__":
    main()
