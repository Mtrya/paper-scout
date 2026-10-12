"""EXPERIMENT 3 -- with signal: does correcting drift cost learning?

Task: the two-step chain of toy_lib (both tokens must land near the context's
target angle).  Reward mode is the paper's default, REINFORCE with
group-centered advantages (Eq. 13); one SGD step per batch, SGD lr as in the
paper's setup (large because the toy's per-step gradient is a per-prefix mean).

Three runners per lane:
  * exact  -- exact expectation under the sampler (deterministic dynamics);
  * sampled -- G rollouts per context, group centering, per-token IS weights,
    reward observation noise (the honest control);
  * sampled with a shared low-rank head (logits = U @ W.T, one W for all
    prefixes), which is what couples prefixes through shared parameters, as in
    a real network.

Usage: uv run --with numpy python3 run_exp3_signal.py
"""

import json
import os

import numpy as np

import toy_lib as T

FIGDIR = os.environ.get("SC_FIGDIR", "figs")
LR = 4.0
STEPS = 300
METHODS = ["vanilla", "SC", "TIS", "MIS", "SC+TIS", "SC+MIS"]
LANES = [
    ("on-policy", dict(tim="none")),
    ("noise 1.5", dict(tim="noise", sigma=1.5)),
    ("noise 3.0", dict(tim="noise", sigma=3.0)),
    ("quant 3b", dict(tim="quant", bits=3)),
    ("stale k=64", dict(tim="stale", stale_k=64)),
]


def cfg_for(method, lane, param="table", seed=0, lr=LR, **extra):
    cfg = dict(V=64, C=4, sigma_r=0.35, seed=seed, init_scale=0.5, lr=lr,
               method=method, reward_mode="group_centered", param=param,
               rank=4)
    cfg.update(lane)
    cfg.update(extra)
    return cfg


def tail_mean(hist, key, n=20):
    return float(np.mean([h[key] for h in hist[-n:]]))


def run_exact(lane, param="table"):
    out = {}
    for m in METHODS:
        _, h = T.run(cfg_for(m, lane, param), steps=STEPS)
        out[m] = {k: [x[k] for x in h] for k in
                  ("step", "reward_trainer", "reward_sampler", "tv_pq",
                   "entropy", "tv_p0", "logit_norm")}
        out[m]["final"] = tail_mean(h, "reward_trainer")
    return out


def run_sampled(lane, param="table", seeds=(0, 1, 2), group=8, steps=STEPS):
    out = {}
    for m in METHODS:
        finals, curves = [], []
        for seed in seeds:
            cfg = cfg_for(m, lane, param, seed=seed)
            _, h = T.run_sampled(cfg, steps=steps, group=group,
                                 reward_noise=0.05, seed=seed)
            finals.append(tail_mean(h, "reward_trainer"))
            curves.append([x["reward_trainer"] for x in h])
        out[m] = {"final_mean": float(np.mean(finals)),
                  "final_std": float(np.std(finals)),
                  "curves": curves,
                  "finals": finals}
    return out


def anatomy(TIM_LANE):
    """Per-prefix drift / signal norms at the initial policy: the paper's claim
    that group centering shrinks drift but does not remove it, and that drift
    lives exactly at the prefixes that carry the learning signal."""
    rows = []
    for lane_name, lane in TIM_LANE:
        for mode in ("group_centered", "const"):
            cfg = cfg_for("vanilla", lane)
            cfg["reward_mode"] = mode
            toy = T.Toy(cfg)
            p1, p2, q1, q2 = toy.dists()
            Rvec1, Rvec2 = toy.advantages(p1, q1, p2, q2)
            dr1, sg1 = T.decompose(p1, q1, Rvec1)
            dr2, sg2 = T.decompose(p2, q2, Rvec2)
            vis = q1[:, :, None]
            rows.append({
                "lane": lane_name, "reward_mode": mode,
                "root_drift": float(np.linalg.norm(dr1)) / 2,
                "root_signal": float(np.linalg.norm(sg1)) / 2,
                "d2_drift": float(np.linalg.norm(dr2 * vis)) / 2,
                "d2_signal": float(np.linalg.norm(sg2 * vis)) / 2,
                "mean_abs_A_d2": float(np.abs(Rvec2).mean()),
                "mean_abs_A_root": float(np.abs(Rvec1).mean()),
            })
    return rows


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    results = {}
    for param in ("table", "lowrank"):
        for lane_name, lane in LANES:
            if param == "lowrank" and lane_name == "quant 3b":
                continue
            ex = run_exact(lane, param)
            sm = run_sampled(lane, param, seeds=(0, 1, 2))
            results[f"{param}|{lane_name}"] = {"exact": ex, "sampled": sm}
            print(f"-- {param} / {lane_name}")
            for m in METHODS:
                print(f"   {m:8s} exact={ex[m]['final']:.3f}  "
                      f"sampled={sm[m]['final_mean']:.3f}"
                      f"±{sm[m]['final_std']:.3f}")

    print("-- drift anatomy at the initial policy (norm of the expected update)")
    anat = anatomy(LANES)
    for r in anat:
        if r["reward_mode"] != "group_centered":
            continue
        print(f"   {r['lane']:11s} root drift={r['root_drift']:.5f} "
              f"root signal={r['root_signal']:.5f} | d2 drift={r['d2_drift']:.5f} "
              f"d2 signal={r['d2_signal']:.5f}")
    const = [r for r in anat if r["reward_mode"] == "const"]
    print(f"   [+1 constant reward] mean root drift="
          f"{np.mean([r['root_drift'] for r in const]):.5f}, "
          f"mean d2 drift={np.mean([r['d2_drift'] for r in const]):.5f}")

    with open(os.path.join(FIGDIR, "exp3_signal.json"), "w") as f:
        json.dump({"results": results, "anatomy": anat,
                   "config": {"lr": LR, "steps": STEPS, "group": 8,
                              "reward_noise": 0.05, "V": 64, "C": 4}},
                  f, indent=1)
    print(f"wrote {FIGDIR}/exp3_signal.json")


if __name__ == "__main__":
    main()
