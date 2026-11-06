#!/usr/bin/env python3
"""Summarize tnas_results.json: rho stats + Probe-3 sweep means vs paper values."""
import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "tnas_results.json"
rows = json.load(open(path))
print(f"n = {len(rows)}")
rho = np.array([r["rho"] for r in rows])
print(f"rho: mean={rho.mean()*100:.2f}% median={np.median(rho)*100:.2f}%  (paper: 5.28% / 4.90%)")
print(f"rho>99.8% retained share: mean rho^2 = {(rho**2).mean():.5f} (paper 0.00382)")

# rand dirs averaged per example
def sweep_mean(dirnames, q_target):
    vals, kls = [], []
    for r in rows:
        sel = [s for s in r["sweeps"] if s["dir"] in dirnames and abs(abs(s["q"]) - q_target) < 1e-6]
        # paper aggregates by sign-aligned q: both signs included
        if sel:
            vals.append(np.mean([s["action_shift"] for s in sel]))
            kls.append(np.mean([s["tf_kl"] for s in sel]))
    return np.mean(vals), np.mean(kls), len(vals)

paper = {  # Table 8 InstructVLA
    "tnas":  {0.5: 0.0734, 1.0: 0.110, 2.0: 0.156},
    "act":   {0.5: 0.0734, 1.0: 0.110, 2.0: 0.155},
    "rand":  {0.5: 0.00861, 1.0: 0.00999, 2.0: 0.0124},
    "nll":   {0.5: 0.0134, 1.0: 0.0197, 2.0: 0.0283},
    "margin":{0.5: 0.0158, 1.0: 0.0198, 2.0: 0.0337},
}
paper_kl2 = {"tnas": 2.20e-2, "act": 2.71e-2, "rand": 3.64e-3, "nll": 5.80e-1, "margin": 4.66e-1}
dirs = {"tnas": ["tnas"], "act": ["act"], "rand": ["rand0", "rand1", "rand2", "rand3"],
        "nll": ["nll"], "margin": ["margin"]}
print("\nq sweep (mean shift | paper) and KL@q=2 (ours | paper):")
for name, keys in dirs.items():
    line = f"{name:7s}"
    for q in (0.5, 1.0, 2.0):
        m, _, _ = sweep_mean(keys, q)
        line += f"  q={q}: {m:.4f}|{paper[name][q]:.4f}"
    _, kl2, _ = sweep_mean(keys, 2.0)
    line += f"   KLq2: {kl2:.2e}|{paper_kl2[name]:.2e}"
    print(line)

# response stats vs paper (9.60 words avg, 178/200 distinct)
words = [len(r["response"].split()) for r in rows]
distinct = len(set(r["response"] for r in rows))
print(f"\nresponses: mean {np.mean(words):.2f} words (paper 9.60), distinct {distinct}/{len(rows)} (paper 178/200)")
