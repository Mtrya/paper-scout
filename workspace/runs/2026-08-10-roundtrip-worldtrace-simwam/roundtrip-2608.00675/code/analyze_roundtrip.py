"""Aggregate round-trip probe arrays into figure data + stats.

Question: does round-trip consistency C_i track forward rollout error E_i?
Findings to quantify per regime r:
  - growth of median E / C / delta with depth
  - Spearman(C, E) and Spearman(delta, E) per depth
  - how much of C is explained by delta (the backward model's own error)
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"
DEPTHS = [5, 10, 20, 40, 80]
REGIMES = ["pendulum", "r0.5", "r3.0", "r10.0", "r20.0", "r28.0"]

summary = {}
for reg in REGIMES:
    d = np.load(os.path.join(OUT, reg, "arrays.npz"))
    entry = {"depths": DEPTHS, "E_med": [], "C_med": [], "delta_med": [],
             "rho_CE": [], "rho_dE": [], "rho_Cd": []}
    for i in DEPTHS:
        E, C, dl = d[f"E_{i}"], d[f"C_{i}"], d[f"delta_{i}"]
        ok = np.isfinite(E) & np.isfinite(C) & np.isfinite(dl)
        E, C, dl = E[ok], C[ok], dl[ok]
        entry["E_med"].append(float(np.median(E)))
        entry["C_med"].append(float(np.median(C)))
        entry["delta_med"].append(float(np.median(dl)))
        entry["rho_CE"].append(float(spearmanr(C, E).statistic) if ok.sum() > 10 else None)
        entry["rho_dE"].append(float(spearmanr(dl, E).statistic) if ok.sum() > 10 else None)
        entry["rho_Cd"].append(float(spearmanr(C, dl).statistic) if ok.sum() > 10 else None)
    summary[reg] = entry
    fmt = lambda xs: [('%.2f' % x) if x is not None else 'nan' for x in xs]
    print(f"{reg}: rho_CE={fmt(entry['rho_CE'])}")
    print(f"      rho_Cd={fmt(entry['rho_Cd'])}")
    print(f"      E_med={['%.2e' % x for x in entry['E_med']]}")
    print(f"      C_med={['%.2e' % x for x in entry['C_med']]}")

json.dump(summary, open(os.path.join(OUT, "roundtrip_summary.json"), "w"), indent=2)
print("saved roundtrip_summary.json")
