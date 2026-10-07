"""Re-render fig_replace.png and fig_probe_steer.png with clean tick labels
(the auto-generated versions had colliding x labels / a garbled log-scale axis).
Reads code/analysis/replace.json and code/analysis/probe.json; writes into ../assets/.
Run from the thread dir:  python code/make_report_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
AN = HERE / "analysis"
OUT = HERE.parent.parent / "assets"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True})

def seed_means(d, fn):
    vals = [fn(v) for v in d.values()]
    return float(np.mean(vals)), float(np.std(vals))

# ---------------------------------------------------------------- fig_replace
rep = json.load(open(AN / "replace.json"))
conds = [
    ("BC",                       "bc",               "#4C7FB8"),
    ("RL",                       "rl",               "#C02739"),
    ("TS only\n(BC + RL TS)",    "ts_only",          "#E8821E"),
    ("non-TS only\n(RL + BC TS)", "nonts_only",      "#7E4FA3"),
    ("top-4 dirs\nof RL TS update", "ts_top4",       "#3E8853"),
    ("RL TS update\nw/o top-4 dirs", "ts_wo_top4",   "#8C6D1F"),
    ("neg. control\n(BC-branch top-4)", "bc_plus_ts_top4_bc", "#888888"),
]
means, stds, labels, colors = [], [], [], []
for lab, key, c in conds:
    m, s = seed_means(rep, lambda v, k=key: v[k]["success"])
    means.append(m); stds.append(s); labels.append(lab); colors.append(c)

fig, ax = plt.subplots(figsize=(9.2, 3.6), dpi=180)
x = np.arange(len(conds))
ax.bar(x, means, yerr=stds, color=colors, capsize=4, error_kw=dict(lw=1.2))
for xi, m in zip(x, means):
    ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.06); ax.set_ylabel("success rate")
ax.set_title("Module replacement: which parameters carry the RL gain")
fig.tight_layout()
fig.savefig(OUT / "fig_replace.png")
plt.close(fig)

# ------------------------------------------------------------- fig_probe_steer
probe = json.load(open(AN / "probe.json"))
p_m, p_s = seed_means(probe, lambda v: v["probe_all_auc"])
r_m, r_s = seed_means(probe, lambda v: v["probe_randomdir_auc"])
sh_m, sh_s = seed_means(probe, lambda v: v["probe_random_auc_mean"])

alphas = [0.25, 0.5, 1.0, 2.0]
def steer_series(policy, kind):
    base_key = "alpha0"
    ms, ss = [], []
    for a in alphas:
        key = f"{kind}_a{a}"
        m, s = seed_means(probe, lambda v, k=key, p=policy.lower(), b=base_key:
                          v[f"steer_{p}"][k] - v[f"steer_{p}"][b])
        ms.append(m); ss.append(s)
    return np.array(ms), np.array(ss)

series = [
    ("RL", "shift_adaptive",  "RL shift (adaptive)",  "#C02739", "-",  "o"),
    ("RL", "shift_fixed",     "RL shift (fixed)",     "#C02739", "--", "s"),
    ("RL", "random_fixed",    "RL random (fixed)",    "#C02739", ":",  "d"),
    ("BC", "shift_adaptive",  "BC shift (adaptive)",  "#4C7FB8", "-",  "o"),
    ("BC", "shift_fixed",     "BC shift (fixed)",     "#4C7FB8", "--", "s"),
    ("BC", "random_fixed",    "BC random (fixed)",    "#4C7FB8", ":",  "d"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6), dpi=180,
                               gridspec_kw={"width_ratios": [1, 1.25]})
bars = [("shift-update\ndirections", p_m, p_s, "#C02739"),
        ("random\ndirections", r_m, r_s, "#8C8C8C"),
        ("shuffled\nlabels", sh_m, sh_s, "#D3D3D3")]
for i, (lab, m, s, c) in enumerate(bars):
    ax1.bar(i, m, yerr=s, color=c, capsize=4, error_kw=dict(lw=1.2))
    ax1.text(i, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
ax1.axhline(0.5, ls="--", c="k", lw=0.8)
ax1.set_xticks(range(3)); ax1.set_xticklabels([b[0] for b in bars], fontsize=9)
ax1.set_ylim(0.4, 1.02); ax1.set_ylabel("ROC-AUC (episode outcome)")
ax1.set_title("Linear probe on shift-update projections")

xi = np.arange(len(alphas))
for pol, kind, lab, c, ls, mk in series:
    m, s = steer_series(pol, kind)
    ax2.errorbar(xi, m, yerr=s, label=lab, color=c, ls=ls, marker=mk,
                 ms=4, lw=1.4, capsize=3)
ax2.axhline(0, c="k", lw=0.8)
ax2.set_xticks(xi); ax2.set_xticklabels([str(a) for a in alphas])
ax2.set_xlabel("steering strength α")
ax2.set_ylabel("success change vs no steering")
ax2.set_title("Steering along the shift direction")
ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "fig_probe_steer.png")
plt.close(fig)
print("wrote", OUT / "fig_replace.png", "and", OUT / "fig_probe_steer.png")
