#!/usr/bin/env python3
"""Gram probe figures: scale comparison + plan link + direction consistency."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

per = json.load(open("gram_local_out/gram_per_anchor.json"))["per_anchor"]
summ = json.load(open("gram_local_out/gram_summary.json"))
epss = [k.split("_")[0] for k in summ if k.endswith("_D_G_median")]
epss = sorted(set(epss), key=lambda s: float(s[3:]))

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), dpi=160)
x = np.arange(len(epss))
for ax, metric, title in [(axes[0], "D_G", "trace-normalized Gram discrepancy $D_G$"),
                          (axes[1], "gain", "sensitivity gain $\\sqrt{tr(G_m)/tr(G_e)}$")]:
    med = [summ[f"{e}_{metric}_median"] for e in epss]
    iqr = [summ[f"{e}_{metric}_iqr"] for e in epss]
    ax.errorbar(x, med, yerr=[np.array(med) - np.array(iqr)[:, 0], np.array(iqr)[:, 1] - np.array(med)],
                fmt="o-", capsize=4, color="#4878cf")
    ax.set_xticks(x)
    ax.set_xticklabels(epss)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("perturbation scale ε", fontsize=9)
axes[0].set_ylim(0, 1.2)

dg = np.array([r[f"{epss[0]}"]["D_G"] for r in per if r["plan_pair"]])
rho = np.array([r["plan_rho"] for r in per if r["plan_pair"]])
ok = ~np.isnan(rho)
axes[2].scatter(dg[ok], rho[ok], s=22, color="#ee854a")
axes[2].set_xlabel("per-pair $D_G$ (ε=0.001)", fontsize=9)
axes[2].set_ylabel("per-pair Plan-Real Spearman ρ", fontsize=9)
axes[2].set_title(f"geometry vs planning (n={int(ok.sum())}, rank corr {summ.get('planlink_spearman_DG_vs_planrho', float('nan')):.3f})", fontsize=9)
fig.tight_layout()
fig.savefig("gram_main.png")
print("saved gram_main.png")
print(json.dumps({k: v for k, v in summ.items() if not k.endswith("iqr")}, indent=1))
