"""Muon 探针 U 形曲线图:平稳尾部 ‖g‖_nuc vs β,叠加漂移项预测。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "results_muon_floor_v2.json").read_text())
rs = d["results"]
b1 = [r for r in rs if r["B"] == 1]
betas = [r["beta"] for r in b1]
floors = [r["tail_avg_gnuc"] for r in b1]
sems = [r["sem"] for r in b1]
drift = [r["drift_term"] for r in b1]
L = d["config"]["L"]; eta = d["config"]["eta"]
# 噪声残余: 用 β=0.9 的观测作为噪声地板锚点, 按 sqrt(1-beta)/sqrt(0.1) 缩放
anchor = next(r["tail_avg_gnuc"] for r in b1 if r["beta"] == 0.9)
noise = [anchor * ( (1-b)/(1-0.9) ) ** 0.5 for b in betas]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
ax.errorbar(betas, floors, yerr=sems, fmt="o-", color="#1a6faf", lw=1.8,
            ms=5, capsize=2.5, label="observed stationary floor (tail avg, 5 seeds)")
ax.plot(betas, drift, "--", color="#c0392b", lw=1.6, label=r"drift bound  $2\beta L\eta/(1-\beta)$")
ax.plot(betas, noise, ":", color="#7f8c8d", lw=1.6,
        label=r"noise scaling  $\propto\sqrt{1-\beta}$ (anchored at β=0.9)")
ax.plot(betas, [dv + nv for dv, nv in zip(drift, noise)], "-", color="#e67e22",
        lw=1.2, alpha=0.8, label="bound sum (drift + noise)")
ax.set_xlabel(r"momentum $\beta$")
ax.set_ylabel(r"tail avg $\|g\|_{\mathrm{nuc}}$")
ax.set_yscale("log")
ax.set_title(f"Muon-style msign on stochastic quadratic (64x32, L={L:.2f}, η={eta:.4f}, B=1)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
out = HERE.parent.parent / "assets" / "muon-u-shape.png"
fig.savefig(out)
print("written", out, out.stat().st_size)
