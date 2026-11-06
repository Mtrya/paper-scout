import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

d = json.load(open("code/tnas_results.json"))
acc = defaultdict(lambda: defaultdict(list))
for ex in d:
    for s in ex["sweeps"]:
        g = "rand" if s["dir"].startswith("rand") else s["dir"]
        acc[g][abs(s["q"])].append((s["action_shift"], s["tf_kl"]))

qs = [0.5, 1, 2, 4]
style = {"tnas": ("#d62728", "o", "tnas (text-aligned)"), "act": ("#ff9896", "s", "act (action jacobian)"),
         "nll": ("#1f77b4", "^", "nll (text grad)"), "margin": ("#2ca02c", "v", "margin (text grad)"),
         "rand": ("#7f7f7f", "x", "rand (4 dirs)")}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
for g, (c, m, lab) in style.items():
    sh = [np.mean([x[0] for x in acc[g][q]]) for q in qs]
    kl = [np.mean([x[1] for x in acc[g][q]]) for q in qs]
    axes[0].plot(qs, sh, marker=m, color=c, label=lab, lw=1.6, ms=5)
    axes[1].scatter(kl, sh, marker=m, color=c, label=lab, s=42)
    axes[1].plot(kl, sh, color=c, lw=0.8, alpha=0.5)
axes[0].set_xlabel("intervention magnitude q (equal-norm)")
axes[0].set_ylabel("action shift (mean per-step L2)")
axes[0].set_yscale("log"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7.5)
axes[1].set_xlabel("response-span teacher-forced KL (log)")
axes[1].set_ylabel("action shift (log)")
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].grid(alpha=0.3)
fig.suptitle("InstructVLA TNAS intervention, 200 Bridge examples (ours)", fontsize=10)
fig.tight_layout()
fig.savefig("runs/2026-11-05-dreamx-primesim-tnas/assets/tnas_probe.png", dpi=170)
print("saved")
