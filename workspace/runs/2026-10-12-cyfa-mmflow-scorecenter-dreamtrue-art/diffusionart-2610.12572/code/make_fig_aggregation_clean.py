#!/usr/bin/env python3
"""从 results/prop1.json 重画聚合残差图(替换 fig2_aggregation.png 的标题重叠版)。

数据全部来自 exp1_prop1.py 的真实测量(30 状态恒等式 + 48 状态聚合 + 5 档求解器步数),
本脚本只做可视化。输出:code/figs/fig2b_aggregation_clean.png 与 run 包 assets/art-aggregation.png。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / "assets"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

d = json.loads((HERE / "results/prop1.json").read_text())
agg = d["aggregation"]
gap = d["solver_gap"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.0))

# Panel A: 16 步求解器(论文设定)下,逐状态 vs 聚合后的引导/残差强度
labels = ["引导 2g", "一致性残差 (2q-1)e"]
per_state = [agg["mean_norm_guidance"], agg["mean_norm_residual"]]
aggregated = [agg["norm_mean_guidance"], agg["norm_mean_residual"]]
x = range(2)
w = 0.36
b1 = ax1.bar([i - w / 2 for i in x], per_state, w, color="#4C90C4", label="逐状态平均强度")
b2 = ax1.bar([i + w / 2 for i in x], aggregated, w, color="#E8762B", label="48 状态聚合后(均值向量的模)")
for bars in (b1, b2):
    for r in bars:
        ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.3f}",
                 ha="center", fontsize=8)
ax1.set_xticks(list(x), labels)
ax1.set_ylabel("输出空间力的模")
ax1.set_ylim(0, 0.26)
ax1.legend(fontsize=8, loc="upper right")
ax1.set_title("逐状态残差≈引导的 20%,聚合后只剩 6.8%\n(相干性:引导 0.66 vs 残差 0.22)", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# Panel B: 残差/引导比随求解器步数
steps = [e["steps"] for e in gap]
per = [e["mean_rel_e"] * 100 for e in gap]
agg_r = [e["aggregate_ratio"] * 100 for e in gap]
ax2.plot(steps, per, "o-", color="#4C90C4", label="逐状态 |e|/|引导| 均值")
ax2.plot(steps, agg_r, "s-", color="#E8762B", label="聚合后残差/引导")
for xi, yi in zip(steps, agg_r):
    ax2.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
ax2.set_xscale("log", base=2)
ax2.set_xticks(steps, [str(s) for s in steps])
ax2.set_xlabel("rollout 求解器步数")
ax2.set_ylabel("残差相对引导的强度 (%)")
ax2.set_title("求解器越粗,聚合残差越压不住\n(4 步时聚合残差仍达 22.9%)", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
out1 = HERE / "figs" / "fig2b_aggregation_clean.png"
out2 = ASSETS / "art-aggregation.png"
fig.savefig(out1, bbox_inches="tight")
fig.savefig(out2, bbox_inches="tight")
print("written:", out1, out2)
