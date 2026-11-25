#!/usr/bin/env python3
"""Draw the two redo figures: judge net-gain criterion + smoothing ceiling."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    try:
        font_manager.fontManager.addfont(path)
    except Exception:
        pass
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

# ---------- figure 1: judge net-gain criterion ----------
e_f, e_s, r, h = 0.8735, 0.5115, 0.1931, 0.0876   # REA Table 14, LIBERO-Pro 590
s = np.linspace(0, 1, 400)
gain_rea = 100 * ((1 - s) * e_f * r - s * e_s * h)
gain_toy = 100 * (1 - s)                            # toy fixture: r=1, h=0, e_f=1
s_star = e_f * r / (e_f * r + e_s * h)

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
ax.plot(s, gain_toy, ls="--", lw=2, color="#c44e52",
        label="玩具夹具：重跑永不毁坏（h=0，r=1）")
ax.plot(s, gain_rea, lw=2.4, color="#2b6cb0",
        label="REA 真实工作点（r=0.19，h=0.09）")
ax.axhline(0, color="#555", lw=1)
ax.axvline(s_star, color="#2b6cb0", lw=1, ls=":")
ax.annotate(f"盈亏平衡 s*≈{s_star:.2f}", xy=(s_star, 0), xytext=(s_star + 0.03, 4.5),
            fontsize=10, color="#2b6cb0")
ax.scatter([0.719], [1.53], zorder=5, color="#2b6cb0", s=55)
ax.annotate("590 集池：+1.5pp", xy=(0.719, 1.53), xytext=(0.62, -3.4), fontsize=10)
ax.scatter([0.514], [8.33], zorder=5, color="#dd8452", s=55)
ax.annotate("困难子集：+8.3pp", xy=(0.514, 8.33), xytext=(0.30, 10.6), fontsize=10)
ax.set_xlabel("首跑成功率 s（池子里本来就能做对的比例）")
ax.set_ylabel("判官门净收益（百分点 / 每咨询回合）")
ax.set_title("同一个判官，收益随池子强弱移动：净收益 = 失败占比×升级率×救回率 − 成功占比×重跑率×毁坏率",
             fontsize=11)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig("judge-net-gain.png")
plt.close(fig)

# ---------- figure 2: label-smoothing ceiling ----------
d = json.load(open("tau_result.json"))
rows = d["rows"]
eps = 0.1
ceil = np.array([1 - eps + eps / r_["n_options"] for r_ in rows])
pmax = np.array([r_["p_max"] for r_ in rows])
corr = np.array([r_["correct"] for r_ in rows])

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
ax.scatter(ceil[corr], pmax[corr], s=16, alpha=0.45, color="#2b6cb0",
           label=f"判对（{int(corr.sum())} 条）")
ax.scatter(ceil[~corr], pmax[~corr], s=16, alpha=0.45, color="#c44e52",
           label=f"判错（{int((~corr).sum())} 条）")
lim = [0.895, 1.0]
ax.plot(lim, lim, color="#555", lw=1, ls="--")
ax.text(0.897, 0.905, "y = x（贴着上限）", fontsize=9, color="#555")
ax.axhline(0.918, color="#dd8452", lw=1.4)
ax.axvline(0.918, color="#dd8452", lw=1.4)
ax.text(0.896, 0.921, "τ_cold = 0.918", fontsize=10, color="#dd8452")
ax.axhspan(0.97, 1.0, color="#c44e52", alpha=0.08)
ax.text(0.955, 0.983, "τ≥0.97：0/600 条样本", fontsize=9, color="#c44e52")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("平滑置信度上限 1 − ε + ε/K（ε=0.1，K=该题候选数）")
ax.set_ylabel("官方权重实测最大置信度（600 条抽样）")
ax.set_title("标签平滑把置信度天花板钉在 1−ε+ε/K：阈值只能贴着天花板下方放",
             fontsize=11)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("smoothing-ceiling.png")
plt.close(fig)
print("figures written")
