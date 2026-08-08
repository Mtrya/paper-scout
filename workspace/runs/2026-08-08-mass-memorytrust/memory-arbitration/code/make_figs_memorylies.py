#!/usr/bin/env python3
"""巡航 2026-08-08 线程 B(记忆仲裁)报告图:p1 模态鸿沟 + p2 图像消融。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

A = "../assets"  # 运行包 assets

CT, CV = "#1f77b4", "#d62728"   # 文本 / 视觉

# ---- 图 1:p1 模态鸿沟(Qwen3-VL-8B 复跑 vs 论文锚点)----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ours = {  # regime: (text F1, vision F1)
    "L1": (1.000, 0.158),
    "L2": (0.986, 0.284),
}
for ax, (reg, (ft, fv)) in zip(axes, ours.items()):
    b = ax.bar(["文本观测", "视觉观测"], [ft, fv], color=[CT, CV], width=0.55)
    for r, v in zip(b, [ft, fv]):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(0.887, color="#555", ls="--", lw=1)
    ax.text(1.42, 0.90, "论文 Qwen3.6 视觉 0.887", fontsize=8, color="#555", ha="right")
    ax.axhline(0.067, color="#777", ls=":", lw=1)
    ax.text(0.5, 0.085, "论文 GLM-5.1 视觉 0.067", fontsize=8, color="#777", ha="center")
    ax.set_title(f"regime {reg}(变化{'全局撒点' if reg=='L1' else '局部成簇'})")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("过期记忆检测 F1")
fig.suptitle("同一个模型,文本近天花板、视觉接近崩塌:Qwen3-VL-8B 复跑 SpatialSTALE", fontsize=12)
fig.tight_layout()
fig.savefig(f"{A}/memorylies-p1-modality-gap.png", dpi=150)
plt.close(fig)

# ---- 图 2:p2 图像消融(flag 率与翻转率)----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
conds = ["正确图", "空白图", "错配图"]
flag = {"L1": [0.248, 0.050, 0.267], "L2": [0.271, 0.067, 0.241]}
stale_ratio = {"L1": 0.094, "L2": 0.141}
x = np.arange(3)
w = 0.36
ax = axes[0]
for i, (reg, vals) in enumerate(flag.items()):
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=f"regime {reg}", color=[CT, CV][i])
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
for i, (reg, sr) in enumerate(stale_ratio.items()):
    ax.axhline(sr, color=[CT, CV][i], ls="--", lw=1, alpha=0.6)
ax.text(1.0, 0.102, "L1 真实 stale 比 9.4%", fontsize=8, color=CT, ha="center")
ax.text(1.0, 0.150, "L2 真实 stale 比 14.1%", fontsize=8, color=CV, ha="center")
ax.set_xticks(x); ax.set_xticklabels(conds)
ax.set_ylabel("条目被判 stale 的比率")
ax.set_title("图像内容如何改变审计输出(n=960/格)")
ax.legend(fontsize=9, loc="upper center"); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
flips = {"L1": [0.248, 0.287], "L2": [0.279, 0.306]}
x2 = np.arange(2)
w2 = 0.36
for i, (reg, vals) in enumerate(flips.items()):
    bars = ax.bar(x2 + (i - 0.5) * w2, vals, w2, label=f"regime {reg}", color=[CT, CV][i])
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
ax.text(0, 0.42, "同批判定的推理文本相似度\nvs 空白图 0.73–0.74", ha="center", fontsize=9, color="#555")
ax.text(1, 0.45, "同批判定的推理文本相似度\nvs 错配图 0.76", ha="center", fontsize=9, color="#555")
ax.set_xticks(x2); ax.set_xticklabels(["vs 空白图", "vs 错配图"])
ax.set_ylabel("逐条目判定翻转率(相对正确图)")
ax.set_ylim(0, 0.55)
ax.set_title("判定翻转 ~25–31%,但推理文本几乎不变")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{A}/memorylies-p2-ablation.png", dpi=150)
plt.close(fig)
print("done")
