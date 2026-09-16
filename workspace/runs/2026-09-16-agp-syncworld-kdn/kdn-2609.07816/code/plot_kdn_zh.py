"""实验 C 报告图:v3/v4 对照 —— 池化不确定度的制度内优势与制度外崩溃。中文版。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

NAMES = [("fixed-0.1", "固定 $\\beta$=0.1(delta 规则)", "#999999", ":"),
         ("gated", "可靠性门控 $\\beta_t$", "#1f77b4", "--"),
         ("kalman-iso", "Kalman 各向同性(池化)", "#d62728", "-"),
         ("kalman-diag", "Kalman 对角(逐通道)", "#2ca02c", "-")]


def load(path):
    d = json.load(open(path))
    return d["curve"], d["drifts"]


def plot_panel(ax, curve, drifts, group, title, ylab=True):
    for key, label, color, ls in NAMES:
        xs = [c["t"] for c in curve]
        ys = [c[f"{key}_{group}"] for c in curve]
        ax.semilogy(xs, ys, ls, color=color, lw=1.8 if "kalman" in key else 1.2, label=label)
    for d in drifts:
        ax.axvline(d, color="black", alpha=0.15, lw=0.8)
    ax.set_title(title, fontsize=11)
    if ylab:
        ax.set_ylabel("回忆 MSE(对数)")
    ax.set_xlabel("流 token 位置")
    ax.set_ylim(5e-4, 3)


fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)

c3, d3 = load("probe_v3_curve.json")
plot_panel(axes[0], c3, d3, "d", "A. 漂移通道 70% 噪声:池化赢")
axes[0].text(0.02, 0.04, "制度内 iso 处处优于 diag", transform=axes[0].transAxes, fontsize=9, color="#d62728")

c4, d4 = load("probe_v4_curve.json")
plot_panel(axes[1], c4, d4, "d", "B. 漂移通道 100% 噪声:池化饿死", ylab=False)
axes[1].text(0.02, 0.04, "iso 漂移误差超过固定-$\\beta$;diag 保持近最优", transform=axes[1].transAxes, fontsize=9, color="#2ca02c")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("漂移组回忆 MSE:逐方向不确定度什么时候重要?", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("kdn_v3v4_contrast_zh.png", bbox_inches="tight")
print("saved kdn_v3v4_contrast_zh.png")

# 第二张:v4 stable 组 + drift 组双线,展示 diag 的全制度近最优
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
plot_panel(axes[0], c4, d4, "s", "C. 稳定通道组(始终可靠)")
plot_panel(axes[1], c4, d4, "d", "D. 漂移通道组(始终噪声)", ylab=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("v4 设置双组对照:对角是唯一在两组都近最优的臂", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("kdn_v4_both_groups_zh.png", bbox_inches="tight")
print("saved kdn_v4_both_groups_zh.png")
