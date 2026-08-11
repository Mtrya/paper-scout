"""Invisible Shortcuts probe figure: MP + displacement across 3 encoders."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]

sig = json.load(open("drafts/rynnvalue-probe/shortcut_siglip.json"))["encoders"]["siglip"]
cd = json.load(open("drafts/rynnvalue-probe/shortcut_clipdinov2.json"))["encoders"]
data = {"SigLIP-SO400M": sig, "CLIP-B/16": cd["clip"], "DINOv2-B": cd["dinov2"]}
names = list(data)
mp = [data[n]["MP_jpeg_acc"] * 100 for n in names]
disp = [data[n]["disp_ratio"] * 100 for n in names]
spd = [data[n]["SPD_delta"] * 100 for n in names]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
colors = ["#c0392b", "#e67e22", "#2980b9"]

ax = axes[0]
bars = ax.bar(names, mp, color=colors, width=0.55)
ax.axhline(20, color="#7f8c8d", ls="--", lw=1.2)
ax.text(2.35, 21.5, "随机基线 20%", fontsize=9, color="#7f8c8d", ha="right")
for b, v in zip(bars, mp):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", fontsize=11, weight="bold")
ax.set_ylabel("JPEG 质量线性可解码率 MP(%)")
ax.set_ylim(0, 68)
ax.set_title("语言监督编码器的特征里\n躺着压缩痕迹,DINOv2 没有", fontsize=11)
ax.grid(alpha=0.25, axis="y")

ax = axes[1]
x = np.arange(3)
b1 = ax.bar(x - 0.19, disp, 0.36, color=colors, alpha=0.85, label="同图 q95↔q30 位移比")
b2 = ax.bar(x + 0.19, spd, 0.36, color="#95a5a6", label="语义预测翻转 SPD Δ")
for b, v in zip(b1, disp):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}%", ha="center", fontsize=10)
for b, v in zip(b2, spd):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}%", ha="center", fontsize=10, color="#7f8c8d")
ax.set_xticks(x, names)
ax.set_ylabel("百分比(%)")
ax.set_title("痕迹挪得动特征(位移),\n但随机指派下翻不动语义(SPD≈0)", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.25, axis="y")

fig.suptitle("不可见捷径探针:imagenette 500 图 × 5 档 JPEG 质量随机指派,三个真实编码器", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("drafts/rynnvalue-probe/shortcut_report.png", dpi=170)
print("ok")
