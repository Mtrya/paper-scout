#!/usr/bin/env python
# SyncWorld 复现+可观性探针 双面板图(中文标注)
# 用法: python3 syncworld_plot.py <pilot_metrics.csv> <probe_analysis.json> <out.png>
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(f)
        break
    except Exception:
        continue
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

pilot_csv, probe_json, out_png = sys.argv[1], sys.argv[2], sys.argv[3]

# ---- 左:pilot 逐段指标 ----
seg_start, seg_psnr, seg_lpips = [], [], []
with open(pilot_csv) as fh:
    for r in csv.DictReader(fh):
        if r["episode"] == "average":
            avg_psnr = float(r["psnr"]); continue
        seg_start.append(int(r["start_frame"]))
        seg_psnr.append(float(r["psnr"]))
        seg_lpips.append(float(r["lpips"]))

# ---- 右:probe 三条件 × 两 episode ----
data = json.load(open(probe_json))
eps = {e["episode"]: e for e in data["episodes"]}
view = "agentview_rgb"
tags = [t for t in ["full", "null", "droprot"] if t in data["tags"]]
tag_label = {"full": "完整标定", "null": "全零标定", "droprot": "抹旋转三槽"}
colors = {"full": "#4C9F70", "null": "#B0B0B0", "droprot": "#C0504D"}
ep_ids = sorted(e for e in eps if any(f"{t}:{view}" in eps[e] for t in tags))
# 每 episode 的 rot 主导度(直接用 rot/trans 原始比)
def ratio(e):
    return eps[e]["rot_motion"] / max(eps[e]["trans_motion"], 1e-9)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

ax1.plot(seg_start, seg_psnr, "-o", ms=3.5, lw=1.4, color="#3A6EA5")
ax1.axhline(avg_psnr, ls="--", lw=1, color="#888888")
ax1.text(seg_start[-1], avg_psnr + 0.15, f"平均 {avg_psnr:.1f}", ha="right", fontsize=9, color="#666666")
ax1.set_xlabel("rollout 起始帧(闭环回写,共 272 帧)")
ax1.set_ylabel("PSNR (dB)")
ax1.set_title("复现:发布权重逐段模拟质量\n(论文口径 35 步 / CFG 5.0,无累积漂移)", fontsize=10.5)

x = np.arange(len(ep_ids))
w = 0.26
for j, tag in enumerate(tags):
    vals = []
    for e in ep_ids:
        m = eps[e].get(f"{tag}:{view}")
        vals.append(m["psnr"] if m else np.nan)
    ax2.bar(x + (j - 1) * w, vals, w * 0.92, label=tag_label[tag], color=colors[tag])
    for xi, v in zip(x + (j - 1) * w, vals):
        if not np.isnan(v):
            ax2.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
labels = []
for e in ep_ids:
    dom = "转动主导" if ratio(e) > 1.5 else "平移主导"
    labels.append(f"episode {e}\n({dom} {ratio(e):.1f}×)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("PSNR (dB)")
ax2.set_title("探针:抹掉旋转标定槽的退化\n是否选择性落在转动主导的 episode 上", fontsize=10.5)
ax2.legend(fontsize=9, loc="lower left")

fig.tight_layout()
fig.savefig(out_png, dpi=160)
print("wrote", out_png)
