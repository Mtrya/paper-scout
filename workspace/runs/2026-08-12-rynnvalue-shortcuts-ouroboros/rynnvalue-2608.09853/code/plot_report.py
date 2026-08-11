"""Report-version RynnValue probe figure: 2x3 panels, Chinese labels."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

main = json.load(open("drafts/rynnvalue-probe/probe_results.json"))
extra = json.load(open("drafts/rynnvalue-probe/probe_results_extra.json"))
dense = json.load(open("drafts/rynnvalue-probe/probe_results_dense.json"))
conds = {**main["conditions"], **extra["conditions"], **dense["conditions"]}
T, FPS = main["total_frames"], main["fps"]


def smooth(y, w=7):
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def xy(name):
    c = conds[name]
    n = len(c["values"])
    pos = np.array(c["eval_positions"]) / (max(c["eval_positions"]) or 1)
    return pos, np.array(c["values"]), np.array(c["true_last_frame_ts"])


fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))

# Panel 1: forward / reversed / truncate
ax = axes[0][0]
for name, color, lab in [("forward", "#2980b9", "正向(对照)"), ("reversed", "#c0392b", "倒放"),
                         ("truncate", "#27ae60", "截断 90%")]:
    pos, v, _ = xy(name)
    ax.plot(pos, smooth(v), color=color, lw=1.8, label=lab)
ax.set_title("倒放:预测剩余时间随播放上升", fontsize=11)
ax.set_xlabel("相对播放位置")
ax.set_ylabel("预测剩余时间 v(秒)")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# Panel 2: frozen40 / frozen80 vs forward
ax = axes[0][1]
pos, v, _ = xy("forward")
ax.plot(pos, smooth(v), color="#2980b9", lw=1.8, label="正向(对照)")
for name, cut, color in [("frozen", 0.4, "#e67e22"), ("frozen80", 0.8, "#8e44ad")]:
    pos, v, _ = xy(name)
    ax.plot(pos, smooth(v), color=color, lw=1.8, label=f"{int(cut*100)}% 处冻结")
    ax.axvline(cut, color=color, ls=":", lw=1)
ax.set_title("画面冻结后,v 仍沿时序惯性下滑", fontsize=11)
ax.set_xlabel("相对播放位置")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# Panel 3: rewind
ax = axes[0][2]
pos, v, ts = xy("rewind")
ax.plot(pos, v, color="#7f8c8d", lw=1.0, alpha=0.5)
ax.plot(pos, smooth(v), color="#c0392b", lw=1.8, label="预测 v")
ax.axvspan(0.6, 0.48, color="#f5b7b1", alpha=0.6)
ax.annotate("回退腿 (+5.3s)", xy=(0.545, 11.5), fontsize=9.5, color="#c0392b", ha="center", weight="bold")
true_rem = (T / FPS) - ts
ax.plot(pos, true_rem, "k--", lw=1.2, label="真实剩余时间")
ax.set_title("60%→30% 回退:v 上跳但幅度被压缩", fontsize=11)
ax.set_xlabel("相对播放位置(构造视频)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.25)

# Panel 4: loopdense — the star panel
ax = axes[1][0]
c = conds["loopdense"]
pos = np.array(c["eval_positions"])
v = np.array(c["values"])
idx = np.array(c["true_last_frame_idx"])
ax.plot(pos, v, color="#7f8c8d", lw=0.9, alpha=0.55)
ax.plot(pos, smooth(v, 5), color="#16a085", lw=2.0)
for lo, hi, lab in [(170, 255, None), (255, 340, None)]:
    ax.axvspan(lo, hi, color="#abebc6", alpha=0.55)
ax.axvspan(85, 170, color="#d6eaf8", alpha=0.55)
for sp, vv, txt in [(183, 8.35, "+2.3s"), (268, 8.5, "+3.9s")]:
    ax.annotate(txt, xy=(sp, vv), fontsize=10, color="#c0392b", ha="center", weight="bold")
ax.text(127, 1.05, "第 1 遍", fontsize=9, ha="center", color="#2980b9")
ax.text(212, 1.05, "第 2 遍(重复)", fontsize=9, ha="center", color="#1e8449")
ax.text(297, 1.05, "第 3 遍(重复)", fontsize=9, ha="center", color="#1e8449")
ax.set_ylim(0, 9.2)
ax.set_title("中段 10% 循环 3 遍:每次重播 v 精准上跳", fontsize=11)
ax.set_xlabel("子视频帧位置(循环区加密采样)")
ax.set_ylabel("预测剩余时间 v(秒)")
ax.grid(alpha=0.25)

# Panel 5: shuffle scatter
ax = axes[1][1]
c = conds["shuffle"]
ts = np.array(c["true_last_frame_ts"])
v = np.array(c["values"])
true_rem = (T / FPS) - ts
ax.scatter(true_rem, v, s=14, alpha=0.65, color="#2980b9")
from scipy.stats import spearmanr
rho_true = spearmanr(true_rem, v).statistic
rho_pos = spearmanr(np.array(c["eval_positions"]), v).statistic
ax.set_title(f"随机打乱:v 跟踪内容而非位置\nρ(v,末帧真实剩余)={rho_true:.2f}  ρ(v,位置)={rho_pos:.2f}",
             fontsize=11)
ax.set_xlabel("末帧的真实剩余时间(秒)")
ax.set_ylabel("预测 v(秒)")
ax.grid(alpha=0.25)

# Panel 6: endpoint bars
ax = axes[1][2]
names = ["forward", "reversed", "frozen", "frozen80", "rewind", "truncate", "loop"]
labels = ["正向", "倒放", "40%冻结", "80%冻结", "回退", "截断90%", "循环"]
v0 = [conds[n]["values"][0] for n in names]
v1 = [conds[n]["values"][-1] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, v0, 0.38, color="#85c1e9", label="起点 v")
ax.bar(x + 0.2, v1, 0.38, color="#e59866", label="终点 v")
ax.axhline(0.38, color="#2980b9", ls="--", lw=1, alpha=0.7)
ax.text(2.1, 0.62, "正向终点=0.38", fontsize=8.5, color="#2980b9")
ax.set_xticks(x, labels, fontsize=9, rotation=20)
ax.set_title("各条件端点取值:近完成/回退可分辨", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")

fig.suptitle("RynnValue-4B 捷径压力测试:同一段 28.5s 示范视频,八种受控扰动", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("drafts/rynnvalue-probe/report_analysis.png", dpi=170)
print("saved; loop bumps:",
      v[list(pos).index(183)] if 183 in pos else "n/a")
