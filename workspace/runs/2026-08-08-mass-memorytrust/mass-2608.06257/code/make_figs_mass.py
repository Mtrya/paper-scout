#!/usr/bin/env python3
"""巡航 2026-08-08 线程 A(MASS)报告图:从 results/*.json 生成 assets 图。"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

R = "results"  # 本脚本位于运行包 code/ 内,results/ 为同级目录
A = "../assets"  # 运行包 assets

ev_t = json.load(open(f"{R}/eval.json"))["typed"]
ev_d = json.load(open(f"{R}/eval_dense.json"))["dense"]
p1 = json.load(open(f"{R}/probe1_drift.json"))
p3 = json.load(open(f"{R}/probe3_attractor.json"))

Hs = [1, 8, 16, 32, 64, 128]

def curve(ev, key):
    return [ev[str(h)][key] for h in Hs]

# ---- 图 1:复现对照(position / semantic vs H,附论文参考点)----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
paper_typed = {1: 99.1, 32: 90.2, 128: 72.3}
paper_dense = {1: 2.7, 32: 0.1, 128: 0.0}

ax = axes[0]
ax.plot(Hs, curve(ev_t, "position"), "o-", color="#1f77b4", label="typed 复现(本巡航)")
ax.plot(Hs, curve(ev_d, "position"), "s-", color="#d62728", label="dense 复现(本巡航)")
ax.plot(list(paper_typed), list(paper_typed.values()), "o--", color="#1f77b4", alpha=0.45, label="typed 论文值")
ax.plot(list(paper_dense), list(paper_dense.values()), "s--", color="#d62728", alpha=0.45, label="dense 论文值")
ax.set_xscale("log", base=2)
ax.set_xlabel("自回归步数 H")
ax.set_ylabel("蛇头位置精度 (%)")
ax.set_title("位置保持:两种递归载体 vs 论文")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(Hs, curve(ev_t, "semantic"), "o-", color="#1f77b4", label="typed 复现")
ax.plot(Hs, curve(ev_d, "semantic"), "s-", color="#d62728", label="dense 复现")
ax.plot(Hs, curve(ev_t, "contradiction"), "o:", color="#2ca02c", label="typed 结构矛盾率")
ax.plot(Hs, curve(ev_d, "contradiction"), "s:", color="#9467bd", label="dense 结构矛盾率")
ax.set_xscale("log", base=2)
ax.set_xlabel("自回归步数 H")
ax.set_ylabel("百分比 (%)")
ax.set_title("聚合语义高位横盘,矛盾率分道扬镳")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{A}/mass-repro-compare.png", dpi=150)
plt.close(fig)

# ---- 图 2:探针 1 漂移解剖 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(p1["head_curve"], color="#1f77b4", label="蛇头位置与真值一致的比例")
ax.plot(p1.get("food_curve", [0] * len(p1["head_curve"])), color="#2ca02c", label="food 集合一致的比例")
ax.set_xlabel("tick")
ax.set_ylabel("比例")
ax.set_title("与真值的分歧:头位置 13 tick 内归零")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
legal = p1.get("legal_move_curve") or p1.get("legal_curve")
if legal is None:
    # 从 agent 的 png 可知曲线存在;尝试其他键名
    keys = [k for k in p1 if "legal" in k]
    legal = p1[keys[0]] if keys else None
if legal:
    ax.plot(legal, color="#8c564b", label="模型步进与引擎合法推进一步一致的比例")
ax.axhline(0.706, color="#8c564b", ls="--", alpha=0.5)
ax.text(len(p1["head_curve"]) * 0.35, 0.73, "均值 70.6%:分歧后仍走引擎合法的平行世界线", fontsize=9, color="#8c564b")
ax.set_xlabel("tick")
ax.set_ylabel("比例")
ax.set_title("分歧≠垃圾:世界仍合法,只是不再是真的")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{A}/mass-probe1-drift.png", dpi=150)
plt.close(fig)

# ---- 图 3:探针 3 吸引子 ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for ep, c in p3["unique_curve"].items():
    xs = [int(1024 / (len(c) - 1) * i) for i in range(len(c))]
    ax.plot(xs, c, marker="o", ms=3, label=f"episode {int(ep)+1}")
for i, t in enumerate(p3["cycle_entry_tick"]):
    ax.axvline(t, color="gray", ls=":", alpha=0.4)
ax.set_xlabel("tick(no-op 动作流,H=1024)")
ax.set_ylabel("累计唯一状态数")
ax.set_title("世界死亡:24–48 tick 进入吸引子,状态停止演化(存活=2,矛盾=0)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{A}/mass-probe3-attractor.png", dpi=150)
plt.close(fig)
print("done")
