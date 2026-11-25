# Generate the three probe figures for the 2026-11-20 fast-slow cruise report.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK HK",
    "Noto Sans CJK JP", "Droid Sans Fallback",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

BLUE, ORANGE, GREEN, RED, GRAY = "#3b6fb6", "#e2813a", "#4a9a5f", "#c34f4f", "#8a8f98"

with open("code/rea-probe/results_a.json") as f:
    RA = json.load(f)
with open("code/rea-probe/results_b.json") as f:
    RB = json.load(f)

# ---------- Figure 1: probe A, gates + judge noise ----------
gates = [
    ("never-fast", "单快策略"), ("never-slow", "单慢策略"), ("exception", "异常门"),
    ("always-same", "盲重试"), ("always", "盲升级"), ("judge", "判官门"),
]
base = {g: next(r for r in RA if r["gate"] == g and r["alpha"] == 0 and r["beta"] == 0) for g, _ in gates}
labels = [zh for _, zh in gates]
succ = [base[g]["success"] for g, _ in gates]
steps = [base[g]["stepsPerEp"] for g, _ in gates]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
ax = axes[0]
colors = [GRAY, GREEN, RED, RED, ORANGE, BLUE]
bars = ax.bar(range(6), succ, color=colors, width=0.62)
for i, (b, s) in enumerate(zip(bars, steps)):
    ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{s:g}步", ha="center",
            va="bottom", fontsize=8.5, color="white", fontweight="bold")
ax.set_xticks(range(6))
ax.set_xticklabels(labels, fontsize=8.8)
ax.set_ylim(0, 1.12)
ax.set_ylabel("成功率")
ax.set_title("(a) 四种门控 vs 判官门（400 seeds/格）", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
alphas = [0, 0.05, 0.1, 0.2, 0.3]
emp = [next(r["success"] for r in RA if r["gate"] == "judge" and r["alpha"] == a and r["beta"] == 0) for a in alphas]
ax.plot(alphas, emp, "o-", color=BLUE, lw=1.8, ms=5, label="判官门实测")
xs = [0, 0.3]
ax.plot(xs, [1 - 0.5 * a for a in xs], "--", color=GRAY, lw=1.2, label=r"$1-p_{jam}\alpha$ 理论线")
ax.set_xlabel(r"假 done 率 $\alpha$（判官谎报完成）")
ax.set_ylabel("成功率")
ax.set_ylim(0.78, 1.03)
ax.set_title("(b) 假 done 线性没收成功率", fontsize=10.5)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
betas = [0, 0.2, 0.5]
stp = [next(r["stepsPerEp"] for r in RA if r["gate"] == "judge" and r["alpha"] == 0 and r["beta"] == b) for b in betas]
bars = ax.bar([str(b) for b in betas], stp, color=BLUE, width=0.5)
for b, s in zip(bars, stp):
    ax.text(b.get_x() + b.get_width() / 2, s + 0.12, f"{s:g}", ha="center", fontsize=9)
ax.set_xlabel(r"假 not-done 率 $\beta$（判官谎报未完成）")
ax.set_ylabel("平均步数/集")
ax.set_ylim(0, 11.5)
ax.set_title("(c) 假 not-done 只耗步数（成功率恒 1.0）", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("drafts/rea_probe_gates.png", dpi=120)
plt.close(fig)

# ---------- Figure 2: probe B, async sweep ----------
age_cats = [0, 1, 3, 6, 12, 24]
brains = [100, 500, 1350, 3000, 6000]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
ax = axes[0]
cmap = [GREEN, "#7cbf8e", "#e0b23a", ORANGE, RED]
for c, bm in zip(cmap, brains):
    ys = []
    for a in age_cats:
        row = next(r for r in RB["sweep"] if r["brainMs"] == bm and r["maxAdviceAgeSteps"] == a)
        ys.append(row["success"])
    ax.plot(range(6), ys, "o-", color=c, lw=1.8, ms=4.5, label=f"大脑 {bm}ms")
ax.set_xticks(range(6))
ax.set_xticklabels([str(a) for a in age_cats])
ax.set_xlabel("建议年龄上限 maxAdviceAgeSteps（步）")
ax.set_ylabel("成功率")
ax.set_yticks([0, 1])
ax.set_yticklabels(["失败", "成功"])
ax.set_ylim(-0.12, 1.15)
ax.set_title("(a) 建议年龄上限必须 ≥ 大脑延迟", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
pairs = [("sync", "同步阻塞"), ("async", "异步（REA 式）")]
x = range(2)
w = 0.34
for off, (mode, zh), color in [(-w / 2, pairs[0], GRAY), (w / 2, pairs[1], BLUE)]:
    vals = [next(m["wallMs"] for m in RB["modes"] if m["mode"] == mode and m["brainMs"] == bm) / 1000 for bm in (1350, 3000)]
    bars = ax.bar([i + off for i in x], vals, width=w, color=color, label=zh)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}s", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(["大脑 1350ms", "大脑 3000ms"])
ax.set_ylabel("墙钟时间（秒）")
ax.set_title("(b) 同步 vs 异步墙钟（同一任务）", fontsize=10.5)
ax.legend(fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("drafts/rea_probe_async.png", dpi=120)
plt.close(fig)

# ---------- Figure 3: theory verification ----------
eq13 = [
    ("ρ=0.20\nfp=fn=0", 4.9957, 5.0),
    ("ρ=0.20\nfp=fn=0.05", 3.9631, 3.9583),
    ("ρ=0.10\nfp=fn=0.10", 5.8846, 5.8966),
    ("ρ=0.05\nfp=0.02,fn=0.3", 11.8073, 11.8644),
]
prop14 = [("ρ=0.2,r=0.01", 0.21, 0.21), ("ρ=0.2,r=0.10", 0.30, 0.30),
          ("ρ=0.05,r=0.02", 0.07, 0.07), ("ρ=0.5,r=0.05", 0.55, 0.55)]
regret = [("ε=0.05\n2ε<gap", 0.0, 0.0, 0.0), ("ε=0.09\n2ε<gap", 0.0, 0.0, 0.0),
          ("ε=0.11\n2ε≥gap", 0.009, 74.464, 44000.0), ("ε=0.20\n2ε≥gap", 0.8541, 12862.124, 80000.0)]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
ax = axes[0]
x = range(4)
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], [e for _, e, _ in eq13], width=w, color=BLUE, label="蒙特卡洛实测")
b2 = ax.bar([i + w / 2 for i in x], [t for _, _, t in eq13], width=w, color=ORANGE, label="Eq.13 公式")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.3g}", ha="center", fontsize=7.6)
ax.set_xticks(list(x))
ax.set_xticklabels([n for n, _, _ in eq13], fontsize=8)
ax.set_ylabel("IIE 增益倍数")
ax.set_ylim(0, 13.5)
ax.set_title("(a) Eq.13：门控收益公式逐点吻合", fontsize=10.5)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
b1 = ax.bar([i - w / 2 for i in x], [e for _, e, _ in prop14], width=w, color=BLUE, label="实测成本比")
b2 = ax.bar([i + w / 2 for i in x], [t for _, _, t in prop14], width=w, color=ORANGE, label=r"$\rho_u+r$ 公式")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.2g}", ha="center", fontsize=7.6)
ax.set_xticks(list(x))
ax.set_xticklabels([n for n, _, _ in prop14], fontsize=8)
ax.set_ylabel(r"筛选成本比 $C_{CI}/C_{react}$")
ax.set_ylim(0, 0.64)
ax.set_title("(b) Prop.14：筛选增益 = ρ+r", fontsize=10.5)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
xr = range(4)
vals = [v for _, v, _, _ in regret]
bars = ax.bar(xr, vals, color=[GREEN, GREEN, ORANGE, RED], width=0.55)
for i, (b, (_, v, emp, bound)) in enumerate(zip(bars, regret)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.1%}" if v else "0",
            ha="center", fontsize=8.6)
    if bound:
        ax.text(i, v + 0.075, f"遗憾 {emp:g}\n界 {bound:g}", ha="center", fontsize=7.2, color="black")
ax.set_xticks(list(xr))
ax.set_xticklabels([n for n, _, _, _ in regret], fontsize=8)
ax.set_ylabel("oracle 违例率")
ax.set_ylim(0, 1.0)
ax.set_title("(c) Thm.18：2ε 跨过 gap 才出现违例", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("drafts/sys12_verify.png", dpi=120)
plt.close(fig)
print("figures done")
