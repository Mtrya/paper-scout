#!/usr/bin/env python3
"""2026-09-17 巡航实验绘图:B(命名×约定)/C(速度+抖动拦截)/D(投掷非平稳)。
用法: uv run --with matplotlib python plot_all.py
"""
import json, glob, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    if os.path.exists(f):
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(OUT, "..", "..", "code"))

# ---------- 图 1:实验 B 条件成功率 vs Show-Harness ----------
sh = {"A\n语义+约定": 100, "B\n语义,无约定": 90, "C\n任意+约定": 95, "D\n任意,无约定": 5}
ours = {}
for cond in "ABCD":
    rs = glob.glob(f"{CODE}/showharness-probe/runs/{cond}/t*/result.json")
    ours[cond] = 100 * sum(json.load(open(f))["score"]["success"] for f in rs) / max(len(rs), 1)
rsE = glob.glob(f"{CODE}/showharness-probe/runs/E/t*/result.json")
oursE = 100 * sum(json.load(open(f))["score"]["success"] for f in rsE) / max(len(rsE), 1)

fig, ax = plt.subplots(figsize=(7.2, 3.4))
keys = list(sh.keys()) + ["D\n任意,无约定"]
x = range(5)
shv = [sh[k] for k in keys[:4]] + [0]   # E 条件论文未做,不画论文柱
ov = [ours[c] for c in "ABCD"] + [oursE]
b1 = ax.bar([i - .21 for i in x[:4]], shv[:4], width=.4, color="#b8c4d8", label="Show-Harness Gemini-3.1 Pro(论文,Fig.10a 约读)")
b2 = ax.bar([i + .21 for i in x], ov, width=.4, color="#d2683c", label="gpt-6-astra(本轮沙盒复测)")
for i, (a, b) in enumerate(zip(shv, ov)):
    if i < 4: ax.text(i - .21, a + 2, f"~{a:.0f}", ha="center", fontsize=8, color="#556")
    ax.text(i + .21, b + 2, f"{b:.0f}", ha="center", fontsize=8, color="#a40")
ax.text(4 - .21, 4, "论文\n未做", ha="center", fontsize=7, color="#99a")
ax.set_xticks(list(x)); ax.set_xticklabels(["A 语义名+约定", "B 仅语义名", "C 任意符号+约定", "D 任意符号", "E 任意符号+仅图像"], fontsize=8.5)
ax.set_ylabel("成功率 (%)"); ax.set_ylim(0, 118)
ax.set_title("动作语义从交互中推断:astra 全部破解(5 条件 × 5 回合)")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_naming.png"), dpi=170); plt.close(fig)

# 每条件调用数
calls = {}
for cond in "ABCDE":
    rs = sorted(glob.glob(f"{CODE}/showharness-probe/runs/{cond}/t*/result.json"))
    calls[cond] = [json.load(open(f))["bridge_calls"] for f in rs]
print("B calls:", calls)

# ---------- 图 2:实验 C 拦截 速度×抖动 ----------
def succ(pattern):
    rs = glob.glob(pattern)
    if not rs: return None, 0
    return sum(json.load(open(f))["score"]["success"] for f in rs), len(rs)

speeds = [1, 2, 4, 8, 16]
astra_v = [succ(f"{CODE}/intercept-probe/runs/v{i}/t*/result.json")[0] for i in range(5)]
orac_v = [1 if i == 0 else None for i in range(5)]
orc = {0: 5, 1: 5, 2: 1, 3: 0, 4: 0}  # oracle 速度扫描(贪心)
# 理论可捕数
theory = {0: 5, 1: 5, 2: 5, 3: 3, 4: 1}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
ax = axes[0]
w = .26
ax.bar([i - w for i in range(5)], [v / 5 * 100 for v in astra_v], width=w, color="#d2683c", label="astra")
ax.bar([i for i in range(5)], [orc[i] / 5 * 100 for i in range(5)], width=w, color="#b8c4d8", label="贪心 oracle")
ax.bar([i + w for i in range(5)], [theory[i] / 5 * 100 for i in range(5)], width=w, color="#7fbf7f", label="全知可捕(格点+半径)")
for i in range(5):
    ax.text(i - w, astra_v[i] / 5 * 100 + 3, str(astra_v[i]), ha="center", fontsize=8, color="#a40")
    ax.text(i, orc[i] / 5 * 100 + 3, str(orc[i]), ha="center", fontsize=8, color="#556")
    ax.text(i + w, theory[i] / 5 * 100 + 3, str(theory[i]), ha="center", fontsize=8, color="#373")
ax.set_xticks(range(5)); ax.set_xticklabels([f"{s}cm/s" for s in speeds], fontsize=9)
ax.set_ylabel("5 回合中捕获数换算 (%)"); ax.set_ylim(0, 118)
ax.set_title("匀速目标:决策与物理边界逐回合吻合(25 局仅 1 漏判)")
ax.legend(fontsize=8)

jts = [5, 15, 30, 60]
astra_j = [succ(f"{CODE}/intercept-probe/runs/j{j}/t*/result.json")[0] for j in jts]
orac_j = {5: 1, 15: 2, 30: 0, 60: 0}       # 贪心 oracle
clair_j = {5: 5, 15: 4, 30: 4, 60: 1}      # 全知可行性上界(analyze_clairvoyant.py)
ax = axes[1]
ax.plot([0] + jts, [5] + [clair_j[j] for j in jts], ":", color="#7fbf7f", label="全知可捕上界", lw=2)
ax.plot([0] + jts, [5] + astra_j, "o-", color="#d2683c", label="astra", lw=2)
ax.plot([0] + jts, [1] + [orac_j[j] for j in jts], "s--", color="#8a97b8", label="贪心 oracle")
for xx, yy in zip([0] + jts, [5] + astra_j):
    ax.text(xx, yy + .18, str(yy), ha="center", fontsize=9, color="#a40")
for xx, yy in zip([0] + jts, [5] + [clair_j[j] for j in jts]):
    ax.text(xx, yy - .38, str(yy), ha="center", fontsize=8, color="#373")
ax.set_xlabel("航向抖动 σ (度/0.5s)"); ax.set_ylabel("5 回合中捕获数")
ax.set_title("不可预测目标(4cm/s):小抖动成陷阱区")
ax.set_ylim(-0.4, 5.6); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figC_intercept.png"), dpi=170); plt.close(fig)
print("C astra_v:", astra_v, "astra_j:", astra_j)

# ---------- 图 3:实验 D 投掷非平稳 ----------
modes = ["d0", "d1", "d2", "d3", "d4"]
att = {}
for m in modes:
    rs = sorted(glob.glob(f"{CODE}/throw-probe/runs/{m}/t*/result.json"))
    att[m] = [json.load(open(f))["score"]["attempts"] for f in rs]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
bp = ax.boxplot([att[m] for m in modes], tick_labels=["σ=0\n(静态)", "0.5", "1.0", "2.0", "4.0"],
                patch_artist=True, widths=.5)
for p in bp["boxes"]: p.set_facecolor("#e8b39a")
ax.set_xlabel("风漂移幅度 σ (m/s² 每投)"); ax.set_ylabel("命中所需投掷数")
ax.set_title("非平稳系统辨识:5/5 全部命中,代价随漂移上升(15 投预算)")
ax.set_ylim(0, 16)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figD_throw.png"), dpi=170); plt.close(fig)
print("D attempts:", att)
print("PLOTS_DONE")
