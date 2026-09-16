#!/usr/bin/env python3
"""绳索族两张图:
figA_rope.png      上排四关场景渲染 + 下排操作扫描成功率(astra vs oracle)
figB_perception.png 感知探针双通道准确率(Q1 操作化/Q2/Q3 × 数值/图像)
用法: uv run --with matplotlib --with pillow python plot_rope.py
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    if os.path.exists(f):
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(OUT, "..", "..", "code"))
PROBE = os.path.join(CODE, "rope-probe")

LEVEL_CN = {"L0": "L0 刚杆摆位", "L1": "L1 链拉直到目标线",
            "L2": "L2 绳重塑为目标曲线", "L3": "L3 绳绕柱≥0.85圈", "L3X": "L3X 同L3但屏蔽环绕数"}

def succ(level):
    rs = glob.glob(f"{PROBE}/runs/{level}/t*/result.json")
    if not rs: return None, 0
    return sum(1 for f in rs if (json.load(open(f)).get("score") or {}).get("success")), len(rs)

oracle_res = json.load(open(f"{PROBE}/oracle/results.json"))
def osucc(level):
    v = [r.get("success") for k, r in oracle_res.items() if k.startswith(level + "/")]
    return (sum(1 for x in v if x), len(v)) if v else (None, 0)

# ---------------- figA ----------------
fig = plt.figure(figsize=(11, 5.4))
for i, L in enumerate(["L0", "L1", "L2", "L3"]):
    ax = fig.add_subplot(2, 4, i + 1)
    ax.imshow(mpimg.imread(f"/tmp/scene_{L}.png"))
    ax.set_title(LEVEL_CN[L], fontsize=10)
    ax.axis("off")
ax = fig.add_subplot(2, 1, 2)
levels = ["L0", "L1", "L2", "L3", "L3X"]
x = range(len(levels)); w = .32
av = [(succ(L)[0] or 0) / 5 * 100 if succ(L)[1] else 0 for L in levels]
ov = [(osucc(L)[0] or 0) / max(osucc(L)[1], 1) * 100 if osucc(L)[1] else 0 for L in levels]
ax.bar([i - w / 2 for i in x], ov, width=w, color="#b8c4d8", label="oracle 先知基线")
ax.bar([i + w / 2 for i in x], av, width=w, color="#d2683c", label="gpt-6-astra")
for i, L in enumerate(levels):
    o = osucc(L); a = succ(L)
    if o[1]: ax.text(i - w / 2, ov[i] + 2, f"{o[0]}/{o[1]}", ha="center", fontsize=9, color="#556")
    if a[1]: ax.text(i + w / 2, av[i] + 2, f"{a[0]}/{a[1]}", ha="center", fontsize=9, color="#a40")
ax.set_xticks(list(x)); ax.set_xticklabels([LEVEL_CN[L] for L in levels], fontsize=9)
ax.set_ylabel("成功率 (%)"); ax.set_ylim(0, 118)
ax.set_title("操作扫描:oracle 全关全过;astra 的失败全部是判据可观测性问题,不是塑形能力")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figA_rope.png"), dpi=170)

# ---------------- figB ----------------
GT = json.load(open(f"{PROBE}/perception/cases.json"))["cases"]
def truth(c, q):
    g = c["ground_truth"]
    if q == "q1": return "yes" if abs(g["winding_about_peg"]) >= g["wound_threshold"] else "no"
    if q == "q2": return g["self_crossings"]
    return "E0" if g["end_dist_to_peg"]["E0"] <= g["end_dist_to_peg"]["E1"] else "E1"

def acc(channel, q, variant=""):
    got = tot = 0
    for c in GT:
        p = f"{PROBE}/perception/runs/{c['case']}/{channel}{variant}/result.json"
        if not os.path.exists(p): continue
        a = json.load(open(p)).get("answer") or {}
        t = truth(c, q); tot += 1
        if q == "q1": got += str(a.get("q1", "")).strip().lower() == t
        elif q == "q2":
            try: got += int(a.get("q2")) == t
            except Exception: pass
        else: got += {"a": "E0", "b": "E1"}.get(str(a.get("q3", "")).strip().lower()) == t
    return got, tot

fig, ax = plt.subplots(figsize=(6.4, 3.4))
groups = [("Q1 绕柱否\n(操作化判据)", "q1", "_q1op"), ("Q2 自交叉数", "q2", ""), ("Q3 哪端更近", "q3", "")]
x = range(len(groups)); w = .32
for off, ch, color, lab in [(-w/2, "numeric", "#4a7fb5", "数值 state 通道"), (w/2, "image", "#d2683c", "渲染图通道")]:
    vals, labels = [], []
    for _, q, var in groups:
        if ch == "image" and q == "q3":
            vals.append(0); labels.append("图上不可判"); continue
        g, t = acc(ch, q, var)
        vals.append(g / max(t, 1) * 100); labels.append(f"{g}/{t}")
    ax.bar([i + off for i in x], vals, width=w, color=color, label=lab)
    for i, (v, lb) in enumerate(zip(vals, labels)):
        ax.text(i + off, v + 3, lb, ha="center", fontsize=8.5,
                color="#333" if lb.endswith("0") and "/" in lb else "#888")
ax.set_xticks(list(x)); ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
ax.set_ylabel("准确率 (%)"); ax.set_ylim(0, 118)
ax.set_title("感知探针:给精确判据后,拓扑性质(环绕)双通道都近乎可读;\n失误集中在 |环绕|=0.500 的刀锋 case 与最密线卷")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figB_perception.png"), dpi=170)
print("PLOTS_DONE")
