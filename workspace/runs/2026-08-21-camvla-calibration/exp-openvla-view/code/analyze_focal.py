"""实验 E 分析:焦距扰动成功率曲线。
输入: focal_results_A.json / focal_results_B.json (+ 可选 PIL 锚点 JSON)
输出: focal-summary.json / focal-success-vs-focal.png(横轴等效焦距偏差 %,分任务两条线 + crop 对照点)
"""
import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
F0 = 45.0  # agentview 默认 fovy(deg)

def focal_dev(fovy_pct, crop_pct):
    """等效焦距偏差 %:f_eff = h/(2 tan(fovy/2));fovy_pct 变化 → tan 反比。
    crop_pct:center-crop c% 再缩放回原尺寸 → f_eff × 1/(1-c/100)。"""
    if crop_pct > 0:
        return 100.0 * (1.0 / (1.0 - crop_pct / 100.0) - 1.0)
    f1 = F0 * (1.0 + fovy_pct / 100.0)
    return 100.0 * (math.tan(math.radians(F0) / 2) / math.tan(math.radians(f1) / 2) - 1.0)

rows = []
for f in sys.argv[2:]:
    with open(f) as fh:
        rows.extend(json.load(fh))

# 按 (task, fovy_pct, crop_pct) 聚合
agg = {}
for r in rows:
    key = (r["task"], r.get("fovy_pct", 0.0), r.get("crop_pct", 0.0))
    agg.setdefault(key, []).append(bool(r["success"]))

summary = {"conditions": []}
for (task, fovy, crop), vals in sorted(agg.items()):
    dev = focal_dev(fovy, crop)
    summary["conditions"].append({
        "task": task, "fovy_pct": fovy, "crop_pct": crop, "focal_dev_pct": round(dev, 3),
        "success_rate": float(np.mean(vals)), "successes": int(sum(vals)), "n": len(vals),
    })
with open(os.path.join(OUT, "focal-summary.json"), "w") as f:
    json.dump(summary, f, indent=1, ensure_ascii=False)

print(f"{'task':>4} {'fovy%':>7} {'crop%':>6} {'focal_dev%':>10} {'SR':>6} {'n':>4}")
for c in summary["conditions"]:
    print(f"{c['task']:>4} {c['fovy_pct']:>7} {c['crop_pct']:>6} {c['focal_dev_pct']:>10} "
          f"{c['success_rate']:>6.2f} {c['n']:>4}")

# ---- 图:横轴等效焦距偏差 %,分任务两条线 + crop 对照点 ----
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for task in sorted({r["task"] for r in rows}):
    xs, ys, ns = [], [], []
    crop_pts = []
    for c in summary["conditions"]:
        if c["task"] != task:
            continue
        if c["crop_pct"] > 0:
            crop_pts.append((c["focal_dev_pct"], c["success_rate"], c["n"]))
        else:
            xs.append(c["focal_dev_pct"])
            ys.append(c["success_rate"])
            ns.append(c["n"])
    order = np.argsort(xs)
    xs = np.array(xs)[order]; ys = np.array(ys)[order]
    ax.plot(xs, ys, "-o", label=f"task{task} fovy")
    for (cx, cy, cn) in crop_pts:
        ax.plot(cx, cy, "s", ms=10, label=f"task{task} crop5% (n={cn})")
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("equivalent focal-length deviation (%)")
ax.set_ylabel("success rate")
ax.set_title("OpenVLA-7B: success rate vs focal-length perturbation (LIBERO-Spatial)")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "focal-success-vs-focal.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved focal-success-vs-focal.png")
print("FOCAL_ANALYSIS_DONE")
