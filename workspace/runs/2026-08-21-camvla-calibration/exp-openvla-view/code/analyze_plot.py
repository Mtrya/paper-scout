"""OpenVLA×LIBERO 视角扰动结果分析+绘图。
输入: results/viewprobe_results.json(逐集记录)+ cam_info.json
输出: results/summary_table.json + results/success_vs_theta.png(dpi150,裁白边)
"""
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/viewprobe_results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUT, exist_ok=True)

with open(RESULTS) as f:
    rows = json.load(f)

tasks = sorted({r["task"] for r in rows})
thetas = sorted({r["theta_deg"] for r in rows})

# ---- 汇总:每个 (task, theta, mode) 的成功率 ----
agg = {}
for r in rows:
    key = (r["task"], r["theta_deg"], r["mode"], r["rescue_sign"])
    agg.setdefault(key, []).append(bool(r["success"]))

summary = {"tasks": {}, "conditions": []}
for (task, theta, mode, sign), vals in sorted(agg.items()):
    sr = float(np.mean(vals)) if vals else None
    n = len(vals)
    summary["conditions"].append({
        "task": task, "theta_deg": theta, "mode": mode, "rescue_sign": sign,
        "success_rate": sr, "successes": int(sum(vals)), "n": n,
    })
    summary["tasks"].setdefault(str(task), {}).setdefault(theta, {})[mode] = {
        "success_rate": sr, "successes": int(sum(vals)), "n": n}

with open(os.path.join(OUT, "summary_table.json"), "w") as f:
    json.dump(summary, f, indent=1, ensure_ascii=False)

# ---- 表格打印 ----
print(f"{'task':>4} {'theta':>6} {'mode':>7} {'sign':>4} {'sr':>7} {'n':>4}")
for c in summary["conditions"]:
    print(f"{c['task']:>4} {c['theta_deg']:>6} {c['mode']:>7} {c['rescue_sign']:>4} "
          f"{(c['success_rate'] if c['success_rate'] is not None else -1):>7.3f} {c['n']:>4}")

# ---- 图:success rate vs theta,raw/rescue 两条线(任务合并或分开)----
if os.path.exists(os.path.join(OUT, "cam_info.json")):
    with open(os.path.join(OUT, "cam_info.json")) as f:
        cam = json.load(f)
else:
    cam = None

for per_task in [False, True]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if not per_task:
        groups = {"all": [t for t in tasks]}
    else:
        groups = {f"task{t}": [t] for t in tasks}
    for gname, tl in groups.items():
        for mode in ["raw", "rescue"]:
            xs, ys, ns = [], [], []
            for th in sorted(thetas):
                if th == 0 and mode != "baseline":
                    continue
                m = mode if th != 0 else "baseline"
                vals = []
                for t in tl:
                    key = (t, th, m, 1.0)
                    if key in agg:
                        vals.extend(agg[key])
                    elif mode == "rescue":
                        key = (t, th, m, -1.0)
                        if key in agg:
                            vals.extend(agg[key])
                if vals:
                    xs.append(th)
                    ys.append(np.mean(vals))
                    ns.append(len(vals))
            lab = {"raw": "raw (no rescue)", "rescue": "rescue Rz(+θ)"}[mode]
            ax.plot(xs, ys, "-o", label=lab if gname == "all" or mode == "raw" else None)
            if per_task:
                pass
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("camera rotation θ (deg, about base z)")
    ax.set_ylabel("success rate")
    ax.set_title(("Success rate vs camera rotation (all tasks)" if not per_task
                  else "Success rate vs camera rotation (per task)"))
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fname = "success_vs_theta.png" if not per_task else "success_vs_theta_per_task.png"
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)

print("ANALYSIS_DONE")
