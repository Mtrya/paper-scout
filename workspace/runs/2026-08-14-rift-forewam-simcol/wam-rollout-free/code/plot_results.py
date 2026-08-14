"""Plot results of the RIFT/ForeWAM toy probe and the simulator-collapse repro.

Consumes results/eval_*.json, results/train_*.json, simcol_results/*.json.
Outputs PNGs for the report into <out>/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False})


def load(p):
    with open(p) as f:
        return json.load(f)


def plot_interventions(evalj, out_dir, tag):
    joint = evalj["joint"]
    order = ["mask", "noise", "frozenpresent", "swap", "shuffle", "finalclean", "original"]
    labels = ["掩码未来读", "噪声值", "冻结当前", "时间交换", "空间乱序", "终态缓存", "原版(rollout)"]
    srs = [joint[o]["sr"] * 100 for o in order]
    ades = [joint[o]["ade"] for o in order]
    x = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    colors = ["#c44", "#c44", "#c44", "#c44", "#c44", "#2a7", "#888"]
    a1.bar(x, srs, color=colors)
    a1.set_xticks(x); a1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    a1.set_ylabel("成功完成率 %"); a1.set_ylim(0, 105)
    a1.set_title("未来缓存干预 × 任务成功率(配对)")
    a2.bar(x, ades, color=colors)
    a2.set_xticks(x); a2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    a2.set_ylabel("EE-ADE(像素)")
    a2.set_title("末端轨迹漂移 vs 原版")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"interventions_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_producers(evalj, out_dir, tag):
    rows = []
    names = {"joint": "rollout(Joint 式)", "currentonly": "纯当前观测",
             "rift-l2": "RIFT-L2(anticipation)", "rift-fm": "RIFT-FM(anticipation)",
             "noiseslots": "ForeWAM 式(噪声槽 prefill)"}
    for k in ["joint", "currentonly", "rift-l2", "rift-fm", "noiseslots"]:
        if k not in evalj or "sr" not in evalj[k]:
            continue
        d = evalj[k]
        rows.append(dict(name=names.get(k, k), sr=d["sr"] * 100,
                         ade=d.get("ade_vs_rollout", d.get("ade")),
                         lat=d["latency_ms_per_chunk"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["sr"] for r in rows], width=0.4, label="成功率 %", color="#3a7")
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in rows], fontsize=8)
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [r["lat"] for r in rows], width=0.4, label="每块延迟 ms", color="#78a")
    ax2.set_ylabel("每块延迟 ms")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("单趟生产者 vs rollout:成功率与延迟(玩具尺度)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"producers_{tag}.png"), dpi=150)
    plt.close(fig)
    # also dump table
    with open(os.path.join(out_dir, f"producers_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def plot_planprobe(evalj, out_dir, tag):
    p = evalj.get("plan_probe", {})
    if not p:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(["到专家A(本模式)", "到专家B(注入模式)"],
           [p["step4_dist_to_expertA"], p["step4_dist_to_expertB"]],
           color=["#888", "#c44"])
    ax.set_ylabel("第 4 步 EE 距离(像素)")
    ax.set_title(f"计划注入:指令 A + 缓存 B(跟随注入比例 {p['frac_following_injected_plan']:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"planprobe_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_simcol(simdir, out_dir):
    rows = {}
    for v in ["single", "vs", "cot"]:
        p = os.path.join(simdir, f"{v}.json")
        if os.path.exists(p):
            rows[v] = load(p)
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    cols = {"single": "#c44", "vs": "#2a7", "cot": "#e90"}
    names = {"single": "单冻结模拟器", "vs": "口头化采样", "cot": "共同训练"}
    for v, d in rows.items():
        steps = [s["step"] for s in d["steps"]]
        axes[0].plot(steps, [s["train_reward"] for s in d["steps"]], color=cols[v],
                     label=names[v], marker="o", ms=3)
        axes[1].plot(steps, [s["distinct2"] for s in d["steps"]], color=cols[v],
                     label=names[v], marker="o", ms=3)
        axes[2].plot(steps, [s["self_bleu2"] for s in d["steps"]], color=cols[v],
                     label=names[v], marker="o", ms=3)
    axes[0].set_title("训练奖励(策略侧)"); axes[0].set_xlabel("REINFORCE 步")
    axes[1].set_title("distinct-2(策略多样性)"); axes[1].set_xlabel("REINFORCE 步")
    axes[2].set_title("self-BLEU-2(坍缩信号)"); axes[2].set_xlabel("REINFORCE 步")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "simcol_curves.png"), dpi=150)
    plt.close(fig)
    # panel table
    panel_rows = {}
    for v, d in rows.items():
        if d.get("panel"):
            last_k = sorted(d["panel"].keys(), key=int)[-1]
            panel_rows[names[v]] = d["panel"][last_k]
    if panel_rows:
        with open(os.path.join(out_dir, "simcol_panel.json"), "w") as f:
            json.dump(panel_rows, f, indent=2, ensure_ascii=False)
        print("simcol panel:", json.dumps(panel_rows))


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "plots"
    tag = sys.argv[2] if len(sys.argv) > 2 else "v1"
    os.makedirs(out_dir, exist_ok=True)
    evalp = f"results/eval_{tag}.json"
    if os.path.exists(evalp):
        ev = load(evalp)
        plot_interventions(ev, out_dir, tag)
        plot_producers(ev, out_dir, tag)
        plot_planprobe(ev, out_dir, tag)
        print("plots done from", evalp)
    plot_simcol("simcol_results", out_dir)
