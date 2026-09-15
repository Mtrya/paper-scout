#!/usr/bin/env python3
"""Make report plots from remote results. Run on remote (venv has matplotlib).
Usage: python make_plots.py <results-dir>
Produces: plot_a1_noise.png, plot_a2_curves.png, plot_b_tvd.png (+bar),
          plot_a3_bar.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(f"{D}/plots", exist_ok=True)
OUT = f"{D}/plots"


def plot_a1():
    p = f"{D}/a1_noise.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    agg = d["agg"]
    teachers = list(agg.keys())

    def pick_val(s, mode):
        if mode == "pooled":
            return ((s["correct"]["noisy"] + s["incorrect"]["noisy"])
                    / max(1, s["usable"]), s["usable"])
        return s[mode]["noisy"] / max(1, s[mode]["n"]), s[mode]["n"]

    short = {t: ("14B" if "14B" in t else ("4B" if "4B" in t else t))
             for t in teachers}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    panels = [("correct", "correct answers get negative adv", "#c44e52"),
              ("incorrect", "incorrect answers get positive adv", "#4c72b0"),
              ("pooled", "pooled noise rate", "#8172b3")]
    for ax_i, (mode, title, color) in enumerate(panels):
        xs, ys, ns = [], [], []
        for t in teachers:
            v, n = pick_val(agg[t], mode)
            xs.append(short[t]); ys.append(v * 100); ns.append(n)
        bars = axes[ax_i].bar(xs, ys, color=color, width=0.5)
        axes[ax_i].bar_label(bars, fmt="%.1f%%", fontsize=8)
        axes[ax_i].set_title(f"{title}(n={','.join(str(n) for n in ns)})",
                             fontsize=9)
        axes[ax_i].set_ylim(0, max(max(ys) * 1.3, 12))
    axes[2].axhline(30.6, ls="--", lw=1, color="gray")
    axes[2].text(0.03, 0.93, "paper, 4B teacher = 30.6%", fontsize=7, color="gray",
                 transform=axes[2].transAxes)
    fig.suptitle("A1 replication: sign noise of teacher supervision "
                 "(Qwen3-1.7B student, 367 usable trajectories)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/plot_a1_noise.png", dpi=150)
    plt.close(fig)


def plot_a2():
    conds = ["opd", "opd-oneshot", "fixed-neg", "opsa", "fixed-pos"]
    names = {"opd": "OPD (teacher)", "opd-oneshot": "OPD 1-query",
             "fixed-neg": "fixed -0.5", "opsa": "OPSA", "fixed-pos": "fixed +0.2"}

    def smooth(ys, w=5):
        out = []
        for i in range(len(ys)):
            lo = max(0, i - w + 1)
            out.append(sum(ys[lo:i + 1]) / (i - lo + 1))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.6))
    for c in conds:
        p = f"{D}/a2_{c}/log.json"
        if not os.path.exists(p):
            continue
        log = json.load(open(p))
        steps = [r["step"] for r in log]
        axes[0].plot(steps, smooth([r.get("mean_resp_len", 0) for r in log]),
                     label=names[c])
        axes[1].plot(steps, [r.get("mean_entropy", 0) for r in log], label=names[c])
        axes[2].plot(steps, [r.get("mean_abs_adv", 1e-6) for r in log], label=names[c])
    axes[0].axhline(2048, ls=":", lw=1, color="gray")
    axes[0].text(1, 2020, "generation cap (2048)", fontsize=7, color="gray")
    axes[0].set_ylim(900, 2120)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("mean response length (smoothed)")
    axes[0].set_title("response length (5-step moving average)")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("mean policy entropy (nats)")
    axes[1].set_title("entropy dynamics")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("mean |A| over all tokens (log)")
    axes[2].set_title("advantage magnitude")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/plot_a2_curves.png", dpi=150)
    plt.close(fig)


def plot_b():
    p = f"{D}/toywam/eval_report.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    acts = sorted(d["per_action"].keys(), key=lambda s: int(s))
    models = ["det", "vae", "zdiff", "oracle"]
    mnames = {"det": "det-MSE", "vae": "VAE-Gauss", "zdiff": "z-diffusion",
              "oracle": "oracle"}
    tvd = {m: [] for m in models}
    cov = {m: [] for m in models}
    for a in acts:
        for m in models:
            tvd[m].append(d["per_action"][a]["models"][m]["tvd"])
            cov[m].append(d["per_action"][a]["models"][m]["coverage"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(acts))
    w = 0.2
    for i, m in enumerate(models):
        axes[0].bar(x + (i - 1.5) * w, tvd[m], w, label=mnames[m])
        axes[1].bar(x + (i - 1.5) * w, [c * 100 for c in cov[m]], w,
                    label=mnames[m])
    axes[0].set_xticks(x); axes[0].set_xticklabels(acts)
    axes[0].set_xlabel("action (initial offset)")
    axes[0].set_ylabel("TVD vs reference")
    axes[0].set_title("outcome-distribution alignment")
    axes[1].set_xticks(x); axes[1].set_xticklabels(acts)
    axes[1].set_ylabel("valid-support coverage (%)")
    axes[1].set_title("outcome support recovery")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(f"{OUT}/plot_b_tvd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # also aggregate
    agg = {m: {"tvd": float(np.mean(tvd[m])), "cov": float(np.mean(cov[m]))}
           for m in models}
    with open(f"{D}/toywam/agg_summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


def plot_a3():
    res = {}
    for tag in ["a3_base", "a3_opsa"]:
        p = f"{D}/{tag}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            res[tag] = {"avg": d.get("avg4", 0) * 100, "pass": d.get("pass4", 0) * 100}
    if not res:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(2)
    w = 0.36
    avg = [res.get("a3_base", {}).get("avg", 0), res.get("a3_opsa", {}).get("avg", 0)]
    ps = [res.get("a3_base", {}).get("pass", 0), res.get("a3_opsa", {}).get("pass", 0)]
    b1 = ax.bar(x - w / 2, avg, w, label="avg@4 (ours)", color="#4c72b0")
    b2 = ax.bar(x + w / 2, ps, w, label="pass@4 (ours)", color="#dd8452")
    ax.bar_label(b1, fmt="%.1f", fontsize=8)
    ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Qwen3-1.7B base", "+ OPSA (released ckpt)"])
    ax.set_ylabel("AIME24 (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("paper (32 samples): avg@32 13.44 -> 48.85, pass@32 40.0 -> 80.0",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/plot_a3_bar.png", dpi=150)
    plt.close(fig)


plot_a1()
plot_a2()
plot_b()
plot_a3()
print("plots done in", OUT)
