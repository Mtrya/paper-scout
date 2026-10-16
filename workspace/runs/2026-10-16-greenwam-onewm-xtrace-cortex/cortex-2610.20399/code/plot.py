"""Figures for the prospective-query probe (reads artifacts/*.json)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

ARMS = ["R-kl", "P-kl", "M-kl", "R-ce", "P-ce"]
NICE = {"R-kl": "R-KL\n(restate)", "P-kl": "P-KL\n(prospective)", "M-kl": "M-KL\n(mixed)",
        "R-ce": "R-CE\n(restate)", "P-ce": "P-CE\n(prospective)"}
COL = {"R-kl": "#4c72b0", "P-kl": "#dd8452", "M-kl": "#55a868", "R-ce": "#8c8c8c", "P-ce": "#c44e52"}


def main():
    s = json.load(open(ART / "summary.json"))
    arms = s["arms"]
    diag = json.load(open(ART / "diag_cross.json"))
    try:
        labels = json.load(open(ART / "rehearsal_labels.json"))
    except FileNotFoundError:
        labels = None
    tbl = json.load(open(ART / "analysis_table.json"))

    present = [a for a in ARMS if a in arms]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))

    # (a) consolidation rate on held-out items, free readout
    ax = axes[0]
    xs = np.arange(len(present))
    w = 0.38
    for k, kind in enumerate(["use", "restate"]):
        vals = [100 * arms[a][f"{kind}/free_CR"] for a in present]
        ax.bar(xs + (k - 0.5) * w, vals, w,
               color=["#e8b48b", "#7f9dc9"][k],
               edgecolor="black", linewidth=0.6,
               label=f"held-out {'application' if kind=='use' else 'recall'} questions")
    ax.axhline(0, color="k", lw=0.8)
    # bootstrap CI of the M - R difference on the use set, drawn at the right
    d = json.load(open(ART / "bootstrap.json"))["M-R"]
    xb = len(present) - 0.5
    base_m = 100 * arms["R-kl"]["use/free_CR"]
    ax.errorbar([xb], [base_m + 100 * d["mean"]], yerr=[[100 * (d["mean"] - d["lo"])], [100 * (d["hi"] - d["mean"])]],
                fmt="o", ms=4, color="#333", ecolor="#333", capsize=3, lw=1.2,
                label="M − R on application questions\n+10pp, 95% CI [−2, +25]")
    for x, a in zip(xs, present):
        for k, kind in enumerate(["use", "restate"]):
            v = 100 * arms[a][f"{kind}/free_CR"]
            ax.text(x + (k - 0.5) * w, v + 1.0, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs, [NICE[a] for a in present], fontsize=8)
    ax.set_ylabel("consolidation rate CR (%)")
    ax.set_ylim(0, 62)
    ax.set_title("(a) recovery of the context gap on held-out\nquestions (3 seeds; free-form readout)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # (b) cross-transfer diagnostic
    ax = axes[1]
    names = ["R-trained\non R qs", "R-trained\non P qs", "P-trained\non P qs", "P-trained\non R qs"]
    vals = [
        100 * diag["R/recovery_on_R"], 100 * diag["R/recovery_on_P"],
        100 * diag["P/recovery_on_P"], 100 * diag["P/recovery_on_R"],
    ]
    cols = ["#4c72b0", "#9db4d6", "#dd8452", "#f0c4a4"]
    ax.bar(names, vals, 0.6, color=cols, edgecolor="black", linewidth=0.6)
    ax.axhline(100, color="k", ls=":", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("recovery of teacher log-prob (%)")
    ax.set_title("(b) fit to the trained questions is complete,\ntransfer across question kinds is partial", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # (c) retention
    ax = axes[2]
    base = 100 * s["reference"]["mmlu"]
    vals = [100 * arms[a]["RR"] * base / 100 for a in present]
    ax.bar(xs, vals, 0.6, color=[COL[a] for a in present], edgecolor="black", linewidth=0.6)
    ax.axhline(base, color="k", ls="--", lw=1.2, label=f"base MMLU {base:.1f}%")
    for x, a in zip(xs, present):
        ax.text(x, vals[xs.tolist().index(x)] + 0.4, f"RR {100*arms[a]['RR']:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(xs, [NICE[a] for a in present], fontsize=8)
    ax.set_ylabel("MMLU accuracy (%)")
    ax.set_ylim(0, max(vals + [base]) * 1.3)
    ax.set_title("(c) non-target capability preserved", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "probe_summary.png", dpi=170)
    print("wrote", OUT / "probe_summary.png")


if __name__ == "__main__":
    main()
