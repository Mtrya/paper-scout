"""Build compact report figures from the preserved QKV result JSON."""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BUCKETS = ["1-16", "17-32", "33-64", "65-128", "129-256", "257-512"]
X = np.array([8, 24, 48, 96, 192, 384])
COLORS = {"mean": "#6b7280", "vala": "#2563eb", "repair": "#059669",
          "raw1": "#f59e0b", "raw2": "#dc2626", "raw3": "#7f1d1d"}


def finite_log(values):
    return [math.log10(v) if math.isfinite(v) and v > 0 else np.nan for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", required=True)
    parser.add_argument("--assets", required=True)
    args = parser.parse_args()
    thread = Path(args.thread)
    assets = Path(args.assets)
    primary = json.loads((thread / "code/qkv_full.json").read_text())
    scales = json.loads((thread / "code/qkv_gdn_scales.json").read_text())
    repair = json.loads((thread / "code/qkv_gdn_repair.json").read_text())
    repair_heldout = json.loads((thread / "code/qkv_gdn_repair_heldout.json").read_text())
    preupdate = json.loads((thread / "code/qkv_ridge_preupdate.json").read_text())
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    ax = axes[0]
    raw = [("0.0", "κ = 0 (VaLA)", COLORS["vala"]),
           ("0.001953125", "κ = 1/512", COLORS["raw1"]),
           ("0.0078125", "κ = 1/128", COLORS["raw2"]),
           ("0.125", "κ = 1/8", COLORS["raw3"])]
    for key, label, color in raw:
        vals = [scales["aggregate"]["methods"][key][b]["nmse"] for b in BUCKETS]
        ax.plot(X, finite_log(vals), marker="o", ms=3, lw=1.5, label=label, color=color)
        for x, value in zip(X, vals):
            if not math.isfinite(value):
                ax.scatter([x], [105], marker="^", s=24, color=color)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-2, 110)
    ax.set_xscale("log", base=2)
    ax.set_xticks(X, [str(x) for x in X])
    ax.set_xlabel("context position (token)")
    ax.set_ylabel("log10 NMSE (overflow marked ▲)")
    ax.set_title("A  Raw centered-key recursion")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    for ax, metric, title, ylabel in [
            (axes[1], "nmse", "B  Spectral clamp: two text splits", "NMSE"),
            (axes[2], "cosine", "C  Direction is a separate cost", "cosine with exact attention")]:
        xpos = np.arange(3)
        for offset, data, label, color in [(-0.18, repair, "initial", "#60a5fa"),
                                           (0.18, repair_heldout, "confirmation", "#f59e0b")]:
            values = [data["aggregate"]["methods"][key]["all"][metric]
                      for key in ("mean", "vala", "spectral_clamp")]
            ax.bar(xpos + offset, values, width=0.36, label=label, color=color)
        ax.set_xticks(xpos, ["value mean", "VaLA", "spectral clamp"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=2)
    fig.savefig(assets / "attention-gdn-stability.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    methods = primary["aggregate"]["ridge"]["0.001"]
    ax = axes[0]
    for key, label, color in [("attention_readout", "keep attention readout", COLORS["repair"]),
                              ("query_readout", "Wq readout", COLORS["raw2"]),
                              ("projected_query_readout", "best per-token q projection", COLORS["raw1"])]:
        ax.plot(X, [methods[key][b]["nmse"] for b in BUCKETS], marker="o", ms=3,
                lw=1.8, label=label, color=color)
    ax.plot(X, [preupdate["aggregate"]["pre_attention"][b]["nmse"] for b in BUCKETS],
            color=COLORS["repair"], ls="--", lw=1.4, label="keep attention, pre-update")
    ax.plot(X, [primary["aggregate"]["baselines"]["mean"][b]["nmse"] for b in BUCKETS],
            color=COLORS["mean"], ls="--", lw=1.4, label="value mean")
    ax.set_xscale("log", base=2)
    ax.set_xticks(X, [str(x) for x in X])
    ax.set_xlabel("context position (token)")
    ax.set_ylabel("NMSE")
    ax.set_title("A  Ridge writing vs TTT reading (ρ = 0.001)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    geometry = primary["aggregate"]["readout_geometry"]
    ax.plot(X, [geometry[b]["cosine"] for b in BUCKETS], marker="o", ms=3,
            lw=1.8, color=COLORS["vala"], label="cos(r, q)")
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("cos(r, q)", color=COLORS["vala"])
    ax2 = ax.twinx()
    ax2.plot(X, [geometry[b]["projection_residual"] for b in BUCKETS], marker="s", ms=3,
             lw=1.6, color=COLORS["raw2"], label="projection residual")
    ax2.set_ylim(0.9, 1.0)
    ax2.set_ylabel("||r − λq|| / ||r||", color=COLORS["raw2"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(X, [str(x) for x in X])
    ax.set_xlabel("context position (token)")
    ax.set_title("B  r = Σ aᵢkᵢ is not aligned with q")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="upper left")

    ax = axes[2]
    layer_ids = [str(x) for x in primary["config"]["layers"]]
    xpos = np.arange(len(layer_ids))
    width = 0.25
    for offset, key, label, color in [(-width, "attention_readout", "keep attention", COLORS["repair"]),
                                      (0, "query_readout", "Wq", COLORS["raw2"]),
                                      (width, "projected_query_readout", "projected q", COLORS["raw1"])]:
        values = []
        for layer in primary["layers"]:
            values.append(layer["ridge"]["0.001"][key]["overall"]["all"]["nmse"])
        ax.bar(xpos + offset, values, width=width, label=label, color=color)
    ax.set_xticks(xpos, [f"layer {x}" for x in layer_ids])
    ax.set_ylabel("NMSE over all positions")
    ax.set_title("C  The gap persists across depth")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=2)
    fig.savefig(assets / "attention-ttt-readout.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
