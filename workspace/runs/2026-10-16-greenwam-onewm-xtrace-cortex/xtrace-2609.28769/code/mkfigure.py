#!/usr/bin/env python3
"""Draw the packet's measurement figure from results/summary.json.

Two panels, both over the eight Triton cubins we spliced on the RTX 3060:

  (a) instruction preservation -- the text-level metric the paper reports, against the
      byte-level metric that also sees a control bit change; both are fractions of the
      original kernel's instructions
  (b) splice success -- how many of the requested trace sites the splicer could instrument
      with a feasible schedule

Usage: mkfigure.py [--json results/summary.json] [--out results/xtrace_sm86.png]
"""

import argparse
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHORT = {
    "awq_gemm_kernel": "awq_gemm",
    "awq_dequantize_kernel": "awq_dequant",
    "_forward_kernel": "fwd",
    "_backward_kernel": "bwd",
    "rotary_kernel": "rotary",
}


def label(row):
    spec = "mem" if row["sites"] == "mem" else "ev8"
    return (f"{SHORT.get(row['kernel'], row['kernel'])}\n"
            f"{row['origin_instructions']}i {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/summary.json")
    ap.add_argument("--out", default="results/xtrace_sm86.png")
    args = ap.parse_args()
    rows = json.load(open(args.json))
    rows = [r for r in rows if r.get("bytes")]
    rows.sort(key=lambda r: (r["kernel"], r["origin_instructions"]))
    labels = [label(r) for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=150)

    # (a) preservation
    ax = axes[0]
    text = [r["bytes"]["text_similarity"] * 100 for r in rows]
    byte = [r["bytes"]["byte_similarity"] * 100 for r in rows]
    ax.bar(x - 0.2, text, 0.4, label="text level (the paper's metric)", color="#3b6ea5")
    ax.bar(x + 0.2, byte, 0.4, label="byte level (control bits included)", color="#c46b2e")
    for xi, (t, b) in enumerate(zip(text, byte)):
        ax.text(xi - 0.2, t + 0.15, f"{t:.1f}", ha="center", fontsize=7)
        ax.text(xi + 0.2, b + 0.15, f"{b:.1f}", ha="center", fontsize=7)
    ax.axhline(94, color="grey", linestyle="--", linewidth=0.8)
    ax.text(-0.45, 94.15, "paper's own floor: 94%", ha="left", fontsize=7, color="grey")
    ax.set_ylim(84, 101.5)
    ax.set_ylabel("original instructions preserved (%)")
    ax.set_title("(a) what the spliced cubin kept", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    # (b) splice success
    ax = axes[1]
    ok = [r["sites_spliced"] / r["sites_requested"] * 100 for r in rows]
    ax.bar(x, ok, 0.55, color="#4c8c4a")
    for xi, r in enumerate(rows):
        ax.text(xi, ok[xi] + 1.5, f"{r['sites_spliced']}/{r['sites_requested']}", ha="center",
                fontsize=7)
    ax.set_ylim(0, 112)
    ax.set_ylabel("trace sites spliced (%)")
    ax.set_title("(b) how many requested sites the splicer could schedule", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Xtrace 0.0.35 (gwatch) splicing Triton kernels on sm_86 (RTX 3060): "
                 "binary preservation and splice success", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
