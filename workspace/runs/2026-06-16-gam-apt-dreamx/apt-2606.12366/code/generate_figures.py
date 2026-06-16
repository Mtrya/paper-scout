#!/usr/bin/env python3
"""
Generate report-facing figures/tables for the APT deep-dive.

Outputs go to: runs/2026-06-16-gam-apt-dreamx/assets/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[4]
THREAD_CODE = Path(__file__).resolve().parent
ASSETS = ROOT / "runs" / "2026-06-16-gam-apt-dreamx" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

SUMMARY = THREAD_CODE / "inspect_apt_model_summary.json"


def fmt_m(n: int) -> str:
    return f"{n/1e6:.2f}M"


def fmt_k(n: int) -> str:
    return f"{n/1e3:.1f}K" if n < 1_000_000 else fmt_m(n)


def generate_param_table() -> Path:
    data = json.loads(SUMMARY.read_text())
    c0 = data["stage0_counts"]
    c1 = data["stage1_counts"]

    rows = [
        ("Total", c0["total"], c1["total"]),
        ("Context encoder", c0["context_encoder"], c1["context_encoder"]),
        ("Diffusion head", c0["diffusion_head"], c1["diffusion_head"]),
        ("  Traj-context attn", c0["  |- traj_context_attn"], c1["  |- traj_context_attn"]),
        ("    Attention blocks", c0["  |    |- attention_blocks"], c1["  |    |- attention_blocks"]),
        ("    Layer gates", c0["  |    |- layer_gates"], c1["  |    |- layer_gates"]),
        ("  Input encoders", c0["  |- input_encoders"], c1["  |- input_encoders"]),
        ("  Final head", c0["  |- final_head"], c1["  |- final_head"]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axis("off")
    ax.set_title("APT ActionExpert parameters: Stage 0 (VA prior) vs Stage 1 (VLA likelihood)",
                 fontsize=12, weight="bold", pad=12)

    table_data = [
        [label, fmt_m(v0), fmt_m(v1), f"+{v1-v0:,}\n({(v1-v0)/v0*100:.1f}%)" if v0 else "—"]
        for label, v0, v1 in rows
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Component", "Stage 0 params", "Stage 1 params", "Delta"],
        loc="center",
        cellLoc="left",
        colColours=["#e8e8e8"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5)

    # Highlight total row
    for j in range(4):
        table[(1, j)].set_facecolor("#d9edf7")
        table[(1, j)].set_text_props(weight="bold")

    out = ASSETS / "apt_param_table.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_architecture_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("APT two-stage action-expert architecture (simplified)",
                 fontsize=13, weight="bold", pad=16)

    def box(x, y, w, h, text, color, text_color="black", fontsize=10):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, weight="bold" if fontsize >= 10 else "normal")

    # Stage 0
    ax.text(2.5, 6.0, "Stage 0 — VA prior (train_stage=0)", ha="center", va="top",
            fontsize=12, weight="bold", color="#1a5276")
    box(0.8, 4.8, 1.8, 0.9, "Qwen3-VL\n(frozen)", "#aed6f1")
    box(3.2, 4.8, 1.8, 0.9, "VLM feature\nprojectors", "#d6eaf8")
    # Half layers
    for i in range(3):
        box(5.0, 5.2 - i*0.55, 1.6, 0.45, f"PRoPE layer {i}", "#f9e79f", fontsize=9)
    ax.text(5.8, 3.85, "N/2 layers\n(no language)", ha="center", va="top", fontsize=9, style="italic")
    ax.text(5.8, 3.5, "...", ha="center", va="top", fontsize=12)

    # Action tokens
    box(7.2, 4.8, 1.6, 0.9, "Action tokens\n(history + noisy)", "#f5b7b1")

    # Arrows stage 0
    ax.annotate("", xy=(3.2, 5.25), xytext=(2.6, 5.25),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(5.0, 5.25), xytext=(5.0, 5.7),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(7.2, 5.25), xytext=(6.6, 5.25),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    # Attention mask icon for stage 0
    ax.text(5.8, 2.9, "Mask: language tokens blocked\nfor vision & action queries",
            ha="center", va="top", fontsize=9, color="#922b21",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#922b21"))

    # Divider
    ax.axhline(y=2.4, color="#555", linestyle="--", linewidth=1.0)
    ax.text(10.5, 2.55, "load_from_pretrain()\nexpands N/2 → N", ha="right", va="bottom",
            fontsize=9, color="#1e8449", style="italic")

    # Stage 1
    ax.text(2.5, 2.2, "Stage 1 — VLA likelihood (train_stage=1)", ha="center", va="top",
            fontsize=12, weight="bold", color="#1a5276")
    box(0.8, 1.0, 1.8, 0.9, "Qwen3-VL\n(frozen/tuned)", "#aed6f1")
    box(3.2, 1.0, 1.8, 0.9, "VLM feature\nprojectors", "#d6eaf8")
    # Interleaved layers
    colors = ["#f9e79f", "#abebc6"] * 3
    labels = ["PRoPE\n(inherited)", "mRoPE\n(new)"] * 3
    for i in range(6):
        y = 1.7 - i*0.28
        box(5.0, y, 1.6, 0.26, labels[i], colors[i], fontsize=8)
        # gated fusion arrow from VLM highway
        ax.annotate("", xy=(5.0, y+0.13), xytext=(5.0, y+0.5),
                    arrowprops=dict(arrowstyle="->", color="#1e8449", lw=0.8,
                                    connectionstyle="arc3,rad=0"))
    ax.text(5.8, 0.05, "N interleaved layers\n(language + vision + action)",
            ha="center", va="bottom", fontsize=9, style="italic")

    # Gate symbol
    ax.text(4.55, 0.65, "σ(w)", ha="center", va="center",
            fontsize=10, color="#1e8449", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1e8449"))

    # Language + action inputs
    box(7.2, 1.35, 1.6, 0.55, "Language tokens", "#a9dfbf")
    box(7.2, 0.65, 1.6, 0.55, "Action tokens", "#f5b7b1")

    # Arrows stage 1
    ax.annotate("", xy=(3.2, 1.45), xytext=(2.6, 1.45),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(5.0, 1.7), xytext=(5.0, 1.9),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(7.2, 1.6), xytext=(6.6, 1.6),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(7.2, 0.9), xytext=(6.6, 0.9),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    # Legend
    legend_items = [
        mpatches.Patch(color="#f9e79f", label="Inherited PRoPE layer (VA prior)"),
        mpatches.Patch(color="#abebc6", label="Inserted mRoPE layer (language alignment)"),
        mpatches.Patch(color="#aed6f1", label="VLM backbone"),
        mpatches.Patch(color="#f5b7b1", label="Action input"),
        mpatches.Patch(color="#a9dfbf", label="Language input"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9, framealpha=0.95)

    out = ASSETS / "apt_architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    p1 = generate_param_table()
    p2 = generate_architecture_diagram()
    print(f"Generated:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
