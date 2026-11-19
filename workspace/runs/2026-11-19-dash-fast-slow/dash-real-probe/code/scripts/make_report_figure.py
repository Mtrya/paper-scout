"""Report version of the main figure (kept separate from the working figure emitted by analyze.py).

Panels: (a) dose-response of the four core arms, (b) channel ablations at delta=8, (c) slow-feature drift and pixel
change.  Everything is read from results/*.csv / drift.json, so the figure is rebuilt without retraining.

    uv run --with numpy --with matplotlib python code/scripts/make_report_figure.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

RES = Path(__file__).resolve().parent.parent / "results"
OUT = RES / "fig_dash_probe_report.png"

ARMS = [
    ("sync_nofresh", "同步训练 · 仅慢视觉", "#6f6f6f", "--", "o"),
    ("sync_fresh", "同步训练 · 加入当前视觉", "#1f6fb4", "-", "o"),
    ("desync8_nofresh", "错位训练 · 仅慢视觉", "#c46a00", "--", "s"),
    ("desync8_fresh", "错位训练 · 加入当前视觉", "#2b8a3e", "-", "s"),
]
CONDS = [("main", "原输入"), ("zero_vis", "慢视觉置零"), ("shuf_vis", "慢视觉换帧"), ("stalefresh", "新视觉换旧")]
DELTAS = [0, 1, 2, 4, 8]


def pick_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC", "Source Han Sans SC"):
        if cand in names:
            return cand
    return None


def report_overlaps(fig) -> None:
    """Flag text/text and text/artist collisions and any ink outside the canvas (the figure is checked by pixel or by
    eye downstream, but collisions are cheap to catch here)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:  # noqa: BLE001
            continue
        if bb.width > 0 and bb.height > 0:
            boxes.append((t, bb))
    bad = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            inter = matplotlib.transforms.Bbox.intersection(a, b)
            if inter is not None and inter.width * inter.height > 10:
                bad += 1
                print(f"[overlap] {boxes[i][0].get_text()[:28]!r} <-> {boxes[j][0].get_text()[:28]!r} "
                      f"area={inter.width * inter.height:.0f}px2")
    fw, fh = fig.canvas.get_width_height()
    if os.environ.get("FIG_DEBUG"):
        for t, bb in boxes:
            if bb.y1 > fh - 15:
                print(f"[top] {t.get_text()[:30]!r} axes={getattr(t.axes, 'get_title', lambda: '?')()!r} "
                      f"bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f})")
    for t, bb in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fw + 1 or bb.y1 > fh + 1:
            bad += 1
            ax_owner = t.axes.get_title() if getattr(t, "axes", None) is not None else "?"
            print(f"[clipped] {t.get_text()[:40]!r} axes={ax_owner!r} bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}) canvas={fw}x{fh}")
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb, ab = leg.get_window_extent(r), ax.get_window_extent(r)
        inside = lb.x0 >= ab.x0 - 1 and lb.x1 <= ab.x1 + 1 and lb.y0 >= ab.y0 - 1 and lb.y1 <= ab.y1 + 1
        if not inside:
            bad += 1
            print(f"[legend-outside] {ax.get_title()!r} legend=({lb.x0:.0f},{lb.y0:.0f},{lb.x1:.0f},{lb.y1:.0f}) "
                  f"axes=({ab.x0:.0f},{ab.y0:.0f},{ab.x1:.0f},{ab.y1:.0f})")
    print(f"[layout] text artists={len(boxes)} collisions/clipped/legend-outside={bad}")


def main() -> int:
    font = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9.4,
                         "axes.titlesize": 10.2, "axes.labelsize": 9.2,
                         "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 7.9})
    agg = list(csv.DictReader(open(RES / "results_agg.csv")))
    drift = json.load(open(RES / "drift.json"))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1.15, 1.0]))

    # (a) dose-response, core arms only
    ax = axes[0]
    for arm, label, color, ls, mk in ARMS:
        rows = {int(r["delta"]): (float(r["nmse_mean"]), float(r["nmse_sd"]))
                for r in agg if r["arm"] == arm and r["cond"] == "main"}
        xs = [d for d in DELTAS if d in rows]
        ys = [rows[d][0] for d in xs]
        es = [rows[d][1] for d in xs]
        ax.errorbar(xs, ys, yerr=es, color=color, ls=ls, marker=mk, ms=4.5, lw=1.6, capsize=2.2,
                    elinewidth=0.9, label=label, zorder=3)
    ax.set_xlabel("慢上下文陈旧 Δ（帧，20 Hz）")
    ax.set_ylabel("动作块 NMSE（越低越好）")
    ax.set_xticks(DELTAS)
    ax.set_xlim(-0.4, 9.2)
    ax.set_ylim(0.195, 0.375)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_title("(a) 陈旧剂量反应：四条核心臂", pad=6)
    ax.annotate("Δ=8 帧 = 0.4 s", xy=(8, 0.205), xytext=(4.6, 0.201), fontsize=8, color="#333333")
    ax.annotate("", xy=(8.0, 0.212), xytext=(7.6, 0.201),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, borderpad=0.4, handlelength=2.2)

    # (b) channel ablations at delta=8
    ax = axes[1]
    n_arm = len(ARMS)
    width = 0.8 / n_arm
    xs = np.arange(len(CONDS))
    for i, (arm, label, color, ls, mk) in enumerate(ARMS):
        vals = []
        for key, _ in CONDS:
            row = next((r for r in agg if r["arm"] == arm and r["cond"] == key and int(r["delta"]) == 8), None)
            vals.append(float(row["nmse_mean"]) if row else np.nan)
        pos = xs - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.92, color=color, label=label, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[1] for c in CONDS], fontsize=8)
    ax.set_ylabel("动作块 NMSE @ Δ=8（越低越好）")
    ax.set_ylim(0, 1.32)
    ax.grid(alpha=0.25, axis="y", lw=0.6, zorder=0)
    ax.set_title("(b) Δ=8 时各通道被替换后的误差", pad=6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, borderpad=0.35, handlelength=1.3)
    ax.annotate("新视觉换旧仅存在于\n带当前视觉的两臂", xy=(3.0, 0.60), fontsize=7.8, color="#444444", ha="center")

    # (c) feature / pixel drift
    ax = axes[2]
    ds = [r["delta"] for r in drift]
    n_layer = len(drift[0]["vis_cos_per_layer"])
    for l in range(n_layer):
        ax.plot(ds, [r["vis_cos_per_layer"][l] for r in drift], lw=0.9, alpha=0.55,
                color="#7a7a7a", label=f"第 {l+1} 个路由层" if l == 0 else None)
    ax.plot(ds, [r["vis_cos"] for r in drift], "-o", color="#222222", lw=1.8, ms=4,
            label="视觉 token 余弦（四层均值）")
    ax.set_xlabel("Δ（帧，20 Hz）")
    ax.set_ylabel("慢视觉 token 与 t 时刻的余弦相似度")
    ax.set_xticks(DELTAS)
    ax.set_yticks([0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
    ax.set_ylim(0.925, 1.005)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, borderpad=0.35)
    ax.set_title("(c) 小幅特征漂移，明显控制后果", pad=6)
    ax2 = ax.twinx()
    ax2.plot(ds, [r["img_mae"] for r in drift], "--s", color="#b03a2e", lw=1.5, ms=4,
             label="像素平均绝对差 / 255")
    ax2.set_ylabel("图像平均绝对差（归一化）", color="#b03a2e")
    ax2.tick_params(axis="y", colors="#b03a2e", labelsize=8)
    ax2.set_yticks([0.0, 0.005, 0.010, 0.015])
    ax2.set_ylim(0, 0.016)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, borderpad=0.35)

    fig.tight_layout(pad=0.6, w_pad=1.6)
    report_overlaps(fig)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    print("font:", font or "default (CJK glyphs unavailable, labels would be tofu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
