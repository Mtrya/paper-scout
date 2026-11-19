"""Figures for the ForceHarness thread: (1) is the reported sigma trustworthy, (2) what does the fast loop buy.

  python figures.py --e1 runs/dev/e1_full.jsonl --e2 runs/dev/e2_full.jsonl --out ../../../runs/2026-11-19-dash-fast-slow/assets
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze

TASK_LABEL = {"weight_sort": "weight_sort (0.15/0.60 kg)", "weight_sort_ood": "weight_sort_ood (0.30/0.45 kg)",
              "com_sort": "com_sort (4 cm)", "com_sort_ood": "com_sort_ood (2.5 cm)"}
ARM_LABEL = {"paper": "paper noise (0.5 N, 0.3 N bias)", "bias1n": "declared 1 N bias",
             "undeclared": "undeclared +0.5 N residual", "force": "full force", "guard": "contact guard only",
             "pose": "no force primitive"}
COLORS = {"paper": "#1f4e79", "bias1n": "#b5651d", "undeclared": "#8b1a1a",
          "force": "#1f4e79", "guard": "#b5651d", "pose": "#8b1a1a"}
NOMINAL_GAP = {"weight_sort": 0.45, "weight_sort_ood": 0.15, "com_sort": 4.0, "com_sort_ood": 2.5}


def load_e1(path):
    return [r for r in (analyze.episode_metrics_e1(json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()) if r]


def load_e2(path):
    return [analyze.episode_metrics_e2(json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def fig_calibration(rows, out):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["arm"])].append(r)
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.75), constrained_layout=True)

    # a, b: reported sigma vs realized error, one panel per unit (kg, cm)
    for ax, family, ylabel in ((axes[0], "weight", "mass: RMS(estimate − truth)  [g]"),
                               (axes[1], "com", "COM: RMS(estimate − truth)  [cm]")):
        scale = 1000.0 if family == "weight" else 1.0
        for (task, arm), rs in groups.items():
            if (family == "weight") != ("weight" in task):
                continue
            key, skey = ("mass_err", "mass_sigma") if family == "weight" else ("com_err", "com_sigma_cm")
            errs = [r[key] for r in rs if r.get(key) is not None]
            sig = float(np.mean([r[skey] for r in rs if r.get(skey) is not None]))
            rms = float(np.sqrt(np.mean(np.square(errs)))) * scale
            ood = task.endswith("_ood")
            ax.scatter(sig * scale, rms, s=46, marker="o", facecolor="none" if ood else COLORS[arm],
                       edgecolor=COLORS[arm], lw=1.2, zorder=3)
            if arm == "paper" and not ood:
                ax.annotate("", xy=(sig * scale, rms), xytext=(0, 0), textcoords="offset points")
        lo, hi = (5, 190) if family == "weight" else (0.12, 2.2)
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
        ax.text(hi * 0.52, hi * 0.62, "reported σ\n= realized error", fontsize=5.8, color="0.4", ha="left")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"mean reported σ  [{'g' if family == 'weight' else 'cm'}]", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(("a. " if family == "weight" else "b. ") + "does σ describe the real error?", fontsize=8.5, loc="left")
        _style(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=COLORS[a], mec=COLORS[a], label=ARM_LABEL[a]) for a in ("paper", "bias1n", "undeclared")]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", label="OOD task")]
    axes[0].legend(handles=handles, fontsize=5.4, frameon=False, loc="upper left")

    # c: coverage of the reported interval
    ax = axes[2]
    xs, labels, colors = [], [], []
    for (task, arm), rs in sorted(groups.items()):
        z = [r["mass_z"] for r in rs if r.get("mass_z") is not None] + [r["com_z"] for r in rs if r.get("com_z") is not None]
        xs.append((float(np.mean([abs(v) <= 1 for v in z])), float(np.mean([abs(v) <= 2 for v in z]))))
        labels.append(f"{task.replace('_sort', '').replace('weight', 'W').replace('com', 'C')}\n{arm}")
        colors.append(COLORS[arm])
    idx = np.arange(len(xs))
    ax.bar(idx - 0.18, [x[0] for x in xs], width=0.34, color=colors, label="within 1σ")
    ax.bar(idx + 0.18, [x[1] for x in xs], width=0.34, color=colors, alpha=0.45, label="within 2σ")
    ax.axhline(0.6827, color="0.3", lw=0.8, ls=":")
    ax.axhline(0.9545, color="0.3", lw=0.8, ls="--")
    ax.text(len(xs) - 0.5, 0.69, "nominal 68%", fontsize=5.2, ha="right", color="0.3")
    ax.text(len(xs) - 0.5, 0.962, "nominal 95%", fontsize=5.2, ha="right", color="0.3")
    ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=4.4, rotation=90)
    ax.set_ylabel("empirical coverage", fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_title("c. does the interval cover the error?", fontsize=8.5, loc="left")
    _style(ax)

    # d: decision accuracy against the margin the reported sigma claims
    ax = axes[3]
    for (task, arm), rs in groups.items():
        if "weight" in task:
            margins = [r["mass_margin_sigma"] for r in rs if r.get("mass_margin_sigma") is not None]
            correct = [bool(r["mass_correct"]) for r in rs if r.get("mass_correct") is not None]
            marker = "o"
        else:
            margins = [r["com_margin_sigma"] for r in rs if r.get("com_margin_sigma") is not None]
            correct = [bool(r["com_correct"]) for r in rs if r.get("com_correct") is not None]
            marker = "s"
        ood = task.endswith("_ood")
        ax.scatter(float(np.mean(margins)), float(np.mean(correct)), s=46, marker=marker,
                   facecolor="none" if ood else COLORS[arm], edgecolor=COLORS[arm], lw=1.2, zorder=3)
    m = np.linspace(0, 8, 200)
    phi = 0.5 * (1 + np.vectorize(math.erf)(m / math.sqrt(2)))
    ax.plot(m, phi, color="0.5", lw=0.9, ls="--")
    ax.text(4.4, 0.60, "Gaussian\nwith the reported σ", fontsize=5.8, color="0.4")
    ax.set_xlabel("decision margin  |estimate − boundary| / σ", fontsize=8)
    ax.set_ylabel("decision accuracy", fontsize=8)
    ax.set_ylim(0.45, 1.02)
    ax.set_title("d. where the σ limit becomes the bottleneck", fontsize=8.5, loc="left")
    _style(ax)

    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def fig_control(rows, out):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["arm"])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)

    ax = axes[0]
    tasks = ["weight_sort", "com_sort"]
    arms = ["force", "guard", "pose"]
    for j, task in enumerate(tasks):
        vals = []
        for arm in arms:
            rs = groups[(task, arm)]
            key, skey = ("mass_err", "mass_sigma") if task == "weight_sort" else ("com_err", "com_sigma_cm")
            v = [r[key] for r in rs if r.get(key) is not None]
            sig = float(np.mean([r[skey] for r in rs if r.get(skey) is not None]))
            vals.append(float(np.sqrt(np.mean(np.square(v)))) / sig)
        ax.bar([j * 4 + k for k in range(3)], vals, color=[COLORS[a] for a in arms], width=0.75)
        for k, v in enumerate(vals):
            ax.text(j * 4 + k, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
    ax.text(5.45, 1.03, "error = reported σ", fontsize=5.6, ha="right", color="0.35")
    ax.set_xticks([1, 5]); ax.set_xticklabels(["weight_sort\n(mass)", "com_sort\n(COM offset)"], fontsize=7)
    ax.set_ylabel("RMS identification error / reported σ", fontsize=8)
    ax.set_title("a. identification quality by control layer", fontsize=8.5, loc="left")
    ax.set_ylim(0, max(1.9, ax.get_ylim()[1]))
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[a]) for a in arms], [ARM_LABEL[a] for a in arms],
              fontsize=5.6, frameon=False, loc="upper left")
    _style(ax)

    ax = axes[1]
    for j, task in enumerate(tasks):
        vals = [float(np.mean([max([x or 0 for x in (r.get("descend_peak_force") or [0])]) for r in groups[(task, arm)]]))
                for arm in arms]
        maxs = [float(np.max([max([x or 0 for x in (r.get("descend_peak_force") or [0])]) for r in groups[(task, arm)]]))
                for arm in arms]
        ax.bar([j * 4 + k for k in range(3)], vals, color=[COLORS[a] for a in arms], width=0.75)
        for k, v in enumerate(vals):
            ax.text(j * 4 + k, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
            ax.plot([j * 4 + k - 0.3, j * 4 + k + 0.3], [maxs[k]] * 2, color="0.15", lw=1.0, zorder=4)
    ax.axhline(25, color="#8b1a1a", lw=0.8, ls="--")   # the primitive's own force cap
    ax.text(5.45, 25.6, "25 N force cap of the primitive", fontsize=5.5, ha="right", color="#8b1a1a")
    ax.text(-0.55, 1.5, "bar = mean of per-episode peak, tick = worst episode", fontsize=5.2, color="0.3")
    ax.set_xticks([1, 5]); ax.set_xticklabels(["weight_sort", "com_sort"], fontsize=7)
    ax.set_ylabel("peak wrist force in the grasp approach (N)", fontsize=8)
    ax.set_title("b. what the fast loop changes: the transient", fontsize=8.5, loc="left")
    _style(ax)

    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def fig_com_tilt(rows, out):
    rs = [r for r in rows if r["task"].startswith("com_sort") and r["arm"] == "paper" and r.get("com_err") is not None]
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    tilt = np.array([abs(r["bar_tilt_deg"] or 0.0) for r in rs])
    true = np.array([r["com_true_cm"] for r in rs])
    est = np.array([r["com_est_cm"] for r in rs])
    sc = ax.scatter(true, est, c=tilt, cmap="viridis", s=30, vmin=0, vmax=45, zorder=3)
    lim = np.array([0, 5.2])
    ax.plot(lim, lim, color="0.5", lw=0.8, ls="--")
    ax.axhline(4.0, color="#1f4e79", lw=0.7, ls=":")
    ax.text(0.25, 4.08, "nominal 4 cm offset", fontsize=5.8, color="#1f4e79")
    ax.set_xlabel("true horizontal lever of the load's COM (cm)", fontsize=8)
    ax.set_ylabel("reported com_offset (cm)", fontsize=8)
    ax.set_title("reported vs true lever, com_sort, paper noise", fontsize=8.5, loc="left")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("|bar tilt| (deg)", fontsize=7); cb.ax.tick_params(labelsize=6)
    _style(ax)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="runs/dev/e1_full.jsonl")
    ap.add_argument("--e2", default="runs/dev/e2_full.jsonl")
    ap.add_argument("--out", default="runs/dev/figs")
    ap.add_argument("--prefix", default="force-")
    ap.add_argument("--episodes", default=None,
                    help="episodes_e1.csv of the run packet: draw only sigma-reliability (no JSONL needed)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.episodes:
        fig_sigma_reliability(load_episodes_csv(args.episodes), out / f"{args.prefix}sigma-reliability.png")
        return
    rows1 = load_e1(args.e1)
    fig_calibration(rows1, out / f"{args.prefix}estimation-calibration.png")
    fig_control(load_e2(args.e2), out / f"{args.prefix}control-ablation.png")
    fig_com_tilt(rows1, out / f"{args.prefix}com-tilt.png")
    fig_sigma_reliability(rows1, out / f"{args.prefix}sigma-reliability.png")



# --------------------------------------------------------------------------- the report's main panel: sigma vs real error
def load_episodes_csv(path):
    """Minimal rows for fig_sigma_reliability from a packet `episodes_e1.csv`."""
    import csv as _csv
    out = []
    for r in _csv.DictReader(Path(path).open()):
        row = dict(task=r["task"], arm=r["arm"], seed=int(r["seed"]), label=r.get("label") or None,
                   grasp_ok=(r.get("grasp_ok") or None))
        for k in ("mass_err", "mass_sigma", "com_err", "com_sigma_cm", "mass_true", "com_true_cm"):
            v = r.get(k) or ""
            row[k] = float(v) if v not in ("", "None") else None
        out.append(row)
    return out


def _cjk():
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def check_text_layout(fig, axes) -> dict:
    """Objective substitute for eyeballing: no two text boxes overlap, every text stays inside the figure, and no data
    point sits under a text box."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        texts.append((s, bb))
    pad = 1.0
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > pad) and (min(a.y1, b.y1) - max(a.y0, b.y0) > pad):
                pairs.append(f"{texts[i][0][:18]!r} x {texts[j][0][:18]!r}")
    under = []
    for ax in axes:
        for coll in ax.collections:
            off = getattr(coll, "get_offsets", lambda: [])()
            if off is None or len(off) == 0:
                continue
            pts = ax.transData.transform(np.asarray(off, dtype=float))
            for (x, y) in pts:
                for s, bb in texts:
                    if bb.x0 - 2 <= x <= bb.x1 + 2 and bb.y0 - 2 <= y <= bb.y1 + 2:
                        under.append(f"{s[:18]!r} over point ({x:.0f},{y:.0f})")
    return dict(texts=len(texts), overlapping_pairs=pairs, points_under_text=under)


def check_png_margins(path) -> dict:
    """The saved image must not cut any ink: the non-white bounding box needs a margin on every side."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    ink = a < 250
    ys, xs = np.where(ink)
    m = dict(left=int(xs.min()), right=int(a.shape[1] - 1 - xs.max()),
             top=int(ys.min()), bottom=int(a.shape[0] - 1 - ys.max()))
    return dict(size=[int(a.shape[1]), int(a.shape[0])], ink_margins_px=m, ok=min(m.values()) >= 2)


def _nice_lim(v: float) -> tuple[float, float]:
    """A round upper limit >= v with a round tick step, so no tick label falls outside the axes."""
    import math as _m
    best = None
    for k in range(-3, 6):
        for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
            step = mult * 10.0 ** k
            n = int(_m.ceil(v / step - 1e-9))
            if n < 3 or n > 9:
                continue
            lim = n * step
            score = (abs(n - 6), lim)
            if best is None or score < best[0]:
                best = (score, lim, step)
    return (best[1], best[2]) if best else (v * 1.2, v * 1.2 / 6)


def fig_sigma_reliability(rows, out):
    _cjk()
    arms = ["paper", "bias1n", "undeclared"]
    shown = {"paper": "论文噪声（0.5 N / 0.3 N 偏置）", "bias1n": "声明 1 N 偏置", "undeclared": "未声明 +0.5 N 残差"}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["arm"])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), constrained_layout=True)
    panels = [("weight", "质量：报告 σ 与真实误差（weight_sort）", "g", 1000.0, "mass_err", "mass_sigma"),
              ("com", "质心：报告 σ 与真实误差（com_sort）", "cm", 1.0, "com_err", "com_sigma_cm")]
    for ax, (family, title, unit, scale, ekey, skey) in zip(axes, panels):
        pts = []
        for arm in arms:
            for task in (f"{family}_sort", f"{family}_sort_ood"):
                rs = groups.get((task, arm), [])
                if family == "weight":      # missed grasps measure an empty gripper, not the cube
                    rs = [r for r in rs if str(r.get("grasp_ok")) != "False"]
                errs = [r[ekey] for r in rs if r.get(ekey) is not None]
                sigs = [r[skey] for r in rs if r.get(skey) is not None]
                if not errs or not sigs:
                    continue
                sig = float(np.mean(sigs)) * scale
                rms = float(np.sqrt(np.mean(np.square(errs)))) * scale
                pts.append(dict(arm=arm, ood=task.endswith("_ood"), x=sig, y=rms,
                                ratio=rms / sig if sig else float("nan"), n=len(errs)))
        lim, step = _nice_lim(max(max(p["x"] for p in pts), max(p["y"] for p in pts)) * 1.10)
        ax.plot([0, lim], [0, lim], color="0.45", lw=0.9, ls="--", zorder=1)
        for p in [p for p in pts if not p["ood"]]:
            ax.scatter(p["x"], p["y"], s=42, marker="o", color=COLORS[p["arm"]], edgecolor=COLORS[p["arm"]], zorder=3)
        for p in [p for p in pts if p["ood"]]:
            ax.scatter(p["x"], p["y"], s=150, marker="o", facecolor="none", edgecolor=COLORS[p["arm"]], lw=1.2, zorder=4)
        ticks = np.arange(0, lim + 1e-9, step)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"报告 σ（{unit}）", fontsize=8.5)
        ax.set_ylabel(f"真实误差 RMS（{unit}）", fontsize=8.5)
        ax.set_title(title, fontsize=9, loc="left")
        _style(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=COLORS[a], mec=COLORS[a], ms=5, label=shown[a]) for a in arms]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", ms=8, label="空心环 = 官方 OOD 档"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=6.4, handletextpad=0.3,
               columnspacing=1.2)
    axes[0].text(0.97, 0.04, "虚线 y = x\n线上：σ 偏小（乐观）\n线下：σ 偏大（保守）", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=5.6, color="0.35", linespacing=1.5)
    report = check_text_layout(fig, list(axes))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report["png"] = check_png_margins(out)
    print("wrote", out, "| layout check:", json.dumps(report, ensure_ascii=False))
    if report["overlapping_pairs"] or report["points_under_text"] or not report["png"]["ok"]:
        raise SystemExit("figure layout check failed: " + json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()
