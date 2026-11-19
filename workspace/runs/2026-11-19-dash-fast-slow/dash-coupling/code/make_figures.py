#!/usr/bin/env python
"""Figures for the DASH Eq.(11)-(12) timing note.  Reads out/results.json and
re-runs only the anchor scenario needed for the gripper trace."""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_timing as S

plt.rcParams.update({
    "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9.5,
})

C = {"truth": "#1a202c", "exact": "#2b6cb0", "shiftpos": "#c53030",
     "ceil": "#b7791f", "floor": "#805ad5", "fixed": "#2f855a", "grid": "#cbd5e0"}
HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default=os.path.join(HERE, "results", "results.json"))
_ap.add_argument("--assets", default=os.path.join(HERE, "..", "..", "assets"))
_ap.add_argument("--data", default=os.path.join(HERE, "data", "aloha_transfer_rows.json"))
ARGS = _ap.parse_args()
OUT = os.path.normpath(ARGS.assets)
os.makedirs(OUT, exist_ok=True)
DATA = ARGS.data
RES = json.load(open(ARGS.results))


def fig_bookkeeping():
    b = RES["bookkeeping"]["const0.6"]
    k = np.arange(len(b["S_k"]))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))

    ax[0].plot(k, b["S_k"], "-o", color=C["exact"], lw=1.8, ms=4,
               label=r"$S_k=\sum_{j\leq k}\rho_j$（写位置）")
    ax[0].plot(k, b["true_shift_at_own_arrival"], "-s", color=C["shiftpos"], lw=1.4, ms=3.5,
               label="第 $k$ 块到达时它自己的年龄")
    ax[0].annotate("", xy=(2, 1.2), xytext=(2, 0.0),
                   arrowprops=dict(arrowstyle="<->", color=C["ceil"], lw=1.2))
    ax[0].text(2.12, 0.55, "1.2 tick\n$=S_2$", color=C["ceil"], fontsize=9)
    ax[0].text(2.12, 1.28, "第 2 块自己的年龄 = 0", color=C["shiftpos"], fontsize=9)
    ax[0].set_xlabel("chunk 序号 $k$（$\\rho=0.6$）")
    ax[0].set_ylabel("tick")
    ax[0].set_title("(a) 累计量 $S_k$ 是写位置，不是当前块的老化量", fontsize=10)
    ax[0].legend(loc="upper left", fontsize=8, frameon=False)
    ax[0].grid(color=C["grid"], lw=0.5, alpha=0.6)

    ms = b["overshift_ms_if_Sk_is_shift"]
    cols = [C["shiftpos"] if i == 2 else "#a0aec0" for i in range(len(ms))]
    ax[1].bar(k, ms, color=cols, width=0.62)
    ax[1].axhline(0, color=C["truth"], lw=0.8)
    ax[1].annotate("把 $S_2=1.2$ 当作\n第 2 块的位移\n= 过移 60 ms", xy=(2, 60), xytext=(3.4, 150),
                   fontsize=9, color=C["shiftpos"],
                   arrowprops=dict(arrowstyle="->", color=C["shiftpos"], lw=1))
    ax[1].set_xlabel("chunk 序号 $k$")
    ax[1].set_ylabel("过移 (ms)")
    ax[1].set_title("(b) 若把 $S_k$ 直接当作位移施加", fontsize=10)
    ax[1].grid(axis="y", color=C["grid"], lw=0.5, alpha=0.6)

    fig.tight_layout()
    p = os.path.join(OUT, "dash-eq12-bookkeeping.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


def fig_trajectory():
    ref = S.Ref(*S.analytic_trajectory())
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for seq in ("const0.6", "cross"):
        rho = S.SCHEDULES[seq](int(1.7 * (ref.ts[-1] / S.T_EXEC)) + 4)
        chunks = S.chunks_from_rho(rho, 2 * S.T_EXEC, S.HORIZON, S.T_EXEC, ref)
        t1 = min(ref.ts[-1] - 0.2, chunks[-1]["a"] + (S.HORIZON - 2) * S.T_EXEC)
        j = 0 if seq == "const0.6" else 1
        axx = ax[j]
        axx.plot(ref.ts, ref.xs[:, 0], color=C["truth"], lw=2.2, label="参考轨迹 $g(t)$", alpha=0.55)
        for arm, col, ls, lab in (("exact_arrival", C["exact"], "-", "Eq.(12) 位置语义"),
                                  ("ceil_clock", C["ceil"], "--", "整数 ceil 时钟"),
                                  ("shiftpos", C["shiftpos"], "-", "$S_k$ 当位移")):
            ts, u, g, lead, live, starved, fs = S.run_arm(chunks, rho, arm, ref, chunks[0]["a"], t1, "exp")
            n = len(ts) if fs is None else max(2, int(round(fs / S.T_EXEC)) + 1)
            axx.plot(ts[:n], u[:n, 0], color=col, ls=ls, lw=1.5, label=lab)
            if fs is not None:
                axx.axvline(ts[0] + fs, color=col, lw=0.8, ls=":")
                axx.text(ts[0] + fs + 0.05, axx.get_ylim()[1] * 0.55, "饥饿", color=col, fontsize=8)
        axx.set_xlabel("时间 (s)")
        axx.set_ylabel("$x$ (m)")
        axx.set_title(f"({'ab'[j]}) $\\rho$ = {seq}", fontsize=10)
        axx.grid(color=C["grid"], lw=0.5, alpha=0.6)
        if j == 0:
            axx.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "dash-trajectory-arms.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


def fig_metrics():
    rows = [r for r in RES["chunk_rows"] if r["weights"] == "exp" and r["ref"] == "analytic"]
    arms = ["exact_arrival", "ceil_clock", "floor_clock", "fixed_clock", "shiftpos"]
    seqs = ["const0.6", "jitter", "cross"]
    names = {"exact_arrival": "位置(精确)", "ceil_clock": "整数 ceil", "floor_clock": "整数 floor",
             "fixed_clock": "固定偏移", "shiftpos": "$S_k$ 当位移"}
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    w = 0.15
    for i, arm in enumerate(arms):
        vals = [next(r["time_err_ticks"] for r in rows if r["arm"] == arm and r["seq"] == s) for s in seqs]
        xs = np.arange(len(seqs)) + (i - 2) * w
        cols = [C["exact"], C["ceil"], C["floor"], C["fixed"], C["shiftpos"]][i]
        ax.bar(xs, vals, width=w, color=cols, label=names[arm])
        for x, v in zip(xs, vals):
            if v > 12:
                ax.text(x, 11.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=cols, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 30)
    ax.axhline(1.0, color=C["truth"], lw=0.9, ls="--")
    ax.text(2.45, 1.08, "1 tick = 50 ms", fontsize=8, color=C["truth"])
    ax.set_xticks(np.arange(len(seqs)))
    ax.set_xticklabels(["恒定 $\\rho=0.6$", "抖动 $\\rho\\in[0.5,1.6]$", "跨整数 $0.9/1.1$+漂移"])
    ax.set_ylabel("等效时间误差 (tick)")
    ax.set_title("各语义下的执行误差（解析轨迹，指数权重，$H=16$）", fontsize=10)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", color=C["grid"], lw=0.5, alpha=0.6)
    fig.tight_layout()
    p = os.path.join(OUT, "dash-arm-error.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


def fig_anchor():
    # rebuild the anchor scenario exactly as sim_timing.main does
    ref = S.Ref(*S.aloha_trajectory(DATA))
    n = int(1.6 * (ref.ts[-1] / S.T_EXEC))
    base_s = 2 * S.T_EXEC + np.arange(n) * S.T_EXEC
    rng = np.random.default_rng(S.SEED + 7)
    L = S.T_EXEC * (1.0 + np.clip(0.25 * rng.standard_normal(n), -0.5, 0.5))
    chunks = S.chunks_from_anchors(base_s, L, S.HORIZON, S.T_EXEC, ref)
    rho = np.concatenate([[1.0], np.diff([c["a"] for c in chunks]) / S.T_EXEC])
    t0, t1 = chunks[0]["a"], min(ref.ts[-1] - 0.2, chunks[-1]["a"] + (S.HORIZON - 2) * S.T_EXEC)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    lab = {"exact_anchor": ("观测时刻锚定（真值）", C["exact"]),
           "exact_arrival": ("到达时刻锚定（$\\rho$ 的用法）", C["shiftpos"]),
           "prefix_ceil": ("到达锚定 + 整数前缀丢弃", C["ceil"])}
    for arm, (name, col) in lab.items():
        ts, u, g, lead, live, starved, fs = S.run_arm(chunks, rho, arm, ref, t0, t1, "exp")
        ax[0].plot(ts, u[:, 6], color=col, lw=1.6, label=name)
        ax[1].bar(["观测锚定", "到达锚定", "整数前缀"][list(lab).index(arm)],
                  [next(r["time_err_ticks"] for r in RES["anchor_rows"]
                        if r["arm"] == arm and r["ref"] == "aloha" and r["scenario"] == "gauss")],
                  color=col, width=0.55)
    ax[0].plot(ref.ts, ref.xs[:, 6], color=C["truth"], lw=2.4, alpha=0.35, label="参考动作 $g(t)$")
    ax[0].set_xlim(4.3, 5.2)
    ax[0].set_xlabel("时间 (s)")
    ax[0].set_ylabel("左夹爪指令")
    ax[0].set_title("(a) 抓取时刻随锚点移动 50 ms（ALOHA 仿真遥操作示范）", fontsize=10)
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    ax[0].grid(color=C["grid"], lw=0.5, alpha=0.6)
    ax[0].axvline(4.66, color=C["truth"], lw=0.8, ls="--")
    ax[0].text(4.665, ax[0].get_ylim()[0] + 0.05, "真值抓取", fontsize=7.5, rotation=90, va="bottom")

    ax[1].set_ylabel("等效时间误差 (tick)")
    ax[1].set_title("(b) 三种锚点的误差：1 tick = 整个动作头延迟", fontsize=10)
    ax[1].grid(axis="y", color=C["grid"], lw=0.5, alpha=0.6)
    vals = [next(r["time_err_ticks"] for r in RES["anchor_rows"]
                 if r["arm"] == a and r["ref"] == "aloha" and r["scenario"] == "gauss")
            for a in ("exact_anchor", "exact_arrival", "prefix_ceil")]
    for x, v in enumerate(vals):
        ax[1].text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax[1].set_ylim(0, 1.25)
    fig.tight_layout()
    p = os.path.join(OUT, "dash-anchor-latency.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    fig_bookkeeping()
    fig_trajectory()
    fig_metrics()
    fig_anchor()
