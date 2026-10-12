"""Figures for the score-centering toy thread. Reads the JSON written by the
three experiment scripts and writes PNGs into the same directory.

Usage: uv run --with numpy --with matplotlib python3 make_plots.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGDIR = os.environ.get("SC_FIGDIR", "figs")
COLORS = {"vanilla": "#c0392b", "SC": "#2471a3", "TIS": "#1e8449",
          "MIS": "#b7950b", "SC+TIS": "#7d3c98", "SC+MIS": "#a04000"}
STYLE = {"vanilla": "-", "SC": "-", "TIS": "--", "MIS": "--",
         "SC+TIS": "-.", "SC+MIS": "-."}


def load(name):
    with open(os.path.join(FIGDIR, name)) as f:
        return json.load(f)


def fig_drift_constant(d, d2=None):
    """d = faithful-scale run (exp2), d2 = amplified run with root-only
    diagnostics (exp2b). Panels (a)-(c) use exp2b at noise sigma=0.6; panel (d)
    uses exp2's noise sweep (all-row entropy, lr=0.1: no collapse)."""
    s2 = d2["lanes"] if d2 else None
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    if s2:
        lane = s2["noise 0.6"]
        for ax, key, ylab, title in (
                (axes[0], "root_entropy", "root-prefix entropy (nats)",
                 "(a) constant +1 reward, σ=0.6, lr=1.0: root rows collapse"),
                (axes[1], "root_tv_pq", "root TV(p, q)",
                 "(b) the mismatch metric ‖p−q‖  (vanilla: 0.225→0.004)"),
                (axes[2], "root_tv_p0", "root TV(p, p₀)",
                 "(c) distance moved: SC stays exactly at 0")):
            for m, h in lane.items():
                ax.plot(h["step"], h[key], STYLE[m],
                        color=COLORS[m], label=m, lw=1.8)
            ax.set_xlabel("SGD step")
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        ax = axes[3]
        for lane_name, ls in (("noise 0.6", "-"), ("noise 2.0", "--"),
                              ("quant 3b", ":"), ("stale k=16", "-.")):
            h = s2[lane_name]["vanilla"]
            ax.plot(h["step"], h["root_entropy"], ls,
                    color=COLORS["vanilla"], lw=1.6, label=f"vanilla {lane_name}")
            h = s2[lane_name]["SC"]
            ax.plot(h["step"], h["root_entropy"], ls,
                    color=COLORS["SC"], lw=1.2, alpha=0.7, label=f"SC {lane_name}")
        ax.set_xlabel("SGD step")
        ax.set_ylabel("root-prefix entropy (nats)")
        ax.set_title("(d) severity: more mismatch → earlier collapse", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_drift_constant.png"), dpi=150)
    plt.close(fig)


def fig_signal(d, anat):
    res = d["results"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [("table|stale k=64", "(a) staleness k=64, per-prefix table"),
              ("table|noise 3.0", "(b) sampler noise σ=3.0, per-prefix table"),
              ("lowrank|noise 3.0", "(c) same noise, shared low-rank head"),
              ("lowrank|stale k=64", "(d) same staleness, shared low-rank head")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        block = res.get(key)
        if block is None:
            continue
        for m, h in block["exact"].items():
            y = h["reward_trainer"]
            ax.plot(range(1, len(y) + 1), y, STYLE[m], color=COLORS[m],
                    label=m, lw=1.8)
        ax.set_xlabel("SGD step")
        ax.set_ylabel("training reward  E_q[R]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_signal.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    keys = [r for r in anat if r["reward_mode"] == "group_centered"]
    labels = [r["lane"] for r in keys]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [r["root_drift"] for r in keys], 0.35, label="root prefix drift",
           color="#c0392b")
    ax.bar(x + 0.2, [r["d2_drift"] for r in keys], 0.35, label="depth-2 prefix drift",
           color="#2471a3")
    ax.plot(x - 0.2, [r["root_signal"] for r in keys], "k_", ms=14, label="root signal")
    ax.plot(x + 0.2, [r["d2_signal"] for r in keys], "k+", ms=10, label="depth-2 signal")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("‖expected update‖ at step 0")
    ax.set_title("(a) drift anatomy, group-centered rewards", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    for m in ("vanilla", "SC", "TIS", "SC+TIS"):
        block = res.get("table|stale k=64", {}).get("sampled")
        if block is None or m not in block:
            continue
        mu = np.array(block[m]["reward_mean"])
        sd = np.array(block[m]["reward_std"])
        ax.plot(mu, color=COLORS[m], label=f"{m} (3 seeds)", lw=1.8)
        ax.fill_between(np.arange(len(mu)), mu - sd, mu + sd, color=COLORS[m],
                        alpha=0.2)
    ax.set_xlabel("SGD step")
    ax.set_ylabel("training reward  E_q[R]")
    ax.set_title("(b) sampled control, staleness k=64 (G=8, 3 seeds)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_signal_anatomy.png"), dpi=150)
    plt.close(fig)


def fig_official(d, spiky=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    lanes = [("flat sd=3.0", "#c0392b", "-"), ("peaky sd=1.5+spike 15", "#2471a3", "-"),
             ("wordlike sd=1.0+spike 25", "#1e8449", "--")]
    if spiky is not None:
        for label, color, style in lanes:
            sub = [r for r in spiky if r["lane"] == label]
            if not sub:
                continue
            ks = sorted({r["k"] for r in sub})
            rf = [np.mean([r["residual_frac"] for r in sub if r["k"] == k]) for k in ks]
            hs = [np.mean([r["head_mass"] for r in sub if r["k"] == k]) for k in ks]
            ax.plot(ks, rf, "o" + style, color=color, lw=1.8,
                    label=f"{label}: residual drift")
            ax.plot(ks, np.array(hs) / 1000.0, ":", color=color, lw=1.2)
    else:
        rows = d["topk_drift"]
        ks = sorted({r["k"] for r in rows})
        ax.plot(ks, [np.mean([r["residual_frac"] for r in rows if r["k"] == k])
                     for k in ks], "o-", color="#c0392b")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel("top-k head size k")
    ax.set_ylabel("residual drift / vanilla drift")
    ax.set_title("(a) top-k centering leaves (0.2–0.5)×tail mass of drift\n"
                 "(dotted: sampler mass in head / 1000, V=4096)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[1]
    align = d["alignment"]
    methods = ["vanilla", "SC", "TIS", "SC+TIS", "MIS", "SC+MIS"]
    drifts = [np.mean([a["drift_norm"] / a["scale"] for a in align
                       if a["method"] == m and a["adv"] == 1.0])
              for m in methods]
    err = [max(a["rel_err"] for a in align
               if a["method"] == m and a["k"] == "full") for m in methods]
    ax.bar(range(len(methods)), drifts, color=[COLORS[m] for m in methods])
    for i, e in enumerate(err):
        ax.text(i, drifts[i] * 1.02 + 0.01, f"err {e:.0e}", ha="center", fontsize=7)
    ax.set_xticks(range(len(methods)), methods, fontsize=8)
    ax.set_ylabel("drift left by the rule / vanilla drift\n(advantage +1, full-vocab; SC bar = its top-k path)")
    ax.set_title("(b) official loss, constant reward, full-vocab centering\n"
                 "(err = |official autodiff − analytic| / drift scale)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_official.png"), dpi=150)
    plt.close(fig)


def main():
    """Reads the compact JSONs (regenerate them with compact_results.py)."""
    for name, fn in (("exp2_drift_compact.json", fig_drift_constant),):
        if os.path.exists(os.path.join(FIGDIR, name)):
            fn(load(name))
        else:
            print(f"skipping {name} (not found)")
    if os.path.exists(os.path.join(FIGDIR, "exp1_official.json")):
        spiky = (load("exp1c_topk_spiky.json")
                 if os.path.exists(os.path.join(FIGDIR, "exp1c_topk_spiky.json"))
                 else None)
        fig_official(load("exp1_official.json"), spiky)
    if os.path.exists(os.path.join(FIGDIR, "exp2b_severity_compact.json")):
        fig_drift_constant(load("exp2_drift_compact.json"),
                           load("exp2b_severity_compact.json"))
    name = "exp3_signal_compact.json"
    if os.path.exists(os.path.join(FIGDIR, name)):
        exp3 = load(name)
        fig_signal(exp3, exp3["anatomy"])
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
