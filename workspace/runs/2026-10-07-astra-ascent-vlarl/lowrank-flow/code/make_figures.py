"""Figures for the C1-C5 report. Reads analysis/*.json + out/seed*/."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = os.environ.get("FIGDIR", "figures")
BLUE, RED, GREEN, ORANGE, PURPLE = "#2166ac", "#b2182b", "#1b7837", "#d95f02", "#762a83"


def load(path):
    with open(path) as f:
        return json.load(f)


def agg(datasets, fn):
    vals = [fn(d) for d in datasets.values() if fn(d) is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def fig_spectrum(W):
    groups = ["AdaRMS", "TimeMLP", "Rest"]
    tags = [("bc_stage", "BC from init (continuous $\\tau$)", GREEN),
            ("rl", "RL from BC (discrete $\\tau$)", RED),
            ("bc_cont", "BC branch (continuous $\\tau$)", BLUE),
            ("bc_disc", "BC branch (discrete $\\tau$)", ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    for gi, g in enumerate(groups):
        ax = axes[gi]
        for tag, label, col in tags:
            curves = []
            for seed, res in W.items():
                if tag not in res:
                    continue
                for k, st in res[tag]["per_matrix"].items():
                    if st.get("group", None) != g and not name_matches(k, g):
                        continue
                    if "sv_norm" in st:
                        curves.append(st["sv_norm"])
            if not curves:
                continue
            L = min(len(c) for c in curves)
            m = np.mean([c[:L] for c in curves], 0)
            s = np.std([c[:L] for c in curves], 0)
            ax.plot(range(L), m, label=label, color=col, lw=1.8)
            ax.fill_between(range(L), m - s, m + s, color=col, alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(g, fontsize=11)
        ax.set_xlabel("singular direction index")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("normalised singular value")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Update spectra by module group (normalised SVD, mean$\\pm$std over matrices/seeds)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_update_spectrum.png", dpi=160)
    plt.close(fig)


def name_matches(key, group):
    if group == "AdaRMS":
        return ".ada." in key
    if group == "TimeMLP":
        return key.startswith("time_mlp")
    return (".ada." not in key) and not key.startswith("time_mlp")


def fig_density_rank(W):
    tags = [("rl", "RL", RED), ("bc_cont", "BC cont.", BLUE),
            ("bc_disc", "BC disc.", ORANGE), ("bc_stage", "BC stage", GREEN)]
    groups = ["AdaRMS", "TimeMLP", "Rest"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    x = np.arange(len(groups))
    wdt = 0.2
    for i, (tag, label, col) in enumerate(tags):
        dens, r95, r95f, esh = [], [], [], []
        for g in groups:
            m, s = agg(W, lambda d, t=tag, g=g: d[t]["groups"][g]["density_1e-05"]
                       if t in d else None)
            dens.append(m or 0)
            m, _ = agg(W, lambda d, t=tag, g=g: d[t]["groups"][g]["r95_mean"]
                       if t in d else None)
            r95.append(m or 0)
            m, _ = agg(W, lambda d, t=tag, g=g: d[t]["groups"][g]["r95_frac_mean"]
                       if t in d else None)
            r95f.append(m or 0)
            m, _ = agg(W, lambda d, t=tag, g=g: d[t]["groups"][g]["energy_share"]
                       if t in d else None)
            esh.append(m or 0)
        axes[0].bar(x + i * wdt, dens, wdt, label=label, color=col)
        axes[1].bar(x + i * wdt, r95, wdt, label=label, color=col)
        axes[2].bar(x + i * wdt, r95f, wdt, label=label, color=col)
        axes[3].bar(x + i * wdt, esh, wdt, label=label, color=col)
    for ax, ttl, yl in zip(axes,
                           ["update density ($|\\Delta w|>10^{-5}$)",
                            "effective rank $r_{95}$",
                            "$r_{95}$ / max rank",
                            "update-energy share"],
                           ["fraction of params", "rank", "fraction", "share"]):
        ax.set_xticks(x + 1.5 * wdt)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(yl, fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_density_rank.png", dpi=160)
    plt.close(fig)


def fig_tau_dirs(W):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    cfg = [("rl", "RL (discrete $\\tau$)", RED, axes[0]),
           ("bc_disc", "BC, discrete $\\tau$", ORANGE, axes[1]),
           ("bc_cont", "BC, continuous $\\tau$", BLUE, axes[2])]
    for tag, label, col, ax in cfg:
        curves = []
        for seed, res in W.items():
            key = f"tau_dirs_{tag}"
            if key not in res:
                continue
            v0 = [v["cos"] for k, v in res[key].items() if k.endswith("|v0")]
            if v0:
                curves.append(np.mean(v0, 0))
        if curves:
            L = min(len(c) for c in curves)
            m = np.mean([c[:L] for c in curves], 0)
            s = np.std([c[:L] for c in curves], 0)
            xs = np.linspace(0, 0.95, L)
            ax.plot(xs, m, color=col, lw=2)
            ax.fill_between(xs, np.maximum(m - s, 0), m + s, color=col, alpha=0.2)
        for k in range(10):
            ax.axvline(k / 10, color="grey", lw=0.6, ls=":", alpha=0.6)
        ax.set_title(f"{label}\ntop AdaRMS input direction vs $c_\\tau$", fontsize=10)
        ax.set_xlabel("denoising timestep $\\tau$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("normalised $|\\cos(v_1, c_\\tau)|$")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_tau_dirs.png", dpi=160)
    plt.close(fig)


def fig_ssg_cos(W):
    keys = [("ssg_rl_vs_bc", "RL vs BC"), ("ssg_disc_vs_bc", "BC disc vs BC"),
            ("ssg_cont_vs_bc", "BC cont vs BC")]
    comps = ["scale", "shift", "gate"]
    cols = {"scale": BLUE, "shift": RED, "gate": GREEN}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, (key, label) in zip(axes, keys):
        data = {c: [] for c in comps}
        for seed, res in W.items():
            if key not in res:
                continue
            for c in comps:
                arr = np.array(res[key]["abs_cos"][c])
                data[c].append(np.abs(arr).mean())
        means = [np.mean(data[c]) if data[c] else 0 for c in comps]
        stds = [np.std(data[c]) if data[c] else 0 for c in comps]
        ax.bar(range(3), means, yerr=stds, color=[cols[c] for c in comps],
               capsize=4, width=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(comps)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}: $|\\cos(\\mathrm{{base}},\\ \\Delta)|$", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        for i, m in enumerate(means):
            ax.text(i, m + 0.03, f"{m:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("mean |cos| over layers x timesteps")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_ssg_cos.png", dpi=160)
    plt.close(fig)


def fig_replace(R):
    order = [("bc", "BC"), ("rl", "RL"), ("ts_only", "RL TS only\n(BC + RL TS)"),
             ("nonts_only", "RL minus TS\n(RL + BC TS)"), ("ts_top4", "BC + top-4 singular\ndirections of RL TS update"),
             ("ts_wo_top4", "BC + rest of\nRL TS update")]
    labels, means, stds = [], [], []
    for k, lab in order:
        vals = [d[k]["success"] for d in R.values() if k in d]
        if not vals:
            continue
        labels.append(lab)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    cols = [BLUE, RED, ORANGE, PURPLE, GREEN, "#8c510a"][:len(labels)]
    ax.bar(range(len(labels)), means, yerr=stds, color=cols, capsize=4, width=0.65)
    for i, m in enumerate(means):
        ax.text(i, m + 0.012, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("success rate")
    ax.set_title("Module replacement: which parameters carry the RL gain", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_replace.png", dpi=160)
    plt.close(fig)


def fig_probe(P):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = ["probe_all_auc", "probe_randomdir_auc", "probe_random_auc_mean"]
    names = ["shift-update\ndirections", "random\ndirections", "shuffled\nlabels"]
    for i, (k, nm) in enumerate(zip(labels, names)):
        vals = [d[k] for d in P.values() if k in d]
        if not vals:
            continue
        axes[0].bar(i, np.mean(vals), yerr=np.std(vals), capsize=4,
                    color=[RED, "#999999", "#dddddd"][i], width=0.55,
                    edgecolor="k", linewidth=0.5)
        axes[0].text(i, np.mean(vals) + 0.015, f"{np.mean(vals):.3f}", ha="center",
                     fontsize=9)
    axes[0].axhline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylim(0.4, 1.02)
    axes[0].set_ylabel("ROC-AUC (episode outcome)")
    axes[0].set_title("Linear probe on shift-update projections", fontsize=10)
    axes[0].grid(alpha=0.25, axis="y")
    # steering curves
    alphas = [0.25, 0.5, 1.0, 2.0]
    for pol, col in [("rl", RED), ("bc", BLUE)]:
        for mode, ls, mlab in [("shift_adaptive", "-", "shift (adaptive)"),
                               ("shift_fixed", "--", "shift (fixed)"),
                               ("random_fixed", ":", "random (fixed)")]:
            xs, ys, es = [], [], []
            for a in alphas:
                key = f"{mode}_a{a}"
                vals = [d[f"steer_{pol}"][key] for d in P.values()
                        if "steer_" + pol in d and key in d["steer_" + pol]]
                if vals:
                    xs.append(a)
                    ys.append(np.mean(vals))
                    es.append(np.std(vals))
            if xs:
                base = np.mean([d[f"steer_{pol}"]["alpha0"] for d in P.values()
                                if f"steer_{pol}" in d])
                axes[1].errorbar(xs, np.array(ys) - base, yerr=es, marker="o",
                                 ls=ls, color=col, label=f"{pol.upper()} {mlab}",
                                 capsize=3, lw=1.6, ms=4)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xticks(alphas)
    axes[1].set_xticklabels([str(a) for a in alphas])
    axes[1].set_xlabel("steering strength $\\alpha$")
    axes[1].set_ylabel("success change vs no steering")
    axes[1].set_title("Steering along the shift direction", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_probe_steer.png", dpi=160)
    plt.close(fig)


def fig_success(seeds):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (k, lab, col) in enumerate([("bc_success", "BC", BLUE),
                                       ("rl_success", "RL", RED)]):
        vals = [s[k] for s in seeds if k in s]
        if not vals:
            continue
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color=col, capsize=5, width=0.55)
        for j, v in enumerate(vals):
            ax.scatter(i + (j - (len(vals) - 1) / 2) * 0.05, v, color="k", s=14, zorder=3)
        ax.text(i, np.mean(vals) + 0.015, f"{np.mean(vals):.3f}", ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BC", "RL"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate (200 episodes)")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("BC $\\rightarrow$ RL (5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_success.png", dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    seeds = []
    for p in sorted(glob.glob("out/seed*/rl_metrics.json")):
        seeds.append(load(p))
    if os.path.exists("analysis/weights.json"):
        fig_spectrum(load("analysis/weights.json"))
        fig_density_rank(load("analysis/weights.json"))
        fig_tau_dirs(load("analysis/weights.json"))
        fig_ssg_cos(load("analysis/weights.json"))
    if os.path.exists("analysis/replace.json"):
        fig_replace(load("analysis/replace.json"))
    if os.path.exists("analysis/probe.json"):
        fig_probe(load("analysis/probe.json"))
    if seeds:
        fig_success(seeds)
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
