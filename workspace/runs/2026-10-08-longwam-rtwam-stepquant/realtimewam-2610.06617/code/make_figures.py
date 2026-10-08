"""Aggregate distillation runs into summary.txt/json and draw the report figures."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM_ORDER = ["bare", "cd", "ta", "cdta", "cdta1"]
ARM_LABEL = {"bare": "bare-1step\n(teacher, untrained)", "cd": "cd-only",
             "ta": "ta-only", "cdta": "cd+ta\nλ=0.2", "cdta1": "cd+ta\nλ=1.0"}
ARM_COLOR = {"bare": "#444444", "cd": "#d1495b", "ta": "#00798c",
             "cdta": "#edae49", "cdta1": "#7b6d8d"}
SUCC_KEYS = ["success_1step", "success_2step", "success_5step", "success_10step",
             "teacher_success_10step", "teacher_success_1step"]
DIAG_KEYS = ["global_err", "global_err_avg", "local_err_data", "local_err_traj",
             "teacher_1step_err", "global_norm_aT"]
HIST_KEYS = ["loss_cd_train", "loss_ta_train", "local_err_data", "local_err_traj",
             "global_err", "global_err_avg", "success_1step"]


def mean_std(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def load(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "s*_*", "*.json"))):
        r = json.load(open(path))
        hist = r.get("history") or []
        for k in ("loss_cd_train", "loss_ta_train"):
            vals = [h[k] for h in hist if k in h]
            if vals:
                r[k + "_final"] = vals[-1]
        runs.setdefault(r["arm"], {})[r["seed"]] = r
    return runs


def summarize(runs, outdir, tag=""):
    seeds = sorted({s for a in runs.values() for s in a})
    arms = [a for a in ARM_ORDER if a in runs]
    summary = {"tag": tag, "seeds": seeds, "arms": {}}
    lines = []
    for a in arms:
        per = runs[a]
        rows = {}
        for key in SUCC_KEYS:
            vals = [r[key] for r in per.values() if key in r]
            if vals:
                rows[key] = mean_std(vals)
        for key in DIAG_KEYS:
            vals = [r["diag"][key] for r in per.values()
                    if key in r.get("diag", {})]
            if vals:
                rows[key] = mean_std(vals)
        # per-tau breakdown of the global endpoint error (mean over seeds) and
        # how much of the paper's 1/t^2-weighted average the top two tau points
        # carry (the weighting makes the low-noise end dominate)
        bks = [r["diag"]["global_data_by_k"] for r in per.values()
               if r.get("diag", {}).get("global_data_by_k")]
        if bks:
            mean_bk = [float(sum(b[k] for b in bks) / len(bks)) for k in range(len(bks[0]))]
            rows["global_by_k_mean"] = mean_bk
            w = [1.0 / (1.0 - k / 10) ** 2 for k in range(len(mean_bk))]
            rows["taweight_share_top2"] = (w[8] * mean_bk[8] + w[9] * mean_bk[9]) / sum(
                a * b for a, b in zip(w, mean_bk))
        # paper Fig.3-right quantity for every arm: 1/t^2-weighted endpoint error
        # on the held-out tau grid, recomputed from the stored per-tau breakdown
        wvals = []
        for r in per.values():
            bk = r.get("diag", {}).get("global_data_by_k")
            if bk:
                w = [1.0 / (1.0 - k / 10) ** 2 for k in range(len(bk))]
                wvals.append(sum(a * b for a, b in zip(w, bk)) / sum(w))
        if wvals:
            rows["ta_loss_eval_weighted"] = mean_std(wvals)
        for key in ("loss_cd_train_final", "loss_ta_train_final"):
            vals = [r[key] for r in per.values() if key in r]
            if vals:
                rows[key] = mean_std(vals)
        # CD-only can be non-monotone: track the best global error reached and
        # the success rate at that step, not just the final point
        best, best_step, best_succ = [], [], []
        for r in per.values():
            hist = [h for h in (r.get("history") or []) if "global_err" in h]
            if not hist:
                continue
            h = min(hist, key=lambda h: h["global_err"])
            best.append(h["global_err"])
            best_step.append(h["step"])
            if "success_1step" in h:
                best_succ.append(h["success_1step"])
        if best:
            rows["global_err_best"] = mean_std(best)
            rows["global_err_best_step"] = mean_std(best_step)
        if best_succ:
            rows["success_at_best_global"] = mean_std(best_succ)
        rows["wall_s"] = mean_std([r["wall_s"] for r in per.values()])
        summary["arms"][a] = {
            "n_seed": len(per), "mean": rows,
            "per_seed": {str(s): {
                **{k: r.get(k) for k in SUCC_KEYS},
                **{k: r.get(k) for k in ("loss_cd_train_final", "loss_ta_train_final")},
                **{k: r["diag"].get(k) for k in DIAG_KEYS}} for s, r in per.items()}}
        def f(key):
            return f"{rows[key][0]:.5f}" if key in rows else "  --   "
        lines.append(
            f"{a:11s} n={len(per)} succ1 {rows['success_1step'][0]:.3f}±{rows['success_1step'][1]:.3f}"
            f"  succ2 {rows.get('success_2step', (float('nan'),))[0]:.3f}"
            f"  succ5 {rows.get('success_5step', (float('nan'),))[0]:.3f}"
            f"  succ10 {rows.get('success_10step', (float('nan'),))[0]:.3f}"
            f"  global {rows['global_err'][0]:.5f}±{rows['global_err'][1]:.5f}"
            f"  global_avg {rows['global_err_avg'][0]:.5f}"
            f"  local {rows['local_err_data'][0]:.5f}"
            f"  loss_cd {f('loss_cd_train_final')}"
            f"  loss_ta {f('loss_ta_train_final')}"
            f"  TAweighted {f('ta_loss_eval_weighted')}")
        if "global_by_k_mean" in rows:
            lines.append(f"{'':11s}      per-tau err^2 by k=0..9: " +
                         " ".join(f"{v:.5f}" for v in rows["global_by_k_mean"]) +
                         f"  | top-2 tau share of 1/t^2 avg {rows['taweight_share_top2']:.2f}")
        if "global_err_best" in rows:
            lines.append(
                f"{'':11s}      global_best {rows['global_err_best'][0]:.5f} @step "
                f"{rows['global_err_best_step'][0]:.0f}"
                f"  succ@best {rows.get('success_at_best_global', (float('nan'),))[0]:.3f}")
    txt = "\n".join(lines)
    print(f"--- {tag}\n{txt}")
    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, f"summary{tag}.txt"), "w").write(txt + "\n")
    json.dump(summary, open(os.path.join(outdir, f"summary{tag}.json"), "w"), indent=1)
    return summary


def fig_arms(runs, outpath, tag):
    arms = [a for a in ARM_ORDER if a in runs]
    n = len(arms)
    fig, axes = plt.subplots(1, 2, figsize=(4.6 + n * 1.05, 4.0))
    ax = axes[0]
    for i, a in enumerate(arms):
        m, s = mean_std([r["success_1step"] for r in runs[a].values()])
        ax.bar(i, m * 100, width=0.62, color=ARM_COLOR[a], alpha=0.92)
        ax.errorbar(i, m * 100, yerr=s * 100, fmt="none", ecolor="k", capsize=3)
        ax.text(i, m * 100 + 1.6, f"{m*100:.1f}", ha="center", fontsize=8.5)
    t10 = np.mean([r["teacher_success_10step"] for r in runs[arms[0]].values()])
    ax.axhline(t10 * 100, ls="--", lw=1.2, color="#2a9d8f",
               label=f"teacher 10-step inference = {t10*100:.1f}%")
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, loc="upper right")
    ax.set_xticks(range(n))
    ax.set_xticklabels([ARM_LABEL[a].replace("\\n", " ") for a in arms],
                       fontsize=7.5, rotation=14, ha="right")
    ax.set_ylabel("PushEnv success (%), 1-step inference")
    ax.set_ylim(60, 100)
    ax.set_title(f"one-step success rate, 200 episodes × "
                 f"{len(runs[arms[0]])} seeds ({tag})", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for i, a in enumerate(arms):
        m, s = mean_std([r["diag"]["global_err"] for r in runs[a].values()])
        ax.bar(i, m, width=0.62, color=ARM_COLOR[a], alpha=0.92)
        ax.errorbar(i, m, yerr=s, fmt="none", ecolor="k", capsize=3)
        ax.text(i, m * 1.15, f"{m:.4f}", ha="center", fontsize=8)
    t1 = np.mean([r["diag"]["teacher_1step_err"] for r in runs["bare"].values()])
    ax.axhline(t1, ls="--", lw=1.2, color="#2a9d8f")
    ax.text(0.97, t1 * 1.07, f"teacher's own 1-step error = {t1:.4f}", fontsize=8,
            ha="right", va="bottom", color="#2a9d8f", transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(n))
    ax.set_xticklabels([ARM_LABEL[a].replace("\\n", " ") for a in arms],
                       fontsize=7.5, rotation=14, ha="right")
    ax.set_ylabel(r"global endpoint error $\|\hat a_S^{1}-\hat a_T^{10}\|^2$")
    ax.set_title("global endpoint error (held-out, τ=0)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_gap(runs, outpath, tag):
    arms = [a for a in ["cd", "ta", "cdta", "cdta1"] if a in runs]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    panels = [(axes[0][0], "loss_cd_train", "CD loss  $\\|f_S-f_{ema}\\|^2$",
               "local objective: CD training loss (interval mean)"),
              (axes[0][1], "loss_ta_train", "TA loss  $\\|v_S-u_T\\|^2$",
               "global objective: TA training loss (interval mean)"),
              (axes[1][0], "local_err_data",
               "local consistency error  $\\|f_S(x_\\tau)-f_S(x_{\\tau+\\Delta})\\|^2$",
               "local consistency error, held-out (student self-consistency)"),
              (axes[1][1], "global_err",
               "global endpoint error  $\\|\\hat a^{1}_S-\\hat a^{10}_T\\|^2$",
               "global endpoint error at τ=0, held-out")]
    for ax, key, ylab, title in panels:
        for a in arms:
            series = []
            for r in sorted(runs[a].values(), key=lambda r: r["seed"]):
                pts = [(h["step"], h[key]) for h in r.get("history") or [] if key in h]
                if pts:
                    series.append(pts)
            if not series:
                continue
            n = min(len(s) for s in series)
            steps = np.array([p[0] for p in series[0][:n]])
            vals = np.array([[p[1] for p in s[:n]] for s in series], dtype=float)
            ax.plot(steps, vals.mean(0), color=ARM_COLOR[a], lw=1.8,
                    label=ARM_LABEL[a].replace("\n", " "))
            if vals.shape[0] > 1:
                ax.fill_between(steps, vals.min(0), vals.max(0), color=ARM_COLOR[a],
                                alpha=0.16, lw=0)
        if key in ("loss_cd_train", "loss_ta_train"):
            ax.set_yscale("log")
        if key == "global_err" and "bare" in runs:
            t1 = np.mean([r["diag"]["teacher_1step_err"] for r in runs["bare"].values()])
            ax.axhline(t1, ls=":", lw=1.4, color="#2a9d8f")
            ax.text(0.03, t1 * 0.9, "teacher init (bare-1step)", fontsize=8.5,
                    color="#2a9d8f", transform=ax.get_yaxis_transform())
            ax.set_yscale("log")
        ax.set_xlabel("distillation step")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.25, which="both")
    axes[0][0].legend(fontsize=8.5, frameon=False)
    fig.suptitle(f"local vs global error during distillation ({tag}, "
                 f"{len(runs[arms[0]])} seeds, band = min–max)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_premise(paths, outpath):
    cands = [(p, label, c) for p, label, c in paths if os.path.exists(p)]
    if not cands:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    for path, label, color in cands:
        d = json.load(open(path))
        sw = d["success_vs_samples"]
        ms = sorted(int(k) for k in sw if k != "1step")
        ax.plot(ms, [sw[str(m)] * 100 for m in ms], "o-", color=color, lw=1.9,
                label=f"{label}: 10-step teacher")
        one = sw["1step"] * 100
        ax.axhline(one, ls=":", lw=1.5, color=color)
        ax.scatter([32], [one], marker="*", s=130, color=color, zorder=5)
        ax.annotate(f"{label}: m=1 → {sw['1']*100:.1f}%,  m=2 → {sw['2']*100:.1f}%,\n"
                    f"1-step (★, m→∞ limit) = {one:.1f}%",
                    xy=(2.2, sw["2"] * 100 - 5.5), fontsize=8, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "1-step"])
    ax.set_xlabel("m = independent 10-step rollouts averaged per decision")
    ax.set_ylabel("PushEnv success (%)")
    ax.set_title("averaging the multi-step endpoint removes its deficit\n"
                 "(same ODE, same integration error, only the noise draw differs)",
                 fontsize=9.5)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1]
    for path, label, color in cands:
        d = json.load(open(path))
        sw = d["success_sweep"]
        xs = sorted(float(k.replace("1step+sigma", "")) for k in sw
                    if k.startswith("1step+sigma"))
        ax.plot(xs, [sw[f"1step+sigma{x}"] * 100 for x in xs], "o-", color=color,
                lw=1.9, label=f"{label}: 1-step + N(0,σ²)")
        ax.axhline(sw["10step"] * 100, ls="--", color=color, lw=1.4, alpha=0.85)
        rms = float(np.sqrt(d["dispersion_1step_vs_10step"]))
        ax.axvline(rms, ls=":", color=color, lw=1.3)
        ax.annotate(f"{label}: 10-step = {sw['10step']*100:.1f}%, "
                    f"rms(1step−10step) = {rms:.3f}", xy=(0.012, 70 if color == "#d1495b" else 64),
                    fontsize=8, color=color)
    ax.set_xlabel("injected Gaussian action noise σ on the 1-step prediction")
    ax.set_ylabel("PushEnv success (%)")
    ax.set_title("cross-check: isotropic noise of the same rms is WORSE than the\n"
                 "real multi-step error, so dispersion alone over-explains",
                 fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/clean")
    ap.add_argument("--tag", default="")
    ap.add_argument("--figs", default="out/figs")
    ap.add_argument("--premise", nargs="*", default=[])
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)
    runs = load(args.root)
    if not runs:
        raise SystemExit(f"no runs under {args.root}")
    summarize(runs, args.figs, args.tag)
    fig_arms(runs, os.path.join(args.figs, f"fig1_arms{args.tag}.png"), args.tag or "clean")
    fig_gap(runs, os.path.join(args.figs, f"fig2_gap{args.tag}.png"), args.tag or "clean")
    if args.premise:
        paths = [(p, os.path.splitext(os.path.basename(p))[0], c)
                 for p, c in zip(args.premise, ["#d1495b", "#00798c"])]
        fig_premise(paths, os.path.join(args.figs, "fig3_premise.png"))
    print("figures ->", args.figs)


if __name__ == "__main__":
    main()
