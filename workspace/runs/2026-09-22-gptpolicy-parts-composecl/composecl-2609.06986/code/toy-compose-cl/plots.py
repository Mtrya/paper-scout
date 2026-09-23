"""Aggregate run_*.json -> figures + summary tables.

Outputs (in the results dir):
  fig1_final_retention.png   six-arm final retention, mean +/- std over seeds
  fig2_decay.png             retention vs memory age R(a), six arms
  fig3_trajectories.png      per-task accuracy vs age, C vs D (merge anatomy)
  fig4_grad_cosine.png       replay/current gradient cosine probe (arm D, task 20)
  fig5_si_mismatch.png       SI penalty at task start + Omega corr (arms E/F)
  aggregates.json            all summary numbers
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_ORDER = ["A_vanilla", "B_merge", "C_replay", "D_replay_merge",
             "E_si_merge", "F_si_replay_merge"]
LABELS = {"A_vanilla": "A vanilla", "B_merge": "B merge",
          "C_replay": "C replay", "D_replay_merge": "D replay+merge",
          "E_si_merge": "E SI+merge", "F_si_replay_merge": "F SI+replay+merge"}


def load(results_dir):
    runs = {}
    for fn in sorted(os.listdir(results_dir)):
        if fn.startswith("run_") and fn.endswith(".json"):
            with open(os.path.join(results_dir, fn)) as f:
                r = json.load(f)
            runs.setdefault(r["arm"], []).append(r)
    return runs


def age_curve(M):
    """R(a) = mean M[i][j] over i-j=a, for a = 0..T-1."""
    T = len(M)
    out = []
    for a in range(T):
        vals = [M[i][j] for i in range(T) for j in range(T)
                if i - j == a and M[i][j] is not None]
        out.append(float(np.mean(vals)))
    return out


def half_life(R):
    r0 = R[0]
    for a in range(1, len(R)):
        if R[a] < r0 / 2:
            frac = (R[a - 1] - r0 / 2) / max(R[a - 1] - R[a], 1e-9)
            return a - 1 + frac
    return float(len(R) - 1)


def decay_slope(M):
    """Pooled regression of M[i][j] on age i-j (per-task acc decay)."""
    xs, ys = [], []
    T = len(M)
    for i in range(T):
        for j in range(i + 1):
            if M[i][j] is not None:
                xs.append(i - j)
                ys.append(M[i][j])
    xs, ys = np.array(xs, float), np.array(ys, float)
    return float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs))


def main(results_dir):
    runs = load(results_dir)
    missing = [a for a in ARM_ORDER if a not in runs]
    if missing:
        print(f"missing arms: {missing}", file=sys.stderr)
    arms = [a for a in ARM_ORDER if a in runs]
    seeds = sorted({r["seed"] for a in arms for r in runs[a]})

    agg = {}
    for a in arms:
        finals = [r["final"] for r in runs[a]]
        agg[a] = dict(finals=finals, mean=float(np.mean(finals)),
                      std=float(np.std(finals)),
                      diag_acc=float(np.mean([r["diag_acc"] for r in runs[a]])),
                      forget=float(np.mean([r["forget"] for r in runs[a]])),
                      half_life=float(np.mean([half_life(age_curve(r["M"]))
                                               for r in runs[a]])),
                      decay_slope=float(np.mean([decay_slope(r["M"])
                                                 for r in runs[a]])))

    print("\n=== final retention (exact match) ===")
    print(f"{'arm':<20}{'per-seed':<22}{'mean':>7}{'std':>7}"
          f"{'acq':>7}{'forget':>8}{'half-life':>10}{'slope':>8}")
    for a in arms:
        g = agg[a]
        print(f"{LABELS[a]:<20}{str([round(x,3) for x in g['finals']]):<22}"
              f"{g['mean']:>7.3f}{g['std']:>7.3f}{g['diag_acc']:>7.3f}"
              f"{g['forget']:>8.3f}{g['half_life']:>10.1f}"
              f"{g['decay_slope']:>8.4f}")

    m = {a: agg[a]["mean"] for a in arms}
    inter = {}
    if all(k in m for k in ("A_vanilla", "B_merge", "C_replay", "D_replay_merge")):
        gain_R = m["C_replay"] - m["A_vanilla"]
        gain_M = m["B_merge"] - m["A_vanilla"]
        gain_RM = m["D_replay_merge"] - m["A_vanilla"]
        inter["R_x_M"] = dict(gain_replay=gain_R, gain_merge=gain_M,
                              gain_sum=gain_R + gain_M, gain_combined=gain_RM,
                              interaction=gain_RM - gain_R - gain_M,
                              super_additive=bool(gain_RM > gain_R + gain_M))
        print("\n=== R x M super-additivity ===")
        print(f"replay alone  {gain_R*100:+.2f} pp | merge alone {gain_M*100:+.2f} pp "
              f"| sum {(gain_R+gain_M)*100:+.2f} pp | combined {gain_RM*100:+.2f} pp "
              f"| interaction {(gain_RM-gain_R-gain_M)*100:+.2f} pp "
              f"-> super-additive: {gain_RM > gain_R + gain_M}")
    if all(k in m for k in ("B_merge", "E_si_merge", "D_replay_merge",
                            "F_si_replay_merge")):
        inter["SI_x_M"] = dict(si_added_to_merge=m["E_si_merge"] - m["B_merge"],
                               si_added_to_replay_merge=m["F_si_replay_merge"]
                               - m["D_replay_merge"])
        print("\n=== SI x M interaction (proxies; no SI-alone arm) ===")
        print(f"SI added to merge-only:   {(m['E_si_merge']-m['B_merge'])*100:+.2f} pp")
        print(f"SI added to replay+merge: {(m['F_si_replay_merge']-m['D_replay_merge'])*100:+.2f} pp")

    # ---- fig 1: final retention bars
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(arms)), [agg[a]["mean"] * 100 for a in arms],
           yerr=[agg[a]["std"] * 100 for a in arms], capsize=4,
           color=["#999", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"])
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_ylabel("final retention (%)")
    ax.set_title("Final retention after all tasks (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fig1_final_retention.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2: decay curves R(age)
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in arms:
        curves = np.array([age_curve(r["M"]) for r in runs[a]])
        ax.plot(range(curves.shape[1]), curves.mean(0) * 100,
                label=LABELS[a])
    ax.set_xlabel("memory age (tasks learned since)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Retention vs memory age R(a)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fig2_decay.png"), dpi=150)
    plt.close(fig)

    # ---- fig 3: per-task trajectories, C vs D
    T = len(runs[arms[0]][0]["M"])
    show = [j for j in (0, 4, 9, 14, 19) if j < T - 1]
    fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 3),
                             sharey=True)
    colors = {"C_replay": "#1b9e77", "D_replay_merge": "#d95f02"}
    for ax, j in zip(axes, show):
        for a, style in (("C_replay", "-o"), ("D_replay_merge", "-s")):
            if a not in runs:
                continue
            for k, r in enumerate(runs[a]):
                ys = [r["M"][j + a_][j] * 100 for a_ in range(T - j)]
                lbl = LABELS[a] if (j == show[0] and k == 0) else None
                ax.plot(range(len(ys)), ys, style, color=colors[a],
                        alpha=0.45 + 0.45 * (k == 0), label=lbl,
                        markersize=3)
        ax.set_title(f"task {j+1}", fontsize=10)
        ax.set_xlabel("age")
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Merge anatomy: task accuracy vs age, C (shared) vs D (merged)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fig3_trajectories.png"), dpi=150)
    plt.close(fig)

    # ---- fig 4: gradient cosine probe (arm D, task 20)
    diags = [r["grad_diag"] for r in runs.get("D_replay_merge", [])
             if r["grad_diag"]["cos"]]
    if diags:
        cos = np.concatenate([d["cos"] for d in diags])
        rep_self = np.concatenate([d["rep_selfcos"] for d in diags])
        sft_self = np.concatenate([d["sft_selfcos"] for d in diags])
        rep_norm = np.concatenate([d["rep_norm"] for d in diags])
        sft_norm = np.concatenate([d["sft_norm"] for d in diags])
        agg["grad_probe"] = dict(
            n=int(len(cos)), cos_mean=float(cos.mean()),
            cos_std=float(cos.std()),
            cos_neg_frac=float((cos < 0).mean()),
            rep_selfcos_mean=float(rep_self.mean()),
            sft_selfcos_mean=float(sft_self.mean()),
            rep_norm_mean=float(rep_norm.mean()),
            sft_norm_mean=float(sft_norm.mean()),
            norm_ratio=float(rep_norm.mean() / max(sft_norm.mean(), 1e-12)))
        print("\n=== gradient probe (arm D, task 20) ===")
        print(f"cos(g_replay, g_sft): mean {cos.mean():+.3f} std {cos.std():.3f} "
              f"| frac<0 {(cos<0).mean():.2f} (n={len(cos)})")
        print(f"self-consistency: replay {rep_self.mean():+.3f} "
              f"| sft {sft_self.mean():+.3f}")
        print(f"grad norms: replay {rep_norm.mean():.3f} sft {sft_norm.mean():.3f} "
              f"ratio {rep_norm.mean()/max(sft_norm.mean(),1e-12):.3f}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(cos, bins=30, alpha=0.7, label="cos(g_replay, g_sft)")
        ax.hist(rep_self, bins=30, alpha=0.5,
                label="cos(g_replay_t, g_replay_t+1)")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("cosine similarity")
        ax.set_title(f"Replay gradient probe, arm D task 20 "
                     f"(mean cos {cos.mean():+.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "fig4_grad_cosine.png"), dpi=150)
        plt.close(fig)

    # ---- fig 5: SI coordinate mismatch (arms E/F)
    si_runs = {a: [r["si_log"] for r in runs[a] if r["si_log"]]
               for a in ("E_si_merge", "F_si_replay_merge") if a in runs}
    si_runs = {a: v for a, v in si_runs.items() if v}
    if si_runs:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        for a, logs in si_runs.items():
            pen = np.array([l["penalty_at_start"] for l in logs])
            corr = np.array([l["omega_corr"] for l in logs])
            rd = np.array([l["ref_dist"] for l in logs])
            x = np.arange(1, pen.shape[1] + 1)
            axes[0].plot(x, pen.mean(0), "-o", markersize=3, label=LABELS[a])
            axes[1].plot(np.arange(2, corr.shape[1] + 2), corr.mean(0),
                         "-o", markersize=3, label=LABELS[a])
            axes[2].plot(np.arange(2, rd.shape[1] + 2), rd.mean(0),
                         "-o", markersize=3, label=LABELS[a])
            agg[a]["si_penalty_at_start_mean"] = float(pen[:, 1:].mean())
            agg[a]["si_omega_corr_mean"] = float(corr.mean())
        axes[0].set_title("SI penalty at task start\n(>0 after merge = ref mismatch)")
        axes[1].set_title("corr(Omega_in, omega_new)\n(old importance vs fresh-coord importance)")
        axes[2].set_title("mean (theta_init - theta*_old)^2\nat task start")
        for ax in axes:
            ax.set_xlabel("task")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "fig5_si_mismatch.png"), dpi=150)
        plt.close(fig)
        print("\n=== SI x merge coordinate mismatch ===")
        for a in si_runs:
            print(f"{LABELS[a]}: penalty-at-start mean "
                  f"{agg[a]['si_penalty_at_start_mean']:.4f}, "
                  f"Omega-vs-fresh-importance corr mean "
                  f"{agg[a]['si_omega_corr_mean']:+.3f}")

    agg["_interactions"] = inter
    with open(os.path.join(results_dir, "aggregates.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nwrote {os.path.join(results_dir, 'aggregates.json')}")
    return agg, inter


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
