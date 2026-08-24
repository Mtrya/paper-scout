"""
Scenario 1 - Observability of the augmented state (extrinsic + focal length).

The core scientific question of experiment D: under which motion shapes is the
focal length jointly observable with the extrinsics? Classical camera
self-calibration says pure translation leaves f coupled with translation scale
/ scene depth (f ~ t), and rotation excites f. We test this in the eye-to-hand
fixed-camera setting (camera static, known base-frame feature points moving):

Motions (EE point only, so the point-cloud degeneracies are clean):
  line          : feature traces a straight segment (collinear cloud)
  line_frontal  : straight segment parallel to the camera image plane
                  (analytic prediction: exact f <-> t_z coupling family)
  plane         : Lissajous spanning a plane (coplanar cloud)
  volume        : 3-D Lissajous (space-spanning cloud)
  rot_arc       : pure rotation: yaw-only sweep (circular arc, coplanar)
  wave          : yaw + pitch sweep (curved 2-D patch, non-coplanar)

Per motion, over a sliding window of the augmented (7-D) Fisher information at
the TRUE parameters, we report: the eigenvalue spectrum (rank / nullity), the
raw focal information F_ff, the f-marginal information (Schur complement, i.e.
focal information after the extrinsics are marginalised), and the implied
joint f-CRB sigma_f = 100/sqrt(f_marg) %. The null directions for line /
line_frontal are verified against two analytic families:
  - rotation about the motion-line axis through a line point (exp-A family)
    with zero f component;
  - the f <-> t_z coupling: camera shifted along its optical axis with a
    log-f change, delta = (0, 0, 0, -w_z e_z, 1).

An EKF illustration on line motion (EE only, f init +10%) shows the filter
cannot recover f along the null direction while translation partially
converges.

Usage:  python scenario1_observability.py
Outputs: figures/fig1_observability.png, figures/fig1_line_ekf.png,
         results/scenario1.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

import sim_intrinsic as sim
from analyze import median_quantile, savefig, set_style

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

DT = 0.05
T = 60.0
SIGMA = 1.0
SEEDS = [0, 1, 2, 3]
MOTIONS = ["line", "line_frontal", "plane", "volume", "rot_arc", "wave"]
FIM_WINDOW = 120

MOTION_LABELS = {
    "line": "line (1-D, collinear)",
    "line_frontal": "frontal line (f~t_z coupling)",
    "plane": "plane Lissajous (2-D)",
    "volume": "volume Lissajous (3-D)",
    "rot_arc": "rot_arc (yaw only, arc)",
    "wave": "wave (yaw+pitch, 2-D patch)",
}


def null_cos_line(motion, seed, F):
    """Cos-similarity of the analytic null families vs the FIM null space."""
    if motion not in ("line", "line_frontal"):
        return 0.0, 0.0
    feats, meta = sim.build_features(motion, T, DT, seed, use_elbow=False)
    ua = sim.analytic_null_extrinsic_line(meta["v"], meta["p0"])
    uf = sim.analytic_null_focal_frontal_line(meta["v"], meta["p0"],
                                              *sim.true_extrinsic())
    nulls, _ = sim.null_space(F)
    if len(nulls) == 0:
        return 0.0, 0.0
    # projection onto the null space
    proj_a = np.linalg.norm(np.stack([(u @ ua) * u for u in nulls]).sum(0))
    proj_f = np.linalg.norm(np.stack([(u @ uf) * u for u in nulls]).sum(0))
    return float(proj_a), float(proj_f)


def main():
    set_style()
    R_true, t_true = sim.true_extrinsic()
    results = {}
    fig, axes = plt.subplots(len(MOTIONS), 2, figsize=(12, 15))
    for row, kind in enumerate(MOTIONS):
        min_eigs, f_margs, f_ffs, nullities = [], [], [], []
        cos_a, cos_f = [], []
        for seed in SEEDS:
            feats, _ = sim.build_features(kind, T, DT, seed, use_elbow=False)
            fim = sim.per_step_fim(feats, [(R_true, t_true)] * len(feats),
                                   500.0, 64.0, 64.0, SIGMA)
            ana = sim.windowed_fim_analysis(fim, window=FIM_WINDOW)
            tail = slice(-25, None)
            min_eigs.append(ana["min_eig"][tail].min())
            f_margs.append(ana["f_marginal"][tail].mean())
            f_ffs.append(ana["f_ff"][tail].mean())
            nullities.append(int(np.median(ana["nullity"][tail])))
            F = fim.sum(0)
            ca, cf = null_cos_line(kind, seed, F)
            cos_a.append(ca)
            cos_f.append(cf)
            # per-seed curves for the panels
            if seed == 0:
                t = ana["eigvals"][:, -1].copy()
                eig_trace0 = ana["eigvals"]
                fcrb0 = ana["sigma_f_crb_pct"]
        min_eigs = np.array(min_eigs)
        f_margs = np.array(f_margs)
        f_ffs = np.array(f_ffs)
        nullities = np.array(nullities)
        results[kind] = {
            "fim_min_eig": float(np.median(min_eigs)),
            "fim_nullity": int(np.median(nullities)),
            "f_ff": float(np.median(f_ffs)),
            "f_marginal": float(np.median(f_margs)),
            "sigma_f_crb_pct": float(100.0 / np.sqrt(max(np.median(f_margs), 1e-30))),
            "null_cos_extrinsic_line": float(np.mean(cos_a)),
            "null_cos_focal_tz": float(np.mean(cos_f)),
            "per_seed": [{"min_eig": float(m), "nullity": int(n),
                          "f_marginal": float(fm)} for m, n, fm in
                         zip(min_eigs, nullities, f_margs)],
        }

        # panel 1: eigenvalue spectrum band (full windowed spectrum, seed 0)
        ax = axes[row, 0]
        ev = eig_trace0
        med, lo, hi = median_quantile(ev.T)
        ax.plot(np.arange(1, len(med) + 1) * DT, med, lw=1.4, color="tab:blue")
        ax.fill_between(np.arange(1, len(med) + 1) * DT, lo, hi,
                        alpha=0.25, color="tab:blue")
        ax.set_yscale("log")
        ax.set_ylabel("windowed FIM eigenvalues")
        ax.set_title(f"{MOTION_LABELS[kind]}: eigenvalue spectrum")
        ax.grid(True, alpha=0.3)
        ax.text(0.03, 0.03,
                f"min eig = {results[kind]['fim_min_eig']:.2e}\n"
                f"nullity = {results[kind]['fim_nullity']}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="wheat", alpha=0.8))

        # panel 2: f-CRB (marginal) vs time
        ax = axes[row, 1]
        ax.plot(np.arange(1, len(fcrb0) + 1) * DT, fcrb0, lw=1.4, color="tab:red")
        ax.set_yscale("log")
        ax.set_ylabel("joint f-CRB (%, log f)")
        ax.set_title("focal marginal info (Schur): CRB vs time")
        ax.grid(True, alpha=0.3)
        s = results[kind]
        txt = f"F_ff raw = {s['f_ff']:.2e}\nf-marginal = {s['f_marginal']:.2e}\n"
        if kind in ("line", "line_frontal"):
            txt += (f"null cos: rot-about-line {s['null_cos_extrinsic_line']:.3f} | "
                    f"f~t_z {s['null_cos_focal_tz']:.3f}")
        else:
            txt += f"sigma_f CRB = {s['sigma_f_crb_pct']:.2f}%"
        ax.text(0.03, 0.55, txt, transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="wheat", alpha=0.8))
        if row == 0:
            axes[row, 0].legend(["windowed spectrum (median+IQR, 4 seeds)"],
                                fontsize=8, loc="upper right")
        if row == len(MOTIONS) - 1:
            for c in range(2):
                axes[row, c].set_xlabel("time (s)")
    fig.suptitle("Augmented (R, t, f) observability vs motion shape (EE point, sigma=1 px, "
                 "sliding window 6 s; f in log space)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    savefig(fig, os.path.join(FIGDIR, "fig1_observability.png"))

    # --- EKF illustration on line motion: f cannot converge ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 3.8))
    seed = 0
    feats, _ = sim.build_features("line", 120.0, DT, seed, use_elbow=False)
    T_true = sim.true_extrinsic()
    rec = sim.run_filter_simulation(
        feats, [T_true] * len(feats), np.full(len(feats), 500.0), 64.0, 64.0,
        SIGMA, sim.initial_guess(T_true, seed=seed), 550.0,
        sim.initial_covariance(7, logf_std=0.10),
        q_diag=np.array([3e-9] * 3 + [3e-10] * 3 + [1e-9]), seed=seed)
    t = rec["t"]
    for ax, key, scale, lab in [
        (axes2[0], "rot_err", 180 / np.pi, "rotation error (deg)"),
        (axes2[1], "trans_err", 100, "translation error (cm)"),
        (axes2[2], "f_err_pct", 1.0, "focal error (%)"),
    ]:
        ax.plot(t, rec[key] * scale, lw=1.4, color="tab:blue")
        ax.set_yscale("log")
        ax.set_xlabel("time (s)")
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
    axes2[2].axhline(10.0, color="k", ls=":", lw=1)
    axes2[2].text(0.5, 0.75, "init +10%", transform=axes2[2].transAxes, fontsize=8)
    axes2[1].text(0.03, 0.05,
                  "translation partially\nconverges (16-19 cm);\nf stays at ~+10%",
                  transform=axes2[1].transAxes, fontsize=8,
                  bbox=dict(facecolor="wheat", alpha=0.8))
    axes2[0].text(0.03, 0.05,
                  "rotation error stays at\nthe init projection (~19 deg):\n"
                  "the exp-A null direction is\ninherited by the augmented state",
                  transform=axes2[0].transAxes, fontsize=8,
                  bbox=dict(facecolor="wheat", alpha=0.8))
    fig2.suptitle("Line motion (EE only): the 2-D augmented null space blocks f recovery "
                  "(f init +10%, Q floor)", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig2, os.path.join(FIGDIR, "fig1_line_ekf.png"))

    with open(os.path.join(RESDIR, "scenario1.json"), "w") as f:
        json.dump({"motions": MOTION_LABELS, "results": results}, f, indent=2)

    print("=== Scenario 1: augmented (R,t,f) observability vs motion ===")
    for kind in MOTIONS:
        s = results[kind]
        extra = ""
        if kind in ("line", "line_frontal"):
            extra = (f" | null cos: rot-line {s['null_cos_extrinsic_line']:.3f}, "
                     f"f~t_z {s['null_cos_focal_tz']:.3f}")
        print(f"[{kind:12s}] min_eig={s['fim_min_eig']:.2e} nullity={s['fim_nullity']} "
              f"f_ff={s['f_ff']:.2e} f_marg={s['f_marginal']:.2e} "
              f"sigma_f_crb={s['sigma_f_crb_pct']:.2f}%{extra}")
    print("figures ->", os.path.join(FIGDIR, "fig1_observability.png"),
          os.path.join(FIGDIR, "fig1_line_ekf.png"))


if __name__ == "__main__":
    main()
