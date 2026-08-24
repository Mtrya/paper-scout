"""
Scenario 2 - Joint estimation accuracy (extrinsic + focal length).

Question: how much does adding f to the online state degrade the extrinsic
estimate, and can a +/-10% focal initialisation be corrected online?

Volume motion + EE+elbow, sigma=1 px, 4 seeds, T=180 s. Configs:
  (a) extrinsic-only EKF (6-D, f known)            - baseline (exp A quality)
  (b) joint EKF, f init 0%                         - coupling cost at correct f
  (c) joint EKF, f init +10% / -10% (big P0)       - the hard cold-ish start
  (d) joint EKF, f init +10% with extrinsic held   - "f re-calibration" (small
                                                      extrinsic P0)
  (e) batch GN (7-D) from the same init as (c)     - reference optimum

Theoretical anchor: the 6-D vs 7-D Fisher CRB at the true parameters (180 s).

The headline: the joint problem is dominated by an f ~ translation coupling
(translation along the viewing direction absorbs focal information). Joint CRB:
translation degrades ~50x vs extrinsic-only, and a +/-10% focal init drives the
online filter into a biased local optimum (the exp-A lesson recurs for f),
while batch GN reaches the (noise-shifted) CRB.

Usage:  python scenario2_joint.py
Outputs: figures/fig2_joint_accuracy.png, results/scenario2.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

import sim_intrinsic as sim
from ekf_intrinsic import measurement_jacobian
from analyze import savefig, set_style

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

DT = 0.05
T = 180.0
SIGMA = 1.0
SEEDS = [0, 1, 2, 3]

# extrinsic-only filter (exp A machinery) for the baseline
import ekf6
from ekf6 import HandEyeEKF, project_points


def run_extrinsic_only(feats, sigma, T0, P0, q_diag, seed):
    rng = np.random.default_rng(seed)
    R_true, t_true = sim.true_extrinsic()
    ekf = HandEyeEKF(sim.make_K(), T0, P0, q_diag=q_diag)
    errs = []
    for fts in feats:
        pix = project_points(sim.make_K(), R_true, t_true, fts) + \
            rng.normal(0.0, sigma, size=(len(fts), 2))
        ekf.predict()
        ekf.update(fts, pix, sigma)
        Re = ekf.T_R @ R_true.T
        te = ekf.T_t - Re @ t_true
        errs.append(sim.se3_log(Re, te))
    errs = np.array(errs)
    return errs


def steady(err, frac=0.3):
    i0 = int(len(err) * (1 - frac))
    return err[i0:]


def main():
    set_style()
    R_true, t_true = sim.true_extrinsic()
    feats_map = {se: sim.build_features("volume", T, DT, se, use_elbow=True)[0]
                 for se in SEEDS}
    T_seq = [(R_true, t_true)]
    Q_EX = np.array([3e-9] * 3 + [3e-10] * 3)      # exp-A Q floor
    Q_JOINT = np.array([3e-9] * 3 + [3e-10] * 3 + [1e-9])

    # --- theoretical CRB (volume + elbow, 180 s, sigma=1) ---
    feats0 = feats_map[0]
    H7 = np.concatenate([measurement_jacobian(500.0, 64.0, 64.0, R_true, t_true, f, 7)
                         for f in feats0], axis=0)
    F6i = np.linalg.inv(H7[:, :6].T @ H7[:, :6])
    F7i = np.linalg.inv(H7.T @ H7)
    crb = {
        "ext_only_rot_deg": float(np.sqrt(F6i[0, 0] + F6i[1, 1] + F6i[2, 2]) * 180 / np.pi),
        "ext_only_trans_cm": float(np.sqrt(F6i[3, 3] + F6i[4, 4] + F6i[5, 5]) * 100),
        "joint_rot_deg": float(np.sqrt(F7i[0, 0] + F7i[1, 1] + F7i[2, 2]) * 180 / np.pi),
        "joint_trans_cm": float(np.sqrt(F7i[3, 3] + F7i[4, 4] + F7i[5, 5]) * 100),
        "joint_f_pct": float(np.sqrt(F7i[6, 6]) * 100),
    }

    results = {"crb": crb, "configs": {}}
    recs = {}

    def fin(rec):
        return {k: rec[k][int(len(rec["t"]) * 0.7):] for k in
                ("rot_err", "trans_err", "f_err_pct")}

    # (a) extrinsic-only baseline
    cfg = "ext_only"
    outs = {}
    for se in SEEDS:
        errs = run_extrinsic_only(feats_map[se], SIGMA,
                                  sim.initial_guess((R_true, t_true), seed=se),
                                  sim.initial_covariance(6), Q_EX, se)
        outs[se] = errs
    results["configs"][cfg] = {
        "rot_deg": float(np.mean([np.linalg.norm(steady(e)[:, :3], axis=1).mean()
                                  for e in outs.values()]) * 180 / np.pi),
        "trans_cm": float(np.mean([np.linalg.norm(steady(e)[:, 3:], axis=1).mean()
                                   for e in outs.values()]) * 100),
    }
    recs[cfg] = outs
    # normalise the raw arrays into the plot dict shape
    recs[cfg] = {
        se: {"rot_err": errs[:, :3], "trans_err": errs[:, 3:],
             "f_err_pct": np.zeros(len(errs))}
        for se, errs in outs.items()
    }

    # (b) joint, f init 0%
    cfg = "joint_f0"
    outs = {}
    for se in SEEDS:
        rec = sim.run_filter_simulation(
            feats_map[se], [T_seq[0]] * len(feats_map[se]),
            np.full(len(feats_map[se]), 500.0), 64.0, 64.0, SIGMA,
            sim.initial_guess((R_true, t_true), seed=se), 500.0,
            sim.initial_covariance(7, logf_std=0.05), q_diag=Q_JOINT, seed=se)
        outs[se] = rec
    results["configs"][cfg] = {
        "rot_deg": float(np.mean([fin(r)["rot_err"].mean() for r in outs.values()]) * 180 / np.pi),
        "trans_cm": float(np.mean([fin(r)["trans_err"].mean() for r in outs.values()]) * 100),
        "f_pct": float(np.mean([np.abs(fin(r)["f_err_pct"]).mean() for r in outs.values()])),
        "nees": float(np.mean([r["nees"][int(0.7 * len(r["t"])):].mean() for r in outs.values()])),
    }
    recs[cfg] = outs

    # (c) joint, f init +10% / -10% (big P0): the hard case
    for fb, f0 in (("joint_f+10", 550.0), ("joint_f-10", 450.0)):
        cfg = fb
        outs = {}
        for se in SEEDS:
            rec = sim.run_filter_simulation(
                feats_map[se], [T_seq[0]] * len(feats_map[se]),
                np.full(len(feats_map[se]), 500.0), 64.0, 64.0, SIGMA,
                sim.initial_guess((R_true, t_true), seed=se), f0,
                sim.initial_covariance(7, logf_std=0.10), q_diag=Q_JOINT, seed=se)
            outs[se] = rec
        results["configs"][cfg] = {
            "rot_deg": float(np.mean([fin(r)["rot_err"].mean() for r in outs.values()]) * 180 / np.pi),
            "trans_cm": float(np.mean([fin(r)["trans_err"].mean() for r in outs.values()]) * 100),
            "f_pct": float(np.mean([np.abs(fin(r)["f_err_pct"]).mean() for r in outs.values()])),
            "nees": float(np.mean([r["nees"][int(0.7 * len(r["t"])):].mean() for r in outs.values()])),
        }
        recs[cfg] = outs

    # (d) joint, f init +10% with extrinsic held (small extrinsic P0): re-calibration
    cfg = "joint_f+10_hold"
    outs = {}
    P0_hold = np.diag([np.deg2rad(0.5) ** 2] * 3 + [0.005 ** 2] * 3 + [0.10 ** 2])
    for se in SEEDS:
        rec = sim.run_filter_simulation(
            feats_map[se], [T_seq[0]] * len(feats_map[se]),
            np.full(len(feats_map[se]), 500.0), 64.0, 64.0, SIGMA,
            (R_true.copy(), t_true.copy()), 550.0, P0_hold,
            q_diag=np.array([1e-10] * 3 + [1e-11] * 3 + [1e-9]), seed=se, n_iters=5)
        outs[se] = rec
    results["configs"][cfg] = {
        "rot_deg": float(np.mean([fin(r)["rot_err"].mean() for r in outs.values()]) * 180 / np.pi),
        "trans_cm": float(np.mean([fin(r)["trans_err"].mean() for r in outs.values()]) * 100),
        "f_pct": float(np.mean([np.abs(fin(r)["f_err_pct"]).mean() for r in outs.values()])),
        "nees": float(np.mean([r["nees"][int(0.7 * len(r["t"])):].mean() for r in outs.values()])),
    }
    recs[cfg] = outs

    # (e) batch GN reference from the +10% init
    from batch_gn_intrinsic import batch_gn
    bf, brot, btr = [], [], []
    for se in SEEDS:
        _, e7 = batch_gn(feats_map[se], SIGMA, seed=se, theta0_deg=30.0, d0=0.3,
                         f_bias=0.10)
        bf.append(e7[6])
        brot.append(np.linalg.norm(e7[:3]))
        btr.append(np.linalg.norm(e7[3:6]))
    results["configs"]["batch_gn_f+10"] = {
        "rot_deg": float(np.mean(brot) * 180 / np.pi),
        "trans_cm": float(np.mean(btr) * 100),
        "f_pct": float(np.mean(np.abs(bf)) * 100),
    }

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"ext_only": "tab:gray", "joint_f0": "tab:blue",
              "joint_f+10": "tab:red", "joint_f-10": "tab:orange",
              "joint_f+10_hold": "tab:green"}
    t = np.arange(len(feats_map[0])) * DT
    for cfg, col in colors.items():
        for se in SEEDS:
            rec = recs[cfg][se]
            lw = 0.7 if se else 1.4
            re = rec["rot_err"] if rec["rot_err"].ndim == 1 else \
                np.linalg.norm(rec["rot_err"], axis=1)
            te = rec["trans_err"] if rec["trans_err"].ndim == 1 else \
                np.linalg.norm(rec["trans_err"], axis=1)
            axes[0].plot(t, re * 180 / np.pi, color=col, lw=lw, alpha=0.5)
            axes[1].plot(t, te * 100, color=col, lw=lw, alpha=0.5)
            if cfg in ("joint_f0", "joint_f+10", "joint_f-10", "joint_f+10_hold"):
                axes[2].plot(t, np.abs(rec["f_err_pct"]), color=col, lw=lw, alpha=0.5)
    labels = {
        "ext_only": "extrinsic-only (f known)",
        "joint_f0": "joint, f init 0%",
        "joint_f+10": "joint, f init +10%",
        "joint_f-10": "joint, f init -10%",
        "joint_f+10_hold": "joint, +10%, extrinsic held",
    }
    for ax, key, lab in [(axes[0], "rot_err", "rotation error (deg)"),
                         (axes[1], "trans_err", "translation error (cm)"),
                         (axes[2], "f_err_pct", "|focal error| (%)")]:
        ax.set_yscale("log")
        ax.set_xlabel("time (s)")
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
    for cfg, col in colors.items():
        axes[0].plot([], [], color=col, lw=1.6, label=labels[cfg])
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot([], [], color="tab:green", lw=1.6, ls="--",
                 label="batch GN ref (f+10%%): trans %.2f cm, f %.2f%%" % (
                     results["configs"]["batch_gn_f+10"]["trans_cm"],
                     results["configs"]["batch_gn_f+10"]["f_pct"]))
    axes[1].legend(fontsize=8, loc="upper right")
    axes[2].text(0.03, 0.55,
                 "CRB: ext-only trans 0.10 cm  ->  joint trans %.2f cm\n"
                 "joint f CRB = %.2f %%\n"
                 "big-P0 + f init 10%% -> biased local optimum\n"
                 "(NEES %.0f); extrinsic held -> f %.2f %%" % (
                     crb["joint_trans_cm"], crb["joint_f_pct"],
                     results["configs"]["joint_f+10"]["nees"],
                     results["configs"]["joint_f+10_hold"]["f_pct"]),
                 transform=axes[2].transAxes, fontsize=8,
                 bbox=dict(facecolor="wheat", alpha=0.8))
    fig.suptitle("Joint (R,t,f) estimation, volume motion + EE+elbow, sigma=1 px, "
                 "T=180 s, 4 seeds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, os.path.join(FIGDIR, "fig2_joint_accuracy.png"))

    with open(os.path.join(RESDIR, "scenario2.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=== Scenario 2: joint estimation accuracy ===")
    print("CRB (180 s, volume+elbow):", json.dumps(crb, indent=1))
    for cfg in results["configs"]:
        s = results["configs"][cfg]
        print(f"[{cfg:16s}] rot {s['rot_deg']:.3f} deg | trans {s['trans_cm']:.3f} cm | "
              + (f"f {s['f_pct']:.3f} % | nees {s.get('nees', float('nan')):.0f}"
                 if "f_pct" in s else ""))
    print("figure ->", os.path.join(FIGDIR, "fig2_joint_accuracy.png"))


if __name__ == "__main__":
    main()
