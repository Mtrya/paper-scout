"""
Scenario 4 - Plug-and-play: cold start of the full 9-D state after a camera
swap.

The pipeline swaps in another camera of the same model: the focal length is
off by +3% (catalog value 500 px vs true 515 px), the principal point is off
by a few pixels (true (66.5, 62.2) vs nominal (64, 64)), and the extrinsic is
completely different (the new unit is mounted at a different pose). The 9-D
filter (R, t, log f, cx, cy) starts from factory-nominal knowledge: f=500,
(cx, cy)=(64, 64), and the extrinsic only roughly known (10 deg / 15 cm off
the new truth - a "place the camera and eyeball the mount" cold start).

Questions:
  - how long does each parameter take to converge, per motion shape?
  - what is the minimal "self-calibration dance" (motion + duration)?
  - what happens from a larger extrinsic mismatch (25 deg / 35 cm): the online
    filter stays biased, and batch GN (9-D) is shown as the rescue path.

Convergence thresholds are CRB-aware: at T=480 s (volume, EE+elbow) the joint
CRBs are rot 0.68 deg / trans 3.0 cm / f 0.99% / cx 4.9 px / cy 3.6 px, so the
"converged" thresholds are ~2 sigma: rot<1.5 deg, trans<7 cm, f<3%,
cx<12 px, cy<9 px. The convergence time of a parameter is the LAST time it
crosses below the threshold (error stays below afterwards).

Usage:  python scenario4_plugplay.py
Outputs: figures/fig4_plugplay.png, results/scenario4.json
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
SIGMA = 1.0
SEEDS = [0, 1, 2]

# new camera geometry
CAM2_EYE = np.array([0.25, -1.85, 0.66])
CAM2_TARGET = np.array([0.28, 0.05, 0.10])
F2 = 515.0          # +3% vs catalog 500
CX2, CY2 = 66.5, 62.2

THR = {"rot": np.deg2rad(1.5), "trans": 0.07, "f": 3.0, "cx": 12.0, "cy": 9.0}
Q_9 = np.array([3e-9] * 3 + [3e-10] * 3 + [1e-9, 1e-9, 1e-9])
P0_COLD = sim.initial_covariance(9, rot_deg=15.0, trans=0.2,
                                 logf_std=0.12, px_std=6.0)


def new_camera():
    return sim.look_at(CAM2_EYE, CAM2_TARGET), CAM2_EYE.copy()


def run_cold(kind, T, seed, theta0_deg=10.0, d0=0.15):
    R2, t2 = new_camera()
    feats, _ = sim.build_features(kind, T, DT, seed, use_elbow=True)
    T0 = sim.initial_guess((R2, t2), theta0_deg=theta0_deg, d0=d0, seed=seed)
    return sim.run_filter_simulation(
        feats, [(R2, t2)] * len(feats), np.full(len(feats), F2),
        CX2, CY2, SIGMA, T0, 500.0, P0_COLD, cx0=64.0, cy0=64.0,
        q_diag=Q_9, seed=seed, n_iters=5)


def conv_time(v, thr, dt=DT):
    """Last time the series crosses below the threshold (stays below after)."""
    below = np.abs(np.asarray(v)) < thr
    if not below.any():
        return None
    idx = np.where(~below)[0]
    if len(idx) == 0:
        return 0.0
    return (idx[-1] + 1) * dt


def main():
    set_style()
    R2, t2 = new_camera()
    results = {"thresholds": {k: (v if k == "rot" else v) for k, v in THR.items()},
               "camera": {"f_true": F2, "cx_true": CX2, "cy_true": CY2,
                          "eye": CAM2_EYE.tolist(), "target": CAM2_TARGET.tolist()}}
    # per-motion CRBs (volume, EE+elbow) for the threshold justification
    feats0, _ = sim.build_features("volume", 480.0, DT, 0, use_elbow=True)
    H = np.concatenate([measurement_jacobian(F2, CX2, CY2, R2, t2, f, 9)
                        for f in feats0], axis=0)
    Fi = np.linalg.inv(H.T @ H)
    results["crb_480s"] = {
        "rot_deg": float(np.sqrt(Fi[0, 0] + Fi[1, 1] + Fi[2, 2]) * 180 / np.pi),
        "trans_cm": float(np.sqrt(Fi[3, 3] + Fi[4, 4] + Fi[5, 5]) * 100),
        "f_pct": float(np.sqrt(Fi[6, 6]) * 100),
        "cx_px": float(np.sqrt(Fi[7, 7])),
        "cy_px": float(np.sqrt(Fi[8, 8])),
    }

    motions = ["volume", "wave", "plane"]
    recs = {}
    for T in (240.0, 480.0):
        for kind in motions:
            outs = {}
            for se in SEEDS:
                rec = run_cold(kind, T, se)
                outs[se] = rec
            recs[(kind, T)] = outs
            keys = ("rot", "trans", "f", "cx", "cy")

            def arr_of(se, k):
                r = outs[se]
                return {"rot": r["rot_err"], "trans": r["trans_err"],
                        "f": np.abs(r["f_err_pct"]), "cx": np.abs(r["cx_err"]),
                        "cy": np.abs(r["cy_err"])}[k]

            times = {k: [conv_time(arr_of(se, k), THR[k]) for se in SEEDS]
                     for k in keys}
            i0 = int(0.7 * len(outs[SEEDS[0]]["t"]))
            steady = {k: float(np.mean([arr_of(se, k)[i0:].mean() for se in SEEDS]))
                      for k in keys}
            steady["rot_deg"] = steady.pop("rot") * 180 / np.pi
            steady["trans_cm"] = steady.pop("trans") * 100
            steady["f_pct"] = steady.pop("f")
            steady["cx_px"] = steady.pop("cx")
            steady["cy_px"] = steady.pop("cy")
            # ALL-params convergence: max over {rot, trans, f} of per-seed times
            all_t = [max(t for t in (times["rot"][i], times["trans"][i],
                                     times["f"][i]) if t is not None)
                     if all(x is not None for x in (times["rot"][i], times["trans"][i],
                                                    times["f"][i])) else None
                     for i in range(len(SEEDS))]
            results[f"{kind}_{int(T)}s"] = {
                "steady": steady,
                "conv_time_s": {k: [x for x in v] for k, v in times.items()},
                "conv_mean_s": {k: float(np.mean([x for x in v if x is not None]))
                                if any(x is not None for x in v) else None
                                for k, v in times.items()},
                "all_rot_trans_f_s": all_t,
            }
            print(f"[{kind:7s} T={T:4.0f}] steady: rot {steady['rot_deg']:.2f} deg | "
                  f"trans {steady['trans_cm']:.2f} cm | f {steady['f_pct']:.2f}% | "
                  f"cx {steady['cx_px']:.2f} | cy {steady['cy_px']:.2f}")
            print(f"          conv last-cross (s): "
                  f"rot {times['rot']} | trans {times['trans']} | f {times['f']} | "
                  f"cx {times['cx']} | cy {times['cy']} | ALL(rot,trans,f) {all_t}")

    # minimal dance: volume at 480 s, max over seeds of the ALL time
    all_480 = results["volume_480s"]["all_rot_trans_f_s"]
    results["minimal_dance"] = {
        "motion": "volume",
        "T_s": 480.0,
        "all_rot_trans_f_s": all_480,
        "dance_s": float(np.max([t for t in all_480 if t is not None]))
        if any(t is not None for t in all_480) else None,
    }
    print("\nminimal dance (volume, rot<1.5deg, trans<7cm, f<3%):",
          results["minimal_dance"])

    # --- large-mismatch cold start: online vs batch GN ---
    rec = run_cold("volume", 480.0, 0, theta0_deg=25.0, d0=0.35)
    i0 = int(0.7 * len(rec["t"]))
    results["big_mismatch_online"] = {
        "rot_deg": float(rec["rot_err"][i0:].mean() * 180 / np.pi),
        "trans_cm": float(rec["trans_err"][i0:].mean() * 100),
        "f_pct": float(np.abs(rec["f_err_pct"][i0:]).mean()),
        "cx_px": float(np.abs(rec["cx_err"][i0:]).mean()),
        "cy_px": float(np.abs(rec["cy_err"][i0:]).mean()),
        "nees": float(rec["nees"][i0:].mean()),
    }
    print("big-mismatch online (25deg/0.35m):", results["big_mismatch_online"])

    from batch_gn_intrinsic import batch_gn
    feats_b, _ = sim.build_features("volume", 480.0, DT, 0, use_elbow=True)
    _, e9 = batch_gn(feats_b, SIGMA, seed=0, theta0_deg=25.0, d0=0.35,
                     f_bias=0.03, cx_bias=2.5, cy_bias=-1.8, ndim=9)
    results["big_mismatch_batch_gn"] = {
        "rot_deg": float(np.linalg.norm(e9[:3]) * 180 / np.pi),
        "trans_cm": float(np.linalg.norm(e9[3:6]) * 100),
        "f_pct": float(np.abs(e9[6]) * 100),
        "cx_px": float(np.abs(e9[7])),
        "cy_px": float(np.abs(e9[8])),
    }
    print("big-mismatch batch GN (9-D):", results["big_mismatch_batch_gn"])

    # --- figure ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    cols = [("rot_err", "rotation error (deg)", 180 / np.pi, THR["rot"]),
            ("trans_err", "translation error (cm)", 100, THR["trans"]),
            ("f_err_pct", "|focal error| (%)", 1.0, THR["f"]),
            ("cx_err", "|cx error| (px)", 1.0, THR["cx"]),
            ("cy_err", "|cy error| (px)", 1.0, THR["cy"])]
    mcolors = {"volume": "tab:blue", "wave": "tab:orange", "plane": "tab:green"}
    for k, (key, lab, scale, thr) in enumerate(cols):
        ax = axes[k // 3, k % 3]
        for kind in motions:
            for se in SEEDS:
                rec = recs[(kind, 480.0)][se]
                lw = 0.8 if se else 1.5
                ax.plot(rec["t"], np.abs(rec[key]) * scale, lw=lw, alpha=0.55,
                        color=mcolors[kind])
        ax.axhline(thr * scale, color="k", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("time (s)")
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
    for kind in motions:
        axes[0, 0].plot([], [], color=mcolors[kind], lw=1.6, label=kind)
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 2]
    ax.axis("off")
    d = results["minimal_dance"]
    txt = (f"self-calibration dance (cold start, 9-D)\n\n"
           f"volume 480 s: ALL(rot,trans,f) conv =\n  {d['all_rot_trans_f_s']} s\n"
           f"(rot<1.5 deg, trans<7 cm, f<3%)\n\n"
           f"principal point is the slowest: 480 s\n"
           f"CRB cx {results['crb_480s']['cx_px']:.1f} px | "
           f"cy {results['crb_480s']['cy_px']:.1f} px\n"
           f"(cx ~ t_x coupling: keep the factory\n"
           f"principal point, estimate only f)\n\n"
           f"25 deg/35 cm mismatch: online stuck\n"
           f"(f {results['big_mismatch_online']['f_pct']:.1f}%, NEES "
           f"{results['big_mismatch_online']['nees']:.0f});\n"
           f"batch GN (9-D) rescues: f "
           f"{results['big_mismatch_batch_gn']['f_pct']:.2f}%, "
           f"trans {results['big_mismatch_batch_gn']['trans_cm']:.2f} cm")
    ax.text(0.0, 1.0, txt, va="top", fontsize=9,
            bbox=dict(facecolor="wheat", alpha=0.9))
    fig.suptitle("Plug-and-play cold start after a camera swap (f +3%, principal point off, "
                 "new extrinsic; 9-D IEKF, factory-nominal init 10deg/15cm)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, os.path.join(FIGDIR, "fig4_plugplay.png"))

    with open(os.path.join(RESDIR, "scenario4.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("figure ->", os.path.join(FIGDIR, "fig4_plugplay.png"))


if __name__ == "__main__":
    main()
