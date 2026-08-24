"""Quick diagnostic: single-seed trajectories for the 9-D cold start.

Print per-parameter convergence behaviour (median over the last 10% vs the
3% / 7 cm / 1.5 deg thresholds) and the late-time trend of f / cx / cy, so the
README narrative can be written against actual traces.  Not part of the run.
"""
import numpy as np
import sim_intrinsic as sim
import scenario4_plugplay as s4

DT = 0.05


def summarize(kind, T, seed, theta0_deg=10.0, d0=0.15):
    rec = s4.run_cold(kind, T, seed, theta0_deg=theta0_deg, d0=d0)
    t = rec["t"]
    i0 = int(0.9 * len(t))
    f_last = rec["f_err_pct"][i0:]
    out = {
        "rot_deg": float(rec["rot_err"][i0:].mean() * 180 / np.pi),
        "trans_cm": float(rec["trans_err"][i0:].mean() * 100),
        "f_pct": float(np.abs(f_last).mean()),
        "f_pct_start_mid_end": [float(rec["f_err_pct"][0]), float(rec["f_err_pct"][len(t)//2]),
                                float(f_last[-1])],
        "cx_px": float(np.abs(rec["cx_err"][i0:]).mean()),
        "cy_px": float(np.abs(rec["cy_err"][i0:]).mean()),
    }
    # trend of |f| in the second half: still decreasing?
    j0, j1 = len(t)//2, len(t)
    f2 = np.abs(rec["f_err_pct"][j0:j1])
    out["f_mean_2nd_half_pct"] = float(f2.mean())
    out["f_mean_last10pct_pct"] = float(np.abs(f_last).mean())
    return out


if __name__ == "__main__":
    for kind, se in (("volume", 0), ("volume", 1), ("volume", 2), ("wave", 1)):
        r = summarize(kind, 480.0, se)
        print(f"{kind} seed{se}: rot {r['rot_deg']:.2f} deg | trans {r['trans_cm']:.2f} cm | "
              f"|f| {r['f_pct']:.2f}% (start/mid/end {r['f_pct_start_mid_end']}) | "
              f"|f| 2nd-half {r['f_mean_2nd_half_pct']:.2f}% / last10 {r['f_mean_last10pct_pct']:.2f}% | "
              f"cx {r['cx_px']:.2f} | cy {r['cy_px']:.2f}")
    # long-run f trend for volume seed 0: does f eventually dip below 3%?
    for T in (600.0, 900.0):
        r = s4.run_cold("volume", T, 0)
        i0 = int(0.9 * len(r["t"]))
        print(f"volume seed0 T={T}: |f| last10% {np.abs(r['f_err_pct'][i0:]).mean():.2f}% | "
              f"last value {r['f_err_pct'][-1]:+.2f}% | trans last {r['trans_err'][i0:].mean()*100:.1f} cm | "
              f"rot last {r['rot_err'][i0:].mean()*180/np.pi:.2f} deg | cx {np.abs(r['cx_err'][i0:]).mean():.1f} | "
              f"cy {np.abs(r['cy_err'][i0:]).mean():.1f}")
