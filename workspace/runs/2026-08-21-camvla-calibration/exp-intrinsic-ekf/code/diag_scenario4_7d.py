"""Diagnostic: does the 9-D (f+cx+cy) cold start fail to correct f where the
7-D (f only, principal point known) version succeeds?

Runs the scenario-4 cold start (volume 480 s, f0=500 vs true 515, extrinsic
10 deg/15 cm off, big P0) with a 7-D state (no cx/cy estimation), 3 seeds, and
prints the f / trans / rot trajectories summary for comparison with the 9-D
results in scenario4.json.
"""
import numpy as np
import sim_intrinsic as sim
import scenario4_plugplay as s4

DT = 0.05


def run_cold7(kind, T, seed):
    R2, t2 = s4.new_camera()
    feats, _ = sim.build_features(kind, T, DT, seed, use_elbow=True)
    T0 = sim.initial_guess((R2, t2), theta0_deg=10.0, d0=0.15, seed=seed)
    P0 = sim.initial_covariance(7, rot_deg=15.0, trans=0.2, logf_std=0.12)
    return sim.run_filter_simulation(
        feats, [(R2, t2)] * len(feats), np.full(len(feats), s4.F2),
        s4.CX2, s4.CY2, s4.SIGMA, T0, 500.0, P0, cx0=None, cy0=None,
        q_diag=s4.Q_9[:7], seed=seed, n_iters=5)


if __name__ == "__main__":
    for kind in ("volume", "wave"):
        for se in (0, 1, 2):
            r = run_cold7(kind, 480.0, se)
            i0 = int(0.9 * len(r["t"]))
            t = r["t"]
            print(f"7D {kind} seed{se}: |f| last10 "
                  f"{np.abs(r['f_err_pct'][i0:]).mean():.2f}% | f start/mid/end "
                  f"[{r['f_err_pct'][0]:.2f}, {r['f_err_pct'][len(t)//2]:.2f}, "
                  f"{r['f_err_pct'][-1]:.2f}] | trans {r['trans_err'][i0:].mean()*100:.2f} cm | "
                  f"rot {r['rot_err'][i0:].mean()*180/np.pi:.2f} deg")
