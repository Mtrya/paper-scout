"""Diagnostic: is the wrong-f signature visible in the PRE-update innovation
(not the post-update residual)?  The post-update residual in scenario3's
run_backlash stays flat at ~0.94 px for all f biases, so the README's claim
"innovation RMS rises with f bias" needs a check against the pre-update
innovation z - h(x_prior).
"""
import numpy as np
import sim_intrinsic as sim
import ekf6
from ekf6 import HandEyeEKF, project_points


def run_backlash_innov(f_bias, seed=0):
    T = 120.0
    feats, _ = sim.build_features("volume", T, 0.05, seed, use_elbow=True)
    R_true, t_true = sim.true_extrinsic()
    rng = np.random.default_rng(seed)
    K6 = sim.make_K(500.0 * (1.0 + f_bias))
    ekf = HandEyeEKF(K6, sim.initial_guess((R_true, t_true), seed=seed),
                     np.diag([0.6 ** 2] * 3 + [0.5 ** 2] * 3),
                     q_diag=np.array([3e-9] * 3 + [3e-10] * 3))
    pre_innov, post_res = [], []
    for fts in feats:
        pix = project_points(sim.make_K(), R_true, t_true, fts) + \
            rng.normal(0.0, 1.0, size=(len(fts), 2))
        ekf.predict()
        # PRE-update innovation: predicted with the wrong-f model at the prior state
        pix_prior = project_points(K6, ekf.T_R, ekf.T_t, fts)
        pre_innov.append(np.sqrt(np.mean((pix - pix_prior) ** 2)))
        ekf.update(fts, pix, 1.0)
        pix_hat = project_points(K6, ekf.T_R, ekf.T_t, fts)
        post_res.append(np.sqrt(np.mean((pix - pix_hat) ** 2)))
    pre_innov = np.array(pre_innov)
    post_res = np.array(post_res)
    i0 = int(0.7 * len(pre_innov))
    return pre_innov[i0:].mean(), post_res[i0:].mean()


if __name__ == "__main__":
    for b in (0.0, 0.02, 0.05, -0.05):
        pre, post = run_backlash_innov(b)
        print(f"f bias {b*100:+.0f}%: PRE-update innov RMS {pre:.3f} px | "
              f"post-update residual RMS {post:.3f} px")
