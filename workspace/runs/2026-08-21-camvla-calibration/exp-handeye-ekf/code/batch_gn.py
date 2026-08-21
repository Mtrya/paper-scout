"""
Batch Gauss-Newton on SE(3) for the hand-eye extrinsic (reference solver).

Used as a sanity reference: the same observation data solved off-line with a
nonlinear least-squares should reach the Cramer-Rao bound, whereas the online
single-point EKF can lock into a biased local optimum.
"""
from __future__ import annotations

import numpy as np

import sim
from ekf import project_points, measurement_jacobian, se3_exp, se3_log


def batch_gn(features, sigma, seed=0, theta0_deg=30.0, d0=0.3, max_iter=100):
    """Least squares over all observations: min ||u - proj(K, T, p)||^2 / sigma^2.

    features : list of (P,3) base-frame feature points (one entry per step)
    Returns (n_iters, error6) with error6 = left error state of T vs truth.
    """
    rng = np.random.default_rng(seed)
    T_true = sim.true_extrinsic()
    K = sim.make_K()
    pix_all, pts_all = [], []
    for f in features:
        pix_all.append(
            project_points(K, *T_true, f) + rng.normal(0.0, sigma, size=(len(f), 2))
        )
        pts_all.append(f)
    pix_all = np.concatenate(pix_all)
    pts_all = np.concatenate(pts_all)

    R, t = sim.initial_guess(T_true, theta0_deg=theta0_deg, d0=d0, seed=seed)
    for it in range(max_iter):
        r = pix_all.ravel() - project_points(K, R, t, pts_all).ravel()
        H = measurement_jacobian(K, R, t, pts_all)
        delta = np.linalg.solve(H.T @ H, H.T @ r)
        Rd, td = se3_exp(delta)
        R = Rd @ R
        t = Rd @ t + td
        if np.linalg.norm(delta) < 1e-10:
            break
    Re = R @ T_true[0].T
    te = t - Re @ T_true[1]
    return it + 1, se3_log(Re, te)
