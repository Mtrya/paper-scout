"""
Batch Gauss-Newton on SE(3) x focal length (reference solver).

Same observation data solved off-line with nonlinear least squares over the
7-D parameter vector (R, t, log f): a sanity reference for the online filter
(scenario 2). Uses the left SE(3) perturbation for the extrinsic and a
multiplicative perturbation for f, iterated to convergence.
"""
from __future__ import annotations

import numpy as np

import sim_intrinsic as sim
from ekf_intrinsic import (measurement_jacobian, project, se3_exp, se3_log)

MAX_ITER = 200


def batch_gn(features, sigma, seed=0, theta0_deg=30.0, d0=0.3,
             f_bias=0.10, cx0=64.0, cy0=64.0, cx_bias=0.0, cy_bias=0.0,
             ndim=7, max_iter=MAX_ITER):
    """Least squares over all observations:
        min sum_t || u_t - proj(f, cx, cy, R, t, p_t) ||^2 / sigma^2
    over (R, t, log f [, cx, cy]). ndim = 7 or 9. Returns (n_iters, errN)
    with errN the left error state of the estimate vs truth (rotation,
    translation, log f [, dcx, dcy]).
    """
    rng = np.random.default_rng(seed)
    R_true, t_true = sim.true_extrinsic()
    f_true = sim.CAM_F
    cx_t, cy_t = 64.0, 64.0

    pix_all, pts_all = [], []
    for fts in features:
        pix_all.append(project(f_true, cx_t, cy_t, R_true, t_true, fts)
                       + rng.normal(0.0, sigma, size=(len(fts), 2)))
        pts_all.append(fts)
    pix_all = np.concatenate(pix_all)
    pts_all = np.concatenate(pts_all)

    R, t = sim.initial_guess((R_true, t_true), theta0_deg=theta0_deg,
                             d0=d0, seed=seed)
    f = f_true * (1.0 + f_bias)
    cx, cy = cx0 + cx_bias, cy0 + cy_bias
    for it in range(max_iter):
        r = pix_all.ravel() - project(f, cx, cy, R, t, pts_all).ravel()
        H = measurement_jacobian(f, cx, cy, R, t, pts_all, ndim)
        delta = np.linalg.solve(H.T @ H, H.T @ r)
        Rd, td = se3_exp(delta[:6])
        R = Rd @ R
        t = Rd @ t + td
        f = f * np.exp(delta[6])
        if ndim == 9:
            cx += delta[7]
            cy += delta[8]
        if np.linalg.norm(delta) < 1e-10:
            break
    Re = R @ R_true.T
    te = t - Re @ t_true
    err6 = se3_log(Re, te)
    errN = np.concatenate([err6, [np.log(f / f_true)]])
    if ndim == 9:
        errN = np.concatenate([errN, [cx - cx_t, cy - cy_t]])
    return it + 1, errN
