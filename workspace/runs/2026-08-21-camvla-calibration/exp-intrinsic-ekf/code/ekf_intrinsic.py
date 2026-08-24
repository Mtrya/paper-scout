"""
Error-state IEKF on SE(3) x focal length (x principal point) for online
hand-eye extrinsic + intrinsic estimation.

State error (7-D core):
    delta = (dtheta (3), dt (3), dlogf (1))
Optional (9-D, for the plug-and-play scenario):
    delta = (dtheta (3), dt (3), dlogf (1), dcx (1), dcy (1))

Perturbations (left multiplicative for SE(3), multiplicative for f, additive
for the principal point):
    T = Exp(delta_se3) o T_hat
    f = f_hat * exp(dlogf)      (positive focal length, scale-invariant noise)
    cx = cx_hat + dcx ; cy = cy_hat + dcy

Measurement: pixel projections of known base-frame points (EE point,
optionally the elbow joint) under iid Gaussian pixel noise,
    u = proj(f, cx, cy, R, t, p_b) + N(0, sigma^2 I).

Jacobians are central finite differences of the projection w.r.t. the full
error state (7 or 9 parameters -> cheap, robust to convention errors). The
measurement update is iterated (IEKF) to tolerate the deliberately large
initialisation errors used in the experiments.

This module mirrors exp-handeye-ekf/code/ekf.py; the extrinsic-only filter
of experiment A is the special case ndim = 6 with f fixed.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# SO(3) / SE(3) helpers
# --------------------------------------------------------------------------
def hat(v):
    """Skew-symmetric matrix of a 3-vector."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def so3_exp(w):
    """Rodrigues: rotation matrix from rotation vector w (rad)."""
    w = np.asarray(w, dtype=float)
    th = np.linalg.norm(w)
    if th < _EPS:
        return np.eye(3) + hat(w)
    a = w / th
    K = hat(a)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def so3_log(R):
    """Rotation vector (rad) from rotation matrix; safe near identity and pi."""
    R = np.asarray(R, dtype=float)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-8:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
    if np.pi - th < 1e-6:
        _, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        u = V[:, -1]
        if u @ np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) < 0:
            u = -u
        return u * np.pi
    s = np.sin(th)
    return th / (2.0 * s) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def so3_left_jacobian(w):
    """Left Jacobian J_l(w) of SO(3) (D Exp(w)/Dw), 3x3."""
    w = np.asarray(w, dtype=float)
    th = np.linalg.norm(w)
    if th < _EPS:
        return np.eye(3) + 0.5 * hat(w) + (1.0 / 6.0) * (hat(w) @ hat(w))
    a = w / th
    K = hat(a)
    return (
        np.eye(3)
        + ((1.0 - np.cos(th)) / th**2) * hat(w)
        + ((th - np.sin(th)) / th**3) * (hat(w) @ hat(w))
    )


def se3_exp(delta):
    """Exp(delta) for delta=(dtheta, dt); returns (R, t) of the 4x4 block."""
    dtheta, dt = delta[:3], delta[3:]
    R = so3_exp(dtheta)
    t = so3_left_jacobian(dtheta) @ dt
    return R, t


def se3_log(R, t):
    """Log of (R, t) in SE(3) under the left convention; returns 6-vector."""
    dtheta = so3_log(R)
    J = so3_left_jacobian(dtheta)
    dt = np.linalg.solve(J, t)
    return np.concatenate([dtheta, dt])


def rotate_error_angle(R1, R2):
    """Angle (rad) of the smallest rotation taking R1 to R2."""
    return np.linalg.norm(so3_log(R1.T @ R2))


# --------------------------------------------------------------------------
# Camera model (f, cx, cy explicit)
# --------------------------------------------------------------------------
def project(f, cx, cy, T_R, T_t, points_b):
    """Project base-frame points (P,3) to pixels (P,2)."""
    points_b = np.asarray(points_b, dtype=float)
    pc = (points_b - T_t) @ T_R  # R^T (p - t)
    z = pc[:, 2]
    u = f * pc[:, 0] / z + cx
    v = f * pc[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def measurement_jacobian(f, cx, cy, T_R, T_t, points_b, ndim, eps=1e-6):
    """Central-difference Jacobian of stacked pixels (2P,) w.r.t. the error
    state delta (ndim = 7 or 9). Evaluated at (T_R, T_t, f, cx, cy)."""
    P = len(points_b)
    H = np.zeros((2 * P, ndim))
    for j in range(6):  # SE(3) left perturbation
        e = np.zeros(6)
        e[j] = eps
        Rp, tp = se3_exp(e)
        up = project(f, cx, cy, Rp @ T_R, Rp @ T_t + tp, points_b)
        Rm, tm = se3_exp(-e)
        um = project(f, cx, cy, Rm @ T_R, Rm @ T_t + tm, points_b)
        H[:, j] = ((up - um) / (2.0 * eps)).ravel()
    # log-focal-length perturbation (multiplicative); only for ndim >= 7
    if ndim >= 7:
        up = project(f * np.exp(eps), cx, cy, T_R, T_t, points_b)
        um = project(f * np.exp(-eps), cx, cy, T_R, T_t, points_b)
        H[:, 6] = ((up - um) / (2.0 * eps)).ravel()
    if ndim == 9:
        up = project(f, cx + eps, cy, T_R, T_t, points_b)
        um = project(f, cx - eps, cy, T_R, T_t, points_b)
        H[:, 7] = ((up - um) / (2.0 * eps)).ravel()
        up = project(f, cx, cy + eps, T_R, T_t, points_b)
        um = project(f, cx, cy - eps, T_R, T_t, points_b)
        H[:, 8] = ((up - um) / (2.0 * eps)).ravel()
    return H


# --------------------------------------------------------------------------
# Error-state IEKF with augmented intrinsic state
# --------------------------------------------------------------------------
class IntrinsicEKF:
    """IEKF estimating (T_bc, f [, cx, cy]) from point-pixel correspondences.

    The extrinsic part is a left-perturbed SE(3) nominal plus 6-D error; the
    focal length is tracked in log space (f = f_hat * exp(dlogf)) so it stays
    positive and its uncertainty is scale-invariant. P is ndim x ndim with the
    last diagonal block covering log f (and cx/cy in 9-D mode).
    """

    def __init__(self, f0, cx0, cy0, T0, P0, q_diag=None, n_iters=3,
                 jac_eps=1e-6, estimate_principal=False):
        self.f = float(f0)
        self.cx = float(cx0)
        self.cy = float(cy0)
        self.T_R, self.T_t = T0  # nominal estimate
        self.P = np.asarray(P0, dtype=float)
        self.ndim = self.P.shape[0]
        self.q_diag = q_diag  # per-step process noise (ndim,) or None (static)
        self.n_iters = n_iters
        self.jac_eps = jac_eps

    def predict(self):
        if self.q_diag is not None:
            self.P = self.P + np.diag(self.q_diag)

    def update(self, points_b, pixels, sigma):
        """Measurement update with `points_b` (P,3) and noisy pixels (P,2)."""
        P = len(points_b)
        m = 2 * P
        Rm = np.eye(m) * sigma**2
        Pc = self.P
        R, t = self.T_R, self.T_t
        f, cx, cy = self.f, self.cx, self.cy

        # Iterated (Gauss-Newton style) relinearisation of the measurement.
        for _ in range(self.n_iters):
            pix = project(f, cx, cy, R, t, points_b)
            r = pixels.ravel() - pix.ravel()
            H = measurement_jacobian(f, cx, cy, R, t, points_b,
                                     self.ndim, self.jac_eps)
            S = H @ Pc @ H.T + Rm
            Kk = Pc @ H.T @ np.linalg.solve(S, np.eye(m))
            d = Kk @ r
            Rd, td = se3_exp(d[:6])
            R = Rd @ R
            t = Rd @ t + td
            f = f * np.exp(d[6])
            if self.ndim == 9:
                cx += d[7]
                cy += d[8]

        # Final covariance update (Joseph form) with the final linearisation.
        H = measurement_jacobian(f, cx, cy, R, t, points_b,
                                 self.ndim, self.jac_eps)
        S = H @ Pc @ H.T + Rm
        Kk = Pc @ H.T @ np.linalg.solve(S, np.eye(m))
        I = np.eye(self.ndim)
        Pc = (I - Kk @ H) @ Pc @ (I - Kk @ H).T + Kk @ Rm @ Kk.T

        self.T_R, self.T_t, self.P = R, t, Pc
        self.f, self.cx, self.cy = f, cx, cy
