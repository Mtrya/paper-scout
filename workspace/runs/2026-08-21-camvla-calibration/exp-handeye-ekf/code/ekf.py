"""
Error-state EKF / IEKF on SE(3) for online hand-eye extrinsic estimation.

State
-----
T_bc in SE(3) maps camera coordinates to base (robot) coordinates:
    p_b = R @ p_c + t
with R a rotation matrix and t the camera position in the base frame.
This is the eye-to-hand extrinsic to be estimated online.

The filter keeps a nominal estimate T_hat plus a 6-D error state
    delta = (dtheta in R^3, dt in R^3)
with a LEFT multiplicative perturbation:
    T = Exp(delta) o T_hat,   Exp(delta) = [[Rot(dtheta), J_l(dtheta) dt], [0, 1]]
so a base-frame point p_b maps to camera frame by
    p_c(delta) = R_hat^T ( Rot(-dtheta) @ p_b - J_l(-dtheta) @ dt - t_hat ).

Measurement
-----------
At each step we observe the pixel projections of known base-frame points
(the EE point, optionally the elbow joint point) under iid Gaussian pixel
noise, i.e. u = proj(K, R^T (p_b - t)) + N(0, sigma^2 I).

Jacobians are computed by central finite differences of the measurement
function with respect to delta (only 6 parameters -> cheap, robust to
convention errors). The measurement update is iterated (IEKF) so that the
filter tolerates the deliberately large initialisation error used in the
experiments (30 deg / 0.3 m).
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
        # near pi: (R + I)/2 = u u^T up to sign ambiguity; take largest eigenvector
        _, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        u = V[:, -1]
        # sign convention: make the log consistent with the skew part where possible
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
# Camera model
# --------------------------------------------------------------------------
def project_points(K, T_R, T_t, points_b):
    """Project base-frame points (P,3) to pixels (P,2) under T_bc=(R,t)."""
    points_b = np.asarray(points_b, dtype=float)
    pc = (points_b - T_t) @ T_R  # R^T (p - t), rows are camera-frame points
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def measurement_jacobian(K, T_R, T_t, points_b, eps=1e-6):
    """Central-difference Jacobian of stacked pixels (2P,) w.r.t. the
    left-perturbation error state delta (6,). Evaluated at (T_R, T_t)."""
    P = len(points_b)
    H = np.zeros((2 * P, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = eps
        Rp, tp = se3_exp(e)
        up = project_points(K, Rp @ T_R, Rp @ T_t + tp, points_b)
        Rm, tm = se3_exp(-e)
        um = project_points(K, Rm @ T_R, Rm @ T_t + tm, points_b)
        H[:, j] = ((up - um) / (2.0 * eps)).ravel()
    return H


# --------------------------------------------------------------------------
# Error-state EKF (with iterated measurement update)
# --------------------------------------------------------------------------
class HandEyeEKF:
    """IEKF on SE(3) estimating T_bc from point-pixel correspondences."""

    def __init__(self, K, T0, P0, q_diag=None, n_iters=3, jac_eps=1e-6):
        self.K = K
        self.T_R, self.T_t = T0  # nominal estimate
        self.P = np.asarray(P0, dtype=float)  # 6x6 error-state covariance
        self.q_diag = q_diag  # per-step process noise (6,) or None (static)
        self.n_iters = n_iters
        self.jac_eps = jac_eps

    def predict(self):
        if self.q_diag is not None:
            self.P = self.P + np.diag(self.q_diag)

    def update(self, points_b, pixels, sigma):
        """Measurement update with `points_b` (P,3) and noisy pixels (P,2)."""
        P = len(points_b)
        Rm = np.eye(2 * P) * sigma**2
        Pc = self.P
        R, t = self.T_R, self.T_t

        # Iterated (Gauss-Newton style) relinearisation of the measurement.
        for _ in range(self.n_iters):
            pix = project_points(self.K, R, t, points_b)
            r = pixels.ravel() - pix.ravel()
            H = measurement_jacobian(self.K, R, t, points_b, self.jac_eps)
            S = H @ Pc @ H.T + Rm
            Kk = Pc @ H.T @ np.linalg.solve(S, np.eye(2 * P))
            d = Kk @ r
            Rd, td = se3_exp(d)
            R = Rd @ R
            t = Rd @ t + td

        # Final covariance update (Joseph form) with the final linearisation.
        H = measurement_jacobian(self.K, R, t, points_b, self.jac_eps)
        S = H @ Pc @ H.T + Rm
        Kk = Pc @ H.T @ np.linalg.solve(S, np.eye(2 * P))
        I6 = np.eye(6)
        Pc = (I6 - Kk @ H) @ Pc @ (I6 - Kk @ H).T + Kk @ Rm @ Kk.T

        self.T_R, self.T_t, self.P = R, t, Pc
