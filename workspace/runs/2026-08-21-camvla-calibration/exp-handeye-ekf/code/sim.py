"""
Simulator for the hand-eye EKF experiment.

A 3-DOF spatial arm (yaw about z at the base, then two pitch joints) with
analytic forward/inverse kinematics; a pinhole camera (K known) placed
eye-to-hand; the camera extrinsic T_bc (camera -> base) is the unknown to be
estimated. Known base-frame feature points (the EE point, optionally the
elbow joint point) are projected to pixels under the true extrinsic, corrupted
by iid Gaussian pixel noise, and fed to the EKF at the arm control rate.

Motions
-------
- line_motion    : EE oscillates along a straight segment (1-D excitation)
- plane_motion   : Lissajous curve spanning a plane (2-D excitation)
- volume_motion  : 3-D Lissajous with incommensurate frequencies (3-D excitation)
Each generator consumes a seed and produces random orientation / frequencies /
phases, so different seeds give genuinely different trials.

Observability expectations (single point feature, known base-frame positions):
- 1-D line motion leaves a 1-parameter unobservable family: rotation of the
  camera about the motion-line axis (coupled with a specific translation), so
  the sliding-window Fisher information has an exact null direction.
- 2-D planar / 3-D volume motions give full rank (classical PnP with known
  scale), differing in conditioning and convergence speed.
"""
from __future__ import annotations

import numpy as np

from ekf import (
    HandEyeEKF,
    measurement_jacobian,
    project_points,
    se3_exp,
    se3_log,
)

LINKS = (0.30, 0.30, 0.15)  # l1, l2, l3 (m)

CAM_F = 500.0  # focal length (px)
CAM_IMG = 128  # square image size (px)

# Nominal geometry used by all scenarios (base frame).
C0 = np.array([0.36, 0.0, 0.26])  # EE motion centroid
CAM_EYE = np.array([0.0, -2.2, 0.50])  # true camera position
CAM_TARGET = np.array([0.33, 0.0, 0.14])  # aim point (between elbow and EE)


def make_K(f=CAM_F, size=CAM_IMG):
    return np.array([[f, 0.0, size / 2], [0.0, f, size / 2], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# Arm kinematics
# --------------------------------------------------------------------------
def _Ry(a):
    # positive angle rotates +x toward +z (consistent with the planar IK)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_fk(q):
    """Forward kinematics: q=(q1,q2,q3) -> dict of base-frame feature points.

    p_elbow is joint-2 origin, p_wrist is joint-3 origin, p_ee the end point.
    """
    q1, q2, q3 = q
    l1, l2, l3 = LINKS
    A = _Rz(q1)
    J2 = A @ np.array([l1, 0.0, 0.0])
    J3 = A @ (np.array([l1, 0.0, 0.0]) + _Ry(q2) @ np.array([l2, 0.0, 0.0]))
    ee = A @ (
        np.array([l1, 0.0, 0.0])
        + _Ry(q2) @ (np.array([l2, 0.0, 0.0]) + _Ry(q3) @ np.array([l3, 0.0, 0.0]))
    )
    return dict(p_ee=ee, p_elbow=J2, p_wrist=J3)


def arm_ik(p, elbow_up=True):
    """Analytic inverse kinematics for the 3-DOF arm; None if unreachable."""
    l1, l2, l3 = LINKS
    x, y, z = np.asarray(p, dtype=float)
    r = np.hypot(x, y)
    q1 = np.arctan2(y, x)
    x2, z2 = r - l1, z
    d = np.hypot(x2, z2)
    if d < 1e-9 or d > l2 + l3 - 1e-9 or d < abs(l2 - l3) + 1e-9:
        return None
    cq3 = np.clip((x2**2 + z2**2 - l2**2 - l3**2) / (2.0 * l2 * l3), -1.0, 1.0)
    q3 = np.arccos(cq3) if elbow_up else -np.arccos(cq3)
    phi = np.arctan2(z2, x2)
    psi = np.arctan2(l3 * np.sin(q3), l2 + l3 * np.cos(q3))
    q2 = phi - psi
    return np.array([q1, q2, q3])


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Rotation matrix whose columns are the camera axes in base frame."""
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def true_extrinsic():
    """T_bc_true = (R, t): camera look-at geometry for the nominal scene."""
    R = look_at(CAM_EYE, CAM_TARGET)
    return R, CAM_EYE.copy()


# --------------------------------------------------------------------------
# Motion generators  (all return (times, trajectory (N,3) of EE base positions))
# --------------------------------------------------------------------------
def _orthonormal_basis(rng):
    e1 = rng.normal(size=3)
    e1 /= np.linalg.norm(e1)
    e2 = rng.normal(size=3)
    e2 -= e1 * (e1 @ e2)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return e1, e2, e3


def line_motion(c0, A, T, dt, seed):
    """EE oscillates along a single straight segment through c0."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    f = 0.10 + 0.05 * rng.uniform()
    ph = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(0.0, T, dt)
    return t, c0 + A * np.outer(np.sin(2 * np.pi * f * t + ph), v)


def plane_motion(c0, A, T, dt, seed):
    """Lissajous curve spanning a plane (two incommensurate frequencies)."""
    rng = np.random.default_rng(seed)
    e1, e2, _ = _orthonormal_basis(rng)
    f1 = 0.10 + 0.03 * rng.uniform()
    f2 = 0.16 + 0.03 * rng.uniform()
    p1 = rng.uniform(0.0, 2.0 * np.pi)
    p2 = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(0.0, T, dt)
    traj = (
        c0
        + A * np.outer(np.cos(2 * np.pi * f1 * t + p1), e1)
        + A * np.outer(np.sin(2 * np.pi * f2 * t + p2), e2)
    )
    return t, traj


def volume_motion(c0, A, T, dt, seed):
    """3-D Lissajous with incommensurate frequencies: covers a box, direction-rich."""
    rng = np.random.default_rng(seed)
    e1, e2, e3 = _orthonormal_basis(rng)
    freqs = rng.choice([0.07, 0.10, 0.13, 0.16, 0.19, 0.22], size=3, replace=False)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    t = np.arange(0.0, T, dt)
    traj = c0.copy()
    for j, (e, f, p) in enumerate(zip((e1, e2, e3), freqs, phases)):
        traj = traj + A * np.outer(np.cos(2 * np.pi * f * t + p + j * np.pi / 2), e)
    return t, traj


def arm_motion_from_trajectory(traj, dt):
    """IK every target, then FK: returns per-step feature points.

    Returns (pts_ee (N,3), pts_elbow (N,3), ok_mask (N,) bool).
    """
    pts_ee = np.zeros((len(traj), 3))
    pts_elbow = np.zeros((len(traj), 3))
    ok = np.zeros(len(traj), dtype=bool)
    for i, p in enumerate(traj):
        q = arm_ik(p)
        if q is None:
            continue
        fk = arm_fk(q)
        pts_ee[i] = fk["p_ee"]
        pts_elbow[i] = fk["p_elbow"]
        ok[i] = True
    return pts_ee, pts_elbow, ok


# --------------------------------------------------------------------------
# Drift sequence (scenario 3)
# --------------------------------------------------------------------------
def drift_sequence(T, dt, seed, t_bump, step_rot_deg, step_trans, a_rot, a_trans,
                   wander_rot_deg_per_s, wander_trans_per_sqrt_s):
    """Camera drift: static, then a step at t_bump, then a random walk.

    Returns (N, 6) with rows (dtheta, dt) such that
        T_true(t) = Exp(delta(t)) o T_true(0)  (left perturbation, base frame).
    """
    rng = np.random.default_rng(seed)
    N = int(round(T / dt))
    tb = int(round(t_bump / dt))
    wrot = np.deg2rad(wander_rot_deg_per_s)
    d = np.zeros((N, 6))
    for i in range(1, N):
        d[i] = d[i - 1]
        if i > tb:
            d[i, :3] += rng.normal(0.0, wrot * np.sqrt(dt), 3)
            d[i, 3:] += rng.normal(0.0, wander_trans_per_sqrt_s * np.sqrt(dt), 3)
        if i == tb + 1:
            d[i, :3] += np.deg2rad(step_rot_deg) * a_rot
            d[i, 3:] += step_trans * a_trans
    return d


# --------------------------------------------------------------------------
# Filter driver
# --------------------------------------------------------------------------
def run_filter_simulation(features_per_step, T_true_per_step, K, sigma, T0, P0,
                          q_diag=None, seed=0, n_iters=3, fim_window=120,
                          null_basis=None, dt=1.0):
    """Run the IEKF over a simulated observation sequence; return a record dict.

    features_per_step : list of (P,3) base-frame feature points per step
    T_true_per_step   : list of (R, t) true extrinsics per step
    fim_window        : sliding window (steps) for the Fisher information
    null_basis        : optional 6-vector; its projection is recorded per step

    Returns a dict with per-step arrays:
        t, rot_err, trans_err, cov_trace, fim_mineig, err6, null_proj
    """
    rng = np.random.default_rng(seed)
    N = len(features_per_step)
    ekf = HandEyeEKF(K, T0, P0, q_diag=q_diag, n_iters=n_iters)

    # Per-step Fisher information at the TRUE extrinsic, then sliding-window sums.
    fim_i = np.zeros((N, 6, 6))
    for i in range(N):
        R, t = T_true_per_step[i]
        H = measurement_jacobian(K, R, t, features_per_step[i])
        fim_i[i] = H.T @ H / sigma**2
    cum = np.concatenate([np.zeros((1, 6, 6)), np.cumsum(fim_i, axis=0)], axis=0)

    rec = {k: [] for k in ("t", "rot_err", "trans_err", "cov_trace",
                           "fim_mineig", "err6", "null_proj", "P")}
    for i in range(N):
        TR, Tt = T_true_per_step[i]
        pts = features_per_step[i]
        pix = project_points(K, TR, Tt, pts) + rng.normal(
            0.0, sigma, size=(len(pts), 2)
        )
        ekf.predict()
        ekf.update(pts, pix, sigma)

        # left error state: T_est = Exp(err6) o T_true
        Re = ekf.T_R @ TR.T
        te = ekf.T_t - Re @ Tt
        err6 = se3_log(Re, te)

        a = i + 1
        b = max(0, a - fim_window)
        mineig = np.linalg.eigvalsh(cum[a] - cum[b])[0]

        rec["t"].append(i * dt)
        rec["rot_err"].append(np.linalg.norm(err6[:3]))
        rec["trans_err"].append(np.linalg.norm(err6[3:]))
        rec["cov_trace"].append(np.trace(ekf.P))
        rec["fim_mineig"].append(mineig)
        rec["err6"].append(err6)
        rec["P"].append(ekf.P.copy())
        if null_basis is not None:
            rec["null_proj"].append(err6 @ np.asarray(null_basis, dtype=float))

    rec["rot_err"] = np.array(rec["rot_err"])
    rec["trans_err"] = np.array(rec["trans_err"])
    rec["cov_trace"] = np.array(rec["cov_trace"])
    rec["fim_mineig"] = np.array(rec["fim_mineig"])
    rec["err6"] = np.array(rec["err6"])
    rec["P"] = np.stack(rec["P"])
    rec["nees"] = np.array(
        [e @ np.linalg.solve(P, e) for e, P in zip(rec["err6"], rec["P"])]
    )
    rec["t"] = np.array(rec["t"])
    if null_basis is not None:
        rec["null_proj"] = np.array(rec["null_proj"])
    else:
        rec["null_proj"] = None
    return rec


def build_trial(motion_kind, T, dt, seed, use_elbow, A=None):
    """Build the feature-point sequence for a motion kind (EE [+ elbow])."""
    c0 = C0
    if motion_kind == "line":
        A = A if A is not None else 0.07
        _, traj = line_motion(c0, A, T, dt, seed)
    elif motion_kind == "plane":
        A = A if A is not None else 0.07
        _, traj = plane_motion(c0, A, T, dt, seed)
    elif motion_kind == "volume":
        A = A if A is not None else 0.06
        _, traj = volume_motion(c0, A, T, dt, seed)
    else:
        raise ValueError(motion_kind)
    pts_ee, pts_elbow, ok = arm_motion_from_trajectory(traj, dt)
    if not ok.all():
        raise RuntimeError(f"unreachable IK targets: {int((~ok).sum())}")
    feats = []
    for i in range(len(traj)):
        if use_elbow:
            feats.append(np.stack([pts_ee[i], pts_elbow[i]]))
        else:
            feats.append(pts_ee[i][None, :])
    return feats


def initial_guess(T_true, theta0_deg=30.0, d0=0.3, seed=0):
    """T0 = Exp(delta0) o T_true with delta0 = (30 deg about axis a, 0.3 m along b)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    b = rng.normal(size=3)
    b /= np.linalg.norm(b)
    delta0 = np.concatenate([np.deg2rad(theta0_deg) * a, d0 * b])
    R, t = se3_exp(delta0)
    R0 = T_true[0] @ R
    t0 = T_true[0] @ np.asarray(t) + T_true[1]
    return R0, t0


def initial_covariance():
    """P0 with 1-sigma ~ 0.6 rad / 0.5 m per axis (covers the 30 deg, 0.3 m offset)."""
    return np.diag([0.6**2, 0.6**2, 0.6**2, 0.5**2, 0.5**2, 0.5**2])
