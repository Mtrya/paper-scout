"""
Simulator for the intrinsic-parameter EKF experiment (experiment D).

Extension of exp-handeye-ekf/sim.py: the same 3-DOF spatial arm (analytic
FK/IK) and eye-to-hand pinhole camera, but the focal length f (and optionally
the principal point cx, cy) is now part of the online estimate. The camera
geometry constants are kept identical to experiment A so results are directly
comparable.

Motions
-------
- line           : EE oscillates along a straight segment (1-D excitation)
- line_frontal   : EE oscillates along a segment parallel to the camera image
                   plane (analytically: exact f <-> t_z coupling family)
- plane          : Lissajous curve spanning a plane (2-D excitation)
- volume         : 3-D Lissajous, incommensurate frequencies (3-D excitation)
- rot_arc        : pure rotation: yaw-only sweep (EE traces a circular arc,
                   point cloud coplanar) -- "rotation without depth coverage"
- wave           : yaw + pitch combined sweep (EE traces a curved 2-D patch,
                   point cloud non-coplanar) -- "rotation with depth coverage"

For rot_arc / wave the joint trajectory is generated directly in joint space
(FK), because a pure-q1 sweep is not naturally an IK target sequence.
"""
from __future__ import annotations

import numpy as np

import ekf_intrinsic as ekfi
from ekf_intrinsic import IntrinsicEKF, measurement_jacobian, project, se3_exp, se3_log

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
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_fk(q):
    """FK: q=(q1,q2,q3) -> dict of base-frame feature points."""
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
    """Analytic IK for the 3-DOF arm; None if unreachable."""
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
# Motion generators
# --------------------------------------------------------------------------
def _orthonormal_basis(rng):
    e1 = rng.normal(size=3)
    e1 /= np.linalg.norm(e1)
    e2 = rng.normal(size=3)
    e2 -= e1 * (e1 @ e2)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return e1, e2, e3


def line_motion(c0, A, T, dt, seed, v=None):
    """EE oscillates along a straight segment through c0 (optionally along v)."""
    rng = np.random.default_rng(seed)
    if v is None:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
    f = 0.10 + 0.05 * rng.uniform()
    ph = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(0.0, T, dt)
    return t, c0 + A * np.outer(np.sin(2 * np.pi * f * t + ph), v), v


def plane_motion(c0, A, T, dt, seed):
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
    rng = np.random.default_rng(seed)
    e1, e2, e3 = _orthonormal_basis(rng)
    freqs = rng.choice([0.07, 0.10, 0.13, 0.16, 0.19, 0.22], size=3, replace=False)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    t = np.arange(0.0, T, dt)
    traj = c0.copy()
    for j, (e, f, p) in enumerate(zip((e1, e2, e3), freqs, phases)):
        traj = traj + A * np.outer(np.cos(2 * np.pi * f * t + p + j * np.pi / 2), e)
    return t, traj


def rot_arc_motion(c0, T, dt, seed, sweep_deg=15.0, f_sweep=0.07):
    """Pure rotation: yaw-only sweep about the pose reaching c0.

    The EE traces a circular arc (coplanar point cloud): rotation of the arm
    with almost no depth coverage as seen from the fixed camera.
    """
    q0 = arm_ik(c0)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(0.0, T, dt)
    q1 = q0[0] + np.deg2rad(sweep_deg) * np.sin(2 * np.pi * f_sweep * t + ph)
    q = np.stack([q1, np.full_like(q1, q0[1]), np.full_like(q1, q0[2])], axis=1)
    return t, q


def wave_motion(c0, T, dt, seed, q1_amp_deg=15.0, q2_amp_deg=28.0,
                f1=0.055, f2=0.103):
    """Rotation with depth coverage: yaw + pitch sweep with incommensurate
    frequencies. The EE traces a curved 2-D patch (non-coplanar)."""
    q0 = arm_ik(c0)
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0.0, 2.0 * np.pi)
    p2 = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(0.0, T, dt)
    q1 = q0[0] + np.deg2rad(q1_amp_deg) * np.sin(2 * np.pi * f1 * t + p1)
    q2 = q0[1] + np.deg2rad(q2_amp_deg) * np.sin(2 * np.pi * f2 * t + p2)
    q = np.stack([q1, q2, np.full_like(q1, q0[2])], axis=1)
    return t, q


def _fk_traj(q_seq):
    """Feature points (EE, elbow) from a joint-space sequence via FK."""
    N = len(q_seq)
    pts_ee = np.zeros((N, 3))
    pts_elbow = np.zeros((N, 3))
    for i, qi in enumerate(q_seq):
        fk = arm_fk(qi)
        pts_ee[i] = fk["p_ee"]
        pts_elbow[i] = fk["p_elbow"]
    return pts_ee, pts_elbow


def arm_motion_from_trajectory(traj, dt):
    """IK every target, then FK: returns per-step feature points."""
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
# Motion -> feature-point sequences
# --------------------------------------------------------------------------
def build_features(motion_kind, T, dt, seed, use_elbow, A=None,
                   check_visibility=True, cx=64.0, cy=64.0):
    """Per-step list of (P,3) base-frame feature points for a motion kind.

    Returns (features, meta) where meta carries motion geometry needed for the
    analytic observability checks (line direction / line point for `line`,
    `line_frontal`), None otherwise.
    """
    R_true, t_true = true_extrinsic()
    if motion_kind == "line":
        A = A if A is not None else 0.07
        _, traj, v = line_motion(C0, A, T, dt, seed)
        pts_ee, pts_elbow, ok = arm_motion_from_trajectory(traj, dt)
        meta = {"v": v, "p0": C0}
    elif motion_kind == "line_frontal":
        # line parallel to the camera image plane: v perpendicular to the
        # camera optical axis (camera y-axis, expressed in base frame)
        A = A if A is not None else 0.07
        v = R_true @ np.array([0.0, 1.0, 0.0])
        _, traj, v = line_motion(C0, A, T, dt, seed, v=v)
        pts_ee, pts_elbow, ok = arm_motion_from_trajectory(traj, dt)
        meta = {"v": v, "p0": C0}
    elif motion_kind == "plane":
        A = A if A is not None else 0.07
        _, traj = plane_motion(C0, A, T, dt, seed)
        pts_ee, pts_elbow, ok = arm_motion_from_trajectory(traj, dt)
        meta = None
    elif motion_kind == "volume":
        A = A if A is not None else 0.06
        _, traj = volume_motion(C0, A, T, dt, seed)
        pts_ee, pts_elbow, ok = arm_motion_from_trajectory(traj, dt)
        meta = None
    elif motion_kind == "rot_arc":
        _, q_seq = rot_arc_motion(C0, T, dt, seed)
        pts_ee, pts_elbow = _fk_traj(q_seq)
        ok = np.ones(len(q_seq), dtype=bool)
        meta = None
    elif motion_kind == "wave":
        _, q_seq = wave_motion(C0, T, dt, seed)
        pts_ee, pts_elbow = _fk_traj(q_seq)
        ok = np.ones(len(q_seq), dtype=bool)
        meta = None
    else:
        raise ValueError(motion_kind)

    if not ok.all():
        raise RuntimeError(f"[{motion_kind}] unreachable IK targets: {int((~ok).sum())}")

    if check_visibility:
        f = CAM_F
        feats_probe = pts_ee if not use_elbow else np.concatenate(
            [pts_ee, pts_elbow], axis=0)
        pix = project(f, cx, cy, R_true, t_true, feats_probe)
        if pix.min() < 2.0 or pix.max() > 126.0:
            raise RuntimeError(
                f"[{motion_kind}] feature leaves the image: pix in "
                f"[{pix.min():.1f}, {pix.max():.1f}]")

    feats = []
    for i in range(len(pts_ee)):
        if use_elbow:
            feats.append(np.stack([pts_ee[i], pts_elbow[i]]))
        else:
            feats.append(pts_ee[i][None, :])
    return feats, meta


# --------------------------------------------------------------------------
# Drift sequences
# --------------------------------------------------------------------------
def drift_extrinsic_sequence(T, dt, seed, t_bump, step_rot_deg, step_trans,
                             a_rot, a_trans, wander_rot_deg_per_s,
                             wander_trans_per_sqrt_s):
    """Extrinsic drift: static, a step at t_bump, then a random walk.

    Returns (N, 6) rows (dtheta, dt) with T_true(t) = Exp(delta(t)) o T_true(0).
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


def drift_f_sequence(T, dt, seed, kind, t_bump=None, ramp_frac=0.02,
                     ramp_time=1800.0, step_frac=0.05, jitter_frac=0.15):
    """Focal-length drift; returns per-step true f.

    kind = "ramp": linear thermal drift of `ramp_frac` (e.g. 2%) over
           `ramp_time` seconds plus a small random-walk jitter.
    kind = "step": constant f, then a `step_frac` jump at t_bump (lens swap).
    """
    rng = np.random.default_rng(seed)
    N = int(round(T / dt))
    f = np.full(N, CAM_F)
    if kind == "ramp":
        rate = ramp_frac / ramp_time  # per second
        jit = rate * np.sqrt(dt) * jitter_frac  # per-step jitter std
        for i in range(1, N):
            f[i] = f[i - 1] * (1.0 + rate * dt) * (1.0 + rng.normal(0.0, jit))
    elif kind == "step":
        tb = int(round(t_bump / dt))
        f[tb:] = f[tb:] * (1.0 + step_frac)
    else:
        raise ValueError(kind)
    return f


# --------------------------------------------------------------------------
# Filter driver (augmented state)
# --------------------------------------------------------------------------
def run_filter_simulation(features_per_step, T_true_per_step, f_true_per_step,
                          cx_true, cy_true, sigma, T0, f0, P0, cx0=None, cy0=None,
                          q_diag=None, seed=0, n_iters=3, dt=1.0):
    """Run the augmented IEKF over a simulated observation sequence.

    Pixels are generated with the TRUE camera (f_true, cx_true, cy_true); the
    filter estimates the extrinsic + intrinsics (P0 is 7x7 or 9x9) starting
    from (T0, f0 [, cx0, cy0]). Returns a per-step record dict with t,
    rot_err, trans_err, logf_err, f_err_pct, cx_err, cy_err, errN, P, nees.
    """
    rng = np.random.default_rng(seed)
    N = len(features_per_step)
    cx0 = cx_true if cx0 is None else cx0
    cy0 = cy_true if cy0 is None else cy0
    ndim = np.asarray(P0).shape[0]
    estimate_principal = ndim == 9
    ekf = IntrinsicEKF(f0, cx0, cy0, T0, P0, q_diag=q_diag, n_iters=n_iters,
                       estimate_principal=estimate_principal)

    rec = {k: [] for k in ("t", "rot_err", "trans_err", "logf_err",
                           "f_err_pct", "cx_err", "cy_err", "errN", "P")}
    for i in range(N):
        TR, Tt = T_true_per_step[i]
        fT = f_true_per_step[i]
        pts = features_per_step[i]
        pix = project(fT, cx_true, cy_true, TR, Tt, pts) + rng.normal(
            0.0, sigma, size=(len(pts), 2)
        )
        ekf.predict()
        ekf.update(pts, pix, sigma)

        Re = ekf.T_R @ TR.T
        te = ekf.T_t - Re @ Tt
        err6 = se3_log(Re, te)
        errN = np.concatenate([err6, [np.log(ekf.f / fT)]])
        if ndim == 9:
            errN = np.concatenate([errN, [ekf.cx - cx_true, ekf.cy - cy_true]])

        rec["t"].append(i * dt)
        rec["rot_err"].append(np.linalg.norm(err6[:3]))
        rec["trans_err"].append(np.linalg.norm(err6[3:]))
        rec["logf_err"].append(errN[6])
        rec["f_err_pct"].append(100.0 * (ekf.f / fT - 1.0))
        rec["cx_err"].append(errN[7] if ndim == 9 else 0.0)
        rec["cy_err"].append(errN[8] if ndim == 9 else 0.0)
        rec["errN"].append(errN)
        rec["P"].append(ekf.P.copy())

    rec["t"] = np.array(rec["t"])
    rec["rot_err"] = np.array(rec["rot_err"])
    rec["trans_err"] = np.array(rec["trans_err"])
    rec["logf_err"] = np.array(rec["logf_err"])
    rec["f_err_pct"] = np.array(rec["f_err_pct"])
    rec["cx_err"] = np.array(rec["cx_err"])
    rec["cy_err"] = np.array(rec["cy_err"])
    rec["errN"] = np.array(rec["errN"])
    rec["P"] = np.stack(rec["P"])
    rec["nees"] = np.array(
        [e @ np.linalg.solve(P, e) for e, P in zip(rec["errN"], rec["P"])]
    )
    return rec


# --------------------------------------------------------------------------
# Fisher-information analysis (augmented state)
# --------------------------------------------------------------------------
def per_step_fim(features_per_step, T_true_per_step, f_true, cx_true, cy_true,
                 sigma, ndim=7):
    """Per-step augmented FIM matrices (ndim x ndim) at the TRUE state."""
    N = len(features_per_step)
    out = np.zeros((N, ndim, ndim))
    for i in range(N):
        H = measurement_jacobian(f_true, cx_true, cy_true, *T_true_per_step[i],
                                 features_per_step[i], ndim)
        out[i] = H.T @ H / sigma**2
    return out


def windowed_fim_analysis(fim_i, window=120):
    """Sliding-window FIM spectra + f-marginal information.

    Returns dict with per-window arrays: eigvals (sorted ascending), min_eig,
    f_ff (raw FIM diagonal for log f), f_marginal (Schur complement), and the
    f-direction singular values.
    """
    N = len(fim_i)
    ndim = fim_i.shape[1]
    cum = np.concatenate([np.zeros((1, ndim, ndim)), np.cumsum(fim_i, axis=0)],
                         axis=0)
    eigvals = np.zeros((N, ndim))
    f_ff = np.zeros(N)
    f_marg = np.zeros(N)
    for a in range(1, N + 1):
        b = max(0, a - window)
        F = cum[a] - cum[b]
        eigvals[a - 1] = np.linalg.eigvalsh(F)
        f_ff[a - 1] = F[6, 6]
        Fee = F[:6, :6]
        Fef = F[:6, 6]
        # Schur complement via rank-restricted pseudo-inverse: the extrinsic
        # block can be singular (degenerate motion leaves an extrinsic null
        # direction); only the observable extrinsic directions are marginalised.
        # (rcond is the RELATIVE cutoff: sv < rcond * s_max are zeroed.)
        s = np.linalg.svd(Fee, compute_uv=False)
        f_marg[a - 1] = F[6, 6] - Fef @ np.linalg.pinv(Fee, rcond=1e-8) @ Fef
    # numerical rank / nullity: eigenvalues < 1e-8 of the max eigenvalue
    scale = eigvals.max(axis=1, keepdims=True)
    nullity = (eigvals < 1e-8 * np.maximum(scale, 1e-30)).sum(axis=1)
    return {
        "eigvals": eigvals,
        "min_eig": eigvals[:, 0],
        "nullity": nullity,
        "f_ff": f_ff,
        "f_marginal": f_marg,
        "sigma_logf_crb": 1.0 / np.sqrt(np.clip(f_marg, 1e-300, None)),
        "sigma_f_crb_pct": 100.0 / np.sqrt(np.clip(f_marg, 1e-300, None)),
        "f_singular": np.sqrt(np.clip(f_ff, 1e-300, None)),
    }


def null_space(F, thresh=1e-8):
    """Orthonormal basis of the (near-)null space of F (eigenvectors with
    eigenvalue < thresh * max)."""
    w, V = np.linalg.eigh(F)
    scale = w.max() if w.max() > 0 else 1.0
    mask = w < thresh * scale
    return V[:, mask].T, w


def analytic_null_extrinsic_line(v, p0):
    """Exp-A family for a line through p0 along v: rotation of the camera
    about the motion-line axis through p0. In the left-perturbation error
    state this is  delta = (beta v, -beta v x p0, 0); the translation part is
    J_l^-1 (I - Rot(v,beta)) p0 ~ -beta v x p0 to first order (the camera
    position does NOT enter). Null = (v, -v x p0, 0)."""
    u = np.concatenate([v, -np.cross(v, p0), [0.0]])
    return u / np.linalg.norm(u)


def analytic_null_focal_frontal_line(v, p0, R_true, t_true):
    """f <-> t_z coupling for a line parallel to the image plane (v _|_ optical
    axis): null = (0, 0, 0, -w_z * e_z, 1) with e_z the camera optical axis in
    base frame and w_z = (p0 - t) . e_z the line depth."""
    e_z = R_true @ np.array([0.0, 0.0, 1.0])
    w_z = float((p0 - t_true) @ e_z)
    u = np.concatenate([np.zeros(3), -w_z * e_z, [1.0]])
    return u / np.linalg.norm(u)


# --------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------
def initial_guess(T_true, theta0_deg=30.0, d0=0.3, seed=0):
    """T0 = Exp(delta0) o T_true with delta0 = (30 deg about random axis,
    0.3 m along random direction)."""
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


def initial_covariance(ndim=7, rot_deg=30.0, trans=0.5, logf_std=0.10,
                       px_std=4.0):
    """P0 for the augmented state (last diag = log-f variance, or cx/cy px^2)."""
    p = np.diag([np.deg2rad(rot_deg) ** 2] * 3 + [trans**2] * 3)
    p = np.pad(p, ((0, ndim - 6), (0, ndim - 6)))
    if ndim >= 7:
        p[6, 6] = logf_std**2
    if ndim == 9:
        p[7, 7] = px_std**2
        p[8, 8] = px_std**2
    return p
