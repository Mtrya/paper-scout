"""Blob world: minimal 3D reaching task with a pinhole camera rig.

Setup
-----
- Base-frame 3D workspace: EE point and target point sampled uniformly in a cube.
- Oracle: base-frame proportional controller (walks toward target, clipped step,
  small noise). Generates the demonstration rollouts used for behavior cloning.
- Camera: pinhole, known intrinsics; the whole camera rig (position + orientation)
  rotates around the base z-axis by `yaw`. Rendering is 64x64 RGB: EE = green blob,
  target = red blob, plus a few fixed gray distractors. Blob apparent size scales
  with 1/depth (perspective), so depth is weakly observable from blob size.

Frame conventions
-----------------
- Camera frame: x right, y down (image v), z forward (view direction).
- R_bc(yaw): camera-to-base rotation (3x3, columns = camera axes in base frame).
- R_cb(yaw) = R_bc(yaw)^T: base-to-camera.
- Projection: p_c = R_cb (p_b - t_b); u = f*x_c/z_c + cx, v = f*y_c/z_c + cy.
- Action frames: Base head predicts base-frame delta directly; Cam head predicts
  camera-frame delta, synthesized at execution with R_bc(yaw_estimate).
"""
from __future__ import annotations

import numpy as np

# ---- task parameters -------------------------------------------------------
V_MAX = 0.08          # actuator limit: step clipped to +-V_MAX
SUCCESS_THRESH = 0.08 # success if final distance < this
MAX_STEPS = 50        # per-episode budget (rollout & oracle demos)
PROP_GAIN = 0.35      # oracle proportional gain (distance-multiplied, clipped)
ORACLE_NOISE = 0.006  # per-step noise added to oracle deltas
DYN_NOISE = 0.004     # actuation noise added at execution

WS_LO, WS_HI = -0.5, 0.5  # workspace cube

# ---- camera ---------------------------------------------------------------
IMG = 64
FOCAL = 60.0
CX = CY = (IMG - 1) / 2.0
CAM_DIST = 2.5        # camera distance from origin
CAM_HEIGHT = 0.6      # camera height above x-y plane
BLOB_RADIUS = 0.15    # world radius of EE/target blobs
DIST_RADIUS = 0.12    # world radius of distractors
BLOB_PEAK = 1.0
NEAR = 0.1            # points closer than this (along view) are not rendered

# fixed distractors in base frame (gray blobs) - give the view identifiable structure
DISTRACTORS = np.array([
    [0.35, 0.00, 0.25],
    [-0.30, 0.25, -0.20],
    [0.00, -0.35, 0.05],
])
DIST_COLOR = np.array([0.45, 0.45, 0.45], dtype=np.float64)
EE_COLOR = np.array([0.0, 1.0, 0.0], dtype=np.float64)     # green
TGT_COLOR = np.array([1.0, 0.0, 0.0], dtype=np.float64)    # red

# canonical camera frame axes (columns of R_bc(0))
_t0 = np.array([0.0, -CAM_DIST, CAM_HEIGHT])
_zc0 = -_t0 / np.linalg.norm(_t0)          # view direction (camera -> origin)
_xc0 = np.cross(_zc0, np.array([0.0, 0.0, 1.0]))  # right
_xc0 /= np.linalg.norm(_xc0)
_yc0 = np.cross(_zc0, _xc0)                # down (image v)
_R0 = np.stack([_xc0, _yc0, _zc0], axis=1)
_T0 = _t0.copy()


def rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_extrinsics(yaw: float):
    """Return (R_bc, t_b): camera-to-base rotation and camera position in base frame."""
    R = rz(yaw) @ _R0
    t = rz(yaw) @ _T0
    return R, t


def project(R_bc: np.ndarray, t_b: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Project a base-frame point to camera frame coords (or None if behind camera)."""
    p_c = R_bc.T @ (p_b - t_b)
    if p_c[2] < NEAR:
        return None
    return p_c


def render(ee_pos: np.ndarray, tgt_pos: np.ndarray, yaw: float,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Render a 64x64x3 float32 image in [0,1]. rng unused (deterministic render)."""
    R_bc, t_b = camera_extrinsics(yaw)
    img = np.zeros((IMG, IMG, 3), dtype=np.float32)
    uu, vv = np.meshgrid(np.arange(IMG), np.arange(IMG))
    blobs = [(ee_pos, EE_COLOR, BLOB_RADIUS),
             (tgt_pos, TGT_COLOR, BLOB_RADIUS)]
    for d in DISTRACTORS:
        blobs.append((d, DIST_COLOR, DIST_RADIUS))
    for pos, color, radius in blobs:
        p_c = project(R_bc, t_b, pos)
        if p_c is None:
            continue
        u0 = FOCAL * p_c[0] / p_c[2] + CX
        v0 = FOCAL * p_c[1] / p_c[2] + CY
        if not (-6 <= u0 <= IMG + 6 and -6 <= v0 <= IMG + 6):
            continue
        sig = FOCAL * radius / p_c[2]
        d2 = (uu - u0) ** 2 + (vv - v0) ** 2
        mask = BLOB_PEAK * np.exp(-d2 / (2.0 * sig * sig))
        for c in range(3):
            img[..., c] += mask * color[c]
    return np.clip(img, 0.0, 1.0)


def sample_state(rng: np.random.Generator):
    """Sample EE and target uniformly in the cube, away from the fixed distractors."""
    while True:
        ee = rng.uniform(WS_LO, WS_HI, 3)
        tgt = rng.uniform(WS_LO, WS_HI, 3)
        d_ee = np.min(np.linalg.norm(DISTRACTORS - ee, axis=1))
        d_tg = np.min(np.linalg.norm(DISTRACTORS - tgt, axis=1))
        if d_ee > 0.18 and d_tg > 0.18:
            return ee, tgt


def oracle_delta(p_ee: np.ndarray, p_tgt: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Base-frame proportional controller step with small noise."""
    d = p_tgt - p_ee
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.zeros(3)
    step = float(np.clip(PROP_GAIN * n, 0.0, V_MAX))
    return d / n * step + rng.normal(0.0, ORACLE_NOISE, 3)


def step_world(p_ee: np.ndarray, delta_b: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Apply base-frame delta (clipped) plus small actuation noise.

    The EE is clamped to the workspace cube (joint limits analog), so a
    consistently wrong policy stalls at the boundary instead of escaping."""
    clipped = np.clip(delta_b, -V_MAX, V_MAX)
    p = p_ee + clipped + rng.normal(0.0, DYN_NOISE, 3)
    return np.clip(p, WS_LO, WS_HI)


def gen_episode(rng: np.random.Generator):
    """Generate an oracle demo. Returns states (list of base positions), deltas
    (list of base-frame oracle deltas), and target position."""
    ee, tgt = sample_state(rng)
    states, deltas = [], []
    p = ee.copy()
    for _ in range(MAX_STEPS):
        states.append(p.copy())
        d = oracle_delta(p, tgt, rng)
        deltas.append(d)
        p = step_world(p, d, rng)
        if np.linalg.norm(p - tgt) < SUCCESS_THRESH:
            break
    return states, deltas, tgt
