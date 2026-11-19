"""Data access for the DASH real-weight probe on Jack97/ForceHarness-Data (raw NPZ form).

Layout facts taken from the dataset card and from forceharness/data/demos.py in the ForceHarness repo:

  * ``raw/<task>/episode_<seed>.npz``: ``action`` (T,7) normalised OSC delta, ``kp`` (T,6) stiffness,
    proprioception arrays with T+1 frames (frame i is what the controller saw before action i),
    ``img_<cam>_jpeg``/``img_<cam>_off`` JPEG frames (128x128, quality 90, BGR, upside down as robosuite renders),
    ``meta_json`` with ``scene0`` (the noisy object poses the controller was given).
  * The LeRobot part of the same repo stores ``state`` as proprio 11 (eef_pos 3 + rot6d 6 + gripper 2) followed by
    the scene vector (5 per object: pos 3 + sin/cos yaw).  The scene vector is ground-truth object pose, so the
    probe reads proprio only; ``FULL_STATE_DIM=26`` exists solely for the leak-control arm.

Everything here is pure numpy + PIL so it runs without the ForceHarness package.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

CAMS = ("agentview", "robot0_eye_in_hand")
PROPRIO_DIM = 11
FULL_STATE_DIM = 26
ACTION_DIM = 13
H = 8
DELTA_MAX = 8
FPS = 20

# Verbatim from ForceHarness scripts/baselines/pi05/convert_to_lerobot.py (these strings are model inputs).
PROMPTS = {
    "weight_sort": "pick up the cube, judge its weight, and put it in the heavy or light zone",
    "com_sort": "pick up the bar at its center, judge its balance, and put it in the balanced or offset zone",
}
TASKS = ["weight_sort", "com_sort"]


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = x * x + y * y + z * z + w * w
    s = np.where(n > 0, 2.0 / np.maximum(n, 1e-12), 0.0)
    c0 = np.stack([1 - s * (y * y + z * z), s * (x * y + z * w), s * (x * z - y * w)], -1)
    c1 = np.stack([s * (x * y - z * w), 1 - s * (x * x + z * z), s * (y * z + x * w)], -1)
    return np.concatenate([c0, c1], -1).astype(np.float32)


def proprio(eef_pos: np.ndarray, eef_quat: np.ndarray, gripper_qpos) -> np.ndarray:
    p = np.asarray(eef_pos, np.float32)
    g = np.asarray(gripper_qpos if gripper_qpos is not None else [], np.float32)
    if g.size == 0 or g.shape[:-1] != p.shape[:-1]:
        g = np.zeros(p.shape[:-1] + (2,), np.float32)
    return np.concatenate([p, quat_to_rot6d(eef_quat), g[..., :2]], -1)


def load_episode(path: Path) -> Dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if not k.startswith("img_") and k != "meta_json"}
    d["meta"] = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    d["jpeg"] = {c: (z[f"img_{c}_jpeg"], z[f"img_{c}_off"]) for c in d["meta"].get("cams", [])}
    T = len(d["action"])
    d["T"] = T
    d["proprio"] = proprio(d["eef_pos"][:T], d["eef_quat"][:T], d["gripper_qpos"][:T])
    adim = 7 if not np.isnan(d["action"][:, 6]).any() else 6
    d["actions13"] = np.concatenate([d["action"][:, :adim], d["kp"]], -1).astype(np.float32)
    if adim == 6:
        raise ValueError(f"{path}: 6-D OSC task, not part of this probe")
    return d


def decode_frame(d: Dict, cam: str, i: int) -> np.ndarray:
    """Upright RGB uint8 128x128.  The recorder stores BGR JPEG frames upside down."""
    buf, off = d["jpeg"][cam]
    raw = np.asarray(buf[off[i]:off[i + 1]], dtype=np.uint8).tobytes()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.ascontiguousarray(np.asarray(img)[::-1])


def split_episodes(data_root: Path, n_train: int, n_val: int, seed: int = 0):
    """Episode-disjoint split; every episode goes entirely to one side (no frame-level leakage)."""
    out = {}
    rng = np.random.default_rng(seed)
    seeds_file = data_root / "seeds.json"
    known = json.loads(seeds_file.read_text()) if seeds_file.exists() else None
    for task in TASKS:
        if known is not None and task in known:
            files = [data_root / "raw" / task / f"episode_{s:05d}.npz" for s in sorted(known[task])]
        else:
            files = sorted((data_root / "raw" / task).glob("episode_*.npz"),
                           key=lambda p: int(p.stem.split("_")[1]))
        perm = rng.permutation(len(files))
        val = [files[i] for i in sorted(perm[:n_val])]
        train = [files[i] for i in sorted(perm[n_val:n_val + n_train])]
        out[task] = dict(train=train, val=val)
    return out


def index_json(data_root: Path, split: Dict) -> Dict:
    out = {}
    for task, parts in split.items():
        out[task] = {k: [p.name for p in v] for k, v in parts.items()}
    return out
