#!/usr/bin/env python3
"""Experiment: quantify recovery-like content and internal consistency of HiFi-UMI-2K.

Conjecture (from the Xiaomi-vs-HiFi-UMI thread): UMI-style data's weakness is
not nominal trajectories but the recovery/correction distribution. HiFi-UMI
claims to "deliberately collect rare failure-and-recovery episodes" but reports
no number. This script produces that number, plus two consistency checks the
paper implies but never shows:

1. ACTION-STATE SHIFT: actions are declared "absolute_next_state_target", so
   action[t] must equal state[t+1] within float tolerance.
2. RECOVERY CONTENT: recovery/correction behavior is operationalized as
   direction reversals of the smoothed end-effector velocity (angle between
   consecutive velocity vectors > REVERSAL_ANGLE), which is what
   overshoot-backoff-retry looks like kinematically. We count reversal events
   per episode and per 10s of motion, and the fraction of frames in
   near-contact slow-motion (a proxy for precision phases).
3. TRAJECTORY SMOOTHNESS: position jitter (high-frequency displacement noise)
   as a rough upper bound consistent with the claimed ~3 mm pose accuracy.

Input: one LeRobot v3 data parquet (chunk-0000, 1125 episodes) + episodes meta.
"""
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hifi-data/data.parquet"
EPS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hifi-data/episodes.parquet"

FPS = 25.0
SMOOTH_WIN = 5            # 0.2 s velocity smoothing
REVERSAL_ANGLE = 120.0    # degrees between consecutive smoothed velocities
MIN_SPEED = 0.01          # m/s; below this the direction is meaningless

df = pd.read_parquet(DATA)
eps = pd.read_parquet(EPS)
print(f"frames={len(df)}, episodes={df['episode_index'].nunique()}")

state = np.stack(df["observation.state"].to_numpy())   # [N,20] right|left 10d
action = np.stack(df["action"].to_numpy())             # [N,20]
ep = df["episode_index"].to_numpy()
valid = df["valid.frame"].to_numpy()

# ---------- check 1: action[t] == state[t+1] (within episode) ----------
same_ep = ep[1:] == ep[:-1]
lhs = action[:-1][same_ep]
rhs = state[1:][same_ep]
diff = np.abs(lhs - rhs)
pos_diff = np.maximum(diff[:, 0:3], diff[:, 10:13]).max()
grip_diff = np.maximum(diff[:, 9], diff[:, 19]).max()
print("\n[1] action[t] vs state[t+1] (within-episode frames:", same_ep.sum(), ")")
print(f"    max |pos diff|      = {pos_diff:.6f} m")
print(f"    max |gripper diff|  = {grip_diff:.6f} rad")
print(f"    mean |pos diff|     = {np.maximum(diff[:,0:3], diff[:,10:13]).mean():.8f} m")

# ---------- per-episode kinematics ----------
def analyze_episode(g):
    s = np.stack(g["observation.state"].to_numpy())
    v = g["valid.frame"].to_numpy()
    out = {}
    for side, sl in (("right", slice(0, 3)), ("left", slice(10, 13))):
        p = s[:, sl]
        dp = np.diff(p, axis=0) * FPS                    # velocity m/s
        # smoothing
        k = np.ones(SMOOTH_WIN) / SMOOTH_WIN
        dps = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), 0, dp)
        speed = np.linalg.norm(dps, axis=1)
        moving = speed > MIN_SPEED
        cosang = np.sum(dps[1:] * dps[:-1], axis=1) / (
            speed[1:] * speed[:-1] + 1e-12)
        rev = (cosang < np.cos(np.radians(REVERSAL_ANGLE))) & moving[1:] & moving[:-1]
        dur = len(p) / FPS
        out[side] = dict(
            reversals=int(rev.sum()),
            dur=dur,
            move_frac=float(moving.mean()),
            rev_per_10s=float(rev.sum() / max(dur / 10.0, 1e-9)),
            slow_frac=float((speed < 0.05).mean()),       # near-contact proxy
            jitter=float(np.median(np.linalg.norm(dp - dps, axis=1))),
            speed_med=float(np.median(speed[moving])) if moving.any() else 0.0,
        )
    return out

rows = []
for epi, g in df.groupby("episode_index"):
    r = analyze_episode(g)
    row = {"episode": epi, "len_s": len(g) / FPS}
    for side in ("right", "left"):
        for key, val in r[side].items():
            row[f"{side}_{key}"] = val
    rows.append(row)
R = pd.DataFrame(rows)

print("\n[2] recovery-content statistics (n=%d episodes)" % len(R))
for side in ("right", "left"):
    r10 = R[f"{side}_rev_per_10s"]
    print(f"    {side}: reversals/10s  median={r10.median():.2f}  "
          f"p10={r10.quantile(.1):.2f}  p90={r10.quantile(.9):.2f}  max={r10.max():.2f}")
zero_rev = ((R["right_reversals"] == 0) & (R["left_reversals"] == 0)).mean()
print(f"    episodes with ZERO reversals on both hands: {zero_rev:.1%}")
high = ((R["right_rev_per_10s"] > 1.0) | (R["left_rev_per_10s"] > 1.0)).mean()
print(f"    episodes with >1 reversal/10s (recovery-rich): {high:.1%}")

print("\n[3] smoothness / jitter")
j = np.concatenate([R["right_jitter"], R["left_jitter"]])
print(f"    median per-frame jitter (unsmoothed-minus-0.2s-smooth): {np.median(j)*1000:.2f} mm")
sp = np.concatenate([R["right_speed_med"], R["left_speed_med"]])
print(f"    median moving speed: {np.median(sp):.3f} m/s")
print(f"    episode duration: median={R['len_s'].median():.1f}s  "
      f"p90={R['len_s'].quantile(.9):.1f}s")

R.to_csv("/tmp/hifi-data/episode_kinematics.csv", index=False)
print("\nsaved /tmp/hifi-data/episode_kinematics.csv")
