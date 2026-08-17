#!/usr/bin/env python3
"""omega-HOME probe v2: concurrency (leg vs arm joint activity), teleop fidelity, retries.

Signal map (G1 29-dof order in this dataset: 12 leg + 3 waist + 14 arm, verified by ptp):
  loco activity  = mean |d(body_q[:, :12])|/dt + |base_ang_vel_z|
  manip activity = mean |d(body_q[:, 15:])|/dt + hand-point speed (vr_3point)
"""
import glob, json, sys, os, csv, collections
import numpy as np
import h5py

ROOT = "code/omega-home"
OUT = sys.argv[1] if len(sys.argv) > 1 else "drafts/omega_probe.csv"

def episode_stats(path):
    with h5py.File(path, "r") as h:
        t = h["sonic/ros_timestamp"][:].astype(np.float64)
        n = len(t)
        dt = np.diff(t); dt_pos = dt[dt > 0]
        dur = t[-1] - t[0]
        fps = (n - 1) / dur if dur > 0 else np.nan  # span-based; timestamps quantized to 0.02s grid
        dtc = np.clip(dt, 1e-3, None)

        bq = h["sonic/body_q"][:].astype(np.float64)
        qv = np.abs(np.diff(bq, axis=0)) / dtc[:, None]          # (N-1,29) joint speeds
        leg_v = qv[:, :12].mean(axis=1)
        waist_v = qv[:, 12:15].mean(axis=1)
        arm_v = qv[:, 15:].mean(axis=1)
        ang_z = np.abs(h["sonic/base_ang_vel"][:-1, 2])

        vp = h["sonic/vr_3point_position"][:].reshape(n, 3, 3)
        hand_v = (np.linalg.norm(np.diff(vp[:, 1:3], axis=0), axis=2).max(axis=1) / dtc)

        # thresholds: rad/s for joints, rad/s for yaw, m/s for hands
        loco = np.maximum(leg_v / 0.15, ang_z / 0.15)             # normalized [~0,inf]
        mani = np.maximum(arm_v / 0.15, hand_v / 0.15)
        loco_on = loco > 1.0
        mani_on = mani > 1.0
        any_on = loco_on | mani_on
        frac_loco = loco_on.mean(); frac_mani = mani_on.mean()
        frac_conc = (loco_on & mani_on).mean()
        conc_motion = (loco_on & mani_on).sum() / any_on.sum() if any_on.sum() else np.nan

        # teleop tracking delay + error
        qt = h["sonic/body_q_target"][:]; qm = h["sonic/body_q_measured"][:]
        lags, errs = [], []
        med = np.median(dt_pos) if len(dt_pos) else 0.04
        for j in range(qt.shape[1]):
            a, b = qt[:, j], qm[:, j]
            if np.std(a) < 1e-3: continue
            errs.append(float(np.sqrt(np.mean((a - b) ** 2))))
            c = np.correlate(b - b.mean(), a - a.mean(), mode="full")
            lag = int(np.argmax(c)) - (n - 1)
            if -25 <= lag <= 25: lags.append(lag * med * 1000)
        lag_ms = float(np.median(lags)) if lags else np.nan
        rmse = float(np.mean(errs)) if errs else np.nan

        # grasp cycles from trigger (full close 1 -> open 0 transitions)
        cyc = 0
        for k in ("action/left_grip", "action/right_grip"):
            if k in h:
                g = h[k][:, 0]
                closed = g > 0.9
                cyc += int(np.sum((~closed[1:]) & closed[:-1]))  # close->open
        task = path.split("code/omega-home/")[1].split("/")[0]
        return dict(task=task, ep=os.path.basename(os.path.dirname(path)),
                    frames=n, dur_s=round(dur,2), fps=round(fps,2),
                    frac_loco=round(frac_loco,4), frac_mani=round(frac_mani,4),
                    frac_concurrent=round(frac_conc,4), conc_given_motion=round(conc_motion,4) if conc_motion==conc_motion else "",
                    teleop_lag_ms=round(lag_ms,1) if lag_ms==lag_ms else "", track_rmse_rad=round(rmse,4),
                    grasp_cycles=cyc)

rows = []
files = sorted(glob.glob(f"{ROOT}/*/*/state_action.hdf5"))
for i, f in enumerate(files):
    try:
        rows.append(episode_stats(f))
    except Exception as e:
        print("fail", f, repr(e), file=sys.stderr)
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(files)}", file=sys.stderr)

with open(OUT, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", OUT, len(rows))

byt = collections.defaultdict(list)
for r in rows: byt[r["task"]].append(r)
for tk, rs in sorted(byt.items()):
    g = lambda k: np.nanmean([r[k] for r in rs if r[k] != ""])
    print(f"{tk:24s} n={len(rs):3d} dur={g('dur_s'):6.1f}s fps={g('fps'):5.1f} "
          f"loco={g('frac_loco'):.2f} mani={g('frac_mani'):.2f} conc={g('frac_concurrent'):.3f} "
          f"conc|motion={g('conc_given_motion'):.3f} lag={g('teleop_lag_ms'):6.1f}ms "
          f"rmse={g('track_rmse_rad'):.3f} grasps={g('grasp_cycles'):.1f}")
