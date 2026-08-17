#!/usr/bin/env python3
"""omega-HOME report plots: concurrency timeseries + per-task summary + teleop lag."""
import glob, csv, collections
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "drafts/omega_plots"
import os; os.makedirs(OUT, exist_ok=True)

# ---- 1. example episodes: time series of loco vs manip activity ----
def series(path):
    with h5py.File(path, "r") as h:
        t = h["sonic/ros_timestamp"][:].astype(np.float64)
        dt = np.clip(np.diff(t), 1e-3, None)
        bq = h["sonic/body_q"][:].astype(np.float64)
        qv = np.abs(np.diff(bq, axis=0)) / dt[:, None]
        leg = qv[:, :12].mean(axis=1)
        arm = qv[:, 15:].mean(axis=1)
        angz = np.abs(h["sonic/base_ang_vel"][:-1, 2])
        vp = h["sonic/vr_3point_position"][:].reshape(len(t), 3, 3)
        hand = np.linalg.norm(np.diff(vp[:, 1:3], axis=0), axis=2).max(axis=1) / dt
        tt = t[1:] - t[0]
    return tt, leg, arm, angz, hand

eps = {
    "mop_floor": sorted(glob.glob("code/omega-home/mop_floor/*/state_action.hdf5"))[3],
    "pick_and_place_apple": sorted(glob.glob("code/omega-home/pick_and_place_apple/*/state_action.hdf5"))[3],
}
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=False)
for ax, (name, path) in zip(axes, eps.items()):
    tt, leg, arm, angz, hand = series(path)
    ax.plot(tt, leg, label="leg joints |dq| (rad/s)", lw=1.2)
    ax.plot(tt, arm, label="arm joints |dq| (rad/s)", lw=1.2)
    ax.plot(tt, angz, label="base |yaw rate| (rad/s)", lw=1.0, alpha=0.8)
    ax.plot(tt, hand, label="teleop hand speed (m/s)", lw=1.0, alpha=0.8)
    ax.axhline(0.15, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.set_title(name.replace("_", " "), fontsize=11)
    ax.set_ylabel("activity")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.4)
fig.tight_layout()
fig.savefig(f"{OUT}/timeseries_concurrency.png", dpi=150)
plt.close(fig)

# ---- 2. per-task summary bars + teleop lag ----
rows = list(csv.DictReader(open("drafts/omega_probe.csv")))
byt = collections.defaultdict(list)
for r in rows:
    byt[r["task"]].append(r)
tasks = sorted(byt)
def agg(t, k):
    return np.array([float(r[k]) for r in byt[t] if r[k] != ""])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
x = np.arange(len(tasks))
conc = [agg(t, "conc_given_motion").mean() for t in tasks]
axes[0].bar(x, conc, color="#3a7ca5")
axes[0].set_xticks(x, [t.replace("_", "\n") for t in tasks], fontsize=8)
axes[0].set_ylabel("P(loco ∧ manip | any motion)")
axes[0].set_title("concurrency share of motion frames")
lags = [agg(t, "teleop_lag_ms") for t in tasks]
axes[1].boxplot(lags, positions=x, showfliers=False)
axes[1].set_xticks(x, [t.replace("_", "\n") for t in tasks], fontsize=8)
axes[1].set_ylabel("teleop tracking lag (ms)")
axes[1].set_title("target→measured joint lag")
rmse = [agg(t, "track_rmse_rad") for t in tasks]
axes[2].boxplot(rmse, positions=x, showfliers=False)
axes[2].set_xticks(x, [t.replace("_", "\n") for t in tasks], fontsize=8)
axes[2].set_ylabel("tracking RMSE (rad)")
axes[2].set_title("command tracking error")
fig.tight_layout()
fig.savefig(f"{OUT}/task_summary.png", dpi=150)
plt.close(fig)
print("saved", OUT)
