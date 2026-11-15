#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, help="One exported NPZ or a directory of episodes")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.episodes.glob("episode_*.npz")) if args.episodes.is_dir() else [args.episodes]
    if not paths:
        raise ValueError(f"No episodes found under {args.episodes}")
    lifts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            lift = np.asarray(data["object_pose"])[:, 0, 2]
            lift = lift - lift[0]
            fps = float(data["fps"]) if "fps" in data else 20.0
        lifts.append(lift)
    if len({len(lift) for lift in lifts}) != 1:
        raise ValueError("Episodes must have the same length for a median trajectory")
    lifts = np.asarray(lifts)
    time = np.arange(lifts.shape[1]) / fps
    median = np.median(lifts, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=180)
    ax.plot(time, lifts.T, color="#195c8c", linewidth=0.9, alpha=0.24)
    ax.plot(time, median, color="#123f5f", linewidth=2.4, label=f"Median of {len(paths)} rollouts")
    ax.axhline(0.10, color="#a33b31", linestyle="--", linewidth=1.4, label="10 cm lift threshold")
    ax.axhline(0.05, color="#b78427", linestyle=":", linewidth=1.6, label="5 cm hold threshold")
    ax.fill_between(time, 0.05, median, where=median >= 0.05, color="#4e8e68", alpha=0.14)
    ax.set(xlabel="Time (s)", ylabel="Object height relative to start (m)", xlim=(0, time[-1]), ylim=(-0.02, max(0.18, float(lifts.max()) + 0.01)))
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Public-reference expert: ten lift, hold, and return trajectories")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print({"episodes": len(paths), "frames_per_episode": lifts.shape[1], "fps": fps})


if __name__ == "__main__":
    main()
