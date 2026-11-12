#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
CAMERAS = ["main", "right", "left_wrist", "right_wrist"]


def contiguous_runs(labels):
    runs = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            if labels[start] > 0:
                runs.append({"subtask_index": int(labels[start]), "start": start, "end": index - 1})
            start = index
    return runs


def read_selected_frames(path, selected):
    import os
    import re

    selected = sorted(set(selected))
    binary = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    probe = subprocess.run([binary, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", probe.stderr)
    if not match:
        raise RuntimeError(probe.stderr)
    width, height = map(int, match.groups())
    expression = "+".join(f"eq(n\\,{index})" for index in selected)
    process = subprocess.run(
        [binary, "-i", str(path), "-vf", f"select={expression}", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    )
    frame_size = width * height * 3
    if len(process.stdout) != len(selected) * frame_size:
        raise RuntimeError(f"decoded {len(process.stdout) / frame_size} frames, expected {len(selected)}")
    frames = {
        index: np.frombuffer(process.stdout[offset * frame_size:(offset + 1) * frame_size], dtype=np.uint8).reshape(height, width, 3).copy()
        for offset, index in enumerate(selected)
    }
    return frames, {"size": (width, height)}


def draw_hand(ax, points, intrinsics, color):
    if np.all(points == 0) or not np.all(np.isfinite(points)):
        return 0
    fx, fy, cx, cy = intrinsics[:4]
    z = points[:, 2]
    uv = np.column_stack([fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy])
    valid = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
    for a, b in BONES:
        if valid[a] and valid[b]:
            ax.plot(uv[[a, b], 0], uv[[a, b], 1], color=color, lw=1.6, alpha=0.9)
    ax.scatter(uv[valid, 0], uv[valid, 1], s=8, c=color, edgecolors="white", linewidths=0.25)
    return int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.dataset
    info = json.loads((root / "meta/info.json").read_text())
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    data = table.to_pydict()
    subtasks = {r["subtask_index"]: r["subtask"] for r in pq.read_table(root / "meta/subtasks.parquet").to_pylist()}
    task = pq.read_table(root / "meta/tasks.parquet").to_pylist()[0]["task"]
    episode = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    labels = np.asarray(data["subtask_index"])
    timestamps = np.asarray(data["timestamp"])
    runs = contiguous_runs(labels)
    for run in runs:
        run["subtask"] = subtasks[run["subtask_index"]]
        run["timestamp_start"] = float(timestamps[run["start"]])
        run["timestamp_end"] = float(timestamps[run["end"]])
    wanted = [1, 2, 4, 6]
    selected_runs = []
    for index in wanted:
        candidates = [r for r in runs if r["subtask_index"] == index]
        selected_runs.append(max(candidates, key=lambda r: r["end"] - r["start"]))
    selected_frames = [(r["start"] + r["end"]) // 2 for r in selected_runs]
    video_meta = {}
    frames_by_camera = {}
    for camera in CAMERAS:
        path = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        frames, meta = read_selected_frames(path, selected_frames)
        frames_by_camera[camera] = frames
        video_meta[camera] = {k: str(v) for k, v in meta.items() if k in ["size", "fps", "codec", "duration"]}
    left = np.asarray(data["observation.state.hand_left_cam"], dtype=np.float32).reshape(-1, 21, 3)
    right = np.asarray(data["observation.state.hand_right_cam"], dtype=np.float32).reshape(-1, 21, 3)
    camera_position = np.asarray(data["observation.state.camera_position"], dtype=np.float32)
    camera_quaternion = np.asarray(data["observation.state.camera_rotation"], dtype=np.float32)
    left_conf = np.asarray(data["left_hand_pose_confidence"], dtype=np.float32)
    right_conf = np.asarray(data["right_hand_pose_confidence"], dtype=np.float32)
    intrinsics = episode["camera_intrinsics"][:8]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 7.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.3, 1.0], hspace=0.22, wspace=0.12, top=0.88)
    short_tasks = {1: "Pick flowers", 2: "Gather stems", 3: "Pick rubber band", 4: "Wrap band", 5: "Secure band", 6: "Place bouquet"}
    for column, (run, frame_index) in enumerate(zip(selected_runs, selected_frames)):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(frames_by_camera["main"][frame_index])
        visible_left = draw_hand(ax, left[frame_index], intrinsics, "#55A868")
        visible_right = draw_hand(ax, right[frame_index], intrinsics, "#C44E52")
        ax.set_title(f"{timestamps[frame_index]:.1f}s · {short_tasks[run['subtask_index']]}", fontsize=10, loc="left")
        ax.text(10, 710, f"projected joints L/R: {visible_left}/{visible_right}", color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
        ax.axis("off")
    ax = fig.add_subplot(grid[1, 0:2])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for run in runs:
        sl = slice(run["start"], run["end"] + 1)
        color = colors[run["subtask_index"] % 10]
        for hand, points, style in [("L", left, "-"), ("R", right, "--")]:
            valid = ~np.all(points[sl] == 0, axis=(1, 2))
            xyz = points[sl][valid, 0, :]
            if len(xyz):
                ax.plot(xyz[:, 0], xyz[:, 2], style, color=color, lw=1.5, alpha=0.85)
        ax.text(float(np.mean(left[sl, 0, 0])), float(np.mean(left[sl, 0, 2])), str(run["subtask_index"]), color=color, fontsize=9)
    ax.set(xlabel="Camera-frame wrist x (m)", ylabel="Camera-frame wrist z (m)", title="Wrist trajectories; numbers mark T2 subtasks")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax = fig.add_subplot(grid[1, 2:4])
    ax.plot(timestamps, left_conf, color="#55A868", lw=1, label="left hand")
    ax.plot(timestamps, right_conf, color="#C44E52", lw=1, label="right hand")
    for run in runs:
        ax.axvspan(run["timestamp_start"], run["timestamp_end"], color=colors[run["subtask_index"] % 10], alpha=0.1)
        ax.text(run["timestamp_start"], 1.04, str(run["subtask_index"]), color=colors[run["subtask_index"] % 10])
    ax.set(xlabel="Episode time (s)", ylabel="Tracking confidence", ylim=(0, 1.08), title="Tracking quality and contiguous T2 intervals")
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle(task, fontsize=12, x=0.02, ha="left")
    fig.savefig(args.output / "human1m_aligned_episode.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    contact_path = args.output / "human1m_wrist_grid.png"
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for column, (run, frame_index) in enumerate(zip(selected_runs, selected_frames)):
        for row, camera in enumerate(["left_wrist", "right_wrist"]):
            axes[row, column].imshow(frames_by_camera[camera][frame_index])
            axes[row, column].set_title(f"{camera} · {timestamps[frame_index]:.1f}s")
            axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(contact_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "task": task,
        "frames": table.num_rows,
        "fps": info["fps"],
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "timestamp_max_step_seconds": float(np.diff(timestamps).max()),
        "timestamp_min_step_seconds": float(np.diff(timestamps).min()),
        "camera_position_finite": bool(np.all(np.isfinite(camera_position))),
        "camera_rotation_finite": bool(np.all(np.isfinite(camera_quaternion))),
        "hand_mask_values": sorted(int(x) for x in np.unique(data["hand_mask"])),
        "left_confidence_quantiles": np.quantile(left_conf, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
        "right_confidence_quantiles": np.quantile(right_conf, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
        "left_zero_position_frames": int(np.all(left == 0, axis=(1, 2)).sum()),
        "right_zero_position_frames": int(np.all(right == 0, axis=(1, 2)).sum()),
        "video_meta": video_meta,
        "subtask_runs": runs,
        "selected_frames": selected_frames,
        "camera_intrinsics_first_view": intrinsics,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
