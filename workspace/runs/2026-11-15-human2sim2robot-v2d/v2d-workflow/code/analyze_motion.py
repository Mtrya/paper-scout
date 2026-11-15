#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def position_stats(values, fps):
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim == 3 and positions.shape[1] == 1:
        positions = positions[:, 0, :]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected one 3D position per frame, got {positions.shape}")
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    displacement = positions - positions[0]
    return {
        "shape": list(positions.shape),
        "start": positions[0].tolist(),
        "end": positions[-1].tolist(),
        "axis_range": np.ptp(positions, axis=0).tolist(),
        "max_rise_from_start_m": float(displacement[:, 2].max()),
        "step_distance_m": stats(steps),
        "step_speed_m_per_s": stats(steps * fps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motion", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(args.motion).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Expected one sequence row, got {len(rows)}")
    row = rows[0]
    fps = float(row["fps"])
    joints = np.asarray(row["robot_joint_positions"], dtype=np.float64)
    contact = np.asarray(row["hand_contact_active"], dtype=np.float64)
    if contact.shape == (2, len(joints)):
        contact = contact.T
    if contact.shape != (len(joints), 2):
        raise ValueError(f"Expected two per-frame contact signals, got {contact.shape}")
    active = contact > 0.5
    any_active = active.any(axis=1)
    transitions = np.diff(np.pad(any_active.astype(np.int32), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    durations = (ends - starts) / fps
    result = {
        "label": args.label,
        "path": str(args.motion),
        "metadata": {
            key: row.get(key)
            for key in (
                "schema_version", "motion_kind", "source_dataset", "fps",
                "coord_frame", "object_name", "source_kind",
            )
        },
        "frames": len(joints),
        "duration_seconds": len(joints) / fps,
        "robot_joint_shape": list(joints.shape),
        "joint_step_l2_rad": stats(np.linalg.norm(np.diff(joints, axis=0), axis=1)),
        "contact": {
            "shape": list(contact.shape),
            "active_frame_fraction_per_side": active.mean(axis=0).tolist(),
            "any_contact_frame_fraction": float(any_active.mean()),
            "contact_bout_count": len(durations),
            "longest_contact_bout_seconds": float(durations.max(initial=0.0)),
        },
        "object_body_position": position_stats(row["object_body_position"], fps),
        "object_root_position": position_stats(row["object_root_position"], fps),
    }
    for key in ("ik_error_per_frame", "ik_num_iterations"):
        if row.get(key) is not None:
            result[key] = stats(row[key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
