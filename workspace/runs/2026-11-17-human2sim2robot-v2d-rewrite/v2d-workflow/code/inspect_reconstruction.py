#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def pose_metrics(transforms, valid):
    transforms = np.asarray(transforms, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    selected = transforms[valid]
    result = {
        "valid_frames": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "finite_fraction": float(np.isfinite(transforms).mean()),
    }
    if len(selected) < 2:
        return result
    positions = selected[:, :3, 3]
    rotations = selected[:, :3, :3]
    relative = rotations[1:] @ rotations[:-1].transpose(0, 2, 1)
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    result.update({
        "translation_range_m": np.ptp(positions, axis=0).tolist(),
        "translation_step_m": stats(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
        "rotation_step_degrees": stats(angles),
        "start_position_m": positions[0].tolist(),
        "end_position_m": positions[-1].tolist(),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bundle / "result.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    vertices = 0
    faces = 0
    with (args.bundle / "mesh.obj").open(errors="replace") as stream:
        for line in stream:
            vertices += line.startswith("v ")
            faces += line.startswith("f ")
    result = {
        "frames": int(manifest["n_frames"]),
        "array_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "mesh": {"vertices": vertices, "faces": faces, **manifest.get("mesh", {})},
        "object_mesh_vertex_scale_applied": manifest["sources"].get("object_mesh_vertex_scale_applied"),
        "camera": pose_metrics(arrays["camera_to_world_transform"], arrays["camera_is_valid"]),
        "object_in_camera": pose_metrics(arrays["object_to_camera_transform"], arrays["object_is_valid"]),
        "object_in_world": pose_metrics(arrays["object_to_world_transform"], arrays["object_is_valid"]),
    }
    object_positions = arrays["object_to_camera_transform"][:, :3, 3]
    for side in ("left", "right"):
        wrist = arrays[f"hand_{side}_wrist_to_camera_transform"]
        valid = arrays[f"hand_{side}_is_valid"] & arrays["object_is_valid"]
        distances = np.linalg.norm(wrist[:, :3, 3] - object_positions, axis=1)[valid]
        result[f"hand_{side}"] = {
            "valid_fraction": float(arrays[f"hand_{side}_is_valid"].mean()),
            "scale": stats(arrays[f"hand_{side}_scale"]),
            "wrist_to_object_origin_distance_m": stats(distances),
            "distance_interpretation": "Origin distance is a diagnostic, not a surface-contact measurement.",
        }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "frames": result["frames"],
        "mesh": result["mesh"],
        "object_in_camera": result["object_in_camera"],
        "hand_left": result["hand_left"],
        "hand_right": result["hand_right"],
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
