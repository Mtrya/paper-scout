#!/usr/bin/env python3
import json
import math
from pathlib import Path


def main():
    shape_noise = 5.0
    shape_reference = 11.0
    velocity = 2.0
    step = 0.25
    shape_t = 1.0
    t = 0.5
    denoised = shape_noise + step * velocity
    shape_path = (1 - shape_t) * shape_noise + shape_t * shape_reference
    alpha = 0.9
    guided = (1 - alpha) * denoised + alpha * shape_path
    fixed = 0.0 * denoised + 1.0 * shape_path

    previous_pose = 4.0
    pose_path = (1 - t) * 0.0 + t * previous_pose
    pose_denoised = previous_pose + step * 1.0
    pose_alpha = 0.8
    pose_guided = (1 - pose_alpha) * pose_denoised + pose_alpha * pose_path

    hand_depth_h = 0.5
    hand_depth_m = 2.0
    scale = hand_depth_h / hand_depth_m
    hand_h = (0.1, 0.0, 0.5)
    hand_m = (0.2, 0.0, 2.0)
    object_m = (0.4, 0.1, 2.2)
    object_target = tuple(h + scale * (o - hm) for h, o, hm in zip(hand_h, object_m, hand_m))

    reference_preserved = math.isclose(fixed, shape_reference, abs_tol=1e-12)
    assert reference_preserved
    assert guided != shape_reference
    assert pose_guided < pose_denoised
    assert all(math.isclose(actual, expected, abs_tol=1e-12) for actual, expected in zip(object_target, (0.15, 0.025, 0.55)))
    result = {
        "shape": {
            "denoised": denoised,
            "reference_path": shape_path,
            "alpha": alpha,
            "guided": guided,
            "alpha_one_exactly_preserves_reference": reference_preserved,
        },
        "pose": {
            "previous": previous_pose,
            "denoised": pose_denoised,
            "alpha": pose_alpha,
            "guided": pose_guided,
            "moves_toward_previous_pose": pose_guided < pose_denoised,
        },
        "metric_anchor": {
            "scale": scale,
            "object_target": object_target,
            "formula": "c_hand_H + (z_hand_H/z_hand_M) * (c_obj_M - c_hand_M)",
        },
    }
    output = Path(__file__).with_name("geometry_probe.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
