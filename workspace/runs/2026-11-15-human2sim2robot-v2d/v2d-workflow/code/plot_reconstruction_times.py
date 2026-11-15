#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "v2d.anycalib.lib.video_to_calibration": "AnyCalib intrinsics",
    "v2d.moge.lib.video_to_depth": "MoGe depth",
    "v2d.droid_slam.lib.video_to_slam": "DROID-SLAM",
    "v2d.wilor.lib.video_to_hands": "WiLoR hands",
    "v2d.grounding_dino.lib.image_to_object_bboxes": "GroundingDINO",
    "v2d.wilor.lib.render_prompt_masks": "Hand prompts",
    "v2d.sam2.lib.video_to_masks": "SAM2 masks",
    "v2d.sam3d.lib.image_to_mesh": "SAM3D mesh",
    "v2d.foundation_pose.lib.run_estimate_mesh_scale": "Mesh scale",
    "v2d.foundation_pose.lib.run_video_to_poses": "Object tracking",
    "v2d.foundation_pose.lib.run_ekf_smoothing": "EKF smoothing",
    "v2d.hamer.lib.masks_to_hands": "HaMeR hands",
    "v2d.hamer.lib.render_hands_video": "Hand overlay",
    "v2d.hamer.lib.align_hands": "Hand–object alignment",
}
REFERENCE_STAGES = {
    "v2d.anycalib.lib.video_to_calibration",
    "v2d.grounding_dino.lib.image_to_object_bboxes",
    "v2d.sam3d.lib.image_to_mesh",
    "v2d.foundation_pose.lib.run_estimate_mesh_scale",
}
HAND_STAGES = {
    "v2d.wilor.lib.video_to_hands",
    "v2d.wilor.lib.render_prompt_masks",
    "v2d.hamer.lib.masks_to_hands",
    "v2d.hamer.lib.render_hands_video",
    "v2d.hamer.lib.align_hands",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    stages = sorted((item for item in data["successful_modules"] if item.get("role", "reconstruction") == "reconstruction"), key=lambda item: item["wall_seconds"])
    labels = [LABELS.get(item["module"], item["module"].rsplit(".", 1)[-1]) for item in stages]
    values = [item["wall_seconds"] for item in stages]
    colors = ["#6f8faf" if item["module"] in REFERENCE_STAGES else "#7c9a78" if item["module"] in HAND_STAGES else "#c58b5c" for item in stages]

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=140)
    bars = ax.barh(labels, values, color=colors, height=0.68)
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)
    ax.set_xlabel("Wall time on one RTX 4090 (seconds)")
    ax.set_title(f"Perception cost for a {data['source_video_seconds']:g} s egocentric video\n{data['observed_single_gpu_wall_seconds']:.1f} s across measured modules = {data['observed_wall_per_source_second']:.1f}× source duration", fontsize=13, pad=14)
    ax.set_xlim(0, max(values) * 1.20)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#6f8faf"),
        plt.Rectangle((0, 0), 1, 1, color="#c58b5c"),
        plt.Rectangle((0, 0), 1, 1, color="#7c9a78"),
    ]
    ax.legend(handles, ["Reference-frame asset", "Sequence processing", "Hand processing"], loc="lower right", frameon=False)
    fig.text(0.125, 0.012, "First successful run per module; downloads, setup, failed retries and host-side alignment excluded.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
