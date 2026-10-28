import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def number(pattern, text):
    return float(re.search(pattern, text, re.MULTILINE).group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    clips = json.loads((args.results / "offline.json").read_text())
    vae = (args.results / "vae_rice.log").read_text()
    branch = (args.results / "branch_scale.log").read_text()
    summary = {
        "n_example_clips": len(clips),
        "camera_path_below_2cm": sum(v["camera"]["trans_path_m"] < 0.02 for v in clips.values()),
        "camera_rotation_below_1deg": sum(v["camera"]["max_rot_deg"] < 1 for v in clips.values()),
        "camera_path_median_m": float(np.median([v["camera"]["trans_path_m"] for v in clips.values()])),
        "camera_rotation_median_deg": float(np.median([v["camera"]["max_rot_deg"] for v in clips.values()])),
        "skeleton_ink_median": float(np.median([v["skeleton"]["ink_frac"] for v in clips.values()])),
        "rice_vae": {
            "roundtrip_ink_iou": number(r"mean ink IoU=([0-9.]+)", vae),
            "shift_2px_over_blank": number(r"dx=\s*2px\s+\|\|dz\|\|/\|\|z_skeleton-z_blank\|\| = ([0-9.]+)", vae),
            "shift_32px_over_blank": number(r"dx=\s*32px\s+\|\|dz\|\|/\|\|z_skeleton-z_blank\|\| = ([0-9.]+)", vae),
            "static_moment_exactly_zero": "frame0 moment all-zero=True" in vae,
        },
        "branch_rms_over_video": {
            "camera": number(r"cam/vid\(noise\) = ([0-9.]+)", branch),
            "skeleton": number(r"skel/vid\(noise\) = ([0-9.]+)", branch),
        },
        "intervention_rms_over_video": {
            "synthetic_camera_15deg_0p5m": number(r"\[gain\] camera .* = ([0-9.]+)$", branch),
            "skeleton_right_40px": number(r"\[gain\] skeleton .* = ([0-9.]+)$", branch),
        },
    }
    assert summary["n_example_clips"] == 18
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.3, 1]})
    for name, row in clips.items():
        ax0.scatter(row["camera"]["trans_path_m"], row["camera"]["max_rot_deg"], s=48, color="#31688e", alpha=0.85)
    labels = {"agibot_rice_cooker": "rice cooker", "xp10m_cut_cardboard": "cut cardboard", "xp10m_jigsaw_puzzle": "jigsaw", "xp_teleop_playdoh_tub": "PlayDoh"}
    for name, label in labels.items():
        row = clips[name]
        ax0.annotate(label, (row["camera"]["trans_path_m"], row["camera"]["max_rot_deg"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax0.axvline(0.02, color="#999999", linestyle="--", linewidth=1)
    ax0.axhline(1, color="#999999", linestyle="--", linewidth=1)
    ax0.set_xlabel("Camera translation path (m)")
    ax0.set_ylabel("Maximum rotation from first frame (degrees)")
    ax0.set_title("18 released example clips: camera-motion dose")
    ax0.text(0.98, 0.03, "5 below 2 cm; 6 below 1°", transform=ax0.transAxes, ha="right", fontsize=9)
    ax0.grid(alpha=0.15)

    names = ["Camera\n(condition)", "Skeleton\n(condition)", "Camera change\n(first 21 frames)", "Skeleton change\n(right 40 px)"]
    values = [summary["branch_rms_over_video"]["camera"], summary["branch_rms_over_video"]["skeleton"], summary["intervention_rms_over_video"]["synthetic_camera_15deg_0p5m"], summary["intervention_rms_over_video"]["skeleton_right_40px"]]
    colors = ["#31688e", "#35b779", "#fdb515", "#fd8d3c"]
    bars = ax1.bar(np.arange(4), values, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", fontsize=9)
    ax1.set_xticks(np.arange(4), names, fontsize=9)
    ax1.set_ylabel("RMS / video-token projection RMS")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("Condition strength and intervention size")
    ax1.text(0.03, 0.96, "5B midtrain; random video latent", transform=ax1.transAxes, va="top", fontsize=9, color="#555555")
    ax1.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
