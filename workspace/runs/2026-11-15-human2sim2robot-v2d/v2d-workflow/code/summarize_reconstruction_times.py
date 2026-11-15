#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


VISUALIZATION_MODULES = {
    "v2d.hamer.lib.render_hands_video",
    "v2d.hamer.lib.render_hands_aligned_video",
}
MODULE_ROLES = {
    "v2d.wilor.lib.tracks_interpolate": "post_alignment",
    "v2d.gsplat_refinement.lib.refine_simple": "optional_refinement",
    "v2d.geocalib.lib.video_to_calibration": "reconstruction_packaging",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-seconds", type=float, default=13.0)
    args = parser.parse_args()

    successful = {}
    previews = []
    for line in args.log.read_text().splitlines():
        record = json.loads(line)
        module = record["module"]
        if record["status"] != "success":
            continue
        if not module.startswith("v2d.") or module.endswith(".download_weights"):
            continue
        if module in VISUALIZATION_MODULES:
            previews.append({
                "module": module,
                "image": record["image"],
                "wall_seconds": record["wall_seconds"],
            })
            continue
        successful.setdefault(module, {
            "image": record["image"],
            "wall_seconds": record["wall_seconds"],
            "role": MODULE_ROLES.get(module, "reconstruction"),
        })

    stages = [{"module": module, **values} for module, values in successful.items()]
    core = sum(stage["wall_seconds"] for stage in stages if stage["role"] == "reconstruction")
    production = sum(stage["wall_seconds"] for stage in stages)
    preview = sum(stage["wall_seconds"] for stage in previews)
    role_totals = {}
    for stage in stages:
        role = stage["role"]
        role_totals[role] = round(role_totals.get(role, 0.0) + stage["wall_seconds"], 3)
    result = {
        "source_video_seconds": args.source_seconds,
        "successful_modules": stages,
        "observed_single_gpu_wall_seconds": round(core, 3),
        "core_wall_seconds_through_hand_object_alignment": round(core, 3),
        "observed_wall_per_source_second": round(core / args.source_seconds, 6),
        "production_role_wall_seconds": role_totals,
        "all_successful_production_wall_seconds": round(production, 3),
        "diagnostic_previews": previews,
        "diagnostic_preview_wall_seconds": round(preview, 3),
        "scope": "First successful timing for each production module in the log. Downloads, environment setup, failed attempts and cache hits are excluded. The core total ends at hand-object alignment. Interpolation, optional refinement and final bundle packaging are separate production roles. Every successful diagnostic visualization invocation is summed separately and excluded from production costs.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
