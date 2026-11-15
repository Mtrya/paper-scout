#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from groot_finetune.closed_loop.evaluation import evaluate_lift_hold
from groot_finetune.task_profile import load_task_profile


def evaluate_source(name, root, profile):
    episodes = []
    for path in sorted(root.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            object_pose = np.asarray(data["object_pose"])
            if object_pose.ndim != 3 or object_pose.shape[-1] != 7:
                raise ValueError(f"{path}: unexpected object_pose shape {object_pose.shape}")
            result = evaluate_lift_hold(
                object_pose[:, 0, 2],
                config=profile.evaluator,
            )
            episodes.append({
                "episode": path.name,
                "timeout_eligible": bool(np.asarray(data["source_success"]).item()),
                "offline_task_success": result.success,
                "metrics": asdict(result),
            })
    return {
        "name": name,
        "root": str(root),
        "episode_count": len(episodes),
        "timeout_eligible_count": sum(item["timeout_eligible"] for item in episodes),
        "lift_hold_success_count": sum(
            item["metrics"]["lift_hold_success"] for item in episodes
        ),
        "offline_task_success_count": sum(
            item["offline_task_success"] for item in episodes
        ),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply the official task evaluator to pre-action object heights in exported source rollouts."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-profile", type=Path)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional export directory; defaults to the selected directory.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    profile_path = args.task_profile or run_root / "contracts/task_profile.json"
    profile = load_task_profile(profile_path)
    sources = args.source or [f"selected={run_root / 'selected'}"]
    parsed = []
    for source in sources:
        name, separator, value = source.partition("=")
        if not separator or not name or not value:
            parser.error("--source must have the form NAME=PATH")
        parsed.append((name, Path(value).resolve()))
    result = {
        "run_root": str(run_root),
        "task_profile": str(profile_path),
        "task_id": profile.task_id,
        "evaluator": profile.evaluator.as_dict(),
        "interpretation": "Offline task metrics of source expert rollouts, not closed-loop VLA evaluation. Object heights are pre-action samples from the exported trajectories.",
        "sources": [evaluate_source(name, root, profile) for name, root in parsed],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
