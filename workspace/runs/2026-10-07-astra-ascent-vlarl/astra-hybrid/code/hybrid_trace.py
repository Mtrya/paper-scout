"""Print one hybrid episode as a compact step table (for reports and audits).

Usage: .venv/bin/python hybrid_trace.py runs/T1_hybrid-biased_s0_20261007-..._failed [--notes]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def waypoint_summary(arguments: dict) -> str:
    try:
        if isinstance(arguments.get("target"), dict):
            xyz = arguments["target"]["pose_xyzquat"][:3]
            return "(" + ",".join(f"{v:.3f}" for v in xyz) + ")"
        poses = arguments.get("poses") or []
        if poses:
            first = poses[0]["pose_xyzquat"][:3]
            last = poses[-1]["pose_xyzquat"][:3]
            text = "(" + ",".join(f"{v:.3f}" for v in first) + ")"
            if len(poses) > 1:
                text += f"->({','.join(f'{v:.3f}' for v in last)})"
            return f"{text} x{len(poses)}"
        if "gripper" in arguments:
            return f"gripper={arguments['gripper']}"
    except (KeyError, TypeError, IndexError):
        pass
    return "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--notes", action="store_true", help="print model notes for REPLACE steps")
    args = parser.parse_args()
    root = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines() if line.strip()]
    evaluation = next((e for e in events if e["event"] == "sim_evaluation"), {})
    print(f"# {root.name}")
    print(f"success={evaluation.get('success')} detail={evaluation.get('detail')}")
    print(f"{'step':>4} {'stage':>16} {'choice':>8}  {'proposal':<16}{'waypoint':<26} {'model':<16}waypoint")
    for event in events:
        if event["event"] != "hybrid_step":
            continue
        proposal, model = event["proposal"], event["model_action"]
        print(f"{event['step']:>4} {str(event.get('stage') or '-'):>16} {event['choice']:>8}  "
              f"{proposal['name']:<16}{waypoint_summary(proposal['arguments']):<26} "
              f"{model['name']:<16}{waypoint_summary(model['arguments'])}")
        if args.notes and event["choice"] == "replace":
            note = (model["arguments"] or {}).get("note") or (model["arguments"] or {}).get("summary") or ""
            if note:
                print(f"     note: {note}")
    terminal = next((e for e in events if e["event"] == "terminal"), None)
    if terminal:
        print(f"terminal: {terminal['name']} at step {terminal['step']} :: "
              f"{json.dumps(terminal['arguments'], ensure_ascii=False)[:200]}")
    if any(e["event"] == "budget_exhausted" for e in events):
        print("budget_exhausted: yes")


if __name__ == "__main__":
    main()
