#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser(
        description="Extract stage status and wall time without host, user, or absolute-path metadata."
    )
    parser.add_argument("run_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text())
    stages = []
    for name, stage in manifest.get("stages", {}).items():
        attempts = []
        for attempt in stage.get("attempts", []):
            started = attempt.get("started_at")
            finished = attempt.get("finished_at")
            duration = None
            if started and finished:
                duration = (parse_time(finished) - parse_time(started)).total_seconds()
            attempts.append({
                "number": attempt.get("number"),
                "status": attempt.get("status"),
                "exit_code": attempt.get("exit_code"),
                "duration_seconds": duration,
            })
        stages.append({
            "name": name,
            "status": stage.get("status"),
            "attempts": attempts,
        })
    result = {"stages": stages}
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
