#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["V2D_WORKROOT"]) / "video_to_data"))

from tools.e2e.config import resolve_config, update_stage_config
from tools.e2e.runner import Runner
from tools.e2e.stages import finetune_stages


def main():
    parser = argparse.ArgumentParser(
        description="Run only the official LeRobot conversion, statistics, and audit stages."
    )
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.7-3B")
    args = parser.parse_args()
    if args.epochs <= 0 or args.global_batch_size <= 0:
        parser.error("epochs and global batch size must be positive")

    config = resolve_config(args.config)
    workflow = config.workflow
    parameters = {
        "epochs": args.epochs,
        "global_batch_size": args.global_batch_size,
        "action_horizon": int(config.data["contracts"]["embodiment"]["action_horizon"]),
        "base_model": args.base_model,
    }
    config = update_stage_config(
        config,
        "finetune",
        parameters,
        workflow=parameters,
        persist=not args.dry_run,
    )
    stages = finetune_stages(config, include_open_loop=False)[:3]
    if [stage.name for stage in stages] != [
        "finetune.convert",
        "finetune.statistics",
        "finetune.audit-dataset",
    ]:
        raise RuntimeError("upstream finetune stage contract changed; refusing to continue")
    Runner(
        config,
        dry_run=args.dry_run,
        force=args.force,
    ).run_many(stages)
    if not args.dry_run:
        audit_path = Path(config.run_root) / "dataset_audit.json"
        audit = json.loads(audit_path.read_text())
        print(json.dumps({"audit_path": str(audit_path), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
