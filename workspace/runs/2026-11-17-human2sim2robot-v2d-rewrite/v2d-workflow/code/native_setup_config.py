#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["V2D_WORKROOT"]) / "video_to_data"))

from tools.e2e.config import resolve_config, update_stage_config
from tools.e2e.runner import Runner, Stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--isaac-groot-dir", type=Path, required=True)
    parser.add_argument("--sequence-id", default="tissue_box_simple")
    args = parser.parse_args()
    config = resolve_config(args.config)
    inputs = {
        "mano_dir": args.mano_dir,
        "isaac_groot_dir": args.isaac_groot_dir,
    }
    parameters = {
        "mano_dir": args.mano_dir,
        "isaac_groot_dir": args.isaac_groot_dir,
        "image_version": "native",
        "gpu": 0,
        "skip_reconstruction": False,
        "accept_nvidia_model_eula": True,
    }
    workflow = None
    if config.workflow.get("sequence_id") != args.sequence_id:
        workflow = {"sequence_id": args.sequence_id}
    config = update_stage_config(
        config, "setup", parameters, inputs=inputs, workflow=workflow,
    )
    Runner(config).run(Stage(
        "setup.native-python",
        (),
        inputs=tuple(inputs.values()),
        description="native reconstruction environment and downloaded real weights",
    ))


if __name__ == "__main__":
    main()
