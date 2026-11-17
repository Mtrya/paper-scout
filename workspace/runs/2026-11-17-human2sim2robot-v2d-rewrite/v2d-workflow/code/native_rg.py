#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

WORKROOT = Path(os.environ.get("V2D_WORKROOT", Path(__file__).resolve().parents[3]))
REPOSITORY = WORKROOT / "video_to_data"


def main():
    raw = sys.argv[1:]
    if "--" in raw:
        split = raw.index("--")
        front, command = raw[:split], raw[split + 1:]
    else:
        front, command = raw, []
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("version", nargs="?", default="native")
    parser.add_argument("gpu", nargs="?", default="0")
    parser.add_argument("--run-data", type=Path)
    parser.add_argument("--mano-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workdir", default="/workspace/video_to_data/robotic_grounding")
    parser.add_argument("--recreate-on-mount-change", action="store_true")
    args = parser.parse_args(front)
    if args.action == "build":
        print("Using the native Isaac Lab environment; no image build is needed")
        return 0
    if args.action != "e2e-run" or not args.run_data:
        raise RuntimeError("native Isaac runner supports build and e2e-run")

    if not command:
        raise RuntimeError("e2e-run requires a command")
    mounts = {
        "/workspace/video_to_data": str(REPOSITORY),
        "/workspace/e2e": str(args.run_data.resolve()),
    }
    if args.mano_dir:
        mounts["/workspace/mano"] = str(args.mano_dir.resolve())
    if args.checkpoint:
        mounts["/workspace/checkpoints/" + args.checkpoint.name] = str(args.checkpoint.resolve())

    def host_path(value):
        for destination, source in mounts.items():
            value = value.replace(destination, source)
        return value

    child_env = os.environ.copy()
    if command and command[0] == "env":
        command.pop(0)
    while command and "=" in command[0] and not command[0].startswith("/"):
        key, value = command.pop(0).split("=", 1)
        child_env[key] = host_path(value)
    if not command:
        return 0
    if command[0] in ("python", "python3"):
        python = os.environ.get("V2D_ISAAC_PYTHON")
        if python:
            command[0] = python
        elif os.environ.get("ISAACLAB_PATH"):
            command[:1] = [str(Path(os.environ["ISAACLAB_PATH"]) / "isaaclab.sh"), "-p"]
    command = [host_path(item) for item in command]
    workdir = Path(host_path(args.workdir))
    child_env["V2D_NATIVE"] = "1"
    python_path = [str(REPOSITORY / "robotic_grounding")]
    python_path.extend(
        host_path(item)
        for item in child_env.get("PYTHONPATH", "").split(os.pathsep)
        if item
    )
    child_env["PYTHONPATH"] = os.pathsep.join(python_path)
    subprocess.run(command, cwd=workdir, env=child_env, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print("native Isaac runner: " + str(exc), file=sys.stderr)
        raise SystemExit(getattr(exc, "returncode", 1))
