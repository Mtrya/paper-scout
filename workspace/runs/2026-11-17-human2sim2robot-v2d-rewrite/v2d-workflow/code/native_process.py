#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKROOT = Path(os.environ.get("V2D_WORKROOT", Path(__file__).resolve().parents[1]))
RECONSTRUCTION = WORKROOT / "video_to_data/reconstruction"


def main(argv):
    if not argv:
        return 2
    if argv[0] == "version":
        print("V2D native Python process adapter; no container runtime")
        return 0
    if argv[0] == "build":
        print("Native environment is already installed; no image build is needed")
        return 0
    if argv[:2] == ["image", "inspect"]:
        print("Native Python environment")
        return 0
    if argv[0] != "run":
        raise RuntimeError("native adapter supports only run, build, version, and image inspect")

    args = list(argv[1:])
    environment = []
    volumes = []
    workdir = None
    entrypoint = None
    valued = {
        "-e", "--env", "-v", "--volume", "-w", "--workdir", "--entrypoint",
        "--user", "--gpus", "--runtime", "--network", "--shm-size", "--name",
        "--group-add",
    }
    while args:
        item = args[0]
        if item in ("--rm", "-i", "-t", "-it", "-ti", "--interactive", "--tty"):
            args.pop(0)
        elif item in valued:
            if len(args) < 2:
                raise RuntimeError("missing value for " + item)
            value = args[1]
            args = args[2:]
            if item in ("-e", "--env"):
                environment.append(value)
            elif item in ("-v", "--volume"):
                volumes.append(value)
            elif item in ("-w", "--workdir"):
                workdir = value
            elif item == "--entrypoint":
                entrypoint = value
        elif item.startswith("--env="):
            environment.append(item.split("=", 1)[1])
            args.pop(0)
        elif item.startswith("--volume="):
            volumes.append(item.split("=", 1)[1])
            args.pop(0)
        elif item.startswith("--workdir="):
            workdir = item.split("=", 1)[1]
            args.pop(0)
        elif item.startswith("--entrypoint="):
            entrypoint = item.split("=", 1)[1]
            args.pop(0)
        elif any(item.startswith(option + "=") for option in valued):
            args.pop(0)
        elif item.startswith("-"):
            raise RuntimeError("unsupported native run option: " + item)
        else:
            break
    if not args:
        raise RuntimeError("native run requires an image name and a command")
    image = args[0]
    args = args[1:]
    if entrypoint:
        args.insert(0, entrypoint)
    if not args:
        raise RuntimeError("native run requires a command")

    mounts = []
    for volume in volumes:
        parts = volume.split(":")
        if len(parts) >= 2 and parts[1].startswith("/"):
            mounts.append((parts[1].rstrip("/"), str(Path(parts[0]).resolve())))

    def host_path(value):
        if not isinstance(value, str):
            return value
        for destination, source in sorted(mounts, key=lambda pair: -len(pair[0])):
            if value == destination or value.startswith(destination + "/"):
                return source + value[len(destination):]
        return value

    child_env = os.environ.copy()
    for item in environment:
        if "=" in item:
            key, value = item.split("=", 1)
            child_env[key] = host_path(value)
        elif item in child_env:
            child_env[item] = host_path(child_env[item])
    compatibility = []
    if image == "v2d_foundation_pose":
        child_env.pop("LD_LIBRARY_PATH", None)
    if image == "v2d_sam3d":
        compatibility.extend([
            str(WORKROOT / "third_party/utils3d-sam3d"),
            str(WORKROOT / "third_party/MoGe-sam3d"),
        ])
    if child_env.get("PYTHONPATH"):
        compatibility.append(child_env["PYTHONPATH"])
    if compatibility:
        child_env["PYTHONPATH"] = os.pathsep.join(compatibility)
    if child_env.get("PYTHONPATH"):
        child_env["PYTHONPATH"] = os.pathsep.join(
            host_path(item) for item in child_env["PYTHONPATH"].split(os.pathsep)
        )
    child_env["V2D_NATIVE"] = "1"
    child_env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if args[0] in ("python", "python3", "/usr/local/bin/python", "/usr/bin/python3"):
        args[0] = os.environ.get("V2D_NATIVE_PYTHON", sys.executable)
    args = [host_path(item) for item in args]
    selected_workdir = host_path(workdir) if workdir else str(RECONSTRUCTION)
    module = args[args.index("-m") + 1] if "-m" in args else args[0]
    started = time.monotonic()
    returncode = None
    try:
        returncode = subprocess.run(args, cwd=selected_workdir, env=child_env, check=True).returncode
        return returncode
    finally:
        seconds = time.monotonic() - started
        record = {
            "image": image,
            "module": module,
            "status": "success" if returncode == 0 else "failed",
            "returncode": returncode,
            "wall_seconds": round(seconds, 3),
        }
        timing_log = Path(os.environ.get("V2D_NATIVE_TIMING_LOG", WORKROOT / "logs/native-process.jsonl"))
        timing_log.parent.mkdir(parents=True, exist_ok=True)
        with timing_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"native module {module}: {record['status']}, {seconds:.3f}s")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print("native process adapter: " + str(exc), file=sys.stderr)
        raise SystemExit(getattr(exc, "returncode", 1))
