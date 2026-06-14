#!/usr/bin/env python3
"""
Minimal probe of EurekAgent's secure-evaluation mechanism.

Runs the official grader server (src/eval_grader/server.py) against the
official circle-packing evaluator, submits a candidate generated from the
provided initial.py, and verifies that:

1. The grader server writes controller-owned result files
   (best_result.jsonl, intermediate_results.jsonl, eval_feedback/latest_feedback.json).
2. The temporary submission file is removed after grading.
3. The server rejects an invalid submission (missing description).

This demonstrates the "permissions engineering" boundary: agents can submit
candidates and read feedback, but cannot touch the grader or the authoritative
result files.

Usage (from the repo root):
    cd code/eurekagent-2606.13662
    ../../runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/probe_venv/bin/python \
        ../../runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/code/secure_grader_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = WORKSPACE_ROOT / "code" / "eurekagent-2606.13662"
SRC_ROOT = REPO_ROOT / "src"
WORKSPACE = Path(__file__).resolve().parent.parent / "probe_workspace"
HIDDEN_EVAL = REPO_ROOT / "examples" / "circle_packing" / "hidden_eval_dir"
TOKEN = "probe-token-12345"
PORT = 19876
GRADER_URL = f"http://127.0.0.1:{PORT}"


def ensure_workspace() -> None:
    """Create a fresh probe workspace."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "approach_details" / "test" / "submissions").mkdir(parents=True)


def generate_candidate() -> dict[str, Any]:
    """Generate a candidate using the official initial.py."""
    sys.path.insert(0, str(REPO_ROOT / "examples" / "circle_packing"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "initial", REPO_ROOT / "examples" / "circle_packing" / "initial.py"
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        centers, radii, sum_radii = module.run_packing()
    finally:
        sys.path.pop(0)

    return {
        "description": (
            "Constructor-based 26-circle packing with proportional radius scaling "
            "used as a baseline probe of the secure grader."
        ),
        "centers": centers.tolist(),
        "radii": radii.tolist(),
        "sum_radii": float(sum_radii),
    }


def wait_for_health(timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{GRADER_URL}/healthz", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def start_grader() -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.eval_grader.server",
            "--workspace-root",
            str(WORKSPACE),
            "--hidden-eval-dir",
            str(HIDDEN_EVAL),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--token",
            TOKEN,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not wait_for_health():
        proc.send_signal(signal.SIGTERM)
        raise RuntimeError("grader server did not become healthy")
    return proc


def submit(candidate_path: Path) -> dict[str, Any]:
    """Submit a candidate via the official client."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    env["EUREKA_SECURE_SUBMIT_URL"] = GRADER_URL
    env["EUREKA_SECURE_SUBMIT_TOKEN"] = TOKEN
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval_grader.client",
            "--approach-dir",
            str(WORKSPACE / "approach_details" / "test"),
            "--submission",
            str(candidate_path),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"submit failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def inspect_artifacts() -> dict[str, Any]:
    approach_dir = WORKSPACE / "approach_details" / "test"
    files = {
        "best_result": approach_dir / "best_result.jsonl",
        "intermediate": approach_dir / "intermediate_results.jsonl",
        "feedback": approach_dir / "eval_feedback" / "latest_feedback.json",
    }
    out: dict[str, Any] = {}
    for name, path in files.items():
        out[name] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
            "content": json.loads(path.read_text(encoding="utf-8")) if path.exists() else None,
        }
    return out


def main() -> None:
    print("=" * 70)
    print("EurekAgent secure-grader probe")
    print("=" * 70)

    ensure_workspace()
    candidate = generate_candidate()
    candidate_path = WORKSPACE / "approach_details" / "test" / "submissions" / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    print(f"\n[1] Generated candidate with sum_radii={candidate['sum_radii']:.6f}")

    print("[2] Starting official grader server ...")
    grader = start_grader()

    try:
        print("[3] Submitting valid candidate ...")
        response = submit(candidate_path)
        print(json.dumps(response, indent=2))

        if not response.get("valid"):
            raise RuntimeError("valid submission was rejected")

        print("\n[4] Inspecting controller-written result files ...")
        artifacts = inspect_artifacts()
        for name, info in artifacts.items():
            print(f"  {name}: exists={info['exists']}, size={info['size']} bytes")
            if info["exists"]:
                print(f"    score={info['content'].get('score')}, valid={info['content'].get('valid')}")

        print(f"\n[5] Checking temporary submission file was removed: {not candidate_path.exists()}")

        print("\n[6] Submitting invalid candidate (missing description) ...")
        bad_path = WORKSPACE / "approach_details" / "test" / "submissions" / "bad.json"
        bad_path.write_text(json.dumps({"centers": [], "radii": []}), encoding="utf-8")
        try:
            submit(bad_path)
            print("  ERROR: invalid submission was accepted")
        except RuntimeError as exc:
            print(f"  Correctly rejected: {exc}")

    finally:
        print("\n[7] Stopping grader server ...")
        grader.send_signal(signal.SIGTERM)
        try:
            grader.wait(timeout=5)
        except subprocess.TimeoutExpired:
            grader.kill()

    print("\nProbe complete. Artifacts preserved under:")
    print(f"  {WORKSPACE}")


if __name__ == "__main__":
    main()
