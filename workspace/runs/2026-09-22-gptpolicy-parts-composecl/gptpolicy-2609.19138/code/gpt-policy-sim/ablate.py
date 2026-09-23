"""Ablation driver: 3 tasks x 4 contexts x 3 seeds of real Codex episodes.

Resumable: finished cells are appended to results/cells.jsonl and skipped on
rerun. Three parallel workers; per-cell stdout/stderr lands in results/logs/.
Ground truth comes from each run's sim_evaluation event, never from exit codes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = RESULTS / "logs"
CELLS = RESULTS / "cells.jsonl"
TIMEOUT_S = 1800  # 60 decisions x ~15 s typical; generous headroom
WORKERS = 3

GRID = [
    (task, context, seed)
    for task in ("T1", "T2", "T3")
    for context in ("none", "target", "demo-video", "demo-action")
    for seed in (0, 1, 2)
]

_lock = threading.Lock()


def done_cells() -> set[tuple[str, str, int]]:
    if not CELLS.is_file():
        return set()
    cells = set()
    for line in CELLS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cells.add((row["task"], row["context"], row["seed"]))
    return cells


def find_run_dir(task: str, context: str, seed: int, before: set[Path]) -> str | None:
    prefix = f"{task}_{context}_s{seed}_"
    candidates = [p for p in (HERE / "runs").glob(f"{prefix}*") if p not in before]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime).relative_to(HERE))


def cell_record(run_dir: str | None) -> dict:
    """Extract ground truth and stats from a finished (or killed) run dir."""
    record = {"success": None, "decisions": None, "elapsed_s": None, "status": None, "error": None}
    if run_dir is None:
        return record
    root = HERE / run_dir
    events_file = root / "events.jsonl"
    if events_file.is_file():
        events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
        decisions = [e for e in events if e["event"] == "model_decision"]
        record["decisions"] = len(decisions)
        for event in events:
            if event["event"] == "sim_evaluation":
                record["success"] = bool(event.get("success"))
                record["detail"] = event.get("detail")
        terminal = next((e for e in events if e["event"] == "terminal"), None)
        if terminal:
            record["terminal"] = terminal.get("name")
    status_file = root / "status.json"
    if status_file.is_file():
        status = json.loads(status_file.read_text())
        record["status"] = status.get("state")
        record["elapsed_s"] = status.get("elapsed_s")
        record["error"] = status.get("error")
    return record


def run_cell(task: str, context: str, seed: int) -> dict:
    log_path = LOGS / f"{task}_{context}_s{seed}.log"
    before = set((HERE / "runs").glob(f"{task}_{context}_s{seed}_*"))
    started = time.time()
    command = [
        str(HERE / ".venv/bin/python"), "run_episode.py",
        "--task", task, "--context", context, "--seed", str(seed),
        "--max-decisions", "60",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# cell {task} {context} seed={seed} started {time.strftime('%H:%M:%S')}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, 9)
            process.wait(timeout=10)
        log.write(f"\n# cell finished rc={process.returncode} timeout={timed_out} "
                  f"wall={time.time() - started:.0f}s\n")
    time.sleep(1.0)  # let the recorder's directory rename settle
    run_dir = find_run_dir(task, context, seed, before)
    record = {
        "task": task, "context": context, "seed": seed,
        "run_dir": run_dir, "timeout": timed_out,
        "wall_s": round(time.time() - started, 1),
        "returncode": process.returncode,
        **cell_record(run_dir),
    }
    if timed_out and record["success"] is not True:
        record["success"] = False
        record["status"] = "timeout"
    with _lock:
        with CELLS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    finished = done_cells()
    todo = [cell for cell in GRID if cell not in finished]
    print(f"[ablate] {len(finished)} cells done, {len(todo)} to run", flush=True)
    if "--dry-run" in sys.argv:
        for cell in todo:
            print("  todo:", cell)
        return
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for record in pool.map(lambda cell: run_cell(*cell), todo):
            print(f"[ablate] {record['task']} {record['context']} s{record['seed']}: "
                  f"success={record['success']} decisions={record['decisions']} "
                  f"wall={record['wall_s']:.0f}s timeout={record['timeout']} "
                  f"({time.time() - started:.0f}s elapsed)", flush=True)
    print(f"[ablate] all cells finished in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
