"""Measure uni-robot-agent's scheduler overlap with synthetic tools.

No model or robot is involved: this isolates the harness's two concurrency
levels (background calls and per-resource queues) from model quality.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "code" / "rua1103" / "uni-robot-agent"
sys.path.insert(0, str(ROOT))

from uni_robot_agent.execution.parallel import ParallelToolExecutor
from uni_robot_agent.execution.scheduler import ToolExecutionPolicy
from uni_robot_agent.execution.serial import SerialToolExecutor
from uni_robot_agent.tools.toolkit import ToolDef, Toolkit, ToolOutput


def build():
    toolkit = Toolkit()
    toolkit.register_tools([
        ToolDef(
            name="slow_exclusive",
            spec={"name": "slow_exclusive", "description": "sleep on arm", "input_schema": {"type": "object", "properties": {}}},
            handler=lambda **_: (time.sleep(1.0), ToolOutput(extra={"done": time.monotonic()}))[1],
        ),
        ToolDef(
            name="slow_parallel",
            spec={"name": "slow_parallel", "description": "sleep on vision", "input_schema": {"type": "object", "properties": {}}},
            handler=lambda **_: (time.sleep(1.0), ToolOutput(extra={"done": time.monotonic()}))[1],
        ),
    ])
    policies = {
        "slow_exclusive": ToolExecutionPolicy(resource_key="arm", concurrency="exclusive", return_mode="background"),
        "slow_parallel": ToolExecutionPolicy(resource_key="vision", concurrency="parallel", return_mode="background"),
    }
    return toolkit, policies


def run_serial(n=4):
    toolkit, _ = build()
    ex = SerialToolExecutor(toolkit)
    t0 = time.perf_counter()
    for _ in range(n):
        ex.call("slow_exclusive", {})
    return time.perf_counter() - t0


def run_blocking(n=4, reasoning_s=0.25):
    toolkit, _ = build()
    ex = SerialToolExecutor(toolkit)
    t0 = time.perf_counter()
    for _ in range(n):
        ex.call("slow_exclusive", {})
        time.sleep(reasoning_s)
    return time.perf_counter() - t0


def run_pipelined(n=4, reasoning_s=0.25):
    toolkit, policies = build()
    ex = ParallelToolExecutor(toolkit, policies, max_parallel=4)
    t0 = time.perf_counter()
    jobs = []
    for _ in range(n):
        jobs.append(ex.submit("slow_exclusive", {}))
        time.sleep(reasoning_s)  # simulated model reasoning between submissions
    for job in jobs:
        ex.job_result(job.job_id, timeout=30)
    return time.perf_counter() - t0


def run_parallel(n=4):
    toolkit, policies = build()
    ex = ParallelToolExecutor(toolkit, policies, max_parallel=4)
    t0 = time.perf_counter()
    jobs = [ex.submit("slow_parallel", {}) for _ in range(n)]
    for job in jobs:
        ex.job_result(job.job_id, timeout=30)
    return time.perf_counter() - t0


def run_mixed(n=4):
    toolkit, policies = build()
    ex = ParallelToolExecutor(toolkit, policies, max_parallel=8)
    t0 = time.perf_counter()
    jobs = []
    for _ in range(n):
        jobs.append(ex.submit("slow_exclusive", {}))
        jobs.append(ex.submit("slow_parallel", {}))
    for job in jobs:
        ex.job_result(job.job_id, timeout=30)
    return time.perf_counter() - t0


def main():
    out = {
        "serial_4x1s": round(run_serial(), 3),
        "blocking_4x1s_reasoning0.25": round(run_blocking(), 3),
        "pipelined_4x1s_reasoning0.25": round(run_pipelined(), 3),
        "blocking_4x1s_reasoning1.5": round(run_blocking(reasoning_s=1.5), 3),
        "pipelined_4x1s_reasoning1.5": round(run_pipelined(reasoning_s=1.5), 3),
        "parallel_4x1s": round(run_parallel(), 3),
        "mixed_4exclusive_4parallel": round(run_mixed(), 3),
    }
    out["ideal_serial"] = 4.0
    out["ideal_pipelined"] = 4.0
    out["ideal_parallel"] = 1.0
    out["ideal_mixed"] = 5.0
    out["hidden_reasoning_s"] = round(out["blocking_4x1s_reasoning0.25"] - out["pipelined_4x1s_reasoning0.25"], 3)
    print(json.dumps(out, indent=2))
    dest = Path(__file__).resolve().parent / "out" / "sched.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
