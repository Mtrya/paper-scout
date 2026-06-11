#!/usr/bin/env python3
"""Probe: inspect how ALE-Bench Session enforces resource limits and time scale.

This script verifies the resource-usage gating logic that shapes the
"long-horizon" vs "massively parallel search" debate in the paper."""

import inspect
from code.alebench.src.ale_bench.session import Session, AleBenchFunction, CHECK_RESOURCE_USAGE_FIELDS

print("=== ALE-Bench Session Resource Gating ===\n")

# Show which resource fields are checked for each action type
print("Resource fields checked per action:")
for func, fields in CHECK_RESOURCE_USAGE_FIELDS.items():
    print(f"  {func.value:20s} -> {fields}")

print()

# Show the public_eval cooldown logic (same_time_scale simulates contest submission interval)
source = inspect.getsource(Session.next_public_eval_time.fget)
print("Public-eval cooldown logic (same_time_scale):")
for line in source.splitlines():
    print("  " + line)

print()

# Show private_eval rank estimation
source = inspect.getsource(Session.estimate_rank_and_performance)
print("Rank/Performance estimation (private eval):")
for line in source.splitlines():
    print("  " + line)
