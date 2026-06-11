#!/usr/bin/env python3
"""Probe: inspect the ALE-Agent / scaffold evaluation loop.

Shows how repeated sampling, self-refinement, and private evaluation
are stitched together in __main__.py and scaffolds.py."""

import inspect
from code.alebench.src.ale_bench_eval.__main__ import evaluate_contest
from code.alebench.src.ale_bench_eval.scaffolds import run_repeated_sampling, run_self_refinement
from code.alebench.src.ale_bench_eval.evaluate import run_private_evaluation

print("=== ALE-Agent Scaffold Loop ===\n")

# The four phases of evaluate_contest
src = inspect.getsource(evaluate_contest)
print("evaluate_contest() phases (from __main__.py):")
for line in src.splitlines():
    if "Phase" in line or "repeated_sampling" in line or "self_refinement" in line or "private_evaluation" in line:
        print("  " + line.strip())

print()

# How repeated sampling parallelizes LLM calls
src = inspect.getsource(run_repeated_sampling)
print("run_repeated_sampling() parallelism signature:")
for line in src.splitlines()[:25]:
    print("  " + line)

print()

# How self-refinement accumulates message history
src = inspect.getsource(run_self_refinement)
print("run_self_refinement() history accumulation:")
for line in src.splitlines()[:30]:
    print("  " + line)
