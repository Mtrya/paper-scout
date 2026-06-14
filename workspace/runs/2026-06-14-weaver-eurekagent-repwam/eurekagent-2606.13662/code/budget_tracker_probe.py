#!/usr/bin/env python3
"""
Minimal probe of EurekAgent's budget-engineering layer.

Uses the official TokenTracker (src/token_tracker.py) to show how the engine:

1. Aggregates token usage across multiple sessions and token types
   (input, output, cache_read, cache_creation).
2. Computes cost from per-million-token prices.
3. Flags when a configured cost limit is exceeded.

This is a pure-Python reconstruction of the accounting logic that the
SessionManager uses to kill sessions when the run budget is exhausted.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "code" / "eurekagent-2606.13662" / "src"
sys.path.insert(0, str(SRC_ROOT))

from token_tracker import TokenTracker, cost_stats, token_usage_dict


def main() -> None:
    # GLM-5.1-ish illustrative pricing (not official; used to demonstrate math).
    tracker = TokenTracker(
        _input_price=2.0,
        _cache_creation_price=0.5,
        _cache_read_price=0.2,
        _output_price=6.0,
    )

    # Simulate three parallel implement sessions plus a propose session.
    sessions = {
        "propose": {
            "input_tokens": 120_000,
            "output_tokens": 30_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 50_000,
        },
        "implement_1": {
            "input_tokens": 200_000,
            "output_tokens": 60_000,
            "cache_read_input_tokens": 80_000,
            "cache_creation_input_tokens": 0,
        },
        "implement_2": {
            "input_tokens": 180_000,
            "output_tokens": 45_000,
            "cache_read_input_tokens": 60_000,
            "cache_creation_input_tokens": 0,
        },
        "implement_3": {
            "input_tokens": 220_000,
            "output_tokens": 70_000,
            "cache_read_input_tokens": 100_000,
            "cache_creation_input_tokens": 0,
        },
    }

    for name, usage in sessions.items():
        tracker.update_session(name, usage, message_id=f"msg_{name}")

    totals = token_usage_dict(tracker)
    cost = cost_stats(tracker, currency="USD")

    print("=" * 60)
    print("EurekAgent budget tracker probe")
    print("=" * 60)
    print("\nPer-session usage:")
    for name in sessions:
        u = tracker.session_usage(name)
        print(f"  {name}: in={u.input_tokens:,}, out={u.output_tokens:,}, "
              f"cache_r={u.cache_read_input_tokens:,}, cache_c={u.cache_creation_input_tokens:,}")
    print("\nAggregated totals:")
    for k, v in totals.items():
        print(f"  {k}: {v:,}")
    print("\nComputed cost:")
    for k, v in cost.items():
        if isinstance(v, float):
            print(f"  {k}: ${v:.4f}")
        else:
            print(f"  {k}: {v}")

    cost_limit = 10.0
    current = cost["total_cost"]
    print(f"\nCost limit = ${cost_limit:.2f}; current = ${current:.4f}")
    print(f"Would kill sessions (cost_exceeded): {current >= cost_limit}")

    print("\nProbe complete.")


if __name__ == "__main__":
    main()
