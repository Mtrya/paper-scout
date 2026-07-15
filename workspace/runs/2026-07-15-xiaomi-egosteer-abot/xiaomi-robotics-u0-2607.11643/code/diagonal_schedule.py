#!/usr/bin/env python3
"""Probe: visualize FlashAR anti-diagonal decoding schedule.

The paper (and the FlashAR+ extension in Xiaomi-Robotics-U0) generates image
tokens in parallel along anti-diagonals: position (r, c) belongs to step r+c.
This script prints the schedule and shows the reduction in serial steps versus
standard raster-order AR.

Usage:
    python code/diagonal_schedule.py --height 32 --width 32
"""

from __future__ import annotations

import argparse
import sys


def diagonal_schedule(height: int, width: int):
    """Return list of anti-diagonal position lists for an HxW grid."""
    steps = []
    for d in range(height + width - 1):
        positions = []
        for r in range(max(0, d - width + 1), min(height, d + 1)):
            c = d - r
            positions.append((r, c))
        steps.append(positions)
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()

    h, w = args.height, args.width
    steps = diagonal_schedule(h, w)

    print(f"Grid: {h}x{w}")
    print(f"Raster AR serial steps: {h * w}")
    print(f"FlashAR anti-diagonal serial steps: {len(steps)}")
    print(f"Theoretical speed-up: {h * w / len(steps):.2f}x")
    print()
    print("First few diagonals:")
    for i, positions in enumerate(steps[:6]):
        print(f"  step {i}: {positions}")
    print("  ...")
    print("Last diagonal:")
    print(f"  step {len(steps)-1}: {steps[-1]}")

    # Sanity check: every position appears exactly once.
    flat = [p for step in steps for p in step]
    assert len(flat) == h * w and len(set(flat)) == h * w
    return 0


if __name__ == "__main__":
    sys.exit(main())
