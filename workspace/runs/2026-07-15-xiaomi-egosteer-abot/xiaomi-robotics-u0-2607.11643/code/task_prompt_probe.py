#!/usr/bin/env python3
"""Probe: extract structured prompt templates from the released inference configs.

The paper claims a structured control formulation that disentangles workspace,
background, foreground irrelevant objects, target objects, and lighting.  This
script loads the public task configs and prints the system prompts and example
templates to verify that claim against the released code.

Usage:
    python code/task_prompt_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "code" / "xiaomi-robotics-u0"
sys.path.insert(0, str(ROOT))

from configs.tasks import scene_gen, transfer  # type: ignore


def main() -> int:
    print("=" * 60)
    print("Embodied Transfer system prompt")
    print("=" * 60)
    print(transfer.TRANSFER_SYSTEM_PROMPT)
    print()

    print("=" * 60)
    print("Embodied Scene Generation system prompt")
    print("=" * 60)
    print(scene_gen.SCENE_SYSTEM_PROMPT)
    print()

    print("=" * 60)
    print("Example scene-generation prompt (first example)")
    print("=" * 60)
    print(scene_gen.EXAMPLES[0]["text_prompt"])
    print()

    print("=" * 60)
    print("Example transfer prompt (first example)")
    print("=" * 60)
    print(transfer.EXAMPLES[0]["text_prompt"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
