#!/usr/bin/env python3
"""
Extract representative APT code snippets from the cloned repository into a
single markdown file for the thread.  The snippets are chosen to answer:

  * How is the action expert implemented?
  * How are the two stages realized (layer expansion, language masking,
    gated fusion, FiLM)?
  * How does Stage-1 expand into Stage-2 at checkpoint load time?
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
APT_ROOT = ROOT / "code" / "APT"
OUT_DIR = Path(__file__).resolve().parent

# File -> list of (label, first_line, last_line)  (1-based, inclusive)
SNIPPETS = {
    APT_ROOT / "apt" / "action_expert.py": [
        ("prepare_attention_mask: language-vs-action masking", 18, 71),
        ("HybridAttentionLayers: layer expansion, gated fusion, FiLM", 124, 172),
        ("ActionExpert: two-stage docstring", 452, 465),
        ("ActionExpert: train_stage switches inputs/masks", 749, 774),
    ],
    APT_ROOT / "apt" / "vla.py": [
        ("VLA.parameter_groups: separate LR/WD for VLM and actor", 51, 99),
        ("VLA.load_from_pretrain: expand Stage-0 -> Stage-1", 101, 139),
    ],
}


def extract(path: Path, ranges: list[tuple[str, int, int]]) -> str:
    lines = path.read_text().splitlines()
    chunks = [f"## {path.relative_to(ROOT)}\n"]
    for label, start, end in ranges:
        chunks.append(f"### {label} (lines {start}-{end})\n")
        chunks.append("```python")
        for i in range(start - 1, min(end, len(lines))):
            chunks.append(lines[i])
        chunks.append("```\n")
    return "\n".join(chunks)


def main():
    out_path = OUT_DIR / "action_expert_snippets.md"
    parts = ["# APT Action Expert — Curated Code Snippets\n"]
    for file_path, ranges in SNIPPETS.items():
        if not file_path.exists():
            print(f"[WARN] missing {file_path}", file=sys.stderr)
            continue
        parts.append(extract(file_path, ranges))
    out_path.write_text("\n".join(parts))
    print(f"Wrote snippets to {out_path}")


if __name__ == "__main__":
    main()
