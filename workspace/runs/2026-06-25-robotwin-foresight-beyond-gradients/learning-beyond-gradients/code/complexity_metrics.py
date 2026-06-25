#!/usr/bin/env python3
"""Compute lightweight complexity metrics for the main heuristic policies."""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path("/home/betelgeuse/Documents/paper-scout/workspace/code/learning-beyond-gradients")
OUT = Path(__file__).resolve().parent / "complexity_metrics.json"

POLICIES = [
    ("Breakout", REPO / "atari/breakout/heuristic_breakout.py"),
    ("Ant", REPO / "mujoco/ant/heuristic_ant.py"),
    ("HalfCheetah", REPO / "mujoco/halfcheetah/heuristic_halfcheetah_v5.py"),
    ("VizDoom D3", REPO / "vizdoom/heuristic_vizdoom_d3_cv.py"),
    ("Montezuma 400", REPO / "atari/montezuma/heuristic_montezuma_400_policy.py"),
]


def count_lines(path: Path) -> dict[str, int]:
    total = 0
    code = 0
    blank = 0
    comment = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            stripped = line.strip()
            if not stripped:
                blank += 1
            elif stripped.startswith("#"):
                comment += 1
            else:
                code += 1
    return {"total": total, "code": code, "blank": blank, "comment": comment}


def ast_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return {"functions": 0, "classes": 0, "error": str(e)}
    funcs = len([n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
    classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
    dataclasses = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
    return {"functions": funcs, "classes": classes}


def rough_config_fields(path: Path) -> int:
    """Count fields in dataclass-style config classes by looking for class-level assignments."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # count AnnAssign/Assign at class level that are not methods
            for item in node.body:
                if isinstance(item, (ast.AnnAssign, ast.Assign)):
                    total += 1
    return total


def main() -> None:
    results = []
    for name, path in POLICIES:
        lines = count_lines(path)
        asts = ast_counts(path)
        configs = rough_config_fields(path)
        results.append(
            {
                "policy": name,
                "path": str(path.relative_to(REPO.parent)),
                "total_lines": lines["total"],
                "code_lines": lines["code"],
                "functions": asts["functions"],
                "classes": asts["classes"],
                "config_fields": configs,
            }
        )
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("| Policy | total lines | code lines | functions | classes | config fields |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        print(
            f"| {r['policy']} | {r['total_lines']} | {r['code_lines']} | {r['functions']} | {r['classes']} | {r['config_fields']} |"
        )


if __name__ == "__main__":
    main()
