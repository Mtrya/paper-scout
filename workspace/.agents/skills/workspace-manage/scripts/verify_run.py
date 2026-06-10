#!/usr/bin/env python3
"""Verify a Paper Scout run packet and final workspace hygiene."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FIGURE_ANCHOR_RE = re.compile(r"\[\[figure-anchor:([^\]]+)\]\]")
RUN_RESERVED = {"assets", "report.docxxml", "checklist.md", "README.md"}
THREAD_BLOCKER_SHAPE = {"BLOCKER.md"}
THREAD_EVIDENCE_SHAPES = (
    {"README.md", "code"},
    {"README.md", "patches"},
    {"README.md", "code", "patches"},
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_run_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def has_non_empty_file(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.rglob("*"):
        if child.is_file() and child.stat().st_size > 0:
            return True
    return False


def visible_entries(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.iterdir(), key=lambda item: item.name)


def verify_thread(path: Path, errors: list[str]) -> None:
    names = {entry.name for entry in visible_entries(path)}

    if names == THREAD_BLOCKER_SHAPE:
        if not (path / "BLOCKER.md").is_file():
            errors.append(f"{path}: BLOCKER.md must be a file")
        return

    if names not in THREAD_EVIDENCE_SHAPES:
        expected = "BLOCKER.md, or README.md with code/ and/or patches/"
        errors.append(f"{path}: invalid thread shape; expected {expected}")
        return

    readme = path / "README.md"
    if not readme.is_file():
        errors.append(f"{path}: README.md must be a file")

    for evidence_dir in ("code", "patches"):
        evidence_path = path / evidence_dir
        if evidence_dir in names:
            if not evidence_path.is_dir():
                errors.append(f"{evidence_path}: must be a directory")
            elif not has_non_empty_file(evidence_path):
                errors.append(f"{evidence_path}: must contain at least one non-empty file")


def verify_run(run_path: Path, mode: str) -> tuple[list[str], int, int]:
    errors: list[str] = []
    root = workspace_root()
    runs_root = root / "runs"

    if not run_path.exists():
        return [f"{run_path}: run path does not exist"], 0, 0
    if not run_path.is_dir():
        return [f"{run_path}: run path must be a directory"], 0, 0

    try:
        run_path.relative_to(runs_root.resolve())
    except ValueError:
        errors.append(f"{run_path}: run path must be under {runs_root}")

    report = run_path / "report.docxxml"
    checklist = run_path / "checklist.md"
    assets = run_path / "assets"

    if not report.is_file():
        errors.append(f"{report}: required report file is missing")
        anchor_count = 0
    else:
        anchors = {
            match.strip()
            for match in FIGURE_ANCHOR_RE.findall(report.read_text(encoding="utf-8"))
            if match.strip()
        }
        anchor_count = len(anchors)
        if anchor_count < 2:
            errors.append(f"{report}: expected at least 2 unique figure anchors, found {anchor_count}")

    if not checklist.is_file():
        errors.append(f"{checklist}: required checklist file is missing")

    if not assets.is_dir():
        errors.append(f"{assets}: required assets directory is missing")

    thread_count = 0
    for entry in visible_entries(run_path):
        if entry.name in RUN_RESERVED:
            continue
        if not entry.is_dir():
            errors.append(f"{entry}: unexpected run-level file")
            continue
        thread_count += 1
        verify_thread(entry, errors)

    if thread_count == 0:
        errors.append(f"{run_path}: expected at least one thread directory")

    if mode == "final":
        verify_final_hygiene(root, run_path, errors)

    return errors, thread_count, anchor_count


def verify_final_hygiene(root: Path, run_path: Path, errors: list[str]) -> None:
    allowed_marker = {"README.md"}
    for scratch_name in ("code", "drafts"):
        scratch = root / scratch_name
        if not scratch.is_dir():
            errors.append(f"{scratch}: required workspace directory is missing")
            continue
        leftovers = [entry for entry in visible_entries(scratch) if entry.name not in allowed_marker]
        if leftovers:
            names = ", ".join(str(item.relative_to(root)) for item in leftovers[:10])
            suffix = "" if len(leftovers) <= 10 else f" and {len(leftovers) - 10} more"
            errors.append(f"{scratch}: must contain only README.md in final mode; found {names}{suffix}")

    index = root / "runs" / "INDEX.md"
    if not index.is_file():
        errors.append(f"{index}: required index file is missing")
        return
    index_text = index.read_text(encoding="utf-8")
    if run_path.name not in index_text:
        errors.append(f"{index}: final mode expects an entry mentioning {run_path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a Paper Scout run packet.")
    parser.add_argument("run_path", help="Run packet path, for example runs/2026-06-07-cosmos3-grail-qwenvla")
    parser.add_argument("--mode", choices=("prepublish", "final"), default="prepublish")
    args = parser.parse_args(argv)

    run_path = resolve_run_path(args.run_path)
    errors, thread_count, anchor_count = verify_run(run_path, args.mode)

    if errors:
        print(f"FAIL {args.mode}: {args.run_path}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS {args.mode}: {args.run_path}")
    print(f"- threads: {thread_count}")
    print(f"- figure anchors: {anchor_count}")
    print("- assets: present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
