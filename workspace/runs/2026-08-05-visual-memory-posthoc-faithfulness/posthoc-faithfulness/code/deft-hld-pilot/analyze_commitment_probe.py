#!/usr/bin/env python3
"""Summarize and visualize the DEFT free-text commitment intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit("no records")

    for record in records:
        record["choices"] = {
            condition: payload.get("choice")
            for condition, payload in record["turn2"].items()
        }

    valid = records
    n = len(valid)

    changed_withheld = sum(
        record["choices"]["withheld"] != record["choices"]["original"] for record in valid
    )
    changed_counterfactual = sum(
        record["choices"]["counterfactual"] != record["choices"]["original"] for record in valid
    )
    followed_counterfactual = sum(
        record["choices"]["counterfactual"] == record["counterfactual_target"] for record in valid
    )
    newly_followed_counterfactual = sum(
        record["choices"]["counterfactual"] == record["counterfactual_target"]
        and record["choices"]["original"] != record["counterfactual_target"]
        for record in valid
    )
    withheld_choice_rate = sum(
        record["choices"]["withheld"] is not None for record in valid
    )
    original_accuracy = sum(
        record["choices"]["original"] == record["oracle"] for record in valid
    )
    withheld_accuracy = sum(
        record["choices"]["withheld"] == record["oracle"] for record in valid
    )
    counterfactual_accuracy = sum(
        record["choices"]["counterfactual"] == record["oracle"] for record in valid
    )

    summary = {
        "n": n,
        "changed_withheld": changed_withheld,
        "changed_counterfactual": changed_counterfactual,
        "followed_counterfactual": followed_counterfactual,
        "newly_followed_counterfactual": newly_followed_counterfactual,
        "withheld_choice_rate": withheld_choice_rate,
        "original_accuracy": original_accuracy,
        "withheld_accuracy": withheld_accuracy,
        "counterfactual_accuracy": counterfactual_accuracy,
        "records": [
            {
                "row_idx": record["row_idx"],
                "oracle": record["oracle"],
                "counterfactual_target": record["counterfactual_target"],
                "choices": record["choices"],
                "original_hld": record["original_hld"],
                "counterfactual_hld": record["counterfactual_hld"],
            }
            for record in valid
        ],
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.1, 2.6))
    ax_top = fig.add_subplot(grid[0])
    metrics = [
        ("HLD removed\nchoice changed", changed_withheld),
        ("HLD reversed\nchoice changed", changed_counterfactual),
        ("Reversed target\nnewly selected", newly_followed_counterfactual),
    ]
    ax_top.bar(
        range(3),
        [value / n for _, value in metrics],
        color=["#8aa4c6", "#e58b74", "#bd5d55"],
        width=0.62,
    )
    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("Fraction of 6 scenes")
    ax_top.set_xticks(range(3), [name for name, _ in metrics])
    for idx, (_, value) in enumerate(metrics):
        ax_top.text(idx, value / n + 0.04, f"{value}/{n}", ha="center", weight="bold")
    ax_top.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1])
    columns = ["Oracle", "Original", "HLD withheld", "HLD reversed", "Reverse target"]
    ax.set_xlim(0, len(columns))
    ax.set_ylim(0, n + 1.55)
    ax.invert_yaxis()
    ax.axis("off")
    for col, name in enumerate(columns):
        ax.text(col + 0.5, 0.5, name, ha="center", va="center", weight="bold")
    for row_no, record in enumerate(valid, start=1):
        values = [
            record["oracle"],
            record["choices"]["original"],
            record["choices"]["withheld"],
            record["choices"]["counterfactual"],
            record["counterfactual_target"],
        ]
        for col, value in enumerate(values):
            if value is None:
                face = "#d6d9dd"
            elif col in (0, 4):
                face = "#edf1f5"
            elif value == record["oracle"]:
                face = "#b9dfc5"
            elif value == record["counterfactual_target"]:
                face = "#f0b4aa"
            else:
                face = "#f3dfad"
            ax.add_patch(Rectangle((col, row_no), 1, 1, facecolor=face, edgecolor="white"))
            ax.text(col + 0.5, row_no + 0.5, value or "∅", ha="center", va="center", fontsize=12, weight="bold")
        ax.text(-0.08, row_no + 0.5, f"row {record['row_idx']}", ha="right", va="center", fontsize=9)
    ax.text(
        0,
        n + 1.32,
        "green = oracle; red = injected reverse target; amber = another option; ∅ = no final choice",
        fontsize=9,
        color="#4f5964",
    )
    fig.suptitle("Does Turn-2 trajectory choice causally use the Turn-1 commitment?", fontsize=15, weight="bold")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
