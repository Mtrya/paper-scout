#!/usr/bin/env python3
"""Aggregate the phase-localized post-hoc steering experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ORDER = (
    "prefill",
    "prefill_all",
    "early_reasoning_4",
    "reasoning",
    "answer",
    "all_decode",
    "orthogonal",
)
LABELS = {
    "prefill": "Prefill-last",
    "prefill_all": "Prefill-all",
    "early_reasoning_4": "First 4 reasoning states",
    "reasoning": "Full reasoning",
    "answer": "Answer only",
    "all_decode": "All decoding",
    "orthogonal": "Orthogonal control",
}
COLORS = {
    "prefill": "#2f6690",
    "prefill_all": "#4d908e",
    "early_reasoning_4": "#f2a541",
    "reasoning": "#d97706",
    "answer": "#7c3aed",
    "all_decode": "#b91c1c",
    "orthogonal": "#7b8794",
}


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def exact_sign_p(left_only: int, right_only: int) -> float:
    """Two-sided exact sign test for paired binary outcomes."""
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    extreme = min(left_only, right_only)
    tail = sum(math.comb(discordant, k) for k in range(extreme + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("interventions", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    probe = json.loads(args.summary.read_text())
    records = [json.loads(line) for line in args.interventions.read_text().splitlines() if line.strip()]
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["mode"], abs(float(record["alpha"])))].append(record)

    aggregate: dict[str, dict[str, dict]] = {}
    for mode in ORDER:
        aggregate[mode] = {}
        for alpha in sorted({abs(float(record["alpha"])) for record in records}):
            subset = grouped[(mode, alpha)]
            flips = sum(record["answer"] == record["target_answer"] for record in subset)
            parsed = sum(record["answer"] in {"yes", "no"} for record in subset)
            low, high = wilson(flips, len(subset))
            aggregate[mode][str(alpha)] = {
                "n": len(subset),
                "flips": flips,
                "flip_rate": flips / len(subset) if subset else None,
                "parsed": parsed,
                "parsed_rate": parsed / len(subset) if subset else None,
                "wilson95": [low, high],
            }

    minimal_phase_examples = []
    by_key = {(r["index"], r["mode"], abs(float(r["alpha"]))): r for r in records}
    for record in records:
        if record["mode"] != "reasoning" or record["answer"] != record["target_answer"]:
            continue
        alpha = abs(float(record["alpha"]))
        prefill = by_key.get((record["index"], "prefill", alpha))
        answer = by_key.get((record["index"], "answer", alpha))
        if prefill and answer and prefill["answer"] == record["baseline_answer"] and answer["answer"] == record["baseline_answer"]:
            minimal_phase_examples.append(record)

    paired = {}
    for alpha in sorted({abs(float(record["alpha"])) for record in records}):
        paired[str(alpha)] = {}
        indices = sorted({record["index"] for record in records})
        for left, right in (
            ("prefill_all", "orthogonal"),
            ("early_reasoning_4", "orthogonal"),
            ("reasoning", "orthogonal"),
            ("reasoning", "early_reasoning_4"),
            ("reasoning", "answer"),
            ("all_decode", "reasoning"),
        ):
            both = left_only = right_only = neither = 0
            for index in indices:
                left_record = by_key.get((index, left, alpha))
                right_record = by_key.get((index, right, alpha))
                if not left_record or not right_record:
                    continue
                left_flip = left_record["answer"] == left_record["target_answer"]
                right_flip = right_record["answer"] == right_record["target_answer"]
                if left_flip and right_flip:
                    both += 1
                elif left_flip:
                    left_only += 1
                elif right_flip:
                    right_only += 1
                else:
                    neither += 1
            paired[str(alpha)][f"{left}_vs_{right}"] = {
                "both": both,
                "left_only": left_only,
                "right_only": right_only,
                "neither": neither,
                "exact_two_sided_p": exact_sign_p(left_only, right_only),
            }

    result = {
        "probe": probe,
        "n_intervention_records": len(records),
        "aggregate": aggregate,
        "paired": paired,
        "reasoning_only_flip_examples": minimal_phase_examples[:12],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    ax = axes[0]
    layers = list(range(len(probe["layer_aucs"])))
    ax.plot(layers, probe["layer_aucs"], color="#235789", lw=2.2, marker="o", ms=3)
    ax.axhline(0.5, color="#9aa5b1", lw=1, ls="--", label="Chance")
    ax.axhline(
        probe["permutation_max_auc_p95"],
        color="#d97706",
        lw=1.4,
        ls=":",
        label="95th percentile of max-layer random-label AUC",
    )
    ax.scatter([probe["best_layer"]], [probe["best_auc"]], s=65, color="#b91c1c", zorder=5)
    ax.annotate(
        f"best layer {probe['best_layer']}\nAUC {probe['best_auc']:.3f}",
        (probe["best_layer"], probe["best_auc"]),
        xytext=(8, -34),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set(title="The eventual answer is decodable before written reasoning", xlabel="Decoder layer", ylabel="Held-out AUC")
    ax.set_ylim(0.42, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    focus_alpha = 4.0
    rows = [aggregate[mode][str(focus_alpha)] for mode in ORDER]
    rates = [row["flip_rate"] for row in rows]
    lows = [row["wilson95"][0] for row in rows]
    highs = [row["wilson95"][1] for row in rows]
    yerr = [[rate - low for rate, low in zip(rates, lows)], [high - rate for rate, high in zip(rates, highs)]]
    positions = list(range(len(ORDER)))
    ax.bar(
        positions,
        rates,
        yerr=yerr,
        color=[COLORS[mode] for mode in ORDER],
        capsize=3,
        width=0.72,
    )
    for position, rate, high, row in zip(positions, rates, highs, rows):
        ax.text(position, high + 0.035, f"{row['parsed']}/{row['n']} parsed", ha="center", va="bottom", fontsize=7.3)
    ax.set(
        title="Brief pre-CoT steering barely moves answers; sustained steering does",
        xlabel="Intervention scope at |α| = 4",
        ylabel="Forced answer-flip rate",
        xticks=positions,
        xticklabels=[LABELS[mode] for mode in ORDER],
    )
    ax.set_ylim(-0.03, 1.13)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Phase-localized causal test on Qwen3-VL-8B-Instruct · BIG-Bench Sports Understanding", fontsize=13.5, weight="bold")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=190, bbox_inches="tight")


if __name__ == "__main__":
    main()
