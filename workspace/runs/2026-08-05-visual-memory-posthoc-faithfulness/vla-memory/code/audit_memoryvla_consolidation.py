#!/usr/bin/env python3
"""Audit the algebra of MemoryVLA's released adjacent-token merge rule.

This is a mechanism probe, not a policy evaluation.  It mirrors the released
CogMemBank consolidation rule for a single token per time step, while tracking
the otherwise-discarded provenance of every merged slot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Slot:
    timestamp: int
    feature: np.ndarray
    weights: dict[int, float]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def consolidate(features: np.ndarray, capacity: int) -> list[Slot]:
    bank: list[Slot] = []
    for timestep, feature in enumerate(features):
        bank.append(Slot(timestep, feature.copy(), {timestep: 1.0}))
        while len(bank) > capacity:
            similarities = [cosine(bank[i].feature, bank[i + 1].feature) for i in range(len(bank) - 1)]
            index = int(np.argmax(similarities))
            left, right = bank[index], bank[index + 1]
            weights = {key: 0.5 * value for key, value in left.weights.items()}
            for key, value in right.weights.items():
                weights[key] = weights.get(key, 0.0) + 0.5 * value
            # Mirrors memory_vla.py exactly: arithmetic mean, earlier timestamp.
            bank[index] = Slot(left.timestamp, 0.5 * (left.feature + right.feature), weights)
            bank.pop(index + 1)
    return bank


def smooth_episode(rng: np.random.Generator, timesteps: int, dimension: int) -> np.ndarray:
    """A slow visual random walk with occasional task-phase jumps."""
    features = np.empty((timesteps, dimension), dtype=np.float64)
    state = rng.normal(size=dimension)
    for timestep in range(timesteps):
        if timestep and timestep % 32 == 0:
            state += 0.9 * rng.normal(size=dimension)
        state += 0.08 * rng.normal(size=dimension)
        features[timestep] = state / np.linalg.norm(state)
    return features


def slot_metrics(slot: Slot) -> dict[str, float | int]:
    indices = np.asarray(sorted(slot.weights), dtype=np.float64)
    weights = np.asarray([slot.weights[int(index)] for index in indices], dtype=np.float64)
    uniform = 1.0 / len(indices)
    centroid = float(np.dot(indices, weights))
    effective_n = float(1.0 / np.square(weights).sum())
    return {
        "timestamp": slot.timestamp,
        "start": int(indices.min()),
        "end": int(indices.max()),
        "frames": len(indices),
        "weighted_centroid": centroid,
        "timestamp_error": abs(slot.timestamp - centroid),
        "max_to_uniform_weight_ratio": float(weights.max() / uniform),
        "min_to_uniform_weight_ratio": float(weights.min() / uniform),
        "effective_frames": effective_n,
        "effective_fraction": effective_n / len(indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_metrics: list[dict[str, float | int]] = []
    worst: dict[str, object] | None = None
    for episode in range(args.episodes):
        bank = consolidate(smooth_episode(rng, args.timesteps, args.dimension), args.capacity)
        for slot_index, slot in enumerate(bank):
            metrics = slot_metrics(slot)
            metrics.update({"episode": episode, "slot": slot_index})
            all_metrics.append(metrics)
            score = float(metrics["max_to_uniform_weight_ratio"])
            if worst is None or score > float(worst["max_to_uniform_weight_ratio"]):
                worst = {**metrics, "weights": slot.weights}

    assert worst is not None
    numeric = {
        key: np.asarray([float(row[key]) for row in all_metrics])
        for key in (
            "frames",
            "timestamp_error",
            "max_to_uniform_weight_ratio",
            "min_to_uniform_weight_ratio",
            "effective_fraction",
        )
    }
    result = {
        "source": "MemoryVLA commit d732ea9072bc063399ccc817aed74ab172eb50be",
        "scope": "single-token algebraic mirror of CogMemBank._consolidate_with_token_merge",
        "configuration": vars(args) | {"output": str(args.output)},
        "slots_observed": len(all_metrics),
        "summary": {
            key: {
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
                "min": float(values.min()),
            }
            for key, values in numeric.items()
        },
        "worst_slot": worst,
        "interpretation_boundary": (
            "The unequal leaf weights and stale timestamps follow from the released merge rule; "
            "the random-walk frequency statistics do not estimate downstream policy impact."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(result["worst_slot"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
