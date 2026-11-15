#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def unit(values):
    norm = math.sqrt(dot(values, values))
    return tuple(value / norm for value in values)


def support(directions, wrenches):
    return [max(dot(direction, wrench) for wrench in wrenches) for direction in directions]


def main():
    rng = random.Random(7)
    directions = [unit(tuple(rng.gauss(0, 1) for _ in range(6))) for _ in range(512)]
    contacts = [((0.10, 0.0, 0.0), 0.5), ((-0.10, 0.0, 0.0), 0.5)]
    wrenches = []
    for position, friction in contacts:
        for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
            force = unit((friction * math.cos(angle), friction * math.sin(angle), 1.0))
            wrenches.append(force + cross(position, force))

    reference = support(directions, wrenches)
    permuted = support(directions, list(reversed(wrenches)))
    permutation_delta = max(abs(a - b) for a, b in zip(reference, permuted))
    shifted = list(wrenches)
    force = shifted[0][:3]
    shifted[0] = force + cross((0.20, 0.0, 0.0), force)
    shifted_support = support(directions, shifted)
    normalized = [wrench[:3] + tuple(value / 0.15 for value in wrench[3:]) for wrench in wrenches]
    normalized_support = support(directions, normalized)
    moment_arm_delta = max(abs(a - b) for a, b in zip(reference, shifted_support))
    radius_delta = max(abs(a - b) for a, b in zip(reference, normalized_support))

    assert permutation_delta == 0
    assert moment_arm_delta > 0
    assert radius_delta > 0
    result = {
        "direction_samples": len(directions),
        "contacts": len(contacts),
        "primitive_wrenches": len(wrenches),
        "support_permutation_max_abs_delta": permutation_delta,
        "moment_arm_change_max_abs_support_delta": moment_arm_delta,
        "radius_normalization_max_abs_support_delta": radius_delta,
        "support_range": [min(reference), max(reference)],
        "interpretation": "The support representation is invariant to contact ordering while retaining sensitivity to moment arms and their radius normalization.",
    }
    output = Path(__file__).with_name("wrench_support_probe.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
