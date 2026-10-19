#!/usr/bin/env python3
"""Two-point scaling audit for Natura-1, compared with Dyna-2's published laws.

Natura-1 reports relative changes from ~44k to 1M hours of human video:
  held-out human: Action MSE -47.7%, Action MAE -37.3%
  held-out real robot: MSE -67.7%, MAE -49.3%
It does not publish absolute values, error bars, or intermediate rungs.

For a power law q(D) = q0 * (D/D0)^alpha, a relative change f gives
  alpha = ln(1 + f) / ln(D1/D0).
With only two points this is an exact fit with zero residual degrees of
freedom: it is a descriptive exponent, not a tested scaling law.
"""
from __future__ import annotations

import math

D0 = 44_000.0
D1 = 1_000_000.0
RATIO = D1 / D0

NATURA = {
    "human Action MSE": -0.477,
    "human Action MAE": -0.373,
    "robot Action MSE": -0.677,
    "robot Action MAE": -0.493,
}

DYNA2_EXP = {
    "human acc@0.5": 0.0203,
    "human acc@0.1": 0.0606,
    "zero-shot robot acc@0.5": 0.1390,
}


def exponent(rel_change: float) -> float:
    """Power-law exponent from a relative change over D0 -> D1."""
    return math.log1p(rel_change) / math.log(RATIO)


def per_doubling(alpha: float) -> float:
    """Fractional change per doubling of data."""
    return 2.0**alpha - 1.0


def main() -> None:
    print(f"data ratio D1/D0 = {RATIO:.3f}  (ln = {math.log(RATIO):.4f})")
    print()
    print("Natura-1: two-point exponents (exact fit, zero residual dof)")
    print(f"{'metric':<22} {'change':>8} {'alpha':>9} {'per 2x':>9}")
    for name, f in NATURA.items():
        a = exponent(f)
        print(f"{name:<22} {f*100:>7.1f}% {a:>9.4f} {per_doubling(a)*100:>8.1f}%")
    print()
    print("Dyna-2: published power-law exponents (D in hours)")
    print(f"{'metric':<28} {'alpha':>9} {'per 2x':>9}")
    for name, a in DYNA2_EXP.items():
        print(f"{name:<28} {a:>+9.4f} {per_doubling(a)*100:>+8.1f}%")
    print()
    n_human = exponent(NATURA["human Action MSE"])
    n_robot = exponent(NATURA["robot Action MSE"])
    print(f"Natura robot/human MSE exponent ratio: {n_robot/n_human:.2f}x")
    print(
        "Dyna-2 robot/human acc@0.5 ratio:      "
        f"{DYNA2_EXP['zero-shot robot acc@0.5']/DYNA2_EXP['human acc@0.5']:.2f}x"
    )
    print()
    print("Dyna-2's laws evaluated over the same 44k -> 1M span:")
    for name, a in DYNA2_EXP.items():
        print(f"  {name:<28} relative change = {RATIO**a - 1:+.1%}")
    print()
    print("Illustrative extrapolation of Natura's two-point fit to 10M hours")
    print("(NOT evidence; shown only to make the weakness of two points concrete):")
    for name, f in NATURA.items():
        a = exponent(f)
        pred = (10_000_000.0 / D1) ** a - 1.0
        print(f"  {name:<22} additional change = {pred:+.1%}")
    print()
    print("Caveats:")
    print("  - Natura's curve is 'as training advances', not a data-only ladder;")
    print("    compute and checkpoint schedule co-vary with hours.")
    print("  - Metrics are teacher-forced action errors, not closed-loop success.")
    print("  - MSE and MAE give different exponents (-0.21 vs -0.15 human;")
    print("    -0.36 vs -0.22 robot), so even Natura's two points do not lie on")
    print("    one scale factor: the error distribution changes shape.")
    print("  - No absolute values, error bars, task breakdown, or R^2 are published.")
    print("  - Dyna-2 uses 4 rungs (1k/10k/100k/1M), 4 metrics, R^2 0.865-0.926,")
    print("    and a controlled data-only ladder; still self-reported, no code/weights.")


if __name__ == "__main__":
    main()
