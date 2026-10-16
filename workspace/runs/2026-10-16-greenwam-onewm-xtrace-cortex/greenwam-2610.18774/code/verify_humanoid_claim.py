#!/usr/bin/env python3
"""Turn the Green humanoid protocol of the paper (Appendix C.2) into integer counts and test the headline.

The paper reports instruction success per split (Equation 14) for four policies on 45 in-distribution and
25 out-of-distribution instructions per policy, and the abstract says GreenWAM (95.6%) "succeeds more often
than policies adapted from pi0.5 (84.4%) and from Cosmos 3 Edge (88.9%)". This script recovers the integer
counts the percentages imply and asks how strong that statement is.

    python3 verify_humanoid_claim.py
"""

from __future__ import annotations

from math import comb, sqrt

ID_N, OOD_N = 45, 25
REPORTED = {  # policy: (ID success, OOD success), from Section 6.5
    "GreenWAM (teacher)": (0.956, 0.840),
    "Cosmos 3 Edge policy": (0.889, 0.800),
    "pi0.5": (0.844, 0.760),
    "GreenWAM (distilled)": (0.911, 0.800),
}


def fisher_two_sided(a, b, c, d):
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    lo, hi = max(0, c1 - r2), min(r1, c1)

    def p(k):
        return comb(r1, k) * comb(r2, c1 - k) / comb(n, c1)

    obs = p(a)
    return sum(p(k) for k in range(lo, hi + 1) if p(k) <= obs + 1e-12)


def diff_ci(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return d, d - 1.96 * se, d + 1.96 * se


def main() -> int:
    counts = {}
    print(f"{'policy':24s} {'ID':>10s} {'OOD':>10s}")
    for name, (pid, pood) in REPORTED.items():
        sid, sood = round(pid * ID_N), round(pood * OOD_N)
        counts[name] = (sid, sood)
        print(f"{name:24s} {sid:3d}/{ID_N} {100*sid/ID_N:5.1f}% {sood:3d}/{OOD_N} {100*sood/OOD_N:5.1f}%")

    g_id, g_ood = counts["GreenWAM (teacher)"]
    print()
    for other in ("Cosmos 3 Edge policy", "pi0.5"):
        o_id, o_ood = counts[other]
        p_id = fisher_two_sided(g_id, ID_N - g_id, o_id, ID_N - o_id)
        p_ood = fisher_two_sided(g_ood, OOD_N - g_ood, o_ood, OOD_N - o_ood)
        d_id, lo_id, hi_id = diff_ci(g_id, ID_N, o_id, ID_N)
        d_ood, lo_ood, hi_ood = diff_ci(g_ood, OOD_N, o_ood, OOD_N)
        print(f"GreenWAM vs {other}")
        print(f"  ID : {g_id}/{ID_N} vs {o_id}/{ID_N}  diff {100*d_id:+.1f} pp, 95% CI [{100*lo_id:+.1f}, {100*hi_id:+.1f}], Fisher p = {p_id:.3f}")
        print(f"  OOD: {g_ood}/{OOD_N} vs {o_ood}/{OOD_N}  diff {100*d_ood:+.1f} pp, 95% CI [{100*lo_ood:+.1f}, {100*hi_ood:+.1f}], Fisher p = {p_ood:.3f}")
    print()
    print("Reading: with 45 ID and 25 OOD instructions the ID gap to pi0.5 is 5 instructions and the Fisher")
    print("test does not separate the policies at 95% (the CI includes 0). The ordering is consistent across")
    print("both splits and the distilled student keeps it, but the headline rests on a small instruction set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
