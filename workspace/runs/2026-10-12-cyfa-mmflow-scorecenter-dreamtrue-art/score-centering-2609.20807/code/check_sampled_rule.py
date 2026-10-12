"""Verification: the sampled rule (G rollouts per context, group-centered
advantages, per-token IS weights) must have the exact expectation that the
deterministic dynamics uses, up to the 1 - 1/G group-centering factor.

Reported as a z-score of the Monte-Carlo mean against the exact value, in units
of the aggregate block standard error, plus the cosine between the two update
tensors. Usage: uv run --with numpy python3 check_sampled_rule.py
"""

import numpy as np

import toy_lib as T

if __name__ == "__main__":
    rows = T.check_sampled_vs_exact(group=8, blocks=30, per_block=200)
    bad = 0
    for r in rows:
        flag = "" if max(r["d1_z"], r["d2_z"]) < 4.0 else "  <-- CHECK"
        bad += flag != ""
        print(f"{r['lane']:9s} {r['method']:8s} "
              f"d1: z={r['d1_z']:5.2f} cos={r['d1_cos']:.5f}   "
              f"d2: z={r['d2_z']:5.2f} cos={r['d2_cos']:.5f}{flag}")
    print(f"\nmax z = {max(max(r['d1_z'], r['d2_z']) for r in rows):.2f} "
          f"over {len(rows)} cases; min cos = "
          f"{min(min(r['d1_cos'], r['d2_cos']) for r in rows):.5f}; "
          f"rows above z=4: {bad}")
