"""Calibrate the frozen base: per-family / per-level success, format validity, timing.

    python base_rate.py 80 1234
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict

import harness as h
import taskgen as tg


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    mdl = h.Model.load()
    cfg = h.Config()
    stream = tg.build_stream(seed=seed, n_tasks=n, n_phases=4)
    agg = defaultdict(Counter)
    t0 = time.time()
    tok_total = 0
    for i, t in enumerate(stream):
        ep, rec = h.run_episode(mdl, t, cfg)
        agg[t.family]["n"] += 1
        agg[t.family]["succ"] += rec["success"]
        agg[t.family]["fmt"] += rec["format_valid_rate"]
        agg[t.family]["inv"] += rec["invalid_turns"]
        agg[t.family]["turns"] += rec["n_turns"]
        agg[f"level{t.level}"]["n"] += 1
        agg[f"level{t.level}"]["succ"] += rec["success"]
        agg["ALL"]["n"] += 1
        agg["ALL"]["succ"] += rec["success"]
        agg["ALL"]["fmt"] += rec["format_valid_rate"]
        agg["ALL"]["inv"] += rec["invalid_turns"]
        agg["ALL"]["turns"] += rec["n_turns"]
        tok_total += rec["n_resp_tokens"]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} overall succ={agg['ALL']['succ']/agg['ALL']['n']:.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    print("\nfamily        n  succ   fmt   inv/turn  turns/task")
    for k in ("strxform", "calc", "world", "listops", "level0", "level1", "level2",
              "level3", "ALL"):
        a = agg[k]
        if not a["n"]:
            continue
        print(f"{k:12s} {a['n']:3d} {a['succ']/a['n']:.3f} {a['fmt']/a['n']:.3f} "
              f"{a['inv']/max(1,a['n']):.2f}     {a['turns']/a['n']:.2f}")
    print(f"\nwall={time.time()-t0:.0f}s  resp_tokens/task={tok_total/n:.0f}  "
          f"base rate={agg['ALL']['succ']/agg['ALL']['n']:.3f}")


if __name__ == "__main__":
    main()
