#!/usr/bin/env python3
"""Summarise the six A2 final-eval arms, overall and split by MATH difficulty.

Usage: python a2_final_breakdown.py results/a2_*_final.json
Pairs each arm's per-question attempts with the level metadata written by
math500_levels.py (results/math500_levels.json).
"""
import json
import statistics
import sys

ARMS = ["base", "opd", "opd-oneshot", "fixed-neg", "fixed-pos", "opsa"]
MAXNEW = 2048


def load_levels():
    try:
        d = json.load(open("results/math500_levels.json"))
        return {o["i"]: o["level"] for o in d}
    except Exception:
        return {}


def summarise(rows, levels, sel=None):
    sub = [r for r in rows if sel is None or levels.get(r["i"]) in sel]
    if not sub:
        return None
    n_att = sum(len(r["attempts"]) for r in sub)
    n_ok = sum(sum(r["attempts"]) for r in sub)
    lens = [l for r in sub for l in r["lens"]]
    return dict(n=len(sub), passk=sum(r["ok"] for r in sub) / len(sub),
                avgk=n_ok / max(1, n_att),
                meanlen=statistics.mean(lens) if lens else 0,
                trunc=sum(1 for l in lens if l >= MAXNEW) / max(1, len(lens)))


def main():
    levels = load_levels()
    easy = {1, 2}
    hard = {3, 4, 5}
    print(f"{'arm':12s} {'n':>3s} {'pass@4':>7s} {'avg@4':>7s} {'len':>6s} {'trunc':>6s}"
          f" | {'n_easy':>6s} {'pass@4':>7s} {'avg@4':>7s}"
          f" | {'n_hard':>6s} {'pass@4':>7s} {'avg@4':>7s}")
    store = {}
    for arm in ARMS:
        p = f"results/a2_{arm}_final.json"
        try:
            rows = json.load(open(p))["results"]
        except Exception:
            print(f"{arm:12s}  (missing)")
            continue
        all_ = summarise(rows, levels)
        ez = summarise(rows, levels, easy)
        hd = summarise(rows, levels, hard)
        store[arm] = dict(all=all_, easy=ez, hard=hd)
        f6 = lambda x, k: (f"{x[k]:6.3f}" if x else "   -  ")
        print(f"{arm:12s} {all_['n']:3d} {all_['passk']:7.3f} {all_['avgk']:7.3f} "
              f"{all_['meanlen']:6.0f} {all_['trunc']:6.2f} | "
              f"{(ez or {}).get('n', 0):6d} {f6(ez, 'passk')} {f6(ez, 'avgk')} | "
              f"{(hd or {}).get('n', 0):6d} {f6(hd, 'passk')} {f6(hd, 'avgk')}")
    json.dump(store, open("results/a2_breakdown.json", "w"), indent=1)


if __name__ == "__main__":
    main()
