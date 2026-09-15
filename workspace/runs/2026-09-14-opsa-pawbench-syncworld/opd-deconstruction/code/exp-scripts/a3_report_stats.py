#!/usr/bin/env python3
"""Report-side statistics for the A3 released-checkpoint evaluation.

Reads results/a3_*.json (written incrementally by a3_eval.py) and prints
pass@k / avg@k together with a truncation breakdown, since len == max_new
means the response was cut off before its final answer box.
"""
import json
import statistics
import sys

MAXNEW = 16384


def stats(path):
    d = json.load(open(path))
    rows = d["results"]
    n = len(rows)
    k = len(rows[0]["attempts"])
    pk = d.get("passk", d.get("pass4"))
    ak = d.get("avgk", d.get("avg4"))
    lens = [l for r in rows for l in r.get("lens", [])]
    n_trunc = sum(1 for l in lens if l >= MAXNEW)
    # per-question, restrict to attempts that ended naturally
    nat_ok, nat_n = 0, 0
    tr_ok, tr_n = 0, 0
    for r in rows:
        for ok, l in zip(r["attempts"], r.get("lens", [])):
            if l >= MAXNEW:
                tr_n += 1
                tr_ok += ok
            else:
                nat_n += 1
                nat_ok += ok
    ok_lens = [l for r in rows for ok, l in zip(r["attempts"], r.get("lens", [])) if ok]
    bad_lens = [l for r in rows for ok, l in zip(r["attempts"], r.get("lens", [])) if not ok]
    print(f"--- {path}  ({d.get('model', d.get('cond', '?'))})")
    print(f"questions          {n}   (k={k})")
    print(f"pass@{k} / avg@{k}   {pk:.3f} / {ak:.3f}")
    if lens:
        print(f"truncated @{MAXNEW}   {n_trunc}/{len(lens)} = {n_trunc/len(lens):.1%}")
        print(f"  natural-end acc  {nat_ok}/{nat_n} = {nat_ok/max(1,nat_n):.1%}")
        print(f"  truncated  acc   {tr_ok}/{tr_n} = {tr_ok/max(1,tr_n):.1%}")
        print(f"len overall        median {statistics.median(lens):.0f}  mean {statistics.mean(lens):.0f}")
    if ok_lens:
        print(f"len correct        median {statistics.median(ok_lens):.0f}  mean {statistics.mean(ok_lens):.0f}")
    if bad_lens:
        print(f"len incorrect      median {statistics.median(bad_lens):.0f}  mean {statistics.mean(bad_lens):.0f}")
    solved = [r["i"] for r in rows if r["ok"]]
    print(f"solved questions   {solved}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=16384)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    MAXNEW = a.max_new
    for p in a.paths:
        stats(p)
