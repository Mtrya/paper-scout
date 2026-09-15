#!/usr/bin/env python3
"""Compare the 16384-budget A3 results with the 32768-budget extension.

Usage: python a3_ext_compare.py results/a3_opsa.json results/a3_opsa_ext.json
"""
import json
import re
import sys

CAP_OLD, CAP_NEW = 16384, 32768


def load(p):
    d = json.load(open(p))
    return {r["i"]: r for r in d["results"]}


def top_repeat(text, n=60, min_count=3):
    """crude loop detector: most frequent n-char window and its count"""
    if len(text) < n * 2:
        return text, 1
    seen = {}
    best, bc = "", 0
    for i in range(0, len(text) - n, max(1, n // 3)):
        w = text[i:i + n]
        seen[w] = seen.get(w, 0) + 1
        if seen[w] > bc:
            best, bc = w, seen[w]
    return best, bc


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    for i, r in sorted(new.items()):
        o = old.get(i)
        print(f"\n===== question {i}  label={r['label']!r}")
        if o is not None:
            print(f"  old @{CAP_OLD}: attempts={o['attempts']} lens={o['lens']}")
        print(f"  new @{CAP_NEW}: attempts={r['attempts']} lens={r['lens']}")
        texts = r.get("texts", [])
        for j, (ok, ln, t) in enumerate(zip(r["attempts"], r["lens"], texts)):
            boxed = re.findall(r"\\boxed\{", t)
            rep, cnt = top_repeat(t)
            tail = re.sub(r"\s+", " ", t[-220:])
            head = re.sub(r"\s+", " ", t[:220])
            print(f"  -- sample {j}: ok={ok} len={ln} cap_hit={ln >= CAP_NEW} "
                  f"n_boxed={len(boxed)} top_repeat={cnt}x {rep[:50]!r}")
            print(f"     head: {head}")
            print(f"     tail: {tail}")


if __name__ == "__main__":
    main()
