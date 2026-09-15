#!/usr/bin/env python3
"""Where does the final \\boxed{} answer appear inside each generation?"""
import json
import sys

BOX = "\\boxed{"


def main():
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/a3_opsa_ext.json"))
    for r in d["results"]:
        print(f"q {r['i']} label={r['label']!r} ok={r['ok']}")
        for j, (ok, ln, t) in enumerate(zip(r["attempts"], r["lens"], r.get("texts", []))):
            pos = t.rfind(BOX)
            frac = pos / len(t) if t else -1
            where = "none" if pos < 0 else f"{frac * 100:.0f}%"
            print(f"   s{j} ok={ok} len={ln} n_box={t.count(BOX)} "
                  f"last_box_at={pos} chars ({where}) ~token≈{int(frac * ln)}")


if __name__ == "__main__":
    main()
