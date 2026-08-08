"""Extract verbatim probe-2 cases: entries that are ground-truth stale, judged
valid under an ablated image, whose reasoning describes image content consistent
with the memory claim rather than reality.

Usage: python extract_cases.py <model> [--condition blank|mismatch] [--n 5]
"""
from __future__ import annotations

import argparse
import json
import os

from analyze import DATA, RAW, parse_judgments, load_model_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--condition", default="blank")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(DATA, "tasks.jsonl"))}
    gold = json.load(open(os.path.join(DATA, "gold.json")))
    recs = load_model_raw(os.path.join(RAW, f"{args.model}.jsonl"))

    cases = []
    for tid, g in gold.items():
        t = tasks[tid]
        if t["probe"] != "p2" or t["condition"] != args.condition:
            continue
        if tid not in recs:
            continue
        pred = parse_judgments(recs[tid]["response"])
        if pred is None:
            continue
        for mid, is_stale in g.items():
            p = pred.get(mid)
            if p is None:
                continue
            # truly stale but judged valid, with a fluent visual claim
            if is_stale and not p["is_stale"] and len(p["reasoning"]) > 40:
                score = sum(k in p["reasoning"].lower() for k in
                            ("image", "shows", "blue", "frozen", "light", "cell", "matches", "consistent"))
                mem_text = None
                cases.append((score, t["seed"], t["regime"], mid, p, recs[tid]["response"]))
    cases.sort(key=lambda x: -x[0])
    for score, seed, regime, mid, p, raw in cases[: args.n]:
        print("=" * 70)
        print(f"seed={seed} regime={regime} condition={args.condition} {mid} gold=STALE pred=valid")
        print(f"confidence={p['confidence']}")
        print("reasoning:", json.dumps(p["reasoning"], ensure_ascii=False))


if __name__ == "__main__":
    main()
