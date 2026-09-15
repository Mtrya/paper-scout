#!/usr/bin/env python3
"""Attach MATH-500 difficulty/subject metadata to our eval subset.

Our data/math500-subset.jsonl keeps only prompt+label, so we re-fetch the
source dataset (HuggingFaceH4/MATH-500) and match problems by text, giving a
per-question level (1-5) and subject. That lets us look at accuracy on the
easy nested subset without re-running anything.
"""
import json
import os
import re
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def norm(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    rows = [json.loads(l) for l in open("data/math500-subset.jsonl")]
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        src = [dict(r) for r in ds]
    except Exception as e:
        print("datasets failed:", e, file=sys.stderr)
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl",
                            repo_type="dataset")
        src = [json.loads(l) for l in open(p)]
    print("MATH-500 rows:", len(src))

    index = {}
    for r in src:
        key = norm(r["problem"])
        index[key[-90:]] = r      # tail-of-problem keys are distinctive
    out, miss = [], 0
    for i, d in enumerate(rows):
        content = norm(d["prompt"][0]["content"])
        hit = None
        for k, v in index.items():
            if k in content:
                hit = v
                break
        if hit is None:
            miss += 1
            out.append({"i": i, "level": None, "subject": None})
        else:
            out.append({"i": i, "level": hit.get("level"), "subject": hit.get("subject")})
    print("matched", len(rows) - miss, "/", len(rows))
    with open("results/math500_levels.json", "w") as f:
        json.dump(out, f, indent=1)
    from collections import Counter
    print(Counter(o["level"] for o in out))


if __name__ == "__main__":
    main()
