"""Remove truncated (finish_reason=length) or empty records from a raw jsonl so
the idempotent runner will re-run those task_ids on the next invocation.

Usage: python purge_failed.py results/raw/<model>.jsonl
"""
import json
import sys

path = sys.argv[1]
kept, dropped = [], 0
with open(path) as f:
    for line in f:
        r = json.loads(line)
        if r.get("finish_reason") == "length" or not r.get("response", "").strip():
            dropped += 1
        else:
            kept.append(line)
with open(path, "w") as f:
    f.writelines(kept)
print(f"kept {len(kept)}, dropped {dropped}")
