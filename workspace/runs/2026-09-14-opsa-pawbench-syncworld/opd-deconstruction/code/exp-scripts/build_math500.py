#!/usr/bin/env python3
"""Build data/math500-subset.jsonl (100 rows) in the same schema as
dapo-math-17k.jsonl: {"prompt": [{"role":"user","content": <instruction+problem>}],
"label": "<answer>"}  with the identical instruction wrapper."""
import glob
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset",
                  local_dir="data/math500-dl")

rows_in = [json.loads(l) for l in open("data/math500-dl/test.jsonl")]
print("rows:", len(rows_in), "cols:", list(rows_in[0].keys()))

INSTR = ("Solve the following math problem step by step. The last line of "
         "your response should be of the form Answer: \\boxed{$Answer} where "
         "$Answer is the answer to the problem.\n\n")

out = []
for r in rows_in[:100]:
    out.append({"prompt": [{"role": "user", "content": INSTR + str(r["problem"])}],
                "label": str(r["answer"])})
with open("data/math500-subset.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("wrote data/math500-subset.jsonl:", len(out), "rows")
print("first label:", out[0]["label"])
