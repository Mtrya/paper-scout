#!/usr/bin/env python3
"""(1) Reproduce one generate_batch call and check prompt-prefix contamination.
(2) Count cached pairs that actually contain a \\boxed{} answer."""
import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys
sys.path.insert(0, ".")
from a1_noise import chat_text, generate_batch, load_model  # noqa: E402

rows = [json.loads(l) for l in open("data/dapo-math-17k.jsonl")][:120]
cache = json.load(open("results/a1_sampled.json"))
cache = {int(k): v for k, v in cache.items()}

# ---- (2) boxed coverage in the cache ----
n_ok_box = n_bad_box = 0
by = {"correct": [0, 0], "incorrect": [0, 0]}   # [with_box, total]
for q, rec in cache.items():
    for bucket in ("correct", "incorrect"):
        for text in rec[bucket]:
            has = bool(re.search(r"\\boxed\{", text))
            by[bucket][1] += 1
            by[bucket][0] += int(has)
            n_ok_box += int(has)
            n_bad_box += int(not has)
print(f"cache pairs with boxed: {n_ok_box}, without: {n_bad_box}")
for b, (wb, tot) in by.items():
    print(f"  {b}: {wb}/{tot} have boxed")

# ---- (1) generation mechanism test ----
model, tok = load_model("models/Qwen3-1.7B")
idx = [0, 5, 9, 10]     # a few questions with different prompt lengths
ptexts = [chat_text(tok, rows[i]["prompt"]) for i in idx]
print("prompt lens:", [len(p) for p in ptexts])
outs = generate_batch(model, tok, ptexts, max_new=16)
for i, (p, (text, r)) in zip(idx, zip(ptexts, outs)):
    p_clean = re.sub(r"<\|[^|]*\|>", "", p)
    print(f"q={i}: resp_len_tok={len(r)} starts_with_prompt_text="
          f"{text.startswith(p)} starts_with_clean={text.startswith(p_clean)} "
          f"head={text[:60]!r}")
