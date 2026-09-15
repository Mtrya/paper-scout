#!/usr/bin/env python3
"""Quantify the generate_batch slicing bug: stored responses = prompt?+response."""
import json
import re
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/Qwen3-1.7B", trust_remote_code=True)

rows = [json.loads(l) for l in open("data/dapo-math-17k.jsonl")][:120]
cache = json.load(open("results/a1_sampled.json"))
cache = {int(k): v for k, v in cache.items()}


def chat_text(messages):
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


def strip_special(s):
    return re.sub(r"<\|[^|]*\|>", "", s)


n_total = n_prefixed = n_clean = 0
examples = []
for q, rec in cache.items():
    p = chat_text(rows[q]["prompt"])
    p_clean = strip_special(p)
    for bucket in ("correct", "incorrect"):
        for text in rec[bucket]:
            n_total += 1
            if text.startswith(p):
                n_prefixed += 1
            elif text.startswith(p_clean):
                n_prefixed += 1
                if len(examples) < 2:
                    examples.append((q, bucket, p_clean[:80], text[:80]))
            else:
                n_clean += 1

print(f"samples={n_total} prefixed={n_prefixed} clean={n_clean} "
      f"({100.0*n_prefixed/max(1,n_total):.1f}% contaminated)")
for q, b, pc, tc in examples:
    print(f"--- q={q} {b}:\nprompt_clean: {pc!r}\nstored_head:  {tc!r}")

# where does the answer live in a contaminated sample?
q0 = next(iter(cache))
rec = cache[q0]
for b in ("correct", "incorrect"):
    if rec[b]:
        t = rec[b][0]
        m = re.search(r"\\boxed\{", t)
        print(f"q={q0} {b}: len={len(t)} boxed_at={m.start() if m else None} "
              f"prompt_len={len(chat_text(rows[q0]['prompt']))}")
        break
