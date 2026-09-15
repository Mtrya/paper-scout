#!/usr/bin/env python3
"""Is the A2 eval floor a property of the model or of the decoding config?

Evaluates the *untrained* student on a handful of MATH-500 questions under
three configs and reports accuracy plus the format-compliance rate (does the
response contain a \\boxed{} at all — our grader requires one).
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a2_train import chat_text, load  # noqa: E402
from verify import verify_answer  # noqa: E402

CONFIGS = [
    ("eval cfg (temp .7/k20/p.8, 2048)", dict(temp=0.7, top_k=20, top_p=0.8, max_new=2048)),
    ("train cfg (temp 1.0/full, 2048)", dict(temp=1.0, top_k=0, top_p=1.0, max_new=2048)),
    ("eval cfg, 4096 tokens", dict(temp=0.7, top_k=20, top_p=0.8, max_new=4096)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--data", default="data/math500-subset.jsonl")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--k", type=int, default=1)
    args = ap.parse_args()

    student, tok = load(args.model)
    student.eval()
    rows = [json.loads(l) for l in open(args.data)][: args.n]
    for name, cfg in CONFIGS:
        t0 = time.time()
        ok = tot = boxed = 0
        lens, ex = [], []
        for d in rows:
            ptext = chat_text(tok, d["prompt"])
            label = str(d["label"])
            ids = tok(ptext, add_special_tokens=False).input_ids
            batch = torch.tensor([ids] * args.k, device=student.device)
            with torch.no_grad():
                out = student.generate(
                    batch, attention_mask=torch.ones_like(batch),
                    max_new_tokens=cfg["max_new"], do_sample=True,
                    temperature=cfg["temp"], top_k=cfg["top_k"],
                    top_p=cfg["top_p"],
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for row in out:
                resp = row[len(ids):].tolist()
                text = tok.decode(resp, skip_special_tokens=True)
                lens.append(len(resp))
                has_box = "\\boxed{" in text
                boxed += has_box
                if has_box and len(ex) < 3:
                    ex.append(text[-260:])
                ok += verify_answer(text, label)
                tot += 1
        print(f"== {name}: acc {ok}/{tot} = {ok/tot:.2f}  boxed-anywhere "
              f"{boxed}/{tot} = {boxed/tot:.2f}  mean_len {sum(lens)/len(lens):.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for e in ex:
            print("   ...", e.replace("\n", " ")[-200:])


if __name__ == "__main__":
    main()
