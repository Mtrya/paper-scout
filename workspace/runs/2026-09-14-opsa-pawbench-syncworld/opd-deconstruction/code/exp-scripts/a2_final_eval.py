#!/usr/bin/env python3
"""Re-evaluate a2 conditions' final checkpoints on a larger eval subset.

The in-loop eval (20 questions, k=4, early break) is too noisy to separate
conditions (base rate ~5%, +-1 question = +-5pp). This script batches the k
attempts of the same prompt and reports avg@k / pass@k on n=100 questions.

Usage:
  python a2_final_eval.py --cond opd --ckpt results/a2_opd/ckpt_40.pt \
      --data data/math500-subset.jsonl --n 100 --k 4 --out results/a2_opd_final.json
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a2_train import chat_text, load  # noqa: E402
from verify import verify_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--data", default="data/math500-subset.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    student, tok = load(args.model)
    if args.ckpt not in ("", "none"):   # "--ckpt none" evaluates the untouched base model
        student.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    student.eval()

    rows = [json.loads(l) for l in open(args.data)][: args.n]
    t0 = time.time()
    results = []
    for i, d in enumerate(rows):
        ptext = chat_text(tok, d["prompt"])
        label = str(d["label"])
        ids = tok(ptext, add_special_tokens=False).input_ids
        batch_ids = torch.tensor([ids] * args.k, device=student.device)
        with torch.no_grad():
            out = student.generate(
                batch_ids, attention_mask=torch.ones_like(batch_ids),
                max_new_tokens=args.max_new, do_sample=True,
                temperature=0.7, top_k=20, top_p=0.8,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        atts = []
        lens = []
        for row in out:
            resp = row[len(ids):].tolist()
            lens.append(len(resp))
            atts.append(verify_answer(tok.decode(resp, skip_special_tokens=True), label))
        results.append({"i": i, "ok": any(atts), "attempts": atts, "lens": lens})
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            n_att = sum(len(r["attempts"]) for r in results)
            n_ok = sum(sum(r["attempts"]) for r in results)
            print(f"[{time.time()-t0:.0f}s] {i+1}/{len(rows)} "
                  f"pass@{args.k}={sum(r['ok'] for r in results)/len(results):.3f} "
                  f"avg@{args.k}={n_ok/n_att:.3f}", flush=True)
            with open(args.out, "w") as f:
                json.dump({"cond": args.cond, "ckpt": args.ckpt, "n": len(results),
                           "k": args.k, "results": results,
                           "passk": sum(r["ok"] for r in results) / len(results),
                           "avgk": n_ok / n_att}, f, indent=2)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
