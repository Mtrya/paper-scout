#!/usr/bin/env python3
"""A3: released-checkpoint evaluation on AIME24 (avg@4), paper decoding.

Usage:
  python a3_eval.py --model Tuwhy/Qwen3-1.7B-OPSA --data aime-2024.jsonl \
      --n 30 --k 4 --max-new 16384 --out results/a3_opsa_1p7b.json
  python a3_eval.py --model Qwen/Qwen3-1.7B ... (base reference)
"""
import argparse, json, os, signal, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# official slime/VERL math grader, vendored from the OPSA repo's slime fork so
# our verification matches the paper's own eval pipeline (boxed extraction +
# mathd normalization / sympy equivalence)
from math_utils import grade_answer_verl


class _Timeout(Exception):
    pass


def _handler(signum, frame):  # noqa: ARG001
    raise _Timeout()


def verify_official(text, label, timeout=8.0):
    """grade_answer_verl under a hard timeout (sympy parsing can blow up on
    pathological LaTeX; a hang would stall the whole eval loop)."""
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return bool(grade_answer_verl(text, label))
    except _Timeout:
        return False
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="aime-2024.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=16384)
    ap.add_argument("--temp", type=float, default=0.7)
    # restrict to a subset of original question indices (comma-separated),
    # e.g. re-run only the questions that hit the token cap at half budget
    ap.add_argument("--indices", default="")
    ap.add_argument("--save-text", action="store_true",
                    help="also record the decoded responses (for qualitative "
                         "checks and within-problem diversity)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2", trust_remote_code=True)
    model.eval()
    if hasattr(model.generation_config, "enable_thinking"):
        model.generation_config.enable_thinking = False

    with open(args.data) as f:
        rows = [json.loads(l) for l in f][: args.n]
    if args.indices:
        keep = {int(x) for x in args.indices.split(",") if x.strip()}
        rows = [(i, d) for i, d in enumerate(rows) if i in keep]
    else:
        rows = list(enumerate(rows))
    results = []
    t0 = time.time()
    for i, d in rows:
        msgs, label = d["prompt"], str(d["label"])
        ptext = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
        ids = tok(ptext, add_special_tokens=False).input_ids
        rec = {"i": i, "label": label, "ok": False, "attempts": [], "lens": []}
        # k attempts of the SAME prompt: batch them (no padding needed) —
        # single-sequence decoding runs at ~33 tok/s, which made capped
        # generations take 30+ min per question
        batch_ids = torch.tensor([ids] * args.k, device=model.device)
        with torch.no_grad():
            out = model.generate(
                batch_ids, attention_mask=torch.ones_like(batch_ids),
                max_new_tokens=args.max_new, do_sample=True,
                temperature=args.temp, top_k=20, top_p=0.8,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for row in out:
            resp = row[len(ids):].tolist()
            text = tok.decode(resp, skip_special_tokens=True)
            ok = verify_official(text, label)
            rec["attempts"].append(ok)
            rec["lens"].append(len(resp))
            if args.save_text:
                rec.setdefault("texts", []).append(text)
            if ok:
                rec["ok"] = True
        results.append(rec)
        done = sum(1 for r in results if r["ok"])
        n_att = sum(len(r["attempts"]) for r in results)
        n_ok = sum(sum(r["attempts"]) for r in results)
        print(f"[{time.time()-t0:.0f}s] {i+1}/{len(rows)} solved={done} "
              f"pass@{args.k}={done/max(1,i+1):.3f} "
              f"avg@{args.k}={n_ok/max(1,n_att):.3f}", flush=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "results": results,
                       "pass4": done / max(1, i + 1),
                       "avg4": n_ok / max(1, n_att)}, f, indent=2)
    n_att = sum(len(r["attempts"]) for r in results)
    n_ok = sum(sum(r["attempts"]) for r in results)
    print(f"FINAL pass@{args.k}: {sum(r['ok'] for r in results) / len(results):.3f}  "
          f"avg@{args.k}: {n_ok / n_att:.3f}")


if __name__ == "__main__":
    main()
