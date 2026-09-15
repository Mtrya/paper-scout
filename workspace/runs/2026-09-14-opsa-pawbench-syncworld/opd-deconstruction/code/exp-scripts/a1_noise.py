#!/usr/bin/env python3
"""A1: OPD teacher-noise measurement (Fig 2a replication), batched.

For each of N questions: sample student responses until >=1 correct and >=1
incorrect (cap MAX_ATT), then score each response's answer-token span with each
teacher: A_i = log pi_t(y_i|y_<i) - log pi_s(y_i|y_<i). A trajectory's answer
supervision is "noisy" when mean advantage over the boxed answer span
disagrees in sign with correctness. Reports noise rates per teacher.

Usage:
  python a1_noise.py --student /path/models/Qwen3-1.7B \
      --teachers /path/models/Qwen3-4B-Instruct /path/models/Qwen3-14B-Instruct \
      --data dapo-math-17k.jsonl --n 250 --out results/a1_noise.json
"""
import argparse, gc, json, os, re, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import last_boxed_content, verify_answer
from verify_fast import verify_fast

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2", trust_remote_code=True)
    model.eval()
    if hasattr(model.generation_config, "enable_thinking"):
        model.generation_config.enable_thinking = False
    return model, tok


def chat_text(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


def pad_left(list_of_ids, pad_id):
    L = max(len(x) for x in list_of_ids)
    padded, masks = [], []
    for x in list_of_ids:
        padded.append([pad_id] * (L - len(x)) + x)
        masks.append([0] * (L - len(x)) + [1] * len(x))
    return torch.tensor(padded), torch.tensor(masks)


@torch.no_grad()
def generate_batch(model, tok, prompt_texts, max_new=4096, temp=1.0):
    """Return list of (text, token_ids) per prompt."""
    import time as _t
    ids = [tok(p, add_special_tokens=False).input_ids for p in prompt_texts]
    padded, masks = pad_left(ids, tok.pad_token_id)
    L = padded.size(1)
    print(f"[gb] batch={len(ids)} lens={[len(i) for i in ids]} start", flush=True)
    t0 = _t.time()
    gen = model.generate(
        padded.to(model.device), attention_mask=masks.to(model.device),
        max_new_tokens=max_new, do_sample=True, temperature=temp,
        top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id)
    print(f"[gb] generate done in {_t.time()-t0:.1f}s", flush=True)
    out = []
    for g in gen:
        r = g[L:].tolist()      # left-padded batch: response starts after the pad length L
        out.append((tok.decode(r, skip_special_tokens=True), r))
    return out


@torch.no_grad()
def score_batch(model, tok, pairs):
    """pairs: list of (prompt_text, response_text). Return per-pair list of
    (sampled-token logp tensor, offsets list).

    Two fixes vs the first version (2026-09-15):
    - store only the logp AT the sampled token ((Lr,) floats) instead of the
      full (Lr, V) row — 505 pairs x 1.2GB of full-vocab rows OOM-killed the
      100GB container earlier;
    - index with the left-pad offset (pad_left pads the shorter rows), the
      first version sliced as if nothing were padded."""
    results = []
    for i in range(0, len(pairs), 2):
        chunk = pairs[i:i + 2]
        pids = [tok(p, add_special_tokens=False).input_ids for p, _ in chunk]
        rids = [tok(r, add_special_tokens=False).input_ids for _, r in chunk]
        seqs = [p + r for p, r in zip(pids, rids)]
        padded, masks = pad_left(seqs, tok.pad_token_id)
        L = padded.size(1)
        out = model(padded.to(model.device), attention_mask=masks.to(model.device))
        logits = out.logits.float()                     # (B, L, V), transient
        logz = torch.logsumexp(logits, dim=-1)          # (B, L)
        for b, (p, r) in enumerate(zip(pids, rids)):
            Lp, Lr = len(p), len(r)
            if Lr == 0:
                results.append((torch.zeros(0), []))
                continue
            start = (L - (Lp + Lr)) + Lp - 1            # first pos predicting resp
            idx = torch.tensor(r, device=logits.device)
            tok_logit = logits[b, start:start + Lr].gather(1, idx[:, None]).squeeze(1)
            sampled_logp = tok_logit - logz[b, start:start + Lr]
            offsets, cur = [], ""
            for t in r:
                nxt = tok.decode([t], add_special_tokens=False)
                offsets.append((len(cur), len(cur) + len(nxt)))
                cur += nxt
            results.append((sampled_logp.cpu(), offsets))
    return results


def answer_span_tokens(response_text, offsets):
    m = last_boxed_content(response_text)
    if m is None:
        return []
    start, end = m
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end]


def strip_prompt_tail_prefix(p_clean, s, min_k=8, max_k=1500):
    """Responses cached before the generate_batch slicing fix (left-pad
    artifact) start with a copy of the prompt's TAIL (up to the whole prompt).
    Strip the longest matching tail so scoring sees prompt + response only.
    Returns (text, stripped_chars)."""
    hi = min(len(p_clean), len(s), max_k)
    for k in range(hi, min_k - 1, -1):
        if s.startswith(p_clean[-k:]):
            return s[k:], k
    return s, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="/models/Qwen3-1.7B")
    ap.add_argument("--teachers", nargs="+", default=[])
    ap.add_argument("--data", default="dapo-math-17k.jsonl")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--max-att", type=int, default=6)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--out", default="results/a1_noise.json")
    ap.add_argument("--sample-cache", default="results/a1_sampled.json")
    ap.add_argument("--fast", action="store_true",
                    help="use fast latex2sympy verifier for sampling")
    args = ap.parse_args()
    if args.fast:
        verify_answer = verify_fast

    with open(args.data) as f:
        rows = [json.loads(l) for l in f][: args.n]
    questions = [(d["prompt"], str(d["label"])) for d in rows]

    smodel, stok = load_model(args.student)

    # sample until 1 correct + 1 incorrect per question (capped), cached
    sampled = {}
    ptexts = {qi: chat_text(stok, msgs) for qi, (msgs, _) in enumerate(questions)}
    t0 = time.time()
    if os.path.exists(args.sample_cache):
        with open(args.sample_cache) as f:
            sampled = json.load(f)
        sampled = {int(k): v for k, v in sampled.items()}
        remaining = {q for q in range(len(questions))
                     if not (sampled.get(q, {}).get("correct")
                             and sampled.get(q, {}).get("incorrect"))}
        print(f"loaded {len(sampled)} cached samples; {len(remaining)} remain",
              flush=True)
    else:
        remaining = set(range(len(questions)))
    for attempt in range(args.max_att):
        if not remaining:
            break
        qids = sorted(remaining)
        for g in range(0, len(qids), args.group):
            grp = qids[g:g + args.group]
            outs = generate_batch(smodel, stok, [ptexts[q] for q in grp],
                                  max_new=args.max_new)
            for q, (text, _) in zip(grp, outs):
                ok = verify_answer(text, questions[q][1])
                rec = sampled.setdefault(q, {"correct": [], "incorrect": []})
                rec["correct" if ok else "incorrect"].append(text)
                if rec["correct"] and rec["incorrect"]:
                    remaining.discard(q)
        os.makedirs(os.path.dirname(args.sample_cache), exist_ok=True)
        with open(args.sample_cache, "w") as f:
            json.dump(sampled, f)
        done = len(questions) - len(remaining)
        print(f"[{time.time()-t0:.0f}s] attempt {attempt+1}: {done}/{len(questions)} "
              f"done ({len(remaining)} remaining)", flush=True)

    # scoring phase: pool all sampled responses. Responses cached before the
    # generate_batch slicing fix carry a leading copy of the prompt TAIL
    # (left-pad artifact, ~3/4 of batches); strip it so the scored context is
    # prompt + response only.
    pairs = []   # (qi, ok, prompt_text, response_text)
    n_stripped, strip_lens = 0, []
    for qi, rec in sampled.items():
        p = ptexts[qi]
        p_clean = re.sub(r"<\|[^|]*\|>", "", p)
        for ok, texts in [(True, rec["correct"]), (False, rec["incorrect"])]:
            for text in texts:
                text, k = strip_prompt_tail_prefix(p_clean, text)
                if k:
                    n_stripped += 1
                    strip_lens.append(k)
                pairs.append((qi, ok, p, text))
    print(f"pool: {len(pairs)} (question, response) pairs; "
          f"stripped prompt-tail prefix on {n_stripped} "
          f"(mean {sum(strip_lens)/max(1,len(strip_lens)):.0f} chars)", flush=True)

    # scoring must fit on one 48GB card: score the student first and free it,
    # then load teachers ONE at a time (14B bf16 alone is 28GB; keeping all
    # three models resident OOMs at ~45GB allocated — 2026-09-14 crash).
    # Token scores are cached so aggregation bugs do not cost a re-score.
    score_cache = "results/a1_scores.pt"
    cached = None
    if os.path.exists(score_cache):
        try:
            cached = torch.load(score_cache, map_location="cpu")
            if cached.get("n_pairs") != len(pairs):
                print(f"score cache stale ({cached.get('n_pairs')} != {len(pairs)}), rescoring", flush=True)
                cached = None
        except Exception as e:  # noqa: BLE001
            print(f"score cache unreadable ({e}), rescoring", flush=True)
            cached = None
    if cached is not None:
        s_scores, teacher_scores = cached["s"], cached["t"]
        print(f"loaded cached token scores for {len(s_scores)} pairs", flush=True)
    else:
        s_scores = score_batch(smodel, stok, [(p, r) for _, _, p, r in pairs])
        del smodel
        gc.collect()
        torch.cuda.empty_cache()
        teacher_scores = {}
        for t in args.teachers:
            m, tok_ = load_model(t)
            teacher_scores[t] = score_batch(m, tok_, [(p, r) for _, _, p, r in pairs])
            del m
            gc.collect()
            torch.cuda.empty_cache()
            torch.save({"n_pairs": len(pairs), "s": s_scores,
                        "t": teacher_scores}, score_cache)
            print(f"saved token scores through {t}", flush=True)

    results = []
    for idx, (qi, ok, _, rtext) in enumerate(pairs):
        resp_logp, offsets = s_scores[idx]
        span = answer_span_tokens(rtext, offsets)
        rec = {"q": qi, "ok": ok, "n_ans_tok": len(span)}
        for tname in args.teachers:
            t_logp, _ = teacher_scores[tname][idx]
            if len(t_logp) != len(resp_logp):
                rec[tname] = {"mean_adv": None, "n_ans": 0}
                continue
            adv = (t_logp - resp_logp)
            sel = adv[span] if span else torch.zeros(0)
            rec[tname] = {"mean_adv": float(sel.mean()) if len(sel) else 0.0,
                          "n_ans": len(span),
                          "n_neg": int((sel < 0).sum().item()),
                          "mean_abs": float(sel.abs().mean()) if len(sel) else 0.0}
        results.append(rec)

    agg = {}
    for tname in args.teachers:
        stats = {"correct": {"noisy": 0, "n": 0}, "incorrect": {"noisy": 0, "n": 0},
                 "usable": 0}
        for rec in results:
            a = rec.get(tname)
            if a is None or a["mean_adv"] is None or a["n_ans"] == 0:
                continue
            stats["usable"] += 1
            key = "correct" if rec["ok"] else "incorrect"
            stats[key]["n"] += 1
            if (rec["ok"] and a["mean_adv"] < 0) or \
               (not rec["ok"] and a["mean_adv"] > 0):
                stats[key]["noisy"] += 1
        agg[tname] = stats
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"agg": agg, "n_results": len(results),
                   "n_pairs": len(pairs)}, f, indent=2)
    print(json.dumps(agg, indent=2))
    for tname in args.teachers:
        s = agg[tname]
        tot = s["usable"]
        if tot:
            rate = (s["correct"]["noisy"] + s["incorrect"]["noisy"]) / tot
            print(f"{tname}: usable={tot} "
                  f"correct_noise={s['correct']['noisy']}/{s['correct']['n']} "
                  f"incorrect_noise={s['incorrect']['noisy']}/{s['incorrect']['n']} "
                  f"overall={rate:.3f}")


if __name__ == "__main__":
    main()
