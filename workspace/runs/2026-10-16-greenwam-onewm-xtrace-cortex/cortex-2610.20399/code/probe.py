"""Prospective-query probe for Cortex (arXiv 2610.20399).

Question: do rehearsal queries that *anticipate future use* of a context buy
anything over rehearsal queries that *restate* the context, when both are
otherwise identical (same generator, same count, same teacher, same optimizer)?

Pipeline
  gen    : Qwen2.5-3B-Instruct writes rehearsal questions for arm R (restate)
           and arm P (prospective), from the same document.
  build  : the pre-update model answers every question *with* the document in
           context (Cortex Eq. 4: y ~ M(.|C, q)); a light filter drops
           meta-questions ("according to the document ...").
  train  : FFN-only LoRA on Qwen3-0.6B, student without context, teacher with
           context; loss = truncated-KL distillation (Cortex Eq. 8) or CE.
  eval   : 4-way forced choice over candidate answer strings, on two held-out
           sets (restate / use) plus an MMLU retention panel.
"""

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_data import CONTEXT, EVAL_SETS, GEN_PROMPTS

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

STUDENT_ID = "Qwen/Qwen3-0.6B"
GENERATOR_ID = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_BASE = "You are an internal engineering assistant for a software team."
SYSTEM_CTX = (
    "You are an internal engineering assistant for a software team. "
    "The following internal documentation is provided for reference.\n\n"
    "<document>\n{ctx}\n</document>"
)

TEMPERATURE = 2.0  # KL temperature, Cortex Table 5

torch.backends.cuda.matmul.allow_tf32 = True


# --------------------------------------------------------------------------- io
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------------ model
def load_student(device="cuda", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(STUDENT_ID)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    return tok, model


def chat(tok, messages, add_generation_prompt=False):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def encode_with_span(tok, prefix, suffix):
    """Tokens for prefix+suffix plus the (start, end) span of suffix tokens."""
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tok(prefix + suffix, add_special_tokens=False)["input_ids"]
    if full_ids[: len(pre_ids)] != pre_ids:
        return None
    return full_ids, len(pre_ids), len(full_ids)


# --------------------------------------------------------------- scoring (eval)
@torch.no_grad()
def score_candidates(model, tok, question, candidates, ctx=None, device="cuda"):
    """Sum of log P(candidate | prompt) for each candidate. Prompt = one user turn."""
    sysmsg = SYSTEM_CTX.format(ctx=ctx) if ctx else SYSTEM_BASE
    messages = [{"role": "system", "content": sysmsg}, {"role": "user", "content": question}]
    prefix = chat(tok, messages, add_generation_prompt=True)

    seqs, spans = [], []
    for cand in candidates:
        enc = encode_with_span(tok, prefix, " " + cand)
        if enc is None:
            raise RuntimeError("prefix/suffix tokenization mismatch")
        ids, s, e = enc
        seqs.append(ids)
        spans.append((s, e))

    maxlen = max(len(s) for s in seqs)
    pad_id = tok.pad_token_id or tok.eos_token_id
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    shifts = []
    for i, ids in enumerate(seqs):
        shift = maxlen - len(ids)
        shifts.append(shift)
        input_ids[i, shift:] = torch.tensor(ids)
        attn[i, shift:] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    logprobs = F.log_softmax(logits, dim=-1)

    scores = []
    for i, (s, e) in enumerate(spans):
        total = 0.0
        for j in range(s, e):
            total += logprobs[i, shifts[i] + j - 1, seqs[i][j]].item()
        scores.append(total)
    return scores


def shuffled_options(question, seed):
    q, correct, distractors = question
    cands = [correct] + list(distractors)
    rng = random.Random(seed)
    order = list(range(len(cands)))
    rng.shuffle(order)
    shuffled = [cands[i] for i in order]
    return q, shuffled, shuffled.index(correct)


def eval_set(model, tok, items, ctx=None, seed=0, device="cuda", tag="", mode="free"):
    """mode='free'  -> candidate answer strings continue the assistant turn
       mode='letter'-> options are listed in the prompt, we score the letters A-D"""
    correct, margins, per_item = 0, [], []
    for k, item in enumerate(items):
        q, cands, ci = shuffled_options(item, seed * 1000 + k)
        if mode == "letter":
            letters = ["A", "B", "C", "D"]
            body = q + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(cands))
            prompt_q = body + "\nAnswer with a single letter (A, B, C or D)."
            sc = score_candidates(model, tok, prompt_q, letters, ctx=ctx, device=device)
        else:
            prompt_q = q
            sc = score_candidates(model, tok, prompt_q, cands, ctx=ctx, device=device)
        pred = max(range(len(sc)), key=lambda i: sc[i])
        ok = pred == ci
        correct += ok
        others = [sc[i] for i in range(len(sc)) if i != ci]
        margins.append(sc[ci] - max(others))
        per_item.append({"q": q, "cands": cands, "correct_idx": ci, "scores": sc, "ok": bool(ok)})
    n = len(items)
    return {
        "tag": tag,
        "mode": mode,
        "acc": correct / n,
        "n": n,
        "mean_margin": sum(margins) / n,
        "items": per_item,
    }


# --------------------------------------------------------------------- MMLU RR
def load_mmlu(n=150, seed=0):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ds)), n))
    items = []
    for i in idx:
        row = ds[i]
        choices = list(row["choices"])
        items.append(
            {
                "q": row["question"],
                "choices": choices,
                "answer": choices[row["answer"]],
                "subject": row["subject"],
            }
        )
    return items


MMLU_LETTERS = ["A", "B", "C", "D"]


def eval_mmlu(model, tok, items, device="cuda"):
    correct = 0
    for item in items:
        body = item["q"] + "\n" + "\n".join(
            f"{MMLU_LETTERS[i]}. {c}" for i, c in enumerate(item["choices"])
        )
        q = body + "\nAnswer with a single letter (A, B, C or D)."
        sc = score_candidates(model, tok, q, list(MMLU_LETTERS), ctx=None, device=device)
        pred = MMLU_LETTERS[max(range(4), key=lambda i: sc[i])]
        correct += pred == MMLU_LETTERS[item["choices"].index(item["answer"])]
    return {"acc": correct / len(items), "n": len(items)}


# ---------------------------------------------------------------- rehearsal gen
def parse_pairs(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for obj in arr:
        if isinstance(obj, dict) and "q" in obj and "a" in obj:
            out.append({"q": str(obj["q"]).strip(), "g": str(obj["a"]).strip()})
    return out


META = re.compile(r"\b(document|docs|passage|context|text above|above|provided information|according to)\b", re.I)


def clean_questions(pairs, limit):
    seen, out = set(), []
    for p in pairs:
        q = p["q"].strip()
        if not q or len(q.split()) < 4 or META.search(q):
            continue
        key = re.sub(r"[^a-z0-9]+", "", q.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def cmd_gen(args):
    from transformers import AutoModelForCausalLM as AM

    tok = AutoTokenizer.from_pretrained(GENERATOR_ID)
    model = AM.from_pretrained(GENERATOR_ID, dtype=torch.bfloat16).to("cuda").eval()
    out = {}
    for arm in args.arms:
        prompt = GEN_PROMPTS[arm].format(n=args.per_call, ctx=CONTEXT)
        pairs, tries = [], 0
        while len(pairs) < args.n and tries < 8:
            tries += 1
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to("cuda")
            torch.manual_seed(1000 * tries + {"R": 1, "P": 2}.get(arm, 3))
            gen = model.generate(
                **enc, max_new_tokens=2200, do_sample=True, temperature=0.9, top_p=0.95,
                repetition_penalty=1.05,
            )
            raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            got = parse_pairs(raw)
            log(f"arm {arm} try {tries}: parsed {len(got)} pairs")
            pairs.extend(got)
        out[arm] = clean_questions(pairs, args.n)
        log(f"arm {arm}: kept {len(out[arm])} questions")
    dump(ART / "rehearsal_questions.json", out)
    del model
    torch.cuda.empty_cache()


def cmd_build(args):
    """Teacher answers (with context) for every rehearsal question."""
    tok, model = load_student()
    qs = load_json(ART / "rehearsal_questions.json")
    data = {}
    for arm, items in qs.items():
        rows = []
        for it in items:
            msgs = [
                {"role": "system", "content": SYSTEM_CTX.format(ctx=CONTEXT)},
                {"role": "user", "content": it["q"]},
            ]
            text = chat(tok, msgs, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=32, do_sample=False, temperature=None)
            ans = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rows.append({"q": it["q"], "gen_answer": it.get("g", ""), "teacher_answer": ans})
        data[arm] = rows
        log(f"built {arm}: {len(rows)} pairs")
        for r in rows[:3]:
            log("   Q: " + r["q"][:110] + "  ||  A: " + r["teacher_answer"][:60])
    dump(ART / "rehearsal_data.json", data)


# ------------------------------------------------------------------- training
def ffn_lora(model, r=8, alpha=16):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=["gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def build_batch(tok, rows, ctx, device):
    """Return (batch dict for teacher, batch dict for student, spans)."""
    t_ids, s_ids = [], []
    for r in rows:
        s_prefix = chat(tok, [{"role": "system", "content": SYSTEM_BASE},
                              {"role": "user", "content": r["q"]}], add_generation_prompt=True)
        t_msgs = [{"role": "system", "content": SYSTEM_CTX.format(ctx=ctx)},
                  {"role": "user", "content": r["q"]}]
        t_prefix = chat(tok, t_msgs, add_generation_prompt=True)
        s_enc = encode_with_span(tok, s_prefix, " " + r["teacher_answer"])
        t_enc = encode_with_span(tok, t_prefix, " " + r["teacher_answer"])
        if s_enc is None or t_enc is None:
            continue
        t_ids.append(t_enc)
        s_ids.append(s_enc)
    return t_ids, s_ids


def pad_batch(items, pad_id, device):
    maxlen = max(len(x[0]) for x in items)
    B = len(items)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    spans = []
    for i, (seq, s, e) in enumerate(items):
        shift = maxlen - len(seq)
        ids[i, shift:] = torch.tensor(seq)
        attn[i, shift:] = 1
        spans.append((s + shift, e + shift))
    return ids.to(device), attn.to(device), spans


@torch.no_grad()
def teacher_logits(model, ids, attn):
    with model.disable_adapter():
        out = model(input_ids=ids, attention_mask=attn)
    return out.logits.float()


def cmd_train(args):
    tok, model = load_student()
    data = load_json(ART / "rehearsal_data.json")
    if args.arm == "M":
        rows = data["R"][: args.per_arm] + data["P"][: args.per_arm]
    else:
        rows = data[args.arm][: args.n]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    model = ffn_lora(model, r=args.rank, alpha=2 * args.rank)
    model.to("cuda")
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    log(f"arm {args.arm} seed {args.seed}: {len(rows)} examples, {n_train/1e6:.3f}M trainable LoRA params")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    pad_id = tok.pad_token_id or tok.eos_token_id
    batch_size = args.batch
    steps_per_epoch = max(1, len(rows) // batch_size)
    total_steps = steps_per_epoch * args.epochs

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos"
    )

    step = 0
    losses = []
    for epoch in range(args.epochs):
        for b in range(steps_per_epoch):
            chunk = rows[b * batch_size : (b + 1) * batch_size]
            t_items, s_items = build_batch(tok, chunk, CONTEXT, "cuda")
            if not t_items:
                continue
            t_ids, t_attn, t_spans = pad_batch(t_items, pad_id, "cuda")
            s_ids, s_attn, s_spans = pad_batch(s_items, pad_id, "cuda")
            with torch.no_grad():
                t_logits = teacher_logits(model, t_ids, t_attn)

            if args.loss == "kl":
                out_s = model(input_ids=s_ids, attention_mask=s_attn)
                s_gathered = [out_s.logits[i, s - 1 : e - 1].float() for i, (s, e) in enumerate(s_spans)]
                t_gathered = [t_logits[i, s - 1 : e - 1].float() for i, (s, e) in enumerate(t_spans)]
                del out_s, t_logits
                loss, npos = 0.0, 0
                for s_g, t_g in zip(s_gathered, t_gathered):
                    for j in range(s_g.shape[0]):
                        p_t = F.softmax(t_g[j] / TEMPERATURE, dim=-1)
                        logp_s = F.log_softmax(s_g[j] / TEMPERATURE, dim=-1)
                        loss = loss + (p_t * (torch.log(p_t + 1e-12) - logp_s)).sum() * (TEMPERATURE ** 2)
                        npos += 1
                loss = loss / max(1, npos)
            else:  # cross entropy on the teacher-forced answer
                out_s = model(input_ids=s_ids, attention_mask=s_attn)
                s_gathered = [out_s.logits[i, s - 1 : e - 1].float() for i, (s, e) in enumerate(s_spans)]
                del out_s
                loss, npos = 0.0, 0
                for i, s_g in enumerate(s_gathered):
                    s, e = s_spans[i]
                    for j in range(s_g.shape[0]):
                        loss = loss - F.log_softmax(s_g[j], dim=-1)[s_ids[i, s + j]]
                        npos += 1
                loss = loss / max(1, npos)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % 20 == 0:
                log(f"  step {step}/{total_steps} loss {sum(losses[-20:])/20:.4f}")

    outdir = ART / f"lora_{args.arm}_{args.loss}_s{args.seed}"
    model.save_pretrained(outdir)
    dump(ART / f"train_{args.arm}_{args.loss}_s{args.seed}.json",
         {"arm": args.arm, "loss": args.loss, "seed": args.seed, "n_examples": len(rows),
          "trainable_params": n_train, "steps": total_steps, "final_loss": losses[-1],
          "mean_last20": sum(losses[-20:]) / 20, "questions": [r["q"] for r in rows]})
    log(f"saved {outdir}")


# ------------------------------------------------------------------- classify
CLASSIFY_PROMPT = (
    "Classify a question that was written about a software document.\n"
    "RECALL = the question asks what the document says or states (a value, name, rule or definition).\n"
    "USE = the question asks what to do in a situation: choosing, configuring, applying, debugging or deciding.\n"
    "Reply with exactly one word, RECALL or USE.\n\n"
    "Question: {q}\nLabel:"
)


def cmd_classify(args):
    from transformers import AutoModelForCausalLM as AM

    tok = AutoTokenizer.from_pretrained(GENERATOR_ID)
    model = AM.from_pretrained(GENERATOR_ID, dtype=torch.bfloat16).to("cuda").eval()
    data = load_json(ART / "rehearsal_data.json")
    out = {}
    for arm, rows in data.items():
        labels = []
        for r in rows:
            text = tok.apply_chat_template(
                [{"role": "user", "content": CLASSIFY_PROMPT.format(q=r["q"])}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=4, do_sample=False)
            raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            lab = "USE" if "USE" in raw.upper() else ("RECALL" if "RECALL" in raw.upper() else "?")
            labels.append(lab)
        out[arm] = labels
        n_use = sum(l == "USE" for l in labels)
        log(f"arm {arm}: {n_use}/{len(labels)} labelled USE (prospective), {len(labels)-n_use} RECALL")
        for r, l in list(zip(rows, labels))[:5]:
            log(f"   [{l}] {r['q'][:100]}")
    dump(ART / "rehearsal_labels.json", out)
    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------------------- diagnose
@torch.no_grad()
def answer_logp(model, tok, q, answer, ctx=None, device="cuda"):
    """Mean log P(answer tokens | q, system) per token."""
    sysmsg = SYSTEM_CTX.format(ctx=ctx) if ctx else SYSTEM_BASE
    prefix = chat(tok, [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}],
                  add_generation_prompt=True)
    enc = encode_with_span(tok, prefix, " " + answer)
    if enc is None:
        return float("nan")
    ids, s, e = enc
    inp = torch.tensor([ids], device=device)
    logits = model(input_ids=inp).logits.float()
    logp = F.log_softmax(logits[0], dim=-1)
    tot = sum(logp[j - 1, ids[j]].item() for j in range(s, e))
    return tot / max(1, e - s)


def cmd_diag(args):
    """Does an adapter trained on one question kind also cover the other kind?"""
    from peft import PeftModel

    tok, model = load_student()
    data = load_json(ART / "rehearsal_data.json")
    out = {}
    for arm in ("R", "P"):
        for q_arm in ("R", "P"):
            rows = data[q_arm]
            v_t = [answer_logp(model, tok, r["q"], r["teacher_answer"], ctx=CONTEXT) for r in rows]
            v_0 = [answer_logp(model, tok, r["q"], r["teacher_answer"]) for r in rows]
            out[f"vanilla/{arm}qs_teacherctx/{q_arm}"] = sum(v_t) / len(v_t)
            out[f"vanilla/{arm}qs_nocx/{q_arm}"] = sum(v_0) / len(v_0)
        log(f"vanilla: {arm}-questions logp teacher {out[f'vanilla/{arm}qs_teacherctx/{arm}']:.3f} "
            f"no-ctx {out[f'vanilla/{arm}qs_nocx/{arm}']:.3f}")

    for arm in ("R", "P"):
        path = ART / f"lora_{arm}_kl_s0"
        m = PeftModel.from_pretrained(model, path, adapter_name="diag")
        m.eval()
        for q_arm in ("R", "P"):
            rows = data[q_arm]
            lp = [answer_logp(m, tok, r["q"], r["teacher_answer"]) for r in rows]
            mean = sum(lp) / len(lp)
            ctx_lp = out[f"vanilla/{arm}qs_teacherctx/{q_arm}"]
            base_lp = out[f"vanilla/{arm}qs_nocx/{q_arm}"]
            out[f"{arm}/{arm}qs_nocx/{q_arm}"] = mean
            out[f"{arm}/recovery_on_{q_arm}"] = (mean - base_lp) / (ctx_lp - base_lp)
            log(f"adapter {arm} on {q_arm}-questions: logp {mean:.3f} "
                f"(base {base_lp:.3f}, teacher {ctx_lp:.3f}) -> recovery "
                f"{out[f'{arm}/recovery_on_{q_arm}']:.3f}")
        m.delete_adapter("diag")
    dump(ART / "diag_cross.json", out)


# ------------------------------------------------------------------- evaluate
def eval_all_sets(model, tok, label, mmlu=None, device="cuda", shuffles=(0, 1)):
    """Both readouts x both option shuffles; returns a flat dict of results."""
    res = {}
    for kind, items in EVAL_SETS.items():
        for mode in ("free", "letter"):
            accs, margins, first = [], [], None
            for sh in shuffles:
                r = eval_set(model, tok, items, ctx=None, seed=sh, tag=label, mode=mode)
                accs.append(r["acc"])
                margins.append(r["mean_margin"])
                first = first or r
            res[f"{label}/{kind}/{mode}"] = {
                "acc": sum(accs) / len(accs),
                "acc_per_shuffle": accs,
                "mean_margin": sum(margins) / len(margins),
                "n": first["n"],
                "items": first["items"],
            }
    if mmlu is not None:
        r = eval_mmlu(model, tok, mmlu)
        r["tag"] = label
        res[f"{label}/mmlu"] = r
    return res


def cmd_eval(args):
    from peft import PeftModel

    tok, model = load_student()
    results = {}

    log("reference: vanilla (no context)")
    results.update(eval_all_sets(model, tok, "vanilla"))
    log("reference: with context")
    for kind, items in EVAL_SETS.items():
        for mode in ("free", "letter"):
            accs, margins, first = [], [], None
            for sh in (0, 1):
                r = eval_set(model, tok, items, ctx=CONTEXT, seed=sh, tag="ctx", mode=mode)
                accs.append(r["acc"])
                margins.append(r["mean_margin"])
                first = first or r
            results[f"ctx/{kind}/{mode}"] = {
                "acc": sum(accs) / len(accs), "acc_per_shuffle": accs,
                "mean_margin": sum(margins) / len(margins), "n": first["n"],
                "items": first["items"],
            }
        log(f"  ctx {kind}: free {results[f'ctx/{kind}/free']['acc']:.3f} "
            f"letter {results[f'ctx/{kind}/letter']['acc']:.3f}")

    mmlu = load_mmlu(n=args.mmlu_n, seed=7)
    results["vanilla/mmlu"] = eval_mmlu(model, tok, mmlu)
    log(f"  vanilla mmlu: {results['vanilla/mmlu']['acc']:.3f} (n={len(mmlu)})")

    for run in args.runs:
        arm, loss, seed = run.split(":")
        adapter = ART / f"lora_{arm}_{loss}_s{seed}"
        log(f"evaluating {adapter.name}")
        m = PeftModel.from_pretrained(model, adapter, adapter_name="probe")
        m.eval()
        res = eval_all_sets(m, tok, f"{arm}-{loss}-s{seed}", mmlu=mmlu)
        for kind in EVAL_SETS:
            for mode in ("free", "letter"):
                r = res[f"{arm}-{loss}-s{seed}/{kind}/{mode}"]
                log(f"  {kind:8s} {mode:6s} acc {r['acc']:.3f} margin {r['mean_margin']:.3f}")
        log(f"  mmlu {res[f'{arm}-{loss}-s{seed}/mmlu']['acc']:.3f}")
        results.update(res)
        m.delete_adapter("probe")

    dump(ART / f"eval_{args.tag}.json", results)


# --------------------------------------------------------------------- report
def kendall_tau(a, b):
    """tau-a over pairs; a, b are equal-length score vectors."""
    num = den = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = a[i] - a[j]
            s2 = b[i] - b[j]
            if s1 == 0 or s2 == 0:
                continue
            den += 1
            num += 1 if (s1 > 0) == (s2 > 0) else -1
    return num / den if den else float("nan")


def shift_agreement(res, ref_items_key, tag):
    """Mean Kendall tau between a model's candidate scores and the teacher's
    (context-conditioned) candidate scores, per item, on the held-out sets."""
    out = {}
    for kind in EVAL_SETS:
        for mode in ("free", "letter"):
            key = f"{tag}/{kind}/{mode}"
            if key not in res:
                continue
            taus = []
            for it_a, it_t in zip(res[key]["items"], ref_items_key[f"ctx/{kind}/{mode}"]["items"]):
                taus.append(kendall_tau(it_a["scores"], it_t["scores"]))
            out[f"{kind}/{mode}"] = sum(taus) / len(taus)
    return out


def cmd_report(args):
    import glob

    runs = {}
    for path in sorted(glob.glob(str(ART / "eval_*.json"))):
        tag = Path(path).stem.replace("eval_", "")
        runs[tag] = load_json(path)

    summary = {"runs": {}, "reference": {}}
    ref = runs[sorted(runs)[0]]
    for kind in EVAL_SETS:
        for mode in ("free", "letter"):
            summary["reference"][f"{kind}/{mode}"] = {
                "vanilla": ref[f"vanilla/{kind}/{mode}"]["acc"],
                "ctx": ref[f"ctx/{kind}/{mode}"]["acc"],
                "gap": ref[f"ctx/{kind}/{mode}"]["acc"] - ref[f"vanilla/{kind}/{mode}"]["acc"],
                "vanilla_margin": ref[f"vanilla/{kind}/{mode}"]["mean_margin"],
            }
    summary["reference"]["mmlu"] = ref["vanilla/mmlu"]["acc"]
    # teacher-agreement baselines (Kendall tau on candidate-score vectors)
    summary["reference"]["tau"] = {
        "vanilla": shift_agreement(ref, ref, "vanilla"),
        "ctx": shift_agreement(ref, ref, "ctx"),
    }

    for tag, res in runs.items():
        run_ids = sorted({k.split("/")[0] for k in res if not k.startswith(("vanilla/", "ctx/"))})
        for run_id in run_ids:
            entry = summary["runs"].setdefault(run_id, {"seeds": {}})
            for key, val in res.items():
                if not key.startswith(run_id + "/"):
                    continue
                parts = key.split("/")
                if len(parts) == 2:
                    entry["RR"] = val["acc"] / ref["vanilla/mmlu"]["acc"]
                    continue
                _, kind, mode = parts
                v0 = ref[f"vanilla/{kind}/{mode}"]["acc"]
                vc = ref[f"ctx/{kind}/{mode}"]["acc"]
                entry[f"{kind}/{mode}"] = {
                    "acc": val["acc"],
                    "margin": val["mean_margin"],
                    "CR": (val["acc"] - v0) / (vc - v0) if vc != v0 else float("nan"),
                    "delta_margin": val["mean_margin"] - ref[f"vanilla/{kind}/{mode}"]["mean_margin"],
                }
            sub = {k: v for k, v in res.items()
                   if k.startswith(("vanilla/", "ctx/")) or k.startswith(run_id + "/")}
            tau = shift_agreement(sub, ref, run_id)
            for k, t in tau.items():
                tv = summary["reference"]["tau"]["vanilla"][k]
                tc = summary["reference"]["tau"]["ctx"][k]
                entry[k]["tau"] = t
                entry[k]["CR_tau"] = (t - tv) / (tc - tv) if tc != tv else float("nan")
    # arm-level means over seeds
    arms = {}
    for run_id, entry in summary["runs"].items():
        a = "-".join(run_id.split("-")[:2])
        entry["arm"] = a
        arms.setdefault(a, []).append(entry)
    grouped = {}
    for a, entries in arms.items():
        g = {"n_runs": len(entries)}
        for k in ("restate/free", "use/free", "restate/letter", "use/letter"):
            g[k + "_CR"] = sum(e[k]["CR"] for e in entries) / len(entries)
            g[k + "_CR_tau"] = sum(e[k]["CR_tau"] for e in entries) / len(entries)
            g[k + "_acc"] = sum(e[k]["acc"] for e in entries) / len(entries)
            g[k + "_margin"] = sum(e[k]["margin"] for e in entries) / len(entries)
            g[k + "_tau"] = sum(e[k]["tau"] for e in entries) / len(entries)
        g["RR"] = sum(e.get("RR", float("nan")) for e in entries) / len(entries)
        grouped[a] = g
    summary["arms"] = grouped
    dump(ART / "summary.json", summary)
    for a, g in grouped.items():
        log(f"{a} (n={g['n_runs']}): "
            f"CR_tau use {100*g['use/letter_CR_tau']:.1f}% restate {100*g['restate/letter_CR_tau']:.1f}% | "
            f"CR_acc(free) use {100*g['use/free_CR']:.1f}% restate {100*g['restate/free_CR']:.1f}% | RR {100*g['RR']:.1f}%")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--arms", nargs="+", default=["R", "P"])
    g.add_argument("--n", type=int, default=32)
    g.add_argument("--per_call", type=int, default=14)
    g.set_defaults(func=cmd_gen)

    b = sub.add_parser("build")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("classify")
    c.set_defaults(func=cmd_classify)

    d = sub.add_parser("diag")
    d.set_defaults(func=cmd_diag)

    t = sub.add_parser("train")
    t.add_argument("--arm", required=True)
    t.add_argument("--loss", default="kl", choices=["kl", "ce"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n", type=int, default=32)
    t.add_argument("--per_arm", type=int, default=16)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-4)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--tag", required=True)
    e.add_argument("--mmlu_n", type=int, default=150)
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("report")
    r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
