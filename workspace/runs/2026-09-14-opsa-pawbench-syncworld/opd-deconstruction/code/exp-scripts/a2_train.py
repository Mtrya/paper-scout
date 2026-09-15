#!/usr/bin/env python3
"""A2: OPD mechanism check — OPD vs fixed-negative vs OPSA vs fixed-positive.

Hand-rolled policy-gradient loop (no Megatron/SGLang). Student rollouts are
scored token-wise; advantages by condition:
  opd      : A_i = log pi_t - log pi_s on all response tokens
  fixed-neg: A_i = -0.5 on the 20% lowest student-logp tokens, 0 elsewhere
  opsa     : A_i = -0.5 - (H_i-H_min)/(2(H_max-H_min)) on the 20% lowest
  fixed-pos: A_i = +0.2 on the 20% lowest (collapse control)
Loss = -mean(A_i * log pi_s(y_i)) over selected positions (detached A).
Eval every N steps: avg@4 on a fixed eval subset.

Usage:
  python a2_train.py --condition opsa --student Qwen/Qwen3-0.6B \
      --teacher Qwen/Qwen3-1.7B --steps 80 --out results/a2_opsa
"""
import argparse, json, os, random, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import verify_answer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load(name, grad=False, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2", trust_remote_code=True)
    model.eval()
    if not grad:
        for p in model.parameters():
            p.requires_grad_(False)
    if hasattr(model.generation_config, "enable_thinking"):
        model.generation_config.enable_thinking = False
    return model, tok


def chat_text(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


def rollouts(model, tok, prompts, max_new=2048, batch=4, temp=1.0,
             top_k=0, top_p=1.0):
    """Generate one response per prompt; returns list of (text, token_ids)."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        ids = [tok(p, add_special_tokens=False).input_ids for p in chunk]
        padded, masks = pad_left(ids, tok.pad_token_id or tok.eos_token_id)
        L = padded.size(1)
        with torch.no_grad():
            gen = model.generate(
                padded.to(model.device), attention_mask=masks.to(model.device),
                max_new_tokens=max_new, do_sample=True, temperature=temp,
                top_k=top_k, top_p=top_p,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for g in gen:
            resp = g[L:].tolist()   # left-padded batch: response starts after pad len
            out.append((tok.decode(resp, skip_special_tokens=True), resp))
    return out


def pad_left(list_of_ids, pad_id):
    L = max(len(x) for x in list_of_ids)
    padded, masks = [], []
    for x in list_of_ids:
        p = [pad_id] * (L - len(x)) + x
        m = [0] * (L - len(x)) + [1] * len(x)
        padded.append(p)
        masks.append(m)
    return torch.tensor(padded), torch.tensor(masks)


def batch_forward(model, tok, prompt_texts, resp_ids_list):
    """Forward full prompt+response, return per-token logp at sampled token,
    entropy at each response position, and logits positions. Returns tensors
    packed per sample with response-length masks."""
    all_logp, all_ent, all_lens = [], [], []
    for i in range(0, len(prompt_texts), 2):
        chunk_p = prompt_texts[i:i + 2]
        chunk_r = resp_ids_list[i:i + 2]
        pids = [tok(p, add_special_tokens=False).input_ids for p in chunk_p]
        seqs = [p + r for p, r in zip(pids, chunk_r)]
        padded, masks = pad_left(seqs, tok.pad_token_id or tok.eos_token_id)
        with torch.no_grad():
            out = model(padded.to(model.device),
                        attention_mask=masks.to(model.device))
        logits = out.logits.float()  # (B, L, V)
        logz = torch.logsumexp(logits, dim=-1)           # (B, L)
        L = padded.size(1)
        for b, (p, r) in enumerate(zip(pids, chunk_r)):
            Lp, Lr = len(p), len(r)
            if Lr == 0:
                continue
            start = (L - (Lp + Lr)) + Lp - 1              # pad-aware first resp pos
            idx = torch.tensor(r, device=logits.device)
            tok_logit = logits[b, start:start + Lr].gather(1, idx[:, None]).squeeze(1)
            sampled_logp = tok_logit - logz[b, start:start + Lr]
            pos_logp = logits[b, start:start + Lr] - logz[b, start:start + Lr, None]
            ent = -(pos_logp.exp() * pos_logp).sum(-1)
            all_logp.append(sampled_logp.cpu())
            all_ent.append(ent.cpu())
            all_lens.append(Lr)
    return all_logp, all_ent, all_lens


def select_lowest_global(logp_list, frac=0.2):
    """Official OPSA semantics: concat all valid tokens across the batch,
    take floor(frac*N) lowest-logp tokens globally. Returns per-sample index
    lists."""
    flat = torch.cat([lp for lp in logp_list if len(lp) > 0])
    n = flat.numel()
    if n == 0:
        return [[] for _ in logp_list]
    k = max(1, int(frac * n))
    k = min(k, n)
    _, order = torch.sort(flat, stable=True)
    sel = set(order[:k].tolist())
    positions, offset = [], 0
    for lp in logp_list:
        m = len(lp)
        positions.append([i - offset for i in sel if offset <= i < offset + m])
        offset += m
    return positions


def train_step(student, teacher, tok, prompts, condition, opt, frac=0.2,
               grad_clip=1.0, max_new=2048, batch=4):
    """One OPD-style training step. Returns metrics dict. Optimizer state is
    held by the caller so it persists across steps."""
    # 1. rollout (chat-template the message lists first; rollouts() takes text)
    ptexts = [chat_text(tok, m) for m in prompts]
    texts_ids = rollouts(student, tok, ptexts, max_new=max_new, batch=batch)
    texts = [t for t, _ in texts_ids]
    resp_ids = [r for _, r in texts_ids]
    # 2. student logp & entropy over own rollouts
    s_logp, s_ent, s_lens = batch_forward(student, tok, ptexts, resp_ids)
    # 3. teacher logp (opd / opd-oneshot)
    t_logp = None
    if condition in ("opd", "opd-oneshot"):
        t_logp, _, _ = batch_forward(teacher, tok, ptexts, resp_ids)
    # 4. per-sample advantages (tensors with same lengths)
    advs = []
    neg_fracs = []
    global_pos = None
    if condition not in ("opd", "opd-oneshot"):
        global_pos = select_lowest_global(s_logp, frac)
    for b in range(len(s_logp)):
        lp = s_logp[b]
        ent = s_ent[b]
        if condition in ("opd", "opd-oneshot"):
            a = t_logp[b] - lp
            advs.append(a)
            if len(a):
                neg_fracs.append(float((a < 0).float().mean().item()))
        else:
            a = torch.zeros_like(lp)
            pos = global_pos[b]
            if condition == "fixed-neg":
                a[pos] = -0.5
            elif condition == "fixed-pos":
                a[pos] = 0.2
            elif condition == "opsa":
                if len(pos) == 0:      # sample with no valid response tokens
                    advs.append(a)
                    continue
                H = ent[pos]
                Hmin, Hmax = H.min(), H.max()
                if Hmax - Hmin < 1e-8:
                    a[pos] = -0.5
                else:
                    a[pos] = -0.5 - (H - Hmin) / (2 * (Hmax - Hmin))
            advs.append(a)
    # 5. loss = -mean(A_i * logp_i) over selected (nonzero-advantage) positions
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)
    opt.zero_grad()
    total, count, losses = torch.tensor(0.0, device=student.device), 0, []
    for i in range(0, len(prompts), 2):
        chunk_p = prompts[i:i + 2]
        chunk_ptext = [chat_text(tok, m) for m in chunk_p]
        chunk_r = resp_ids[i:i + 2]
        chunk_a = advs[i:i + 2]
        pids = [tok(p, add_special_tokens=False).input_ids for p in chunk_ptext]
        seqs = [p + r for p, r in zip(pids, chunk_r)]
        padded, masks = pad_left(seqs, tok.pad_token_id or tok.eos_token_id)
        out = student(padded.to(student.device), attention_mask=masks.to(student.device))
        logp_all = torch.log_softmax(out.logits.float(), dim=-1)
        chunk_loss = None
        for b, (p, r, a) in enumerate(zip(pids, chunk_r, chunk_a)):
            Lp, Lr = len(p), len(r)
            if Lr == 0:
                continue
            sel = (a != 0).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            pos_logp = logp_all[b, Lp - 1: Lp - 1 + Lr]
            lp_sel = pos_logp[sel, torch.tensor([r[i] for i in sel.tolist()],
                                                device=student.device)]
            loss = -(a[sel].to(student.device) * lp_sel).sum() / len(sel)
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            total = total + loss.detach()
            count += len(sel)
        if chunk_loss is not None:
            (chunk_loss / len(prompts)).backward()  # once per chunk: the graph is per-chunk
    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
    opt.step()
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return {
        "n_tokens_selected": count,
        "loss": float(total.item()),
        "mean_resp_len": float(sum(s_lens) / max(1, len(s_lens))),
        "mean_abs_adv": float(torch.cat([a.abs() for a in advs]).mean().item()) if advs and all(len(a) for a in advs) else 0.0,
        "mean_entropy": float(torch.cat([e for e in s_ent]).mean().item()) if s_ent and all(len(e) for e in s_ent) else 0.0,
        "neg_adv_frac": float(sum(neg_fracs) / max(1, len(neg_fracs))),
    }


@torch.no_grad()
def evaluate(model, tok, prompts, labels, k=4, max_new=4096, temp=0.7,
             top_k=20, top_p=0.8):
    correct = 0
    total = 0
    for msgs, lab in zip(prompts, labels):
        ptext = chat_text(tok, msgs)
        for _ in range(k):
            ids = tok(ptext, add_special_tokens=False).input_ids
            out = model.generate(torch.tensor([ids], device=model.device),
                                 max_new_tokens=max_new, do_sample=True,
                                 temperature=temp, top_k=top_k, top_p=top_p,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(out[0][len(ids):].tolist(), skip_special_tokens=True)
            if verify_answer(text, lab):
                correct += 1
                break
        total += 1
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["opd", "opd-oneshot", "fixed-neg", "opsa",
                             "fixed-pos"])
    ap.add_argument("--student", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--teacher", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train-data", default="dapo-math-17k.jsonl")
    ap.add_argument("--eval-data", default="math500-subset.jsonl")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/a2_{args.condition}"
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.train_data) as f:
        rows = [json.loads(l) for l in f]
    random.shuffle(rows)
    train_rows = [(d["prompt"], str(d["label"])) for d in rows[:args.n_train]]
    if args.condition == "opd-oneshot":
        train_rows = [train_rows[0]]  # single query, repeated every step
    with open(args.eval_data) as f:
        erows = [json.loads(l) for l in f]
    eval_rows = [(d["prompt"], str(d["label"])) for d in erows[:args.n_eval]]
    eval_prompts = [p for p, _ in eval_rows]
    eval_labels = [l for _, l in eval_rows]

    student, tok = load(args.student)
    teacher = None
    if args.condition in ("opd", "opd-oneshot"):
        teacher, _ = load(args.teacher)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-6)

    log = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        idx = [(step - 1) * args.batch + i for i in range(args.batch)]
        prompts = [train_rows[i % len(train_rows)][0] for i in idx]
        t0 = time.time()
        metrics = train_step(student, teacher, tok, prompts, args.condition,
                             opt, frac=args.frac, max_new=args.max_new,
                             batch=args.batch)
        metrics["step"] = step
        metrics["secs"] = time.time() - t0
        if step % args.eval_every == 0 or step == args.steps:
            acc = evaluate(student, tok, eval_prompts, eval_labels)
            metrics["eval_avg4"] = acc
            torch.save(student.state_dict(), f"{args.out}/ckpt_{step}.pt")
            print(f"[{time.time()-t_start:.0f}s] step {step}: {metrics}", flush=True)
        else:
            print(f"[{time.time()-t_start:.0f}s] step {step}: {metrics}", flush=True)
        log.append(metrics)
        with open(f"{args.out}/log.json", "w") as f:
            json.dump(log, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
