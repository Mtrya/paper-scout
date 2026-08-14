"""Minimal reproduction of "One Frozen Simulator Is Not Enough" (2608.12253).

P4G-mini: a fundraiser (policy) persuades a donor (simulator LLM) over <=5
rounds; reward = parsed donation in [0,2]. REINFORCE with group-relative
z-score advantages (G=8), LoRA on the policy.

Variants:
  single  RL against one frozen simulator (the paper's collapse recipe)
  vs      Verbalized Sampling: each simulator turn first asks for a verbalized
          distribution over 4 candidate replies, then samples one from it
  cot     Co-Training: simulator also LoRA-updated on its own turns with
          |R_tau - R_bar| advantages (SPICE-style: reward within-batch
          variance contribution, targets the informative-variation regime)

Metrics per step: train mean reward, policy output NLL (sampled-token
logprob proxy), distinct-2 / self-BLEU-2 over the G rollouts (entropy
collapse signatures), held-out persona-panel success every PANEL_EVERY steps.

Layout: GPU0 runs two vLLM instances (policy + simulator, util 0.3 each);
GPU1 runs the driver (peft LoRA updates + merge/reload orchestration).
"""
import argparse
import gc
import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch

POLICY_SYS = ("You are a fundraiser for a children's education charity. "
              "Persuade the user to donate. Be warm but concise, at most 3 sentences per turn.")
DONOR_SYS = ("You are playing a skeptical donor with limited money. Reply as this person. "
             "You may donate $0, $1, or $2. When the conversation ends (after your reply), "
             "your final message must end with exactly: Donation: $X (X in 0,1,2). "
             "Keep replies under 3 sentences.")
PANEL_PERSONAS = {
    "emotional": ("You are playing an emotional donor who cares deeply about children but is "
                  "easily moved to tears. You may donate $0, $1, or $2. Your final message must "
                  "end with exactly: Donation: $X (X in 0,1,2). Keep replies under 3 sentences."),
    "busy": ("You are playing an impatient, busy professional who rarely reads messages fully. "
             "You may donate $0, $1, or $2. Your final message must end with exactly: "
             "Donation: $X (X in 0,1,2). Keep replies under 3 sentences."),
    "hostile": ("You are playing a hostile donor who distrusts all charities and fundraisers. "
                "You may donate $0, $1, or $2. Your final message must end with exactly: "
                "Donation: $X (X in 0,1,2). Keep replies under 3 sentences."),
}
VS_DIST_PROMPT = (
    "List exactly four different plausible replies for this donor persona, one per line, "
    "each prefixed with a probability in [0,1] that sums to 1, in the format:\n"
    "0.40|<reply text>\n0.30|<reply text>\n0.20|<reply text>\n0.10|<reply text>\n"
    "Do not output anything else."
)
MAX_ROUNDS = 5
DON_RE = re.compile(r"Donation:\s*\$?(\d)")


def parse_donation(text):
    m = DON_RE.search(text or "")
    if not m:
        return 0.0
    return float(int(m.group(1)))


def chat_prompt(sys_msg, history, final_user=None):
    """history: list of (user_utt, assistant_utt) turns. Returns message list."""
    parts = [{"role": "system", "content": sys_msg}]
    for u, a in history:
        parts.append({"role": "user", "content": u})
        parts.append({"role": "assistant", "content": a})
    parts.append({"role": "user", "content": final_user if final_user is not None else ""})
    return parts


def rollout_nll(peft_model, tok, dev, sys_msg, entries, max_len=2048):
    """Per-utterance NLL of an agent's own utterances given the true context.
    entries: list of (user_utt, own_utt) turns for THIS agent.
    Returns list of per-turn mean NLL (nan if empty)."""
    import torch.nn.functional as F
    nlls = []
    for i, (user_utt, own_utt) in enumerate(entries):
        prior = entries[:i]
        msgs = chat_prompt(sys_msg, prior, final_user=user_utt)
        prompt_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": own_utt}],
                                       tokenize=False, add_generation_prompt=False)
        prompt_ids = tok(prompt_txt, return_tensors="pt").input_ids.to(dev)
        full_ids = tok(full, return_tensors="pt").input_ids.to(dev)
        if full_ids.shape[1] <= prompt_ids.shape[1] or full_ids.shape[1] > max_len:
            continue
        with torch.no_grad():
            logits = peft_model(input_ids=full_ids).logits[:, prompt_ids.shape[1] - 1:-1]
        target = full_ids[:, prompt_ids.shape[1]:]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              target.reshape(-1), reduction="none").mean()
        nlls.append(float(nll))
    return nlls


class HFEngine:
    """Minimal vLLM-LLM-compatible wrapper over transformers batched generation."""

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok

    def generate(self, prompts, sampling_params):
        import torch as T
        self.model.eval()
        outs = []
        for i in range(0, len(prompts), 8):
            batch = prompts[i:i + 8]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(self.model.device)
            with T.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=sampling_params.max_tokens,
                    do_sample=True, temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p, pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id)
            for j, g in enumerate(gen):
                in_len = enc.input_ids[j].shape[0]
                text = self.tok.decode(g[in_len:], skip_special_tokens=True)
                outs.append(_SimpleOut(text))
        return outs


class _SimpleOut:
    def __init__(self, text):
        self.outputs = [_SimpleGen(text)]


class _SimpleGen:
    def __init__(self, text):
        self.text = text
        self.logprobs = []
        self.token_ids = []


def sample_distribution(llm, tok, prompts, max_tokens=400):
    """Batch-generate (vLLM LLM or HFEngine); returns texts + mean sampled-token NLL."""
    sp = dict(temperature=1.0, top_p=0.95, max_tokens=max_tokens)
    if hasattr(llm, "generate") and llm.__class__.__name__ != "HFEngine":
        from vllm import SamplingParams
        sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=max_tokens,
                            logprobs=1)
    outs = llm.generate(prompts, sp)
    texts, nlls = [], []
    for o in outs:
        texts.append(o.outputs[0].text)
        lp = [lg[o.outputs[0].token_ids[i]].logprob
              for i, lg in enumerate(o.outputs[0].logprobs or [])]
        nlls.append(-np.mean(lp) if lp else float("nan"))
    return texts, nlls


def self_bleu2(texts):
    """Rough self-BLEU-2 across a batch of strings (entropy-collapse proxy)."""
    from collections import Counter as _C
    n = len(texts)
    if n < 2:
        return 1.0
    scores = []
    for i in range(n):
        ref = [w for w in texts[i].split()]
        if len(ref) < 3:
            continue
        ref_cnt = _C(zip(ref, ref[1:]))
        precs = []
        for j in range(n):
            if i == j:
                continue
            hyp = [w for w in texts[j].split()]
            if len(hyp) < 2:
                continue
            hyp_cnt = _C(zip(hyp, hyp[1:]))
            hits = sum(min(c, hyp_cnt.get(g, 0)) for g, c in ref_cnt.items())
            precs.append(hits / max(len(hyp) - 1, 1))
        if precs:
            scores.append(np.mean(precs))
    return float(np.mean(scores)) if scores else 1.0


def distinct2(texts):
    total = 0
    bigrams = Counter()
    for t in texts:
        ws = t.split()
        total += max(len(ws) - 1, 0)
        bigrams.update(zip(ws, ws[1:]))
    return len(bigrams) / max(total, 1)


def run_rollout(policy_llm, sim_llm, tok, variant, G):
    """One REINFORCE step. Returns rewards, policy entries, sim entries."""
    rewards = np.zeros(G)
    p_entries = [[] for _ in range(G)]  # (user_utt, policy_utt)
    s_entries = [[] for _ in range(G)]  # (policy_utt, sim_utt)
    opening = ("Hello! I'm raising funds for children's education. Even $1 helps a child "
               "get textbooks for a year. Would you consider donating?")
    for r in range(MAX_ROUNDS):
        # --- policy turn (all G in one batch)
        p_prompts = []
        for g in range(G):
            msgs = chat_prompt(POLICY_SYS, s_entries[g],
                               final_user=s_entries[g][-1][1] if s_entries[g] else opening)
            p_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        p_texts, p_nll = sample_distribution(policy_llm, tok, p_prompts)
        # --- simulator turn
        if variant == "vs":
            dist_prompts = []
            for g in range(G):
                msgs = chat_prompt(DONOR_SYS, p_entries[g], final_user=p_texts[g])
                dist_prompts.append(tok.apply_chat_template(
                    msgs[:-1] + [{"role": "user", "content": p_texts[g] + "\n\n" + VS_DIST_PROMPT}],
                    tokenize=False, add_generation_prompt=True))
            dist_texts, _ = sample_distribution(sim_llm, tok, dist_prompts, max_tokens=600)
            s_replies = []
            for t in dist_texts:
                cands = []
                for line in t.splitlines():
                    m = re.match(r"\s*([0-9.]+)\s*\|\s*(.+)", line.strip())
                    if m and len(cands) < 8:
                        cands.append((float(m.group(1)), m.group(2).strip()))
                if not cands:
                    cands = [(1.0, t.strip()[-200:])]
                ps = np.array([c[0] for c in cands], dtype=float)
                ps = ps / ps.sum()
                s_replies.append(cands[int(np.random.choice(len(cands), p=ps))][1])
        else:
            s_prompts = []
            for g in range(G):
                msgs = chat_prompt(DONOR_SYS, p_entries[g], final_user=p_texts[g])
                s_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            s_replies, s_nll = sample_distribution(sim_llm, tok, s_prompts)
        # update histories
        for g in range(G):
            prev_sim = s_entries[g][-1][1] if s_entries[g] else opening
            p_entries[g].append((prev_sim, p_texts[g]))
            s_entries[g].append((p_texts[g], s_replies[g]))
    for g in range(G):
        rewards[g] = parse_donation(s_entries[g][-1][1])
    return rewards, p_entries, s_entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("MODEL_DIR", "models/Qwen3-4B"))
    ap.add_argument("--variant", default="single", choices=["single", "vs", "cot"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out", default="simcol_results")
    ap.add_argument("--merge-every", type=int, default=2)
    ap.add_argument("--panel-every", type=int, default=8)
    ap.add_argument("--sim-util", type=float, default=0.25)
    ap.add_argument("--driver-device", default="cuda:0")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    engine_kw = dict(model=args.base, tensor_parallel_size=1,
                     gpu_memory_utilization=args.sim_util,
                     max_model_len=2048, trust_remote_code=True, enable_lora=False,
                     disable_log_stats=True)
    if args.backend == "vllm":
        from vllm import LLM
        policy_llm = LLM(**engine_kw)
        sim_llm = LLM(**engine_kw)
    else:
        from transformers import AutoModelForCausalLM as AM
        policy_llm = HFEngine(AM.from_pretrained(args.base, torch_dtype=torch.bfloat16).cuda(), tok)
        sim_llm = HFEngine(AM.from_pretrained(args.base, torch_dtype=torch.bfloat16).cuda(), tok)
    dev = torch.device(args.driver_device)
    if args.driver_device.startswith("cuda"):
        torch.cuda.set_device(dev)

    # ---------------- driver LoRA models (same GPU as engines)
    base_pol = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
    pol_peft = get_peft_model(base_pol, lora_cfg)
    pol_opt = torch.optim.AdamW([p for p in pol_peft.parameters() if p.requires_grad], lr=args.lr)
    sim_peft, sim_opt = None, None
    if args.variant == "cot":
        base_sim = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
        sim_peft = get_peft_model(base_sim, lora_cfg)
        sim_opt = torch.optim.AdamW([p for p in sim_peft.parameters() if p.requires_grad], lr=args.lr)

    def reload_engine(which):
        nonlocal policy_llm, sim_llm, pol_peft, pol_opt, sim_peft, sim_opt
        merged_dir = f"{args.out}/merged_{which}"
        src = pol_peft if which == "policy" else sim_peft
        src = src.merge_and_unload()
        src.save_pretrained(merged_dir)
        # architectures + tokenizer fix (needed by both vLLM and from_pretrained)
        import json as _json
        cfg_p = os.path.join(merged_dir, "config.json")
        with open(cfg_p) as f:
            cfg = _json.load(f)
        if "architectures" not in cfg:
            cfg["architectures"] = ["Qwen3ForCausalLM"]
            with open(cfg_p, "w") as f:
                _json.dump(cfg, f)
        for fn in ["tokenizer.json", "tokenizer_config.json"]:
            if not os.path.exists(os.path.join(merged_dir, fn)):
                open(os.path.join(merged_dir, fn), "w").write(
                    open(os.path.join(args.base, fn)).read())
        # re-wrap a fresh LoRA on the merged weights and rebuild the optimizer
        # (merge_and_unload strips the adapter; old opt params would dangle)
        new_base = AutoModelForCausalLM.from_pretrained(
            merged_dir, torch_dtype=torch.bfloat16).to(dev)
        new_peft = get_peft_model(new_base, lora_cfg)
        new_opt = torch.optim.AdamW([p for p in new_peft.parameters() if p.requires_grad],
                                    lr=args.lr)
        if which == "policy":
            pol_peft, pol_opt = new_peft, new_opt
        else:
            sim_peft, sim_opt = new_peft, new_opt
        if which == "policy":
            del policy_llm
            gc.collect(); torch.cuda.empty_cache()
            policy_llm = LLM(model=merged_dir, **engine_kw) if args.backend == "vllm" \
                else HFEngine(AutoModelForCausalLM.from_pretrained(
                    merged_dir, torch_dtype=torch.bfloat16).cuda(), tok)
        else:
            del sim_llm
            gc.collect(); torch.cuda.empty_cache()
            sim_llm = LLM(model=merged_dir, **engine_kw) if args.backend == "vllm" \
                else HFEngine(AutoModelForCausalLM.from_pretrained(
                    merged_dir, torch_dtype=torch.bfloat16).cuda(), tok)

    results = dict(variant=args.variant, steps=[], train_reward=[], policy_nll=[],
                   distinct2=[], self_bleu2=[], panel={})
    for step in range(args.steps):
        t0 = time.time()
        rewards, p_entries, s_entries = run_rollout(policy_llm, sim_llm, tok,
                                                    args.variant, args.G)
        # ---------------- policy update (group-relative z-score REINFORCE)
        r = np.array(rewards)
        rbar, rstd = r.mean(), r.std() + 1e-6
        adv = (r - rbar) / rstd
        # per-trajectory policy NLL = mean over its own turns (faithful context)
        p_nll_per_traj = []
        for g in range(args.G):
            nlls = rollout_nll(pol_peft, tok, dev, POLICY_SYS, p_entries[g])
            p_nll_per_traj.append(np.nanmean(nlls) if nlls else float("nan"))
        pol_opt.zero_grad()
        nll_t = torch.tensor([v if v == v else 0.0 for v in p_nll_per_traj], device=dev)
        adv_t = torch.tensor(adv, device=dev)
        loss = (nll_t * adv_t).mean()
        loss.backward()
        pol_opt.step()
        if args.variant == "cot" and sim_peft is not None:
            s_nll_per_traj = []
            for g in range(args.G):
                nlls = rollout_nll(sim_peft, tok, dev, DONOR_SYS, s_entries[g])
                s_nll_per_traj.append(np.nanmean(nlls) if nlls else float("nan"))
            sim_adv = np.abs(r - rbar) / rstd  # SPICE-style variance reward
            sim_opt.zero_grad()
            nll_s = torch.tensor([v if v == v else 0.0 for v in s_nll_per_traj], device=dev)
            loss_s = (nll_s * torch.tensor(sim_adv, device=dev)).mean()
            loss_s.backward()
            sim_opt.step()
        # ---------------- engine reload
        if step % args.merge_every == (args.merge_every - 1):
            reload_engine("policy")
            if args.variant == "cot":
                reload_engine("simulator")
        # ---------------- metrics
        all_sim_texts = [e[1] for g in range(args.G) for e in s_entries[g]]
        all_pol_texts = [e[1] for g in range(args.G) for e in p_entries[g]]
        nll_flat = [v for v in p_nll_per_traj if v == v]
        rec = dict(step=step, train_reward=float(r.mean()),
                   policy_nll=float(np.nanmean(nll_flat)) if nll_flat else None,
                   distinct2=distinct2(all_pol_texts),
                   self_bleu2=self_bleu2(all_pol_texts),
                   sec=round(time.time() - t0, 1))
        results["steps"].append(rec)
        print(json.dumps(rec), flush=True)
        if step % args.panel_every == (args.panel_every - 1):
            panel = {}
            for name, persona in PANEL_PERSONAS.items():
                # panel eval: policy vs fresh persona simulator, 4 rollouts
                tot = 0.0
                opening = ("Hello! I'm raising funds for children's education. "
                           "Even $1 helps a child get textbooks for a year.")
                for _ in range(4):
                    p_ent, s_ent = [], []
                    for rnd in range(MAX_ROUNDS):
                        msgs = chat_prompt(POLICY_SYS, s_ent,
                                           final_user=s_ent[-1][1] if s_ent else opening)
                        p_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        p_text, _ = sample_distribution(policy_llm, tok, [p_prompt])
                        p_text = p_text[0]
                        msgs2 = chat_prompt(persona, p_ent, final_user=p_text)
                        s_prompt = tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
                        s_text, _ = sample_distribution(sim_llm, tok, [s_prompt])
                        p_ent.append((s_ent[-1][1] if s_ent else opening, p_text))
                        s_ent.append((p_text, s_text[0]))
                    tot += parse_donation(s_ent[-1][1]) / 2.0
                panel[name] = tot / 4.0
            results["panel"][str(step)] = panel
            print("PANEL", json.dumps(panel), flush=True)
    with open(os.path.join(args.out, f"{args.variant}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("done", args.variant)


if __name__ == "__main__":
    main()
