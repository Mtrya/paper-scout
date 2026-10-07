"""Streaming online-TTT harness for the five/six arm study.

One process per arm.  Arms differ only in the update operator; generation, task order,
turn budget, optimizer and hyper-parameters are shared.

Arm definitions
---------------
frozen          : no parameter update (also produces the Fixed-Generation cache).
loop-imitate    : ungated hard imitation (CE) on every completed episode.
rft             : hard imitation on verifier-accepted episodes only.
ascent          : forward-KL distillation from the frozen initial model, whose input is
                  privileged hindsight (task + success statement + valid-turn responses);
                  invalid-action / unparsed turns are dropped from the teacher input only.
rft-settlement  : rft + a Settlement gate -- the candidate state must not raise NLL on a
                  held-out real-text probe, otherwise the whole buffer update is rolled back.
loop-imitate-frozen-gen
                : ungated hard imitation whose training targets come from the frozen base
                  (Fixed Generation), never from the learner's own outputs.

LoRA is implemented directly (not peft) so that "disable the adapter" is exactly the frozen
initial model, and so that snapshot / rollback is a tensor copy.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import taskgen as tg
import probes

MODEL_ID = os.environ.get("TTT_MODEL", "Qwen/Qwen3-0.6B")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CTX_CAP = 2048          # keep at most this many trailing context tokens when training


# --------------------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32,
                                               device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32,
                                               device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            h = F.linear(x.float(), self.lora_A)
            out = out + (F.linear(h, self.lora_B) * self.scale).to(out.dtype)
        return out


def apply_lora(model, r=16, alpha=32, targets=LORA_TARGETS):
    n = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
                n += 1
    return n


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def lora_params(model):
    ps = []
    for m in lora_modules(model):
        ps += [m.lora_A, m.lora_B]
    return ps


def set_lora_enabled(model, flag: bool):
    for m in lora_modules(model):
        m.enabled = flag


def snapshot_lora(model):
    return [(m.lora_A.detach().clone(), m.lora_B.detach().clone()) for m in lora_modules(model)]


def restore_lora(model, snap):
    with torch.no_grad():
        for m, (a, b) in zip(lora_modules(model), snap):
            m.lora_A.copy_(a.to(m.lora_A.device))
            m.lora_B.copy_(b.to(m.lora_B.device))


# --------------------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------------------

class Model:
    def __init__(self, model, tok, device):
        self.model = model
        self.tok = tok
        self.device = device

    @classmethod
    def load(cls, model_id=MODEL_ID, device="cuda", attn="sdpa"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation=attn)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, tok, device)

    def enc(self, text):
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def render(self, system, messages):
        msgs = [{"role": "system", "content": system}] + [
            {"role": r, "content": c} for r, c in messages]
        out = None
        for kw in ({"tokenize": True, "add_generation_prompt": True,
                    "enable_thinking": False, "return_dict": False},
                   {"tokenize": True, "add_generation_prompt": True},
                   {"tokenize": True, "add_generation_prompt": True}):
            try:
                out = self.tok.apply_chat_template(msgs, **kw)
                break
            except TypeError:
                continue
        if isinstance(out, dict):                       # transformers >= 5 BatchEncoding
            out = out["input_ids"]
        elif hasattr(out, "keys") and "input_ids" in out:
            out = out["input_ids"]
        return list(out)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=96, stop_strings=("</cmd>", "</answer>")):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _Stop(StoppingCriteria):
            def __init__(self, tok, strings, prompt_len):
                self.tok, self.strings, self.prompt_len = tok, strings, prompt_len

            def __call__(self, input_ids, scores, **kw):
                gen = input_ids[0, self.prompt_len:]
                if gen.numel() == 0:
                    return False
                txt = self.tok.decode(gen, skip_special_tokens=True)
                return any(s in txt for s in self.strings)

        ids = torch.tensor([prompt_ids], device=self.device)
        out = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_Stop(self.tok, stop_strings, ids.shape[1])]),
            output_scores=True, return_dict_in_generate=True,
        )
        seq = out.sequences[0, ids.shape[1]:].tolist()
        scores = out.scores
        text = self.tok.decode(seq, skip_special_tokens=True)
        entropies = []
        for s in scores:
            lp = torch.log_softmax(s[0].float(), dim=-1)
            entropies.append(float(-(lp.exp() * lp).sum()))
        first_logits = scores[0][0].float() if len(scores) else None
        return text, seq, first_logits, entropies

    @torch.no_grad()
    def nll(self, texts, max_tokens=48):
        total, ntok = 0.0, 0
        for t in texts:
            ids = self.enc(t)[:max_tokens]
            if len(ids) < 4:
                continue
            inp = torch.tensor([ids], device=self.device)
            logits = self.model(input_ids=inp,
                                attention_mask=torch.ones_like(inp)).logits[0].float()
            lp = torch.log_softmax(logits[:-1], dim=-1)
            tgt = inp[0, 1:]
            total += float(-lp[torch.arange(len(tgt)), tgt].sum())
            ntok += len(tgt)
        return total / max(1, ntok)

    @torch.no_grad()
    def logits_at(self, ids, positions):
        inp = torch.tensor([ids], device=self.device)
        out = self.model.model(input_ids=inp, attention_mask=torch.ones_like(inp))
        h = out.last_hidden_state[0][positions]
        return self.model.lm_head(h).float()


# --------------------------------------------------------------------------------------
# supervision
# --------------------------------------------------------------------------------------

@dataclass
class Turn:
    ctx_ids: list
    resp_ids: list
    valid: bool


@dataclass
class Episode:
    task_id: str
    family: str
    level: int
    success: bool
    turns: list
    task_body: str
    valid_turns: list


@dataclass
class Config:
    arm: str = "frozen"
    tag: str = ""
    seed: int = 1234
    n_tasks: int = 160
    n_phases: int = 4
    eval_every: int = 20
    buffer_tasks: int = 8
    steps_per_update: int = 4
    lr: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    max_new_tokens: int = 96
    grad_clip: float = 1.0
    nll_probe_n: int = 50
    nll_probe_tokens: int = 48
    val_per_family: int = 5
    max_tokens_per_microbatch: int = 2048
    max_rows_per_microbatch: int = 6
    out_dir: str = "results"
    fixed_gen_dir: str = ""
    max_seconds: float = 3000.0
    eval_only: bool = False


def _trunc_ctx(ctx):
    return ctx[-CTX_CAP:] if len(ctx) > CTX_CAP else ctx


def _pad_batch(rows, pad_id, device):
    """Left-pad rows to a common length.

    ``resp_pos[i][k]`` is the position whose *logit predicts* ``tgt[i][k]``: in a causal LM
    the logit at position ``j`` predicts token ``j+1``, so the positions run from the last
    context token to the second-to-last response token.
    """
    maxlen = max(len(p) + len(t) for p, t in rows)
    B = len(rows)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((B, maxlen), dtype=torch.long)
    resp_pos, tgt = [], []
    for i, (p, t) in enumerate(rows):
        seq = p + t
        ids[i, maxlen - len(seq):] = torch.tensor(seq, dtype=torch.long)
        mask[i, maxlen - len(seq):] = 1
        start = maxlen - len(t) - 1                     # logit at `start` predicts t[0]
        resp_pos.append(list(range(start, start + len(t))))
        tgt.append(t)
    return ids.to(device), mask.to(device), resp_pos, tgt


def _logits_at_positions(model, ids, mask, resp_pos):
    # left padding requires explicit mask-aware positions: transformers builds
    # position_ids as arange(L) when none is given, which is wrong for left-padded rows.
    pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos_ids)
    hs = out.last_hidden_state
    h = torch.cat([hs[i, pos, :] for i, pos in enumerate(resp_pos)], dim=0)
    return model.lm_head(h).float()


def imit_loss(model, batch, pad_id, device):
    """Mean cross-entropy on response tokens only (hard targets)."""
    ids, mask, resp_pos, tgt = _pad_batch([(_trunc_ctx(r[0]), r[1]) for r in batch],
                                          pad_id, device)
    logits = _logits_at_positions(model, ids, mask, resp_pos)
    labels = torch.tensor([t for tt in tgt for t in tt], device=device)
    return F.cross_entropy(logits, labels)


def ascent_loss(model, batch, pad_id, device):
    """Forward KL(student || teacher) at student response positions; teacher sees z (+) ctx."""
    s_ids, s_mask, s_pos, _ = _pad_batch([(_trunc_ctx(r[0]), r[1]) for r in batch],
                                         pad_id, device)
    t_ids, t_mask, t_pos, _ = _pad_batch([(r[2] + _trunc_ctx(r[0]), r[1]) for r in batch],
                                         pad_id, device)
    with torch.no_grad():
        set_lora_enabled(model, False)
        q_logits = _logits_at_positions(model, t_ids, t_mask, t_pos)
        set_lora_enabled(model, True)
    p_logits = _logits_at_positions(model, s_ids, s_mask, s_pos)
    lp = F.log_softmax(p_logits, dim=-1)
    lq = F.log_softmax(q_logits, dim=-1)
    return (lq.exp() * (lq - lp)).sum(-1).mean()


def chunk_rows(rows, max_tokens=2048, max_rows=6, max_resp=512):
    """Group rows into micro-batches bounded in tokens, rows and scored positions."""
    rows = sorted(rows, key=lambda r: len(r[0]) + len(r[1]))
    out, cur, cur_len, cur_resp = [], [], 0, 0
    for r in rows:
        L, R = len(r[0]) + len(r[1]), len(r[1])
        if cur and (cur_len + L > max_tokens or cur_resp + R > max_resp
                    or len(cur) >= max_rows):
            out.append(cur)
            cur, cur_len, cur_resp = [], 0, 0
        cur.append(r)
        cur_len += L
        cur_resp += R
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------------------
# episode execution
# --------------------------------------------------------------------------------------

@torch.no_grad()
def seq_kl(mdl: Model, ctx_ids, resp_ids):
    """Per-token KL between the frozen base and the current policy over a response.

    Both directions are returned; recomputed from scratch at a fixed generation context.
    """
    if not resp_ids:
        return None, None
    ids, mask, pos, _ = _pad_batch([(_trunc_ctx(ctx_ids), resp_ids)],
                                   mdl.tok.pad_token_id or 0, mdl.device)
    set_lora_enabled(mdl.model, False)
    q = _logits_at_positions(mdl.model, ids, mask, pos)
    set_lora_enabled(mdl.model, True)
    p = _logits_at_positions(mdl.model, ids, mask, pos)
    lp = F.log_softmax(p, dim=-1)
    lq = F.log_softmax(q, dim=-1)
    return (float((lp.exp() * (lp - lq)).sum(-1).mean()),
            float((lq.exp() * (lq - lp)).sum(-1).mean()))


def run_episode(mdl: Model, task: tg.Task, cfg: Config):
    system = tg.SYSTEM
    messages = list(tg.demo_messages(task.family)) + [("user", task.prompt())]
    st = task.new_state()
    turns, valid_texts = [], []
    invalid_count = 0
    ent_first, ent_all = [], []
    kl_pairs = None
    seq_pairs = (None, None)
    success = False
    t0 = time.time()
    ctx_ids = mdl.render(system, messages)
    for turn_i in range(task.max_turns):
        text, new_ids, first_logits, ents = mdl.generate(ctx_ids, cfg.max_new_tokens)
        kind, payload = tg.parse_turn(text)
        valid, env_text, finished, ok = tg.step_env(task, st, kind, payload)
        turns.append(Turn(ctx_ids=list(ctx_ids), resp_ids=list(new_ids), valid=valid))
        if valid:
            valid_texts.append(text)
        else:
            invalid_count += 1
        if ents:
            ent_first.append(ents[0])
            ent_all.extend(ents)
        if turn_i == 0 and first_logits is not None:
            with torch.no_grad():
                set_lora_enabled(mdl.model, False)
                base_logits = mdl.logits_at(ctx_ids, [len(ctx_ids) - 1])
                set_lora_enabled(mdl.model, True)
            p = torch.log_softmax(first_logits, dim=-1)
            q = torch.log_softmax(base_logits, dim=-1)
            kl_pairs = (float((p.exp() * (p - q)).sum()), float((q.exp() * (q - p)).sum()))
            seq_pairs = seq_kl(mdl, ctx_ids, list(new_ids))
        messages.append(("assistant", text))
        if finished:
            success = bool(ok)
            break
        messages.append(("user", f"Environment: {env_text}"))
        ctx_ids = mdl.render(system, messages)
    ep = Episode(task_id=task.tid, family=task.family, level=task.level, success=success,
                 turns=turns, task_body=task.prompt(), valid_turns=valid_texts)
    rec = {
        "task_id": task.tid, "family": task.family, "level": task.level,
        "success": int(success), "n_turns": len(turns),
        "invalid_turns": invalid_count,
        "format_valid_rate": (len(turns) - invalid_count) / max(1, len(turns)),
        "first_tok_entropy": (sum(ent_first) / len(ent_first)) if ent_first else None,
        "mean_tok_entropy": (sum(ent_all) / len(ent_all)) if ent_all else None,
        "kl_cur_base": kl_pairs[0] if kl_pairs else None,
        "kl_base_cur": kl_pairs[1] if kl_pairs else None,
        "seq_kl_cur_base": seq_pairs[0],
        "seq_kl_base_cur": seq_pairs[1],
        "n_resp_tokens": sum(len(t.resp_ids) for t in turns),
        "wall_s": round(time.time() - t0, 3),
    }
    return ep, rec


def hindsight_z(mdl: Model, ep: Episode, use_filter: bool = True) -> list:
    """Privileged information z = task goal + success statement + (filtered) responses."""
    responses = [t for t in (ep.valid_turns if use_filter else [tl for tl in ep.turns])]
    if use_filter:
        body = "\n".join(ep.valid_turns)
    else:
        body = ""
    text = (
        "The following episode was completed successfully.\n"
        f"Task: {ep.task_body}\n"
        + (f"Successful responses:\n{body}\n" if body else "")
        + "Now reproduce the same episode.\n"
    )
    return mdl.enc(text)


@torch.no_grad()
def evaluate_val(mdl: Model, val_tasks, cfg: Config):
    recs = [run_episode(mdl, t, cfg)[1] for t in val_tasks]
    n = max(1, len(recs))
    skc = [r["seq_kl_cur_base"] for r in recs if r["seq_kl_cur_base"] is not None]
    skb = [r["seq_kl_base_cur"] for r in recs if r["seq_kl_base_cur"] is not None]
    return {
        "val_success": sum(r["success"] for r in recs) / n,
        "val_format_valid": sum(r["format_valid_rate"] for r in recs) / n,
        "val_first_tok_entropy": sum((r["first_tok_entropy"] or 0.0) for r in recs) / n,
        "val_mean_tok_entropy": sum((r["mean_tok_entropy"] or 0.0) for r in recs) / n,
        "val_invalid_turn_rate": sum(
            r["invalid_turns"] / max(1, r["n_turns"]) for r in recs) / n,
        "val_turns": sum(r["n_turns"] for r in recs) / n,
        "val_tokens": sum(r["n_resp_tokens"] for r in recs) / n,
        "val_seq_kl_cur_base": (sum(skc) / len(skc)) if skc else None,
        "val_seq_kl_base_cur": (sum(skb) / len(skb)) if skb else None,
    }


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def opt_state_clone(opt):
    """Deep copy of the optimizer state dict, so a rejected update can roll it back exactly."""
    import copy
    return copy.deepcopy(opt.state_dict())


def load_fixed_gen_episodes(traj_path, lo, hi):
    eps = []
    with open(traj_path) as fh:
        for line in fh:
            d = json.loads(line)
            if lo <= d["task_idx"] < hi:
                turns = [Turn(ctx_ids=t["ctx"], resp_ids=t["resp"], valid=t["valid"])
                         for t in d["turns"]]
                eps.append(Episode(task_id=d["task_id"], family=d.get("family", ""),
                                   level=d.get("level", 0), success=bool(d["success"]),
                                   turns=turns, task_body=d.get("body", ""),
                                   valid_turns=d.get("valid_texts", [])))
    return eps


# --------------------------------------------------------------------------------------
# buffer update
# --------------------------------------------------------------------------------------

def buffer_update(mdl, opt, params, buffer, cfg, probe_texts, fixed_cache, buf_lo, buf_hi):
    """Apply one buffer's worth of updates; returns a log record."""
    t_up = time.time()
    src = buffer
    if cfg.arm == "loop-imitate-frozen-gen":
        src = load_fixed_gen_episodes(fixed_cache, buf_lo, buf_hi)
    if cfg.arm in ("loop-imitate", "loop-imitate-frozen-gen"):
        use = src
    elif cfg.arm in ("rft", "rft-settlement", "ascent"):
        use = [e for e in src if e.success]
    else:
        use = []
    info = {"arm": cfg.arm, "buf_first": buf_lo, "buf_last": buf_hi - 1,
            "buffer_n": len(buffer), "gated_n": len(use), "lr": cfg.lr}
    if not use:
        info["skipped"] = True
        info["wall_s"] = round(time.time() - t_up, 2)
        return info, None
    rows = [(t.ctx_ids, t.resp_ids, hindsight_z(mdl, e, True) if cfg.arm == "ascent" else None)
            for e in use for t in e.turns]
    groups = chunk_rows(rows, cfg.max_tokens_per_microbatch, cfg.max_rows_per_microbatch)
    snap = snapshot_lora(mdl.model) if cfg.arm == "rft-settlement" else None
    opt_snap = opt_state_clone(opt) if cfg.arm == "rft-settlement" else None
    pre_nll = mdl.nll(probe_texts, cfg.nll_probe_tokens) if cfg.arm == "rft-settlement" else None
    tot_pos = sum(len(r[1]) for r in rows)
    losses, gns = [], []
    for _step in range(cfg.steps_per_update):
        opt.zero_grad(set_to_none=True)
        step_loss, gn = 0.0, 0.0
        for g in groups:
            w = sum(len(r[1]) for r in g) / max(1, tot_pos)
            batch = [tuple(r) for r in g]
            loss = (ascent_loss(mdl.model, batch, mdl.tok.pad_token_id or 0, mdl.device)
                    if cfg.arm == "ascent"
                    else imit_loss(mdl.model, batch, mdl.tok.pad_token_id or 0, mdl.device))
            (loss * w).backward()                      # free each micro-batch graph as we go
            step_loss += float(loss.detach()) * w
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        opt.step()
        losses.append(step_loss)
        gns.append(gn)
    info["loss"] = losses
    info["grad_norm"] = gns
    info["n_rows"] = len(rows)
    info["n_pos"] = tot_pos
    if cfg.arm == "rft-settlement":
        post_nll = mdl.nll(probe_texts, cfg.nll_probe_tokens)
        info["pre_nll"], info["post_nll"] = pre_nll, post_nll
        rejected = post_nll > pre_nll
        info["rejected"] = bool(rejected)
        if rejected:
            restore_lora(mdl.model, snap)
            opt.load_state_dict(opt_snap)
    info["wall_s"] = round(time.time() - t_up, 2)
    return info, (None if cfg.arm == "rft-settlement" else None)


# --------------------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------------------

def run_arm(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    stem = f"{cfg.arm}{cfg.tag}"
    stream_path = os.path.join(cfg.out_dir, f"{stem}_stream.jsonl")
    val_path = os.path.join(cfg.out_dir, f"{stem}_eval.jsonl")
    upd_path = os.path.join(cfg.out_dir, f"{stem}_updates.jsonl")
    ckpt_path = os.path.join(cfg.out_dir, f"{stem}_latest.pt")
    traj_path = os.path.join(cfg.out_dir, f"{stem}_trainids.jsonl")
    fixed_cache = (os.path.join(cfg.fixed_gen_dir, "frozen_trainids.jsonl")
                   if cfg.fixed_gen_dir else "")
    with open(os.path.join(cfg.out_dir, f"{stem}_config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    mdl = Model.load()
    n_lora = apply_lora(mdl.model, cfg.lora_r, cfg.lora_alpha)
    params = lora_params(mdl.model)
    trainable = sum(p.numel() for p in params)
    print(f"[{cfg.arm}] lora on {n_lora} modules, {trainable/1e6:.2f}M trainable params",
          flush=True)

    torch.manual_seed(cfg.seed)
    val_tasks = tg.build_val_set(seed=999, n_per_family=cfg.val_per_family)
    stream = tg.build_stream(seed=cfg.seed, n_tasks=cfg.n_tasks, n_phases=cfg.n_phases)
    probe_texts = probes.REAL_TEXT[: cfg.nll_probe_n]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)

    # ---- resume -------------------------------------------------------------------
    done = 0
    if os.path.exists(stream_path):
        with open(stream_path) as f:
            done = sum(1 for _ in f)
    resume_at = 0
    if os.path.exists(ckpt_path) and done > 0:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["task_idx"] <= done:
            resume_at = int(ck["task_idx"])
            restore_lora(mdl.model, ck["lora"])
            if ck.get("opt") is not None:
                opt.load_state_dict(ck["opt"])
            print(f"[{cfg.arm}] resume at task {resume_at} (jsonl had {done})", flush=True)
    if resume_at == 0:
        for p in (stream_path, val_path, upd_path, traj_path):
            if os.path.exists(p):
                os.remove(p)
    elif done > resume_at:
        with open(stream_path) as f:
            lines = f.readlines()[:resume_at]
        with open(stream_path, "w") as f:
            f.writelines(lines)

    if resume_at == 0:
        nll0 = mdl.nll(probe_texts, cfg.nll_probe_tokens)
        r = {"task_idx": 0, **evaluate_val(mdl, val_tasks, cfg), "probe_nll": nll0,
             "n_updates": 0, "n_rejected": 0, "wall_s": 0.0}
        append_jsonl(val_path, r)
        print(f"[{cfg.arm}] step0 val={r['val_success']:.3f} nll={nll0:.4f}", flush=True)
        if cfg.eval_only:
            return

    t_start = time.time()
    buffer: list[Episode] = []
    n_updates = n_rejected = 0
    traj_fh = open(traj_path, "a")
    try:
        for idx in range(resume_at, cfg.n_tasks):
            task = stream[idx]
            ep, rec = run_episode(mdl, task, cfg)
            ep.task_idx = idx
            rec["task_idx"] = idx
            rec["arm"] = cfg.arm
            traj_fh.write(json.dumps({
                "task_idx": idx, "task_id": ep.task_id, "family": ep.family,
                "level": ep.level, "success": ep.success, "body": ep.task_body,
                "valid_texts": ep.valid_turns,
                "turns": [{"ctx": t.ctx_ids, "resp": t.resp_ids, "valid": t.valid}
                          for t in ep.turns]}) + "\n")
            traj_fh.flush()
            buffer.append(ep)
            append_jsonl(stream_path, rec)

            if len(buffer) >= cfg.buffer_tasks:
                info, _ = buffer_update(mdl, opt, params, buffer, cfg, probe_texts,
                                        fixed_cache, idx + 1 - len(buffer), idx + 1)
                n_updates += 1
                n_rejected += int(bool(info.get("rejected")))
                append_jsonl(upd_path, info)
                buffer = []
                torch.save({"task_idx": idx + 1, "arm": cfg.arm,
                            "lora": [(a.detach().cpu(), b.detach().cpu())
                                     for a, b in snapshot_lora(mdl.model)],
                            "opt": opt.state_dict() if cfg.arm != "frozen" else None},
                           ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)

            if (idx + 1) % 5 == 0 or (idx + 1) == cfg.n_tasks:
                print(f"[{cfg.arm}] {idx+1}/{cfg.n_tasks} buf={len(buffer)} "
                      f"upd={n_updates} rej={n_rejected} t={time.time()-t_start:.0f}s",
                      flush=True)

            if (idx + 1) % cfg.eval_every == 0 or (idx + 1) == cfg.n_tasks:
                ev = evaluate_val(mdl, val_tasks, cfg)
                ev.update({"task_idx": idx + 1,
                           "probe_nll": mdl.nll(probe_texts, cfg.nll_probe_tokens),
                           "n_updates": n_updates, "n_rejected": n_rejected,
                           "wall_s": round(time.time() - t_start, 1)})
                append_jsonl(val_path, ev)
                print(f"[{cfg.arm}] EVAL@{idx+1} val={ev['val_success']:.3f} "
                      f"nll={ev['probe_nll']:.4f} fmt={ev['val_format_valid']:.3f}", flush=True)

            if cfg.max_seconds and (time.time() - t_start) > cfg.max_seconds:
                print(f"[{cfg.arm}] time budget hit at task {idx+1}", flush=True)
                break
    finally:
        traj_fh.close()
    print(f"[{cfg.arm}] DONE updates={n_updates} rejected={n_rejected} "
          f"wall={time.time()-t_start:.0f}s", flush=True)
