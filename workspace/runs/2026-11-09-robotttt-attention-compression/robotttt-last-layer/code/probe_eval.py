"""Shared held-out evaluation and fast-state diagnostics for the paired arms.

Every arm is scored on the SAME held-out episodes, drawn once from a fixed
generator seed. A short-horizon score is the prefix of the same rollout used
for the long-horizon score, so T=512 and T=1024 numbers are the same episodes
cut at different points, not two independent draws.
"""
import math

import torch

from task import sample_episode_batch

SAT = 0.95  # actions with max|a| >= SAT are trivially predictable; excluded

# context buckets (total elapsed steps t), same definitions as the original
# probe so numbers stay comparable with the previous run
BUCKETS_512 = [(8, 16), (16, 32), (32, 64), (64, 128), (128, 256),
               (256, 384), (384, 512)]
BUCKETS_1024 = [(256, 384), (384, 512), (512, 640), (640, 768),
                (768, 896), (896, 1024)]


def buckets_for(T):
    """Context buckets fully contained in a horizon of T steps."""
    base = BUCKETS_512 if T <= 512 else BUCKETS_1024
    return [b for b in base if b[1] <= T]


def sample_heldout(episodes=64, T=1024, batch=32, seed=999, device="cpu"):
    """One fixed episode set shared by every arm and every horizon."""
    if episodes % batch:
        raise ValueError("episodes must be a multiple of batch")
    gen = torch.Generator(device=device).manual_seed(seed)
    toks, acts = [], []
    for _ in range(episodes // batch):
        tok, act = sample_episode_batch(batch, T, device=device, gen=gen)
        toks.append(tok)
        acts.append(act)
    return torch.cat(toks, 0), torch.cat(acts, 0)


@torch.no_grad()
def evaluate(model, tokens, act, horizons, batch=32):
    """R^2 = 1 - SS_res / SS_act over unsaturated steps (max|a| < 0.95).

    The model is run once over the longest horizon; shorter horizons are
    prefixes of those same rollouts. The mixers are causal, so a prefix's
    outputs are identical either way.
    """
    horizons = sorted(set(horizons))
    maxT = horizons[-1]
    if tokens.shape[1] < maxT:
        raise ValueError(f"episodes are {tokens.shape[1]} steps, need {maxT}")
    model.eval()
    acc = {}
    for h in horizons:
        acc[h] = {"se": 0.0, "a2": 0.0, "n": 0, "un_se": 0.0,
                  "un_a2": 0.0, "un_n": 0,
                  "bucket": {b: [0.0, 0.0, 0] for b in buckets_for(h)}}
    for s in range(0, tokens.shape[0], batch):
        tk = tokens[s:s + batch, :maxT]
        ac = act[s:s + batch, :maxT]
        pred, _ = model(tk, create_graph=False)
        for h in horizons:
            p, a = pred[:, :h], ac[:, :h]
            se = ((p - a) ** 2).mean(-1)          # (B, h)
            a2 = (a ** 2).mean(-1)
            unsat = a.abs().amax(-1) < SAT
            m8 = unsat.clone()
            m8[:, :8] = False                     # skip warm-up
            acc[h]["se"] += se.sum().item()
            acc[h]["a2"] += a2.sum().item()
            acc[h]["n"] += se.numel()
            acc[h]["un_se"] += se[m8].sum().item()
            acc[h]["un_a2"] += a2[m8].sum().item()
            acc[h]["un_n"] += int(m8.sum().item())
            for b in buckets_for(h):
                m = unsat.clone()
                m[:, :b[0]] = False
                m[:, b[1]:] = False
                acc[h]["bucket"][b][0] += se[m].sum().item()
                acc[h]["bucket"][b][1] += a2[m].sum().item()
                acc[h]["bucket"][b][2] += int(m.sum().item())
    out = {}
    for h in horizons:
        r2, counts = {}, {}
        for b, (bse, bav, bn) in acc[h]["bucket"].items():
            name = f"{b[0]}-{b[1]}"
            counts[name] = bn
            r2[name] = None if bn == 0 else 1.0 - bse / max(bav, 1e-12)
        out[str(h)] = {
            "horizon": h,
            "r2_by_context_bucket": r2,
            "bucket_counts": counts,
            "r2_unsat": (None if acc[h]["un_n"] == 0
                         else 1.0 - acc[h]["un_se"] / max(acc[h]["un_a2"], 1e-12)),
            "global_nmse": acc[h]["se"] / max(acc[h]["a2"], 1e-12),
            "metric": ("R^2 on unsaturated steps (max|a|<0.95), bucketed by "
                       "context position; r2_unsat aggregates t>=8"),
        }
    return out


@torch.no_grad()
def inner_diag(model, tokens, act, T=None, batch=16):
    """Per-timestep inner (self-supervised) loss and fast-state drift.

    Layer 0 and layer 1 are both replayed; the inner loss is the same
    pre-update regression loss the inner loop minimises. Drift is relative
    ||W_t - W_0|| / ||W_0|| and is reported three ways:

      drift_all  -- global ratio over every carried state tensor. For the
                    last-readout arm this includes W1/b1/b2, whose movement
                    is exactly zero, so the number is diluted.
      drift_upd  -- global ratio over the tensors the inner loop updates
                    (W1,b1,W2,b2 for ttt; W2 only for ttt_last).
      drift_by_key -- per-tensor series.

    static_abs_change records max_t ||W_t - W_0||_F for every tensor the
    inner loop must not touch; it is exactly 0.0 for the last-readout arm.
    """
    model.eval()
    T = tokens.shape[1] if T is None else min(T, tokens.shape[1])
    tokens, act = tokens[:batch, :T], act[:batch, :T]
    B = tokens.shape[0]
    layers_out = []
    x = model.embed(tokens)
    for layer in model.layers:
        W0 = {k: v.detach().clone() for k, v in layer.init_state(B).items()}
        n0 = {k: v.float().pow(2).sum().item() for k, v in W0.items()}
        fast = set(layer.fast_keys())
        rec = {"fast_keys": list(layer.fast_keys()),
               "static_keys": [k for k in W0 if k not in fast],
               "inner_loss": [], "drift_all": [], "drift_upd": [],
               "drift_by_key": {k: [] for k in W0},
               "static_abs_change": {k: 0.0 for k in W0 if k not in fast}}

        def hook(t, W, loss, rec=rec, W0=W0, n0=n0, fast=fast, B=B):
            rec["inner_loss"].append(round(loss.detach().item() / B, 6))
            num_all = num_upd = den_all = den_upd = 0.0
            for k, v in W.items():
                d2 = (v.float() - W0[k].float()).pow(2).sum().item()
                # relative drift is undefined for zero-initialised tensors
                rec["drift_by_key"][k].append(
                    None if n0[k] < 1e-12
                    else round(math.sqrt(d2 / n0[k]), 6))
                num_all += d2
                den_all += n0[k]
                if k in fast:
                    num_upd += d2
                    den_upd += n0[k]
                else:
                    rec["static_abs_change"][k] = max(
                        rec["static_abs_change"][k], math.sqrt(d2))
            rec["drift_all"].append(
                round(math.sqrt(num_all / max(den_all, 1e-12)), 6))
            rec["drift_upd"].append(
                round(math.sqrt(num_upd / max(den_upd, 1e-12)), 6))

        y, _ = layer(x, None, create_graph=False, step_hook=hook)
        layers_out.append(rec)
        x = x + y
    pred = model.head(model.ln_f(x))
    per_pos = ((pred - act) ** 2).mean(-1).mean(0)
    return {"T": T, "batch": B, "layers": layers_out,
            "per_pos_mse": [round(v, 6) for v in per_pos.tolist()]}
