"""Autoregressive rollout + state metrics for typed and dense carriers.

Metrics (direct state, App. B.3/B.4 style):
  semantic      token/cell accuracy over the full canonical state
  active        accuracy over populated (non-PAD / non-empty) positions
  position      head-cell accuracy over snakes alive in GT
  count         exact entity-count match (alive snakes + food)
  full_exact    complete canonical state match
  contradiction structural violations in the raw prediction
"""
import numpy as np
import torch

from codec import (PAD, N_STATE_TOKENS, PREFIX_LEN, CELL_BASE, allowed_tokens,
                   build_prefix, decode_state_tokens, check_contradiction,
                   N_FOOD)
from gen_data import unpack_fields
from model_dense import (fields_to_grid, carry_to_grid, fields_to_target,
                         S0_HEAD, S0_BODY, FOOD)


# ------------------------------------------------------------- typed ----

@torch.no_grad()
def greedy_decode_states(model, prefixes, dev):
    """prefixes: int64 (B, PREFIX_LEN). Returns int64 (B, 421) with mask-enforced
    canonical decoding (greedy, dynamic selector)."""
    B = prefixes.shape[0]
    prefixes = torch.from_numpy(prefixes).to(dev)
    out = np.empty((B, N_STATE_TOKENS), dtype=np.int64)
    past = None
    offset = 0
    cur = prefixes
    pcount = np.full(B, 2)
    for j in range(N_STATE_TOKENS):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits, past = model.step(cur, past, offset)
        offset += cur.shape[1]
        lg = logits[:, -1].float().cpu().numpy()  # (B, V)
        for b in range(B):
            allowed = allowed_tokens(j, out[b, :j] if j else [], pcount[b])
            nxt = allowed[np.argmax(lg[b, allowed])]
            out[b, j] = nxt
            if j == 4 + N_FOOD:
                from codec import PCOUNT_BASE
                pcount[b] = int(nxt - PCOUNT_BASE) + 1
        cur = torch.from_numpy(out[:, j:j + 1]).to(dev)
    return out


def typed_metrics(pred_tok, gt_fields, gt_tok):
    """One tick. pred_tok (421,), gt fields dict, gt_tok (421,)."""
    sem = float((pred_tok == gt_tok).mean())
    nz = gt_tok != PAD
    active = float((pred_tok[nz] == gt_tok[nz]).mean()) if nz.any() else 1.0
    pred_f = decode_state_tokens(pred_tok)
    heads, alive_gt = [], 0
    for i in range(gt_fields["player_count"]):
        g = gt_fields["slots"][i]
        if not g["alive"]:
            continue
        alive_gt += 1
        p = pred_f["slots"][i]
        ok = (not p["inactive"]) and p["length"] > 0 and int(p["body"][0]) == int(g["body"][0])
        heads.append(float(ok))
    position = float(np.mean(heads)) if heads else 1.0
    alive_pred = sum(1 for i in range(pred_f["player_count"])
                     if pred_f["slots"][i]["alive"] and not pred_f["slots"][i]["inactive"])
    count = float(alive_pred == alive_gt and len(pred_f["food"]) == len(gt_fields["food"]))
    full = float((pred_tok == gt_tok).all())
    contra = float(len(check_contradiction(pred_f)) > 0)
    return dict(semantic=sem, active=active, position=position, count=count,
                full_exact=full, contradiction=contra)


@torch.no_grad()
def rollout_typed(model, episodes, horizon, dev, stream="recorded",
                  noop=False, empty_spawn=False):
    """Greedy autoregressive rollout from s_0 of each episode.

    Returns list of (horizon, 421) predicted token states per episode.
    stream='recorded' uses recorded joint actions + exogenous spawns.
    noop=True forces all-no-op actions; empty_spawn=True forces empty spawns.
    """
    B = len(episodes)
    cur_fields = [unpack_fields(ep["fields_init"]) for ep in episodes]
    preds = [np.empty((horizon, N_STATE_TOKENS), np.int64) for _ in episodes]
    for h in range(horizon):
        prefixes = []
        for b, ep in enumerate(episodes):
            if noop:
                actions = [0] * 8
                spawns = []
            else:
                actions = [int(a) for a in ep["actions"][h]]
                spawns = [] if empty_spawn else [int(c) for c in ep["spawns"][h] if c >= 0]
            prefixes.append(build_prefix(cur_fields[b], actions, spawns))
        out = greedy_decode_states(model, np.stack(prefixes), dev)
        for b in range(B):
            preds[b][h] = out[b]
            cur_fields[b] = decode_state_tokens(out[b])
    return preds


# ------------------------------------------------------------- dense ----

def parse_dense(cls, alives):
    """Parse raw class grid -> per-snake head cells, body cells, food count."""
    res = dict(food=int((cls == FOOD).sum()), snakes=[])
    for i in range(2):
        heads = np.argwhere(cls == S0_HEAD + 2 * i)
        bodies = np.argwhere(cls == S0_BODY + 2 * i)
        res["snakes"].append(dict(alive=bool(alives[i]), heads=heads, bodies=bodies))
    return res


def dense_contradiction(parsed):
    """Structural check on the raw dense prediction."""
    for s in parsed["snakes"]:
        if not s["alive"]:
            if len(s["heads"]) or len(s["bodies"]):
                return True
            continue
        if len(s["heads"]) != 1:
            return True
        if len(s["bodies"]):
            cells = {tuple(c) for c in s["bodies"]}
            hy, hx = s["heads"][0]
            nbr = any((abs(cy - hy) + abs(cx - hx)) == 1 for cy, cx in cells)
            if not nbr:
                return True
            # body must form one 4-connected chain from the head
            seen, stack = set(), [(hy, hx)]
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    c = (cy + dy, cx + dx)
                    if c in cells and c not in seen:
                        seen.add(c); stack.append(c)
            if seen != cells:
                return True
    return False


@torch.no_grad()
def rollout_dense(model, episodes, horizon, dev):
    """Raw-prediction feedback rollout. Returns per-episode lists of carries."""
    B = len(episodes)
    carries = []
    for ep in episodes:
        f0 = unpack_fields(ep["fields_init"])
        cls, heads, alives = fields_to_target(f0)
        carries.append(dict(cls=cls, headings=heads, alives=alives))
    outs = [[] for _ in episodes]
    for h in range(horizon):
        xs = []
        for b, ep in enumerate(episodes):
            actions = [int(a) for a in ep["actions"][h]]
            spawns = [int(c) for c in ep["spawns"][h] if c >= 0]
            tick = int(unpack_fields(ep["fields_t"][h])["tick"])
            xs.append(carry_to_grid(carries[b], actions, spawns, tick))
        x = torch.from_numpy(np.stack(xs)).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            cell, head, alive = model(x)
        cls = cell.float().argmax(1).cpu().numpy()
        heads = head.float().argmax(-1).cpu().numpy()
        alives = alive.float().argmax(-1).cpu().numpy()
        for b in range(B):
            carries[b] = dict(cls=cls[b], headings=heads[b], alives=alives[b])
            outs[b].append(carries[b])
    return outs


def dense_metrics(carry, gt_fields):
    cls = carry["cls"]
    gt_cls, gt_heads, gt_alives = fields_to_target(gt_fields)
    sem = float((cls == gt_cls).mean())
    nz = gt_cls != 0
    active = float((cls[nz] == gt_cls[nz]).mean()) if nz.any() else 1.0
    parsed = parse_dense(cls, carry["alives"])
    heads_acc, alive_gt = [], 0
    for i in range(gt_fields["player_count"]):
        g = gt_fields["slots"][i]
        if not g["alive"]:
            continue
        alive_gt += 1
        hs = parsed["snakes"][i]["heads"]
        gy, gx = divmod(int(g["body"][0]), 48)
        heads_acc.append(float(len(hs) == 1 and hs[0][0] == gy and hs[0][1] == gx))
    position = float(np.mean(heads_acc)) if heads_acc else 1.0
    alive_pred = sum(1 for s in parsed["snakes"] if s["alive"])
    count = float(alive_pred == alive_gt and parsed["food"] == (gt_cls == FOOD).sum())
    full = float((cls == gt_cls).all())
    contra = float(dense_contradiction(parsed))
    return dict(semantic=sem, active=active, position=position, count=count,
                full_exact=full, contradiction=contra)
