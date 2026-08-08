"""Probe 2 — determinism isolation.

The det.npz episodes have NO stochasticity in dynamics: eps=0 policy and a
hash-of-state deterministic food respawn, so s_{t+1} is a pure function of
(s_t, a_t). Any divergence in greedy rollout is therefore dynamics learning
error, not irreducible spawn randomness. Same per-tick comparison as probe 1.
"""
import argparse, json
import numpy as np
import torch

from codec import encode_state_tokens, decode_state_tokens, build_prefix
from gen_data import load_split, unpack_fields, pack_fields
from model_typed import LogicEngine
from rollout import greedy_decode_states
from probe1_drift import head_err


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_typed.pt")
    ap.add_argument("--data", default="data/det.npz")
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    episodes = load_split(args.data)
    model = LogicEngine().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()
    H = args.horizon

    cur = [unpack_fields(ep["fields_init"]) for ep in episodes]
    exact = np.zeros(H)
    head_ok = np.zeros(H)
    first_div, first_head_div = [], []
    prev_div = [False] * len(episodes); prev_hdiv = [False] * len(episodes)
    for h in range(H):
        prefixes = []
        for b, ep in enumerate(episodes):
            spawns = [int(c) for c in ep["spawns"][h] if c >= 0]
            prefixes.append(build_prefix(cur[b], ep["actions"][h], spawns))
        out = greedy_decode_states(model, np.stack(prefixes), dev)
        for b, ep in enumerate(episodes):
            pred_f = decode_state_tokens(out[b])
            gt = unpack_fields(ep["fields_t"][h + 1])
            same = np.array_equal(pack_fields(pred_f), pack_fields(gt))
            he = head_err(pred_f, gt)
            exact[h] += same
            head_ok[h] += (not he)
            if not prev_div[b] and not same:
                first_div.append(h + 1); prev_div[b] = True
            if not prev_hdiv[b] and he:
                first_head_div.append(h + 1); prev_hdiv[b] = True
            cur[b] = pred_f

    n = len(episodes)
    res = dict(n_episodes=n, horizon=H,
               first_divergence_ticks=first_div,
               first_head_divergence=first_head_div,
               first_head_div_median=float(np.median(first_head_div)) if first_head_div else None,
               fraction_no_head_divergence=(n - len(first_head_div)) / n,
               exact_curve=(exact / n).tolist(),
               head_curve=(head_ok / n).tolist())
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/probe2_determinism{args.tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("curve")}, indent=1))


if __name__ == "__main__":
    main()
