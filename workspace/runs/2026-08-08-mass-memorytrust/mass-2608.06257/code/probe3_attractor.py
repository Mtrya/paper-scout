"""Probe 3 — attractor analysis (App. C.3 protocol).

No-op action stream + explicit empty spawns from held-out initial states,
long horizon (H=1024 or 4096). Per episode: unique-state count (excluding the
tick nibbles), cycle entry + period, roster survival, structural validity.
Paper reference: 20k N=2 checkpoint visits ~277 unique states through H=4096.
"""
import argparse, json
import numpy as np
import torch

from codec import decode_state_tokens, build_prefix, check_contradiction
from gen_data import load_split, unpack_fields
from model_typed import LogicEngine
from rollout import greedy_decode_states


def state_key(tok):
    """Canonical state identity excluding the 4 tick nibbles."""
    return tuple(tok[4:].tolist())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_typed.pt")
    ap.add_argument("--data", default="data/val.npz")
    ap.add_argument("--horizon", type=int, default=4096)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    episodes = load_split(args.data)[:args.episodes]
    model = LogicEngine().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()
    H = args.horizon

    cur = [unpack_fields(ep["fields_init"]) for ep in episodes]
    seen = [dict() for _ in episodes]          # state_key -> first step
    alive_final, invalid = [], 0
    unique_counts, cycle_entries, cycle_periods = [], [], []
    unique_curve = [[] for _ in episodes]
    for h in range(H):
        prefixes = [build_prefix(cur[b], [0] * 8, []) for b in range(len(episodes))]
        out = greedy_decode_states(model, np.stack(prefixes), dev)
        for b in range(len(episodes)):
            k = state_key(out[b])
            if h % 128 == 0 or h == H - 1:
                unique_curve[b].append(len(seen[b]) + (0 if k in seen[b] else 1))
            if k not in seen[b]:
                seen[b][k] = h + 1
            cur[b] = decode_state_tokens(out[b])
            if h == H - 1:
                f = cur[b]
                alive_final.append(sum(s["alive"] for s in f["slots"] if not s["inactive"]))
                invalid += len(check_contradiction(f)) > 0
        if (h + 1) % 512 == 0:
            print(f"  step {h+1}/{H}, unique so far: {[len(s) for s in seen]}", flush=True)

    for b in range(len(episodes)):
        unique_counts.append(len(seen[b]))
        # cycle: final state's first occurrence
        final_k = state_key(out[b])
        entry = seen[b][final_k]
        cycle_entries.append(entry)
        cycle_periods.append(H - entry)

    res = dict(n_episodes=len(episodes), horizon=H,
               unique_states=unique_counts,
               cycle_entry_tick=cycle_entries,
               cycle_period=cycle_periods,
               alive_at_end=alive_final, invalid_at_end=invalid,
               unique_curve={str(b): unique_curve[b] for b in range(len(episodes))})
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/probe3_attractor{args.tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "unique_curve"}, indent=1))


if __name__ == "__main__":
    main()
