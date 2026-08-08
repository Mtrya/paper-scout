"""Direct logic evaluation (App. B.4 style): greedy AR feedback, per-horizon metrics.

Evaluates typed and/or dense checkpoints on a split (default: val, 16 episodes),
horizons 1..128, reports aggregate table + per-tick curves to results/.
"""
import argparse, json
import numpy as np
import torch

from codec import encode_state_tokens
from gen_data import load_split, unpack_fields
from model_typed import LogicEngine
from model_dense import DenseCNN
from rollout import (rollout_typed, typed_metrics, rollout_dense, dense_metrics)

HORIZONS = [1, 8, 16, 32, 64, 128]
METRICS = ["semantic", "active", "position", "count", "full_exact", "contradiction"]


def eval_typed(ckpt, episodes, dev, horizon=128, batch_eps=None):
    model = LogicEngine().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    model.eval()
    # preds[b][h] is s_{h+1}; GT at the same tick is fields_t[h+1]
    gt_toks = [[encode_state_tokens(unpack_fields(ep["fields_t"][h + 1]))
                for h in range(horizon)] for ep in episodes]
    gt_fields = [[unpack_fields(ep["fields_t"][h + 1]) for h in range(horizon)]
                 for ep in episodes]
    per_tick = np.zeros((horizon, len(METRICS)))
    preds = rollout_typed(model, episodes, horizon, dev)
    for b, ep in enumerate(episodes):
        for h in range(horizon):
            m = typed_metrics(preds[b][h], gt_fields[b][h], gt_toks[b][h])
            for k, name in enumerate(METRICS):
                per_tick[h, k] += m[name]
    per_tick /= len(episodes)
    return per_tick, preds


def eval_dense(ckpt, episodes, dev, horizon=128):
    model = DenseCNN().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    model.eval()
    gt_fields = [[unpack_fields(ep["fields_t"][h + 1]) for h in range(horizon)]
                 for ep in episodes]
    per_tick = np.zeros((horizon, len(METRICS)))
    outs = rollout_dense(model, episodes, horizon, dev)
    for b, ep in enumerate(episodes):
        for h in range(horizon):
            m = dense_metrics(outs[b][h], gt_fields[b][h])
            for k, name in enumerate(METRICS):
                per_tick[h, k] += m[name]
    per_tick /= len(episodes)
    return per_tick, outs


def summarize(per_tick):
    rows = {}
    for H in HORIZONS:
        rows[H] = {name: round(100 * float(per_tick[H - 1, k]), 1)
                   for k, name in enumerate(METRICS)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typed", default=None)
    ap.add_argument("--dense", default=None)
    ap.add_argument("--data", default="data/val.npz")
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    episodes = load_split(args.data)
    result = {}
    if args.typed:
        pt, _ = eval_typed(args.typed, episodes, dev, args.horizon)
        result["typed"] = summarize(pt)
        np.save(f"results/typed_curve{args.tag}.npy", pt)
        print("typed:", json.dumps(result["typed"], indent=1))
    if args.dense:
        pt, _ = eval_dense(args.dense, episodes, dev, args.horizon)
        result["dense"] = summarize(pt)
        np.save(f"results/dense_curve{args.tag}.npy", pt)
        print("dense:", json.dumps(result["dense"], indent=1))
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(f"results/eval{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
