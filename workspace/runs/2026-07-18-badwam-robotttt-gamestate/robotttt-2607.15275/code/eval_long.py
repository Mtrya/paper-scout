"""Evaluate trained checkpoints at 2x their training horizon (T=1024):
does the recurrent/fast-weight state stay stable far beyond the training
window? Metric: R^2 on unsaturated steps, bucketed by context position.
"""
import json, sys
import torch

from task import sample_episode_batch
from models import Policy

BUCKETS = [(256, 384), (384, 512), (512, 640), (640, 768), (768, 896),
           (896, 1024)]
SAT = 0.95
T = 1024


def eval_long(ckpt_path, mixer):
    ckpt = torch.load(ckpt_path, weights_only=False)
    args = ckpt["args"]
    model = Policy(mixer, d=args["d"], layers=args["layers"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    gen = torch.Generator().manual_seed(999)
    bse = {b: 0.0 for b in BUCKETS}
    bav = {b: 0.0 for b in BUCKETS}
    with torch.no_grad():
        for _ in range(2):  # 2 x 32 = 64 episodes
            tokens, act = sample_episode_batch(32, T, gen=gen)
            pred, _ = model(tokens, create_graph=False)
            se = ((pred - act) ** 2).mean(-1)
            a2 = (act ** 2).mean(-1)
            mask = act.abs().amax(-1) < SAT
            for b in BUCKETS:
                m = mask.clone()
                m[:, :b[0]] = False
                m[:, b[1]:] = False
                bse[b] += se[m].sum().item()
                bav[b] += a2[m].sum().item()
    return {f"{b[0]}-{b[1]}": 1.0 - bse[b] / max(bav[b], 1e-12)
            for b in BUCKETS}


if __name__ == "__main__":
    torch.set_num_threads(4)
    out = {}
    for mixer, name in [("ttt", "ttt_T512"), ("delta", "delta_T512"),
                        ("attn", "attn_T512"), ("ttt", "ttt_T64")]:
        r = eval_long(f"results/{name}.pt", mixer)
        out[name] = r
        print(f"{name:12s}", {k: round(v, 3) for k, v in r.items()}, flush=True)
    with open("results/eval_long_T1024.json", "w") as f:
        json.dump(out, f)
