"""SimWAM checkpoint census + RL delta analysis (local, CPU, mmap).

Questions:
  1. What did the release actually ship? (parameter census by module group:
     video expert / action expert / VAE / T5 / other)
  2. Do video and action experts share parameters? (paper: "share no
     parameters and interact only through a unified attention interface")
  3. SimWAM.pt vs SimWAM-RL.pt: which tensors moved? Paper says RL updated
     only rank-32 LoRA adapters on the action expert's attention projections.
     If merged into the base weights, deltas should be confined to action
     expert attention q/k/v/o; the video expert should be bitwise identical.
"""
import json
import sys
from collections import defaultdict

import torch

CKPT_DIR = "checkpoints"
OUT = "analysis/results.json"


def census(sd, name):
    groups = defaultdict(lambda: [0, 0])  # group -> [n_tensors, n_params]
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        parts = k.split(".")
        # classify
        if "action" in k.lower():
            g = "action_expert"
        elif "vae" in k.lower():
            g = "vae"
        elif "text" in k.lower() or "t5" in k.lower():
            g = "text_encoder"
        elif "proprio" in k.lower():
            g = "proprio_encoder"
        else:
            g = "video_expert_or_shared"
        groups[g][0] += 1
        groups[g][1] += v.numel()
    total = sum(g[1] for g in groups.values())
    print(f"[{name}] top-level payload groups:")
    for g, (nt, np_) in sorted(groups.items()):
        print(f"  {g:24s} tensors={nt:5d} params={np_/1e9:.3f}B")
    print(f"  {'TOTAL':24s} params={total/1e9:.3f}B")
    return {g: {"tensors": nt, "params": np_} for g, (nt, np_) in groups.items()}


def prefix_histogram(sd, depth=4):
    hist = defaultdict(int)
    for k, v in sd.items():
        if torch.is_tensor(v):
            hist[".".join(k.split(".")[:depth])] += v.numel()
    return dict(sorted(hist.items(), key=lambda kv: -kv[1])[:25])


def main():
    import os
    os.makedirs("analysis", exist_ok=True)
    il = torch.load(f"{CKPT_DIR}/SimWAM.pt", map_location="cpu", mmap=True, weights_only=False)
    rl = None
    rl_path = f"{CKPT_DIR}/SimWAM-RL.pt"
    try:
        rl = torch.load(rl_path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as e:
        print(f"RL checkpoint unavailable ({e}); census only, no delta.")

    print("IL payload keys:", list(il.keys()))
    results = {"payload_keys": {"il": list(il.keys())},
               "meta": {k: il.get(k) for k in il.keys() if not isinstance(il.get(k), dict)}}
    if rl is not None:
        print("RL payload keys:", list(rl.keys()))
        results["payload_keys"]["rl"] = list(rl.keys())

    for tag, ck in [("il", il), ("rl", rl)]:
        if ck is None:
            continue
        for sub in ck.keys():
            if isinstance(ck[sub], dict):
                print(f"\n== {tag}/{sub} ==")
                results[f"census_{tag}_{sub}"] = census(ck[sub], f"{tag}/{sub}")
                results[f"hist_{tag}_{sub}"] = prefix_histogram(ck[sub])

    if rl is None:
        json.dump(results, open(OUT, "w"), indent=2, default=str)
        print("saved", OUT)
        return
    # delta analysis on the shared sub-dict
    common = None
    for sub in il.keys():
        if isinstance(il[sub], dict) and sub in rl and isinstance(rl[sub], dict):
            common = sub
            break
    moved, identical, only_il, only_rl = [], 0, [], []
    il_sd, rl_sd = il[common], rl[common]
    for k in il_sd:
        if k not in rl_sd:
            only_il.append(k)
            continue
        a, b = il_sd[k], rl_sd[k]
        if not torch.is_tensor(a):
            continue
        if a.shape != b.shape:
            moved.append((k, "SHAPE", None))
            continue
        d = (a.float() - b.float()).norm().item()
        if d == 0:
            identical += 1
        else:
            moved.append((k, float(d), float(a.float().norm())))
    for k in rl_sd:
        if k not in il_sd:
            only_rl.append(k)

    print(f"\n== delta ({common}) == identical={identical} moved={len(moved)} "
          f"only_il={len(only_il)} only_rl={len(only_rl)}")
    moved_sorted = sorted(moved, key=lambda x: -(x[1] if x[1] else 0))
    for m in moved_sorted[:40]:
        rel = (m[1] / m[2]) if m[2] else None
        print(f"  {m[0]:80s} dL2={m[1]:.4f} rel={rel and round(rel,5)}")

    results["delta"] = {
        "sub": common, "identical": identical,
        "moved": [{"key": m[0], "dL2": m[1], "base_norm": m[2]} for m in moved_sorted],
        "only_il": only_il, "only_rl": only_rl,
    }
    json.dump(results, open(OUT, "w"), indent=2, default=str)
    print("saved", OUT)


if __name__ == "__main__":
    main()
