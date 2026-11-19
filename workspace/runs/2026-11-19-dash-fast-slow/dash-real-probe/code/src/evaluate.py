"""Evaluate a trained arm on the held-out episodes: fixed-Δ open-loop chunk error + channel ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DashHead  # noqa: E402
from packed import PackedSplit  # noqa: E402
from train import ARMS, to_torch  # noqa: E402

H = 8
DELTAS = (0, 1, 2, 4, 8)
NOISE_SEED = 20261119
OTHER = {"weight_sort": "com_sort", "com_sort": "weight_sort"}


def build_head(train_ds, norm, cfg, ckpt_path):
    state_dim = norm.get("state_dim", 11) if cfg["leak"] else 11
    head = DashHead(D=512, n_blocks=4, n_heads=8, horizon=H, act_dim=13, n_layers=train_ds.n_layers,
                    n_vis=train_ds.n_vis, vlm_dim=train_ds.dim, state_dim=state_dim,
                    use_fresh=cfg["fresh"]).cuda()
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head


@torch.no_grad()
def run(head, ds, anchors, norm, cfg, delta, cond, text_pure=None, batch=512, img_delta=None,
        tasks=None):
    errs, errs0 = [], []
    full_noise = np.random.default_rng(NOISE_SEED + delta * 97).standard_normal(
        (anchors.size, H, 13)).astype(np.float32)
    for s in range(0, anchors.size, batch):
        idx = anchors[s:s + batch]
        b = ds.gather(idx, delta, horizon=H, need_img=cfg["fresh"], img_delta=img_delta)
        if cond == "zero_vis":
            b["vis"] = np.zeros_like(b["vis"])
        elif cond == "shuf_vis":
            perm = np.random.default_rng(7 + s).permutation(b["vis"].shape[0])
            b["vis"] = b["vis"][perm]
        elif cond == "zero_txt":
            b["txt"] = np.zeros_like(b["txt"])
        elif cond == "other_txt":
            ts = tasks[s:s + batch]
            b["txt"] = np.stack([text_pure[OTHER[t]] for t in ts])
        noise = torch.from_numpy(full_noise[s:s + idx.size]).cuda()
        t = to_torch(b, norm, cfg["leak"], cfg["fresh"])
        kvs = head.encode(t["vis"], t["txt"], t.get("img"))
        pred = head.sample(kvs, t["state"], noise)
        sq = (pred - t["act"]) ** 2
        errs.append(sq.mean(1).cpu().numpy())        # (n, 13) mean over horizon
        errs0.append(sq[:, 0].cpu().numpy())         # (n, 13) first executed step
    return np.concatenate(errs, 0), np.concatenate(errs0, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--text-pure", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = ARMS[args.arm]
    norm = json.loads(Path(args.norm).read_text())
    train = PackedSplit(Path(args.packed), "train")
    val = PackedSplit(Path(args.packed), "val")
    head = build_head(train, norm, cfg, args.ckpt)
    anchors = val.anchors(stride=args.stride)
    if args.limit:
        anchors = anchors[: args.limit]
    ep_of = val.episode_of(anchors)
    anchor_task = np.array([val.episodes[e]["task"] for e in ep_of])
    tp = np.load(args.text_pure)
    text_pure = {t: tp[t].astype(np.float32) / np.asarray(norm["txt_scale"], np.float32)[:, None]
                 for t in OTHER}

    out = Path(args.out)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    meta = dict(arm=args.arm, seed=args.seed, n_anchors=int(anchors.size), stride=args.stride,
                deltas=list(DELTAS), noise_seed=NOISE_SEED, anchors=anchors.tolist(),
                tasks=anchor_task.tolist(), episodes=[val.episodes[e] for e in np.unique(ep_of)])
    (out / "pred" / f"meta_{args.arm}_s{args.seed}.json").write_text(json.dumps(meta))

    results = {}
    for d in DELTAS:
        e, e0 = run(head, val, anchors, norm, cfg, d, "main", tasks=anchor_task)
        np.savez_compressed(out / "pred" / f"{args.arm}_s{args.seed}_main_d{d}.npz", err=e, err0=e0)
        results[f"main_d{d}"] = dict(nmse=float(e.mean()), per_dim=np.round(e.mean(0), 5).tolist())
        print(f"[eval] {args.arm} s{args.seed} main d={d} nmse={e.mean():.4f}", flush=True)

    for d in (0, 8):
        for cond in ("zero_vis", "shuf_vis", "zero_txt", "other_txt"):
            e, _ = run(head, val, anchors, norm, cfg, d, cond, text_pure=text_pure, tasks=anchor_task)
            np.savez_compressed(out / "pred" / f"{args.arm}_s{args.seed}_{cond}_d{d}.npz", err=e)
            results[f"{cond}_d{d}"] = dict(nmse=float(e.mean()))
            print(f"[eval] {args.arm} s{args.seed} {cond} d={d} nmse={e.mean():.4f}", flush=True)
        if cfg["fresh"]:
            e, _ = run(head, val, anchors, norm, cfg, d, "main", img_delta=d, tasks=anchor_task)
            np.savez_compressed(out / "pred" / f"{args.arm}_s{args.seed}_stalefresh_d{d}.npz", err=e)
            results[f"stalefresh_d{d}"] = dict(nmse=float(e.mean()))
            print(f"[eval] {args.arm} s{args.seed} stalefresh d={d} nmse={e.mean():.4f}", flush=True)

    # constant (train-mean) predictor: the NMSE=1 anchor by construction
    anchor_ep = val.episode_of(anchors)
    cat = val.act[anchors[:, None] + np.arange(H)[None, :]].astype(np.float32)
    cat = (cat - np.asarray(norm["act_mean"], np.float32)) / np.asarray(norm["act_std"], np.float32)
    mean_err = (cat ** 2).mean(1)
    np.savez_compressed(out / "pred" / f"constpred_{args.arm}_s{args.seed}.npz", err=mean_err)
    results["constpred_d0"] = dict(nmse=float(mean_err.mean()))
    print(f"[eval] {args.arm} s{args.seed} const-predictor nmse={mean_err.mean():.4f}", flush=True)

    (out / "pred" / f"summary_{args.arm}_s{args.seed}.json").write_text(json.dumps(results, indent=1))
    print("[eval done]", args.arm, args.seed, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
