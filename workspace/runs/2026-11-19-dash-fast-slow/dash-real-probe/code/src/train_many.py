"""Train every arm/seed in one process so the 28 GB feature cache is loaded once.

The head is tiny, so the per-run cost is dominated by data movement; a single process with a shared RAM cache and
sequential runs beats several parallel processes that each re-copy the split and compete for memory bandwidth.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DashHead  # noqa: E402
from packed import PackedSplit, H  # noqa: E402
from train import ARMS, gather, to_torch  # noqa: E402

CORE = ["sync_nofresh", "sync_fresh", "desync8_nofresh", "desync8_fresh"]
EXTRA = ["desync4_nofresh", "desync4_fresh", "leak_nofresh", "leak_fresh"]


def train_one(arm, seed, train, val, norms, args, probe):
    cfg = ARMS[arm]
    norm = norms["leak"] if cfg["leak"] else norms["plain"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(1000 + seed)
    state_dim = norm.get("state_dim", 11) if cfg["leak"] else 11
    head = DashHead(D=512, n_blocks=4, n_heads=8, horizon=H, act_dim=13, n_layers=train.n_layers,
                    n_vis=train.n_vis, vlm_dim=train.dim, state_dim=state_dim, use_fresh=cfg["fresh"]).cuda()
    n_params = sum(p.numel() for p in head.parameters())
    n_fresh = sum(p.numel() for p in head.resnet.parameters()) if cfg["fresh"] else 0
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    anchors = train.anchors(stride=1)
    steps_per_epoch = max(1, math.ceil(anchors.size / args.batch))
    total = steps_per_epoch * args.epochs
    warm = min(100, total // 10)

    def lr_at(i):
        if i < warm:
            return args.lr * (i + 1) / warm
        p = (i - warm) / max(1, total - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    log = dict(arm=arm, seed=seed, cfg=cfg, epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd,
               n_params=n_params, n_fresh_params=n_fresh, n_anchors=int(anchors.size),
               steps_per_epoch=steps_per_epoch, steps=[])
    t0 = time.time()
    gstep = 0
    for ep in range(args.epochs):
        head.train()
        perm = rng.permutation(anchors)
        ep_loss, nb = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            if idx.size < 4:
                continue
            delta = (rng.integers(0, cfg["dmax"] + 1, size=idx.size) if cfg["dmax"] > 0
                     else np.zeros(idx.size, np.int64))
            b = gather(train, idx, delta, cfg["leak"], cfg["fresh"])
            t = to_torch(b, norm, cfg["leak"], cfg["fresh"])
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)
            kvs = head.encode(t["vis"], t["txt"], t.get("img"))
            loss = head.loss(kvs, t["state"], t["act"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
            gstep += 1
        head.eval()
        with torch.no_grad():
            vls = {}
            for d in (0, 8):
                b = gather(val, probe, np.full(probe.size, d), cfg["leak"], cfg["fresh"])
                t = to_torch(b, norm, cfg["leak"], cfg["fresh"])
                kvs = head.encode(t["vis"], t["txt"], t.get("img"))
                vls[f"val_flow_loss_d{d}"] = float(head.loss(kvs, t["state"], t["act"]).item())
        rec = dict(epoch=ep, train_loss=ep_loss / max(1, nb), elapsed=round(time.time() - t0, 1), **vls)
        log["steps"].append(rec)
    log["wall_s"] = round(time.time() - t0, 1)
    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=head.state_dict(), arm=arm, seed=seed, cfg=cfg,
                    n_params=n_params, n_fresh_params=n_fresh), out / "ckpt" / f"{arm}_s{seed}.pt")
    (out / "logs" / f"{arm}_s{seed}.json").write_text(json.dumps(log, indent=1))
    print(f"[run] {arm} s{seed} loss {log['steps'][-1]['train_loss']:.4f} "
          f"val_d0 {log['steps'][-1]['val_flow_loss_d0']:.4f} val_d8 {log['steps'][-1]['val_flow_loss_d8']:.4f} "
          f"params {n_params} fresh {n_fresh} wall {log['wall_s']}s", flush=True)
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=CORE + EXTRA)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    args = ap.parse_args()

    train = PackedSplit(Path(args.packed), "train").to_ram()
    val = PackedSplit(Path(args.packed), "val").to_ram()
    norms = dict(plain=json.loads((Path(args.packed) / "norm.json").read_text()),
                 leak=json.loads((Path(args.packed) / "norm_leak.json").read_text()))
    va = val.anchors(stride=4)
    np.random.default_rng(0).shuffle(va)
    probe = va[:1024]
    print(f"[data] train {train.n} frames {train.anchors().size} anchors, val {val.n} frames, "
          f"probe {probe.size}", flush=True)
    for arm in args.arms:
        for s in args.seeds:
            ck = Path(args.out) / "ckpt" / f"{arm}_s{s}.pt"
            if ck.exists():
                print(f"[skip] {ck} exists", flush=True)
                continue
            train_one(arm, s, train, val, norms, args, probe)
    print("[all done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
