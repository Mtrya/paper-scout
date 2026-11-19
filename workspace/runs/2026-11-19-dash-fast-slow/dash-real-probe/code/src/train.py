"""Train one arm/seed of the 2x2 (desync training x fresh vision) controlled probe."""
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

ARMS = {
    # name:                 fresh,  desync dmax, leak full state
    "sync_nofresh":        dict(fresh=False, dmax=0, leak=False),
    "sync_fresh":          dict(fresh=True,  dmax=0, leak=False),
    "desync8_nofresh":     dict(fresh=False, dmax=8, leak=False),
    "desync8_fresh":       dict(fresh=True,  dmax=8, leak=False),
    "desync4_nofresh":     dict(fresh=False, dmax=4, leak=False),
    "desync4_fresh":       dict(fresh=True,  dmax=4, leak=False),
    "leak_nofresh":        dict(fresh=False, dmax=0, leak=True),
    "leak_fresh":          dict(fresh=True,  dmax=0, leak=True),
}


_SCALES = {}


def _scale_tensors(norm, device="cuda"):
    key = (id(norm), device)
    if key not in _SCALES:
        _SCALES[key] = (torch.tensor(norm["vis_scale"], device=device).view(1, -1, 1, 1),
                        torch.tensor(norm["txt_scale"], device=device).view(1, -1, 1),
                        torch.tensor(norm["state_mean"], device=device),
                        torch.tensor(norm["state_std"], device=device),
                        torch.tensor(norm["act_mean"], device=device),
                        torch.tensor(norm["act_std"], device=device))
    return _SCALES[key]


def to_torch(batch, norm, leak: bool, need_img: bool, device="cuda"):
    """Cast and normalise on the GPU: doing the fp16->fp32 conversion and the per-batch division on 10 CPUs costs
    more than the whole training step."""
    vs, ts, sm, ss, am, asd = _scale_tensors(norm, device)
    vis = torch.from_numpy(batch["vis"]).to(device, non_blocking=True).float() / vs
    txt = torch.from_numpy(batch["txt"]).to(device, non_blocking=True).float() / ts
    state_np = batch["state"] if not leak else np.concatenate([batch["state"], batch["scene"]], 1)
    state = (torch.from_numpy(np.ascontiguousarray(state_np)).to(device).float() - sm) / ss
    act = (torch.from_numpy(batch["act"]).to(device, non_blocking=True).float() - am) / asd
    out = dict(vis=vis, txt=txt, state=state, act=act)
    if need_img:
        img = torch.from_numpy(batch["img"]).to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float()
        out["img"] = img.div_(255.0).sub_(0.5).div_(0.5)
    return out


def gather(ds, idx, delta, leak, need_img, img_delta=None):
    return ds.gather(idx, delta, horizon=H, need_img=need_img, img_delta=img_delta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 60 batches/epoch")
    ap.add_argument("--cache", action="store_true", help="load the split into RAM before training")
    args = ap.parse_args()

    cfg = ARMS[args.arm]
    norm = json.loads(Path(args.norm).read_text())
    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)

    train = PackedSplit(Path(args.packed), "train")
    val = PackedSplit(Path(args.packed), "val")
    if args.cache:
        train.to_ram()
        val.to_ram()
    anchors = train.anchors(stride=1)
    val_anchors = val.anchors(stride=4)
    rng.shuffle(val_anchors)
    probe = val_anchors[:1024]

    assert (not cfg["leak"]) or norm.get("leak"), "leak arm needs norm_leak.json"
    state_dim = norm.get("state_dim", 11) if cfg["leak"] else 11
    head = DashHead(D=512, n_blocks=4, n_heads=8, horizon=H, act_dim=13, n_layers=train.n_layers,
                    n_vis=train.n_vis, vlm_dim=train.dim, state_dim=state_dim,
                    use_fresh=cfg["fresh"]).cuda()
    n_params = sum(p.numel() for p in head.parameters())
    n_fresh = sum(p.numel() for p in head.resnet.parameters()) if cfg["fresh"] else 0
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, math.ceil(anchors.size / args.batch))
    if args.smoke:
        steps_per_epoch = 40
    total = steps_per_epoch * args.epochs
    warm = min(100, total // 10)

    def lr_at(i):
        if i < warm:
            return args.lr * (i + 1) / warm
        p = (i - warm) / max(1, total - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    log = dict(arm=args.arm, seed=args.seed, cfg=cfg, epochs=args.epochs, batch=args.batch, lr=args.lr,
               wd=args.wd, n_params=n_params, n_fresh_params=n_fresh, n_train_frames=train.n,
               n_anchors=int(anchors.size), steps_per_epoch=steps_per_epoch, steps=[],
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
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
            delta = rng.integers(0, cfg["dmax"] + 1, size=idx.size) if cfg["dmax"] > 0 else np.zeros(idx.size, np.int64)
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
        # fixed-subset validation flow loss at delta 0 and delta 8
        head.eval()
        vls = {}
        with torch.no_grad():
            for d in (0, min(8, cfg["dmax"]) if cfg["dmax"] else 8):
                idx = probe[:512]
                b = gather(val, idx, np.full(idx.size, d), cfg["leak"], cfg["fresh"])
                t = to_torch(b, norm, cfg["leak"], cfg["fresh"])
                kvs = head.encode(t["vis"], t["txt"], t.get("img"))
                vls[f"val_flow_loss_d{d}"] = float(head.loss(kvs, t["state"], t["act"]).item())
        rec = dict(epoch=ep, train_loss=ep_loss / max(1, nb), elapsed=round(time.time() - t0, 1), **vls)
        log["steps"].append(rec)
        print("[train]", json.dumps(rec), flush=True)
    torch.save(dict(state_dict=head.state_dict(), arm=args.arm, seed=args.seed, cfg=cfg,
                    n_params=n_params, n_fresh_params=n_fresh),
               out / "ckpt" / f"{args.arm}_s{args.seed}.pt")
    log["wall_s"] = round(time.time() - t0, 1)
    (out / "logs" / f"{args.arm}_s{args.seed}.json").write_text(json.dumps(log, indent=1))
    print(f"[done] {args.arm} seed{args.seed} params={n_params} fresh={n_fresh} wall={log['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
