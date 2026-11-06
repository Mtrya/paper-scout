#!/usr/bin/env python3
"""Attentive-probe regression on precomputed features (DreamX B.4 protocol).

Protocol: freeze representation, train a common attentive probe.
  per-token Linear(D->768) + LayerNorm + GELU
  + learned temporal embedding per latent-frame position
  flatten (T*N) tokens -> learned-query attentive pooler -> Linear head.
Train: Huber(delta=1). Report: MSE on normalized actions (LARYBench metric),
plus position/orientation/gripper group MSE.

Features: directory of per-sample npz, each with 'tokens' (T,N,D) fp16/fp32.
Manifest CSV: sample_id,src_img,tgt_img,action (action npy = (5,7) absolute).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Normalization stats copied from LARYBench regression/main.py (absolute mode).
CALVIN_MEAN = np.array([0.03993005, -0.1113833, 0.50033228, 1.04580053, -0.08165425, 1.58390577, -0.08441296])
CALVIN_STD = np.array([0.14403107, 0.09919957, 0.05518382, 2.89455128, 0.13053949, 0.57474015, 0.99643086])
VLABENCH_MEAN = np.array([0.02959172, 0.39965126, 0.25788001, 1.0136418, -0.38189239, -0.13961033, 0.54842541])
VLABENCH_STD = np.array([0.15966601, 0.13813421, 0.1362726, 2.70449661, 0.63210694, 1.72885403, 0.49764945])

GROUPS = {"position": [0, 1, 2], "orientation": [3, 4, 5], "gripper": [6]}


class FeatureDataset(Dataset):
    def __init__(self, manifest_csv, feature_dir, dataset, feature_key="tokens", pool4=False, preload=False, meanpool=False):
        self.df = pd.read_csv(manifest_csv)
        self.feature_dir = feature_dir
        self.dataset = dataset
        self.feature_key = feature_key
        self.pool4 = pool4
        self.meanpool = meanpool
        if dataset == "calvin":
            self.mean, self.std = CALVIN_MEAN, CALVIN_STD
        else:
            self.mean, self.std = VLABENCH_MEAN, VLABENCH_STD
        self.rows = []
        for _, r in self.df.iterrows():
            fp = os.path.join(feature_dir, str(r["sample_id"]).replace("/", "_") + ".npz")
            if os.path.exists(fp):
                self.rows.append((fp, r["action"]))
        print(f"{manifest_csv}: {len(self.rows)}/{len(self.df)} samples have features")
        self.cache = None
        if preload:
            t0 = time.time()
            toks, acts = [], []
            for fp, act_path in self.rows:
                toks.append(self._load_tok(fp))
                acts.append(self._load_act(act_path))
            self.cache = (toks, acts)
            print(f"preloaded {len(toks)} features in {time.time()-t0:.0f}s")

    def _load_tok(self, fp):
        z = np.load(fp)
        tok = z[self.feature_key].astype(np.float32)  # (T,N,D)
        if self.pool4:
            T, N, D = tok.shape
            g = int(round(N ** 0.5))
            t4 = torch.from_numpy(tok.reshape(T, g, g, D).transpose(0, 3, 1, 2).copy())
            tok = F.adaptive_avg_pool2d(t4, (2, 2)).permute(0, 2, 3, 1).reshape(T, 4, D).numpy()
        if self.meanpool:
            tok = tok.mean(axis=1, keepdims=True)  # (T,1,D)
        return tok

    def _load_act(self, act_path):
        act = np.load(act_path).astype(np.float32).reshape(-1)  # 35
        return ((act - np.tile(self.mean, 5)) / np.tile(self.std, 5)).astype(np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.cache is not None:
            tok, act = self.cache[0][i], self.cache[1][i]
        else:
            fp, act_path = self.rows[i]
            tok, act = self._load_tok(fp), self._load_act(act_path)
        T = tok.shape[0]
        return torch.from_numpy(tok.reshape(T * tok.shape[1], tok.shape[2])), torch.from_numpy(act), T


def collate(batch):
    toks, acts, ts = zip(*batch)
    n_max = max(t.shape[0] for t in toks)
    D = toks[0].shape[1]
    x = torch.zeros(len(toks), n_max, D)
    mask = torch.zeros(len(toks), n_max, dtype=torch.bool)
    for i, t in enumerate(toks):
        x[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    return x, torch.stack(acts), mask, torch.tensor(ts)


class AttentiveProbe(nn.Module):
    def __init__(self, in_dim, out_dim=35, d=768, n_temporal=16):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU())
        self.temporal_emb = nn.Parameter(torch.randn(n_temporal, 1, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x, mask, ts):
        h = self.proj(x)  # (B, T*N, d)
        B, S, d = h.shape
        t0 = int(ts[0])
        assert bool((ts == t0).all()), "uniform T per batch expected"
        n = S // t0
        h = h + self.temporal_emb[:t0].reshape(t0, 1, d).expand(t0, n, d).reshape(1, S, d)
        q = self.query.expand(B, -1, -1)
        key_padding = ~mask
        pooled, _ = self.attn(q, h, h, key_padding_mask=key_padding)
        return self.head(self.out_norm(pooled.squeeze(1)))


class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return self.ffn(x) + x


class ActionExpertMLP(nn.Module):
    """LARYBench official regression probe (regression/main.py): flat LN->4096->2xResBlock->35."""

    def __init__(self, in_dim, out_dim=35, hidden=4096, num_blocks=2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden) for _ in range(num_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, x, mask=None, ts=None):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def evaluate(model, loader, device, flat=False):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y, mask, ts in loader:
            if flat:
                p = model(x.to(device).flatten(1))
            else:
                p = model(x.to(device), mask.to(device), ts.to(device))
            preds.append(p.cpu())
            tgts.append(y)
    p, t = torch.cat(preds), torch.cat(tgts)
    mse = F.mse_loss(p, t).item()
    p5 = p.view(-1, 5, 7)
    t5 = t.view(-1, 5, 7)
    groups = {g: float(((p5[:, :, idx] - t5[:, :, idx]) ** 2).mean()) for g, idx in GROUPS.items()}
    return mse, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--train_features", required=True)
    ap.add_argument("--val_features", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--feature_key", default="tokens")
    ap.add_argument("--probe", choices=["attn", "mlp"], default="attn", help="mlp = LARYBench official ActionExpertMLP on flattened tokens")
    ap.add_argument("--meanpool", action="store_true", help="mean-pool each latent frame's tokens to one vector")
    ap.add_argument("--pool4", action="store_true", help="adaptive-pool each latent frame's token grid to 2x2 (DreamX budget)")
    ap.add_argument("--preload", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = FeatureDataset(args.train_manifest, args.train_features, args.dataset, args.feature_key, args.pool4, args.preload, args.meanpool)
    val_ds = FeatureDataset(args.val_manifest, args.val_features, args.dataset, args.feature_key, args.pool4, args.preload, args.meanpool)
    nw = 0 if args.preload else 4
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=nw)

    sample_x, _, _, _ = next(iter(train_ld))
    val_bs = min(512, max(8, 100000 // sample_x.shape[1]))
    val_ld = DataLoader(val_ds, batch_size=val_bs, shuffle=False, collate_fn=collate, num_workers=nw)
    in_dim = sample_x.shape[-1]
    flat = args.probe == "mlp"
    if flat:
        in_dim = sample_x.shape[1] * sample_x.shape[2]
    model = (ActionExpertMLP(in_dim) if flat else AttentiveProbe(in_dim)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"in_dim={in_dim} probe params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_steps = args.epochs * len(train_ld)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * n_steps))))
    best = float("inf")
    log = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y, mask, ts in train_ld:
            if flat:
                p = model(x.to(device).flatten(1))
            else:
                p = model(x.to(device), mask.to(device), ts.to(device))
            loss = F.huber_loss(p, y.to(device), delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        mse, groups = evaluate(model, val_ld, device, flat)
        log.append({"epoch": ep, "val_mse": mse, **{f"mse_{k}": v for k, v in groups.items()}})
        best = min(best, mse)
        print(f"epoch {ep}: val_mse={mse:.4f} groups={ {k: round(v,4) for k,v in groups.items()} } ({time.time()-t0:.0f}s)")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag, "dataset": args.dataset, "seed": args.seed,
                   "best_val_mse": best, "final_val_mse": log[-1]["val_mse"], "log": log,
                   "n_train": len(train_ds), "n_val": len(val_ds), "in_dim": in_dim}, f, indent=2)
    print(f"best val MSE {best:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
