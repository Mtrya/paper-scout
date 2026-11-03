"""Probe what the k=1 router selects, on held-out samples.

Outputs a small JSON so the report's mechanism claims are traceable:
- token-level hits near the goal / distractor, selection centre distance;
- a linear cue classifier (goal token vs distractor token) trained and tested
  on disjoint halves;
- a linear goal-position readout from the four selected-token features.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from synth_wam import WAM, render, sample_id, sample_ood, DEVICE

SIZE = 48


def token_coords():
    c = -1 + (torch.arange(6, device=DEVICE) + 0.5) * (2 / 6)
    return torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(36, 2)


def nearest(p, coords):
    d = (p.unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
    return d.argmin(1), d.min(1).values


def probe_stats(model, g, d):
    coords = token_coords()
    with torch.no_grad():
        tokens = model.encoder(render(g, d))
        _, w, _ = model.head(tokens)
    sel = w.argmax(-1)  # B,H
    sel_coords = coords[sel]
    dg = (sel_coords - g[:, None]).norm(dim=-1)
    dd = (sel_coords - d[:, None]).norm(dim=-1)
    return tokens, sel, {
        "sel_goal_frac": (dg < 0.30).float().mean().item(),
        "sel_dist_frac": (dd < 0.30).float().mean().item(),
        "sel_center_dist": sel_coords.norm(dim=-1).mean().item(),
        "sel_goal_mean_dist": dg.mean().item(),
    }


def fit_linear(x, y, steps=800, lr=0.01):
    n, dim = x.shape
    m = torch.nn.Linear(dim, y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.mse_loss(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def r2(pred, y):
    ss_res = ((pred - y) ** 2).sum()
    ss_tot = ((y - y.mean(0, keepdim=True)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = WAM(topk=1).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()
    torch.manual_seed(0)
    g, d = sample_id(1024)
    tokens, sel, stats = probe_stats(model, g, d)
    coords = token_coords()
    gi, _ = nearest(g, coords)
    di, _ = nearest(d, coords)

    # cue classifier on disjoint halves: goal token vs distractor token
    feats = torch.cat([tokens[torch.arange(1024), gi], tokens[torch.arange(1024), di]], 0)
    labels = torch.cat([torch.ones(1024), torch.zeros(1024)]).to(DEVICE)
    perm = torch.randperm(len(feats), device=DEVICE)
    tr, te = perm[:1024], perm[1024:]
    clf = torch.nn.Linear(128, 1).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(800):
        loss = F.binary_cross_entropy_with_logits(clf(feats[tr]).squeeze(), labels[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((clf(feats[te]).squeeze() > 0) == labels[te]).float().mean().item()

    # goal-position readout from the four selected token features
    sel_feats = tokens[torch.arange(1024).unsqueeze(1), sel].reshape(1024, -1)
    perm = torch.randperm(1024, device=DEVICE)
    tr, te = perm[:512], perm[512:]
    head = fit_linear(sel_feats[tr], g[tr])
    with torch.no_grad():
        r2_id = r2(head(sel_feats[te]), g[te])
    g2, d2 = sample_ood(512)
    tokens2, sel2, stats2 = probe_stats(model, g2, d2)
    sel_feats2 = tokens2[torch.arange(512).unsqueeze(1), sel2].reshape(512, -1)
    with torch.no_grad():
        r2_ood = r2(head(sel_feats2), g2)

    out = {
        "n_id": 1024,
        "stats_id": stats,
        "stats_ood": stats2,
        "cue_probe_acc": acc,
        "goal_readout_r2_id": r2_id,
        "goal_readout_r2_ood": r2_ood,
    }
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
