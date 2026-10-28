#!/usr/bin/env python3
"""Post-hoc delta probe: frozen features -> future state DELTA (sT - s0), per saved checkpoint."""
import glob
import json
import os
import torch
from train import Renderer, sample_batch
from model_visual import VisualPolicy


@torch.no_grad()
def main():
    device = "cuda"
    eps = torch.load("data/demos_probe.pt", weights_only=False)
    stats = torch.load("data/stats.pt", weights_only=False)
    renderer = Renderer(device)
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    out = {}
    for ck in sorted(glob.glob("out/*_seed*.pt")):
        name = os.path.basename(ck).replace(".pt", "")
        model = VisualPolicy().to(device)
        model.load_state_dict(torch.load(ck, weights_only=False), strict=False)
        model.eval()
        gen = torch.Generator().manual_seed(99)

        def collect(n):
            idx = torch.randint(0, len(eps), (n,), generator=gen).tolist()
            ts = [int(torch.randint(0, len(eps[i]["actions"]) - 8, (1,), generator=gen)) for i in idx]
            s0 = torch.stack([eps[i]["states"][t] for i, t in zip(idx, ts)]).to(device)
            sT = torch.stack([eps[i]["states"][t + 8] for i, t in zip(idx, ts)]).to(device)
            f0 = renderer(s0)
            feat = model.foresight_features(f0).mean(dim=1).float()
            y = ((sT - s0) / std).float()
            return feat, y

        Xf, Yf = collect(2000)
        Xv, Yv = collect(1000)
        Xf = torch.cat([Xf, torch.ones(2000, 1, device=device)], 1)
        Xv = torch.cat([Xv, torch.ones(1000, 1, device=device)], 1)
        lam = 1e-2 * Xf.shape[0]
        W = torch.linalg.solve(Xf.T @ Xf + lam * torch.eye(Xf.shape[1], device=device), Xf.T @ Yf)
        pred = Xv @ W
        ss_res = ((Yv - pred) ** 2).sum(0)
        ss_tot = ((Yv - Yv.mean(0)) ** 2).sum(0)
        r2 = 1 - (ss_res / ss_tot.clamp_min(1e-9)).mean()
        out[name] = float(r2)
        print(f"{name}: delta R2 {r2:.3f}", flush=True)
    json.dump(out, open("delta_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
