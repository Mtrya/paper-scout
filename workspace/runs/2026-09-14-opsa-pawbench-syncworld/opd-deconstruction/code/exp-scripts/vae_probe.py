#!/usr/bin/env python3
"""Diagnose the toy-VAE: is z carrying outcome information at all?

Measures on the trained results/toywam/vae.pt:
  A) KL per dim / total                 (collapse check)
  B) reconstruction BCE with mu / prior / posterior-sample
  C) linear R^2  z(=mu) -> true final column
  D) spread of readout columns when sampling z ~ N(0,I) at eval (the eval path)
  E) spread when decoding the posterior mean of real videos (upper bound:
     does the latent+decoder COMBINATION represent the outcome at all)
"""
import numpy as np
import torch
import torch.nn.functional as F

from b_toy_wam import (ACTIONS, DEVICE, Encoder, Decoder, SIZE,
                       make_batch, readout_last_frame, simulate)

Z_DIM = 16
CK = "results/toywam/vae.pt"

ck = torch.load(CK, map_location=DEVICE)
enc = Encoder(Z_DIM).to(DEVICE)
dec = Decoder(Z_DIM).to(DEVICE)
enc.load_state_dict(ck["enc"])
dec.load_state_dict(ck["dec"])
enc.eval(); dec.eval()

rng = np.random.default_rng(0)
x, a, cols = make_batch(512, rng)
cols_np = cols.cpu().numpy()

with torch.no_grad():
    mu, lv = enc(x)
    kl_dim = -0.5 * (1 + lv - mu ** 2 - lv.exp())        # (B, 16)
    print("== A) KL ==")
    print("per-dim mean:", np.round(kl_dim.mean(0).cpu().numpy(), 4).tolist())
    print("total mean: %.4f" % kl_dim.sum(1).mean().item())
    print("mu std per-dim:", np.round(mu.std(0).cpu().numpy(), 4).tolist())
    print("lv mean per-dim:", np.round(lv.mean(0).cpu().numpy(), 4).tolist())

    print("== B) reconstruction BCE ==")
    z_prior = torch.randn_like(mu)
    z_post = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
    for name, z in (("mu", mu), ("prior", z_prior), ("post", z_post)):
        rec = F.binary_cross_entropy_with_logits(dec(z, a), x).item()
        print(f"  rec[{name}] = {rec:.5f}")

    print("== C) linear probe z(mu) -> final column ==")
    Zm = mu.cpu().numpy()
    Z1 = np.hstack([Zm, np.ones((len(Zm), 1))])
    w, *_ = np.linalg.lstsq(Z1, cols_np.astype(np.float64), rcond=None)
    pred = Z1 @ w
    y = cols_np.astype(np.float64)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  R2 = {r2:.4f}")

    print("== D) z ~ N(0,I) sampling (the eval path) ==")
    for act in (0, 4):
        c0 = 15 + act
        x0 = np.zeros((1, 1, 32, 32), dtype=np.float32)
        x0[0, 0, 1:3, c0 - 1:c0 + 2] = 1.0
        x0t = torch.tensor(x0).to(DEVICE)
        at = torch.tensor([float(act)], device=DEVICE)
        x0b = x0t.expand(256, 1, 32, 32)
        ab = at.expand(256)
        z = torch.randn(256, Z_DIM, device=DEVICE)
        vv = dec(z, ab)
        cs = readout_last_frame(vv).cpu().numpy()
        hist = np.bincount(cs, minlength=SIZE)
        print(f"  action {act}: distinct={len(set(cs.tolist()))} "
              f"hist={hist[hist > 0].tolist()} cols={np.nonzero(hist)[0].tolist()}")

    print("== E) decode posterior-mean of REAL videos (upper bound) ==")
    for act in (0, 4):
        c0 = 15 + act
        xs = []
        for _ in range(512):
            f, _c = simulate(c0, rng)
            xs.append(f)
        xr = torch.tensor(np.stack(xs)).unsqueeze(1).to(DEVICE)
        ar = torch.full((512,), float(act), device=DEVICE)
        mu_r, _ = enc(xr)
        vr = dec(mu_r, ar)
        cs = readout_last_frame(vr).cpu().numpy()
        hist = np.bincount(cs, minlength=SIZE)
        print(f"  action {act}: distinct={len(set(cs.tolist()))} "
              f"cols={np.nonzero(hist)[0].tolist()} counts={hist[hist > 0].tolist()}")
        # how well does mu decode reproduce the true video of each sample?
        rec = F.binary_cross_entropy_with_logits(vr, xr).item()
        print(f"  action {act}: BCE(dec(mu(real)), real) = {rec:.5f}")
print("done")
