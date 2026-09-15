#!/usr/bin/env python3
"""Dump per-arm empirical outcome histograms for action 0, so the report can
state exactly which support columns are covered/missed."""
import json
import numpy as np
import torch

from b_toy_wam import (ACTIONS, DEVICE, Decoder, DetNet, SIZE, ZDiff,
                       readout_last_frame, reference_probs, sample_zdiff,
                       simulate)

Z_DIM = 16
c0 = 15
a = 0
n_roll = 500
ref = reference_probs(c0)

det = DetNet().to(DEVICE)
det.load_state_dict(torch.load("results/toywam/det.pt", map_location=DEVICE))
det.eval()
vck = torch.load("results/toywam/vae.pt", map_location=DEVICE)
dec = Decoder(Z_DIM).to(DEVICE)
dec.load_state_dict(vck["dec"])
dec.eval()
znet = ZDiff(Z_DIM).to(DEVICE)
znet.load_state_dict(torch.load("results/toywam/zdiff.pt", map_location=DEVICE))
znet.eval()

x0 = np.zeros((1, 1, 32, 32), dtype=np.float32)
x0[0, 0, 1:3, c0 - 1:c0 + 2] = 1.0
x0t = torch.tensor(x0).to(DEVICE)
at = torch.tensor([float(a)], device=DEVICE)

rng = np.random.default_rng(1234)
cols = {"det": [], "vae": [], "zdiff": [], "oracle": []}
for i in range(0, n_roll, 64):
    b = min(64, n_roll - i)
    x0b = x0t.expand(b, 1, 32, 32)
    ab = at.expand(b)
    with torch.no_grad():
        cols["det"].append(readout_last_frame(det(x0b, ab)).cpu())
        z = torch.randn(b, Z_DIM, device=DEVICE)
        cols["vae"].append(readout_last_frame(dec(z, ab)).cpu())
        zz = sample_zdiff(znet, dec, x0b, ab, n=1).squeeze(1)
        cols["zdiff"].append(readout_last_frame(dec(zz, ab)).cpu())
    oc = [simulate(c0, rng)[1] for _ in range(b)]
    cols["oracle"].append(torch.tensor(oc))

support = np.nonzero(ref > 0)[0]
print("support columns:", support.tolist())
print("reference probs:", {int(c): round(float(ref[c]), 5) for c in support})
for m, parts in cols.items():
    allc = torch.cat(parts).numpy()
    emp = np.bincount(allc, minlength=SIZE) / len(allc)
    covered = [int(c) for c in support if emp[c] > 0]
    missed = [int(c) for c in support if emp[c] == 0]
    print(f"\n[{m}] covered={covered}")
    print(f"       missed={missed}")
    print("       hist(support)=", {int(c): round(float(emp[c]), 4) for c in support})
