#!/usr/bin/env python3
"""Why do all three arms read out at the right edge? Render & inspect."""
import numpy as np
import torch
import torch.nn.functional as F

from b_toy_wam import (DEVICE, Decoder, DetNet, ZDiff, SIZE, N_FRAMES,
                       readout_last_frame, sample_zdiff, simulate)

CH = " .:-=+*#%@"


def ascii_frame(f):
    return "\n".join("".join(CH[min(9, int(v * 9.5))] for v in row) for row in f)


def show(name, frames):
    print(f"--- {name} ---")
    for t in (0, 8, 15):
        print(f"[frame {t}]")
        print(ascii_frame(frames[t]))
    print("colsum(frame15)=",
          frames[15].sum(0).astype(int).tolist())


# ---- sanity: readout on REAL videos ----
c0 = 15
rng = np.random.default_rng(0)
xs, true_cols = [], []
for _ in range(64):
    f, c = simulate(c0, rng)
    xs.append(f); true_cols.append(c)
xt = torch.tensor(np.stack(xs)).unsqueeze(1).to(DEVICE)
rd = readout_last_frame(xt).cpu().numpy()
true_cols = np.array(true_cols)
print("readout(real) == true final col: %d/64" % (rd == true_cols).sum())
print("  readout[:12]=", rd[:12].tolist())
print("  true   [:12]=", true_cols[:12].tolist())
print("  readout-true histogram:",
      np.bincount((rd - true_cols + 8).astype(int), minlength=17).tolist(), "(index-8 = offset)")

# reference real video at c0=15
np.random.seed(0)
f_ref, c_ref = simulate(c0, np.random.default_rng(7))
print(f"\nreference real video: final col={c_ref}")
show("REAL (simulator)", f_ref)

# eval-style x0 (as in eval_all)
x0 = np.zeros((1, 1, 32, 32), dtype=np.float32)
x0[0, 0, 1:3, c0 - 1:c0 + 2] = 1.0
x0t = torch.tensor(x0).to(DEVICE)
at = torch.tensor([0.0], device=DEVICE)
print("\neval x0 vs real frame0 of a fresh sample:")
f_new, c_new = simulate(c0, np.random.default_rng(11))
print("  |eval_x0 - real_frame0| =", float(np.abs(x0[0, 0] - f_new[0]).sum()))
show("eval x0 (input to all arms)", np.stack([x0[0, 0]] * N_FRAMES))

det = DetNet().to(DEVICE)
det.load_state_dict(torch.load("results/toywam/det.pt", map_location=DEVICE))
det.eval()
with torch.no_grad():
    vd = torch.sigmoid(det(x0t, at))[0, 0].cpu().numpy()
show("DET (mean predictor)", vd)

vck = torch.load("results/toywam/vae.pt", map_location=DEVICE)
dec = Decoder(16).to(DEVICE)
dec.load_state_dict(vck["dec"])
dec.eval()
with torch.no_grad():
    z = torch.randn(4, 16, device=DEVICE)
    vv = torch.sigmoid(dec(z, at.expand(4)))[:, 0].cpu().numpy()
for i in range(2):
    show(f"VAE prior sample {i}", vv[i])

znet = ZDiff(16).to(DEVICE)
znet.load_state_dict(torch.load("results/toywam/zdiff.pt", map_location=DEVICE))
znet.eval()
with torch.no_grad():
    zs = sample_zdiff(znet, dec, x0t.expand(4, 1, 32, 32), at.expand(4), n=1).squeeze(1)
    vz = torch.sigmoid(dec(zs, at.expand(4)))[:, 0].cpu().numpy()
for i in range(2):
    show(f"ZDIFF sample {i}", vz[i])
