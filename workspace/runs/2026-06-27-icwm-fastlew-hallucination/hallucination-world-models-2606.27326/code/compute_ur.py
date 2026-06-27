#!/usr/bin/env python3
"""
Compute the tokenizer round-trip residual (u_r) on real MMBench2 frames using
the released combined-model tokenizer checkpoint.

This is the first of the three label-free hallucination predictors from
"Hallucination in World Models is Predictable and Preventable" (arXiv:2606.27326).

u_r = RMS( z_pred - Encoder(Decoder(z_pred)) )

A latent whose decoded frame falls off the tokenizer manifold produces a large
u_r; the paper uses this as a detector for perceptual hallucination.

Inputs (configurable via CLI):
  --tokenizer_ckpt  path to released tokenizer.pt
  --png_strip       path to an MMBench2 PNG strip (e.g. val/cup-catch-0.png)
  --n_frames        how many consecutive frames to evaluate

Outputs a CSV with per-frame u_r, reconstruction MSE/PSNR, and a simple
"high residual" flag (> median).
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image

# The official model definition is dependency-light (only torch + torchvision).
# A copy of the release model.py is preserved in this directory so the probe is
# self-contained; the original lives at https://github.com/nicklashansen/mmbench2.
from model import Encoder, Decoder, Tokenizer, temporal_patchify, temporal_unpatchify


def _strip_prefix(sd, pfx):
    return {k[len(pfx):] if k.startswith(pfx) else k: v for k, v in sd.items()}


def load_tokenizer(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {}) or {}

    H = int(a.get("H", 224))
    W = int(a.get("W", 224))
    C = int(a.get("C", 3))
    patch = int(a.get("patch", 14))
    d_model = int(a.get("d_model", 512))
    n_heads = int(a.get("n_heads", 8))
    depth = int(a.get("depth", 12))
    n_latents = int(a.get("n_latents", 64))
    d_bottleneck = int(a.get("d_bottleneck", 64))
    dropout = float(a.get("dropout", 0.0))
    mlp_ratio = float(a.get("mlp_ratio", 4.0))
    time_every = int(a.get("time_every", 1))

    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=d_model,
        n_latents=n_latents,
        n_patches=n_patches,
        n_heads=n_heads,
        depth=depth,
        d_bottleneck=d_bottleneck,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    dec = Decoder(
        d_bottleneck=d_bottleneck,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        n_latents=n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
    )
    tok = Tokenizer(enc, dec).to(device)

    sd = ckpt.get("model", ckpt)
    if not isinstance(list(sd.values())[0], torch.Tensor):
        raise KeyError(f"Could not find a state dict in checkpoint keys={list(ckpt.keys())}")
    for pfx in ("_orig_mod.", "module.", "dynamics.", "dyn."):
        sd = _strip_prefix(sd, pfx)
    tok.load_state_dict(sd, strict=True)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    return tok, dict(H=H, W=W, C=C, patch=patch, n_latents=n_latents, d_bottleneck=d_bottleneck)


def load_png_strip(path: str, n_frames: int):
    """Load the first n_frames from a horizontally-concatenated PNG strip."""
    img = read_image(path)  # (3, 224, 224 * N), uint8
    C, H, W_total = img.shape
    frame_w = H  # square frames
    n_total = W_total // frame_w
    n_frames = min(n_frames, n_total)
    frames = img[:, :, : n_frames * frame_w]
    frames = frames.view(C, H, n_frames, frame_w).permute(2, 0, 1, 3)  # (N, C, H, W)
    return frames.unsqueeze(0).float() / 255.0  # (1, N, C, H, W)


def compute_ur_for_frame(tok, frames_btcwh, frame_idx, info):
    """Run encode-decode-encode on one frame and return metrics."""
    H, W, C, patch = info["H"], info["W"], info["C"], info["patch"]
    frame = frames_btcwh[:, frame_idx: frame_idx + 1]  # (1, 1, C, H, W)
    patches = temporal_patchify(frame, patch)  # (1, 1, Np, Dp)

    with torch.no_grad():
        z, _ = tok.encoder(patches)  # (1, 1, n_latents, d_bottleneck)
        recon_patches = tok.decoder(z)  # (1, 1, Np, Dp)
        z_recon, _ = tok.encoder(recon_patches)

    u_r = (z - z_recon).pow(2).mean().sqrt().item()

    recon = temporal_unpatchify(recon_patches, H=H, W=W, C=C, patch=patch)
    mse = F.mse_loss(recon, frame).item()
    psnr = -10.0 * torch.log10(torch.tensor(mse).clamp_min(1e-10)).item()

    # Motion-normalized u_r needs a previous latent; for a strip we use the
    # preceding frame (or zeros for frame 0).
    if frame_idx > 0:
        prev = frames_btcwh[:, frame_idx - 1: frame_idx]
        prev_patches = temporal_patchify(prev, patch)
        with torch.no_grad():
            z_prev, _ = tok.encoder(prev_patches)
        motion = (z - z_prev).pow(2).mean().sqrt().item()
    else:
        motion = 1e-3
    u_r_norm = u_r / max(motion, 1e-3)

    return dict(
        frame=frame_idx,
        u_r=u_r,
        u_r_norm=u_r_norm,
        motion=motion,
        recon_mse=mse,
        recon_psnr=psnr,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_ckpt", required=True)
    p.add_argument("--png_strip", required=True)
    p.add_argument("--n_frames", type=int, default=24)
    p.add_argument("--out_csv", default="ur_results.csv")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tok, info = load_tokenizer(args.tokenizer_ckpt, device)
    print(f"Loaded tokenizer: H={info['H']} W={info['W']} patch={info['patch']} "
          f"n_latents={info['n_latents']} d_bottleneck={info['d_bottleneck']}")

    frames = load_png_strip(args.png_strip, args.n_frames).to(device)
    print(f"Loaded PNG strip: {frames.shape}")

    rows = []
    for i in range(frames.shape[1]):
        rows.append(compute_ur_for_frame(tok, frames, i, info))
        if i % 4 == 0:
            print(f"  frame {i:3d}: u_r={rows[-1]['u_r']:.4f}  "
                  f"u_r_norm={rows[-1]['u_r_norm']:.2f}  "
                  f"PSNR={rows[-1]['recon_psnr']:.2f}")

    median_ur = sorted(r["u_r"] for r in rows)[len(rows) // 2]
    for r in rows:
        r["high_residual"] = int(r["u_r"] > median_ur)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")

    print(f"\nSummary: median u_r={median_ur:.4f}; "
          f"max u_r={max(r['u_r'] for r in rows):.4f}; "
          f"min recon PSNR={min(r['recon_psnr'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
