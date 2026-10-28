#!/usr/bin/env python
"""Probe what the Wan2.2 video VAE preserves of the Ropedia W0 conditions.

The camera Pluecker map and the rendered skeleton video enter the DiT through the video VAE
(48 latent channels, spatial stride 16, temporal stride 4) and are added to the video tokens by a
zero-initialised Conv3d. This script measures the information that reaches that Conv3d:

  1. round-trip the skeleton video through the VAE and inspect what survives visually;
  2. shift the skeleton by 4/8/16/32 px and measure how far the latent moves, relative to
     (a) the distance from a blank condition and (b) the distance between two real conditions;
  3. report the same for the Pluecker map, plus the fact that the moment channels vanish for a
     static camera (o = 0 in the first-frame reference frame).

usage: python probe_vae_condition.py --clip examples/agibot_rice_cooker --out /tmp/probe
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import imageio
from PIL import Image

from ropedia_w0.models.wan_video_vae import WanVideoVAE38
from ropedia_w0.utils.data.camera_plucker import compute_plucker_from_camera_params

H, W = 480, 640


def load_vae(path, device="cuda", dtype=torch.float32):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" in sd:
        sd = sd["model_state"]
    sd = {("model." + k if not k.startswith("model.") else k): v for k, v in sd.items()}
    vae = WanVideoVAE38(z_dim=48, dim=160).eval()
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"[vae] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return vae.to(device=device, dtype=dtype), dtype


def read_mp4(path, idxs, size=(W, H)):
    r = imageio.get_reader(path)
    out = []
    try:
        for i in idxs:
            out.append(Image.fromarray(r.get_data(i)).convert("RGB").resize(size))
    finally:
        r.close()
    return out


def to_tensor(frames):
    """list[PIL] -> (1,3,T,H,W) in [-1,1], the Wan VAE input convention."""
    arr = np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in frames])
    x = torch.from_numpy(arr).permute(3, 0, 1, 2) * 2 - 1
    return x.unsqueeze(0)


def to_uint8(video):
    x = video[0].detach().float().cpu()
    x = ((x + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0).numpy()
    return (x * 255).round().astype(np.uint8)


def ink_mask(u8, thr=60):
    return [f.max(axis=2) > thr for f in u8]


def shift_frames(frames, dx, dy=0):
    out = []
    for f in frames:
        im = Image.new("RGB", f.size, (0, 0, 0))
        im.paste(f, (dx, dy))
        out.append(im)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default="models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--start", type=int, default=0, help="first source frame of the window")
    ap.add_argument("--tiled", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae, dtype = load_vae(args.vae, dev)

    idxs = list(range(args.start, args.start + args.frames))
    frames = read_mp4(os.path.join(args.clip, "skeleton.mp4"), idxs)
    skel = to_tensor(frames).to(device=dev, dtype=dtype)
    print(f"[in] skeleton {tuple(skel.shape)}  (B,C,T,H,W)", flush=True)

    with torch.no_grad():
        z_skel = vae.encode(skel, device=dev, tiled=args.tiled)
        rec_skel = vae.decode(z_skel, device=dev, tiled=args.tiled)
    B, C, T, h, w = z_skel.shape
    print(f"[latent] skeleton {tuple(z_skel.shape)}  C={C} T={T} h={h} w={w}  "
          f"(spatial stride {W // w}x, temporal {args.frames}/{T})", flush=True)

    in_u8, rec_u8 = to_uint8(skel), to_uint8(rec_skel)
    m_in, m_rec = ink_mask(in_u8), ink_mask(rec_u8)
    iou = []
    for a, b in zip(m_in, m_rec):
        u = (a | b).sum()
        iou.append(float((a & b).sum() / u) if u else 1.0)
    print(f"[roundtrip] ink fraction in={np.mean([m.mean() for m in m_in]):.5f} "
          f"rec={np.mean([m.mean() for m in m_rec]):.5f}  mean ink IoU={np.mean(iou):.3f}", flush=True)
    # how much of the finger structure survives: connected components per hand region
    try:
        import scipy.ndimage as ndi
        for i in (0, len(idxs) // 2):
            ci = ndi.label(m_in[i])[1]
            cr = ndi.label(m_rec[i])[1]
            print(f"  frame {i}: connected ink components  in={ci} rec={cr}", flush=True)
    except Exception as e:
        print(f"  [warn] scipy ndimage unavailable: {e}", flush=True)

    # Blank reference: all-black video, i.e. "a condition channel exists but carries nothing".
    with torch.no_grad():
        z_blank = vae.encode(torch.zeros_like(skel), device=dev, tiled=args.tiled)
    base_scale = (z_skel - z_blank).pow(2).mean().sqrt().item()
    print(f"[latent] ||z_skeleton - z_blank|| = {base_scale:.4f}   "
          f"||z_skeleton|| = {z_skel.pow(2).mean().sqrt().item():.4f}", flush=True)

    # Shift sensitivity: how far does the latent move when the skeleton moves dx pixels?
    for dx in (2, 4, 8, 16, 32):
        sh = shift_frames(frames, dx)
        with torch.no_grad():
            z_sh = vae.encode(to_tensor(sh).to(device=dev, dtype=dtype), device=dev, tiled=args.tiled)
        d = (z_sh - z_skel).pow(2).mean().sqrt().item()
        cos = torch.nn.functional.cosine_similarity(z_sh.flatten(), z_skel.flatten(), dim=0).item()
        print(f"[shift] dx={dx:3d}px  ||dz||/||z_skeleton-z_blank|| = {d / base_scale:.3f}   cos={cos:.4f}", flush=True)

    # Pluecker: how much of the map is nonzero, and does the moment vanish for a static camera?
    plucker = compute_plucker_from_camera_params(
        os.path.join(args.clip, "data.npz"), H, W, num_frames=args.frames, dtype=torch.float32)
    p = plucker.permute(1, 0, 2, 3).unsqueeze(0)          # (1,6,T,H,W)
    mom, direc = p[:, :3], p[:, 3:]
    print(f"[plucker] |moment| mean={mom.abs().mean():.6f} max={mom.abs().max():.6f}   "
          f"|direction| mean={direc.abs().mean():.4f}   "
          f"frame0 moment all-zero={bool((mom[:, :, 0].abs().max() == 0).item())}", flush=True)
    with torch.no_grad():
        z_mom = vae.encode(mom.to(device=dev, dtype=dtype), device=dev, tiled=args.tiled)
        z_dir = vae.encode(direc.to(device=dev, dtype=dtype), device=dev, tiled=args.tiled)
        z_mom_blank = vae.encode(torch.zeros_like(mom).to(device=dev, dtype=dtype), device=dev, tiled=args.tiled)
    print(f"[latent] moment  ||z-z_blank||={((z_mom - z_mom_blank).pow(2).mean().sqrt()).item():.5f}  "
          f"||z||={z_mom.pow(2).mean().sqrt().item():.4f}   "
          f"direction ||z||={z_dir.pow(2).mean().sqrt().item():.4f}", flush=True)
    print(f"[latent] condition energy (RMS of the additive branch input): "
          f"moment={((z_mom - z_mom_blank).pow(2).mean().sqrt()).item():.4f} "
          f"direction={(z_dir - z_mom_blank).pow(2).mean().sqrt().item():.4f} "
          f"skeleton={base_scale:.4f}", flush=True)

    for i in (0, len(idxs) // 2, len(idxs) - 1):
        pair = np.concatenate([in_u8[i], rec_u8[i]], axis=1)
        Image.fromarray(pair).save(os.path.join(args.out, f"skeleton_roundtrip_f{i:03d}.png"))
    mom_v = mom[0].detach().float().cpu()
    mom_v = (mom_v - mom_v.min()) / (mom_v.max() - mom_v.min() + 1e-8)
    Image.fromarray((mom_v.permute(1, 2, 3, 0)[0].numpy() * 255).astype(np.uint8)).save(
        os.path.join(args.out, "plucker_moment_f000.png"))
    print("[done]", args.out, flush=True)


if __name__ == "__main__":
    main()
