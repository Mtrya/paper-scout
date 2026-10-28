#!/usr/bin/env python
"""How loud is the condition branch, relative to the video tokens it is added to?

`patchify` computes h(0) = P_vid(z_tau) + P_cam(z_cam) + P_skel(z_skel) at every denoising step,
so the two condition branches are a fixed additive bias on top of whatever the video latent carries.
This script measures, using the released weights and the Wan2.2 VAE, without diffusion sampling:

  ratio_*        ||P_* z_*|| / ||P_vid z|| for z ~ N(0,1), matching the initial sampling noise
  gain_cam       ||P_cam z_cam(A) - P_cam z_cam(B)|| / ||P_cam z_cam(A)||
                 for A = the clip's own camera and B = --alt_camera, both truncated to --frames
  gain_skel      same for a skeleton shifted by 40 px

usage: python probe_branch_scale.py --ckpt weights/w0-5b-midtrain.safetensors \
           --vae models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth --clip examples/agibot_rice_cooker \
           --alt_camera variants/synthcam_RICE/data.npz --alt_skeleton variants/skelshift_RICE/skeleton.mp4
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import imageio
from PIL import Image
from safetensors import safe_open

from ropedia_w0.models.wan_video_vae import WanVideoVAE38
from ropedia_w0.utils.data.camera_plucker import compute_plucker_from_camera_params

H, W = 480, 640


def load_vae(path, device):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" in sd:
        sd = sd["model_state"]
    sd = {("model." + k if not k.startswith("model.") else k): v for k, v in sd.items()}
    vae = WanVideoVAE38(z_dim=48, dim=160).eval()
    vae.load_state_dict(sd, strict=False)
    return vae.to(device=device, dtype=torch.float32)


def branch_weights(ckpt):
    out = {}
    with safe_open(ckpt, framework="pt") as f:
        for k in ("camera_patch_embed.weight", "skeleton_patch_embed.weight", "patch_embedding.weight"):
            out[k] = f.get_tensor(k).float()
    return out


def conv3d_apply(weight, z):
    """Apply a Conv3d(kernel=stride=patch, bias=False) weight to a (B,C,T,H,W) tensor.

    The released branches use a kernel depth of 1, so the temporal axis is untouched and the same
    operation is a 2x2 conv2d applied per frame (fp32 conv3d with this kernel has no CUDA path).
    """
    import torch.nn.functional as F
    B, C, T, H, W = z.shape
    if weight.shape[2] == 1:
        w2 = weight[:, :, 0]
        out = F.conv2d(z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), w2,
                       stride=(w2.shape[2], w2.shape[3]))
        return out.reshape(B, T, -1, out.shape[-2], out.shape[-1]).permute(0, 2, 1, 3, 4)
    return F.conv3d(z, weight, stride=(weight.shape[2], weight.shape[3], weight.shape[4]))


def load_skel(path, n):
    r = imageio.get_reader(path)
    frames = []
    try:
        for i in range(n):
            frames.append(Image.fromarray(r.get_data(i)).convert("RGB").resize((W, H)))
    finally:
        r.close()
    arr = np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in frames])
    return (torch.from_numpy(arr).permute(3, 0, 1, 2) * 2 - 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae", default="models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--alt_camera", default=None)
    ap.add_argument("--alt_skeleton", default=None)
    ap.add_argument("--frames", type=int, default=21)   # 21 frames -> 6 latent frames, cheap
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    vae = load_vae(args.vae, dev)
    bw = {k: v.to(dev) for k, v in branch_weights(args.ckpt).items()}
    print(f"[ckpt] cam {tuple(bw['camera_patch_embed.weight'].shape)}  "
          f"skel {tuple(bw['skeleton_patch_embed.weight'].shape)}  "
          f"vid {tuple(bw['patch_embedding.weight'].shape)}", flush=True)

    n = args.frames
    pl = compute_plucker_from_camera_params(os.path.join(args.clip, "data.npz"), H, W,
                                            num_frames=n, dtype=torch.float32)
    z_cam = vae.encode(pl.permute(1, 0, 2, 3).unsqueeze(0)[:, :3].to(dev), device=dev)
    z_dir = vae.encode(pl.permute(1, 0, 2, 3).unsqueeze(0)[:, 3:].to(dev), device=dev)
    z_cam_full = torch.cat([z_cam, z_dir], dim=1)                     # (1,96,T',h,w)
    z_skel = vae.encode(load_skel(os.path.join(args.clip, "skeleton.mp4"), n).to(dev), device=dev)

    B, C, T, h, w = z_cam_full.shape
    torch.manual_seed(0)
    z_noise = torch.randn(1, 48, T, h, w, device=dev)                  # what the DiT sees at t = T


    with torch.no_grad():
        out_cam = conv3d_apply(bw["camera_patch_embed.weight"], z_cam_full.to(dtype))
        out_skel = conv3d_apply(bw["skeleton_patch_embed.weight"], z_skel.to(dtype))
        vid_noise = conv3d_apply(bw["patch_embedding.weight"], z_noise)
        rms = lambda t: t.pow(2).mean().sqrt().item()
        print(f"[tokens] {tuple(out_cam.shape)}  rms cam={rms(out_cam):.4f} skel={rms(out_skel):.4f} "
              f"vid(noise)={rms(vid_noise):.4f}", flush=True)
        print(f"[ratio] cam/vid(noise) = {rms(out_cam) / rms(vid_noise):.4f}   "
              f"skel/vid(noise) = {rms(out_skel) / rms(vid_noise):.4f}", flush=True)
        # relative to the mean video token norm (not the deviation), which is what a token sees
        vid_mean_norm = vid_noise.pow(2).sum(-1).mean().sqrt().item()
        cam_mean_norm = out_cam.pow(2).sum(-1).mean().sqrt().item()
        print(f"[token-norm] |P_cam z| / |P_vid z_noise| = {cam_mean_norm / vid_mean_norm:.4f}", flush=True)

        if args.alt_camera:
            pl2 = compute_plucker_from_camera_params(args.alt_camera, H, W, num_frames=n, dtype=torch.float32)
            m2 = vae.encode(pl2.permute(1, 0, 2, 3).unsqueeze(0)[:, :3].to(dev), device=dev)
            d2 = vae.encode(pl2.permute(1, 0, 2, 3).unsqueeze(0)[:, 3:].to(dev), device=dev)
            out_cam2 = conv3d_apply(bw["camera_patch_embed.weight"], torch.cat([m2, d2], 1).to(dtype))
            diff = (out_cam2 - out_cam)
            cos = torch.nn.functional.cosine_similarity(out_cam2.flatten(), out_cam.flatten(), 0).item()
            print(f"[gain] camera (own vs alternative, first {n} frames): |dP|/|P| = {rms(diff) / rms(out_cam):.4f}"
                  f"   cos = {cos:.4f}   |dP|/|P_vid(noise)| = {rms(diff) / rms(vid_noise):.4f}", flush=True)

        if args.alt_skeleton:
            z_skel2 = vae.encode(load_skel(args.alt_skeleton, n).to(dev), device=dev)
            out_skel2 = conv3d_apply(bw["skeleton_patch_embed.weight"], z_skel2.to(dtype))
            diff = out_skel2 - out_skel
            cos = torch.nn.functional.cosine_similarity(out_skel2.flatten(), out_skel.flatten(), 0).item()
            print(f"[gain] skeleton (+40 px): |dP|/|P| = {rms(diff) / rms(out_skel):.4f}   cos = {cos:.4f}"
                  f"   |dP|/|P_vid(noise)| = {rms(diff) / rms(vid_noise):.4f}", flush=True)


if __name__ == "__main__":
    main()
