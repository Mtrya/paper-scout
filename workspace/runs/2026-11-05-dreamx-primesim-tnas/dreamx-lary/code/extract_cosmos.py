#!/usr/bin/env python3
"""Extract Cosmos-Predict2.5-2B features for LARYBench regression samples.

Arms:
  vae: VAE latents of the 2-frame clip -> tokens (T_lat, H*W, 16)
  dit: frozen DiT intermediate features at t=sigma_cond (clean, observed-frames
       convention) -> per-layer tokens (T_lat, H/16*W/16, 2048) for layers
       6,13,20,27 saved as keys layer6..layer27 in one npz.

Input convention for dit arm: x = [z(16), cond_mask(1)] with
concat_padding_mask=True (the model appends the 18th padding-mask channel),
matching the released 2B checkpoint's 18-channel x_embedder (2048x72).
cond_mask=1 marks the observed frames as given (img2vid conditioning);
--cond_mask 0 and --sigma override for convention pilots.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cosmos-predict2.5"))

from cosmos_patch import patch_environment
patch_environment()

VAE_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
VAE_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cosmos-weights")


def load_models(res, layers):
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE_
    from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, SACConfig, CheckpointMode

    vae = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                  attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
    vae.load_state_dict(torch.load(os.path.join(WEIGHTS, "tokenizer.pth"), map_location="cpu", weights_only=False), strict=True)
    vae = vae.to(dtype=torch.bfloat16, device="cuda").eval()

    dit = MiniTrainDIT(
        max_img_h=res, max_img_w=res, max_frames=128,
        in_channels=17, out_channels=16, patch_spatial=2, patch_temporal=1,
        model_channels=2048, num_blocks=28, num_heads=16,
        concat_padding_mask=True, pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop", use_adaln_lora=True, adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=False, atten_backend="torch",
        use_crossattn_projection=True, crossattn_proj_in_channels=100352,
        crossattn_emb_channels=1024, sac_config=SACConfig(mode=CheckpointMode.NONE),
    )
    sd = torch.load(os.path.join(WEIGHTS, "post_trained_ema_bf16.pt"), map_location="cpu", weights_only=False)
    sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "_extra_state" not in k]
    unexpected = [k for k in unexpected if "_extra_state" not in k]
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    dit = dit.to(dtype=torch.bfloat16, device="cuda").eval()

    cr1 = torch.load(os.path.join(WEIGHTS, "cr1_empty_string_text_embeddings.pt"),
                     map_location="cpu", weights_only=False)
    cr1 = cr1.to(device="cuda", dtype=torch.bfloat16)
    if cr1.dim() == 2:
        cr1 = cr1.unsqueeze(0)
    return vae, dit, cr1, layers


def load_pair(src, tgt, res):
    # LARYBench release ships only the stride-5 endpoint frames; the causal VAE
    # needs T = 1 (mod 4), so encode [src, tgt, tgt, tgt, tgt]: latent frame 0
    # = src, latent frame 1 = static tgt tail. The src->tgt transition lives
    # between the two latent frames.
    imgs = []
    for p in (src, tgt):
        img = Image.open(p).convert("RGB").resize((res, res), Image.LANCZOS)
        imgs.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 * 2 - 1)
    frames = [imgs[0], imgs[1], imgs[1], imgs[1], imgs[1]]
    return torch.stack(frames, dim=1)  # (3, 5, H, W)


@torch.no_grad()
def extract_batch(vae, dit, cr1, batch_imgs, args, scale):
    x = batch_imgs.to(device="cuda", dtype=torch.bfloat16)  # (B,3,2,H,W)
    z = vae.encode(x, scale=scale)  # (B,16,T_lat,h,w)
    B, C, T, h, w = z.shape
    if args.arm == "vae":
        toks = z.float().permute(0, 2, 3, 4, 1).reshape(B, T, h * w, C)  # (B,T,HW,16)
        return {"tokens": toks.cpu().half().numpy()}

    if args.sigma > 0:
        g = torch.Generator(device="cuda").manual_seed(args.noise_seed)
        noise = torch.randn(z.shape, generator=g, device="cuda", dtype=torch.bfloat16)
        z_in = z + args.sigma * noise
    else:
        z_in = z
    cond = torch.full((B, 1, T, h, w), float(args.cond_mask), device="cuda", dtype=torch.bfloat16)
    x_in = torch.cat([z_in, cond], dim=1)  # (B,17,T,h,w); model appends padding channel -> 18
    t = torch.full((B,), args.sigma if args.sigma > 0 else 1e-4, device="cuda", dtype=torch.bfloat16)
    padding_mask = torch.zeros(B, 1, h, w, device="cuda", dtype=torch.bfloat16)
    fps = torch.full((B,), 16.0, device="cuda", dtype=torch.bfloat16)
    from cosmos_predict2._src.predict2.conditioner import DataType
    _, feats = dit(x_in, timesteps_B_T=t, crossattn_emb=cr1.expand(B, -1, -1), fps=fps,
                   padding_mask=padding_mask, data_type=DataType.VIDEO,
                   intermediate_feature_ids=args.layers)
    out = {}
    for lid, f in zip(args.layers, feats):
        # f: (B, T*225, 2048) -> (B, T, 225, 2048)
        out[f"layer{lid}"] = f.float().reshape(B, T, -1, f.shape[-1]).cpu().half().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arm", choices=["vae", "dit"], required=True)
    ap.add_argument("--res", type=int, default=240)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20, 27])
    ap.add_argument("--sigma", type=float, default=0.0, help="0 => clean (t=1e-4)")
    ap.add_argument("--cond_mask", type=float, default=1.0)
    ap.add_argument("--noise_seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.iloc[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    scale = [torch.tensor(VAE_MEAN, dtype=torch.bfloat16, device="cuda"),
             1.0 / torch.tensor(VAE_STD, dtype=torch.bfloat16, device="cuda")]
    vae, dit, cr1, layers = load_models(args.res, args.layers)

    todo = []
    for _, r in df.iterrows():
        name = str(r["sample_id"]).replace("/", "_") + ".npz"
        if not os.path.exists(os.path.join(args.out_dir, name)):
            todo.append((name, r["src_img"], r["tgt_img"]))
    print(f"{len(todo)}/{len(df)} samples to extract (skip-existing)")

    t0 = time.time()
    done = 0
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i : i + args.batch_size]
        imgs = torch.stack([load_pair(s, t, args.res) for _, s, t in chunk])
        out = extract_batch(vae, dit, cr1, imgs, args, scale)
        for j, (name, _, _) in enumerate(chunk):
            payload = {k: v[j] for k, v in out.items()}
            np.savez(os.path.join(args.out_dir, name), **payload)
        done += len(chunk)
        if done % 160 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"{done}/{len(todo)}  {el:.0f}s  {done/max(el,1):.1f}/s", flush=True)
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
