#!/usr/bin/env python3
"""WorldCycle-style reversible-cycle probe on ABot-World-0.

Runs per-block action protocols on the streaming inference path and saves
per-protocol videos + per-block keyframes for cycle-consistency analysis.

Protocols (block = 12 output frames @12fps):
  p0_yaw_base        J x4                         (functional baseline)
  p1_trans_cycle     W x4 -> S x4                 (translation inverse cycle)
  p2_yaw_cycle       J x4 -> L x4                 (yaw inverse cycle; prefix identical to p0)
  p3_repeat_cycle    (W x2 -> S x2) x4            (repeated cycles, phase-aligned drift)
  p4_long_cycle      W x8 -> S x8                 (forward leg exceeds ~7s rolling window)
  p5_return_then_yaw W x4 -> S x4 -> J x4         (functional state equivalence after return)
"""
import os, sys, json, hashlib, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import torch
import imageio

# Stub lightx2v_kernel (only needed by fp4/fp8 quant paths we never touch).
import types as _types
def _stub_factory(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    def _stub(*a, **k):
        raise RuntimeError(f"lightx2v_kernel stubbed: {name} called")
    return _stub
for _name in ("lightx2v_kernel", "lightx2v_kernel.gemm"):
    _m = _types.ModuleType(_name)
    _m.__getattr__ = _stub_factory
    sys.modules.setdefault(_name, _m)

from web_client.config import KEY_ORDER, STREAM_HEIGHT, STREAM_WIDTH, VIDEO_FPS
from web_client.pipeline_loader import get_pipeline, decode_block_to_frames

PROTOCOLS = {
    "p0_yaw_base": ["J"] * 4,
    "p1_trans_cycle": ["W"] * 4 + ["S"] * 4,
    "p2_yaw_cycle": ["J"] * 4 + ["L"] * 4,
    "p3_repeat_cycle": (["W"] * 2 + ["S"] * 2) * 4,
    "p4_long_cycle": ["W"] * 8 + ["S"] * 8,
    "p5_return_then_yaw": ["W"] * 4 + ["S"] * 4 + ["J"] * 4,
}

REF_IMAGE = "web_client/datasets/images/example.png"
OUT_DIR = Path(os.environ.get("CYCLE_OUT", "outputs/cycle_probe"))


def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    pipeline, config, device = get_pipeline(
        vae_type="taew2_2", use_fp8_gemm=False, quant_type=None
    )
    dtype = torch.bfloat16
    num_fpb = int(pipeline.num_frame_per_block)
    _vae = pipeline.encoder if pipeline.encoder is not None else pipeline.vae
    up = getattr(_vae, "upsampling_factor", 8)
    latent_shape = (1, num_fpb, _vae.z_dim, STREAM_HEIGHT // up, STREAM_WIDTH // up)
    print(f"[probe] num_fpb={num_fpb} latent_shape={latent_shape}", flush=True)

    img_hash = hashlib.md5(Path(REF_IMAGE).read_bytes()).hexdigest()[:16]
    ref_cache_dir = str(_ROOT / "outputs" / "ref_image_cache" / img_hash)
    assert Path(ref_cache_dir).is_dir(), f"ref cache missing: {ref_cache_dir}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for name, keys_seq in PROTOCOLS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        pipeline.set_prompts(["| unknown |"], device=device)
        pipeline.set_ref_latent_mask_from_exists_paths(ref_dir=ref_cache_dir, device=device)
        pipeline.reset_stream(1, dtype=dtype, device=device, initial_latent=None)

        frames_all = []
        block_last_frames = []  # last frame of each block (uint8 HxWx3)
        for b, key in enumerate(keys_seq):
            noise_block = torch.randn(latent_shape, device=device, dtype=dtype)
            if b == 0:
                pipeline.set_first_frame_latent(
                    REF_IMAGE, height=STREAM_HEIGHT, width=STREAM_WIDTH, device=device
                )
            pipeline.set_act(
                {k: (k == key) for k in KEY_ORDER},
                height=STREAM_HEIGHT, width=STREAM_WIDTH,
                num_frames=num_fpb, device=device,
            )
            lat = pipeline.generate_next_block(noise_block)
            block_frames = decode_block_to_frames(pipeline, lat)
            frames_all.extend(block_frames)
            block_last_frames.append(block_frames[-1])
            print(f"[{name}] block {b + 1}/{len(keys_seq)} key={key} done", flush=True)

        vid_path = OUT_DIR / f"{name}.mp4"
        writer = imageio.get_writer(str(vid_path), fps=VIDEO_FPS, format="FFMPEG",
                                    codec="libx264",
                                    ffmpeg_params=["-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p"])
        for f in frames_all:
            writer.append_data(f)
        writer.close()

        kf_dir = OUT_DIR / f"{name}_blocks"
        kf_dir.mkdir(exist_ok=True)
        for i, bf in enumerate(block_last_frames):
            imageio.imwrite(str(kf_dir / f"block_{i:03d}.jpg"), bf, quality=90)

        manifest[name] = {
            "keys": keys_seq,
            "num_blocks": len(keys_seq),
            "frames": len(frames_all),
            "video": str(vid_path),
            "runtime_s": round(time.time() - t0, 1),
        }
        with open(OUT_DIR / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[{name}] done in {manifest[name]['runtime_s']}s", flush=True)

    print("ALL_PROTOCOLS_DONE", flush=True)


if __name__ == "__main__":
    main()
