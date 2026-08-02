"""Wan2.2-TI2V-5B physics probe: generation half.

PhiZero (arXiv:2607.28624) reports Physics-IQ Verified IQ-Score 41.2 for its
reason-then-render pipeline, vs 21.2 for the Wan2.2-5B base model it builds on.
This script reproduces the BASE model side of that comparison on 8 Physics-IQ
scenes: I2V from the last frame of each conditioning clip + the original
Physics-IQ prompt, 121 frames (5s @ 24fps).

The 8 scenes pick one per physics failure mode we care about:
  0008 ball-hits-duck       impact transfer (the scene PhiZero itself demos)
  0032 balls-collide        two-body collision
  0053 double-cradle        Newton's cradle (momentum conservation)
  0065 fill-glass-red-drink fluid pouring / volume
  0089 liquid-overfill      fluid overflow
  0140 paper-smoke          smoke / thermodynamics
  0146 roll-behind-box      object permanence under occlusion
  0182 unstable-block-stack statics / toppling

Usage (on the Inspire notebook, cwd = wan22-probe/):
  .venv/bin/python code/gen_probe.py [--only 0008,0146] [--size 1280x704]
"""

import argparse
import json
import logging
import os
import sys

import cv2
import torch
from decord import VideoReader
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "Wan2.2"))

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

SCENES = {
    "0008": ("ball-hits-duck",
             "A light beige coffee table with a small yellow rubber ducky on it. "
             "A mustard yellow couch is in the background. There is a black pipe on "
             "one end of the table and a brown tennis ball rolls out of it towards "
             "the rubber ducky. Static shot with no camera movement."),
    "0032": ("balls-collide",
             "A light-colored wooden tabletop with two pipes at the edges. A blue and "
             "yellow tennis ball roll out of the pipes and towards each other. Static "
             "shot with no camera movement."),
    "0053": ("double-cradle",
             "A Newton's cradle device on the table and two of the metal balls are "
             "held up by a blue handled grabber tool. The claw releases the two balls. "
             "Static shot with no camera movement."),
    "0065": ("fill-glass-red-drink",
             "A glass beverage dispenser filled with a bright red liquid is set up on "
             "a woven basket and is pouring the liquid into a clear glass on a wooden "
             "table. Static shot with no camera movement."),
    "0089": ("liquid-overfill",
             "A bright red liquid being poured from a dispenser into a glass which is "
             "placed on a dark baking tray on a wooden table. Static shot with no "
             "camera movement."),
    "0140": ("paper-smoke",
             "A piece of folded paper is placed on a glass cutting board. The paper is "
             "being burnt and white smoke is emitting from it. Static shot with no "
             "camera movement."),
    "0146": ("roll-behind-box",
             "A small white lampshade is on a light wood surface. A grey tennis ball "
             "rolls out of the black tube sitting on the table and rolls on the table "
             "towards the right. Static shot with no camera movement."),
    "0182": ("unstable-block-stack",
             "A grabber tool carefully placing a blue wooden block on top of a yellow "
             "block which is balanced on a red block forming an L shape. Static shot "
             "with no camera movement."),
}

COND = os.path.join(
    BASE, "data", "split-videos_conditioning_24FPS",
    "{sid}_conditioning-videos_24FPS_perspective-center_take-1_trimmed-{slug}.mp4")
OUT = os.path.join(BASE, "outputs", "gen", "{sid}_{slug}.mp4")
FIRST = os.path.join(BASE, "outputs", "gen", "{sid}_{slug}_init.png")


def last_frame(path):
    # decord: the NGC opencv build lacks the codec for these mp4s
    vr = VideoReader(path)
    frame = vr[len(vr) - 1].asnumpy()  # RGB
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated scene ids")
    ap.add_argument("--size", default="1280x704")
    ap.add_argument("--frame-num", type=int, default=121)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    w, h = map(int, args.size.split("x"))
    only = set(args.only.split(",")) if args.only else set(SCENES)

    cfg = WAN_CONFIGS["ti2v-5B"]
    logging.info("loading WanTI2V pipeline ...")
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=os.path.join(BASE, "models", "Wan2.2-TI2V-5B"),
        device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=False, convert_model_dtype=True)  # dit is fp32 on disk; -> bf16

    results = {}
    for sid in sorted(only):
        slug, prompt = SCENES[sid]
        cond = COND.format(sid=sid, slug=slug)
        out = OUT.format(sid=sid, slug=slug)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        init = last_frame(cond)
        Image.fromarray(init).save(FIRST.format(sid=sid, slug=slug))
        logging.info(f"[{sid}] generating {w}x{h} x{args.frame_num} ...")
        video = pipe.generate(
            prompt,
            img=Image.fromarray(init),
            size=(w, h),
            max_area=w * h,
            frame_num=args.frame_num,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=args.steps,
            guide_scale=5.0,
            seed=args.seed,
            offload_model=True)  # 48GB card OOMs after ~1 scene without it
        save_video(tensor=video[None], save_file=out, fps=24, nrow=1,
                   normalize=True, value_range=(-1, 1))
        results[sid] = out
        logging.info(f"[{sid}] saved {out}")
        torch.cuda.empty_cache()

    with open(os.path.join(BASE, "outputs", "gen", "gen_manifest.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("GEN_DONE")


if __name__ == "__main__":
    main()
