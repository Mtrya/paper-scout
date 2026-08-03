"""Wan2.2-TI2V-5B VIPE probe: generation over edited conditioning frames.

Same protocol as gen_probe.py (runs/2026-08-02): I2V from the (edited) last
conditioning frame + the original Physics-IQ prompt, 121 frames 1280x704,
seed 42, offload_model=True. Only the init image changes.

Usage (on the Inspire notebook, cwd = wan22-probe/):
  .venv/bin/python code/gen_vipe.py --edit arrow [--only 0008,0146]
"""

import argparse
import json
import logging
import os
import sys

import torch
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

EDIT = os.path.join(BASE, "outputs", "vipe_edits", "{sid}_{slug}_{edit}.png")
OUT = os.path.join(BASE, "outputs", "gen_vipe", "{edit}", "{sid}_{slug}.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", required=True, choices=["arrow", "highlight", "sketch"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--size", default="1280x704")
    ap.add_argument("--frame-num", type=int, default=121)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    w, h = map(int, args.size.split("x"))
    only = set(args.only.split(",")) if args.only else set(SCENES)

    cfg = WAN_CONFIGS["ti2v-5B"]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("loading WanTI2V pipeline ...")
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=os.path.join(BASE, "models", "Wan2.2-TI2V-5B"),
        device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=False, convert_model_dtype=True)

    results = {}
    for sid in sorted(only):
        slug, prompt = SCENES[sid]
        edit_path = EDIT.format(sid=sid, slug=slug, edit=args.edit)
        out = OUT.format(sid=sid, slug=slug, edit=args.edit)
        if os.path.exists(out):
            logging.info(f"[{sid}] edit={args.edit} exists, skip")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        img = Image.open(edit_path).convert("RGB")
        logging.info(f"[{sid}] edit={args.edit} generating {w}x{h} x{args.frame_num} ...")
        video = pipe.generate(
            prompt,
            img=img,
            size=(w, h),
            max_area=w * h,
            frame_num=args.frame_num,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=args.steps,
            guide_scale=5.0,
            seed=args.seed,
            offload_model=True)
        save_video(tensor=video[None], save_file=out, fps=24, nrow=1,
                   normalize=True, value_range=(-1, 1))
        results[sid] = out
        logging.info(f"[{sid}] saved {out}")
        torch.cuda.empty_cache()

    manifest = os.path.join(BASE, "outputs", "gen_vipe", args.edit,
                            "gen_manifest.json")
    with open(manifest, "w") as f:
        json.dump(results, f, indent=2)
    print("GEN_DONE")


if __name__ == "__main__":
    main()
