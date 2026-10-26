#!/usr/bin/env python
"""Compose the report figures for the Ropedia W0 condition audit.

Each row is one case; the label carries the case name plus its measured numbers, so a reader can see
the metric and the pixels in the same panel. Frames are sampled at 0/20/40/60/80 by default.

usage: python make_figures.py --dir outputs/audit --metrics outputs/metrics.json \
           --rows rice_full,rice_noskel,rice_none --gt_name rice_full --out fig_conditions.png
"""
import argparse, json, os

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

H, W = 480, 640


def read(path, idxs):
    r = imageio.get_reader(path)
    try:
        n = int(r.count_frames())
        out = []
        for i in idxs:
            im = Image.fromarray(r.get_data(min(i, n - 1))).convert("RGB").resize((W, H))
            out.append(np.asarray(im).astype(np.float32) / 255.0)
    finally:
        r.close()
    return out


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def tile(img, text, size=18):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except Exception:
        font = ImageFont.load_default()
    w = int(d.textlength(text, font=font)) + 10 if text else 0
    d.rectangle([0, 0, w, size + 10], fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 255, 255), font=font)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gt_name", default=None)
    ap.add_argument("--frames", default="0,20,40,60,80")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_width", type=int, default=1600)
    args = ap.parse_args()
    idxs = [int(x) for x in args.frames.split(",")]
    met = json.load(open(args.metrics)) if args.metrics else {}
    rows = [r for r in args.rows.split(",") if r]
    label = {}
    for r in rows:
        m = met.get(r)
        corr = m["motion_profile_corr"] if m else None
        corr_text = f"{corr:.2f}" if corr is not None else "n/a"
        label[r] = (f"{r}  PSNR {m['psnr_mean']:.2f}  SSIM {m['ssim_mean']:.3f}  "
                    f"motion x{m['motion_ratio']:.2f}  corr {corr_text}") if m else r
    tiles = []
    for r in rows:
        frames = read(os.path.join(args.dir, f"{r}_pred.mp4"), idxs)
        tiles.append([tile(f, label[r] if i == 0 else "") for i, f in enumerate(frames)])
    if args.gt_name:
        gt = read(os.path.join(args.dir, f"{args.gt_name}_gt.mp4"), idxs)
        tiles.insert(0, [tile(f, f"GT ({args.gt_name})" if i == 0 else "") for i, f in enumerate(gt)])
    h, w = tiles[0][0].size[1], tiles[0][0].size[0]
    cols = max(len(t) for t in tiles)
    canvas = Image.new("RGB", (w * cols, h * len(tiles)), (0, 0, 0))
    for ri, row in enumerate(tiles):
        for ci, im in enumerate(row):
            canvas.paste(im, (ci * w, ri * h))
    if canvas.width > args.max_width:
        ratio = args.max_width / canvas.width
        canvas = canvas.resize((args.max_width, int(canvas.height * ratio)), Image.LANCZOS)
    canvas.save(args.out, optimize=True)
    print(f"[fig] {args.out} {canvas.size} {os.path.getsize(args.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
