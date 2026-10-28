#!/usr/bin/env python3
"""RynnLAM camera-vs-content decoupling probe on the released RynnLAM encoder.

For each LIBERO frame pair (t, t+4), we keep f0 fixed and transform the second
frame with synthetic camera motion (translation / rotation / zoom) or real
content change (later frame / different episode). We then measure the relative
change of each descriptor component: motion tokens z^m (512), global action
latent z^a (64), camera latent z^c (32).

Decoupling claim under test (RynnVLA-Latent 2610.34356): camera motion should
mostly move z^c, while manipulation content should mostly move z^a / z^m.
If GRL decoupling works: Delta z^a(camera) << Delta z^a(content), and
Delta z^c(camera) is clearly above the identity floor.

Run: ../scout-gpu/bin/python probe_decoupling.py --device cuda --episodes 6
"""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

STRIDE = 4
COND_TRANSLATE = [(8, 0), (-8, 0), (0, 8), (0, -8), (16, 0), (24, 0)]
COND_ROTATE = [5.0, -5.0, 10.0, -10.0]
COND_ZOOM = [1.1, 1.25]


def load_frames(path, size=224):
    df = pd.read_parquet(path)
    col = "image" if "image" in df.columns else "observation.images.image"
    frames = []
    for b in df[col]:
        if isinstance(b, dict):
            b = b.get("bytes", b.get("data"))
        img = Image.open(io.BytesIO(b)).convert("RGB").resize((size, size), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    return frames


def transform(img, kind, param):
    """img: PIL image; returns transformed PIL image (canvas stays 224x224)."""
    if kind == "translate":
        dx, dy = param
        return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                             resample=Image.BILINEAR, fillcolor=(20, 20, 20))
    if kind == "rotate":
        return img.rotate(param, resample=Image.BILINEAR, fillcolor=(20, 20, 20))
    if kind == "zoom":
        s = param
        w, h = img.size
        nw, nh = int(w * s), int(h * s)
        big = img.resize((nw, nh), Image.BILINEAR)
        left, top = (nw - w) // 2, (nh - h) // 2
        return big.crop((left, top, left + w, top + h))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--pairs-per-ep", type=int, default=3)
    ap.add_argument("--out", default="decoupling_results.json")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RynnVLA-Latent"))
    from rynnlam.inference import RynnLAMEncoder

    enc = RynnLAMEncoder(os.path.join(os.path.dirname(__file__), "..", "models", "RynnLAM.pt"),
                         device=args.device, precision="bf16" if args.device == "cuda" else "fp32")
    rows = []
    for ei in range(args.episodes):
        path = os.path.join(os.path.dirname(__file__), "data", f"episode_{ei:06d}.parquet")
        frames = load_frames(path)
        if len(frames) < 3 * STRIDE + 2:
            continue
        ts = np.linspace(0, len(frames) - 2 * STRIDE - 1, args.pairs_per_ep).astype(int)
        for t in ts:
            f0 = Image.fromarray((frames[t] * 255).astype(np.uint8))
            fT = Image.fromarray((frames[t + STRIDE] * 255).astype(np.uint8))
            fLater = Image.fromarray((frames[t + 2 * STRIDE] * 255).astype(np.uint8))
            fOther = Image.fromarray((frames[(t + 7 * STRIDE) % len(frames)] * 255).astype(np.uint8))
            conditions = [("identity", None, fT), ("content_later", None, fLater),
                          ("content_other", None, fOther)]
            for dx, dy in COND_TRANSLATE:
                conditions.append((f"translate_{dx},{dy}", None, transform(fT, "translate", (dx, dy))))
            for r in COND_ROTATE:
                conditions.append((f"rotate_{r}", None, transform(fT, "rotate", r)))
            for z in COND_ZOOM:
                conditions.append((f"zoom_{z}", None, transform(fT, "zoom", z)))
            base = None
            for name, _, imgB in conditions:
                pair = np.stack([
                    np.asarray(f0, dtype=np.float32) / 255.0,
                    np.asarray(imgB, dtype=np.float32) / 255.0,
                ])[None]
                feats = enc(pair, representation="ktoken_zcam").squeeze(0).numpy()
                zm, za, zc = feats[:512], feats[512:576], feats[576:608]
                if name == "identity":
                    base = (zm, za, zc)
                    deltas = (0.0, 0.0, 0.0)
                else:
                    deltas = tuple(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8))
                                   for a, b in zip((zm, za, zc), base))
                rows.append({"episode": ei, "t": int(t), "condition": name,
                             "d_zm": deltas[0], "d_za": deltas[1], "d_zc": deltas[2]})
        print(f"episode {ei} done ({len(rows)} rows)", flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)
    # summary
    df = pd.DataFrame(rows)
    summ = []
    for name, g in df.groupby("condition"):
        kind = ("identity" if name == "identity" else
                "content" if name.startswith("content") else
                "translate" if name.startswith("translate") else
                "rotate" if name.startswith("rotate") else "zoom")
        summ.append({"condition": name, "kind": kind,
                     "d_zm": float(g.d_zm.mean()), "d_za": float(g.d_za.mean()),
                     "d_zc": float(g.d_zc.mean()), "n": len(g)})
    out = {"rows_summary": summ, "n_rows": len(rows)}
    json.dump(out, open(args.out.replace(".json", "_summary.json"), "w"), indent=1)
    for s in summ:
        print(f"{s['condition']:>18} d_zm={s['d_zm']:.4f} d_za={s['d_za']:.4f} d_zc={s['d_zc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
