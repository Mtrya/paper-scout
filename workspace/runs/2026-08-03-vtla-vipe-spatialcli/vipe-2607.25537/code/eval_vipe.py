"""VIPE x Physics-IQ: evaluation half for the edited-conditioning-frame generations.

Identical protocol to eval_probe.py (original Physics-IQ pipeline order: motion
masks at NATIVE resolution, then resize to 960x540 official target). Evaluates
each edit condition's 8 videos against the same real testing clips and official
real masks, writing per-edit metrics.json + strips + energy curves so that
orig vs arrow/highlight/sketch can be compared cell-by-cell.

Usage (on the Inspire notebook, cwd = wan22-probe/):
  .venv/bin/python code/eval_vipe.py --edit arrow [--only 0008,0146]
"""

import argparse
import json
import os

import cv2
import numpy as np
from decord import VideoReader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SLUGS = {
    "0008": "ball-hits-duck", "0032": "balls-collide", "0053": "double-cradle",
    "0065": "fill-glass-red-drink", "0089": "liquid-overfill",
    "0140": "paper-smoke", "0146": "roll-behind-box",
    "0182": "unstable-block-stack",
}

REAL = os.path.join(BASE, "data", "split-videos_testing_24FPS",
                    "{sid}_testing-videos_24FPS_perspective-center_take-1_trimmed-{slug}.mp4")
RMASK = os.path.join(BASE, "data", "video-masks_real_24FPS",
                     "{sid}_video-masks_24FPS_perspective-center_take-1_trimmed-{slug}.mp4")

EVAL_SIZE = (960, 540)
KERNEL = np.ones((5, 5), np.uint8)


def read_video(path, size=None, gray=False):
    vr = VideoReader(path)
    frames = []
    for i in range(len(vr)):
        f = vr[i].asnumpy()[:, :, ::-1]
        if size is not None:
            f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        if gray:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(f)
    return frames


def motion_masks(gray_frames):
    blur = [cv2.GaussianBlur(g, (5, 5), 0) for g in gray_frames]
    avg = blur[0].astype("float")
    masks = [np.zeros_like(blur[0])]
    for g in blur[1:]:
        cv2.accumulateWeighted(g, avg, 0.3)
        diff = cv2.absdiff(g, cv2.convertScaleAbs(avg))
        _, b = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, KERNEL)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, KERNEL)
        masks.append(b)
    return masks


def resize_masks(masks, size):
    out = []
    for m in masks:
        if m.dtype == bool:
            m = m.astype(np.uint8) * 255
        out.append(cv2.resize(m, size, interpolation=cv2.INTER_AREA) > 127)
    return out


def energy_curve(gray_frames):
    e = [0.0]
    for a, b in zip(gray_frames[:-1], gray_frames[1:]):
        e.append(float(cv2.absdiff(a, b).mean()))
    return e


def st_iou(gen_masks, real_masks):
    vals = []
    for gb, rb in zip(gen_masks, real_masks):
        union = np.logical_or(gb, rb).sum()
        if union == 0:
            continue
        vals.append(float(np.logical_and(gb, rb).sum() / union))
    return float(np.mean(vals)) if vals else 0.0, len(vals)


def weighted_spatial_iou(gen_masks, real_masks):
    wg = np.mean(gen_masks, axis=0)
    wr = np.mean(real_masks, axis=0)
    num = np.minimum(wg, wr).sum()
    den = np.maximum(wg, wr).sum()
    return float(num / den) if den > 0 else 0.0


def save_strip(real_frames, gen_frames, path, tag):
    idxs = [0, 20, 40, 60, 80, 95]
    n = min(len(real_frames), len(gen_frames))
    rows = []
    for frames in (real_frames, gen_frames):
        row = [frames[min(int(i * (n - 1) / 100), n - 1)] for i in idxs]
        rows.append(np.concatenate(row, axis=1))
    strip = np.concatenate(rows, axis=0)
    cv2.putText(strip, "REAL", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(strip, tag, (8, 26 + strip.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(path, strip)


def save_curve(e_real, e_gen, path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    plt.plot(e_real, label="real", lw=1.5)
    plt.plot(e_gen, label=tag, lw=1.5)
    plt.xlabel("frame (24fps)")
    plt.ylabel("mean |dframe|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", required=True, choices=["arrow", "highlight", "sketch"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else set(SLUGS)
    gen_dir = os.path.join(BASE, "outputs", "gen_vipe", args.edit)
    eval_dir = os.path.join(BASE, "outputs", "eval_vipe", args.edit)
    os.makedirs(eval_dir, exist_ok=True)

    metrics = {}
    for sid in sorted(only):
        slug = SLUGS[sid]
        gen_p = os.path.join(gen_dir, f"{sid}_{slug}.mp4")
        if not os.path.exists(gen_p):
            print(f"[{sid}] missing {gen_p}, skip")
            continue
        gen_native = read_video(gen_p, None)
        real_native = read_video(REAL.format(sid=sid, slug=slug), None)
        rmask_native = read_video(RMASK.format(sid=sid, slug=slug), None, gray=True)
        n = min(len(gen_native), len(real_native), len(rmask_native))

        gen_gray_native = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in gen_native[:n]]
        gmasks = resize_masks(motion_masks(gen_gray_native), EVAL_SIZE)
        rmasks = resize_masks([m > 127 for m in rmask_native[:n]], EVAL_SIZE)

        gen = [cv2.resize(f, EVAL_SIZE, interpolation=cv2.INTER_AREA) for f in gen_native[:n]]
        real = [cv2.resize(f, EVAL_SIZE, interpolation=cv2.INTER_AREA) for f in real_native[:n]]
        gen_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in gen]
        real_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in real]

        st, n_used = st_iou(gmasks, rmasks)
        ws = weighted_spatial_iou(gmasks, rmasks)
        mse = float(np.mean([
            np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
            for a, b in zip(gen, real)]))
        e_real = energy_curve(real_gray)
        e_gen = energy_curve(gen_gray)

        save_strip(real, gen, os.path.join(eval_dir, f"{sid}_{slug}_strip.png"), args.edit.upper())
        save_curve(e_real, e_gen, os.path.join(eval_dir, f"{sid}_{slug}_energy.png"),
                   f"Wan2.2-5B ({args.edit})")
        metrics[sid] = dict(slug=slug, frames=n, frames_iou=n_used,
                            st_iou=round(st, 4), weighted_s_iou=round(ws, 4),
                            mse=round(mse, 5),
                            energy_real_peak=round(max(e_real), 2),
                            energy_gen_peak=round(max(e_gen), 2),
                            energy_gen_mean=round(float(np.mean(e_gen)), 3),
                            energy_real_mean=round(float(np.mean(e_real)), 3))
        print(f"[{sid}] {metrics[sid]}", flush=True)
        del gen_native, real_native, rmask_native, gen_gray_native, gmasks, rmasks

    if metrics:
        avg = {k: round(float(np.mean([m[k] for m in metrics.values()])), 4)
               for k in ("st_iou", "weighted_s_iou", "mse")}
        avg["iq_orig_estimate"] = round(100 * (avg["st_iou"] + avg["weighted_s_iou"]) / 2, 1)
        metrics["_average"] = avg
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
