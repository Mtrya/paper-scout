"""VIPE x Physics-IQ: mechanically edited conditioning frames.

VIPE (arXiv:2607.25537) shows that editing the input task image (sketch ->
photorealistic) systematically improves video-model reasoning on ABSTRACT
inputs. Our Physics-IQ probe (runs/2026-08-02) found Wan2.2-5B's failure mode
is "moves plenty, but moves without consequences". This script asks the
converse question on REAL inputs: can edits to the conditioning frame change
the physical consequences the model generates?

Three edit conditions per scene:
  arrow     lime arrow(s) showing the physically correct motion (a visual
            scaffold of the consequence)
  highlight lime ellipse around the moving object(s) at the conditioning
            boundary -- attention guidance only, no direction
  sketch    cv2.pencilSketch: real -> sketch, the REVERSE of VIPE's
            sketch -> real direction; expected to degrade if VIPE's
            directionality is real

Provenance note: arrow positions/directions and highlight placements are
hand-verified against the ground-truth testing clips (MANUAL table below).
A fully mechanical optical-flow estimator (estimate_motion) was tried first;
its raw output is kept in the manifest as provenance, but it misplaces hints
on scenes with static lead-ins (0146), tool transients (0053) and
path-smearing fast movers (0008), so the hand-verified table is authoritative.
The privileged information is used ONLY to place the hint, never in
generation: prompts, seed, sampler and protocol are identical to the
unedited baseline.

Runs on the Inspire notebook, cwd = wan22-probe/.
  .venv/bin/python code/make_vipe_edits.py [--only 0008,0146]
"""

import argparse
import json
import os

import cv2
import numpy as np
from decord import VideoReader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENES = {
    "0008": "ball-hits-duck", "0032": "balls-collide", "0053": "double-cradle",
    "0065": "fill-glass-red-drink", "0089": "liquid-overfill",
    "0140": "paper-smoke", "0146": "roll-behind-box",
    "0182": "unstable-block-stack",
}
COND = os.path.join(
    BASE, "data", "split-videos_conditioning_24FPS",
    "{sid}_conditioning-videos_24FPS_perspective-center_take-1_trimmed-{slug}.mp4")
REAL = os.path.join(
    BASE, "data", "split-videos_testing_24FPS",
    "{sid}_testing-videos_24FPS_perspective-center_take-1_trimmed-{slug}.mp4")
OUTDIR = os.path.join(BASE, "outputs", "vipe_edits")

FLOW_SIZE = (960, 540)
LIME = (50, 255, 50)
BLACK = (0, 0, 0)

# Hand-verified against ground-truth testing clips, FLOW_SIZE coords.
# arrows: list of (position, direction); highlight: (center, half-axes).
MANUAL = {
    "0008": {"arrows": [((250, 195), (1, 0))],        # ball rolls right
             "highlight": ((250, 195), (95, 65))},
    "0032": {"arrows": [((348, 210), (1, 0)),          # blue rolls right
                        ((665, 208), (-1, 0))],        # yellow rolls left
             "highlight": ((505, 210), (240, 75))},
    "0053": {"arrows": [((345, 358), (0.8, 0.6))],     # released balls swing down-right
             "highlight": ((470, 385), (185, 90))},
    "0065": {"arrows": [((490, 265), (0, 1))],         # stream pours down
             "highlight": ((500, 350), (125, 165))},
    "0089": {"arrows": [((565, 65), (0, 1))],          # stream pours down
             "highlight": ((565, 220), (105, 150))},
    "0140": {"arrows": [((505, 285), (0, -1))],        # smoke rises
             "highlight": ((505, 250), (135, 165))},
    "0146": {"arrows": [((175, 230), (1, 0))],         # ball rolls right
             "highlight": ((175, 230), (85, 60))},
    "0182": {"arrows": [((640, 185), (0.35, 1))],      # tower topples down-right
             "highlight": ((560, 300), (205, 205))},
}


def last_frame(path):
    vr = VideoReader(path)
    return vr[len(vr) - 1].asnumpy()[:, :, ::-1]  # RGB -> BGR


def gray_at(path, idx, size):
    vr = VideoReader(path)
    f = vr[idx].asnumpy()[:, :, ::-1]
    f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


def estimate_motion(real_path, n_frames=54, step=6):
    """Fully mechanical estimator (kept for provenance; see module docstring
    for why it is not authoritative)."""
    idxs = list(range(0, n_frames, step))
    grays = [gray_at(real_path, i, FLOW_SIZE) for i in idxs]
    pairs = []
    for a, b in zip(grays, grays[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(fl, axis=-1)
        thr = max(np.percentile(mag, 99.0), 0.75)
        mask = mag > thr
        if mask.sum() < 50:
            thr = np.percentile(mag, 97.0)
            mask = mag > thr
        energy = float(mag[mask].sum()) if mask.any() else 0.0
        pairs.append({"fl": fl, "mag": mag, "mask": mask, "energy": energy})

    max_e = max(p["energy"] for p in pairs)
    onset = next((p for p in pairs if p["energy"] >= 0.25 * max_e),
                 max(pairs, key=lambda p: p["energy"]))
    peak = max(pairs, key=lambda p: p["energy"])

    ys, xs = np.nonzero(onset["mask"])
    cx, cy = float(xs.mean()), float(ys.mean())
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    vx = float(np.median(peak["fl"][:, :, 0][peak["mask"]]))
    vy = float(np.median(peak["fl"][:, :, 1][peak["mask"]]))
    mmean = float(peak["mag"][peak["mask"]].mean())
    return {"centroid": [cx, cy], "vec": [vx, vy], "bbox": bbox,
            "motion_mag": mmean}


def scale_up(pt, native):
    sx = native[1] / FLOW_SIZE[0]
    sy = native[0] / FLOW_SIZE[1]
    return pt[0] * sx, pt[1] * sy


def draw_arrows(img, arrows):
    h, w = img.shape[:2]
    for pos, d in arrows:
        cx, cy = scale_up(pos, (h, w))
        dx, dy = d
        n = max(np.hypot(dx, dy), 1e-6)
        L = 0.16 * w
        ux, uy = dx / n * L, dy / n * L
        p0 = (int(cx - ux * 0.15), int(cy - uy * 0.15))
        p1 = (int(cx + ux), int(cy + uy))
        cv2.arrowedLine(img, p0, p1, BLACK, 26, tipLength=0.18)
        cv2.arrowedLine(img, p0, p1, LIME, 16, tipLength=0.18)
    return img


def draw_highlight(img, center, axes):
    h, w = img.shape[:2]
    cx, cy = scale_up(center, (h, w))
    ax = max(axes[0] * (w / FLOW_SIZE[0]), 40)
    ay = max(axes[1] * (h / FLOW_SIZE[1]), 40)
    cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, BLACK, 22)
    cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, LIME, 13)
    return img


def to_sketch(img):
    small = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    gray, _ = cv2.pencilSketch(small, sigma_s=60, sigma_r=0.07,
                               shade_factor=0.05)
    sk = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.resize(sk, (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_AREA)


def contact_sheet(imgs, labels, path):
    tiles = []
    for im, lab in zip(imgs, labels):
        t = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.putText(t, lab, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    BLACK, 8, cv2.LINE_AA)
        cv2.putText(t, lab, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else set(SCENES)
    os.makedirs(OUTDIR, exist_ok=True)

    meta = {}
    for sid in sorted(only):
        slug = SCENES[sid]
        init = last_frame(COND.format(sid=sid, slug=slug))
        spec = MANUAL[sid]
        arrow = draw_arrows(init.copy(), spec["arrows"])
        center, axes = spec["highlight"]
        high = draw_highlight(init.copy(), center, axes)
        sketch = to_sketch(init.copy())
        paths = {}
        for name, im in [("arrow", arrow), ("highlight", high),
                         ("sketch", sketch)]:
            p = os.path.join(OUTDIR, f"{sid}_{slug}_{name}.png")
            cv2.imwrite(p, im)
            paths[name] = p
        grid_path = os.path.join(OUTDIR, f"{sid}_{slug}_grid.jpg")
        contact_sheet([init, arrow, high, sketch],
                      ["orig", "arrow", "highlight", "sketch"], grid_path)
        meta[sid] = {"slug": slug, "manual": spec, "paths": paths,
                     "grid": grid_path,
                     "flow_estimate_provenance": estimate_motion(
                         REAL.format(sid=sid, slug=slug))}
        print(f"[{sid}] edits written (manual-verified)")

    with open(os.path.join(OUTDIR, "edits_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("EDITS_DONE")


if __name__ == "__main__":
    main()
