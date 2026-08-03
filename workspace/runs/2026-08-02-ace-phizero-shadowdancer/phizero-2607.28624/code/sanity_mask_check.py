"""Sanity check (corrected order): our motion_masks() at NATIVE 4K on the real
testing video, resized to 960x540, vs the official provided mask video.
High IoU => pipeline consistent; low gen-vs-real numbers then reflect the
model, not a protocol bug."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_probe import read_video, motion_masks, resize_masks, EVAL_SIZE
import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = {"0008": "ball-hits-duck", "0053": "double-cradle"}
for sid, slug in SCENES.items():
    real = read_video(os.path.join(BASE, "data", "split-videos_testing_24FPS",
        f"{sid}_testing-videos_24FPS_perspective-center_take-1_trimmed-{slug}.mp4"), None)
    rmask = read_video(os.path.join(BASE, "data", "video-masks_real_24FPS",
        f"{sid}_video-masks_24FPS_perspective-center_take-1_trimmed-{slug}.mp4"), None, gray=True)
    n = min(len(real), len(rmask))
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in real[:n]]
    mine = resize_masks(motion_masks(gray), EVAL_SIZE)
    theirs = resize_masks([m > 127 for m in rmask[:n]], EVAL_SIZE)
    vals = []
    for ab, bb in zip(mine, theirs):
        u = np.logical_or(ab, bb).sum()
        if u:
            vals.append(np.logical_and(ab, bb).sum() / u)
    print(f"[{sid}] self-consistency IoU = {np.mean(vals):.4f} over {len(vals)} frames")
print("SANITY_DONE")
