#!/usr/bin/env python
"""What the released example clips actually put into the two condition channels.

For every clip under --examples this measures, without running the world model:
  * the camera trajectory the Pluecker map encodes (path length, rotation, how many of the six
    channels are non-zero, and which fraction of frames move at all);
  * how much of the image the skeleton condition covers (ink fraction, bounding box);
  * whether the text prompt already names the manipulation, which is what a naive
    "--disable_skeleton" ablation leaves in place.

usage: python audit_offline.py --examples examples --out offline.json
"""
import argparse, json, os, re

import numpy as np
import torch

from ropedia_w0.utils.data.camera_plucker import compute_plucker_from_camera_params, load_camera_params

H, W = 480, 640
ACTION_WORDS = ("hand", "hands", "arm", "arms", "gripper", "fingers", "press", "grasp", "pick",
                "place", "lower", "lift", "hold", "push", "drop", "open", "close", "move", "cut",
                "peel", "stack", "assemble", "hang", "sort", "put", "reach", "pull")


def camera_stats(npz):
    z = load_camera_params(npz)
    T = np.asarray(z["extrinsics_c2w"])
    rel = np.linalg.inv(T[0]) @ T
    t = rel[:, :3, 3]
    path = float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())
    R = T[:, :3, :3]
    ang = []
    for i in range(len(R)):
        c = (np.trace(R[0].T @ R[i]) - 1) / 2
        ang.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    pl = compute_plucker_from_camera_params(npz, H, W, dtype=torch.float64).numpy()
    mom, direc = pl[:, :3], pl[:, 3:]
    return dict(frames=int(len(T)), trans_path_m=round(path, 3),
                max_rot_deg=round(max(ang), 1), end_trans_m=round(float(np.linalg.norm(t[-1])), 3),
                moment_abs_mean=float(np.abs(mom).mean()), moment_nonzero_frac=float((np.abs(mom) > 1e-9).mean()),
                direction_abs_mean=float(np.abs(direc).mean()),
                direction_frame_std=float(direc.std(axis=0).mean()))


def skeleton_stats(path):
    import imageio
    r = imageio.get_reader(path)
    try:
        n = int(r.count_frames())
        idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ink, bbox = [], []
        for i in idxs:
            f = np.asarray(r.get_data(i))[..., :3].astype(np.float32) / 255.0
            m = f.max(axis=2) > 0.24
            ink.append(float(m.mean()))
            if m.any():
                ys, xs = np.nonzero(m)
                bbox.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    finally:
        r.close()
    areas = []
    for b in bbox:
        areas.append((b[2] - b[0]) * (b[3] - b[1]) / (W * H))
    return dict(frames=n, ink_frac=round(float(np.mean(ink)), 5),
                bbox_area_frac=round(float(np.mean(areas)), 4) if areas else 0.0, bbox=bbox[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="examples")
    ap.add_argument("--out", default="offline.json")
    args = ap.parse_args()
    rows = {}
    for clip in sorted(os.listdir(args.examples)):
        d = os.path.join(args.examples, clip)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "data.npz")):
            continue
        with open(os.path.join(d, "prompt.txt"), encoding="utf-8") as f:
            prompt = f.read().strip()
        n_action = sum(1 for w in ACTION_WORDS if re.search(r"\b" + w, prompt.lower()))
        rows[clip] = dict(prompt_chars=len(prompt), prompt=prompt[:160],
                          prompt_action_words=n_action,
                          camera=camera_stats(os.path.join(d, "data.npz")),
                          skeleton=skeleton_stats(os.path.join(d, "skeleton.mp4")))
        c, s = rows[clip]["camera"], rows[clip]["skeleton"]
        print(f"{clip:28s} cam: {c['trans_path_m']:.3f}m {c['max_rot_deg']:5.1f}deg  "
              f"moment-nonzero {c['moment_nonzero_frac']:.3f}  |  skel ink {s['ink_frac']:.4f} "
              f"bbox {s['bbox_area_frac']:.3f}  |  action words {n_action}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    cam_paths = [r["camera"]["trans_path_m"] for r in rows.values()]
    rots = [r["camera"]["max_rot_deg"] for r in rows.values()]
    print(f"\n{len(rows)} clips: camera path median {np.median(cam_paths):.3f} m, "
          f"{sum(1 for p in cam_paths if p < 0.02)}/{len(rows)} below 2 cm; "
          f"max rotation median {np.median(rots):.1f} deg, {sum(1 for r in rots if r < 1)}/{len(rows)} below 1 deg")
    print(f"skeleton ink fraction median {np.median([r['skeleton']['ink_frac'] for r in rows.values()]):.4f} "
          f"of pixels; prompts naming the action: "
          f"{sum(1 for r in rows.values() if r['prompt_action_words'] > 0)}/{len(rows)}")


if __name__ == "__main__":
    main()
