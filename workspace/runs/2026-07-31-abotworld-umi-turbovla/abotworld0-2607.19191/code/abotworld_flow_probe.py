#!/usr/bin/env python3
"""Camera-motion controllability probe for ABot-World-0 long rollouts.

Question (paper never tests): does action-following decay with rollout length,
even while visual quality stays acceptable?

Protocol used during generation (per-block actions, 1 block = 1 s @ 12 fps):
  32-block cycle: J x8 (yaw left), W x8 (forward), L x8 (yaw right), W x8.

Metric per block:
  yaw blocks  -> sign(mean horizontal optical flow) vs commanded direction
  walk blocks -> expansion score (radial divergence of the flow field) vs
                 baseline measured on the first walk phase

Outputs: per-block CSV + agreement curve PNG.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAMES_PER_BLOCK = 12  # 12 fps video, 1 block = 1 s


def block_action(i: int) -> str:
    phase = (i // 8) % 4
    return ["J", "W", "L", "W"][phase]


def flow_stats(prev_gray: np.ndarray, gray: np.ndarray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    fx = flow[..., 0].mean()
    fy = flow[..., 1].mean()
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    rx, ry = xs - cx, ys - cy
    norm = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
    # radial divergence: positive = expansion (camera moving forward)
    expansion = ((flow[..., 0] * rx + flow[..., 1] * ry) / norm).mean()
    return fx, fy, expansion


def main(video_path: str, out_prefix: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"cannot read {video_path}")
    prev = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)

    per_frame = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        per_frame.append(flow_stats(prev, gray))
        prev = gray
        idx += 1
    cap.release()
    n_frames = len(per_frame) + 1
    print(f"frames={n_frames} -> blocks={n_frames // FRAMES_PER_BLOCK}")

    rows = []
    for b in range(n_frames // FRAMES_PER_BLOCK):
        seg = per_frame[b * FRAMES_PER_BLOCK: (b + 1) * FRAMES_PER_BLOCK]
        if not seg:
            continue
        fx = float(np.mean([s[0] for s in seg]))
        fy = float(np.mean([s[1] for s in seg]))
        ex = float(np.mean([s[2] for s in seg]))
        rows.append(dict(block=b, action=block_action(b), fx=fx, fy=fy, expansion=ex))

    # yaw agreement: J should give positive fx, L negative (or vice versa;
    # determine the sign convention from the FIRST cycle, then keep it fixed)
    first_j = next(r for r in rows if r["action"] == "J")
    j_sign = 1.0 if first_j["fx"] >= 0 else -1.0
    for r in rows:
        if r["action"] == "J":
            r["agree"] = float(np.sign(r["fx"]) == np.sign(j_sign) or abs(r["fx"]) < 0.05)
        elif r["action"] == "L":
            r["agree"] = float(np.sign(r["fx"]) == -np.sign(j_sign) or abs(r["fx"]) < 0.05)
        else:
            r["agree"] = np.nan

    # walk response: expansion normalized by the first walk phase
    first_w = [r["expansion"] for r in rows[:32] if r["action"] == "W"]
    w0 = float(np.mean(first_w)) if first_w else 1.0
    for r in rows:
        r["expansion_rel"] = r["expansion"] / w0 if abs(w0) > 1e-6 else np.nan

    csv_path = Path(out_prefix).with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("block,action,fx,fy,expansion,agree,expansion_rel\n")
        for r in rows:
            f.write(f"{r['block']},{r['action']},{r['fx']:.4f},{r['fy']:.4f},"
                    f"{r['expansion']:.4f},{r['agree']},{r['expansion_rel']:.4f}\n")

    # windowed agreement over time (16-block windows)
    yaw_rows = [r for r in rows if not np.isnan(r["agree"])]
    win, t_agree, v_agree = 16, [], []
    for s in range(0, len(yaw_rows) - win + 1, 4):
        t_agree.append(yaw_rows[s]["block"])
        v_agree.append(float(np.mean([r["agree"] for r in yaw_rows[s:s + win]])))
    walk_rows = [r for r in rows if r["action"] == "W"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot([r["block"] for r in rows], [r["fx"] for r in rows], lw=0.8)
    axes[0].set_ylabel("mean flow x / block")
    axes[0].axhline(0, color="k", lw=0.4)
    axes[1].plot(t_agree, v_agree, marker="o", ms=3)
    axes[1].set_ylabel("yaw agreement (16-blk win)")
    axes[1].set_ylim(0, 1.05)
    axes[2].plot([r["block"] for r in walk_rows],
                 [r["expansion_rel"] for r in walk_rows], marker=".", ms=3)
    axes[2].set_ylabel("walk expansion (rel. to 1st phase)")
    axes[2].set_xlabel("block (1 block = 1 s)")
    axes[2].axhline(1.0, color="k", lw=0.4, ls="--")
    fig.suptitle("ABot-World-0 action-following vs rollout time")
    fig.tight_layout()
    png_path = Path(out_prefix).with_suffix(".png")
    fig.savefig(png_path, dpi=140)

    summary = dict(
        n_blocks=len(rows), j_sign=j_sign,
        yaw_agree_first_quarter=float(np.mean([r["agree"] for r in yaw_rows[:len(yaw_rows)//4]])),
        yaw_agree_last_quarter=float(np.mean([r["agree"] for r in yaw_rows[-len(yaw_rows)//4:]])),
        walk_exp_first=float(np.mean([r["expansion_rel"] for r in walk_rows[:4]])),
        walk_exp_last=float(np.mean([r["expansion_rel"] for r in walk_rows[-4:]])),
    )
    print(json.dumps(summary, indent=2))
    with open(Path(out_prefix).with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
