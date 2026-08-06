#!/usr/bin/env python3
"""Cycle-consistency analysis for ABot-World-0 cycle probe videos.

Metrics per protocol (block-level keyframes, last frame of each block):
  ESC : endpoint state closure  = dist(first block keyframe of block0, last block keyframe)
  RPS : reverse-path symmetry   = mean dist over mirrored pairs (i, 2m-i) for inverse cycles
  RCS : repeated-cycle stability= mean dist of phase-aligned keyframes across cycles (p3)

Frame distance = mean 2D displacement (px) of SIFT good matches, normalized by
(H+W) as in WorldCycle eq.(4); pairs with <10 matches get the image diagonal.
Also reports match counts, SSIM, and global-flow yaw rate for J/L segments.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/cycle_probe")
sift = cv2.SIFT_create()

DIAG = None


def load(p):
    img = cv2.imread(str(p))
    return img


def frame_dist(a, b):
    """WorldCycle-style normalized mean displacement + match stats."""
    global DIAG
    H, W = a.shape[:2]
    if DIAG is None:
        DIAG = float(np.hypot(H, W))
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return {"d_norm": 1.0, "d_px": DIAG, "matches": 0}
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return {"d_norm": 1.0, "d_px": DIAG, "matches": len(good)}
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    # robust: keep inliers via homography RANSAC when possible
    if len(good) >= 8:
        _, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 5.0)
        if inl is not None and inl.sum() >= 8:
            m = inl.ravel().astype(bool)
            pa, pb = pa[m], pb[m]
    disp = np.linalg.norm(pa - pb, axis=1)
    return {"d_norm": float(disp.mean() / (H + W)), "d_px": float(disp.mean()),
            "matches": int(len(disp))}


def ssim(a, b):
    from skimage.metrics import structural_similarity
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(ga, gb))


def yaw_rate(frames_dir, blocks):
    """mean global horizontal flow (px/frame, +right) across given block indices."""
    vals = []
    prev = None
    for i in blocks:
        img = load(frames_dir / f"block_{i:03d}.jpg")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (320, 176))
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            vals.append(float(np.median(flow[..., 0])))
        prev = g
    return float(np.mean(vals)) if vals else None


def main():
    manifest = json.load(open(OUT / "manifest.json"))
    results = {}
    for name, meta in manifest.items():
        kf = OUT / f"{name}_blocks"
        n = meta["num_blocks"]
        imgs = [load(kf / f"block_{i:03d}.jpg") for i in range(n)]
        r = {"keys": meta["keys"], "num_blocks": n}
        r["esc"] = frame_dist(imgs[0], imgs[-1])
        r["ssim_first_last"] = ssim(imgs[0], imgs[-1])
        keys = meta["keys"]
        # mirrored pairs for inverse cycles: forward m blocks, reverse m blocks
        m = n // 2
        if name in ("p1_trans_cycle", "p2_yaw_cycle", "p4_long_cycle"):
            pairs = []
            for i in range(m):
                j = n - 1 - i  # mirrored reverse block
                pairs.append(frame_dist(imgs[i], imgs[j]))
            r["rps"] = float(np.mean([p["d_norm"] for p in pairs]))
            r["rps_pairs"] = pairs
        if name == "p3_repeat_cycle":
            # cycles of length 4 blocks (W W S S); compare phase-aligned across cycles
            L, K = 4, n // 4
            ds = []
            for k in range(1, K):
                for i in range(L):
                    ds.append(frame_dist(imgs[i], imgs[k * L + i])["d_norm"])
            r["rcs"] = float(np.mean(ds))
            # endpoint closure per cycle
            r["per_cycle_esc"] = [
                frame_dist(imgs[0], imgs[(k + 1) * L - 1])["d_norm"] for k in range(K)
            ]
        # consecutive-block drift profile (error growth along rollout)
        r["drift_profile"] = [frame_dist(imgs[0], imgs[i])["d_norm"] for i in range(1, n)]
        # yaw functional check
        if name == "p0_yaw_base":
            r["yaw_rate"] = yaw_rate(kf, list(range(n)))
        if name == "p5_return_then_yaw":
            r["yaw_rate_after_return"] = yaw_rate(kf, [8, 9, 10, 11])
            r["yaw_rate_baseline_fresh"] = yaw_rate(kf, [0, 1, 2, 3])  # W segment, should ~0
        results[name] = r
        print(f"[{name}] ESC={r['esc']['d_norm']:.4f} SSIM={r['ssim_first_last']:.3f} "
              + (f"RPS={r.get('rps', 0):.4f} " if "rps" in r else "")
              + (f"RCS={r.get('rcs', 0):.4f}" if "rcs" in r else ""), flush=True)
    with open(OUT / "cycle_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
