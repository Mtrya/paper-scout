#!/usr/bin/env python
"""Metrics and contact sheets for the Ropedia W0 action-interface audit.

Per video: PSNR / SSIM against the ground-truth window, motion energy, and an image-space global
motion estimate from phase correlation (a proxy for camera-induced image motion; it is not a metric
pose, see the README).

usage:
  python audit_metrics.py --dir outputs/audit --out metrics.json --sheet sheet.png \
      --rows name1,name2,... --frames 0,20,40,60,80
"""
import argparse, json, os, sys

import numpy as np
import imageio
from PIL import Image, ImageDraw

H, W = 480, 640


def read_video(path):
    r = imageio.get_reader(path)
    try:
        frames = [np.asarray(f).astype(np.float32) / 255.0 for f in r]
    finally:
        r.close()
    out = []
    for f in frames:
        if f.shape[:2] != (H, W):
            f = np.asarray(Image.fromarray((f * 255).astype(np.uint8)).resize((W, H))).astype(np.float32) / 255.0
        out.append(f[..., :3])
    return np.stack(out)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


def ssim_frame(a, b):
    from scipy.ndimage import gaussian_filter
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = gaussian_filter(a, 1.5)
    mu_b = gaussian_filter(b, 1.5)
    saa = gaussian_filter(a * a, 1.5) - mu_a ** 2
    sbb = gaussian_filter(b * b, 1.5) - mu_b ** 2
    sab = gaussian_filter(a * b, 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(s.mean())


def phase_shift(a, b):
    """Global translation between two grey frames via phase correlation (pixels)."""
    ga = a.mean(2) - a.mean()
    gb = b.mean(2) - b.mean()
    win = np.outer(np.hanning(ga.shape[0]), np.hanning(ga.shape[1]))
    Fa, Fb = np.fft.rfft2(ga * win), np.fft.rfft2(gb * win)
    R = Fa * np.conj(Fb)
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R)
    idx = np.unravel_index(np.argmax(r), r.shape)
    dy = idx[0] if idx[0] <= r.shape[0] // 2 else idx[0] - r.shape[0]
    dx = idx[1] if idx[1] <= r.shape[1] // 2 else idx[1] - r.shape[1]
    return float(dx), float(dy)


def motion_stats(v):
    d = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2, 3))          # per-frame motion energy
    shifts = [phase_shift(v[i], v[i + 1]) for i in range(len(v) - 1)]
    sh = np.array(shifts)
    path = float(np.linalg.norm(sh, axis=1).sum())
    end = float(np.linalg.norm(sh.sum(0)))
    return dict(motion_mean=float(d.mean()), motion_profile=d.tolist(),
                cam_path_px=path, cam_end_px=end,
                cam_shifts=[[round(float(x), 2), round(float(y), 2)] for x, y in sh])


def corner_shift(f0, ft, max_dist=3.0):
    """Mean displacement of the image corners under the ORB+RANSAC homography frame0 -> frame t.

    This is an image-space proxy dominated by static background, not a metric camera pose.
    Returns mean corner displacement, inlier ratio, and signed mean x/y displacement.
    """
    import cv2
    g0 = cv2.cvtColor(f0.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gt = cv2.cvtColor(ft.astype(np.float32), cv2.COLOR_RGB2GRAY)
    g0 = (g0 * 255).astype(np.uint8)
    gt = (gt * 255).astype(np.uint8)
    orb = cv2.ORB_create(2000)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(gt, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = sorted(bf.match(d0, d1), key=lambda x: x.distance)[:200]
    if len(m) < 8:
        return None
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, max_dist)
    if H is None:
        return None
    h, w = g0.shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    delta = proj - corners.reshape(-1, 2)
    return (float(np.linalg.norm(delta, axis=1).mean()), float(mask.mean()),
            float(delta[:, 0].mean()), float(delta[:, 1].mean()))


def homog_track(v, frames=(20, 40, 60, 80)):
    out = {}
    for t in frames:
        if t < len(v):
            r = corner_shift(v[0], v[t])
            out[t] = None if r is None else {
                "corner_shift_px": round(r[0], 2),
                "corner_dx_px": round(r[2], 2),
                "corner_dy_px": round(r[3], 2),
                "inlier_fraction": round(r[1], 3),
            }
    return out


def analyse(pred, gt):
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    ps = [psnr(pred[i], gt[i]) for i in range(1, n)]              # frame 0 is clamped to GT
    ss = [ssim_frame(pred[i].mean(2), gt[i].mean(2)) for i in range(1, n, 4)]
    mp, mg = motion_stats(pred), motion_stats(gt)
    pred_profile = np.asarray(mp["motion_profile"])
    gt_profile = np.asarray(mg["motion_profile"])
    corr = None
    if len(pred_profile) > 2 and pred_profile.std() > 0 and gt_profile.std() > 0:
        corr = float(np.corrcoef(pred_profile, gt_profile)[0, 1])
    hp, hg = homog_track(pred), homog_track(gt)
    return dict(psnr_mean=float(np.mean(ps)), psnr_frames=[round(x, 2) for x in ps],
                ssim_mean=float(np.mean(ss)),
                motion_mean=mp["motion_mean"], motion_mean_gt=mg["motion_mean"],
                motion_ratio=mp["motion_mean"] / (mg["motion_mean"] + 1e-9),
                motion_profile_corr=corr,
                cam_path_px=mp["cam_path_px"], cam_path_px_gt=mg["cam_path_px"],
                cam_end_px=mp["cam_end_px"], cam_end_px_gt=mg["cam_end_px"],
                homog_pred=hp, homog_gt=hg)


def label(im, text, scale=1):
    im = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 260 * scale, 22 * scale], fill=(0, 0, 0))
    d.text((4 * scale, 4 * scale), text, fill=(255, 255, 255))
    return im


def sheet(rows, frames, out, max_width=1600):
    tiles = []
    for name, path in rows:
        v = read_video(path)
        tiles.append([label(v[f], f"{name} f{f}") for f in frames if f < len(v)])
    h, w = tiles[0][0].size[1], tiles[0][0].size[0]
    cols = max(len(t) for t in tiles)
    canvas = Image.new("RGB", (w * cols, h * len(tiles)), (0, 0, 0))
    for r, row in enumerate(tiles):
        for c, im in enumerate(row):
            canvas.paste(im, (c * w, r * h))
    if canvas.width > max_width:
        ratio = max_width / canvas.width
        canvas = canvas.resize((max_width, int(canvas.height * ratio)), Image.LANCZOS)
    canvas.save(out)
    print(f"[sheet] {out} {canvas.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory with <name>_pred.mp4 and <name>_gt.mp4")
    ap.add_argument("--names", required=True, help="comma-separated case names")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--sheet_rows", default=None, help="comma-separated names, in row order")
    ap.add_argument("--frames", default="0,20,40,60,80")
    ap.add_argument("--gt_name", default=None, help="case name whose _gt.mp4 to use for every row")
    args = ap.parse_args()

    names = [n for n in args.names.split(",") if n]
    res = {}
    for name in names:
        pred_path = os.path.join(args.dir, f"{name}_pred.mp4")
        gt_path = os.path.join(args.dir, f"{(args.gt_name or name)}_gt.mp4")
        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            print(f"[skip] {name} (missing video)", flush=True)
            continue
        res[name] = analyse(read_video(pred_path), read_video(gt_path))
        r = res[name]
        print(f"{name:28s} PSNR={r['psnr_mean']:.2f} SSIM={r['ssim_mean']:.3f} "
              f"motion={r['motion_mean']:.4f} (gt {r['motion_mean_gt']:.4f}, ratio {r['motion_ratio']:.2f}) "
              f"corr={r['motion_profile_corr'] if r['motion_profile_corr'] is not None else 'n/a'} "
              f"cam_path={r['cam_path_px']:.0f}px (gt {r['cam_path_px_gt']:.0f})",
              flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[done] {args.out}")

    if args.sheet:
        rows = []
        for name in (args.sheet_rows or args.names).split(","):
            p = os.path.join(args.dir, f"{name}_pred.mp4")
            if os.path.exists(p):
                rows.append((name, p))
        gtp = os.path.join(args.dir, f"{(args.gt_name or names[0])}_gt.mp4")
        if os.path.exists(gtp):
            rows.insert(0, ("GT", gtp))
        frames = [int(x) for x in args.frames.split(",")]
        sheet(rows, frames, args.sheet)


if __name__ == "__main__":
    main()
