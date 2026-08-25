#!/usr/bin/env python3
"""AllTracker probe on DROID exterior videos (Hydra-0 training-side interface audit).

For each episode:
  1. dense AllTracker forward pass (visibility + trajectory maps)
  2. locate gripper cluster WITHOUT masks: corr(per-pixel speed, EE speed)
  3. fit affine map EE(3D) -> track(2D) with RANSAC; residual series = tracker jitter
  4. visibility decay + background stationarity census

Usage: probe_tracker_droid.py <droid_dir> <alltracker_repo> <weights> <out_dir> [n_episodes]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

droid_dir = Path(sys.argv[1])
repo = Path(sys.argv[2])
weights = Path(sys.argv[3])
out_dir = Path(sys.argv[4])
n_episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 6
out_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo))

from nets.alltracker import Net  # noqa: E402

DEVICE = torch.device("cuda")


def load_episode(ep):
    import cv2
    frames = sorted((ep / "ext1").glob("*.jpg"))
    imgs = []
    for f in frames:
        img = cv2.imread(str(f))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cp = np.load(ep / "observation_cartesian_position.npy").astype(np.float64)
    grip = np.load(ep / "observation_gripper_position.npy").astype(np.float64)
    n = min(len(imgs), len(cp))
    return np.stack(imgs[:n]), cp[:n], grip[:n]


@torch.no_grad()
def track(model, imgs):
    rgbs = torch.from_numpy(imgs).float().permute(0, 3, 1, 2)[None].to(DEVICE)
    B, T, C, H, W = rgbs.shape
    flows, visconf, _, _ = model.forward_sliding(rgbs, iters=8, sw=None, is_training=False)
    import utils.basic
    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device="cuda:0").float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)
    traj = flows.cuda() + grid_xy  # 1,T,2,H,W
    return traj[0].cpu().numpy(), visconf[0].cpu().numpy()  # T,2,H,W ; T,2,H,W


def ransac_affine(X, y, iters=400, thr_px=8.0, rng=None):
    """X: (N,3) EE, y: (N,2) track. Returns residual series + inlier mask."""
    rng = rng or np.random.RandomState(0)
    N = len(X)
    Xa = np.concatenate([X, np.ones((N, 1))], 1)  # N,4
    best = None
    for _ in range(iters):
        idx = rng.choice(N, 6, replace=False)
        A, *_ = np.linalg.lstsq(Xa[idx], y[idx], rcond=None)
        r = np.linalg.norm(Xa @ A - y, axis=1)
        inl = r < thr_px
        if best is None or inl.sum() > best[1].sum():
            best = (A, inl)
    A, inl = best
    A, *_ = np.linalg.lstsq(Xa[inl], y[inl], rcond=None)
    resid = np.linalg.norm(Xa @ A - y, axis=1)
    return resid, inl, A


def main():
    model = Net(seqlen=16)
    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict(sd["model"], strict=True)
    model.eval().to(DEVICE)

    episodes = sorted([d for d in droid_dir.iterdir() if (d / "ext1").exists()])[:n_episodes]
    summary = []
    for ep in episodes:
        name = ep.name
        imgs, cp, grip = load_episode(ep)
        T, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        traj, visconf = track(model, imgs)  # T,2,H,W ; T,2,H,W
        vis = visconf[:, 0]  # visibility channel guess; check stats
        conf = visconf[:, 1]

        ee_speed = np.linalg.norm(np.diff(cp[:, :3], axis=0), axis=1)  # T-1
        traj_d = np.diff(traj, axis=0)  # T-1,2,H,W
        pix_speed = np.linalg.norm(traj_d, axis=1)  # T-1,H,W

        # gripper cluster: corr(pixel speed, ee speed)
        es = (ee_speed - ee_speed.mean()) / (ee_speed.std() + 1e-9)
        ps = pix_speed.reshape(T - 1, -1)
        ps = (ps - ps.mean(0)) / (ps.std(0) + 1e-9)
        corr_map = (ps * es[:, None]).mean(0).reshape(H, W)
        top_idx = np.unravel_index(np.argsort(corr_map.ravel())[-200:], (H, W))

        # best pixel track
        flat_best = np.argmax(corr_map)
        by, bx = np.unravel_index(flat_best, (H, W))
        gtrack = traj[:, :, by, bx]  # T,2
        gvis = vis[:, by, bx]

        resid, inl, A = ransac_affine(cp[:, :3], gtrack)
        bg_mask = np.zeros((H, W), bool)
        bg_mask[: H // 5, : W // 5] = True  # top-left corner as background probe
        bg_disp = np.linalg.norm(traj[:, :, bg_mask] - traj[:1, :, bg_mask], axis=1)  # T,nbg
        rec = {
            "episode": name,
            "n_frames": int(T),
            "ee_speed_max": float(ee_speed.max()),
            "gripper_pix": [int(bx), int(by)],
            "gripper_corr": float(corr_map[by, bx]),
            "affine_resid_median": float(np.median(resid)),
            "affine_resid_p95": float(np.percentile(resid, 95)),
            "affine_inlier_frac": float(inl.mean()),
            "gripper_vis_frac": float((gvis > 0.5).mean()),
            "vis_frac_t_end": float((vis[-1] > 0.5).mean()),
            "bg_disp_median_end": float(np.median(bg_disp[-1])),
            "conf_mean": float(conf.mean()),
        }
        summary.append(rec)
        print(json.dumps(rec), flush=True)

        # save artifacts: corr map + overlay + residual series
        np.save(out_dir / f"{name}_corrmap.npy", corr_map)
        np.save(out_dir / f"{name}_resid.npy", resid)
        np.save(out_dir / f"{name}_gtrack.npy", gtrack)
        # overlay: frame 0 and mid frame with top-corr pixels
        import cv2
        for t in [0, T // 2, T - 1]:
            img = imgs[t].copy()
            ys, xs = top_idx
            vis_now = vis[t][ys, xs] > 0.5
            for y, x, v in zip(ys, xs, vis_now):
                c = (0, 255, 0) if v else (0, 0, 255)
                cv2.circle(img, (int(x), int(y)), 1, c, -1)
            cv2.circle(img, (int(gtrack[t, 0]), int(gtrack[t, 1])), 4, (255, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{name}_overlay_t{t}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    (out_dir / "tracker_summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
