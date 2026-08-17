#!/usr/bin/env python3
"""PSG-JEPA identifiability probe (runs on the Inspire notebook).

For each checkpoint (baseline LeWM vs PSG-JEPA):
  A. Per-dimension linear ridge probe: frozen latent z_t -> each observation dim,
     episode-level train/test split, per-dim Pearson r on held-out episodes.
  B. Dataset-side stats: per-dim variance and pixel-visibility
     |corr(|d obs_d|, pixel diff)| -- tests whether the identifiability gap tracks
     what forward prediction could even see in pixels.
  C. Open-loop rollout MSE: 3 context frames, recursive predictor rollout with the
     logged action sequence, latent MSE vs encoder truth at model steps {5,15,30}.

OGBench cube h5 is FLAT: all frames in one table with ep_len/ep_offset marking
episode boundaries. Observation layout (28-d):
  joint_pos [0:6] | joint_vel [6:12] | effector+gripper [12:19] | privileged [19:28]

Usage (on notebook):
  cd $W/repo && PYTHONPATH=. ../.venv/bin/python ../probe_psg.py \
    --ckpt ../out_baseline/...ckpt --name lewm --ckpt ../out_psg/...ckpt --name psg \
    --h5 /tmp/stablewm/datasets/ogbench/cube_single_expert.h5 --out ../probe_out
"""
import argparse, json, os
import numpy as np
import torch
import h5py
import hdf5plugin  # registers compression filters; pixels dataset needs this

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAMESKIP = 5
GROUPS = {"joint_pos": (0, 6), "joint_vel": (6, 12), "effector": (12, 19), "privileged": (19, 28)}

def prep_pixels(pix_u8, img_size=224):
    x = pix_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    if x.shape[-1] != img_size:
        x = torch.nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear")
    return x

@torch.no_grad()
def encode_frames(model, pix_t, bs=32, device="cuda"):
    model = model.to(device).eval()
    outs = []
    for i in range(0, len(pix_t), bs):
        chunk = pix_t[i:i + bs].to(device)
        out = model.encode({"pixels": chunk.unsqueeze(0)})
        outs.append(out["emb"][0].float().cpu())
    return torch.cat(outs)

def ridge_fit_r(Xtr, Ytr, Xte, Yte, alpha=1.0):
    Xtr_ = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    W = np.linalg.solve(Xtr_.T @ Xtr_ + alpha * np.eye(Xtr_.shape[1]), Xtr_.T @ Ytr)
    Xte_ = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    P = Xte_ @ W
    rs = []
    for d in range(Yte.shape[1]):
        a, b = P[:, d], Yte[:, d]
        rs.append(float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-9 and np.std(a) > 1e-9 else float("nan"))
    return rs

@torch.no_grad()
def rollout_mse(model, pix_t, acts, steps=(5, 15, 30), device="cuda"):
    """3 context frames -> recursive rollout with logged actions; MSE vs encoder truth."""
    T = len(pix_t)
    maxk = max(steps)
    if T < 3 + maxk:
        return None
    model = model.to(device).eval()
    info = {"pixels": pix_t[:3].unsqueeze(0).to(device),
            "action": torch.from_numpy(acts[:3]).float().unsqueeze(0).to(device)}
    out = model.encode(info)
    emb = out["emb"].clone()                       # (1,3,D)
    truth = encode_frames(model, pix_t[:3 + maxk]) # (3+maxk, D) on cpu
    cur = emb
    act_hist = info["action"]
    res = {}
    for k in range(1, maxk + 1):
        a_next = torch.from_numpy(acts[2 + k:2 + k + 1]).float().unsqueeze(0).to(device)
        act_hist = torch.cat([act_hist, a_next], dim=1)
        act_emb = model.action_encoder(act_hist[:, -3:])
        pred = model.predict(cur[:, -3:], act_emb)[:, -1:]
        cur = torch.cat([cur, pred], dim=1)
        if k in steps:
            res[k] = float(torch.nn.functional.mse_loss(pred[0, 0].cpu(), truth[2 + k]))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--max-episodes", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    h = h5py.File(args.h5, "r")
    ep_len = h["ep_len"][:]
    ep_off = h["ep_offset"][:]
    n_ep = min(len(ep_len), args.max_episodes)
    print("episodes total:", len(ep_len), "using:", n_ep, flush=True)

    rng = np.random.default_rng(0)
    n_test = max(1, int(n_ep * args.test_frac))
    idx = rng.permutation(n_ep)
    test_set = set(idx[:n_test].tolist())

    Pix, Obs, Act, splits = [], [], [], []
    PixDiff, DObs = [], []
    for ei in range(n_ep):
        s, e = int(ep_off[ei]), int(ep_off[ei] + ep_len[ei])
        pix = h["pixels"][s:e:FRAMESKIP]
        obs = h["observation"][s:e:FRAMESKIP].astype(np.float32)
        # training feeds the action encoder frameskip*act_dim flattened chunks
        # (effective_act_dim = frameskip * action_dim, cf. train.py)
        act_full = h["action"][s:e].astype(np.float32)
        T = min(len(pix), len(obs), act_full.shape[0] // FRAMESKIP)
        pix, obs = pix[:T], obs[:T]
        act = act_full[: T * FRAMESKIP].reshape(T, -1)
        Pix.append(prep_pixels(pix))
        Obs.append(obs)
        Act.append(act)
        splits.append("test" if ei in test_set else "train")
        PixDiff.append(np.abs(np.diff(pix.astype(np.float32), axis=0)).mean(axis=(1, 2, 3)))
        DObs.append(np.abs(np.diff(obs, axis=0)))
        if (ei + 1) % 50 == 0:
            print("loaded", ei + 1, flush=True)
    obs_all = np.concatenate(Obs)
    D = obs_all.shape[1]
    print("obs dim:", D, flush=True)

    var = obs_all.var(axis=0)
    pd_all = np.concatenate(PixDiff)
    do_all = np.concatenate(DObs)
    vis = [float(abs(np.corrcoef(do_all[:, d], pd_all)[0, 1])) if np.std(do_all[:, d]) > 1e-9 else float("nan")
           for d in range(D)]
    json.dump({"var": var.tolist(), "pixel_visibility": vis,
               "groups": {k: [a, b] for k, (a, b) in GROUPS.items()}},
              open(f"{args.out}/dataset_stats.json", "w"), indent=1)

    results = {}
    for name, ck in zip(args.name, args.ckpt):
        print("==", name, ck, flush=True)
        model = torch.load(ck, map_location="cpu", weights_only=False)
        lat = []
        for i, pix_t in enumerate(Pix):
            lat.append(encode_frames(model, pix_t).numpy())
            if (i + 1) % 50 == 0:
                print("  encoded", i + 1, flush=True)
        Ztr = np.concatenate([l for l, s in zip(lat, splits) if s == "train"])
        Ytr = np.concatenate([o for o, s in zip(Obs, splits) if s == "train"])
        Zte = np.concatenate([l for l, s in zip(lat, splits) if s == "test"])
        Yte = np.concatenate([o for o, s in zip(Obs, splits) if s == "test"])
        rs = ridge_fit_r(Ztr, Ytr, Zte, Yte)
        entry = {"per_dim_r": rs,
                 "group_r": {g: float(np.nanmean(rs[a:b])) for g, (a, b) in GROUPS.items() if b <= D}}
        print("  group r:", {k: round(v, 3) for k, v in entry["group_r"].items()}, flush=True)

        model = model.cuda().eval()
        roll = {5: [], 15: [], 30: []}
        test_eps = [i for i, s in enumerate(splits) if s == "test"]
        for i in test_eps[:24]:
            r = rollout_mse(model, Pix[i], Act[i])
            if r:
                for k, v in r.items():
                    roll[k].append(v)
        entry["rollout_mse"] = {str(k): float(np.mean(v)) for k, v in roll.items() if v}
        print("  rollout MSE:", entry["rollout_mse"], flush=True)
        results[name] = entry
        del model
        torch.cuda.empty_cache()

    json.dump(results, open(f"{args.out}/probe_results.json", "w"), indent=1)
    print("saved", args.out, flush=True)

if __name__ == "__main__":
    main()
