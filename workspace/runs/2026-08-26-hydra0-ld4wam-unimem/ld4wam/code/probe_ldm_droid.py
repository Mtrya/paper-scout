#!/usr/bin/env python3
"""LDM frozen probe on DROID (OOD for LD4WAM's LDM).

Encodes exterior-camera transitions with frozen LDM.pt and measures how much
action-relevant information the latent carries on an unseen domain:
  B1: shallow-MLP regression of delta-EE (6D) from 512-d latent
      vs DINOv3 raw-feature baseline vs shuffle control
  B2: motion-vs-appearance retrieval (motion transitions only)

v2 diagnostics (v1 showed r2≈0 for BOTH LDM and DINOv3 -> suspect pipeline):
  - target sanity: per-dim raw ranges/scaled std/quantiles, glitch & rpy-wrap census
  - glitch filter: drop transitions with any |scaled delta| > 8 (>8cm or >37deg/frame)
  - learning-free: corr(||z||, ||delta-EE||) for LDM latent and DINOv3 diff
  - train-fit r2 (optimization/capacity check)
  - static filtering in levels: all / |d|>0.5 / top-50% motion

Usage: probe_ldm_droid.py <droid_dir> <ldm_repo> <ckpt> <dinov3_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

droid_dir = Path(sys.argv[1])
ldm_repo = Path(sys.argv[2])
ckpt = Path(sys.argv[3])
dinov3_dir = Path(sys.argv[4])
out_dir = Path(sys.argv[5])
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ldm_repo))
import os
os.environ["DINOV3_MODEL_PATH"] = str(dinov3_dir)

from PIL import Image
from torchvision.transforms import v2 as T

DEVICE = torch.device("cuda")
STRIDES = [1, 2, 4]
WIN = 8
SCALE = np.array([0.01, 0.01, 0.01, 0.08, 0.08, 0.08], dtype=np.float32)  # m, rad


def load_episode(ep):
    frames = sorted((ep / "ext1").glob("*.jpg"))
    cp = np.load(ep / "observation_cartesian_position.npy").astype(np.float64)
    n = min(len(frames), len(cp))
    return frames[:n], cp[:n]


def build_clips(frames, cp, stride):
    """Return clip frame-lists and per-transition delta-EE (6D, scaled)."""
    clips, targets, tindex = [], [], []
    idxs = list(range(0, len(frames), stride))
    for s in range(0, len(idxs) - WIN, WIN - 1):
        sel = idxs[s:s + WIN]
        if len(sel) < WIN:
            break
        clips.append([frames[i] for i in sel])
        d = np.diff(cp[sel], axis=0)  # (7, 6) per-transition deltas
        targets.append(d)
        tindex.append(sel)
    return clips, targets, tindex


tfm = T.Compose([
    T.Lambda(lambda p: Image.open(p).convert("RGB")),
    T.ToImage(),
    T.Resize((224, 224)),
    T.ToDtype(torch.float32, scale=True),
])


@torch.no_grad()
def encode_clips(model, clips, bs=16):
    latents = []
    feats = []
    for i in range(0, len(clips), bs):
        batch = torch.stack([torch.stack([tfm(f) for f in c]) for c in clips[i:i + bs]])
        batch = batch.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = model.inference(batch, return_reconstructions=False,
                                  return_quantized_actions=True)
            qa = res["quantized_actions"]  # (B, 7, 4, 4, 32)
            B = qa.shape[0]
            latents.append(qa.reshape(B, WIN - 1, -1).float().cpu())
            # DINOv3 baseline: raw frame tokens diff (mean-pooled patch tokens)
            frame_tokens, _ = model.encode(batch)  # (B, T, hp, wp, D)
            ft = frame_tokens
            if ft.dim() == 5:
                ft = ft.mean(dim=(2, 3))  # (B, T, D)
            elif ft.dim() == 4:
                ft = ft.mean(dim=2)
            diffs = ft[:, 1:] - ft[:, :-1]
            feats.append(diffs.float().cpu())
    return torch.cat(latents), torch.cat(feats)


class Probe(nn.Module):
    def __init__(self, din, dout=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, dout),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, Ytr, Xva, Yva, epochs=60, seed=0):
    torch.manual_seed(seed)
    din, dout = Xtr.shape[1], Ytr.shape[1]
    probe = Probe(din, dout).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xva, Yva = [t.to(DEVICE) for t in (Xtr, Ytr, Xva, Yva)]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]
            loss = ((probe(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        pred = probe(Xva)
        mse_per_dim = ((pred - Yva) ** 2).mean(0).cpu().numpy()
        var = Yva.var(dim=0).cpu().numpy() + 1e-12
        r2 = 1 - mse_per_dim / var
        # train-fit 检查:训练集子样上能拟合吗(不能则是优化/容量问题)
        k = max(1, len(Xtr) // 20000)
        pred_tr = probe(Xtr[::k])
        mse_tr = ((pred_tr - Ytr[::k]) ** 2).mean(0).cpu().numpy()
        var_tr = Ytr[::k].var(dim=0).cpu().numpy() + 1e-12
        r2_tr = float(np.mean(1 - mse_tr / var_tr))
    return float(mse_per_dim.mean()), mse_per_dim.tolist(), r2.tolist(), r2_tr


def main():
    from configs.config import MODEL_CFG
    from ldm.latent_dynamics_model import LDM
    cfg = dict(MODEL_CFG)
    cfg["dinov3_model_path"] = str(dinov3_dir)
    model = LDM(**cfg)
    inc = model.load_weights(ckpt, map_location="cpu", strict=False, verbose=True)
    model = model.eval().to(DEVICE)
    n_miss = len(inc.missing_keys) if inc else -1
    n_unexp = len(inc.unexpected_keys) if inc else -1
    print(f"[load] missing={n_miss} unexpected={n_unexp}")

    episodes = sorted([d for d in droid_dir.iterdir() if (d / "ext1").exists()])
    print(f"[data] {len(episodes)} episodes")

    results = {"n_missing": n_miss, "n_unexpected": n_unexp, "strides": {}}

    # ---- 目标诊断:cartesian_position 的量纲/范围/异常跳变(学习前先审目标) ----
    diag = {}
    for stride in STRIDES:
        ds = []
        for ep in episodes:
            frames, cp = load_episode(ep)
            idxs = list(range(0, len(frames), stride))
            ds.append(np.diff(cp[idxs], axis=0))
        Draw = np.concatenate(ds)
        Ds = Draw / SCALE
        absDs = np.abs(Ds)
        outlier = (absDs > 8).any(1)              # >8cm 或 >37deg/帧:遥控毛刺或 rpy 环绕
        rot_wrap = (np.abs(Draw[:, 3:]) > 1.5).any(1)  # 旋转维 >86deg/帧:典型环绕跳变
        diag[str(stride)] = {
            "n": int(len(Ds)),
            "raw_min": [round(float(v), 4) for v in Draw.min(0)],
            "raw_max": [round(float(v), 4) for v in Draw.max(0)],
            "scaled_std": [round(float(v), 3) for v in Ds.std(0)],
            "scaled_abs_p50": [round(float(v), 3) for v in np.quantile(absDs, 0.5, axis=0)],
            "scaled_abs_p99": [round(float(v), 3) for v in np.quantile(absDs, 0.99, axis=0)],
            "outlier_frac": round(float(outlier.mean()), 5),
            "rot_wrap_frac": round(float(rot_wrap.mean()), 5),
        }
        print(f"[diag s{stride}] n={len(Ds)} outlier={outlier.mean():.2%} rot_wrap={rot_wrap.mean():.2%} "
              f"std={np.round(Ds.std(0), 2).tolist()} p50={np.round(np.quantile(absDs, 0.5, axis=0), 2).tolist()} "
              f"raw_range=({np.round(Draw.min(0), 3).tolist()} ~ {np.round(Draw.max(0), 3).tolist()})", flush=True)
    results["target_diag"] = diag

    all_lat, all_tgt, all_epid = None, None, None
    for stride in STRIDES:
        cache = out_dir / f"cache_s{stride}.npz"  # 缓存的是剔毛刺后的数据;改阈值要删缓存
        if cache.exists():
            z = np.load(cache)
            L = torch.from_numpy(z["L"]); D = torch.from_numpy(z["D"])
            Y = torch.from_numpy(z["Y"]); epid = z["epid"]
            print(f"[stride {stride}] cache hit: {len(Y)} transitions", flush=True)
        else:
            lat_all, dv3_all, tgt_all, epid_all = [], [], [], []
            for ei, ep in enumerate(episodes):
                frames, cp = load_episode(ep)
                clips, targets, _ = build_clips(frames, cp, stride)
                if not clips:
                    continue
                lat, dv3 = encode_clips(model, clips)
                # target: per-window (7, 6) scaled
                tgt = torch.tensor(np.array(targets), dtype=torch.float32) / torch.tensor(SCALE)
                flat_lat = lat.reshape(-1, lat.shape[-1])
                flat_dv3 = dv3.reshape(-1, dv3.shape[-1])
                flat_tgt = tgt.reshape(-1, 6)
                good = (flat_tgt.abs() <= 8).all(-1)  # 剔毛刺/环绕跳变,保留静态帧
                lat_all.append(flat_lat[good]); dv3_all.append(flat_dv3[good])
                tgt_all.append(flat_tgt[good])
                epid_all += [ei] * int(good.sum())
            L = torch.cat(lat_all); D = torch.cat(dv3_all); Y = torch.cat(tgt_all)
            epid = np.array(epid_all)
            np.savez_compressed(cache, L=L.numpy(), D=D.numpy(), Y=Y.numpy(), epid=epid)
        # split by episode (contiguous) to avoid temporal leakage
        val_eps = set(np.random.RandomState(0).choice(len(episodes), max(1, len(episodes) // 5), replace=False).tolist())
        va = np.array([e in val_eps for e in epid])
        tr = ~va

        # 学习免费诊断:latent 范数与运动幅度的相关性(信息存在性直接证据)
        mag = Y.norm(dim=-1).numpy()
        corr_lat = float(np.corrcoef(L.norm(dim=-1).numpy(), mag)[0, 1])
        corr_dv3 = float(np.corrcoef(D.norm(dim=-1).numpy(), mag)[0, 1])
        print(f"[stride {stride}] kept {len(Y)} (train {int(tr.sum())} val {int(va.sum())}) "
              f"corr||z||={corr_lat:.3f} corr||dv3diff||={corr_dv3:.3f}", flush=True)

        sres = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
                "corr_znorm_mag": corr_lat, "corr_dv3norm_mag": corr_dv3,
                "target_var": [round(float(v), 4) for v in Y.var(dim=0).numpy()]}
        # 静态过滤分档:全量(仅剔毛刺) / >0.5 / >1.0 / 运动前 50%(stride>1 只跑两头)
        levels = [("all", 0.0), ("mov05", 0.5), ("bigmo", 1.0), ("top50", float(np.quantile(mag, 0.5)))]
        if stride > 1:
            levels = [levels[0], levels[3]]
        for tag, thr in levels:
            km = mag > thr
            ktr, kva = km & tr, km & va
            if kva.sum() < 200 or ktr.sum() < 1000:
                print(f"  [{tag}] 样本不足(train {int(ktr.sum())} val {int(kva.sum())}),跳过", flush=True)
                continue
            lm = train_probe(L[ktr], Y[ktr], L[kva], Y[kva])
            dm = train_probe(D[ktr], Y[ktr], D[kva], Y[kva])
            sres[f"ldm_{tag}"] = {"mse": lm[0], "r2": lm[2], "r2_train": lm[3]}
            sres[f"dv3_{tag}"] = {"mse": dm[0], "r2": dm[2], "r2_train": dm[3]}
            print(f"  [{tag} n={int(km.sum())}] LDM mse={lm[0]:.3f} r2={np.mean(lm[2]):.3f} (train {lm[3]:.3f}) | "
                  f"DINOv3 mse={dm[0]:.3f} r2={np.mean(dm[2]):.3f} (train {dm[3]:.3f})", flush=True)
        # 幅度对照:||Δxyz|| 与 ||Δrot|| 对任意固定坐标系旋转不变。
        # 若幅度可解码而逐维方向不可 -> "逐集外参错位"解释 floor;
        # 若幅度同样 floor -> "latent 本身缺运动信息"的解释增强。
        Ymag = torch.stack([Y[:, :3].norm(dim=-1), Y[:, 3:].norm(dim=-1)], dim=1)
        sres["mag_target_var"] = [round(float(v), 4) for v in Ymag.var(dim=0).numpy()]
        for tag, thr in levels:
            km = mag > thr
            ktr, kva = km & tr, km & va
            if kva.sum() < 200 or ktr.sum() < 1000:
                continue
            lmag = train_probe(L[ktr], Ymag[ktr], L[kva], Ymag[kva], epochs=30)
            sres[f"ldm_{tag}_mag"] = {"r2": lmag[2], "r2_train": lmag[3]}
            print(f"  [{tag}] LDM magnitude r2={np.mean(lmag[2]):.3f} (train {lmag[3]:.3f})", flush=True)
        # shuffle control: latent i predicts target of shuffled j
        rng = np.random.RandomState(1)
        Ysh = Y[rng.permutation(len(Y))]
        sres["shuffle_mse"] = train_probe(L[tr], Ysh[tr], L[va], Ysh[va], epochs=10)[0]
        print(f"  [shuffle floor] mse={sres['shuffle_mse']:.3f}", flush=True)
        results["strides"][str(stride)] = sres
        if stride == 1:
            all_lat, all_tgt, all_epid = L, Y, epid

    # B2: motion-vs-appearance retrieval on stride-1 latents
    # 只在运动迁移(幅度>0.5)上做——静态帧会让检索退化成"零向量互相像"。
    L1, Y1 = all_lat, all_tgt
    mag1 = Y1.norm(dim=-1).numpy()
    mov = np.where(mag1 > 0.5)[0]
    rng = np.random.RandomState(2)
    if len(mov) > 30000:
        mov = rng.choice(mov, 30000, replace=False)
    L1m, Y1m, ep1m = L1[mov], Y1[mov], all_epid[mov]
    L1n = nn.functional.normalize(L1m, dim=-1)
    sim = L1n @ L1n.T
    n_q = min(200, len(L1m))
    qidx = rng.choice(len(L1m), n_q, replace=False)
    motion_cos, appear_cos, cross_motion, same_motion = [], [], [], []
    for qi in qidx:
        s = sim[qi].clone(); s[qi] = -2
        top = s.topk(10).indices.cpu().numpy()
        qe = ep1m[qi]
        # motion agreement: cosine between scaled delta-EE vectors
        a = Y1m[qi].numpy(); b = Y1m[top].numpy()
        na, nb = np.linalg.norm(a) + 1e-9, np.linalg.norm(b, axis=1) + 1e-9
        mc = (b @ a) / (na * nb)
        motion_cos.append(mc.mean())
        # appearance decoy: same episode different transition — is it retrieved?
        same = np.array([ep1m[j] == qe for j in top])
        appear_cos.append(float(same.mean()))
        if (~same).any():
            cross_motion.append(float(mc[~same].mean()))  # 跨集(外观不同)运动一致性
        if same.any():
            same_motion.append(float(mc[same].mean()))
    results["retrieval"] = {
        "motion_cos_top10": float(np.mean(motion_cos)),
        "same_episode_frac_top10": float(np.mean(appear_cos)),
        "motion_cos_cross_episode": float(np.mean(cross_motion)) if cross_motion else None,
        "motion_cos_same_episode": float(np.mean(same_motion)) if same_motion else None,
        "n_queries": n_q, "n_bank": int(len(L1m)),
    }
    print(f"[retrieval] motion_cos={results['retrieval']['motion_cos_top10']:.3f} "
          f"cross_ep={results['retrieval']['motion_cos_cross_episode']:.3f} "
          f"same_ep={results['retrieval']['motion_cos_same_episode']:.3f} "
          f"same_ep_frac={results['retrieval']['same_episode_frac_top10']:.3f}")

    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))
    print("[done]", out_dir / "probe_results.json")


if __name__ == "__main__":
    main()
