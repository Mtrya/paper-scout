"""WAM interface toy, 2026-08-26 run: two experiments on one substrate.

Forked from runs/2026-08-14 wam-rollout-free joint_wam_v3.py (same 2-link arm
world, same 3D-UNet video FM + action-transformer backbone, same dataset).

Experiment C (LAWA anchor hypothesis):
  08-14 toy found learned anticipation tokens (rift-fm, FM-supervised toward
  continuous teacher bottleneck features) score 0% while ForeWAM noise slots
  reach 7.5%. LAWA (2608.24882) claims latent-action intentions DO work when
  anchored to a codebook from action-free video pretraining. Single-variable
  test: `lawatok` = rift-fm with the FM target quantized to a k-means codebook
  (K=256) built from video-only teacher features. Compare full-data and
  few-shot (10% episodes) against currentonly / rift-fm / noiseslots / joint.

Experiment D (mini-WorldEcho):
  WorldEcho (2608.24885) diagnoses that expert-only action-conditioned WMs
  fail off-expert action queries. Replicate in the toy where EE ground truth
  is free (no tracker): train an action-conditioned video model (ACWM) on
  expert demos, query it with demonstrated / cross-state / perturbed / random
  action sequences, measure EE-ADE between generated frames (EE read out by
  skeleton fitting) and simulator replay. Ablations:
    acwm      expert-only training
    acwm-cov  + coverage: FM also on perturbed-action replay videos
    acwm-ie   expert FM only + intervention-effect loss on paired branches
              (same obs, same noise endpoint, different actions)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

BOTTLENECK_T = 3
L1, L2 = 16.0, 13.0


# ================================================================ model
def sigma_schedule(u):
    return 5 * u / (1 + 4 * u)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32,
                                                      device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock3D(nn.Module):
    def __init__(self, ch, tch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(tch, ch)

    def forward(self, x, temb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(F.silu(temb))[:, :, None, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class ActionBlock(nn.Module):
    def __init__(self, d=256, heads=8, mlp=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.sa_qkv = nn.Linear(d, 3 * d)
        self.sa_proj = nn.Linear(d, d)
        self.ca_q = nn.Linear(d, d)
        self.ca_k = nn.Linear(d, d)
        self.ca_v = nn.Linear(d, d)
        self.ca_proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.heads = heads

    def _attn(self, q, K, V, drop=False):
        heads, d = self.heads, q.shape[-1]
        B, Q, _ = q.shape
        qh = q.reshape(B, Q, heads, d // heads).permute(0, 2, 1, 3)
        kh = K.reshape(B, -1, heads, d // heads).permute(0, 2, 1, 3)
        vh = V.reshape(B, -1, heads, d // heads).permute(0, 2, 1, 3)
        att = torch.softmax(torch.einsum("bhqd,bhsd->bhqs", qh, kh)
                            * (d // heads) ** -0.5, dim=-1)
        if drop:
            att = att * 0
        return torch.einsum("bhqs,bhsd->bhqd", att, vh).reshape(B, Q, d)

    def forward(self, h, cache_k, cache_v, drop_fut=False):
        h1 = self.ln1(h)
        qkv = self.sa_qkv(h1)
        d = h.shape[-1]
        h = h + self.sa_proj(self._attn(qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]))
        h = h + self.ca_proj(self._attn(self.ca_q(self.ln2(h)),
                                        self.ca_k(cache_k), self.ca_v(cache_v),
                                        drop=drop_fut))
        h = h + self.mlp(self.ln3(h))
        return h


class ToyJointV3(nn.Module):
    """v3 backbone + optional action-chunk conditioning of the video branch."""

    def __init__(self, act_len=16, act_dim=3, base=64, tch=256, a_d=256,
                 a_depth=4, n_modes=2):
        super().__init__()
        self.act_len, self.act_dim, self.base = act_len, act_dim, base
        self.temb = nn.Sequential(nn.Linear(base, tch), nn.SiLU(), nn.Linear(tch, tch))
        self.mode_emb = nn.Embedding(n_modes, a_d)
        self.mode_emb_v = nn.Embedding(n_modes, base * 4)
        chs = [base, base * 2, base * 4]
        self.in_conv = nn.Conv3d(3, chs[0], 3, padding=1)
        self.down1 = nn.Sequential(ResBlock3D(chs[0], tch), nn.Conv3d(chs[0], chs[1], 3, stride=2, padding=1))
        self.down2 = nn.Sequential(ResBlock3D(chs[1], tch), nn.Conv3d(chs[1], chs[2], 3, stride=2, padding=1))
        self.bottleneck = ResBlock3D(chs[2], tch)
        self.cache_proj_k = nn.Linear(chs[2], a_d)
        self.cache_proj_v = nn.Linear(chs[2], a_d)
        self.up1 = nn.Sequential(ResBlock3D(chs[2], tch),
                                 nn.ConvTranspose3d(chs[2], chs[1], 3, stride=2, padding=1, output_padding=1))
        self.up2 = nn.Sequential(ResBlock3D(chs[1], tch),
                                 nn.ConvTranspose3d(chs[1], chs[0], 3, stride=2, padding=1, output_padding=1))
        self.out_block = ResBlock3D(chs[0], tch)
        self.out_conv = nn.Conv3d(chs[0], 3, 3, padding=1)
        self.act_in = nn.Linear(act_dim, a_d)
        self.act_blocks = nn.ModuleList([ActionBlock(a_d) for _ in range(a_depth)])
        self.action_head = nn.Linear(a_d, act_dim)
        # NEW: action chunk -> video branch conditioning (for ACWM variants)
        self.act_cond = nn.Linear(act_len * act_dim, tch)

    def _te(self, sigma):
        return self.temb(timestep_embedding(sigma, self.base))

    def _video_forward(self, x, sigma, mode, act_seq=None):
        """x: (B,3,9,H,W) [obs clean | 8 noised future] in [-1,1].
        act_seq: (B,act_len,act_dim) or None."""
        B = x.shape[0]
        temb = self._te(sigma)
        if act_seq is not None:
            temb = temb + self.act_cond(act_seq.reshape(B, -1))
        mode_ctx = self.mode_emb_v(mode)[:, None, :, None, None]
        h0 = self.in_conv(x)
        h1 = self.down1[0](h0, temb)
        h1 = self.down1[1](h1)
        h2 = self.down2[0](h1, temb)
        h2 = self.down2[1](h2)
        hb = self.bottleneck(h2, temb)
        Bc, Cb, Tb, Hb, Wb = hb.shape
        bm = hb.permute(0, 2, 3, 4, 1).reshape(B, Tb * Hb * Wb, Cb)
        bm = bm + mode_ctx.reshape(B, 1, -1).repeat(1, Tb * Hb * Wb, 1)
        cache_k = self.cache_proj_k(bm)
        cache_v = self.cache_proj_v(bm)
        u1 = self.up1[0](hb, temb)
        u1 = self.up1[1](u1)
        u1 = u1[:, :, : h1.shape[2]] + h1
        u2 = self.up2[0](u1, temb)
        u2 = self.up2[1](u2)
        u2 = u2[:, :, : h0.shape[2]] + h0
        h = self.out_block(u2, temb)
        vel = self.out_conv(h)
        return vel, (cache_k, cache_v), bm

    def joint_forward(self, x, act_x, sigma, mode, record=False):
        vel_v, cache, _ = self._video_forward(x, sigma, mode)
        h = self.act_in(act_x) + self.mode_emb(mode)[:, None]
        for blk in self.act_blocks:
            h = blk(h, cache[0], cache[1])
        return vel_v, self.action_head(h), cache

    def action_forward(self, cache, act_x, mode, drop_fut=False):
        h = self.act_in(act_x) + self.mode_emb(mode)[:, None]
        for blk in self.act_blocks:
            h = blk(h, cache[0], cache[1], drop_fut=drop_fut)
        return self.action_head(h)

    def anticipation_prefill(self, obs_frames, E, mode):
        x = torch.cat([obs_frames.permute(0, 2, 1, 3, 4), E], dim=2)
        _, cache, bm = self._video_forward(x, torch.zeros(1, device=x.device), mode)
        return bm, cache


# ================================================================ simulator
class ArmSim:
    def __init__(self, bg, start, target, H=64, W=64):
        self.bg, self.H, self.W = bg, H, W
        self.base = np.array([W * 0.5, H - 6])
        self.t1, self.t2, self.grip = 0.0, -0.5, 0.0
        self.cube = np.array(start, dtype=float)
        self.target = np.array(target, dtype=float)
        self.attached = False

    def set_state(self, t1, t2, grip, cube, attached=None):
        self.t1, self.t2, self.grip = float(t1), float(t2), float(grip)
        self.cube = np.array(cube, dtype=float)
        self.attached = bool(grip > 0.5) if attached is None else attached

    def ee(self):
        j1 = self.base + L1 * np.array([np.sin(self.t1), -np.cos(self.t1)])
        return j1 + L2 * np.array([np.sin(self.t1 + self.t2),
                                   -np.cos(self.t1 + self.t2)])

    def step(self, action):
        self.t1, self.t2 = float(action[0]), float(action[1])
        self.grip = float(action[2])
        if self.grip > 0.5:
            if not self.attached and np.linalg.norm(self.cube - self.ee()) < 7:
                self.attached = True
            if self.attached:
                self.cube = self.ee().copy()
                self.cube[1] = min(self.cube[1], self.H - 16)
        else:
            self.attached = False
        return self.render()

    def render(self):
        img = self.bg.copy()
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        d.rectangle([0, self.H - 10, self.W, self.H - 4], fill=(90, 90, 90))
        cs = 7
        cx, cy = int(self.cube[0]), int(self.cube[1])
        d.rectangle([cx - cs // 2, cy - cs // 2, cx + cs // 2, cy + cs // 2],
                    fill=(255, 0, 0))
        pts = [tuple(self.base),
               tuple(self.base + L1 * np.array([np.sin(self.t1), -np.cos(self.t1)])),
               tuple(self.ee())]
        d.line([pts[0], pts[1]], fill=(200, 200, 200), width=3)
        d.line([pts[1], pts[2]], fill=(200, 200, 200), width=3)
        return np.asarray(pil)

    def success(self, tol=8.0):
        return bool(np.abs(self.cube - self.target).max() < tol)


def make_sim(npz):
    z = np.load(npz)
    return ArmSim(z["bg"], z["start"], z["target"]), z


# ================================================================ EE readout
def draw_arm_into(img, theta1, theta2, gripper, color=(200, 200, 200), width=3):
    """Dataset-style arm drawing (copied from gen_joint_data.py so that replay
    renders match the training distribution exactly)."""
    H, W = img.shape[:2]
    base = np.array([W * 0.5, H - 6])
    j1 = base + L1 * np.array([np.sin(theta1), -np.cos(theta1)])
    a2 = theta1 + theta2
    ee = j1 + L2 * np.array([np.sin(a2), -np.cos(a2)])
    pts = [tuple(base), tuple(j1), tuple(ee)]
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.line([pts[0], pts[1]], fill=color, width=width)
    d.line([pts[1], pts[2]], fill=color, width=width)
    d.ellipse([pts[0][0]-3, pts[0][1]-3, pts[0][0]+3, pts[0][1]+3], fill=color)
    d.ellipse([pts[1][0]-2, pts[1][1]-2, pts[1][0]+2, pts[1][1]+2], fill=color)
    g = 4 if gripper > 0.5 else 8
    d.line([(ee[0]-g, ee[1]), (ee[0]+g, ee[1])], fill=color, width=2)
    d.line([(ee[0]-g, ee[1]-3), (ee[0]-g, ee[1]+3)], fill=color, width=2)
    d.line([(ee[0]+g, ee[1]-3), (ee[0]+g, ee[1]+3)], fill=color, width=2)
    return np.asarray(pil)


def render_state(bg, t1, t2, grip, cube):
    img = bg.copy()
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 64 - 10, 64, 64 - 4], fill=(90, 90, 90))
    cx, cy = int(cube[0]), int(cube[1])
    d.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 0, 0))
    return draw_arm_into(np.asarray(pil), t1, t2, grip)


def fk_ee(t1, t2, H=64, W=64):
    base = np.array([W * 0.5, H - 6])
    j1 = base + L1 * np.array([np.sin(t1), -np.cos(t1)])
    ee = j1 + L2 * np.array([np.sin(t1 + t2), -np.cos(t1 + t2)])
    return base, j1, ee


def _line_pixels(p0, p1):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    return np.stack([np.round(xs), np.round(ys)], axis=-1).astype(int)


def skeleton_pixels(t1, t2, H=64, W=64):
    base, j1, ee = fk_ee(t1, t2, H, W)
    px = np.concatenate([_line_pixels(base, j1), _line_pixels(j1, ee)], axis=0)
    ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return px[ok]


def arm_mask(frame, bg=None):
    """Arm pixels: bright gray-ish. If bg given, also require clear difference
    from background (grad backgrounds can reach ~180 gray)."""
    f = frame.astype(np.int32)
    mx = f.max(axis=-1)
    mn = f.min(axis=-1)
    m = (mx > 150) & ((mx - mn) < 30)
    if bg is not None:
        d = np.abs(f - bg.astype(np.int32)).max(axis=-1)
        m = m & (d > 50)
    return m


def fit_joints(frame, bg=None, H=64, W=64):
    """Grid-search (t1,t2) maximizing skeleton/mask Chamfer agreement.
    Returns ((t1,t2), score) or (None, 1e9) if no arm found."""
    from scipy.ndimage import distance_transform_edt  # noqa
    mask = arm_mask(frame, bg)
    if mask.sum() < 8:
        return None, 1e9
    edt = distance_transform_edt(~mask)
    ys, xs = np.nonzero(mask)
    mask_pts = np.stack([xs, ys], axis=-1)

    edt = distance_transform_edt(~mask)
    ys, xs = np.nonzero(mask)
    mask_pts = np.stack([xs, ys], axis=-1)

    def d1(t1, t2):
        sk = skeleton_pixels(t1, t2, H, W)
        if len(sk) == 0:
            return 1e9
        return edt[sk[:, 1], sk[:, 0]].mean()

    def chamfer_full(t1, t2):
        sk = skeleton_pixels(t1, t2, H, W)
        if len(sk) == 0:
            return 1e9
        dd1 = edt[sk[:, 1], sk[:, 0]].mean()
        sk_set = np.zeros((H, W), bool)
        sk_set[sk[:, 1], sk[:, 0]] = True
        d2 = distance_transform_edt(~sk_set)[ys, xs].mean() if len(ys) else 0.0
        return dd1 + 0.5 * d2

    # stage 1: coarse grid scored by d1 only (cheap), keep top-K candidates
    cands = []
    for t1 in np.linspace(-1.9, 1.9, 39):
        for t2 in np.linspace(-2.9, 2.9, 59):
            cands.append((d1(t1, t2), t1, t2))
    cands.sort()
    # stage 2: rescore top candidates with full symmetric chamfer
    best, bs = None, 1e9
    for _, t1, t2 in cands[:40]:
        s = chamfer_full(t1, t2)
        if s < bs:
            bs, best = s, (t1, t2)
    if best is None:
        return None, 1e9
    t1c, t2c = best
    for span in (0.1, 0.02):
        for t1 in np.linspace(t1c - span * 2, t1c + span * 2, 9):
            for t2 in np.linspace(t2c - span * 2, t2c + span * 2, 9):
                s = chamfer_full(t1, t2)
                if s < bs:
                    bs, best = s, (t1, t2)
        t1c, t2c = best
    return best, bs


# ================================================================ data
class JointDataset(torch.utils.data.Dataset):
    def __init__(self, root, files):
        self.files = files
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(os.path.join(self.root, self.files[i]))
        frames = torch.from_numpy(z["frames"]).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        action = torch.from_numpy(z["action"]).float()
        mode = int(z["mode"])
        rng = np.random.default_rng(i)
        f = int(rng.integers(0, 8))
        obs = frames[f:f + 1].clone()
        fut = frames[f + 1:f + 9].clone()
        act = action[2 * f:2 * f + 16].clone()
        return obs, fut, act, mode, f, self.files[i]


def to_model_space(t):
    return t * 2 - 1


def replay_frames(z, f, actions, T=8):
    """Replay `actions` (16,3) from episode state at frame f; return rendered
    future frames (T,64,64,3) uint8 (dataset-style rendering) and per-frame EE
    traj (T,2). Phase convention verified against the dataset: step s applies
    actions[s] (last action held for s>=len); future frame f+k is the state
    after s=2k (readout at s=2,4,...,16 reproduces dataset frames f+1..f+8
    with mean abs pixel diff ~1.0 vs ~3.9 for the odd-phase alternative)."""
    t1, t2, grip = z["joints"][f]
    cube = z["cube"][f].astype(float).copy()
    attached = bool(grip > 0.5)
    frames, ees = [], []
    for s in range(2 * T + 2):
        a = actions[min(s, len(actions) - 1)]
        t1, t2, grip = float(a[0]), float(a[1]), float(a[2])
        base = np.array([32.0, 58.0])
        j1 = base + L1 * np.array([np.sin(t1), -np.cos(t1)])
        ee = j1 + L2 * np.array([np.sin(t1 + t2), -np.cos(t1 + t2)])
        if grip > 0.5:
            if not attached and np.linalg.norm(cube - ee) < 7:
                attached = True
            if attached:
                cube = ee.copy()
                cube[1] = min(cube[1], 48)
        else:
            attached = False
        if s % 2 == 0 and s > 0:
            frames.append(render_state(z["bg"], t1, t2, grip, cube))
            ees.append(ee.copy())
    return np.stack(frames[:T]), np.array(ees[:T])


# ================================================================ C: lawatok
@torch.no_grad()
def build_codebook(model, dl, dev, K=256, n_samples=800, iters=25, seed=0):
    """K-means over teacher bottleneck features of clean [obs|fut] (video-only:
    actions never touched). Returns (K, Cb) codebook."""
    model.eval()
    feats = []
    for i, (obs, fut, act, mode, _, _) in enumerate(dl):
        if i * obs.shape[0] >= n_samples:
            break
        obs = to_model_space(obs).to(dev)
        fut = to_model_space(fut).to(dev)
        mode = mode.to(dev)
        x0 = torch.cat([obs, fut], dim=1).permute(0, 2, 1, 3, 4)
        _, _, bm = model._video_forward(x0, torch.zeros(obs.shape[0], device=dev), mode)
        feats.append(bm.reshape(-1, bm.shape[-1]).float().cpu())
    X = torch.cat(feats)
    print(f"[codebook] features {tuple(X.shape)}", flush=True)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:K]
    C = X[idx].clone()
    for it in range(iters):
        d = torch.cdist(X, C)
        a = d.argmin(dim=1)
        for k in range(K):
            sel = a == k
            if sel.any():
                C[k] = X[sel].mean(0)
        if it % 5 == 0:
            print(f"[codebook] iter {it} mean_dist={d.min(1).values.mean():.4f}", flush=True)
    return C


def quantize(bm, codebook):
    d = torch.cdist(bm.float(), codebook.to(bm.device).float())
    return codebook.to(bm.device)[d.argmin(dim=-1)]


# ================================================================ training
def train_policy_variant(model, dl, dev, steps, lr, variant, E=None,
                         codebook=None, out_every=500, log_prefix=""):
    """Policy-side variants (C). lawatok == rift-fm with codebook-quantized FM target."""
    params = list(model.parameters()) + ([E] if E is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    step, t0 = 0, time.time()
    curve = []
    dl_iter = iter(dl)
    while step < steps:
        try:
            obs, fut, action, mode, _, _ = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            continue
        model.train()
        obs = to_model_space(obs).to(dev)
        fut = to_model_space(fut).to(dev)
        action = action.to(dev); mode = mode.to(dev)
        B = obs.shape[0]
        obs_v = obs.permute(0, 2, 1, 3, 4)  # (B,3,1,H,W)
        fut_p = fut.permute(0, 2, 1, 3, 4)
        u = torch.rand(B, device=dev)
        sig = sigma_schedule(u)
        eps_v = torch.randn_like(fut_p)
        eps_a = torch.randn(action.shape, device=dev)
        xv = torch.cat([obs_v, (1 - sig)[:, None, None, None, None] * fut_p
                        + sig[:, None, None, None, None] * eps_v], dim=2)
        xa = (1 - sig)[:, None, None] * action + sig[:, None, None] * eps_a

        if variant == "joint":
            vel_v, vel_a, _ = model.joint_forward(xv, xa, sig, mode)
            loss = F.mse_loss(vel_v[:, :, 1:], eps_v - fut_p) + F.mse_loss(vel_a, eps_a - action)
        elif variant == "currentonly":
            _, cache, _ = model._video_forward(xv, sig, mode)
            vel_a = model.action_forward(cache, xa, mode, drop_fut=True)
            loss = F.mse_loss(vel_a, eps_a - action)
        elif variant in ("rift-fm", "lawatok", "noiseslots"):
            vel_v, _, _ = model._video_forward(xv, sig, mode)
            if variant == "noiseslots":
                noise_fut = torch.randn(B, 3, 8, obs.shape[-2], obs.shape[-1], device=dev)
                x_pre = torch.cat([obs_v, noise_fut], dim=2)
                _, cache, _ = model._video_forward(x_pre, torch.ones(B, device=dev), mode)
                vel_a = model.action_forward(cache, xa, mode)
                loss = F.mse_loss(vel_v[:, :, 1:], eps_v - fut_p) + F.mse_loss(vel_a, eps_a - action)
            else:
                E_b = E.expand(B, -1, -1, -1, -1)
                bm, cache = model.anticipation_prefill(obs, E_b, mode)
                vel_a = model.action_forward(cache, xa, mode)
                loss = F.mse_loss(vel_v[:, :, 1:], eps_v - fut_p) + F.mse_loss(vel_a, eps_a - action)
                with torch.no_grad():
                    x0 = torch.cat([obs, fut], dim=1).permute(0, 2, 1, 3, 4)
                    _, _, bm_t = model._video_forward(x0, torch.zeros(B, device=dev), mode)
                    if variant == "lawatok":
                        bm_t = quantize(bm_t, codebook)
                eps_f = torch.randn_like(bm_t)
                u2 = torch.rand(B, device=dev)
                sig2 = sigma_schedule(u2)
                X = (1 - sig2)[:, None, None] * bm_t + sig2[:, None, None] * eps_f
                psi_in = torch.cat([X, model.sup_proj(bm),
                                    model._te(sig2)[:, None].expand(-1, bm.shape[1], -1)], dim=-1)
                loss = loss + F.mse_loss(model.fm_head(psi_in), eps_f - bm_t)
        else:
            raise ValueError(variant)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % out_every == 0:
            curve.append([step, float(loss)])
            print(f"[{log_prefix}{variant}] step {step}/{steps} loss={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
            t0 = time.time()
    return curve


def train_acwm(model, dl, data_root, dev, steps, lr, variant, perturb=0.15,
               lam_ie=1.0, out_every=500):
    """Experiment D variants: acwm / acwm-cov / acwm-ie. Video FM conditioned
    on action chunk; cov adds perturbed-action replay videos to FM; ie adds
    paired intervention-effect loss (same obs, same noise, different actions)
    WITHOUT adding the perturbed video to the FM loss."""
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    step, t0 = 0, time.time()
    dl_iter = iter(dl)
    rng = np.random.default_rng(0)
    while step < steps:
        try:
            obs, fut, action, mode, f_idx, fname = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            continue
        model.train()
        obs = to_model_space(obs).to(dev)
        fut = to_model_space(fut).to(dev)
        action = action.to(dev); mode = mode.to(dev)
        B = obs.shape[0]
        obs_v2 = obs.permute(0, 2, 1, 3, 4)  # (B,3,1,H,W)
        fut_p = fut.permute(0, 2, 1, 3, 4)

        def fm_loss_on(obs_b, vid_fut_p, act_b):
            u = torch.rand(obs_b.shape[0], device=dev)
            sig = sigma_schedule(u)
            eps_v = torch.randn_like(vid_fut_p)
            xv = torch.cat([obs_b.permute(0, 2, 1, 3, 4), (1 - sig)[:, None, None, None, None] * vid_fut_p
                            + sig[:, None, None, None, None] * eps_v], dim=2)
            vel_v, _, _ = model._video_forward(xv, sig, mode[:obs_b.shape[0]] if mode.shape[0] == obs_b.shape[0] else mode, act_seq=act_b)
            return F.mse_loss(vel_v[:, :, 1:], eps_v - vid_fut_p)

        loss = fm_loss_on(obs, fut_p, action)

        if variant in ("acwm-cov", "acwm-ie"):
            # build perturbed-action replay batch (CPU render)
            vids_b, acts_b, obs_b_list, modes_b = [], [], [], []
            for b in range(B):
                z = np.load(os.path.join(data_root, "train", fname[b]))
                f = int(f_idx[b])
                a_exp = z["action"][2 * f:2 * f + 16].astype(np.float32)
                a_pert = a_exp + rng.normal(0, perturb, a_exp.shape).astype(np.float32)
                a_pert[:, 2] = a_exp[:, 2]  # keep gripper discrete
                frames_b, _ = replay_frames(z, f, a_pert)
                vids_b.append(torch.from_numpy(frames_b).float().div(255).permute(0, 3, 1, 2))
                acts_b.append(torch.from_numpy(a_pert))
                obs_b_list.append(z["frames"][f].copy())
                modes_b.append(int(z["mode"]))
            fut_b = to_model_space(torch.stack(vids_b)).to(dev)  # (B,8,3,H,W) in [-1,1]
            fut_b = fut_b.permute(0, 2, 1, 3, 4)
            act_b = torch.stack(acts_b).to(dev)
            obs_b = to_model_space(torch.from_numpy(np.stack(obs_b_list)).float().div(255)).to(dev).permute(0, 3, 1, 2).unsqueeze(1)
            mode_b = torch.tensor(modes_b, device=dev)
            if variant == "acwm-cov":
                loss = loss + fm_loss_on(obs_b, fut_b, act_b)
            else:  # acwm-ie
                # paired IE: same obs, same noise endpoint, same sigma;
                # v_A - v_B aligned to x0_B - x0_A
                u = torch.rand(B, device=dev)
                sig = sigma_schedule(u)
                eps = torch.randn_like(fut_p)  # shared noise
                xv_a = torch.cat([obs_v2, (1 - sig)[:, None, None, None, None] * fut_p
                                  + sig[:, None, None, None, None] * eps], dim=2)
                xv_b_same = torch.cat([obs_v2, (1 - sig)[:, None, None, None, None] * fut_b
                                       + sig[:, None, None, None, None] * eps], dim=2)
                v_a, _, _ = model._video_forward(xv_a, sig, mode, act_seq=action)
                v_b, _, _ = model._video_forward(xv_b_same, sig, mode, act_seq=act_b)
                delta_pred = v_a[:, :, 1:] - v_b[:, :, 1:]
                delta_tgt = fut_b - fut_p  # = x0_B - x0_A
                loss = loss + lam_ie * F.mse_loss(delta_pred, delta_tgt)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % out_every == 0:
            print(f"[{variant}] step {step}/{steps} loss={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
            t0 = time.time()
    return


# ================================================================ rollout & eval
@torch.no_grad()
def rollout_joint(model, sim, mode, steps=10, seed=0, max_chunks=8, exec_k=4, device="cuda"):
    torch.manual_seed(seed)
    ee_traj = []
    obs = sim.render()
    for chunk in range(max_chunks):
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).float().div(255).to(device)
        obs_t = to_model_space(obs_t)[None].permute(0, 3, 1, 2).unsqueeze(2)
        H, W = obs.shape[:2]
        xv = torch.randn(1, 3, 8, H, W, device=device)
        xa = torch.randn(1, model.act_len, model.act_dim, device=device)
        dt = 1.0 / steps
        for k in range(steps):
            sig = torch.full((1,), 1.0 - k * dt, device=device)
            x = torch.cat([obs_t, xv], dim=2)
            vel_v, vel_a, cache = model.joint_forward(x, xa, sig, torch.tensor([mode], device=device))
            xv = xv - vel_v[:, :, 1:] * dt
            xa = xa - vel_a * dt
        act = xa[0].cpu().numpy()
        for s in range(exec_k):
            obs = sim.step(act[s])
            ee_traj.append(sim.ee())
        if sim.success():
            break
    return sim.success(), np.array(ee_traj)


@torch.no_grad()
def rollout_producer(model, sim, mode, variant, E, steps=10, seed=0, max_chunks=8,
                     exec_k=4, device="cuda"):
    torch.manual_seed(seed)
    obs = sim.render()
    for chunk in range(max_chunks):
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).float().div(255).to(device)
        obs_t = to_model_space(obs_t)[None].permute(0, 3, 1, 2).unsqueeze(2)
        H, W = obs.shape[:2]
        _ = torch.randn(1, 3, 8, H, W, device=device)
        if variant == "noiseslots":
            noise_fut = torch.randn(1, 3, 8, H, W, device=device)
            x_pre = torch.cat([obs_t, noise_fut], dim=2)
            _, cache, _ = model._video_forward(x_pre, torch.ones(1, device=device), torch.tensor([mode], device=device))
        elif variant == "currentonly":
            x_pre = torch.cat([obs_t, torch.randn(1, 3, 8, H, W, device=device)], dim=2)
            _, cache, _ = model._video_forward(x_pre, torch.zeros(1, device=device), torch.tensor([mode], device=device))
        else:
            _, cache = model.anticipation_prefill(obs_t.permute(0, 2, 1, 3, 4), E, torch.tensor([mode], device=device))
        xa = torch.randn(1, model.act_len, model.act_dim, device=device)
        dt = 1.0 / steps
        for k in range(steps):
            sig = torch.full((1,), 1.0 - k * dt, device=device)
            vel_a = model.action_forward(cache, xa, torch.tensor([mode], device=device),
                                         drop_fut=(variant == "currentonly"))
            xa = xa - vel_a * dt
        act = xa[0].cpu().numpy()
        for s in range(exec_k):
            obs = sim.step(act[s])
        if sim.success():
            break
    return sim.success()


@torch.no_grad()
def gen_future_video(model, obs_np, act_seq, mode, steps=10, device="cuda"):
    """ACWM: generate 8 future frames conditioned on obs + action chunk."""
    obs_t = torch.from_numpy(np.ascontiguousarray(obs_np)).float().div(255).to(device)
    obs_t = to_model_space(obs_t)[None].permute(0, 3, 1, 2).unsqueeze(2)
    H, W = obs_np.shape[:2]
    xv = torch.randn(1, 3, 8, H, W, device=device)
    act_t = torch.from_numpy(act_seq).float().to(device)[None]
    dt = 1.0 / steps
    for k in range(steps):
        sig = torch.full((1,), 1.0 - k * dt, device=device)
        x = torch.cat([obs_t, xv], dim=2)
        vel_v, _, _ = model._video_forward(x, sig, torch.tensor([mode], device=device), act_seq=act_t)
        xv = xv - vel_v[:, :, 1:] * dt
    vid = ((xv[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    return (vid * 255).astype(np.uint8)


def eval_policy_variant(model, variant, E, data_root, dev, n=100):
    files = sorted(os.listdir(os.path.join(data_root, "val")))[:n]
    srs = []
    for f in files:
        sim, z = make_sim(os.path.join(data_root, "val", f))
        mode = int(z["mode"])
        if variant == "joint":
            ok, _ = rollout_joint(model, sim, mode, device=dev)
        else:
            ok = rollout_producer(model, sim, mode, variant, E, device=dev)
        srs.append(bool(ok))
    return dict(sr=float(np.mean(srs)), n=len(srs))


def eval_acwm(model, data_root, dev, n=60, perturb=0.15, seed=1):
    """mini-WorldEcho: four query categories, EE-ADE (px) + arm-loss rate."""
    rng = np.random.default_rng(seed)
    files = sorted(os.listdir(os.path.join(data_root, "val")))[:n]
    cats = ["demo", "xstate", "perturb", "random"]
    out = {c: dict(ade=[], armloss=[]) for c in cats}
    for i, f in enumerate(files):
        z = np.load(os.path.join(data_root, "val", f))
        f0 = 0
        obs_np = z["frames"][f0]
        a_demo = z["action"][2 * f0:2 * f0 + 16].astype(np.float32)
        z2 = np.load(os.path.join(data_root, "val", files[(i + 37) % len(files)]))
        a_xstate = z2["action"][2 * f0:2 * f0 + 16].astype(np.float32)
        a_pert = a_demo + rng.normal(0, perturb, a_demo.shape).astype(np.float32)
        a_pert[:, 2] = a_demo[:, 2]
        a_rand = a_demo + rng.normal(0, 0.45, a_demo.shape).astype(np.float32)
        a_rand[:, 2] = (rng.random(16) > 0.5).astype(np.float32)
        queries = dict(demo=a_demo, xstate=a_xstate, perturb=a_pert, random=a_rand)
        for c, aq in queries.items():
            vid = gen_future_video(model, obs_np, aq, int(z["mode"]), device=dev)
            _, ee_gt_f = replay_frames(z, f0, aq)  # (8,2) per-frame GT
            ee_gen, fails = [], 0
            for fr in vid:
                fit, score = fit_joints(fr, bg=z["bg"])
                if fit is None or score > 3.0:
                    fails += 1
                    ee_gen.append([np.nan, np.nan])
                else:
                    _, _, ee = fk_ee(fit[0], fit[1])
                    ee_gen.append(ee)
            ee_gen = np.array(ee_gen)  # (8,2), one per frame
            valid = ~np.isnan(ee_gen[:, 0])
            if valid.any():
                ade = float(np.mean(np.linalg.norm(ee_gen[valid] - ee_gt_f[valid], axis=1)))
            else:
                ade = float("nan")
            out[c]["ade"].append(ade)
            out[c]["armloss"].append(fails / len(vid))
    summ = {}
    for c in cats:
        summ[c] = dict(ade=float(np.nanmean(out[c]["ade"])),
                       ade_med=float(np.nanmedian(out[c]["ade"])),
                       armloss=float(np.mean(out[c]["armloss"])),
                       n_valid=int(np.sum(~np.isnan(out[c]["ade"]))))
    return summ


# ================================================================ main
def load_joint_teacher(path, dev, base=64):
    model = ToyJointV3(base=base)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="joint_data")
    ap.add_argument("--out", default="results_if")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--teacher", default="results/joint_v3.pt")
    ap.add_argument("--codebook", default="codebook_v1.pt")
    # C
    ap.add_argument("--c-variants", nargs="+", default=[])
    ap.add_argument("--c-steps", type=int, default=12000)
    ap.add_argument("--c-fewshot-n", type=int, default=0, help=">0: use only N train episodes")
    ap.add_argument("--c-eval-n", type=int, default=100)
    ap.add_argument("--c-tag", default="c1")
    # D
    ap.add_argument("--d-variants", nargs="+", default=[])
    ap.add_argument("--d-steps", type=int, default=8000)
    ap.add_argument("--d-eval-n", type=int, default=60)
    ap.add_argument("--d-tag", default="d1")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    all_files = sorted(os.listdir(os.path.join(args.data, "train")))

    # ---------------- experiment C
    if args.c_variants:
        n_train = args.c_fewshot_n if args.c_fewshot_n > 0 else None
        files = all_files[:n_train] if n_train else all_files
        ds = JointDataset(os.path.join(args.data, "train"), files)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                         num_workers=2, drop_last=True)
        codebook = None
        if any(v == "lawatok" for v in args.c_variants):
            cb_path = os.path.join(args.out, args.codebook)
            if os.path.exists(cb_path):
                codebook = torch.load(cb_path)
                print(f"[codebook] loaded {tuple(codebook.shape)}", flush=True)
            else:
                teacher = load_joint_teacher(args.teacher, dev, base=args.base)
                codebook = build_codebook(teacher, dl, dev)
                torch.save(codebook, cb_path)
                del teacher
        results = {}
        for variant in args.c_variants:
            tag = args.c_tag
            ckpt_path = os.path.join(args.out, f"{variant}_{tag}.pt")
            if not args.eval_only and not os.path.exists(ckpt_path):
                torch.manual_seed(0)
                model = ToyJointV3(base=args.base).to(dev)
                E = None
                if variant in ("rift-fm", "lawatok"):
                    E = nn.Parameter(torch.randn(1, 3, 8, 64, 64, device=dev) * 0.02)
                    model.sup_proj = nn.Linear(args.base * 4, args.base * 4).to(dev)
                    model.fm_head = nn.Sequential(
                        nn.Linear(args.base * 8 + 256, 512), nn.SiLU(),
                        nn.Linear(512, args.base * 4)).to(dev)
                train_policy_variant(model, dl, dev, args.c_steps, args.lr, variant,
                                     E=E, codebook=codebook, log_prefix=f"{tag}/")
                torch.save({"model": model.state_dict(),
                            "E": E.detach().cpu() if E is not None else None}, ckpt_path)
                print(f"[{variant}] saved", flush=True)
            # eval
            model = ToyJointV3(base=args.base)
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "rift" in variant or "lawatok" in variant:
                    model.sup_proj = nn.Linear(args.base * 4, args.base * 4)
                    model.fm_head = nn.Sequential(nn.Linear(args.base * 8 + 256, 512), nn.SiLU(),
                                                  nn.Linear(512, args.base * 4))
                model.load_state_dict(ckpt["model"], strict=False)
                E = ckpt["E"]
                model = model.to(dev).eval()
                if E is not None:
                    E = E.to(dev)
                res = eval_policy_variant(model, variant, E, args.data, dev, n=args.c_eval_n)
                results[variant] = res
                print(f"[eval {tag}/{variant}] {json.dumps(res)}", flush=True)
                del model
                torch.cuda.empty_cache()
        with open(os.path.join(args.out, f"eval_{args.c_tag}.json"), "w") as fo:
            json.dump(results, fo, indent=2)

    # ---------------- experiment D
    if args.d_variants:
        ds = JointDataset(os.path.join(args.data, "train"), all_files)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                         num_workers=2, drop_last=True)
        results = {}
        for variant in args.d_variants:
            ckpt_path = os.path.join(args.out, f"{variant}_{args.d_tag}.pt")
            if not args.eval_only and not os.path.exists(ckpt_path):
                torch.manual_seed(0)
                model = ToyJointV3(base=args.base).to(dev)
                train_acwm(model, dl, args.data, dev, args.d_steps, args.lr, variant)
                torch.save({"model": model.state_dict()}, ckpt_path)
                print(f"[{variant}] saved", flush=True)
            model = ToyJointV3(base=args.base)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            model = model.to(dev).eval()
            res = eval_acwm(model, args.data, dev, n=args.d_eval_n)
            results[variant] = res
            print(f"[eval {args.d_tag}/{variant}] {json.dumps(res)}", flush=True)
            del model
            torch.cuda.empty_cache()
        with open(os.path.join(args.out, f"eval_{args.d_tag}.json"), "w") as fo:
            json.dump(results, fo, indent=2)


if __name__ == "__main__":
    main()
