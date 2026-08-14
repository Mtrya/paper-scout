"""Toy-scale RIFT/ForeWAM protocol, v3: conv video branch + attention cache interface.

v1/v2 used a full attention DiT for video generation; at toy scale its video
flow loss plateaued (~0.64) and closed-loop success stayed near zero. The
morning's GeniWorld probe showed the same-scale 3D-UNet reaches ~0.08 video
MSE, so v3 keeps the protocol (record-edit-replay interventions, paired
EE-ADE, one-pass producers, plan injection) but:

- video branch: 3D-UNet flow matching on [obs(clean) | 8 noised future frames]
- the "future cache" = bottleneck feature map K/V that action tokens attend
  through cross-attention at 4 transformer layers (per denoising step)
- interventions edit the bottleneck cache (mask / spatial shuffle / temporal
  swap / noise / frozen-present / final-clean replay)
- producers: anticipation tokens (learned future-frame input, bottleneck
  features supervised by L2 or conditional FM) and noise-slot single prefill
  at sigma=1 (ForeWAM-style)

Variants: joint / currentonly / rift-l2 / rift-fm / noiseslots
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


# Input video is [obs | 8 future] = 9 frames; two stride-2 downsamples give
# bottleneck time dim Tb = ceil(ceil(9/2)/2) = 3 (frames 0,1,2 where frame 0
# still carries the obs side). edit_cache and plan injection rely on this.
BOTTLENECK_T = 3


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
    """Self-attention over action tokens + cross-attention to video bottleneck."""

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
        h = h + self.sa_proj(self._attn(self.sa_qkv(self.ln1(h))[..., :h.shape[-1]],
                                        self.sa_qkv(self.ln1(h))[..., h.shape[-1]:2 * h.shape[-1]],
                                        self.sa_qkv(self.ln1(h))[..., 2 * h.shape[-1]:]))
        h = h + self.ca_proj(self._attn(self.ca_q(self.ln2(h)),
                                        self.ca_k(cache_k), self.ca_v(cache_v),
                                        drop=drop_fut))
        h = h + self.mlp(self.ln3(h))
        return h


class ToyJointV3(nn.Module):
    def __init__(self, act_len=16, act_dim=3, base=64, tch=256, a_d=256,
                 a_depth=4, n_modes=2):
        super().__init__()
        self.act_len, self.act_dim, self.base = act_len, act_dim, base
        self.temb = nn.Sequential(nn.Linear(base, tch), nn.SiLU(), nn.Linear(tch, tch))
        self.mode_emb = nn.Embedding(n_modes, a_d)
        self.mode_emb_v = nn.Embedding(n_modes, base * 4)  # video-side mode ctx, dim = bottleneck ch
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
        # action branch
        self.act_in = nn.Linear(act_dim, a_d)
        self.act_blocks = nn.ModuleList([ActionBlock(a_d) for _ in range(a_depth)])
        self.action_head = nn.Linear(a_d, act_dim)

    def _te(self, sigma):
        return self.temb(timestep_embedding(sigma, self.base))

    def _video_forward(self, x, sigma, mode, cache_out=True):
        """x: (B,3,9,H,W) [obs clean | 8 noised future] in [-1,1].
        Returns (vel (B,3,9,H,W), cache_k, cache_v, bottleneck_map)."""
        B = x.shape[0]
        temb = self._te(sigma)
        mode_ctx = self.mode_emb_v(mode)[:, None, :, None, None]  # (B,1,Cb,1,1)
        h0 = self.in_conv(x)
        h1 = self.down1[0](h0, temb)
        h1 = self.down1[1](h1)
        h2 = self.down2[0](h1, temb)
        h2 = self.down2[1](h2)
        hb = self.bottleneck(h2, temb)  # (B,4C,2,8,8) for 64x64... with T=9: (B,4C,5,8,8)? keep generic
        # cache from bottleneck: flatten spatial-time positions
        Bc, Cb, Tb, Hb, Wb = hb.shape
        bm = hb.permute(0, 2, 3, 4, 1).reshape(B, Tb * Hb * Wb, Cb)
        bm = bm + mode_ctx.reshape(B, 1, -1).repeat(1, Tb * Hb * Wb, 1)  # mode conditioning
        cache_k = self.cache_proj_k(bm)
        cache_v = self.cache_proj_v(bm)
        u1 = self.up1[0](hb, temb)
        u1 = self.up1[1](u1)
        u1 = u1[:, :, : h1.shape[2]] + h1  # T=9 odd: transpose conv gives 6, clip to 5
        u2 = self.up2[0](u1, temb)
        u2 = self.up2[1](u2)
        u2 = u2[:, :, : h0.shape[2]] + h0  # gives 10, clip to 9
        h = self.out_block(u2, temb)
        vel = self.out_conv(h)
        return vel, (cache_k, cache_v), bm

    def joint_forward(self, x, act_x, sigma, mode, record=False):
        vel_v, cache, _ = self._video_forward(x, sigma, mode, cache_out=True)
        h = self.act_in(act_x) + self.mode_emb(mode)[:, None]
        for blk in self.act_blocks:
            h = blk(h, cache[0], cache[1])
        return vel_v, self.action_head(h), cache

    def action_forward(self, cache, act_x, mode, drop_fut=False):
        h = self.act_in(act_x) + self.mode_emb(mode)[:, None]
        for blk in self.act_blocks:
            h = blk(h, cache[0], cache[1], drop_fut=drop_fut)
        return self.action_head(h)

    def anticipation_prefill(self, obs_frames, E, mode, record=True):
        """obs_frames: (B,1,3,H,W); E: learned (B,3,8,H,W) future-frame input.
        Returns (bm, cache)."""
        x = torch.cat([obs_frames.permute(0, 2, 1, 3, 4), E], dim=2)  # (B,3,9,H,W)
        _, cache, bm = self._video_forward(x, torch.zeros(1, device=x.device), mode)
        return bm, cache


# ================================================================ simulator
class ArmSim:
    def __init__(self, bg, start, target, H=64, W=64):
        self.bg, self.H, self.W = bg, H, W
        self.base = np.array([W * 0.5, H - 6])
        self.L1, self.L2 = 16.0, 13.0
        self.t1, self.t2, self.grip = 0.0, -0.5, 0.0
        self.cube = np.array(start, dtype=float)
        self.target = np.array(target, dtype=float)
        self.attached = False

    def ee(self):
        j1 = self.base + self.L1 * np.array([np.sin(self.t1), -np.cos(self.t1)])
        return j1 + self.L2 * np.array([np.sin(self.t1 + self.t2),
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
               tuple(self.base + self.L1 * np.array([np.sin(self.t1), -np.cos(self.t1)])),
               tuple(self.ee())]
        d.line([pts[0], pts[1]], fill=(200, 200, 200), width=3)
        d.line([pts[1], pts[2]], fill=(200, 200, 200), width=3)
        return np.asarray(pil)

    def success(self, tol=8.0):
        return bool(np.abs(self.cube - self.target).max() < tol)


# ================================================================ data
class JointDataset(torch.utils.data.Dataset):
    def __init__(self, root, n=None):
        self.files = sorted(os.listdir(root))[:n]
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(os.path.join(self.root, self.files[i]))
        frames = torch.from_numpy(z["frames"]).float() / 255.0  # (T,H,W,3)
        frames = frames.permute(0, 3, 1, 2)  # (T,3,H,W)
        action = torch.from_numpy(z["action"]).float()
        mode = int(z["mode"])
        rng = np.random.default_rng(i)
        f = int(rng.integers(0, 8))
        obs = frames[f:f + 1].clone()
        fut = frames[f + 1:f + 9].clone()
        act = action[2 * f:2 * f + 16].clone()
        return obs, fut, act, mode


def to_model_space(t):
    return t * 2 - 1  # [0,1] -> [-1,1]


# ================================================================ training
def train_variant(model, dl, dev, steps, lr, variant, E=None, out_every=200,
                  sup_proj=None):
    params = list(model.parameters()) + ([E] if E is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    step, t0 = 0, time.time()
    curve = []
    while step < steps:
        for obs, fut, action, mode in dl:
            if step >= steps:
                break
            model.train()
            obs = to_model_space(obs).to(dev)
            fut = to_model_space(fut).to(dev)
            action = action.to(dev); mode = mode.to(dev)
            B = obs.shape[0]
            x0 = torch.cat([obs, fut], dim=1).permute(0, 2, 1, 3, 4)  # (B,3,9,H,W)
            fut_p = fut.permute(0, 2, 1, 3, 4)  # (B,3,8,H,W)
            u = torch.rand(B, device=dev)
            sig = sigma_schedule(u)
            eps_v = torch.randn_like(x0[:, :, 1:])  # (B,3,8,H,W)
            eps_a = torch.randn(action.shape, device=dev)
            xv = torch.cat([obs, (1 - sig)[:, None, None, None, None] * fut
                            + sig[:, None, None, None, None]
                            * eps_v.permute(0, 2, 1, 3, 4)],
                           dim=1).permute(0, 2, 1, 3, 4)  # (B,3,9,H,W)
            xa = (1 - sig)[:, None, None] * action + sig[:, None, None] * eps_a

            if variant == "joint":
                vel_v, vel_a, _ = model.joint_forward(xv, xa, sig, mode)
                loss = F.mse_loss(vel_v[:, :, 1:], eps_v - fut_p) + F.mse_loss(vel_a, eps_a - action)
            elif variant == "currentonly":
                _, cache, _ = model._video_forward(xv, sig, mode)
                vel_a = model.action_forward(cache, xa, mode, drop_fut=True)
                loss = F.mse_loss(vel_a, eps_a - action)
            elif variant in ("rift-l2", "rift-fm"):
                vel_v, _, _ = model._video_forward(xv, sig, mode)
                E_b = E.expand(B, -1, -1, -1, -1)
                bm, cache = model.anticipation_prefill(obs, E_b, mode)
                vel_a = model.action_forward(cache, xa, mode)
                loss = F.mse_loss(vel_v[:, :, 1:], eps_v - fut_p) + F.mse_loss(vel_a, eps_a - action)
                # teacher bottleneck features from clean future
                with torch.no_grad():
                    _, _, bm_t = model._video_forward(x0, torch.zeros(B, device=dev), mode)
                if variant == "rift-l2":
                    loss = loss + F.mse_loss(sup_proj(bm), bm_t.detach())
                else:
                    eps_f = torch.randn_like(bm_t)
                    u2 = torch.rand(B, device=dev)
                    sig2 = sigma_schedule(u2)
                    X = (1 - sig2)[:, None, None] * bm_t + sig2[:, None, None] * eps_f
                    psi_in = torch.cat([X, sup_proj(bm), model._te(sig2)[:, None].expand(-1, bm.shape[1], -1)], dim=-1)
                    loss = loss + F.mse_loss(model.fm_head(psi_in), eps_f - bm_t)
            elif variant == "noiseslots":
                vel_v, _, _ = model._video_forward(xv, sig, mode)
                noise_fut = torch.randn(B, 3, 8, obs.shape[-2], obs.shape[-1], device=dev)
                x_pre = torch.cat([obs.permute(0, 2, 1, 3, 4), noise_fut], dim=2)
                _, cache, _ = model._video_forward(x_pre, torch.ones(B, device=dev), mode)
                vel_a = model.action_forward(cache, xa, mode)
                loss = F.mse_loss(vel_v[:, :, 1:], eps_v - fut_p) + F.mse_loss(vel_a, eps_a - action)
            else:
                raise ValueError(variant)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % out_every == 0:
                curve.append([step, float(loss)])
                print(f"[{variant}] step {step}/{steps} loss={loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                t0 = time.time()
    return curve


# ================================================================ rollout & eval
@torch.no_grad()
def rollout_joint(model, sim, mode, steps=10, seed=0, max_chunks=8, exec_k=4,
                  device="cuda", record=False, time_it=False):
    torch.manual_seed(seed)
    ee_traj, records, times = [], [], []
    obs = sim.render()
    for chunk in range(max_chunks):
        t_start = time.time()
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).float().div(255).to(device)
        obs_t = to_model_space(obs_t)[None].permute(0, 3, 1, 2).unsqueeze(2)  # (1,3,1,H,W)
        H, W = obs.shape[:2]
        xv = torch.randn(1, 3, 8, H, W, device=device)
        xa = torch.randn(1, model.act_len, model.act_dim, device=device)
        eps_a0 = xa.clone()
        chunk_caches = []
        dt = 1.0 / steps
        for k in range(steps):
            sig = torch.full((1,), 1.0 - k * dt, device=device)
            x = torch.cat([obs_t, xv], dim=2)
            vel_v, vel_a, cache = model.joint_forward(
                x, xa, sig, torch.tensor([mode], device=device), record=record)
            xv = xv - vel_v[:, :, 1:] * dt
            xa = xa - vel_a * dt
            if record:
                chunk_caches.append(cache)
        act = xa[0].cpu().numpy()
        for s in range(exec_k):
            obs = sim.step(act[s])
            ee_traj.append(sim.ee())
        if time_it:
            if device.type != "cpu":
                torch.cuda.synchronize()
            times.append(time.time() - t_start)
        records.append(dict(obs_t=obs_t, eps_a0=eps_a0, caches=chunk_caches))
        if sim.success():
            break
    return sim.success(), np.array(ee_traj), records, times


@torch.no_grad()
def replay_action(model, sim, mode, records, intervention, steps=10, exec_k=4,
                  device="cuda", plan_cache=None, time_it=False):
    ee_traj, times = [], []
    obs = sim.render()
    for chunk, rec in enumerate(records):
        t_start = time.time()
        xa = rec["eps_a0"].clone()
        dt = 1.0 / steps
        for k in range(steps):
            sig = torch.full((1,), 1.0 - k * dt, device=device)
            caches = rec["caches"][k]
            if plan_cache is not None and chunk == 0:
                # Inject B's future-side bottleneck positions while keeping A's
                # obs-side (bottleneck frame 0) intact: plan injection semantics.
                pb_k, pb_v = plan_cache[k]
                n0 = caches[0].shape[1] // BOTTLENECK_T
                ck = torch.cat([caches[0][:, :n0], pb_k[:, n0:]], dim=1)
                cv = torch.cat([caches[1][:, :n0], pb_v[:, n0:]], dim=1)
                caches = (ck, cv)
            if intervention == "finalclean":
                caches = rec["caches"][-1]
            ck, cv = edit_cache(caches, intervention)
            vel_a = model.action_forward((ck, cv), xa, torch.tensor([mode], device=device),
                                         drop_fut=(intervention == "mask"))
            xa = xa - vel_a * dt
        act = xa[0].cpu().numpy()
        for s in range(exec_k):
            obs = sim.step(act[s])
            ee_traj.append(sim.ee())
        if time_it:
            if device.type != "cpu":
                torch.cuda.synchronize()
            times.append(time.time() - t_start)
        if sim.success():
            break
    return sim.success(), np.array(ee_traj), times


def edit_cache(caches, intervention):
    ck, cv = caches
    if intervention in ("none", "inject", "finalclean"):
        return ck, cv
    B, S, D = ck.shape
    n_frames, n_pos = BOTTLENECK_T, S // BOTTLENECK_T
    assert n_pos * n_frames == S, "cache length incompatible with bottleneck T"
    rng = np.random.default_rng(0)
    if intervention == "noise":
        cv = torch.randn_like(cv) * cv.std()
    elif intervention == "frozenpresent":
        cv = cv.reshape(B, n_frames, n_pos, D)
        cv = cv[:, :1].repeat(1, n_frames, 1, 1).reshape(B, S, D)  # obs frame tiles over future
    elif intervention == "shuffle":
        cv = cv.reshape(B, n_frames, n_pos, D)
        idx = rng.permutation(n_pos)
        cv = cv[:, :, idx].reshape(B, S, D)
    elif intervention == "swap":
        cv = cv.reshape(B, n_frames, n_pos, D)
        idx = rng.permutation(n_frames)
        cv = cv[:, idx].reshape(B, S, D)
    return ck, cv


@torch.no_grad()
def rollout_producer(model, sim, mode, variant, E, steps=10, seed=0,
                     max_chunks=8, exec_k=4, device="cuda", time_it=False):
    torch.manual_seed(seed)
    ee_traj, times = [], []
    obs = sim.render()
    for chunk in range(max_chunks):
        t_start = time.time()
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).float().div(255).to(device)
        obs_t = to_model_space(obs_t)[None].permute(0, 3, 1, 2).unsqueeze(2)
        H, W = obs.shape[:2]
        _ = torch.randn(1, 3, 8, H, W, device=device)  # pair RNG
        if variant == "noiseslots":
            noise_fut = torch.randn(1, 3, 8, H, W, device=device)
            x_pre = torch.cat([obs_t, noise_fut], dim=2)
            _, cache, _ = model._video_forward(x_pre, torch.ones(1, device=device), torch.tensor([mode], device=device))
        elif variant == "currentonly":
            x_pre = torch.cat([obs_t, torch.randn(1, 3, 8, H, W, device=device)], dim=2)
            _, cache, _ = model._video_forward(x_pre, torch.zeros(1, device=device), torch.tensor([mode], device=device))
        else:
            _, cache = model.anticipation_prefill(obs_t.permute(0, 2, 1, 3, 4), E,
                                                  torch.tensor([mode], device=device))
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
            ee_traj.append(sim.ee())
        if time_it:
            if device.type != "cpu":
                torch.cuda.synchronize()
            times.append(time.time() - t_start)
        if sim.success():
            break
    return sim.success(), np.array(ee_traj), times


def ee_ade(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n = min(len(a), len(b))
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


def expert_ee(joints, H=64, W=64):
    base = np.array([W * 0.5, H - 6])
    L1, L2 = 16.0, 13.0
    steps = []
    for s in range(32):
        f = s / 2.0
        f0, u = int(np.floor(f)), f - int(np.floor(f))
        f1 = min(f0 + 1, len(joints) - 1)
        j = joints[f0] * (1 - u) + joints[f1] * u
        j1 = base + L1 * np.array([np.sin(j[0]), -np.cos(j[0])])
        ee = j1 + L2 * np.array([np.sin(j[0] + j[1]), -np.cos(j[0] + j[1])])
        steps.append(ee)
    return np.array(steps)


def make_sim(npz):
    z = np.load(npz)
    return ArmSim(z["bg"], z["start"], z["target"]), z


def load_variant(out_dir, tag, variant, base=64):
    model = ToyJointV3(base=base)
    if variant in ("rift-l2", "rift-fm"):
        model.sup_proj = nn.Linear(base * 4, base * 4)
        model.fm_head = nn.Sequential(nn.Linear(base * 8 + 256, 512), nn.SiLU(),
                                      nn.Linear(512, base * 4))
    ckpt = torch.load(os.path.join(out_dir, f"{variant}_{tag}.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    E = ckpt["E"]
    return model, E


def run_eval(args):
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_files = sorted(os.listdir(os.path.join(args.data, "val")))[:args.eval_n]
    out = {}
    variants = ["joint"] + [v for v in args.variants if v != "joint"]
    joint_model, _ = load_variant(args.out, args.tag, "joint", base=args.base)
    joint_model = joint_model.to(dev).eval()
    for variant in variants:
        if variant == "joint":
            model = joint_model
        else:
            model, E = load_variant(args.out, args.tag, variant, base=args.base)
            model = model.to(dev).eval()
            if E is not None:
                E = E.to(dev)  # E is stored detached on cpu; move with the model
        print(f"=== eval {variant}", flush=True)
        if variant == "joint":
            row = {iv: dict(sr=[], ade=[]) for iv in
                   ["original", "mask", "noise", "frozenpresent", "shuffle", "swap", "finalclean"]}
            lat = dict(original=[])
            for f in val_files:
                sim, z = make_sim(os.path.join(args.data, "val", f))
                mode = int(z["mode"])
                ok, traj, recs, times = rollout_joint(model, sim, mode, seed=0,
                                                      device=dev, record=True, time_it=True)
                row["original"]["sr"].append(ok)
                row["original"]["ade"].append(0.0)
                lat["original"] += times
                for iv in ["mask", "noise", "frozenpresent", "shuffle", "swap", "finalclean"]:
                    sim2, _ = make_sim(os.path.join(args.data, "val", f))
                    ok2, traj2, _ = replay_action(model, sim2, mode, recs, iv, device=dev)
                    row[iv]["sr"].append(ok2)
                    row[iv]["ade"].append(ee_ade(traj2, traj))
            summ = {}
            for iv, d in row.items():
                summ[iv] = dict(sr=float(np.mean(d["sr"])),
                                ade=float(np.nanmean(d["ade"])), n=len(d["sr"]))
            summ["latency_ms_per_chunk"] = float(np.mean(lat["original"]) * 1000)
            out["joint"] = summ
            print(json.dumps(summ, indent=2), flush=True)
            out["plan_probe"] = run_plan_probe(model, dev, args)
        else:
            srs, ades, lats = [], [], []
            for f in val_files:
                sim, z = make_sim(os.path.join(args.data, "val", f))
                mode = int(z["mode"])
                sim_ref, _ = make_sim(os.path.join(args.data, "val", f))
                _, traj_ref, _, _ = rollout_joint(joint_model, sim_ref, mode, seed=0, device=dev)
                sim2, _ = make_sim(os.path.join(args.data, "val", f))
                ok, traj, times = rollout_producer(model, sim2, mode, variant, E,
                                                   seed=0, device=dev, time_it=True)
                srs.append(ok)
                ades.append(ee_ade(traj, traj_ref))
                lats += times
            summ = dict(sr=float(np.mean(srs)),
                        ade_vs_rollout=float(np.nanmean(ades)), n=len(srs),
                        latency_ms_per_chunk=float(np.mean(lats) * 1000))
            out[variant] = summ
            print(json.dumps(summ, indent=2), flush=True)
        if variant != "joint":
            del model
            torch.cuda.empty_cache()
    with open(os.path.join(args.out, f"eval_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("eval done ->", os.path.join(args.out, f"eval_{args.tag}.json"))


@torch.no_grad()
def run_plan_probe(model, dev, args, n=60):
    pairs = sorted(os.listdir(os.path.join(args.data, "planprobe")))
    pairs = [p for p in pairs if p.endswith("_A.npz")][:n]
    dA4, dB4, dA_ctrl, n_ok, n_followB = [], [], [], 0, 0
    for pa in pairs:
        base = pa[:-len("_A.npz")]
        pB = os.path.join(args.data, "planprobe", base + "_B.npz")
        simA, zA = make_sim(os.path.join(args.data, "planprobe", pa))
        _, trajA, recsA, _ = rollout_joint(model, simA, 0, seed=0, device=dev, record=True)
        simB, zB = make_sim(pB)
        _, _, recsB, _ = rollout_joint(model, simB, 1, seed=0, device=dev, record=True)
        simI, _ = make_sim(os.path.join(args.data, "planprobe", pa))
        okI, trajI, _ = replay_action(model, simI, 0, recsA, "inject", device=dev,
                                      plan_cache=recsB[0]["caches"])
        simC, _ = make_sim(os.path.join(args.data, "planprobe", pa))
        _, trajC, _ = replay_action(model, simC, 0, recsA, "none", device=dev)
        eA = expert_ee(zA["joints"])
        eB = expert_ee(zB["joints"])
        k = min(4, len(trajI) - 1, len(trajA) - 1)
        dA4.append(np.linalg.norm(trajI[k] - eA[k]))
        dB4.append(np.linalg.norm(trajI[k] - eB[k]))
        dA_ctrl.append(ee_ade(trajC, trajA))
        n_ok += okI
        if len(trajI) > 4:
            e_t = min(len(trajI), len(eA), len(eB))
            adeA = np.mean(np.linalg.norm(trajI[:e_t] - eA[:e_t], axis=1))
            adeB = np.mean(np.linalg.norm(trajI[:e_t] - eB[:e_t], axis=1))
            n_followB += adeB < adeA
    return dict(step4_dist_to_expertA=float(np.mean(dA4)),
                step4_dist_to_expertB=float(np.mean(dB4)),
                control_replay_ade=float(np.mean(dA_ctrl)),
                injected_success_rate=float(n_ok / len(pairs)),
                frac_following_injected_plan=float(n_followB / len(pairs)),
                n=len(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="joint_data")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-n", type=int, default=80)
    ap.add_argument("--variants", nargs="+",
                    default=["joint", "currentonly", "rift-l2", "rift-fm", "noiseslots"])
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    if not args.eval_only:
        ds_tr = JointDataset(os.path.join(args.data, "train"),
                             n=40 if args.smoke else None)
        batch = 8 if args.smoke else args.batch  # smoke has 24 samples; batch32 would drop to 0
        dl = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True,
                                         num_workers=2, drop_last=True)
        for variant in args.variants:
            torch.manual_seed(0)
            model = ToyJointV3(base=args.base).to(dev)
            E = None
            if variant in ("rift-l2", "rift-fm"):
                E = nn.Parameter(torch.randn(1, 3, 8, 64, 64, device=dev) * 0.02)
                model.sup_proj = nn.Linear(args.base * 4, args.base * 4).to(dev)
                model.fm_head = nn.Sequential(
                    nn.Linear(args.base * 8 + 256, 512), nn.SiLU(),
                    nn.Linear(512, args.base * 4)).to(dev)
            steps = 30 if args.smoke else args.steps
            curve = train_variant(model, dl, dev, steps, args.lr, variant, E,
                                  out_every=200, sup_proj=getattr(model, "sup_proj", None))
            torch.save({"model": model.state_dict(),
                        "E": E.detach().cpu() if E is not None else None},
                       os.path.join(args.out, f"{variant}_{args.tag}.pt"))
            print(f"[{variant}] saved", flush=True)
        print("training done")
    run_eval(args)


if __name__ == "__main__":
    main()
