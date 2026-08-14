"""Toy-scale reconstruction of the RIFT / ForeWAM "drop the rollout" mechanism.

Joint video+action flow-matching transformer on synthetic 2-link arm transport
clips (gen_joint_data.py). Action tokens read future video tokens through
per-layer K/V caches; the video->action attention mask keeps the cache
action-independent (RIFT's protocol).

Variants (shared transformer weights per variant):
  joint         rollout baseline: video+action co-denoise, action reads the
                EVOLVING future cache (Fast-WAM-Joint analog)
  currentonly   action-only model, no future read (Fast-WAM current-only analog)
  rift-l2       anticipation tokens E at future positions, single prefill,
                L2 supervision of E states -> clean future patch tokens
  rift-fm       same, conditional-flow-matching supervision (RIFT full recipe)
  noiseslots    ForeWAM-style: pure-noise future slots, single prefill at
                sigma=1, action loss backprops through the cache

Interventions on `joint` (paired with Original, same seeds/noise):
  mask / noise / frozen-present / spatial-shuffle / temporal-swap /
  final-clean-replay  -> EE-ADE vs Original + success rate

Plan probe: condition on mode A, inject a recorded mode-B future cache at the
first chunk -> does the action follow the instruction or the injected plan?

Usage: python joint_wam_probe.py --data joint_data --steps 6000 --tag v1
       python joint_wam_probe.py --data joint_data --eval-only --tag v1
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


# ================================================================ model
def sigma_schedule(u):
    """RIFT-style shifted schedule: u ~ U[0,1) -> sigma."""
    return 5 * u / (1 + 4 * u)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32,
                                                      device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    def __init__(self, d=256, heads=8, mlp=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.heads = heads

    def attn(self, q, kv_sources):
        """q: (B,Q,d); kv_sources: list of (K,V) (B,S_i,d). Returns (B,Q,d)."""
        heads, d = self.heads, q.shape[-1]
        B, Q, _ = q.shape
        qh = q.reshape(B, Q, heads, d // heads).permute(0, 2, 1, 3)
        ks, vs = [], []
        for K, V in kv_sources:
            ks.append(K.reshape(B, -1, heads, d // heads).permute(0, 2, 1, 3))
            vs.append(V.reshape(B, -1, heads, d // heads).permute(0, 2, 1, 3))
        K = torch.cat(ks, dim=2)
        V = torch.cat(vs, dim=2)
        att = torch.softmax(torch.einsum("bhqd,bhsd->bhqs", qh, K)
                            * (d // heads) ** -0.5, dim=-1)
        return torch.einsum("bhqs,bhsd->bhqd", att, V).reshape(B, Q, d)


class ToyJointWAM(nn.Module):
    """obs: 17 tokens (16 patches + 1 mode token); fut: n_fut=128 patch tokens (8 frames);
    act: 32 chunk tokens. obs attends obs; fut attends obs+fut; act attends all.
    """

    def __init__(self, n_obs=17, n_fut=128, act_len=16, d=256, heads=8, depth=6,
                 patch=16, in_ch=3, act_dim=3, n_modes=2, mlp=1024):
        super().__init__()
        self.n_obs, self.n_fut, self.act_len = n_obs, n_fut, act_len
        self.d, self.depth, self.act_dim, self.patch = d, depth, act_dim, patch
        p = patch
        self.patch_embed = nn.Linear(p * p * in_ch, d)
        self.mode_emb = nn.Embedding(n_modes, d)
        self.pos = nn.Parameter(torch.randn(n_obs + n_fut + act_len, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(depth)])
        self.act_in = nn.Linear(act_dim, d)
        self.video_head = nn.Linear(d, p * p * in_ch)
        self.action_head = nn.Linear(d, act_dim)

    def _te(self, sigma):
        return self.temb(timestep_embedding(sigma, self.d))[:, None]  # (B,1,d)

    def _obs_tokens(self, obs_frames, mode):
        """obs_frames (B,1,H,W,3) in [0,1] -> (B,17,d)."""
        B = obs_frames.shape[0]
        flat = obs_frames[:, 0].reshape(B, 16, self.patch * self.patch * 3)
        obs_tok = self.patch_embed(flat)
        return torch.cat([obs_tok, self.mode_emb(mode)[:, None]], dim=1)

    def _fut_tokens(self, fut):
        """fut: raw (B,T,H,W,3) or patch values (B,n_fut,p^2*3). Returns
        patch-embedded tokens (B, n_fut, d)."""
        if fut.dim() == 5:
            B, T, H, W, _ = fut.shape
            fut = fut.reshape(B, T * 16, self.patch * self.patch * 3)
        return self.patch_embed(fut)  # (B, n_fut, d)

    def _fut_pix(self, fut):
        """fut: raw (B,T,H,W,3) -> pixel patch values (B, n_fut, p^2*3)."""
        B, T, H, W, _ = fut.shape
        return fut.reshape(B, T * 16, self.patch * self.patch * 3)

    # ---------------- forward paths
    def joint_forward(self, obs_frames, fut_frames, act_x, sigma, mode, record=False):
        """Video+action co-denoise (joint rollout). fut_frames: raw noised."""
        B = obs_frames.shape[0]
        h_obs = self._obs_tokens(obs_frames, mode) + self.pos[:self.n_obs][None] + self._te(sigma)
        h_fut = self._fut_tokens(fut_frames) + self.pos[self.n_obs:self.n_obs + self.n_fut][None] + self._te(sigma)
        h_act = self.act_in(act_x) + self.pos[self.n_obs + self.n_fut:][None] + self._te(sigma)
        cache = []
        for blk in self.blocks:
            ho, hf, ha = blk.ln1(h_obs), blk.ln1(h_fut), blk.ln1(h_act)
            qkv_o, qkv_f, qkv_a = blk.qkv(ho), blk.qkv(hf), blk.qkv(ha)
            K_o, V_o = qkv_o[..., self.d:2 * self.d], qkv_o[..., 2 * self.d:]
            K_f, V_f = qkv_f[..., self.d:2 * self.d], qkv_f[..., 2 * self.d:]
            K_a, V_a = qkv_a[..., self.d:2 * self.d], qkv_a[..., 2 * self.d:]
            if record:
                cache.append((K_o, V_o, K_f, V_f))
            h_obs = h_obs + blk.proj(blk.attn(qkv_o[..., :self.d], [(K_o, V_o)]))
            h_obs = h_obs + blk.mlp(blk.ln2(h_obs))
            h_fut = h_fut + blk.proj(blk.attn(qkv_f[..., :self.d],
                                              [(K_o, V_o), (K_f, V_f)]))
            h_fut = h_fut + blk.mlp(blk.ln2(h_fut))
            h_act = h_act + blk.proj(blk.attn(qkv_a[..., :self.d],
                                              [(K_o, V_o), (K_f, V_f), (K_a, V_a)]))
            h_act = h_act + blk.mlp(blk.ln2(h_act))
        return self.video_head(h_fut), self.action_head(h_act), cache

    def video_prefill(self, obs_frames, fut_vals, sigma, mode, fut_is_raw, record=False):
        """Single video-branch prefill. fut_vals: raw noised frames (B,T,H,W,3)
        if fut_is_raw else patch values (B,n_fut,p^2*3). Returns (vel_fut, cache)."""
        B = obs_frames.shape[0]
        h_obs = self._obs_tokens(obs_frames, mode) + self.pos[:self.n_obs][None] + self._te(sigma)
        h_fut = self._fut_tokens(fut_vals) + self.pos[self.n_obs:self.n_obs + self.n_fut][None] + self._te(sigma)
        cache = []
        for blk in self.blocks:
            ho, hf = blk.ln1(h_obs), blk.ln1(h_fut)
            qkv_o, qkv_f = blk.qkv(ho), blk.qkv(hf)
            K_o, V_o = qkv_o[..., self.d:2 * self.d], qkv_o[..., 2 * self.d:]
            K_f, V_f = qkv_f[..., self.d:2 * self.d], qkv_f[..., 2 * self.d:]
            if record:
                cache.append((K_o, V_o, K_f, V_f))
            h_obs = h_obs + blk.proj(blk.attn(qkv_o[..., :self.d], [(K_o, V_o)]))
            h_obs = h_obs + blk.mlp(blk.ln2(h_obs))
            h_fut = h_fut + blk.proj(blk.attn(qkv_f[..., :self.d],
                                              [(K_o, V_o), (K_f, V_f)]))
            h_fut = h_fut + blk.mlp(blk.ln2(h_fut))
        return self.video_head(h_fut), cache

    def anticipation_prefill(self, obs_frames, E, mode, record=False):
        """RIFT path: learned anticipation tokens E (B,n_fut,d) at future
        positions, no noise (sigma=0). Returns (states, cache)."""
        B = obs_frames.shape[0]
        h_obs = self._obs_tokens(obs_frames, mode) + self.pos[:self.n_obs][None]
        h_fut = E + self.pos[self.n_obs:self.n_obs + self.n_fut][None]
        cache = []
        for blk in self.blocks:
            ho, hf = blk.ln1(h_obs), blk.ln1(h_fut)
            qkv_o, qkv_f = blk.qkv(ho), blk.qkv(hf)
            K_o, V_o = qkv_o[..., self.d:2 * self.d], qkv_o[..., 2 * self.d:]
            K_f, V_f = qkv_f[..., self.d:2 * self.d], qkv_f[..., 2 * self.d:]
            if record:
                cache.append((K_o, V_o, K_f, V_f))
            h_obs = h_obs + blk.proj(blk.attn(qkv_o[..., :self.d], [(K_o, V_o)]))
            h_obs = h_obs + blk.mlp(blk.ln2(h_obs))
            h_fut = h_fut + blk.proj(blk.attn(qkv_f[..., :self.d],
                                              [(K_o, V_o), (K_f, V_f)]))
            h_fut = h_fut + blk.mlp(blk.ln2(h_fut))
        return h_fut, cache

    def action_forward(self, cache, act_x, sigma, drop_fut=False):
        """Action denoising reading cached obs/fut K/V. drop_fut = true mask
        (fut source removed from attention entirely)."""
        B = act_x.shape[0]
        h_act = self.act_in(act_x) + self.pos[self.n_obs + self.n_fut:][None] + self._te(sigma)
        for li, blk in enumerate(self.blocks):
            ha = blk.ln1(h_act)
            qkv_a = blk.qkv(ha)
            Q_a, K_a, V_a = qkv_a[..., :self.d], qkv_a[..., self.d:2 * self.d], qkv_a[..., 2 * self.d:]
            K_o, V_o, K_f, V_f = cache[li]
            srcs = [(K_o, V_o), (K_a, V_a)]
            if not drop_fut:
                srcs.insert(1, (K_f, V_f))
            h_act = h_act + blk.proj(blk.attn(Q_a, srcs))
            h_act = h_act + blk.mlp(blk.ln2(h_act))
        return self.action_head(h_act)


# ================================================================ simulator
class ArmSim:
    """Scripted closed-loop sim: apply (t1,t2,grip); cube sticks when grasped."""

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
    """Window sampling: obs at frame f, 8 future frames, 16-step action chunk.
    Closed-loop execution needs obs from arbitrary mid-episode frames, so the
    dataset draws windows instead of full episodes."""

    def __init__(self, root, n=None):
        self.files = sorted(os.listdir(root))[:n]
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(os.path.join(self.root, self.files[i]))
        frames = torch.from_numpy(z["frames"]).float() / 255.0
        action = torch.from_numpy(z["action"]).float()
        mode = int(z["mode"])
        rng = np.random.default_rng(i)  # deterministic per index
        f = int(rng.integers(0, 8))  # obs frame 0..7 (future f+1..f+8 stays in range)
        obs = frames[f:f + 1].clone()
        fut = frames[f + 1:f + 9].clone()  # 8 future frames
        act = action[2 * f:2 * f + 16].clone()  # 16-step chunk
        return obs, fut, act, mode


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
            obs = obs.to(dev); fut = fut.to(dev)
            action = action.to(dev); mode = mode.to(dev)
            B = obs.shape[0]
            u = torch.rand(B, device=dev)
            sig = sigma_schedule(u)
            eps_v = torch.randn_like(fut)
            eps_a = torch.randn(action.shape, device=dev)
            xv = (1 - sig)[:, None, None, None, None] * fut + sig[:, None, None, None, None] * eps_v
            xa = (1 - sig)[:, None, None] * action + sig[:, None, None] * eps_a
            # video flow in patch-pixel space (matches video_head output)
            Y_pix = model._fut_pix(fut)
            eps_pix = model._fut_pix(eps_v)
            Y_patch = model._fut_tokens(fut)  # d-space, for anticipation supervision

            if variant == "joint":
                vel_v, vel_a, _ = model.joint_forward(obs, xv, xa, sig, mode)
                loss = F.mse_loss(vel_v, eps_pix - Y_pix) + F.mse_loss(vel_a, eps_a - action)
            elif variant == "currentonly":
                _, cache = model.video_prefill(obs, fut, sig, mode, fut_is_raw=True, record=True)
                vel_a = model.action_forward(cache, xa, sig, drop_fut=True)
                loss = F.mse_loss(vel_a, eps_a - action)
            elif variant in ("rift-l2", "rift-fm"):
                # forward A: native video flow (teacher-forced)
                vel_v, _ = model.video_prefill(obs, xv, sig, mode, fut_is_raw=True, record=False)
                # forward B: deployment-matched action via anticipation cache
                E_b = E.expand(B, -1, -1)
                states, cache = model.anticipation_prefill(obs, E_b, mode, record=True)
                vel_a = model.action_forward(cache, xa, sig)
                loss = F.mse_loss(vel_v, eps_pix - Y_pix) + F.mse_loss(vel_a, eps_a - action)
                if variant == "rift-l2":
                    loss = loss + F.mse_loss(sup_proj(states), Y_patch.detach())
                else:
                    eps_f = torch.randn_like(Y_patch)
                    u2 = torch.rand(B, device=dev)
                    sig2 = sigma_schedule(u2)
                    X = (1 - sig2)[:, None, None] * Y_patch + sig2[:, None, None] * eps_f
                    psi_in = torch.cat([X, sup_proj(states), model._te(sig2).expand(-1, model.n_fut, -1)], dim=-1)
                    loss = loss + F.mse_loss(model.fm_head(psi_in), eps_f - Y_patch)
            elif variant == "noiseslots":
                vel_v, _ = model.video_prefill(obs, xv, sig, mode, fut_is_raw=True, record=False)
                noise_fut = torch.randn(B, model.n_fut, model.patch * model.patch * 3, device=dev)
                _, cache = model.video_prefill(obs, noise_fut, torch.ones(B, device=dev),
                                               mode, fut_is_raw=False, record=True)
                vel_a = model.action_forward(cache, xa, sig)
                loss = F.mse_loss(vel_v, eps_pix - Y_pix) + F.mse_loss(vel_a, eps_a - action)
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
    """Closed-loop episode with the joint rollout policy.
    Returns (success, ee_traj, records, wall_seconds_per_chunk).
    records[chunk] = dict(obs_t, eps_a0, caches: list per denoising step of
    per-layer (K_obs,V_obs,K_fut,V_fut))."""
    torch.manual_seed(seed)
    ee_traj, records, times = [], [], []
    obs = sim.render()
    for chunk in range(max_chunks):
        t_start = time.time()
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).float().div(255).to(device)[None, None]
        H, W = obs.shape[:2]
        xv = torch.randn(1, 8, H, W, 3, device=device)
        xa = torch.randn(1, model.act_len, model.act_dim, device=device)
        eps_a0 = xa.clone()
        chunk_caches = []
        dt = 1.0 / steps
        for k in range(steps):
            sig = torch.full((1,), 1.0 - k * dt, device=device)
            vel_v, vel_a, cache = model.joint_forward(
                obs_t, xv, xa, sig, torch.tensor([mode], device=device), record=record)
            vel_v = vel_v.reshape(1, 8, H, W, 3)
            xv = xv - vel_v * dt
            xa = xa - vel_a * dt
            if record:
                chunk_caches.append(cache)
        act = xa[0].cpu().numpy()
        for s in range(exec_k):
            obs = sim.step(act[s])
            ee_traj.append(sim.ee())
        if time_it:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t_start)
        records.append(dict(obs_t=obs_t, eps_a0=eps_a0, caches=chunk_caches))
        if sim.success():
            break
    return sim.success(), np.array(ee_traj), records, times


@torch.no_grad()
def replay_action(model, sim, records, intervention, steps=10, exec_k=4,
                  device="cuda", plan_cache=None, time_it=False):
    """Replay episode with same seeds/noise, edited future cache.
    interventions: none|mask|noise|frozenpresent|shuffle|swap|finalclean|inject
    plan_cache: per-denoising-step cache list replacing chunk 0's cache."""
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
                # inject only the FUTURE part of B's cache; keep A's obs
                # (instruction) side, so instruction vs injected plan disagree
                caches = [(c_obs[0], c_obs[1], p[2], p[3])
                          for c_obs, p in zip(caches, plan_cache[k])]
            if intervention == "finalclean":
                caches = rec["caches"][-1]
            edited = edit_cache(caches, intervention)
            vel_a = model.action_forward(edited, xa, sig,
                                         drop_fut=(intervention == "mask"))
            xa = xa - vel_a * dt
        act = xa[0].cpu().numpy()
        for s in range(exec_k):
            obs = sim.step(act[s])
            ee_traj.append(sim.ee())
        if time_it:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t_start)
        if sim.success():
            break
    return sim.success(), np.array(ee_traj), times


def edit_cache(caches, intervention):
    """caches: per-layer list of (K_obs,V_obs,K_fut,V_fut). Returns edited list."""
    if intervention in ("none", "inject"):
        return caches
    out = []
    rng = np.random.default_rng(0)  # fixed per call: same edit across layers
    for (K_o, V_o, K_f, V_f) in caches:
        if intervention == "mask":
            out.append((K_o, V_o, K_f, V_f))  # dropped in action_forward
        elif intervention == "noise":
            V_f = torch.randn_like(V_f) * V_f.std()
            out.append((K_o, V_o, K_f, V_f))
        elif intervention == "frozenpresent":
            B, S, D = V_f.shape
            V_f = V_o.mean(dim=1, keepdim=True).expand(B, S, D).contiguous()
            out.append((K_o, V_o, K_f, V_f))
        elif intervention == "shuffle":
            B, S, D = V_f.shape
            n_frames, n_p = 8, S // 8
            V_f = V_f.reshape(B, n_frames, n_p, D)
            idx = rng.permutation(n_p)
            V_f = V_f[:, :, idx].reshape(B, S, D)
            out.append((K_o, V_o, K_f, V_f))
        elif intervention == "swap":
            B, S, D = V_f.shape
            n_frames, n_p = 8, S // 8
            V_f = V_f.reshape(B, n_frames, n_p, D)
            idx = rng.permutation(n_frames)
            V_f = V_f[:, idx].reshape(B, S, D)
            out.append((K_o, V_o, K_f, V_f))
        else:
            out.append((K_o, V_o, K_f, V_f))
    return out


@torch.no_grad()
def rollout_producer(model, sim, mode, variant, E, steps=10, seed=0,
                     max_chunks=8, exec_k=4, device="cuda", time_it=False):
    """One-pass producers: rift-* (anticipation E) / noiseslots (noise slots) /
    currentonly (obs-only cache). RNG consumption order matches rollout_joint
    (video noise drawn and discarded first) so action noise is paired."""
    torch.manual_seed(seed)
    ee_traj, times = [], []
    obs = sim.render()
    for chunk in range(max_chunks):
        t_start = time.time()
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).float().div(255).to(device)[None, None]
        H, W = obs.shape[:2]
        _ = torch.randn(1, 8, H, W, 3, device=device)  # pair RNG with rollout_joint
        if variant == "noiseslots":
            noise_fut = torch.randn(1, model.n_fut, model.patch * model.patch * 3, device=device)
            _, cache = model.video_prefill(obs_t, noise_fut, torch.ones(1, device=device),
                                           torch.tensor([mode], device=device),
                                           fut_is_raw=False, record=True)
        elif variant == "currentonly":
            _, cache = model.video_prefill(obs_t, torch.randn(1, 8, H, W, 3, device=device),
                                           torch.zeros(1, device=device),
                                           torch.tensor([mode], device=device),
                                           fut_is_raw=True, record=True)
        else:
            _, cache = model.anticipation_prefill(obs_t, E, torch.tensor([mode], device=device), record=True)
        xa = torch.randn(1, model.act_len, model.act_dim, device=device)
        dt = 1.0 / steps
        for k in range(steps):
            sig = torch.full((1,), 1.0 - k * dt, device=device)
            vel_a = model.action_forward(cache, xa, sig,
                                         drop_fut=(variant == "currentonly"))
            xa = xa - vel_a * dt
        act = xa[0].cpu().numpy()
        for s in range(exec_k):
            obs = sim.step(act[s])
            ee_traj.append(sim.ee())
        if time_it:
            if torch.cuda.is_available():
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
    """16 keyframe joints -> 32-step EE trajectory (matches the action chunk)."""
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


# ================================================================ eval driver
def make_sim(npz):
    z = np.load(npz)
    return ArmSim(z["bg"], z["start"], z["target"]), z


def load_variant(out_dir, tag, variant, d=256, depth=6):
    model = ToyJointWAM(d=d, depth=depth)
    if variant in ("rift-l2", "rift-fm"):
        model.sup_proj = nn.Linear(d, d)
        model.fm_head = nn.Sequential(nn.Linear(d * 3, 512), nn.SiLU(),
                                      nn.Linear(512, d))
    ckpt = torch.load(os.path.join(out_dir, f"{variant}_{tag}.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    E = ckpt["E"]
    if E is not None:
        E = E.to(next(model.parameters()).device)
    return model, E


def run_eval(args):
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_files = sorted(os.listdir(os.path.join(args.data, "val")))[:args.eval_n]
    out = {}
    variants = ["joint"] + [v for v in args.variants if v != "joint"]

    # joint model serves as the rollout reference for all producer variants
    joint_model, _ = load_variant(args.out, args.tag, "joint",
                                  d=args.dim, depth=args.depth)
    joint_model = joint_model.to(dev).eval()

    for variant in variants:
        if variant == "joint":
            model = joint_model
        else:
            model, E = load_variant(args.out, args.tag, variant,
                                    d=args.dim, depth=args.depth)
            model = model.to(dev).eval()
        print(f"=== eval {variant}", flush=True)

        if variant == "joint":
            # Original + intervention battery (paired)
            row = {iv: dict(sr=[], ade=[]) for iv in
                   ["original", "mask", "noise", "frozenpresent", "shuffle",
                    "swap", "finalclean"]}
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
                    ok2, traj2, _ = replay_action(model, sim2, recs, iv,
                                                  device=dev)
                    row[iv]["sr"].append(ok2)
                    row[iv]["ade"].append(ee_ade(traj2, traj))
            summ = {}
            for iv, d in row.items():
                summ[iv] = dict(sr=float(np.mean(d["sr"])),
                                ade=float(np.nanmean(d["ade"])),
                                n=len(d["sr"]))
            summ["latency_ms_per_chunk"] = float(np.mean(lat["original"]) * 1000)
            out["joint"] = summ
            print(json.dumps(summ, indent=2), flush=True)
            # plan probe
            plan = run_plan_probe(model, dev, args)
            out["plan_probe"] = plan
        else:
            srs, ades, lats = [], [], []
            for f in val_files:
                sim, z = make_sim(os.path.join(args.data, "val", f))
                mode = int(z["mode"])
                # rollout reference from the joint model (paired seed)
                sim_ref, _ = make_sim(os.path.join(args.data, "val", f))
                _, traj_ref, _, _ = rollout_joint(joint_model, sim_ref, mode,
                                                  seed=0, device=dev, record=False)
                sim2, _ = make_sim(os.path.join(args.data, "val", f))
                ok, traj, times = rollout_producer(model, sim2, mode, variant, E,
                                                   seed=0, device=dev, time_it=True)
                srs.append(ok)
                ades.append(ee_ade(traj, traj_ref))
                lats += times
            summ = dict(sr=float(np.mean(srs)),
                        ade_vs_rollout=float(np.nanmean(ades)),
                        n=len(srs),
                        latency_ms_per_chunk=float(np.mean(lats) * 1000))
            out[variant] = summ
            print(json.dumps(summ, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, f"eval_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("eval done ->", os.path.join(args.out, f"eval_{args.tag}.json"))


@torch.no_grad()
def run_plan_probe(model, dev, args, n=60):
    """Mode-A episode with mode-B cache injected at chunk 0.
    Metrics: EE distance at step 4 to expertA vs expertB; full-episode SR;
    plus control injection of same-mode cache (replay fidelity)."""
    pairs = sorted(os.listdir(os.path.join(args.data, "planprobe")))
    pairs = [p for p in pairs if p.endswith("_A.npz")][:n]
    dA4, dB4, dA_ctrl, n_ok, n_followB = [], [], [], 0, 0
    for pa in pairs:
        base = pa[:-len("_A.npz")]
        pB = os.path.join(args.data, "planprobe", base + "_B.npz")
        # A original (mode 0)
        simA, zA = make_sim(os.path.join(args.data, "planprobe", pa))
        okA, trajA, recsA, _ = rollout_joint(model, simA, 0, seed=0, device=dev,
                                             record=True)
        # B rollout from same start (mode 1), record chunk-0 caches
        simB, zB = make_sim(pB)
        _, _, recsB, _ = rollout_joint(model, simB, 1, seed=0, device=dev, record=True)
        # inject B's chunk-0 cache into A episode
        simI, _ = make_sim(os.path.join(args.data, "planprobe", pa))
        okI, trajI, _ = replay_action(model, simI, recsA, "inject", device=dev,
                                      plan_cache=recsB[0]["caches"])
        # control: replay A with its own caches (fidelity check)
        simC, _ = make_sim(os.path.join(args.data, "planprobe", pa))
        _, trajC, _ = replay_action(model, simC, recsA, "none", device=dev)
        # expert trajectories
        eA = expert_ee(zA["joints"])
        eB = expert_ee(zB["joints"])
        k = min(4, len(trajI) - 1, len(trajA) - 1)
        dA4.append(np.linalg.norm(trajI[k] - eA[k]))
        dB4.append(np.linalg.norm(trajI[k] - eB[k]))
        dA_ctrl.append(ee_ade(trajC, trajA))
        n_ok += okI
        # "follows B" if injected trajectory is closer to B than to A overall
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


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="joint_data")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--variants", nargs="+",
                    default=["joint", "currentonly", "rift-l2", "rift-fm", "noiseslots"])
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    if not args.eval_only:
        ds_tr = JointDataset(os.path.join(args.data, "train"),
                             n=40 if args.smoke else None)
        dl = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch,
                                         shuffle=True, num_workers=2, drop_last=True)
        results = {}
        for variant in args.variants:
            torch.manual_seed(0)
            model = ToyJointWAM(d=args.dim, depth=args.depth).to(dev)
            E, sup_proj = None, None
            if variant in ("rift-l2", "rift-fm"):
                E = nn.Parameter(torch.randn(1, model.n_fut, args.dim, device=dev) * 0.02)
                model.sup_proj = nn.Linear(args.dim, args.dim).to(dev)
                model.fm_head = nn.Sequential(
                    nn.Linear(args.dim * 3, 512), nn.SiLU(),
                    nn.Linear(512, args.dim)).to(dev)
            steps = 30 if args.smoke else args.steps
            curve = train_variant(model, dl, dev, steps, args.lr, variant, E,
                                  out_every=200, sup_proj=getattr(model, "sup_proj", None))
            results[variant] = dict(curve=curve)
            torch.save({"model": model.state_dict(),
                        "E": E.detach().cpu() if E is not None else None},
                       os.path.join(args.out, f"{variant}_{args.tag}.pt"))
            print(f"[{variant}] saved", flush=True)
        with open(os.path.join(args.out, f"train_{args.tag}.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("training done")

    run_eval(args)


if __name__ == "__main__":
    main()
