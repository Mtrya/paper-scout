"""Synthetic world-action testbed for action-head readout.

A minimal WAM: a shared visual encoder trained with a next-frame video loss, and
an action head that reads the encoder tokens. The action head is either dense
(all tokens) or top-k sparse (CWAM-style routing). The scene contains one
action-relevant object (the goal, marked with a white cue) and one nuisance
object (a look-alike without the cue). In-distribution training correlates the
nuisance position with the goal; out-of-distribution evaluation decorrelates
them. Video-only OOD adaptation and sparse action demos are toggles.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 48
H = 4
MAX_STEP = 0.25


def render(goal, dist, size=SIZE):
    """goal, dist: (B,2) in [-1,1]. Agent is fixed at the image center."""
    b = goal.shape[0]
    ys = torch.linspace(-1, 1, size, device=goal.device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    img = torch.zeros(b, 3, size, size, device=goal.device)
    img[:, 0] = 0.07
    img[:, 1] = 0.07
    img[:, 2] = 0.09
    grid = (((xx * size / 2).round() % 8 == 0) | ((yy * size / 2).round() % 8 == 0)).float()
    img = img + 0.035 * grid.unsqueeze(0).unsqueeze(0)

    def square(px, py, half, color):
        px = px.view(-1, 1, 1)
        py = py.view(-1, 1, 1)
        m = ((xx - px).abs() < half) & ((yy - py).abs() < half)
        for c in range(3):
            img[:, c] = torch.where(m, torch.full_like(img[:, c], color[c]), img[:, c])

    # agent at center
    square(torch.zeros(b, device=goal.device), torch.zeros(b, device=goal.device), 0.10, (0.1, 0.9, 0.2))
    # goal and distractor: same red body
    square(goal[:, 0], goal[:, 1], 0.16, (0.95, 0.15, 0.15))
    square(dist[:, 0], dist[:, 1], 0.16, (0.95, 0.15, 0.15))
    # cue on the goal only
    square(goal[:, 0], goal[:, 1], 0.05, (1.0, 1.0, 1.0))
    return img


def next_obs(goal, dist):
    step = (0.5 * goal).clamp(-MAX_STEP, MAX_STEP)
    return render(goal - step, dist - step)


def sample_id(b, device=DEVICE):
    goal = (torch.rand(b, 2, device=device) * 2 - 1) * 0.8
    off = torch.randn(b, 2, device=device) * 0.10
    dist = (goal + off).clamp(-0.9, 0.9)
    return goal, dist


def sample_ood(b, device=DEVICE, min_sep=0.5):
    goal = (torch.rand(b, 2, device=device) * 2 - 1) * 0.8
    dist = (torch.rand(b, 2, device=device) * 2 - 1) * 0.8
    for _ in range(20):
        far = (goal - dist).norm(dim=1) < min_sep
        if not far.any():
            break
        dist[far] = (torch.rand(int(far.sum()), 2, device=device) * 2 - 1) * 0.8
    return goal, dist


def action_target(goal):
    b = goal.shape[0]
    p = torch.zeros(b, 2, device=goal.device)
    out = []
    for _ in range(H):
        a = (0.5 * (goal - p)).clamp(-MAX_STEP, MAX_STEP)
        out.append(a / MAX_STEP)
        p = p + a
    return torch.stack(out, dim=1)


def rollout_success(pred, goal):
    p = torch.zeros(goal.shape[0], 2, device=goal.device)
    for k in range(H):
        p = p + (pred[:, k] * MAX_STEP).clamp(-MAX_STEP, MAX_STEP)
    return ((p - goal).norm(dim=1) < 0.10).float(), (p - goal).norm(dim=1)


class Encoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.pos = nn.Parameter(torch.randn(1, 36, width) * 0.02)

    def forward(self, img):
        f = self.net(img)
        b, c, h, w = f.shape
        return f.flatten(2).transpose(1, 2) + self.pos


class VideoDecoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.proj = nn.Linear(width, 128)
        self.up = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, tokens):
        b = tokens.shape[0]
        f = self.proj(tokens).transpose(1, 2).reshape(b, 128, 6, 6)
        f = F.interpolate(f, size=(12, 12), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.up(f))


class ActionHead(nn.Module):
    def __init__(self, width=128, topk=36, router_mode="learned", fixed_tokens=None):
        super().__init__()
        self.topk = topk
        self.router_mode = router_mode
        self.fixed_tokens = fixed_tokens
        self.width = width
        self.q = nn.Parameter(torch.randn(1, H, width) * 0.02)
        self.q_proj = nn.Linear(width, width)
        self.bias = nn.Parameter(torch.zeros(36))
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, tokens):
        b = tokens.shape[0]
        q = self.q_proj(self.q).expand(b, -1, -1)
        logits = torch.einsum("btd,bnd->btn", q, tokens) / math.sqrt(self.width) + self.bias
        w_full = logits.softmax(dim=-1)
        if self.router_mode == "fixed":
            idx = torch.tensor(self.fixed_tokens, device=tokens.device)
            w = torch.zeros_like(w_full)
            for h in range(min(H, len(idx))):
                w[:, h, idx[h]] = 1.0
            return self.mlp(torch.einsum("btn,bnd->btd", w, tokens)), w, torch.zeros((), device=tokens.device)
        if self.topk >= tokens.shape[1]:
            w = w_full
            sparsity = torch.zeros((), device=tokens.device)
        else:
            top = w_full.topk(self.topk, dim=-1).indices
            mask = torch.zeros_like(w_full).scatter_(-1, top, 1.0)
            w = w_full * mask
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            sparsity = (1.0 - (w_full * mask).sum(dim=-1)).mean()
        read = torch.einsum("btn,bnd->btd", w, tokens)
        return self.mlp(read), w, sparsity


class WAM(nn.Module):
    def __init__(self, width=128, topk=36, with_video=True, router_mode="learned", fixed_tokens=None):
        super().__init__()
        self.encoder = Encoder(width)
        self.head = ActionHead(width, topk, router_mode, fixed_tokens)
        self.video = VideoDecoder(width) if with_video else None

    def forward(self, img):
        tokens = self.encoder(img)
        actions, w, sparsity = self.head(tokens)
        video = self.video(tokens) if self.video is not None else None
        return actions, video, w, sparsity


def make_batch(kind, b):
    if kind == "id":
        return sample_id(b)
    return sample_ood(b)


def evaluate(model, kind, n=1024, seed=0):
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    gs, ds = [], []
    left = n
    while left > 0:
        b = min(256, left)
        if kind == "id":
            g = (torch.rand(b, 2, device=DEVICE, generator=gen) * 2 - 1) * 0.8
            d = (g + torch.randn(b, 2, device=DEVICE, generator=gen) * 0.06).clamp(-0.9, 0.9)
        else:
            g = (torch.rand(b, 2, device=DEVICE, generator=gen) * 2 - 1) * 0.8
            d = (torch.rand(b, 2, device=DEVICE, generator=gen) * 2 - 1) * 0.8
            far = (g - d).norm(dim=1) < 0.5
            for _ in range(20):
                if not far.any():
                    break
                d[far] = (torch.rand(int(far.sum()), 2, device=DEVICE, generator=gen) * 2 - 1) * 0.8
                far = (g - d).norm(dim=1) < 0.5
        gs.append(g); ds.append(d); left -= b
    g = torch.cat(gs); d = torch.cat(ds)
    with torch.no_grad():
        img = render(g, d)
        pred, vid, w, _ = model(img)
        # nuisance reliance: move distractor while keeping goal fixed
        d2 = (torch.rand_like(d) * 2 - 1) * 0.8
        pred2, _, _, _ = model(render(g, d2))
        reliance = (pred - pred2).norm(dim=-1).mean()
        succ, dist_err = rollout_success(pred, g)
        mse = F.mse_loss(pred, action_target(g)).item()
        vmse = F.mse_loss(vid, next_obs(g, d)).item() if vid is not None else float("nan")
        # routing mass near goal vs distractor
        if model.head.topk < 36:
            wm = w.mean(dim=1)  # b, 36
            c = -1 + (torch.arange(6, device=DEVICE) + 0.5) * (2 / 6)
            coords = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(36, 2)
            # token centers roughly at pixel centers of the 6x6 grid
            gp = g.unsqueeze(1)  # b,1,2
            dp = d.unsqueeze(1)
            dg = (coords.unsqueeze(0) - gp).norm(dim=-1)
            dd = (coords.unsqueeze(0) - dp).norm(dim=-1)
            near_g = (dg < 0.30).float()
            near_d = (dd < 0.30).float()
            mass_g = (wm * near_g).sum(1) / wm.sum(1).clamp_min(1e-9)
            mass_d = (wm * near_d).sum(1) / wm.sum(1).clamp_min(1e-9)
            sel = w.argmax(-1)
            sel_coords = coords[sel]
            sel_goal = ((sel_coords - g[:, None]).norm(dim=-1) < 0.30).float().mean()
            sel_dist = ((sel_coords - d[:, None]).norm(dim=-1) < 0.30).float().mean()
            sel_center = sel_coords.norm(dim=-1).mean()
        else:
            mass_g = mass_d = torch.tensor(float("nan"))
            sel_goal = sel_dist = sel_center = torch.tensor(float("nan"))
    model.train()
    return {
        "success": succ.mean().item(),
        "dist_err": dist_err.mean().item(),
        "action_mse": mse,
        "video_mse": vmse,
        "nuisance_reliance": reliance.item(),
        "mass_goal": mass_g.mean().item(),
        "mass_dist": mass_d.mean().item(),
        "sel_goal_frac": sel_goal.item(),
        "sel_dist_frac": sel_dist.item(),
        "sel_center_dist": sel_center.item(),
    }


def train(args):
    torch.manual_seed(args.seed)
    fixed = [int(x) for x in args.fixed_tokens.split(",")] if args.fixed_tokens else None
    model = WAM(topk=args.topk, with_video=not args.no_video, router_mode=args.router_mode,
                fixed_tokens=fixed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.pretrain_steps + args.train_steps
    step = 0
    t0 = time.time()

    def lr_at(s):
        warm = min(1.0, s / 50)
        return args.lr * warm * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    def set_lr(s):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)

    # video-only pretraining on ID scenes
    if not args.no_video:
        for s in range(args.pretrain_steps):
            set_lr(step); step += 1
            g, d = sample_id(args.batch)
            img = render(g, d)
            _, vid, _, _ = model(img)
            loss = F.mse_loss(vid, next_obs(g, d))
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 100 == 0:
                print(f"pretrain {s} video {loss.item():.5f}", flush=True)

    # action training: ID actions, optional OOD videos
    for s in range(args.train_steps):
        set_lr(step); step += 1
        g, d = sample_id(args.batch)
        img = render(g, d)
        pred, vid, _, sparse = model(img)
        loss = F.mse_loss(pred, action_target(g))
        if not args.no_video:
            loss = loss + args.video_weight * F.mse_loss(vid, next_obs(g, d))
        if args.ood_video > 0:
            g2, d2 = sample_ood(args.ood_video)
            img2 = render(g2, d2)
            _, vid2, _, _ = model(img2)
            loss = loss + args.video_weight * F.mse_loss(vid2, next_obs(g2, d2))
        if args.topk < 36:
            loss = loss + args.sparse_weight * sparse
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0:
            print(f"train {s} loss {loss.item():.5f} act {F.mse_loss(pred, action_target(g)).item():.5f}", flush=True)

    # optional sparse OOD action grounding: fine-tune the head only
    if args.ood_demos > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        if model.video is not None:
            for p in model.video.parameters():
                p.requires_grad = False
        # fixed demo set, one (or a few) episodes per task
        demo_g, demo_d = [], []
        for _ in range(args.ood_demos):
            g, d = sample_ood(max(8, args.batch // max(1, args.ood_demos)))
            demo_g.append(g); demo_d.append(d)
        demo_g = torch.cat(demo_g); demo_d = torch.cat(demo_d)
        opt2 = torch.optim.AdamW(model.head.parameters(), lr=args.lr * 0.3)
        for s in range(args.demo_steps):
            # half OOD demos, half ID replay
            go = demo_g[torch.randint(len(demo_g), (args.batch // 2,), device=DEVICE)]
            do = demo_d[torch.randint(len(demo_d), (args.batch // 2,), device=DEVICE)]
            gi, di = sample_id(args.batch - len(go))
            g = torch.cat([go, gi]); d = torch.cat([do, di])
            pred, _, _, sparse = model(render(g, d))
            loss = F.mse_loss(pred, action_target(g))
            if args.topk < 36:
                loss = loss + args.sparse_weight * sparse
            opt2.zero_grad(); loss.backward(); opt2.step()
        for p in model.parameters():
            p.requires_grad = True

    res = {
        "args": vars(args),
        "id": evaluate(model, "id"),
        "ood": evaluate(model, "ood"),
        "time_s": time.time() - t0,
    }
    return res, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=36)
    ap.add_argument("--ood_video", type=int, default=0)
    ap.add_argument("--ood_demos", type=int, default=0)
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--pretrain_steps", type=int, default=300)
    ap.add_argument("--train_steps", type=int, default=600)
    ap.add_argument("--demo_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--video_weight", type=float, default=0.5)
    ap.add_argument("--sparse_weight", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--router_mode", type=str, default="learned")
    ap.add_argument("--fixed_tokens", type=str, default="0,5,30,35")
    args = ap.parse_args()
    res, model = train(args)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        if args.save_model:
            torch.save(model.state_dict(), args.out.replace(".json", ".pt"))


if __name__ == "__main__":
    main()
