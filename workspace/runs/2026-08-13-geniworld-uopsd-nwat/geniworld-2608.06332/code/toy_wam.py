"""Toy flow-matching world model: 4 action-interface ablations on synthetic arm videos.

Mirrors the GeniWorld vs Ctrl-World interface question at toy scale:
  cond variants (all on the same 3D-UNet backbone, pixel-space flow matching):
    numeric        : end-effector trajectory (T,3) -> MLP -> frame-wise cross-attn
                     (Ctrl-World style numeric conditioning)
    concat_static  : static arm render (mean pose, repeated) channel-concatenated
                     (spatial grounding only, no motion)
    concat_shuffle : per-frame arm renders with permuted order channel-concatenated
                     (marginal pixel stats match, temporal motion destroyed)
    concat_motion  : per-frame arm renders channel-concatenated (GeniWorld)

Questions:
  Q1 (paper's claim): does motion-render conditioning converge faster / predict
      better than numeric conditioning?
  Q2 (paper's blind spot): is the gain from spatial grounding (static/shuffle) or
      from the motion signal itself (motion > shuffle)?
  Q3 (OOD): do render-conditioned variants generalize to unseen backgrounds?
  Q4 (few-step): does render conditioning degrade less under few Euler steps?

Usage: python toy_wam.py --data arm_data --cond all --steps 8000 --tag v1
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
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------- model
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
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


class FrameCrossAttn(nn.Module):
    """Spatial tokens attend to the full sequence of per-frame action tokens."""
    def __init__(self, ch, ctx=64, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv3d(ch, ch, 1)
        self.k = nn.Linear(ctx, ch)
        self.v = nn.Linear(ctx, ch)
        self.proj = nn.Conv3d(ch, ch, 1)
        self.heads = heads

    def forward(self, x, act):  # x:(B,C,T',H',W') act:(B,T,ctx)
        B, C, Tp, Hp, Wp = x.shape
        d = C // self.heads
        h = self.norm(x)
        q = self.q(h).reshape(B, self.heads, d, Tp, Hp * Wp).permute(0, 1, 3, 4, 2)  # B,h,P,HW,d
        T = act.shape[1]
        k = self.k(act).reshape(B, self.heads, T, d)                                # B,h,T,d
        v = self.v(act).reshape(B, self.heads, T, d)                                # B,h,T,d
        att = torch.softmax(torch.einsum("bhznd,bhtd->bhznt", q, k) / d ** 0.5, dim=-1)
        out = torch.einsum("bhznt,bhtd->bhznd", att, v)                             # B,h,P,HW,d
        out = out.reshape(B, C, Tp, Hp, Wp)
        return x + self.proj(out)


class ToyWAM(nn.Module):
    def __init__(self, cond="concat_motion", in_ch=3, act_dim=3, base=64):
        super().__init__()
        self.cond = cond
        use_concat = cond.startswith("concat")
        self.use_numeric = cond == "numeric"
        in_ch = in_ch + (3 if use_concat else 0)
        tch = base * 4
        self.temb = nn.Sequential(nn.Linear(base, tch), nn.SiLU(), nn.Linear(tch, tch))
        self.act_mlp = nn.Sequential(nn.Linear(act_dim, 64), nn.SiLU(), nn.Linear(64, 64)) if self.use_numeric else None

        chs = [base, base * 2, base * 4]
        self.in_conv = nn.Conv3d(in_ch, chs[0], 3, padding=1)
        self.down1 = nn.Sequential(ResBlock3D(chs[0], tch), nn.Conv3d(chs[0], chs[1], 3, stride=2, padding=1))
        self.down2 = nn.Sequential(ResBlock3D(chs[1], tch), nn.Conv3d(chs[1], chs[2], 3, stride=2, padding=1))
        self.bottleneck = ResBlock3D(chs[2], tch)
        self.cross = FrameCrossAttn(chs[2]) if self.use_numeric else None
        self.up1 = nn.Sequential(ResBlock3D(chs[2], tch),
                                 nn.ConvTranspose3d(chs[2], chs[1], 3, stride=2, padding=1, output_padding=1))
        self.up2 = nn.Sequential(ResBlock3D(chs[1], tch),
                                 nn.ConvTranspose3d(chs[1], chs[0], 3, stride=2, padding=1, output_padding=1))
        self.out_block = ResBlock3D(chs[0], tch)
        self.out_conv = nn.Conv3d(chs[0], 3, 3, padding=1)

    def forward(self, x, t, act):
        temb = self.temb(timestep_embedding(t, self.temb[0].in_features))
        if self.use_numeric:
            act_tok = self.act_mlp(act)  # (B,T,64)
        h0 = self.in_conv(x)
        h1 = self.down1[0](h0, temb)
        h1 = self.down1[1](h1)
        h2 = self.down2[0](h1, temb)
        h2 = self.down2[1](h2)
        hb = self.bottleneck(h2, temb)
        if self.use_numeric:
            hb = self.cross(hb, act_tok)
        u1 = self.up1[0](hb, temb)
        u1 = self.up1[1](u1)
        u1 = u1 + h1
        u2 = self.up2[0](u1, temb)
        u2 = self.up2[1](u2)
        u2 = u2 + h0
        h = self.out_block(u2, temb)
        return self.out_conv(h)


# ---------------------------------------------------------------- data
class ArmDataset(Dataset):
    def __init__(self, root, cond, n=None):
        self.cond = cond
        self.files = sorted(os.listdir(root))[:n]
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(os.path.join(self.root, self.files[i]))
        frames = torch.from_numpy(z["frames"]).float() / 255.0 * 2 - 1  # (T,H,W,3)
        frames = frames.permute(3, 0, 1, 2).contiguous()  # (3,T,H,W)
        renders = torch.from_numpy(z["renders"]).float() / 255.0 * 2 - 1
        renders = renders.permute(3, 0, 1, 2).contiguous()
        numeric = torch.from_numpy(z["numeric"]).float()  # (T,3)
        cube = torch.from_numpy(z["cube"]).float()
        if self.cond == "numeric":
            return frames, numeric, cube
        if self.cond == "concat_motion":
            act = renders
        elif self.cond == "concat_static":
            act = torch.from_numpy(z["static"]).float() / 255.0 * 2 - 1
            act = act.permute(3, 0, 1, 2).contiguous()
        elif self.cond == "concat_shuffle":
            act = torch.from_numpy(z["shuffled"]).float() / 255.0 * 2 - 1
            act = act.permute(3, 0, 1, 2).contiguous()
        else:
            raise ValueError(self.cond)
        return frames, act, cube


def mse(a, b):
    return float(((a - b) ** 2).mean())


@torch.no_grad()
def sample(model, cond, x0, act, steps=50, device="cuda"):
    """Euler flow-matching sampling conditioned on act."""
    model.eval()
    B = x0.shape[0]
    z = torch.randn_like(x0)
    dt = 1.0 / steps
    for i in range(steps):
        s = i / steps
        t = torch.full((B,), s, device=device)
        inp = torch.cat([z, act], dim=1) if cond.startswith("concat") else z
        v = model(inp, t, act)
        z = z + v * dt
    return z


def cube_err(pred, gt_cube, device="cuda"):
    """Red-cube centroid error in final predicted frame vs scripted truth (pixels)."""
    errs = []
    pred = ((pred + 1) / 2).clamp(0, 1).detach().cpu().numpy()  # (B,3,T,H,W)
    gt = gt_cube.cpu().numpy()
    for b in range(pred.shape[0]):
        f = pred[b, :, -1].transpose(1, 2, 0)  # (H,W,3)
        mask = (f[:, :, 0] > 0.6) & (f[:, :, 1] < 0.4) & (f[:, :, 2] < 0.4)
        ys, xs = np.nonzero(mask)
        if len(xs) < 3:
            errs.append(float("nan"))
            continue
        c = np.array([xs.mean(), ys.mean()])
        errs.append(float(np.linalg.norm(c - gt[b, -1])))
    return [e for e in errs if e == e]  # drop nan


def evaluate(model, loader, cond, device, steps=50):
    model.eval()
    tot_mse, tot_ssim, n = 0.0, 0.0, 0
    errs = []
    with torch.no_grad():
        for batch in loader:
            if cond == "numeric":
                x0, act, cube = [b.to(device) for b in batch]
            else:
                x0, act, cube = [b.to(device) for b in batch]
            pred = sample(model, cond, x0, act, steps=steps, device=device)
            tot_mse += mse(pred, x0) * x0.shape[0]
            n += x0.shape[0]
            errs += cube_err(pred, cube, device)
    return tot_mse / n, float(np.mean(errs)) if errs else float("nan")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="arm_data")
    ap.add_argument("--cond", default="all", choices=["numeric", "concat_static", "concat_shuffle", "concat_motion", "all"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-steps", type=int, nargs="+", default=[50, 10, 5])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    conds = ["numeric", "concat_static", "concat_shuffle", "concat_motion"] if args.cond == "all" else [args.cond]
    os.makedirs(args.out, exist_ok=True)
    results = {}

    for cond in conds:
        torch.manual_seed(0)
        np.random.seed(0)
        dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = ToyWAM(cond=cond).to(dev)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        ds_tr = ArmDataset(os.path.join(args.data, "train"), cond)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
        ds_val = ArmDataset(os.path.join(args.data, "val"), cond, n=64)
        dl_val = DataLoader(ds_val, batch_size=args.batch, num_workers=2)
        ds_ood1 = ArmDataset(os.path.join(args.data, "ood_stripes"), cond, n=64)
        dl_ood1 = DataLoader(ds_ood1, batch_size=args.batch, num_workers=2)
        ds_ood2 = ArmDataset(os.path.join(args.data, "ood_dots"), cond, n=64)
        dl_ood2 = DataLoader(ds_ood2, batch_size=args.batch, num_workers=2)

        print(f"[{cond}] params={n_params:.1f}M, train clips={len(ds_tr)}")
        curve = []
        t0 = time.time()
        step = 0
        while step < args.steps:
            for batch in dl_tr:
                if step >= args.steps:
                    break
                model.train()
                if cond == "numeric":
                    x0, act, _ = [b.to(dev) for b in batch]
                else:
                    x0, act, _ = [b.to(dev) for b in batch]
                B = x0.shape[0]
                t = torch.rand(B, device=dev)
                z = torch.randn_like(x0)
                xt = (1 - t)[:, None, None, None, None] * z + t[:, None, None, None, None] * x0
                inp = torch.cat([xt, act], dim=1) if cond.startswith("concat") else xt
                v = model(inp, t, act)
                target = x0 - z
                loss = F.mse_loss(v, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                if step % 500 == 0:
                    val_mse, val_ce = evaluate(model, dl_val, cond, dev, steps=50)
                    curve.append([step, val_mse, val_ce])
                    print(f"[{cond}] step {step}/{args.steps} val_mse={val_mse:.4f} cube_err={val_ce:.1f} ({time.time()-t0:.0f}s)")
                    t0 = time.time()

        # final evaluations
        out = {"cond": cond, "params_m": n_params, "steps": args.steps, "curve": curve}
        for name, dl in [("val", dl_val), ("ood_stripes", dl_ood1), ("ood_dots", dl_ood2)]:
            for es in args.eval_steps:
                m, ce = evaluate(model, dl, cond, dev, steps=es)
                out[f"{name}_mse@{es}"] = m
                out[f"{name}_cube_err@{es}"] = ce
                print(f"[{cond}] {name} @{es} steps: mse={m:.4f} cube_err={ce:.1f}")
        results[cond] = out

    with open(os.path.join(args.out, f"results_{args.tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved results to", os.path.join(args.out, f"results_{args.tag}.json"))


if __name__ == "__main__":
    main()
