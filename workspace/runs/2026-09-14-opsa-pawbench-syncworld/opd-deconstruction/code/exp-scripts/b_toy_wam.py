#!/usr/bin/env python3
"""B: stochastic-outcome world — does training objective decide the outcome
distribution? (toy pachinko world model, PAWBench-style measurement)

A ball falls through T peg rows; at each row it shifts left/right by 1 with
prob (1-p_r)/p_r. Final column = c0 + sum(shifts) ~ Binomial(T, p_r) shifted.
Action a sets c0 = 15 + a. Reference conditional distribution is analytic.

Three world-model variants, same decoder family, different training objective:
  det    : deterministic MSE over frames         (point estimate)
  vae    : Gaussian-latent VAE, ELBO             (unimodal prior)
  zdiff  : diffusion over the VAE latent         (flexible prior)
Measure on 500 repeated rollouts per (x0, a): outcome TVD vs reference,
valid-support coverage, and frame sharpness.

Commands:
  python b_toy_wam.py train-det
  python b_toy_wam.py train-vae
  python b_toy_wam.py train-zdiff
  python b_toy_wam.py eval-all
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_ROWS = 8
P_R = 0.4
N_FRAMES = 16
SIZE = 32
ACTIONS = [-2, -1, 0, 1, 2]
DEVICE = "cuda"
SEED = 0


# ---------------- simulator ----------------
def simulate(c0, rng):
    """Stochastic part is only the final column (binomial deflections); the
    path is a deterministic straight glide from start to the final column, so
    a decoder that knows the final column can render the video perfectly."""
    frames = np.zeros((N_FRAMES, SIZE, SIZE), dtype=np.float32)
    shifts = 0
    for _ in range(T_ROWS):
        shifts += 1 if rng.random() < P_R else -1
    final = c0 + shifts
    for t in range(N_FRAMES):
        col = int(round(c0 + (final - c0) * t / (N_FRAMES - 1)))
        x = min(max(col, 1), SIZE - 2)
        y = min(max(t, 1), SIZE - 2)
        frames[t, y - 1:y + 2, x - 1:x + 2] = 1.0
    return frames, final


def reference_probs(c0):
    """Reference outcome distribution over columns."""
    n = 2 * T_ROWS + 1
    probs = np.zeros(n)
    for k in range(T_ROWS + 1):
        p = math.comb(T_ROWS, k) * (P_R ** k) * ((1 - P_R) ** (T_ROWS - k))
        probs[k] = p
    # outcome c0 + (2k - T) => index (col - (c0-T)) = 2k => k = idx/2
    out = np.zeros(SIZE)
    for k in range(T_ROWS + 1):
        col = c0 + 2 * k - T_ROWS
        if 0 <= col < SIZE:
            out[col] = probs[k]
    return out / out.sum()


def draw_batch(c0s, rng):
    xs = []; cols = []
    for c0 in c0s:
        f, c = simulate(c0, rng)
        xs.append(f)
        cols.append(c)
    return torch.tensor(np.stack(xs)).unsqueeze(1), torch.tensor(cols, dtype=torch.long)


# ---------------- models ----------------
class Encoder(nn.Module):
    """Video encoder: q(z | full 16-frame video)."""
    def __init__(self, z_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, (3, 4, 4), (2, 2, 2), (1, 1, 1)), nn.ReLU(),
            nn.Conv3d(16, 32, (3, 4, 4), (2, 2, 2), (1, 1, 1)), nn.ReLU(),
            nn.Conv3d(32, 64, (3, 4, 4), (1, 2, 2), (1, 1, 1)), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.mu = nn.Linear(64, z_dim)
        self.lv = nn.Linear(64, z_dim)

    def forward(self, video):
        # video: (B, 1, T, H, W)
        h = self.net(video)
        return self.mu(h), self.lv(h)


class Decoder(nn.Module):
    """z + action -> N_FRAMES frames. Wider bottleneck (8x8x64)."""
    def __init__(self, z_dim=16):
        super().__init__()
        self.z_proj = nn.Sequential(nn.Linear(z_dim + 1, 512), nn.ReLU(),
                                    nn.Linear(512, 8 * 8 * 64), nn.ReLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, N_FRAMES, 3, 1, 1))

    def forward(self, z, a):
        h = self.z_proj(torch.cat([z, a.unsqueeze(-1)], dim=-1))
        h = h.view(h.size(0), 64, 8, 8)
        out = self.up(h)                    # (B, T, 32, 32)
        return out.unsqueeze(1)   # raw logits (B, 1, T, 32, 32)


class DetNet(nn.Module):
    """Deterministic: x0 + action -> frames."""
    def __init__(self):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(128 + 1, 512), nn.ReLU(),
                               nn.Linear(512, 8 * 8 * 64), nn.ReLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, N_FRAMES, 3, 1, 1))

    def forward(self, x0, a):
        h = self.fe(x0)
        h = self.h(torch.cat([h, a.unsqueeze(-1)], dim=-1))
        h = h.view(h.size(0), 64, 8, 8)
        return self.up(h).unsqueeze(1)   # raw logits (B, 1, T, 32, 32)


class ZDiff(nn.Module):
    """Tiny MLP diffusion over z, conditioned on (x0 embedding, a)."""
    def __init__(self, z_dim=16, cond_dim=32):
        super().__init__()
        self.x0_emb = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim + 1 + 1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, z_dim))

    def forward(self, z, t_scalar, x0, a):
        c = self.x0_emb(x0)
        t = t_scalar.unsqueeze(-1)
        return self.net(torch.cat([z, c, a.unsqueeze(-1), t], dim=-1))


def beta_schedule(T=100):
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    a_cum = np.cumprod(alphas)
    return (torch.tensor(betas, dtype=torch.float32, device=DEVICE),
            torch.tensor(alphas, dtype=torch.float32, device=DEVICE),
            torch.tensor(a_cum, dtype=torch.float32, device=DEVICE))


# ---------------- training ----------------
def make_batch(batch=256, rng=None):
    rng = rng or np.random.default_rng()
    a = rng.choice(ACTIONS, size=batch)
    c0s = [15 + int(x) for x in a]
    x, cols = draw_batch(c0s, rng)
    return x.to(DEVICE), torch.tensor(a, dtype=torch.float32, device=DEVICE), cols.to(DEVICE)


def train_det(args):
    model = DetNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    rng = np.random.default_rng(SEED)
    for step in range(1, args.steps + 1):
        x, a, _ = make_batch(256, rng)
        x0 = x[:, :, 0]
        pred = model(x0, a)
        loss = F.binary_cross_entropy_with_logits(pred, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"det {step}/{args.steps} loss={loss.item():.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), f"{args.out}/det.pt")


def train_vae(args):
    enc = Encoder(args.z).to(DEVICE)
    dec = Decoder(args.z).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-4)
    rng = np.random.default_rng(SEED)
    for step in range(1, args.steps + 1):
        x, a, _ = make_batch(256, rng)
        mu, lv = enc(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        rec = dec(z, a)
        recon = F.binary_cross_entropy_with_logits(rec, x)
        kl_dim = -0.5 * (1 + lv - mu ** 2 - lv.exp())      # (B, z_dim)
        # free bits: floor each dim's KL so the encoder cannot collapse z to the prior
        kl = torch.clamp(kl_dim, min=0.2).sum(-1).mean()
        beta = args.beta * min(1.0, step / 3000.0)  # KL warmup vs collapse
        loss = recon + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"vae {step}/{args.steps} rec={recon.item():.4f} kl_raw={kl_dim.sum(-1).mean().item():.4f} "
                  f"kl_fb={kl.item():.4f} beta={beta:.3f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()},
               f"{args.out}/vae.pt")


def train_zdiff(args):
    ck = torch.load(f"{args.out}/vae.pt", map_location=DEVICE)
    enc = Encoder(args.z).to(DEVICE)
    dec = Decoder(args.z).to(DEVICE)
    enc.load_state_dict(ck["enc"])
    dec.load_state_dict(ck["dec"])
    enc.eval(); dec.eval()
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    net = ZDiff(args.z).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    betas, alphas, a_cum = beta_schedule(100)
    rng = np.random.default_rng(SEED)
    for step in range(1, args.steps + 1):
        x, a, _ = make_batch(128, rng)
        x0 = x[:, :, 0]
        with torch.no_grad():
            mu, lv = enc(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            t = torch.randint(0, 100, (x.size(0),), device=DEVICE)
            abar = a_cum[t].unsqueeze(-1)
            noise = torch.randn_like(z)
            zt = torch.sqrt(abar) * z + torch.sqrt(1 - abar) * noise
        eps = net(zt, t.float() / 99.0, x0, a)
        loss = F.mse_loss(eps, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"zdiff {step}/{args.steps} loss={loss.item():.4f}", flush=True)
    torch.save(net.state_dict(), f"{args.out}/zdiff.pt")


@torch.no_grad()
def sample_zdiff(net, dec, x0, a, n=1):
    betas, alphas, a_cum = beta_schedule(100)
    z = torch.randn(x0.size(0), n, 16, device=DEVICE)
    a_rep = a.unsqueeze(1).expand(-1, n)
    x0_rep = x0.unsqueeze(1).expand(-1, n, 1, 32, 32).reshape(-1, 1, 32, 32)
    for t in range(99, -1, -1):
        tt = torch.full((z.numel() // z.size(-1),), t / 99.0, device=DEVICE)
        eps = net(z.reshape(-1, 16), tt, x0_rep, a_rep.reshape(-1))
        eps = eps.reshape(x0.size(0), n, 16)
        abar = a_cum[t]
        coef = betas[t] / torch.sqrt(1 - abar).clamp(min=1e-5)
        mean = (z - coef * eps) / torch.sqrt(alphas[t])
        if t > 0:
            z = mean + torch.sqrt(betas[t]) * torch.randn_like(z)
        else:
            z = mean
    return z  # (B, n, z_dim)


def readout_last_frame(videos):
    """Terminal column = argmax over columns of the last frame's row-summed
    occupancy. Operates on probabilities (sigmoid), not raw logits: logits
    from untrained border regions sit near 0 and would beat the genuinely lit
    columns whose background rows are strongly negative. The 3-wide smoothing
    matches the 3-pixel ball blob so the argmax lands on its centre (a plain
    argmax ties across the blob and picks the left edge)."""
    last = torch.sigmoid(videos[:, :, -1]).squeeze(1)   # (B, 32, 32)
    colmass = last.sum(dim=1)                            # (B, 32)
    smooth = F.conv1d(colmass.unsqueeze(1),
                      torch.ones(1, 1, 3, device=colmass.device) / 3.0,
                      padding=1).squeeze(1)
    return smooth.argmax(dim=1)


def eval_all(args):
    z_dim = args.z
    n_roll = args.n_roll
    det = DetNet().to(DEVICE)
    det.load_state_dict(torch.load(f"{args.out}/det.pt", map_location=DEVICE))
    det.eval()
    vck = torch.load(f"{args.out}/vae.pt", map_location=DEVICE)
    dec = Decoder(z_dim).to(DEVICE)
    dec.load_state_dict(vck["dec"])
    dec.eval()
    znet = ZDiff(z_dim).to(DEVICE)
    znet.load_state_dict(torch.load(f"{args.out}/zdiff.pt", map_location=DEVICE))
    znet.eval()

    rng = np.random.default_rng(1234)
    report = {"config": {"T": T_ROWS, "p_r": P_R, "actions": ACTIONS,
                         "n_roll": n_roll}, "per_action": {}}
    for a in ACTIONS:
        c0 = 15 + a
        ref = reference_probs(c0)
        x0 = np.zeros((1, 1, 32, 32), dtype=np.float32)
        # first frame of the episode: ball at top center col c0
        x0[0, 0, 1:3, c0 - 1:c0 + 2] = 1.0
        x0t = torch.tensor(x0).to(DEVICE)
        at = torch.tensor([float(a)], device=DEVICE)
        cols = {"det": [], "vae": [], "zdiff": [], "oracle": []}
        for i in range(0, n_roll, 64):
            b = min(64, n_roll - i)
            x0b = x0t.expand(b, 1, 32, 32)
            ab = at.expand(b)
            with torch.no_grad():
                vd = det(x0b, ab)
            cols["det"].append(readout_last_frame(vd).cpu())
            with torch.no_grad():
                z = torch.randn(b, z_dim, device=DEVICE)
                vv = dec(z, ab)
            cols["vae"].append(readout_last_frame(vv).cpu())
            z = sample_zdiff(znet, dec, x0b, ab, n=1).squeeze(1)
            vp = dec(z, ab)
            cols["zdiff"].append(readout_last_frame(vp).cpu())
            # oracle samples from simulator
            ocols = []
            for _ in range(b):
                _, c = simulate(c0, rng)
                ocols.append(c)
            cols["oracle"].append(torch.tensor(ocols))
        per_model = {}
        for m, parts in cols.items():
            allcols = torch.cat(parts).numpy()
            emp = np.bincount(allcols, minlength=SIZE) / len(allcols)
            tvd = 0.5 * np.abs(emp - ref).sum()
            support = ref > 0
            coverage = float((emp[support] > 0).sum()) / support.sum()
            per_model[m] = {"tvd": float(tvd), "coverage": coverage,
                            "top3": emp.argsort()[::-1][:3].tolist()}
        report["per_action"][str(a)] = {"c0": c0, "models": per_model,
                                        "ref_top3": ref.argsort()[::-1][:3].tolist()}
        print(json.dumps({str(a): per_model}, indent=1), flush=True)
    with open(f"{args.out}/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("saved", f"{args.out}/eval_report.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["train-det", "train-vae", "train-zdiff"]:
        s = sub.add_parser(name)
        s.add_argument("--steps", type=int, default=30000)
        s.add_argument("--out", default="results/toywam")
        s.add_argument("--z", type=int, default=16)
        s.add_argument("--beta", type=float, default=1.0)
    s = sub.add_parser("eval-all")
    s.add_argument("--out", default="results/toywam")
    s.add_argument("--z", type=int, default=16)
    s.add_argument("--n-roll", type=int, default=500)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    if args.cmd == "train-det":
        train_det(args)
    elif args.cmd == "train-vae":
        train_vae(args)
    elif args.cmd == "train-zdiff":
        train_zdiff(args)
    else:
        eval_all(args)


if __name__ == "__main__":
    main()
