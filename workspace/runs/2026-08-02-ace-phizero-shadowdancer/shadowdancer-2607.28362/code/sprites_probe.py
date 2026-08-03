"""ShadowDancer mechanism probe at toy scale (two-ball sprites).

Question: does cross-shadow prediction yield appearance-invariant dynamics
latents where self-reconstruction (with or without KL regularization) fails?
Mirrors Sec. D.3/D.4 of arXiv:2607.28362 at toy scale.

Setup: two balls per frame; one is static, the other moves with constant
velocity (direction theta, speed 3-6 px/frame). Colors are CONTINUOUS (HSV),
so exact shade information in z reduces reconstruction MSE -- this is what
gives self-reconstruction an incentive to smuggle appearance into z.
A shadow pair shares positions and velocities; colors/background/radii are
resampled. Predicting the next frame requires z to say WHICH ball moves and
WHERE -- information not readable from the context frame alone.

Three LAM variants, identical architecture (z dim 32):
  A    self-reconstruction, beta=0.01   (Olaf-style baseline)
  Ahb  self-reconstruction, beta=1.0    (can regularization substitute pairing?)
  B    cross-shadow prediction, beta=0.01

Probes on frozen z (held-out pairs, ridge regression R^2):
  - color of moving ball     (appearance leak;   0 = no leak)
  - color of static ball     (appearance leak;   0 = no leak)
  - (cos, sin) of direction  (dynamics content;  1 = perfect)
  - cross/self MSE ratio of decoding the target shadow
"""

import json
import math
import colorsys
import numpy as np
import torch
import torch.nn as nn

SEED = 0
IMG = 48
ZDIM = 32
N_TRAIN, N_TEST = 24000, 3000
STEPS, BATCH = 3000, 128
DEVICE = "cpu"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def rand_color():
    h, s, v = rng.uniform(0, 1), rng.uniform(0.6, 1.0), rng.uniform(0.7, 1.0)
    return np.array(colorsys.hsv_to_rgb(h, s, v), np.float32)


def render(balls, bg):
    """balls: list of (x, y, radius, color). Returns CHW float32."""
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img = np.full((IMG, IMG, 3), bg, np.float32)
    for x, y, r, c in balls:
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        img[mask] = c
    return img.transpose(2, 0, 1)


def sample_transition():
    """One dynamics sample; returns (state, move_dir)."""
    while True:
        pa = rng.uniform(9, IMG - 9, 2)
        pb = rng.uniform(9, IMG - 9, 2)
        if np.linalg.norm(pa - pb) > 14:
            break
    theta = rng.uniform(0, 2 * math.pi)
    speed = rng.uniform(8.0, 14.0)  # high stakes: z's MSE gain (~0.02) must
    # outweigh beta*KL (~0.015 at beta=0.01), else collapse is the optimum
    vel = speed * np.array([math.cos(theta), math.sin(theta)])
    pb1 = pb + vel
    if not (6 < pb1[0] < IMG - 6 and 6 < pb1[1] < IMG - 6):
        return sample_transition()
    return pa, pb, pb1, theta


def sample_pair(n):
    """n shadow pairs. Item: (src, tgt) each (x_t, x_t1, moving_rgb, static_rgb, theta)."""
    out = []
    for _ in range(n):
        pa, pb, pb1, theta = sample_transition()
        trans = []
        for _ in range(2):
            ca, cb = rand_color(), rand_color()
            ra, rb = rng.uniform(3.0, 5.5, 2)
            bg = rng.uniform(0.03, 0.15)
            xt = render([(pa[0], pa[1], ra, ca), (pb[0], pb[1], rb, cb)], bg)
            xt1 = render([(pa[0], pa[1], ra, ca), (pb1[0], pb1[1], rb, cb)], bg)
            trans.append((xt, xt1, cb.copy(), ca.copy(), theta))
        out.append(trans)
    return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mu = nn.Linear(128, ZDIM)
        self.lv = nn.Linear(128, ZDIM)
        # start with a low-noise channel (sigma ~0.37) so z is a clean
        # deterministic pathway at init; with sigma=1 the decoder learns to
        # ignore z before the encoder can shape it (absorbing collapse).
        nn.init.constant_(self.lv.bias, -2.0)
        # priming head: used only during the first PRIME_STEPS steps to escape
        # the absorbing "decoder ignores z" state (toy-scale privileged info,
        # identical for all variants; dropped afterwards so the converged z is
        # shaped by each variant's own objective)
        self.aux = nn.Linear(ZDIM, 2)

    def forward(self, xt, xt1):
        h = self.net(torch.cat([xt, xt1], 1))
        return self.mu(h), self.lv(h).clamp(-8, 8)


class Decoder(nn.Module):
    """Concat z as an extra image channel, then a small conv stack."""

    def __init__(self):
        super().__init__()
        self.zmap = nn.Linear(ZDIM, IMG * IMG)
        self.up = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, xt, z):
        # no sigmoid here: saturation at init flattens gradients and kills the
        # z pathway before it can become useful
        zmap = self.zmap(z).view(-1, 1, IMG, IMG)
        return torch.sigmoid(self.up(torch.cat([xt, zmap], 1)))


def batch_from(pairs, idx, cross):
    src = np.stack([pairs[i][0][0] for i in idx])
    src1 = np.stack([pairs[i][0][1] for i in idx])
    tgt = np.stack([pairs[i][1][0] for i in idx])
    tgt1 = np.stack([pairs[i][1][1] for i in idx])
    direc = np.stack([[math.cos(pairs[i][0][4]), math.sin(pairs[i][0][4])]
                      for i in idx])
    t = lambda a: torch.from_numpy(a).to(DEVICE)
    if cross:
        return t(src), t(src1), t(tgt), t(tgt1), t(direc)
    return t(src), t(src1), t(src), t(src1), t(direc)


PRIME_STEPS = 600


def train_variant(name, cross, beta, train_pairs):
    # Optimization escapes only; identical for all variants:
    #  - steps 0..PRIME_STEPS: auxiliary linear head on mu predicts the motion
    #    direction (privileged toy-scale info) so the decoder learns z carries
    #    signal before it can zero its z-input weights;
    #  - beta is 0 until step 800, then ramps to beta by step 1600.
    # Without these, all variants posterior-collapse at this toy scale (the
    # blur-copy solution is an absorbing state for an auxiliary-latent VAE).
    enc, dec = Encoder().to(DEVICE), Decoder().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    # copy baseline: MSE of predicting x_{t+1} = x_t
    idx = rng.integers(0, len(train_pairs), 512)
    _, sx1, tx, tx1, _ = batch_from(train_pairs, idx, cross)
    print(f"  [{name}] copy-baseline mse="
          f"{nn.functional.mse_loss(tx, tx1).item():.4f}", flush=True)
    for step in range(STEPS):
        if step < 800:
            beta_eff = 0.0
        else:
            beta_eff = beta * min(1.0, (step - 800) / 800.0)
        idx = rng.integers(0, len(train_pairs), BATCH)
        sx, sx1, tx, tx1, direc = batch_from(train_pairs, idx, cross)
        mu, lv = enc(sx, sx1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        pred = dec(tx, z)
        mse = nn.functional.mse_loss(pred, tx1)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        loss = mse + beta_eff * kl
        if step < PRIME_STEPS:
            loss = loss + nn.functional.mse_loss(enc.aux(mu), direc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 600 == 0:
            print(f"  [{name}] step {step+1}/{STEPS} mse={mse.item():.4f} kl={kl.item():.3f}", flush=True)
    enc.eval()
    dec.eval()
    return enc, dec


@torch.no_grad()
def encode_all(enc, pairs, side=0):
    xs = torch.from_numpy(np.stack([p[side][0] for p in pairs])).to(DEVICE)
    xs1 = torch.from_numpy(np.stack([p[side][1] for p in pairs])).to(DEVICE)
    mus = []
    for i in range(0, len(xs), 512):
        mu, _ = enc(xs[i:i + 512], xs1[i:i + 512])
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus)


@torch.no_grad()
def recon_mse(enc, dec, pairs, z_side, n=512):
    pairs = pairs[:n]
    tx = torch.from_numpy(np.stack([p[1][0] for p in pairs])).to(DEVICE)
    tx1 = torch.from_numpy(np.stack([p[1][1] for p in pairs])).to(DEVICE)
    zx = torch.from_numpy(np.stack([p[z_side][0] for p in pairs])).to(DEVICE)
    zx1 = torch.from_numpy(np.stack([p[z_side][1] for p in pairs])).to(DEVICE)
    mu, _ = enc(zx, zx1)
    pred = dec(tx, mu)
    return nn.functional.mse_loss(pred, tx1).item()


def ridge_probe(Xtr, Ytr, Xte, Yte):
    """Multi-output ridge regression R^2 on the test set."""
    Xtr_ = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte_ = np.hstack([Xte, np.ones((len(Xte), 1))])
    W = np.linalg.solve(Xtr_.T @ Xtr_ + 1e-3 * np.eye(Xtr_.shape[1]), Xtr_.T @ Ytr)
    pred = Xte_ @ W
    ss_res = ((pred - Yte) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-9)))


def targets(pairs):
    move_rgb = np.stack([p[0][2] for p in pairs])
    stat_rgb = np.stack([p[0][3] for p in pairs])
    direction = np.stack([[math.cos(p[0][4]), math.sin(p[0][4])] for p in pairs])
    return move_rgb, stat_rgb, direction


def main():
    print("generating data ...", flush=True)
    train_pairs = sample_pair(N_TRAIN)
    test_pairs = sample_pair(N_TEST)
    tr_move, tr_stat, tr_dir = targets(train_pairs[:6000])
    te_move, te_stat, te_dir = targets(test_pairs)

    results = {}
    for name, cross, beta in [("A-selfrec-b0.01", False, 0.01),
                              ("Ahb-selfrec-b1.0", False, 1.0),
                              ("B-crosshadow-b0.01", True, 0.01)]:
        print(f"training {name} ...", flush=True)
        enc, dec = train_variant(name, cross, beta, train_pairs)
        ztr = encode_all(enc, train_pairs[:6000])
        zte = encode_all(enc, test_pairs)
        r2_move = ridge_probe(ztr, tr_move, zte, te_move)
        r2_stat = ridge_probe(ztr, tr_stat, zte, te_stat)
        r2_dir = ridge_probe(ztr, tr_dir, zte, te_dir)
        mse_self = recon_mse(enc, dec, test_pairs, z_side=1)
        mse_cross = recon_mse(enc, dec, test_pairs, z_side=0)
        results[name] = dict(beta=beta, cross_shadow=cross,
                             r2_color_moving=round(r2_move, 4),
                             r2_color_static=round(r2_stat, 4),
                             r2_direction=round(r2_dir, 4),
                             mse_self=round(mse_self, 5),
                             mse_cross=round(mse_cross, 5),
                             cross_self_ratio=round(mse_cross / max(mse_self, 1e-9), 4))
        print(f"  [{name}] {results[name]}", flush=True)
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()},
                   f"shadow_sprites_{name}.pt")

    with open("shadow_sprites_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
