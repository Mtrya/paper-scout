#!/usr/bin/env python3
"""Decision-metric alignment probe on our own LeWM vs PSG-JEPA checkpoints.

Implements DA-LeWM (2608.18746) diagnostics on OGBench cube-single:
  1. Plan-Real Spearman (random-mixture candidates; paper's Cube caveat worked
     around by using noise-perturbed demo segments + cross-episode segments +
     Gaussian noise instead of pure randoms -> graded, contact-rich real costs)
  2. CEM-stage Spearman (iter 0 / 15 / elite snapshot of a 30-iter CEM run)
  3. Soft-margin rate p + Kendall tau_a (paper's consistency check)
  4. Claim-1 signature: corr(||dz||, ||da||) on held-out transitions
  5. Latent-cost range ratio (collapse surrogate)

Hypothesis under test: does the PSG grounding head (which repaired per-dim
identifiability in our 2026-08-17 probe) also repair decision-metric
alignment? DA-LeWM's framework predicts NO: state-prediction heads improve
probes; inverse/goal-action heads improve planning geometry.

Run on the Inspire notebook:
  cd $W/repo && PYTHONPATH=. ../.venv/bin/python ../probe_align.py \
    --ckpt ../out_baseline/psgjepa_epoch_10_object.ckpt --name lewm \
    --ckpt ../out_psg/psgjepa_epoch_10_object.ckpt --name psg \
    --h5 ../data/extract/cube_single_expert.h5 --out ../align_out
"""
import argparse, json, os
import numpy as np
import torch
import h5py
import hdf5plugin  # noqa: F401  registers compression filters for pixels

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAMESKIP = 5
H_MODEL = 6                      # model steps per plan (30 raw env steps)
DELTA = H_MODEL * FRAMESKIP
N_CAND = 64
CEM_ITERS = 30
CEM_ELITE = 8

def prep_pixels(pix_u8):
    x = pix_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)

@torch.no_grad()
def encode_frame(model, pix_u8, device="cuda"):
    """Single frame (H,W,3) uint8 -> (D,) latent."""
    x = prep_pixels(pix_u8[None]).unsqueeze(0).to(device)  # (1, T=1, C, H, W)
    return model.encode({"pixels": x})["emb"][0, -1].float().cpu().numpy()

@torch.no_grad()
def latent_terminal_batch(model, ctx_pix, ctx_act, act_blocks, device="cuda"):
    """ctx_pix (3,H,W,3) uint8, ctx_act (3,25), act_blocks (N,H,25).
    Broadcasts context over the candidate batch. Returns (N,D) terminal latents."""
    N = len(act_blocks)
    pix = prep_pixels(ctx_pix).unsqueeze(0).repeat(N, 1, 1, 1, 1).to(device)
    act0 = torch.from_numpy(ctx_act).float().unsqueeze(0).repeat(N, 1, 1).to(device)
    out = model.encode({"pixels": pix, "action": act0})
    cur = out["emb"].clone()
    act_hist = act0
    ab = torch.from_numpy(act_blocks).float().to(device)
    for k in range(ab.shape[1]):
        act_hist = torch.cat([act_hist, ab[:, k:k + 1]], dim=1)
        act_emb = model.action_encoder(act_hist[:, -3:])
        pred = model.predict(cur[:, -3:], act_emb)[:, -1:]
        cur = torch.cat([cur, pred], dim=1)
    return cur[:, -1].float().cpu().numpy()

def latent_costs(model, p, blocks, z_g, bs=64):
    """blocks: list of (H,25). Batched latent rollout -> (N,) costs."""
    arr = np.stack(blocks)
    outs = []
    for i in range(0, len(arr), bs):
        outs.append(latent_terminal_batch(model, p["ctx"], p["ctx_act"], arr[i:i + bs]))
    z = np.concatenate(outs)
    return ((z - z_g) ** 2).sum(axis=1)

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    def ranks(v):
        r = np.argsort(np.argsort(v)).astype(float)
        u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        c = np.zeros(len(u)); np.add.at(c, inv, r)
        return (c / cnt)[inv]
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])

def kendall_tau_a(a, b):
    a, b = np.asarray(a), np.asarray(b)
    conc = disc = 0
    for i in range(len(a)):
        d = a[i] - a[i + 1:]; e = b[i] - b[i + 1:]
        s = np.sign(d) * np.sign(e)
        conc += int((s > 0).sum()); disc += int((s < 0).sum())
    return (conc - disc) / max(1, conc + disc)

class CubeReal:
    """OGBench cube-single with exact state reset; real cost = task-state distance."""
    def __init__(self):
        import gymnasium as gym
        import ogbench  # noqa: F401
        self.env = gym.make("cube-single-v0", render_mode=None)
        self.env.reset()

    def terminal_cost(self, qpos0, qvel0, actions, goal):
        u = self.env.unwrapped
        self.env.reset()
        u.set_state(qpos0.copy(), qvel0.copy())
        for a in actions:
            self.env.step(np.asarray(a, dtype=np.float64))
        info = u.get_step_info()
        bp = info["privileged/block_0_pos"]; by = float(info["privileged/block_0_yaw"][0])
        ee = info["proprio/effector_pos"]
        return (float(np.linalg.norm(bp - goal["block_pos"]))
                + 0.5 * float(np.linalg.norm(ee - goal["ee_pos"]))
                + 0.1 * abs((by - goal["block_yaw"] + np.pi) % (2 * np.pi) - np.pi))

def cem_latent(model, p, act_dim_blocks, z_g, rng):
    """Minimal CEM in latent space. Returns snapshots: iter0 pop, iter15 pop, final elites."""
    mu = np.zeros((H_MODEL, act_dim_blocks), dtype=np.float32)
    sigma = np.ones((H_MODEL, act_dim_blocks), dtype=np.float32) * 0.5
    snaps = {}
    for it in range(CEM_ITERS + 1):
        pop = mu + sigma * rng.normal(size=(N_CAND, H_MODEL, act_dim_blocks)).astype(np.float32)
        costs = latent_costs(model, p, list(pop), z_g)
        if it == 0:
            snaps["random"] = pop.copy()
        if it == 15:
            snaps["mid"] = pop.copy()
        elite = pop[np.argsort(costs)[:CEM_ELITE]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-4
    snaps["elite"] = elite.copy()
    return snaps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", default="align_out")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cem", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    h = h5py.File(args.h5, "r")
    ep_ids = h["id"][:]
    bounds, cur = [], 0
    for i in range(1, len(ep_ids) + 1):
        if i == len(ep_ids) or ep_ids[i] != ep_ids[cur]:
            bounds.append((cur, i)); cur = i
    print("episodes:", len(bounds), flush=True)

    pairs, tries = [], 0
    while len(pairs) < args.pairs and tries < 20000:
        tries += 1
        s, e = bounds[rng.integers(len(bounds))]
        L = (e - s) // FRAMESKIP
        if L < 10 + H_MODEL + 2:
            continue
        t0 = s + int(rng.integers(L // 5, max(L // 5 + 1, L - H_MODEL - 2))) * FRAMESKIP
        if t0 + DELTA >= e:
            continue
        pairs.append((s, e, t0))
    print("pairs:", len(pairs), flush=True)

    P = []
    for s, e, t0 in pairs:
        ctx_idx = np.array([t0 - 2 * FRAMESKIP, t0 - FRAMESKIP, t0])
        ctx_act = h["action"][t0 - 3 * FRAMESKIP:t0].reshape(3, -1).astype(np.float32)
        P.append(dict(
            ctx=h["pixels"][ctx_idx],
            ctx_act=ctx_act,
            goal_pix=h["pixels"][t0 + DELTA],
            gt_act=h["action"][t0:t0 + DELTA].astype(np.float32),
            qpos0=h["qpos"][t0].copy(), qvel0=h["qvel"][t0].copy(),
            goal=dict(block_pos=h["privileged_block_0_pos"][t0 + DELTA].copy(),
                      block_yaw=float(h["privileged_block_0_yaw"][t0 + DELTA][0]),
                      ee_pos=h["proprio_effector_pos"][t0 + DELTA].copy()),
        ))
    print("pair data extracted", flush=True)

    act_std = np.concatenate([p["gt_act"] for p in P[:10]]).std(axis=0) + 1e-6

    def gen_candidates(p):
        gt = p["gt_act"]
        cands = []
        for sig in [0.05, 0.1, 0.2, 0.4]:
            for _ in range(8):
                cands.append(gt + rng.normal(0, sig, gt.shape).astype(np.float32) * act_std)
        n_cross = 0
        while n_cross < 16:
            s2, e2 = bounds[rng.integers(len(bounds))]
            L2 = e2 - s2
            if L2 > DELTA + FRAMESKIP:
                t2 = s2 + int(rng.integers(0, (L2 - DELTA) // FRAMESKIP)) * FRAMESKIP
                cands.append(h["action"][t2:t2 + DELTA].astype(np.float32))
                n_cross += 1
        while len(cands) < N_CAND:
            cands.append((rng.normal(0, 0.5, gt.shape) * act_std).astype(np.float32))
        return cands[:N_CAND]

    # candidates + real costs are model-independent: compute ONCE
    env = CubeReal()
    for pi, p in enumerate(P):
        p["cands"] = gen_candidates(p)
        p["c_real"] = np.array([env.terminal_cost(p["qpos0"], p["qvel0"], c, p["goal"])
                                for c in p["cands"]])
        if pi % 5 == 0:
            print(f"real costs {pi}/{len(P)} done", flush=True)
    json.dump({"c_real": [p["c_real"].tolist() for p in P]},
              open(f"{args.out}/real_costs.json", "w"))
    print("real costs done", flush=True)

    results = {}
    for name, ck in zip(args.name, args.ckpt):
        print("==", name, flush=True)
        model = torch.load(ck, map_location="cpu", weights_only=False).cuda().eval()
        rows = []
        for pi, p in enumerate(P):
            z_g = encode_frame(model, p["goal_pix"])
            blocks = [c.reshape(H_MODEL, -1) for c in p["cands"]]
            c_lat = latent_costs(model, p, blocks, z_g)
            c_real = p["c_real"]
            rho = spearman(c_lat, c_real)
            tau = kendall_tau_a(c_lat, c_real)
            kappa = float(np.dot(c_lat, c_real) / max(1e-12, np.dot(c_real, c_real)))
            eta = np.abs(c_lat - kappa * c_real)
            n_soft = n_tot = 0
            for i in range(len(c_real)):
                for j in range(i + 1, len(c_real)):
                    d = abs(c_real[i] - c_real[j])
                    if d < 1e-9:
                        continue
                    n_tot += 1
                    if kappa * d > eta[i] + eta[j]:
                        n_soft += 1
            row = dict(rho=rho, tau_a=float(tau), soft_p=n_soft / max(1, n_tot),
                       lat_range=float(c_lat.max() / max(1e-12, c_lat.min())))
            if not args.skip_cem:
                snaps = cem_latent(model, p, blocks[0].shape[1], z_g, rng)
                for stage, pop in snaps.items():
                    raw = pop.reshape(len(pop), -1, 5)  # (N,30,5) raw env actions
                    real_snap = np.array([env.terminal_cost(p["qpos0"], p["qvel0"], c, p["goal"])
                                          for c in raw])
                    lat_snap = latent_costs(model, p, list(pop), z_g)
                    row[f"cem_{stage}"] = spearman(lat_snap, real_snap)
            rows.append(row)
            print(f"  pair {pi}: rho={rho:.3f} tau={row['tau_a']:.3f} p={row['soft_p']:.3f}"
                  + (f" cem=({row.get('cem_random'):.3f},{row.get('cem_mid'):.3f},{row.get('cem_elite'):.3f})"
                     if not args.skip_cem else ""), flush=True)
        defined = [r["rho"] for r in rows if not np.isnan(r["rho"])]
        res = dict(plan_real_spearman=float(np.mean(defined)), n_defined=len(defined),
                   tau_a=float(np.mean([r["tau_a"] for r in rows])),
                   soft_p=float(np.mean([r["soft_p"] for r in rows])),
                   lat_range=float(np.mean([r["lat_range"] for r in rows])),
                   per_pair=rows)
        if not args.skip_cem:
            for stage in ["random", "mid", "elite"]:
                vals = [r[f"cem_{stage}"] for r in rows if not np.isnan(r.get(f"cem_{stage}", float("nan")))]
                res[f"cem_{stage}"] = float(np.mean(vals)) if vals else None
        # Claim-1 signature: corr(||dz||, ||da||) on held-out consecutive transitions
        dzs, das = [], []
        trans_idx = []
        tries = 0
        while len(trans_idx) < 200 and tries < 20000:
            tries += 1
            s, e = bounds[rng.integers(len(bounds))]
            if e - s < 2 * FRAMESKIP:
                continue
            t = s + int(rng.integers(1, (e - s) // FRAMESKIP - 1)) * FRAMESKIP
            trans_idx.append(t)
        for t in trans_idx:
            z0 = encode_frame(model, h["pixels"][t - FRAMESKIP])
            z1 = encode_frame(model, h["pixels"][t])
            dzs.append(float(np.linalg.norm(z1 - z0)))
            das.append(float(np.linalg.norm(h["action"][t - FRAMESKIP:t].ravel())))
        res["claim1_corr_dz_da"] = float(np.corrcoef(dzs, das)[0, 1])
        print(name, "Claim-1 corr(||dz||,||da||):", res["claim1_corr_dz_da"], flush=True)
        results[name] = res
        print(name, "Plan-Real:", res["plan_real_spearman"], f"({res['n_defined']}/{len(rows)})",
              flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(results, open(f"{args.out}/align_results.json", "w"), indent=1)
    print("saved", args.out, flush=True)

if __name__ == "__main__":
    main()
