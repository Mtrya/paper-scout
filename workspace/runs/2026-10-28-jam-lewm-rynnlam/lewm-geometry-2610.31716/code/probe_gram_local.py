#!/usr/bin/env python3
"""Local Gram + planning probe on the official LeWM cube checkpoint (local anchors).

Protocol follows 2610.31716 (finite-scale Gram geometry) with our local anchor
episodes from a scripted cube-single expert. One model step = 5 raw env actions.
Also computes DA-LeWM-style Plan-Real Spearman per planning pair on the same
anchors and joins per-pair D_G with per-pair plan rho.

Run (from workspace/code):
  ./scout-gpu/bin/python probe_gram_local.py --anchors 100 --plan-pairs 30 \
      --data cube_anchors.pt --out gram_local_out
"""
import argparse, json, os
import numpy as np
import torch
from load_lewm import load_lewm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FS = 5
N_CAND = 64


def prep(pix):
    x = pix.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)


@torch.no_grad()
def encode_frame(model, pix, device="cuda"):
    x = prep(pix[None]).unsqueeze(0).to(device)
    return model.encode({"pixels": x})["emb"][0, -1].float().cpu().numpy()


@torch.no_grad()
def latent_after_block(model, ctx_pix, ctx_act, act_block, device="cuda"):
    pix = prep(ctx_pix).unsqueeze(0).to(device)
    act0 = torch.from_numpy(ctx_act).float().unsqueeze(0).to(device)
    out = model.encode({"pixels": pix, "action": act0})
    cur = out["emb"].clone()
    ab = torch.from_numpy(act_block).float().view(1, 1, -1).to(device)
    act_hist = torch.cat([act0, ab], dim=1)
    act_emb = model.action_encoder(act_hist[:, -3:])
    pred = model.predict(cur[:, -3:], act_emb)[:, -1:]
    return pred[0, 0].float().cpu().numpy()


@torch.no_grad()
def latent_terminal_batch(model, ctx_pix, ctx_act, act_blocks, device="cuda"):
    N = len(act_blocks)
    pix = prep(ctx_pix).unsqueeze(0).repeat(N, 1, 1, 1, 1).to(device)
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


class CubeReal:
    def __init__(self):
        import gymnasium as gym
        import ogbench  # noqa: F401
        self.env = gym.make("cube-single-v0", render_mode="rgb_array", width=224, height=224)
        self.env.reset(seed=12345)

    def endpoint(self, qpos0, qvel0, actions):
        u = self.env.unwrapped
        u.set_state(qpos0.copy(), qvel0.copy())
        for a in actions:
            self.env.step(np.asarray(a, dtype=np.float64))
        return self.env.render()

    def cost(self, qpos0, qvel0, actions, goal):
        u = self.env.unwrapped
        u.set_state(qpos0.copy(), qvel0.copy())
        for a in actions:
            self.env.step(np.asarray(a, dtype=np.float64))
        info = u.get_step_info()
        bp = info["privileged/block_0_pos"]
        by = float(info["privileged/block_0_yaw"][0])
        ee = info["proprio/effector_pos"]
        return (float(np.linalg.norm(bp - goal["block_pos"]))
                + 0.5 * float(np.linalg.norm(ee - goal["ee_pos"]))
                + 0.1 * abs((by - goal["block_yaw"] + np.pi) % (2 * np.pi) - np.pi))


def gram(F, hs):
    d = len(hs)
    J = np.stack([(F[i][0] - F[i][1]) / (2 * hs[i]) for i in range(d)], axis=1)
    return J.T @ J


def wape(G, Ffun, dirs, hs):
    num = den = 0.0
    for v in dirs:
        u = hs * v
        rhat = float(np.sqrt(max(0.0, u @ G @ u)))
        r = float(np.linalg.norm(Ffun(u)[0] - Ffun(u)[1]) / 2)
        num += abs(rhat - r)
        den += r
    return num / max(den, 1e-12)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=100)
    ap.add_argument("--plan-pairs", type=int, default=30)
    ap.add_argument("--data", default="cube_anchors.pt")
    ap.add_argument("--out", default="gram_local_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", default="0.001,0.01")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eps_data = torch.load(args.data, weights_only=False)
    rng = np.random.default_rng(args.seed + 100)
    # anchors: need t0-15 >= 0 and t0+35 <= L
    cand = []
    for ei, e in enumerate(eps_data):
        L = len(e["actions"])
        for t0 in range(16, L - 36):
            cand.append((ei, t0))
    rng.shuffle(cand)
    anchors = cand[:args.anchors]
    plan_anchors = cand[args.anchors:args.anchors + args.plan_pairs]
    all_anchors = anchors + plan_anchors

    def load(ei, t0):
        e = eps_data[ei]
        return dict(
            ctx=np.stack([e["states"][t0 - 10]["pixels"], e["states"][t0 - 5]["pixels"],
                          e["states"][t0]["pixels"]]),
            ctx_act=e["actions"][t0 - 15:t0].reshape(3, -1).astype(np.float32),
            block=e["actions"][t0:t0 + FS].astype(np.float32),
            gt30=e["actions"][t0:t0 + 30].astype(np.float32),
            qpos0=e["states"][t0]["qpos"].copy(), qvel0=e["states"][t0]["qvel"].copy(),
            goal=dict(block_pos=e["states"][t0 + 30]["info"]["privileged/block_0_pos"].copy(),
                      block_yaw=float(e["states"][t0 + 30]["info"]["privileged/block_0_yaw"][0]),
                      ee_pos=e["states"][t0 + 30]["info"]["proprio/effector_pos"].copy()),
            ei=ei, t0=t0)

    A = [load(ei, t0) for ei, t0 in all_anchors]
    sample_act = np.concatenate([e["actions"] for e in eps_data[:20]])
    Dstd = sample_act.std(axis=0) + 1e-6
    R = np.percentile(sample_act, 99, axis=0) - np.percentile(sample_act, 1, axis=0)
    env = CubeReal()
    dirs = [np.random.default_rng(20261021 + i).normal(size=(8, 5)) for i in range(len(A))]
    dirs = [d / np.linalg.norm(d, axis=1, keepdims=True) for d in dirs]
    epsilons = [float(x) for x in args.eps.split(",")]

    # planning candidates (model-independent real costs first)
    rngp = np.random.default_rng(args.seed)
    for ai, a in enumerate(A):
        if ai < len(anchors):
            continue
        gt = a["gt30"]
        cands = []
        for sig in [0.05, 0.1, 0.2, 0.4]:
            for _ in range(8):
                cands.append(gt + rngp.normal(0, sig, gt.shape).astype(np.float32) * Dstd)
        n_cross = 0
        while n_cross < 16:
            e2 = eps_data[rngp.integers(len(eps_data))]
            L2 = len(e2["actions"])
            t2 = int(rngp.integers(0, L2 - 31))
            cands.append(e2["actions"][t2:t2 + 30].astype(np.float32))
            n_cross += 1
        while len(cands) < N_CAND:
            cands.append((rngp.normal(0, 0.5, gt.shape) * Dstd).astype(np.float32))
        a["cands"] = cands[:N_CAND]
        a["c_real"] = np.array([env.cost(a["qpos0"], a["qvel0"], c, a["goal"]) for c in a["cands"]])
        print(f"real costs {ai - len(anchors)}/{args.plan_pairs}", flush=True)

    model = load_lewm()
    per = []
    for ai, a in enumerate(A):
        is_plan = ai >= len(anchors)
        rec = {"plan_pair": bool(is_plan)}
        base = a["block"]
        for eps in epsilons:
            delta = eps * R / 2.0
            hs = delta / Dstd
            Fm, Fe = {}, {}
            for i in range(5):
                off = np.zeros(5, dtype=np.float32); off[i] = delta[i]
                bm = np.stack([base + off, base - off])
                Fm[i] = [latent_after_block(model, a["ctx"], a["ctx_act"], b) for b in bm]
                Fe[i] = [encode_frame(model, env.endpoint(a["qpos0"], a["qvel0"], b)) for b in bm]
            Gm, Ge = gram(Fm, hs), gram(Fe, hs)
            tm, te = np.trace(Gm), np.trace(Ge)
            Gmn, Gen = Gm / max(tm, 1e-30), Ge / max(te, 1e-30)
            DG = float(np.linalg.norm(Gmn - Gen, "fro"))
            wm, Vm = np.linalg.eigh(Gmn)
            we, Ve = np.linalg.eigh(Gen)
            ang = float(np.degrees(np.arccos(np.clip(abs(Vm[:, -1] @ Ve[:, -1]), 0, 1))))
            gain = float(np.sqrt(tm / max(te, 1e-30)))

            def Fm_fun(u):
                out = []
                for sgn in (1, -1):
                    b = base + (sgn * u * Dstd).astype(np.float32)
                    out.append(latent_after_block(model, a["ctx"], a["ctx_act"], b))
                return out

            def Fe_fun(u):
                out = []
                for sgn in (1, -1):
                    b = base + (sgn * u * Dstd).astype(np.float32)
                    out.append(encode_frame(model, env.endpoint(a["qpos0"], a["qvel0"], b)))
                return out

            dv = dirs[ai]
            rec[f"eps{eps}"] = dict(D_G=DG, top1_angle=ang, gain=gain,
                                    wape_model=wape(Gm, Fm_fun, dv, hs),
                                    wape_env=wape(Ge, Fe_fun, dv, hs),
                                    wape_iso=wape(np.eye(5) * te / 5, Fe_fun, dv, hs),
                                    wape_oracle=wape(Gm * te / max(tm, 1e-30), Fm_fun, dv, hs),
                                    eigvec_model=Vm[:, -1].tolist(), eigvec_env=Ve[:, -1].tolist())
        if is_plan:
            z_g = encode_frame(model, eps_data[a["ei"]]["states"][a["t0"] + 30]["pixels"])
            blocks = [c.reshape(6, -1) for c in a["cands"]]
            z = np.concatenate([latent_terminal_batch(model, a["ctx"], a["ctx_act"],
                                                    np.stack(blocks[i:i + 32]))
                                for i in range(0, len(blocks), 32)])
            c_lat = ((z - z_g) ** 2).sum(axis=1)
            rec["plan_rho"] = spearman(c_lat, a["c_real"])
        per.append(rec)
        if ai % 10 == 0:
            print(f"anchor {ai}/{len(A)} D_G={rec[f'eps{epsilons[0]}']['D_G']:.3f}", flush=True)
        json.dump({"per_anchor": per}, open(f"{args.out}/gram_per_anchor.json", "w"), indent=1)

    res = {}
    for eps in epsilons:
        k = f"eps{eps}"
        for metric in ["D_G", "top1_angle", "gain", "wape_model", "wape_env", "wape_iso", "wape_oracle"]:
            vals = np.array([r[k][metric] for r in per])
            res[f"{k}_{metric}_median"] = float(np.median(vals))
            res[f"{k}_{metric}_iqr"] = [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]
        Vm = np.array([r[k]["eigvec_model"] for r in per])
        Ve = np.array([r[k]["eigvec_env"] for r in per])
        Cm = np.abs(Vm @ Vm.T); Ce = np.abs(Ve @ Ve.T)
        iu = np.triu_indices(len(per), 1)
        res[f"{k}_model_dir_consistency"] = float(Cm[iu].mean())
        res[f"{k}_env_dir_consistency"] = float(Ce[iu].mean())
    rhos = np.array([r["plan_rho"] for r in per if r["plan_pair"]])
    dgs = np.array([r[f"eps{epsilons[0]}"]["D_G"] for r in per if r["plan_pair"]])
    ok = ~np.isnan(rhos)
    res["plan_rho_mean"] = float(np.nanmean(rhos))
    if ok.sum() > 3:
        res["planlink_spearman_DG_vs_planrho"] = spearman(dgs[ok], rhos[ok])
    res["per_anchor_file"] = f"{args.out}/gram_per_anchor.json"
    json.dump(res, open(f"{args.out}/gram_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("iqr")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
