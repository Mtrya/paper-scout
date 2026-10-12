"""Experiment 3: does restoration (not off-policy data) keep forward-process RL alive?

Variants, all trained from the same pretrained roller with the same seeds:
  nft_onpolicy_1u    fresh rollouts, anchor = own rollout field, one update per rollout
  nft_onpolicy_2u    same data, two disjoint minibatch updates (restoration on the 2nd)
  nft_ema_1u         EMA rollout policy (eta=0.5), anchor = the lagged EMA field
  nft_ema_noanchor   EMA rollouts, but the loss target uses the live v_theta (no anchor)
  art_1u             aggregation-first covariance guidance, one update, no mask
  art_2u             paper Algorithm 1 (two updates + trust-region mask)
  art_2u_nomask      two updates without the mask

Instrumentation: reward, endpoint diversity, distance to the reward-optimal endpoint
distribution, the second update's guidance-vs-restoration parameter gradient norms.

Writes results/train.json.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

import toy

OUT = Path(__file__).resolve().parent

STEPS = 16
K_ENDPOINTS = 8
DRAWS = 4
ITERS = 120
EVAL_EVERY = 4
EVAL_ENDPOINTS = 48
LR = 2e-3

VARIANTS = {
    "nft_onpolicy_1u": dict(rollout="policy", anchor="old", updates=1, loss="nft", beta=1.0),
    "nft_onpolicy_2u": dict(rollout="policy", anchor="old", updates=2, loss="nft", beta=1.0),
    "nft_ema_1u": dict(rollout="ema", anchor="ema", updates=1, loss="nft", beta=1.0),
    "nft_ema_noanchor": dict(rollout="ema", anchor="live", updates=1, loss="nft", beta=1.0),
    "art_1u": dict(rollout="policy", anchor="old", updates=1, loss="art", mask=False),
    "art_2u": dict(rollout="policy", anchor="old", updates=2, loss="art", mask=True),
    "art_2u_nomask": dict(rollout="policy", anchor="old", updates=2, loss="art", mask=False),
}

EMA_ETA = 0.5


def rollout_batch(net, cfg, gen):
    """One iteration's rollout: endpoints, rewards, r weights, states and caches."""
    rows = {"c": [], "x0": [], "r": [], "t": [], "eps": [], "v0": []}
    guidance = []
    for c in range(toy.N_PROMPTS):
        endpoints = toy.euler_sample(net, c, K_ENDPOINTS, STEPS, gen)
        rewards = toy.reward(endpoints, c)
        goodness = toy.group_weights(rewards)
        guidance.append(toy.cov_guidance(endpoints, rewards))
        x0 = endpoints.repeat_interleave(DRAWS, 0)
        r = goodness.repeat_interleave(DRAWS, 0)
        eps = torch.randn(x0.shape[0], 2, generator=gen)
        t = torch.rand(x0.shape[0], generator=gen).clamp(1e-3, 1.0)
        xt = (1 - t)[:, None] * x0 + t[:, None] * eps
        rows["c"].append(toy.prompt_tensor(x0.shape[0], c))
        rows["x0"].append(x0)
        rows["r"].append(r)
        rows["t"].append(t)
        rows["eps"].append(eps)
    for key in ("c", "x0", "r", "t", "eps"):
        rows[key] = torch.cat(rows[key])
    rows["xt"] = (1 - rows["t"])[:, None] * rows["x0"] + rows["t"][:, None] * rows["eps"]
    rows["u"] = rows["x0"] - rows["eps"]
    rows["guidance"] = torch.stack(guidance)[rows["c"]]
    return rows


def trust_region_weight(rows, v_now, v0, guidance, delta=1e-5, noise_level=0.8):
    """paper Eq. 7/8 in the shipped form: mean shift of one reference SDE step, then
    m = ||d_mu||^2 / (2 D sigma_ref^2), masked when m >= delta and the move is outward."""
    s = rows["t"]
    step = 1.0 / STEPS
    initial = torch.isclose(s, torch.ones_like(s), atol=1e-5, rtol=0)
    sigma_ref_sq = torch.where(
        initial,
        torch.full_like(s, noise_level**2),
        (noise_level**2 * s / (1 - s).clamp_min(1e-3) * step).clamp_min(1e-12),
    )
    coefficient = torch.where(initial, torch.full_like(s, -step), -step * (1 + 0.5 * noise_level**2))
    d = v_now - v0
    delta_mu_sq = coefficient.square() * d.square().sum(-1)
    kl_proxy = delta_mu_sq / (2 * 2 * sigma_ref_sq)
    outward = (d * (guidance - d)).sum(-1) > 0
    mask = ((kl_proxy >= delta) & outward).to(d.dtype)
    return rows["t"] * (1 - mask), float(mask.mean()), float(kl_proxy.median())


def train_variant(name, cfg, seed=0, iters=ITERS, lr=LR):
    torch.manual_seed(seed)
    net = toy.pretrained(seed=0)
    if cfg["rollout"] == "ema":
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
    else:
        ema = None
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(9000 + seed)
    history, instrument = [], []
    for it in range(iters):
        roller = net if cfg["rollout"] == "policy" else ema
        rows = rollout_batch(roller, cfg, gen)
        chunks = torch.chunk(torch.arange(rows["r"].shape[0]), cfg["updates"])
        with torch.no_grad():
            v0 = roller(rows["xt"], rows["t"], rows["c"])
            if cfg["loss"] == "nft":
                v_tgt = v0 + ((2 * rows["r"] - 1) / cfg["beta"])[:, None] * (rows["u"] - v0)
            else:
                v_tgt = v0 + rows["guidance"]
            weights = rows["t"].clone()
        first_step_norm = None
        for index, chunk in enumerate(chunks):
            pred = net(rows["xt"][chunk], rows["t"][chunk], rows["c"][chunk])
            if cfg["loss"] == "nft":
                if cfg["anchor"] == "live":
                    # no old-policy anchor: target built from the live prediction itself
                    a = (2 * rows["r"][chunk] - 1)
                    loss = (a**2 * (rows["u"][chunk] - pred).square().sum(-1)).mean()
                else:
                    loss = cfg["beta"] ** 2 * (pred - v_tgt[chunk]).square().sum(-1).mean()
            else:
                mask_rate, kl_median = float("nan"), float("nan")
                if cfg["mask"]:
                    w, mask_rate, kl_median = trust_region_weight(
                        {k: rows[k][chunk] for k in ("t",)}, pred, v0[chunk], rows["guidance"][chunk]
                    )
                    w = torch.where(torch.isnan(w), torch.zeros_like(w), w)
                else:
                    w = rows["t"][chunk]
                loss = (w * (pred - v_tgt[chunk]).square().sum(-1)).sum()
            opt.zero_grad()
            loss.backward()
            if index == 1 and cfg["updates"] > 1:
                entry = measure_terms(net, rows, chunk, v0, v_tgt, cfg)
                entry["mask_rate"] = mask_rate
                entry["kl_median"] = kl_median
                instrument.append(entry)
            opt.step()
            if index == 0:
                first_step_norm = float(
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).detach()
                )
        if ema is not None:
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(EMA_ETA).add_(pn, alpha=1 - EMA_ETA)
        if it % EVAL_EVERY == 0 or it == iters - 1:
            history.append(evaluate(net, it, first_step_norm))
    return history, instrument


@torch.no_grad()
def evaluate(net, it, first_step_norm, seed=1234):
    gen = torch.Generator().manual_seed(seed)
    row = {"iter": it, "grad_norm": first_step_norm}
    rewards, hi_rates, spreads, energies = [], [], [], []
    for c in range(toy.N_PROMPTS):
        end = toy.euler_sample(net, c, EVAL_ENDPOINTS, STEPS, gen)
        ref = toy.target_reference(c, EVAL_ENDPOINTS, torch.Generator().manual_seed(77))
        m = toy.endpoint_metrics(end, c, ref)
        rewards.append(m["reward"])
        hi_rates.append(m["hi_mode_rate"])
        spreads.append(m["spread"])
        energies.append(m["energy_to_target"])
    row.update(
        {
            "reward": sum(rewards) / 2,
            "hi_rate": sum(hi_rates) / 2,
            "spread": sum(spreads) / 2,
            "energy_to_target": sum(energies) / 2,
        }
    )
    return row


def measure_terms(net, rows, chunk, v0, v_tgt, cfg):
    """Second update: output-space and parameter-space split of guidance vs restoration."""
    pred = net(rows["xt"][chunk], rows["t"][chunk], rows["c"][chunk])
    d = pred - v0[chunk]
    if cfg["loss"] == "nft":
        a = (2 * rows["r"][chunk] - 1)
        h = rows["u"][chunk] - v0[chunk]
        guide = (-2 * cfg["beta"] * a * (d * h).sum(-1)).sum()
        restore = (cfg["beta"] ** 2 * d.square().sum(-1)).sum()
        params = [p for p in net.parameters() if p.requires_grad]
        gg = torch.autograd.grad(guide, params, retain_graph=True, allow_unused=True)
        gr = torch.autograd.grad(restore, params, allow_unused=True)
        gnorm = float(sum((g**2).sum() for g in gg if g is not None).sqrt())
        rnorm = float(sum((g**2).sum() for g in gr if g is not None).sqrt())
        with torch.no_grad():
            return {
                "guide_output_force": float((2 * cfg["beta"] * a[:, None] * h).norm(dim=-1).mean()),
                "restore_output_force": float((2 * cfg["beta"] ** 2 * d).norm(dim=-1).mean()),
                "guide_param_grad": gnorm,
                "restore_param_grad": rnorm,
                "ratio_output": float((2 * cfg["beta"] ** 2 * d).norm(dim=-1).mean())
                / max(float((2 * cfg["beta"] * a[:, None] * h).norm(dim=-1).mean()), 1e-12),
                "ratio_param": rnorm / max(gnorm, 1e-12),
                "drift_norm": float(d.norm(dim=-1).mean()),
            }
    # ART: L = w ||v - (v0 + G)||^2 = w||G||^2 - 2w<G, d> + w||d||^2, i.e. the same
    # linear (guidance) / quadratic (restoration) split as NFT.
    g = rows["guidance"][chunk]
    t_w = rows["t"][chunk]
    guide = (-2 * t_w * (g * d).sum(-1)).sum()
    restore = (t_w * d.square().sum(-1)).sum()
    params = [p for p in net.parameters() if p.requires_grad]
    gg = torch.autograd.grad(guide, params, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(restore, params, allow_unused=True)
    gnorm = float(sum((x**2).sum() for x in gg if x is not None).sqrt())
    rnorm = float(sum((x**2).sum() for x in gr if x is not None).sqrt())
    with torch.no_grad():
        guide_out = float((t_w * g.norm(dim=-1)).mean())
        restore_out = float((t_w * d.norm(dim=-1)).mean())
        return {
            "guide_output_force": guide_out,
            "restore_output_force": restore_out,
            "guide_param_grad": gnorm,
            "restore_param_grad": rnorm,
            "ratio_output": restore_out / max(guide_out, 1e-12),
            "ratio_param": rnorm / max(gnorm, 1e-12),
            "drift_norm": float(d.norm(dim=-1).mean()),
        }


def main() -> None:
    import sys

    wanted = sys.argv[1:]
    out_path = OUT / "results" / "train.json"
    if wanted and out_path.exists():
        out = json.loads(out_path.read_text())
    else:
        out = {"config": dict(STEPS=STEPS, K_ENDPOINTS=K_ENDPOINTS, DRAWS=DRAWS, ITERS=ITERS,
                              LR=LR, EMA_ETA=EMA_ETA, EVAL_ENDPOINTS=EVAL_ENDPOINTS), "runs": {}}
    for name, cfg in VARIANTS.items():
        if wanted and name not in wanted:
            continue
        history, instrument = train_variant(name, cfg)
        tail = history[-6:]
        out["runs"][name] = {
            "cfg": cfg,
            "history": history,
            "instrument": instrument,
            "final_reward": sum(h["reward"] for h in tail) / len(tail),
            "final_hi_rate": sum(h["hi_rate"] for h in tail) / len(tail),
            "final_energy": sum(h["energy_to_target"] for h in tail) / len(tail),
            "final_spread": sum(h["spread"] for h in tail) / len(tail),
            "reward_max": max(h["reward"] for h in history),
            "reward_min": min(h["reward"] for h in history),
        }
        print(
            f"{name:18s} reward {history[0]['reward']:.3f} -> {out['runs'][name]['final_reward']:.3f} "
            f"(max {out['runs'][name]['reward_max']:.3f}) hi_rate {out['runs'][name]['final_hi_rate']:.3f} "
            f"energy {out['runs'][name]['final_energy']:.3f} spread {out['runs'][name]['final_spread']:.3f}"
        )
        if instrument:
            m = instrument[0]
            print(
                f"    2nd update: output-force guide={m['guide_output_force']:.4f} restore={m['restore_output_force']:.4f} "
                f"ratio={m['ratio_output']:.3f} | param-grad guide={m['guide_param_grad']:.3e} "
                f"restore={m['restore_param_grad']:.3e} ratio={m['ratio_param']:.3f}"
            )
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
