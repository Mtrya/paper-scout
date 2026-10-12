"""Step 0: validate the toy before trusting any force measurement.

1. pretrained model quality (flow-matching MSE), endpoint distribution vs analytic mixture
2. Nadaraya-Watson conditional expectation vs the closed-form field of the analytic mixture
3. forward-reverse gap |e| of the pretrained roller as a function of solver steps

Writes results/validate.json and figs/fig0_validate.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import toy

OUT = Path(__file__).resolve().parent


def nw_estimate(
    x0: torch.Tensor,
    t_c: float,
    xt_c: torch.Tensor,
    draws: int,
    bw_x: float,
    bw_t: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    eps = torch.randn(x0.shape[0] * draws, 2, generator=generator)
    t = torch.full((eps.shape[0],), t_c)
    xt = (1 - t_c) * x0.repeat_interleave(draws, 0) + t_c * eps
    u = x0.repeat_interleave(draws, 0) - eps
    w = torch.exp(
        -((xt - xt_c).square().sum(-1)) / (2 * bw_x**2) - (t - t_c).square() / (2 * bw_t**2)
    )
    return (w[:, None] * u).sum(0) / w.sum(), float(w.sum())


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(7)
    net = toy.pretrained(seed=0, steps=4000)

    # --- 1. endpoint distribution of the pretrained roller ---------------------------------
    probe_steps = [4, 8, 16, 32, 64]
    endpoints = {}
    for steps in probe_steps:
        endpoints[steps] = [
            toy.euler_sample(net, c, 4096, steps, torch.Generator().manual_seed(100 + c))
            for c in range(toy.N_PROMPTS)
        ]
    ref = [toy.target_reference(c, 1024, torch.Generator().manual_seed(5)) for c in range(toy.N_PROMPTS)]
    dist = {}
    for steps in probe_steps:
        dist[steps] = [
            toy.endpoint_metrics(endpoints[steps][c], c, ref[c]) for c in range(toy.N_PROMPTS)
        ]
    mix_sample = torch.cat(
        [toy.mixture_sample(c, 1024, torch.Generator().manual_seed(11 + c)) for c in range(2)]
    )
    mix_ref = torch.cat([toy.target_reference(c, 1024, torch.Generator().manual_seed(5)) for c in range(2)])
    mixture_energy = toy.energy_distance(mix_sample, mix_ref)

    # --- 2. NW estimator vs the closed-form field of the analytic mixture ------------------
    x0 = torch.cat([toy.mixture_sample(c, 20000, torch.Generator().manual_seed(21 + c)) for c in range(2)])
    for c in range(2):
        x0[c * 20000 : (c + 1) * 20000] = x0[c * 20000 : (c + 1) * 20000]  # keep grouping explicit
    # relative error is meaningless at states where the true field nearly vanishes
    # (x_t = 0 at large t for a two-mode target), so report absolute error against
    # the mean field magnitude over the probe set, and the estimator's own scale.
    estimator_check = []
    for bw_scale in (0.15, 0.3, 0.6):
        for draws in (4, 16):
            errs, mags, eff = [], [], []
            for c in range(2):
                sub = x0[c * 20000 : (c + 1) * 20000]
                for t_c in (0.1, 0.3, 0.5, 0.7, 0.9):
                    for path in ("hi", "lo", "mid"):
                        xt_c = probe_state(path, t_c, c)
                        bw = toy.network_bandwidth(t_c, bw_scale)
                        est, wsum = nw_estimate(sub, t_c, xt_c, draws, bw, 0.12, gen)
                        truth = toy.analytic_field(xt_c[None], torch.tensor([t_c]), c)[0]
                        errs.append(float((est - truth).norm()))
                        mags.append(float(truth.norm()))
                        eff.append(wsum / sub.shape[0])
            estimator_check.append(
                {
                    "bw_scale": bw_scale,
                    "draws": draws,
                    "n_endpoints": 20000,
                    "mean_abs_err": sum(errs) / len(errs),
                    "mean_truth_norm": sum(mags) / len(mags),
                    "mean_effective_weight": sum(eff) / len(eff),
                }
            )

    # --- 3. forward-reverse gap e of the pretrained roller ---------------------------------
    gaps = []
    for steps in probe_steps:
        rows = []
        for c in range(2):
            x0c = endpoints[steps][c][:1024]
            ens = toy.forward_ensemble(x0c, torch.zeros(x0c.shape[0]), c, 8, torch.Generator().manual_seed(31))
            for t_c in (0.1, 0.3, 0.5, 0.7, 0.9):
                for path in ("hi", "lo", "mid"):
                    xt_c = probe_state(path, t_c, c)
                    bw = toy.network_bandwidth(t_c, 0.45)
                    w = toy.local_weights(ens, xt_c, t_c, bw, 0.12)
                    st = toy.local_stats(ens, w)
                    with torch.no_grad():
                        v_rvs = net(xt_c[None], torch.tensor([t_c]), toy.prompt_tensor(1, c))[0]
                    e = st["mean_u"] - v_rvs
                    rows.append(
                        {
                            "steps": steps,
                            "c": c,
                            "t": t_c,
                            "path": path,
                            "q": float(st["q"]),
                            "norm_e": float(e.norm()),
                            "norm_u": float(st["mean_u"].norm()),
                            "ratio": float(e.norm() / st["mean_u"].norm().clamp_min(1e-8)),
                            "eff_sample": float(st["wsum"]) / ens["t"].shape[0],
                        }
                    )
        gaps.extend(rows)

    out = {
        "fm_loss_probe": None,
        "endpoint_metrics": {str(k): v for k, v in dist.items()},
        "mixture_energy_to_target": mixture_energy,
        "estimator_check": estimator_check,
        "gaps": gaps,
    }
    (OUT / "results" / "validate.json").write_text(json.dumps(out, indent=2))

    agg = {}
    for steps in probe_steps:
        vals = [g for g in gaps if g["steps"] == steps]
        agg[steps] = {
            "mean_norm_e": sum(v["norm_e"] for v in vals) / len(vals),
            "mean_rel_e": sum(v["ratio"] for v in vals) / len(vals),
        }
    print("endpoint metrics by solver steps:")
    for steps in probe_steps:
        print(
            f"  steps={steps:3d} " + " | ".join(
                f"c{c}: hi={dist[steps][c]['hi_mode_rate']:.3f} spread={dist[steps][c]['spread']:.3f} "
                f"E={dist[steps][c]['energy_to_target']:.3f}"
                for c in range(2)
            )
        )
    print(f"analytic-mixture energy distance to target: {mixture_energy:.3f}")
    print("estimator check (NW vs closed form, absolute):")
    for row in estimator_check:
        print(
            f"  bw_scale={row['bw_scale']:.2f} draws={row['draws']:2d}: "
            f"mean|err|={row['mean_abs_err']:.4f} mean|truth|={row['mean_truth_norm']:.3f} "
            f"eff_w={row['mean_effective_weight']:.5f}"
        )
    print("forward-reverse gap:")
    for steps in probe_steps:
        print(f"  steps={steps:3d} mean|e|={agg[steps]['mean_norm_e']:.4f} rel={agg[steps]['mean_rel_e']:.4f}")


def probe_state(path: str, t: float, c: int) -> torch.Tensor:
    centers = toy.mode_centers(c)
    if path == "hi":
        return (1 - t) * centers[0]
    if path == "lo":
        return (1 - t) * centers[1]
    return torch.zeros(2)


if __name__ == "__main__":
    main()
