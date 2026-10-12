"""Experiment 1: Lemma 1 and Proposition 1 (hidden forward-reverse consistency force).

For a fixed state s_t = (x_t, t, c) we estimate the conditional law of
(r, u_t) under Q_old by Nadaraya-Watson weights on forward-noised rollout
endpoints, then

  * measure F_NFT = -(1/2b) grad_v L_NFT(s_t) by autodiff of the conditional loss,
  * measure the branch forces F_+ (r-weighted) and F_- ((1-r)-weighted),
  * compare against the closed forms 2g + (2q-1)e, g + q e, g - (1-q) e,
  * check Lemma 1 collinearity q(v_fwd^+ - v_fwd^old) = (1-q)(v_fwd^old - v_fwd^-) = g,
  * check the ART objective's residual force at the same states (zero by construction),
  * check cross-state aggregation: what survives when a full batch is summed.

Writes results/prop1.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import toy

OUT = Path(__file__).resolve().parent
T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
PATHS = ("hi", "lo", "mid")
ROLLOUT_STEPS = 16
N_ENDPOINTS = 100_000
DRAWS = 4
BW_X, BW_T = 0.45, 0.12


def probe_state(path: str, t: float, c: int) -> torch.Tensor:
    centers = toy.mode_centers(c)
    if path == "hi":
        return (1 - t) * centers[0]
    if path == "lo":
        return (1 - t) * centers[1]
    return torch.zeros(2)


def build_ensemble(net, c, n_end, draws, steps, seed):
    """Q_old for one prompt: endpoints from the roller, goodness weights r from the
    prompt group, then fresh-noise forward noising to (x_t, u_t)."""
    gen = torch.Generator().manual_seed(seed)
    endpoints = toy.euler_sample(net, c, n_end, steps, gen)
    goodness = toy.group_weights(toy.reward(endpoints, c))
    return toy.forward_ensemble(endpoints, goodness, c, draws, gen)


def conditional_rows(ens, c, v_old_of, beta=1.0, bw_scale=0.45):
    rows = []
    for t_c in T_GRID:
        bw = toy.network_bandwidth(t_c, bw_scale)
        for path in PATHS:
            xt_c = probe_state(path, t_c, c)
            w = toy.local_weights(ens, xt_c, t_c, bw, BW_T)
            st = toy.local_stats(ens, w)
            q, mean_u, g = st["q"], st["mean_u"], st["g"]
            v_old = v_old_of(xt_c, t_c)["v_rvs"].clone()
            e = mean_u - v_old
            f = toy.raw_forces(ens, w, v_old, beta)
            pred = 2 * g + (2 * q - 1) * e
            v_fwd_plus = (w * ens["r"])[:, None] * ens["u"]
            v_fwd_plus = v_fwd_plus.sum(0) / w.sum() / q.clamp_min(1e-8)
            one_minus_q = 1 - q
            v_fwd_minus = (w * (1 - ens["r"]))[:, None] * ens["u"]
            v_fwd_minus = v_fwd_minus.sum(0) / w.sum() / one_minus_q.clamp_min(1e-8)
            rows.append(
                {
                    "c": c,
                    "t": t_c,
                    "path": path,
                    "eff_samples": float(st["wsum"]) / ens["t"].shape[0],
                    "q": float(q),
                    "var_r": float(st["var_r"]),
                    "norm_g": float(g.norm()),
                    "norm_e": float(e.norm()),
                    "norm_mean_u": float(mean_u.norm()),
                    "residual": [(2 * float(q) - 1) * float(v) for v in e.tolist()],
                    "guidance": [2 * float(v) for v in g.tolist()],
                    "F": f["F"].tolist(),
                    "F_pred": pred.tolist(),
                    "F_err": float((f["F"] - pred).norm()),
                    "F_plus": f["F_plus"].tolist(),
                    "F_plus_pred": (g + q * e).tolist(),
                    "F_plus_err": float((f["F_plus"] - (g + q * e)).norm()),
                    "F_minus": f["F_minus"].tolist(),
                    "F_minus_pred": (g - (1 - q) * e).tolist(),
                    "F_minus_err": float((f["F_minus"] - (g - (1 - q) * e)).norm()),
                    "sum_branch_err": float((f["F_plus"] + f["F_minus"] - f["F"]).norm()),
                    "lemma1_pos_err": float((q * (v_fwd_plus - mean_u) - g).norm()),
                    "lemma1_neg_err": float(((1 - q) * (mean_u - v_fwd_minus) - g).norm()),
                    "residual_over_guidance": float(
                        ((2 * q - 1) * e).norm() / (2 * g).norm().clamp_min(1e-8)
                    ),
                    "cos_F_g": float(
                        torch.nn.functional.cosine_similarity(
                            f["F"].unsqueeze(0), (2 * g).unsqueeze(0)
                        ).item()
                    ),
                }
            )
    return rows


def art_residual_at_states(net, guidance):
    """ART force at v_theta = v_theta^(0): -(1/2) grad of w||v - (v0 + G)||^2 is pure guidance."""
    rows = []
    for c in range(toy.N_PROMPTS):
        for t_c in T_GRID:
            for path in PATHS:
                xt_c = probe_state(path, t_c, c)
                with torch.no_grad():
                    v0 = net(xt_c[None], torch.tensor([t_c]), toy.prompt_tensor(1, c))[0]
                target = v0 + guidance[c]
                vp = v0.clone().requires_grad_(True)
                loss = t_c * (vp - target).square().sum()
                grad = torch.autograd.grad(loss, vp)[0]
                force = -grad / 2
                rows.append(
                    {
                        "c": c,
                        "t": t_c,
                        "path": path,
                        "F_art": force.tolist(),
                        "F_art_norm": float(force.norm()),
                        "dead_on_arrival": float((force - t_c * guidance[c]).norm()),
                    }
                )
    return rows


def balanced_ensemble_check(beta=1.0):
    """Exactly balanced conditional ensemble: q = 1/2 statewise, e != 0 yet residual must vanish."""
    v_old = torch.tensor([0.4, -0.2])
    us = torch.tensor([[1.0, 0.5], [1.2, 0.4], [-0.3, -1.0], [-0.4, -1.2]])
    r = torch.tensor([1.0, 1.0, 0.0, 0.0])
    w = torch.ones(4)
    ens = {"u": us, "r": r, "t": torch.zeros(4), "xt": torch.zeros(4, 2)}
    st = toy.local_stats(ens, w)
    f = toy.raw_forces(ens, w, v_old, beta)
    q = float(st["q"])
    g = st["g"]
    e = st["mean_u"] - v_old
    return {
        "q": q,
        "norm_e": float(e.norm()),
        "norm_g": float(g.norm()),
        "F": f["F"].tolist(),
        "F_pred": (2 * g + (2 * q - 1) * e).tolist(),
        "residual_norm": float(((2 * q - 1) * e).norm()),
        "F_err": float((f["F"] - (2 * g + (2 * q - 1) * e)).norm()),
    }


def reward_scale_sweep(net, ens_by_c, gains=(0.2, 1.0, 5.0)):
    """Advantage gain = (reward spread) / (clipping denominator Z_c).

    r = 1/2 + clip((R - Rbar)/(Z_c), -1, 1)/2 with Z_c = 2 clip_max sigma_c, so a
    gain of f means the standardised advantage is amplified f times before
    clipping. Small gain -> r ~ 1/2 everywhere (weak g), large gain -> saturated
    r (extreme |2q-1|).
    """
    out = []
    for gain in gains:
        rows = []
        for c in range(toy.N_PROMPTS):
            ens = dict(ens_by_c[c])
            ens["r"] = 0.5 + ((ens["r"] - 0.5) * gain).clamp(-0.5, 0.5)
            for t_c in T_GRID:
                bw = toy.network_bandwidth(t_c, BW_X)
                for path in PATHS:
                    xt_c = probe_state(path, t_c, c)
                    w = toy.local_weights(ens, xt_c, t_c, bw, BW_T)
                    st = toy.local_stats(ens, w)
                    v_old = net(xt_c[None], torch.tensor([t_c]), toy.prompt_tensor(1, c))[0]
                    e = st["mean_u"] - v_old
                    rows.append(
                        {
                            "t": t_c,
                            "path": path,
                            "q": float(st["q"]),
                            "abs2q1": abs(2 * float(st["q"]) - 1),
                            "norm_g": float(st["g"].norm()),
                            "ratio": float(
                                ((2 * st["q"] - 1) * e).norm() / (2 * st["g"]).norm().clamp_min(1e-8)
                            ),
                            "guidance": (2 * st["g"]).tolist(),
                            "residual": ((2 * st["q"] - 1) * e).tolist(),
                        }
                    )
        out.append(
            {
                "gain": gain,
                "mean_abs2q1": sum(r["abs2q1"] for r in rows) / len(rows),
                "mean_norm_g": sum(r["norm_g"] for r in rows) / len(rows),
                "mean_ratio": sum(r["ratio"] for r in rows) / len(rows),
                "aggregate_ratio": _aggregate_ratio(rows),
                "rows": rows,
            }
        )
    return out


def _aggregate_ratio(rows: list[dict]) -> float:
    """||mean residual|| / ||mean guidance|| over the probed states."""
    gx = sum(r["guidance"][0] for r in rows) / len(rows)
    gy = sum(r["guidance"][1] for r in rows) / len(rows)
    rx = sum(r["residual"][0] for r in rows) / len(rows)
    ry = sum(r["residual"][1] for r in rows) / len(rows)
    return ((rx * rx + ry * ry) ** 0.5) / max((gx * gx + gy * gy) ** 0.5, 1e-12)


def aggregation_analysis(net, ens_by_c, n_states=48, seed=3):
    """What survives aggregation over a realistic rollout batch (K=8 endpoints/prompt)?"""
    gen = torch.Generator().manual_seed(seed)
    guidance_norms, residual_norms = [], []
    guidance_sum = torch.zeros(2)
    residual_sum = torch.zeros(2)
    q_values = []
    for _ in range(n_states):
        c = int(torch.randint(0, 2, (1,), generator=gen))
        t_c = float(torch.rand(1, generator=gen).clamp(1e-2, 0.98))
        xt_c = torch.randn(2, generator=gen) * 0.9
        bw = toy.network_bandwidth(t_c, BW_X)
        w = toy.local_weights(ens_by_c[c], xt_c, t_c, bw, BW_T)
        st = toy.local_stats(ens_by_c[c], w)
        v_old = net(xt_c[None], torch.tensor([t_c]), toy.prompt_tensor(1, c))[0]
        e = st["mean_u"] - v_old
        q = st["q"]
        guidance_sum = guidance_sum + 2 * st["g"]
        residual_sum = residual_sum + (2 * q - 1) * e
        guidance_norms.append(float((2 * st["g"]).norm()))
        residual_norms.append(float(((2 * q - 1) * e).norm()))
        q_values.append(float(q))
    n = float(n_states)
    return {
        "n_states": n_states,
        "norm_mean_guidance": float(guidance_sum.norm()) / n,
        "mean_norm_guidance": sum(guidance_norms) / n,
        "coherence_guidance": float(guidance_sum.norm()) / max(sum(guidance_norms), 1e-9),
        "norm_mean_residual": float(residual_sum.norm()) / n,
        "mean_norm_residual": sum(residual_norms) / n,
        "coherence_residual": float(residual_sum.norm()) / max(sum(residual_norms), 1e-9),
        "residual_to_guidance_aggregate": float(
            residual_sum.norm() / guidance_sum.norm().clamp_min(1e-9)
        ),
        "mean_abs2q1": sum(abs(2 * q - 1) for q in q_values) / n,
        "frac_abs2q1_gt_half": sum(abs(2 * q - 1) > 0.5 for q in q_values) / n,
    }


def solver_gap_sweep(net, steps_list=(4, 8, 16, 32, 64), n_end=50_000, seed=5):
    out = []
    for steps in steps_list:
        gen = torch.Generator().manual_seed(seed)
        rows = []
        for c in range(toy.N_PROMPTS):
            endpoints = toy.euler_sample(net, c, n_end, steps, gen)
            goodness = toy.group_weights(toy.reward(endpoints, c))
            ens = toy.forward_ensemble(endpoints, goodness, c, DRAWS, gen)
            for t_c in T_GRID:
                bw = toy.network_bandwidth(t_c, BW_X)
                for path in PATHS:
                    xt_c = probe_state(path, t_c, c)
                    w = toy.local_weights(ens, xt_c, t_c, bw, BW_T)
                    st = toy.local_stats(ens, w)
                    v_old = net(xt_c[None], torch.tensor([t_c]), toy.prompt_tensor(1, c))[0]
                    e = st["mean_u"] - v_old
                    rows.append(
                        {
                            "t": t_c,
                            "path": path,
                            "norm_e": float(e.norm()),
                            "ratio": float(e.norm() / st["mean_u"].norm().clamp_min(1e-8)),
                            "guidance": (2 * st["g"]).tolist(),
                            "residual": ((2 * st["q"] - 1) * e).tolist(),
                        }
                    )
        out.append(
            {
                "steps": steps,
                "mean_norm_e": sum(r["norm_e"] for r in rows) / len(rows),
                "mean_rel_e": sum(r["ratio"] for r in rows) / len(rows),
                "aggregate_ratio": _aggregate_ratio(rows),
                "rows": rows,
            }
        )
    return out


def main() -> None:
    torch.manual_seed(0)
    net = toy.pretrained(seed=0)
    guidance = []
    for c in range(toy.N_PROMPTS):
        gen = torch.Generator().manual_seed(40 + c)
        pts = toy.euler_sample(net, c, 8, ROLLOUT_STEPS, gen)
        guidance.append(toy.cov_guidance(pts, toy.reward(pts, c)))

    ens_by_c = [
        build_ensemble(net, c, N_ENDPOINTS, DRAWS, ROLLOUT_STEPS, 100 + c)
        for c in range(toy.N_PROMPTS)
    ]

    def v_old_of_factory(c):
        def v_old_of(xt_c, t_c):
            with torch.no_grad():
                v = net(xt_c[None], torch.tensor([t_c]), toy.prompt_tensor(1, c))[0]
            return {"v_rvs": v}

        return v_old_of

    rows = []
    for c in range(toy.N_PROMPTS):
        rows.extend(conditional_rows(ens_by_c[c], c, v_old_of_factory(c)))

    out = {
        "config": {
            "rollout_steps": ROLLOUT_STEPS,
            "n_endpoints": N_ENDPOINTS,
            "draws": DRAWS,
            "bw_x_scale": BW_X,
            "bw_t": BW_T,
            "beta": 1.0,
            "t_grid": list(T_GRID),
            "paths": list(PATHS),
        },
        "guidance_G": [g.tolist() for g in guidance],
        "guidance_norm": [float(g.norm()) for g in guidance],
        "states": rows,
        "art": art_residual_at_states(net, guidance),
        "balanced_ensemble": balanced_ensemble_check(),
        "reward_scale": reward_scale_sweep(net, ens_by_c),
        "aggregation": aggregation_analysis(net, ens_by_c),
        "solver_gap": solver_gap_sweep(net),
    }
    (OUT / "results" / "prop1.json").write_text(json.dumps(out, indent=2))

    print("=== Proposition 1: measured vs closed-form force ===")
    errs = [r["F_err"] for r in rows]
    norms = [float(torch.tensor(r["F"]).norm()) for r in rows]
    print(f"states={len(rows)} max|F - (2g+(2q-1)e)|={max(errs):.2e}  mean={sum(errs)/len(errs):.2e}")
    print(f"mean |F|={sum(norms)/len(norms):.3f}")
    print(f"max branch err={max(r['F_plus_err'] for r in rows):.2e} / {max(r['F_minus_err'] for r in rows):.2e}")
    print(f"max |F+ + F- - F|={max(r['sum_branch_err'] for r in rows):.2e}")
    print(f"max Lemma-1 err={max(max(r['lemma1_pos_err'], r['lemma1_neg_err']) for r in rows):.2e}")
    print("\nstate              q     2q-1   |2g|   (2q-1)|e|  |e|    ratio  cos(F,2g)")
    for r in rows:
        res = float(torch.tensor(r["residual"]).norm())
        print(
            f"  c{r['c']} {r['path']:>3} t={r['t']:.1f}  {r['q']:.3f} {2*r['q']-1:+.3f} "
            f"{r['norm_g']*2:.3f}  {res:.3f}     {r['norm_e']:.3f}  {r['residual_over_guidance']:.3f}  {r['cos_F_g']:+.3f}"
        )
    print("\n=== aggregation over 48 batch-like states ===")
    agg = out["aggregation"]
    for k, v in agg.items():
        if k != "per_sample":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== balanced ensemble (q=1/2 statewise) ===")
    print(out["balanced_ensemble"])
    print("\n=== reward scale sweep ===")
    for row in out["reward_scale"]:
        print(
            f"  gain={row['gain']:.1f}: mean|2q-1|={row['mean_abs2q1']:.3f} "
            f"mean|g|={row['mean_norm_g']:.3f} mean per-state ratio={row['mean_ratio']:.3f} "
            f"aggregate ratio={row['aggregate_ratio']:.4f}"
        )
    print("\n=== solver gap ===")
    for row in out["solver_gap"]:
        print(
            f"  steps={row['steps']:3d}: mean|e|={row['mean_norm_e']:.4f} mean rel={row['mean_rel_e']:.4f} "
            f"aggregate residual/guidance={row['aggregate_ratio']:.4f}"
        )
    print("\n=== ART force at the same states ===")
    print(f"  max |F_art - t G| = {max(r['dead_on_arrival'] for r in out['art']):.2e}")


if __name__ == "__main__":
    main()
