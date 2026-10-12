"""Experiment 2: Proposition 2 algebra (restoration), and what beta actually controls.

Checks, all on fresh rollout batches of the toy roller:
  A. the three equivalent forms of L_NFT (branch form / frozen-target form / drift form)
  B. the output-space gradient identity -grad L = 2 b E[(2r-1)h] - 2 b^2 E[d]
  C. the restoring gradient is exactly 2 b^2 d, and vanishes at d = 0
  D. what beta does: single on-policy update under Adam is beta-blind; restoration
     only enters when the anchor lags (EMA / second update on the same rollout)

Writes results/prop2.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import toy

OUT = Path(__file__).resolve().parent


def batch_rows(net, steps=16, k_endpoints=8, draws=4, seed=0):
    """One rollout batch: (c, x0, r, t, eps) triples as in D_old."""
    gen = torch.Generator().manual_seed(seed)
    rows = {}
    c_all, x0_all, r_all, t_all, eps_all = [], [], [], [], []
    for c in range(toy.N_PROMPTS):
        endpoints = toy.euler_sample(net, c, k_endpoints, steps, gen)
        goodness = toy.group_weights(toy.reward(endpoints, c))
        x0 = endpoints.repeat_interleave(draws, 0)
        r = goodness.repeat_interleave(draws, 0)
        eps = torch.randn(x0.shape[0], 2, generator=gen)
        t = torch.rand(x0.shape[0], generator=gen).clamp(1e-3, 1.0)
        c_all.append(toy.prompt_tensor(x0.shape[0], c))
        x0_all.append(x0)
        r_all.append(r)
        t_all.append(t)
        eps_all.append(eps)
    rows["c"] = torch.cat(c_all)
    rows["x0"] = torch.cat(x0_all)
    rows["r"] = torch.cat(r_all)
    rows["t"] = torch.cat(t_all)
    rows["eps"] = torch.cat(eps_all)
    rows["xt"] = (1 - rows["t"])[:, None] * rows["x0"] + rows["t"][:, None] * rows["eps"]
    rows["u"] = rows["x0"] - rows["eps"]
    return rows


def loss_forms(net, rows, beta, delta=0.7):
    """Compare the three algebraic forms of the NFT loss at a displaced v_theta."""
    with torch.no_grad():
        v_old = net(rows["xt"], rows["t"], rows["c"])
        base = net(rows["xt"], rows["t"], rows["c"])
    torch.manual_seed(11)  # same displacement for every beta
    direction = torch.randn_like(base)
    v_theta = (base + delta * direction).clone().requires_grad_(True)

    r, u = rows["r"], rows["u"]
    a_coef = (2 * r - 1) / beta
    v_tgt = v_old + a_coef[:, None] * (u - v_old)
    d = v_theta - v_old
    h = u - v_old

    branch = (r * (beta * d - h).square().sum(-1) + (1 - r) * (-beta * d - h).square().sum(-1)).mean()
    frozen = beta**2 * (v_theta - v_tgt).square().sum(-1).mean()
    const = (4 * r * (1 - r) * h.square().sum(-1)).mean()
    frozen_plus_const = frozen + const
    drift = (-2 * beta * (2 * r - 1) * (d * h).sum(-1) + beta**2 * d.square().sum(-1)).mean()
    drift_plus_const = drift + h.square().sum(-1).mean()
    quad_only = (beta**2 * d.square().sum(-1)).mean()

    grad_branch = torch.autograd.grad(branch, v_theta, retain_graph=True)[0]
    grad_drift = torch.autograd.grad(drift, v_theta)[0]
    # the batch-mean loss puts a 1/N factor on every row gradient, so the batch force
    # is the row sum of -grad; the paper's per-state force is (2r-1)h - beta d.
    force = -grad_branch.sum(0)
    force_expected = (2 * beta * (2 * r - 1)[:, None] * h - 2 * beta**2 * d).mean(0)
    restoration_force = (-2 * beta**2 * d).mean(0)
    guidance_force = (2 * beta * (2 * r - 1)[:, None] * h).mean(0)

    return {
        "branch": float(branch),
        "frozen_plus_const": float(frozen_plus_const),
        "drift_plus_const": float(drift_plus_const),
        "abs_gap_frozen": float((branch - frozen_plus_const).abs()),
        "rel_gap_frozen": float((branch - frozen_plus_const).abs() / branch.abs()),
        "abs_gap_drift": float((branch - drift_plus_const).abs()),
        "rel_gap_drift": float((branch - drift_plus_const).abs() / branch.abs()),
        "quadratic_only": float(quad_only),
        "grad_abs_gap": float((grad_branch - grad_drift).abs().max()),
        "force_vs_closed_form": float((force - force_expected).abs().max()),
        "force_vs_closed_form_rel": float(
            (force - force_expected).abs().max() / force_expected.abs().max().clamp_min(1e-12)
        ),
        "guidance_force_norm": float(guidance_force.norm()),
        "restoration_force_norm": float(restoration_force.norm()),
        "restoration_over_guidance": float(
            restoration_force.norm() / guidance_force.norm().clamp_min(1e-12)
        ),
    }


def restoration_vanishes(net, rows, beta=1.0):
    """C: the quadratic term beta^2 ||d||^2 restores the field; its force is -beta*d
    per state and it is exactly zero at d = 0 (all in the paper's -1/(2b) grad convention)."""
    with torch.no_grad():
        v_old = net(rows["xt"], rows["t"], rows["c"])
    out = {}
    for drift in (0.0, 0.5):
        shift = torch.full_like(v_old, drift)
        v_theta = (v_old + shift).clone().requires_grad_(True)
        quad = (beta**2 * (v_theta - v_old).square().sum(-1)).mean()
        grad = torch.autograd.grad(quad, v_theta)[0]
        n_rows = v_old.shape[0]
        force = -grad / (2 * beta)
        pop_force = -beta * shift  # same quantity per state, without the batch mean
        out[f"drift_{drift}"] = {
            "quadratic_value": float(quad),
            "force_per_row_norm": float(force.norm()),
            "expected_batch_normalised": float((2 * beta**2 * shift / n_rows * 1.0).norm()),
            "force_err_vs_beta_shift": float((force + beta * shift / n_rows).abs().max()),
            "per_state_force_scale": float(pop_force.abs().max()),
        }
    return out


def train_run(beta, lr, updates, steps, seed, iters=150, k_endpoints=8, draws=4, optimiser="adam"):
    """Minimal NFT trainer: fresh on-policy rollouts, `updates` disjoint minibatches."""
    torch.manual_seed(seed)
    net = toy.pretrained(seed=0)
    opt = (
        torch.optim.Adam(net.parameters(), lr=lr)
        if optimiser == "adam"
        else torch.optim.SGD(net.parameters(), lr=lr)
    )
    rewards, losses = [], []
    for it in range(iters):
        rows = batch_rows(net, steps=steps, k_endpoints=k_endpoints, draws=draws, seed=1000 * seed + it)
        with torch.no_grad():
            v_old = net(rows["xt"], rows["t"], rows["c"])
            v_tgt = v_old + ((2 * rows["r"] - 1) / beta)[:, None] * (rows["u"] - v_old)
        n = rows["r"].shape[0]
        chunks = torch.chunk(torch.arange(n), updates)
        for chunk in chunks:
            pred = net(rows["xt"][chunk], rows["t"][chunk], rows["c"][chunk])
            loss = beta**2 * (pred - v_tgt[chunk]).square().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            gen = torch.Generator().manual_seed(5000 + it)
            end = torch.cat([toy.euler_sample(net, c, 64, steps, gen) for c in range(2)])
            rewards.append(float(torch.stack([toy.reward(end[c * 64 : (c + 1) * 64], c) for c in range(2)]).mean()))
    return rewards, losses


def main() -> None:
    import sys

    torch.manual_seed(0)
    net = toy.pretrained(seed=0)
    rows = batch_rows(net, seed=1)
    identities = {f"beta_{b}": loss_forms(net, rows, b) for b in (0.25, 1.0, 4.0)}
    vanish = restoration_vanishes(net, rows)
    out_path = OUT / "results" / "prop2.json"
    if "identities-only" in sys.argv:
        out = json.loads(out_path.read_text())
        out["loss_identities"] = identities
        out["restoration_vanishes"] = vanish
        out_path.write_text(json.dumps(out, indent=2))
        print("=== A/B/C: NFT loss and gradient identities ===")
        for key, row in identities.items():
            print(
                f"  {key}: rel gap frozen={row['rel_gap_frozen']:.2e} drift={row['rel_gap_drift']:.2e} "
                f"branch-vs-drift grad gap={row['grad_abs_gap']:.2e} "
                f"force err={row['force_vs_closed_form']:.2e} (rel {row['force_vs_closed_form_rel']:.2e})"
            )
            print(
                f"        |guidance force|={row['guidance_force_norm']:.4f} "
                f"|restoration force|={row['restoration_force_norm']:.4f} "
                f"ratio={row['restoration_over_guidance']:.3f}"
            )
        for k, v in vanish.items():
            print(
                f"  restoration {k}: quad={v['quadratic_value']:.4e} |force|={v['force_per_row_norm']:.4e} "
                f"err vs -beta*d/N={v['force_err_vs_beta_shift']:.2e}"
            )
        return

    # D: beta is not a step-size knob.  beta rescales the regression target
    # v_tgt = v_old + ((2r-1)/beta)(u - v_old), so it changes the loss minimiser,
    # while the update-point force E[(2r-1)h] is beta-free (extracted above).
    ITERS_D = 60
    LR_D = 2e-3
    beta_sweep = {"adam": {}, "sgd": {}}
    for beta in (0.25, 1.0, 4.0):
        rewards, _ = train_run(beta, LR_D, updates=1, steps=16, seed=3, iters=ITERS_D)
        beta_sweep["adam"][f"beta{beta}_lr{LR_D}"] = rewards
    for beta, lr in ((1.0, LR_D), (0.25, LR_D), (0.25, LR_D / 0.25), (4.0, LR_D * 4)):
        rewards, _ = train_run(beta, lr, updates=1, steps=16, seed=3, iters=ITERS_D, optimiser="sgd")
        beta_sweep["sgd"][f"beta{beta}_sgd_lr{lr:.1e}"] = rewards

    two_update = {}
    for beta in (0.25, 1.0, 4.0):
        rewards, _ = train_run(beta, LR_D, updates=2, steps=16, seed=3, iters=ITERS_D)
        two_update[f"beta{beta}"] = rewards

    out = {
        "loss_identities": identities,
        "restoration_vanishes": vanish,
        "beta_single_update": beta_sweep,
        "beta_two_update": two_update,
    }
    out_path.write_text(json.dumps(out, indent=2))

    print("=== A/B/C: NFT loss and gradient identities ===")
    for key, row in identities.items():
        print(
            f"  {key}: rel gap frozen={row['rel_gap_frozen']:.2e} drift={row['rel_gap_drift']:.2e} "
            f"branch-vs-drift grad gap={row['grad_abs_gap']:.2e} "
            f"force err={row['force_vs_closed_form']:.2e} (rel {row['force_vs_closed_form_rel']:.2e})"
        )
        print(
            f"        |guidance force|={row['guidance_force_norm']:.4f} "
            f"|restoration force|={row['restoration_force_norm']:.4f} "
            f"ratio={row['restoration_over_guidance']:.3f}"
        )
    print("  restoration term at d=0 / d=0.5 (per-state force = -beta*d):")
    for k, v in vanish.items():
        print(
            f"    {k}: quad={v['quadratic_value']:.4e} |force|={v['force_per_row_norm']:.4e} "
            f"err vs -beta*d/N={v['force_err_vs_beta_shift']:.2e}"
        )
    print("\n=== D: beta sweep, single on-policy update ===")
    for optimiser, curves in beta_sweep.items():
        names = list(curves)
        base = torch.tensor(curves[names[0]])
        for name in names:
            cur = torch.tensor(curves[name])
            print(
                f"  {optimiser:5s} {name:28s} final={cur[-20:].mean():.4f} max|diff vs first|={(cur-base).abs().max():.4e}"
            )
    print("\n=== D': beta sweep, two disjoint minibatch updates ===")
    for name, rewards in two_update.items():
        print(f"  {name}: final={torch.tensor(rewards[-20:]).mean():.4f}")


if __name__ == "__main__":
    main()
