"""Experiment 0: do the shipped DiffusionART kernels implement the paper's objects?

The official repository is cloned at code/DiffusionART in the workspace lab bench.
This script imports its pure-tensor kernels (visual_rl.algorithms) and checks, on
synthetic conditional ensembles in the code's own convention (prediction g = eps - x_0,
x_t = (1 - sigma) x_0 + sigma eps):

  A. nft_loss gradient at the update point equals sum_row c_row (2 r_row - 1) h_row
     with the exact per-row constants the kernel itself reports -> the code's force is
     the paper's F_NFT object, and the branch normalisation cancels at d = 0.
  B. the kernel's restoration coefficient (second derivative in prediction space)
     versus the paper's beta^2 ||d||^2 term: what beta does the shipped code actually use?
  C. art_loss gradient at the update point is pure guidance (no data-dependent residual),
     identical for two conditional ensembles that differ only in their forward gap e.
  D. art_scalar_covariance reproduces paper Eq. 5 (G = Cov / RMS(Cov), unit RMS).

Writes results/kernel_map.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

TOY = Path(__file__).resolve().parent
REPO = TOY.parents[3] / "code" / "DiffusionART" / "src"
sys.path.insert(0, str(REPO))

from visual_rl.algorithms import art_loss, art_scalar_covariance, nft_loss  # noqa: E402

OUT = TOY


def conditional_ensemble(sigma: float, xt: torch.Tensor, x0_block: torch.Tensor, deltas=None):
    """Rows consistent with x_t = (1-sigma) x_0 + sigma eps for a single state x_t."""
    x0 = x0_block.clone()
    if deltas is not None:
        x0 = x0 + deltas
    eps = (xt[None] - (1 - sigma) * x0) / sigma
    return x0, eps


def main() -> None:
    torch.manual_seed(0)
    sigma = 0.3
    xt = torch.tensor([0.5, 0.2])
    base = torch.tensor([[0.9, 0.3], [0.7, -0.1], [-0.4, 0.6], [-0.6, -0.5], [0.2, 0.4], [-0.2, -0.3]])
    x0, eps = conditional_ensemble(sigma, xt, base)
    eps_all = eps
    rows = x0.shape[0]
    r_good = 0.5 + torch.tensor([-5.0, -2.0, 3.0, 5.0, 1.0, -1.0]) / 10.0  # kernel adv/10 mapping
    v_old = torch.tensor([0.05, -0.02]).repeat(rows, 1)
    sig = torch.full((rows,), sigma)
    u_code = eps_all - x0  # code convention: prediction target
    h = u_code - v_old
    q = float(r_good.mean())
    g = ((r_good - q)[:, None] * (u_code - u_code.mean(0))).mean(0)
    e = u_code.mean(0) - v_old[0]

    # --- A: nft kernel force at d = 0 -----------------------------------------------------
    beta = 1.0
    adv = (2 * r_good - 1) * 5.0  # kernel standardises advantages, clip_max=5
    pred = v_old.clone().requires_grad_(True)
    out = nft_loss(pred, v_old, x0, xt.repeat(rows, 1), sig, adv, nft_beta=beta)
    grad = torch.autograd.grad(out["per_item_loss"].sum(), pred)[0]
    prob = out["positive_probability"]
    n_pos = out["positive_weight"]
    n_neg = out["negative_weight"]
    # At v = v_old both branch residuals equal sigma * h, so the row gradient is
    # grad_row = (2 sigma clip / D) * (1 - 2 p) * h_row / mean|h_row|; the row constants
    # therefore cancel between branches and the total force is proportional to
    # sum_row (2 r - 1) h_row / mean|h_row| -- the paper's F_NFT object with a
    # per-row (not per-branch) normalisation.
    row_abs = h.abs().mean(-1, keepdim=True)
    recon = (2 * sigma * 5.0 / 2) * (1 - 2 * prob)[:, None] * h / row_abs
    total_grad = grad.sum(0)
    paper_force = (2 * r_good - 1)[:, None] * h
    row_weights = (paper_force / n_pos[:, None]).sum(0)
    A = {
        "total_grad": total_grad.tolist(),
        "total_force": (-total_grad).tolist(),
        "reconstruction_max_err": float((grad - recon).abs().max()),
        "cos_vs_unnormalised_paper_force": float(
            torch.nn.functional.cosine_similarity((-total_grad).flatten(), paper_force.sum(0).flatten(), dim=0)
        ),
        "cos_vs_rownormalised_paper_force": float(
            torch.nn.functional.cosine_similarity((-total_grad).flatten(), row_weights.flatten(), dim=0)
        ),
        "constant_vs_rownormalised": float(
            (-total_grad).norm() / row_weights.norm().clamp_min(1e-12)
        ),
        "mean_abs_h": float(h.abs().mean()),
        "closed_form_residual": (2 * g + (2 * q - 1) * e).tolist(),
        "empirical_sum_force": paper_force.sum(0).tolist(),
        "empirical_mean_force": (paper_force.sum(0) / rows).tolist(),
        "closed_form_err": float((paper_force.sum(0) / rows - (2 * g + (2 * q - 1) * e)).abs().max()),
    }

    # --- B: kernel restoration coefficient vs paper beta^2 ---------------------------------
    B = []
    for beta in (0.25, 1.0, 4.0):
        for direction in (torch.tensor([1.0, 0.0]), torch.tensor([0.3, -0.8])):
            for step in (0.05,):
                pred = v_old.clone().requires_grad_(True)
                out = nft_loss(pred, v_old, x0, xt.repeat(rows, 1), sig, adv, nft_beta=beta)
                base_loss = out["per_item_loss"].sum()
                g1 = torch.autograd.grad(base_loss, pred, create_graph=True)[0]
                d = direction / direction.norm()
                deriv2 = torch.autograd.grad((g1 * d).sum(), pred)[0]
                hess = float((deriv2 * d).sum())
                predicted = float(
                    (2 * sigma**2 * beta * 5.0 / 2) * (prob / n_pos + (1 - prob) / n_neg).sum()
                    * (1.0 if True else 1.0)
                )
                B.append(
                    {
                        "beta": beta,
                        "direction": d.tolist(),
                        "kernel_hessian": hess,
                        "predicted_from_kernel_weights": predicted,
                        "prediction_err": abs(predicted - hess),
                        "paper_hessian": float(2 * beta**2 * rows),
                        "ratio": hess / (2 * beta**2 * rows),
                    }
                )

    # --- C: art kernel force is ensemble-independent and pure guidance ---------------------
    guidance = torch.tensor([0.6, -0.35])
    g_dir = guidance / guidance.norm()
    target = v_old[0] + guidance
    C_rows = []
    ens_variants = {
        "narrow": (x0, eps_all),
        "spread": (x0 * 0.2 + 0.05, eps_all * 0.2 + 0.05),
        "shifted": (x0 + 0.5, eps_all),
    }
    for name, (x0v, epsv) in ens_variants.items():
        pred = v_old.clone().requires_grad_(True)
        out = art_loss(
            pred,
            v_old,
            torch.full((rows,), sigma),
            guidance.repeat(rows, 1),
            torch.ones(rows),
            ode_velocity_coefficient=torch.full((rows,), -1.0 / 16),
            direction_normalization="rms",
            corrective_mask_enabled=False,
        )
        grad = torch.autograd.grad(out["per_item_loss"].sum(), pred)[0]
        h_v = (epsv - x0v) - v_old
        C_rows.append(
            {
                "ensemble": name,
                "art_force": grad[0].tolist(),
                "art_force_norm": float(grad[0].norm()),
                "art_cos_vs_guidance": float(
                    torch.nn.functional.cosine_similarity(grad[0].unsqueeze(0), g_dir.unsqueeze(0)).item()
                ),
                "nft_force_norm": float(((2 * r_good - 1)[:, None] * h_v).sum(0).norm()),
                "nft_force": ((2 * r_good - 1)[:, None] * h_v).sum(0).tolist(),
                "conditional_e_norm": float(h_v.mean(0).norm()),
            }
        )

    # --- D: covariance normalisation = paper Eq. 5 -----------------------------------------
    endpoints = torch.cat([x0, x0 * 1.7 + 0.3])
    rewards = torch.cat([r_good, r_good * 0.4 - 0.2])
    groups = torch.zeros(endpoints.shape[0], dtype=torch.long)
    cov_out = art_scalar_covariance(
        endpoints, rewards, groups, 1, direction_mode="normalized_aggregate", zero_signal_mode="exact"
    )
    cov_direct = ((rewards - rewards.mean())[:, None] * (endpoints - endpoints.mean(0))).mean(0)
    paper_g = cov_direct / cov_direct.square().mean().sqrt()
    D = {
        "kernel_direction": cov_out["group_directions"].flatten().tolist(),
        "paper_G": paper_g.tolist(),
        "max_err": float((cov_out["group_directions"].flatten() - paper_g).abs().max()),
        "direction_rms": float(cov_out["group_directions"].flatten().square().mean().sqrt()),
        "magnitude": float(cov_out["group_guidance_magnitudes"].item()),
    }

    (OUT / "results" / "kernel_map.json").write_text(
        json.dumps({"A_nft_force": A, "B_restoration": B, "C_art_force": C_rows, "D_covariance": D}, indent=2)
    )

    print("=== A: NFT kernel force at the update point ===")
    print(f"  reconstruction err={A['reconstruction_max_err']:.2e} mean|h|={A['mean_abs_h']:.4f}")
    print(f"  cos(F_kernel, sum (2r-1)h) = {A['cos_vs_unnormalised_paper_force']:.6f}")
    print(f"  cos(F_kernel, row-normalised) = {A['cos_vs_rownormalised_paper_force']:.6f} "
          f"constant={A['constant_vs_rownormalised']:.4f}")
    print(f"  mean force = {[round(x,4) for x in A['empirical_mean_force']]}  "
          f"vs closed form 2g+(2q-1)e = {[round(x,4) for x in A['closed_form_residual']]} "
          f"(max err {A['closed_form_err']:.2e})")
    print("\n=== B: kernel restoration coefficient vs paper beta^2 ===")
    for row in B:
        print(
            f"  beta={row['beta']:<5}: kernel H={row['kernel_hessian']:.4f} "
            f"predicted={row['predicted_from_kernel_weights']:.4f} (err {row['prediction_err']:.1e}) "
            f"paper H={row['paper_hessian']:.4f} ratio={row['ratio']:.4f}"
        )
    print("\n=== C: ART vs NFT force on different conditional ensembles ===")
    for row in C_rows:
        print(
            f"  {row['ensemble']:8s}: |F_art|={row['art_force_norm']:.6f} cos={row['art_cos_vs_guidance']:+.6f} "
            f"|F_nft|={row['nft_force_norm']:.6f} F_nft={[round(x, 3) for x in row['nft_force']]}"
        )
    print("\n=== D: covariance normalisation ===")
    print(f"  kernel G = {[round(x, 6) for x in D['kernel_direction']]} paper G = {[round(x, 6) for x in D['paper_G']]}")
    print(f"  max err={D['max_err']:.2e} rms={D['direction_rms']:.6f} magnitude={D['magnitude']}")


if __name__ == "__main__":
    main()
