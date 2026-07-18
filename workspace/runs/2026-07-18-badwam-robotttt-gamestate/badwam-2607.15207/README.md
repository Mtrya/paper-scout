# BadWAM (arXiv 2607.15207) — thread packet

Paper: *BadWAM: When World-Action Models Dream Right but Act Wrong* (Li, Yang, Wang).
Official code: https://github.com/LiQiiiii/BadWAM (cloned to `code/badwam` during the run; code-only release — no checkpoints, no LIBERO data).

## What was attempted

1. **Code trace** of the official repo (`src/attackwam/attacks.py`, `experiments/attackwam/`).
2. **Toy reconstruction + attack probe**: since the official release ships no checkpoints and needs LIBERO, we rebuilt the attacked object — a minimal WAM — in pure numpy (`code/toy_wam.py`), trained three variants mirroring the paper's three FastWAM variants, and attacked them with a faithful numpy re-implementation of the paper's black-box optimizer (`code/attacks.py`), plus a white-box PGD reference (paper appendix A.4).

## Code-trace findings

- The official objective (`DesynchronizationObjective.components`) is exactly the paper's Eq. 9 Lagrangian: `score = action_weight * ||a_adv − a_clean||₂ − future_weight * ||z_adv − z_clean||₂ − perturb_weight * mean|δ|`. Untargeted by default (maximize action distance); a targeted mode flips the sign toward `clean_action * target_action_scale`. Also exports a `desynchronization_score = action_distance / (future_distance + 1e-6)`.
- Attack classes: `QuerySearchAttack` (1-direction SPSA: Rademacher `u`, `g = (J(δ+σu) − J(δ−σu))/(2σ) · u`, signed-PGD update `δ ← Π(δ + η·sign(g))`), `RandomSearchAttack`, `UniversalPatchAttack` (a *universal*, image-agnostic patch trained over a dataset — the stealthiest variant), plus patch/camera-region masks (`PatchMixin`). Query accounting matches the paper: 1 clean + 2·iters queries per replan (iters=8 → 17 queries, ε=0.06, σ=0.02, step=0.02, appendix A.3).
- Evaluation is LIBERO closed-loop (`experiments/libero/`) with RoboTwin hooks; analysis scripts compute the matched-strength stealth comparison and channel/horizon action statistics.
- Release gaps vs paper: no checkpoints, no dataset statistics, no experiment outputs — the headline numbers (96.5%→43.1%) are not independently reproducible from the release as-is.

## Toy probe design

`code/toy_wam.py`: 2D point-mass reaching with a gripper channel (bang-bang, irreversible mis-grasp = failure, so attacks can break the closed loop rather than just add noise). Observation: 32×32 grayscale render in [-1,1] so the paper's ε=0.06 L∞ budget means the same thing. Three variants:

- `direct` — action head reads the observation encoder latent only (FastWAM action-only inference);
- `joint` — action head reads the same latent the future-video decoder reads (FastWAMJoint);
- `idm` — action head reads *only* the decoded future imagination (FastWAMIDM; strict imagine-then-act).

All training/inference uses exact analytic gradients (no autodiff). Attacks in `code/attacks.py` replicate `QuerySearchAttack` line-for-line, with the paper's defaults; `whitebox_pgd` is the appendix-A.4 gradient-access reference (16 steps, α=0.01).

## Results (all in `code/results.json`; 80 episodes/cell for clean/random/SPSA-m1, 40 for m=16 and white-box; grasp task)

Closed-loop task success under attack (clean = 100% joint/direct, 98.8% idm):

| attack | joint | idm | direct |
|---|---|---|---|
| black-box, paper recipe (m=1, 17 queries/replan), λ=0 | 100% | 98.8% | 100% |
| black-box m=16 (257 queries/replan), λ=0 | 100% | 92.5% | 97.5% |
| white-box PGD, λ=0 | **15%** | **20%** | **35%** |
| white-box PGD, λ=0.1 | 67.5% (D_img −55% vs λ=0) | 75% (D_img −63%) | — |
| white-box PGD, λ=0.3 | 100% (attack neutralized) | 90% | — |

Mechanism analysis (input-Jacobian SVD, `results.json["mechanism"]`):

- Action-head input sensitivity is strongly low-rank (participation ratio ≈ 1.5–1.9) vs the imagination head's broader sensitivity (PR ≈ 7); cross-sensitivity ≈ 0.97–0.99, and only 1–5% of action sensitivity lies outside the imagination's top-99% subspace. So a perfectly stealthy drift (large action shift, zero imagination shift) is geometrically near-impossible — the stealth frontier must trade strength, which is exactly what the λ sweep shows (idm white-box decoupling ratio D_act/D_img rises 0.46→0.62 as λ goes 0→1, joint 0.33→0.38, while absolute attack strength falls).
- The optimizer finds directions with ~6–10× more action shift per unit perturbation than random noise, and the imagination-preserving objective measurably re-aims the same budget into the thin action-sensitive/imagination-insensitive slice (`assets/probe_mechanism.png`).

## What this means for the report

- The paper's central phenomenon is **real and mechanistically understandable**: a shared, low-rank input-sensitivity geometry lets small observation perturbations move actions a lot and futures much less; we reproduced the full strength–stealth frontier on a from-scratch WAM.
- Two honest tensions the paper underplays: (a) in our toy, the *black-box* recipe (m=1, 17 queries/replan) barely dents closed-loop success even at 15× the query budget — the paper's strong black-box results likely lean on real-task fragility and high-dimensional image structure; (b) at ε=0.06 the perturbation is large enough to be visible, and with only 1–5% of action sensitivity outside the imagination subspace, "dreams right but acts wrong" has a hard geometric ceiling — strong preservation eventually neutralizes the attack (toy: λ≥0.3).
- The `direct` variant (no imagination at all) is just as attackable — imagination coupling neither protects nor uniquely exposes the policy; what WAM coupling adds is a *stealth surface* (monitors that check imagined futures can be fooled), not a new vulnerability magnitude.

## Rerun

```bash
cd runs/2026-07-18-badwam-robotttt-gamestate/badwam-2607.15207/code
python3 train_models.py            # trains the three toy WAMs -> models.npz (~2 min, CPU)
python3 run_probe.py attacks joint # closed-loop sweeps for one variant
python3 run_probe.py attacks idm
python3 run_probe.py attacks direct
python3 run_probe.py mechanism     # Jacobian analysis
python3 run_probe.py figures       # writes assets/probe_*.png
```

Requires numpy + matplotlib only. `models.npz` and `results.json` are preserved, so `figures` can be rerun directly.

## Preserved files

- `code/toy_wam.py`, `code/train_models.py`, `code/attacks.py`, `code/run_probe.py`, `code/smoke_train.py`
- `code/models.npz` (trained toy weights), `code/results.json` (all metrics)
- Figures in `../../assets/`: `probe_lambda_frontier.png`, `probe_mechanism.png`, `probe_desync_scatter.png`, `probe_channels.png`, `probe_qualitative.png`, plus paper figures `badwam-paper-fig1-desync-evidence.png`, `badwam-paper-fig5-failure-profile.png`, `badwam-paper-fig12-matched-stealth.png`, `badwam-paper-table1.png`.
