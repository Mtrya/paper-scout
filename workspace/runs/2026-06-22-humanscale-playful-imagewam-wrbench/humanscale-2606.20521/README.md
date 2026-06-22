# HumanScale deep-dive thread

Paper: **HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining**  
arXiv: `2606.20521`  
Repo inspected: https://github.com/DAGroup-PKU/HumanNet (cloned to `code/HumanNet`, HEAD `68eb86d`)  
Thread directory: `runs/2026-06-22-humanscale-playful-imagewam-wrbench/humanscale-2606.20521/`

---

## Research question

The paper asks whether egocentric human video, when filtered and pseudo-labeled, can be a *better* pretraining substrate than teleoperated real-robot data for embodied foundation models, under matched scale and identical post-training/evaluation protocols. This is not just a cost question; it tests whether the coverage advantage of human video outweighs the embodiment-alignment advantage of robot data.

---

## Core findings (with numbers)

The authors pretrain the same autoregressive world-action model (MoT video expert initialized from Wan 2.2, action expert interpolated) on either egocentric video from HumanNet or a multi-embodiment real-robot corpus, both at 5,000 hours, then post-train on 1,500 AgiBot trajectories (15 tasks × 100 demos).

| Split | Metric | Ego @ 5k h | Robot @ 5k h | Wan2.2 no pretrain |
|---|---|---|---|---|
| Seen (ID) | best action loss | **0.0067** | 0.0071 | 0.0103 (inferred) |
| Unseen (OOD) | best action loss | **0.0204** | 0.0254 | 0.0268 (inferred) |
| Real robot | ID success rate | **92.5%** | — | 40.0% |
| Real robot | OOD success rate | **90.0%** | — | 0.0% |

Key takeaways:

1. **Ego scales log-linearly** in both splits. Reported losses fall from 0.0080 → 0.0067 (Seen) and 0.0234 → 0.0204 (Unseen) as pretraining grows from 100 h to 5,000 h. The paper fits `L = a − b ln(D)` and reports `R² = 0.86` (Seen) and `R² = 0.94` (Unseen).
2. **Ego is better on OOD, comparable on ID.** At 5,000 h, ego is only ~5.6% better than robot on Seen loss, but ~19.7% better on Unseen loss. Real-robot pretraining barely beats the no-pretrain baseline on Unseen (0.0254 vs. inferred 0.0268), while ego provides a 24% reduction.
3. **The OOD advantage transfers to physical rollouts.** Ego-pretrained policy keeps 90% success on unseen objects vs. 92.5% ID; the no-pretrain baseline collapses from 40% ID to 0% OOD.

---

## External signals inspected

1. **Cloned the HumanNet repository** (`code/HumanNet`, HEAD `68eb86d`). The repository is advertised as the home of the HumanNet corpus, related models, and release notes.
2. **Read repository docs:**
   - `code/HumanNet/README.md` — repository map and news.
   - `code/HumanNet/docs/humandata.md` — HumanNet dataset overview, with a placeholder loss figure claiming 1,000 h of egocentric video matches or modestly surpasses 100 h of real-robot data and closes the gap to a 20,000 h real-robot baseline (a different, VLA-based validation than HumanScale's WAM).
   - `code/HumanNet/docs/stablevla.md` — documentation for StableVLA, the group's VLA model; unrelated to the HumanScale WAM study.
3. **Walked the source tree.** Relevant directories:
   - `src/model/StableVLA/` — StableVLA training and evaluation code for LIBERO/CALVIN.
   - `src/dataset/rovid-x/` — example scripts for extracting CoTracker trajectories and Video-Depth-Anything depth maps from robot videos, plus a small CSV manifest.
   - `src/dataset/humandata/` — only a `.gitkeep`; the HumanNet corpus and metadata are not released yet.

### What is missing

**The HumanNet repo does not currently contain any HumanScale-specific artifacts.** As of the inspected commit there is no:

- WAM/MoT model code used in the paper,
- data curation or filtering pipeline for the 5,000-hour egocentric subset,
- hand-pose retargeting / pseudo-action labeling code,
- post-training recipe on AgiBot World,
- evaluation harness, checkpoints, or raw loss curves,
- `StableVLA` is present but it is a separate VLA model, not the world-action model studied in HumanScale.

The README explicitly marks the HumanNet preview subset, full corpus, and trained checkpoints as "coming next month" (`[Next Month]` news item and unchecked todo list). This is a genuine blocker for reproducing or auditing the HumanScale experiments from the repository alone.

### Evidence preserved

Because the code is not available, this thread preserves:

- the exact reported numbers and the fitted log-linear scaling law (`code/ego_scaling_probe.py`),
- a record of the repository state at HEAD `68eb86d` and the files that were inspected,
- a note that the repo currently serves as a landing page for future releases rather than a reproducibility package for HumanScale.

---

## Probe: `code/ego_scaling_probe.py`

### What it does

The script reconstructs the scaling story from the paper's reported numbers:

- Fits `L = a − b ln(hours)` to the reported 100 h and 5,000 h losses for both Seen and Unseen splits.
- Reconstructs the implied 1,000 h loss under perfect log-linearity (the paper gives `R²` but not the raw midpoint).
- Infers the Wan2.2 no-pretrain baseline from the paper's stated percentage reductions (35% lower on Seen, 24% lower on Unseen).
- Compares ego vs. robot pretraining at 5,000 h.
- Extrapolates the fitted laws to 20,000 h and 100,000 h to show how slowly the curves would saturate if the log-linear trend held.
- Writes `ego_scaling_table.csv` and, if `matplotlib` is installed, `ego_scaling.png`.

### How to rerun

```bash
cd runs/2026-06-22-humanscale-playful-imagewam-wrbench/humanscale-2606.20521/code
python ego_scaling_probe.py
```

Dependencies: Python ≥ 3.8, `numpy`. `matplotlib` is optional (only used for the diagnostic plot).

### Sample output

```text
Split: Seen
  Fitted law: L = 0.009530 - 0.000332 * ln(hours)
  Reconstructed 1,000 h loss (perfect log-linearity): 0.0072
  Ego vs Robot @ 5,000 h: 5.6% lower (0.0067 vs 0.0071)
  Ego vs Wan2.2 baseline: 35.0% lower (inferred baseline 0.0103)
  Extrapolated 20,000 h loss: 0.0062
  Extrapolated 100,000 h loss: 0.0057

Split: Unseen
  Fitted law: L = 0.026932 - 0.000767 * ln(hours)
  Reconstructed 1,000 h loss (perfect log-linearity): 0.0216
  Ego vs Robot @ 5,000 h: 19.7% lower (0.0204 vs 0.0254)
  Ego vs Wan2.2 baseline: 24.0% lower (inferred baseline 0.0268)
  Extrapolated 20,000 h loss: 0.0193
  Extrapolated 100,000 h loss: 0.0181
```

The CSV table (`ego_scaling_table.csv`) contains the same reconstructed values for easy plotting.

### Caveats

- The 1,000 h value is *reconstructed*, not reported. The paper's actual 1,000 h checkpoint may sit above or below the perfect-fit line; the stated `R²` values (0.86 and 0.94) imply real points are close but not exactly collinear.
- Extrapolation to 100,000 h assumes the log-linear law keeps holding, which the authors themselves caution against; they stopped at 5,000 h due to limited real-robot post-training data.
- The no-pretrain baseline is inferred from percentage wording in the paper, not a directly reported loss.

---

## What the thread means for the report

HumanScale makes a strong, quantified claim: **under matched scale and protocol, egocentric pretraining can beat real-robot pretraining, especially on out-of-distribution generalization.** The numbers are internally consistent and produce a clean log-linear trend.

However, the claim is currently *paper-only*. The advertised repository does not yet expose the model, data pipeline, or evaluation code needed to audit or extend the result. The thread should therefore flag the finding as promising but unverified by external code, and focus the report on the buildable interpretation: **if the curation/retargeting quality is high enough, embodiment alignment can be deferred to a small post-training stage, and the pretraining stage should prioritize coverage and scale.** That is the practical takeaway for data collection strategy.

For follow-up, the most valuable external signals would be:

1. the release of the HumanNet curation and retargeting pipeline,
2. the HumanScale WAM training/evaluation code and checkpoints,
3. independent replication on a different robot embodiment or VLA backbone (the authors note they are already testing VLA variants).

Until then, the report should treat the 5,000 h real-robot vs. egocentric comparison as a strong empirical hypothesis, not a settled engineering recipe.
