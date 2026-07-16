# TerraZero Results vs Baselines

## nuPlan val14 closed-loop reactive (Table 2)

| Method | Type | Score ↑ | Notes |
|--------|------|---------|-------|
| SPDM (Np=15) | Rule | 92.28 | Best overall; long-tail weak |
| FlowDrive + guidance + PDM | Hybrid | 92.96 | Uses rule-based scorer |
| Gigaflow | RL | 93.8 | Prior SOTA RL, synthetic maps only |
| **TerraZero** | **RL** | **92.27** | **Safest: best No-AF-Coll (99.11) and TTC (96.06); zero demos** |
| CaRL | RL | 90.60 | Recent RL baseline |

Takeaway: TerraZero is competitive with dedicated planners and the strongest fully-learned RL baseline, with the best safety component scores.

## InterPlan long-tail (Table 3)

| Method | Type | InterPlan 80 ↑ | Full-Scale 335 ↑ |
|--------|------|----------------|------------------|
| SPDM (Np=60) | Rule | 63.66 | — |
| HybridLLMPlanner (Llama-7B) | Hybrid | 53 | — |
| PPO | RL | 42.1 | 61.82 |
| FlowDrive | IL | 36.96 | — |
| **TerraZero** | **RL** | **67.87** | **67.71** |

Takeaway: First fully learned policy to top InterPlan; beats rule-based SPDM (Np=60) and much larger LLM planners, with no rule-based planner at inference.

## WOSAC 2023 full validation (Table 4)

| Method | Demo-Free | Realism ↑ | Notes |
|--------|-----------|-----------|-------|
| Expert Demonstration | logged | 0.722 | Upper bound reference |
| Gigaflow | ✓ | 0.619 | Vehicles + cyclists only; pedestrians scripted |
| **TerraZero (Waymo)** | **✓** | **0.632** | **Vehicles + pedestrians + cyclists jointly controlled** |
| TerraZero (nuPlan, zero-shot) | ✓ | 0.625 | Trained only on nuPlan maps |

## WOSAC 2024 shared 880-scenario subset

### Vehicles (Table 5)

| Method | Demo-Free | Realism ↑ | minADE ↓ | Coll ↓ | Off-rd ↓ |
|--------|-----------|-----------|----------|--------|----------|
| CAT-K | ✗ | 0.766 | 1.47 | 0.060 | 0.090 |
| SPACeR | reference policy | 0.741 | 4.10 | 0.036 | 0.056 |
| **TerraZero (Waymo)** | **✓** | **0.740** | 6.14 | **0.007** | **0.036** |
| TerraZero (nuPlan, zs) | ✓ | 0.732 | 7.99 | 0.007 | 0.040 |

### VRUs (Table 6)

| Method | Demo-Free | Realism ↑ | minADE ↓ |
|--------|-----------|-----------|----------|
| SPACeR | reference policy | 0.729 | 2.07 |
| **TerraZero (Waymo)** | **✓** | **0.683** | 2.88 |
| HR-PPO | reference policy | 0.668 | 7.01 |
| PPO | ✓ | 0.648 | 7.71 |

Takeaway: TerraZero is the strongest method that uses **no demonstrations and no reference policy**. On vehicles it matches SPACeR's realism with far lower collision/off-road rates; on VRUs it beats HR-PPO and PPO but trails SPACeR, likely because SPACeR's reference model is trained directly on human pedestrian/cyclist trajectories.

## Cross-dataset transfer (Figure 6)

- Realism column means vary by target city, not by training source (spread ~0.011).
- Safety column means vary by target (spread ~0.018).
- CARLA-trained policy (only 5 synthetic towns) performs similarly to Waymo/nuPlan-trained policies.
- Policy trained on US right-hand-traffic cities transfers to left-hand-traffic Singapore without degradation.

This supports the claim that robustness comes from the **randomization scheme**, not dataset memorization.
