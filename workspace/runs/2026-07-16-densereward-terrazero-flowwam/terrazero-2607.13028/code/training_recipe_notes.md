# TerraZero Self-Play Training Recipe

## Core stance: zero demonstrations

- No imitation loss.
- No logged trajectories as supervision.
- No fallback planner at inference.
- Logged data enters only as map geometry and optional initialization.

## Algorithm stack

| Component | Choice | Purpose |
|-----------|--------|---------|
| Base algorithm | PPO | Stable policy-gradient updates |
| Advantage estimator | GAE (λ = 0.95) | Bias-variance tradeoff |
| Off-policy correction | V-trace (ρ̄ = c̄ = 1.0) | Reuse rollouts while policy changes between collection and update |
| Value normalization | PopArt (EMA decay 0.9997) | Handle return magnitudes varying across scenarios |
| Sampling | Prioritized advantage sampling (α = 0.85, β annealed) | Compute efficiency: oversample high-advantage segments, skip near-zero ones |
| Optimization | Adam, lr 5e-4 linear decay, 2 epochs per rollout, 16 minibatches | Standard large-batch RL |
| Precision | bf16 mixed precision | Speed / memory |
| Distributed | DDP / Ray, gradient all-reduce + synchronized normalization stats | Scale across 16–32 A100s |

## V-trace + GAE detail

- Clipped importance ratio from each minibatch stored.
- At start of next epoch, ratios are recomputed to update advantages.
- True terminals zero value bootstrap; truncations preserve saved `V(s_final)`.
- Both cut the λ trace → avoids time-limit bias.

## Policy architecture

### Planner (vehicles only)
- Compact MLP, ~**3.5 M parameters**.
- Ego MLP + Deep Sets encoders for partner/road/traffic entities.
- Concatenated embedding → shared 3-layer 1024-unit MLP → actor + value heads.
- Action: discrete jerk grid.

### Heterogeneous sim agent
- ~**6.7 M parameters**.
- Shared trunk + separate per-type action heads:
  - Vehicles: jerk grid
  - Pedestrians: unicycle grid
  - Cyclists: compact bicycle grid

## Observation space (Appendix C)

| Group | Max entities | Features | Encoder |
|-------|--------------|----------|---------|
| Ego | 1 | agent type, relative goal, speed, dims, collision, steering, accel, stop-sign state, reward params (19), kinematic params (4) | MLP |
| Partners | 20 within 50 m | rel pos, dims, rel heading, speed, type, (optional accel) | Deep Sets |
| Road segments | 200 | rel pos, dims, rel angle, road type (lane/line/edge) | Deep Sets |
| Traffic entities | 16 within 100 m | type, 3D rel pos, signal state one-hot, stop-line endpoints | Deep Sets |

All spatial features in ego-centric frame.

## Reward function (Appendix B)

23-parameter vector. Key terms:
- Goal bonus when within δ_goal and below v_goal.
- Collision penalty (speed-scaled).
- Off-road / road-incursion penalty.
- Comfort penalty on jerk/accel thresholds.
- Lane alignment / centering / reverse-driving penalties.
- Velocity / target-speed terms.
- Stop-line / red-light penalties.
- Per-type weights: vehicles use vehicle column; sim agent uses all three (vehicle/ped/cyclist).

## Breaking self-play symmetry

No frozen-checkpoint league. Instead:
1. **Reward/kinematic randomization** makes co-trained policy agents heterogeneous.
2. **Rule-based NPC mix** (IDM/PDM vehicles, parked cars, construction, pedestrians) forces policy to interact with non-self agents.

## Compute

- Planner: 16 × NVIDIA A100 80 GB.
- Sim agent: 32 × NVIDIA A100 80 GB.
- Training from scratch (reported policies use random initialization, no hybrid/log bootstrapping).

## Stability claims

Authors attribute stability at scale to the combination of:
- Priority sampling (focus compute on informative transitions)
- V-trace (correct off-policy mismatch)
- PopArt (normalize highly variable returns)
- Domain randomization + population play (break symmetry)
- Synchronized normalization statistics across ranks
