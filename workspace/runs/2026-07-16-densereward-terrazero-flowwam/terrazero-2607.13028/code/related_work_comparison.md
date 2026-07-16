# TerraZero vs Closest Related Work

## Object-level driving simulators

| System | Backend | Single-GPU SPS | Heterogeneous agents | Multiple dynamics | Signals | Reactive | Real-world maps |
|--------|---------|----------------|----------------------|-------------------|---------|----------|-----------------|
| Nocturne | C++ | 2K (reported) | Vehicles only | 1 | — | — | Waymo |
| SMARTS | Python | slow (CPU-bound) | Veh/Ped/Cyc | Multiple | ✓ | ✓ | Synthetic (SUMO) |
| Waymax | JAX | — | Veh/Ped/Cyc | 2 | ✓ | ✓ | Waymo |
| GPUDrive | Madrona/CUDA | 11K consumer / 319K server (reported) | Vehicles only | 1 | — | — | Waymo |
| PufferDrive | C | 320K consumer / 745K server | Vehicles only | 1 | — | — | Waymo |
| Gigaflow | — | 530K (reported) | Veh/Ped/Cyc | 1 (bicycle for all) | ✓ | ✓ | Synthetic (CARLA) |
| **TerraZero** | **C** | **560K consumer / 1.3M server / 2.8M 8×node** | **Veh/Ped/Cyc** | **Multiple** | **✓** | **✓** | **Waymo + nuPlan + CARLA** |

## What is genuinely new?

1. **Speed + fidelity combination**: C engine at >1M SPS while retaining heterogeneous agents, multiple dynamics models, and traffic-rule enforcement. Prior fast simulators (PufferDrive, GPUDrive) are narrower.
2. **Procedural generation from real maps**: Uses real-world map geometry only, then procedurally populates scenarios with randomized NPCs, signals, dynamics, rewards. Most prior work trains on logs directly or uses synthetic maps.
3. **Zero-demonstration self-play that works on both planning and sim-agent benchmarks**: One recipe tops InterPlan, is competitive on val14, and produces strong WOSAC sim agents without any logged human trajectories or reference policies.
4. **Cross-dataset / cross-handedness generalization**: Trained policy transfers zero-shot across cities and even from RHT to LHT (Singapore), suggesting it learns lane topology rather than memorizing driving conventions.

## Comparison with specific baselines

### Waymax
- Waymax is a JAX-based simulator/benchmark for planning and sim-agent research.
- TerraZero is a full training stack (simulator + PPO recipe) built for large-scale self-play RL.
- Waymax is benchmark-oriented; TerraZero emphasizes throughput and RL training.

### Nocturne
- Nocturne pioneered object-level driving sim with C++ backend over Waymo.
- TerraZero adds heterogeneous agents, signals, multiple dynamics, procedural generation, and multi-source maps.

### SMARTS
- SMARTS is feature-rich but Python-based and CPU-bound → too slow for billion-step RL.
- TerraZero trades some scenario-spec flexibility for raw throughput.

### GPUDrive / PufferDrive
- Both are high-throughput object-level simulators.
- GPUDrive is GPU-resident (Madrona/CUDA); PufferDrive is C-based.
- Both support only vehicles with a single dynamics model and no signals.
- TerraZero supports vehicles, pedestrians, cyclists, trucks, signals, and rule enforcement at comparable or higher throughput.

### Gigaflow
- Closest antecedent: large-scale self-play RL for driving.
- Differences:
  - Gigaflow trains on synthetic CARLA-derived maps; TerraZero on real-world maps (Waymo, nuPlan).
  - Gigaflow uses one bicycle model for all agent classes; TerraZero uses per-class dynamics.
  - Gigaflow's WOSAC entry uses scripted controller for pedestrians; TerraZero controls pedestrians/cyclists jointly.
  - Gigaflow reports no long-tail benchmark; TerraZero tops InterPlan.
  - Gigaflow policy is larger/context-richer; TerraZero uses a leaner observation set.

### SPACeR
- Same research group (Applied Intuition); related but distinct paper.
- SPACeR anchors self-play to a pretrained tokenized reference model (CAT-K/SMART) via likelihood reward + KL divergence.
- SPACeR needs a reference policy trained on logged data; TerraZero uses no demonstrations/reference at all.
- SPACeR policies are tiny (~65K params) and specialized for sim agents; TerraZero policies are larger (3.5M/6.7M) and serve both planner and sim-agent roles.
- Both compare on WOSAC 2024; TerraZero matches SPACeR on vehicle realism without a reference model.

### TerraTransfer
- Follow-up work by overlapping authors (arXiv 2606.17386).
- Uses TerraZero simulator for Phase 1 self-play, then aligns a vision backbone to the privileged-state policy.
- Explicitly says: "multi-agent self-play in TerraZero, our vectorized simulator."
- Confirms TerraZero is a real internal codebase, shared across Applied Intuition projects.
