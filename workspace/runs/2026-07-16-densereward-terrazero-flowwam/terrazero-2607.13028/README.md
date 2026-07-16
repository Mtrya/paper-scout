# TerraZero Deep-Dive Report

**Paper:** *TerraZero: Procedural Driving Simulation for Zero-Demonstration Self-Play at Scale* (arXiv 2607.13028)  
**Authors:** Zhouchonghao Wu, Akshay Rangesh, Weixin Li, Wei-Jer Chang, Zachary Lee, Tim Wang, Wei Zhan (Applied Intuition / UC Berkeley)  
**Date of investigation:** 2026-07-16  
**Thread:** `runs/2026-07-16-densereward-terrazero-flowwam/terrazero-2607.13028/`

---

## TL;DR

TerraZero is a closed-loop object-level driving simulator + self-play RL training stack from Applied Intuition. Its headline claim is combining **high throughput** (1.3M agent-steps/sec on one server GPU) with **high scenario fidelity** (vehicles, pedestrians, cyclists, signals, multiple dynamics models, traffic-rule enforcement) and **zero-demonstration training**. It tops the InterPlan long-tail benchmark, is competitive on nuPlan val14, and produces strong Waymo Open Sim Agents results without any logged human trajectories or reference policies. **No source code, model weights, or terrabin datasets are publicly available as of this investigation**—only project pages, a PDF/technical report, and benchmark videos.

---

## 1. Simulator Architecture

### Engineering stack

- **C simulation engine** compiled as a CPython extension running on the CPU.
- **PyTorch policy/value network** running on the GPU.
- Built on **PufferLib** vectorization and training framework.
- **Zero-copy data path**: C writes simulation state directly into shared numpy buffers that PyTorch reinterprets as tensors.
- **16-bit observations** to halve bandwidth.
- **Variable-size buffers** sized to actual controlled agents, not padded to max count.
- **NUMA-aware** worker pinning per GPU rank.
- Offline `terrabin` binary format decouples source-specific parsing from runtime.

### CPU/GPU split

| CPU (C engine) | GPU (PyTorch) |
|----------------|---------------|
| Object-level dynamics | Policy/value inference |
| Observation construction | PPO/GAE/V-trace/PopArt updates |
| Reward computation | Gradient updates |
| Traffic-light state machines | Distributed all-reduce |
| Collision/off-road/rule checks | bf16 mixed-precision training |

### Throughput claims

| Hardware | Agent steps/sec |
|----------|-----------------|
| Single consumer GPU | 560 K |
| Single server GPU | **1.30 M** |
| 8× server GPU node | **2.80 M** |

A back-of-envelope check (`code/throughput_sanity_check.py`) shows 1.3M SPS is physically plausible: full-fp16 observations are ~3.5 KB/agent, requiring only ~4.7 GB/s H2D bandwidth, and a 3.5M-param MLP at that rate needs ~9 TFLOP/s. **Independent verification is impossible without the code.**

### Supported fidelity

- Heterogeneous agents: vehicles, pedestrians, cyclists.
- Per-class dynamics: unicycle (pedestrians), bicycle (cars/cyclists), bicycle + tire forces (trucks).
- Traffic rules: collisions, off-road, lane direction, stop signs, red lights.
- Signals: NEMA dual-ring, Christmas (uncoordinated LogNormal), round-robin.
- 3D maps with multi-level filtering (bridges/overpasses).

---

## 2. Procedural Scenario Generation

### Core idea

Use logged data **only for real-world map geometry**, then procedurally populate scenarios.

### Real-world map sources

- **Waymo Open Motion Dataset** — ~576 K scenarios (~487 K train); used for sim-agent/WOSAC.
- **nuPlan** — ~262 K scenarios across Las Vegas, Boston, Pittsburgh, Singapore; used for planner/val14/InterPlan.
- **CARLA towns** — 5 synthetic towns; used for transfer/ablation.

### Composable randomization axes

1. Agent classes (any subset of veh/ped/cyc).
2. Per-class dynamics.
3. Initialization mode: log / random / hybrid.
4. Signal controller.
5. Rule-based NPCs (8 generators).
6. Per-episode randomization of reward weights, kinematic coefficients, bounding-box dims, density.

### Rule-based NPCs

- **Reactive vehicles**: IDM + pure pursuit, or PDM-Closed-style planner; default/assertive/cautious modes.
- **Static actors**: parked cars/buses/trucks, crashed-vehicle clusters (disc/rear-end/T-bone/fan), construction zones (grid/taper/lane block), road debris.
- **Pedestrians**: ORCA crosswalk pedestrians (single/pack/bidirectional/staggered); mid-block jaywalkers.
- **Cyclists**: policy-controlled or log-replay only; no scripted NPC controller.

### Domain randomization

- Kinematic coefficients `(c_throttle, c_steer, c_acc, c_vel)` sampled per agent in `[0.5, 1.5]`.
- 23 reward parameters can be fixed, sampled per agent, or disabled.
- Goal dropout: 30% of vehicles have goal hidden.
- Sampled values are exposed in the observation, enabling reward-conditioned / kinematic-conditioned policies.

---

## 3. Self-Play RL Recipe

### Algorithm

- **PPO** with **GAE** (λ=0.95).
- **V-trace** off-policy corrections (ρ̄ = c̄ = 1.0).
- **PopArt** value normalization (EMA decay 0.9997).
- **Prioritized advantage sampling** (α=0.85, β annealed).
- **Adam**, lr 5e-4 linear decay, 2 epochs per rollout, 16 minibatches.
- **bf16 mixed precision**.
- Distributed with synchronized normalization statistics.

### Network

- Planner: compact MLP, ~**3.5 M params**, ego MLP + Deep Sets encoders for partners/road/traffic, 3-layer 1024-unit shared trunk, discrete jerk action.
- Sim agent: ~**6.7 M params**, shared trunk + per-type heads (jerk for vehicles, unicycle grid for pedestrians, compact bicycle grid for cyclists).

### Key design choices for stability at scale

1. **Compute efficiency over sample efficiency**: discard low-advantage segments, spend compute on aggressive off-policy corrections.
2. **Population play + domain randomization** breaks self-play symmetry instead of a frozen-checkpoint league.
3. **Reward/kinematic conditioning** lets one network adapt to diverse regimes online.
4. **Goal dropout** prevents over-reliance on routed goals.
5. **Synchronized normalization stats** across ranks.

### Compute budget

- Planner: 16 × NVIDIA A100 80 GB.
- Sim agent: 32 × NVIDIA A100 80 GB.
- Trained from scratch (random initialization), no log/hybrid bootstrapping for reported results.

---

## 4. Benchmarks & Results

### InterPlan long-tail (80-scenario split)

| Method | Type | Score ↑ |
|--------|------|---------|
| SPDM (Np=60) | Rule | 63.66 |
| HybridLLMPlanner (Llama-7B) | Hybrid | 53.0 |
| PPO | RL | 42.1 |
| **TerraZero** | **RL** | **67.87** |

First fully learned policy to top InterPlan. Same checkpoint scores **67.71** on the larger 335-scenario set.

### nuPlan val14 closed-loop reactive

| Method | Type | Score ↑ | Notes |
|--------|------|---------|-------|
| Gigaflow | RL | 93.8 | Prior best RL, synthetic maps |
| SPDM (Np=15) | Rule | 92.28 | Best overall; weak on long-tail |
| **TerraZero** | **RL** | **92.27** | **Best safety metrics (No-AF-Coll 99.11, TTC 96.06)** |

### Waymo Open Sim Agents

- **WOSAC 2023** full validation: TerraZero 0.632 vs Gigaflow 0.619 (both demonstration-free).
- **WOSAC 2024 vehicles** (880-scenario subset): TerraZero 0.740, matches SPACeR 0.741, lowest collision/off-road rates.
- **WOSAC 2024 VRUs**: TerraZero 0.683, trails SPACeR 0.729 but beats PPO 0.648 and HR-PPO 0.668.
- Zero-shot nuPlan→Waymo transfer nearly matches Waymo-trained policy.

### Generalization / transfer

- Cross-dataset transfer matrices show performance is determined by **evaluation target**, not training source.
- Policy trained only on right-hand-traffic cities generalizes to left-hand-traffic Singapore.
- CARLA-trained policy (5 synthetic towns) performs on par with policies trained on hundreds of thousands of real scenarios.

---

## 5. Relationship to SPACeR and TerraTransfer

### SPACeR (arXiv 2510.18060, ICLR 2026)

- Same research group (Applied Intuition); overlapping authors.
- **Different method**: SPACeR anchors self-play to a pretrained tokenized reference model (SMART/CAT-K) via likelihood reward + KL divergence.
- SPACeR requires a reference policy trained on logged data; TerraZero uses **no reference policy at all**.
- SPACeR policies are tiny (~65 K params) and specialized for sim agents; TerraZero policies are larger and serve both planner and sim-agent roles.
- TerraZero project page links to SPACeR, treating it as related prior work.
- **No public SPACeR source code** either—only project page and paper.

### TerraTransfer (arXiv 2606.17386)

- Follow-up by overlapping authors.
- Explicitly states it uses **TerraZero as its vectorized simulator** for Phase 1 self-play.
- Phase 2 aligns a vision backbone (DINOv3) to the privileged-state self-play policy.
- Project page: https://zikang-xiong-ai.github.io/terratransfer
- **Confirms TerraZero is a real internal codebase shared across Applied Intuition projects.**
- **No public TerraTransfer source code**.

### Same codebase?

TerraTransfer calls TerraZero "our vectorized simulator," and the TerraZero project page links to TerraTransfer. The three papers share authors and the same high-level simulator design, so they are almost certainly built on a common internal stack. However, **no public repository contains the simulator or training code**, so this cannot be independently verified.

---

## 6. Public Code & Artifacts

### Code

| Resource | URL | What it contains |
|----------|-----|------------------|
| TerraZero project page | https://terra-applied.github.io/TerraZero | HTML demo page, links to paper/videos |
| GitHub `terra-applied/TerraZero` | https://github.com/terra-applied/TerraZero | GitHub Pages source only (`index.html`, `assets`) |
| GitHub `akshay-rangesh-ai/TerraZero` | https://github.com/akshay-rangesh-ai/TerraZero | PDF mirror only |
| HuggingFace `woo-who/terrazero-assets` | https://huggingface.co/datasets/woo-who/terrazero-assets | PDF, benchmark videos, `terra-series` videos |
| SPACeR project page | https://spacer-ai.github.io | HTML demo page |
| GitHub `spacer-ai/spacer-ai.github.io` | https://github.com/spacer-ai/spacer-ai.github.io | GitHub Pages source only |
| TerraTransfer project page | https://zikang-xiong-ai.github.io/terratransfer | HTML demo page |

**Conclusion: no public source code, model weights, terrabin converters, or replay/evaluation scripts as of 2026-07-16.**

### Artifacts preserved in this thread

- `assets/terrazero-technical-report.pdf` — technical report PDF (identical to arXiv paper in content).
- `assets/terrazero-technical-report.txt` — extracted text for searching.
- `assets/sample-interplan-jaywalking.mp4` — example InterPlan jaywalking scenario video.
- `papers/world-models/terrazero-2607.13028.md` — original paper markdown (read-only source).

---

## 7. Claims: Supported vs Unsupported

| Claim | Evidence | Verdict |
|-------|----------|---------|
| C engine + zero-copy + CPU/GPU split | Detailed in paper §3.1 and Figure 2; consistent with throughput plausibility | Supported by text, not independently verified |
| 1.3M SPS on server GPU | Reported in paper, project page; sanity-check plausible | Plausible but unverified |
| Heterogeneous agents, multiple dynamics, signals | Detailed specs in §3.2–3.3, Tables 7–12 | Supported by text |
| Procedural generation from real maps | Algorithm 1 and §4 describe full pipeline | Supported by text |
| Zero demonstrations | Repeated claim; training config uses random init | Supported by text |
| InterPlan SOTA | Table 3 | Supported by reported numbers |
| val14 safety leader | Table 2 | Supported by reported numbers |
| WOSAC demonstration-free leader | Tables 4–6 | Supported by reported numbers; trails reference-anchored SPACeR on VRU realism |
| Cross-dataset / LHD transfer | Figure 6 | Supported by reported numbers |
| Public code availability | GitHub/HF searches find only pages/PDFs/videos | **Unsupported / false** |

---

## 8. Blockers & Open Questions

1. **No public code** — impossible to verify throughput, reproduce training, or inspect implementation details (buffer layout, exact V-trace integration, signal-state machine code, etc.).
2. **No released model weights** — cannot evaluate the policy on new scenarios or run ablations.
3. **No `terrabin` converters or dataset** — cannot regenerate the training/evaluation scenarios.
4. **Server GPU unspecified** in throughput chart; inferred A100 from training section but not confirmed.
5. **Benchmark comparison simulators re-benchmarked by authors** under their setup where not otherwise marked (Table 1).
6. **Single-number SPS** does not capture per-configuration variance (agent count, NPC mix, map density, observation budgets all affect throughput).
7. **Vision bridge** — paper is object-level only; end-to-end perception is left to TerraTransfer.
8. **Sim-to-real dynamics gap** — acknowledged limitation; kinematic randomization helps but does not resolve tire/suspension/aero physics.

---

## 9. What the Parent Report Should Say

TerraZero is a credible, well-engineered attempt to close the throughput–fidelity gap for RL-scale autonomous-driving simulation. Its most important technical contributions are:

1. A **fast C object-level simulator** that keeps features (heterogeneous agents, dynamics, signals, rules) usually found only in slower environments.
2. A **procedural scenario generator** that turns real-world maps into an unbounded long-tail training distribution without relying on logged trajectories.
3. A **stable zero-demonstration self-play recipe** combining PPO, V-trace, PopArt, and priority sampling, producing policies that transfer zero-shot across cities and even driving handedness.

The empirical results are strong and internally consistent: it tops InterPlan, is safe on val14, and is the best demonstration-free method on WOSAC. The comparison with SPACeR is particularly informative: TerraZero shows you can get near-SPACeR vehicle realism without the expensive pretrained tokenized reference model, but SPACeR still leads on VRU realism, likely because its reference model captures human pedestrian/cyclist motion more faithfully.

**Caveats to flag:** no public code or weights; the 1.3M SPS number is author-reported and plausible but unverified; the lean observation set is a design strength but also means the policy does not solve perception. The relationship to TerraTransfer confirms the simulator is real and in active internal use, but external reproducibility is currently zero.

---

## Files in this thread

```
terrazero-2607.13028/
├── README.md                              # this file
├── assets/
│   ├── terrazero-technical-report.pdf     # downloaded technical report
│   ├── terrazero-technical-report.txt     # extracted text
│   └── sample-interplan-jaywalking.mp4    # example scenario video
└── code/
    ├── architecture_diagram.md            # system architecture notes + Mermaid
    ├── scenario_generation_notes.md       # procedural generation breakdown
    ├── training_recipe_notes.md           # RL algorithm + network + hyperparams
    ├── results_comparison_table.md        # benchmark tables and takeaways
    ├── related_work_comparison.md         # Waymax/Nocturne/SMARTS/GPUDrive/PufferDrive/Gigaflow/SPACeR
    ├── code_availability_probe.py         # script checking public repos/HF dataset
    └── throughput_sanity_check.py         # plausibility check for 1.3M SPS claim
```
