# RoboDojo: unified sim-and-real diagnosis for generalist manipulation

**Thread:** `runs/2026-07-14-genception-lhtb-robodojo/robodojo-2607.04434`  
**Paper:** `papers/robotics/robodojo-2607.04434.md` (arXiv 2607.04434)  
**Project page:** https://robodojo-benchmark.com/

## Research questions

1. How does RoboDojo move beyond existing manipulation benchmarks (CALVIN, LIBERO, RoboTwin, RMBench, SimplerEnv, ManipulationNet, RoboChallenge, RoboArena)?
2. What capability gaps does the leaderboard reveal between the best policies and human teleoperation?
3. How transferable are simulation-trained policies to the real-world benchmark, and what does the multi-embodiment design show?

## What RoboDojo actually is

RoboDojo is not just another task collection. It is an attempt to build a **diagnostic loop** that is missing from prior robot-learning evaluation:

- **Simulation side:** 42 tasks on a bimanual ARX X5 platform, split into five capability dimensions — Generalization, Memory, Long-Horizon, Precision, and Open — rather than object/layout reskins. The design explicitly avoids training-test leakage: the Open dimension has no task-specific demonstrations, and Generalization training adds auxiliary domain-randomized "DLC" trajectories that do not contain evaluation task solutions.
- **Real-world side:** 18 tasks across three embodiments (ARX X5, Piper, Piper X) evaluated through `RoboDojo-RealEval`, a standardized hardware station with fixed robot/camera/lighting geometry, a touchscreen reset interface, layout-replay overlays, and cloud-based remote access.
- **Integration layer:** `XPolicyLab` defines a single policy interface (`update_obs`/`get_action`/`reset` plus batched variants) and a WebSocket/MessagePack client-server protocol so the same policy adapter runs in simulation and on the real robot.

Compared with the benchmarks it cites (Appendix B of the paper), RoboDojo is the only one that combines **bimanual tasks, heterogeneous parallel simulation, a standardized multi-embodiment real-world evaluation station, and a unified policy integration layer** in one package. CALVIN and LIBERO are simulation-only and do not stress precision or open-vocabulary transfer. RoboTwin and RMBench focus on single dimensions (generalization and memory). SimplerEnv is sim-to-real correlation rather than a paired deployment benchmark. ManipulationNet, RoboChallenge, and RoboArena are real-world-only and lack the reproducible hardware/evaluation protocol that `RoboDojo-RealEval` enforces.

## External signals inspected

I treated the paper as a seed and went to the artifacts:

1. **Cloned the public repositories** into `code/RoboDojo` and `code/XPolicyLab` (shallow clones, depth 1).
2. **Inspected task specification.** `task/RoboDojo/config/_task.yml` sets default `eval_nums: 50` and per-task overrides; `insert_tubes.yml` and `match_and_pick_from_conveyor.yml` show how object categories, spawn ranges, selection modes (`same`, `unique`, `same_as_label`), and per-instance labels are declared in YAML rather than hard-coded.
3. **Traced heterogeneous parallel simulation.** `env/environment/base_env.py` launches Isaac Lab with `replicate_physics=False`; `env/scene_manager/scene_manager.py` then instantiates distinct object sets per environment index from seed-controlled layouts. `src/eval_client/eval_env.py` adds batched video streaming, resume manifests, PhysX-broken-env detection, and action-dict validation.
4. **Read the policy interface.** `XPolicyLab/model_template.py` and `policy/demo_policy/model.py` define the adapter contract (action keys like `left_arm_joint_state` / `left_ee_pose`, dimensions inferred from `utils/robot/_robot_info.json`), and `client_server/ws/model_client.py` is the environment-side WebSocket client.
5. **Visited the project page.** It confirms the leaderboard exists but is JS-rendered (I could not scrape it). It lists 40+ policies in XPolicyLab, while the paper freezes the leaderboard at 30 simulation / 10 real-world policies on 3 Jul 2026. Notably, the project page still labels XPolicyLab as "Coming Soon" even though the GitHub repository is already public.
6. **Built a leaderboard probe.** `code/leaderboard_probe.py` parses the simulation and real-world tables in the paper markdown, ranks policies, computes per-dimension gaps to human teleop, and produces the CSV/PNG artifacts in `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/`.

Detailed code notes are in `code/inspection_summary.md`.

## Findings

### 1. The human gap is enormous and uneven

Using the probe on the paper's frozen tables (3 Jul 2026):

| Setting | Best policy success | Human teleop | Gap |
|---|---:|---:|---:|
| Simulation (average) | 8.80% (Hy-Embodied-0.5-VLA) | 76.03% | **67.23 pp** |
| Real world (overall) | 12.80% (π₀.₅) | 100.00% | **87.20 pp** |

Per-dimension simulation gaps to human teleop:

| Dimension | Human success | Best policy | Absolute gap | Relative gap |
|---|---:|---:|---:|---:|
| Generalization | 87.83% | 9.33% (Spatial Forcing) | 78.50 pp | 89.4% |
| Precision | 64.00% | 12.00% (X-VLA) | 52.00 pp | 81.2% |
| Long-Horizon | 74.25% | 14.92% (Hy-Embodied-0.5-VLA) | 59.33 pp | 79.9% |
| Memory | 74.33% | 12.11% (Hy-Embodied-0.5-VLA) | 62.22 pp | 83.7% |
| Open | 79.75% | 1.67% (π₀.₅) | 78.08 pp | 97.9% |

**Interpretation:** Even the strongest policies operate in a low-score regime. Open-vocabulary / skill-recombination tasks are essentially unsolved. Precision and memory are the next biggest bottlenecks. The leaderboard is fragmented: Spatial Forcing wins Generalization, X-VLA wins Precision, Hy-Embodied-0.5-VLA wins Long-Horizon, Memory, and Average. No single method dominates, which is exactly the diagnostic signal RoboDojo is designed to produce.

### 2. Scene randomization causes broad collapse

The probe does not parse the standard-vs-random split directly, but the paper's Table 3 shows that most policies lose 65–100% of their standard-setting score when backgrounds, lighting, clutter, and object layouts are randomized. Hy-Embodied-0.5-VLA drops from 21.98 to 1.57 (92.9% relative loss); Spatial Forcing retains more but still only reaches 6.98 random score. This confirms that visual-spatial grounding helps but is nowhere near sufficient for robust deployment.

### 3. Simulation and real-world rankings are only weakly aligned

Among the 10 policies evaluated in both settings:

- π₀.₅ is 3rd in simulation and 1st in real world.
- InternVLA-A1 is 17th in simulation but 2nd in real world.
- GalaxeaVLA is 9th in simulation but 3rd in real world.
- X-VLA is 4th in simulation but 5th in real world.

The real-world benchmark is **not** a paired sim-to-real transfer test; tasks are intentionally not one-to-one aligned. It is a deployment stress test, and the re-ranking shows that simulation success does not automatically imply physical robustness. The multi-embodiment split further shows that performance is not uniform: π₀.₅, for example, succeeds on several Piper tasks but nearly fails on Piper X tasks, suggesting embodiment-specific calibration and action-stability issues rather than pure task understanding.

### 4. Real-world failure modes go beyond success rate

The paper reports action jitter, oscillatory motions, contact instability, and occasional safety-critical behavior (e.g., DM0 producing unstable control signals). This matters because a policy can score partial credit while being physically unsafe — a signal that aggregate metrics alone miss. `RoboDojo-RealEval` is designed to surface this by releasing evaluation videos and using three independent raters.

## Limitations and blockers

- **Could not run the simulator.** The RoboDojo repo is eval-only and requires asset downloads (`scripts/init_assets.sh`) plus an Isaac Sim / Isaac Lab installation that is not available here.
- **Could not evaluate policies.** Running any policy needs checkpoints, policy-specific conda/uv environments, and GPU hardware.
- **Could not reproduce the throughput claim.** The 1.94× heterogeneous-parallel speedup (Table 4) was measured on 8×RTX 4090 GPUs; I only traced the implementation.
- **Latest leaderboard not accessible.** The project page leaderboard is JS-rendered and could not be fetched; the probe uses the frozen paper tables.
- **XPolicyLab status mismatch.** The website still says "Coming Soon" while the GitHub repo is public and functional. This is minor, but it is a documentation inconsistency.

## Takeaway

RoboDojo is the most carefully constructed unified sim-and-real manipulation benchmark I have seen: it has a config-driven task engine, genuinely heterogeneous parallel simulation, a reproducible real-world evaluation station, and an adapter framework that already integrates 30–40 policies. The leaderboard it produces is not flattering for the field. The best simulation policy is 67 percentage points below human teleop, the best real-world policy is 87 points below human, and Open-vocabulary manipulation is near zero across the board. Most importantly, **simulation rank is a noisy predictor of real-world deployability**: policies that look mediocre in sim can outperform stronger sim policies on physical hardware, and vice versa. For the report, the key claim should be that RoboDojo exposes both a large capability gap and a measurement gap — current metrics in simpler benchmarks were hiding how far generalist manipulation still is from reliable physical execution.

## Artifacts produced

- `code/leaderboard_probe.py` — parses paper tables and computes gaps/rankings.
- `code/inspection_summary.md` — notes from the code inspection.
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/sim_ranking.csv`
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/sim_gaps.csv`
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/real_ranking.csv`
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/sim_real_overlap.csv`
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/sim_top10_success.png`
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/sim_dimension_gaps.png`
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/sim_real_scatter.png`
- `runs/2026-07-14-genception-lhtb-robodojo/assets/robodojo/summary.md`
