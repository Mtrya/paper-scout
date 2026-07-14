# RoboDojo / XPolicyLab code inspection notes

I cloned shallow copies (depth 1) of the public repositories into this thread's `code/` directory, inspected the files listed below, and then removed the raw clones per the workspace artifact policy (durable notes and the probe remain). The repositories are:

- https://github.com/RoboDojo-Benchmark/RoboDojo
- https://github.com/XPolicyLab/XPolicyLab

The RoboDojo repository is **eval-only** in this release: it ships the simulator client, benchmark tasks, configs, and result aggregation, while policy integration lives in XPolicyLab. This split is deliberate and matches the paper's claim of a unified interface.

## What I looked at

### RoboDojo simulator and task specification

- `task/RoboDojo/config/_task.yml` — default eval counts (`eval_nums: 50` for most tasks, with per-task overrides), scene/robot/camera defaults, and a task-level override table.
- `task/RoboDojo/config/insert_tubes.yml` and `match_and_pick_from_conveyor.yml` — concrete YAML task specs defining rigid/dynamic object spawn ranges, category indices, selection modes (`same`, `unique`, `same_as_label`), and per-object labels used by reward checks.
- `task/RoboDojo/task_registry.py` — dynamic task loader (`task.{BENCHMARK}.tasks.{task_name}`) that lets new tasks be added without touching the evaluation harness.
- `env/environment/base_env.py` — Isaac Lab launch logic. Key detail for heterogeneous parallelism: `InteractiveSceneCfg(..., replicate_physics=False)` in `launch_sim()`.
- `env/environment/isaac/isaac_rl_env.py` — thin `CustomDirectRLEnv` subclass that calls `self.scene.clone_environments(copy_from_source=True)` in `_setup_scene`.
- `env/scene_manager/scene_manager.py` + `env/scene_manager/layout_manager.py` — the actual heterogeneous-instantiation machinery. `SceneManager.initialize()` loops over `env_idx`, sets the per-env seed, and calls `spawn_scene_objects(env_id, ...)` so each parallel environment gets its own object set, counts, articulation structures, and clutter layouts derived from seed-controlled YAML configs.
- `src/eval_client/eval_env.py` — the simulator-side evaluation wrapper. It manages batched video streaming, resume manifests, per-env seed lists, PhysX-broken-env detection, action validation (`validate_action_dict`), and the websocket policy client (`WsModelClient`).
- `scripts/robodojo.sh` + `scripts/eval_policy.sh` — the public entry points for single-machine, split-machine, smoke, benchmark, and summarize workflows.

### XPolicyLab policy interface

- `model_template.py` — minimal required API: `__init__`, `update_obs`, `update_obs_batch`, `get_action`, `get_action_batch`, `reset`.
- `policy/demo_policy/model.py` — reference adapter showing how a policy maps robot `arm_dim`/`ee_dim` metadata to action dictionaries (`left_arm_joint_state` / `left_ee_pose`, etc.) for both single- and dual-arm robots.
- `client_server/ws/model_client.py` — synchronous environment-side adapter that talks to a policy server over WebSocket; supports `reset`, `update_obs`, `get_action`, and batched variants.
- `README.md` — documents the observation/action data format (poses as `[x,y,z,qw,qx,qy,qz]`, multi-camera `vision/` dict, `state/` dict), the `deploy.yml` runtime config, and the train/eval script naming convention.

## Key takeaways from the code

1. **Heterogeneous parallel simulation is real but not magic.** The code achieves it by (a) disabling physics replication (`replicate_physics=False`) and (b) explicitly constructing per-environment object sets in `SceneManager`. The vectorized batched stepping interface is preserved, but the scene content is not a single cloned template.
2. **XPolicyLab is the actual integration layer.** RoboDojo assumes only that each policy directory contains `eval.sh` and `deploy.yml`; everything else (model loading, observation preprocessing, action formatting) lives in XPolicyLab adapters. This matches the paper's "integrate once, evaluate everywhere" pitch.
3. **The eval harness is production-minded.** Streaming video writers, resume manifests, PhysX fatal-error handling, unstable-layout detection (`check_layout_stability`), and retry loops are all present, which suggests the reported leaderboard numbers are backed by a carefully instrumented pipeline rather than a quick ad-hoc script.

## Blockers / what I could not run

- I did **not** download the 3D assets or run Isaac Sim. The repository requires asset installation via `scripts/init_assets.sh` / `download_data.sh`, and the Isaac Sim/Isaac Lab stack is not installed in this environment.
- I did **not** start any policy servers or real-robot clients; doing so would need policy checkpoints, conda/uv environments, and GPU/robot hardware.
- I did **not** verify that the heterogeneous-parallel speedup numbers in Table 4 reproduce; that would require an 8×RTX 4090 (or equivalent) setup.

These limitations are noted in the thread README and affect only the physical reproduction claims, not the leaderboard gap analysis (which is computed directly from the paper's tables).
