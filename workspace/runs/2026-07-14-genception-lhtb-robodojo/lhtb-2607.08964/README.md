# Long-Horizon-Terminal-Bench: how do you grade a task that takes ninety minutes?

**Run packet:** `runs/2026-07-14-genception-lhtb-robodojo/lhtb-2607.08964`  
**Paper:** arXiv 2607.08964, *Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading*  
**Project page:** https://zli12321.github.io/LHTB/  
**Official repo:** https://github.com/zli12321/LHTB (cloned into `code/lhtb-repo/`)

## Research question

LHTB claims that long-horizon terminal agents need a fundamentally different evaluation signal than short-horizon benchmarks: not binary pass/fail, but dense partial credit over subtasks. I wanted to make that claim concrete by inspecting the actual benchmark mechanics — the Harbor task format, Docker containers, subtask graders, reward formulas, the 46 tasks, hidden verifiers, the Terminus-2 harness, and the leaderboard — and then ask three things:

1. What does dense partial-credit grading actually buy over binary pass/fail at this difficulty level?
2. Are the tasks and verifiers sound (resist memorization, shortcutting, reward hacking)?
3. What failure modes dominate, and what do they imply for agent design?

## What I found at a glance

- The benchmark is real and inspectable. The repo contains 46 Harbor tasks, each with `task.toml`, `instruction.md`, environment, solution, and hidden tests. I read the verifiers for `rush_hour_campaign`, `unison-paper-reproduction`, and `vector-db-iterative-build`, and the reward formulas are explicit, continuous, and in some cases already patched once (the vector-DB reward was rescaled to avoid saturating at 10× speedup).
- All 46 `task.toml` files mark difficulty as `"hard"`, but the paper's Easy/Hard split is empirical, derived from mean model reward. That is a sensible choice: difficulty is not an author label but a measured quantity.
- The repo categories are finer-grained than the paper's nine display categories. The paper's taxonomy (Interactive games, Multimodal & imaging, Software & reverse engineering, Scientific computing, Earth/climate/energy, Systems/performance/security, Research reproduction & ML, APEX workflows, Logic & puzzles) is what matters for reporting; the repo uses 19 low-level tags.
- The published results have evolved since the paper. The arXiv version reports 15 models with GPT-5.5 leading (15.2% at R≥0.95). The project page and repo README report 18–21 models, with Grok 4.5 leading (13/46 = 28% at R≥0.95, mean reward 0.505). The underlying design and task set are unchanged; only the model sweep grew.
- Dense rewards are doing exactly the work the paper claims. At R≥1.0 ten of fifteen paper-reported models solve zero tasks; at R≥0.95 the mean pass rate is only 4.3%. Without partial credit the leaderboard would be a sea of zeros and large tie groups.
- The verifiers are intentionally hard to hack: they replay evidence from the final container state, use hidden seeds/configs, forbid hard-coded constants, check determinism, and in some cases inspect the agent's terminal recording for forbidden actions.
- The dominant failure mode is timeout while still making progress (79% of unresolved runs), not local execution errors. A secondary but important mode is false finish: the agent stops early, often with R≥0.75, because it cannot verify that hidden checks remain unsatisfied.

## Mechanism: how a task is built

### Harbor / Terminal-Bench format

Each task is a Dockerized terminal environment plus a hidden verifier. The repo's expected layout is:

```
tasks/<task-id>/
├── task.toml          # metadata, timeouts, resources, docker image
├── instruction.md     # agent-facing natural-language goal
├── environment/       # Dockerfile + assets + starter project
├── tests/             # hidden verifier (pytest, custom script, or both)
└── solution/          # reference / oracle solution
```

The harness is [Harbor](https://github.com/laude-institute/harbor) with the **Terminus-2** agent. The full-benchmark config (`configs/examples/full_benchmark.yaml`) runs all 46 tasks with one model, `parser_name: json`, proactive summarization at 8k tokens, terminal session recording, and a 5,400-second (90-minute) agent timeout per task.

### Task-level metadata

My probe parsed all 46 `task.toml` files. Key facts:

- **46 tasks, 27 of them give the agent ≥90 minutes** (`timeout_sec ≥ 5400`). The rest are shorter (e.g., `rush_hour_campaign` gives the agent 60 minutes and the verifier 60 minutes).
- **Mean estimated expert time: 231 minutes; mean estimated junior time: 575 minutes.** The benchmark is deliberately expert-level and long.
- **All tasks are tagged `difficulty = "hard"` in the repo metadata.** The paper's Easy/Hard labels come from the empirical mean reward across models (Easy if mean reward ≥ 0.5). This is an important distinction: the benchmark authors treat difficulty as a measured property, not a design label.

The paper's nine reporting categories and counts:

| Category | Count |
|---|---|
| Interactive games | 4 |
| Logic & constraint puzzles | 3 |
| Software & reverse engineering | 7 |
| Scientific computing & simulation | 6 |
| Earth, climate & energy | 6 |
| Multimodal & imaging analysis | 6 |
| Systems, performance & security | 5 |
| Research reproduction & ML | 5 |
| APEX professional workflows | 4 |

No category dominates; the largest is only 7/46.

### Subtask-based grader and reward formula

The grader runs *inside* the container after the agent stops. Each task is decomposed into subtasks `{s_1, …, s_K}` with weights `{w_1, …, w_K}` and normalized sub-scores `r_k ∈ [0, 1]`. The overall reward is

```
R = Σ_k w_k r_k / Σ_k w_k
```

The paper describes three subtask types:

1. **Binary subtasks** — strict Boolean checks (unit tests pass, service responds, script exits cleanly).
2. **Continuous / thresholded subtasks** — linear or tapered partial credit as a metric approaches a target.
3. **Episode-aggregating subtasks** — fraction of episodes/levels that trigger an environment success flag, used for games and campaign-style tasks.

A task is counted as *resolved* if `R ≥ 0.95` (the relaxed threshold) or *perfect* if `R = 1.0`.

I inspected three verifiers to see how this plays out:

#### 1. `rush_hour_campaign/tests/verifier.py`

A hand-played Rush Hour campaign with four 6×6 puzzles. The verifier:
- Parses a JSON submission with move routes per puzzle.
- Simulates every move, checking syntax, board bounds, and collisions.
- Scores each puzzle as
  ```
  score = 0.10 * legal_progress
        + 0.65 * solved_within_max_moves
        + 0.25 * solved_within_target_moves
  ```
  where `legal_progress = min(legal_prefix_length / optimal_moves, 1.0)`.
- Caps the score at 0.2 if the agent is caught using code/scripts/solvers (regex checks on the submission text and the terminal recording).

This is a clean example of dense partial credit: even a few legal moves earn something, but full credit requires solving every puzzle within strict move caps without computational aid.

#### 2. `unison-paper-reproduction/tests/test_outputs.py`

Reproduce a calibrated UNISON fat-tree experiment. The verifier:
- Checks that required source modules exist (`topology.py`, `partition.py`, `scheduler.py`, `simulator.py`, `report.py`, `cli.py`).
- Forbids hard-coded golden numbers and direct reads of `/tests` fixtures.
- Runs a public ground-truth config and checks metrics within tolerance.
- Checks that LP partitioning is non-trivial (`lp_count > thread_count`) and load-balanced.
- Checks same-seed determinism via SHA-256 checksum.
- Runs a hidden config with different seeds, thread counts, and scheduler to ensure generalization.

This shows the hidden-verifier pattern: public checks are easy to pass, but most of the reward is in hidden stress tests that require a real implementation.

#### 3. `vector-db-iterative-build/tests/test_outputs.py`

Build an approximate nearest-neighbor search service. The verifier:
- Generates a 100k-vector hidden dataset.
- Measures recall@10 and QPS against the running agent-built server.
- Computes speedup over brute-force search.
- Maps recall/speedup to a continuous reward with explicit bands:
  - recall < 0.50 → 0.0
  - recall ∈ [0.50, 0.95) → 0.1 × (recall / 0.95)
  - recall ≥ 0.95, speedup < 1× → 0.3
  - recall ≥ 0.95, speedup ∈ [1×, 2×) → 0.3 + 0.2×(speedup−1)
  - recall ≥ 0.95, speedup ∈ [2×, 5×) → 0.5 + 0.3×(speedup−2)/3
  - recall ≥ 0.95, speedup ≥ 5× → 0.8 + 0.2×log_ramp up to 1.0 at 100×.

The file includes a comment explaining that the top band was recently rescaled because the old metric saturated at 10×: "a borderline 10× scored the SAME as an excellent 68× — the metric was blind at the high end." That is direct evidence the authors are actively tuning the reward to preserve discriminative signal at the frontier.

## Research question 1: what does dense partial-credit grading buy?

A lot, at this difficulty level. The paper reports that across 15 models:

- At `R ≥ 1.0`, **10 models solve zero tasks**.
- At `R ≥ 0.95`, the **mean pass rate is 4.3%** (30 passes out of 690 model-task runs).
- **62.8% of runs** achieve partial reward `R ∈ [0.05, 0.95)`.
- **26.1% of runs** reach `R ≥ 0.5`.
- The near-miss band `R ∈ [0.85, 0.95)` contains almost twice as many runs as full passes.

Under binary pass/fail the leaderboard would collapse: ten models tied at zero, and the vast majority of runs indistinguishable from total failure. Mean reward and pass rate are only moderately correlated (Spearman ρ = 0.56), so the two metrics rank models differently. Dense rewards are not a cosmetic tweak; they are the only reason the benchmark produces a gradient at current capability levels.

The updated project-page leaderboard (21 models) tells the same story in starker numbers:

- Best model: Grok 4.5, mean reward 0.505, 13/46 solved (28%).
- 29 of 46 tasks have never been solved by any model.
- ~55% of all runs land below R = 0.25.

## Research question 2: are the tasks and verifiers sound?

Mostly yes, and the repo lets you check. The benchmark uses several anti-gaming mechanisms:

- **Hidden tests dominate the reward.** Public checks validate file formats, CLI behavior, and simple examples with low weight. Most reward comes from hidden stress cases.
- **Dynamic hidden inputs:** nested manifests, gzip+base64 wrappers, renamed fields, missing values, injected noise, rotated/cropped images, anomalous frames, alternative coordinate/time conventions.
- **Replay-based grading for games:** the agent's move log is replayed in a fresh seeded engine; self-reported progress does not count.
- **Anti-shortcut checks:** forbidden snippets, bans on reading `/tests` fixtures, determinism checks, hidden seeds.
- **Process monitoring:** `rush_hour_campaign` scans the terminal recording and submission for evidence that the agent wrote or ran a solver.

There are still things I would want to see in a full audit:

- I did not run the Docker images or the oracle smoke test, so I cannot confirm all 46 build cleanly or that every verifier is free of false negatives.
- Some tasks (e.g., APEX workflows) rely on large external assets (Git LFS zips, videos). I did not inspect those.
- The reward weights per subtask are not always easy to read from the verifier files; they are sometimes embedded in the test logic rather than exposed as a clean weight vector. That makes sensitivity analysis harder.
- The fact that the repo marks every task `"hard"` while the paper derives Easy/Hard empirically is fine, but it means there is no canonical difficulty metadata independent of model performance.

## Research question 3: failure modes and what they imply

The paper's failure-mode decomposition is the most actionable part of the work.

### Timeouts dominate

Across all unresolved runs (`R < 0.95`):

- **79% are timeouts** — the agent is still working when the 90-minute budget expires.
- **19% are early exits** — the agent stops on its own.
- **3% are harness errors**.

Crucially, the timed-out runs are not on the verge of completion: their mean reward ranges from 0.10 to 0.35 across models. The binding constraint is not "can the agent execute a correct step?" but "can it budget a long horizon and finish?"

### False finishes expose weak self-verification

Among 124 early exits, 14 have `R ≥ 0.75`. The agent stopped with substantial time remaining, thinking it was done. Examples:

- Kimi K2.7 Code on `duckdb-optimizer-closure` at R = 0.92.
- GLM 5.2 on `apex-ib244-matter` at R = 0.90.
- Seven models on `apex-law433-matter` between R = 0.80 and 0.87.

This is the long-horizon analogue of the verification failures Terminal-Bench 2.0 identified. After fixing visible errors, the agent must hunt residual defects that have no visible signal. Current agents systematically overestimate completion and under-invest in final verification.

### Implications for agent design

- **Planning and progress tracking:** agents need to maintain and update plans over hundreds of steps, not just react to the last observation.
- **Memory and state preservation:** earlier decisions are forgotten; state drifts.
- **Calibrated stopping:** models need better self-verification, or an explicit "am I really done?" loop against hidden requirements.
- **Efficiency, not just capability:** rankings reflect how much reward a model can accumulate in a fixed budget. Reducing redundant exploration and verification loops may be as valuable as stronger single-step reasoning.

## External signals

- **Official repo cloned** to `code/lhtb-repo/`. I inspected the harness config, two task `task.toml` files, and three verifiers (`rush_hour_campaign`, `unison-paper-reproduction`, `vector-db-iterative-build`).
- **Project page and leaderboard fetched** from https://zli12321.github.io/LHTB/ and https://zli12321.github.io/LHTB/leaderboard.html.
- **Probe written:** `code/probe_lhtb.py` parses all 46 `task.toml` files, the paper appendix task list, and the leaderboard snapshot, then emits JSON/CSV summaries and four plots saved to `assets/lhtb/`.

## Limitations and blockers

- I did not run the actual benchmark (no API keys, no Docker builds executed). Findings about verifier soundness are based on code inspection, not empirical probing.
- The paper, project page, and repo README report slightly different model counts (15, 18, and 21 respectively) and different leading models. I attribute this to the paper being a snapshot and the website being updated; the task set is the same.
- The `task.toml` difficulty field is uniform (`"hard"`), so the probe's difficulty analysis relies on the paper's empirical Easy/Hard split rather than repo metadata.
- Large APEX assets are stored via Git LFS; I did not pull them.

## Takeaway

Long-Horizon-Terminal-Bench is a well-engineered stress test for the thing that actually limits today's agents: sustained execution. The dense reward design is not an afterthought — it is the measurement strategy that makes the benchmark usable at all, because binary grading would turn 46 hard tasks into a flat zero for most models. The verifiers are concrete, hidden, and replay-based, and the repo is open enough to inspect them. The central result is that frontier agents can take many correct local steps but still fail to finish, either by timing out with modest partial progress or by stopping early with weak self-verification. For agent builders, the implication is clear: the next gains are likely to come from better long-horizon planning, memory, progress tracking, and calibrated stopping, not just stronger next-token prediction.

## Preserved evidence

- `code/probe_lhtb.py` — analysis script.
- `code/lhtb-repo/` — cloned official LHTB repository.
- `assets/lhtb/lhtb_summary.json` — task count, category, difficulty, and timeout summary.
- `assets/lhtb/lhtb_tasks.csv` — per-task metadata from repo.
- `assets/lhtb/lhtb_leaderboard.csv` — July 2026 leaderboard snapshot.
- `assets/lhtb/lhtb_category_counts.png` — task counts by repo category.
- `assets/lhtb/lhtb_difficulty_heatmap.png` — difficulty by category (note: all repo tasks are "hard").
- `assets/lhtb/lhtb_leaderboard_solved.png` — solved count per model.
- `assets/lhtb/lhtb_cost_vs_reward.png` — cost vs. mean reward scatter.
- `assets/lhtb/fig1.png` through `fig6.png` — paper figures downloaded during initial scouting.
