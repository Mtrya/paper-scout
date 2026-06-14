# EurekAgent deep-dive thread

**Paper:** *EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery* (arXiv 2606.13662)  
**Run:** 2026-06-14-weaver-eurekagent-repwam  
**Thread ID:** eurekagent-2606.13662

## What was attempted

1. Read the full paper (including appendix) from `papers/agents/eurekagent-2606.13662.md`.
2. Cloned the official implementation from https://github.com/THU-Team-Eureka/EurekAgent into `code/eurekagent-2606.13662/`.
3. Traced the prepare–propose–implement loop and the four environment-engineering dimensions in code:
   - `src/pipeline.py`, `src/graph.py`, `src/nodes/{prepare,propose,implement}_node.py`
   - `src/docker/container.py`, `src/workspace_setup.py`, `.claude/hooks/protect_result_files.py`
   - `src/eval_grader/{server,client,gpu_helpers}.py`
   - `src/token_tracker.py`, `src/session/manager.py`, `src/gpu_policy.py`
   - `src/history.py`, `src/ranking.py`, `src/artifacts.py`
4. Inspected the example task definitions under `examples/circle_packing/`.
5. Built and ran two constructive probes:
   - `code/secure_grader_probe.py` — starts the official grader server for the circle-packing task, submits a valid candidate, verifies controller-owned result files, and checks rejection of an invalid submission.
   - `code/budget_tracker_probe.py` — exercises the official `TokenTracker` cost accounting and cost-limit logic.
6. Compared EurekAgent with AlphaEvolve, AIDE, and TTT-Discover using the paper text and available code/papers.

A full end-to-end run was **not attempted** because it requires a working Claude Code installation, API credentials, and (for kernel tasks) NVIDIA GPUs. The probes isolate two durable, testable mechanisms without those dependencies.

## Key mechanism (made concrete)

### Stage loop

```text
START → entry → prepare → [propose → {implement}_p=1..P ]_r=1..R → end
```

- `prepare` (once): validates the problem, tests the grader, installs deps, writes `prepare/complete.json`.
- `propose` (per round): one Claude Code session writes `round_state/current_round_approaches.jsonl` + one `approach_details/<id>/approach.md` per hypothesis.
- `implement` (per round): up to `P` parallel Claude Code sessions, each confined to `approach_details/<id>/`, submit candidates via `python3 /workspace/eval/eureka_submit.py`.
- After implement, the engine collects `best_result.jsonl` entries, ranks them with the evaluator's `is_better`, and appends them to `round_state/ranked_past_best_solutions.jsonl` for the next round.

### Permission boundaries

| Productive affordance | Constraint |
|---|---|
| Workspace shell, Python venv, web search, browser MCP | Run inside a Docker container; workspace bind-mounted at `/workspace` |
| Access prior-round artifacts | Same-round peer `approach_details/` blocked by `protect_result_files.py` hook |
| GPU use via `gpu_helpers` | GPUs hidden (`CUDA_VISIBLE_DEVICES=""`) unless acquired through `gpu_session()`; lock files in `.gpu_locks/` |
| Submit candidates to grader | Hidden evaluator in separate grader container, read-only; agents cannot read/modify it |
| Read scores | `intermediate_results.jsonl` / `best_result.jsonl` are controller-owned; hook blocks direct writes |

### Artifact schema

- `prepare/complete.json` — stage gate.
- `round_state/current_round_approaches.jsonl` — manifest of hypotheses for the current round.
- `approach_details/<id>/approach.md` — implementation-ready hypothesis.
- `approach_details/<id>/code/` — Git-tracked solution code.
- `approach_details/<id>/submissions/<name>.json` — temporary candidate JSON (deleted after grading).
- `approach_details/<id>/eval_feedback/latest_feedback.json` — latest grader feedback.
- `approach_details/<id>/intermediate_results.jsonl` — append-only scored submissions.
- `approach_details/<id>/best_result.jsonl` — best scored result so far.
- `round_state/ranked_past_best_solutions.jsonl` — cross-round ranked history.

### Budget engineering

- Time: separate per-session limits for propose and implement; `.time/time_budget.json` + `.time/check_time.py` make the agent time-aware; a 5-minute warning is injected if deliverables are missing.
- Cost: `TokenTracker` aggregates input/output/cache tokens across sessions and computes USD cost; `SessionManager._cost_exceeded()` kills sessions when the run-level `cost_limit` is hit.
- Resume: session maps under `session_data/session_maps/` persist elapsed time and status so runs can resume under the remaining budget.

## Evidence preserved

| Path | What it is |
|---|---|
| `code/eurekagent-2606.13662/` | Official cloned repository (HEAD at time of clone) |
| `runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/code/secure_grader_probe.py` | Probe that runs the official grader server + client |
| `runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/code/budget_tracker_probe.py` | Probe that exercises `TokenTracker` cost accounting |
| `runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/probe_venv/` | Isolated venv used by the probes (fastapi, uvicorn, numpy) |
| `runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/probe_workspace/` | Workspace artifacts produced by `secure_grader_probe.py` |

## How to rerun

```bash
cd code/eurekagent-2606.13662

# Secure grader probe
../../runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/probe_venv/bin/python \
  ../../runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/code/secure_grader_probe.py

# Budget tracker probe
../../runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/probe_venv/bin/python \
  ../../runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662/code/budget_tracker_probe.py
```

If the isolated venv is missing, recreate it with:

```bash
cd runs/2026-06-14-weaver-eurekagent-repwam/eurekagent-2606.13662
python3 -m venv probe_venv
probe_venv/bin/pip install fastapi uvicorn numpy
```

## Probe results

### Secure grader probe

- Generated a candidate from the official `initial.py` (sum of radii ≈ 0.96, well below the reported SOTA, as expected for the baseline).
- Started the official FastAPI grader server locally on `127.0.0.1:19876`.
- Submitted the candidate; server returned `valid: true`, score `0.959764...`, and `best_result_updated: true`.
- Confirmed the server wrote:
  - `probe_workspace/approach_details/test/best_result.jsonl`
  - `probe_workspace/approach_details/test/intermediate_results.jsonl`
  - `probe_workspace/approach_details/test/eval_feedback/latest_feedback.json`
- Confirmed the temporary submission file under `submissions/` was deleted after grading.
- Submitted an invalid candidate missing `description`; server correctly rejected it with HTTP 400.

This confirms the central permissions-engineering claim: the agent can submit and read feedback, but the grader and authoritative result files are outside its control.

### Budget tracker probe

- Fed four synthetic sessions into `TokenTracker`.
- Correctly aggregated input/output/cache tokens and computed cost breakdown.
- Verified the cost-limit predicate used by `SessionManager`.

## Code status and limitations

- The repository is well-structured and the documented interfaces match the paper: Docker dual-container runtime, secure grader service, GPU lock helpers, budget tracker, hooks, and resumable LangGraph state machine.
- The probes validate two concrete mechanisms but do **not** validate the full autonomous loop, model behavior, or the reported SOTA results.
- Reproducing the $11 circle-packing result would require running the full prepare–propose–implement loop with Claude Code + GLM-5.1 (or equivalent) for ~9.3 hours of agent time (5 rounds × (20 + 120) min = 700 min) plus API budget. That is outside this run's scope.
- The TriMul and MLE-Bench results require GPUs and held-out data/leaderboards; they were not reproduced.
- The official code is AGPL-3.0.

## Comparison with load-bearing neighbors

### AlphaEvolve (Novikov et al., 2025)
- **Workflow:** prescribed evolutionary loop — population of programs, LLM mutation, evaluation cascade, MAP-Elites/island archive.
- **EurekAgent contrast:** does not prescribe mutation/selection; it gives general-purpose CLI agents a constrained environment (permissions, artifacts, budget, human UI) and lets them choose their own search strategy. EurekAgent's reported math results beat prior AlphaEvolve/OpenEvolve numbers.

### AIDE (Jiang et al., 2025)
- **Workflow:** prescribed tree-search agent with solution tree, feedback loops, and role-specialized agents for ML engineering.
- **EurekAgent contrast:** again avoids a fixed research workflow; targets cross-domain metric-driven tasks (math, kernels, ML) with the same minimal outer loop. AIDE's evaluator is visible to the agent; EurekAgent hides the grader behind a network service and blocks direct score-file writes.

### TTT-Discover (Yuksekgonul et al., 2026)
- **Workflow:** test-time RL that updates the model weights for each problem; uses gpt-oss-120b and costs a few hundred dollars per problem.
- **EurekAgent contrast:** training-free; uses off-the-shelf GLM-5.1 + Claude Code and reports total API costs of <$17 for the three math tasks (down to ~$11 for circle packing). EurekAgent's TriMul A100 result (2005 µs) is ~10.8% faster than the TTT-Discover A100 result reported in the paper.

## Research claims the main report can make

1. **The environment-engineering design is concrete and implemented, not just conceptual.** The official code provides a Docker-isolated dual-container runtime, a token-authenticated grader service, GPU lock helpers, controller-owned result files, a PreToolUse hook that blocks same-round peer access and score-file tampering, and a cross-session cost tracker. A direct probe of the grader server confirms that agents submit candidates and receive scores while the evaluator and result files remain outside their reach.

2. **EurekAgent shifts the research burden from prescribed workflows to environment constraints.** Unlike AlphaEvolve/AIDE/TTT-Discover, it does not embed a specific search algorithm (evolution, tree search, or test-time RL) in the controller. Its outer loop is only prepare → propose → implement; the CLI agents decide how to explore within per-stage time/cost budgets and permission boundaries.

3. **The cost claim is plausible in principle but not independently verified here.** The budget-accounting code is present and correct, and the paper reports per-task time budgets (e.g., 5 propose rounds × 20 min + 5 implement rounds × 120 min for circle packing). A full reproduction would require running the actual agent loop with Claude Code/GLM-5.1, which this run did not do; the reported SOTA numbers should be treated as the authors' results, not as independently reproduced findings.
