# LHTB official repo inspection notes

**Repo:** https://github.com/zli12321/LHTB  
**Cloned to:** `runs/2026-07-14-genception-lhtb-robodojo/lhtb-2607.08964/code/lhtb-repo/`

## Files inspected

### Harness / runner config

- `configs/examples/full_benchmark.yaml`
  - Runs all 46 tasks under Terminus-2 agent.
  - 90-minute agent timeout (`override_timeout_sec: 5400`).
  - JSON action parser, proactive summarization at 8k tokens, terminal session recording.
  - Model default is `openai/gpt-4.1` in the example; actual paper runs swapped the model endpoint.

- `configs/examples/oracle_smoke.yaml`
  - Smoke test that runs reference solutions on 3 tasks without an API key.

### Task format

- `tasks/rush_hour_campaign/task.toml`
  - `schema_version = "1.3"`, agent timeout 3600s, verifier timeout 3600s.
  - Docker image: `zli12321/lhtb-rush_hour_campaign:20260615`.
  - No internet; 2 CPUs, 1 GB RAM, 1 GB storage.

- `tasks/unison-paper-reproduction/task.toml`
  - Agent timeout 5400s (90 min), verifier 900s.
  - Docker image: `zli12321/lhtb-unison-paper-reproduction:20260615`.
  - Internet allowed; 2 CPUs, 4 GB RAM, 8 GB storage.

### Verifiers / graders

1. `tasks/rush_hour_campaign/tests/verifier.py` (and its wrapper `tasks/rush_hour_campaign/verifier.py`)
   - Simulates four 6×6 Rush Hour puzzles from a JSON submission.
   - Partial credit: 10% legal-prefix progress + 65% solved within max moves + 25% solved within target moves.
   - Cuts score to ≤0.2 if the agent is caught running or writing solver code (monitors submission text and terminal recording).

2. `tasks/unison-paper-reproduction/tests/test_outputs.py`
   - Requires six source modules and forbids hard-coded golden numbers.
   - Public ground-truth test with tolerance; hidden config test with different seeds/thread count/scheduler.
   - Checks non-trivial LP partitioning, load balance, same-seed determinism.

3. `tasks/vector-db-iterative-build/tests/test_outputs.py`
   - Hidden 100k-vector ANN evaluation: recall@10 and QPS.
   - Continuous reward bands from 0.0 (recall < 0.5) up to 1.0 (recall ≥ 0.95 and speedup ≥ 100×).
   - Comment notes a recent reward rescaling: old top band saturated at 10×, so strong solutions were indistinguishable from borderline ones.

## Observations

- The repo README reports 21 models in the July 2026 snapshot, with Grok 4.5 leading. This is a superset of the 15 models in the arXiv paper (which had GPT-5.5 leading).
- All 46 `task.toml` files set `difficulty = "hard"`. The paper's Easy/Hard labels are derived empirically from mean model reward.
- Repo categories are finer-grained (19 tags) than the paper's nine display categories.
- Large APEX assets are managed by Git LFS and were not pulled.

## Things not run

- Docker builds / oracle smoke test.
- Actual model evaluation (no API keys).
- Git LFS pull for APEX workflow assets.
