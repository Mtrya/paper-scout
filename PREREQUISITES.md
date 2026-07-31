# Prerequisites

Paper Scout has no installer — but a run touches several external tools. This checklist tells you what each one is for and what degrades if it is missing, so you can decide what your own tracker needs. Everything on this list is optional in the sense that the workflow degrades gracefully; nothing crashes.

## Agent harness

- **A supported agent CLI** — `codex`, `kimi`, or `qoder` (see the case block in `scout.sh`). This is the one hard requirement: it is the runtime.
  - *If missing:* no run. Add your own harness to `scout.sh` instead.

## Core tooling

- **git** — the reading agent branches, commits, pushes, and merges run PRs from inside `workspace/`.
  - *If missing:* runs still happen, but nothing is versioned and `scout.sh` aborts (it requires a clean worktree on `main`).
- **GitHub CLI (`gh`)** — PR creation, CI watching, and merge during finalization.
  - *If missing:* the run finishes locally; the agent reports the branch and commit instead of opening a PR.
- **Python 3** (stdlib only) — runs `verify_run.py`, the machine-checkable run-packet gate.
  - *If missing:* no local verification; the `verify-run` GitHub Action still gates PRs remotely.
- **curl, unzip** — arXiv PDF download and MinerU output handling in the `paper-source` scripts.
  - *If missing:* the PDF reading path fails; the agent falls back to metadata-only scouting.

## Paper sources and reading

- **Hugging Face `hf` CLI** — the default recent-paper pool (trending/daily lists, metadata, keyword search).
  - *If missing:* no default pool. The agent is instructed to use whatever other paper sources the environment offers, but expect thinner scouting.
- **MinerU, local install** — `uv tool install 'mineru[all]'`, plus model weights (`mineru-models-download -s modelscope -m pipeline` if the automatic download fails). Used for full-text reading with reliable figure/table extraction.
  - *If missing:* deep dives lose their main full-text source; the agent falls back to abstract-level reading or other converters, and investigations get shallower.

## Delivery

- **`lark-cli`** — creates the Feishu report doc, inserts media, and DMs the user. Also used by `scout.sh` for abort notifications (with `PAPER_SCOUT_NOTIFY_USER_ID` set).
  - *If missing:* delivery degrades gracefully by design — the report is saved to `runs/<run-id>/report.docxxml`, the path is printed, and the run completes without Feishu. If you don't use Feishu at all, retarget the delivery section of `workspace/AGENTS.md` and the `report-compose` skill.

## Remote compute (optional)

- **Inspire `inspire` CLI + configured account** — routes GPU-heavy research actions (reproductions, ablations, probes) to the Inspire platform when the local machine lacks compute. Project context lives in `workspace/INSPIRE.md` (gitignored — create your own as you start using the platform); command details come from the harness-level `inspire` skill installed by InspireSkill.
  - *If missing:* everything still works; GPU-needing investigations simply stay local, and the agent records a precise blocker when no local action is feasible. If you don't have an Inspire account, delete the `remote-compute` skill and the corresponding bullet in `workspace/AGENTS.md`.

## CI

- **GitHub Actions** — `.github/workflows/verify-run.yml` reruns the run-packet verifier on every PR that touches `workspace/runs/`. Nothing to install; it comes with the repo.
