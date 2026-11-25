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
- **Python 3** (stdlib only) — runs `verify_run.py`, the machine-checkable run-packet gate, and the shared RSS/Atom parser.
  - *If missing:* no local verification or feed parsing; the `verify-run` GitHub Action still gates PRs remotely.
- **curl, unzip** — arXiv PDF download and MinerU output handling in the `paper-source` scripts.
  - *If missing:* the PDF reading path fails; the agent falls back to metadata-only scouting.

## Research sources and reading

- **Hugging Face `hf` CLI** — the default recent-paper pool (trending/daily lists, metadata, keyword search).
  - *If missing:* no default pool. The agent is instructed to use whatever other paper sources the environment offers, but expect thinner scouting.
- **Scientific Spaces RSS** — recent research essays from `spaces.ac.cn`; selected full text comes from the public `spaces-index` mirror or the original site.
  - *If unreachable:* skip this source and continue with the other pools; no local mirror or crawler is required.
- **Research-blog RSS feeds** — thirteen registered researcher, lab, and company blogs, fetched through the same zero-dependency parser and merged into the paper pool.
  - *If a feed is unreachable:* skip that feed, keep the rest, and note the gap; feeds come and go and none is load-bearing.
- **MinerU, local install** — `uv tool install 'mineru[all]'`, plus model weights (`mineru-models-download -s modelscope -m pipeline` if the automatic download fails). Used for full-text reading with reliable figure/table extraction.
  - *If missing:* deep dives lose their main full-text source; the agent falls back to abstract-level reading or other converters, and investigations get shallower.

## Delivery

- **`lark-cli`** — creates the Feishu report doc, inserts media, and DMs the user. Also used by `scout.sh` for abort notifications (with `PAPER_SCOUT_NOTIFY_USER_ID` set).
  - *If missing:* delivery degrades gracefully by design — the report is saved to `runs/<run-id>/report.docxxml`, the path is printed, and the run completes without Feishu. If you don't use Feishu at all, retarget the delivery section of `workspace/AGENTS.md` and the `report-compose` skill.

## Remote compute (optional)

- **Google Colab CLI (`colab`, `google-colab-cli`)** — provisions short-lived single-GPU runtimes (T4/L4/A100/H100) headlessly and runs a local script on them. Used for probes, small reproductions, 4B-class inference, and plotting when the local machine has no GPU; sessions are ephemeral, so the disk cannot be blown up. Usage is metered in monthly compute units (`colab usage`).
  - *If missing or unauthenticated:* record a precise blocker instead of pretending a manual browser session happened. Nothing else degrades; the run reports the gap.
- **Inspire `inspire` CLI + configured account** — routes multi-GPU, long-running, or workspace-resident work to the Inspire platform when Colab's single accelerator or session lifetime does not fit. Project context lives in `workspace/INSPIRE.md` (gitignored — create your own as you start using the platform); command details come from the harness-level `inspire` skill installed by InspireSkill.
  - *If missing:* everything still works; GPU-needing investigations fall back to Colab or stay local, and the agent records a blocker when no backend is available. `lite` runs never depend on it. If you don't have an Inspire account, delete the corresponding bullet in `workspace/AGENTS.md` and keep the 本机 → Colab order.

## CI

- **GitHub Actions** — `.github/workflows/verify-run.yml` reruns the run-packet verifier on every PR that touches `workspace/runs/`. Nothing to install; it comes with the repo.
