# Paper Scout

A living collection of paper-conditioned research reports. Paper Scout is an agent-native workflow that scouts recent papers (default source: Hugging Face), filters hard, investigates the most promising or interesting ones deeply, and delivers a fresh Feishu report for each run. The record accumulates here as runs happen.

There is no code to build and nothing to install. The repository *is* the instance. What it ships is **prompts** — instruction files and skills that are easy to read and tune.

## How a Run Works

1. `scout.sh` starts the reading agent from `workspace/` with a date-stamped `prompt.txt`.
2. The agent reads `workspace/AGENTS.md` (its contract) and loads skills from `workspace/.agents/skills/` as needed: scout the recent pool, filter aggressively, deep-dive a small handful, compose the report, deliver a fresh Feishu doc.
3. The report and preserved evidence are saved under `workspace/runs/<run-id>/`, and the run is logged to `workspace/runs/INDEX.md`.

## Layout

- `prompt.txt` — the run trigger
- `scout.sh` — launches a run
- `workspace/AGENTS.md` — the reading agent's contract (your interests, cadence, Feishu destination)
- `workspace/.agents/skills/` — the scouting, deep-dive, and Feishu-doc skills
- `workspace/code/` — ignored lab bench for cloned repos, probes, patches, and temporary environments
- `workspace/runs/` — delivered reports, preserved evidence packets, and `INDEX.md`
- `AGENTS.md` — contract for coding agents maintaining this repo

`workspace/papers/` is the tracked paper-text cache. `workspace/runs/` holds durable run packets. `workspace/code/` and `workspace/drafts/` hold scratch run data and are gitignored except for their README markers.

## Make It Your Own

Want your own tracker? Clone this, then tell your agent something like: "read this repo and configure a paper tracker for me in the same spirit, tailored to my interests." The agent edits `workspace/AGENTS.md` (interests, exclusions, cadence, Feishu destination) and tunes the skills as needed. No setup script required — it's all prompts.
