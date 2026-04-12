# Paper Scout

Paper Scout is an agent-native paper scouting workflow. It discovers recent papers on Hugging Face, filters hard, investigates the most promising ones, and delivers a fresh Feishu doc for each run.

This repository installs a reusable agent behavior. There is no code to run. The output is a configured workspace and an installed skill that an AI agent uses on a schedule.

## Installation

**If you are a human,** tell your AI agent:

> Read https://github.com/Mtrya/paper-scout/blob/main/SETUP.md and install Paper Scout for me.

**If you are an AI agent,** read `SETUP.md` in this repository. That file is your complete installation guide. Do not start the installation from this README.

## What Gets Installed

After setup:

- `~/.paper-scout/workspace/` — the agent's working directory
- `~/.paper-scout/workspace/AGENTS.md` (or harness equivalent) — the persistent runtime contract
- `~/.paper-scout/prompt.md` — a thin run trigger
- An installed `paper-scout` skill in the agent harness
- Optionally, a scheduled task for automated runs

The repository is not required after installation.

## Repository Contents

- `SETUP.md` — installation guide (read this to install)
- `AGENTS_template.md` — template for the workspace instruction file
- `prompt_template.md` — template for the thin run trigger
- `skill/SKILL.md` — the paper-scouting skill
- `AGENTS.md` — repository contract for agents maintaining this codebase
