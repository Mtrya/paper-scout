# Paper Scout

Paper Scout is an agent-native paper scouting workflow for discovering recent papers, filtering aggressively, investigating the most promising ones, and delivering a fresh Feishu doc for each run.

Its default source is Hugging Face papers via `hf papers`, but the installed workspace instructions can later be adjusted to support other accessible sources. The default operating model is:

- scan a broad recent pool
- shortlist the papers worth noticing
- deeply investigate only a few that justify the extra effort
- deliver one Feishu doc that mixes "what matters now" with deeper analysis

This repository is intentionally small. Its main job is to install a reusable behavior, not a fixed application.

## Repository Contents

- `README.md`: onboarding, setup, installation, and reconfiguration
- `AGENTS_template.md`: the persistent workspace instruction template for `AGENTS.md` / `CLAUDE.md` / `GEMINI.md`
- `skill/SKILL.md`: the runtime method for scouting, filtering, investigation, delivery, logging, and cleanup
- `prompt_template.md`: the template used to generate the user's thin `~/.paper-scout/prompt.md` trigger

## Installation Outcome

After setup, a typical installation should produce:

- `~/.paper-scout/workspace/`
- `~/.paper-scout/workspace/AGENTS.md` or the harness-equivalent workspace instruction file
- `~/.paper-scout/prompt.md`
- an installed `paper-scout` skill in the user's preferred harness
- optionally, a scheduled task for automated runs

The repository itself is not required after installation as long as the generated workspace files and installed skill are in place.

## Setup Contract

If you are an AI agent and a user has asked you to install or configure Paper Scout, this README is your guide. Treat it as the canonical setup contract and follow this flow:

1. Read this README before making changes.
2. Clone or fetch the repository so the other files are accessible. If you are reading this README from a URL, you need access to `AGENTS_template.md`, `prompt_template.md`, and `skill/SKILL.md` from the same repository.
3. Verify the required tools and actual access:
   - `hf` is installed (`uv tool install huggingface_hub`)
   - `hf papers` works
   - `lark-cli` is installed and configured
   - `lark-cli docs` and `lark-cli wiki` work
   - Feishu destination access can be validated
4. Ask the user the onboarding questions in the next section.
5. Create `~/.paper-scout/workspace/` if needed.
6. Generate the workspace instruction file from `AGENTS_template.md`, using the harness-appropriate filename when practical:
   - Codex: `AGENTS.md`
   - Claude: `CLAUDE.md`
   - Gemini: `GEMINI.md`
7. Generate a thin `~/.paper-scout/prompt.md` from `prompt_template.md`.
8. Install `skill/SKILL.md` into the user's harness, typically under a skill directory named `paper-scout`.
9. Optionally install a scheduled task if the user wants automation.

You own this flow end-to-end, but pause for clarification or confirmation whenever the user's preferences, destination, or harness path are unclear.

### Onboarding Style Rules

When asking setup questions, translate this README into plain user-facing language instead of repeating the raw field list.

In particular:

- ask only for the minimum information needed to proceed
- offer sensible defaults and say you will fill in the rest unless the user wants to customize
- explain project-specific terms the first time they appear
- prefer short, natural prompts over long requirement dumps
- defer optional questions such as scheduling until the core installation is complete

Do not respond to a simple install request by pasting every onboarding field back to the user verbatim.

For example:

- say "How often should it run: daily or weekly?" instead of "cadence"
- say "Where in Feishu should new reports be created?" instead of "Feishu destination"
- say "Is it okay to inspect promising repos or project pages without running code?" instead of "code investigation policy"

## Required Tools

Paper Scout assumes the agent executing scouting runs can use:

- `hf` with working `hf papers`
- `lark-cli` with a valid authenticated session and the ability to create docs in the target destination

Fail fast if either tool exists but does not actually work for the intended operation.

## Quick Start

If the user says something like "read this repo and configure it for me", you should:

1. Clone or fetch the repository so the templates and skill are accessible.
2. Verify `hf` and `hf papers`.
3. Verify `lark-cli` and validate the user's Feishu destination.
4. Ask a short plain-language version of the onboarding questions below, using defaults where possible.
5. Generate the workspace instruction file in the Paper Scout workspace.
6. Generate the thin `~/.paper-scout/prompt.md` trigger.
7. Install the Paper Scout skill into the user's harness.
8. Optionally install scheduling.

A good quick-start prompt should usually be closer to:

- what topics should this watch for?
- what style should the writeup have?
- how often should it run: daily or weekly?
- where in Feishu should it create the report?

If the user does not care about the finer details, choose reasonable defaults and continue.

## Onboarding Questions

Collect the following information, but you do not need to ask for all of it in one message:

- research interests / domains
- preferred language
- preferred tone
- desired depth
- cadence: daily or weekly
- Feishu destination
- whether code investigation is allowed
- workspace override if the user does not want the default
- whether scheduling should be installed

The quick-start flow should keep these questions concise, explain unfamiliar terms in plain language, and allow custom instructions only where the user wants finer control.

If the user gives only partial answers, fill the remaining fields with reasonable defaults and confirm the result briefly.

## Guidance For Answers

### Research Interests / Domains

Free-form input is expected. Preserve the user's wording and only normalize it enough to make the generated workspace instructions clear and actionable.

### Preferred Language

Free-form input is acceptable. Common defaults are Chinese, English, or bilingual delivery.

If the user does not care, default to the language the user is already using in the conversation.

### Tone

Offer a few reasonable options and also allow custom instruction. Good defaults include:

- research notebook
- executive brief
- critical reading memo
- practical recommendation memo

### Depth

Offer a few reasonable options and also allow custom instruction. Good defaults include:

- light scan
- balanced
- deep technical

### Cadence

Explain this as how often Paper Scout should run. The default modes are:

- `daily`
- `weekly`

The generated workspace instructions should keep budgets adaptive. If the period is weak, the agent running Paper Scout should do fewer papers rather than pad the output.

### Feishu Destination

Explain this as where new reports should be created in Feishu. The expected destination is a folder or wiki space where a new doc can be created for each run.

You should:

1. ask for the destination URL or token
2. verify that it is a valid destination
3. confirm that the authenticated `lark-cli` session can create docs there

Setup should not silently proceed with an invalid or unverified destination.

### Code Investigation Allowed

Explain this as whether Paper Scout may inspect promising project pages, repos, model cards, or dataset cards while staying read-only. This is free-form but should be recorded clearly in the generated workspace instructions. A useful default is:

- project pages, repos, model cards, and dataset cards may be inspected when a paper looks especially promising
- code should not be run
- models and datasets should not be downloaded

### Workspace Location

The default workspace root is:

`~/.paper-scout/workspace/`

If the user does not request a different location, use this default.

## Prompt Generation Rules

The generated trigger prompt should be written to:

`~/.paper-scout/prompt.md`

If `~/.paper-scout/prompt.md` already exists:

1. summarize the current prompt briefly
2. ask whether to overwrite it, update it, or keep both

The generated prompt should stay thin. It should only:

- state the current date
- identify the workspace path
- remind the agent to use the workspace instruction file and the `paper-scout` skill
- start the scouting run

Do not duplicate stable preferences (interests, exclusions, cadence, tone) in `prompt.md`. Those belong in the workspace instruction file.

## Workspace Instruction File Rules

The persistent runtime contract should be installed into the Paper Scout workspace as the harness-appropriate always-loaded instruction file.

Preferred filenames:

- Codex: `AGENTS.md`
- Claude: `CLAUDE.md`
- Gemini: `GEMINI.md`
- Kimi: `AGENTS.md`

If the harness-specific workspace instruction mechanism is obvious, use it. If not, ask the user.

If the harness does not support an always-loaded workspace instruction file, a fuller `prompt.md` fallback is acceptable, but the preferred design is still:

- stable contract in the workspace instruction file
- thin trigger in `~/.paper-scout/prompt.md`

The generated workspace instruction file should carry the stable configuration, including:

- Paper Scout identity and mission
- required companion skills
- user interests and exclusions
- cadence and effort budget
- language, tone, and depth preferences
- investigation policy (user-configured permissions; the skill defines the boundaries)
- output expectations
- delivery destination
- workspace root and coverage log location

The workspace instruction file should not duplicate the skill's phased workflow, selection criteria, or logging procedures. It should defer to the `paper-scout` skill for method details.

## Skill Installation Rules

The installable skill source in this repository is `skill/SKILL.md`.

Install it into the user's harness under a skill directory named `paper-scout` when practical.

If the harness-specific install location is obvious, use it. If not, ask the user.

Paper Scout also expects companion skills during execution:

- `hf-cli` before source discovery
- `lark-doc` before delivery

Those do not need to be copied from this repository, but the workspace instruction file should remind the agent executing runs to load them.

## Manual Use

After installation, the persistent workspace contract should live in:

`~/.paper-scout/workspace/AGENTS.md`

or the harness-equivalent filename for the user's agent.

The thin run trigger should live at:

`~/.paper-scout/prompt.md`

Runs should execute from:

`~/.paper-scout/workspace/`

The workspace should hold:

- the harness-specific workspace instruction file
- `papers/` — downloaded paper markdown
- `repos/` — cloned repositories for inspection
- `runs/` — per-run notes and scratch artifacts
- `output/` — final Lark-flavored markdown before delivery
- `state/log.md` — the persistent coverage log

## Scheduling

Scheduling is optional and should be offered only after the workspace instruction file, thin run trigger, and skill installation are complete.

When first onboarding a user, treat scheduling as a follow-up preference rather than a required setup question.

If the user wants automation, you may install either:

- `cron`
- `systemd --user`

Scheduling should:

- run from `~/.paper-scout/workspace/`
- use the generated thin `~/.paper-scout/prompt.md` together with the installed workspace instruction file
- preserve the same user-specific configuration as manual runs

Explain what you are installing and confirm the user's preference before making the scheduling change.

## Reconfiguration

If the user later wants to change scope, tone, cadence, destination, or investigation policy, update the workspace instruction file in `~/.paper-scout/workspace/`.

The thin `~/.paper-scout/prompt.md` trigger rarely needs reconfiguration since it carries only the date and workspace path. The installed skill should only be touched if the runtime method itself needs to change.

## Troubleshooting

### `hf` Exists But `hf papers` Fails

Treat this as incomplete setup. Fix access or versioning first instead of generating a prompt that cannot run.

### `lark-cli` Exists But Destination Validation Fails

Do not proceed with delivery setup until the destination is verified and writable.

### The Skill Install Location Is Unclear

Use the obvious harness path if one can be detected confidently. Otherwise ask the user instead of guessing.

### The User Does Not Want Scheduling Yet

That is a normal stopping point. A successful manual installation without scheduling is a complete setup.
