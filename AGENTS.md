# Paper Scout — Repository Contract

This file is for **coding agents**: AI agents that develop and maintain this repository.

If you are a **setup agent** installing Paper Scout for a user, read `SETUP.md` instead.
If you are a **reading agent** executing a scouting run, read the workspace instruction file in `~/.paper-scout/workspace/` instead.

## What This Project Is

Paper Scout installs a reusable agent behavior for discovering recent papers, filtering hard, investigating a few deeply, and delivering a Feishu doc per run.

This repository ships **templates and specifications** that get installed into a user's workspace. It is not an application. There is no source code to compile, no server to deploy, no CLI to build. The chain is: a coding agent maintains the templates, a setup agent installs them for a user, and a reading agent interprets the installed files to produce paper briefs.

What the repo contains:

- `README.md` — the project entry point for humans and AI agents landing here
- `SETUP.md` — the installation guide, read by a setup agent during installation
- `AGENTS_TEMPLATE.md` — the workspace instruction template, rendered into the user's persistent runtime contract
- `PROMPT_TEMPLATE.md` — the thin run-trigger template, rendered into a lightweight prompt
- `skill/SKILL.md` — the main runtime method, installed as a skill into the user's agent harness
- `skill/DEEP_DIVE.md` — the deep investigation sub-skill, invoked by the main skill during Phase 4
- `skill/FEISHU_DOC.md` — the document writing sub-skill, invoked by the main skill during Phase 5
- `DEEP_DIVE_SKILL.md` — an external reference baseline (not installed); kept in the repo as a reference for the coding agent

What the repo produces after installation:

- `~/.paper-scout/workspace/AGENTS.md` (or harness equivalent) — the filled runtime contract
- `~/.paper-scout/prompt.md` — the thin trigger
- An installed `paper-scout` skill (directory depending on user's preferred harness)

## What This Project Is Not

- Not a Python/TypeScript/Rust program. There is no code to run.
- Not a prompt library. The files are behavioral contracts with structure and intent, not interchangeable snippets.
- Not a fixed pipeline. The reading agent exercises judgment within the constraints the templates define.
- Not complete after one install. The templates evolve; the installed workspace files may need regeneration.

## Terminology

Four parties interact through this repository. The word "agent" is overloaded, so this contract uses specific terms consistently.

### User

The human. They discover this project, point an agent at it, answer a few questions, and then wait for paper briefs to appear in their Feishu folder. They do not read the templates. They do not read this file. Most of them will never read `README.md` either — they will say something like "read this repo and set it up for me" and expect an agent to handle the rest.

The user's involvement is:

1. One setup conversation (answer onboarding questions, confirm preferences).
2. Receiving paper briefs in Feishu, with no action required.
3. Occasional reconfiguration ("change my cadence to weekly", "add a new topic").

Everything else is handled by agents.

### Setup Agent

An AI agent that **installs and configures** Paper Scout for a user. It reads `README.md` (often directly from the GitHub URL), asks the user onboarding questions, verifies tools, generates workspace files, installs the skill, and optionally sets up scheduling.

The setup agent:

- runs once (or occasionally, for reconfiguration)
- operates from wherever the user invoked it — a terminal, a chat, a random directory
- has no access to the workspace until it creates one
- reads `SETUP.md` as its primary instruction source
- reads `AGENTS_TEMPLATE.md` and `PROMPT_TEMPLATE.md` as source material for generating files
- produces the installed workspace, the thin prompt, and the skill installation
- is done when the user has a working setup

The setup agent does not execute scouting runs. It does not read the installed workspace instruction file as operating instructions. It creates that file.

### Reading Agent

An AI agent that **executes** paper-scouting runs. It operates from `~/.paper-scout/workspace/`, reads the installed workspace instruction file as its persistent contract, follows the `paper-scout` skill as its method, and uses `prompt.md` as the per-run trigger.

The reading agent:

- runs repeatedly (daily, weekly, or on demand)
- operates from `~/.paper-scout/workspace/` (or other directory if specified)
- has no knowledge of this repository or its templates
- reads the installed workspace instruction file, the skill, and the prompt — not `README.md`
- scouts papers, filters, investigates, writes, delivers to Feishu, and logs
- never modifies this repository

The setup agent and the reading agent are different agents, running at different times, in different directories, with different instructions. They do not share context. The only thing that connects them is the files the setup agent generates and the reading agent consumes.

### Coding Agent

An AI agent that **develops and maintains** this repository. That is you, if you are reading this file.

Coding agents modify templates, rewrite instructions, adjust the skill spec, and improve the setup contract. Every edit a coding agent makes changes how the setup agent and reading agent behave, which changes what users experience.

## Thinking From the User's Perspective

Users do not see the templates. They interact with Paper Scout at two moments:

1. **Setup** — a one-time conversation with the setup agent. The user answers a few questions and confirms preferences. This should feel effortless. The user typed something like "install this for me" and expects the agent to drive the process.
2. **Receiving briefs** — paper briefs appear in their Feishu folder or wiki space. No action required. This is the ongoing value of the project.

There is also an occasional third moment: reconfiguration ("watch for a new topic", "switch to weekly"). This should feel like a quick conversation, not a reinstallation.

When editing any file in this repo, ask:

**How does this change affect what the user sees or experiences?**

Common failure modes to watch for:

- **Making setup feel like work.** The whole point is that the user does almost nothing. If `SETUP.md` is written in a way that causes the setup agent to dump a wall of questions, the user experience is already broken. Write setup instructions that lead to short, natural conversations with sensible defaults.
- **Jargon leaking into user-facing moments.** The setup agent translates `SETUP.md` into conversation. If the setup guide uses internal terminology without explaining how to present it, the setup agent may parrot it. Write setup instructions in a way that naturally produces plain-language questions.
- **Over-specification killing usefulness.** If templates are too rigid, the reading agent produces mechanical output. Users want judgment, not checklists. Leave room for the reading agent to adapt structure to content.
- **Under-specification causing inconsistency.** If templates are too vague, different reading agents will interpret them differently across runs. Users expect a stable personality and consistent quality. Be precise about what matters (standards, constraints, identity) and flexible about what should adapt (structure, length, emphasis).
- **Volume over signal.** Users do not want padding. Every template instruction that encourages completeness over selectivity makes the output worse for the user. Filtering hard is a feature, not a compromise.

### The User's Implicit Quality Bar

A user who installs Paper Scout wants:

- to never think about the scouting process itself
- to learn what matters without reading everything themselves
- honest judgment about which papers deserve their time
- enough depth on the important ones to decide whether to read the original
- a brief they can trust, not one they have to second-guess

If a template change would make the output longer but not more useful, it is probably wrong. If a setup change would require the user to understand internal concepts, it is probably wrong.

## Thinking From the Setup Agent's Perspective

The setup agent reads `SETUP.md` — often fetched directly from a GitHub URL — and uses it to drive a setup conversation with the user. It has no prior knowledge of Paper Scout. Everything it knows comes from what it reads.

When editing `SETUP.md`, ask:

**If an agent reads this cold and talks to a user, what conversation will it produce?**

Common failure modes to watch for:

- **Too many questions.** The user said "install this for me", not "let me fill out a form." If the setup guide presents a long field list without emphasizing defaults, the setup agent will ask for every field. Group questions by necessity: what must be asked, what has good defaults, what can be deferred.
- **Unclear sequencing.** The setup agent needs to verify tools before asking preferences, and ask preferences before generating files. If the guide presents these in the wrong order or mixes them, the setup agent may verify tools after generating files, or ask the user questions it could have inferred.
- **Missing error recovery.** If `hf papers` does not work, the setup agent needs to know what to do. If the guide only says "verify hf papers works" without explaining how to handle failure, different setup agents will handle it differently — some will try to fix it, some will silently skip, some will dump a traceback at the user.
- **Assuming the setup agent has repo access.** The setup agent may be reading `SETUP.md` from a URL. It may not have the repo cloned. If setup requires reading `AGENTS_TEMPLATE.md` or `skill/SKILL.md`, the guide should make clear how to access them — whether by cloning, fetching raw URLs, or other means.
- **Polluting the workspace instruction file.** The setup agent must not leave setup artifacts in the file it generates for the future running agent. `SETUP.md` should explicitly warn against this. See the "Your Role" section in that file.

## Thinking From the Reading Agent's Perspective

The reading agent operates from `~/.paper-scout/workspace/` and interprets the installed files as behavioral instructions. It has never seen this repository. It does not know how it was set up. Its entire world is the workspace instruction file, the skill, and the prompt.

When editing `AGENTS_TEMPLATE.md`, `skill/SKILL.md`, or `PROMPT_TEMPLATE.md`, ask:

**How will a reading agent interpret this instruction, and what will it actually do?**

Common failure modes to watch for:

- **Ambiguous priority signals.** If a template says "prefer novelty, relevance, and practical value" without clarifying their relative weight, different reading agents will rank differently. When priority order matters, say so explicitly.
- **Contradictory instructions across files.** The workspace instruction file, the skill, and the prompt can all give direction. If they disagree, the reading agent has to guess which one wins. Maintain a clear hierarchy: the workspace instruction file is the stable contract, the skill is the method, the prompt is the per-run trigger. Later sources override earlier ones only when they explicitly do so.
- **Instructions that assume capabilities.** Reading agents vary across harnesses. An instruction like "load `hf-cli`" works if the harness has skill loading. If it does not, the reading agent may silently skip it or fail. Instructions that depend on specific capabilities should include a fallback or a fail-fast check.
- **Missing stop conditions.** If a template says "investigate deeply" without bounding the effort, a reading agent may spend disproportionate time on one paper. Always pair open-ended directives with effort boundaries or explicit signals for when to stop.
- **Rigid structure mandates.** If the template prescribes exact section headings and ordering, the reading agent will follow them even when the content does not fit. Prefer outcome descriptions ("the brief should make clear which papers matter and why") over structural mandates ("use these exact headings in this order").

### The Instruction Hierarchy

When a reading agent encounters conflicting guidance, it should resolve using this order:

1. **Per-run prompt** (`prompt.md`) — most specific, wins on run-scoped parameters like date and focus.
2. **Workspace instruction file** (`AGENTS.md` in workspace) — stable user preferences, wins on identity, style, policy.
3. **Skill** (`skill/SKILL.md`) — the method, wins on workflow structure and phase definitions.

Coding agents should keep this hierarchy clean. Do not put run-scoped concerns in the skill. Do not put method details in the workspace instruction file. Do not put stable preferences in the prompt.

## Repository Contracts

### File Purposes and Boundaries

Each file in this repo has a distinct job. Do not let them bleed into each other.

**`README.md`** is the project entry point.

- Audience: humans discovering the project, and AI agents that land on the repo root.
- Contains: a brief project description, the two-audience installation pointer, a "what gets installed" summary, and a file inventory.
- Does not contain: setup flow, onboarding questions, tool requirements, runtime behavior.
- Editing principle: keep it short enough that a human gets the idea immediately and an AI agent gets pointed to the right file without needing to read further.

**`SETUP.md`** is the installation guide.

- Audience: **setup agents** (AI agents performing installation).
- Contains: role-awareness section, installation flow, onboarding questions, tool requirements, file generation rules, troubleshooting.
- Does not contain: runtime behavior, run-phase details, skill logic.
- Editing principle: if a change affects how the setup conversation goes, it belongs here.
- Remember: setup agents often read this from a GitHub URL without cloning the repo. It must be self-contained enough to drive the setup flow, while pointing to the other files when the setup agent needs to read them. It must also clearly warn the setup agent not to leave installation artifacts in the workspace instruction file it generates.

**`AGENTS_TEMPLATE.md`** is the workspace instruction template.

- Audience: **reading agents**, after the setup agent fills the template and installs it.
- Contains: identity, user preferences, policies, delivery destination, workspace rules, run directive.
- Does not contain: setup flow, installation steps, onboarding questions.
- Editing principle: if a change affects how every future run behaves for a given user, it belongs here.
- Remember: the reading agent has never seen this repository. It only sees the filled result. Do not reference template placeholders, setup flow, or repo structure in a way that would confuse a reading agent seeing only the installed file.

**`skill/SKILL.md`** is the main runtime method.

- Audience: **reading agents**, loaded as a skill during execution.
- Contains: phased workflow, investigation boundaries, selection criteria, synthesis guidance, logging rules.
- Does not contain: user-specific preferences, delivery destination, onboarding flow.
- Editing principle: if a change affects how the scouting method works regardless of which user is running it, it belongs here.
- Remember: this is the one file shared identically across all installations. It must not assume any particular user's interests or configuration.

**`skill/DEEP_DIVE.md`** is the deep investigation sub-skill.

- Audience: **reading agents**, invoked by the main `paper-scout` skill during Phase 4.
- Contains: section inventory process, motivation/contribution analysis, method walkthrough, experimental evidence assessment, artifact inspection, bottom-line judgment.
- Does not contain: synthesis, writing style, delivery, or workflow orchestration.
- Editing principle: if a change affects how deeply and systematically a single paper is analyzed, it belongs here.
- Relationship: `skill/SKILL.md` Phase 4 delegates entirely to this skill. Keep them consistent.

**`skill/FEISHU_DOC.md`** is the document writing sub-skill.

- Audience: **reading agents**, invoked by the main `paper-scout` skill during Phase 5.
- Contains: brief structure, section templates, visual-hierarchy guidance, writing quality standards, and the v2 delivery command sequence. Authoring format is Lark DocxXML (`lark-cli` v2).
- Does not contain: analysis logic or investigation procedures. It also does not own the DocxXML syntax — low-level format, escaping, and command flags defer to the installed `lark-doc` skill, so this file does not rot when `lark-cli` changes.
- Editing principle: if a change affects how the final brief is structured or written for Feishu, it belongs here. If it is a low-level DocxXML syntax detail, it belongs in `lark-doc`, not here.
- Relationship: `skill/SKILL.md` Phase 5 delegates writing decisions to this skill. Writing tone and depth are still controlled by the workspace instruction file — this skill controls layout and structural quality.

**`PROMPT_TEMPLATE.md`** is the run-trigger template.

- Audience: **reading agents**, at the start of each run.
- Contains: date, workspace path, focus scope, cadence — just enough to start a run.
- Does not contain: full operating contract, method details, stable preferences.
- Editing principle: this should stay thin. If you are adding more than a few lines, the content probably belongs in the workspace instruction template or the skill instead.

### Editing Standards

When modifying any file:

- Read the file fully before editing. These files are behavioral contracts — partial understanding leads to contradictions.
- Check whether the change belongs in the file you are editing or in a different one. Use the file purposes above.
- After editing, verify that the instruction hierarchy remains consistent: no contradictions between the workspace template, the skill, and the prompt template.
- Preserve the existing voice and density. These files are already concise. Do not inflate them with hedging, repetition, or meta-commentary.
- Do not add instructions to reading agents that you cannot reasonably expect them to follow. If an instruction requires a capability that may not exist, add a fallback.

### What Not to Add to This Repo

- Executable code. The runtime is the reading agent, not a script.
- Example outputs. They become stale fast and risk becoming de facto templates that reading agents imitate rather than think through.
- Per-user configuration. That belongs in the installed workspace, not in the repo.

## Workspace Contracts

The installed workspace at `~/.paper-scout/workspace/` has invariants that the templates must preserve.

### Structural Invariants

The workspace always contains:

```text
workspace/
├── AGENTS.md (or harness equivalent)
├── papers/
├── repos/
├── runs/
├── output/
└── state/
    └── log.md
```

Templates must not rename these directories or introduce alternative layouts without updating all references across all files.

### Behavioral Invariants

These rules must hold across template and skill changes:

- `state/log.md` is the single source of truth for what has been covered. If a coding agent changes how logging works, the reading agent must still be able to determine what was previously deep-dived.
- The workspace instruction file is the stable contract. Regenerating `prompt.md` should not require regenerating the workspace instruction file, and vice versa.
- The prompt stays thin. It provides run-scoped parameters. It does not duplicate the runtime contract.
- One run produces one Feishu doc. The delivery model is append-new, not update-existing.
- The reading agent is read-only during investigation by default. Expanding permissions requires explicit user configuration, not template defaults.

### State Management

`state/log.md` is the only persistent state across runs. Templates and the skill must treat it as:

- append-only (newest entries first when practical)
- the authoritative record of which papers have been deep-dived
- the source for deduplication decisions
- readable by both the run agent and the user

If a coding agent changes the log format, it must ensure backward compatibility or provide migration guidance, because existing installations will have logs in the old format.
