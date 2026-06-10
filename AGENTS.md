# Paper Scout — Repository Contract

This file is for **coding agents**: AI agents that develop and maintain this repository.

If you are a **reading agent** executing a scouting run, read `workspace/AGENTS.md` instead — that is your operating contract.

## What This Project Is

Paper Scout is a single living instance of a paper-conditioned research workflow. It discovers recent papers, filters hard, investigates a few deeply, delivers a Feishu research report per run, and accumulates its record in this repository as runs happen.

There is no application to build and no installer. The repository *is* the instance. Anyone who wants their own tracker can clone this, read the prompts, and tell their agent to "configure a paper tracker in the same spirit, tailored to my interests" — then edit `workspace/AGENTS.md` and the skills directly. What this project ships is **prompts**: instruction files and skills that are easy to read and tune.

What the repo contains:

- `README.md` — the project entry point for humans and agents landing here
- `AGENTS.md` — this file: the contract for coding agents maintaining the repo
- `prompt.txt` — the thin run trigger for a reading-agent run
- `scout.sh` — a launcher that starts a run from `workspace/`
- `workspace/AGENTS.md` — the reading agent's persistent operating contract
- `workspace/.agents/skills/` — the skills the reading agent loads (the method)
- `workspace/code/` — ignored lab bench for cloned repos, probes, patches, experiments, and local environments
- `workspace/runs/` — delivered reports, durable evidence packets, and `INDEX.md` (the readable record + dedup log)

## The Two Agents

The word "agent" is overloaded, so this contract uses two specific terms.

### Reading Agent

The agent that **executes** scouting runs. It operates from `workspace/`, reads `workspace/AGENTS.md` as its persistent contract, follows the skills under `workspace/.agents/skills/` as its method, and uses `prompt.txt` as the per-run trigger. It scouts, filters, investigates, writes, delivers to Feishu, and appends to `runs/INDEX.md`.

### Coding Agent

The agent that **develops and maintains** this repository — that is you, if you are reading this file. Coding agents edit the prompts, the skills, the launcher, and the structure. Every edit changes how the reading agent behaves, which changes what the user receives.

## The Instruction Hierarchy

When the reading agent encounters conflicting guidance, it resolves in this order:

1. **Per-run prompt** (`prompt.txt`) — most specific; wins on run-scoped parameters like date and focus.
2. **Reading-agent contract** (`workspace/AGENTS.md`) — stable user preferences; wins on identity, style, policy, destination.
3. **Skills** (`workspace/.agents/skills/`) — the method; win on how each workflow step is executed.

Keep this hierarchy clean. Do not put run-scoped concerns in a skill. Do not put method details in the reading-agent contract. Do not put stable preferences in the prompt.

Note that the reading-agent contract may deliberately **expand** a skill's defaults (for example, broadening the investigation policy beyond the skill's read-only baseline). That is expected: when the contract and a skill differ, the contract wins.

## File Purposes and Boundaries

Each file has a distinct job. Do not let them bleed into each other.

**`README.md`** — the project entry point. Audience: humans and agents landing on the repo. A brief description, how a run works, and how to fork the spirit. Keep it short.

**`AGENTS.md`** (this file) — the coding-agent contract. How to maintain the repo without breaking the reading agent. No run-phase detail, no user preferences.

**`prompt.txt`** — the run trigger. Date (injected by `scout.sh`) plus the kick-off instruction. Stays thin: if you are adding more than a few lines, the content belongs in `workspace/AGENTS.md` or a skill instead.

**`workspace/AGENTS.md`** — the reading-agent contract. Identity, user preferences, policy, delivery destination, coverage-log rules. Concrete values, not templates. No method detail, no run-phase orchestration.

**`workspace/.agents/skills/workspace-manage/SKILL.md`** — workspace layout, naming rules, artifact tracking, verifier script, cleanup policy, and `runs/INDEX.md` coverage log.

**`workspace/.agents/skills/paper-source/SKILL.md`** — recent-paper pool discovery. Default source is Hugging Face Papers, with commands documented inline; additional sources are added here as the workflow grows.

**`workspace/.agents/skills/paper-deep-dive/SKILL.md`** — deep investigation of a single paper: reading, code inspection, resource-proportional research actions, related-work comparison, and structured analysis notes.

**`workspace/.agents/skills/report-compose/SKILL.md`** — DocxXML research report composition, illustrative-artifact planning, Feishu doc creation, media insertion, user notification, and run-packet report preservation. Uses `lark-doc` and `lark-im` for command details.

**`scout.sh`** — the launcher. Computes the date, stamps `prompt.txt`, starts the harness from `workspace/`. The only file that knows the harness invocation.

## Editing Standards

- Read a file fully before editing it. These are behavioral contracts; partial understanding leads to contradictions.
- Check whether a change belongs in the file you are editing or another one. Use the file purposes above.
- After editing, verify the instruction hierarchy stays consistent — no contradictions between the prompt, the reading-agent contract, and the skills.
- Preserve the existing voice and density. These files are concise; do not inflate them with hedging, repetition, or meta-commentary.
- Do not give the reading agent instructions it cannot reasonably follow. If an instruction depends on a capability that may be absent, add a fallback.

## What Not to Add

- Executable application code. The runtime is the reading agent, not a script.
- Example outputs. They go stale and become de facto templates the reading agent imitates instead of thinking through.
- A setup/install/distribution layer. That architecture was deliberately removed.

## Workspace Invariants

The reading agent's home is `workspace/`. These invariants must hold across prompt and skill changes:

```text
workspace/
├── AGENTS.md          # reading-agent contract
├── .agents/skills/    # the skills (the method)
├── papers/            # tracked downloaded paper markdown cache, organized by area
├── code/              # ignored lab bench for external-signal work
├── drafts/            # ignored working DocxXML and scratch artifacts
└── runs/
    ├── INDEX.md                         # coverage log + dedup source of truth
    └── <run-id>/                        # durable run packet
        ├── report.docxxml
        ├── checklist.md
        ├── assets/
        └── <thread-id>/
            ├── README.md
            ├── code/                    # optional preserved code/probes
            └── patches/                 # optional preserved patches
```

Delivered report sources are archived inside their run packets as `report.docxxml`.

Behavioral invariants:

- `runs/INDEX.md` is the single source of truth for what has been covered. If you change how logging works, the reading agent must still be able to determine what was previously covered from this file.
- The reading-agent contract is stable. Regenerating `prompt.txt` should not require changing `workspace/AGENTS.md`, and vice versa.
- The prompt stays thin — run-scoped parameters only.
- One run produces one Feishu doc. Delivery is append-new, not update-existing.
- The reading agent's investigation permissions are governed by its contract. The skill defines a read-only baseline; the contract may expand it.

If you change the `runs/INDEX.md` format, keep it readable by both the reading agent and the user, and append-only (newest first when practical).
