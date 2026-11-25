# Paper Scout — Repository Contract

This file is for **coding agents**: AI agents that develop and maintain this repository.

If you are a **reading agent** executing a scouting run, read `workspace/AGENTS.md` instead — that is your operating contract.

## What This Project Is

Paper Scout is a single living instance of a paper-conditioned research workflow. It discovers recent papers, filters hard, investigates a few deeply, delivers a Feishu research report per run, and accumulates its record in this repository as runs happen.

Runs come in two streams, chosen per run by the `--stream` flag: `full` (strong model, 1–3 deep threads with real experiments) and `lite` (lighter model or compute, 1–2 focused threads resting on reading, source tracing, derivation, and lightweight checks). Both streams deliver the same report quality bar; only how evidence is produced changes.

There is no application to build and no installer. The repository *is* the instance. Anyone who wants their own tracker can clone this, read the prompts, and tell their agent to "configure a paper tracker in the same spirit, tailored to my interests" — then edit `workspace/AGENTS.md` and the skills directly. What this project ships is **prompts**: instruction files and skills that are easy to read and tune.

What the repo contains:

- `README.md` — the project entry point for humans and agents landing here
- `PREREQUISITES.md` — natural-language checklist of external tools a run touches, with per-tool degradation notes
- `AGENTS.md` — this file: the contract for coding agents maintaining the repo
- `prompt.txt` / `prompt-lite.txt` — the thin run triggers for a reading-agent run (`full` and `lite` streams)
- `scout.sh` — a launcher that starts a run from `workspace/` (`--stream full|lite`, default `full`)
- `workspace/AGENTS.md` — the reading agent's persistent operating contract
- `workspace/.agents/skills/` — the skills the reading agent loads (the method)
- `workspace/memories/` — tracked cross-run memory: a near-stable orientation file plus typed dated entries
- `workspace/code/` — ignored lab bench for cloned repos, probes, patches, experiments, and local environments
- `workspace/runs/` — delivered reports, durable evidence packets, and `INDEX.md` (the readable record + dedup log)
- `.github/workflows/verify-run.yml` — CI gate on run PRs: reruns `verify_run.py --mode final` for every run packet touched by the PR
- `.github/workflows/secret-scan.yml` — CI gate on every PR: runs `scan_secrets.py` over the PR diff
- `scripts/scan_secrets.py` + `scripts/scan_secrets.allowlist` — zero-dependency scanner for secrets, credentials, and personal/machine-identifying info; masks secret matches in output, exits nonzero on findings
- `scripts/git-hooks/` — tracked `pre-commit` (staged secret scan) and `pre-push` (range secret scan + run-packet verification, mirroring CI); enable once per clone with `git config core.hooksPath scripts/git-hooks`

## The Two Agents

The word "agent" is overloaded, so this contract uses two specific terms.

### Reading Agent

The agent that **executes** scouting runs. It operates from `workspace/`, reads `workspace/AGENTS.md` as its persistent contract, follows the skills under `workspace/.agents/skills/` as its method, and uses `prompt.txt` (or `prompt-lite.txt`) as the per-run trigger. It scouts, filters, investigates, writes, delivers to Feishu, and appends to `runs/INDEX.md`.

### Coding Agent

The agent that **develops and maintains** this repository — that is you, if you are reading this file. Coding agents edit the prompts, the skills, the launcher, and the structure. Every edit changes how the reading agent behaves, which changes what the user receives.

## The Instruction Hierarchy

When the reading agent encounters conflicting guidance, it resolves in this order:

1. **Per-run prompt** (`prompt.txt` for `full`, `prompt-lite.txt` for `lite`) — most specific; wins on run-scoped parameters like stream, date, and focus.
2. **Reading-agent contract** (`workspace/AGENTS.md`) — stable user preferences; wins on identity, style, policy, destination.
3. **Skills** (`workspace/.agents/skills/`) — the method; win on how each workflow step is executed.

Keep this hierarchy clean. Do not put run-scoped concerns in a skill. Do not put method details in the reading-agent contract. Do not put stable preferences in the prompt.

Note that the reading-agent contract may deliberately **expand** a skill's defaults (for example, broadening the investigation policy beyond the skill's read-only baseline). That is expected: when the contract and a skill differ, the contract wins.

## File Purposes and Boundaries

Each file has a distinct job. Do not let them bleed into each other.

**`README.md`** — the project entry point. Audience: humans and agents landing on the repo. A brief description, how a run works, and how to fork the spirit. Keep it short.

**`AGENTS.md`** (this file) — the coding-agent contract. How to maintain the repo without breaking the reading agent. No run-phase detail, no user preferences.

**`prompt.txt` / `prompt-lite.txt`** — the run triggers (`full` and `lite` streams). Date (injected by `scout.sh`) plus the stream-scoped kick-off instruction. Stays thin: if you are adding more than a few lines, the content belongs in `workspace/AGENTS.md` or a skill instead.

**`workspace/AGENTS.md`** — the reading-agent contract. Identity, user preferences, policy, delivery destination, coverage-log rules. Concrete values, not templates. No method detail, no run-phase orchestration.

**`workspace/.agents/skills/workspace-manage/SKILL.md`** — workspace layout, naming rules, artifact tracking, verifier script, cleanup policy, thread shapes (including reading-only `README.md` threads), and `runs/INDEX.md` coverage log with per-stream dedup.

**`workspace/.agents/skills/memory-manage/SKILL.md`** — cross-run memory retrieval, typed execution/hypothesis schemas, promotion rules, provenance, and update policy.

**`workspace/.agents/skills/paper-source/SKILL.md`** — recent research-candidate discovery and full-text acquisition. Hugging Face Papers is the default paper source; Scientific Spaces RSS supplies research essays. Commands and fallbacks are documented inline.

**`workspace/.agents/skills/paper-deep-dive/SKILL.md`** — deep investigation of a selected paper or research article: reading, code inspection, resource-proportional research actions chosen by run stream (`full` favors real experiments, `lite` favors source tracing and derivation), related-work comparison, and structured analysis notes.

**`workspace/.agents/skills/report-compose/SKILL.md`** — DocxXML research report composition as a gated pipeline: four-beat concrete-first thread outlines (puzzle → minimal mechanism → selective evidence → updated model), report-level and thread-level outlines, scope-matched writer briefs, explicit model routing for writing and review, four human review gates plus a machine gate, sequential no-lookahead cold reading, Feishu doc creation, media insertion, user notification, and run-packet report preservation. Formatting rules, gate definitions, the anti-AI-tone negative list, and the manual delivery flow live in `references/`; `scripts/push_report.py` automates chunked push, anchor-based image placement, and verification. Uses `lark-doc` and `lark-im` for command details.

**`workspace/.agents/skills/remote-compute/SKILL.md`** — routing GPU-heavy or otherwise compute-bound research actions to the Inspire platform during `full`-stream deep dives, with unchanged evidence-promotion rules; `lite` runs stay on lightweight local or Colab checks. Defers command details to the harness-level `inspire` skill and project context to `workspace/INSPIRE.md`.

**`workspace/INSPIRE.md`** — project-level Inspire platform context (default workspace, image, path conventions, live workloads). Gitignored and local-only — it may accumulate sensitive operational details, so each deployment maintains its own; do not commit it.

**`scout.sh`** — the launcher. Computes the date, selects the prompt and launches the reading agent from `workspace/`. The only file that knows the harness invocation.

## Editing Standards

- Read a file fully before editing it. These are behavioral contracts; partial understanding leads to contradictions.
- Check whether a change belongs in the file you are editing or another one. Use the file purposes above.
- After editing, verify the instruction hierarchy stays consistent — no contradictions between the prompt, the reading-agent contract, and the skills.
- Preserve the existing voice and density. These files are concise; do not inflate them with hedging, repetition, or meta-commentary.
- Workspace instruction files (`prompt.txt`, `workspace/AGENTS.md`, the skills) and delivered reports are written in Chinese; keep them in Chinese when editing. Root-level files stay in English.
- Do not give the reading agent instructions it cannot reasonably follow. If an instruction depends on a capability that may be absent, add a fallback.

## What Not to Add

- Executable application code. The runtime is the reading agent, not a script. Repo guards (`verify_run.py`, `scan_secrets.py`) are verifiers, not application code.
- Example outputs. They go stale and become de facto templates the reading agent imitates instead of thinking through.
- A setup/install/distribution layer. That architecture was deliberately removed.

## Workspace Invariants

The reading agent's home is `workspace/`. These invariants must hold across prompt and skill changes:

```text
workspace/
├── AGENTS.md          # reading-agent contract
├── INSPIRE.md         # project-level Inspire platform context (gitignored, local-only)
├── .agents/skills/    # the skills (the method)
├── memories/
│   ├── MEMORY.md         # near-stable research orientation
│   └── <date>-memory.md  # typed cross-run memory from a scouting run
├── papers/            # tracked paper and research-article markdown cache, organized by area
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

- `runs/INDEX.md` is the single source of truth for what has been covered. Each entry carries `Stream: full` or `Stream: lite` (legacy entries default to `full`), so dedup rules can differ per stream while the file stays readable by both the reading agent and the user. If you change how logging works, the reading agent must still be able to determine what was previously covered from this file.
- A `lite` run skips any previously covered paper; a `full` run skips `full` coverage but may upgrade a paper covered only by `lite`.
- `memories/MEMORY.md` is near-stable orientation, not a per-run retrieval map. Dated memory files hold typed `execution` and `hypothesis` entries with keywords and provenance.
- The reading-agent contract is stable. Regenerating the prompts should not require changing `workspace/AGENTS.md`, and vice versa.
- The prompts stay thin — run-scoped parameters only. `prompt.txt` names the `full` stream, `prompt-lite.txt` the `lite` stream; both streams share one writing quality bar and the same review gates.
- One run produces one Feishu doc. Delivery is append-new, not update-existing.
- Run packets stay light: `verify_run.py` enforces per-file (2 MB) and per-run (20 MB) caps and bans weights/checkpoints. Raw outputs live in the ignored workspace `code/` or are regenerable from preserved scripts.
- A thread may consist of `README.md` alone when its evidence is reading, source tracing, cross-paper triangulation, or derivation; `code/` and `patches/` stay optional in every stream.
- Tracked content stays free of secrets, credentials, and personal/machine-identifying info: `scan_secrets.py` gates this in CI and in the tracked local hooks. False positives go to `scripts/scan_secrets.allowlist` with a justification comment.
- The reading agent's investigation permissions are governed by its contract. The skill defines a read-only baseline; the contract may expand it, and the stream it runs under narrows the compute it may spend.

If you change the `runs/INDEX.md` format, keep it readable by both the reading agent and the user, and append-only (newest first when practical).
