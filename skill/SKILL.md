---
name: paper-scout
description: "Scout a recent pool of papers, filter aggressively, deeply investigate the most promising ones, write Lark DocxXML, and create a fresh Feishu doc for each run."
user-invocable: true
---

# paper-scout

Paper Scout is a two-speed paper reading workflow:

- first, scan broadly to identify what matters in a large recent pool
- then, investigate deeply only where the expected value is high

The goal is not to summarize everything. The goal is to help a user understand what matters now, what deserves attention, and which papers justify deeper investment.

## When To Use This Skill

Use this skill when the user wants a recent-paper brief that combines:

- broad scouting across a large candidate pool
- strong filtering
- a small shortlist worth noticing
- a few deep investigations with stronger analysis
- delivery to Feishu docs

This skill defaults to Hugging Face papers via `hf papers`, but it may be used with other configured sources when the workspace instructions or run trigger allow it.

## Required Companion Skills

Before source discovery, load `hf-cli`.

Before creating Feishu docs, load `lark-doc`.

During Phase 4 (deep investigation), load and follow `paper-scout-deep-dive` (`skill/DEEP_DIVE.md`).

During Phase 5 (synthesis and writing), load and follow `paper-scout-feishu-doc` (`skill/FEISHU_DOC.md`).

Use this `paper-scout` skill as the main operating method throughout the run.

## Core Principles

### Filter Hard

Do not produce a padded report. If the period is weak, cover fewer papers.

### Use Adaptive Effort

Spend little time on mediocre candidates and more time on genuinely promising ones.

### Be Skeptical But Not Cynical

Look for red flags, overstated claims, weak baselines, shallow repos, or missing evidence, but do not force criticism where it is not warranted.

### Investigate Beyond The Paper Selectively

Project pages, repos, model cards, and dataset cards are valuable when a paper looks especially promising. They are not mandatory for every paper.

### Avoid Repeating Prior Deep Dives

Previously deep-dived papers should not be deep-dived again. Previously shortlisted papers may reappear unless they are clearly outdated or no longer relevant.

### Preserve User Trust

Prefer accurate judgment, clear rationale, and honest uncertainty over false confidence.

## Workspace Contract

Default workspace root:

`~/.paper-scout/workspace/`

Recommended structure:

```text
workspace/
├── papers/
├── repos/
├── runs/
├── output/
└── state/
    └── log.md
```

### Directory Roles

- `papers/`: downloaded paper markdown and related lightweight paper artifacts
- `repos/`: cloned repositories for code inspection
- `runs/`: per-run notes, scratch analysis, and temporary structured artifacts
- `output/`: final Lark DocxXML before delivery
- `state/log.md`: the persistent coverage log

## State Rules

Before serious scouting or investigation, inspect `state/log.md`.

Use the log to identify:

- papers already deep-dived
- recently covered time windows
- recent Feishu doc links
- repeated themes or stale follow-ups

Deep-dived papers should not be deep-dived again unless the workspace instructions or current run trigger explicitly override that rule.

## Investigation Boundaries

### Allowed

- download paper markdown locally, for example with `hf papers read <id> > <path>`
- inspect project pages
- inspect linked GitHub repositories
- clone repositories locally
- scan codebases and documentation
- inspect Hugging Face model cards or dataset cards

### Not Allowed By Default

- do not run repositories
- do not download models
- do not download datasets
- do not perform heavyweight execution or benchmarking

If the workspace instructions or current run trigger explicitly expand permissions, follow them. Otherwise keep investigation read-only and lightweight.

## Phase 0: Preflight

Before scouting:

1. Load `hf-cli`.
2. Confirm the workspace root exists or can be created.
3. Inspect `state/log.md` if present.
4. Verify the output and scratch directories are available.
5. Before delivery work later in the run, remember to load `lark-doc`.

If source access or delivery readiness is broken, stop early and explain the blocker.

## Phase 1: Source Discovery

Start from the configured recent-paper source. By default, this is `hf papers`.

Gather a broad recent pool and preserve enough metadata to support later selection, such as:

- title
- paper id and any alternative identifiers
- authors
- abstract or summary snippet
- links to project pages, repos, model cards, or dataset cards when available

The initial pool can be large. Do not investigate each item deeply.

## Phase 2: Fast Scan

Use a lightweight scan to identify promising candidates efficiently.

During fast scan:

- read enough to judge novelty, relevance, plausibility, and likely value
- notice bigger-picture patterns across the pool
- identify papers that are clearly weak, derivative, or outside scope
- surface candidates worth mentioning even if they do not justify a deep dive

At this phase, speed matters. Avoid expensive investigation unless a paper is already emerging as unusually promising.

## Phase 3: Selection

Select two nested sets:

- a shortlist worth noticing
- a smaller set worth deep investigation

Use importance and user relevance as the main criteria. Use diversity as a tie-breaker if the top candidates are too homogeneous.

Signals that strengthen a paper's case include:

- unusually strong or surprising idea
- clear relevance to the user's interests
- practical usefulness
- credible evidence
- signs that the work may influence future papers or workflows
- code or project artifacts that make deeper inspection more worthwhile

Signals that weaken a paper's case include:

- vague contribution
- hype without evidence
- weak or suspicious evaluation
- unclear novelty
- shallow or missing supporting artifacts when they matter

The exact quotas come from the workspace instructions or current run trigger. If the period is weak, select fewer papers.

## Phase 4: Deep Investigation

For each selected deep-dive candidate, load and follow the `paper-scout-deep-dive` skill (`skill/DEEP_DIVE.md`).

That skill defines the full investigation process: section inventory, motivation and contribution analysis, method walkthrough, experimental evidence assessment, artifact inspection, and bottom-line judgment. Follow it in full for every deep-dive paper.

A light summary of what the skill covers:

- **Build a section inventory first.** List every section and where it will appear in the analysis. This prevents important content from being silently skipped.
- **Work through the full paper.** Do not stop at the method section. Experiments, ablations, and appendices contain the evidence that tests the claims.
- **Write specific analysis notes.** The goal is not to summarize. It is to understand the paper well enough to write a genuinely useful brief section.
- **Inspect artifacts selectively.** Repos, model cards, and project pages add value when the paper is genuinely promising. Do not let artifact inspection become a detour.
- **Write a bottom-line judgment.** Is it worth the user's time? Should they track follow-up work? Be direct.

Save analysis notes for each paper to `runs/<paper-id>-deep-dive.md`. These notes feed Phase 5 directly.

Do not let deep investigation turn into a full reproduction effort.

## Phase 5: Synthesis And Writing

Load and follow the `paper-scout-feishu-doc` skill (`skill/FEISHU_DOC.md`).

That skill defines the document structure, visual hierarchy, formatting conventions, and quality standards for the final brief. Follow it for layout and writing decisions.

A light summary of what the skill covers:

- **Open with a synthesis.** The user should understand what mattered this period before reading individual papers.
- **Use a shortlist table** for all shortlisted papers, with specific contribution and relevance columns.
- **Write structured H2 sections** for each deep-dived paper: what it does, how it works, the evidence, and a judgment.
- **Use rich formatting.** Tables, callouts (blockquotes), and horizontal rules make the brief navigable. Avoid flat bullet lists for everything.
- **Be specific, not promotional.** Every claim should be grounded in what the paper actually says or what the analysis found.

Writing tone and depth are controlled by the workspace instructions or current run trigger. The reasoning standard should remain high regardless of style choice.

## Phase 6: Delivery

Before delivery, load `lark-doc`.

Write the brief as Lark DocxXML into `output/` first.

Then create a fresh Feishu doc in the configured destination for this run. The destination is expected to be a folder or wiki space rather than an existing doc to update.

Preserve the resulting document URL or identifier for logging.

## Phase 7: Logging And Optional Cleanup

After successful delivery, append a new entry to `state/log.md`.

Each run entry should record:

- run date and time
- period covered
- resulting Feishu doc link
- shortlisted papers
- deep-dived papers
- useful identifiers for each paper
- a brief rationale for deep-dive selections when practical

The log should be written newest first when convenient for readability.

After logging, optionally perform a light cleanup if clearly safe:

- prune stale scratch files in `runs/`
- prune old downloaded paper markdown if it is no longer useful
- prune old cloned repos if they are large and clearly stale

Do not delete recent or obviously useful material without a good reason.

## Final Checklist

Before finishing a run, confirm:

- required companion skills were loaded at the right times
- the workspace state was checked before serious work
- previously deep-dived papers were not repeated
- the shortlist was filtered aggressively rather than padded
- deep investigation stayed read-only and lightweight
- the final brief reflects both broad scouting and deeper analysis
- a fresh Feishu doc was created
- `state/log.md` was updated

If any of these failed, say so clearly instead of implying the run was complete.
