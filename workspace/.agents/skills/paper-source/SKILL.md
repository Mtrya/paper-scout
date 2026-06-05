---
name: paper-source
description: "Discover and pull a broad recent pool of papers from the configured source. Default is Hugging Face Papers; other sources will be added to this skill as the workflow grows."
user-invocable: true
---

# paper-source

This skill covers how to gather papers.

The default source is Hugging Face Papers.

## What to Gather

For each candidate, preserve enough metadata to support later triage:

- title
- paper id and any alternative identifiers
- authors
- abstract or summary snippet
- links to project pages, repos, model cards, or dataset cards when available

If you persist the candidate pool to disk, write it to `drafts/` (scratch, may be overwritten) — never to `runs/`, which holds only durable notes.

## Source A: Hugging Face Papers

Use the `hf` CLI. The commands below cover the common scouting cases.

List daily papers:

```bash
hf papers list --date YYYY-MM-DD --limit N
```

Read a paper as markdown:

```bash
hf papers read <paper-id> > papers/<area>/<slug>-<paper-id>.md
```

Get paper metadata:

```bash
hf papers info <paper-id>
```

Search papers by keyword:

```bash
hf papers search "<query>" --limit N
```

Use `--format json` when structured output is easier to process.
