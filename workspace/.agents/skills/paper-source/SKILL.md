---
name: paper-source
description: "Discover and pull a broad recent pool of papers from the configured source. Default is Hugging Face Papers; other sources will be added to this skill as the workflow grows."
user-invocable: true
---

# paper-source

This skill covers how to gather papers.

## What to Gather

For each candidate, preserve enough metadata to support later triage:

- title
- paper id and any alternative identifiers
- authors
- abstract or summary snippet
- links to project pages, repos, model cards, or dataset cards when available

If you persist the candidate pool to disk, write it to `drafts/` (scratch, may be overwritten) — never to `runs/`, which holds only durable notes.

## Source A: Hugging Face Papers

Use the `hf` CLI for quick paper scouting and reading.

List papers:

```bash
hf papers ls --sort trending --limit N
hf papers ls --date YYYY-MM-DD --limit N
hf papers ls --week YYYY-Www --limit N
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

## Source B: ArXiv PDF + MinerU

Use ArXiv PDF + MinerU for finer-grained control, better support for same-day papers, and more reliable full-text and figure/table extraction.

End-to-end script:

```bash
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug>
```

The script downloads the PDF to `drafts/`, submits the arXiv PDF URL to MinerU, polls for completion, extracts the result zip under `drafts/`, copies the resulting Markdown to `papers/<area>/<slug>-<paper-id>.md`, then removes the transient PDF, zip, and task/result JSON files. It assumes `curl`, `jq`, `unzip`, and a MinerU token at `~/.config/mineru/token`.

Useful options:

```bash
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug> --model pipeline
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug> --copy-images
```

With `--copy-images`, the script also copies extracted images from `drafts/` to `assets/` for easier reference during deep dives and writing.

If you only need the PDF audit artifact, use:

```bash
.agents/skills/paper-source/scripts/fetch-arxiv-pdf.sh <paper-id> <slug>
```
