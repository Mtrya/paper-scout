---
name: report-compose
description: "Compose, publish, and archive the Paper Scout research report as Lark DocxXML, including structure, illustrative artifacts, local media insertion, Feishu delivery, and user notification. Load lark-doc and lark-im for command details."
user-invocable: false
---

# report-compose

This skill owns the Write and Deliver steps. The output is one canonical research report: deep-dive notes, paper assets, code probes, and MinerU outputs are source traces, not parallel user-facing deliverables.

Load `lark-doc` before drafting or publishing. Follow its DocxXML references for syntax, especially `references/lark-doc-xml.md`. Load `lark-im` before notification. For local image placement, follow `references/figure-embedding.md`.

## Report Contract

Recent papers are research seeds, not the boundary of the report. Start from what is new, then follow the strongest threads: related papers, code, artifacts, toy checks, equations, and buildable questions. The report should make the interesting findings quickly understandable.

Illustrative artifacts are first-class: equations, code snippets, pseudocode, paper figures, curated diagrams, and real tables. Their purpose is not to prove the report true, but to demonstrate the mechanism, result, contrast, or failure mode faster than prose can.

Plain prose is connective tissue. It should orient the reader, explain what to notice, and give a judgment. Do not let the final report become only paraphrase.

## Format Rules

- Author as DocxXML. v2 API. No `--title` flag - use a `<title>` element at the start and do not repeat it in the body.
- Use a sensible `<h1>` / `<h2>` hierarchy. Feishu auto-generates a table of contents.
- Callout children must be block elements (`<p>`, headings, lists). No bare text, no tables or code blocks inside callouts.
- Escape `<`, `>`, `&` as `&lt;`, `&gt;`, `&amp;` in text and code.
- DocxXML is a fragment format with multiple top-level blocks. Do not validate it with a standard single-root XML parser.
- Temporary media anchors must be standalone top-level paragraphs, unique within the doc, and easy to delete after media insertion, for example `<p>[[figure-anchor:paper-slug:overview]]</p>`.

## Structure

Organize by theme, not as a flat list.

```xml
<title>Paper Scout Research Report - YYYY-MM-DD</title>
Opening synthesis (2-4 sentences)
<h1>Theme A</h1>
  mini-synthesis (1-3 sentences)
  shortlist table (lightly-noticed papers only; omit if none)
  <h2>Deep-dive paper</h2> + research narrative + illustrative artifacts
  <hr/>
  <h2>Deep-dive paper</h2> + research narrative + illustrative artifacts
<h1>Theme B</h1>
  ...
Optional: <h1>Cross-cutting observations</h1>
```

### Opening Synthesis

Answer: *What did this run discover?* Name the themes the rest of the report will use.

If one paper was clearly exceptional:

```xml
<callout emoji="✅" background-color="light-green">
  <p><b>Standout:</b> [Title] is the clearest seed for [research thread]. [Why.]</p>
</callout>
```

If the period was weak, say so plainly. Do not pad.

### Theme Sections

One `<h1>` per actual theme. Derive themes from the research threads, not from a fixed taxonomy.

Rules:

- Each paper appears **exactly once**.
- Put deep-dived papers as `<h2>` research sections inside their theme.
- Put lightly-noticed papers as rows in a per-theme shortlist table.
- Fold one-stray-paper themes into a broader theme or a catch-all.
- If the pool does not cluster, use a single `<h1>Highlights</h1>`.

#### Shortlist Table

Four columns: **Paper**, **Key Contribution**, **Why It Matters**, **Links**.

```xml
<table>
  <colgroup>
    <col width="200"/>
    <col width="230"/>
    <col width="230"/>
    <col width="90"/>
  </colgroup>
  <thead>
    <tr>
      <th background-color="light-gray">Paper</th>
      <th background-color="light-gray">Key Contribution</th>
      <th background-color="light-gray">Why It Matters</th>
      <th background-color="light-gray">Links</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Title</b> (Author et al., 2026)</td>
      <td>What they did.</td>
      <td>Why this user should care.</td>
      <td><a href="url">HF</a> / <a href="url">PDF</a></td>
    </tr>
  </tbody>
</table>
```

Keep contribution and relevance distinct. Omit the table if every paper in the theme is deep-dived. Tables with paragraph-length cells do not count as illustrative artifacts.

### Deep Dive Sections

Each deep dive is an `<h2>` inside its theme, separated by `<hr/>`.

Header format:

```xml
<h2>Paper Title</h2>
<p>Author et al. (2026) · <a href="url">HF</a> · <a href="url">PDF</a></p>
```

Write as fluent narrative, not a fixed template. Every deep dive should explain the research thread, the core mechanism, what the investigation found, and the bottom-line judgment.

Use DocxXML features where they carry understanding:

- `<latex>` for equations that clarify the mechanism.
- `<pre lang="python" caption="..."><code>...</code></pre>` for compact code or pseudocode snippets.
- `<grid>` for two-way comparisons.
- `<table>` for structured multi-way comparisons, metrics, or ablations.
- Local image anchors for paper figures, qualitative examples, failure cases, or curated diagrams that should appear near the relevant prose.

### Illustration Plan

Before writing the final draft, keep a scratch illustration plan for each deep-dive section:

- artifact type: equation / code / pseudocode / paper figure / curated diagram / real table
- source: paper asset, codebase, related paper, toy check, or agent-created diagram
- purpose: what the artifact helps the reader understand
- placement: nearby paragraph or temporary media anchor

Aim for at least half of the report's reader value to come from illustrative artifacts when the material supports it. If a deep dive has no useful artifact, the prose should make clear why.

### Cross-cutting Observations

Add a closing `<h1>` only for a pattern that cuts across themes: a convergence, absence, shift, reusable method idea, or open research question. Skip if there is nothing to say.

## Delivery

All `docs` commands carry `--api-version v2`. Create and update as the bot.

Before creating the real doc, run the same create command with `--dry-run`:

```bash
lark-cli docs +create --api-version v2 --dry-run --as bot \
  --content @drafts/report.xml
```

Create the doc:

```bash
lark-cli docs +create --api-version v2 --as bot \
  --content @drafts/report.xml
```

The bot owns the resulting doc. There is **no `--parent-token`** and no configured folder/wiki destination - do not add one. The `<title>` element inside the content sets the document title. Capture `data.document.document_id` and `data.document.url` from the response.

If the report is long enough to risk a single create call becoming unwieldy, create a skeleton first, then append each theme with `docs +update --command append`; dry-run each append before the real update.

After the doc exists, insert local media into the temporary anchors, delete the anchors, and fetch once to verify order. Follow `references/figure-embedding.md` for the exact command shape.

Then load `lark-im` and send the user a direct message containing the doc `url`. A run is complete only once this direct message is sent and confirmed. If recipient resolution or sending fails, stop and report it.

After the DM is confirmed:

1. Archive the delivered DocxXML to `../reports/<YYYY-MM-DD>-<slug>.docxxml`.
2. Preserve the document `url` so `workspace-manage` can record it in `runs/INDEX.md`.

## What Not To Do

- Do not omit or duplicate the `<title>`.
- Do not use `overwrite` to fix a report mid-run. Use `append`, or the block-level edit commands documented in `lark-doc`.
- Do not put bare text, tables, or code blocks inside a `<callout>`.
- Do not write "achieves state-of-the-art." Write the number, benchmark, and comparison.
- Do not use a flat bullet list for the whole report.
- Do not stamp the same five sub-headers onto every deep dive.
- Do not list a deep-dived paper in a shortlist table or use "see deep dive" pointers.
- Do not present one giant shortlist table for the whole pool.
- Do not hardcode the theme list.
- Do not ship a deep dive that restates the abstract.
- Do not leave temporary media anchors visible in the delivered doc.
- Do not skip the DM and consider delivery complete.
- Do not archive to `reports/` before delivery is confirmed.
- Do not invent recipient resolution rules; follow `lark-im`.

## Checklist

- [ ] `lark-doc` loaded
- [ ] `lark-im` loaded before notification
- [ ] DocxXML draft in `drafts/report.xml`
- [ ] `<title>` present, not duplicated
- [ ] Opening synthesis names research themes
- [ ] Each paper appears exactly once
- [ ] Shortlist tables are per-theme and lightly-noticed only
- [ ] Each deep dive has an illustration plan
- [ ] Illustrative artifacts included where they clarify the finding
- [ ] Local media anchors inserted, resolved, deleted, and verified
- [ ] Callouts used only where they earn attention
- [ ] `<`, `>`, `&` escaped per `lark-doc`
- [ ] Create/update commands dry-run before live write
- [ ] Feishu URL captured
- [ ] User DM confirmed
- [ ] Delivered DocxXML archived to `../reports/`
- [ ] URL handed to `workspace-manage` for `runs/INDEX.md`
