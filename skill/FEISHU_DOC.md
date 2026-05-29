---
name: paper-scout-feishu-doc
description: "Structure and write the Paper Scout brief as a Feishu document using lark-cli (v2 DocxXML). Covers document sections, callouts, grids, tables, LaTeX, and the v2 delivery command sequence."
user-invocable: false
---

# paper-scout-feishu-doc

This skill governs how the Paper Scout brief is written and delivered as a Feishu document. It is used during Phase 5 (synthesis and writing) and Phase 6 (delivery) of the main `paper-scout` skill.

This skill owns the **brief's structure and editorial standards** — which sections exist, what goes in each, and the quality bar. It does **not** own the Lark document format. The installed `lark-doc` skill is the authoritative source for DocxXML syntax, escaping, and command flags. Load it before any delivery work and follow its references. This separation is deliberate: when `lark-cli` changes, the format details update in one place, not here.

---

## Before You Start

Load `lark-doc` before any writing or delivery work, and follow its references (`lark-doc-xml.md` for syntax, `lark-doc-create.md` / `lark-doc-update.md` for commands, `lark-doc-style.md` for visual richness).

The current `lark-doc` skill uses the **v2 API with DocxXML** as the default content format. All `docs +create`, `docs +fetch`, and `docs +update` commands must carry `--api-version v2`. Author the brief as DocxXML.

Feishu renders DocxXML with real visual structure — callouts, grids, tables with styled cells, inline LaTeX. Use it. A brief that is nothing but flat bullet lists misses the platform entirely.

**Brief-specific rules (defer to `lark-doc` for everything else):**

- The document title is the `<title>` element at the very start of the content. There is no `--title` flag in v2. Do not repeat the title anywhere else in the body.
- Feishu generates a table of contents automatically from headings. Use a sensible `<h1>`/`<h2>` hierarchy.
- Callout children must be block elements (`<p>`, headings, lists, `<checkbox>`, `<blockquote>`) — not bare text, and not tables or code blocks.
- In DocxXML, tags are never escaped, but `<`, `>`, and `&` inside text content must be written as `&lt;`, `&gt;`, `&amp;`. Paper titles, math, and code snippets frequently contain these — escape them in text, and see `lark-doc-xml.md` for the full rule.

---

## Document Structure

### 1. Opening Synthesis (Required)

Write 2–4 sentences answering: *What mattered this period?*

This is a judgment, not a summary. What was the dominant theme? What was the standout paper? What should the user notice before reading further?

If a paper was clearly exceptional this period, wrap the synthesis in a standout callout:

```xml
<callout emoji="✅" background-color="light-green">
  <p><b>Standout this week:</b> [Title] is the clearest example of [trend/contribution]. [One sentence on why.]</p>
</callout>
```

If the period was weak, say so in plain prose. Do not pad.

---

### 2. Shortlist Table

All shortlisted papers in one table. Use a standard DocxXML `<table>` so each cell can hold links, bold text, and short lists cleanly. Give it a header row in `<thead>` and use `<colgroup>` to set column widths.

Four columns: Paper, Key Contribution, Why It Matters, Links.

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
      <td><b>Title</b> (Author et al., 2025)</td>
      <td>One sentence — the specific thing the paper did.</td>
      <td>One sentence — why this user should care.</td>
      <td><a href="url">HF</a> / <a href="url">PDF</a></td>
    </tr>
  </tbody>
</table>
```

Keep "Key Contribution" and "Why It Matters" distinct — contribution is what they did, relevance is why it matters to this user. Do not write the same sentence twice.

For papers that also appear in a deep dive section below: bold the row title and add `→ see deep dive` in the contribution cell.

---

### 3. Deep Dive Sections

One `<h2>` section per deep-dived paper, separated by `<hr/>`.

#### Header

Start with the paper title as the `<h2>`, and put author, year, and links on the immediately following paragraph:

```xml
<h2>Paper Title</h2>
<p>Author et al. (2025) · <a href="url">HF</a> · <a href="url">PDF</a></p>
```

#### What It Does

One or two sentences — the core claim, stated specifically, not the abstract. Prefer the concrete: "The paper shows that…" or "The authors introduce X, which…".

#### How It Works

The key mechanism. Dense and specific. If the paper has a clear pipeline, describe it step by step with an ordered list (`<ol><li seq="auto">…</li></ol>`). If it has a notable formula or loss term, write it with inline LaTeX:

```xml
<p>The training objective is <latex>\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KL}}</latex>.</p>
```

For a display equation, give it its own centered paragraph:

```xml
<p align="center"><latex>\hat{y} = \arg\max_{y} P(y \mid x; \theta)</latex></p>
```

If the method contrasts two design choices directly, use a two-column grid:

```xml
<grid>
  <column width-ratio="0.5">
    <p><b>Prior approach</b></p>
    <p>Description of what prior work did.</p>
  </column>
  <column width-ratio="0.5">
    <p><b>This paper</b></p>
    <p>Description of what this paper does differently.</p>
  </column>
</grid>
```

#### Evidence

What the key results show and how convincing they are. Be direct:

- "Outperforms X by Y% on Z benchmark" is useful.
- "Achieves state-of-the-art" is not useful.

If there are notable red flags or caveats, use a warning callout:

```xml
<callout emoji="⚠️" background-color="light-yellow">
  <p>Baseline comparisons exclude [Method X], the strongest prior approach in this setting.</p>
</callout>
```

#### Artifacts (If Inspected)

One sentence or a short bullet: what the repo or project page reveals about maturity, usability, or gaps between the paper and the implementation.

#### Judgment

One or two sentences: why this paper was worth the deep dive, and what the user should do with it. Worth reading in full? Worth building on? Worth tracking follow-ups?

---

### 4. Themes And Patterns (Optional)

A short `<h2>` section after all deep dives, only when the pool shows a clear convergence, absence, or shift worth naming.

Write as an observation in prose, not a bullet list. One or two paragraphs.

---

## Callout Reference

Use callouts for things that demand attention, not as decoration. Three types are useful for paper briefs. Callout children must be block elements; see `lark-doc-xml.md` for the supported attributes and color names.

**Standout paper or exceptional finding:**
```xml
<callout emoji="✅" background-color="light-green">
  <p>[Content]</p>
</callout>
```

**Caveat, red flag, or important limitation:**
```xml
<callout emoji="⚠️" background-color="light-yellow">
  <p>[Content]</p>
</callout>
```

**Notable insight or key insight:**
```xml
<callout emoji="💡" background-color="light-blue">
  <p>[Content]</p>
</callout>
```

---

## Delivery Commands

All commands carry `--api-version v2`. Content is DocxXML, passed via `--content` (default `--doc-format xml`). For multi-line content, prefer `--content @path/to/file.xml` or `--content -` (stdin) over inline strings — write the brief to `output/` first and pass the file. See `lark-doc-create.md` / `lark-doc-update.md`.

### Creating The Document

```bash
lark-cli docs +create --api-version v2 \
  --parent-token <token> \
  --content @output/brief.xml
```

The `<title>` element inside the content sets the document title — there is no `--title` flag. The destination is set with `--parent-token <folder-or-wiki-node-token>` (or `--parent-position my_library` for a personal library). The destination type is recorded in the workspace instruction file. Note that v2 uses `--parent-token` for both folders and wiki nodes — there are no separate `--folder-token` / `--wiki-node` / `--wiki-space` flags.

Capture `data.document.document_id` and `data.document.url` from the response.

### Appending Sections For Long Briefs

`lark-doc` recommends creating a skeleton first (title + headings + the opening synthesis + shortlist table), then appending each deep-dive section:

```bash
lark-cli docs +update --api-version v2 \
  --doc "<document_id>" \
  --command append \
  --content @output/section.xml
```

Use the `document_id` from the create response, not the `url`, especially for wiki destinations.

### Typical Sequence

1. Create the doc with the opening synthesis and shortlist table.
2. Append each deep-dive section.
3. Append the themes section if present.
4. Record the document `url` from the create response for the coverage log.

---

## What Not To Do

- Do not omit the `<title>` element — there is no `--title` flag in v2, so the title must live in the content.
- Do not repeat the title as an `<h1>` after the `<title>` — it duplicates the heading.
- Do not put bare text directly inside a `<callout>`; wrap it in `<p>` or another block element. Do not put tables or code blocks inside a callout.
- Do not use a Markdown `>` blockquote expecting a callout. Use the `<callout>` tag.
- Do not use `overwrite` to fix a brief mid-run. Use `append`, or the block-level edit commands documented in `lark-doc-update.md`.
- Do not write "the paper achieves state-of-the-art." Write the number, the benchmark, and the comparison.
- Do not use a flat bullet list for the entire brief. The shortlist is a `<table>`. Deep dives are `<h2>` sections. Judgment paragraphs are prose.
- Do not skip the opening synthesis. The user should understand what mattered before scrolling to individual papers.
- Do not hand-author low-level XML escaping from memory. Follow `lark-doc-xml.md`.

---

## Checklist Before Delivery

- [ ] `lark-doc` loaded and its references followed
- [ ] Content authored as DocxXML; all commands use `--api-version v2`
- [ ] `<title>` element present and matches the configured title pattern; not duplicated in the body
- [ ] Opening synthesis written — 2–4 sentences on what mattered this period
- [ ] Shortlist in a `<table>` with specific contribution and relevance columns
- [ ] Each deep dive: specific contribution, how it works, evidence quality, judgment
- [ ] Callouts used for standout papers and notable caveats — not decoratively
- [ ] `<latex>` used for any equations the paper centers on
- [ ] `<grid>` used for any direct method comparisons worth visualizing
- [ ] `<`, `>`, `&` in text content escaped per `lark-doc-xml.md`
- [ ] Doc created via `docs +create --api-version v2`; long briefs appended via `docs +update --api-version v2 --command append`
- [ ] Document `url` recorded for the coverage log
