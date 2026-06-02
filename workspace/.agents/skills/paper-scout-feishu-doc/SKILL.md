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

The brief is **organized by theme**, not as one flat list. The top-level shape is: a whole-pool synthesis, then one `<h1>` section per theme that emerged this run, with that theme's shortlist and deep dives nested inside it. Derive the themes from the papers each run — do not hardcode them.

### 1. Opening Synthesis (Required)

Write 2–4 sentences answering: *What mattered this period?*

This is a judgment about the whole pool, not a summary. What were the dominant themes? What was the standout paper? What should the user notice before reading further? Name the themes the rest of the doc will use.

If a paper was clearly exceptional this period, wrap the synthesis in a standout callout:

```xml
<callout emoji="✅" background-color="light-green">
  <p><b>Standout this week:</b> [Title] is the clearest example of [trend/contribution]. [One sentence on why.]</p>
</callout>
```

If the period was weak, say so in plain prose. Do not pad.

---

### 2. Theme Sections (The Body)

One `<h1>` per theme that actually emerged from the pool (e.g. "Video & World Models", "Embodied / Robotics", "Multimodal LLMs", "Autonomous Driving"). Each theme section contains, in order:

1. A short **mini-synthesis** (1–3 sentences) on what this theme showed this run.
2. That theme's **shortlist** as a table (or a compact list if only one or two papers).
3. Any **deep dives** for the theme, nested as `<h2>` under the `<h1>`.

Rules:

- Put each paper in **exactly one** theme. Derive themes from the papers; do not force papers into a fixed taxonomy.
- Fold thin themes (one stray paper) into a broader theme or a catch-all rather than creating a one-line section.
- If the pool does not cluster cleanly at all, fall back to a single `<h1>` "Highlights" section containing one shortlist table and the deep dives.

#### Per-theme shortlist table

Use a standard DocxXML `<table>` so each cell can hold links, bold text, and short lists cleanly. Give it a header row in `<thead>` and use `<colgroup>` to set column widths. One table per theme (the papers in that theme only).

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

For papers that are deep-dived in the same theme below: bold the row title and add `→ see deep dive` in the contribution cell.

---

### 3. Deep Dive Subsections (nested under each theme)

Each deep-dived paper is an `<h2>` inside its theme's `<h1>` section, separated by `<hr/>`. These are the substantive part of the brief — they must reflect the deeper analysis from `paper-scout-deep-dive`, not a polished restatement of the abstract. It is fine for a deep dive to be long; verbosity that carries technical content is welcome.

#### Header

Start with the paper title as the `<h2>`, and put author, year, and links on the immediately following paragraph:

```xml
<h2>Paper Title</h2>
<p>Author et al. (2025) · <a href="url">HF</a> · <a href="url">PDF</a></p>
```

#### What It Does

One or two sentences — the core claim, stated specifically, not the abstract. Prefer the concrete: "The paper shows that…" or "The authors introduce X, which…".

#### How It Works

The key mechanism, at enough detail that a reader could approximately reimplement it — the inputs/outputs, the major stages, the key equations or loss terms, and the non-obvious design choices *and why* they were made. Dense and specific; this is the part that proves you understood the paper rather than skimmed it. If the paper has a clear pipeline, describe it step by step with an ordered list (`<ol><li seq="auto">…</li></ol>`). If it has a notable formula or loss term, write it with inline LaTeX:

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

Where a diagram would make the architecture, pipeline, or data flow clearer than prose, add a Feishu whiteboard (see `lark-doc`/`lark-whiteboard` for the `<whiteboard>` block). Use it when it genuinely aids understanding, not as decoration.

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

#### Code & Artifacts

Expected for every deep dive (code inspection is mandatory when a repo exists — see `paper-scout-deep-dive`). Report what reading the implementation revealed: does the code match the paper, any undocumented tricks or default hyperparameters, paper-vs-code discrepancies, and the result of any lightweight check you ran. If no code was available, say so and note how that affects confidence.

#### Judgment

A grounded assessment, as long as it needs to be: how credible the results are, how novel the work is relative to prior art, what (if anything) the code inspection changed about your read, and what the user should do — read in full, build on, track follow-ups, or skip. Be specific; avoid hedging boilerplate.

---

### 4. Cross-Cutting Observations (Optional)

Theme grouping is already the structure of the brief, so this is **not** where themes are introduced. Add a short closing `<h1>` only when there is a pattern that cuts *across* themes worth naming — a convergence, a notable absence, a shift from prior runs, or a scope note about what was filtered out.

Write as an observation in prose, not a bullet list. One or two paragraphs. Skip it entirely if there is nothing cross-cutting to say.

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

All commands carry `--api-version v2`. Content is DocxXML, passed via `--content` (default `--doc-format xml`). For multi-line content, prefer `--content @path/to/file.xml` or `--content -` (stdin) over inline strings — write the brief to `drafts/` first and pass the file. See `lark-doc-create.md` / `lark-doc-update.md`.

### Creating The Document

```bash
lark-cli docs +create --api-version v2 \
  --parent-token <token> \
  --content @drafts/brief.xml
```

The `<title>` element inside the content sets the document title — there is no `--title` flag. The destination is set with `--parent-token <folder-or-wiki-node-token>` (or `--parent-position my_library` for a personal library). The destination type is recorded in the workspace instruction file. Note that v2 uses `--parent-token` for both folders and wiki nodes — there are no separate `--folder-token` / `--wiki-node` / `--wiki-space` flags.

Capture `data.document.document_id` and `data.document.url` from the response.

### Appending Sections For Long Briefs

`lark-doc` recommends creating a skeleton first (title + opening synthesis + the theme `<h1>` headings), then appending each theme's body and deep dives:

```bash
lark-cli docs +update --api-version v2 \
  --doc "<document_id>" \
  --command append \
  --content @drafts/section.xml
```

Use the `document_id` from the create response, not the `url`, especially for wiki destinations.

### Typical Sequence

1. Create the doc with the opening synthesis and the theme `<h1>` headings.
2. For each theme, append its mini-synthesis + shortlist table, then its deep-dive `<h2>` sections.
3. Append the cross-cutting observations section if present.
4. Record the document `url` from the create response for the coverage log.

---

## What Not To Do

- Do not omit the `<title>` element — there is no `--title` flag in v2, so the title must live in the content.
- Do not repeat the title as an `<h1>` after the `<title>` — it duplicates the heading.
- Do not put bare text directly inside a `<callout>`; wrap it in `<p>` or another block element. Do not put tables or code blocks inside a callout.
- Do not use a Markdown `>` blockquote expecting a callout. Use the `<callout>` tag.
- Do not use `overwrite` to fix a brief mid-run. Use `append`, or the block-level edit commands documented in `lark-doc-update.md`.
- Do not write "the paper achieves state-of-the-art." Write the number, the benchmark, and the comparison.
- Do not use a flat bullet list for the entire brief. Themes are `<h1>` sections, each shortlist is a `<table>`, deep dives are `<h2>` sections, and judgment is prose.
- Do not present one giant shortlist table for the whole pool. Split the shortlist by theme, one table per theme.
- Do not hardcode the theme list. Derive it from the pool each run; fall back to a single "Highlights" section if the pool does not cluster.
- Do not ship a deep dive that just restates the abstract and method. It must reflect the reimplementation-level mechanism, the evidence assessment, and what the code revealed.
- Do not skip the opening synthesis. The user should understand what mattered before scrolling to individual papers.
- Do not hand-author low-level XML escaping from memory. Follow `lark-doc-xml.md`.

---

## Checklist Before Delivery

- [ ] `lark-doc` loaded and its references followed
- [ ] Content authored as DocxXML; all commands use `--api-version v2`
- [ ] `<title>` element present and matches the configured title pattern; not duplicated in the body
- [ ] Opening synthesis written — 2–4 sentences on what mattered this period, naming the themes
- [ ] Body organized into per-theme `<h1>` sections (or a single "Highlights" fallback); each paper in exactly one theme
- [ ] Each theme has a mini-synthesis and its own shortlist `<table>` with specific contribution and relevance columns
- [ ] Each deep dive (nested `<h2>`): reimplementation-level mechanism, evidence assessment, code/artifact findings, and a grounded judgment
- [ ] Callouts used for standout papers and notable caveats — not decoratively
- [ ] `<latex>` used for any equations the paper centers on
- [ ] `<grid>` / `<whiteboard>` used where a comparison or diagram genuinely aids understanding
- [ ] `<`, `>`, `&` in text content escaped per `lark-doc-xml.md`
- [ ] Doc created via `docs +create --api-version v2`; long briefs appended via `docs +update --api-version v2 --command append`
- [ ] Document `url` recorded for the coverage log
