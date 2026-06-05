---
name: brief-compose
description: "Write the Paper Scout brief as Lark DocxXML: structure, editorial standards, callouts, tables, and visual hierarchy. Load lark-doc for syntax details."
user-invocable: false
---

# brief-compose

Load `lark-doc` first and follow its DocxXML references.

## Format Rules

- Author as DocxXML. v2 API. No `--title` flag — use a `<title>` element at the start and do not repeat it in the body.
- Use a sensible `<h1>` / `<h2>` hierarchy. Feishu auto-generates a table of contents.
- Callout children must be block elements (`<p>`, headings, lists). No bare text, no tables or code blocks inside callouts.
- Escape `<`, `>`, `&` as `&lt;`, `&gt;`, `&amp;` in text. Follow `lark-doc-xml.md` for full escaping rules.

## Structure

Organize by theme, not as a flat list.

```
<title>Paper Scout Daily Brief - YYYY-MM-DD</title>
Opening synthesis (2–4 sentences)
<h1>Theme A</h1>
  mini-synthesis (1–3 sentences)
  shortlist table (lightly-noticed papers only; omit if none)
  <h2>Deep-dive paper</h2> + narrative
  <hr/>
  <h2>Deep-dive paper</h2> + narrative
<h1>Theme B</h1>
  ...
Optional: <h1>Cross-cutting observations</h1>
```

### Opening synthesis

Answer: *What mattered this period?* Name the themes the rest of the doc will use.

If one paper was clearly exceptional:

```xml
<callout emoji="✅" background-color="light-green">
  <p><b>Standout:</b> [Title] is the clearest example of [trend]. [Why.]</p>
</callout>
```

If the period was weak, say so plainly. Do not pad.

### Theme sections

One `<h1>` per actual theme. Derive from the papers; do not hardcode themes.

Rules:

- Each paper appears **exactly once**.
- Put deep-dived papers as `<h2>` narrative sections inside their theme.
- Put lightly-noticed papers as rows in a per-theme shortlist table.
- Fold one-stray-paper themes into a broader theme or a catch-all.
- If the pool does not cluster, use a single `<h1>Highlights</h1>`.

#### Shortlist table

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
      <td><b>Title</b> (Author et al., 2025)</td>
      <td>What they did.</td>
      <td>Why this user should care.</td>
      <td><a href="url">HF</a> / <a href="url">PDF</a></td>
    </tr>
  </tbody>
</table>
```

Keep contribution and relevance distinct. Omit the table if every paper in the theme is deep-dived.

### Deep dive sections

Each is an `<h2>` inside its theme, separated by `<hr/>`.

Header format:

```xml
<h2>Paper Title</h2>
<p>Author et al. (2025) · <a href="url">HF</a> · <a href="url">PDF</a></p>
```

Write as fluent narrative, not a fixed template. Sub-headings (`<h3>`) only when a long dive genuinely needs sign-posts. Every deep dive must still carry:

- **Core claim** — what the paper actually shows, not the abstract's framing.
- **Mechanism** — enough detail that a reader could approximately reimplement it.
- **Evidence** — concrete numbers, magnitude, where it does not win, red flags. Write "outperforms X by Y% on Z", never "achieves state-of-the-art".
- **Investigation findings** — code/implementation insights or related-work comparison.
- **Grounded verdict** — read / skim / build on / track / skip, with reason.

Use platform features where they earn their place: `<latex>` for centered equations, `<grid>` for two-way comparisons, `<table>` for multi-way comparisons, callouts for standouts or caveats. Judgment over decoration.

### Cross-cutting observations (optional)

Add a closing `<h1>` only for a pattern that cuts across themes — a convergence, absence, or shift. One or two paragraphs of prose. Skip if there is nothing to say.

## Callouts

Use sparingly, only for things that demand attention.

**Standout:**
```xml
<callout emoji="✅" background-color="light-green"><p>[Content]</p></callout>
```

**Caveat / red flag:**
```xml
<callout emoji="⚠️" background-color="light-yellow"><p>[Content]</p></callout>
```

**Key insight:**
```xml
<callout emoji="💡" background-color="light-blue"><p>[Content]</p></callout>
```

## What Not To Do

- Do not omit or duplicate the `<title>`.
- Do not put bare text, tables, or code blocks inside a `<callout>`.
- Do not write "achieves state-of-the-art." Write the number, benchmark, and comparison.
- Do not use a flat bullet list for the whole brief.
- Do not stamp the same five sub-headers onto every deep dive.
- Do not list a deep-dived paper in a shortlist table or use "→ see deep dive" pointers.
- Do not present one giant shortlist table for the whole pool.
- Do not hardcode the theme list.
- Do not ship a deep dive that restates the abstract.
- Do not skip the opening synthesis.
- Do not guess XML escaping. Follow `lark-doc-xml.md`.

## Checklist

- [ ] `lark-doc` loaded
- [ ] DocxXML draft in `drafts/`
- [ ] `<title>` present, not duplicated
- [ ] Opening synthesis names themes
- [ ] Per-theme `<h1>`; each paper exactly once
- [ ] Shortlist tables per theme (lightly-noticed only)
- [ ] Deep dives as fluent narrative with mechanism, evidence, findings, verdict
- [ ] Callouts used only where they earn attention
- [ ] `<`, `>`, `&` escaped per `lark-doc-xml.md`
