---
name: paper-scout-feishu-doc
description: "Structure and write the Paper Scout brief as a Feishu document using lark-cli. Covers the full Lark-flavored Markdown format, document sections, callouts, grids, lark-table, LaTeX, and the delivery command sequence."
user-invocable: false
---

# paper-scout-feishu-doc

This skill governs how the Paper Scout brief is written and delivered as a Feishu document. It is used during Phase 5 (synthesis and writing) and Phase 6 (delivery) of the main `paper-scout` skill.

The document is created using `lark-cli docs +create` with Lark-flavored Markdown. For long briefs, create the document first and append sections with `lark-cli docs +update --mode append`.

---

## Before You Start

Load `lark-doc` before any delivery work.

Feishu renders Lark-flavored Markdown with real visual structure — callouts, grids, tables with rich cell content, inline LaTeX. Use this. A brief that is nothing but flat bullet lists misses the platform entirely.

**Critical rules:**
- Do not repeat the document title inside the markdown. The `--title` flag handles it; the markdown starts directly with body content.
- Feishu generates a table of contents automatically from your headings. Use a sensible H2/H3 hierarchy.
- Callouts cannot contain code blocks or tables. Callouts contain text, headings, lists, and quotes.
- `<lark-table>` cells must each have a blank line before and after their content.

---

## Document Structure

### 1. Opening Synthesis (Required)

Write 2–4 sentences answering: *What mattered this period?*

This is a judgment, not a summary. What was the dominant theme? What was the standout paper? What should the user notice before reading further?

If a paper was clearly exceptional this period, wrap the synthesis in a standout callout:

```html
<callout emoji="✅" background-color="light-green">

**Standout this week:** [Title] is the clearest example of [trend/contribution]. [One sentence on why.]

</callout>
```

If the period was weak, say so in plain prose. Do not pad.

---

### 2. Shortlist Table

All shortlisted papers in one `<lark-table>`. Use `<lark-table>` rather than a Markdown table so each cell can hold links, bold text, and short lists cleanly.

Four columns: Paper, Key Contribution, Why It Matters, Links.

```html
<lark-table column-widths="200,230,230,70" header-row="true">
<lark-tr>
<lark-td>

**Paper**

</lark-td>
<lark-td>

**Key Contribution**

</lark-td>
<lark-td>

**Why It Matters**

</lark-td>
<lark-td>

**Links**

</lark-td>
</lark-tr>
<lark-tr>
<lark-td>

**Title** (Author et al., 2025)

</lark-td>
<lark-td>

One sentence — the specific thing the paper did.

</lark-td>
<lark-td>

One sentence — why the user should care about this.

</lark-td>
<lark-td>

[HF](url) / [PDF](url)

</lark-td>
</lark-tr>
</lark-table>
```

Keep "Key Contribution" and "Why It Matters" distinct — contribution is what they did, relevance is why it matters to this user. Do not write the same sentence twice.

Papers that also appear in a deep dive section below: bold their row title and add `→ see deep dive` in the contribution cell.

---

### 3. Deep Dive Sections

One `## [Title]` section per deep-dived paper, separated by `---`.

#### Header

Start with the paper title as the H2, and put author, year, and links on the immediately following line:

```markdown
## Paper Title

Author et al. (2025) · [HF](url) · [PDF](url)
```

#### What It Does

One or two sentences — the core claim, stated specifically, not the abstract. Start directly: "This paper proposes..." is a valid opener but prefer even more concrete: "The paper shows that..." or "The authors introduce X, which..."

#### How It Works

The key mechanism. Dense and specific. If the paper has a clear pipeline, describe it step by step with a numbered list. If it has a notable formula or loss term, write it with inline LaTeX:

```markdown
The training objective is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KL}}$
```

For block equations:

```markdown
$$
\hat{y} = \arg\max_{y} P(y \mid x; \theta)
$$
```

If the method contrasts two design choices directly, use a two-column grid:

```html
<grid cols="2">
<column>

**Prior approach**

Description of what prior work did.

</column>
<column>

**This paper**

Description of what this paper does differently.

</column>
</grid>
```

#### Evidence

What the key results show and how convincing they are. Be direct:

- "Outperforms X by Y% on Z benchmark" is useful.
- "Achieves state-of-the-art" is not useful.

If there are notable red flags or caveats, use a warning callout:

```html
<callout emoji="⚠️" background-color="light-yellow">

Baseline comparisons exclude [Method X], which is the strongest prior approach in this setting.

</callout>
```

#### Artifacts (If Inspected)

One sentence or a short bullet: what the repo or project page reveals about maturity, usability, or gaps between the paper and the implementation.

#### Judgment

One or two sentences: why this paper was worth the deep dive, and what the user should do with it. Is it worth reading in full? Worth building on? Worth tracking follow-ups?

---

### 4. Themes And Patterns (Optional)

A short H2 section after all deep dives, only when the pool shows a clear convergence, absence, or shift worth naming.

Write as an observation in prose, not a bullet list. One or two paragraphs.

---

## Callout Reference

Use callouts for things that demand attention, not as decoration. Three types are useful for paper briefs:

**Standout paper or exceptional finding:**
```html
<callout emoji="✅" background-color="light-green">

[Content]

</callout>
```

**Caveat, red flag, or important limitation:**
```html
<callout emoji="⚠️" background-color="light-yellow">

[Content]

</callout>
```

**Notable insight or key insight:**
```html
<callout emoji="💡" background-color="light-blue">

[Content]

</callout>
```

Callout content supports text, headings, lists, and quotes. It does not support code blocks or tables.

---

## Delivery Commands

### Creating The Document

```bash
lark-cli docs +create \
  --title "Paper Scout Daily Brief - YYYY-MM-DD" \
  --folder-token <token> \
  --markdown "<opening synthesis and first section>"
```

Replace `--folder-token` with `--wiki-node <token>` or `--wiki-space <id>` depending on what the user's destination is. The destination type is recorded in the workspace instruction file.

### Appending Sections For Long Briefs

For briefs longer than a few sections, create the document with the opening content then append sections one at a time:

```bash
lark-cli docs +update \
  --doc "<doc_id>" \
  --mode append \
  --markdown "<next section>"
```

Use the `doc_id` from the create response, not the `doc_url`, especially when the destination is a wiki node (the URL may be `/wiki/...` form, which behaves differently).

### Typical Sequence

1. Create the doc with the opening synthesis and shortlist table.
2. Append each deep-dive section.
3. Append the themes section if present.
4. Record the `doc_url` from the create response for the coverage log.

---

## What Not To Do

- Do not start the markdown with a `# Title` heading — it duplicates the `--title` parameter.
- Do not use `>` blockquote syntax expecting it to render as a callout. Use the `<callout>` tag.
- Do not put tables or code blocks inside a callout — it is not supported.
- Do not use `overwrite` mode to fix or update a brief. Use `append` or `replace_range`.
- Do not write "the paper achieves state-of-the-art." Write the number, the benchmark, and the comparison.
- Do not use a flat bullet list for the entire brief. The shortlist is a `<lark-table>`. Deep dives are H2 sections. Judgment paragraphs are prose.
- Do not skip the opening synthesis. The user should understand what mattered before scrolling to individual papers.

---

## Checklist Before Delivery

- [ ] `lark-doc` loaded
- [ ] Document title matches configured pattern (`--title` flag, not inside markdown)
- [ ] Opening synthesis written — 2–4 sentences on what mattered this period
- [ ] Shortlist in a `<lark-table>` with specific contribution and relevance columns
- [ ] Each deep dive: specific contribution, how it works, evidence quality, judgment
- [ ] Callouts used for standout papers and notable caveats — not decoratively
- [ ] LaTeX used for any equations the paper centers on
- [ ] Grid used for any direct method comparisons worth visualizing
- [ ] Doc created via `docs +create`; sections appended via `docs +update --mode append` if long
- [ ] `doc_url` or `doc_id` recorded for the coverage log
