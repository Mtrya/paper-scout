---
name: paper-scout-feishu-doc
description: "Structure and write the Paper Scout brief as rich Feishu-rendered markdown. Covers document layout, visual hierarchy, callouts, tables, and section templates."
user-invocable: false
---

# paper-scout-feishu-doc

This skill governs how the Paper Scout brief is written and structured for Feishu delivery. It is used during Phase 5 (synthesis and writing) and Phase 6 (delivery).

Feishu renders markdown with rich visual output: headings create clear hierarchy, tables render as grids, blockquotes render as highlighted callouts, and horizontal rules create visible section breaks. Use these to make the brief scannable and visually organized — not a wall of bullets.

---

## Core Principles

**Vary the structure.** A brief that is entirely flat bullet lists is hard to read. Use headings, tables, callouts, and prose to match the structure to the content.

**Lead with the synthesis.** The user should understand what mattered this period before they read individual papers. Put the top-line picture first.

**Tables for comparison, bullets for lists.** If you are comparing papers, showing a shortlist, or summarizing multiple items with the same attributes, use a table. Use bullets for non-comparative lists of items.

**Callouts for things that demand attention.** If a paper is exceptional, if there is a notable pattern, or if there is a caveat the user must not miss, use a blockquote callout to make it stand out.

**Headings to navigate.** Feishu users can jump to sections. Use H2 for major sections and H3 for individual papers or subsections.

**Fewer words, more specificity.** Do not write "this paper presents an interesting approach to X." Write what the approach actually is. If you cannot say it specifically, you have not understood it well enough.

---

## Document Structure

### Title

Use the configured title pattern from the workspace instructions:

```
Paper Scout Daily Brief - YYYY-MM-DD
Paper Scout Weekly Brief - YYYY-MM-DD
```

### Opening Synthesis (Required)

A 2–4 sentence paragraph or short section that answers: *What mattered this period?*

This is not a table of contents. It is a judgment: what is the dominant theme, what was the standout paper, what should the user pay attention to even before reading further. Write it as a researcher, not a summarizer.

If the period was weak, say so honestly here.

### Shortlist Table

A table of all shortlisted papers — papers worth noticing but not deeply investigated. Columns:

| Paper | Key Contribution | Why It Matters | Links |
|---|---|---|---|
| Title (authors, year) | One sentence, specific | One sentence, user-relevant | [HF](url) / [PDF](url) |

Keep "Key Contribution" and "Why It Matters" distinct: contribution is what the paper did, relevance is why the user should care. Do not write the same thing twice.

If a paper appears in the shortlist but was also deep-dived, mark it clearly (e.g., bold the row, or add a note "→ deep dive below").

### Deep Dive Sections

One H2 section per deep-dived paper. Each section should contain:

**Header line:** Title, authors, year, links inline.

**What it does (1–2 sentences):** The core claim or contribution, stated specifically. Not the abstract.

**How it works:** The key mechanism or approach. Dense and specific — this is where the brief earns its value. Use an H3 subsection if the method is complex enough to deserve it. If the paper has a clear pipeline or algorithm, describe it step by step.

**Evidence:** What the experiments show, how convincing they are, any notable caveats or red flags. Be direct: if the baseline comparisons are weak, say so. If the gains are marginal, say so.

**Repo / artifacts (if inspected):** One sentence or a short bullet on what the code or project page reveals about maturity, usability, or gaps between the paper and the implementation.

**Judgment:** One or two sentences on why this paper was selected for deep investigation and what the user should do with it. Is it worth reading in full? Worth building on? Worth tracking?

Separate deep dive sections with a horizontal rule (`---`).

### Themes And Patterns (Optional)

If the pool this period shows a clear convergence, a notable absence, or a significant shift in direction, add a short H2 section after the deep dives.

Write this as an observation, not a list. "Three of this week's most-cited papers use X approach, suggesting Y is becoming the dominant paradigm" is more useful than "Notable themes: X, Y, Z."

---

## Formatting Reference

**Callout (blockquote)** — use for standout papers, critical caveats, or important patterns:

```markdown
> **Standout this week:** [Paper title] is the clearest example yet of [notable trend].
> The results are unusually well-validated and the code is clean. Worth reading.
```

**Table** — use for the shortlist and any multi-attribute comparisons:

```markdown
| Paper | Contribution | Relevance | Links |
|---|---|---|---|
| **Title** (Author et al., 2025) | Specific claim | Why it matters | [HF](url) |
```

**Horizontal rule** — use between deep dive sections:

```markdown
---
```

**Inline code** — use for method names, model names, dataset names, paper IDs:

```markdown
`GPT-4o`, `LoRA`, `2501.12345`
```

**Bold** — use for key claims, paper titles within prose, and things the user must not miss:

```markdown
The paper's key finding is that **attention heads are redundant beyond depth 12** in this family of models.
```

---

## What Not To Do

- Do not use a flat bullet list for the entire brief. If everything is a bullet, nothing stands out.
- Do not write an abstract-style summary for each paper. The brief is a judgment, not a digest.
- Do not pad with qualifications ("it is worth noting that," "interestingly," "the authors argue that"). Say the thing directly.
- Do not use the same structure for every section regardless of content. A paper with a clean pipeline deserves a step-by-step description; a position paper deserves a different treatment.
- Do not omit the opening synthesis. The user should not have to read the entire brief to understand what mattered.
- Do not skip the shortlist table if there are shortlisted papers. They deserve a concise record even if they did not earn a deep dive.

---

## Checklist Before Delivery

- [ ] Opening synthesis written — covers what mattered this period in 2–4 sentences
- [ ] Shortlist table populated with all shortlisted papers, specific contributions and relevance
- [ ] Each deep dive section: specific contribution, how it works, evidence quality, judgment
- [ ] No flat-bullet-only sections where a table or prose would be clearer
- [ ] Callout used for anything the user must not miss
- [ ] Horizontal rules separating deep dive sections
- [ ] Document title matches configured pattern
- [ ] Written to `output/` before delivery
