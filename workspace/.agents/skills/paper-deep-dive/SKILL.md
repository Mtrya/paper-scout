---
name: paper-deep-dive
description: "Thoroughly investigate a selected paper: read it fully, inspect code and artifacts, choose resource-proportional research actions, compare related work, and write structured analysis notes."
user-invocable: true
---

# paper-deep-dive

Do not summarize. Understand the paper well enough to judge its claims. The code matters more than the prose.

For workspace layout and naming, follow `workspace-manage`.

## Output

Structured notes saved to `runs/<area>/<slug>-<id>-deep-dive.md`:

- what the paper is actually doing
- what is genuinely novel
- core method at near-reimplementation detail
- experimental evidence and credibility
- what the code reveals — matches, undocumented tricks, discrepancies, or explicit note that no code exists
- artifact completeness and research action — classify the release and record the live question, action, resource fit, result, and interpretation or blocker
- comparative positioning against 1–3 key related papers (mandatory when no code)
- illustration candidates for the report — equations, figures, tables, snippets, diagnostics, or experiment artifacts that make the finding easier to understand
- red flags and caveats
- bottom-line judgment: read / skim / build on / track / skip, with reason

Notes do not need polished prose. They need to be accurate and specific.

## Phase D0: Download And Inventory

1. Acquire the paper through `paper-source` (`hf papers read` or arXiv PDF + MinerU) and save the Markdown to `papers/<area>/<slug>-<id>.md`.
2. Read the full paper. Do not skip sections or the appendix.
3. Build a section inventory. Account for every section. Assign each one a home in your analysis or note it under general observations.

Example:

```
§ Abstract         → contribution summary
§ 1 Introduction   → motivation, framing
§ 2 Related Work   → positioning
§ 3 Method         → core mechanism
§ 4 Experiments    → evidence
  § 4.1 Setup      → datasets, baselines, metrics
  § 4.2 Results    → main results
  § 4.3 Ablations  → component validation
§ 5 Conclusion     → limitations, future work
§ Appendix         → integrate where relevant
```

## Phase D1: Motivation And Contribution

- State the problem, gap, and failure mode in the paper's own terms.
- List claimed contributions and tag each: method / insight / benchmark / analysis.
- Capture the core insight in 1–2 sentences: what did the authors observe that made this work possible?

## Phase D2: Method

Explain the method densely enough that someone could roughly reimplement it.

- Inputs and outputs.
- Major stages and how they connect.
- Non-obvious design choices and why the authors chose them.
- Frameworks, libraries, and implementation signals that affect reproducibility.
- Fill in skipped derivations when needed; note when you are supplementing rather than quoting.

## Phase D3: Experiments And Evidence

- **Setup:** datasets, baselines, metrics. Fair? Appropriate?
- **Main results:** what the key tables and figures actually show, magnitude, practical meaning, conditions where the method loses.
- **Ablations:** what components are tested, what removing each reveals, whether the study is convincing.
- **Red flags:** cherry-picked comparisons, missing baselines, gains within noise, evaluation favoring the method, abstract claims overshooting results.

## Phase D4: Code And Artifact Investigation

Mandatory whenever a repository exists. Do not trust the paper's description of its own method.

**GitHub repos:**

1. Clone into `repos/<area>/<repo-name>/`. Use `--depth 1` if large. Fall back to web browsing only if cloning fails.
2. Read the README.
3. Trace the core implementation end to end. Locate the key equation, loss, or algorithm in the source.
4. Look for what the paper omits: undocumented tricks, defaults, clamps, normalizations, data filtering, baseline implementations.
5. Note paper-vs-code discrepancies and repo maturity (last commit, activity).
6. Convert live questions into a research action. Choose the strongest feasible action: code-path trace, config/default audit, dataset/sample inspection, diagnostic script, partial reproduction, ablation, or full experiment when justified.

**Hugging Face model / dataset cards:**

- Check downloads / community activity.
- Verify the card matches the paper's claims.
- Inspect a small dataset sample if it sharpens the analysis.
- Note limitations the paper does not mention.

If no code exists, state that explicitly and explain how it affects confidence.

Classify artifact completeness: reproducible artifact / architecture release / partial artifact / no usable artifact. Do not use an incomplete artifact as a reason to stop at critique. When meaningful, create a project under `repos/<area>/<slug>-check/`, set up a local Python venv, write scripts, install dependencies proportionate to the action, and execute the diagnostic or experiment. Good actions might be reimplementing an attention mask, token-packing rule, loss, optimizer step, metric, config/model-shape assertion, small synthetic-input path, ablation of a code branch, or benchmark subset. If no meaningful action fits the available resources, say why.

Use this loop in the notes:

```text
Question -> Action -> Result -> Interpretation -> New Question -> ...
```

## Phase D-RW: Situate Against Related Work

Mandatory when no usable code exists; encouraged otherwise.

1. Identify 1–3 load-bearing related papers: key baselines, closest competing approach, or method directly improved on. Use the paper's citations and `papers/<area>/` first.
2. Pull missing ones into `papers/<area>/`. Read at scan depth — abstract, method, headline results.
3. Record comparative reasoning: what is genuinely new, whether comparisons are fair, what the paper omits about its lineage.

If a related paper itself raises a load-bearing question, do a bounded dive on it — but keep the scope tight. 1–3 papers, scan depth.

## Phase D5: Assessment

Write a bottom-line covering novelty, credibility, relevance, and priority. End with a specific call: read / skim / build on / track / skip.

## What Not To Do

- Do not pad with obvious restatements. If the abstract says "we propose X," explain what X is.
- Do not treat the abstract as ground truth. Check whether the paper delivers.
- Do not stop at the method section. Experiments and ablations test claims.
- Do not skip the appendix.
- Do not skip code inspection when a repo exists.
- Do not perform cargo-cult reproduction. Spend resources when the expected insight justifies them.
- Do not let related-work exploration become a literature review.

## Checklist

- [ ] Section inventory complete
- [ ] Motivation and contribution in the paper's own terms
- [ ] Core method at reimplementation detail
- [ ] Evidence assessed, not transcribed
- [ ] Artifact completeness classified; code inspected or absence explicitly noted; discrepancies recorded
- [ ] Situated against 1–3 related papers when no code exists
- [ ] Research action loop recorded: question, action, resource fit, result, interpretation or blocker
- [ ] Illustration candidates recorded for report composition
- [ ] Red flags noted
- [ ] Bottom-line judgment with specific priority call
- [ ] Notes saved to `runs/<area>/<slug>-<id>-deep-dive.md`
