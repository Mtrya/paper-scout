---
name: paper-scout-deep-dive
description: "Given a paper already selected for deep investigation, read it thoroughly, inspect linked artifacts, and produce structured analysis notes that feed into the Paper Scout brief."
user-invocable: false
---

# paper-scout-deep-dive

This skill governs the deep investigation phase for a single paper. It is invoked from the `paper-scout` skill during Phase 4, once a paper has been selected for deep investigation.

The goal is not to summarize the paper. The goal is to understand it well enough to write a genuinely useful analysis: what it actually claims, whether those claims hold up, what it reveals about the state of the field, and whether it changes what the user should read, build, or think about.

Loading the paper into context is only the **first** step. A deep dive that restates the abstract and method at a high level is a failure, even if well written. Genuine understanding here means reconstructing the method at near-reimplementation detail, interrogating the evidence rather than transcribing it, and — whenever code exists — reading the actual implementation. **The code matters more than the prose: it is the ground truth for what the paper actually does.**

---

## Input

A paper already identified for deep investigation, with:

- paper id (HuggingFace paper id or arxiv id)
- title and abstract already read during fast scan
- any linked project pages, repos, model cards, or dataset cards noted during selection

---

## Output

Structured analysis notes saved to `runs/<paper-id>-deep-dive.md`, covering:

- what the paper is actually doing
- what is genuinely novel
- the core method and how it works, at enough detail that a reader could approximately reimplement it
- the experimental evidence and how credible it is
- **what the code reveals** — whether the implementation matches the paper, undocumented tricks or defaults, and any paper-vs-code discrepancies (or an explicit note that no code was found)
- red flags, caveats, and open questions
- a bottom-line judgment: why this paper was worth the deep dive

These notes feed directly into Phase 5 synthesis. They do not need to be polished prose — they need to be accurate and specific.

---

## Phase D0: Download And Inventory

Before analyzing:

1. Download the paper markdown: `hf papers read <id> > papers/<id>.md`
2. Read the full paper. Do not skip sections.
3. Before starting the analysis, build a **section inventory**: list every section and subsection, note its core content in one sentence, and record where it will appear in your analysis notes. This prevents important content from being silently dropped.

A section inventory looks like this:

```
§ Abstract         → analysis: contribution summary
§ 1 Introduction   → analysis: motivation, problem framing
§ 2 Related Work   → analysis: positioning, prior art
§ 3 Method         → analysis: core mechanism
  § 3.1 ...        → analysis: [specific subsection]
§ 4 Experiments    → analysis: experimental evidence
  § 4.1 Setup      → analysis: datasets, baselines, metrics
  § 4.2 Results    → analysis: main results
  § 4.3 Ablations  → analysis: component validation
§ 5 Conclusion     → analysis: limitations, future work
§ Appendix (if any)→ analysis: integrate into relevant sections
```

After building the inventory, check: is every section accounted for? If a section has no clear home in your analysis, find one or note that it is covered in a general observations section. Do not leave sections unaddressed.

---

## Phase D1: Motivation And Contribution

Answer these questions based strictly on what the paper says:

**What problem does this paper solve?**
- What is the specific gap or failure mode being addressed?
- What would go wrong without this paper's contribution?
- If the paper gives a concrete example or failure case, use it.

**What are the claimed contributions?**
- List the contributions as the paper states them, not your paraphrase.
- For each contribution, note whether it is a new method, a new insight, a new benchmark, a new analysis, or a combination.

**What is the core insight or idea?**
- In one or two sentences: what did the authors observe or realize that made this work possible?
- What is the key design decision that distinguishes this from prior work?

---

## Phase D2: Method

Work through the method in enough detail that someone who has not read the paper could understand what it does and roughly how to implement it.

**Overall pipeline:**
- What are the inputs and outputs?
- What are the major components or stages?
- How do the components connect?

**Core mechanism:**
- For any non-obvious design decisions, explain not just what they do but why the authors chose them.
- If the paper skips derivation steps, fill in the reasoning. Note when you are supplementing rather than quoting.
- If the paper includes an algorithm or pseudocode, work through it step by step.

**Implementation signals:**
- What frameworks, libraries, or architectures does the implementation depend on?
- Are there non-standard operations or tricks that would affect reproducibility?
- Does the appendix contain additional implementation details that matter?

Write the method section densely and specifically. Do not replace technical content with vague summaries.

---

## Phase D3: Experiments And Evidence

**Setup:**
- What datasets are used? Are they standard benchmarks or novel?
- What baselines are compared against? Are the comparisons fair?
- What metrics are used? Are they appropriate for the claimed contribution?

**Main results:**
- What do the key tables and figures actually show?
- What is the magnitude of the improvement? Is it practically meaningful?
- Are there conditions under which the method does not win?

**Ablations:**
- What components does the ablation study test?
- What does removing each component reveal about what is actually doing the work?
- Are the ablations thorough enough to be convincing?

**Red flags:**
- Are there cherry-picked comparisons or missing baselines?
- Are the gains within noise, or are they meaningful?
- Does the evaluation setup favor the proposed method in ways that may not generalize?
- Are claims in the abstract supported by the experimental results, or do they overshoot?

---

## Phase D4: Code And Artifact Inspection (Mandatory When Code Exists)

These papers were selected *because* they are promising, so artifact inspection is not optional triage — it is a required part of every deep dive.

**Code inspection is mandatory whenever a repository is present.** Do not rely on the paper's prose description of its own method; read what the implementation actually does. If you cannot find any code, state that explicitly in the notes and explain how it affects your confidence.

**For GitHub repos — always clone and read the implementation:**

1. Clone the repo (`git clone`) into `repos/<paper-id>/` and inspect it locally. Repo size is rarely a reason to skip — clone shallowly (`--depth 1`) if it is large, and only fall back to browsing on the web if cloning genuinely fails.
2. Read the README for setup instructions and claimed functionality.
3. Open the core implementation file(s) and trace the method end to end: does the code actually compute what the paper describes? Locate the key equation, loss term, or algorithm in the source.
4. Look for what the paper does **not** say: undocumented tricks, default hyperparameters, clamps/normalizations, data filtering, baseline implementations, and anything that would change how you read the results.
5. Note every significant discrepancy between the paper and the code, and check the last commit date / activity as a maturity signal.
6. Where it is cheap and decisive, run a **lightweight** check — a small script, a CPU-only sanity run, or a tiny snippet — to confirm or challenge a specific claim. Stay within the lightweight-execution limits (no training, no full benchmarks, no heavy downloads). Report exactly what you ran and what it showed.

**For model and dataset cards on Hugging Face:**

- Check download numbers and community activity as a proxy for real-world uptake.
- Check whether the card accurately describes what the paper claims.
- Inspect a small sample of any released dataset where it would sharpen the analysis.
- Note any limitations or warnings that the paper does not mention.

**What to look for:**
- Does the implementation match the described method, exactly or only loosely?
- Is the code usable by someone other than the authors?
- Are there signs of a rushed or incomplete release?
- Does the repo reveal practical limitations or hidden assumptions not mentioned in the paper?

---

## Phase D5: Assessment

Write a bottom-line assessment covering:

**Novelty:** Is the contribution genuinely new, or is it an incremental improvement on a well-known idea? Is the novelty claimed in the paper proportionate to what the work actually delivers?

**Credibility:** Do the experiments support the claims? Are there methodological concerns that should temper confidence in the results?

**Relevance to the user:** Given the user's interests and exclusions, does this paper matter? Is it actionable, informative, or field-shaping in ways the user should know about?

**Priority judgment:** Should the user read this paper? Should they track the follow-up work? Should they try to use or build on this? Be specific about why.

---

## What Not To Do

- Do not pad analysis with summary of obvious content. If the abstract says "we propose X," do not write "the authors propose X" in your analysis — explain what X is.
- Do not treat the abstract as a reliable guide to what the paper actually does. Check whether the paper delivers what the abstract promises.
- Do not stop at the method section. Experiments and ablations are where claims get tested.
- Do not skip the appendix. Important implementation details, full hyperparameter tables, and additional experiments often live there.
- Do not skip code inspection when a repo exists, and do not substitute the paper's description of the code for actually reading it. If the code confirms the paper and adds nothing new, say so briefly — but only after you have looked.
- Do not let code inspection turn into a reproduction project. Read the implementation and run only the smallest checks that change your assessment; stay within the lightweight-execution limits.

---

## Checklist Before Finishing

- [ ] Section inventory completed — no sections unaccounted for
- [ ] Motivation and contribution clearly stated in the paper's own terms
- [ ] Core method explained at near-reimplementation detail, not vaguely summarized
- [ ] Experimental evidence assessed, not just transcribed
- [ ] Code cloned and inspected when a repo exists (or absence of code stated explicitly); paper-vs-code discrepancies noted
- [ ] Any lightweight verification run is reported with what it showed
- [ ] Red flags noted where present; absence of red flags noted where absent
- [ ] Bottom-line assessment written with a specific priority judgment
- [ ] Notes saved to `runs/<paper-id>-deep-dive.md`
