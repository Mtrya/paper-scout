---
name: paper-deep-dive
description: "Investigate a selected paper or paper-inspired research question by reading the paper, seeking external signals, preserving evidence, and extracting report-ready insight."
user-invocable: true
---

# paper-deep-dive

Do not summarize from the outside. Enter the paper as a researcher: understand the core idea, make it concrete, and learn something by tracing, running, reimplementing, deriving, comparing, or building a small probe. Critique is welcome when it is earned by investigation, but critique is not the goal.

For workspace layout, thread packets, and verifier rules, follow `workspace-manage`.

## Thread Packet

A thread is a durable unit of investigation under `runs/<run-id>/<thread-id>/`. It may be one paper, a comparison across papers, a method question, a code path, or a buildable idea inspired by the paper.

Use one of these shapes:

- `README.md` + `code/`
- `README.md` + `patches/`
- `README.md` + `code/` + `patches/`
- `BLOCKER.md`

`README.md` explains what was attempted, what evidence was preserved, how to rerun it when applicable, and what the result means for the report. `BLOCKER.md` is for the rare case where no meaningful external signal could be preserved; it should state the blocker plainly enough that the report can be honest about it.

## Acquire And Read

1. Acquire the paper through `paper-source` (`hf papers read` or arXiv PDF + MinerU) and save the Markdown to `papers/<area>/<slug>-<id>.md`.
2. Read the full paper, including appendix.
3. Identify the paper's problem, claimed contribution, core mechanism, key evidence, and the questions that matter for this run.

## Make The Method Concrete

Explain the method densely enough that you could roughly reimplement or modify it:

- inputs and outputs
- major stages and how they connect
- non-obvious design choices
- equations, losses, metrics, prompts, data transforms, or control loops that carry the method
- what the reported experiments actually demonstrate and what remains untested

Fill in skipped derivations when needed; note when you are supplementing rather than quoting.

## Seek External Signals

Insight comes from the paper plus external signals. Let the investigation iterate: ask a live question, take the strongest feasible constructive action, inspect the result, reinterpret the thread, and decide what question follows.

Prefer actions such as:

- trace official code end to end
- run a small path through the implementation
- inspect configs, preprocessing, data samples, checkpoints, or evaluation scripts
- profile or probe a component
- reimplement the core mechanism on a toy case
- derive a missing detail
- compare with a related implementation or paper
- run a small ablation, metric check, or benchmark subset

When official code or artifacts exist, use them as the strongest signal. Clone or work under `code/`, using explicit command working directories and direct environment binaries rather than relying on persistent `cd` or virtualenv activation.

If official artifacts are partial or unusable, do not stop at that observation. Reimplement the core idea at a humble scale, inspect related implementations, or read related papers that can test what is new. If none of those actions is meaningful within the run's resources, preserve `BLOCKER.md`.

Related papers count as external signals when they answer a real question: whether a claimed baseline comparison is fair, whether the mechanism is genuinely new, whether a prior method already solves the same problem, or whether a related implementation clarifies the missing code.

Use subagents when the work splits cleanly, for example one agent tracing code while another checks related papers or builds a small probe.

## Compare The Neighborhood

Situate the thread against 1-3 load-bearing neighbors when it sharpens the finding: key baselines, closest competing approaches, method predecessors, or related codebases. Keep this bounded. The goal is triangulation, not a literature review.

## Extract Report Material

Capture material that will make the report insight-dense and scannable:

- the research question and what the run learned
- the core mechanism in words, equations, pseudocode, or compact snippets
- paper figures, result tables, experiment figures, or produced artifacts that clarify the story
- limitations, blockers, and uncertainty that affect what the report can responsibly claim
- the research takeaway: what this thread teaches the run, what the report should claim because of it, and what remains unresolved

## Preserve Evidence

Before report delivery, promote durable work from `code/` into the thread packet:

- agent-written probes, reimplementations, wrappers, metric checks, synthetic experiments, or compact runnable examples go under `runs/<run-id>/<thread-id>/code/`
- patches against official code go under `runs/<run-id>/<thread-id>/patches/`
- report-facing media and small result artifacts go under `runs/<run-id>/assets/`
- raw cloned repos, venvs, large logs, caches, and scratch remain ignored and are cleaned after durable evidence is promoted

Run the workspace verifier as directed by `workspace-manage`; if it fails, fix the packet rather than weakening the evidence.

## What Not To Do

- Do not deliver paraphrase as investigation.
- Do not treat the abstract as ground truth.
- Do not skip code inspection when usable code exists.
- Do not stop at "no code"; try a reconstruction, related implementation, or related-paper triangulation.
- Do not perform cargo-cult reproduction. Spend resources when the expected insight justifies them.
- Do not let related-work exploration become a literature review.
