## Agents' Last Exam (ALE): The Benchmark Is Stronger Than the Framing

**Sun et al., UC Berkeley — arXiv:2606.05405**

ALE is the most ambitious agent benchmark yet: 1,490 task instances across 55 subfields, sourced from 250+ domain experts and anchored in the O*NET/SOC 2018 occupational taxonomy. It introduces rigorous quality control, deterministic artifact-based verification, and a rolling-evaluation design (150 public / 1,017 private tasks) to combat contamination.

The results are sobering. Codex with GPT-5.5 scores only 42.4% on the easiest tier and 8.6% on the hardest; most agents record near-zero on the "Last-Exam" tier. Yet the paper's loftiest claims warrant scrutiny.

**The five-hour cap is a hard truncation, not a generous limit.** Claude Code needs 72 wall-clock hours just for the 59-task easiest tier. Timed-out runs score 25% lower than those that finish naturally, suppressing scores for slow-but-thorough agents.

**The failure taxonomy inverts the paper's central hypothesis.** Understanding failures (31%) and Approach failures (47%) together account for ~78% of cases. Agents do not fail because they cannot click buttons or run shell commands; they fail because they lack domain knowledge and default to ad-hoc Bash scripts instead of the intended professional software. The benchmark was built around GUI+CLI orchestration as the hard problem, but the data reveal that domain knowledge—not tool use—is the binding constraint.

**Harness engineering matters far less than the backbone model.** Appendix D.4 shows that varying the model produces an 18.0 pp spread in pass rate, while varying the harness produces only 5–6 pp. ALE-Claw, a stripped-down Python port of OpenClaw with no memory or plugin framework, performs within 2 pp of full product harnesses. The bottleneck is the foundation model, not the scaffolding.

**The "economically valuable" framing is overstated.** ALE tasks are authentic expert workflows, but evaluated as isolated, single-shot deliverables with fixed requirements and no feedback loops. That is a real capability gap, but not the same as economically valuable work, which is embedded in organizational context and iterative refinement.

**Cross-benchmark positioning is clarifying.** ALE-CLI is substantially harder than Terminal-Bench: Codex scores 82% on Terminal-Bench but only 26% on ALE-CLI. The gap reflects domain breadth and end-to-end complexity, not merely terminal vs. GUI.

Bottom line: ALE sets a new standard for agent evaluation. Its most important finding is that frontier agents lack domain knowledge, not tool coordination. But the GDP-relevant framing is premature. ALE is an excellent midterm; the final exam is still to come.
