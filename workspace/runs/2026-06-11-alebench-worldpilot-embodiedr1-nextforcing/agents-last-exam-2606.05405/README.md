# Agents' Last Exam (ALE) — Deep-Dive Analysis

**Paper:** Agents' Last Exam (arXiv:2606.05405)  
**Authors:** Yiyou Sun, Xinyang Han, Weichen Zhang, et al. (UC Berkeley)  
**Date:** 2026-06-11  
**Investigator:** Paper Scout

---

## Top-Line Thesis

ALE is the most ambitious agent benchmark to date—1,490 task instances across 55 subfields, sourced from 250+ industry experts, with a rigorous verification pipeline and a living (rolling) evaluation design. It succeeds as a *measurement instrument* where prior work failed: broad coverage, authentic workflows, deterministic scoring, and genuine headroom (best agent: 26.2% overall pass rate). 

But the paper's framing—"economically valuable," "GDP-relevant impact," "last exam"—oversells what the benchmark actually measures. ALE evaluates whether a solitary agent, given a static task description and up to five hours, can produce a single deliverable that matches a pre-specified reference. That is a real capability, but it is not the same as economically valuable work, which is embedded in organizational context, client feedback loops, iterative refinement, and maintenance. The benchmark's own failure taxonomy reveals that the dominant bottleneck is *domain knowledge*, not tool use or agent architecture—yet the benchmark's design (GUI+CLI Generalist CUA) was built around the assumption that execution surface would be the harder part. And the five-hour cap, far from being generous, is a hard truncation that hides just how long agents would actually take: Claude Code needs 72 wall-clock hours just for the easiest tier.

The deepest insight from ALE is not that agents fail at long-horizon computer use. It is that *simplified agent harnesses perform nearly as well as full product harnesses*, and *model choice explains roughly 3× more variance than harness choice*. The bottleneck is the foundation model's reasoning and domain knowledge, not the scaffolding around it. That is a genuinely important finding—but it cuts against the benchmark's implicit narrative that we need richer agent architectures to close the gap.

---

## 1. The Five-Hour Cap: A Hard Truncation, Not a Generous Limit

### What the paper says

Each run is capped at five hours wall-clock. The overall timeout rate is 4.3%, ranging from ~1% for lightweight harnesses to 7% for OpenClaw (Appendix D.2). Runs that time out have a mean score of 20.8, compared with 27.7 for runs that ended earlier.

### What the numbers actually reveal

The cap looks generous until you look at the *total wall-clock time for a tier*:

| Configuration | Near-Term wall-clock | Full-Spectrum wall-clock | Last-Exam wall-clock |
|---|---|---|---|
| Codex + GPT-5.5 | 30h | 23h | 29h |
| Claude Code + Sonnet 4.6 | **72h** | 38h | **72h** |
| Claude Code + Opus 4.7 | 17h | 15h | 19h |
| ALE-Claw + Opus 4.7 | 21h | 20h | **49h** |

These are totals across 59, 55, and 35 tasks respectively. For Near-Term alone, Claude Code + Sonnet 4.6 needs 72 hours—an average of **73 minutes per task**. But many individual tasks take far longer. The cap truncates the tail of the distribution.

The timeout analysis (Appendix D.2) says timed-out runs score 20.8 vs 27.7 for non-timed-out runs—a 25% score penalty. This understates the problem. If an agent times out at 5 hours, it has not merely "scored lower"; it has been *forcibly stopped* before completing a workflow that the benchmark authors themselves describe as taking experts "days or weeks" (Section 2.3). The 4.3% timeout rate is a lower bound on how many tasks genuinely exceed five hours, because some agents may terminate early (giving up) rather than run to the cap.

The cap is pragmatic—evaluation must finish—but it has a systematic effect: it suppresses scores for slow-but-steady agents (like Claude Code + Sonnet 4.6, which takes 72h for Near-Term) relative to fast agents (like Droid + Opus 4.6, which takes 23h total but scores lower). The benchmark partially rewards speed over thoroughness. This is not a flaw if speed is part of economic value, but the paper does not defend the five-hour threshold as economically grounded.

**Non-obvious implication:** If agents were given 10 or 24 hours, the pass rates would likely rise non-trivially for the slowest configurations, but probably not for the hardest tier—where even frontier models score 0–8.6%. The cap matters most at the margin, not for the core finding that the hardest tier is unsaturated.

---

## 2. Failure Taxonomy: Domain Knowledge, Not Execution, Is the Bottleneck

### The headline numbers

For Claude Code + Opus 4.7 failures (Figure 9d / Appendix D.3):

- **Understanding failures: 31%** (Domain Knowledge Gap 25%, Hallucination/Fabrication 6%)
- **Approach failures: 47%** (Wrong Strategy 30%, Incomplete/Abandoned 17%)
- **Execution failures: 22%** (Implementation Bug 8%, Output Format Error 10%, GUI/Browser Failure 4%)
- **Infrastructure/Timeout: excluded from this breakdown**

Understanding + Approach = ~78% of classified failures.

### What this actually means

The authors write: "Lacking specialized knowledge, agents default to ad-hoc scripts instead of the intended domain software, reinforcing the GUI-underutilization pattern above" (Section 4.2). This is the single most important sentence in the paper.

ALE was designed around the concept of the *Generalist Computer-Use Agent* (GCUA)—an agent that unifies CLI and GUI capabilities across five functional layers (Brain, Eyes, Body, Hands, Feet). The implicit hypothesis was that the hard part is *orchestration*: interleaving shell commands, GUI interactions, file manipulation, and web research within a single workflow. The failure taxonomy refutes this. Agents do not fail because they cannot click the right button or execute the right shell command. They fail because they do not know *which* software to use, *how* to configure it, or *what* the correct domain-specific approach looks like.

Consider the tool-usage data: 34% of public task instances designate graphical software as the primary tool, yet GUI usage stays small across most configurations (Section 4.2). Agents are not struggling with GUI actions; they are *avoiding* GUI tools because they do not understand the domain workflow well enough to know why SolidWorks is the right choice for a task, or how to navigate its feature tree once opened. They fall back to Bash/CLI substitutes because that is what they know.

This is a subtle but important inversion. The benchmark's architectural contribution—GUI-as-Tool, CUA MCP bridge, sub-agent dispatch—solves a real problem (how to evaluate GUI+CLI agents uniformly), but it is not the binding constraint on performance. The constraint is the model's internal knowledge of 55 professional domains.

**Non-obvious implication:** If you fixed every harness bug and gave agents perfect GUI coordination, you would likely still face the majority of the same failures, since ~78% are classified as Understanding or Approach rather than Execution. The marginal return to better agent architecture is lower than the marginal return to better domain knowledge in the foundation model—or to retrieval-augmented generation that can pull in domain-specific documentation and examples.

---

## 3. "Economically Valuable": A Claim That Needs Honest Scrutiny

### What the authors claim

ALE tasks are "economically valuable, real world tasks" (Abstract), "sourced from real professional practice" (Section 2.1), anchored in O*NET/SOC 2018, and intended to close "the gap between benchmark success and GDP-relevant impact" (Section 1). The framing is that if agents can pass ALE, they can perform work that currently employs human professionals.

### What is actually true

The tasks are authentic in a narrow but important sense: they were contributed by domain experts who have actually performed them, using real software, producing real deliverables. The quality-control pipeline (first-pass review, engineer dry-runs, expert committee peer review) is the most rigorous I have seen in any agent benchmark. ALE is not synthetic, not crowdsourced from lay workers, and not padded with easy tasks.

But economic value is not just about producing a single correct deliverable in isolation. Real professional work has properties that ALE deliberately strips away:

1. **Requirements negotiation.** ALE tasks come with a fixed description, fixed inputs, and a fixed evaluation rubric. Real clients do not.
2. **Iterative refinement.** Experts revise deliverables based on feedback. ALE scores a single shot.
3. **Organizational context.** Real workflows involve handoffs, consistency checks, version control, and coordination across roles. ALE tasks are atomic.
4. **Maintenance and adaptation.** Economic value often comes from ongoing relationships, not one-off deliverables.
5. **Error recovery.** ALE tasks that time out or fail are simply scored low. Real professionals recover from errors, clarify ambiguities, and renegotiate scope.

The O*NET anchor gives the benchmark *face validity*—these tasks map to occupations that exist in the federal taxonomy. But face validity is not economic validation. The authors do not show that the tasks they selected are representative of the *time allocation* or *economic output* of those occupations. ALE might over-index on tasks that are easy to verify (CAD toolpaths, financial workbooks) and under-index on tasks whose value comes from judgment, persuasion, or contextual adaptation.

**Honest assessment:** ALE measures *technical deliverable production* across a broad professional taxonomy. That is a genuine and important capability gap. But calling it "economically valuable" or "GDP-relevant" overreaches. A more accurate framing would be: "If agents can pass ALE, they can produce single-shot technical artifacts that resemble professional deliverables, in controlled conditions, with pre-specified requirements." That is still a high bar—and still far from saturated—but it is not the same as economic transformation.

---

## 4. Cost, Time, and Token Data: The Scalability Problem

### The numbers

From Table 1 and Appendix D.5:

| Configuration | Total API Cost (all tiers) | Total Wall-Clock | Total Tokens |
|---|---|---|---|
| Codex + GPT-5.5 | ~$560 | ~82h | ~581M |
| ALE-Claw + GPT-5.5 | ~$307 | ~48h | ~331M |
| Claude Code + Sonnet 4.6 | ~$378 | ~181h | ~822M |
| ALE-Claw + Opus 4.7 | ~$1,141 | ~90h | ~1,350M |

Per-task averages: $3–10 per task, tens of minutes to hours each (Section 4.1).

### What this reveals

First, evaluation is expensive. The 150-task public set costs hundreds of dollars per configuration. The full 1,490-task pool would cost thousands. This is not a criticism—real workflows should be expensive to evaluate—but it is a practical constraint on how often the community can run full evaluations.

Second, **cost and performance are only loosely correlated** (Appendix D.5). ALE-Claw + GPT-5.5 achieves the highest overall mean score (48.0%) at $307, while the same harness with Opus 4.7 spends 3.7× more ($1,141) for 6.0 pp lower score. Cursor + GPT-5.5 reaches 41.7% for only $177. Token volume does not predict performance either: ALE-Claw + Opus 4.7 burns 1,350M tokens while Cursor + Opus 4.7 uses 446M tokens and scores slightly higher.

Third, the data expose an **efficiency paradox**. Claude Code + Sonnet 4.6 takes 181 hours total wall-clock (72h for Near-Term alone) but scores only 35.3% overall. Droid + Opus 4.6 takes 23 hours but scores 27.3%. There is no clear Pareto frontier: agents are not trading off speed and accuracy in a predictable way. Some are just slow *and* bad.

**Non-obvious implication:** The cost data make ALE a poor fit for rapid iterative development. If you are tuning a model or harness, you cannot afford to run the full benchmark frequently. The three-tier design (Near-Term / Full-Spectrum / Last-Exam) is a pragmatic response, but it means that most community evaluations will target only the 59-task Near-Term tier. The Full-Spectrum and Last-Exam tiers, which are the most distinctive contributions of the benchmark, will be evaluated rarely—mostly by well-funded labs. This creates a risk that the benchmark becomes two separate instruments: a 59-task leaderboard for routine comparison, and a 91-task milestone test that few run.

---

## 5. ALE-CLI vs. Terminal-Bench: Why the Gap Is 3×

### The comparison

- **Terminal-Bench** (ICLR 2026, arXiv:2601.11868): ~100 terminal-centric tasks, mostly coding, security, math, file operations. Codex + GPT-5.5 scores **82%**.
- **ALE-CLI** (Linux-only subset of ALE): 106 tasks across 55 subdomains. Codex + GPT-5.5 scores **26.4% overall** (42.9% Near-Term, 21.4% Full-Spectrum, 4.5% Last-Exam).

### Why the gap is so large

The paper notes that "ALE-CLI tasks are substantially harder and require longer agent sessions" (Section 4.1). But there are three deeper reasons:

**1. Domain breadth, not just terminal vs. GUI.** Terminal-Bench tasks are concentrated in coding, security, and scientific computing—domains where frontier LLMs have extensive training data. ALE-CLI spans finance, manufacturing, biomolecular design, visual media, agriculture, and more. Even when stripped of GUI requirements, the tasks require domain-specific knowledge that models lack.

**2. Task duration and complexity.** Terminal-Bench tasks have a median completion time under 20 minutes for successful runs (their Figure 10). ALE tasks routinely take hours. The difference is not merely scale; it is that ALE tasks are *end-to-end deliverables* rather than single-function implementations or bug fixes.

**3. Verification strictness.** Terminal-Bench uses unit-test-style verification (pytest pass, exit code check). ALE uses gate-and-score: a binary precondition must pass before any quality metric is evaluated. In manufacturing/gcode, for example, a collision/gouge gate forces the score to 0 regardless of geometric similarity (Appendix C.3.3). This is more punishing than typical software test suites.

**Non-obvious implication:** The 3× gap suggests that Terminal-Bench, while valuable, is not a proxy for general agent capability. It is a proxy for *coding-adjacent* capability. The coding domain is an outlier where agents perform best, partly because compilers give sharper reward signals and the task surface is mature. ALE's CLI subset is a harder test precisely because it forces agents out of the coding comfort zone.

---

## 6. Harness vs. Backbone: The Model Dominates

### The evidence

Appendix D.4 (Figure 12) isolates model and harness effects:

- **Fixed harness (OpenClaw), varying model:** 18.0 pp spread in overall pass rate (Grok 4.3 at 5.3% to GPT-5.5 at 23.3%).
- **Fixed backbone (GPT-5.5), varying harness:** 6.0 pp spread (five harnesses, 19.3–25.3%).
- **Fixed backbone (Opus 4.7), varying harness:** 5.3 pp spread (three harnesses, 14.7–20.0%).

Model choice accounts for roughly **3× the variance** of harness choice.

### What this means for the field

ALE-Claw is a stripped-down harness: it removes long-term memory, user customization, multi-channel gateways, and the plugin framework, reducing the system prompt by ~65% (Appendix C.4). Yet it performs comparably to Codex, Claude Code, and Cursor when paired with the same model. ALE-Claw + GPT-5.5 scores 24.2% overall vs. Codex + GPT-5.5 at 26.2%—a 2.0 pp gap for a vastly simpler system.

This is a striking result. The agent-product industry (Claude Code, Codex, Cursor, Droid) is built on the premise that sophisticated harnesses—context management, sub-agents, memory, tool routing—matter for performance. ALE suggests that, at least for single-task benchmark evaluation, they do not matter much. The backbone model's reasoning and domain knowledge swamp the harness differences.

**Caveat:** This finding may not generalize to *interactive* or *multi-session* deployment, where memory, user preference learning, and cron/heartbeat systems (the features ALE-Claw strips out) are plausibly more important. ALE evaluates isolated benchmark runs, not ongoing assistant relationships.

**Non-obvious implication:** If you are a researcher with limited budget, you get more bang for your buck by tuning or upgrading your foundation model than by engineering a fancier harness. The field may be over-investing in agent scaffolding relative to model capability. Conversely, if you believe that long-horizon autonomy requires richer architecture, ALE's single-task design may not be the right testbed to demonstrate it.

---

## 7. Cross-Benchmark Positioning

ALE is correctly positioned as a superset of prior work. The coverage map (Table 2 / Figure 3) shows that even the *union* of 16 major prior benchmarks leaves 13 of 55 ALE subdomains entirely uncovered. Key differentiators:

| Benchmark | Breadth (ALE subdomains) | Real workflows | Deterministic verification | Human grading |
|---|---|---|---|---|
| MMLU, GPQA, HLE | Narrow (exam topics) | No | Yes | No |
| SWE-bench | 1 (software engineering) | Yes (GitHub issues) | Yes | No |
| OSWorld | 3–4 (general computer use) | Curator-authored | Partial | No |
| Terminal-Bench | 3–4 (coding/security/math) | Curator-authored | Yes | No |
| GAIA | 2–3 (general assistant) | Curator-authored | Partial | No |
| GDPval | 16 | Real projects | No | Yes |
| RLI | 14 | Real projects | No | Yes |
| **ALE** | **55 (all covered)** | **Yes (expert-sourced)** | **Yes** | **No** |

ALE's genuine innovations are: (1) the O*NET-anchored taxonomy, which forces systematic coverage rather than ad hoc domain selection; (2) the expert-sourcing pipeline, which ensures tasks reflect actual practice; (3) the gate-and-score verification design, which replaces human grading with deterministic, artifact-based checks; and (4) the rolling-evaluation strategy (150 public / 1,017 private), which mitigates contamination over time.

The closest contemporaries, GDPval and RLI, share the "real work" ambition but rely on expensive human grading and cover smaller industry slices. ALE's automated verification is a genuine engineering achievement that enables scale. The tradeoff is that some aspects of quality (aesthetic judgment, strategic soundness, client satisfaction) are harder to capture with deterministic checks. The authors acknowledge this and reserve LLM judges for cases where no deterministic alternative exists, using narrow yes/no probes rather than holistic scoring (Appendix C.3.4).

---

## 8. Limitations and Open Questions

### Contamination and rolling evaluation
The 150/1,490 public split is a good start, but 150 tasks is small. If frontier models are trained on the public tasks, the private pool's advantage may erode quickly. The rolling-evaluation promise—periodically rotating private tasks into the public set—depends on sustained curation effort. The paper does not specify the rotation cadence or governance.

### Difficulty-tier construction
The three tiers are described as "Near-Term" (agents can partially complete), "Full-Spectrum" (one task per subdomain), and "Last-Exam" (hardest workflows). But the assignment criteria are opaque. Are tiers assigned by expert judgment, pilot-agent performance, or some hybrid? Without knowing the assignment rule, it is hard to interpret the 0% pass rates on Last-Exam: are these genuinely unsolved problems, or just problems where the evaluation gate was set especially tight?

### GUI-underutilization as a design tension
The failure taxonomy reveals that agents avoid GUI tools due to domain ignorance, not GUI incompetence. But the benchmark's design (GUI-as-Tool via CUA MCP bridge) was built on the assumption that GUI capability is the missing link. There is a tension here: the benchmark invests heavily in GUI infrastructure, but the data suggest that better GUI tools would not move the needle much. Should future versions of ALE de-emphasize GUI and focus on domain-knowledge retrieval?

### Cost barrier to entry
At $300–1,100 per full evaluation, ALE is accessible to well-funded labs but not to individual researchers or small teams. The Near-Term tier ($100–200) is more reasonable, but it evaluates only 59 tasks. The benchmark risks becoming an instrument for big labs while the broader community optimizes on cheaper proxies (Terminal-Bench, SWE-bench).

### Generalization beyond single-shot deliverables
ALE tasks are atomic and self-contained. The hardest problems in real professional work—scoping ambiguous requirements, managing client expectations, maintaining consistency across a portfolio of deliverables—are not captured. The authors might argue that you must walk before you run, and ALE is the "walk" phase. Fair enough, but the GDP-relevant framing implies running.

---

## 9. Probe Artifacts and Code Investigation

### Repository investigation
The open-source framework is available at `github.com/rdi-berkeley/agents-last-exam` (Apache-2.0 for code, CC-BY-4.0 for data). Key findings from code inspection:

- **ALE-Claw** (`ale_run/agents/ale_claw/`) is a ~20KB Python harness (deployer.py) that implements the core action loop, tool bindings, context compaction, and GUI delegation. It is intentionally minimal and reads as a clean reference implementation.
- **Task structure** (`tasks/<domain>/<task>/`) follows a consistent `main.py` pattern with `load()`, `start()`, and `evaluate()` lifecycle methods. Evaluation scripts live in per-task `scripts/` directories and are uploaded to the VM for execution.
- **Public task inventory** includes concrete examples like `equity_research_summary`, `sec_10k_financial_parsing`, `manufacturing/gcode`, and `game/mota_reproduction`. The diversity is real—these are not rehashed coding tasks.
- **CUA MCP bridge** exposes 14 desktop-action tools (screenshot, click, type, scroll, key presses) as ordinary tools in the agent loop. The integration is clean but confirms that GUI actions are treated as first-class tool calls, not as a separate modality.

### Terminal-Bench comparison probe
Terminal-Bench (arXiv:2601.11868) tasks have median completion times under 20 minutes and concentrate on coding, security, and math. ALE tasks span domains where median expert time is hours. The 3× performance gap (82% vs. 26.4%) is therefore not a fair comparison of "same difficulty, different interface" but rather a comparison of "narrow coding tasks vs. broad professional workflows." Both benchmarks are valid; they measure different things.

---

## 10. Summary of Claims: Supported vs. Unsupported

| Claim | Verdict | Evidence |
|---|---|---|
| ALE has the broadest industry coverage of any agent benchmark | **Supported** | 55 subdomains, 13 clusters, coverage map shows 13 subdomains uncovered by union of 16 prior benchmarks (Table 2) |
| Tasks are sourced from real professional practice | **Supported** | 960 external submissions + 530 commissioned tasks, expert advisory committee, multi-round QC (Section 2.3) |
| Verification is deterministic and artifact-based | **Supported** | Gate-and-score design, code-based judges by default, narrow LLM probes when unavoidable (Appendix C.3) |
| Frontier agents score below 50% even on the easiest tier | **Supported** | Best configuration: 42.4% pass on Near-Term (Table 1) |
| The hardest tier is far from saturated | **Supported** | Best configuration: 8.6% pass on Last-Exam; many configurations at 0% (Table 1) |
| Tasks are "economically valuable" | **Partially supported / overstated** | Tasks resemble professional deliverables, but evaluated in isolation without organizational context, feedback, or maintenance |
| ALE measures "GDP-relevant impact" | **Unsupported / overstated** | No evidence linking ALE pass rates to GDP, employment, or actual economic output |
| GUI capability is the critical missing link | **Contradicted by own data** | Failure taxonomy shows domain knowledge (not GUI execution) is the bottleneck |
| Agent harness engineering matters greatly | **Weakly supported** | Harness variation explains ~6 pp spread; model variation explains ~18 pp (Appendix D.4) |
| Five hours is sufficient for meaningful evaluation | **Partially supported** | 4.3% timeout rate is low, but total wall-clock times (72h for a tier) reveal the cap truncates slow agents |

---

## 11. Bottom Line

ALE is a landmark benchmark. Its scale, expert involvement, verification rigor, and rolling-evaluation design set a new standard for agent evaluation. The technical execution is impressive, and the 150 public tasks are a genuine community resource.

But the paper's loftiest claims—"economically valuable," "GDP-relevant," "last exam"—are not yet earned by the evidence. What ALE actually shows is that frontier agents struggle to produce single-shot technical deliverables across a broad professional taxonomy, and that the binding constraint is domain knowledge in the foundation model, not the sophistication of the agent harness.

That is still a profound finding. It redirects research attention from agent scaffolding toward model capability and domain-specific knowledge acquisition. And it establishes a durable, hard-to-game benchmark that the field can use for years. But if you are looking for evidence that AI is about to automate professional work at scale, ALE does not provide it. It provides evidence that AI cannot yet automate the *first step* of that work—the production of an isolated deliverable under controlled conditions.

The real "last exam" is not ALE. It is whether agents can operate in the messy, iterative, socially embedded context where economic value is actually created. ALE is an excellent *midterm*. We are not ready for the final.
