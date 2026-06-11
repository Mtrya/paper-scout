# ALE-Bench Deep-Dive — 2506.09050

**Run:** 2026-06-11-alebench-worldpilot-embodiedr1-nextforcing  
**Paper:** ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering  
**Authors:** Yuki Imajuku et al. (Sakana AI, University of Tokyo, AtCoder)  
**Repo:** `code/alebench/` | **Paper:** `papers/control/alebench-2506.09050.md`

---

## 1. What ALE-Bench Measures and Why It Matters

ALE-Bench is the first large-scale benchmark for **score-based algorithmic programming contests** — specifically the AtCoder Heuristic Contest (AHC). Unlike pass/fail benchmarks (IOI, Codeforces, SWE-Bench), AHC problems are NP-hard optimization tasks with no known exact solution: package-delivery routing, crew scheduling, factory planning, power-grid balancing. Contestants submit code that produces a score, and rankings are determined by that score on a hidden test set.

The benchmark is co-developed with AtCoder Inc. and ships with:
- 40 real contest problems (full) + 10-problem lite subset
- Official problem statements, scorers, visualizers, and human leaderboards
- A Python framework that replicates the contest environment (time limits, Docker sandbox, public/private evaluation, visualization server)
- Performance metrics drawn directly from AtCoder's Elo-like rating system

**Why this matters now:**
- Classic coding benchmarks are saturating. Frontier LLMs already rival human experts on exact-solution contests.
- Score-based contests are **open-ended** — even after AI surpasses the best humans, scores can keep improving, enabling continuous evaluation.
- Each problem maps to a real industrial optimization domain. The tasks are CPU-bound (2–10s per case), require deep domain reasoning, and reward iterative refinement over hours or weeks.

---

## 2. Key Findings from the Paper

### One-Shot (No Iteration)
- **o3-high** is the only model to break average performance 1000 (1044). Most models cluster between 550–850.
- Even the best models achieve ≥1600 performance on ≤5% of problems and never hit 2000.
- **C++20** dominates Python3 and Rust in average performance, though the gap is nuanced — Rust slightly beats C++20 on ≥1600-rate, suggesting language proficiency matters beyond raw speed.

### Iterative Refinement (4 Hours, Self-Refine with Summarization)
- **o4-mini-high** leads with 1520 average performance, rating 2104 (top 11.8% of humans by rating).
- All four tested models (GPT-4.1 mini, o4-mini-high, Gemini 2.5 Pro, DeepSeek-R1) improve by 400+ points over one-shot.
- Score trajectories show steady improvement; code size grows monotonically, suggesting incremental feature addition rather than random sampling.

### ALE-Agent Scaffolding (Lite Subset)
- **Base** (sequential self-refine, no domain prompts) ≈ iterative-refinement performance.
- **+ Method 1** (domain-knowledge prompts + 3-turn refinement) gives a modest boost.
- **+ Method 1&2** (full beam search, width 30) achieves **2880 performance on AHC039** — equivalent to **5th place in the actual contest**. This was verified by submitting the solution to AtCoder post-hoc.
- **OpenHands** (general-purpose agent) barely improves over one-shot and often exits early, underscoring that generic scaffolding fails in this domain.

### The Human Gap
- **Rating is misleading for AI.** o4-mini-high's rating (2104, top 11.8%) overstates its skill. Humans in that rating band average 56.2% of problems ≥1600; o4-mini-high achieves only 32.5% — closer to the 1600–1700 human band.
- LLMs show **high variance**: they spike to ≥2400 on 5% of problems but crater on others. Humans at the same rating are far more consistent.
- **Short contests > long contests for AI.** Models generate 100–1000 solution variants in a few hours — impossible for humans. This "parallel search" effect flatters short-contest performance, while long contests (where humans develop deep, problem-specific insights) remain harder.

---

## 3. Code Investigation

### 3.1 Benchmark Structure (`ale_bench/session.py`, `data.py`)

The core abstraction is a **`Session`** that wraps a single problem and enforces contest rules:

```python
class Session:
    def public_eval(self, code, code_language) -> Result
    def private_eval(self, code, code_language) -> tuple[Result, int, int]  # result, rank, performance
    def case_eval(self, input_str, code, ...) -> Result
    def case_gen(self, seed, gen_kwargs) -> str | list[str]
    def local_visualization(self, input_str, output_str) -> Image | list[Image]
```

Key design decisions observed in the code:
- **Resource gating:** `Session` tracks a `ResourceUsage` object (`num_case_gen`, `num_case_eval`, `num_call_public_eval`, `execution_time_case_eval`). Each action checks against `maximum_resource_usage`. For `public_eval`, if `use_same_time_scale` is enabled, a cooldown enforces the original contest submission interval (5 min for short contests, 30 min for long). This is the mechanism that ostensibly forces "long-horizon" pacing.
- **Rank estimation:** `private_eval` computes rank by inserting the agent's aggregate score into the original contest leaderboard and linearly interpolating performance. For problems with relative scoring (e.g., `ahc016`, `ahc017`), it recalculates normalized scores against all human participants.
- **Docker sandbox:** Every compilation and execution runs inside a language-specific Docker container (C++17/20/23, Python, Rust) on a single vCPU with 2 GiB RAM, replicating AtCoder's judge environment.

### 3.2 Tool Wrappers (`tool_wrappers/case_runner.py`, `code_runner.py`)

The execution pipeline is a three-stage Docker dance:

1. **Compile** (`run_compile_container`): writes code to a temp file, mounts it read-only, compiles inside the judge image. Syntax errors in Python/Julia are treated as compilation errors.
2. **Run** (`run_batch_run_container`): mounts the compiled binary + input file, runs with `timeout` and `prlimit` for wall/CPU time enforcement, captures output via GNU `time` in JSON format.
3. **Judge** (`run_batch_judge_container`): runs the Rust `tester` binary against the output to extract `Score = N`. If the score line is missing → `WRONG_ANSWER`.

Case evaluation is parallelized via `ThreadPoolExecutor` in `run_cases()`, with `num_workers` configurable (the paper uses up to 13 workers per problem).

### 3.3 ALE-Agent Scaffolding (`ale_bench_eval/scaffolds.py`, `prompts/`)

The evaluation harness in `__main__.py` orchestrates a four-phase pipeline:

```
Phase 1: Repeated Sampling  → N independent LLM calls (parallelized)
Phase 2: Selection          → Pick best/median by public-eval score
Phase 3: Self-Refinement    → Iterative improvement with feedback + summarization
Phase 4: Private Evaluation → Run best checkpoints from each phase on hidden tests
```

**Repeated sampling** (`run_repeated_sampling`):
- Dispatches up to `max_repeated_sampling_workers` concurrent LLM calls.
- Each call receives the initial problem prompt. No history is shared across samples.
- Evaluates each sample on public cases (or a custom subset if `n_public_cases` is set).

**Self-refinement** (`run_self_refinement`):
- Loads the best sample as the starting point.
- For each iteration, constructs a feedback message from the public evaluation result (score or error details) and appends it to the conversation history.
- Uses **summarization** to avoid context overflow: after the first generation, the LLM must output a summary in a ````md```` block. Subsequent prompts replace the full history with: (problem statement) + (summary) + (best code + feedback) + (latest code + feedback).
- If the model overflows context (`MaxTokenError`), refinement stops.

**Prompt design** (`prompts/texts.py`, `builder.py`):
- System prompt: "You are a world-class algorithm engineer... asked to solve a heuristic problem, known as an NP-hard problem."
- Initial prompt asks the model to analyze the problem, then implement in a specific language block (e.g., ````cpp ... ````).
- Feedback prompt template instructs the model to "analyze this given feedback and list what insights can be gained from it," then refine.
- Domain-knowledge prompts (Method 1) are injected stochastically during ALE-Agent expansion, covering simulated annealing state/neighbor design, beam search, and speedup strategies.

### 3.4 ALE-Agent Tree Search (Appendix B → inferred from scaffolds.py + paper)

ALE-Agent proper is **not** just the sequential self-refine in `scaffolds.py`. It layers a best-first search tree on top:

- Each node = a solution (source code).
- **Selection:** priority = acceptance ratio on 50 public cases + score. Tabu mechanism prevents re-expanding the same parent.
- **Expansion:** beam width of 30. The LLM is prompted with (current code, best-so-far code, trajectory summary, stochastic domain guidance) and generates refinements over **three conversational turns**. The best of the three turns becomes a child node.
- The 30 children are evaluated asynchronously. Because reasoning models (o-series, Gemini 2.5) can take minutes per response, the parallelism amortizes latency.

This is, in effect, an **LLM-driven genetic algorithm** with beam selection and domain-informed mutation prompts. The "long-horizon" claim rests on whether this tree search constitutes genuine iterative insight accumulation or simply a more efficient parallel sweep.

---

## 4. Critical Claims to Verify: Long-Horizon Problem Solving or Parallel Search?

This is the central conceptual tension in the paper. The authors address it directly in **Section 5.4** with a targeted experiment.

### The Experiment
- Select 6 problems where o4-mini-high achieved ≥2000 in iterative refinement.
- Run **150 independent One-Shot trials** per problem (matching the ~150 code generations in iterative refinement).
- Compare the max score from independent search vs. the iterative-refinement max.

### The Evidence

| Problem | Iter-Refine | One-Shot Max | Gap |
|---------|-------------|--------------|-----|
| ahc005  | 2107        | 1722         | -385 |
| ahc006  | 2472        | 2174         | -298 |
| ahc012  | 2236        | 1388         | -848 |
| ahc020  | 2545        | 1731         | -814 |
| ahc041  | 2306        | 1911         | -395 |
| ahc044  | 2150        | 1831         | -319 |

The authors conclude: *"simply exploring a large number of independent ideas is insufficient... success demands the ability to refine and improve solutions based on feedback."*

### Our Assessment

**The claim holds, but with important caveats:**

1. **Peak vs. distribution:** The gap at the *maximum* is real and large (300–800 points). However, the paper does not report the *mean* of the 150 one-shot trials vs. the iterative mean. If independent search has a long tail of low scores, the mean gap could be even larger — or, if iterative refinement also has high variance, the comparison is less clean than it appears.

2. **The mechanism is not human-like long-horizon reasoning.** The iterative-refinement protocol in the code (`run_self_refinement`) is essentially: (a) generate code, (b) evaluate, (c) feed score/error back, (d) regenerate. There is **no persistent state** beyond the conversation summary and the "best code so far." The LLM does not maintain a running algorithm design document, a profiler log, or a mathematical analysis. It is a local hill-climbing loop in code space, not a deep strategic investigation.

3. **ALE-Agent's tree search is parallel search with memory.** The beam width of 30 means 30 independent trajectories are explored concurrently. The "feedback" that propagates is the public-eval score, not a semantic understanding of why an algorithm failed. The tabu list and best-first selection are classical search heuristics; the LLM is acting as a mutation operator.

4. **The authors are honest about this.** Section 5.4 notes that o4-mini-high generated **>100 solutions** in 4 hours, and ALE-Agent generated **~1000** — "unrealistic for humans, even in long-format contests." They explicitly frame AI's strength as "sheer volume" rather than sophistication. This is commendable transparency.

**Verdict:** The benchmark *does* measure something beyond brute-force independent sampling — feedback-driven mutation improves the peak. But the "long-horizon" framing should be read as "many iterations of generate-test-mutate" rather than "months of accumulated human insight." The benchmark is valuable precisely because it captures this intermediate regime.

---

## 5. What This Run Learned That Wasn't Obvious from the Abstract

1. **The rating system is a trap.** AtCoder's rating is designed for humans: it rewards high peaks and does not heavily penalize catastrophic failures. This makes it a poor primary metric for AI. The paper's insistence on **average performance** and **performance distribution** as the true signals is well-founded and easy to overlook in headline numbers.

2. **Domain scaffolding >>> generic agents.** OpenHands (a strong general-purpose coding agent) achieved marginal gains over one-shot and often quit early. ALE-Agent's domain-specific prompts (SA neighbor design, beam-width tuning) and tree-search structure are responsible for the leap to contest-top-5 performance. This suggests that "agentic" gains are highly domain-dependent.

3. **Lite version is not a free shortcut.** The lite/full correlation is strong (Pearson r=0.917), but the regression slope is ~0.7, meaning lite is systematically harder. Direct cross-version comparisons are misleading. The lite subset was deliberately curated to be harder and more diverse.

4. **C++ proficiency is part of the task.** The benchmark faithfully replicates AtCoder, where execution speed matters. LLMs that generate better C++ (tighter loops, appropriate data structures) get higher scores not because they solved the problem more cleverly, but because their code runs more iterations within the time limit. This conflates algorithm design with implementation efficiency — a feature, not a bug, since real algorithm engineering has the same property.

5. **Real contest validation happened.** The ALE-Agent AHC039 solution (5th place) was actually submitted to AtCoder and confirmed. An in-development agent also participated live in AHC046 as user `fishylene`, placing 154th with performance 1915. This grounds the benchmark in reality better than most synthetic evaluations.

6. **The framework is remarkably production-grade.** The Docker sandboxing, Rust toolchains, polars-based leaderboard math, visualization server orchestration, and checkpoint/resume logic in `__main__.py` show serious engineering. The code is not a throwaway research artifact; it is designed for sustained use.

---

## 6. Limitations and Honest Assessment

### From the Paper
- **Small dataset:** 40 problems. The authors argue variance is smoothed by long-horizon accumulation, but this remains a thin statistical basis compared to e.g. LiveCodeBench or APPS.
- **No error bars:** Single runs per configuration due to cost. Appendix C.5.1 shows 95% CIs for a subset of one-shot models; o3-high's CI is ±47, which is tight, but iterative-refinement runs are unreplicated.
- **No multimodal experiments:** Image inputs are supported in the code (`use_image`, `statement_images`) but were not evaluated.
- **No tool-use experiments:** Input generation and web visualizers are exposed in the `Session` API, but the paper did not test LLMs using them autonomously via tool-calling.
- **Contamination check is weak:** A scatter plot of performance vs. contest date with a cutoff line is suggestive but not conclusive. Given that AHC problems are publicly discussed in Japanese competitive programming communities, training-data leakage is plausible for some models.

### From This Investigation
- **The "long-horizon" claim is overstated in the abstract.** The body of the paper is more careful, but a casual reader could come away thinking LLMs are doing weeks-long strategic reasoning. They are doing high-throughput local search with feedback.
- **ALE-Agent results are lite-only.** The full ALE-Agent (Method 1&2) was only run on 10 problems. The headline 2880 performance is on a single problem. We don't know how consistently it performs across the full 40.
- **Cost is a hidden barrier.** o4-mini-high costs ~$7/problem in iterative refinement; Gemini 2.5 Pro costs ~$11/problem. A full 40-problem sweep with ALE-Agent could run into hundreds of dollars per model, limiting reproducibility.
- **Human comparison is slightly unfair to humans.** AI agents get instant compilation feedback, can generate hundreds of test cases, and never get tired or discouraged. Human contestants in AHC operate under psychological and ergonomic constraints that are not modeled.

### Bottom Line
ALE-Bench is a **genuinely new and well-executed benchmark** that fills a real gap. It convincingly shows that frontier LLMs are competent but inconsistent algorithm engineers — strong novices, not experts. The gap is narrowest on well-understood problem templates (simulated annealing for routing) and widest on problems requiring ad-hoc evaluation functions or deep structural insights. The framework is solid enough to serve as a platform for the next generation of agent research, provided the community invests in expanding the problem pool and running replicated, statistically powered evaluations.

---

## 7. Code Probes

Small inspection scripts written during this investigation are preserved in `code/`:

- `probe_session_resources.py` — Dumps the `Session` resource-gating fields and public-eval cooldown logic.
- `probe_scaffold_loop.py` — Prints the phase structure of `evaluate_contest()` and the parallelism signatures of repeated sampling / self-refinement.
