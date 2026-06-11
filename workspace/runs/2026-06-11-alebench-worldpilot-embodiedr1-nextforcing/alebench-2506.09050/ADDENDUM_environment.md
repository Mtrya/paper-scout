# Addendum: Is the ALE-Bench Gap Due to an "Alien" Environment?

**Question:** Do frontier LLMs underperform on ALE-Bench because the benchmark environment — Docker sandbox, Rust scorer, C++ pipeline, Session API, score-based evaluation — is unfamiliar or hostile to them? Or is the gap genuinely about reasoning?

**Conclusion (up front):** The environment is not alien, unfamiliar, or a significant source of friction. The performance gap is overwhelmingly a reasoning limitation. LLMs consistently produce syntactically valid, compilable code that runs without error; they simply fail to invent the problem-specific heuristics, neighbor functions, and evaluation criteria that separate mediocre from expert solutions. Every strand of evidence in the paper and the code supports this.

---

## 1. The Environment Is Standard Competitive Programming Infrastructure

Reading the code and prompts reveals nothing exotic.

**The prompts are textbook competitive-programming style.** In `src/ale_bench_eval/prompts/texts.py`, the system prompt reads:

> "You are a world-class algorithm engineer, and you are very good at programming. Now, you are participating in a programming contest. You are asked to solve a heuristic problem, known as an NP-hard problem."

The implementation prompt then asks the model to analyze the problem and implement in a specified language block (e.g. ` ```cpp ``` `). It lists available libraries by name and version (AC Library, Boost, numpy, scipy, etc.). This is the exact same format used in hundreds of coding benchmarks (APPS, CodeContests, LiveCodeBench) and in millions of training examples. There is no novel DSL, no arcane protocol, and no hidden state machine.

**The Session API is a thin Python wrapper around standard compilation and execution.** `session.py` exposes six methods: `code_run`, `case_gen`, `case_eval`, `case_gen_eval`, `public_eval`, and `local_visualization`. Under the hood (`tool_wrappers/case_runner.py`), each evaluation is a three-step Docker pipeline:

1. Compile the source inside a language-specific container (standard `g++` / `rustc` / `python` commands).
2. Run the binary with `timeout` and `prlimit` for resource enforcement.
3. Pass the output to a Rust `tester` binary that extracts `Score = N`.

This is identical to how Kaggle, Codeforces, or any university auto-grader works. An LLM never interacts with the Session API directly in the one-shot and iterative-refinement experiments; the scaffold calls it on the model's behalf and feeds back the score or error message in natural language.

**The languages and toolchains are exactly what LLMs are trained on.** C++20 (GCC 12.2), Python 3.11, Rust 1.70 — these are mainstream versions. The Docker images install standard libraries (Boost, AC Library, numpy, scipy, networkx, torch, etc.) that appear constantly in open-source repositories. There is no proprietary framework or custom runtime.

---

## 2. One-Shot Retries Prove the Bottleneck Is Not the Environment

The one-shot protocol (Appendix C.2) is the most direct test of the "alien environment" hypothesis. The setup:

> "After each code generation, a public evaluation was performed. If the result was ACCEPTED, no further generation for that problem was permitted. Otherwise, feedback (per defined templates) was provided, and the model attempted to revise its code. If an ACCEPTED status was not achieved after five attempts, the last generated code was used for private evaluation."

If the environment were alien — if models were struggling with Docker quirks, Rust scorer formatting, or compilation flags — we would expect to see them burning through all five retries on compilation or runtime errors, and achieving near-zero scores. That is not what happens.

**Models achieve ACCEPTED status on the vast majority of problems on the first or second try.** Table 1 shows o3-high scores ≥400 (the lowest non-trivial tier) on 97.5% of problems. Even GPT-4o mini, a small model, hits ACCEPTED on 75% of problems. The failures that do occur are not environmental: Appendix C.3 notes that Gemini 2.5 Pro occasionally runs "very close to the time limit (e.g., 1.98s for a 2s limit)" and gets TIME_LIMIT_EXCEEDED on a few cases — a classic algorithmic efficiency issue, not an environment misunderstanding. DeepSeek-R1's failures in iterative refinement stem from "issues with instruction following" (failing to generate the required summary block), which is a capability gap, not an environment gap.

**The critical observation:** models routinely produce runnable code that scores poorly. If the bottleneck were compilation, formatting, or sandbox unfamiliarity, the retry data would show repeated COMPILATION_ERROR or WRONG_ANSWER verdicts. Instead, the bottleneck is the *score* itself — the algorithmic quality of the heuristic. The environment is doing its job perfectly; the model's reasoning is not.

---

## 3. OpenHands Failed Because the Problems Are Hard, Not Because the Environment Is Confusing

OpenHands (v0.34.0) is a general-purpose coding agent with file-system access, web browsing, and tool use. The authors gave it every conceivable advantage: exact compile commands, a visualization server, the full problem statement, and explicit instructions. The OpenHands prompt (Appendix C.4) even says:

> "After you implement your solution, **you must ask user (not exit) to evaluate your temporary solution**... Even if your solution get accepted, you must refine your solution or try another approach to get a better score."

Despite this hand-holding, OpenHands "often exited prematurely, indicating difficulties with improving" (Section 5.3). The cost data confirm this: OpenHands cost only $0.15–$3.25 per problem, versus $2.14–$11.13 for sequential iterative refinement and $7.64 for ALE-Agent. It quit early because it could not figure out *what to do next* — not because it misunderstood how to compile C++ or how to call `public_eval`.

This is strong evidence that generic agent scaffolding fails on ALE-Bench for the same reason generic agents fail on hard math problems: the domain requires deep, problem-specific insight (designing a simulated annealing neighborhood, crafting a custom evaluation function, choosing a beam width) that cannot be solved by file manipulation and web search. The environment was not the obstacle; the reasoning required was.

---

## 4. Language Comparison Points to Execution Efficiency, Not Familiarity

Table 2 shows C++20 outperforming Python3 and Rust in average performance (668 vs 624 vs 611). A naïve "familiarity" hypothesis would predict C++20 dominates because LLMs see more C++ in training. But the paper explicitly attributes the C++20 lead to speed, not syntax comfort:

> "C++20 may generate effective solutions more quickly, but it does so with higher token consumption per response."

More tellingly, Rust slightly *exceeds* C++20 on problems with ≥1600 performance (1.1% vs 1.0%). If raw training-data familiarity were the driver, C++ — by far the most common competitive programming language in corpora — should dominate at every threshold. The fact that Rust is competitive at the high end undercuts the familiarity argument and supports the efficiency argument: compiled languages allow more iterations within the CPU time limit, and the exact choice between C++ and Rust matters less than the fact that both are fast.

Python3's lower scores are consistent with this interpretation. Python is the language LLMs are arguably *most* familiar with, yet it performs worst on average. The bottleneck is not "Can the model write valid Python?" — it clearly can — but "Can the model's Python solution run enough iterations in 2 seconds to be competitive?" The answer is often no.

---

## 5. Visualization Is a Missing Modality, Not an Environmental Barrier

The paper notes: "Our primary experimental setup relied solely on text-based feedback and did not utilize the visualization, except for the OpenHands." Image inputs are supported in the code (`use_statement_image`, `local_visualization`) but were not evaluated in the main experiments.

This is a genuine experimental gap. It is plausible that visualizations — especially for geometric routing or packing problems — could help models build better intuitions. However, this does not mean the current text-only environment is "alien." It means the authors ran an ablated study (text-only) and found that even with text-only feedback, models perform at a novice-to-intermediate level. The OpenHands experiment, which *did* use the visualization server, still failed to improve over one-shot. If visualizations were the missing key, OpenHands — the only agent with browser access to the visualizer — should have outperformed sequential refinement. It did not.

Appendix C.6 candidly flags this as a limitation: "We have not used the image input function in our experiments, and further verification is needed in this area." The honest framing is "untested modality," not "hostile environment."

---

## 6. Appendix A.4: The Paper's Own Limitations Are Environmental, Not Alien

Appendix A.4 lists three limitations:

1. **Differences in judging environments** (older contests ran on slightly slower hardware). The authors re-evaluated and found "little change" in scores.
2. **Use of new resources** (modern algorithms not available when the contest was held). This is a fairness issue for human comparison, not an LLM handicap.
3. **Problem-set contamination** (public problems may be in training data). The contamination check found no effect.

None of these mention API confusion, prompt misalignment, or toolchain unfamiliarity. The authors do not consider the environment alien because it is not.

---

## 7. The Genuine Insight: LLMs Are Novice Heuristic Designers

Synthesizing the evidence, the ALE-Bench gap has a clear diagnosis:

**Frontier LLMs are competent but novice algorithm engineers.** They read problem statements fluently, write syntactically correct C++/Python/Rust, compile and run it without drama, and even iterate on feedback. What they cannot yet do reliably is the creative core of heuristic contest programming: inventing a problem-specific state representation, designing a neighborhood function that exploits the problem structure, crafting an ad-hoc evaluator that guides local search, and knowing when to switch from simulated annealing to beam search to greedy construction.

Appendix C.5.3 makes this explicit:

> "AI systems achieved relatively high performance on routing-type problems... In contrast, performance on planning-type problems... was lower. This appears to be due to the need for problem-specific neighborhood structures and evaluation functions, which are more difficult for the AI systems to construct automatically."

This is not an environment problem. It is a reasoning problem. The ALE-Agent result — 5th place on AHC039 using simulated annealing — proves that when the right domain scaffolding (method prompts + tree search) is layered on top, the environment friction disappears entirely and the model can reach expert-level performance on specific problem types. The scaffold supplies what the model lacks: strategic direction for heuristic design. The benchmark environment itself is a faithful, transparent, and largely transparent replication of a well-known contest format.

**Final verdict:** The "alien environment" hypothesis is refuted by the data. The Session API is familiar Python; the prompts are standard competitive programming; the toolchain is mainstream; models compile and run successfully on the first try; and the one-shot retry data show ACCEPTED verdicts with poor scores, not repeated environment failures. The gap is in algorithmic invention, not in environmental adaptation.
