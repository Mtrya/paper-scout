# Deep-dive: Harness Handbook (arXiv 2607.13285)

**Paper**: *Harness Handbook: Making Evolving Agent Harnesses Readable, Navigable, and Editable* — Tencent HY LLM Frontier + Indiana U + UMD + UGA + NUS. Local copy: `papers/agents/harness-handbook-2607.13285.md` (full text incl. appendices A–E).

**Thread question**: is behavior-centric organization of a codebase feasible, and does the paper's core mechanism (behavior localization via an L1–L3 handbook + state registers) hold up when you try to build a tiny one yourself?

---

## 1. Pipeline mechanics (as understood from the full text incl. appendix)

**Representation** (Fig 1): an L1–L3 document tree `D` + a cross-stage state-register view `Z`.
- L1 system overview (architecture, execution model, stages, global data flow).
- L2 one component overview per execution stage (purpose, I/O, dependencies, local state).
- L3 one "unit card" per leaf. In function-as-leaf mode the L3 card schema (App. D.1.4) contains: `locator_role`, `stage_context`, `synopsis`, `interface` {signature, params, `reads_state`, returns, side_effects}, `execution_flow` (2–8 steps), `design_decisions` (1–5), `relations` {callers, core_callees, config_state_sources, results_to, siblings, `register_interactions` (read/write/clear/reset per register)}. Plus a source locator `file:start-end`.
- Two invariants: **progressive disclosure** (only descend when needed) and **behavior–implementation alignment** (every L3 locator must revalidate against current source; failures are *frozen*, not guessed).

**Construction** (Algorithm 2, App. A), leaf mode `g ∈ {function, file}` fixed for the handbook's lifetime:
- **Phase I (deterministic, no LLM)**: language adapters parse the repo → program graph `G`: internal-function nodes (qualname, file, signature, line range, enclosing class, observed state access), named external *boundary* nodes, call edges kept only when the target resolves internally or to a named boundary; **unresolved calls go to an audit log, never guessed**.
- **Phase II (LLM structuring)**: function-as-leaf starts from a trusted hand-authored seed skeleton `S0` and assigns each function (whole, or split into contiguous *regions* if it serves several behavioral roles) to stages via an LLM proposer + skeptic-reviewer loop (prompts D.1.1/D.1.2; reviewers "lean APPROVE so the pipeline keeps making progress"), iterating until fixpoint or budget. file-as-leaf instead writes one *file card* per file (D.1.3), then *infers* the stage skeleton from cards+graph (variants oneshot/doctor/agent), one primary stage per file + ≤2 secondary.
- **Phase III**: synthesize L1→L2→L3 top-down (function mode, bounded actor–critic–reflexion per node) or bottom-up rollup (file mode); build register view `Z`; **validate every L3 locator against the repo, freeze invalid ones**; package `H = (V, D, Z, g, K_g)` where `K_g` is the machine-readable resync state (graph, skeleton, organization state, config `Θ`, generation cache `B`, and `S0` in function mode).

**BGPD localization** (App. B.1): read L1/L2 → select stages → follow `Z` to add state-coupled stages → pick L3 entries → expand candidates along call edges (boundaries: context only, never edit sites) → **verify every candidate against the live repo** → evidence set `Ê_q` = (file, anchor, current excerpt).

**Modification + resync** (Algorithm 1, App. B): planner emits edit plan `P` as verbatim old/new blocks + action declarations `Γ` (modify/add/remove; rename = remove+add). A separate executor (no shell, no dir listing) applies `P`; any non-empty diff `Δ` triggers `Resync_g`: reparse → align old/new graph (function body fingerprints ignoring line numbers; moves treated as unchanged, locator line-offsets rolled forward) → if skeleton still valid, scoped update of affected entries + regenerate only affected L3/L2/L1 nodes (cache reuse); else full rebuild. Model calls in resync limited to four semantic steps; everything else deterministic. Unclassifiable content is frozen/recorded, never guessed.

## 2. Artifact availability — mostly absent

- Project page https://ruhan-wang.github.io/Harness-Handbook/ is a polished product page (incl. an unreleased "Handbook Studio" workbench demo). **No code repo, no constructed handbooks, no 60 modification requests, no plans/traces, no judge outputs.** The arXiv HTML links no GitHub either (checked).
- What *is* public: the **Terminus-2 harness itself** inside Harbor (`github.com/harbor-framework/harbor`, cloned to `code/harbor`, commit `d3e606d`, 2026-07-15; `src/harbor/agents/terminus_2/` = 5 .py files, 3,914 LOC; paper says 6 files / 103 functions — snapshot drift). The planner framework **NexAU** exists (`github.com/nex-agi/NexAU`).
- What cannot be obtained: DeepSeek-V4-Pro (planner + one judge), GPT-5.5 and Opus 4.8 (judges/reference-plan generators) — none publicly available. **The headline numbers are not externally reproducible** even if Tencent later releases the pipeline; Appendix E's single walkthrough is the only executable evidence shown.

## 3. The probe: a mini handbook on the real Terminus-2

`code/probe_handbook.py` (stdlib `ast` only, ~480 lines) mirrors the three phases on the real Terminus-2 source, replacing the LLM proposal/review loop with hand-written keyword rules + call-graph propagation. Rerun:

```bash
git clone --depth 1 https://github.com/harbor-framework/harbor code/harbor   # once
python3 code/probe_handbook.py <path-to>/code/harbor/src/harbor/agents/terminus_2 ./handbook-out
```

Full console output: `code/probe-output.txt`; generated mini-handbook: `code/handbook-out/` (`overview.md`, `index.md`, `registers.md`, `stages/stage-*.md`).

### What worked

- **Phase I replication is easy and genuinely useful**: 107 internal functions, 195 resolved internal call edges (paper: 257 on their snapshot), 318 unresolved calls logged (mostly `list.append`, `logger.debug`, `str.join` — correctly not guessed), 68 `self._*` state registers after filtering method references. A tiny `self._x = ClassName(...)` / annotated-attribute / parameter-annotation type inference recovers 10 cross-file edges (Terminus2→TmuxSession→AsciinemaHandler), the load-bearing edges for BGPD's expansion step.
- **The state-register view is the star.** The behavior-localization demo uses the paper's own Terminus-2 Q1 ("mark task_complete three times"). Handbook path: request → "Main Agent Loop / completion gate" stage → register `_pending_completion` → exactly 3 functions / 7 line hits (`__init__` L305, `_reset_per_run_state` L1553, `_run_agent_loop` L1405–1417 + L1532), verified 3/3 against source. This **exactly matches the paper's Appendix E answer key** (their L292/L1574/L1427–1440/L1552–1559, modulo snapshot drift). Naive `grep -rn task_complete` returns **41 hits across 5 files** (prompt templates ×9, both parsers ×20, loop ×12) — mirrored implementations and doc strings, i.e. real noise even in a 4k-LOC repo; and `grep _pending_completion` only works if you already know the identifier. The register view converts a behavior question into an identifier you didn't know to grep for.

### What broke / failure modes observed first-hand

- **The completion gate is not a function.** It is ~15 lines inside the 310-line `_run_agent_loop`. My function-granularity locator (`terminus_2.py:1231-1540`) is far too coarse; this is precisely why the paper needs its *contiguous-region* mechanism in function-as-leaf mode. Confirmed: regions are load-bearing, not a nicety.
- **Heuristic stage assignment is where the errors live.** My keyword rules mis-assigned `_run_agent_loop` to "Prompt & Template Assembly" because its parameter is named `original_instruction` (fixed by excluding signatures from matching); `Terminus2.setup` stayed unmapped. This validates the paper's design instinct: Phase II needs semantic judgment (their proposer–reviewer loop), not rules — but it also means **the handbook's quality is only as good as that LLM loop, and the paper never ablates structuring quality** (e.g. vs. cheap embeddings/keyword assignment or an ablated no-review loop).
- **Static analysis ceiling**: my resolver reaches 195/~257 of the paper's edges. The missing quarter hides in local-variable aliases, callbacks, container-stored callables, inherited methods (`BaseAgent`), and dynamic dispatch — `_query_llm` both uses `getattr` and is wrapped in `@retry` (tenacity), so its real call/edge semantics are dynamic. The paper's "unresolved calls are logged, not guessed" is the right call, but at harness scale the audit log will be large.
- **Registers are shallow heuristics**: `self._*` captures instance state but misses module-level globals, config dicts, and state passed through parameters — several of Terminus-2's real couplings travel through `chat`/parameter passing, invisible to the register view.

## 4. Evaluation design — earned caveats

1. **Judges**: three LLMs score 0–5 per dimension (weighted 0.5/0.25/0.25, rescaled to 0–100), win if |Δ| ≥ δ=3. Positives: judges build a leakage-safe answer key from pristine source *before* seeing plans (D.3.1), and the direction is consistent across all three judges. Negatives: all judges are unreleased models with unknown calibration; δ=3 on a 100-scale derived from 0.5-step scores is noise-level; no inter-judge agreement stats, no human spot-check, no score distributions — win rates without variance.
2. **Reference-plan circularity**: Recall/Precision/F1 compare DeepSeek-V4-Pro's plans against reference plans *written by Opus 4.8 and GPT-5.5* — i.e. "does the weaker planner agree with strong models", with the strong models assumed correct. These reference plans are themselves unverified LLM outputs on the same repos; if a reference plan misses a cold-path site (exactly the SH category), the handbook arm is penalized for being *more* complete. Codex GPT-5.5-reference F1 gain is only +5.0 (vs +15.2 vs Opus) — reference-model disagreement of this size is itself evidence that "ground truth" is soft. Wrong↓ metric inherits the same caveat.
3. **Baseline arm**: same planner with read-only repo tools (read, in-file search, dir listing). Fair for isolating the handbook's effect, but it is a *strawman-of-omission*: no repo-map (Aider-style), no RAG/embedding retrieval, no Agentless-style hierarchical localization baseline. The +10–19pp win rates are "handbook vs. naked exploration", not "handbook vs. best alternative index". Terminus-2 (6 files, ~4k LOC) fits largely in a modern context window, so the Terminus-2 gains (+18.9pp) are the surprising claim and the least convincing without stronger baselines.
4. **Power**: 30 requests/harness × 3 judges; slices (Q/CF/SH × E/M/H) leave ~10 requests per cell — Figure 5's per-slice gains (3.7–33.3pp) are well inside binomial noise per cell (±~18pp at n=10, 95% CI). Aggregate direction is consistent everywhere, which mitigates but doesn't quantify.
5. **Synthetic requests**: the 60 requests are author-constructed ("built from the same behavioral intents but phrased against each harness's own code", App. C.2), not sampled from real issue/PR history, and difficulty labels are hand-assigned. That controls for confounds but risks being tuned to what the handbook is good at (scattered/mirror/cold-path sites); no held-out natural request distribution is tested.
6. **Evaluator-independent concern**: planner, one judge, and reference plans all share DeepSeek-V4-Pro lineage (planner = judge #3). They do note the direction holds per-judge, which is the right check.
7. **NexAU verdict**: the planner framework is public (`github.com/nex-agi/NexAU`, HTTP 200 verified) and Harbor/Terminus-2 is public, so the *harnesses* are obtainable; but the planner model (DeepSeek-V4-Pro) and all three judges are unreleased, so the pipeline is only partially rebuildable.

## 5. Neighbors and honest novelty assessment

- **Aider repo-map**: tree-sitter symbols + PageRank over references, computed per prompt, organized by file/symbol. No behavior layer, no stages, no state registers, no persistence/resync invariant. Handbook = repo-map re-organized around execution stages + maintained as a living artifact.
- **Agentless / SWE-bench localization**: already does coarse-to-fine (file → class/function → edit block) — BGPD's *navigation pattern* is not new. What is new: the persistent behavior-centric intermediate artifact with source-verified locators, the explicit cross-stage **state-register view** (the piece that empirically catches mirrored/cold-path sites, cf. our probe), and the **resync invariant** (frozen-not-guessed locators, fingerprint-based roll-forward, diff-triggered scoped updates) making the artifact trustworthy across an evolution loop.
- **Code as Agent Harness (2605.18747)**, which they cite: defines the harness as the executable stateful layer — Handbook is the *reading/editing interface* to that abstraction. Complementary, correctly positioned.
- **For the Heuristic Learning thread**: the paper is the missing maintenance layer for HL-style executable-policy systems. An HS whose policy is code evolves via repeated coding-agent edits; the bottleneck the paper names (behavior localization before each edit) is exactly the HL update step's cost center, and the resynced handbook is a plausible "shared behavioral memory" between successive update iterations (their stated next step is harness self-evolving). The probe shows the valuable 20% (state registers + source-verified locators) is cheap to build; the expensive 80% is LLM stage-structuring quality, which the paper doesn't ablate.

## 6. Evidence index

- `code/probe_handbook.py` — the probe (stdlib only).
- `code/probe-output.txt` — full console output of the run above.
- `code/handbook-out/` — generated mini-handbook (overview/index/registers + 10 stage files).
- `../../assets/fig1-handbook-structure.png` — paper Figure 1 (L1–L3 representation; verified PNG 996×561, arXiv `x1.png`).
- `../../assets/fig2-construction-pipeline.png` — paper Figure 2 (three-phase construction; verified PNG 996×254, arXiv `x2.png`).
- Scratch clone: `code/harbor` (workspace-root, git-ignored), commit `d3e606d9f7d1e111bb22d3d820ebed03ec300eb3`.
