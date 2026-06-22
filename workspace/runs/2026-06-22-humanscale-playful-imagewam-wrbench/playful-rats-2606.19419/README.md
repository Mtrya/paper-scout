# Deep-dive thread: Playful Agentic Robot Learning (arXiv 2606.19419)

**Paper:** *Playful Agentic Robot Learning* — Junyi Zhang et al.  
**Project page:** https://Playful-RATs.github.io  
**Public code:** https://github.com/Playful-RATs/rats (cloned to `code/rats` for inspection)

---

## Research question

Can a Code-as-Policy robot agent acquire reusable manipulation skills *before* it is given downstream tasks, by treating self-directed play as a deliberate skill-learning stage rather than random exploration? And does the resulting skill library actually help later task solving in a way that is cheaper/better than simply spending the same compute on more test-time retries?

---

## What the paper claims

The paper introduces **RATS** (Robotics Agent Teams), a multi-agent Code-as-Policy system that runs a play-time loop:

1. **Propose** a task — chosen to be novel yet learnable.
2. **Plan, write, execute** robot code.
3. **Verify** intermediate progress and **diagnose** failures.
4. **Retry** with dense feedback and **distill** successes into a reusable code skill library.

At test time the learned library is frozen and retrieved into a standard Code-as-Policy agent (CaP-Agent0) or used by the full RATS execution team. Key empirical claims:

- LIBERO-PRO: +20.6 percentage points over CaP-Agent0 (23.2% → 43.8%).
- MolmoSpaces: +17.0 pp over CaP-Agent0 (21.0% → 38.0%).
- Cross-environment transfer: LIBERO-learned skills plugged into CaP-Agent0 improve RoboSuite by +8.9 pp, and sim-learned skills improve real-world Franka tasks by +8.8 pp, with no fine-tuning.
- Curiosity matters: random play under the same 50-iteration budget gives only small gains; a compute-matched baseline that spends the tokens reactively at test time improves only modestly (23.2% → 26.0%) versus proactive play (32.3%).

---

## External signals

### Project page and public repository

The project page links to a public GitHub repository. I cloned it to `code/rats` and inspected the implementation that underpins the paper's claims:

- `rats/agents/curiosity_scoring.py` — analytical Goldilocks score (`novelty * competence_frontier`), Wilson-lower-bound reliability, retry-bonus and recent-failure penalty.
- `rats/agents/task_proposer.py` — candidate generation, curriculum gating, catalog/pick-reliability vetoes, MolmoSpaces grounding.
- `rats/agents/planner.py`, `policy_writer.py`, `failure_diagnoser.py`, `per_step_verifier.py`, `verifier.py`, `memory_curator.py`, `skill_proposer.py` — the WEVD loop.
- `skill_library/library.py` — skill lifecycle (experimental → verified → deprecated), dependency closure, static AST validation, tier-gated retrieval.
- `skill_library/*.json` — seed primitives and learned skill snapshots (e.g., `molmospaces_formula_play.json`, `libero_formula_warmup5_iter050.json`).

The repository is real, runnable, and includes setup instructions for LIBERO-PRO, MolmoSpaces, RoboSuite transfer, and real-Frankia transfer. The code I preserved in this thread is self-contained and does not depend on the heavy repo dependencies.

### Closest neighbors and what is genuinely new

- **CaP-X / Code-as-Policy agents** are task-driven: the agent receives a language instruction and retries until it solves it. RATS adds an *upstream play phase* with autonomous task proposal, intrinsic motivation, and a persistent skill library. The paper's ablations show that simply giving CaP-Agent0 more test-time retries (compute-matched to 15 turns) gains only +2.8 pp, while RATS skills gain +9.1 pp, so the value is not just "more inference."
- **Voyager** also builds an automatic curriculum and code skill library, but in open-ended Minecraft. RATS transfers the same idea to physically grounded robot Code-as-Policy, with per-step verifiers and a failure-diagnosis loop tailored to manipulation.
- **Goal babbling / curiosity-driven developmental robotics** (Oudeyer et al., Baranes & Oudeyer) selects goals in sensorimotor or feature spaces. RATS lifts that principle to *language-programmed* goals and executes them as code policies, then distills successes into named, reusable functions rather than collecting trajectory data.

The novelty is therefore the combination: **autonomous language/task proposal at the competence frontier + structured multi-agent execution feedback + code skill lifecycle + cross-environment transfer without model fine-tuning.**

---

## What the probe demonstrates

I wrote two small, runnable probes in `code/`:

### `goldilocks_probe.py`

Reimplements the analytical task-selection score from `rats/agents/curiosity_scoring.py` and the paper's Sec. 3.2:

- **Novelty** `N(τ)`: mean of `1 / sqrt(N(o,s) + 1)` over object-skill pairs.
- **Competence** `r̄(τ)`: mean Wilson-lower-bound reliability of required skills.
- **Frontier** `F(τ) = 4 · r̄ · (1 − r̄)`: the Goldilocks parabola peaking at `r̄ ≈ 0.5`.
- **Score** `N(τ) · F(τ)`.

It first reproduces the Table 7 candidate trace from MolmoSpaces iteration 15 and confirms that the **tissue-box lift** wins because it sits on the competence frontier (`r̄ ≈ 0.19`, `F ≈ 0.62`) rather than on mastered Close/Open tasks (`r̄ ≈ 0.9`, `F ≈ 0.36`) or unsupported Place-in tasks (`r̄ ≈ 0.05`, `F ≈ 0.19`).

It then runs a deterministic 50-iteration abstract simulation starting from the primitive/learned-skill mix reported in Table 5. The simulation shows how the score evolves: tasks that are repeatedly practiced lose novelty and, if mastered, move off the frontier, so the selector migrates toward underexplored object-skill combinations. The script writes `goldilocks_sim_log.json` for inspection.

Run it:

```bash
python runs/2026-06-22-humanscale-playful-imagewam-wrbench/playful-rats-2606.19419/code/goldilocks_probe.py
```

### `wevd_minimal.py`

Implements a minimal Write-Execute-Verify-Diagnose loop for the MolmoSpaces drawer-opening example from Appendix C.2 / Figure 9. The mock environment and deterministic agents demonstrate:

- A first attempt fails because the grasp is not axis-aligned.
- The **per-step verifier** localizes the failure to step 3.
- The **failure diagnoser** emits a concrete fix: call the learned helpers `get_axis_aligned_pull_direction` and `select_grasp_for_pulling`, then align the grasp.
- The second attempt succeeds, showing how dense feedback converts a sparse failure into a targeted retry.

Run it:

```bash
python runs/2026-06-22-humanscale-playful-imagewam-wrbench/playful-rats-2606.19419/code/wevd_minimal.py
```

---

## Takeaway for the report

RATS makes a persuasive case that **play is a practical pre-training stage for embodied coding agents**. The key mechanism is not open-ended novelty maximization; it is a curiosity objective that actively targets the competence frontier, backed by a multi-agent execution loop that turns failures into dense, code-level feedback. The public repository confirms that the formulas, lifecycle rules, and agent prompts in the paper are implemented faithfully.

The most important result for the report is the **compute-matched ablation**: play-time skill acquisition beats spending the same tokens on extra test-time retries. That suggests the gains come from *structure* (named, reusable skills) rather than *scale* (more inference). The cross-environment and real-world transfer numbers are still small in absolute terms, but they are positive with no fine-tuning, which supports the paper's bet on a plug-and-play code skill library.

What remains unresolved: the system is expensive (≈30M tokens for 50 play iterations), relies heavily on VLM verification, and the real-world evaluation is small. It also inherits the brittleness of Code-as-Policy retrieval — the paper notes cases where retrieved skills hurt performance. So the practical question is not whether play helps, but whether the cost and verification burden are worth it relative to collecting a modest amount of human demonstration or running targeted curriculum design.
