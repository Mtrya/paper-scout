# Deep-dive thread: Guava — An Effective and Universal Harness for Embodied Manipulation

**Run:** `2026-06-18-guava-kairos-omniagent`  
**Paper:** arXiv:2606.18363  
**Project page:** https://guava-harness.github.io  
**Thread directory:** `runs/2026-06-18-guava-kairos-omniagent/guava-2606.18363/`

---

## 1. Research question chased

The paper asks: *what makes an effective harness for embodied manipulation, and can that harness transfer embodied capabilities to small open-source VLMs?*  
I focused on the first half in a runnable form: **Are Guava's three claimed ingredients—iterative ReAct loops, semantic action abstractions, and multimodal observations—actually necessary and sufficient for robust toy manipulation?**  I also inspected what the data-distillation pipeline claims to do and what cannot be checked because the official code is not yet public.

---

## 2. What the paper claims

- **Three design principles** for an embodied harness (Figure 1):
  1. **Iterative perception–reasoning–action loops** (ReAct-style) instead of one-shot plans.
  2. **Semantic action abstractions** (`grasp(object)`, `align(...)`, `move(x,y,z)`, etc.) so the VLM plans at the object/task level and leaves geometry to low-level controllers.
  3. **Multimodal observations**: an RGB image plus a compact symbolic state (object positions, gripper pose, last-action result).
- **Distillation pipeline**: A 4B Qwen3.5-4B model is trained on **1,934 simulation trajectories** (62% success, 38% recovery) collected from a frontier VLM (GPT-5.4) acting under the Guava harness. Training is two-stage: supervised fine-tuning, then **GRPO only on the two hardest long-horizon tasks** with sparse success rewards.
- **Results**: Guava-Agent-4B matches or beats GPT-5.4 and a concurrent CaP-Agent0 baseline in both simulation and zero-shot real-world Franka experiments.

Key figures captured for the report:

- `runs/2026-06-18-guava-kairos-omniagent/assets/guava-overview-figure1.jpg` — the three ingredients and real-world zero-shot deployment grid.
- `runs/2026-06-18-guava-kairos-omniagent/assets/guava-react-ablation-figure2-left.jpg` — ReAct vs. single-turn planning in the paper's Robosuite ablation.
- `runs/2026-06-18-guava-kairos-omniagent/assets/guava-data-engine-figure3.jpg` — the frontier-VLM data engine and training-data distribution.

---

## 3. External signals gathered

### Project page & code
- The project page (https://guava-harness.github.io) repeats the paper narrative and shows videos/galleries, but **does not host code or a GitHub repository**. No public repo was found via web search on 2026-06-18. The paper itself lists implementation details only in the appendix (tool definitions, prompts, hyperparameters).
- This means the exact prompt templates, the SAM3/segmentation-to-grasp pipeline, the Robosuite task definitions, and the GRPO training code are currently **not independently verifiable**.

### Related harness/code-as-policy work
- **Code-as-Policies (Liang et al., 2022)**: Google Research released the original Colabs and demo; the paper notes CaP uses one-shot hierarchical code generation, which Guava argues is brittle for long-horizon manipulation because it cannot react to execution failure.
- **CaP-X / CaP-Agent0 (Fu et al., 2026)**: A public repo exists at https://github.com/capgym/cap-x. It is a much larger open benchmark (CaP-Gym, CaP-Bench, CaP-RL) and CaP-Agent0 also uses multi-turn interaction. Guava compares against it as a baseline; CaP-Agent0 is the closest concurrent system with available code.
- **Maestro (Shi et al., 2025)**: Cited by Guava as another concurrent VLM-orchestration harness. A matching public code repository was **not located quickly**; only the NeurIPS workshop paper reference exists.

### Paper's own caveats
- The authors note they cannot correct **tool-level errors** (bad SAM3 segmentation, invalid grasp proposals), only detect and retry.
- Current setup uses a **single fixed camera**, so occlusion can break perception.
- Spatial reasoning tasks (pushing, arranging by order) remain the hardest and show VLM shortcuts.

---

## 4. What the probe found

I built a minimal, pure-Python probe in `code/guava_probe.py` that instantiates a Guava-style harness on a toy tabletop scenario: **place the red cube in the basket**. The agent is a hand-written ReAct policy (a stand-in for the VLM) that calls semantic tools and receives both a text state and an ASCII top-down visual observation.

### The probe demonstrates
1. **Closed-loop recovery from grasp failure.** The first `grasp(red_cube)` is allowed to fail stochastically. The agent sees the closed-empty gripper, releases, re-aligns, and retries—exactly the failure-recovery behavior the paper highlights in Figure 12.
2. **Semantic abstractions reduce geometric burden.** A low-level ablation that must issue raw `(x,y,z)` moves repeatedly misses the object because it lacks `align(...)` and clearance reasoning; it fails 20/20 trials.
3. **One-shot planning is brittle.** An open-loop baseline that executes a fixed sequence without observing execution feedback fails whenever the first grasp slips.
4. **Multimodal observations matter for disambiguation.** A text-only ablation strips color labels; the agent cannot tell red cube from blue cube and satisfies the wrong goal (placing the blue cube), so task success is 0/20.

### Representative output

```text
Condition: Guava-style: ReAct + semantic tools + multimodal obs
Success: True  |  Steps used: 7
...
Think: Recovery: last grasp failed, opening gripper before retry.
Tool:  release()
...
Think: Gripper is aligned and open; grasp target.
Tool:  grasp(red_cube)
Result: grasped red_cube
...

Monte-Carlo summary (20 trials, seed 100-119)
  ReAct + semantic + multimodal       success 20/20  avg_steps 6.0
  One-shot                            success 12/20  avg_steps 4.0
  Low-level moves                     success  0/20  avg_steps 20.0
  Text-only                           success  0/20  avg_steps 6.0
```

A full run is saved in `code/sample_output.txt`.

### What this is *not*
The probe is not a reproduction of Guava-Agent-4B. It does not train a 4B model, does not use real vision, and does not run Robosuite. It is a **mechanistic sanity check** that the harness design choices the paper identifies as important are individually load-bearing even in a tiny environment.

---

## 5. How to run the probe

No dependencies beyond Python 3.8+ are required.

```bash
cd runs/2026-06-18-guava-kairos-omniagent/guava-2606.18363/code
python3 guava_probe.py
```

The script prints a verbose trace of the Guava-style agent and short summaries of three ablations, followed by a 20-trial Monte-Carlo comparison.

---

## 6. Limitations & blockers

- **No official code.** The project page has no repo; the data engine, training code, prompts, and real-robot deployment scripts cannot be inspected or rerun. The probe substitutes a toy environment and a hand-coded policy.
- **Teacher model details are thin.** The paper says GPT-5.4 generated the 1,934 trajectories, but the exact system prompt, scene-randomization range, and perturbation set are described only at a high level.
- **GRPO recipe is unusually targeted.** RL is applied only to the two hardest long-horizon tasks. This is a reasonable efficiency choice, but without code it is hard to tell whether the reported gains generalize or overfit to those two tasks.
- **Vision is simulated.** The probe's "visual" observation is an ASCII grid. It captures the *concept* of multimodal grounding but not the difficulty of real RGB-D perception or SAM3 segmentation.
- **Comparison with CaP-X is incomplete.** CaP-X has public code; a head-to-head run on Guava's exact tasks was not attempted because Guava's task configs are not released.

---

## 7. Key takeaways for the report

1. **Guava's contribution is primarily harness design, not a new model.** The paper makes a strong case that *how* you wrap a VLM (ReAct loop + semantic tools + multimodal obs) matters as much as the model size. The probe confirms each ingredient is individually load-bearing.
2. **The data engine is the most under-documented but most important part.** Fewer than 2K trajectories producing a competitive 4B agent is only credible if the frontier teacher, perturbations, and recovery curation are as careful as described. Without code, this remains a claim.
3. **Official code is not yet available.** Report this plainly. The project page and paper give enough detail to understand the method, but not to reproduce it.
4. **CaP-X is the nearest open comparator.** It already provides a public benchmark, multi-turn agents, and GRPO training. Guava differentiates itself by distilling into a small VLM *under the same harness* rather than treating the harness as a test-time-only scaffold, but the overlap is large enough that the two should be discussed together.
5. **Real-world zero-shot transfer is the headline result.** If reproducible, the fact that a 4B model trained only in simulation deploys directly on a Franka is the strongest signal. The probe cannot validate that; it can only show that the harness mechanism is plausible.

---

*Prepared by the Paper Scout deep-dive subagent for the Guava thread, 2026-06-18.*
