# Addendum: What Does "Reasoning" Actually Mean in Embodied-R1.5?

**Date:** 2026-06-11  
**Focus:** A second-pass investigation into whether the paper's claims about "reasoning," "adaptive thinking," and "System 2 / System 1" are supported by the text and the code.

---

## Core Insight

**The RL training objective in Embodied-R1.5 provides no mechanism to improve reasoning quality, and no evidence is presented that it does so.** The reward function evaluates only the final answer inside `<answer>` tags and discards the reasoning text entirely. The "adaptive thinking" the paper celebrates is therefore not a learned property of multi-task RL; it is at best an SFT imitation artifact, at worst a spurious correlation between task complexity and response length, and — most critically — **explicitly prompted rather than emergent.** The authors' own code contradicts their prose on this point.

---

## 1. The "Adaptive Thinking" Claim Is Undermined by the Training Code

### What the paper says

In §4.2.1, the authors write:

> "We only constrain the output format (requiring answers within `<answer></answer>` tags) **without forcing explicit reasoning chains**. Since the RL reward evaluates only the final answer quality without rewarding intermediate reasoning, the model **naturally learns to allocate computation on demand**..."

This is presented as an emergent behavior discovered by RL.

### What the code says

Every RL training prompt is wrapped by a hardcoded template in `EasyR1/verl/utils/dataset.py` (lines 35–41):

```python
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)
```

This is not "only constraining the output format." It is an **explicit reasoning instruction injected into every prompt at training time.** The model is not "naturally learning" to think; it is being told to think across hundreds of thousands of RL rollouts. Longer traces on complex tasks are at least partially explained by the fact that complex questions elicit more verbalized reasoning when a model is explicitly instructed to reason — not by a learned computation-allocation strategy.

### The missing evidence

The paper offers **zero quantitative evidence** for adaptive thinking: no reasoning token distributions by task type, no ablation removing the thinking instruction, and no correlation between reasoning length and reward. The claim is an after-the-fact observation dressed up as an RL discovery.

---

## 2. The RL Reward Cannot Reward Reasoning — Only Answers

The reward implementation in `embodied_reward.py` makes this explicit. The `extract_answer()` function (lines 282–286) strips everything outside `<answer>` tags, and `accuracy_reward()` (lines 906–980) evaluates **only** the extracted answer string. For every task type — multiple choice, numerical, math, spatial grounding, point, trace, open-ended — the reasoning text is **completely ignored**.

This means: **during RL, the model receives gradient signals only for final-answer correctness.** A beautifully structured reasoning chain that concludes with a slightly wrong coordinate gets the same low reward as gibberish followed by the wrong coordinate. Nonsensical reasoning followed by the correct answer receives full reward. The optimizer cannot distinguish good reasoning from bad reasoning; it can only select for answers that happen to be correct.

The paper is honest about the setup, but draws an unsupported conclusion. An outcome-only reward cannot teach a model to reason better; it can only select for text that looks like reasoning while satisfying an answer constraint. The "structured thinking processes" in Appendix C.4 are therefore **not evidence that the model was trained to reason** — they are evidence that it was trained to produce reasoning-shaped text.

---

## 3. The System 2 / System 1 Framing Is a Metaphor, Not an Architecture

The paper describes the VLA extension as "a dual-system architecture (System 2 for reasoning, System 1 for action generation)" (§2.2). In reality, the architecture is a standard Qwen3-VL-8B backbone plus a DiT-B flow-matching action head attached to intermediate VLM hidden states. The action head generates continuous actions via cross-attention on VLM features, **not by parsing explicit reasoning tokens.** There is no architectural separation between "reasoning" and "action" subsystems in the compute graph. The "System 2 / System 1" language is a conceptual metaphor, not a description of the architecture.

Separately, **the paper does not ablate whether reasoning text improves VLA performance.** The strong VLA results (92.4% on SimplerEnv) demonstrate that a VLM with rich spatial-semantic representations provides good features for an action head — valuable, but not the same as showing that "reasoning drives action." Without an ablation that removes reasoning tokens and measures the impact, we cannot know whether the reasoning text is causally important or merely correlated with correct answers.

---

## 4. The Reflection Loop Is Retry, Not Reasoning

The released `inference/reflection.py` implements a 3-round planning→reflection loop:

```python
def plan_with_reflection(client, case, max_rounds=3):
    for round_idx in range(1, max_rounds + 1):
        plan_text = inference(client, planning_case)
        reflection_text = inference(client, reflection_case)
        if "correct" in reflection_text.lower() and "incorrect" not in reflection_text.lower():
            return plan_text
        feedback_text = reflection_text
    return plan_text
```

The reflection "verdict" is crude string matching with no parsing of reflection content, no state machine, no memory buffer, and no grounding in execution outcomes. This is not a reasoning system; it is a **sampling-and-retry mechanism** with a lightweight filter. The sophisticated PGC framework described in §5 is not present in the released repository.

---

## 5. The SFT Stage Does Not Prime Reasoning Either

The SFT config (`scripts/train/sft_config.yaml`) uses template `qwen3_vl_nothink` — a LLaMA-Factory template that disables the native reasoning/thinking tokens of Qwen3. The explicit "think first" instruction is introduced **only at the RL stage** via `QUESTION_TEMPLATE`.

This means the "reasoning" behavior is not an inherited property of the base model fine-tuned on embodied data. It is **elicited by explicit prompting during RL** on a model whose SFT stage did not emphasize reasoning. That the model produces coherent reasoning traces at all is a testament to the general capabilities of the Qwen3-VL backbone, not to a deliberate reasoning curriculum.

---

## 6. What Is Demonstrated vs. What Is Claimed

| Claim | Evidence | Verdict |
|---|---|---|
| "Adaptive thinking" emerges from RL | No quantitative evidence; contradicted by explicit prompt injection | **Unsupported** |
| RL optimizes reasoning quality | Reward discards reasoning text; only answers are scored | **False** |
| System 2 / System 1 dual-system | Standard VLM + action head; no ablation on reasoning text | **Metaphor, not architecture** |
| Closed-loop PGC autonomy | Only basic retry demo released; no harness code | **Unverifiable** |
| Strong benchmark performance | Extensive, reproducible via EmbodiedEvalKit | **Genuine** |

The authors deserve credit for the genuine findings: benchmark results, data engineering, MBPO normalization, and open release. But the "reasoning" framing oversells what the experiments validate. The paper measures **benchmark accuracy**, not **reasoning quality**. It demonstrates that an 8B VLM can excel at embodied VQA, pointing, and planning, and that these capabilities transfer to VLA training. It does **not** demonstrate that the model reasons about physical tasks in a verifiable, causal sense — and its own training code shows why it could not have learned to do so through RL.

To substantiate the reasoning claim, the necessary experiments are straightforward: ablate the thinking instruction, measure reasoning token counts and their correlation with performance, evaluate reasoning chains independently of final answers, and show that the action head degrades when reasoning tokens are removed. None of these appear in the paper. Until they do, "reasoning" in Embodied-R1.5 should be read as **pattern-matched text generation that correlates with correct answers** — impressive in breadth, but not evidence of genuine cognitive computation allocation.
