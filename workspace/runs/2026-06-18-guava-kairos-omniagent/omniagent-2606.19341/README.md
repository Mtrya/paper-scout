# OmniAgent Deep-Dive Thread

**Paper:** Native Active Perception as Reasoning for Omni-Modal Understanding  
**arXiv:** 2606.19341 | **Repo:** https://github.com/harryhsing/OmniAgent  
**Thread:** `runs/2026-06-18-guava-kairos-omniagent/omniagent-2606.19341/`

## Research Question

How does OmniAgent turn long-video understanding into a POMDP with strict information distillation, and is the TAURA objective in the released code actually implemented as turn-level entropy rescaling?

## Findings

### 1. Video understanding is framed as a POMDP with a strict transient/persistent split

The agent maintains two pieces of state:

- **Transient percept** $\mathcal{E}_k$: raw media returned by the environment (frames, audio, clip).
- **Persistent memory** $\mathcal{M}_k$: the textual reasoning trace.

At each turn the policy samples an Observation–Thought–Action triplet conditioned on $(\mathcal{M}_{k-1}, \mathcal{E}_{k-1})$, distills $\mathcal{E}_{k-1}$ into a textual $O_k$, and then the environment *purges* the old raw percept before fetching the next one. This is the core decoupling: context cost is meant to scale with the reasoning trace, not with video duration.

The prompt template in the repo makes this explicit:

> "Media Persistence: Once media is returned, it becomes a TEXT PLACEHOLDER in the next turn."  
> "Your `observation` must be an exhaustive, high-fidelity log. Once media is omitted, you will 'forget' any detail not recorded here."

This matches Algorithm 1 in the paper and is enforced in `agent_system/environments/env_package/video_env.py` (curated in `code/curated/agent_system-video_env.py`).

### 2. The OTA loop is implemented as an asynchronous pool rollout

`agent_system/multi_turn_rollout/rollout_loop.py` (curated) implements `pool_async_multi_turn_loop`: a dynamic pool of environments where finished trajectories are evicted immediately so the GPU batch never waits for stragglers. Each loop iteration:

1. Pre-processes the current observation history (with media) into model inputs.
2. Generates one OTA response per active env.
3. Calls `envs.step_selected()` to execute the action and return the next raw percept.
4. Stores the (prompt + response) pair for training; the environment feedback is **not** appended to the training tokens (the "farewell message" is intentionally dropped).

The action grammar is exactly the paper's four actions:

```json
{"type": "get_frames", "start": float, "end": float, "num": int}
{"type": "get_audio",  "start": float, "end": float}
{"type": "get_clip",   "start": float, "end": float}
{"type": "answer",     "content": "string"}
```

### 3. TAURA is in the code and matches the paper's equation

The repo does **not** grep for the string "TAURA"; the mechanism is controlled by config flags `entropy_weight` / `entropy_soft_weight`. The TAURA variant is `examples/omniagent_train/train_TAURA.sh` (curated), which sets:

```bash
algorithm.adv_estimator=grpo
+algorithm.entropy_weight=true
+algorithm.entropy_soft_weight=false
+algorithm.entropy_top_ratio=null
+algorithm.entropy_seq_top_ratio=null
```

The actual rescaling lives in `verl/trainer/ppo/core_algos.py`:

```python
def get_entropy_weights(entropy, response_mask, group_indices, eps=1e-8):
    # h_gi = mean entropy of this response (i.e., this turn)
    h_gi = turn_entropy_sum / (turn_token_count + eps)
    # group mean over all turns of all trajectories for the same query
    group_mean = group_h_sum / (group_traj_count + eps)
    w_gi = h_gi / (group_mean[g] + eps)
    return final_w * mask
```

And it is applied in `verl/trainer/ppo/ray_trainer.py`:

```python
if need_entropy_weight:
    w = get_entropy_weights(entropy=entropys, response_mask=response_mask,
                            group_indices=batch.non_tensor_batch['uid'])
    batch.batch["entropy_weights"] = w.detach()

# later, in compute_advantage():
if "entropy_weights" in data.batch:
    data.batch["advantages"] = data.batch["advantages"] * w
```

Because each training sample is one **turn** of one trajectory, `uid` is the query group and `traj_uid` is the trajectory ID, `get_entropy_weights` computes exactly the paper's Equation 4:

$$
\hat A_{i,k} = A_i \cdot \frac{H_{i,k}}{\frac{1}{N_{\mathcal G}} \sum_j \sum_m H_{j,m}}
$$

Vanilla GRPO (the baseline script `train_GRPO.sh`) differs only by `entropy_weight=false`, which leaves the trajectory-level advantage broadcast uniformly to every token.

### 4. Surprises / caveats from reading the code

- **Name mismatch:** the released code never calls it TAURA; it is just "entropy weight". The paper's name appears only in README/docs.
- **No KL / no entropy regularizer:** both `train_TAURA.sh` and `train_GRPO.sh` set `entropy_coeff=0.000` and `use_kl_loss=False`, consistent with the paper's RL details.
- **Retry-on-format-error mode:** the rollout loop supports a `RETRY_ON_FORMAT_ERROR` path where invalid JSON actions are kept in history but physically deleted from the training batch, letting the model learn recovery without contaminating gradients.
- **TITO (token-in/token-out) assembly:** for Qwen2.5-Omni the code avoids re-tokenizing the full chat history each turn by reusing token segments generated by vLLM. This is an engineering necessity for long trajectories, not a method claim.

## Runnable Probe

`code/omniagent_probe.py` is a self-contained, CPU-only demonstration of the two core ideas.

### Part A: tiny active-perception POMDP

A 1-D hidden-target search where an agent emits Observation–Thought–Action JSON, receives noisy signal samples, narrows its belief, and finally answers. Raw observations are not carried forward; only the agent's own belief text is. Run:

```bash
cd code
python omniagent_probe.py --part pomdp --seed 42
```

Typical output:

```
Turn 1  Action: look in [0.0, 100.0]
Turn 2  Action: look in [46.7, 86.7]
Turn 3  Action: look in [52.0, 68.0]
...
Turn 6  Action: answer 62.0
Final guess: 62.0 | target: 63.9 | reward: 0.903
Mean reward over 5 episodes: 0.959 (std=0.033)
```

### Part B: TAURA advantage rescaling vs. vanilla GRPO

Synthesises turn-level entropies with one high-entropy "discovery" turn per trajectory and compares the per-token advantages. Run:

```bash
python omniagent_probe.py --part taura --seed 42
```

Typical result:

```
High-entropy turns: 15
Low-entropy turns : 15
High-entropy |adv| GRPO  : 0.947
High-entropy |adv| TAURA : 1.237
Low-entropy |adv| GRPO   : 0.832
Low-entropy |adv| TAURA  : 0.554

TAURA boosts high-entropy turn magnitude by ~1.31x
```

This confirms the intended asymmetry: high-entropy discovery turns receive stronger gradients, while routine low-entropy turns are down-weighted.

## Blockers for Local Full-Model Demo

The Gradio demo `demo/omniagent_demo_pro.py` can in principle run locally, but requires:

1. A GPU with enough VRAM for Qwen2.5-Omni-7B under vLLM (the paper uses A100s).
2. `ffmpeg` installed and available on PATH.
3. A checkout of the released SFT or RL checkpoint (`harryhsing/OmniAgent-SFT-7B` or `OmniAgent-RL-7B`).
4. The full repo dependency stack (vLLM, Ray, transformers, qwen-omni-utils, etc.).

This workspace has no GPU and the checkpoints are not cached, so we did not run the full model. The probe above isolates the method's mechanics without the model.

## Curated Artifacts

- `code/omniagent_probe.py` — runnable probe
- `code/curated/agent_system-rollout_loop.py` — async OTA rollout loop
- `code/curated/agent_system-video_env.py` — environment with action grammar and media purging
- `code/curated/verl-core_algos.py` — GRPO/TAURA advantage and `get_entropy_weights`
- `code/curated/verl-ray_trainer.py` — trainer that applies entropy weights
- `code/curated/train_TAURA.sh` and `train_GRPO.sh` — launch configs
- `../assets/omniagent-figure1-ota-framework.jpg` — Figure 1 from the paper
- `../assets/omniagent-figure2-test-time-scaling.jpg` — Figure 2
- `../assets/omniagent-figure3-accuracy-vs-frames.jpg` — Figure 3
- `../assets/omniagent-figure5-entropy-fork.jpg` — Figure 5(b), entropy spike at fork step

## Report Takeaways

1. **Active perception as reasoning is the central move.** OmniAgent is not a tool-using wrapper around a passive VLM; the same model generates observations, reasoning, and media requests, and raw media is deliberately discarded each turn. This is what lets a 7B model beat a 72B passive model on LVBench while using ~73% fewer frames.
2. **TAURA is implemented and matches the paper.** The released code rescales turn-level advantages by per-turn mean token entropy, with the group normalization taken over all turns of all trajectories for the same query. The only difference from the paper's notation is the internal flag name.
3. **Test-time scaling is emergent and bounded.** The agent takes more turns when the query is harder, but average turns saturate around 11.7 even when the horizon is 52, suggesting the policy learns to stop once evidence is sufficient rather than burning budget.
4. **Practical insight for the report:** the real win is not just fewer frames, but *interpretable, query-conditional search*. The OTA trace makes the agent's evidentiary chain inspectable, which is a meaningful contrast to dense passive models and to black-box reasoning models.
