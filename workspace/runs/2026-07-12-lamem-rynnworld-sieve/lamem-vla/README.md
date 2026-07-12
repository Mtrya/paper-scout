# LaMem-VLA deep-dive

**Paper:** *LaMem-VLA: Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation* (arXiv 2607.07608)  
**Authors:** Hongyu Qu, Jianzhe Gao, Xiaobin Hu, Shaohuan Yang, Xinlei Yu, Rui Yan, Wenguan Wang, Xiangbo Shu, Shuicheng Yan  
**Official code:** https://github.com/quhongyu/LaMem-VLA — currently empty ("Codes will be released soon").

## Research question

Most VLAs predict actions from a single Markovian frame.  Memory-augmented variants usually keep history outside the model’s native token space: either as extra context-window frames, or as a retrieved bank fed to the policy head.  LaMem-VLA asks whether history can instead live *inside* the VLA’s continuous latent reasoning — as tokens that attend to, and are attended by, the current observation, language instruction, and action queries.

The answer proposed here is a four-stage latent-memory loop: **curator → seeker → condenser → weaver**.

## Core mechanism, in my own words

At every timestep the model sees the current RGB frame and a language instruction, encodes them into visual tokens **X**_t and instruction tokens **I**, and appends learnable action queries **Q**^action.  The trick is what happens before the VLM processes that sequence.

1. **Curator** maintains two *memory vaults*:
   - A **short-term vault** of key/value units.  The value is a compact set of visual tokens produced by an SE-bottleneck compressor from the current frame; the key is the mean of that value.  Each timestep a new unit is appended.
   - A **long-term vault** of action hidden states **H**^action (the output of the action-query slots).  These are stored directly, one unit per timestep, without an explicit value/key split.
   - When a vault exceeds capacity **L**, the most redundant *adjacent* pair is found by cosine similarity of their keys and merged by averaging.  This is a simple online compression rule that favors keeping diverse, temporally separated evidence.

2. **Seeker** builds a context-aware query from the current visual + instruction tokens, appends learnable query slots, and updates only those slots with a small masked transformer.  The mean-pooled query **q**_t is used to retrieve the top-**K** units from both vaults by cosine similarity.  The retrieval is discrete and non-differentiable.

3. **Condenser** takes the retrieved evidence (which can be long and redundant) and squeezes it into a *fixed* number of latent memory tokens: **L**_s short-term tokens and **L**_l long-term tokens.  It does this by prepending the context query **Q**_t and a set of learnable memory slots, then running a small transformer and reading off the final slots.  This decouples the variable size of the vault from the fixed-size VLM input.

4. **Weaver** injects the condensed memory into the VLM sequence itself:

   ```
   S_t = [M^short + b_s ; M^long + b_l ; X_t ; I ; Q^action]
   ```

   The `+`s are learnable source embeddings broadcast over each memory stream.  Because the memory tokens are ordinary tokens in the same embedding space, the action-query outputs become *memory-grounded action tokens*.  Those tokens then condition a diffusion action expert (DDIM, 10 steps in the paper) that denoises a 16-step 7-DoF action chunk.

The key architectural move is that memory is not side information; it is part of the same self-attention graph as perception, language, and action intent.

## What I built

Because the official repository is empty, I wrote a compact runnable PyTorch reconstruction of the token-flow described above: `code/lamem_probe.py`.  It is *not* a trained model and does not reproduce the reported numbers; it is a shape-and-mechanism probe that makes the method concrete.

The probe implements:
- SE-bottleneck compression of visual tokens into short-term memory values.
- Online key/value vaults with adjacent-pair merging when capacity is exceeded.
- A masked query builder that produces the retrieval vector **q**_t.
- Non-differentiable top-**K** retrieval from both vaults.
- Two independent memory formers that condense retrieved evidence into fixed-length memory slots.
- The weaver sequence `[M^short + b_s ; M^long + b_l ; X_t ; I ; Q^action]`.
- A tiny surrogate diffusion expert that emits a `(B, 16, 7)` action chunk.

Running it rolls out an 8-step episode and prints the growing then capped vault sizes and the action-token shapes, then checks that gradients flow all the way back through the memory-augmented sequence.

### How to rerun

A CPU-only PyTorch environment is enough.  From the workspace root:

```bash
# One-time: create an isolated environment (code/ is ignored by git)
uv venv -p 3.12 code/lamem-vla/.venv
uv pip install --python code/lamem-vla/.venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cpu

# Run the probe
code/lamem-vla/.venv/bin/python \
    runs/2026-07-12-lamem-rynnworld-sieve/lamem-vla/code/lamem_probe.py
```

If you prefer a standard `venv`/`pip` workflow, any Python 3.10–3.12 environment with `torch` installed will run the same file.

### What the probe shows

A typical run looks like this:

```text
Running LaMem-VLA probe on cpu

--- Episode roll-out ---
t=0: action_chunk (1, 16, 7), z_action (1, 4, 256), short_vault_size=1, long_vault_size=1
t=1: action_chunk (1, 16, 7), z_action (1, 4, 256), short_vault_size=2, long_vault_size=2
t=2: action_chunk (1, 16, 7), z_action (1, 4, 256), short_vault_size=3, long_vault_size=3
t=3: action_chunk (1, 16, 7), z_action (1, 4, 256), short_vault_size=4, long_vault_size=4
t=4: action_chunk (1, 16, 7), z_action (1, 4, 256), short_vault_size=4, long_vault_size=4
...
Dummy loss backward succeeded.
Gradient on action queries: True
Gradient on source embedding short: True
```

The vaults grow to capacity 4 and then stay bounded, confirming that the memory interface is constant-size regardless of episode length.  The backward pass reaching the source embeddings and action queries confirms that memory is part of the differentiable input sequence.

## Key claimed results

| Benchmark | Metric | LaMem-VLA | Closest memory baseline | Strong non-memory baseline |
|-----------|--------|-----------|------------------------|---------------------------|
| SimplerEnv-Bridge (WidowX) | avg. success | **73.9%** | MemoryVLA 71.9% | CogACT 57.3%, π₀ 69.2% |
| LIBERO (Franka) | avg. over 5 suites | **97.6%** | MemoryVLA 96.5% | CogACT 93.2%, π₀ 94.2% (first four suites) |
| LIBERO Long-10 | success | **95.8%** | MemoryVLA 93.4% | — |
| LIBERO Long-90 | success | **97.0%** | MemoryVLA 95.6% | — |

The ablations are arguably the most important numbers:
- Removing both memory streams drops SimplerEnv to 57.3% and LIBERO-90 to 92.1% (the CogACT-like baseline).
- Removing only short-term memory drops to 65.6% / 95.4%; removing only long-term drops to 64.6% / 94.8%.
- Feeding retrieved evidence as *policy-side conditioning* reaches 71.9% / 94.8% — better than no memory, but below the 73.9% / 97.0% achieved by prepending condensed latent memory tokens into the VLM sequence.
- Retrieval budget matters: **K = 8** is best; **K = 12** hurts slightly, suggesting compression quality matters.
- Memory-token budget matters: **(L_s, L_l) = (8, 4)** balances performance and context length.

## Findings, limitations, and open questions

**What is solid.**  The architectural idea is crisp and cleanly factorized.  The ablations isolate the design choices: dual streams > single stream, latent-native integration > policy-side conditioning, and condensing > raw retrieval.  The SimplerEnv-Bridge gain over CogACT (+16.6 points) is large enough that it is unlikely to be noise, and the LIBERO suite-wide sweep is consistent.

**What I could not verify from the paper text.**
- The exact SE-bottleneck architecture, the precise masked-attention patterns inside the query builder and memory formers, and the initialization of empty vaults are all in the appendix, which the arXiv HTML/markdown download did not include.  My probe fills these gaps with reasonable defaults and notes them as interpretive choices.
- The long-term vault stores action hidden states **H**^action, but at the very first timestep there is no prior action hidden state.  The paper does not say how the vault is bootstrapped (zero initialization?  first-frame action query output?).  My probe simply starts with empty vaults and lets the first step retrieve zero tensors.
- The retrieval operation is non-differentiable.  In practice the gradient will not flow through the top-**K** selection; only the condenser and query builder are learned.  This is consistent with the paper’s statement, but it means the model cannot learn *which* vault unit to retrieve in a fully end-to-end way.
- The reported results are simulation-only.  The authors note real-world experiments are planned for the next version.

**Potential concerns.**
- The adjacent-pair merging rule is O(L) per write and relies on cosine similarity of compressed keys.  It works as a cheap online clustering heuristic, but it is not guaranteed to preserve the most task-relevant evidence; a learned write policy could be stronger.
- LIBERO numbers are already very high (mid-90s for the best baselines), so the 1.1-point lead over MemoryVLA is meaningful but not dramatic.  SimplerEnv-Bridge is the more discriminating benchmark here.
- The comparison to π₀ is complicated by inputs: the starred π₀ results use proprioception and wrist camera, while LaMem-VLA uses only a single third-person RGB frame.  The paper correctly flags this, but readers should not treat the π₀ gap as apples-to-apples without that caveat.

## Suggested report claims

1. **LaMem-VLA makes robotic history context-native.**  It stores, retrieves, and consumes memory as ordinary tokens in the VLA embedding sequence, not as an external policy-side bank.  This is the paper’s central conceptual contribution.
2. **The four-stage loop is what enables the gain.**  The ablations show that the *combination* of dual-scale memory and latent-native injection outperforms either memory stream alone and outperforms policy-side conditioning.
3. **The empirical headline is strong on real-to-sim generalization.**  SimplerEnv-Bridge 73.9% is a clear jump over MemoryVLA (71.9%), π₀ (69.2%), and especially the CogACT baseline (57.3%).
4. **The method is still simulation-bound and implementation-sparse.**  With no released code and no appendix in the downloaded source, several important details (exact compressor, attention masks, vault bootstrap) remain underspecified; the real-world transfer story is pending.
