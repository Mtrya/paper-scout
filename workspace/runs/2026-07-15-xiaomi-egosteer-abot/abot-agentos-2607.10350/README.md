# ABot-AgentOS (arXiv:2607.10350) — Thread Packet

**Investigator:** Paper Scout research subagent  
**Run:** 2026-07-15-xiaomi-egosteer-abot  
**Thread:** abot-agentos-2607.10350  

## What was attempted

This thread deep-dives *ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory* by AMAP CV Lab. The paper proposes a deliberative agent layer that sits between foundation VLMs/VLAs and robot hardware, with three public-facing contributions:

1. **Agent OS architecture** — edge-cloud LLM routing, Agent Harness (Verification-aware ReAct, Context Management, Skill Evolvement), and a plugin Skills/Tools layer.
2. **EmbodiedWorldBench** — an executable long-horizon embodied benchmark in UnrealZoo.
3. **Universal Multi-modal Graph Memory** — typed, source-grounded graph memory with hybrid retrieval and failure-driven lifelong self-evolution via gated runtime "evo-assets".

I read the full 9,800-line Markdown conversion, searched for code/benchmark/project-page artifacts, cloned the advertised GitHub repo, and built a small runnable probe that reconstructs the memory-graph mechanism.

## Evidence preserved

- `code/memory_graph_probe.py` — runnable Python reconstruction of the typed graph memory, hybrid seed-selection retriever (Eq. 2 from the paper), evidence-subgraph expansion, and retrieval-trace generation. Run with:
  ```bash
  cd runs/2026-07-15-xiaomi-egosteer-abot/abot-agentos-2607.10350/code
  python memory_graph_probe.py
  ```
  It writes synthetic egocentric observations (the paper's "I adopted a Maltese dog yesterday" example) into nodes/edges and answers queries by retrieving evidence subgraphs.
- `code/external_signals.json` — summary of what external artifacts exist or are missing.
- This README records blockers and what the report should claim.

## External signals found

| Artifact | Status | Notes |
|----------|--------|-------|
| arXiv abstract/page | ✅ Exists | https://arxiv.org/abs/2607.10350 |
| GitHub repo | ⚠️ Placeholder | https://github.com/amap-cvlab/ABot-AgentOS exists but contains only a 10-line README referencing a non-existent architecture image; no code, no benchmark, no model weights. |
| Project page | ❌ 404 | https://amap-cvlab.github.io/ABot-AgentOS linked from arXiv returns HTTP 404. |
| EmbodiedWorldBench | ❌ Not released | Paper says "future work will report the complete benchmark evaluation and release EmbodiedWorldBench for open research use." |
| Training pipeline artifacts | ❌ Not released | Section 4 training pipeline is described as deployment-oriented; paper states "we focus on the method and do not release private data or production results." |
| Related ABot repos | ⚠️ Related, not this system | ABot-Navigation, ABot-Manipulation, ABot-PhysWorld, ABot-Explorer exist with code, but none implement AgentOS or the benchmark. |

## What the probe shows

The probe is deliberately small but faithful to the paper's stated mechanism:

- Memory is a typed graph 𝒢 = (𝒱, ℰ) with node types `session`, `semantic_event`, `entity`, `place`, `evidence`.
- Raw observations are compressed into source-grounded records with `time_ref`, `place`, `source_id`, `adapter_version`, `extractor_model`, and confidence.
- Hybrid retrieval follows Eq. (2): `s(q,v) = λ_sem s_sem + λ_lex s_lex + λ_meta s_meta + λ_type s_type`.
- Seed nodes are expanded along typed edges (`location`, `observation`, `provenance`, `temporal_order`, `identity`) to build an auditable evidence subgraph.
- The retrieval trace exposes which nodes/edges supported an answer, which is the prerequisite for the paper's self-evolution loop.

The probe is not the full system; it does not include the Agent Harness, edge-cloud routing, Skill Runner, Verifier, EmbodiedWorldBench, or the RL training pipeline.

## Key quantitative results from the paper

- **Agent evaluation (EmbodiedWorldBench subset, Table 1):**
  - ReAct/Qwen3.6-Plus baseline: TSR 49.97%, GCR 57.95%
  - ABot-AgentOS/Qwen3.6-Plus: TSR 61.96%, GCR 68.79% (+11.99 TSR, +10.84 GCR)
  - ABot-AgentOS/DeepSeek-V4-Pro: TSR 68.18%, GCR 74.62% (+6.22 TSR over same architecture)

- **Memory evaluation (Static system):**
  - LoCoMo: 87.5 overall (near human 87.9)
  - OpenEQA EM-EQA: 59.9 with 24 frames (listed scene-memory baselines ≤ 57.8)
  - Mem-Gallery: 88.6 overall
  - NExT-QA: 76.5 Acc@All
  - EgoLifeQA: 65.4 average retrieving only 1 frame

- **Self-evolution gains:**
  - LoCoMo +1.2 → 88.7
  - OpenEQA +1.2 → 60.4
  - Mem-Gallery +0.4 → 89.0
  - NExT-QA +4.1 Acc@All
  - EgoLife +0.8 → 66.2

## Limitations and uncertainties

1. **Artifacts are largely unavailable.** The GitHub repo is a placeholder, the project page is 404, and the benchmark/training artifacts are not released. The agent and memory results cannot be independently reproduced from public code.
2. **Agent evaluation is on a subset.** The paper calls the EmbodiedWorldBench agent numbers "initial system validation" rather than a full leaderboard, and does not report per-task variance or confidence intervals.
3. **Memory baselines are not fully controlled.** The paper acknowledges that external baseline protocols differ in backbone, frame budget, judge model, and subset; the strongest controlled comparison is Static vs. +Self-evo under the same implementation.
4. **Self-evolution uses ground-truth answers post-hoc.** The split-wise protocol prevents leakage, but the correctness signal in these experiments is benchmark ground truth, not real-world environmental feedback.
5. **Privacy classification claim is thin.** "Over 99% accuracy" for cloud-upload decisions is stated without dataset size, model, or test procedure.
6. **Model names are future-dated.** The paper cites "Qwen3.6-Plus" and "GPT-5.4", which do not correspond to publicly known models as of the paper's date. This may be a typo, internal naming, or speculative naming; the report should not treat these as confirmed public models.

## What the report should say

- ABot-AgentOS is best understood as a *system architecture paper* rather than a reproducible open-source release. Its main value is a coherent design that unifies several trends in embodied agents: hierarchical LLM control, verification-aware execution, typed multi-modal memory, and lifelong self-evolution.
- The Agent Harness decomposition (Main LLM → Skill Runner → Verifier, with edge-cloud routing) is a sensible response to the reasoning-execution gap, and the quantitative subset suggests it helps, but the effect size should be reported as preliminary.
- The graph-memory results are the strongest and most concrete contribution. The system achieves competitive or near-human numbers across five memory benchmarks, and the self-evolution protocol is technically well-designed (no current-split leakage, gated promotion, JSON DSL assets only).
- The comparison with related work is fair at a high level: it positions itself above ReAct/Voyager-style agents by adding verification, above MemGPT/Mem0 by adding typed graph structure and provenance, and above OSWorld-style digital benchmarks by focusing on physical execution.
- The report should flag the artifact gap honestly: project page 404, placeholder GitHub repo, and unreleased benchmark mean the community cannot yet build on the full system. The memory-graph probe in this thread at least makes the central mechanism concrete.

## How to rerun

```bash
cd runs/2026-07-15-xiaomi-egosteer-abot/abot-agentos-2607.10350/code
python memory_graph_probe.py
```

No dependencies beyond Python 3.10+ standard library.
