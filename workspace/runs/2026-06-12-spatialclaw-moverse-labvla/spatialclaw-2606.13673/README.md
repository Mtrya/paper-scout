# SpatialClaw Deep-Dive Thread

**Paper:** SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning (arXiv 2606.13673)  
**Thread directory:** `runs/2026-06-12-spatialclaw-moverse-labvla/spatialclaw-2606.13673/`  
**Investigator:** Paper Scout sub-agent  
**Date:** 2026-06-12

---

## 1. What was attempted

We investigated SpatialClaw's central claim: **code is the right action interface for spatial reasoning agents**, because a persistent, multi-turn Python kernel lets a VLM compose perception tools, inspect intermediate results, and revise its analysis in ways that single-pass code execution or structured tool-calls cannot.

Specific actions taken:

1. **External signal search**
   - Queried Hugging Face Papers (`hf papers info 2606.13673`) — confirmed project page and GitHub repo exist.
   - Fetched the project page at https://spatialclaw.github.io/.
   - Cloned the official implementation from https://github.com/NVlabs/SpatialClaw into `code/spatialclaw-2606.13673/`.
   - Searched for and cloned related spatial-agent code: `pySpatial` (single-pass code) and `VADAR` (project-page only, no code released).

2. **Code tracing**
   - Read the agent loop (`spatial_agent/workflow.py`), the persistent Jupyter kernel manager (`spatial_agent/kernel/manager.py`), the AST safety sandbox (`spatial_agent/kernel/safety.py`), the execute/feedback nodes, the system prompt builder, and the tool wrappers for reconstruction and geometry.
   - Identified the five-stage loop (plan → code → execute → feedback → reflect/continue) and the six kernel entry points (`InputImages`, `Metadata`, `tools`, `show`, `vlm`, `ReturnAnswer`).

3. **Related-work comparison**
   - `CodeAct` (Wang et al., ICML 2024): general code-as-action for LLM agents; SpatialClaw instantiates the same principle for spatial reasoning with domain-specific design (persistent kernel, per-frame typed containers, coordinate-system prompt discipline, BEV rendering).
   - `pySpatial` (Luo et al., ICLR 2026): single-pass Python program generation; the LLM writes one program, executes it once, then a second VLM answers from the output. No per-step revision.
   - `SpaceTools-Toolshed` (Chen et al., CVPR 2026): structured JSON tool-calls with RL fine-tuning; less flexible composition.
   - `VADAR` (Marsili et al., CVPR 2025): generates a dynamic Pythonic API then synthesizes a single program; no persistent multi-turn kernel.

4. **Probe / reconstruction**
   - Built a minimal, self-contained reconstruction of the SpatialClaw mechanism in `code/`:
     - `mock_kernel.py`: persistent Python namespace with stdout capture and variable tracking.
     - `mock_tools.py`: mock SAM3, Depth-Anything-3 reconstruction, geometry, and mask tools.
     - `safety.py`: lightweight AST sandbox.
     - `probe.py`: scripted agent loop demonstrating segmentation → reconstruction → metric distance computation → `ReturnAnswer`.
   - The probe runs without GPUs, VLMs, or Jupyter and produces a saved input image plus step-by-step feedback.

---

## 2. What was found

### 2.1 Core mechanism

SpatialClaw's action interface has three defining properties:

- **Code as the action medium.** Each step is one executable Python cell, not a JSON tool call. This lets the agent freely compose perception outputs with NumPy/SciPy/Matplotlib operations that were not anticipated in a fixed tool schema.
- **Persistent state.** The IPython kernel keeps variables (masks, depth maps, point clouds, plots, partial results) across steps. Later cells can reuse and refine earlier outputs.
- **Closed-loop feedback.** After each cell the agent sees stdout, variable summaries, tracebacks, and images registered via `show()`. It can revise masks, recompute, or cross-check before committing to `ReturnAnswer()`.

The implementation confirms the paper's architectural description exactly:

- `workflow.py` builds a LangGraph with nodes `init → plan → llm_step → execute → feedback → reflection`.
- `kernel/manager.py` wraps a real `jupyter_client.AsyncKernelManager`; variables persist via `store_history=True` and are introspected with `get_variables()`.
- `safety.py` AST-checks every generated cell before execution, blocking forbidden modules, builtins, and file-I/O patterns.
- `init_node.py` injects `InputImages`, `Metadata`, `tools`, `vlm`, `feedback`, and `ReturnAnswer` into the kernel namespace and monkey-patches `plt.show()` so figures flow into `show()`.

### 2.2 Empirical claims check

The paper reports 59.9% average accuracy across 20 benchmarks, +11.2 pp over SpaceTools-Toolshed, with consistent gains across six backbones (Qwen3.5/3.6, Gemma4) and no benchmark- or model-specific tuning. The repo ships all 20 benchmark loaders and SLURM launch managers to reproduce every table. We did not run the full evaluation (it requires vLLM, GPU perception servers, SAM3/Depth-Anything-3 weights, and SLURM), but the code structure matches the reported design.

Key ablations visible in the code:

- `config.prompt_section_ablations` allows excluding/overriding prompt sections, confirming the paper's claim that the same prompt is used everywhere.
- `ToolsModule.get_all_prompt_descriptions_static()` toggles GPU tools via `config.tools_to_use`, supporting the ablation that removes perception tools or utility wrappers.

### 2.3 Contrast with related work

| Method | Interface | Persistence | Step-by-step revision | Key limitation SpatialClaw addresses |
|---|---|---|---|---|
| CodeAct | general Python actions | multi-turn | yes | not specialized for 3D/4D spatial reasoning |
| pySpatial | single-pass Python program | one-shot | no | commits to full strategy before seeing any output |
| SpaceTools-Toolshed | JSON tool calls | multi-turn | yes | fixed tool schema limits emergent compositions |
| VADAR | dynamic API + single program | one-shot | no | no per-step inspection of intermediate evidence |
| **SpatialClaw** | Python cell per step | persistent kernel | yes | composes + inspects + revises in one loop |

The `pySpatial` code (`agent/codeAgent/query.py`, `agent/codeAgent/execute.py`) confirms it is single-pass: `generate_code_from_query()` → `parse_LLM_response()` → `execute_code()` → `answer()`. The generated code defines a single `program(scene)` function.

The `VADAR` GitHub repository (https://github.com/glab-caltech/VADAR) contains only the project-page HTML; no implementation was available to trace.

---

## 3. Probe / reconstruction

The probe lives in `runs/2026-06-12-spatialclaw-moverse-labvla/spatialclaw-2606.13673/code/`.

Files:

- `mock_kernel.py` — minimal persistent Python workspace.
- `mock_tools.py` — mock `SAM3`, `Reconstruct`, `Geometry`, `Mask`, `ToolsModule`.
- `safety.py` — AST sandbox.
- `probe.py` — scripted agent loop for the scenario *"Which object is closer to the camera — the red car or the blue bicycle?"*.
- `probe_input.jpg` — synthetic input image generated by the probe.

How to run:

```bash
cd runs/2026-06-12-spatialclaw-moverse-labvla/spatialclaw-2606.13673/code
.venv/bin/python probe.py
```

The probe demonstrates:

1. Pre-loading the kernel with `InputImages`, `tools`, `np`, `show()`, and `ReturnAnswer`.
2. Step 1: segment both objects with mock SAM3 and `show()` the masks.
3. Step 2: reconstruct metric 3D geometry.
4. Step 3: compute 3D centroids, camera-to-object distances, and `show()` a BEV visualization.
5. Step 4: submit `ReturnAnswer("car")`.

Expected output ends with:

```
[FINAL ANSWER SUBMITTED] car
Result: car is closer to the camera.
```

This is a *humble reconstruction*: it replaces the real VLM with scripted cells and the real perception models with geometry-aware mocks. Its purpose is to make the action-interface mechanism concrete and reproducible without the full SpatialClaw infrastructure.

---

## 4. Main research takeaway

The strongest finding is that **the action-interface design is the independent variable driving SpatialClaw's gains**, not tool coverage or model-specific tuning.

Evidence:

- The paper's ablation shows that removing utility wrappers (`tools.Mask`, `tools.Geometry`) loses only −0.5 pp, because the agent can reimplement them on the fly with NumPy/SciPy.
- Removing the perception tools entirely still beats the no-tool baseline by +2.7 pp, isolating the contribution of the code-as-action interface itself.
- The code confirms the interface is implemented as a generic loop (LangGraph + Jupyter kernel + prompt sections) with no benchmark-specific branches.
- The largest gains are in categories requiring chained geometric computation across frames/viewpoints (camera motion, multi-view reasoning, relative direction), precisely where per-step composition and revision matter.

The broader implication: for spatial reasoning, the agent's action space should be as expressive as Python, not as restrictive as a fixed JSON tool menu, and it should be evaluated in a persistent loop rather than a single program. This shifts the engineering focus from "which tools do we expose?" to "how do we let the agent compose, inspect, and revise whatever computation the question demands?"

---

## 5. Limitations and blockers

### Blockers encountered

- **Full reproduction requires infrastructure we did not run.** The official implementation depends on:
  - vLLM-served backbones (Qwen3.5/3.6, Gemma4).
  - A GPU perception-tool server wrapping SAM3 and Depth-Anything-3.
  - SLURM for the launch managers (or a multi-GPU machine for the direct CLI).
  - Pre-downloaded model weights and third-party git submodules.
  We therefore did not reproduce the 20-benchmark numbers and treat the empirical claims as unverified by our run.

- **VADAR code unavailable.** The VADAR repository contains only the project-page HTML; we could not trace its implementation. The comparison relies on the paper and project-page description.

### Limitations of the probe

- The probe uses scripted agent cells, not a real VLM, so it cannot demonstrate genuine emergent tool composition.
- Mock tools use synthetic geometry, not real segmentation or reconstruction, so numerical results are illustrative.
- It does not exercise error recovery, timeout handling, or the planner node.

### Limitations of SpatialClaw itself (per the paper)

- The remaining bottleneck is perception quality, not the action interface. Failure-mode analysis attributes the largest share of errors to VLM hallucinations and tool limitations.
- The framework assumes a VLM strong enough to write correct Python; weaker code-generation models may not benefit as much.

---

## 6. Assets and figures for the report

Best illustrative material from the paper and this thread:

1. **Figure 2** (`papers/agents/spatialclaw-2606.13673.md` or project page) — the three action interfaces side-by-side (single-pass code, structured tool-call, SpatialClaw). This is the conceptual centerpiece.
2. **Figure 3** — the five-stage agentic loop (plan → code → execute → feedback → answer). Good for explaining the mechanism.
3. **Table 2 / action-interface ablation** — shows No-tool 53.4, Single-pass code 55.2, Structured tool-call 56.7, SpatialClaw 59.9 on Gemma4-31B. Direct evidence that the interface itself drives gains.
4. **Figure 5** — primitive-usage heatmap showing distance questions use KDTree/norms and direction questions use dot products. Demonstrates spontaneous, task-adaptive composition.
5. **Figure 6** — attribution of wins over structured tool-call: 52.2% code composition, 19.5% control flow. Makes the mechanism concrete.
6. **Thread probe artifacts:**
   - `code/probe_input.jpg` — synthetic input image with red car and blue bicycle.
   - `code/probe.py` terminal output (or a screenshot) — shows multi-step feedback and `ReturnAnswer`.
   These are useful for a "how it works under the hood" sidebar.

Local copies of paper figures are in `runs/2026-06-12-spatialclaw-moverse-labvla/assets/spatialclaw-2606.13673/`.

---

## 7. File paths summary

- Official code clone: `code/spatialclaw-2606.13673/`
- pySpatial code clone (comparison): `code/pyspatial-2603.00905/`
- VADAR project-page clone (no code): `code/vadar-2502.06787/`
- Thread README: `runs/2026-06-12-spatialclaw-moverse-labvla/spatialclaw-2606.13673/README.md`
- Probe code: `runs/2026-06-12-spatialclaw-moverse-labvla/spatialclaw-2606.13673/code/`
- Probe input image: `runs/2026-06-12-spatialclaw-moverse-labvla/spatialclaw-2606.13673/code/probe_input.jpg`
- Paper markdown: `papers/agents/spatialclaw-2606.13673.md`
- Paper figures/assets: `runs/2026-06-12-spatialclaw-moverse-labvla/assets/spatialclaw-2606.13673/`
