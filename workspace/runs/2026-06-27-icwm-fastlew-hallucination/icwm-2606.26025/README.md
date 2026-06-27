# Deep thread: In-Context World Modeling for Robotic Control (arXiv:2606.26025)

## What was investigated

"In-Context World Modeling for Robotic Control" reframes VLA generalization as a test-time **system identification** problem. The paper argues that standard VLAs condition only on the current observation and language instruction, implicitly marginalizing over the system configuration ψ (camera viewpoint, robot morphology). ICWM instead prepends a short prefix of **self-generated, task-agnostic interaction clips** `(o_i^s, a_i, o_i^e)` so the model can infer ψ in-context, without parameter updates or task-specific demonstrations.

This thread asked three concrete questions:

1. Is there an official implementation, model, or dataset artifact?
2. How would ICWM be realized on top of existing open-source components (LIBERO, OpenVLA, FAST)?
3. Does the paper's core information-theoretic claim hold in a minimal synthetic setting?

## Code/artifact status

**No official ICWM code was found.** The paper is dated June 24, 2026. Searches of arXiv, GitHub, Papers with Code, and the authors' home institutions turned up no public repository, project page, or model checkpoint. The closest public artifacts are the building blocks the paper cites:

- **LIBERO** benchmark (`Lifelong-Robot-Learning/LIBERO`) — simulation suite and multi-camera rendering code.
- **OpenVLA** (`openvla/openvla`) — open VLA training/inference codebase; shows the one-image-per-sequence input format that ICWM would have to extend.
- **FAST** action tokenizer (`physical-intelligence/fast`, used via `Physical-Intelligence/openpi`) — DCT+BPE action tokenization, which the paper uses instead of OpenVLA's naive 256-bin discretization.

These repos were cloned into `/home/betelgeuse/Documents/paper-scout/workspace/code/` for inspection (see `libero-icwm-probe` and `openvla-icwm-probe`). The durable probe scripts derived from that inspection live in this thread under `code/`.

## Constructive research action

I preserved two runnable probes and the implementation notes from inspecting OpenVLA and LIBERO.

### 1. `code/icwm_data_construction.py` — training sample & probing protocol

This script makes the ICWM data construction concrete:

- Builds a training sample by prepending **N=5** task-agnostic interaction clips sampled from a pool of trajectories.
- Each clip is `(start_image, action, end_image)`.
- Formats the task query as `[current_image, language_instruction, target_action]`.
- Implements the test-time **active probing phase**: 20 random workspace probes, from which 5 clips are sampled as context at each inference step (matching the paper's protocol in Appendix D).

Run:

```bash
cd code
python icwm_data_construction.py
```

It prints the token-sequence layout and saves `outputs/icwm_sequence_summary.json` and `outputs/task_query_image.png`.

### 2. `code/icwm_information_probe.py` — Proposition 1 diagnostic

This is a synthetic check of the paper's claim that interaction context carries strictly more information about ψ than a single observation. It builds a minimal 2-D "camera projection" world where ψ is an azimuthal viewpoint and the robot observes a scalar horizontal projection. It then compares the posterior entropy over ψ after:

- one noisy observation, versus
- a sequence of 5 known probing actions plus their resulting observations.

Example output:

```text
Prior entropy over 14 viewpoints: 3.807 bits
Posterior entropy after single observation: 2.186 bits
Posterior entropy after T=5 interaction steps: 1.560 bits
Information gain (single obs): 1.621 bits
Information gain (interaction): 2.248 bits
Interaction context carries 0.627 extra bits about psi
```

Run:

```bash
cd code
python icwm_information_probe.py
```

Results are saved to `outputs/information_probe.json`.

### 3. Implementation notes from OpenVLA / LIBERO inspection

**OpenVLA input format.** The PrismaticVLM forward (see `prismatic/models/vlms/prismatic.py`, lines 367–396) assumes one image per sequence: it splits `input_ids` at position 1, inserts projected image patch embeddings between the first token and the remainder, and masks image tokens in the loss. To support ICWM's N context images + 1 task image, this logic would need to be extended so that every image placeholder token receives its own `pixel_values` slice. The paper's KV-cache remark (Sec. 6.4) only holds once that multi-image input pipeline exists.

**LIBERO multi-view rendering.** Camera positions are hard-coded in each problem's `_setup_camera` method (e.g. `libero/libero/envs/problems/libero_tabletop_manipulation.py`, lines 187–211). The paper trains on 8 azimuthal angles `{30°, 60°, 90°, 120°, 240°, 270°, 300°, 330°}` and tests on 6 OOD angles `{45°, 135°, 225°, 255°, 285°, 315°}`. Reproducing their cross-view protocol therefore requires replaying the stored MuJoCo states and re-rendering from each camera pose — the `scripts/create_dataset.py` playback loop is the natural starting point, but with the camera pose varying per output dataset.

## Key paper numbers / quotes

- Core formulation: standard VLA as `π_θ(a_t | o_t, l)` (Eq. 1); ideal policy as `π_θ^*(a_t | o_t, l, ψ)` (Eq. 2).
- Training loss: `L = -log π_θ(a_t | Ψ(T), o_t, l)` (Eq. 7), where `Ψ(T)` are the hidden states induced by the interaction context.
- N=5 context clips; action chunk size 5; backbone Qwen2.5-VL-3B; FAST action tokenizer (Sec. 5.1).
- LIBERO OOD success rate: ICWM 25.0% vs. Multi-View BC 19.8% (+13.0% relative), vs. Explicit Configuration (+9.5%).
- Largest gains on LIBERO-Long: +29.9% seen, +26.3% unseen over MV.
- Real robot: standard VLA drops from 68% to 17% on viewpoint shift; ICWM recovers much of that gap (Sec. 5.3, Fig. 5).
- Ablations (Tab. 1): removing images from context causes −56.4% average collapse; false context (180° offset) performs worse than no context, indicating the model genuinely uses the context for configuration inference.
- Latency: baseline 0.112s/step; ICWM N=3 0.165s, N=5 0.185s on RTX 4090 (Sec. 6.4).

## Key preserved files

```text
runs/2026-06-27-icwm-fastlew-hallucination/icwm-2606.26025/
├── README.md
├── code/
│   ├── icwm_data_construction.py      # training sample + probing protocol
│   ├── icwm_information_probe.py      # Proposition 1 synthetic diagnostic
│   └── outputs/                       # generated by the probes
│       ├── icwm_sequence_summary.json
│       ├── information_probe.json
│       └── task_query_image.png
```

The upstream repos inspected (not durable, ignored by git) live in `workspace/code/libero-icwm-probe` and `workspace/code/openvla-icwm-probe`.

## Findings for the report

1. **ICWM is a data-format and inference-protocol idea, not a new architecture.** The paper's main technical move is conditioning the VLA on self-generated interaction triplets. The model itself is a standard autoregressive VLA (Qwen2.5-VL-3B + FAST). This makes the idea cheap to add to existing training stacks, but also means the heavy lifting is in the data pipeline and multi-image input handling.

2. **No official artifacts exist yet.** The result depends entirely on the paper's tables and figures; independent verification will require re-implementing the multi-view LIBERO data synthesis and the Qwen2.5-VL + FAST training pipeline from scratch. That is feasible but non-trivial.

3. **The information-theoretic justification is sound in principle.** The synthetic probe confirms that interaction context reduces posterior entropy over ψ compared with a single observation. Whether a 3B-parameter VLA can reliably extract that extra information from raw pixels across 14 camera angles is the empirical question the paper claims to answer affirmatively.

4. **The most important caveat is the limited resolution of the ablations.** The paper shows that context content matters and that false context hurts, which supports genuine conditioning on ψ. But it does not isolate how much of the gain comes from (a) better viewpoint estimation versus (b) a learned "calibration" of action-image correspondence, or (c) simply more tokens at test time. A follow-up could probe the hidden representation `Ψ(T)` more directly than the t-SNE in Fig. 7.

## Suggested report angle / takeaway

ICWM points to a simple but underexplored design axis for VLAs: **use the context window for system identification, not just task specification.** The paper's strongest result is that task-agnostic random probing already buys a large chunk of viewpoint generalization, with no parameter updates and no human demonstrations. The key tension to foreground is between the method's appealing simplicity and the currently missing public artifacts: the idea is easy to describe, but reproducing the LIBERO cross-view numbers will require rebuilding a non-trivial data and training pipeline. If the claim holds, it suggests future VLAs should be trained with interaction prefixes as a default, not as an afterthought.
