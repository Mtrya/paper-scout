# WRBench deep-dive: Current World Models Lack a Persistent State Core

**Paper:** arXiv:2606.20545 — *Current World Models Lack a Persistent State Core*  
**Authors:** Jinpeng Lu, Dexu Zhu, Haoyuan Shi, et al.  
**Code:** https://github.com/JinPLu/WRBench  
**Project page:** https://jinplu.github.io/WRBench  
**Thread packet:** `runs/2026-06-22-humanscale-playful-imagewam-wrbench/wrbench-2606.20545/`

## Research question

WRBench asks whether a video world model keeps an internal world state that evolves independently of the camera. The probe is simple but sharp: the prompt fixes an event (e.g., *a cat jumps onto the bed*), the camera turns away, and later returns. If the model is maintaining a persistent world state, the cat should be on the bed when the view comes back. If it is merely rendering plausible frames on demand, the returning target is likely to be frozen, reset, misplaced, or erased.

The key twist is **attribution**: a single "did it look right?" score cannot tell whether the camera failed to turn, the world stopped while unseen, or the return was correct. WRBench therefore decomposes evaluation into a chain of six diagnostics, keeping each failure mode legible.

## Key diagnosis: a missing persistent state core

Across 23 models and 9,600 generated videos, the paper finds a recurring **preservation–access–re-observed-consistency gap**:

- Models can render visually coherent frames (D2, D3, D4).
- With explicit trajectories or source-video conditioning, they can usually swing the camera and bring the target back into view (D1, D5 support).
- But once the target is back and judgeable, the event endpoint is often wrong (D6).

The failure is sharpest for **in-place state change** (e.g., folding a blanket, sitting down) rather than relocation. Relocation gives the model a new coordinate and static anchors; in-place change offers no anchor, so the altered state drifts or is silently reset during the hidden interval. Scaling Wan 2.1 from 1.3B to 14B raises re-observation support but *lowers* conditional re-observed-state consistency, showing that endpoint persistence is not a by-product of scale or visual quality.

The diagnosis is that current systems act as **view-conditioned renderers** with memory for *where* to look back, not *what changed* while hidden. The missing ingredient is a persistent state core: a representation that writes event outcomes (contact, containment, posture, placement) and reads them back after occlusion.

## What code was inspected

A partial mirror of the WRBench repository is under `code/WRBench-repo/`. Full `git clone` and archive downloads timed out in this environment (large repository, unstable connection to GitHub), so the files needed for this deep dive were fetched individually from `https://raw.githubusercontent.com/JinPLu/WRBench/main/`. The inspected components are:

| Path | What it does |
|------|--------------|
| `src/wrbench/data/natural25/scene_events_25x4.csv` | The 25 scene families crossed with the four event tiers (none / spatial / state-only / full). |
| `src/wrbench/data/natural25/variants.jsonl` | Pre-generated 400-row event-view record set: 25 families × 4 event tiers × 4 camera gaps. |
| `src/wrbench/data/natural25/families.jsonl` | Scene metadata, first-frame guidance, primary/interactor/anchor objects. |
| `src/wrbench/data/natural25/t2v_layout_anchors.jsonl` + `t2v_event_tails.jsonl` | Text-only prompt material for T2V/prompt-only models. |
| `src/wrbench/data/natural25/prompt_profiles/t2v_layout_anchor.json` | Policy that strips style tokens and keeps only layout/event facts. |
| `src/wrbench/data/natural25/camera_scopes/t2v_rotation_stress_30_60.json` | Formal T2V camera scope: static, yaw30/60 LR/RL. |
| `src/wrbench/eval/d1/d1_camera.py` | Requested-camera precision: loads certified target trajectories, compares recovered VGGT-Ω poses, and reports per-row camera accuracy. |
| `src/wrbench/eval/d1/d1_camalign.py` | Prompt-camera alignment for prompt-only/API rows (common-yaw / static-hold). |
| `src/wrbench/eval/d2/extract_d2_dinov2_consistency_features.py` | Visual integrity via DINOv2 global + center/motion-crop temporal consistency. |
| `src/wrbench/eval/scoring/prompts_v2_probe.py` | Yes/no VLM probe catalog for D3–D6, including gate and score probes. |
| `src/wrbench/eval/scoring/run_local_qwen35_probe_logprob_scorer.py` | Runs Qwen-3.5/3-VL to score probes from next-token Yes/No log-probabilities and aggregates dimension scores. |
| `src/wrbench/eval/scoring/run_local_qwen3vl_video_evidence.py` | Re-observation gate: structured visibility evidence from Qwen3-VL. |
| `src/wrbench/eval/runtime.py` | Orchestrates the full D1–D6 pipeline and table build. |
| `scripts/run_natural25_generation.py` | Batch generator that compiles Natural-25 tasks into model-native camera payloads. |
| `src/wrbench/eval/contract/latest_d1_d6_metric_contract.json` | Frozen metric contract: score fields, denominators, gate rules. |

## The evaluation pipeline, explained

WRBench evaluates each generated video as **evidence for a specific event under a specific viewpoint intervention**, not as a standalone quality sample.

### 1. Camera control (D1)

- **D1-CamPrec** (*requested-camera precision*) applies to models that receive an explicit trajectory. WRBench compares the requested OpenCV camera-to-world path against the trajectory recovered from the generated video by VGGT-Ω. It checks sign/direction for yaw/pan motions and stillness for static shots.
- **D1-CamAlign** (*prompt-camera alignment*) is a separate diagnostic for prompt-only/API models. It checks whether the video follows the intended common-yaw direction or stays static, rather than strict pose error.

The two are **never merged**: they answer different questions for different control interfaces.

### 2. Visual integrity (D2)

Before any world-state judgment, WRBench asks whether the frames are trustworthy. It samples frames at 3 fps (up to 24), extracts DINOv2 features, and combines:

- **Global continuity**: cosine similarity between first and last CLS tokens.
- **Local continuity**: bidirectional best-match patch-token similarity between adjacent frames, aggregated at the 20th percentile so localized collapse (cuts, ghosting, identity drift) is penalized even when the median frame pair looks fine.

D2 is an evidence-quality floor, not a physics or event-correctness score.

### 3. Visible consistency (D3 / D4)

While the target is in view, Qwen-3.5 answers compact yes/no probes:

- **D3 visible spatial**: Is the subject/target in a plausible world-space position, contact, support, or path relative to stable anchors?
- **D4 visible state**: Does the subject actually perform/reach the required action, pose, or state change, and maintain it?

Each dimension mixes positive probes (intended evidence present) and negative probes (counter-evidence absent). The dimension score is the average of polarity-adjusted next-token probabilities:

```
p_yes = exp(L_yes) / (exp(L_yes) + exp(L_no))
e =  p_yes      for positive probes
 e = 1 - p_yes  for negative probes
M = mean(e over probes)
```

### 4. Re-observation support (the gate for D5/D6)

D5 and D6 are only posed when the video actually creates a hidden-and-returned interval. A separate Qwen3-VL gate decides whether:

- the prompt-critical subject/state becomes unjudgeable for a meaningful interval, and
- later evidence makes it judgeable again.

If the target never leaves, never returns, or returns unidentifiable, the row is marked **N/A** for D5/D6. This is crucial: absence of judgeable return evidence is recorded as *insufficient support*, not as a failed state test.

### 5. Re-observed consistency (D5 / D6)

On the subset that passes the gate, the same probe style is reused:

- **D5 re-observed spatial**: Does the returned subject occupy a plausible world-region position relative to pre-hidden evidence and anchors?
- **D6 re-observed state**: Does the returned action/state look like the event continued through the hidden interval rather than freezing, resetting, or skipping?

### Why the denominators matter

The benchmark reports **support rate** and **conditional metric** separately:

- `reobservation_support` = (# judgeable rows) / (# all generated rows).
- `reobserved_state` = mean(D6 score) **only over judgeable rows**.

This prevents two common confusions:

1. A model that keeps the target in frame has high visible scores but near-zero support; collapsing support into the score would make it look like it passes the hidden-state test when it never took it.
2. A model with beautiful frames but wrong returns would get a high D2/D3/D4 and a low conditional D6; averaging them together would hide the failure.

The conditional mean is therefore a proper measure of **endpoint binding**, conditional on the test actually being posed.

## The probe in this packet

`code/probe_wrbench.py` is a zero-GPU diagnostic. It:

1. Loads `scene_events_25x4.csv`, `variants.jsonl`, and `families.jsonl`.
2. Verifies the 25 × 4 event-tier × 4 camera-gap factorial design.
3. Prints sample event-view records for one family (`bedroom_cat_bed_jump`) across all four event tiers.
4. Demonstrates the denominator mechanics using Gen3C's Table 2 numbers (73.0% support, 0.640 conditional D6).
5. Emits the actual VLM prompts for a D3 visible-spatial probe and a D5 re-observation gate probe.

### Rerun

```bash
cd runs/2026-06-22-humanscale-playful-imagewam-wrbench/wrbench-2606.20545/code
python3 probe_wrbench.py
```

The script adds `WRBench-repo/src` to `sys.path` so it can import `wrbench.eval.scoring.prompts_v2_probe` without installing the package.

### Sample output (abridged)

```
Natural-25 factorial design checks passed:
  families=25  variants=400
  event tiers={'none': 100, 'spatial': 100, 'state_only': 100, 'full': 100}
  camera gaps={'none': 100, 'static': 100, 'yaw_LR': 100, 'yaw_RL': 100}
  cross count per (event, camera) cell=25

--- Denominator mechanics for Gen3C ---
  reported re-observation support = 73.0%
  reported conditional re-observed-state score = 0.640
  assumed total generated outputs = 400
  judgeable rows (D5/D6 denominator) = 73.0% x 400 ≈ 292
  sum of judgeable D6 scores = 0.640 x 292 ≈ 186.9
  conditional mean = 186.9 / 292 = 0.640
  => D6 is averaged ONLY over rows that pass the re-observation gate.
```

The full run also prints the D3 and D5 probe prompts, showing how the evaluation encodes the world-state question into a strict yes/no VLM call.

## Report takeaway

WRBench is not another holistic video-quality leaderboard. It is an **evidence-attribution benchmark**: it uses viewpoint change as an intervention on observability and asks whether the same evolving world state is supported before, during, and after the hidden interval.

The empirical result is that current video world models fail the re-observed-state test systematically, especially for in-place state changes, and that this failure is **orthogonal to visual fidelity, camera precision, and scale** within the evaluated scope. The path forward, suggested by the paper, is to build models with an explicit state writer that records hidden event endpoints and a training objective that supervises endpoint persistence across occlusion — a "what-memory" to complement the existing "where-memory" of geometry, appearance, and camera carriers.

For this run, the most actionable follow-up is to treat WRBench's D5/D6 conditional metrics as a first-class target in model training, not just a post-hoc diagnostic, and to use the released preference-pair exports to reward endpoint binding directly.
