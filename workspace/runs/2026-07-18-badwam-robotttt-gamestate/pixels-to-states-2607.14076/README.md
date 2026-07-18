# Thread Packet: From Pixels to States (arXiv:2607.14076)

**Paper**: *From Pixels to States: Rethinking Interactive World Models as Game Engines* — Zian Meng, Shuwei Shi, Mingliang Zhai, Jiaming Tan, Chuanhao Li, Kaipeng Zhang (Alaya Lab / Shanda), v1 2026-07-15.
**Type**: position/analysis paper + data engine (no model trained).
**Investigation date**: 2026-07-18. Investigator: paper-scout subagent.

## What this packet contains

| Path | What |
|---|---|
| `code/classify_landscape.py` | Encodes the four-dimension classification of 6 recent WAMs (with evidence quotes) → emits `classification.json`, `classification.md`, `taxonomy_matrix.png`. Rerun: `python3 classify_landscape.py` (numpy+matplotlib only). |
| `code/classification.json` / `code/classification.md` | The structured analysis. The judgments are the content; quotes are pulled from the full-text caches in `papers/world-models/`. |
| `code/implicit_state_demo.py` | Mechanistic toy experiment: implicit-window vs explicit-state outcome prediction in a synthetic HP-3 boss fight. Rerun: `python3 implicit_state_demo.py` (~1 min, CPU). |
| `code/demo_results.json` / `code/forgetting_curve.png` | Demo output: implicit model decays 0.95→chance as state-change spacing passes the context window; explicit model stays at 1.00. |
| `code/taxonomy_matrix.png` | 6 papers × 4 dimensions matrix, colored by the position paper's families. |
| `../../assets/` | Copies of the two figures + classification table for the report. |

## The paper's lens, precisely

A conventional game engine runs a recurrent **action → state → observation** loop: the player
input first updates an explicit game state (HP, stamina, cooldowns, boss phase, position) according
to rules; only then is the frame rendered from state. Four dimensions follow:

- **D1 player action control** — how intent is represented before it drives generation.
  Families: *geometric trajectories* (camera poses/Plücker), *motor signals* (raw keyboard/mouse,
  device-defined or latent actions), *semantic events* (language-specified, scene- to entity-level).
- **D2 game state dynamics** — how state is represented and updated.
  Families: *entangled in observations* (state-free pixel prediction), *learned latents* (recurrent
  compressive state), *explicit descriptions* (symbolic/textual state, transition = reasoning).
- **D3 state-observation persistence** — consequences must survive long horizons.
  Families: *memory as stored observations* (temporally or spatially indexed past-copy retrieval),
  *memory as estimates of the present* (memory updated to track a changing world).
- **D4 real-time interactive generation** — the paper's sharpest distinction: **control latency**
  (input → visible response; minimize) vs **consequence latency** (action → rule-governed outcome;
  must be *accurate*, not early). Families: *reducing generation latency* (distillation/streaming/
  systems), *reducing conditioning latency* (mid-rollout condition switching).

## Headline findings of this investigation

1. **The "state stays implicit" diagnosis holds across the recent WAM landscape.** Of 6 recent
   systems classified (MIRA, ActWorld, Kairos, DreamX-World, GigaWorld-1, + BadWAM as an
   analysis-side datapoint), **zero maintain an explicit symbolic state in the generation loop**.
   Explicit state exists only as *data/eval* (MIRA's privileged-state probes; WildWorld;
   EgoCS-400K; AnimeGamer) — never as the variable that drives generation. See the matrix figure:
   the "explicit symbolic state" color appears in the legend but in no cell.
2. **Strongest counterexamples, ranked**: (a) **Kairos** — GLA recurrent state matrix with formal
   error-accumulation bounds is a real carried state, but uninterpretable and trained by visual
   prediction only (exactly the paper's "learned latents" caveat: unreliable for visually-invisible
   rule variables). (b) **ActWorld** — event-routed persistent memory is the closest thing to
   "estimates of the present," but update decisions are learned saliency, not rule-grounded.
   (c) **MIRA** — probes show implicit latents DO track true physics locally (ball/car positions);
   implicit state suffices for fast continuous dynamics, nothing carries irreversible conditions.
3. **Mechanistic demo** (`implicit_state_demo.py`): in a synthetic boss fight where HP is
   visually invisible and "the same strike kills or merely wounds," a windowed implicit-state model
   decays from 0.95 to chance (0.5) outcome-prediction accuracy as the spacing of state-changing
   events exceeds the context window W; an explicit carried scalar state with a *learned* linear
   transition (recovered exactly: h′ = 1.000·h − 1.000·hit) stays at 1.00. Diagnostic confirms the
   mechanism: implicit accuracy tracks the number of state-change events still inside the window.
4. **GigaWorld-1 independently corroborates the premise** with 324k rollouts: evaluator quality is
   dominated by long-horizon action-faithful state consistency, not short-term visual realism —
   yet its own answer is past-copy memory (a never-evicted first-frame anchor), not an evolving state.
5. **BadWAM adds a complementary failure**: even when a WAM carries an imagined future, nothing
   binds the executed action to it (96.5%→43.1% success under drift attack). The action-state-
   observation loop is not closed on either side.

## Dataset status (checked 2026-07-18)

- **Not released.** arXiv v1 has no code/data links; no project page, GitHub, or HF dataset found.
  Paper says "as a resource for state-aware game world modeling" — aspirational for now.
- Same team (Alaya/Shanda; Meng, Shi, Li, Zhang) previously built **WildWorld** (Monster Hunter:
  Wilds, 108M frames, 450+ actions, explicit per-frame states, project page existed but 404s as of
  this check). Track record suggests intent to release, but WildWorld's page being down is a
  cautionary signal about longevity.
- **Capture design (reusable)**: engine instrumentation exports JSON action/state records per tick;
  ReShade split-screen shader shows RGB + depth-buffer side by side with UI passes disabled; OBS
  Studio logs per-frame system timestamps; common clock aligns engine records to video frames.
  Zero extra hardware; crowdsourcers just play. Slot captions aggregate per-tick records into
  fixed-window structured text; semantic captions from Qwen3-VL-235B conditioned on frames +
  records. This recipe is directly transplantable to any moddable PC game and cheap.
- **Robotics reusability**: the *pipeline* is the contribution for us — frame-aligned
  (action, explicit state, RGB+depth) triples are exactly what robot WAM work lacks (teleop logs +
  proprio state + multi-cam). The *content* (boss fights) transfers to combat/game WAMs, not to
  manipulation; but "slot captions as next-state supervision for language-based state transition"
  is a concrete recipe for conditioning generation on explicit state.

## Limitations of the paper itself

- **No model, no experiments.** The position is argued, never demonstrated; the dataset ships with
  no baseline numbers (not even a video-prediction baseline on the new data).
- Dimensions are not independent (D2 vs D3 overlap heavily — memory *is* the state mechanism in
  most systems); families within a dimension overlap too (ActWorld is cited in both action and
  memory families).
- The engine analogy has a silent gap: engines make state *causal* (rendering reads state), while
  all proposed "explicit state" integrations merely *condition* on state — the paper does not
  discuss how to make generated pixels actually respect a state variable (a hard, unsolved problem).
- Boss-encounter-only scope; ReShade depth from the buffer is not ground-truth simulation depth.
- Consequence-latency framing (D4) is genuinely novel but gets no operationalization (no metric).

## Figure-worthy artifacts (in `../../assets/`)

1. `taxonomy_matrix.png` — 6 WAMs × 4 dimensions; the empty "explicit state" family tells the story.
2. `forgetting_curve.png` — implicit vs explicit state outcome prediction vs event spacing / W.
3. `classification.md` — the full table with evidence quotes (report-ready).
