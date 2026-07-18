# WAM landscape classified on the Pixels-to-States four-dimension lens

Lens: arXiv:2607.14076. D1 = player action control, D2 = game state dynamics,
D3 = state-observation persistence, D4 = real-time interactive generation.

| Paper (arXiv) | D1 action control | D2 state dynamics | D3 persistence | D4 real-time |
|---|---|---|---|---|
| **MIRA (Multiplayer Interactive WM)** (2607.05352) | motor signals | implicit (state entangled in observations) | none (rolling window only) | real-time (generation-latency reduction) |
| **ActWorld** (2606.17730) | motor + semantic (hybrid) | implicit (state entangled in observations) | estimates of the present (updating memory) | real-time (generation-latency reduction) |
| **Kairos** (2606.16533) | motor + semantic (hybrid) | learned latent state (recurrent) | running-summary latent memory | real-time (generation-latency reduction) |
| **DreamX-World 1.0** (2606.16993) | motor + semantic (hybrid) | implicit (state entangled in observations) | stored observations (past-copy memory) | real-time (generation-latency reduction) |
| **GigaWorld-1** (2607.02642) | geometric trajectories | implicit (state entangled in observations) | stored observations (past-copy memory) | near/offline (not play-rate) |
| **BadWAM (attack analysis)** (2607.15207) | motor signals | learned latent state (recurrent) | none (rolling window only) | near/offline (not play-rate) |

## MIRA (Multiplayer Interactive WM) — arXiv:2607.05352 (game (Rocket League, 4-player))
- **D1 action control** — *motor signals*: Per-player controller action streams (9 discrete high-frequency actions); multiplayer conditioning must attribute scene changes to the correct agent.
  - evidence: 'conditions on the action streams of multiple agents, learning to attribute changes in the scene to the correct player'; action probe on frozen DINOv3 features reaches 0.84 mAP over the nine actions.
- **D2 state dynamics** — *implicit (state entangled in observations)*: 5B latent diffusion over a fixed rolling window of T=20 latents (~2 s at 10 Hz). No separate state variable. Privileged game state (ball/car, 50-dim) is logged ONLY as probe ground truth.
  - evidence: 'the model holds a fixed-size window of T = 20 latents, rolling the window by 1 each step'; 'we also log privileged game state... instead it serves as ground truth for evaluation, letting us probe'.
- **D3 persistence** — *none (rolling window only)*: No persistent memory beyond the rolling context. Hour-scale stability comes from diffusion-forcing-style training (noised history), not from any carried state. Probe shows rolled-out car/ball positions track true physics LOCALLY.
  - evidence: 'a game-state probe further confirms that the rolled-out car and ball positions track the true physics'; 'distributional quality holds steady out to five minutes' — quality, not state persistence.
- **D4 real-time** — *real-time (generation-latency reduction)*: 20 fps on a single B200: few-step diffusion distillation + streaming KV-cache + rolling context.
  - evidence: 'generating 20 frames per second on a single Nvidia B200 GPU'.
- **verdict note**: Purest modern example of the state-free design the position paper critiques — and the authors know it: the probe machinery exists precisely because state is implicit and must be verified post-hoc.

## ActWorld — arXiv:2606.17730 (general indoor scenes (synthetic interaction data))
- **D1 action control** — *motor + semantic (hybrid)*: Per-frame WASD/arrow keyboard-mouse control PLUS per-chunk semantic captions (40 action categories, 6 interaction-phase labels via CoT VLM annotation).
  - evidence: 'under per-frame keyboard and mouse control (WASD + arrow keys)'; 'annotate each chunk offline with a dedicated description and the structured labels'.
- **D2 state dynamics** — *implicit (state entangled in observations)*: No explicit state; object state lives only in video latents + memory tokens. The paper itself diagnoses 'action-forgetting' as a memory-design failure, not a missing state variable.
  - evidence: 'recency-biased history compression in existing world models discards the event-transition frames that causally determine subsequent object states'.
- **D3 persistence** — *estimates of the present (updating memory)*: Hierarchical action-aware memory: local bank routes frames by interaction importance (EAFR, contact-phase prior beats recency); persistent bank carries event-update and object-identity (DINOv3 anchor) tokens beyond the eviction horizon.
  - evidence: 'a persistent memory bank that maintains event-update and object-identity tokens across long rollouts'; pixels-to-states itself cites ActWorld in its 'memory as estimates of the present' family.
- **D4 real-time** — *real-time (generation-latency reduction)*: Chunk-AR, 3-step adversarial DMD distillation (Helios recipe); 33-frame chunks (~1.4 s @24 fps) 'land in a fraction of a second'. Real-time claimed; exact FPS not reported.
  - evidence: 'an adversarial DMD-style distillation that reduces the 50-step teacher to a 3-step generator'.
- **verdict note**: Strongest counterexample-adjacent system: it moves memory from past-copy toward present-estimate — but the 'state' is still unverifiable latent tokens, and update decisions are left to learned saliency, exactly the ungrounded-update concern the position paper raises.

## Kairos — arXiv:2606.16533 (physical AI / robot + game inputs)
- **D1 action control** — *motor + semantic (hybrid)*: Joint World-Action MoT: Video DiT + smaller Action DiT generates future action tokens; inputs listed as camera control, natural language, keyboard/mouse, trajectory directives.
  - evidence: 'The Action DiT predicts future action tokens... approximately one-fifth of the Video DiT'; 'camera control, natural language commands, keyboard/mouse operations, or trajectory directives'.
- **D2 state dynamics** — *learned latent state (recurrent)*: Gated Linear Attention state matrix S_t acts as an explicit-architecture, continuously carried latent state with gated delta update; formal bounds claim limited error accumulation. But the state is uninterpretable and not rule-grounded.
  - evidence: 'The Gated Linear Attention (GLA) state matrix S_t acts as a compressed latent memory'; 'the GLA mechanism, with its gated delta updates, serves as a persistent memory that tracks the world state over long horizons'.
- **D3 persistence** — *running-summary latent memory*: SWA (local) + dilated SWA (mid) + GLA (global contractive memory); claims object permanence and 'Necessity of Persistent Latent States' (App. B.2).
  - evidence: 'gated linear attention maintains persistent global memory... mathematically guaranteeing state propagation across extended horizons'.
- **D4 real-time** — *real-time (generation-latency reduction)*: 4-step distilled 480P model reaches real-time on A800; 720P 5 s clip costs 43 s on 1 GPU / 9 s on 4 GPUs — deployment co-designed but not 720p play-rate.
  - evidence: '480P video generation on the Nvidia A800 achieves real-time performance'; '43 seconds on 1 GPU and 9 seconds on 4 GPUs' (720P, 5 s).
- **verdict note**: The most serious challenge to 'state stays implicit': Kairos carries a persistent recurrent state and even proves bounds about it. Yet it matches the position paper's 'learned latents' family verdict: no interpretability, and unreliable for visually-invisible rule variables (remaining HP) since the state is trained by visual prediction only.

## DreamX-World 1.0 — arXiv:2606.16993 (photorealistic + game-style + stylized)
- **D1 action control** — *motor + semantic (hybrid)*: Camera trajectories via E-PRoPE (projective positional encoding, ~30% latency cut vs PRoPE) + composable multi-entity Event Instruction Tuning.
  - evidence: 'Event Instruction Tuning adds composable event control'; 'E-PRoPE... retaining comparable trajectory-following performance while reducing inference latency by approximately 30%'.
- **D2 state dynamics** — *implicit (state entangled in observations)*: Wan2.2-based few-step autoregressive generator; no separate state representation anywhere in the stack.
  - evidence: 'initialized from Wan2.2... converted into a few-step autoregressive world model using causal forcing, DMD-style distillation, and long-rollout training'.
- **D3 persistence** — *stored observations (past-copy memory)*: Memory-Conditioned Scene Persistence: retrieves earlier views by camera geometry; residual recycling tolerates imperfect memory latents. Exactly the past-copy family — good for revisits of static layout, silent on irreversible change.
  - evidence: 'Memory-Conditioned Scene Persistence retrieves earlier views through camera-geometry-based retrieval, while residual recycling makes the conditioning path less sensitive to imperfect memory latents'.
- **D4 real-time** — *real-time (generation-latency reduction)*: Up to 16 FPS on 8x RTX 5090 via mixed-precision DiT, residual reuse, 75%-pruned VAE decode, async pipeline parallelism.
  - evidence: 'reaches up to 16 FPS on eight RTX 5090 GPUs'.
- **verdict note**: Textbook illustration of the position paper's stored-observation critique: geometry-retrieved memory guarantees the past reappears faithfully — including, potentially, a past that should no longer exist (the 'skill demolishes a building, memory restores it intact' failure).

## GigaWorld-1 — arXiv:2607.02642 (robot manipulation (policy evaluator))
- **D1 action control** — *geometric trajectories*: Explicit spatially-aligned control maps — EE-pose map (head view) + ray maps (wrist views) — channel-concatenated with the noisy latent; best of 4 action encodings in controlled comparison.
  - evidence: 'the strongest result comes from channel-concatenated control maps... a unified pixel-aligned representation derived from calibrated robot and camera geometry'.
- **D2 state dynamics** — *implicit (state entangled in observations)*: No symbolic state; headline finding is that evaluator quality is dominated by long-horizon action-faithful consistency, i.e., implicit-state fidelity over rollouts.
  - evidence: 'evaluator quality is dominated by long-horizon, action-faithful rollout consistency rather than short-term visual realism'.
- **D3 persistence** — *stored observations (past-copy memory)*: Hierarchical history buffer: a NEVER-EVICTED first-frame anchor + short/mid/long-range memories. The anchor freezes initial scene identity — the polar opposite of an updating state.
  - evidence: 'the anchor is never evicted during memory updates, each generation step retains access to the original appearance statistics'; 'reliable evaluators require persistent memory for long-horizon rollout' (Finding 10).
- **D4 real-time** — *near/offline (not play-rate)*: Offline closed-loop evaluator; throughput not a design goal (324k simulated rollouts run offline).
  - evidence: 'closed-loop rollout in world models... until task termination' — no real-time claim anywhere.
- **verdict note**: Independently confirms, with 324k rollouts of evidence, the position paper's core empirical premise: long-horizon STATE consistency (not visual realism) is what makes a world model useful — yet its own answer is past-copy memory, not an explicit evolving state.

## BadWAM (attack analysis) — arXiv:2607.15207 (robot WAMs (LIBERO, RoboTwin))
- **D1 action control** — *motor signals*: WAMs output action chunks a_{t:t+H-1} conditioned on observation + instruction (+ imagined future).
  - evidence: 'its world-prediction module first imagines a future, and its action module then produces an action sequence conditioned on that imagination'.
- **D2 state dynamics** — *learned latent state (recurrent)*: The 'state' is the imagined latent/decoded future z_{t+1:t+K} — and the paper shows it can be adversarially desynchronized from the executed action.
  - evidence: 'the model may still produce plausible future imaginations, yet execute actions that cause the task to fail' (96.5% -> 43.1% success under action-only attack).
- **D3 persistence** — *none (rolling window only)*: Not a persistence mechanism paper; chunk-horizon imagination only.
  - evidence: n/a
- **D4 real-time** — *near/offline (not play-rate)*: Runs at policy-eval rate; real-time generation not the subject.
  - evidence: n/a
- **verdict note**: Complements the position paper from the safety side: even when a model carries an explicit imagined future, nothing BINDS action outcomes to that state — the action-state-observation loop is open. Implicit state is not just forgetful; its coupling to action is unverifiable.
