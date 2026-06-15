# μ₀: A Scalable 3D Interaction-Trace World Model — Investigation Notes

Paper: *μ₀: A Scalable 3D Interaction-Trace World Model* (arXiv 2606.13769)  
Authors: Seungjae Lee et al. (UMD + SNU)  
Project page: https://mu0-wm.github.io/ (code marked "Coming soon" as of 2026-06-15)

## Core question

Can a world model learn a reusable, embodiment-agnostic motion prior from unlabeled videos by predicting an intermediate 3D-trace representation instead of dense pixels or robot-specific actions?

## What μ₀ does

μ₀ sits between two extremes:
- **Pixel-space world models** (Cosmos, Sora, etc.) — scalable from video, but waste capacity on appearance and often fail on metric geometry / contact.
- **Vision-Language-Action models** (π₀, π₀.5, etc.) — directly executable, but require embodiment-specific action labels that are expensive and non-transferable.

μ₀ instead predicts **3D interaction traces**: future trajectories of semantic keypoints on objects, tools, hands, and contact regions. The same trace forecast can be consumed by different downstream action experts for different robots.

## Method breakdown

### 1. TraceExtract (offline data engine)

Converts heterogeneous human/robot videos into `(observation, trace, language)` supervision.

| Stage | What it does | Why it matters |
|---|---|---|
| Semantic keypoint sampling | DINOv2 patch features → entity clusters → spatially-diverse keypoints per entity | Avoids background-heavy fixed grids; catches tool tips and contact patches |
| Global-local 3D reconstruction | VGGT sparse global anchors + dense per-chunk reconstruction + SE(3) alignment back to global frame | Handles long egocentric videos and camera motion without compounding drift |
| Progressive 3D tracking | TAPIP3D in world-space, propagating each entity's last known 3D position across chunk boundaries | Preserves identities across chunks |
| Event-centric captioning | Savitzky-Golay smoothed acceleration → peaks/valleys → motion chunks → VLM captions start/mid/end frames | Binds language to local motion events, not just whole episodes |

Output tuples: `(I_t, l_c, Q_t, T_ref^{t-h:t+H})`, where `Q_t` is a variable set of semantic query keypoints and `T_ref` is their reference-frame 3D trace.

### 2. μ₀ model

- **Backbone**: pretrained SmolVLM2-2.2B prefix (frozen), with an optional metric-depth pathway through a separate stem that shares deeper SigLIP layers.
- **Trace Expert**: 20-layer transformer, 0.5× VLM width, permutation-equivariant over the unordered query set. Cross-attends to VLM KV cache.
- **Target representation**: each query's future is represented as **D=10 cubic B-spline control points** (anchor-relative, per-axis rescaled). This replaces dense H=32 waypoints.
- **Training objective**: conditional flow matching over control points, plus:
  - validity/done prediction for occlusion/track loss,
  - rigidity loss using DINO cluster identities to preserve intra-part geometry.
- **Inference**: 4-step Euler integration on the control-point velocity field, then decode via B-spline basis.

### 3. Trace-conditioned action expert

After video-only pretraining, μ₀ is **frozen**. An action expert is trained on robot demonstrations that:
1. takes a single partial-denoising step of μ₀ starting from noise,
2. extracts intermediate trace-expert hidden states as motion tokens,
3. fuses them into VLM features via gated cross-attention,
4. predicts continuous action chunks via flow matching (π₀.5-style architecture).

This keeps the motion prior embodiment-agnostic while limiting action supervision to the target robot head.

## Reconstruction probe

Because the official code is not yet released, I reconstructed the B-spline trace target from the paper (Appendix B.2) and verified that a degree-3 B-spline with D=10 control points accurately compresses an H=32 noisy 3D trace.

Script: `code/mu0_trace_recon.py` (copied into this thread's `code/`).  
Artifact: `assets/mu0_bspline_recon.png`

Key result on synthetic spiral+noise data:
- raw trace: 32 × 3 = 96 scalars per query
- B-spline target: 10 × 3 = 30 scalars per query  
- reconstruction MSE ≈ 0.0028, confirming the compactness/smoothness trade-off claimed in the paper.

## Results and claims

### Trace prediction
μ₀ reports best top-5 ADE/FDE/DTW in both 2D and 3D on the paper's trace benchmark. Inference latency is 0.29 s on an A6000, ~2.9× faster than Track2Act (0.85 s).

### Downstream robot control
- **RoboCasa365 sim**: 30.25% average success, above π₀ (25.25%) and TraceGen+action-expert (23%), but below π₀.5 (42%). The authors note π₀.5 is not data-matched because it uses large-scale action-labeled pretraining.
- **Real UR3 tasks**: 91.7% average success, surpassing π₀ (+20.0 pp), π₀.5 (+11.7 pp), and TraceGen (+10.0 pp).

### Scaling
Both model scaling (342M → 568M → 2.59B) and data scaling (5% → 20% → 100%) show monotonic improvements on top-5 DTW, suggesting the trace objective is not yet saturated.

## Critical notes / open questions

1. **No code released yet.** The project page lists "Code (Coming soon!)". The method reconstruction above is based on the paper and appendix; the real implementation details (e.g., exact VGGT/TAPIP3D hyperparameters, the TraceExtract pipeline code, the action-expert checkpoint) cannot be verified independently today.
2. **Supervision pipeline is complex.** TraceExtract depends on VGGT reconstruction, DINOv2 clustering, TAPIP3D tracking, and a VLM captioner. Errors in any stage feed into training. The paper acknowledges this as a limitation.
3. **Real-world evaluation is narrow.** Three tabletop tasks on one UR3 setup. Claims about cross-embodiment scalability await broader validation (mobile manipulators, dexterous hands, longer horizons).
4. **Forces and contact modes are not explicit.** The trace representation captures geometry and motion, not force/tactile information, which may limit fine manipulation.
5. **Comparison to π₀.5 is apples-to-oranges on data.** The paper is honest about this, but it means the headline "competitive with VLAs despite no action supervision" is strongest on the real-robot subset, not the simulation benchmark.

## Relation to the run's other thread

HyVLA-0.5 (the other deep-dive) takes the opposite pretraining philosophy: it relies on 10K hours of high-precision, action-labeled human demonstrations collected with custom hardware. μ₀ instead tries to learn motion from unlabeled video and only uses action labels at the small action-expert head. The two papers bracket the current debate on whether scalable robot learning needs large action-labeled corpora or can bootstrap from video.

## Artifacts preserved

- `code/mu0_trace_recon.py` — small NumPy/SciPy reconstruction of the B-spline trace target.
- `../assets/mu0_bspline_recon.png` — figure showing raw 3D trace vs. B-spline reconstruction.
