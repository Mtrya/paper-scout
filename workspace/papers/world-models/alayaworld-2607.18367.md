
##### Report GitHub Issue


Content selection saved. Describe the issue below:


# AlayaWorld: Interactive Long-Horizon World Modeling - Full Technical Report



###### Abstract


Unlike conventional video game development, which relies on labor-intensive pipelines for asset production, animation, physics, and programming, video world models generate interactive environments from user inputs instantly. It enable us to create customized, explorable, and continuously evolving virtual world from text, an image, or video. Realizing this vision requires four tightly coupled capabilities: interaction, persistent spatiotemporal consistency, stable long-horizon generation, and efficient response. We present AlayaWorld , an interactive long-horizon video world model that generates 24-fps video at 540p and 720p. Built on a 15B video diffusion transformer, AlayaWorld generates short latent chunks autoregressively under camera trajectories and switchable text prompts. Its bounded visual context combines a persistent sink frame, compressed temporal history, geometry-aligned spatial memory, and recent-frame conditioning. To reduce long-term drift, the model is trained with corrupted histories and prediction residuals collected from its own roll-outs. We further introduce a discrete autoregressive distillation formulation that combines distribution-matching distillation, self-forcing++, and consistency distillation, reducing inference from approximately 30 sampling steps to four steps per chunk. On iWorld-Bench, AlayaWorld achieves the best performance over long-horizon generation. Conceived as a full-stack, open-source, and long-term project, AlayaWorld is intended to provide an extensible foundation for future research on interactive video world models.

https://alaya-lab.github.io/AlayaWorld/ \Code https://github.com/AlayaLab/AlayaWorld \video https://www.youtube.com/watch?v=n0jIEg7taTI \contact kaipeng.zhang@shanda.com

**Figure 1 : Interactive world simulation across diverse scenes.**


## 1 Introduction


Reality be rent. Synapse break. Banishment, this world!

— Watch Love, Chunibyo & Other Delusions

The dream of engaging with a reality unconstrained by physical laws has captivated humanity for generations. Among existing forms of entertainment, video games have come closest to realizing this vision by creating interactive virtual worlds that respond dynamically to player actions. However, modern 3D game development depends on a prolonged and tightly coupled production pipeline involving concept design, asset creation, animation, rendering, physics, gameplay programming, testing, and optimization. The resulting time and labor costs make personalized, rapidly evolving, and effectively unbounded interactive worlds prohibitively expensive.

Video world models offer a fundamentally different route mao2025yume ; Mao_2026_CVPR ; he2025matrix ; hyworld2025 . Rather than manually creating assets and programming interaction rules, such models generate virtual worlds directly from text, images, videos, and user controls. Visual appearance, motion, viewpoint changes, and certain forms of interaction are encoded jointly in the model parameters and realized through a continuous generative process.

Turning a video generator into a genuinely interactive world model, however, requires four tightly coupled capabilities. Interaction requires the model to respond accurately to camera trajectories and evolving user intent. Consistency requires the world to preserve its spatial structure and visual identity across viewpoint changes and delayed revisits. Stability requires long autoregressive roll-outs without progressively accumulating blur, illumination shifts, or geometric drift. Efficiency requires sufficiently low generation latency and rapid response to new control signals. These capabilities cannot be addressed independently: broader interaction makes consistency harder to preserve, longer roll-outs amplify residual errors, and aggressive acceleration may further compromise visual stability.

In this work, we present AlayaWorld , an interactive long-horizon video world model. AlayaWorld is built on a 15B video diffusion transformer and generates 24-fps video at 540p and 720p. It synthesizes the world autoregressively in short latent chunks under continuous camera trajectories and dynamically switchable text prompts, supporting both controllable navigation and prompt-driven open-ended actions. To preserve scene information over long roll-outs, AlayaWorld uses a bounded visual context that combines a global scene anchor, compressed recent history, geometry-aligned spatial memory, and recent-frame conditioning. Temporal memory maintains local dynamics and frame-to-frame continuity, while spatial memory stores past observations and reprojects them into the target view when previously visited regions are revisited. Because the context size remains bounded, the model generates each new chunk with approximately constant computational cost as the roll-out grows.

AlayaWorld is trained in three stages. We first adapt a general bidirectional video prior to world-modeling data composed of real-world videos, gameplay recordings, and generated events. The model is then converted into an autoregressive generator equipped with camera control and spatiotemporal memory. To mitigate long-horizon drift, we train with corrupted histories and replay prediction residuals collected from the model’s own roll-outs, teaching it to recover from imperfect context. Finally, we introduce a discrete autoregressive distillation method that combines distribution-matching distillation, self-forcing++, and consistency distillation, reducing inference from approximately 30 sampling steps to four steps per chunk while preserving the complete control and memory stack.

We evaluate AlayaWorld on iWorld-Bench fang2026iworld across generation quality, trajectory following, and memory ability. AlayaWorld achieves the best overall performance. Qualitative results further demonstrate controllable navigation, consistent revisitation, prompt-driven actions, and stable long-horizon generation across diverse scenes.

AlayaWorld still represents the world primarily through visual observations, estimated geometry, and visual memory. Its understanding of object state, physical causality, and long-term task structure therefore remains limited to their visible consequences.

Conceived as a full-stack, open-source, and long-term project, AlayaWorld is intended to provide an extensible foundation for future research on interactive video world models.


## 2 Training Data


Data is the interface through which a video world model learns two coupled capabilities: photorealistic scene evolution and controllable camera-conditioned navigation. We therefore construct a corpus that is deliberately heterogeneous along three axes: visual domain, motion geometry, and supervision fidelity. The corpus combines real-world captures , which anchor the model in natural appearance, scene layout, and capture artifacts, with synthetic renderings , which provide scalable access to controlled camera motion, long-tail interactions, and action-driven dynamics. This design follows recent video world-model pipelines that mix real, synthetic, and action-conditioned data to improve both appearance diversity and controllability ( li2025sekai ; mao2025yume ; he2025cameractrl2 ) .

All sources are normalized into a single training record consisting of a video, per-frame camera intrinsics and pose when available or recoverable, and a hierarchical caption aligned to the clip timeline. The resulting corpus contains 222 147 222\,147 clips from seven sources, including two internally curated sources (MUGEN and GameVerse), summarized in Table 1 . Figure 2 illustrates the same mixture qualitatively. Each row shows temporally ordered frames from one clip, so the row should be read as a short trajectory rather than as independent images. Horizontally, the samples expose the camera motions the model is trained to follow: forward walking, indoor traversal, panoramic sweeps, third-person navigation, and event-centric viewpoint changes. Vertically, they show the domain coverage induced by the mixture, spanning crowded streets, domestic interiors, real-estate walkthroughs, panoramic reprojections, game-engine renderings, and generative event videos.

**Figure 2 : Representative training samples from the seven data sources, grouped by source type. Each row contains four temporally ordered frames from a single clip, making both the appearance diversity of the corpus and the camera motion available for supervision visible within one view.**


### 2.1 Data Mixture



#### Real-world captures.


The real-world portion of the corpus supplies physical appearance, natural camera shake, sensor artifacts, and scene layouts that are difficult to cover with simulation alone. We aggregate five complementary sources. Sekai-Real ( li2025sekai ) contributes first-person urban walking trajectories; SpatialVid ( wang2025spatialvid ) adds short clips with dense indoor camera motion; RealEstate10K ( zhou2018realestate10k ) provides real-estate walkthroughs with broad indoor coverage; and DL3DV ( ling2024dl3dv ) contributes long, contiguous multi-view walkthroughs of real environments. We keep DL3DV as a standalone source because its clips are scene-level traversals rather than fixed-length Internet-video segments. Finally, MUGEN is our internally curated dataset, built from YouTube videos, with various camera trajectories and accurate annotation. For all real-world sources without camera metadata, we recover per-frame intrinsics and poses using ViPE ( huang2025vipe ) ;


#### Synthetic renderings.


The synthetic portion complements real captures with camera and action control that is difficult to obtain at Internet scale. GameVerse is our internal large-scale gameplay corpus, which contains 124 116 124\,116 clips of approximately 66 s 66\text{\,}\mathrm{s} each. GenEvent contributes 6490 6490 event-centric clips, averaging 10.6 s 10.6\text{\,}\mathrm{s} , synthesized by a generative video model. These clips expose the model to open-domain, action-triggered events that are underrepresented in both curated real-world captures and game footage.

**Table 1: Training data composition. The mixture balances real captures for photorealism and geometry with synthetic renderings for scalable camera and action controllability. † denotes internally curated sources.**

| Source | Type | #Clips |
| Sekai-Real (walking) | real, FPV | 21 561 21\,561 |
| SpatialVid | real, indoor | 23 210 23\,210 |
| RealEstate10K | real, indoor | 17 429 17\,429 |
| DL3DV | real, walkthrough | 7905 7905 |
| MUGEN † | real, FPV | 21 436 21\,436 |
| GameVerse † | synthetic, game | 124 116 124\,116 |
| GenEvent † | synthetic, event | 6490 6490 |
| Total |  | 222 147 222\,147 |


### 2.2 Curation Pipeline


A heterogeneous corpus is only useful if quality decisions are made consistently across sources. We therefore use a unified, stage-based filtering pipeline rather than source-specific heuristics. The pipeline operates over a shared manifest in three phases: ingest builds the manifest, run evaluates all filtering stages over a dependency DAG, and select applies the final all-stages-must-pass rule. Expensive features are computed once, cached in the manifest, and reused by downstream gates; optional global rank-cut and near-duplicate dedup stages can be inserted before final selection.


#### Shared feature cache.


The filtering backbone uses a single-decode design. For each clip, one NVDEC pass decodes frames and one RAFT ( teed2020raft ) forward pass estimates optical flow. From these outputs we compute frame-level summaries, including luminance, frame differences, and border statistics, together with flow-derived motion features such as flow-per-second, temporal variance, and directional variance. All rule-based stages read these cached scores at millisecond cost, avoiding repeated video decoding and making large-scale threshold sweeps practical.


#### Filtering stages.


Each clip must pass six classes of gates. First, technical validation checks decode integrity, minimum resolution ( ≥ 720 \geq\!720 p), duration ( ≥ 3 \geq\!3 s), frame rate ( 24 24 – 65 65 fps), and codec membership (H.264/HEVC/AV1). Second, photometric validation rejects clips with extreme exposure or excessive black borders, using a maximum border ratio of 0.10 0.10 . Third, shot-boundary filtering combines classical cut/dissolve detection with OmniShotCut ( wang2026omnishotcut ) , ensuring that each retained training sample is a single continuous shot. Fourth, motion analysis removes static clips, scores temporal consistency, assigns camera-motion buckets (static, pan, gameplay, mixed), and detects pose-free camera shake. Fifth, text and interface suppression removes subtitles, watermarks, and game HUD overlays using EasyOCR-based text detection ( jaidedai2020easyocr ) and a pixel-stability UI mask with maximum overlay ratio 0.04 0.04 . Sixth, person control applies a YOLO11 ( khanam2024yolo11 ) detector to bound foreground-human count and screen occupancy.

For sources with estimated camera trajectories, we additionally apply a pose stability gate. The gate scores trajectory jitter, peak acceleration, median reconstruction residual, and long-horizon drift, with source-specific anchors because walking videos and game footage occupy different motion scales. Perceptual-quality estimators (COVER ( he2024cover ) , VBench ( huang2024vbench ) ) and SigLIP2/CLIP/V-JEPA2 embeddings ( tschannen2025siglip2 ; radford2021clip ; assran2025vjepa2 ) support the optional global rank-cut and near-duplicate dedup stages.


#### Selection protocol.


The final training set consists only of clips that pass every active gate; these surviving clips form the *_filtered splits reported in Table 1 . We calibrate thresholds per source by profiling a ∼ 200 \sim\!200 -clip slice before enabling a gate, since score distributions differ substantially across capture conditions and game genres. This calibration makes filtering strict enough to remove failure modes while preserving source-specific motion statistics.


### 2.3 Training-Oriented Caption Annotation


A single global caption is insufficient for training a controllable world model: at inference time the model receives localized, time-varying instructions, while a clip-level caption provides no aligned temporal supervision. We therefore annotate each clip with a two-level schema that explicitly separates global context from segment-level dynamics. Captions are generated by a vision-language model, with Kimi-K2.6 ( kimiteam2026kimi26 ) as the default backend and Gemini ( comanici2025gemini25 ) and Gemma ( gemmateam2025gemma3 ) as alternatives. The model consumes frames sampled at 1 1 – 2 2 fps, each tagged with an explicit [mm:ss] timestamp, which encourages temporal segmentation rather than a single collapsed description.


#### Video-level context.


At the clip level, we annotate a compact set of global attributes: weather, time of day, location type, camera perspective, camera motion, and video style. These attributes are drawn from a deliberately small vocabulary, reduced from 59 to 26 values, so they can serve both as conditioning tokens and as reliable keys for data balancing, for example by location_type or camera_perspective .


#### Segment-level tracks.


Each clip is partitioned into timestamped segments, represented by time_range_s . Within each segment, we annotate separate semantic tracks rather than a single entangled sentence: primary subject motion, environmental dynamics, static scene attributes, and camera viewpoint/motion (Table 2 ). The separation between subject and camera tracks is crucial. It allows the annotation to distinguish, for example, a walking subject from a dolly-in camera, an orbit from a turning agent, or a static subject under a panning viewpoint. The descriptive tracks are fused into a natural-language full_prompt of 5–9 present-tense sentences for the text encoder, together with a 15–45-word short_prompt used for caption dropout and multi-caption augmentation.

**Table 2: Per-segment annotation tracks. The descriptive tracks are fused into full_prompt and short_prompt ; camera_path provides a discrete camera-trajectory control signal independent of subject motion.**

| Field | Type | Role |
| subject_motion | free text | primary-agent motion |
| environment_motion | free text | dynamics of entities, lighting, and weather |
| static_scene | free text | time-invariant scene attributes |
| camera_description | free text | viewpoint, framing, motion, and stability |
| full_prompt | free text | fused caption for the text encoder |
| short_prompt | free text | compact caption for dropout and augmentation |
| camera_path | enum (16) | discrete camera-trajectory control target |


## 3 AlayaWorld


AlayaWorld is an interactive world model built on top of the LTX-2.3. The public LTX-2.3 checkpoint is a 22 22 B multimodal model; we remove its audio module, leaving a ∼ \sim 13B video DiT that constitutes our backbone.

Training proceeds in three stages: (i) bidirectional pre-training , which adapts the general video prior to our domain with a full-parameter fine-tune; (ii) autoregressive training , which grafts a history-compression module, a spatial memory, and camera control onto the backbone, and stabilises long roll-outs with anti-drift training; and (iii) post-training acceleration , which distils the many-step teacher into a 4 4 -step student via Distribution-Matching Distillation (DMD) combined with self-forcing++ and consistency distillation.

**Figure 3 : Training stages.**


### 3.1 Formulation


AlayaWorld generates video autoregressively in the VAE latent space , chunk by chunk. The causal video VAE encodes a clip into a latent sequence partitioned into chunks { z 1 , z 2 , … } \{z_{1},z_{2},\dots\} , each a block of K = 4 K{=}4 latent frames. At step i i the control input is the chunk’s target camera trajectory π i \pi_{i} — a sequence of absolute camera poses, one per latent frame of the chunk — together with an optional chunk-level text prompt y i y_{i} ; a prompt switched in at a chunk boundary drives prompt-driven actions such as combat or spell-casting. Generation factorises causally as

|  | p θ ​ ( z 1 : N ∣ π 1 : N , y 1 : N ) = ∏ i = 1 N p θ ​ ( z i ∣ z < i , π ≤ i , y i ) , p_{\theta}\!\left(z_{1:N}\mid\pi_{1:N},y_{1:N}\right)=\prod_{i=1}^{N}p_{\theta}\!\left(z_{i}\mid z_{<i},\,\pi_{\leq i},\,y_{i}\right), |  | (1) |

so every chunk sees only the past ( z < i , π ≤ i z_{<i},\pi_{\leq i} ) and never leaks future information. The two conditioning modalities enter by different routes: the camera trajectory is injected as a compact per-frame condition — the relative pose between consecutive frames — through an adaptive layer-norm (AdaLN) camera-control module, whereas the visual past z < i z_{<i} enters as an in-context token prefix .

**Figure 4 : Formulation.**

Conditioning as an in-context prefix. The backbone is a single self-attention DiT. The visual context is not supplied through a separate cross-attention bank; instead, for chunk i i the model prepends four clean ( σ = 0 \sigma{=}0 , noise-free) conditioning streams to the K K noised target frames, forming one token sequence

|  | S i = [ s ⏟ sink ; h i ⏟ temporal memory ; g i ⏟ spatial memory ; n i ⏟ nearby / I2V ; z i τ ⏟ target ] , S_{i}=\big[\;\underbrace{s}_{\text{sink}}\;;\;\underbrace{h_{i}}_{\text{temporal memory}}\;;\;\underbrace{g_{i}}_{\text{spatial memory}}\;;\;\underbrace{n_{i}}_{\text{nearby / I2V}}\;;\;\underbrace{z_{i}^{\tau}}_{\text{target}}\;\big], |  | (2) |

which is processed by full (non-causal) self-attention; the whole prefix is sliced off after the last transformer block and only the target segment is denoised. The four streams differ in source and role:

- • Sink s s — a single clean latent frame, patch-embedded and pinned at RoPE temporal position 0 , held fixed across all chunks as a global identity/appearance anchor. During training it is drawn as a remote frame (at least 8 8 latent frames from the target), which prevents the model from extrapolating the next chunk directly from it and thereby increases its reliance on the camera-control signal.

Sink s s — a single clean latent frame, patch-embedded and pinned at RoPE temporal position 0 , held fixed across all chunks as a global identity/appearance anchor. During training it is drawn as a remote frame (at least 8 8 latent frames from the target), which prevents the model from extrapolating the next chunk directly from it and thereby increases its reliance on the camera-control signal.

- • Temporal memory h i = H ϕ ​ ( w i ) h_{i}=H_{\phi}(w_{i}) — a compressed temporal history . A history-compression module H ϕ H_{\phi} (following Frame Preservation) encodes a sliding window w i = z i − L : i w_{i}=z_{i-L:i} of the last L = 6 L{=}6 latent frames into a lightweight embedding, recomputed every chunk; its tokens are injected directly, bypassing the patch embedder.

Temporal memory h i = H ϕ ​ ( w i ) h_{i}=H_{\phi}(w_{i}) — a compressed temporal history . A history-compression module H ϕ H_{\phi} (following Frame Preservation) encodes a sliding window w i = z i − L : i w_{i}=z_{i-L:i} of the last L = 6 L{=}6 latent frames into a lightweight embedding, recomputed every chunk; its tokens are injected directly, bypassing the patch embedder.

- • Spatial memory g i g_{i} — a geometry-aligned rendering of past views into the current view (Eq. ( 3 )), giving the generator concrete visual evidence for the queried viewpoint.

Spatial memory g i g_{i} — a geometry-aligned rendering of past views into the current view (Eq. ( 3 )), giving the generator concrete visual evidence for the queried viewpoint.

- • Nearby / I2V condition n i n_{i} — the single most-recent latent frame (the last frame of w i w_{i} ), patch-embedded and placed immediately before the target; this is the image-to-video conditioning frame that carries full-resolution frame-to-frame continuity.

Nearby / I2V condition n i n_{i} — the single most-recent latent frame (the last frame of w i w_{i} ), patch-embedded and placed immediately before the target; this is the image-to-video conditioning frame that carries full-resolution frame-to-frame continuity.

Building and rendering the spatial memory. Following GEN3C, the spatial memory maintains an explicit cache ℬ = { ( I j , D j , π j ) } \mathcal{B}=\{(I_{j},D_{j},\pi_{j})\} — each previously generated frame I j I_{j} with its monocular depth D j D_{j} (Depth-Anything-3) and camera pose π j \pi_{j} , keyed by global frame index — and renders it along the target camera trajectory. The AdaLN camera condition is the per-frame increment of the same trajectory, so the cache and the camera control share one pose source. To render the cache into chunk i i (target camera π i \pi_{i} ):

- 1. Retrieve up to 10 10 frames by greedy maximum-coverage selection: each candidate’s depth is unprojected to world points and projected into π i \pi_{i} ; a z-buffer with occlusion tolerance δ = 0.1 \delta{=}0.1 marks which target pixels it covers, and frames are chosen to maximise the number of newly-covered pixels.

Retrieve up to 10 10 frames by greedy maximum-coverage selection: each candidate’s depth is unprojected to world points and projected into π i \pi_{i} ; a z-buffer with occlusion tolerance δ = 0.1 \delta{=}0.1 marks which target pixels it covers, and frames are chosen to maximise the number of newly-covered pixels.

- 2. Warp each selected frame into the target view by forward splatting, u ′ = π i ​ ( π j − 1 ​ ( u , D j ​ ( u ) ) ) , u^{\prime}=\pi_{i}\!\big(\,\pi_{j}^{-1}(u,\,D_{j}(u))\,\big), (3) i.e. unproject pixel u u of frame j j to a 3D point via its depth and camera, then reproject into π i \pi_{i} ; per-pixel occlusion is resolved by nearest depth and multiple sources are fused into one warped image I ~ i \tilde{I}_{i} plus a binary coverage mask M i M_{i} .

Warp each selected frame into the target view by forward splatting,

|  | u ′ = π i ​ ( π j − 1 ​ ( u , D j ​ ( u ) ) ) , u^{\prime}=\pi_{i}\!\big(\,\pi_{j}^{-1}(u,\,D_{j}(u))\,\big), |  | (3) |

i.e. unproject pixel u u of frame j j to a 3D point via its depth and camera, then reproject into π i \pi_{i} ; per-pixel occlusion is resolved by nearest depth and multiple sources are fused into one warped image I ~ i \tilde{I}_{i} plus a binary coverage mask M i M_{i} .

- 3. Inject : I ~ i \tilde{I}_{i} is VAE-encoded to g i g_{i} and placed at the target’s RoPE coordinates; the coverage mask M i M_{i} becomes a self-attention key bias so uncovered (i.e. never-observed) regions are ignored rather than trusted.

Inject : I ~ i \tilde{I}_{i} is VAE-encoded to g i g_{i} and placed at the target’s RoPE coordinates; the coverage mask M i M_{i} becomes a self-attention key bias so uncovered (i.e. never-observed) regions are ignored rather than trusted.

- 4. Update : once chunk i i is generated it is decoded to pixels, its depth is estimated, and ( I i , D i , π i ) (I_{i},D_{i},\pi_{i}) is appended to ℬ \mathcal{B} .

Update : once chunk i i is generated it is decoded to pixels, its depth is estimated, and ( I i , D i , π i ) (I_{i},D_{i},\pi_{i}) is appended to ℬ \mathcal{B} .

This gives long-range spatial consistency — revisiting a previously seen place renders it consistently — well beyond the 6 6 -frame temporal window. Writing c i = ( s , h i , g i , n i , π ≤ i , y i ) c_{i}=(s,h_{i},g_{i},n_{i},\pi_{\leq i},y_{i}) for the full per-chunk conditioning — where the four context streams s , h i , g i , n i s,h_{i},g_{i},n_{i} are all functions of the past z < i z_{<i} — the causal factor p θ ​ ( z i ∣ z < i , π ≤ i , y i ) p_{\theta}(z_{i}\mid z_{<i},\pi_{\leq i},y_{i}) of Eq. ( 1 ) is realised as p θ ​ ( z i ∣ c i ) p_{\theta}(z_{i}\mid c_{i}) .

Roll-out. One interactive step is therefore: (1) read the camera trajectory π i \pi_{i} (and optional prompt y i y_{i} ); (2) build the prefix S i S_{i} of Eq. ( 2 ) — pin the sink s s , recompute the temporal memory h i h_{i} over the last 6 6 latents, render the spatial memory into g i g_{i} , and set the nearby frame n i n_{i} ; (3) sample the next chunk z ^ i ∼ p θ ( ⋅ ∣ c i ) \hat{z}_{i}\sim p_{\theta}(\cdot\mid c_{i}) ; (4) stream-decode z ^ i \hat{z}_{i} to pixels, slide the history window forward, and append the new frames to the spatial-memory cache ℬ \mathcal{B} ; (5) advance i ← i + 1 i\leftarrow i+1 . Because the context is a bounded rolling window (a fixed sink, a 6 6 -frame history, and a capped set of 10 10 rendered cache frames), the compute per chunk is constant and the horizon N N is in principle unbounded, giving arbitrarily long interactive generation.


### 3.2 Bidirectional Model Pre-Training – Establishing the General Video Prior


The base LTX-2.3 model is a general text-to-video prior. This stage adapts it to our world-model domain through a full-parameter fine-tune, so that the whole backbone absorbs the new visual and temporal statistics while retaining the base model’s generative prior. The model stays fully bidirectional here; no memory or control mechanism is introduced yet. Fine-tuning runs at 24 24 fps under a mix of 540 540 p and 720 720 p resolutions, on variable-length clips of up to 20 20 seconds — the temporal range the base model was trained on. To prime the backbone for the conditional generation used later, clips are trained under a mixture of image-, video-, and text-conditioned objectives.

The training data is a weighted mixture dominated by a balanced scene / camera-pose corpus, complemented by AAA-gameplay recordings, real first-person walkthroughs, and magic-event clips for prompt-driven actions such as combat, spell casting, and monster summoning; a per-sample weighted sampler balances the dominant source. Text conditioning is chosen by the length of the underlying video segment: segments longer than 20 20 seconds are described only by a single high-level whole-clip summary, whereas segments within 20 20 seconds randomly alternate between that summary and a detailed per-segment caption. In either case the descriptive content inside the <camera> tag is randomly dropped, so that the model does not over-rely on the textual camera description — camera control is instead supplied by a dedicated signal in the later stages. We adopt an adaptive sigma-shift schedule whose flow-matching timestep shift scales with clip length, allocating the denoising budget appropriately across durations, and close the stage with a short low- σ \sigma refinement pass that sharpens fine detail.


### 3.3 Autoregressive Model Training – Control and Memory Integration


This stage turns the bidirectional generator into an autoregressive, controllable world model that rolls out chunk-by-chunk, conditioned on its own past, a persistent memory, and a user-supplied camera trajectory. It is carried out in two phases with different training regimes.

History pre-training. The first phase keeps the backbone frozen and trains, via a LoRA adapter, the history-compression module H ϕ H_{\phi} — which, following Frame Preservation , compresses the recent frame history into a lightweight embedding. Given a history window { z 1 , … , z L } \{z_{1},\dots,z_{L}\} , we mask it by perturbing each frame with its own noise level, z ~ i = ( 1 − σ i ) ​ z i + σ i ​ ϵ i \tilde{z}_{i}=(1-\sigma_{i})z_{i}+\sigma_{i}\epsilon_{i} with σ i ∼ 𝒰 ​ ( 0.2 , 1 ) \sigma_{i}\sim\mathcal{U}(0.2,1) , and ask the backbone (a rectified-flow velocity field v θ v_{\theta} ) to reconstruct a short target window z Ω z_{\Omega} conditioned on the compressed masked history, under the flow-matching objective

|  | ℒ 2a = 𝔼 ∥ v θ ( z Ω τ , τ ∣ H ϕ ( z ~ ) ) − ( ϵ − z Ω ) ∥ 2 2 , z Ω τ = ( 1 − τ ) z Ω + τ ϵ , τ ∼ 𝒰 ( 0 , 1 ) . \mathcal{L}_{\text{2a}}=\mathbb{E}\big\|\,v_{\theta}\!\big(z_{\Omega}^{\tau},\tau\mid H_{\phi}(\tilde{z})\big)-(\epsilon-z_{\Omega})\big\|_{2}^{2},\qquad z_{\Omega}^{\tau}=(1-\tau)z_{\Omega}+\tau\epsilon,\ \ \tau\sim\mathcal{U}(0,1). |  | (4) |

The target-window length is varied so the module serves both short and long horizons. This phase uses no control signal, pre-training the memory pathway in isolation.

Full-stack fine-tuning. Starting from the history-pretrained weights, the second phase is a full-parameter supervised fine-tune: the backbone is unfrozen and trained in full, together with three dedicated modules — the history-compression module, the camera-control module, and a next forcing head xu2026next . Its components are:

- • Camera control. Each per-frame relative-pose increment Δ ​ π \Delta\pi of the camera trajectory is Fourier-embedded per axis, concatenated over the six pose components, and passed through an MLP; the result modulates the tokens through AdaLN: c cam = MLP ​ ( ⨁ k = 1 6 PE ​ ( Δ ​ π k ) ) , e ← e + c cam , c_{\text{cam}}=\mathrm{MLP}\Big(\textstyle\bigoplus_{k=1}^{6}\mathrm{PE}(\Delta\pi_{k})\Big),\qquad e\leftarrow e+c_{\text{cam}}, (5) where e e is the timestep embedding from which the AdaLN scale and shift are produced. Per-axis scales are calibrated from the motion statistics of the real data.

Camera control. Each per-frame relative-pose increment Δ ​ π \Delta\pi of the camera trajectory is Fourier-embedded per axis, concatenated over the six pose components, and passed through an MLP; the result modulates the tokens through AdaLN:

|  | c cam = MLP ​ ( ⨁ k = 1 6 PE ​ ( Δ ​ π k ) ) , e ← e + c cam , c_{\text{cam}}=\mathrm{MLP}\Big(\textstyle\bigoplus_{k=1}^{6}\mathrm{PE}(\Delta\pi_{k})\Big),\qquad e\leftarrow e+c_{\text{cam}}, |  | (5) |

where e e is the timestep embedding from which the AdaLN scale and shift are produced. Per-axis scales are calibrated from the motion statistics of the real data.

- • Temporal memory. The history-compression module from the first phase is kept trainable and recomputed each chunk over the sliding history window, with history-dropout for robustness.

Temporal memory. The history-compression module from the first phase is kept trainable and recomputed each chunk over the sliding history window, with history-dropout for robustness.

- • Spatial memory. Following GEN3C, an explicit cache stores previously generated frames with their monocular depth (Depth-Anything-3) and camera pose, and renders them into the current view by maximum-coverage retrieval, giving long-range spatial consistency beyond the history window.

Spatial memory. Following GEN3C, an explicit cache stores previously generated frames with their monocular depth (Depth-Anything-3) and camera pose, and renders them into the current view by maximum-coverage retrieval, giving long-range spatial consistency beyond the history window.

- • Next forcing . An auxiliary head reinforces frame-to-frame causal continuity by predicting the next chunk from the backbone’s intermediate features: hidden states hooked from several layers are fused into a feature F F and, together with the noised next chunk, decoded by a small head f ψ f_{\psi} into a velocity, supervised at a shifted (higher) noise level τ ~ = 10 ​ τ 1 + 9 ​ τ \tilde{\tau}=\tfrac{10\,\tau}{1+9\,\tau} , ℒ nf = ‖ f ψ ​ ( F , z 0 + , τ ~ , τ ~ ) − ( ϵ − z 0 + ) ‖ 2 2 , ℒ = ℒ flow + 0.5 ​ ℒ nf , \mathcal{L}_{\text{nf}}=\big\|\,f_{\psi}\big(F,\ z^{+,\tilde{\tau}}_{0},\ \tilde{\tau}\big)-(\epsilon-z^{+}_{0})\big\|_{2}^{2},\qquad\mathcal{L}=\mathcal{L}_{\text{flow}}+0.5\,\mathcal{L}_{\text{nf}}, (6) where z 0 + z^{+}_{0} is the next chunk and ℒ flow \mathcal{L}_{\text{flow}} is the main flow-matching objective of the chunk being generated.

Next forcing . An auxiliary head reinforces frame-to-frame causal continuity by predicting the next chunk from the backbone’s intermediate features: hidden states hooked from several layers are fused into a feature F F and, together with the noised next chunk, decoded by a small head f ψ f_{\psi} into a velocity, supervised at a shifted (higher) noise level τ ~ = 10 ​ τ 1 + 9 ​ τ \tilde{\tau}=\tfrac{10\,\tau}{1+9\,\tau} ,

|  | ℒ nf = ‖ f ψ ​ ( F , z 0 + , τ ~ , τ ~ ) − ( ϵ − z 0 + ) ‖ 2 2 , ℒ = ℒ flow + 0.5 ​ ℒ nf , \mathcal{L}_{\text{nf}}=\big\|\,f_{\psi}\big(F,\ z^{+,\tilde{\tau}}_{0},\ \tilde{\tau}\big)-(\epsilon-z^{+}_{0})\big\|_{2}^{2},\qquad\mathcal{L}=\mathcal{L}_{\text{flow}}+0.5\,\mathcal{L}_{\text{nf}}, |  | (6) |

where z 0 + z^{+}_{0} is the next chunk and ℒ flow \mathcal{L}_{\text{flow}} is the main flow-matching objective of the chunk being generated.

The data mixture follows the pre-training stage, with the per-source proportions retuned for this phase.

Anti-drift training strategy. Long autoregressive roll-outs accumulate error, so we train the model to tolerate a corrupted past. Both mechanisms below act on the temporal-memory , spatial-memory , and nearby context tokens (the sink is kept clean):

- • Helios drift simulation yuan2026helios degrades the context in latent space with one of three artefact types that mimic what a roll-out drifts into — additive noise z ↦ ( 1 − σ ) ​ z + σ ​ ϵ z\mapsto(1-\sigma)z+\sigma\epsilon (with σ ∼ 𝒰 ​ ( 0 , ρ ) \sigma\sim\mathcal{U}(0,\rho) , ρ \rho a corruption strength), a down/up-sampling blur z ↦ up ​ ( down ​ ( z ; r ) ) z\mapsto\mathrm{up}(\mathrm{down}(z;r)) ( r ∼ 𝒰 ​ ( 0.9 , 1 ) r\sim\mathcal{U}(0.9,1) ), and a saturation shift z ↦ ( z − z ¯ ) ​ α + z ¯ z\mapsto(z-\bar{z})\,\alpha+\bar{z} ( α ∼ 𝒰 ​ ( 0.3 , 1.7 ) \alpha\sim\mathcal{U}(0.3,1.7) ) — where a noise-or-blur step is optionally followed by a saturation step.

Helios drift simulation yuan2026helios degrades the context in latent space with one of three artefact types that mimic what a roll-out drifts into — additive noise z ↦ ( 1 − σ ) ​ z + σ ​ ϵ z\mapsto(1-\sigma)z+\sigma\epsilon (with σ ∼ 𝒰 ​ ( 0 , ρ ) \sigma\sim\mathcal{U}(0,\rho) , ρ \rho a corruption strength), a down/up-sampling blur z ↦ up ​ ( down ​ ( z ; r ) ) z\mapsto\mathrm{up}(\mathrm{down}(z;r)) ( r ∼ 𝒰 ​ ( 0.9 , 1 ) r\sim\mathcal{U}(0.9,1) ), and a saturation shift z ↦ ( z − z ¯ ) ​ α + z ¯ z\mapsto(z-\bar{z})\,\alpha+\bar{z} ( α ∼ 𝒰 ​ ( 0.3 , 1.7 ) \alpha\sim\mathcal{U}(0.3,1.7) ) — where a noise-or-blur step is optionally followed by a saturation step.

- • Error bank li2025stablevideoinfinity keeps the model’s own reconstruction residuals δ = z ^ 0 − z 0 \delta=\hat{z}_{0}-z_{0} (with z ^ 0 = z τ − τ ​ v θ \hat{z}_{0}=z^{\tau}-\tau\,v_{\theta} ) in a buffer bucketed by chunk length and noise level, and replays them additively into the context (and the target latent), z ← z + γ ​ δ z\leftarrow z+\gamma\,\delta , so the model learns to recover from the failure modes it actually produces at inference.

Error bank li2025stablevideoinfinity keeps the model’s own reconstruction residuals δ = z ^ 0 − z 0 \delta=\hat{z}_{0}-z_{0} (with z ^ 0 = z τ − τ ​ v θ \hat{z}_{0}=z^{\tau}-\tau\,v_{\theta} ) in a buffer bucketed by chunk length and noise level, and replays them additively into the context (and the target latent), z ← z + γ ​ δ z\leftarrow z+\gamma\,\delta , so the model learns to recover from the failure modes it actually produces at inference.

The two are scheduled through an error-bank warm-up: before the bank has filled, only Helios drift is applied (at a fixed probability); once the bank is warmed up it takes priority and the Helios probability is lowered, the two being mutually exclusive within any step.


### 3.4 Post-Training – Inference Acceleration


The model from the autoregressive stage is strong but needs many ( ∼ \sim 30 30 ) sampling steps per chunk, too slow for interactive use. We distil it into a 4 4 -step student. Inspired by the joint consistency-distillation and distribution-matching principle of Causal-rCM zheng2026causal , we introduce a discrete distillation formulation tailored to our autoregressive world model. Unlike the continuous-time formulation of Causal-rCM, our discrete formulation avoids Jacobian-vector-product computation. We further incorporate self-forcing++ to account for the distribution shift induced by autoregressive roll-out. The student retains the full control and memory stack, and is optimized using distribution-matching distillation yin2024improved , self-forcing++ cui2025self , and consistency distillation song2023consistency .

- • Distribution-Matching Distillation. The student is trained to match the teacher’s output distribution through a real and a fake (critic) score. Both are served by the same score backbone via LoRA swapping rather than a second network — the critic LoRA off gives the real (teacher) score, on gives the fake (critic) score — and the critic is updated more frequently than the student (a two-timescale update rule) so that it stays ahead.

Distribution-Matching Distillation. The student is trained to match the teacher’s output distribution through a real and a fake (critic) score. Both are served by the same score backbone via LoRA swapping rather than a second network — the critic LoRA off gives the real (teacher) score, on gives the fake (critic) score — and the critic is updated more frequently than the student (a two-timescale update rule) so that it stays ahead.

- • Self-forcing++. Instead of distilling on teacher-forced clips, the student rolls out its own multi-chunk trajectories and is scored against the teacher along that self-generated path (with ground-truth context and detached history). This closes the train/inference gap of autoregressive generation and is the key to seam-free chunk continuation.

Self-forcing++. Instead of distilling on teacher-forced clips, the student rolls out its own multi-chunk trajectories and is scored against the teacher along that self-generated path (with ground-truth context and detached history). This closes the train/inference gap of autoregressive generation and is the key to seam-free chunk continuation.

- • Consistency distillation. A consistency loss between adjacent noise levels on a grid of 50 50 matches the student’s prediction at a higher noise level to an EMA copy of itself at the neighbouring lower level, which stabilises the few-step solution and suppresses brightness/appearance flicker at chunk boundaries.

Consistency distillation. A consistency loss between adjacent noise levels on a grid of 50 50 matches the student’s prediction at a higher noise level to an EMA copy of itself at the neighbouring lower level, which stabilises the few-step solution and suppresses brightness/appearance flicker at chunk boundaries.

Concretely, DMD pushes the student distribution p θ , τ p_{\theta,\tau} toward the data distribution p data , τ p_{\mathrm{data},\tau} by the score-difference gradient

|  | ∇ θ D KL ​ ( p θ , τ ∥ p data , τ ) = − 𝔼 ​ [ ( s real ​ ( z ^ i τ , τ ∣ c i ) − s fake ​ ( z ^ i τ , τ ∣ c i ) ) ​ ∂ z ^ i ∂ θ ] , \nabla_{\theta}\,D_{\mathrm{KL}}\!\left(p_{\theta,\tau}\,\|\,p_{\mathrm{data},\tau}\right)=-\,\mathbb{E}\Big[\big(s_{\mathrm{real}}(\hat{z}_{i}^{\tau},\tau\mid c_{i})-s_{\mathrm{fake}}(\hat{z}_{i}^{\tau},\tau\mid c_{i})\big)\,\tfrac{\partial\hat{z}_{i}}{\partial\theta}\Big], |  | (7) |

where z ^ i \hat{z}_{i} is a chunk from the student’s own self-roll-out and the two scores are served by the same score backbone with the critic LoRA off ( s real s_{\mathrm{real}} , teacher) or on ( s fake s_{\mathrm{fake}} , trainable critic). The consistency-distillation term enforces trajectory-invariant outputs against an EMA target θ − \theta^{-} ,

|  | ℒ cm = 𝔼 ​ [ d ​ ( G θ ​ ( z i τ , τ ∣ c i ) , G θ − ​ ( z i τ ′ , τ ′ ∣ c i ) ) ] , τ ′ < τ , \mathcal{L}_{\mathrm{cm}}=\mathbb{E}\,\big[\,d\big(G_{\theta}(z_{i}^{\tau},\tau\mid c_{i}),\;G_{\theta^{-}}(z_{i}^{\tau^{\prime}},\tau^{\prime}\mid c_{i})\big)\big],\qquad\tau^{\prime}<\tau, |  | (8) |

with d d a Huber distance; the combined objective is ℒ DMD + 0.5 ​ ℒ cm \mathcal{L}_{\mathrm{DMD}}+0.5\,\mathcal{L}_{\mathrm{cm}} . The student itself is a LoRA on the frozen backbone, and the temporal and spatial memory are kept frozen in this stage.

The distilled model generates at 4 4 sampling steps per chunk with the same 24 ​ fps 24\penalty 10000\ \mathrm{fps} output, full camera control, temporal memory and spatial memory as its teacher, at a small fraction of the inference cost.


## 4 Results



### 4.1 Experimental Setup


Benchmarks. We evaluate AlayaWorld on iWorld-Bench fang2026iworld , following its evaluation protocol for the Action Control and Memory Ability tasks. The evaluation covers three dimensions: Generation Quality, Trajectory Following, and Memory Ability. These dimensions jointly assess visual quality and consistency, the smoothness and accuracy of action-conditioned trajectories, and the model’s ability to preserve spatial and visual consistency along loop-closure trajectories. In addition, we evaluate AlayaWorld on the standardized WorldMark test suite xu2026worldmark through the World Model Arena, where model outputs generated from identical reference images and action sequences are compared through side-by-side blind human-preference evaluations across Visual Quality, Control Alignment, and World Consistency, with the resulting votes aggregated into Elo ratings. The evaluation results are publicly available at https://warena.ai/ .

Models. We compare against representative open-source video world models, including Cosmos agarwal2025cosmos , HunyuanVideo-1.5 hunyuanvideo2025 , Yume 1.5 Mao_2026_CVPR , Matrix-Game 2.0 he2025matrix , and HY-World 1.5 hyworld2025 . AlayaWorld is fine-tuned from LTX-2.3 hacohen2024ltx , performing autoregressive generation at 720p/540p, where each chunk is produced with four denoising steps and corresponds to roughly one second of video.


### 4.2 Quantitative Results


**Table 3 : Results on iWorld-Bench across three evaluation dimensions. All metrics are in [ 0 , 1 ] [0,1] , and higher is better.**

| Metric | NVIDIA Cosmos | HunyuanVideo-1.5 | WAN 2.2 | YUME 1.5 | Matrix-Game 2.0 | HY-World 1.5 | AlayaWorld |
| Generation Quality |
| Image Quality | 0.6778 | 0.7128 | 0.5545 | 0.6232 | 0.4851 | 0.6675 | 0.6620 |
| Brightness Consistency | 0.6952 | 0.7027 | 0.3886 | 0.3810 | 0.2963 | 0.8051 | 0.9492 |
| Color Temp. Constraint | 0.7170 | 0.7477 | 0.3411 | 0.4165 | 0.2937 | 0.7819 | 0.9379 |
| Sharpness Retention | 0.4363 | 0.5545 | 0.3428 | 0.4023 | 0.4149 | 0.6634 | 0.8361 |
| Trajectory Following |
| Motion Smoothness | 0.9907 | 0.9908 | 0.9557 | 0.9765 | 0.9848 | 0.9921 | 0.9924 |
| Trajectory Accuracy | 0.4955 | 0.6844 | 0.6514 | 0.7113 | 0.7008 | 0.7472 | 0.7985 |
| Memory Ability |
| Memory Symmetry | 0.3738 | 0.6336 | 0.4480 | 0.5276 | 0.3311 | 0.8481 | 0.8871 |
| Trajectory Alignment | 0.6419 | 0.6449 | 0.5703 | 0.5988 | 0.6362 | 0.6776 | 0.7018 |

Table 3 reports the quantitative results on iWorld-Bench across Generation Quality, Trajectory Following, and Memory Ability. All results are obtained using the distilled autoregressive AlayaWorld model at 480p resolution, matching the resolution of the initial frames provided by the benchmark, with each chunk generated in four sampling steps. Before inference, we apply an automated, semantics-preserving prompt adaptation procedure to reformulate the benchmark instructions into the prompt style used during training. Despite the substantially reduced sampling budget, AlayaWorld achieves the best performance on most metrics, demonstrating strong visual consistency, trajectory controllability, and long-term memory.

For Generation Quality, AlayaWorld substantially outperforms the competing methods in brightness consistency, color-temperature constraint, and sharpness retention. These improvements indicate that AlayaWorld effectively mitigates visual drift, illumination fluctuations, and sharpness degradation during autoregressive generation. Although it does not achieve the highest image-quality score, its overall generation quality remains competitive with existing video and interactive world models.

For Trajectory Following, AlayaWorld achieves the best results in both motion smoothness and trajectory accuracy, showing that the generated observations respond smoothly and accurately to the input action trajectories. AlayaWorld also consistently leads on both Memory Ability metrics, demonstrating its ability to preserve visual appearance and spatial structure when revisiting previously observed regions. These results validate the effectiveness of the proposed spatial and temporal memory mechanisms for stable long-horizon interactive generation.


### 4.3 Qualitative Results


We further present qualitative results to demonstrate the interactive generation capabilities of AlayaWorld. All examples are generated autoregressively by AlayaWorld and cover four representative aspects: camera-controllable navigation, prompt-driven actions, consistent world generation, and long-horizon stability.

Figure 5 shows the camera-control results. Given user-specified camera trajectories, AlayaWorld generates smooth viewpoint changes under diverse translation and rotation commands. The generated videos follow the intended camera motion while preserving coherent scene geometry, object layout, and visual appearance, demonstrating accurate and stable camera-conditioned navigation.

**Figure 5 : Qualitative results of camera-controlled generation. AlayaWorld follows diverse camera trajectories while maintaining coherent scene geometry and visual appearance.**

Beyond navigation, AlayaWorld supports open-ended semantic interactions through dynamically switchable text prompts. As shown in Figure 6 , users can introduce new actions and events, such as spell casting, combat, object appearance, and scene transformation, during an ongoing autoregressive rollout. The newly specified content emerges naturally in subsequent chunks while the existing scene context and previously generated content remain visually coherent.

**Figure 6 : Qualitative results of prompt-driven actions. AlayaWorld introduces newly requested actions and events during autoregressive generation while preserving the existing scene context.**

Figure 7 demonstrates the world-consistency capability of AlayaWorld under leave-and-return trajectories. After exploring previously unseen regions, the camera returns to an earlier viewpoint. AlayaWorld preserves the overall scene layout, object identity, textures, and structural details of the revisited region. These results indicate that the combination of temporal and geometry-aligned spatial memory effectively supports persistent scene representation beyond the recent context window.

**Figure 7 : Qualitative results of consistent world generation under leave-and-return trajectories. AlayaWorld preserves scene structure and visual identity when revisiting previously observed regions.**

Finally, Figure 8 presents long-horizon autoregressive generation results. AlayaWorld maintains coherent scene evolution, stable visual quality, and smooth temporal transitions over extended rollouts. In particular, the generated videos exhibit limited accumulation of blur, illumination shifts, color drift, and structural degradation, showing the effectiveness of the drift-aware training strategy for stable long-horizon generation.

**Figure 8 : Qualitative results of long-horizon autoregressive generation. AlayaWorld maintains stable visual quality and coherent scene evolution over extended rollouts.**


## 5 Conclusion


We presented AlayaWorld , an open-source interactive long-horizon video world model. Rather than constructing virtual worlds through conventional game-development pipelines, AlayaWorld explores an alternative paradigm in which interactive environments are synthesized directly from user inputs and evolve continuously through autoregressive generation.

We argue that interactive world modeling is fundamentally defined by four tightly coupled properties: interaction, consistency, stability, and efficiency. Rather than addressing these properties independently, AlayaWorld integrates them within a unified autoregressive framework in which several design choices simultaneously benefit multiple objectives. The bounded visual context supports both controllable navigation and persistent world memory, drift-aware training improves long-horizon robustness, and discrete few-step distillation together with short chunk-wise generation enables low-latency interaction while preserving responsiveness to evolving camera trajectories and text prompts. Experiments on iWorld-Bench demonstrate strong performance across generation quality, trajectory following, and memory ability. Conceived as a full-stack, open-source, and long-term project, AlayaWorld is intended to provide an extensible foundation for future research on interactive video world models and generative reality.


## 6 Contributions and Acknowledgments


Within each role category, authors are listed in alphabetical order by their first names.

Core Lead: Kaipeng Zhang

Lead: Chuanhao Li

Core Contributor: Chuanhao Li, Kaipeng Zhang, Yifan Zhan, Yongtao Ge, Yuanyang Yin

Contributor: Jiaming Tan, Kang He, Liaoyuan Fan, Mingliang Zhai, Ruicong Liu, Xiaojie Xu, Xuangeng Chu, Zhen Li, Zhengyuan Lin, Zhixiang Wang, Zian Meng, Zihui Gao


## References
