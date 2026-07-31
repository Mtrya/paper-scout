# Infinite Interactive World Rollout on a Single Desktop GPU

ABot-World Team

July 2026

## Abstract

We present ABot-World-0, an action-conditioned video world model for real-time, longhorizon closed-loop interaction, supported by a multi-source data infrastructure spanning AAA games, simulation engines, and internet videos to learn controllable world dynamics. WorldExplorer performs agent-driven collection guided by training feedback, while a unified pipeline applies 14 deterministic quality checks, VLM-based assessment, and synchronized action and text annotation. We progressively distill a bidirectional actionconditioned teacher into a causal student through teacher forcing and ODE distillation, and introduce LongForcing to align long student self-rollouts with an extended-horizon teacher, mitigating accumulated distribution shift and autoregressive drift. Raw keyboard actions provide a unified control interface for scene roaming and third-person character interaction, while reference-character memory provides persistent appearance cues for identity consistency during third-person rollouts. For deployment, we co-design a streaming inference stack with a lightweight VAE decoder, efficient attention, memory-aware scheduling, and low-bit DiT inference. Across optimized low-bit configurations, ABot-World-0 streams 720P video at up to 16 FPS on a single NVIDIA RTX 5090 desktop GPU, with 1.2 s action-to-first-frame latency and approximately 19 GiB peak VRAM. Experiments on WorldRoamBench and extended interactive rollouts demonstrate competitive controllability and coherent long-horizon world evolution. https://github.com/amapcvlab/ABot-World

![](images/97b41743ac107f6f9489d4825c2dd70f43cd6bb35bd9c2b7cec2bb65b8ab9a3e.jpg)  
Figure 1: ABot-World-0 turns a single NVIDIA RTX 5090 GPU into a real-time interactive world simulator, enabling infinite action-conditioned world rollout at 720P and up to 16 FPS with 1.2 s action-to-first-frame latency within a peak-VRAM budget of approximately 19 GiB.

## 1 Introduction

Artificial intelligence is moving from generating static content toward generating worlds that can be entered, controlled, and continuously evolved. The goal is not a sequence of visually plausible clips, but a persistent generative environment in which actions change state, the resulting observations inform subsequent actions, and the world remains coherent as interaction continues. World models [1–7] are therefore a step toward general-purpose simulators: they must represent the current state of a world, predict how it evolves under interventions, and provide a substrate for planning, learning, creation, and embodied intelligence. This is a systems-level challenge rather than a single generation objective. Visual fidelity remains necessary, but controllability, state persistence, rollout stability, latency, throughput, and memory footprint must be optimized jointly. Rapid advances in large-scale video generation [8–16] provide increasingly powerful visual and dynamical priors, yet turning those priors into responsive, accessible worlds requires solving the closed loop.

Recent systems have established important parts of this agenda. Genie [2] showed that playable environments can be learned from internet gameplay through inferred latent actions, while Genie 2 and Genie 3 [3, 17] demonstrated increasingly capable real-time dynamic environments. GameNGen [4], Oasis [5], WHAM [18], Runway GWM [19], Cosmos [6], and Waymo World Model [20] explore complementary points in the design space of games, open worlds, physical AI, and driving. Yet scaling a video model alone does not resolve the four coupled bottlenecks of interactive world modeling: obtaining broad, temporally coherent data with reliable action supervision; representing user intent across both camera navigation and embodied character control; preventing the generated history from drifting as it becomes the next input; and deploying the entire generation-and-decoding stack at interactive speed on practical hardware. A system can be strong on any one axis while remaining unusable as a local, persistent world simulator.

The data problem is especially fundamental. Passive internet video offers visual diversity but rarely exposes synchronized controls; game recordings provide exact inputs but can be stylistically narrow; simulations offer geometry and controllability but require deliberate trajectory design. We treat these sources as complementary rather than interchangeable. Our data infrastructure combines AAA game data, simulation-engine data, and real-world internet video. WorldExplorer, an agent-driven collection system, produces synchronized multimodal game and simulation trajectories and reallocates collection effort from training feedback. A source-aware but unified processing pipeline then performs 14 deterministic checks across six quality dimensions, VLM-based semantic assessment, action annotation, and structured language annotation. This turns dataset construction into a closed-loop part of world-model development: the data distribution is not merely collected once, but is actively shaped to expose the motion, viewpoint, environment, and action regimes in which the model remains weak.

We build on this infrastructure with ABot-World-0, an action-conditioned video world model designed as an end-to-end answer to the control–consistency–efficiency coupling. The model uses a unified, framesynchronous keyboard action interface rather than a separate latent-action interface; source-native controls and pose-derived pseudo-actions are mapped into the same action space during data construction, which is naturally available to users at inference time. A unified action representation covers observer-style scene roaming and actor-style character motion, while reference-character memory provides persistent appearance information for long third-person rollouts. These choices make the control channel part of the dynamics model itself rather than an external post-processing interface.

The learning pipeline separates visual-dynamics quality from the causal constraint required online. We first train a bidirectional teacher on the multi-source action-video corpus, where full-horizon generation provides a high-quality target for action-conditioned dynamics. We then progressively convert it into a causal student through teacher forcing and ODE distillation, which preserves the deployable information pattern while reducing the denoising budget. Crucially, local distillation does not by itself solve closed-loop generation: every student prediction changes the visual context for subsequent predictions, causing a growing mismatch between short-horizon training states and long-horizon inference states. We introduce LongForcing, a final distribution-matching stage that supervises long student self-rollouts with an extended-horizon bidirectional teacher. By extending distribution-level supervision over student selfrollouts, LongForcing provides corrective signals for long-horizon rollout contexts that short-horizon objectives cover only weakly.

Real-time interaction also requires systems co-design beyond few-step sampling. ABot-World-0 couples its causal model with a lightweight VAE decoder, memory-aware module scheduling, low-bit DiT inference, efficient low-precision attention, and bounded local-context KV caching. Across its optimized low-bit operating envelope, the resulting 720P streaming pipeline reaches up to 16 FPS with 1.2 s actionto-first-frame latency on a single NVIDIA RTX 5090 while operating within a peak-VRAM budget of approximately 19 GiB. The reported latency covers the complete generation-and-decoding path from receiving a user action to making the first decoded response frame available, rather than reporting sampling speed in isolation.

## Our contributions are fourfold:

• We present ABot-World-0, a unified action-conditioned video world model that uses raw keyboard inputs to support both scene roaming and character control, with reference-character memory providing complementary appearance cues during third-person interaction.

• We develop a multi-source data infrastructure for interactive world modeling, including WorldExplorer’s training-feedback-driven collection loop, synchronized multimodal capture, multi-stage quality control, and unified action and text annotation.

• We introduce a progressive bidirectional-to-causal training pipeline and LongForcing, which extends distribution-level teacher supervision to long student self-rollouts to mitigate accumulated autoregressive drift.

• We demonstrate a deployment-oriented streaming stack whose optimized low-bit operating envelope delivers 720P output at up to 16 FPS with 1.2 s action-to-first-frame latency within a peak-VRAM budget of approximately 19 GiB on one RTX 5090, and evaluate controllability, visual quality, physical plausibility, and temporal memory on WorldRoamBench together with extended interactive rollouts.

Together, these components frame interactive world modeling as a full-stack capability rather than a single generative objective. ABot-World-0 is a step toward local visual simulators that can be continuously explored, controlled, and improved for interactive creation, agent learning, and embodied-AI research.

## 2 Related Work

## 2.1 Bidirectional Video Generation

Recent advances in generative modeling have greatly improved the quality of video synthesis, with diffusion models becoming a dominant paradigm [8, 9, 11]. Early video diffusion models commonly adopt U-Net-based architectures, while recent methods increasingly employ diffusion transformers with spatiotemporal attention mechanisms [14, 15, 21–24]. By modeling spatial and temporal dependencies over the whole clip, these methods [14, 16, 25] achieve strong performance in visual quality and temporal coherence. Such full-clip bidirectional modeling is also well suited to tasks such as interpolation, infilling, and bounded generation, where multiple temporal constraints need to be considered.

However, such full-clip modeling is often enabled by bidirectional attention, where each frame can access both past and future context. While beneficial for temporal consistency, this design couples frames across the entire sequence, making it difficult to generate and output frames incrementally in low-latency or streaming scenarios. This limitation motivates recent causal or autoregressive video diffusion methods, which attempt to convert or distill bidirectional video diffusion models into sequential generators, trading part of the global context for more efficient online generation.

## 2.2 Autoregressive Video Generation

To overcome the limitations of non-causal models, autoregressive (AR) video generation methods [10, 13, 26–28] produce frames sequentially, conditioning each new segment only on previously generated content. By enforcing a causal constraint, these methods enable low-latency streaming and, when paired with bounded context or caching, reduce the memory overhead associated with long-video synthesis. Early AR models primarily relied on discrete token-based transformers, while recent works [29–33] have integrated causal structures into diffusion and flow-based frameworks [8, 34, 35] to achieve superior visual quality.

The causal structure of AR generation makes it well suited for practical long-video synthesis, as it enables streaming generation. This makes AR generation an appealing paradigm for scalable video generation. In this work, we therefore focus on the AR paradigm and address the key challenge of improving longhorizon stability.

## 2.3 Long Video Generation

Autoregressive models suffer from drift beyond the training horizon, manifesting as progressive quality degradation, identity inconsistency, motion stagnation, or collapse to static frames after a modest number of autoregressive rollouts.

Temporal drift is closely tied to exposure bias and error accumulation in AR rollouts. To mitigate this, methods like Diffusion Forcing [29], PA-VDM [36], and Rolling Forcing [33] employ heterogeneous noise scheduling during training [37–39], while training-free approaches like FIFO-Diffusion [40] extend pretrained models via sliding latent windows but still suffer eventual degradation without explicit rollout alignment. Alternatively, Self-Forcing and its variants like LongLive [41], Self-Forcing++ [42] supervise models under their own rollout distribution using causal attention, KV re-caching, and distillation for long video generation. Subsequent work further addresses fine-grained mismatches: Causal Forcing [43] tack les architectural gaps between bidirectional teachers and causal students, Context Forcing [44] leverages long-context teachers with Slow–Fast Memory, HiAR [45] performs hierarchical denoising at matched noise levels, and Diagonal Distillation [46] jointly optimizes temporal chunks and denoising steps. In this work, we use a progressive bidirectional-to-causal pipeline: a teacher trained on long action-conditioned sequences provides high-quality dynamics supervision, and LongForcing extends that supervision to long student self-rollouts to reduce accumulated drift.

Even with rollout-aligned objectives, long-range drift persists because dense attention over full history is infeasible. While naive truncation of sliding windows or KV caches often causes identity loss and motion stagnation, recent memory-centric methods improve stability via RoPE stabilization, deep attention sinks, structured KV memory [47–50]. LongLive-2.0 [51] further preserves identity across prompt switches using multi-shot attention sinks, while training-free extensions (e.g., FLEX [52], PackForcing [53]) alleviate drift through positional correction or cache partitioning. Consequently, long video generation has evolved from merely causalizing diffusion models into a comprehensive challenge spanning rollout alignment, memory design, positional extrapolation, distillation, and efficient deployment.

## 2.4 Interactive Video Generation

Interactive video generation conditions video dynamics on user actions, providing a foundation for interactive worlds, game-like simulators, and embodied AI benchmarks [7, 17, 54, 55]. A common approach encodes 6-DoF camera extrinsics with dedicated networks [56–58], while another injects dense Plücker ray maps or frustum features aligned with latent tokens [59–62]. These methods typically represent camera motion using calibrated trajectories in a global or initial-frame coordinate system. Over long rollouts, accumulated poses may move beyond the training distribution, while periodic re-anchoring can introduce inconsistencies across temporal segments. Camera-aware positional-encoding methods [63–65] incorporate camera geometry or pose into token positional representations. In contrast, we condition generation on discrete keyboard actions that specify local, incremental controls for both character movement and camera motion, rather than calibrated global camera trajectories. These actions are interpreted relative to the current character or camera state and drawn from a fixed action space, keeping the control representation bounded and naturally aligned with real-time user interaction during long-horizon rollouts [7, 54, 55].

## 2.5 Real-Time Video Generation

Real-time interactive video generation is governed by two coupled metrics: latency, the delay between a user action and the corresponding visual update, and throughput, the sustained frame rate the system can maintain over long rollouts [7, 41, 66, 67]. Latency is shaped primarily by the denoising budget and the temporal unit of generation, as standard diffusion requires many score-network passes per frame and distilled few-step models [31, 32, 41, 51, 66, 68–70] reduce this cost. Frame-wise autoregressive designs [4, 31, 41] minimize action-to-pixel latency because each new control can influence the very next emitted frame. However, they require one full denoising cycle per frame, which sharply limits throughput and increases sequential dependence across the rollout. Chunk-wise generation [39, 71, 72] instead de noises a chunk of frames jointly, amortizing attention and VAE decoding to achieve much higher FPS and stronger short-range temporal coherence, at the cost of a bounded control delay within the current chunk. Throughput, in turn, depends on how efficiently each temporal unit is produced and decoded: amortizing attention and VAE decoding across multiple frames within each chunk raises FPS, while low-bit weights and KV caches [51, 73], kernel fusion, and sequence or tensor parallelism [7, 74, 75] reduce memory and execution overhead. Asynchronous streaming decode [51] can additionally overlap generation with display. In this work, we advocate for a deployment-oriented formulation of interactive video generation that jointly optimizes latency, throughput, and memory efficiency. Our method adopts a chunk-wise streaming pipeline with progressive visual delivery. By co-designing lightweight VAE decoding, memory-aware scheduling, low-bit DiT inference, efficient attention, and optimized positional encoding, our optimized deployment envelope supports 720P interactive streaming at up to 16 FPS with 1.2 s action-to-first-frame latency within a peak-VRAM budget of approximately 19 GiB on a single NVIDIA RTX 5090 desktop GPU.

## 3 Data Infrastructure

## 3.1 Overview

ABot-World treats data production as an integral part of model development rather than as a one-time preprocessing stage. The construction of a high-fidelity, action-controllable world model fundamentally depends on large-scale, high-quality training data spanning diverse environments, viewpoints, and motion patterns. Our training corpus draws from three complementary sources: AAA game data, simulation engine data, and real-world internet data. Detailed characteristics and collection methods for each source are described in Section 3.2.

As shown in Figure 2, our data infrastructure processes all three sources through three progressive stages:

Collection. Game and simulation data are collected via WorldExplorer, our unified agent-driven system (Section 3.2.2), which navigates virtual environments and captures synchronized multimodal signals with closed-loop training feedback. Internet data undergoes a separate video segmentation and pose estimation pipeline (Section 3.2.5).

Quality Filtering. Raw collected data passes through a multi-stage filtering pipeline (Section 3.3) that combines 14 deterministic signal checks spanning six quality dimensions with vision-language model (VLM)-based semantic assessment, eliminating technically flawed and semantically invalid samples.

![](images/0a9907c278b0e716dfc0de2e61e749114d95585210f879305a7a3e93ea425575.jpg)  
Figure 2: Data Pipeline Overview. An end-to-end pipeline couples agent-driven and internet-scale collection, multi-stage quality filtering, and structured multimodal annotation. Curated data trains the world model, whose evaluation feeds a training-feedback loop that drives targeted re-collection and continual data refinement.

Annotation. Surviving clips are enriched with action labels, structured natural language scene descriptions, and semantic tags (Section 3.4) for both action-conditioned and text-conditioned training. The pipeline follows a unified, source-independent acquisition and processing design while preserving source native raw control signals when available and converting source-specific action information into a canonical representation for model training.

Our data infrastructure is distinguished from existing approaches by three key advantages:

• Fully automated agent-driven collection with training-feedback-driven adaptive rebalancing, transforming data production from passive batch processing into an active servo system tightly coupled to training objectives.

• A comprehensive quality control pipeline—14 distinct checks spanning six quality dimensions supplemented by VLM-based semantic assessment—is applied systematically to maximize the effective data ratio.

• The combination of deterministic action annotation (for game and simulation data via direct API access and trajectory design) with a unified multi-source processing flow provides high-quality labels, while real-world videos contribute diversity through pose-estimated pseudo-labels.

## 3.2 Data Collection

## 3.2.1 Data Sources and Collection Paradigms

Our training data spans three complementary sources. AAA game data provides rich, high-fidelity 3D environments with precisely synchronized action and observation signals, though inherently limited to specific titles and visual styles. Simulation engine data offers geometrically precise synthetic data with deterministic action labels and full environmental control; notably, our street aerial photography and streetscanning data reconstructed via ABot-3DGS [76, 77] yield photorealistic real-world scenes derived from proprietary, non-public assets. Real-world internet data introduces natural camera dynamics, diverse lighting, and domain generalization signals difficult to replicate synthetically, but lacks ground-truth action labels and requires noisy pose estimation.

Three collection paradigms have been developed for interactive environment data collection, each with distinct trade-offs. Manual recording (human gameplay) produces authentic, naturalistic behavior but suffers from severe scalability bottlenecks, fragmented trajectories, and high labor cost. Rule-based automated collection (scripted exploration in both game and simulation environments) operates continuously with deterministic annotations at low marginal cost, but is fundamentally constrained by its pre-defined scripts—it cannot autonomously discover novel scenarios, adapt its strategy based on training feedback, or dynam ically rebalance data distribution. Agent-driven collection, where reinforcement-learning or imitationlearning agents autonomously navigate interactive environments, enables purposeful, goal-directed exploration and produces smooth continuous trajectories at scale, while remaining prone to narrow exploration or reward-hacking failure modes and fundamentally inapplicable to passively consumed data such as internet videos.

## 3.2.2 WorldExplorer: Data Closed-Loop Collection System

To address the limitations of existing paradigms—particularly the rigidity of rule-based scripts and the narrow exploration and lack of training feedback in conventional agent-driven approaches—we develop WorldExplorer, a unified agent-driven collection system applicable to both game and simulation environments. WorldExplorer provides a modular, environment-agnostic architecture with source-specific adapters for each backend, comprising four core components.

Navigation Agent. A navigation agent autonomously traverses virtual environments through a multiphase goal selection strategy, progressively relaxing exploration criteria—from prioritizing entirely unexplored regions, to searching nearby areas, to a forward-movement fallback with collision detection— achieving high scene coverage with low revisit rates. This strategy applies to any interactive virtual environment with a navigable mesh, whether a live game world or a reconstructed 3DGS scene.

Parallel Capture Pipeline. Video frames, camera parameters (position, rotation, field of view (FOV), focal length), control inputs, environment state, and metadata are acquired simultaneously through parallel pipelines. All modalities are synchronized via millisecond-precision timestamps, with cross-modal alignment error below 33 ms at 30 FPS, and each recording session is segmented post-hoc into training clips.

Task Template System. Collection is organized into structured task categories to ensure behavioral distribution completeness—ranging from standard navigation and free exploration, to focused observation of landmarks, to targeted coverage of long-tail scenarios rarely occurring during natural interaction but critical for robust training. Scene configuration is parameterized across multiple dimensions—geography, weather, time of day, traffic density, viewpoint mode, and vehicle type—enabling fine-grained control over the data distribution.

Training-Feedback-Driven Closed-Loop. WorldExplorer’s most distinguishing feature is its closedloop, distribution-aware design. Rather than operating with fixed collection ratios, a training monitoring module continuously tracks per-category performance metrics, a weakness diagnosis component identifies under-performing scene-action combinations via cross-dimensional scoring, and an adaptive strategy generator updates collection ratios while maintaining minimum coverage floors to prevent catastrophic forgetting. The agent dynamically adapts through weighted task template selection and real-time scene parameter adjustment, all without manual intervention. The full loop—training feedback → weakness diagnosis → strategy adaptation → agent-directed production—transforms data collection from passive batch production into an active intelligent servo system tightly coupled to training objectives throughout the model development lifecycle.

## 3.2.3 Game Data Collection

Game data constitutes the primary and largest source of our training corpus. WorldExplorer is applied to multiple AAA titles across diverse genres (open-world exploration, urban driving, equestrian traversal), naturally spanning both first-person and third-person viewpoints. Consistent with our unified, data-driven training paradigm, the pipeline supports both first- and third-person viewpoints within a shared collection and processing framework, while retaining viewpoint-specific annotations where applicable. This design enables a single model architecture to learn action-conditioned dynamics across both viewpoints without requiring separate viewpoint-specific models. Each recording session produces RGB video at 1920×1080 resolution alongside synchronized per-frame camera poses, control signals, and environment metadata.

A distinctive advantage of game data is the viewpoint diversity it naturally provides. First-person perspectives deliver direct ego-motion signals closely aligned with the camera’s optical axis, while third-person perspectives offer observable character dynamics and surrounding spatial context that enrich the model’s understanding of agent-environment interaction. Moreover, the direct API access eliminates the label noise inherent in pose-estimation-based approaches, making game data the highest-quality supervision signal in our training corpus.

Source-native control signals are captured directly from the game’s runtime API with ground-truth precision and synchronized with each video frame. These source-specific controls are then translated into the canonical action representation used throughout the data pipeline.

## 3.2.4 Simulation Data Collection

To augment trajectory diversity and provide geometric supervision signals beyond what game recordings can offer [78], we construct a synthetic pipeline using two complementary rendering backends: Unreal Engine (UE) for photorealistic scene rendering with full environmental control, and a 3D Gaussian Splatting (3DGS) backend powered by ABot-3DGS [76, 77]. ABot-3DGS reconstructs highly photorealistic scenes from multi-view imagery—optionally accepting LiDAR and pre-processed photogrammetry point clouds as geometric priors—and is applied to our proprietary street aerial photography and street-scanning data to produce large-scale outdoor urban environments and massive indoor scene coverage. These geometrically accurate real-world scene representations are derived from proprietary, non-public assets.

WorldExplorer executes trajectories on these 3D assets through two complementary modes: (a) procedural path generation, employing geometric pattern synthesis with multi-point interpolation; and (b) real-world trajectory import, mapping motion paths captured from physical devices into the 3D scene, preserving authentic human browsing behaviors such as repeated scanning and natural handheld jitter. Each trajectory undergoes collision detection before rendering, and action labels are deterministically derived by projecting translational and rotational displacements onto camera basis vectors and binarizing them into discrete action signals, yielding frame-aligned annotations.

## 3.2.5 Internet Data Collection

For real-world internet data, agent-driven collection is not applicable since the content is passively consumed rather than interactively generated. We instead employ a video-based pipeline to extract training signals from diverse internet sources, including driving footage, walking tours, and aerial videos. Raw videos are first segmented into training clips based on scene boundaries and content coherence, ensuring each clip captures a consistent environment or activity. 6-DoF camera trajectories are then recovered as pseudo-labels via pose estimation methods—selected based on scene characteristics such as indoor/outdoor setting, static/dynamic content, and texture richness—to handle the diverse visual conditions encountered in internet videos.

Action intent labels are derived by projecting frame-to-frame translational and rotational displacements onto camera basis vectors and binarizing via thresholding, producing discrete action signals aligned with video frames. While these pseudo-labels carry estimation noise absent from game and simulation ground truth, internet data substantially expands training diversity by introducing natural camera dynamics, realworld lighting variations, and domain generalization signals difficult to replicate synthetically.

## 3.3 Quality Filtering

Raw collected data inevitably contains quality issues ranging from technical artifacts (frame drops, resolution mismatches) to semantic problems (user interface overlays, death sequences, geometry glitches). We implement a unified multi-stage filtering pipeline comprising 14 distinct checks across six quality dimensions—(1) file integrity, (2) visual validity, (3) geometric consistency, (4) game state correctness, (5) action-label alignment, and (6) metadata quality—supplemented by VLM-based semantic assessment, to produce the final training-ready dataset. The pipeline operates progressively across three stages, with each stage addressing a distinct category of quality concerns. Checks execute sequentially—early-stage format validation enables rapid rejection before computationally intensive analyses.

Stage 1: File Integrity. Fast deterministic checks—file pair existence, frame count consistency, resolution compliance, frame drop detection—eliminate fundamentally flawed recordings with high throughput. Format-level errors trigger clip-level rejection as they render samples unusable for downstream action conditioned model training.

Stage 2: Visual Validity, Geometric Consistency & Game State Correctness. Content-level analysis detects semantically invalid segments via VLM-based screening (user interface overlays, loading screens, popups, rendering anomalies), geometric anomaly detection (vertical displacement jumps for terrain clipping, camera-through-object analysis), game state signal processing (death sequence excision, cutscene and map-boundary removal), and action-label alignment verification. Third-person character visibility is assessed and flagged (not rejected) for downstream weighting.

Stage 3: Metadata Quality. Surviving clips receive metadata annotations—action-pose consistency scores, screen color shift flags—that serve as soft signals for training-time sample weighting and curriculum scheduling rather than hard rejection, allowing the model to still learn effectively from imperfect but informative training samples.

## 3.4 Data Annotation

Surviving clips are annotated for action-conditioned and text-conditioned world model training. The pipeline is designed to be source-agnostic where possible, while exploiting source-specific signals wherever they are available.

Action Labels. Action labels are obtained through source-specific mechanisms and converted into a shared canonical format. For game data, source-native control signals are captured directly from the game’s runtime API with ground-truth precision and synchronized with each video frame. For simulation data, labels derive deterministically from designed trajectories (Section 3.2.4). For internet videos, labels derive from estimated poses via displacement projection and thresholding (Section 3.2.5). These pseudolabels carry estimation noise but enable a substantial expansion of training data diversity. All three sources therefore provide a common action representation for unified model training.

Scene Descriptions. Beyond action labels, we employ a large vision-language model (VLM) to generate structured natural language descriptions for each clip. These descriptions serve as conditional inputs during text-conditioned generation training, enabling the model to respond to high-level semantic prompts (e.g., “a sunny drive through a coastal highway”) in addition to low-level action controls.

The VLM generates descriptions that capture the overall scene composition, environmental characteristics, weather and lighting conditions, and notable dynamic events, while deliberately omitting camera movement information to decouple motion control from scene generation—a design choice that helps prevent the model from conflating textual descriptions with camera trajectory signals. For game data, each caption is prefixed with a game identifier token, enabling the model to learn source-specific visual styles and rendering conventions.

Semantic Tagging. Each clip is additionally annotated with a set of structured semantic tags extracted by the VLM, including scene attributes (indoor/outdoor, urban/rural, building types), weather conditions, time of day, vehicle type, and viewpoint mode. These tags enable stratified sampling during training, ensuring balanced exposure to diverse environmental conditions, and facilitate targeted fine-tuning or evaluation on specific subsets.

Person Identity. For third-person perspective data, we additionally extract structured person identity representations: from each third-person clip, we identify frames where the character is clearly visible and unoccluded, then crop person thumbnails oriented toward four cardinal viewing directions (frontal, rear, left-lateral, right-lateral) based on the relative camera-to-character angle. From these directional thumbnails, we further synthesize a canonical frontal portrait via image-based face completion, producing a standardized identity reference.

These annotations support identity-conditioned generation in third-person scenarios and provide identity references for character-consistent video synthesis. Together, the curated multi-source corpus supports the complete ABot-World training pipeline, while model evaluation feeds subsequent WorldExplorer collection and closes the data–model loop shown in Figure 2.

## 4 ABot-World-0

This section presents the complete ABot-World-0 pipeline. We first introduce the general problem formulation (Section 4.1), which defines the interactive world modeling task, including the action-conditioned inputs and training objectives. We then describe bidirectional teacher training (Section 4.2), which equips the pretrained video generator with action-conditioned dynamic modeling capability through large-scale interactive video data. Next, we present causal distillation (Section 4.3), which transfers the bidirectional generation capability into a causal autoregressive model that supports efficient online interaction and long-horizon rollout. To further improve rollout stability, we introduce LongForcing, a long-horizon distribution-matching strategy that extends teacher supervision over longer temporal contexts and mitigates accumulated autoregressive drift. Finally, we detail our inference acceleration pipeline (Section 4.4).

In summary, this section presents three core technical components:

1. An action-controllable video world model for unified interactive generation. ABot-World-0 uses raw keyboard inputs as the sole action interface for both scene roaming and third-person character interaction, while reference-character memory provides complementary appearance conditioning to support character identity consistency during third-person interaction.

2. A progressive bidirectional-to-causal training pipeline for efficient and stable rollout. Teacher forcing and ODE distillation progressively transfer the bidirectional teacher into a causal few-step student, while LongForcing aligns long student self-rollouts with an extended-horizon teacher to mitigate accumulated distribution shift and autoregressive drift.

3. A full-stack inference co-design for real-time single-GPU deployment. The deployment stack combines few-step diffusion with lightweight VAE decoding, low-bit DiT inference, efficient attention and positional encoding, and memory-aware scheduling to enable real-time interactive generation within the compute and memory constraints of a single desktop GPU.

## 4.1 Formulation

A world model fundamentally aims to predict the future state of an environment from past observations and agent interactions. In our setting, the environment state is represented by video, and the model predicts future visual observations conditioned on previously observed frames, user actions, and multimodal conditions such as high-level textual instructions and reference images. We formulate this task as an interactive world modeling problem, which captures three key requirements: long-horizon video rollout, action conditioning, and temporal consistency over extended horizons.

Formally, let $\mathbf { v } _ { 0 : t - 1 }$ and $\mathbf { a } _ { t : t + L - 1 }$ denote the available visual history and future action sequence, respectively. Let c denote the multimodal conditions comprising a text prompt and reference images. We aim to model

$$
p _ { \theta } \big ( \mathbf { v } _ { t : t + L - 1 } \mid \mathbf { v } _ { 0 : t - 1 } , \mathbf { a } _ { t : t + L - 1 } , \mathbf { c } \big ) ,\tag{1}
$$

where $\mathbf { v } _ { t : t + L - 1 }$ denotes the next video chunk of length L. By iteratively predicting and appending future chunks, the model can autoregressively roll out long-horizon trajectories.

Our training pipeline consists of two main phases as shown in Figure 3, each corresponding to a different conditional modeling objective:

1. Bidirectional Teacher Learning. In the first phase, we train a bidirectional teacher to generate a full-horizon video conditioned on an initial frame, the complete action sequence, and multimodal conditions. Specifically, given the initial frame v , actions $\mathbf { a } _ { 1 : T }$ , and multimodal conditions c, the teacher is trained to model

$$
p _ { \phi } ^ { \mathrm { b i } } ( \mathbf { v } _ { 1 : T } \mid v _ { 0 } , \mathbf { a } _ { 1 : T } , \mathbf { c } ) .\tag{2}
$$

The bidirectional architecture generates the target sequence jointly, allowing information to propagate across the entire temporal horizon rather than enforcing a strictly causal dependency. This non-causal formulation allows full-horizon information exchange and provides a strong teacher for visual consistency, motion quality, and action alignment. Starting from a pretrained video generation backbone, we fine-tune the teacher on large-scale video–action data with action conditioning, reference-character memory, and long-sequence training.

2. Causal Student Learning via Distillation. In the second phase, we distill the trained bidirectional teacher into a causal autoregressive generator whose objective matches the online rollout setting in Eq. (1). Specifically, the student models

$$
p _ { \theta } ^ { \mathrm { c a u s a l } } \bigl ( \mathbf { v } _ { t : t + L - 1 } \mid \mathbf { v } _ { 0 : t - 1 } , \mathbf { a } _ { t : t + L - 1 } , \mathbf { c } \bigr ) ,\tag{3}
$$

predicting each future video chunk using only past visual observations, the corresponding action sequence, and the multimodal conditions, without access to future frames.

Because the bidirectional teacher relies on full-sequence generation and iterative denoising, it cannot be directly deployed for interactive autoregressive rollout. We therefore adopt a progressive three-stage distillation pipeline. First, teacher forcing transfers the teacher’s knowledge to an autoregressive student under causal visual conditioning. Second, ODE distillation compresses multi-step autoregressive denoising into few-step prediction by learning the mapping from intermediate noisy latents to clean endpoints defined by the frozen Stage 1 causal model’s probability-flow ODE under the same causal inputs. Third, we introduce LongForcing, which performs distribution matching against an extendedhorizon teacher. The longer supervision horizon covers later portions of student self-rollouts, where prediction errors have had more opportunities to accumulate. This provides distribution-level corrective supervision beyond short-horizon training and reduces long-term drift during autoregressive rollout.

The following sections describe each stage in detail.

![](images/83ae0c615ca1633c24b5ced5a8bf5abcafc2b23bf38c8d2a274d1513f5769360.jpg)  
Figure 3: Overall training pipeline of ABot-World-0. We first adapt a pretrained video generator into a highquality bidirectional action-conditioned teacher. The teacher is then progressively converted into a causal autoregressive student through Teacher Forcing, Causal ODE Distillation, and LongForcing, enabling few-step low-latency inference and stable long-horizon interactive rollout.

## 4.2 Bidirectional Teacher Training with Action Control Injection

Pretrained video generation models possess rich world priors such as visual appearance, physical plausibility, and scene composition, but lack action-conditioned dynamic knowledge, i.e., the mapping from discrete control inputs to corresponding visual state transitions. To bridge this gap, we train a bidirectional video generation model by injecting action control signals and reference images into the pretrained Wan2.2 backbone [16]. Through full-parameter fine-tuning on large-scale multi-source video–action data with temporal augmentation, our model learns the correspondence between actions and visual dynamics, providing a strong foundation for subsequent causal autoregressive adaptation.

## 4.2.1 Action Control Injection

We use raw keyboard input as the sole interactive control signal. Specifically, each frame-level action is represented as an 8-dimensional multi-hot vector corresponding to 8 discrete keys: W/A/S/D for character or camera movement and $\mathbb { I } / \mathbb { J } / \mathbb { K } / \mathbb { I }$ for camera rotation. Let

$$
\mathbf { a } _ { 1 : T } = \{ a _ { 1 } , \ldots , a _ { T } \} , \qquad a _ { t } \in \{ 0 , 1 \} ^ { 8 } ,\tag{4}
$$

denote the per-frame action sequence for a video of length T. Compared with methods that rely on estimated camera poses or learned latent actions, such raw keyboard signals are directly available from game recordings and align naturally with user intent at inference time.

To match the temporal compression of the video tokenizer, we pack every 4 consecutive frame-level actions along the channel dimension, consistent with the VAE temporal patch size of 4. Concretely, the packed action token sequence is

$$
\tilde { \mathbf { a } } _ { 1 : T / 4 } = \{ \tilde { a } _ { 1 } , \dots , \tilde { a } _ { T / 4 } \} , \qquad \tilde { a } _ { \tau } \in \{ 0 , 1 \} ^ { 3 2 } ,\tag{5}
$$

where

$$
{ \tilde { a } } _ { \tau } = \mathrm { C o n c a t } ( a _ { 4 \tau - 3 } , a _ { 4 \tau - 2 } , a _ { 4 \tau - 1 } , a _ { 4 \tau } ) .\tag{6}
$$

Thus, each temporal action token has dimension $8 \times 4 = 3 2$ and is aligned with one latent frame after VAE compression.

We then feed the packed action sequence into an Action Control Adapter $\mathcal { F } _ { \psi }$ . We adopt a camera adapter architecture consisting of PixelUnshuffle, a convolution layer whose kernel size and stride are both set to the DiT spatial patch size, followed by residual convolution blocks. The downsampling factor of PixelUnshuffle is chosen to match the spatial compression ratio of the VAE, so that the adapter output is strictly aligned with the spatiotemporal resolution of the DiT patchified latent tokens. Denoting the noised video latents by z and their patch embeddings by PatchEmbed(z), the action-conditioned latent tokens are given by

$$
\hat { \mathbf { z } } = \mathrm { P a t c h E m b e d } ( \mathbf { z } ) + \mathcal { F } _ { \psi } ( \tilde { \mathbf { a } } _ { 1 : T / 4 } ) .\tag{7}
$$

This design enables direct token-wise additive injection of action information into the DiT backbone.

We use additive injection at the patchify stage throughout our model.

## 4.2.2 Identity-Preserving Conditioning

Video world models may suffer from identity drift during long-horizon third-person rollout, where the character appearance gradually deviates despite preserving motion dynamics. To address this issue, we introduce a reference-character memory module composed of multiple canonical reference images of the controllable character.

The reference images are encoded by the same VAE as the rollout frames and converted into identitymemory tokens, which are prepended to the video-token sequence before the DiT backbone. We assign fixed negative temporal RoPE indices to memory tokens and non-negative indices to video tokens, separating static identity information from the generated trajectory. Furthermore, we adopt an asymmetric memory–video attention pattern, where video tokens attend to memory tokens while memory tokens remain isolated from video tokens. This enables persistent identity retrieval throughout autoregressive rollout and improves character consistency over long horizons.

![](images/47c19f58d85c75a8623f2e12e70ba3d0c096d33c8ca11795c95bd14ecf58117b.jpg)  
Figure 4: ABot-World-0 Model Architecture. ABot-World-0 incorporates action control and reference-character memory into a pretrained video DiT. Keyboard actions guide dynamics, while reference images help preserve character identity.

## 4.3 Causal Progressive Student Distillation

The bidirectional teacher cannot be directly deployed for real-time interaction due to its full-sequence generation and multi-step denoising. We therefore distill it into a causal generator through a three-stage progressive pipeline: (1) Teacher forcing, which converts the bidirectional teacher into an autoregressive student under causal visual conditioning; (2) ODE distillation, which reduces the autoregressive denoising process to few-step inference while preserving generation quality; and (3) LongForcing, which performs long-horizon distribution matching with an extended-horizon teacher to improve rollout stability and mitigate accumulated drift during autoregressive generation. By exposing the student to a broader range of long-horizon student-rollout contexts where rollout errors may accumulate, LongForcing provides corrective supervision beyond short training horizons. Each stage builds on the previous one, progressively transforming the high-quality but non-causal teacher into an efficient causal world model suitable for real-time interactive rollout.

## 4.3.1 Stage 1: Teacher Forcing

Rather than training the causal student from scratch, we initialize it from the trained bidirectional teacher and adapt the model through a causal training objective and attention masking strategy. While the bidirectional teacher jointly denoises the entire video sequence with access to the full temporal context, the causal student is trained to generate future video chunks using only previously observed visual context.

Specifically, during teacher forcing, the history frames are provided as clean ground-truth latents, while the target chunk is corrupted according to the diffusion process. A causal attention mask restricts the model from accessing future visual contexts beyond the available history, thereby matching the information structure required for autoregressive inference. This formulation retains the visual priors and long-range dynamics knowledge learned by the bidirectional teacher while adapting the model to causal generation.

Formally, for a rollout step starting at time t, the student models

$$
p _ { \theta } ^ { \mathrm { c a u s a l } } \bigl ( \mathbf { v } _ { t : t + L - 1 } \mid \mathbf { v } _ { 0 : t - 1 } , \mathbf { a } _ { t : t + L - 1 } , \mathbf { c } \bigr ) ,\tag{8}
$$

where $\mathbf { v } _ { 0 : t - 1 }$ is used as clean visual conditioning and $\mathbf { v } _ { t : t + L - 1 }$ is converted into noisy latent targets for diffusion training. By relying on ground-truth history at this stage, the objective provides a stable adaptation from bidirectional trajectory modeling to causal autoregressive rollout.

## 4.3.2 Stage 2: ODE Distillation

The causal diffusion model obtained in the previous stage still requires iterative denoising at each autoregressive rollout step. To reduce inference latency, we apply causal ODE distillation, which trains a few-step model to approximate the probability-flow ODE induced by the Stage 1 causal model.

A key requirement is that the reference model and the distilled model follow the same causal factorization. We therefore freeze the Stage 1 causal diffusion model, parameterized by $\theta _ { \mathrm { c } }$ , and condition both models on the same causal context:

$$
\mathcal { C } _ { t } = \left( \mathbf { v } _ { 0 : t - 1 } , \mathbf { a } _ { t : t + L - 1 } , \mathbf { c } \right) ,\tag{9}
$$

where $\mathbf { v } _ { 0 : t - 1 }$ denotes the clean visual history, $\mathbf { a } _ { t : t + L - 1 }$ denotes the target action sequence, and c denotes the multimodal conditions comprising a text prompt and reference images.

Let $\mathbf { z } _ { s } ^ { \mathrm { c } }$ denote an intermediate latent at noise level s along the probability-flow ODE trajectory of the Stage 1 causal model. Its corresponding clean endpoint is

$$
\mathbf { z } _ { 0 } ^ { \mathrm { c } } = \Phi _ { \theta _ { \mathrm { c } } , s  0 } ( \mathbf { z } _ { s } ^ { \mathrm { c } } ; \mathcal { C } _ { t } ) ,\tag{10}
$$

where $\Phi _ { \theta _ { \mathrm { c } } , s  0 }$ denotes integration of the causal model’s probability-flow ODE from noise level s to the clean endpoint. The distilled model is trained to directly predict this endpoint:

$$
\mathcal { L } _ { \mathrm { O D E } } = \mathbb { E } _ { s , \mathbf { z } _ { s } ^ { \mathrm { c } } , \mathcal { C } _ { t } } \left[ \lVert f _ { \boldsymbol { \theta } } \left( \mathbf { z } _ { s } ^ { \mathrm { c } } , s , \mathcal { C } _ { t } \right) - \mathrm { s g } \left( \mathbf { z } _ { 0 } ^ { \mathrm { c } } \right) \rVert _ { 2 } ^ { 2 } \right] ,\tag{11}
$$

where sg(·) denotes the stop-gradient operation.

Because the intermediate latent and its endpoint are obtained from the same causal ODE trajectory under identical conditioning, they define a consistent flow map for each autoregressive prediction step. Moreover, the distillation target depends only on the causal context available during deployment and does not require future visual observations. The resulting model can therefore approximate the original causal diffusion endpoint map with substantially fewer denoising steps while preserving its autoregressive rollout behavior.

## 4.3.3 Stage 3: LongForcing

After ODE distillation, the causal student can perform autoregressive rollout with only a few denoising steps. However, it still faces a fundamental long-horizon challenge: since each prediction is conditioned on previously generated frames, small distributional errors can accumulate and gradually shift the rollout toward long-horizon student-rollout contexts that are rarely encountered during training. These long-horizon student-rollout contexts receive limited distribution-matching supervision, causing the rollout distribution to gradually drift away from the desired world dynamics.

We therefore introduce LongForcing, a final long-horizon distribution-matching stage that improves rollout stability by extending the temporal horizon of teacher supervision. Unlike ODE distillation, which learns clean-endpoint flow maps under fixed causal conditioning, LongForcing aligns the student and teacher at the level of long-horizon conditional video distributions. The key insight is that long-horizon stability depends not only on accurate local transitions, but also on whether the closed-loop rollout distribution remains within the temporal region covered by teacher supervision.

Longer student self-rollouts expose the model to long-horizon student-rollout contexts in which small prediction errors have had more opportunities to accumulate. LongForcing trains on these self-generated trajectories and applies distribution-level corrective supervision from an extended-horizon teacher during the final Distribution Matching Distillation (DMD) stage [68]. This encourages the student to maintain plausible long-horizon dynamics without requiring explicit frame-level trajectory matching. Compared with short-horizon teacher supervision, the extended horizon provides distribution-level guidance over a longer temporal range and is designed to mitigate progressive degradation in visual quality and actionconditioned dynamics.

## 4.4 Full-Stack Co-Design for Real-Time World Rollout

Few-step distillation substantially reduces the denoising cost of each autoregressive prediction, but it is not sufficient by itself to deliver real-time interactive world rollout on a single consumer-grade GPU. As the denoising budget is compressed, the dominant system bottlenecks shift toward chunk-wise latent decoding, transformer computation, attention and positional-encoding overhead, context-memory traffic, and runtime model residency.

We therefore formulate real-time deployment as a full-stack co-design problem spanning the temporal generation granularity, VAE decoder, low-precision DiT execution, attention and RoPE kernels, bounded context memory, and runtime module scheduling. Rather than optimizing an isolated operator, our objective is to jointly balance three deployment-critical properties: sustained generation throughput, actionto-first-frame latency, and peak GPU memory usage. This co-designed inference stack enables real-time, high-resolution, long-horizon interactive generation on a single NVIDIA RTX 5090 GPU.

## 4.4.1 Deployment Objective and Runtime Metrics

ABot-World-0 adopts chunk-wise causal generation, jointly producing a chunk of latent frames and decoding the complete chunk before delivering the resulting frames to the streaming runtime. We define action-to-first-frame latency as the wall-clock time from a user keypress until the corresponding inference

Table 1: Deployment setting and operating envelope of ABot-World-0 on a single NVIDIA RTX 5090.
<table><tr><td>Item</td><td>Specification</td></tr><tr><td>GPU</td><td>1× NVIDIA RTX 5090</td></tr><tr><td>Batch size</td><td>1</td></tr><tr><td>Resolution</td><td>1280 × 704</td></tr><tr><td>Inference</td><td>Chunk-wise streaming</td></tr><tr><td>Throughput</td><td>Up to 16 FPS</td></tr><tr><td>Action-to-first-frame latency</td><td>1.2 s</td></tr><tr><td>Peak VRAM</td><td>≤ 19.3 GiB</td></tr></table>

chunk has been decoded and its first response frame becomes available.

The table reports the overall deployment envelope rather than a single precision-specific operating point. Detailed speed and memory trade-offs across FP8 and more aggressive low-bit configurations are reported in Table 2.

## 4.4.2 Chunk-Wise Streaming and Runtime Co-Design

A lightweight VAE decoder reduces first-frame latency and peak memory usage. Although the DiT remains the dominant computational component, VAE decoding still contributes non-negligibly to firstframe latency and memory consumption at high resolution. Inspired by TAEHV [79], we simplify and prune the decoder architecture to construct a lightweight VAE decoding path, denoted LightVAE, substantially reducing decoding time and peak memory usage.

Because the complete latent chunk must be decoded before its response frames become available to the streaming runtime, reducing chunk decoding time directly lowers the action-to-first-frame latency.

Memory-aware scheduling improves single-GPU deployment feasibility. To further reduce GPU memory pressure, we adopt memory-aware runtime scheduling inspired by the module-swapping utilities of FramePack [80]. Instead of requiring all components of the inference pipeline to remain simultaneously resident on the GPU, model modules are staged according to their execution order and memory requirements. This strategy reduces peak VRAM usage without modifying the model architecture, while limiting data-transfer overhead along the latency-critical inference path.

Fast-RoPE reduces temporal positional-encoding overhead. Long-horizon autoregressive generation repeatedly applies temporal positional encoding over a moving local-attention context. We adopt Fast-RoPE to improve temporal positional encoding during streaming generation. Specifically, we re-anchor temporal RoPE within the local attention window to reduce repeated RoPE computation over the full visible context as the rollout advances. We additionally use a Triton-based RoPE kernel to accelerate the rotation operation. Together, these optimizations reduce positional-encoding overhead during long horizon interactive inference.

## 4.4.3 Low-Precision Compute and Context-Memory Co-Design

Low-bit mixed-precision inference improves DiT throughput and memory efficiency. To reduce the computational and memory-bandwidth cost of DiT inference, we apply low-bit quantization following the deployment practices of LightX2V [81]. FP8 serves as our default quality-oriented operating point and provides substantial improvements in GEMM throughput and memory-bandwidth efficiency.

We additionally evaluate more aggressive low-bit formats to characterize the upper-throughput region of the deployment envelope. Concretely, the compute-intensive linear layers of the DiT backbone are quantized, while numerically sensitive components such as the VAE and text encoder remain in higher precision. This mixed-precision design provides configurable trade-offs in inference throughput and memory usage while retaining higher precision for numerically sensitive components.

Efficient attention kernels accelerate the dominant transformer operators. Attention computation constitutes a major part of the per-step latency of the diffusion transformer, particularly for high-resolution video generation with long token sequences. We therefore adopt SageAttention2 [82] as an efficient attention backend. It replaces conventional attention execution without requiring model retraining or architectural modification, reducing the runtime cost of a dominant transformer operator.

Bounded KV caching prevents unbounded context growth, while cache quantization can further reduce its fixed cost. For long-horizon streaming generation, a naive full-history KV cache grows continuously with the generated context and eventually becomes a practical memory and bandwidth bottleneck. ABot-World-0 avoids this unbounded growth through a bounded local-context KV cache with rolling eviction, keeping the cache footprint independent of the total rollout duration.

Nevertheless, even under a bounded-context design, the resident key and value tensors remain a substantial source of memory consumption and memory-bandwidth traffic at high resolution. To further reduce this cost, we explore KV-cache quantization, which compresses cached key and value tensors while preserving their usability for subsequent attention computation.

Following recent advances in online vector quantization [83], this strategy can reduce both the fixed cache footprint and the bandwidth pressure of local-context attention. KV-cache quantization can therefore complement low-bit DiT inference and improve the scalability of long-horizon generation under constrained GPU memory.

## 4.4.4 End-to-End System Analysis

Table 2: System-level breakdown across optimization variants on a single NVIDIA RTX 5090 GPU at 1280 × 704 resolution and batch size 1.
<table><tr><td>Configuration</td><td>DiT time (ms/chunk)</td><td>VAE time (ms/chunk)</td><td>FPS ↑</td><td>VRAM (GiB) ↓</td></tr><tr><td>Base</td><td>1</td><td></td><td>OOM</td><td>OOM</td></tr><tr><td>+ SageAttention2</td><td></td><td></td><td>OOM</td><td>OOM</td></tr><tr><td>+ SageAttention2 + LightVAE</td><td>1191.081</td><td>78.276</td><td>9.117</td><td>20.491</td></tr><tr><td> $+ \mathrm { S a g e A t t e n t i o n 2 } + \mathrm { L i g h t V A E } + \mathrm { F P 8 }$ </td><td>845.180</td><td>75.980</td><td>12.405</td><td>15.925</td></tr><tr><td> $\mathrm { + S a g e A t t e n t i o n 2 + L i g h t V A E + F P 8 + F a s t - R o P E }$ </td><td>786.871</td><td>71.730</td><td>13.269</td><td>19.281</td></tr><tr><td> $+ \mathrm { S a g e A t t e n t i o n 2 } + \mathrm { L i g h t V A E + M X F P 6 + F a s t - R o P E }$ </td><td>718.281</td><td>85.994</td><td>14.098</td><td>18.287</td></tr><tr><td> $+ \mathrm { S a g e A t t e n t i o n 2 } + \mathrm { L i g h t V A E } + \mathrm { M X F P 4 } + \mathrm { F a s t - R o P E }$ </td><td>638.843</td><td>72.957</td><td>15.831</td><td>17.148</td></tr></table>

Each inference chunk contains 3 latent frames and produces 12 decoded video frames. DiT and VAE times are reported per inference chunk, FPS denotes generation throughput, VRAM denotes peak GPU memory usage in GiB, and OOM denotes out of memory. Table 2 demonstrates that real-time deployment emerges from full-stack optimization rather than the acceleration of a single operator. Both the Base configuration and the SageAttention2-only variant run out of memory, showing that a faster attention kernel alone is insufficient to make the complete high-resolution pipeline deployable on a single desktop GPU. Memory feasibility requires the joint optimization of transformer execution, video decoding, numerical precision, and runtime model residency.

Introducing LightVAE yields the first feasible configuration, achieving 9.117 FPS with 20.491 GiB of peak VRAM. This result indicates that the original VAE is an important bottleneck for memory feasibility, while VAE decoding remains a non-negligible component of the chunk-response latency.

Building on this configuration, quantizing the DiT backbone provides substantial gains. FP8 reduces the DiT time from 1191.1 ms to 845.2 ms per chunk, increases generation throughput from 9.117 to 12.405 FPS, and reduces peak memory usage from 20.491 GiB to 15.925 GiB. Adding Fast-RoPE further reduces the DiT time to 786.9 ms and raises throughput to 13.269 FPS. The measured peak memory increases to

19.281 GiB in this configuration, highlighting that peak VRAM is determined by the complete runtime configuration rather than by the cost of an individual operator alone.

We use FP8 as the default quality-oriented configuration, while more aggressive low-bit formats extend the upper-throughput operating envelope. Across the optimized low-bit configurations, ABot-World-0 reaches up to 16 FPS while peak VRAM remains below 19.3 GiB.

Overall, the results validate our central systems insight: few-step generation does not automatically translate into real-time interaction. Real-time, high-resolution world rollout requires the full-stack co-design of temporal generation, latent decoding, transformer arithmetic, attention and positional encoding, contextmemory management, and runtime scheduling. Together, these optimizations turn the distilled causal model into a practical interactive world simulator that can run continuously on a single desktop GPU.

Quantization-aware training may further improve the speed–memory–quality trade-off. We leave this direction for future work.

## 5 Evaluation

We evaluate ABot-World-0 through three complementary views: quantitative comparison on WorldRoam-Bench, a 60-second temporal ablation of LongForcing, and qualitative stress tests spanning hour- and day-scale rollouts, out-of-domain control, and physical interaction. WorldRoamBench measures action controllability, trajectory following, visual quality, physical plausibility, and temporal memory, while the extended rollouts examine model behavior beyond the benchmark horizon. The LongForcing ablation directly evaluates visual error accumulation during autoregressive rollout, and the qualitative cases illustrate extended interaction, identity persistence, control generalization, and plausible physical effects. Together, these experiments characterize the controllability, coherence, and state persistence of ABot-World-0 over prolonged interaction.

## 5.1 WorldRoamBench Evaluation

To evaluate ABot-World-0 in interactive, long-horizon scenarios, we use WorldRoamBench [84], an openworld benchmark for controllable video world models. WorldRoamBench assesses four critical dimensions: action controllability, visual plausibility, physical consistency, and temporal memory retention.

## 5.1.1 Quantitative Results

Table 3 reports concrete WorldRoamBench sub-dimensions only. We compare against Genie 3 [3], HappyOyster [85], LingBot-World [86], and HY-World 1.5 [87] following the benchmark protocol.

Table 3: WorldRoamBench quantitative comparison on selected sub-dimensions of Action, Visual, Physics, and Memory, with headers colored accordingly. Best results are shown in bold, and second-best results are underlined.
<table><tr><td>Model</td><td>Size</td><td>Strict Acc.</td><td>Partial Acc.</td><td>Traj. Score</td><td>Aesthetic</td><td>Imaging</td><td>Mechanics</td><td>Memory</td></tr><tr><td>Genie 3</td><td>一</td><td>0.4700</td><td>0.6608</td><td>0.6719</td><td>0.4711</td><td>0.4757</td><td>0.5454</td><td>0.6073</td></tr><tr><td>HappyOyster</td><td></td><td>0.5317</td><td>0.7631</td><td>0.7737</td><td>0.5235</td><td>0.4377</td><td>0.5395</td><td>0.6309</td></tr><tr><td>LingBot-World</td><td>14B</td><td>0.3235</td><td>0.4198</td><td>0.4094</td><td>0.2898</td><td>0.2875</td><td>0.2777</td><td>0.3006</td></tr><tr><td>HY-World 1.5</td><td>8.3B</td><td>0.1640</td><td>0.2088</td><td>0.2015</td><td>0.1400</td><td>0.1236</td><td>0.1115</td><td>0.1562</td></tr><tr><td>ABot-World-0</td><td>5B</td><td>0.5266</td><td>0.7290</td><td>0.6752</td><td>0.5039</td><td>0.4651</td><td>0.5223</td><td>0.5041</td></tr></table>

ABot-World-0 achieves strong action fidelity, trajectory following, visual quality, physical mechanics, and memory scores.

## 5.1.2 Qualitative Analysis

Beyond its competitive quantitative performance, ABot-World-0 demonstrates three qualitative capabilities that are critical for practical interactive world models: sustained controllability and scene coherence over extremely long rollouts, a unified control interface that generalizes across diverse scenes and controllable characters, and plausible physical responses to actions and environmental constraints. Across the evaluated cases, ABot-World-0 remains responsive to user commands while preserving recognizable scene structure, actor and object identity, and motion continuity, without rapidly degenerating into frozen motion, repetitive textures, or severe visual drift. These results suggest that ABot-World-0 learns persistent and stateful interactive dynamics rather than merely generating sequences of locally plausible short clips.

Figures 5, 6, and 7 demonstrate the model’s long-horizon generation capability. The hour-scale results contain five independent rollouts visualized as timestamped keyframe strips. Despite the extended generation horizon, ABot-World-0 preserves coherent environments, viewpoints, and controllable actors while continuously responding to the input action stream. The day-scale stress tests further expose the model to severe accumulated-error conditions, and the sampled checkpoints retain recognizable scene structure and active motion instead of collapsing into texture noise, repetitive content, or static frames.

Figure 8 evaluates out-of-domain (OOD) control generalization, where both the environment and the controllable character fall outside the training distribution. The examples span diverse scenes and controllable characters under a unified action representation. Across these OOD settings, the generated motion consistently follows the input action sequence while remaining coherent with the surrounding environment, demonstrating that the learned action conditioning generalizes to novel scene–character combinations.

Figure 9 highlights plausible physical consequences that emerge during interaction. For example, a person pushes a cardboard box away, footsteps induce temporally consistent disturbances in water, a person walking through snow leaves persistent footprints, first-person motion is blocked by a wall, and a character collides with a railing without passing through it. These cases involve collisions, contact effects, persistent environmental changes, and geometric constraints that are not specified by the action labels alone. Although ABot-World-0 is not explicitly trained with symbolic physical rules or real-world colli sion annotations, it produces plausible physical responses from large-scale interactive video experience.

![](images/da3f2c5f3beaaa0cfc4507e98328f678888a094fd6dc5859a284b56be93781c2.jpg)  
Third-person fantasy adventure gameplay scene of a young blond hero standing on a stone path …

Figure 5: Hour-scale long-horizon generation examples. Each rollout is shown as a timestamped keyframe strip, demonstrating sustained controllability and scene coherence over one-hour interactive rollouts.

![](images/10d0dc2a27c20745f4d893dfe0ba51b1bf44d8fcec3c18f27a8e8e6b2903f7e2.jpg)  
Third-person fantasy gameplay scene of a white-haired swordsman moving along a steep grassy mountainside, …

Figure 6: Day-scale long-horizon generation examples, Set I. Timestamped keyframes from two day-scale rollout examples show recognizable scene structure, viewpoint consistency, and active motion at sampled checkpoints over extended generation.

![](images/d12eee5dedde1589e37cae75d9407afb4f8d74944f1f2c2b66be3acbebb90554.jpg)  
Third-person dark fantasy gameplay scene of a white-haired hunter climbing a bleak rocky slope under stormy skies, …

![](images/695f497a3211003511004555e359bc57f83c4b658fee15ef3bb11c7cfa63e0cf.jpg)  
Third-person snowy wilderness gameplay scene of an explorer in a bright orange jacket running across …  
Figure 7: Day-scale long-horizon generation examples, Set II. Two additional timestamped rollout examples maintain recognizable scene structure and active motion at sampled checkpoints under substantial accumulated-error conditions.

![](images/9bbd4ff2d4dcc63048d38d3f2da50d37dbb2029de6abb29143ae490b9494392b.jpg)

![](images/be778ddd9075ef0c2cbe2c369fe7e0eb0ff3483fb9839b4cb520d553ca68247c.jpg)  
Third-person gameplay scene of a man in a dark leather jacket standing in the middle of a wide, wet city boulevard, …

![](images/d67ac80975e23c3301125d90c570c0c1b9e8d24a81e3afd630d695073741a7a8.jpg)  
Third-person gameplay scene of a red-robed warrior holding a long polearm in a snowy temple courtyard, …

![](images/0c2c3c749fd4e9185917068a64d187d68bc414d4c04fef960993dfa007915eb5.jpg)

![](images/fd01b0a1ef665b26ef956c9c42d74f688856b58bd07bb4b12c076a28de365cd3.jpg)  
Third-person gameplay scene of a man standing on a sunny urban sidewalk beside a modern shopping street, …

Figure 8: Out-of-domain (OOD) control generalization across diverse scenes and controllable characters. Each row presents an interactive rollout in which both the environment and the controllable subject are outside the training distribution. Using a unified action interface, ABot-World-0 generates action-consistent character motion while preserving coherence with the surrounding scene.

![](images/440b4a2305a091197148219ed60ff05bc6aa6cc556e1a4d8b9f327fae114078b.jpg)  
A man walks down a snow-covered village street while carrying a cardboard box, …

![](images/5fd8aa12fc417a8e8b3a83e425ae72418fea52017dd42f0695ead940a8164a80.jpg)  
A man stands in a shallow desert river, rocky banks and palm trees on both sides, warm sunlight, …

![](images/5787dc7117a84b0a80fc22bec41ac8cac4da4f9e19e561528af2d298e439d921.jpg)  
A character in an orange winter parka standing on a snowy shoreline beneath vivid green aurora lights, …

![](images/f3c3b8e101f1db92f9fd44ccc1b5fa28182ddcfed9f78bbd8020d922676a788d.jpg)  
A character walking along a narrow concrete path beside a tall gray wall, modern campus-like …

![](images/16623f16dd86fad462e62e9183b1d429d10a3851b0c1351edfb109c6d97fd10a.jpg)  
A backpacked character climbing a metal staircase through a snowy mountain pass, …

Figure 9: Physical-interaction examples. The cases include object collision, water traces, snow tracks, wall blocking, and collision with a railing without interpenetration.

## 5.2 Effect of LongForcing on Long-Horizon Rollouts

We compare LongForcing with a Causal-Forcing-style baseline [43] adapted to our interactive world model under the same rollout protocol. Both variants train on histories produced by the student’s own autoregressive rollout and apply DMD in the final post-training stage. LongForcing differs by using extended-horizon teacher supervision during this stage rather than the shorter-horizon supervision used by the baseline. We evaluate both variants over 60-second rollouts using the HPSv3 score [88], highsaturation pixel ratio, perceptual blur score, and patch repeat ratio. Higher HPSv3 is better; lower values are preferred for the other three metrics.

![](images/fdf40ed8bc0f9110d5eb08a2a8056b9713673493b8b1c583d7b55c0fda517de4.jpg)

![](images/af1356fda7dd49ebdd9fda0a57a7e8c94cc8bff82afb5318b53596d4b75f9d25.jpg)

![](images/fdb5e53a9cb5f68cbab4975e9d15f10f8d0cb7ef6e895d1bbfcba03862c9432e.jpg)

![](images/9817601f0d453ddd37a2d85649a07f11bce5e7dc54c6e4cddd2686f7156285fe.jpg)  
Figure 10: Framewise comparison of our Causal-Forcing-style baseline and LongForcing over 60-second rollouts. Both variants use student self-rollouts and final-stage DMD; LongForcing additionally uses extended-horizon teacher supervision. The curve labeled Causal Forcing denotes the adapted Causal-Forcing-style baseline used in this report. Higher HPSv3 values are better, while lower values are better for high-saturation pixel ratio, perceptual blur score, and patch repeat ratio. Curves are averaged over the evaluation set.

As shown in Figure 10, the differences between the two variants become more pronounced during later portions of the rollout. The Causal-Forcing-style baseline shows a progressive decrease in HPSv3 together with higher saturation, blur, and patch repetition in the second half of the rollout. LongForcing retains a higher HPSv3 score and lower artifact-related metrics over the same interval, indicating less visual error accumulation during autoregressive generation.

## 6 Discussion and Future Work

Explicit actions admit simple conditioning. Keyboard inputs are discrete, temporally aligned with the video, and directly reflect user intent. For these explicit control signals, we use additive injection at the patch-embedding stage to provide reliable control while preserving the visual prior of the pretrained model. More elaborate conditioning may remain useful for ambiguous signals such as latent actions, continuous camera trajectories, or semantic instructions.

Long-horizon drift is better understood as a distribution-shift problem. As autoregressive generation proceeds, small errors accumulate in the visual context and gradually move the model beyond the states covered during short-horizon training. Sink-based context stabilization and fixed reference frames can delay this process, but excessive anchoring may keep the model too close to its initial observation, restricting motion and scene evolution. LongForcing takes a different route: it trains on long student self-rollouts and uses an extended-horizon teacher to regularize the resulting distribution. This allows the model to continue imagining new content while remaining stable. The coherent sampled checkpoints in our day-scale rollouts are consistent with the objective of training-time distribution alignment as a promising route to long-horizon generation.

Real-time interaction is more than few-step sampling. Another practical lesson is that reducing denoising steps does not by itself produce an interactive system. Once the DiT is distilled, VAE decoding, attention computation, memory transfer, and the memory footprint and bandwidth demands of the KV cache remain material systems costs. Optimizing these components together enables ABot-World-0 to sustain real-time inference on a single desktop GPU. For interactive world models, action-to-first-frame latency, sustained throughput, and memory footprint are therefore more meaningful than the number of denoising steps alone.

Future work. These observations point to several natural extensions. Richer actions and semantic events may require more structured conditioning; multi-scale LongForcing and persistent scene memory may further improve long-horizon consistency; and more efficient decoding and context management could extend high-resolution real-time generation to a broader range of consumer hardware.

## 7 Conclusion

ABot-World-0 brings action-conditioned video world modeling to real-time desktop deployment. Trained on multi-source action-video data spanning AAA games, simulation engines, and internet videos, a single model supports both scene navigation and character control across different environments and controllable subjects. Reference-character conditioning provides persistent appearance cues during long-horizon thirdperson generation.

The training pipeline starts from a bidirectional teacher that learns action-conditioned visual dynamics, and progressively converts it into an efficient causal generator through teacher forcing and ODE distillation. LongForcing further aligns the student’s long self-rollouts with an extended-horizon teacher, extending distribution-level supervision to long-horizon student-rollout contexts encountered during repeated autoregressive generation. The 60-second rollout evaluation, which extends beyond the training horizon, shows that LongForcing reduces visual error accumulation relative to the Causal-Forcing-style baseline, supporting improved visual stability over the evaluated rollouts.

The optimized deployment stack combines a lightweight VAE decoder, low-bit inference, efficient attention, and memory-aware scheduling to deliver 720P streaming at up to 16 FPS with 1.2 s action-to-firstframe latency on a single RTX 5090, while peak VRAM remains below 19.3 GiB across the optimized low-bit operating points. Results on WorldRoamBench show competitive performance in action following, trajectory following, visual quality, physical mechanics, and memory retention. Notably, sampled checkpoints from day-scale rollouts preserve recognizable visual quality, active dynamics, and scene coherence without observable collapse at the evaluated timestamps.

ABot-World-0 demonstrates that unified interactive control, stable long-horizon inference, and consumergrade real-time deployment can be achieved within a single video world model. By continuously extending the generated world beyond predefined scene boundaries, it provides an open and practical foundation for interactive content creation, game simulation, agent learning, and embodied-AI research.

## 8 Contributors

Project Sponsors: Mu Xu and Ning Guo.

Foundation Model Team: Fan Jiang<sup>†</sup>, Zhaoxu Sun, Mengchao Wang, Ziyu Zhu, Chiyu Wang, Yunpeng Zhang, Wenlin Liu, Yun Wang, Xue Zheng, Rui Sun, and Junfeng Ni.

Data Team: Hongyu Pan<sup>†</sup>, Zhongxu Sun, Fei Yu, Zengye Ge, Mengmeng Du, Nianfei Fan, Mingchao Sun, Yu Liu, and Yongchang.

AI Infra Team: Yanqing Zhu<sup>†</sup>, Jiahang Wang, Ning Ying, Yuze Xuan, and Di Yang.

Benchmark Team: Zhicheng Liu<sup>†</sup>, Zhe Gao, Tingbing Xu, Jiacheng Sui, and Wenjin Yang.

Engineering Team: Junnan Lai<sup>†</sup>, Nianfei Fan, Shufeng Liu, Mengmeng Du, Yuan Liu, Zheng Zhou, Yingliang Peng, Dawei Cao, Kaifeng Sheng, Yuxiang Cai, and Fei Lu.

## Acknowledgements

We sincerely thank Chengzhen Yu, Chong Sun, Chunnuo Gong, Jian Zhang, Qianwei Wang, and Yu Lei (listed alphabetically by first name) for their invaluable support, insightful feedback, and contributions throughout the development of this project.

## References

[1] Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, and Sanja Fidler. Learning to simulate dynamic environments with GameGAN. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[2] Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In International Conference on Machine Learning (ICML), 2024.

[3] Google DeepMind. Genie 3. https://deepmind.google/models/genie/, 2025.

[4] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In International Conference on Learning Representations (ICLR), 2025.

[5] Decart and Etched. Oasis: A universe in a transformer. https://oasis-model.github. io/, 2024.

[6] NVIDIA, Niket Agarwal, et al. Cosmos world foundation model platform for physical AI. arXiv preprint arXiv:2501.03575, 2025.

[7] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan. RELIC: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.

[8] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

[9] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-A-Video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

[10] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. CogVideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

[11] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.

[12] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations (ICLR), 2023.

[13] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. VideoPoet: A large language model for zero-shot video generation. In International Conference on Machine Learning (ICML), 2024.

[14] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[15] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

[16] Team Wan. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[17] Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, et al. Genie 2: A large-scale foundation world model. Google DeepMind Blog, 2024.

[18] Anssi Kanervisto, Dave Bignell, Linda Yilin Wen, Martin Grayson, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Tabish Rashid, Tim Pearce, Yuhan Cao, Abdelhak Lemkhenter, Chentian Jiang, Gavin Costello, Gunshi Gupta, Marko Tot, Shu Ishida, Tarun Gupta, Udit Arora, Ryen W. White, Sam Devlin, Cecily Morrison, and Katja Hofmann. World and human action models towards gameplay ideation. Nature, 638(8051):656–663, 2025.

[19] Runway. Introducing GWM-1. Runway Research, 2025.

[20] Chiyu Max Jiang, Xander Masotto, and Bo Sun. The Waymo World Model: A new frontier for autonomous driving simulation. Waymo Blog, 2026.

[21] Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, and Mingyu Ding. VDT: General-purpose video diffusion transformers via mask modeling. In International Conference on Learning Representations, 2024.

[22] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.

[23] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-Video-T2V technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.

[24] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

[25] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[26] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. GODIVA: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.

[27] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic VQGAN and time-sensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022.

[28] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. VideoGPT: Video generation using VQ-VAE and transformers. arXiv preprint arXiv:2104.10157, 2021.

[29] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion Forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[30] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.

[31] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Frédo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22963– 22974, 2025.

[32] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self Forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

[33] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling Forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.

[34] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[35] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[36] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models, 2025.

[37] Guibin Chen, Dixuan Li, Sen Yang, Bin Zhu, Bohan Tan, Mingyu Wang, et al. SkyReels-V2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

[38] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. AR-Diffusion: Asynchronous video generation with auto-regressive diffusion, 2025.

[39] Akio Kodaira, Tingbo Hou, Ji Hou, Markos Georgopoulos, Felix Juefei-Xu, Masayoshi Tomizuka, and Yue Zhao. StreamDiT: Real-time streaming text-to-video generation, 2026.

[40] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. FIFO-Diffusion: Generating infinite videos from text without training. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[41] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. LongLive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

[42] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-Forcing++: Towards minute-scale high-quality video generation, 2025.

[43] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal Forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. In International Conference on Machine Learning (ICML), 2026. arXiv:2602.02214.

[44] Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context Forcing: Consistent autoregressive video generation with long context, 2026.

[45] Kai Zou, Dian Zheng, Hongbo Liu, Tiankai Hang, Bin Liu, and Nenghai Yu. HiAR: Efficient autoregressive long video generation via hierarchical denoising, 2026.

[46] Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Ming-Hsuan Yang, and Weiyang Liu. Streaming autoregressive video generation via Diagonal Distillation. In ICLR, 2026.

[47] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. LoL: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.

[48] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep Forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

[49] Zengqun Zhao, Yanzuo Lu, Ziquan Liu, Jifei Song, Jiankang Deng, and Ioannis Patras. Relax Forcing: Relaxed KV-Memory for consistent long video generation. arXiv preprint arXiv:2603.21366, 2026.

[50] Youngrae Kim, Qixin Hu, C-C Jay Kuo, and Peter A Beerel. MemRoPE: Training-free infinite video generation via evolving memory tokens. arXiv preprint arXiv:2603.12513, 2026.

[51] Yukang Chen, Luozhou Wang, Wei Huang, Shuai Yang, Bohan Zhang, Yicheng Xiao, Ruihang Chu, Weian Mao, Qixin Hu, Shaoteng Liu, et al. LongLive-2.0: An NVFP4 parallel infrastructure for long video generation. arXiv preprint arXiv:2605.18739, 2026.

[52] Jia Li, Xiaomeng Fu, Xurui Peng, Weifeng Chen, Youwei Zheng, Tianyu Zhao, Jiexi Wang, Fangmin Chen, Xing Wang, and Hayden Kwok-Hay So. Train short, inference long: Training-free horizon extension for autoregressive video generation. arXiv preprint arXiv:2602.14027, 2026.

[53] Xiaofeng Mao, Shaohao Rui, Kaining Ying, Bo Zheng, Chuanhao Li, Mingmin Chi, and Kaipeng Zhang. PackForcing: Short video training suffices for long video sampling and long context inference. arXiv preprint arXiv:2603.25730, 2026.

[54] Ruili Feng, Han Zhang, Zhilei Shu, Zhantao Yang, Longxiang Tang, Zhicai Wang, Andy Zheng, Jie Xiao, Zhiheng Liu, Ruihang Chu, et al. The Matrix: Infinite-horizon world generation with real-time moving control. Advances in Neural Information Processing Systems, 38:87318–87344, 2026.

[55] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

[56] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.

[57] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. ReCamMaster: Camera-controlled generative rendering from a single video. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14834–14844, 2025.

[58] Zhening Huang, Hyeonho Jeong, Xuelin Chen, Yulia Gryaditskaya, Tuanfeng Y Wang, Joan Lasenby, and Chun-Hao Huang. SpaceTimePilot: Generative rendering of dynamic scenes across space and time. arXiv preprint arXiv:2512.25075, 2025.

[59] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pages 313–331. Springer, 2024.

[60] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

[61] Hongfei Zhang, Kanghao Chen, Zixin Zhang, Harold Haodong Chen, Yuanhuiyi Lyu, Yuqi Zhang, Shuai Yang, Kun Zhou, and Yingcong Chen. DualCamCtrl: Dual-branch diffusion model for geometry-aware camera-controlled video generation. arXiv preprint arXiv:2511.23127, 2025.

[62] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable Virtual Camera: Generative view synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12405–12414, 2025.

[63] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems, 38:15984–16009, 2026.

[64] Chunyang Li, Yuanbo Yang, Jiahao Shao, Hongyu Zhou, Katja Schwarz, and Yiyi Liao. ReRoPE: Repurposing RoPE for relative camera control. arXiv preprint arXiv:2602.08068, 2026.

[65] Yiming Wang, Qihang Zhang, Shengqu Cai, Tong Wu, Jan Ackermann, Zhengfei Kuang, Yang Zheng, Frano Rajic, Siyu Tang, and Gordon Wetzstein. BulletTime: Decoupled control of time and ˇ camera pose for video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18319–18330, 2026.

[66] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. Advances in Neural Information Processing Systems, 38:41061–41086, 2026.

[67] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. LTX-Video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

[68] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[69] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

[70] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252. PMLR, 2023.

[71] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. MAGI-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

[72] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, and Xun Huang. MotionStream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.

[73] Haocheng Xi, Shuo Yang, Yilong Zhao, Muyang Li, Han Cai, Xingyang Li, Yujun Lin, Zhuoyang Zhang, Jintao Zhang, Xiuyu Li, et al. Quant VideoGen: Auto-regressive long video generation via 2-bit KV-Cache quantization. arXiv preprint arXiv:2602.02958, 2026.

[74] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. DeepSpeed Ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.

[75] Jiacheng Yang, Jun Wu, Yaoyao Ding, Zhiying Xu, Yida Wang, and Gennady Pekhimenko. Swift-Fusion: Scalable sequence parallelism for distributed inference of diffusion transformers on GPUs. In Proceedings of the ACM Conference on AI and Agentic Systems, CAIS ’26, pages 1037–1050, New York, NY, USA, 2026. Association for Computing Machinery.

[76] AMAP, Alibaba. Yunjing AMAP: AI-powered 3D reconstruction and digital twin platform. https: //yunjing.amap.com/, 2023. Accessed: 2026-07.

[77] Mingchao Sun, Luyang Tang, Yu Liu, Xu Yan, Zhan Li, Yunwei Zhang, Fei Yu, Zengye Ge, Yumin Liu, Jiacheng Zhang, et al. ABot-3DWorld 0: A universal world model to explore any 3D space. arXiv preprint arXiv:2607.11673, 2026.

[78] Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, and Tong He. DeepVerse: 4D autoregressive video generation as a world model. arXiv preprint arXiv:2506.01103, 2025.

[79] Ollin Boer Bohan. TAEHV: Tiny AutoEncoder for Hunyuan Video. GitHub repository, 2025.

[80] Lvmin Zhang. FramePack: Practical video diffusion with frame context packing. GitHub repository, 2025.

[81] LightX2V Contributors. LightX2V: Light video generation inference framework. https:// github.com/ModelTC/lightx2v, 2025.

[82] Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. SageAttention2: Efficient attention with thorough outlier smoothing and per-thread INT4 quantization. arXiv preprint arXiv:2411.10958, 2024.

[83] Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. TurboQuant: Online vector quantization with near-optimal distortion rate. arXiv preprint arXiv:2504.19874, 2025.

[84] Ting-Bing Xu, Jiacheng Sui, Zhe Gao, Kewei Shi, Wenjin Yang, Zhicheng Liu, Zhaoxu Sun, Mingchao Sun, Hongyu Pan, Fan Jiang, Mu Xu, Qi Fan, Yang Gao, Yong Li, and Baoquan Chen. WorldRoamBench: An open-world benchmark for long-horizon stability of interactive world models, 2026.

[85] Alibaba Cloud. Alibaba launches HappyOyster, a world model product for real-time immersive creation and interaction. Alibaba Cloud Blog, 2026.

[86] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[87] Tencent Hunyuan. HY-World 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. Technical report, Tencent Hunyuan, 2025.

[88] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. HPSv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15086–15095, 2025.