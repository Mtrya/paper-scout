Title: WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory

URL Source: https://arxiv.org/html/2607.02517

Markdown Content:
Hanlin Wang 1,2 Hao Ouyang 2 Qiuyu Wang 2 Wen Wang 3 Qingyan Bai 1,2

Ka Leong Cheng 2 Yue Yu 1,2 Yixuan Li 4,2 Yihao Meng 1,2 Zichen Liu 1,2

Yanhong Zeng 2 Yujun Shen 2 Qifeng Chen 1

1 HKUST 2 Ant Group 3 ZJU 4 CUHK

###### Abstract

We present WorldDirector, a highly controllable video world model framework designed for persistent dynamic object memory and unrestricted viewpoint exploration. Unlike existing world models that entangle physical dynamics with pixel rendering and rely on continuous visual observation to sustain motion, our framework explicitly decouples semantic motion orchestration from visual generation. By leveraging an LLM to coordinate 3D trajectories with camera movements and subsequently employing these orchestrated trajectories as control signals for video generation, our approach ensures strict physical logic and appearance stability, successfully preserving the exact visual identities of dynamic entities even when they re-enter the scene after prolonged periods out of view. Experimental results demonstrate that our method supports the synthesis of complex and extended events with unprecedented controllability and persistent dynamic object memory.

![Image 1: [Uncaptioned image]](https://arxiv.org/html/2607.02517v1/x1.png)

Figure 1: Controllable world simulation with persistent dynamic memory via WorldDirector. By decoupling 3D semantic orchestration from latent video synthesis, our framework autoregressively generates long-horizon videos via causal chunks, ensuring rigorous dynamic memory and object permanence. Please refer to the video results on our [project page](https://worlddirector.github.io/) for intuitive demonstrations.

## 1 Introduction

The landscape of video generation is undergoing a profound transformation, transitioning from passive pixel synthesis [[7](https://arxiv.org/html/2607.02517#bib.bib7 "Stable video diffusion: scaling latent video diffusion models to large datasets"), [5](https://arxiv.org/html/2607.02517#bib.bib8 "Lumiere: a space-time diffusion model for video generation"), [27](https://arxiv.org/html/2607.02517#bib.bib9 "VideoPoet: a large language model for zero-shot video generation"), [13](https://arxiv.org/html/2607.02517#bib.bib10 "SEINE: short-to-long video diffusion model for generative transition and prediction")] to interactive environment simulation[[8](https://arxiv.org/html/2607.02517#bib.bib3 "Video generation models as world simulators"), [9](https://arxiv.org/html/2607.02517#bib.bib4 "Genie: generative interactive environments"), [11](https://arxiv.org/html/2607.02517#bib.bib11 "GameGen-x: interactive open-world game video generation"), [20](https://arxiv.org/html/2607.02517#bib.bib12 "Matrix-game 2.0: an open-source, real-time, and streaming interactive world model"), [33](https://arxiv.org/html/2607.02517#bib.bib13 "Yume: an interactive world generation model"), [38](https://arxiv.org/html/2607.02517#bib.bib16 "Advancing open-source world models"), [36](https://arxiv.org/html/2607.02517#bib.bib15 "WorldPlay: towards long-term geometric consistency for real-time interactive world modeling")]. A cornerstone of this paradigm shift is memory: the ability to maintain consistency of static scenes and the continuous movements of dynamic objects, whether they are visible in the frame or out-of-view. While recent methods have achieved remarkable success in preserving static scene consistency through memory retrieval or contextual conditioning[[50](https://arxiv.org/html/2607.02517#bib.bib39 "Context as memory: scene-consistent interactive long video generation with memory retrieval"), [46](https://arxiv.org/html/2607.02517#bib.bib40 "WORLDMEM: long-term consistent world simulation with memory"), [21](https://arxiv.org/html/2607.02517#bib.bib37 "StreamingT2V: consistent, dynamic, and extendable long video generation from text"), [35](https://arxiv.org/html/2607.02517#bib.bib38 "FreeNoise: tuning-free longer video diffusion via noise rescheduling"), [44](https://arxiv.org/html/2607.02517#bib.bib54 "Infinite-world: scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory")], a crucial area remains largely unexplored: “Object Permanence” and “Dynamic Object Memory”. Specifically, this entails that dynamic entities persistently exist and execute their physical movements independent of camera visibility. Consequently, whenever the dynamic objects reappear in the frame, their newly updated positions and states should be accurately observed.

To achieve this, we argue that a world simulator with robust dynamic memory must be built upon two foundational pillars. First, entities must exhibit independent motion. Their trajectories should follow continuous physical logic unconstrained by camera visibility, ensuring that unobserved dynamics progress naturally. Second, the system must guarantee strict appearance consistency. When a hidden entity re-enters the frame, its visual identity and fine details must remain entirely intact without distortion. Satisfying these two criteria is the prerequisite for elevating unpredictable video generation to the level of persistent world simulation. Driven by this goal, several methods have been proposed to realize world simulators equipped with dynamic memory recently. One framework[[16](https://arxiv.org/html/2607.02517#bib.bib43 "LiveWorld: simulating out-of-sight dynamics in generative video world models")] introduces a monitor-based mechanism to address out-of-sight dynamics by registering explicit “monitors” that autonomously track and fast-forward the temporal progression of unobserved active entities. However, this explicit tracking system scales poorly and incurs prohibitive computational overhead in scenarios involving multiple dynamic entities. Conversely, another approach[[12](https://arxiv.org/html/2607.02517#bib.bib42 "Out of sight but not out of mind: hybrid memory for dynamic video world models")] tracks dynamic features but delegates trajectory extrapolation entirely to internal generative priors. While this implicit estimation might suffice for brief occlusions, it fails during prolonged camera diversions or intricate dynamic interactions. Relying on generative weights to guess continuous physical evolution without a dedicated orchestration mechanism inevitably leads to trajectory collapse, frozen states, or severe identity errors upon re-entry.

To overcome the aforementioned limitations and fulfill the two foundational pillars, we introduce WorldDirector. Our primary insight is to explicitly decouple the motion planning of dynamic objects from the video synthesis process. By leveraging controllable generation paradigms, we transmit semantic-level planning results as conditions to the generative model, thereby realizing a persistent world simulator equipped with robust dynamic memory. This architecture not only guarantees the independent and continuous movements of dynamic objects, but also provides high controllability, enabling users to independently dictate the specific actions and semantic behaviors of multiple distinct dynamic entities. Specifically, we employ an LLM to act as a central orchestrator, which translates user instructions into 3D bounding box and camera trajectories. These spatial plans are subsequently projected into 2D bounding box sequences, providing location conditions for video synthesis. To prevent identity distortion when a hidden entity re-enters the frame, we propose an Appearance Binding mechanism that injects RGB dynamic object features from context as visual anchors. For granular state control, a spatial-aware cross-attention mechanism[[41](https://arxiv.org/html/2607.02517#bib.bib47 "The world is your canvas: painting promptable events with reference images, trajectories, and text")] routes entity-specific text prompts to their corresponding regions. Integrated within a causal autoregressive architecture, these mechanisms ensure extended video generation with strict dynamic memory.

Extensive evaluations demonstrate that WorldDirector synthesizes highly controllable dynamic scenarios while rigorously maintaining dynamic memory across extended sequences. By ensuring object permanence and appearance consistency after prolonged out-of-view intervals, our approach transcends passive video generation and represents a significant step toward interactive and persistent world simulators with unprecedented dynamic object memory.

## 2 Related Works

### 2.1 Foundation Video Models and World Simulators

Generative video synthesis has progressed rapidly with diffusion and transformer architectures[[7](https://arxiv.org/html/2607.02517#bib.bib7 "Stable video diffusion: scaling latent video diffusion models to large datasets"), [5](https://arxiv.org/html/2607.02517#bib.bib8 "Lumiere: a space-time diffusion model for video generation"), [27](https://arxiv.org/html/2607.02517#bib.bib9 "VideoPoet: a large language model for zero-shot video generation"), [13](https://arxiv.org/html/2607.02517#bib.bib10 "SEINE: short-to-long video diffusion model for generative transition and prediction")]. Beyond pixel fidelity, the field is increasingly shifting toward video world models for simulating interactive environments. Pioneering works such as Sora[[8](https://arxiv.org/html/2607.02517#bib.bib3 "Video generation models as world simulators")], Genie[[9](https://arxiv.org/html/2607.02517#bib.bib4 "Genie: generative interactive environments")], Oasis[[15](https://arxiv.org/html/2607.02517#bib.bib5 "Oasis: a universe in a transformer")], and DIAMOND[[1](https://arxiv.org/html/2607.02517#bib.bib6 "Diffusion for world modeling: visual details matter in atari")] treat generative models as rudimentary physics engines, with further advances in game-like simulators[[11](https://arxiv.org/html/2607.02517#bib.bib11 "GameGen-x: interactive open-world game video generation"), [20](https://arxiv.org/html/2607.02517#bib.bib12 "Matrix-game 2.0: an open-source, real-time, and streaming interactive world model")] and long-sequence interactive generators[[33](https://arxiv.org/html/2607.02517#bib.bib13 "Yume: an interactive world generation model"), [22](https://arxiv.org/html/2607.02517#bib.bib14 "Relic: interactive video world model with long-horizon memory"), [36](https://arxiv.org/html/2607.02517#bib.bib15 "WorldPlay: towards long-term geometric consistency for real-time interactive world modeling")]. However, relying on generative models to implicitly memorize object states, actions, and appearances overloads their capacity: when active entities exit the camera’s field of view, these entangled models fail to sustain dynamics, causing objects to freeze or vanish. Our work addresses this by explicitly decoupling semantic motion orchestration from visual rendering.

### 2.2 Controllable Video Generation

To move beyond random generation, controllable synthesis has been widely explored. Image control mechanisms[[51](https://arxiv.org/html/2607.02517#bib.bib17 "Adding conditional control to text-to-image diffusion models"), [34](https://arxiv.org/html/2607.02517#bib.bib18 "T2i-adapter: learning adapters to dig out more controllable ability for text-to-image diffusion models"), [25](https://arxiv.org/html/2607.02517#bib.bib20 "Vace: all-in-one video creation and editing"), [2](https://arxiv.org/html/2607.02517#bib.bib21 "Scaling instruction-based video editing with a high-quality synthetic dataset")] have been extended to video[[52](https://arxiv.org/html/2607.02517#bib.bib19 "ControlVideo: training-free controllable text-to-video generation")]. For spatial and motion control, Boximator[[42](https://arxiv.org/html/2607.02517#bib.bib23 "Boximator: generating rich and controllable motions for video synthesis")] and GLIGEN[[30](https://arxiv.org/html/2607.02517#bib.bib22 "GLIGEN: open-set grounded text-to-image generation")] leverage bounding boxes, while others target camera trajectories[[19](https://arxiv.org/html/2607.02517#bib.bib24 "CameraCtrl: enabling camera control for text-to-video generation")] or motion tracking[[49](https://arxiv.org/html/2607.02517#bib.bib26 "DragNUWA: fine-grained control in video generation by integrating text, image, and trajectory"), [43](https://arxiv.org/html/2607.02517#bib.bib25 "MotionCtrl: a unified and flexible motion controller for video generation")]. More recently, dense or point-trajectory guidance has emerged as a flexible interface for fine-grained, entity-level motion control[[53](https://arxiv.org/html/2607.02517#bib.bib27 "Tora: trajectory-oriented diffusion transformer for video generation"), [45](https://arxiv.org/html/2607.02517#bib.bib28 "DragAnything: motion control for anything using entity representation"), [18](https://arxiv.org/html/2607.02517#bib.bib29 "Motion prompting: controlling video generation with motion trajectories"), [28](https://arxiv.org/html/2607.02517#bib.bib30 "MagicMotion: controllable video generation with dense-to-sparse trajectory guidance"), [40](https://arxiv.org/html/2607.02517#bib.bib31 "Levitor: 3d trajectory oriented image-to-video synthesis"), [14](https://arxiv.org/html/2607.02517#bib.bib32 "Wan-Move: motion-controllable video generation via latent trajectory guidance")]. Though effective in short clips, these methods lack the autoregressive memory required for long-horizon simulation. Our framework adopts the spatial-aware cross-attention of GLIGEN[[30](https://arxiv.org/html/2607.02517#bib.bib22 "GLIGEN: open-set grounded text-to-image generation")] within a persistent memory architecture, enabling consistent control across extended temporal windows.

### 2.3 Memory Mechanisms in Video World Models

Memory underpins temporal coherence beyond the immediate context window. Long video generators[[21](https://arxiv.org/html/2607.02517#bib.bib37 "StreamingT2V: consistent, dynamic, and extendable long video generation from text"), [35](https://arxiv.org/html/2607.02517#bib.bib38 "FreeNoise: tuning-free longer video diffusion via noise rescheduling")] use sliding windows but struggle with extended occlusion. Prior work preserves static-scene consistency via FOV retrieval[[50](https://arxiv.org/html/2607.02517#bib.bib39 "Context as memory: scene-consistent interactive long video generation with memory retrieval"), [46](https://arxiv.org/html/2607.02517#bib.bib40 "WORLDMEM: long-term consistent world simulation with memory")] or 3D representations[[29](https://arxiv.org/html/2607.02517#bib.bib41 "VMem: consistent interactive video scene generation with surfel-indexed view memory")], yet assumes a static world. Object permanence, defined as objects persisting and evolving when unobserved, remains a core challenge in physical reasoning[[48](https://arxiv.org/html/2607.02517#bib.bib44 "CLEVRER: collision events for video representation and reasoning"), [4](https://arxiv.org/html/2607.02517#bib.bib45 "PHYRE: a new benchmark for physical reasoning"), [6](https://arxiv.org/html/2607.02517#bib.bib46 "Revisiting feature prediction for learning visual representations from video")], and is even harder for active entities in complex scenes. Recent approaches employ implicit hybrid memory tokens[[12](https://arxiv.org/html/2607.02517#bib.bib42 "Out of sight but not out of mind: hybrid memory for dynamic video world models")] or external monitors for out-of-sight dynamic simulation[[16](https://arxiv.org/html/2607.02517#bib.bib43 "LiveWorld: simulating out-of-sight dynamics in generative video world models")]. However, internal priors risk trajectory collapse during prolonged diversions, while external monitors are computationally prohibitive. Coupled with an Appearance Binding mechanism, our LLM-orchestrated approach delivers controllable generation and robust dynamic object memory, offering a scalable path to object permanence in world exploration.

## 3 Method

![Image 2: Refer to caption](https://arxiv.org/html/2607.02517v1/x2.png)

Figure 2: Overview of WorldDirector. An LLM orchestrates 3D trajectories that are projected into 2D Location Conditions for causal chunk generation. Location (\mathcal{B}) and Appearance (\mathcal{A}) conditions are channel-concatenated with the noisy latent, while historical Context (\mathcal{M}) is sequence-concatenated. During generation, temporal drop is applied and an asymmetric attention routing prevents noise from polluting the context memory.

This section outlines our data curation pipeline (Section[3.1](https://arxiv.org/html/2607.02517#S3.SS1 "3.1 Data Curation Pipeline ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory")), model design (Section[3.2](https://arxiv.org/html/2607.02517#S3.SS2 "3.2 Building Controllable World Simulator with Persistent Dynamic Memory ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory")), training objective and inference workflow (Section[3.4](https://arxiv.org/html/2607.02517#S3.SS4 "3.4 Training and Inference ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory")). An overview of WorldDirector is illustrated in Figure[2](https://arxiv.org/html/2607.02517#S3.F2 "Figure 2 ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). We use the LingBot-World-Base model[[38](https://arxiv.org/html/2607.02517#bib.bib16 "Advancing open-source world models")] as our foundation model.

### 3.1 Data Curation Pipeline

We introduce a tailored data curation pipeline to address the foundational requirements outlined in Section[1](https://arxiv.org/html/2607.02517#S1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). Specifically, this pipeline constructs comprehensive training tuples that encapsulate: 2D bounding boxes for dynamic spatial grounding, appearance references for fine-grained visual conditioning, object-centric captions detailing behavioral dynamics, and contextual signals to facilitate causal generation. The proposed pipeline comprises the following key components:

Dynamic Object Tracking and Entity-Based Captioning. To address the scarcity of real-world data featuring dynamic entities exiting and re-entering the field of view (FOV), we developed a game-based platform to generate 15-second videos with precise camera parameters, deliberately scripted to induce target disappearances and reappearances. We employ SAM3[[10](https://arxiv.org/html/2607.02517#bib.bib48 "Sam 3: segment anything with concepts")] to extract 2D bounding box trajectories; its robust re-identification seamlessly tracks objects despite temporary FOV exits, ensuring highly reliable annotations. For training, we sample a contiguous 5-second window from each video that maximizes the number of newly visible objects (absent in the first frame but appearing later) to specifically capture critical re-entry scenarios. The remaining 10 seconds function as a candidate pool to provide historical appearance and spatiotemporal context. Finally, we superimpose unique color-coded bounding boxes onto the source frames to preserve instance identities and feed these visually augmented sequences into Qwen2.5-VL-72B[[3](https://arxiv.org/html/2607.02517#bib.bib49 "Qwen2. 5-vl technical report")] to generate fine-grained textual captions of each entity’s action dynamics.

Dual-Conditioning Preparation for Dynamic Objects. We construct two conditioning videos for each training sequence. First, to encode spatio-temporal trajectories and provide positional priors, we generate a spatial location condition video by filling each dynamic object’s 2D bounding box with a unique color identifier against a zero-initialized background. Second, to ensure appearance consistency for re-entering objects regardless of their absence duration, we introduce an appearance conditioning video. Specifically, for an object a at frame t with bounding box \text{box}_{a,t}, we retrieve a reference \text{box}_{a,t^{\prime}} of the identical object from the 10-second candidate pool that minimizes aspect ratio divergence relative to \text{box}_{a,t}. The image region within \text{box}_{a,t^{\prime}} is then cropped, spatially resampled, and mapped onto the coordinates of \text{box}_{a,t} in the appearance video, directly equipping the model with exact visual features at designated spatio-temporal indices.

Static and Dynamic Context Retrieval. To better support robust causal inference and preserve spatio-temporal consistency, we retrieve context through a dual-perspective approach. For static scenes, we follow [[50](https://arxiv.org/html/2607.02517#bib.bib39 "Context as memory: scene-consistent interactive long video generation with memory retrieval")] to retrieve the top-K frames maximizing Field of View (FoV) overlap. For dynamic objects, we introduce a greedy algorithm prioritizing frames with active object identities within the current temporal chunk. To ensure uniform spatio-temporal distribution, we enforce a minimum temporal stride of four frames between selected frames. Ranked lists derived from both strategies are then interleaved and deduplicated to yield the final N memory frame indices. The detailed selection procedure is outlined in Appendix[B](https://arxiv.org/html/2607.02517#A2 "Appendix B Details of Static and Dynamic Context Retrieval. ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory").

### 3.2 Building Controllable World Simulator with Persistent Dynamic Memory

Building upon our established data curation pipeline, we present WorldDirector, a framework that reconceptualizes video generation as a controllable world simulation with persistent dynamic memory. We formulate this objective as a conditional denoising process guided by multi-modal structural priors. Formally, let V\in\mathbb{R}^{T\times 3\times H\times W} denote a training video sequence comprising T frames. The generative process is conditioned on a composite tuple \mathcal{T}=\{\mathcal{B},\mathcal{A},\mathcal{P},\mathcal{M}\}:

*   •
Location Condition\mathcal{B}\in\mathbb{R}^{T\times 3\times H\times W} encodes the precise spatiotemporal trajectories of 2D bounding boxes for all entities, rendered as identity-preserving, color-coded masks.

*   •
Appearance Condition\mathcal{A}\in\mathbb{R}^{T\times 3\times H\times W} provides sparse RGB features derived from contextual frames to maintain dynamic object appearance consistency across the sequence.

*   •
Multi-Granularity Prompts\mathcal{P}=\{p_{\text{global}},p_{1},p_{2},\dots,p_{k}\} consists of a global prompt p_{\text{global}} summarizing the overall video narrative, coupled with fine-grained textual descriptions \{p_{i}\}_{i=1}^{k} detailing the specific semantic behaviors of k dynamic entities.

*   •
Contextual Memory Frames\mathcal{M} represents contextual frames retrieved via a dual-stream selection strategy, paired with their corresponding location and appearance conditioning to align feature dimensions with the current generation window, thereby anchoring the generated content within the broader global scene.

In this section, we elaborate on how these multi-modal conditioning priors are leveraged to accomplish a dynamic memory-augmented world simulation.

#### 3.2.1 Control of Spatial Location and Visual Appearance

To achieve a controllable world model with strict dynamic consistency, we extend the LingBot-World-Base architecture with auxiliary feature channels for spatial (\mathcal{B}) and appearance (\mathcal{A}) constraints, enabling high-fidelity free-exploration simulations with high physical fidelity. Specifically, \mathcal{B} employs instance-specific color-coded masks to explicitly distinguish multiple entities, serving as a deterministic geometric prior for their trajectories, shapes, and orientations. Simultaneously, \mathcal{A} anchors historical visual features to ensure identity coherence, preventing visual degradation when entities re-enter the camera view. To prevent the model from over-relying on \mathcal{A} and generating unnatural sliding artifacts where entities merely translate without exhibiting proper articulated motion, we introduce a Temporal Drop Mechanism. For each dynamic entity, we preserve a dense sequence of the initial 16 frames immediately following its entry into the view. Subsequently, we employ a sparse sampling strategy, retaining only one reference frame per six-frame interval. This information bottleneck compels the model to synthesize natural object movements driven by trajectories and semantic captions, utilizing \mathcal{A} strictly as an identity anchor.

Architecturally, both \mathcal{B} and \mathcal{A} are encoded by a 3D VAE into latent tokens and concatenated with the noisy latent sequence along the feature dimension:

z_{\text{in}}=\text{Conv3D}\Big(z_{t}\oplus\mathcal{E}(\mathcal{B})\oplus\mathcal{E}\big(\mathcal{D}_{\tau}(\mathcal{A})\big)\Big),(1)

where z_{t} denotes the noisy latent, and \mathcal{E}(\cdot) represents the pre-trained 3D VAE encoder. \oplus denotes channel concatenation. \mathcal{D}_{\tau}(\cdot) formulates our Temporal Drop Mechanism, defined as:

\mathcal{D}_{\tau}\big(\mathcal{A}_{t}^{(i)}\big)=\begin{cases}\mathcal{A}_{t}^{(i)},&\text{if }k^{(i)}<16\\
\mathcal{A}_{t}^{(i)},&\text{if }k^{(i)}\geq 16\text{ and }(k^{(i)}-16)\pmod{6}=0\\
\mathbf{0},&\text{otherwise}.\end{cases}(2)

Here, \mathcal{A}_{t}^{(i)} represents the appearance condition for entity i at global frame t, and k^{(i)}\geq 0 is its instance-specific relative frame index (i.e., the number of frames elapsed since entity i newly entered the view). The full appearance conditioning feature \mathcal{D}_{\tau}(\mathcal{A}) aggregates these processed entity-level representations, where masked entities are replaced by null embeddings \mathbf{0}. The fused latent is then processed by a dedicated layer \text{Conv3D}(\cdot), where the channel weights corresponding to \mathcal{E}\big(\mathcal{D}_{\tau}(\mathcal{A})\big) are initialized from the first-frame processor for RGB identity transfer, whereas those for \mathcal{E}(\mathcal{B}) are zero-initialized to learn trajectory-guided generation as a residual process.

#### 3.2.2 Contextual Integration

To preserve the consistency of both static scenes and dynamic objects during causal chunk generation, we integrate the retrieved context \mathcal{M} through sequence-level concatenation by prepending the context frames to the noisy latent sequence along the temporal axis. To enforce structural alignment, the location and appearance conditioning associated with each context frame are concatenated along the feature dimension, strictly mirroring the input formulation of the training segment. Furthermore, to explicitly disentangle these historical anchors from the current generative chunk, the time steps used for Rotary Position Embedding (RoPE) of the context frames are shifted by an offset substantially exceeding the maximum training sequence length, establishing a definitive frequency boundary within the RoPE representation space. To prevent the noisy training latent from polluting the high-fidelity context \mathcal{M}, we impose an asymmetric attention mask where context tokens exclusively self-attend to remain stable, noise-free references. This allows the model to leverage historical priors without compromising contextual integrity. Finally, to equip the model with the capability to directly generate the initial sequence chunk from scratch, we randomly discard the contextual information \mathcal{M} with a probability of 30% during the training phase.

### 3.3 Camera Injection and Spatial-Aware Text Control

To accurately model perspective variations and effectively leverage contextual camera information during generation, we first convert the camera poses of all context frames and the current video chunk into relative camera poses with respect to the first frame of the current generated chunk. Following Wan[[39](https://arxiv.org/html/2607.02517#bib.bib50 "Wan: open and advanced large-scale video generative models")], we utilize Plücker coordinates to encode these relative intrinsic and extrinsic parameters. Next, we apply a spatial downsampling to this representation, followed by a series of convolutional modules to extract multi-level camera motion embeddings. These embeddings are subsequently injected into each Diffusion Transformer (DiT) block via an adaptive normalization layer.

For textual condition injection, it is crucial to guarantee that entity-specific captions are precisely grounded in their corresponding spatial regions. To achieve this, we adopt the Spatial-Aware Weighted Cross-Attention mechanism from [[41](https://arxiv.org/html/2607.02517#bib.bib47 "The world is your canvas: painting promptable events with reference images, trajectories, and text")]. Rather than computing cross-attention uniformly across the entire frame, this scheme identifies the visual tokens encompassed by each entity’s 2D bounding box trajectory. We then apply a targeted spatial weight bias to the pre-softmax attention logits between these localized visual tokens and the specific text tokens describing that entity. By doing so, it effectively mitigates semantic leakage and facilitates fine-grained control over multiple dynamic objects within the synthesized scene.

### 3.4 Training and Inference

We follow the flow matching framework[[31](https://arxiv.org/html/2607.02517#bib.bib1 "Flow matching for generative modeling"), [17](https://arxiv.org/html/2607.02517#bib.bib2 "Scaling rectified flow transformers for high-resolution image synthesis")] to perform post-training using the mean squared error (MSE) loss. The training objective is applied exclusively to the current target segment, while the historical context remains non-noisy and serves solely as a reference. Formally, let x_{1} denote the ground-truth latent of the target video chunk and x_{0}\sim\mathcal{N}(0,I) be the random noise. At a sampled timestep t\in[0,1], the training input for the target portion is x_{tgt,t}=tx_{1}+(1-t)x_{0}, with the corresponding ground-truth velocity defined as v_{t}=x_{1}-x_{0}. As described in Section [3.2](https://arxiv.org/html/2607.02517#S3.SS2 "3.2 Building Controllable World Simulator with Persistent Dynamic Memory ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), the model u receives a concatenated sequence [x_{ctx},x_{tgt,t}], where x_{ctx} represents the clean context tokens. The training objective is formulated as:

\mathcal{L}=\mathbb{E}_{x_{0},x_{1},t,\Omega}\left[\sum_{i\in\mathcal{I}_{tgt}}\left\|u(x_{t},t,\Omega;\theta)_{i}-v_{t,i}\right\|^{2}\right],(3)

where \Omega=\{\mathcal{B},\mathcal{A},\mathcal{P},\mathcal{M}\} is the union of all location, appearance, text, and contextual conditions, and \mathcal{I}_{tgt} denotes the set of token indices belonging to the current video segment. By restricting the loss calculation to \mathcal{I}_{tgt}, we ensure that the model learns to synthesize new content anchored by the clean memory of previous frames without attempting to reconstruct the already-determined context. During inference, our method operates in two primary stages as described below: World Planning via LLM and Causal Chunk-Based Generation. Further details regarding the inference implementation are provided in Appendix[C](https://arxiv.org/html/2607.02517#A3 "Appendix C Details of Inference System ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory").

World Planning via LLM. We first estimate the 3D bounding boxes of target dynamic objects in the given initial image to provide a foundational spatial context for the LLM, which then forecasts continuous 3D box trajectories—comprising both spatial coordinates and orientations—alongside our designed camera path. This trajectory planning encompasses not only the entities present in the initial frame but also those that appear later. Objects absent from the initial frame are synthesized based on their captions when they first enter the camera view. Subsequent generations are then conditioned on these initial outputs to maintain appearance consistency. These 3D trajectories are then projected onto the 2D image plane to yield a sequence of 2D bounding boxes, formulating a spatial condition \mathcal{B} that strictly aligns with the location conditioning format employed during our training phase.

Causal Chunk-Based Generation. To facilitate computationally efficient long-horizon world exploration, we introduce an autoregressive chunk-based generation strategy (detailed in Appendix[C](https://arxiv.org/html/2607.02517#A3 "Appendix C Details of Inference System ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory")). The projected 2D location condition \mathcal{B} is partitioned into contiguous temporal chunks. During the first chunk generation, the process relies exclusively on the initial reference frame for the appearance condition \mathcal{A}, with an empty memory context \mathcal{M}. For all subsequent chunks, we recursively construct \mathcal{A} and retrieve \mathcal{M} from the continuously updated pool of previously generated chunks. This causal loop explicitly preserves entity identities and spatiotemporal consistency throughout the dynamic simulation, ultimately facilitating arbitrary-length world exploration.

## 4 Experiments

Implementation Details We build WorldDirector on the pre-trained LingBot-World-Base model[[38](https://arxiv.org/html/2607.02517#bib.bib16 "Advancing open-source world models")]. All training videos are pre-processed to a fixed resolution of 832\times 480 pixels at 16 fps. For conditioning, the context length is set to N=10 frames, with each frame independently encoded via the pre-trained 3D VAE. Our model is trained for 3,000 steps utilizing a global batch size of 64 and a constant learning rate of 1\times 10^{-5}. During inference, we leverage Gemini[[37](https://arxiv.org/html/2607.02517#bib.bib51 "Gemini: a family of highly capable multimodal models")] as the orchestrator to plan 3D trajectories and states for all dynamic entities. Subsequently, the full-length video is partitioned into five-second segments and generated chunk by chunk in an autoregressive manner. Comprehensive prompt templates for the LLM are detailed in the supplementary material.

Baselines. We compare WorldDirector with state-of-the-art causal interactive world models: Yume 1.5[[32](https://arxiv.org/html/2607.02517#bib.bib52 "Yume-1.5: a text-controlled interactive world generation model")], which uses uniform temporal downsampling for memory; HY-World 1.5[[24](https://arxiv.org/html/2607.02517#bib.bib53 "HY-world 1.5: a systematic framework for interactive world modeling with real-time latency and geometric consistency")], applying FOV-based attention on mixed data to achieve memorization; Infinite World[[44](https://arxiv.org/html/2607.02517#bib.bib54 "Infinite-world: scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory")], which achieves memorization through hierarchical context compression; LingBot-World-Fast[[38](https://arxiv.org/html/2607.02517#bib.bib16 "Advancing open-source world models")], leveraging causal attention for infinite generation; and HyDRA[[12](https://arxiv.org/html/2607.02517#bib.bib42 "Out of sight but not out of mind: hybrid memory for dynamic video world models")], which utilizes spatiotemporal retrieval for maintaining off-screen character motion.

Evaluation Protocol. To evaluate our method, we use our data pipeline to construct a test set of 100 video samples featuring novel scenes and subjects that are unseen during training. Following HyDRA[[12](https://arxiv.org/html/2607.02517#bib.bib42 "Out of sight but not out of mind: hybrid memory for dynamic video world models")], we evaluate our model using PSNR, SSIM, and LPIPS to measure overall reconstruction fidelity via pixel-wise analysis, along with VBench’s[[23](https://arxiv.org/html/2607.02517#bib.bib55 "Vbench: comprehensive benchmark suite for video generative models")] Subject and Background Consistency for frame-level coherence. We also adopt Dynamic Subject Consistency (DSC) by cropping YOLO-detected bounding boxes of dynamic objects and computing their average DINO and CLIP similarities with their contextual counterparts. This metric effectively captures dynamic object consistency, especially for off-screen reappearance.

### 4.1 Comparisons

Quantitative Results.

Table 1: Quantitative results. The best and runner-up are in bold and underlined.

Method PSNR↑SSIM↑LPIPS↓Subject Consistency↑Background Consistency↑DSC_DINO↑DSC_CLIP↑
Yume1.5 14.391 0.455 0.425 0.898 0.919 0.765 0.898
HY-World 14.782 0.418 0.398 0.923 0.931 0.758 0.911
Infinite-World 14.574 0.431 0.406 0.934 0.908 0.773 0.913
LingBot-World 14.116 0.409 0.412 0.887 0.911 0.736 0.891
HyDRA 13.421 0.352 0.439 0.855 0.902 0.632 0.877
Ours 18.127 0.502 0.359 0.891 0.909 0.769 0.917

As reported in Table[1](https://arxiv.org/html/2607.02517#S4.T1 "Table 1 ‣ 4.1 Comparisons ‣ 4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), WorldDirector achieves state-of-the-art performance across all three reconstruction metrics. This stems from our location conditioning, which captures continuous object positions and reflects camera poses, facilitating more accurate generation that aligns with the ground truth. For the VBench results, Yume, HY-World, and Infinite-World attain the best performance. However, analyzing the generated videos indicates that this is largely because they generate less subject or camera motion, giving them an inherent advantage when calculating these metrics. Even though these methods also have an inherent advantage on the DSC metric due to their limited motion, our method still attains superior results. This proves our method’s strong capability in preserving dynamic consistency while producing highly dynamic generations.

![Image 3: Refer to caption](https://arxiv.org/html/2607.02517v1/x3.png)

Figure 3: Qualitative comparison with baselines. Note that HyDRA uses the initial 10s of our results as a reference video for its generation. Please refer to the video results on our [project page](https://worlddirector.github.io/) for intuitive demonstrations.

Qualitative Results. We show a qualitative comparison result in Figure[3](https://arxiv.org/html/2607.02517#S4.F3 "Figure 3 ‣ 4.1 Comparisons ‣ 4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). Since HyDRA requires a reference video for motion extraction, we use the first 10s of our result to prompt its subsequent 5s generation. We leveraged Gemini to script a specific scenario: a man stands stationary and then walks away; concurrently, the camera pans left (moving the man out of frame) and later pans back to reveal his reappearance. Comparisons against baselines yield the following observations: (1) Limited dynamic generation: Yume, HY-World, and Infinite-World render the man stationary even though the prompt specifies that the man walks into the distance. (2) Identity inconsistency: While LingBot-World and HyDRA capture the man’s movement, they struggle with identity preservation. Lingbot-World exhibits slight appearance degradation despite keeping the man in-frame, while HyDRA generates a completely new identity upon the man’s reappearance. (3) Insufficient control: Due to the lack of Location Condition, all baselines fail to properly synchronize camera and object dynamics with the user’s design. Lingbot-World automatically generates camera translation to ensure the man remains in the shot; Infinite-World executes camera controls correctly but misses object motion; HyDRA directly ignores the man in the distance and generates a new man walking in front of the camera. In contrast, by explicitly conditioning on location and appearance conditions, our method accurately generates the user-expected scene and maintains the consistency of the man reappearing after a long period of disappearance. We show more qualitative comparison results in Appendix[F](https://arxiv.org/html/2607.02517#A6 "Appendix F More Qualitative Comparisons ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory").

### 4.2 Ablation Studies and Promptable World Events

![Image 4: Refer to caption](https://arxiv.org/html/2607.02517v1/x4.png)

Figure 4: Ablation on Appearance Condition. We conduct experiments on a case involving complex character movements and multiple pose changes. The findings highlight the significance of the Appearance Condition for preserving dynamic consistency.

![Image 5: Refer to caption](https://arxiv.org/html/2607.02517v1/x5.png)

Figure 5: A generation example of Promptable World Events.

Table 2: Quantitative ablation results on Appearance Condition.

Method PSNR↑SSIM↑LPIPS↓Subject Consistency↑Background Consistency↑DSC_DINO↑DSC_CLIP↑
No \mathcal{A}16.764 0.469 0.385 0.878 0.898 0.693 0.882
No \mathcal{A} + routing 17.461 0.486 0.372 0.881 0.901 0.686 0.886
Ours 18.127 0.502 0.359 0.891 0.909 0.769 0.917

Ablation Studies. We investigate whether the model can implicitly maintain visual consistency without the explicit Appearance Condition. Assuming the unique color-coded masks in the Location Condition could guide appearance retrieval from the contextual frames, we observe that the model fails to autonomously leverage this context, causing severe identity loss for re-entering dynamic objects (Figure[4](https://arxiv.org/html/2607.02517#S4.F4 "Figure 4 ‣ 4.2 Ablation Studies and Promptable World Events ‣ 4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), second row). Attempting to resolve this without introducing new condition channels, we applied a heuristic self-attention routing strategy to amplify the attention weights between current and contextual dynamic object tokens sharing the same identity. Although this explicit bias captures general styles (the color of people’s apparel remains consistent), it fundamentally disrupts the pre-trained latent distribution, inducing severe artifacts, blurring, and the loss of fine-grained textures (Figure[4](https://arxiv.org/html/2607.02517#S4.F4 "Figure 4 ‣ 4.2 Ablation Studies and Promptable World Events ‣ 4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), third row). We attribute these failures to imbalances in pixel distributions in the training data. Since static backgrounds account for the majority of pixels and dominate the MSE loss, the model struggles to implicitly learn the complex mappings required for high-fidelity consistency in small dynamic regions. Results in Table [2](https://arxiv.org/html/2607.02517#S4.T2 "Table 2 ‣ 4.2 Ablation Studies and Promptable World Events ‣ 4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory") also show that all metrics drop without the Appearance Condition. This confirms explicitly injecting Appearance Condition is necessary for dynamic memory. We also conduct ablations on Dynamic Context and Appearance Condition Drop in Appendix[D](https://arxiv.org/html/2607.02517#A4 "Appendix D Ablation on Dynamic Context and Appearance Condition Drop Mechanism ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory").

Promptable World Events. Our framework is not constrained by the entities present in the initial frame. The LLM can freely populate the simulated world by defining identities, entrance timings, and 3D motion trajectories for novel objects. Upon first entering the camera view, their appearance and movements are synthesized directly from text prompts and appended to the Appearance Condition pool to ensure subsequent temporal consistency. As shown in Figure[5](https://arxiv.org/html/2607.02517#S4.F5 "Figure 5 ‣ 4.2 Ablation Studies and Promptable World Events ‣ 4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), this paradigm enables the simultaneous choreography of multiple emerging entities alongside unconstrained camera exploration. Consequently, rather than merely extrapolating existing video content, our approach provides a highly controllable mechanism for open-ended scene generation and dynamic environment simulation.

## 5 Conclusion

We present WorldDirector, a novel framework for free exploration and flexible event design in video world models while preserving rigorous dynamic memory. By decoupling semantic orchestration from latent synthesis, WorldDirector empowers LLMs to plan complex 3D trajectories and open-world events. Abstract planning is visually realized via causal chunk-based context routing, utilizing spatial and appearance conditioning. Experiments confirm our approach maintains rigorous dynamic consistency, establishing a highly controllable paradigm for future video world models.

Limitation. Relying on synthetic game data introduces a domain gap that occasionally restricts visual fidelity (e.g., unnatural locomotion or blurry faces). Future work will incorporate real-world datasets to bridge this gap and enhance overall visual realism.

## References

*   [1]E. Alonso, A. Jelley, V. Micheli, A. Kanervisto, A. Storkey, T. Pearce, and F. Fleuret (2024)Diffusion for world modeling: visual details matter in atari. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [2]Q. Bai, Q. Wang, H. Ouyang, Y. Yu, H. Wang, W. Wang, K. L. Cheng, S. Ma, Y. Zeng, Z. Liu, et al. (2025)Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742. Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [3]S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025)Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. Cited by: [§3.1](https://arxiv.org/html/2607.02517#S3.SS1.p2.1 "3.1 Data Curation Pipeline ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [4]A. Bakhtin, L. van der Maaten, J. Johnson, L. Gustafson, and R. Girshick (2019)PHYRE: a new benchmark for physical reasoning. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [5]O. Bar-Tal, H. Chefer, O. Tov, C. Herrmann, R. Paiss, S. Zada, A. Ephrat, J. Hur, G. Liu, A. Raj, et al. (2024)Lumiere: a space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [6]A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas (2024)Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research (TMLR). Cited by: [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [7]A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, V. Jampani, and R. Rombach (2023)Stable video diffusion: scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [8]T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh (2024)Video generation models as world simulators. External Links: [Link](https://openai.com/research/video-generation-models-as-world-simulators)Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [9]J. Bruce, M. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. (2024)Genie: generative interactive environments. In International Conference on Machine Learning (ICML), Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [10]N. Carion, L. Gustafson, Y. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, et al. (2025)Sam 3: segment anything with concepts. arXiv preprint arXiv:2511.16719. Cited by: [§3.1](https://arxiv.org/html/2607.02517#S3.SS1.p2.1 "3.1 Data Curation Pipeline ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [11]H. Che, X. He, Q. Liu, C. Jin, and H. Chen (2025)GameGen-x: interactive open-world game video generation. In International Conference on Learning Representations (ICLR), Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [12]K. Chen, D. Liang, X. Zhou, Y. Ding, X. Liu, P. Wan, and X. Bai (2026)Out of sight but not out of mind: hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p2.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§4](https://arxiv.org/html/2607.02517#S4.p2.1 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§4](https://arxiv.org/html/2607.02517#S4.p3.1 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [13]X. Chen, Y. Wang, L. Zhang, S. Zhuang, X. Ma, J. Yu, Y. Wang, D. Lin, Y. Qiao, and Z. Liu (2024)SEINE: short-to-long video diffusion model for generative transition and prediction. In International Conference on Learning Representations (ICLR), Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [14]R. Chu, Y. He, Z. Chen, S. Zhang, X. Xu, B. Xia, D. Wang, H. Yi, X. Liu, H. Zhao, Y. Liu, Y. Zhang, and Y. Yang (2025)Wan-Move: motion-controllable video generation via latent trajectory guidance. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [15]Decart, J. Quevedo, Q. McIntyre, S. Campbell, X. Chen, and R. Wachen (2024)Oasis: a universe in a transformer. Note: Project page External Links: [Link](https://oasis-model.github.io/)Cited by: [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [16]Z. Duan, J. Xia, Z. Zhang, W. Zhang, G. Zhou, C. Gou, Y. He, F. Chen, X. Zhang, and L. Liu (2026)LiveWorld: simulating out-of-sight dynamics in generative video world models. arXiv preprint arXiv:2603.07145. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p2.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [17]P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. (2024)Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML), Cited by: [§3.4](https://arxiv.org/html/2607.02517#S3.SS4.p1.8 "3.4 Training and Inference ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [18]D. Geng, C. Herrmann, J. Hur, F. Cole, S. Zhang, T. Pfaff, T. Lopez-Guevara, C. Doersch, Y. Aytar, M. Rubinstein, C. Sun, O. Wang, A. Owens, and D. Sun (2025)Motion prompting: controlling video generation with motion trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [19]H. He, Y. Xu, Y. Guo, G. Wetzstein, B. Dai, H. Li, and C. Yang (2024)CameraCtrl: enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101. Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [20]X. He, C. Peng, Z. Liu, B. Wang, Y. Zhang, Q. Cui, F. Kang, B. Jiang, M. An, et al. (2025)Matrix-game 2.0: an open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [21]R. Henschel, L. Khachatryan, H. Poghosyan, D. Hayrapetyan, V. Tadevosyan, Z. Wang, S. Navasardyan, and H. Shi (2025)StreamingT2V: consistent, dynamic, and extendable long video generation from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [22]Y. Hong, Y. Mei, C. Ge, Y. Xu, Y. Zhou, S. Bi, Y. Hold-Geoffroy, M. Roberts, M. Fisher, E. Shechtman, et al. (2025)Relic: interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040. Cited by: [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [23]Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, et al. (2024)Vbench: comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§4](https://arxiv.org/html/2607.02517#S4.p3.1 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [24]T. HunyuanWorld (2025)HY-world 1.5: a systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint. Cited by: [§4](https://arxiv.org/html/2607.02517#S4.p2.1 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [25]Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu (2025)Vace: all-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [26]A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W. Lo, et al. (2023)Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), Cited by: [§C.1](https://arxiv.org/html/2607.02517#A3.SS1.p1.1 "C.1 World Planning via LLM ‣ Appendix C Details of Inference System ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [27]D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, G. Schindler, R. Hornung, V. Birodkar, J. Yan, M. Chiu, K. Somandepalli, H. Akbari, Y. Alon, Y. Cheng, J. V. Dillon, A. Gupta, M. Hahn, A. Hauth, D. Hendon, A. Martinez, D. Minnen, M. Sirotenko, K. Sohn, X. Yang, H. Adam, M. Yang, I. Essa, H. Wang, D. A. Ross, B. Seybold, and L. Jiang (2024)VideoPoet: a large language model for zero-shot video generation. In Proceedings of the 41st International Conference on Machine Learning (ICML), Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [28]Q. Li, Z. Xing, R. Wang, H. Zhang, Q. Dai, and Z. Wu (2025)MagicMotion: controllable video generation with dense-to-sparse trajectory guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [29]R. Li, P. Torr, A. Vedaldi, and T. Jakab (2025)VMem: consistent interactive video scene generation with surfel-indexed view memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [30]Y. Li, H. Liu, Q. Wu, F. Mu, J. Yang, J. Gao, C. Li, and Y. J. Lee (2023)GLIGEN: open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [31]Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2023)Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), Cited by: [§3.4](https://arxiv.org/html/2607.02517#S3.SS4.p1.8 "3.4 Training and Inference ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [32]X. Mao, Z. Li, C. Li, X. Xu, K. Ying, T. He, J. Pang, Y. Qiao, and K. Zhang (2025)Yume-1.5: a text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096. Cited by: [§4](https://arxiv.org/html/2607.02517#S4.p2.1 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [33]X. Mao, S. Lin, Z. Li, C. Li, W. Peng, T. He, J. Pang, M. Chi, Y. Qiao, and K. Zhang (2025)Yume: an interactive world generation model. arXiv preprint arXiv:2507.17744. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [34]C. Mou, X. Wang, L. Xie, Y. Wu, J. Zhang, Z. Qi, and Y. Shan (2024)T2i-adapter: learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence (AAAI), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [35]H. Qiu, M. Xia, Y. Zhang, Y. He, X. Wang, Y. Shan, and Z. Liu (2023)FreeNoise: tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [36]W. Sun, H. Zhang, H. Wang, J. Wu, Z. Wang, Z. Wang, Y. Wang, J. Zhang, T. Wang, and C. Guo (2025)WorldPlay: towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.1](https://arxiv.org/html/2607.02517#S2.SS1.p1.1 "2.1 Foundation Video Models and World Simulators ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [37]G. Team, R. Anil, S. Borgeaud, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. (2023)Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. Cited by: [§C.1](https://arxiv.org/html/2607.02517#A3.SS1.p1.1 "C.1 World Planning via LLM ‣ Appendix C Details of Inference System ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§4](https://arxiv.org/html/2607.02517#S4.p1.3 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [38]R. Team, Z. Gao, Q. Wang, Y. Zeng, J. Zhu, K. L. Cheng, Y. Li, H. Wang, Y. Xu, S. Ma, et al. (2026)Advancing open-source world models. arXiv preprint arXiv:2601.20540. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§3](https://arxiv.org/html/2607.02517#S3.p1.1 "3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§4](https://arxiv.org/html/2607.02517#S4.p1.3 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§4](https://arxiv.org/html/2607.02517#S4.p2.1 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [39]T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. (2025)Wan: open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314. Cited by: [§3.3](https://arxiv.org/html/2607.02517#S3.SS3.p1.1 "3.3 Camera Injection and Spatial-Aware Text Control ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [40]H. Wang, H. Ouyang, Q. Wang, W. Wang, K. L. Cheng, Q. Chen, Y. Shen, and L. Wang (2025)Levitor: 3d trajectory oriented image-to-video synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [41]H. Wang, H. Ouyang, Q. Wang, Y. Yu, Y. Meng, W. Wang, K. L. Cheng, S. Ma, Q. Bai, Y. Li, et al. (2025)The world is your canvas: painting promptable events with reference images, trajectories, and text. arXiv preprint arXiv:2512.16924. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p3.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§3.3](https://arxiv.org/html/2607.02517#S3.SS3.p2.1 "3.3 Camera Injection and Spatial-Aware Text Control ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [42]J. Wang, Y. Zhang, J. Zou, Y. Zeng, G. Wei, L. Yuan, and H. Li (2024)Boximator: generating rich and controllable motions for video synthesis. arXiv preprint arXiv:2402.01566. Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [43]Z. Wang, Z. Yuan, X. Wang, T. Chen, M. Xia, P. Luo, and Y. Shan (2024)MotionCtrl: a unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [44]R. Wu, X. He, M. Cheng, T. Yang, Y. Zhang, Z. Kang, X. Cai, X. Wei, C. Guo, C. Li, et al. (2026)Infinite-world: scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory. arXiv preprint arXiv:2602.02393. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§4](https://arxiv.org/html/2607.02517#S4.p2.1 "4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [45]W. Wu, Z. Li, Y. Gu, R. Zhao, Y. He, D. J. Zhang, M. Z. Shou, Y. Li, T. Gao, and D. Zhang (2024)DragAnything: motion control for anything using entity representation. In European Conference on Computer Vision (ECCV), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [46]Z. Xiao, Y. Lan, Y. Zhou, W. Ouyang, S. Yang, Y. Zeng, and X. Pan (2025)WORLDMEM: long-term consistent world simulation with memory. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [47]L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao (2024)Depth anything v2. Advances in Neural Information Processing Systems (NeurIPS). Cited by: [§C.1](https://arxiv.org/html/2607.02517#A3.SS1.p1.1 "C.1 World Planning via LLM ‣ Appendix C Details of Inference System ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [48]K. Yi, C. Gan, Y. Li, P. Kohli, J. Wu, A. Torralba, and J. B. Tenenbaum (2020)CLEVRER: collision events for video representation and reasoning. In International Conference on Learning Representations (ICLR), Cited by: [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [49]S. Yin, C. Wu, J. Liang, J. Shi, H. Li, G. Ming, and N. Duan (2023)DragNUWA: fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089. Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [50]J. Yu, J. Bai, Y. Qin, Q. Liu, X. Wang, P. Wan, D. Zhang, and X. Liu (2025)Context as memory: scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141. Cited by: [§1](https://arxiv.org/html/2607.02517#S1.p1.1 "1 Introduction ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§2.3](https://arxiv.org/html/2607.02517#S2.SS3.p1.1 "2.3 Memory Mechanisms in Video World Models ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), [§3.1](https://arxiv.org/html/2607.02517#S3.SS1.p4.2 "3.1 Data Curation Pipeline ‣ 3 Method ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [51]L. Zhang, A. Rao, and M. Agrawala (2023)Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [52]Y. Zhang, Y. Wei, D. Jiang, X. Zhang, W. Zuo, and Q. Tian (2023)ControlVideo: training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077. Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 
*   [53]Z. Zhang, J. Liao, M. Li, Z. Dai, B. Qiu, S. Zhu, L. Qin, and W. Wang (2025)Tora: trajectory-oriented diffusion transformer for video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§2.2](https://arxiv.org/html/2607.02517#S2.SS2.p1.1 "2.2 Controllable Video Generation ‣ 2 Related Works ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). 

## Appendix A Training and Compute Details.

In this section, we provide a more comprehensive breakdown of the training configuration and computational resources.

The training process for WorldDirector is conducted on a high-performance cluster utilizing 8 compute nodes. Each node is equipped with 8 NVIDIA A100 (80GB) GPUs, amounting to a total of 64 GPUs. To maximize memory efficiency and training throughput across this large-scale distributed setup, we employ Fully Sharded Data Parallel (FSDP) alongside activation checkpointing. This system-level optimization ensures that the memory footprint of the 3D VAE encodings, diffusion transformer blocks, and multi-modal conditioning channels is efficiently distributed, preventing out-of-memory bottlenecks when processing high-resolution video chunks and extended context memory.

As formulated in our method, the model processes training videos at a resolution of 832\times 480 pixels at 16 fps, with the context length explicitly set to N=10 frames. We optimize the flow matching objective utilizing the AdamW optimizer with a constant learning rate of 1\times 10^{-5}. To accelerate computation while maintaining numerical stability during the denoising process, the training is conducted using BFloat16 (BF16) mixed precision. Operating with a global batch size of 64, the model undergoes 3,000 optimization steps. Under the 64-GPU distributed configuration, the entire post-training pipeline takes approximately 72 hours (3 days) to fully converge.

## Appendix B Details of Static and Dynamic Context Retrieval.

Algorithm 1 Static and Dynamic Context Retrieval

1:Candidate Context Frames

\mathcal{F}
, Training Frames

\mathcal{V}
, Camera Poses

\mathcal{C}
,

2: 2D Bounding Boxes

\mathcal{B}
, Context Length

N

3:Static and Dynamic Context

\mathcal{M}

4:procedure Static_Context(

\mathcal{F},\,\mathcal{V},\,\mathcal{C}
)

5:for each candidate

c\in\mathcal{F}
do

6:

\mathrm{score}(c)\leftarrow\max_{v\in\mathcal{V}}\,\mathrm{FoV\_Overlap}(\mathcal{C}_{c},\,\mathcal{C}_{v})

7:end for

8:return

\mathcal{F}
sorted by

\mathrm{score}(\cdot)
descending

9:end procedure

10:procedure Dynamic_Context(

\mathcal{F},\,\mathcal{B},\,N,\,\mathcal{V}
)

11: Initialize coverage

\mathrm{cnt}[i]\leftarrow 0
for each dynamic entity

i
appears in

\mathcal{V}

12:while

|\mathrm{selected}|<N
do

13: Find entity

i^{*}=\arg\min_{i}\,\mathrm{cnt}[i]
\triangleright least-covered entity so far

14: Select frame

f^{*}\in\mathcal{F}
with largest

\mathrm{area}(\mathcal{B}^{(i^{*})}_{f})
\triangleright best visible context for entity i^{*}

15: Add

f^{*}
to

\mathrm{selected}
. Remove

f^{*}
from

\mathcal{F}

16:

A_{\max}\leftarrow\max_{e}\,\mathrm{area}(\mathcal{B}^{(e)}_{f^{*}})
\triangleright largest bbox area in f^{*}, used as normalizer

17:for each entity

j
in

f^{*}
do

18:

\mathrm{cnt}[j]\mathrel{+}=\mathrm{area}(\mathcal{B}^{(j)}_{f^{*}})\,/\,A_{\max}
\triangleright normalized bbox area as visibility weight

19:end for

20:end while

21:return

\mathrm{selected}

22:end procedure

23:

\mathcal{P}_{\mathrm{cam}}\leftarrow
Static_Context(

\mathcal{F},\,\mathcal{V},\,\mathcal{C}
)

24:

\mathcal{P}_{\mathrm{box}}\leftarrow
Dynamic_Context(

\mathcal{F},\,\mathcal{B},\,N
)

25:

\mathcal{M}\leftarrow[\,]

26:for

k=0,1,\ldots
until

|\mathcal{M}|=N
do

27: Append

\mathcal{P}_{\mathrm{cam}}[k]
to

\mathcal{M}
if not already in

\mathcal{M}

28: Append

\mathcal{P}_{\mathrm{box}}[k]
to

\mathcal{M}
if not already in

\mathcal{M}

29:end for

30:return

\mathcal{M}
sorted by temporal order

In this section, we elaborate on the specific implementation details of the static and dynamic context retrieval mechanism (as outlined in Algorithm[1](https://arxiv.org/html/2607.02517#alg1 "Algorithm 1 ‣ Appendix B Details of Static and Dynamic Context Retrieval. ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory")). This algorithm aims to select N memory frames from a candidate frame set \mathcal{F} to construct the final context set \mathcal{M}. The inputs primarily include the candidate frames \mathcal{F}, the current training frames \mathcal{V}, the camera poses \mathcal{C}, and the 2D bounding boxes of dynamic entities \mathcal{B}. The retrieval process consists of two parallel scoring modules and a subsequent interleaving fusion module:

Static Context Retrieval. The static retrieval module (Static_Context) aims to find contextual frames with the highest viewpoint overlap to provide comprehensive static background information. For each frame c within the candidate set \mathcal{F}, the algorithm calculates the Field of View (FoV) overlap between its camera pose \mathcal{C}_{c} and all training frame poses \mathcal{C}_{v}, taking the maximum overlap value as the candidate’s score. Subsequently, all candidate frames are sorted in descending order based on these scores to generate the static context candidate list \mathcal{P}_{\mathrm{cam}}.

Dynamic Context Retrieval. To ensure balanced spatio-temporal coverage of dynamic objects, we maintain a coverage counter \mathrm{cnt}[i] (initialized to 0) for each dynamic entity i\in\mathcal{V}. In each greedy iteration, we identify the least-covered entity i^{*} and retrieve the frame f^{*}\in\mathcal{F} that maximizes its visible 2D bounding box area. We then update the coverage for all entities j present in f^{*} by adding their bounding box areas, normalized by the maximum bounding box area A_{\max} in f^{*}. This process repeats until sufficient frames are gathered, yielding the dynamic list \mathcal{P}_{\mathrm{box}}.

Interleaving and Fusion. We alternately append frames from \mathcal{P}_{\mathrm{cam}} and \mathcal{P}_{\mathrm{box}} to the final memory set \mathcal{M}. During this step, duplicates are discarded, and a minimum temporal stride of four frames is rigidly enforced to guarantee a uniform distribution. Interleaving terminates once |\mathcal{M}|=N, and the context frames are returned in chronological order.

## Appendix C Details of Inference System

Our inference system comprises two main components: World Planning via LLM and Causal Chunk-Based Generation. In this section, we elaborate on the specific details of these operations.

### C.1 World Planning via LLM

We employ Gemini[[37](https://arxiv.org/html/2607.02517#bib.bib51 "Gemini: a family of highly capable multimodal models")] as the core semantic engine for world planning. Specifically, given an initial frame, we first select the dynamic objects of interest. We leverage SAM[[26](https://arxiv.org/html/2607.02517#bib.bib56 "Segment anything")] and DepthAnything v2[[47](https://arxiv.org/html/2607.02517#bib.bib57 "Depth anything v2")] to roughly estimate the 3D bounding boxes of these objects and establish the initial orientations for both the entities and the camera.

This structured information is then fed into the LLM, prompting it to analytically plan the corresponding 3D trajectories based on our customized narrative design. An example of the prompt we utilize is structured as follows:

> You are an expert with strong 3D spatial imagination capabilities. Given the following information:
> 
> {
>   "coordinate_system": "OpenGL (X-right, Y-up, Z-backward)",
>   "camera_position": {"position": [0.0, 0.0, 0.0]},
>   "camera_intrinsics": {
>     "fx": 565.4046, "fy": 565.4046, "cx": 416.0, "cy": 240.0,
>     "image_width": 832, "image_height": 480, "fov_v_deg": 46
>   },
>   "ground_height_y": [-1.248],
>   "bboxes_3d": [
>     {
>       "bbox_3d": {
>         "center": [-1.7175, -0.4237, -3.4715],
>         "dimensions": [0.6799, 1.8, 0.442],
>         "rotation_yaw_deg": 90,
>         "prompt": "A woman walks on the road."
>       }
>     },
>     {
>       "bbox_3d": {
>         "center": [3.227, -0.8014, -7.1893],
>         "dimensions": [1.2749, 1.1321, 3.1872],
>         "rotation_yaw_deg": 90,
>         "prompt": "A car first keep still, then starts driving on the road."
>       }
>     }
>   ]
> }
> 
> Here, "coordinate_system" indicates the 3D coordinate system; "camera_position" refers to the initial camera location; "camera_intrinsics" specifies the camera parameters; "ground_height_y" is the y-coordinate of the ground; "bboxes_3d" contains information for multiple subjects. For each "bbox_3d", "center" represents the initial 3D center coordinates, "dimensions" denotes the actual width, height, and length of the object, and "rotation_yaw_deg" is the initial yaw angle. The "prompt" provides the textual description for the trajectory generation.
> 
> 
> User Instruction: Please help me generate the corresponding 3D bbox trajectories and camera poses for these subjects. The total duration is 15s at 16 fps. The initial camera position is at the origin, with the pose as an identity matrix facing the -Z direction.
> 
> 
> 0-5s: The camera and the car remain stationary. The woman walks forward along the +X direction, reaching the edge of the camera view at 5s.5-10s: At 5-6s, the camera rotates from -Z to +X. From 6-10s, the camera moves forward along +X, overtaking the woman. The woman continues walking along +X, while the car remains stationary.10-15s: At 10s, the camera stops. From 10-11s, it rotates from +X to -Z, and from 11-15s, it remains strictly stationary. The woman walks along +X and re-enters the camera view at 11s, then continues walking within the frame. The car starts driving along +X at 10s, enters the camera view at 11s, and exits at 14s.
> 
> 
> Return the Python code to generate the above 3D bbox trajectories and camera poses, and visualize them. Simultaneously, for each generated frame, project the 3D bboxes onto the camera plane using the current camera pose to generate 2D bboxes. Write the projected 2D bboxes into the final output and visualize them.

Beyond planning trajectories for objects explicitly selected in the initial frame, users can also define motion paths for completely novel objects within the prompt. The LLM demonstrates remarkable capability in automatically synthesizing physically plausible kinematics for these newly introduced entities (i.e., Promptable World Events).

Consequently, we obtain the complete 3D trajectories of all dynamic objects and their corresponding 2D bounding box sequences projected onto the camera plane. These 2D projection sequences directly serve as the deterministic Spatial Location Condition (\mathcal{B}) applied in the subsequent generative stage.

### C.2 Causal Chunk-Based Generation

Building upon the projected 2D Location Condition and camera trajectories from the planning phase, we execute the video synthesis in a causal autoregressive manner, as detailed in Algorithm[2](https://arxiv.org/html/2607.02517#alg2 "Algorithm 2 ‣ C.2 Causal Chunk-Based Generation ‣ Appendix C Details of Inference System ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory").

Algorithm 2 Causal Chunk-Based Generation

1:Location condition sequence

\mathcal{B}_{1:T}
, Captions

\mathcal{P}
, Initial reference frame

I_{0}
, Total frames

T
, Chunk size

K
, Camera Poses

\mathcal{C}_{1:T}

2:Generated continuous video stream

V

3:procedure Causal_Generation(

\mathcal{B}_{1:T},\,\mathcal{P},\,I_{0},\,T,\,K,\,\mathcal{C}_{1:T}
)

4: Partition

\mathcal{B}_{1:T}
into

N=T/K
chunks:

\{\mathcal{B}^{(1)},\dots,\mathcal{B}^{(N)}\}

5: Partition

\mathcal{C}_{1:T}
into

N=T/K
chunks:

\{\mathcal{C}^{(1)},\dots,\mathcal{C}^{(N)}\}

6: Initialize video buffer

V\leftarrow I_{0}

7:for

n=1,2,\ldots,N
do

8:if

n=1
then

9:

\mathcal{A}\leftarrow
Appearance_condition_generation(

\mathcal{B},\,I_{0}
)

10:

\mathcal{M}\leftarrow\emptyset
\triangleright no historical context for the first chunk

11:else

12:

\mathcal{A}\leftarrow
Appearance_condition_generation(

\mathcal{B},\,V
)

13:

\mathcal{M}\leftarrow
Context_Retrieval(

\mathcal{B},\,\mathcal{C},\,V
)

14:end if

15:

I_{start}\leftarrow V_{\text{last}}

16:\triangleright use the last frame of buffer V as initial frame for current chunk generation

17:

V^{(n)}\leftarrow
WorldDirector(

\mathcal{B}^{(n)},\,\mathcal{P},\,\mathcal{A},\,\mathcal{M},\,\mathcal{C}^{(n)},\,I_{start}
)

18:

V\leftarrow V\cup V^{(n)}[1:]
\triangleright append generated chunk without first frame to the buffer

19:end for

20:return

V

21:end procedure

Given the complete spatial location sequence \mathcal{B}_{1:T} and camera poses \mathcal{C}_{1:T}, we first partition them into N sequential chunks of size K. The generative process maintains a global continuous video buffer V, which is initialized with the starting reference frame I_{0}.

During the iterative generation, the conditioning strategy adapts based on the temporal state. For the first chunk (n=1), the model extracts the Appearance Condition (\mathcal{A}) directly from I_{0} guided by the Location Condition \mathcal{B}, while the historical context memory (\mathcal{M}) remains empty as there is no preceding temporal information. For all subsequent chunks (n>1), the framework dynamically constructs \mathcal{A} and retrieves the historical Context (\mathcal{M}) from the previously generated video buffer V. This retrieval mechanism strictly leverages the location constraints \mathcal{B} and camera parameters \mathcal{C} to fetch precise dynamic entity identities and static background anchors.

Crucially, to guarantee temporal smoothness at the chunk boundaries, the last frame of the current buffer (V_{\text{last}}) serves as the conditional initial frame (I_{start}) for generating the next chunk V^{(n)}. The core diffusion model, WorldDirector, processes these multimodal conditions (\mathcal{B}^{(n)},\mathcal{P},\mathcal{A},\mathcal{M},\mathcal{C}^{(n)},I_{start}) to synthesize the current segment. Finally, we append the generated frames to V—excluding the overlapping first frame to prevent redundancy—thereby progressively unrolling the long-horizon video stream without inherent length limitations.

## Appendix D Ablation on Dynamic Context and Appearance Condition Drop Mechanism

We further evaluate the efficacy of the retrieved dynamic context and the Temporal Drop Mechanism. Despite the strong visual priors from the Appearance Condition, ablating the dynamic context stream confirms the necessity of retrieving dynamic objects within the contextual memory. As shown in Figure[S1](https://arxiv.org/html/2607.02517#A4.F1 "Figure S1 ‣ Appendix D Ablation on Dynamic Context and Appearance Condition Drop Mechanism ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), relying solely on Appearance Condition for re-entering dynamic entities degrades identity preservation; the model generates semantically similar but non-identical entities. This demonstrates that dynamic context is indispensable for temporally anchoring the specific object identity across causal chunks. Furthermore, we validate the necessity of the Temporal Drop Mechanism. Removing this exposes the network to dense, frame-by-frame appearance references, which induces severe motion rigidity (e.g., characters "sliding" rather than walking naturally, as depicted in Figure[S1](https://arxiv.org/html/2607.02517#A4.F1 "Figure S1 ‣ Appendix D Ablation on Dynamic Context and Appearance Condition Drop Mechanism ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory")). This evidence substantiates our design: the Temporal Drop Mechanism effectively prevents overfitting to static reference images, compelling the model to synthesize fluid, text-driven dynamics rather than executing rigid image warping.

![Image 6: Refer to caption](https://arxiv.org/html/2607.02517v1/x6.png)

Figure S1: Ablation on Dynamic Context and Appearance Condition Drop Mechanism. We highly recommend viewing the video results on our [project page](https://worlddirector.github.io/) for a more intuitive demonstration.

![Image 7: Refer to caption](https://arxiv.org/html/2607.02517v1/x7.png)

Figure S2: Flexible Viewpoint Control.WorldDirector supports diverse exploration paradigms. Top: A pure third-person view tracking a running dog with a 360^{\circ} panoramic sweep. Bottom: A dynamic viewpoint switch from a third-person tracking shot to an independent first-person backward movement.

## Appendix E Flexible Viewpoint Control

By explicitly incorporating the spatial location condition, our framework intrinsically supports flexible viewpoint control, enabling seamless transitions between first- and third-person exploration paradigms. Specifically, during 3D trajectory planning, anchoring the 2D bounding box of a target dynamic entity near the center of the camera’s field of view yields a third-person perspective. Conversely, decoupling the camera trajectory from dynamic objects allows for independent first-person navigation. As illustrated in Figure[S2](https://arxiv.org/html/2607.02517#A4.F2 "Figure S2 ‣ Appendix D Ablation on Dynamic Context and Appearance Condition Drop Mechanism ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"), the first scenario demonstrates pure third-person exploration, where the camera follows a running dog while simultaneously performing a continuous 360^{\circ} panoramic sweep of the surrounding scene. The second scenario highlights dynamic viewpoint switching within a single sequence: the initial two temporal chunks maintain a third-person perspective following a human character, whereas the third chunk smoothly transitions to a first-person view as the camera detaches and moves backward. These capabilities underscore the model’s profound flexibility in directing simulated environments.

## Appendix F More Qualitative Comparisons

![Image 8: Refer to caption](https://arxiv.org/html/2607.02517v1/x8.png)

Figure S3: Qualitative comparison with baselines. Note that HyDRA uses the initial 10s of our results as a reference video for its generation. We highly recommend viewing the video results on our [project page](https://worlddirector.github.io/) for a more intuitive demonstration.

![Image 9: Refer to caption](https://arxiv.org/html/2607.02517v1/x9.png)

Figure S4: Qualitative comparison with baselines. Note that HyDRA uses the initial 10s of our results as a reference video for its generation. We highly recommend viewing the video results on our [project page](https://worlddirector.github.io/) for a more intuitive demonstration.

We provide additional qualitative comparison results in Figure[S3](https://arxiv.org/html/2607.02517#A6.F3 "Figure S3 ‣ Appendix F More Qualitative Comparisons ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory") and Figure [S4](https://arxiv.org/html/2607.02517#A6.F4 "Figure S4 ‣ Appendix F More Qualitative Comparisons ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory") to further evaluate the baselines. The observations remain consistent with our main findings in Section [4.1](https://arxiv.org/html/2607.02517#S4.SS1 "4.1 Comparisons ‣ 4 Experiments ‣ WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory"). Specifically, Yume, HY-World, and Infinite-World tend to generate significantly less subject motion. LingBot-World successfully produces highly dynamic results that align well with the textual prompts. However, it lacks fine-grained interactive control precision, making it difficult to strictly match the user’s specific scenario designs. HyDRA consistently exhibits a strong bias towards generating a prominent subject walking directly in front of the camera, which is likely an artifact of its training data distribution. In contrast, our method accurately executes the user’s intended spatial layout and maintains precise interactive control and dynamic memory.

## Appendix G Impact Statement

This paper focuses on the technical advancements in controllable video world simulation with persistent dynamic memory. The work aims to enhance applications in virtual reality, gaming, film-making, and interactive design, which could have positive societal implications in these domains. However, this study does not directly address potential societal impacts, including possible negative consequences such as malicious or unintended uses (e.g., generating deceptive or fake video content), fairness considerations, privacy concerns, or security risks that might arise from the application of this generative technology. The paper primarily presents foundational technical research and does not discuss the commercial deployment of the technology or specific mitigation strategies for these negative impacts.

## Appendix H Responsible Release and Safeguards

Because WorldDirector is a highly controllable generative video model driven by an LLM orchestrator, we plan a staged and documented release. We will release the inference codebase, LLM prompt templates, and pre-trained model checkpoints strictly intended for academic research and evaluation.

For downstream applications, we strongly recommend combining WorldDirector with established safety mechanisms that fall outside the scope of this foundational paper. These include LLM prompt safety filters to prevent malicious planning, generated-video watermarking, content provenance metadata, and deployment-time monitoring. Given that our framework facilitates the continuous generation of long-horizon events with persistent entities, it inherently lowers the barrier for creating complex, logically coherent synthetic scenarios. Therefore, these external safeguards remain an important consideration to mitigate the general risks of misuse associated with video generation technologies.

