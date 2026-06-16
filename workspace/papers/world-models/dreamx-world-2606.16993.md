# DreamX-World 1.0: A General-Purpose Interactive World Model

DreamX Team

![](images/01192f0687144453a3946096dd22da280a5f0ebf85263d8e8483bcdecb4610a6.jpg)

DreamX-World 1.0 is a general-purpose interactive text/image-to-video world model for controllable long-horizon generation. It supports camera navigation, revisits to previously observed regions, and promptable events across photorealistic, game-style, and stylized domains. Our data engine combines camera-accurate Unreal Engine rendering, action-rich gameplay recordings, and real-world videos with recovered camera geometry. For camera control, we introduce E-PRoPE, a lightweight variant of projective positional encoding that retains PRoPE’s projective camera geometry while applying camera-aware attention to spatially reduced tokens. We convert a bidirectional video generator into a few-step autoregressive world model using causal forcing, DMD-style distillation, and long-rollout training. Training on self-generated long-horizon contexts exposes the model to its own generated history and reduces the style and color drift that accumulates across autoregressive chunks. Memory-Conditioned Scene Persistence retrieves earlier views through camera-geometry-based retrieval, while residual recycling makes the conditioning path less sensitive to imperfect memory latents. Event Instruction Tuning adds composable event control, and reinforcement learning alignment recovers camera control and visual quality after distillation. With mixed-precision DiT execution, residual reuse, 75%-pruned VAE decoding, and asynchronous pipeline parallelism, DreamX-World 1.0 reaches up to 16 FPS on eight RTX 5090 GPUs. On our 5-second basic evaluation, DreamX-World 1.0 achieves a camera-control score of 73.75 and an overall score of 84.76, outperforming HY-WorldPlay 1.5 and LingBot-World in overall score, which achieve 80.79 and 80.45, respectively.

GitHub: https://github.com/AMAP-ML/DreamX-World Project Page: https://dreamx-world.github.io

![](images/5377c6143426742f176864b6f9cb54528f610d676e89d158360298e0b7469957.jpg)  
Figure 1 DreamX-World 1.0 generates interactive videos with precise camera and event control across photorealistic, game-style, and stylized visual domains.

## 1 Introduction

World models extend video generation from passive visual synthesis to interactive simulation. While modern video diffusion models can produce high-quality short clips, interactive systems must additionally respond to user controls and preserve scene state over long horizons (Team Wan et al., 2025; Robbyant Team, 2026; Team HunyuanWorld, 2025). Changing the camera trajectory should reveal a consistent scene rather than produce an unrelated plausible view; revisiting a location should preserve its layout and identity; and prompted events should modify the existing world state. These requirements become particularly difficult when a single model must operate across photorealistic, game-style, and stylized domains.

Building such a model requires videos spanning different visual domains, together with reliable camera, action, and event annotations. No single data source provides this coverage at sufficient scale, motivating the combination of synthetic, game, and real-world data. Beyond the data problem, three coupled technical challenges remain.

First, camera control must translate a prescribed trajectory into consistent viewpoint changes across scenes with different scales and motion distributions (Bahmani et al., 2025). The conditioning mechanism must provide sufficient geometric accuracy without substantially increasing the cost of the video backbone. Second, interactive generation must preserve scene content beyond the local context window. Once an earlier view leaves the context, the model may render a different plausible scene when that region is revisited (Hong et al., 2025). Autoregressive generation compounds this problem because small prediction errors accumulate into appearance, style, and color drift (Huang et al., 2025). Finally, continuous interaction imposes a latency requirement that is absent from offline video generation. Reducing the number of diffusion steps improves throughput, but aggressive distillation can weaken visual quality, camera control, and rollout stability (Zhu et al., 2026). A practical system must therefore reduce both sampling and decoding costs while retaining the capabilities of the bidirectional model.

We present DreamX-World 1.0, a general-purpose interactive world model initialized from Wan2.2 (Team Wan et al., 2025). Its training pipeline progressively introduces camera conditioning, non-local scene memory, event interaction, and autoregressive long-video generation, followed by post-distillation alignment. The data engine combines Unreal Engine trajectories with exact camera geometry, gameplay recordings with action-rich dynamics, and real-world videos with recovered camera poses. After geometric filtering and normalization, these sources provide a common training representation across photorealistic, game-style, and stylized domains.

![](images/e7af4344d6b2c4d4a72ffdf18b078c2ba3ff50882079d168bc53b8da36dcb731.jpg)  
Figure 2 System overview of DreamX-World 1.0. The pipeline integrates camera-accurate data, efficient control, autoregressive distillation, long-range memory, interaction alignment, and optimized serving.

Camera control is introduced through E-PRoPE, an efficient variant of PRoPE (Li et al., 2025a). It computes projective attention on spatially reduced tokens, retaining comparable trajectory-following performance while reducing inference latency by approximately 30%. Once camera motion is controllable, a memory-conditioned stage addresses a different failure mode: previously observed content may change after leaving the local context window. Camera-geometry-based retrieval supplies non-local visual evidence, while residual recycling

improves robustness to imperfect memory latents.

Event Instruction Tuning subsequently adds structured multi-entity event control through the text-conditioning interface. For streaming generation, the bidirectional video model is converted into a few-step autoregressive generator using causal forcing, DMD, and long student rollouts (Zhu et al., 2026; Yin et al., 2024b; Huang et al., 2025). This training exposes the model to generated history and reduces the style and color drift that accumulates across chunks.

Because distillation can reduce visual diversity, motion quality, and camera controllability, the final training stage applies reinforcement learning to the DMD-distilled model. Camera-control and video-quality rewards guide a conservative update while preserving the few-step autoregressive interface. Mixed-precision DiT execution, residual reuse, pruned VAE decoding, and asynchronous pipeline parallelism then enable streaming generation at up to 16 FPS on eight RTX 5090 GPUs.

## Our main contributions are:

Efficient and controllable world generation. We build a multi-source data system and introduce E-PRoPE for camera-controlled generation. By applying projective conditioning to spatially reduced tokens, E-PRoPE retains comparable trajectory-following performance to PRoPE with approximately 30% lower inference latency.

• Long-horizon generation with scene persistence. Geometry-guided memory preserves earlier observations during camera revisits, while autoregressive distillation and long-rollout training reduce the style and color drift accumulated across generated chunks. Reinforcement learning further improves the camera control and visual quality of the DMD-distilled model without changing its few-step inference interface.

• Real-time streaming deployment. Together with mixed-precision DiT inference, residual reuse, pruned VAE decoding, and asynchronous pipeline parallelism, the system reaches up to 16 FPS on eight RTX 5090 GPUs.

## 2 Data

![](images/bffa0b52e28162e8d5ef98e6992639c150c8a55dc3c38b57cb3726aa35c2ccd2.jpg)  
Figure 3 UE data generation pipeline. Trajectories are collected and validated online, then rendered offline with camera poses, actions, and metadata. The runtime layer supports distributed rendering and failure recovery.

Training interactive world models requires diverse videos with reliable camera and action annotations. Realworld data provides visual diversity but often lacks such annotations, whereas synthetic and game data offer precise control but have limited domain coverage. We therefore combine UE-generated, real-world, and game data in a unified pipeline for annotation, geometric processing, and quality control.

## 2.1 Data Curation

## 2.1.1 UE-generated Data

We synthesize a substantial portion of the training data in UE5 (Unreal Engine 5), which provides controllable camera and agent motion together with ground-truth annotations.

The UE data comprises three observation modes: first-person, third-person, and event subsets, which share a unified output schema. The first-person subset records free camera exploration, the third-person subset follows a moving character, and the event subset captures object interactions and visible state changes. A distinctive property of the UE subset is its per-frame ground-truth annotation: each frame includes a discrete action vector encoded as keyboard-style control signals (WASD for translation, IJKL for rotation) and camera pose (position and Euler angles). Third-person clips additionally record the character’s world position and heading, enabling joint reasoning over camera and agent motion.

UE clips are generated through a two-stage pipeline: first-person and third-person engines explore scenes to discover high-quality trajectories, which are then rendered offline. This decoupled design avoids wasting computational resources on invalid or low-motion clips.

First-person Free-camera Generation. An free camera explores each scene using UE’s NavMesh navigation system. It navigates toward sampled goals with randomized heading and pitch adjustments. Collision checking, minimum-duration and path-length constraints, and stuck detection are used to reject invalid trajectories. Accepted trajectories are replayed for offline rendering.

Third-person Character-driven Generation. A rigged character follows navigation paths while a follow camera records the motion. The camera uses smooth tracking, collision avoidance, and occlusion handling to maintain visibility of the character.

Rendering and Runtime Management. Validated trajectories are rendered with UE’s Movie Render Queue and stored with their poses, actions, and metadata. Rendering jobs are distributed across multiple GPUs with checkpoint resumption and automatic failure recovery.

## 2.1.2 Real-world and Game Data

We collect real-world videos from SpatialVID (Wang et al., 2025), RealEstate10K (Zhou et al., 2018), Sekai (Li et al., 2025c), and DL3DV (Ling et al., 2024). Camera poses are sparsely estimated on several key frames using MegaSaM (Li et al., 2025d) and interpolated through the pipeline in Section 2.2. Game data is collected from Sekai-Game (Li et al., 2025c) and OmniWorld-Game (Zhou et al., 2025). Their engine-exported poses are converted to the same camera coordinate system as the UE and real-world data.

## 2.2 Data Annotation and Filtering

Our data processing pipeline consists of three-stage quality control: basic filtering, geometric camera-pose cleaning, and video captioning and attribute tagging.

Basic Filtering. We remove clips with insufficient duration or frame rate, excessive overlaid text, black borders, or limited visual change. Visual change is measured using the cosine similarity between CLIP embeddings of the first and last frames.

Geometric Camera-pose Cleaning. For real-world videos, sparse camera poses are densified using SLERP for rotation and linear interpolation for translation. Each trajectory is normalized and checked for inconsistent intrinsics, translation spikes, rapid rotations, vertical jitter, and invalid orientations.

Video Captioning and Attribute Tagging. Each clip is annotated with a global caption describing the scene, subjects, actions, and temporal changes. Retained clips are further tagged with aesthetic quality, motion intensity, scene category, visual style, subject type, and motion category. The motion category distinguishes static scenes with camera motion (3D) from scenes containing both camera and object motion (4D). These annotations and tags are used for filtering and training.

![](images/70494f0f0958f67bc75e01d4f2ddaa3a290158876591f9b64f695e15e3d52fb7.jpg)

## 2.3 Event Instruction Data

Figure 4 Overview of the cleaning, filtering, and attribute tagging pipeline.

For Event Instruction Tuning, we select high-quality clips that contain visible state changes from the cleaned and tagged data pool and annotate them with structured

event descriptions. Following a hierarchical captioning strategy, each annotation pairs a holistic global description with dense, time-aligned entity-level event records. The global description summarizes the static scene context and overall temporal evolution, while each event record specifies the entity reference, event predicate, spatial anchor, and temporal interval of a dynamic change. For composable-event examples, each participating object is assigned its own event record, and inter-object interactions (e.g., collision, handoff) are explicitly described in the global caption. The dataset mixes single-object events and composable events so that the model learns to ground both atomic and compositional instructions.

## 3 Progressive Training Pipeline

We initialize DreamX-World 1.0 from the Wan2.2-TI2V model (Team Wan et al., 2025) and progressively adapt it for camera control, memory conditioning, event interaction, and autoregressive generation.

## 3.1 Camera-Aware Training

Since Wan2.2-TI2V does not take camera trajectories as input, we first train the bidirectional model on poseannotated videos to support explicit 6-DoF camera control. Projective Positional Encoding (PRoPE) (Li et al., 2025a) introduces a relative conditioning mechanism by introducing both inter-camera frustum relationships and camera-agnostic token positions (e.g., RoPE (Su et al., 2024)) directly into transformer self-attention blocks. Mathematically, for a sequence of input tokens $X = \{ x _ { s } \} _ { s = 1 } ^ { S } , X \in R ^ { S \times d } , x _ { s } \in R ^ { d }$ , it defines a per-token matrix $D _ { s } ^ { P R o \ " P E } \in \mathbb { R } ^ { d \times d }$ composed of two complementary submatrices of shape $\begin{array} { l } { \frac { d } { 2 } \times \frac { d } { 2 } } \end{array}$ :

$$
D _ { s } ^ { P R o P E } = \left[ \begin{array} { c c } { D _ { s } ^ { P r o j } } & { 0 } \\ { 0 } & { D _ { s } ^ { R o P E } } \end{array} \right] .
$$

The first submatrix, $D _ { s } ^ { P r o j }$ , encodes the full projective camera geometry using the world-to-image projection matrix. The second submatrix, $D _ { s } ^ { R o P E }$ , replicates the standard rotary position embeddings (RoPE). The matrix $D _ { s } ^ { P R o P E }$ is applied to the attention query and key tokens via matrix multiplication.

Directly applying PRoPE can be expensive on long videos for both training and inference, as it requires adding extra attention modules to the DiT backbone in a layer-wise manner, nearly doubling the overall computational cost. We argue that PRoPE primarily captures the view-dependent high-level semantics. Therefore, computing attention over the full-resolution video token set, which contains complete semantics, is unnecessary. To this end, we introduce a lightweight variant of PRoPE as displayed in Figure 5. The key idea is to focus on a downsampled set of tokens. Specifically, we downsample the PRoPE attention input tokens $X ^ { P R o P E }$ along the spatial dimension and project them into a lower-dimensional query/key/value space, yielding $X ^ { P R o \overline { { P } } E } \in \mathring { \mathbb { R } } ^ { N \times d ^ { \prime } }$ where $d ^ { \prime } < d$ and N is the number of downsampled tokens. This makes camera control highly efficient during both training and inference while retaining most of the controllability. For example, given a 5-second 720P video, the VAE of Wan2.2 5B maps it to $S = 1 8 4 8 0$ tokens, while we downsample it to N = 4096 tokens before computing PRoPE attention, indicating a more than 4.5x spatial downsampling ratio. Consequently, it significantly reduces the training and inference time by approximately 50% and 30% respectively.

Efficient PRoPE (E-PRoPE). Additionally, we omit the $D _ { s } ^ { R o P E }$ component and use only the projective submatrix $D _ { s } ^ { P r o j }$ , since the original attention in the DiT backbone already provides sufficient spatiotemporal inductive bias for fine-grained semantic modeling. After PRoPE attention, we upsample the output tokens back to the original resolution and simply add them to the original DiT attention output. With standard denoising objective, we train PRoPE modules by freezing the DiT backbone and only backpropagating the gradients to the PRoPE parameters. We compare the performance of PRoPE and E-PRoPE in Table 1.

Interestingly, during inference, we empirically find that the downstream model can still leverage the pre-trained PRoPE component in a plug-and-play manner even when trained without it, suggesting the strong and robust geometric bias of PRoPE.

![](images/17028739d0493e12a954d3008fdf64eb19eb4a90d6ee824892b3e68b7798f133.jpg)  
Figure 5 Our Efficient PRoPE (E-PRoPE) component is attached to each DiT attention layer. We neglect the adaLN modulation for simplicity.

Table 1 Comparing PRoPE and E-PRoPE on Omni-WorldBench (Wu et al., 2026). E-PRoPE achieves comparable camera control performance to the full PRoPE while being more computationally efficient. Latency is measured as the average time (in seconds) to generate a 5-second video at $1 2 8 0 \times 7 2 0$ resolution on 8 NVIDIA H20 GPUs.
<table><tr><td>Model</td><td>Camera Control</td><td>Image Quality 1</td><td>Dynamic Degree</td><td>Transition Detect 个</td><td>Temporal Flicker 个</td><td>Motion Smooth.</td><td>Latency</td></tr><tr><td>PRoPE</td><td>73.89</td><td>66.15</td><td>87.5</td><td>96.67</td><td>96.02</td><td>98.65</td><td>80</td></tr><tr><td>E-PRoPE</td><td>73.75</td><td>66.75</td><td>85.83</td><td>98.33</td><td>96.17</td><td>98.79</td><td>59</td></tr></table>

## 3.2 Memory-Conditioned Scene Persistence

This training stage targets inconsistency when the model generates the current frame. As shown in Fig. 6, we train the Diffusion Transformer (DiT) to use both recent context and retrieved memory frames.

## 3.2.1 Memory Conditioning

We use two sources of conditioning. Recent history frames are the latest denoised latent frames before the target window, while memory frames are predicted clean latent frames retrieved from earlier history. The target frames are the noisy latent frames that the model needs to denoise and supervise. For a latent video sequence $z _ { 1 : T } ^ { 0 }$ with camera signals $\pi _ { 1 : T }$ , we sample target latent frames $z _ { \mathcal { C } }$ , recent history latent frames $z _ { \mathcal { H } }$ and memory latent frames $z _ { \mathcal { M } }$ , and form the model input sequence

$$
z _ { \mathrm { p a c k } } = [ z _ { \mathcal { M } } \mid z _ { \mathcal { H } } \mid z _ { \mathcal { C } } ^ { \tau } ] ,\tag{1}
$$

latents, $z _ { \mathcal { C } } ^ { \tau }$ are target latents with diffusion noise level $\tau ,$ and $[ \cdot | \cdot ]$ denotes concatenation along the token dimension. During training, we update the model parameters with standard rectified flow training objective and only compute loss on target latent frames. We also compare other memory injection mechanism, such as

![](images/37dee131685ed5a97d437daf92961239c21005227c2a7ccd908464e54e041ce8.jpg)  
Figure 6 Training framework for Memory-Conditioned Scene Persistence. Geometry-based retrieval selects non-local memory frames. The memory frames are packed with recent history frames and target frames into the same DiT self-attention stream. Supervision is applied only to the target prediction. The residual-recycling path perturbs conditioning tokens without changing the supervised target.

cross-attention and VACE-style conditioning (Jiang et al., 2025). However, these approaches consistently produces worse generation performance.

## 3.2.2 Memory Retrieval

We retrieve memory frames by geometry-based clues. In specific, we use camera pose and view overlap to choose history frames that are highly relevant to the target view, instead of selecting frames only by temporal distance. After retrieval, each memory frame is added with RoPE embedding that corresponds to its original temporal location in the generated video when packed into the input sequence. This prevents distant memory frames from being treated as the immediate frames temporally adjacent to the target frames. For large time gaps, we use a lightweight temporal-position treatment, inspired by long-range positional encoding methods such as NTK-aware RoPE scaling, YaRN (Peng et al., 2023), and randomized positional encodings (Ruoss et al., 2023).

## 3.2.3 Exposure Bias in Memory

Memory conditioning has an exposure-bias problem, which originates from the gap between training and inference. During training, those conditioning frames are sampled from training data, while, during inference, the conditioning frames are generated by model and contain prediction errors.

We mitigate this train-test gap by adopting the error injection approach proposed in Stable Video Infinity (SVI) (Li et al., 2025b). We perturb only the conditioning tokens while keeping the target latent clean. At this point, the model learns to draw upon the sampled memory frames when it helps and fall back to its learned prior when the memory frames contain explicit errors.

## 3.3 Event Instruction Tuning for Composable Events

Recent interactive world models expose event-related controls at different granularities. LingBot-World (Robbyant Team, 2026) demonstrates promptable global and local world events, HY-WorldPlay 1.5 (Team HunyuanWorld, 2025; Sun et al., 2025) supports text-triggered dynamic events during streaming generation, and Yume-1.5 (Mao et al., 2025b) introduces text-controlled event editing. Matrix-Game 3.0 (Wang et al., 2026c) focuses instead on action-conditioned long-horizon memory and real-time streaming. Related spatially controllable video-generation work, such as Omni-Effects (Mao et al., 2025a), studies compositional visual effects specified by category and location, but does not target persistent interactive world events under navigation. However, existing public systems do not provide explicit composable events—multiple objects appearing, acting, and interacting within the same generation under structured event instructions (Table 2). Composable events are essential for realistic world simulation, where meaningful state changes rarely involve a single object in isolation: a traffic scene requires pedestrians, vehicles, and signals to respond concurrently, and an indoor scene may involve multiple characters acting and reacting to one another. DreamX-World 1.0 introduces an Event Instruction Tuning stage that closes this gap: users describe the coarse region or relation in which each object appears, what it does, and how objects interact, and the model responds to all specified events in a single forward pass.

Table 2 Comparison of event-control capabilities across interactive world models. Composable events subsume multi-object and multi-event settings: the model handles multiple objects with distinct actions and mutual interactions in one generation.
<table><tr><td>Model</td><td>Promptable Events</td><td>Object-Level Events</td><td>Region-Guided Events</td><td>Multi-Entity Composition</td><td>Inter-Object Interaction</td></tr><tr><td>LingBot-World</td><td></td><td></td><td>△</td><td>△</td><td></td></tr><tr><td>HY-WorldPlay 1.5</td><td>√</td><td></td><td>X</td><td>△</td><td>X</td></tr><tr><td>Matrix-Game 3.0</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Yume-1.5</td><td></td><td></td><td>X</td><td>X</td><td>X</td></tr><tr><td>DreamX-World 1.0</td><td></td><td></td><td></td><td></td><td></td></tr></table>

△indicates qualitative or partial support without structured per-object event instructions. Object-level events denote single  
object/category events; multi-entity composition requires distinct entities and actions in one structured instruction. Region-guided events refer to coarse regions or relative placement rather than coordinate-level localization.

Training. Using the event instruction data described in Section 2.3, we fine-tune the full DiT while keeping the architecture unchanged. Event semantics enter exclusively through the text-conditioning interface, where structured event instructions are rendered as natural-language prompts covering the global scene and perentity dynamics. The tuning mixture combines event-instruction samples with non-event training clips, which preserves the model’s general world-generation capability while adding responsiveness to promptable events. We use conservative updates and strict gradient clipping to avoid disrupting the pretrained visual and motion priors.

## 3.4 Autoregressive Long Video Generation and Distillation

![](images/3fedc06f7a4b809615b70ceb40ce3959638a6b53d46ccf36ee6c1960aad55704.jpg)  
Figure 7 DMD-forcing for camera-controlled long-video distillation. The E-PRoPE AR student is distilled from the bidirectional E-PRoPE teacher through DMD supervision over local temporal windows sampled from long videos, while preserving the streaming autoregressive sampling interface.

To enable few-step long-video generation, we distill a bidirectional model into an autoregressive generator that can stream from generated history while preserving visual quality and camera controllability.

We train few-step autoregressive model using causal forcing (Huang et al., 2025; Zhu et al., 2026) on large-scale high-quality video data, while keeping it close to the original visual distribution of the bidirectional model. Subsequently, following LongLive (Yang et al., 2025), we further adapt the model on long sequences with long rollouts and local temporal windows, using Infinity-RoPE (Yesiltepe et al., 2025) to support extended autoregressive context and reduce long-video failures such as identity drift, background mutation, and weakened prompt or motion control.

For camera-controlled generation, we incorporate an E-PRoPE branch into the long-video T2V student and distill the resulting few-step camera-controlled autoregressive model from a bidirectional E-PRoPE teacher. Although this branch enables camera control, chunk-wise inference may still lead to reduced motion smoothness and degraded camera controllability over long sequences.

We therefore repeat the long-video DMD training to improve this behavior. Figure 7 illustrates the DMDforcing pipeline, in which camera-controlled student rollouts are matched to the bidirectional E-PRoPE teacher over local temporal windows sampled from long videos.

To preserve I2V quality, we perform I2V DMD distillation using the bidirectional E-PRoPE model as the teacher. The first latent frame of each sampled DMD window is decoded by the VAE and fed to the teacher as the image condition, enabling the teacher to supervise the camera-controlled AR student over local temporal windows of long videos. With this design, the model can perform stable long-duration inference for videos up to one minute while maintaining camera controllability and temporal coherence across chunks.

## 3.5 Reinforcement Learning

![](images/798fcf525e8af40d595e3dd75753e4ef407ad225d63c379b8bda7473865b1c65.jpg)  
Figure 8 RL training overview. The model first produces long-horizon autoregressive rollouts, then samples short clips for video-quality and camera-control reward-model evaluation. The fused reward drives a moderated DiffusionNFT soft update, decoupling rollout horizon from the optimization window.

DMD distillation enables efficient autoregressive generation, but reducing the number of denoising steps can degrade video generation quality and weaken camera controllability. After DMD distillation, we further train the model with reinforcement learning (RL) (Schulman et al., 2017; Chu et al., 2026) as a post-training stage to enhance video quality and strengthen camera following.

RL after DMD distillation needs to be applied carefully. Because the model already runs with very few denoising steps, strong reward updates can make RL training highly unstable. To keep training stable, we adopt a gradual update strategy, allowing the model to change step by step. With this update schedule, RL training proceeds stably and avoids early collapse.

For each text-image-camera condition, the current model generates several long-horizon rollout candidates. The full rollout preserves the autoregressive context, and short clips sampled from candidates are used for reward computation and DiffusionNFT training (Zheng et al., 2025; Zhang et al., 2026) (Figure 8). Because the reward models operate on short temporal windows, the RL update backpropagates through sampled clips rather than the full rollout, keeping GPU memory within a practical range. We adopt two reward models: one measures horizontal translation and rotation accuracy, and the other evaluates the visual quality of generated clips. KL regularization balances the two rewards and keeps the updated model close to the original DMD-distilled model.

RL post-training makes the model better at following camera commands and improves the visual quality of the generated videos. At the same time, the model retains the key strengths of DMD distillation: long-horizon video generation remains stable, and the few-step inference efficiency is not compromised.

## 4 Inference Acceleration

For interactive deployment, our inference pipeline enables streaming autoregressive generation while further accelerating DiT denoising and VAE decoding to reduce latency.

Our deployment target is interactive streaming generation with bounded latency. We first define the shared chunk-wise autoregressive inference interface for camera-controlled T2V and I2V, then summarize the optimizations that make it run in real time. In our asynchronous deployment, 8 RTX 5090 GPUs jointly execute DiT denoising and VAE decoding, reaching up to 16 FPS.

## 4.1 Autoregressive Streaming Inference

![](images/1ed38a9b328b34be4189166992007b9bb3ccd4fc526c0ac6ccb19b7e834aaafb.jpg)  
Figure 9 Autoregressive streaming inference. The distilled sampler generates chunks from noise, updates the rolling KV cache, and uses chunk-relative camera controls.

As shown in Figure 9, the video is generated chunk by chunk: each chunk starts from noise, is denoised by the distilled few-step sampler under the text prompt, the chunk-relative camera trajectory, and the rolling KV cache, and then writes its generated tokens back into the cache for subsequent chunks. This keeps inference streaming-friendly because the model only needs to carry a autoregressive history rather than regenerate previous video content.

For camera-controlled I2V, the inference procedure is almost identical to T2V. The only difference is the first chunk: its first frame is replaced by the input image, which anchors the generated video to the reference frame. All later chunks follow the same rolling-cache procedure as T2V and continue from the accumulated generated history.

Camera controls are represented in a chunk-relative form, following E-PRoPE relative camera conditioning (Li et al., 2025a). The first chunk uses poses relative to its first frame, while each later chunk uses poses relative to the last frame of the previous chunk. This chunk-local relative parameterization keeps the camera condition aligned with the current autoregressive context and prevents the conditioning signal from weakening over long sequences.

## 4.2 Inference Optimizations

For DiT denoising, we combine precision, parallelism, kernel, and timestep-level optimizations. The attention layers use INT8 SageAttention (Zhang et al., 2025), while FFN layers are quantized to FP8 with AngelSlim (Cen et al., 2026). Long spatio-temporal token sequences are sharded across GPUs with sequence parallelism, synchronizing only the required attention and normalization statistics to reduce per-GPU activation memory while preserving full-sequence computation. We further rewrite frequent Transformer-block operators as fused Triton kernels, combining elementwise operations, layout transforms, and small reductions to reduce intermediate allocations and kernel-launch overhead. Following TeaCache (Liu et al., 2025a), we also reuse denoising residuals in empirically stable timestep regions, skipping selected Transformer-block forward passes when adjacent-step residuals change only marginally.

For VAE decoding, we adopt the Matrix-Game 3.0 VAE (Wang et al., 2026c) as the VAE decoder with a 75% pruning ratio, reducing single-chunk decoding to approximately 0.25 seconds. After the first iteration, torch.compile further reduces later decoding latency. We also follow ParaVAE (RiseAI-Sys, 2026): the latent video is split mainly along height, each GPU decodes a local patch, and the decoded patches are gathered into the final video, reducing peak per-GPU memory.

For serving, we use asynchronous pipeline parallelism to overlap the VAE decoding of chunk k with the control reception, KV-cache update, and DiT denoising of chunk k + 1. This hides most VAE latency behind diffusion computation and enables continuous decoded-chunk emission for real-time interaction.

## 5 Evaluation

![](images/ed5160c515e04cd954e85c06059a1a58c57ab05ea9e80fe999b18dbeb6511d09.jpg)  
Figure 10 Qualitative results of DreamX-World-1.0-5B. Each row shows five uniformly sampled keyframes from a generated video sequence under different scene types and camera controls.

Since the evaluation of interactive world models is still at an exploratory stage, we devise our own evaluation suite that jointly probes camera controllability, perceptual quality, long-horizon behavior, and memory consistency. We first inspect the model qualitatively. Fig. 10 illustrates generated sequences across diverse scene types, camera trajectories, and visual styles. As the camera moves through the scene, the model produces smooth, temporally coherent transitions while maintaining high visual fidelity across heterogeneous scenes and styles. Additional samples and interactive demos are available on our project page1. We then turn to the quantitative evaluation detailed below.

We compare DreamX-World-1.0-5B against two representative open-source world models—HY-WorldPlay 1.5 (Team HunyuanWorld, 2025) (8B) and LingBot-World (Robbyant Team, 2026) (14B)—through quantitative evaluation on three complementary axes: basic evaluation on 5-second clips covering camera controllability and visual quality (table 3), long-horizon evaluation on ∼30-second rollouts (table 4), and memory consistency via revisit-based protocols (section 5.3). We additionally conduct a blind human preference study (section 5.4).

## 5.1 Basic Evaluation

We begin with a basic evaluation on standard 5-second generated videos, the most commonly used inference duration, focusing on camera controllability and visual quality. Extended evaluation over longer horizons is reported in table 4.

Camera Control Metric. We follow the camera controllability evaluation in WorldScore (Duan et al., 2025), but replace its pose estimator with MegaSaM (Li et al., 2025d) for more accurate camera pose recovery. Beyond adopting the evaluation protocol, we further enrich the evaluation camera trajectories and test data. On the trajectory side, we augment the original set with commonly encountered real-world camera motions such as upward tilt, downward tilt, and diagonal movement, which are under-represented in existing benchmarks yet frequent in practical interactive scenarios. On the data side, our evaluation set covers a broad spectrum of visual domains including AI-generated imagery, stylized content, and simulation-rendered scenes, ensuring that controllability is assessed across a wide range of appearance distributions. The camera control error is computed as:

$$
e _ { \mathrm { c a m e r a } } = { \sqrt { e _ { \theta } \cdot e _ { t } } } ,\tag{2}
$$

where $e _ { \theta }$ and et are scale-invariant rotation and translation errors with respect to the ground-truth trajectory, respectively. We compute the camera errors across all frames of the generated video V, which are subsequently normalized to yield a final score, where a higher score denotes superior camera controllability.

Visual Quality metrics. Following Omni-WorldBench (Wu et al., 2026), we assess visual quality from multiple complementary dimensions: imaging quality, temporal flickering, motion smoothness, dynamic degree, and transition detection. Together, these metrics comprehensively evaluate the generated videos in terms of frame-level clarity, inter-frame stability, motion intensity, and temporal coherence.

Artifact Detection Metric. Complementing the visual quality metrics above, we further introduce a multimodal large language model (Gemini-3.1-Pro) for artifact detection, leveraging its strong perceptual reasoning to flag visible defects in the generated frames. This metric focuses on critical defects and failures during the generation process, such as duplicated limbs, objects instantaneously vanishing, and geometric pass-through, among others. We sample frames from each generated video at 2 FPS and prompt the VLM to output a binary pass/fail judgment per sampled frame. Each test case is evaluated twice and averaged, and the final artifact score is the mean pass rate across the evaluation set.

Table 3 Quantitative comparisons on basic metrics. All scores are normalized to [0, 100] with higher being better.
<table><tr><td>Model</td><td>Params</td><td>Camera↑</td><td> Quality ↑</td><td>Trans.个</td><td>Flicker个</td><td></td><td> Smooth.个 Dynamic个</td><td>Artifact 个</td><td>Overall 个</td></tr><tr><td>HY-WorldPlay 1.5</td><td>8B</td><td>65.12</td><td>68.23</td><td>98.33</td><td>96.45</td><td>99.05</td><td>66.67</td><td>71.66</td><td>80.79</td></tr><tr><td> LingBot-World</td><td>14B</td><td>71.73</td><td>67.76</td><td>85.00</td><td>94.94</td><td>97.06</td><td>88.33</td><td>58.33</td><td>80.45</td></tr><tr><td>DreamX-World-1.0-5B</td><td>5B</td><td>73.75</td><td>66.75</td><td>98.33</td><td>96.17</td><td>98.79</td><td>85.83</td><td>73.75</td><td>84.76</td></tr></table>

The overall score is the average of all individual metrics. As shown in Tab. 3, our method achieves the highest camera control score and the best overall score while maintaining competitive visual quality. The combination of E-PRoPE camera conditioning and RL-based alignment enables precise trajectory adherence while maintaining competitive perceptual quality. In terms of motion naturalness, our model generates richer and more physically plausible dynamics, benefiting from the diverse camera coverage in the UE data engine and the forcing-based training that encourages robust temporal evolution.

## 5.2 Long-Horizon Evaluation

We further extend the evaluation to approximately 30-second generated rollouts to measure how each metric behaves under the long-horizon regime. We adopt the same set of metrics as the basic evaluation.

Table 4 Long-horizon evaluation on 30s generation rollouts. All scores are normalized to [0, 100] with higher being better.
<table><tr><td>Model</td><td>Params</td><td>Camera↑</td><td>Quality个</td><td>Trans.个</td><td>Flicker个</td><td></td><td>Smooth.个 Dynamic↑ Artifact个</td><td></td><td>Overall个</td></tr><tr><td>HY-WorldPlay 1.5</td><td>8B</td><td>65.86</td><td>63.02</td><td>91.00</td><td>97.00</td><td>99.11</td><td> 52.00</td><td>14.00</td><td>68.85</td></tr><tr><td> LingBot-World</td><td>14B</td><td>63.76</td><td>60.81</td><td>54.00</td><td>96.59</td><td>97.86</td><td>87.00</td><td>12.00</td><td>67.43</td></tr><tr><td>DreamX-World-1.0-5B</td><td>5B</td><td>62.03</td><td>64.11</td><td>80.00</td><td>96.35</td><td>98.41</td><td>75.00</td><td>17.00</td><td>70.41</td></tr></table>

As shown in table 4, DreamX-World-1.0-5B achieves the highest overall score (70.41), outperforming both HY-WorldPlay 1.5 (68.85, 8B) and LingBot-World (67.43, 14B). Our model attains the best imaging quality and artifact detection scores, demonstrating that the forcing-based architecture sustains higher visual fidelity over long horizons than larger competing models.

## 5.3 Memory Evaluation via Revisit Consistency

Existing world-model benchmarks such as WorldScore (Duan et al., 2025) and Omni-WorldBench (Wu et al., 2026) primarily assess short-term properties—visual quality, temporal flickering, and camera controllability— without requiring the agent to return to a previously visited region. Distributional metrics such as FVD and FID capture overall generation quality but cannot reveal whether a model remembers the specific scene it generated moments ago. In interactive settings, however, users inevitably revisit earlier locations, making long-horizon spatial memory a critical yet largely unevaluated capability.

To address this gap, we evaluate memory through revisit consistency, which detects frame pairs visiting the same spatial location at different times and checks whether the generated observations agree. A practical challenge is that imprecise camera control can introduce slight viewpoint offsets between revisit pairs, producing apparent inconsistencies unrelated to memory. We therefore adopt a multi-level metric suite so that each level captures a distinct facet of memory while offering progressively greater robustness to such offsets: pixel-level fidelity measures exact appearance preservation, perceptual consistency reflects human-perceived similarity, semantic identity captures high-level scene content, place recognition identifies the same location across viewpoint changes, and geometric structure verifies local feature correspondence. A temporal-smoothness metric further ensures that memory is not achieved at the cost of incoherent transitions.

Trajectory Construction. We construct camera trajectories that explicitly induce revisits using simple navigation primitives (forward/backward translation, left/right rotation). As shown in Fig. 11, three complementary templates are used: (1) an out-and-back path that revisits with nearly identical orientation, testing appearance stability; (2) a closed-loop path that returns to the starting pose, testing globally consistent layout under loop closure; and (3) a translation-rotation path that introduces heading changes, testing place-identity preservation under viewpoint variation.

![](images/efef4586889543f9280cf78913cda278e33eba0d8a3a03dc133465cc8be98648.jpg)

![](images/86d71fe1df37d49ac70aaa31065b5105e69d76ff4e3edec7773d13557020b3ec.jpg)

![](images/b02cf14dc75a258e5199e65f6812724acdf6af93b9f9cda8f66b34b98429462b.jpg)  
Figure 11 Bird’s-eye view of the three evaluation trajectory templates. Color encodes temporal progression from start (blue) to end (red); arrows indicate camera movement direction. (a) Out-and-back: translates laterally (D×3) then reverses (A×3), returning with identical orientation. (b) Translation–rotation: combines translation with heading changes (W·S·L·R·R·L·L), revisiting from a different yaw angle. (c) Closed-loop: traverses a rectangular path back to the exact starting pose. The ⋆ marker denotes the revisit point that coincides with the start.

Revisit Pair Detection. From camera extrinsics, we extract position $\mathbf { t } = \left( t _ { x } , t _ { y } , t _ { z } \right)$ and yaw θ for each frame. A revisit pair (i, j) requires

$$
\begin{array} { r } { | \theta _ { i } - \theta _ { j } | \leq \tau _ { \theta } , \qquad \| \mathbf { t } _ { i } - \mathbf { t } _ { j } \| _ { 2 } \leq \tau _ { t } , } \end{array}\tag{3}
$$

with $\tau _ { \theta } = 2 ^ { \circ } , \tau _ { t } = 0 . 1$ , and a minimum temporal gap $| j - i | \geq \lfloor 0 . 2 T \rfloor$ to focus on long-horizon memory.   
Among multiple candidates, we select the pair with the smallest weighted pose distance $\lvert \theta _ { i } - \theta _ { j } \rvert + 1 0 \lvert \lvert \mathbf { t } _ { i } - \mathbf { t } _ { j } \rvert \rvert _ { 2 }$ .

Metrics. For each revisit pair $( I _ { i } , I _ { j } )$ we compute metrics spanning all five aspects. (i) Pixel-level fidelity. PSNR and SSIM measure signal-to-noise ratio and local structural similarity, respectively, providing strict low-level references sensitive to any scene-content drift. (ii) Perceptual consistency. LPIPS (Zhang et al., 2018) measures perceptual distance in a deep feature space, capturing whether the two frames look different to a human observer; lower values indicate higher consistency. (iii) Semantic identity. DINO-Sim computes cosine similarity between frozen DINOv2 (Oquab et al., 2023) features of the two frames; a drop signals altered semantic content upon revisitation. (iv) Place recognition. VPR-Sim uses global descriptors from MutualVPR (Gu et al., 2026b), which is trained to retrieve the same location across viewpoint changes, thereby reducing the confounding effect of camera-control error. (v) Geometric structure. SP-Match detects up to 1024 SuperPoint (DeTone et al., 2018) keypoints per frame and matches them with LightGlue (Lindenberger et al., 2023); we report the matching ratio $r _ { \mathrm { m a t c h } } = N _ { \mathrm { m a t c h } } / \operatorname* { m i n } ( N _ { i } , N _ { j } )$ . We additionally report CLIP-Video (Radford et al., 2021), the average CLIP similarity between consecutive frames, as a complementary temporal-smoothness measure.

Gain-based Scoring. Absolute similarity scores can be inflated by slow camera movement rather than genuine memory. We therefore sample non-revisit baseline pairs with a matched temporal-gap distribution and report all metrics as gains: $S _ { \mathrm { r e v i s i t } } - S _ { \mathrm { b a s e l i n e } }$ for similarity metrics, $S _ { \mathrm { b a s e l i n e } } - S _ { \mathrm { r e v i s i t } }$ for LPIPS, so that positive values consistently indicate better memory. CLIP-Video is reported as an absolute value.

Table 5 Memory consistency evaluation on 10-second generated videos. Metrics span pixel-level fidelity (∆PSNR, ∆SSIM), perceptual consistency (∆LPIPS), semantic identity (∆DINO-Sim), place recognition (∆VPR-Sim), and geometric structure (∆SP-Match). All gains are over non-revisit baselines; CLIP-V is absolute. Higher is better for all columns.
<table><tr><td>Model</td><td></td><td></td><td></td><td></td><td></td><td>△PSNR △SSIM △LPIPS △DINO-Sim △VPR-Sim △SP-Match CLIP-V</td><td></td></tr><tr><td> LingBot-World</td><td>0.61</td><td>0.019</td><td>0.039</td><td>0.090</td><td>0.100</td><td>0.088</td><td>0.987</td></tr><tr><td>HY-WorldPlay 1.5</td><td>3.19</td><td>0.079</td><td>0.202</td><td>0.200</td><td>0.110</td><td>0.251</td><td>0.992</td></tr><tr><td>DreamX-World-1.0-5B</td><td>3.92</td><td>0.098</td><td>0.232</td><td>0.246</td><td>0.142</td><td>0.216</td><td>0.991</td></tr></table>

As shown in Table 5, DreamX-World-1.0-5B achieves the highest gains on pixel-level, perceptual, semantic, and place-recognition metrics, demonstrating stronger memory at every level of abstraction. HY-WorldPlay 1.5 leads on SP-Match and CLIP-Video, while LingBot-World shows lower gains across all revisit metrics.

## 5.4 Human Preference Study

To complement automatic metrics, we conduct a blind side-by-side human preference study. Each trial compares DreamX-World-1.0-5B with one baseline under the same prompt, initial condition, camera/action trajectory, and playback setting. The model identities are anonymized and the left-right order is randomized. Assessors report whether DreamX-World-1.0-5B wins, ties, or loses along four dimensions: overall preference, camera control, visual quality, and artifact detection. We report all percentages from the perspective of DreamX-World-1.0-5B.

![](images/fb588120fe26162fe6dbef33ad212195391b2c55ec3630df82e9d22d18471079.jpg)  
Figure 12 Human preference study comparing DreamX-World-1.0-5B with HY-WorldPlay 1.5 and LingBot-World. Each horizontal stacked bar reports Win/Tie/Lose percentages from the perspective of DreamX-World-1.0-5B under blind side-by-side comparison. DreamX-World-1.0-5B is preferred in overall preference, visual quality, and artifact detection, while camera-control judgments are close with many ties, indicating comparable perceived controllability.

As shown in figure 12, DreamX-World-1.0-5B is preferred more often in overall preference, achieving win/tie/loss rates of 57.5/14.4/28.1 against HY-WorldPlay 1.5 and 61.9/10.6/27.5 against LingBot-World. The preference is also consistent in visual quality, where DreamX-World-1.0-5B obtains 57.5% and 61.3% win rates against the two baselines, and in artifact detection, where it wins 59.4% and 56.2% of comparisons. Camera-control judgments are closer, with higher tie rates, suggesting comparable perceived controllability among the systems under side-by-side viewing. Overall, the human study supports the automatic evaluation: DreamX-World-1.0-5B improves perceptual quality and artifact robustness while maintaining competitive camera controllability.

## 6 Related Work

Video Generation and Interactive World Models. Diffusion models (Ho et al., 2020; Song et al., 2021; Peebles and Xie, 2023; Song et al., 2021) have become a standard approach to video generation, progressing from simple image synthesis (Chu et al., 2025; Lei et al., 2023, 2026) to complex cinematic video generation (Team Wan et al., 2025; Feng et al., 2026). Current interactive world models extend this setting by conditioning future observations on user inputs or agent actions. Early systems primarily target gaming scenarios. Genie (Bruce et al., 2024) learns action-conditioned environments from videos, GameNGen (Valevski et al., 2024) demonstrates real-time neural game simulation, and GameGen-X (Che et al., 2025) supports interactive open-world game generation. More recent systems move toward general-domain video worlds with camera navigation, streaming generation, or object interaction (Team HunyuanWorld, 2025; Robbyant Team, 2026; Hong et al., 2025; Wang et al., 2026c; Zhao et al., 2026a; Gu et al., 2026a). Our work follows this general direction and studies camera control, long-horizon generation, scene memory, event control, real-time interaction, and world model evaluation.

Camera-controlled Video Generation. Camera conditioning has been introduced through explicit motion features, camera embeddings, and geometric representations. MotionCtrl separates camera and object motion control (Wang et al., 2024), while CameraCtrl injects camera trajectories into pretrained video diffusion models (He et al., 2025). AC3D analyzes the representation and training choices required for 3D camera control in video DiTs (Bahmani et al., 2025). PRoPE instead incorporates camera intrinsics and extrinsics directly into self-attention as a projective relative positional encoding (Li et al., 2025a). Following this, we further introduce E-PRoPE, which applies camera-aware attention to a spatially reduced token set, significantly improving computation efficiency.

Long-horizon Generation and Memory. Autoregressive video generation enables streaming but introduces exposure bias and accumulated errors. Self-Forcing (Huang et al., 2025) address this by directly training on model outputs, while Stable Video Infinity (Li et al., 2025b) adds prediction error injection to model input to reduce its reliance on clean frames, mitigating the negative impact of imperfect context during inference. LongLive and Infinity-RoPE primarily study long-context autoregressive generation (Yang et al., 2025; Yesiltepe et al., 2025). Causal Forcing identifies the mismatch between bidirectional teachers and causal students and proposes distillation from an autoregressive teacher (Zhu et al., 2026; Zhao et al., 2026b). Beyond recent context, memory-based methods retrieve or compress earlier observations to preserve scene identity during revisits (Yu et al., 2025; Sun et al., 2025; Hong et al., 2025; Yu et al., 2026; Wang et al., 2026c). Our training pipeline combines autoregressive distillation and rollout-based correction with geometry-guided retrieval and error-recycled memory conditioning.

Efficient Sampling. Existing popular few-step diffusion reduces sampling cost primarily through distribution matching distillation (Yin et al., 2024b,a). Interactive video systems combine such distillation with causal generation and KV caching to produce frames incrementally (Huang et al., 2025; Yang et al., 2025; Zhu et al., 2026). Complementary systems techniques reduce the cost of individual model components (Zhang et al., 2025; Wang et al., 2026a; Liu et al., 2025a; RiseAI-Sys, 2026): SageAttention provides low-precision attention (Zhang et al., 2025), TeaCache reuses intermediate results across diffusion timesteps (Liu et al., 2025a), and ParaVAE distributes VAE decoding across devices (RiseAI-Sys, 2026). We combine few-step autoregressive generation with quantized DiT execution, residual reuse, parallel VAE decoding, and asynchronous serving.

Reinforcement Learning. Reinforcement learning enables diffusion models to optimize non-differentiable objectives such as perceptual quality and prompt alignment. DDPO formulates denoising as a sequential decision process (Black et al., 2023), while Flow-GRPO extends online policy optimization to flow-matching models (Liu et al., 2025b). DiffusionNFT instead applies negative-aware fine-tuning directly to the forward process, avoiding reverse-process likelihood estimation and solver-specific training (Zheng et al., 2025). However, aggressive reward optimization can reduce generative diversity (Chen et al., 2026). For video world models, reinforcement learning must additionally preserve temporal coherence during autoregressive rollout. WorldCompass combines clip-level sampling with interaction and visual-quality rewards (Wang et al., 2026b), while Astrolabe applies forward-process reinforcement learning to distilled autoregressive video models using streaming rollouts and local optimization windows (Zhang et al., 2026). Our reinforcement learning stage follows this setting: long rollouts provide temporal context, short clips serve as reward-bearing units, and camera-control and video-quality rewards guide a conservative update that preserves the distilled prior.

World Model Evaluation World-model evaluation has gradually moved beyond frame-level video quality (Chen et al., 2025; Ling et al., 2025) toward controllability, consistency, and interactive response. World-Score (Duan et al., 2025) evaluates world generation under prescribed camera trajectories along controllability, quality, and dynamics. Omni-WorldBench(Wu et al., 2026) focuses on whether user interactions produce the intended outcomes and intermediate state transitions. WBench (Ying et al., 2026) further considers multi-turn navigation, subject actions, event editing, and perspective switching, measuring interaction adherence, consistency, and physical compliance. Complementary to these benchmarks, our evaluation separately examines short-video camera control and visual quality, long-horizon autoregressive generation, and revisit consistency over extended trajectories.

## 7 Limitations

Although DreamX-World 1.0 improves interaction, controllability, and efficiency, several challenges remain. First, long-horizon visual and geometric consistency is still difficult: generated worlds may drift drastically in object appearance or layout after extended interaction. Second, control signals like caption, camera and event may conflict when an event produces visual content that is incompatible with the future observations in certain world settings specified by the caption. Third, automatic evaluation of world models remains imperfect; benchmarks such as Omni-WorldBench V2 are important, but human and task-based evaluation will remain necessary for open-ended interaction.

## 8 Conclusion

We presented DreamX-World 1.0, a general-purpose interactive world model. The central lesson is that world modeling is a full-stack problem: data curation, training, evaluation, and inference acceleration must be organized and improved from a global perspective. Driven by this full-stack perspective, DreamX-World 1.0 establishes a promising foundation for the next generation of interactive world models.

Future work. Two directions are especially promising. First, character-centric world models. It primarily focuses on maintaining persistent character identity, coordinating character actions with freely moving cameras, and supporting richer multi-character interactions over long horizons. Second, native audio-visual world models. It jointly generates synchronized speech, ambient sound, and action-dependent audio, while also utilizing sound as an interactive signal for events and scene dynamics. Together with stronger memory and physical reasoning, these extensions would move world models toward more embodied, expressive, and immersive simulation.

## Authors

Team Members. Team members are listed alphabetically by last name (and by first name where last names are identical). The ordering does not indicate relative contributions.

Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, Geng Li, Jifan Li, Ruimin Lin, Qingfeng Shi, Bingze Song, Lei Sun, Jing Tang, Ruitian Tian, Jun Wang, Jiahong Wu, Pengfei Zhang, Shen Zhang, and Jiashu Zhu.

Acknowledgements. We thank the AMAP-ML team for computational infrastructure, engineering support, and discussions that made this work possible.

## References

Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B. Lindell, and Sergey Tulyakov. AC3D: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673, 2025.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024.

Rui Cen, QiangQiang Hu, Hong Huang, Hong Liu, Song Liu, Xin Luo, Lin Niu, Yifan Tan, Decheng Wu, Linchuan Xie, Rubing Yang, Guanghua Yu, and Jianchen Zhu. Angelslim: A more accessible, comprehensive, and efficient toolkit for large model compression. arXiv preprint arXiv:2602.21233, 2026.

Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. In ICLR, volume 2025, pages 37546–37593, 2025.

Chubin Chen, Sujie Hu, Jiashu Zhu, Meiqi Wu, Jintao Chen, Yanxun Li, Nisha Huang, Chengyu Fang, Jiahong Wu, Xiangxiang Chu, et al. Taming preference mode collapse via directional decoupling alignment in diffusion reinforcement learning. In CVPR, pages 12775–12786, 2026.

Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. FingER: Content aware fine-grained evaluation with reasoning for AI-generated videos. In ACM MM, pages 3517–3526, 2025. doi: 10.1145/3746027.3755102.

Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. In ICCV, pages 18475–18486, October 2025.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. GPG: A simple and strong reinforcement learning baseline for model reasoning. In ICLR, 2026.

Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperPoint: Self-supervised interest point detection and description. In CVPR Workshops, pages 224–236, 2018.

Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983, 2025.

Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Fangyuan Mao, Haiyang Guo, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. Enhancing train-free infinite-frame generation for consistent long videos. arXiv preprint arXiv:2605.18233, 2026.

Bohai Gu, Taiyi Wu, Yueyang Yuan, Jian Liu, Xiaocheng Lu, Dazhao Du, Jie Zhang, Jinxiang Lai, Shuai Yang, Xiaotong Zhao, Alan Zhao, and Song Guo. Worldcraft: From camera navigation to object manipulation in interactive video world models. arXiv preprint arXiv:2605.25077, 2026a.

Qiwen Gu, Xufei Wang, Junqiao Zhao, Siyue Tao, Tiantian Feng, Ziqiao Wang, and Guang Chen. Mutualvpr: A mutual learning framework for resolving supervision inconsistencies via adaptive clustering. In NeurIPS, pages 2899–2922, 2026b.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for video diffusion models. In ICLR, 2025.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, volume 33, pages 6840–6851, 2020.

Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan. RELIC: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. In ICCV, 2025.

Jiachen Lei, Qinglong Wang, Peng Cheng, Zhongjie Ba, Zhan Qin, Zhibo Wang, Zhenguang Liu, and Kui Ren. Masked diffusion models are fast distribution learners. arXiv preprint arXiv:2306.11363, 2023.

Jiachen Lei, Keli Liu, Julius Berner, Haiming Yu, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. There is no vae: End-to-end pixel-space generative modeling via self-supervised pre-training. arXiv preprint arXiv:2510.12586, 2026.

Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. NeurIPS, 2025a.

Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025b.

Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Zhixiang Wang, Yuwei Wu, Tong He, Jiangmiao Pang, Yu Qiao, Yunde Jia, and Kaipeng Zhang. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025c.

Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos. arXiv preprint arXiv:2412.04463, 2025d.

Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In ICCV, pages 17627–17638, 2023.

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, 2024.

Xinrang Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. VMBench: A benchmark for perception-aligned video motion generation. arXiv preprint arXiv:2503.10076, 2025.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In CVPR, 2025a.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online reinforcement learning. arXiv preprint arXiv:2505.05470, 2025b.

Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-Effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025a.

Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025b.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.

RiseAI-Sys. ParaVAE: A parallelism distributed 3d vae for efficient vae training and inference with slicing and tiling optimization. https://github.com/RiseAI-Sys/ParaVAE/, 2026.

Robbyant Team. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

Anian Ruoss, Grégoire Delétang, Tim Genewein, Jordi Grau-Moya, Róbert Csordás, Mehdi Bennani, Shane Legg, and Joel Veness. Randomized positional encodings boost length generalization of transformers. arXiv preprint arXiv:2305.16843, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2021.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

Team HunyuanWorld. HY-World 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. Technical report, 2025.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.

Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Yi Zhang, Chang Zeng, Yanxi Zhou, Xiao-Xiao Long, Hao Zhu, Zhaoxiang Zhang, Xun Cao, and Yao Yao. Spatialvid: A large-scale video dataset with spatial annotations. arXiv preprint arXiv:2509.09676, 2025.

Yifei Wang, Yueqi Wang, Zhenrui Yue, Huimin Zeng, Yong Wang, Ismini Lourentzou, Zhengzhong Tu, Xiangxiang Chu, and Julian McAuley. Fasa: Frequency-aware sparse attention. arXiv preprint arXiv:2602.03152, 2026a.

Zehan Wang, Tengfei Wang, Haiyu Zhang, Xuhui Zuo, Junta Wu, Haoyuan Wang, Wenqiang Sun, Zhenwei Wang, Chenjie Cao, Hengshuang Zhao, et al. Worldcompass: Reinforcement learning for long-horizon world models. arXiv preprint arXiv:2602.09022, 2026b.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, 2024.

Zile Wang, Zexiang Liu, Jaixing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026c.

Meiqi Wu, Zhixin Cai, Fufangchen Zhao, Xiaokun Feng, Rujing Dang, Bingze Song, Ruitian Tian, Jiashu Zhu, Jiachen Lei, Hao Dou, Jing Tang, Lei Sun, Jiahong Wu, Xiangxiang Chu, Zeming Liu, and Kaiqi Huang. Omni-worldbench: Towards a comprehensive interaction-centric evaluation for world models. arXiv preprint arXiv:2603.22212, 2026.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. arXiv preprint arXiv:2511.20649, 2025.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024a.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024b.

Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation, 2026. https://arxiv.org/abs/2605.25874.

Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. In ACM SIGGRAPH Asia, 2025.

Wei Yu, Runjia Qian, Yumeng Li, Liquan Wang, Songheng Yin, Sri Siddarth Chakaravarthy P, Dennis Anthony, Yang Ye, Yidi Li, Weiwei Wan, and Animesh Garg. Mosaicmem: Hybrid spatial memory for controllable video world models. arXiv preprint arXiv:2603.17117, 2026.

Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In ICLR, 2025.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586–595, 2018.

Songchun Zhang, Zeyue Xue, Siming Fu, Jie Huang, Xianghao Kong, Yue Ma, Haoyang Huang, Nan Duan, and Anyi Rao. Astrolabe: Steering forward-process reinforcement learning for distilled autoregressive video models. arXiv preprint arXiv:2603.17051, 2026.

Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, Fan Bao, and Jun Zhu. minWM: A full-stack open-source framework for real-time interactive video world models. arXiv preprint arXiv:2605.30263, 2026a.

Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026b.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. In ACM SIGGRAPH, 2018.

Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, Mingyu Liu, Dingning Liu, Jiange Yang, Zhoujie Fu, Junyi Chen, Chunhua Shen, Jiangmiao Pang,

Kaipeng Zhang, and Tong He. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.