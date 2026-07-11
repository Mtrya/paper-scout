Title: 4D Embodied World Models for Robotic Manipulation

URL Source: https://arxiv.org/html/2607.06559

Markdown Content:
1]DAMO Academy, Alibaba Group, 2]Hong Kong Embodied AI Lab, 3]CUHK, 4]Hupan Lab \contribution[*]Equal contribution \contribution[†]Corresponding author

(July 7, 2026)

###### Abstract

Robotic manipulation in the open world requires not only recognizing what a scene looks like, but also anticipating how its 3D structure moves under interaction. We argue that synchronized RGB, depth, and optical flow (RGB-DF) provide a physically grounded representation that captures the underlying 4D dynamics of a scene. Compared to 2D pixel videos, this multi-modal synergy aligns visual appearance with geometric structure and temporal motion, creating a representation space significantly closer to low-level end-effector actions demanded by robotic systems, narrowing the gap between world prediction and policy learning. Building on this insight, we introduce RynnWorld-4D, a generative model that co-produces future RGB frames, depth maps, and optical flow from a single RGB-D image and a language instruction within one unified diffusion process. This 4D world model features a tri-branch architecture that integrates cross-modal attention with frame-wise 3D RoPE, ensuring that appearance, geometry, and motion evolve consistently. To supply training data at scale, we curate Rynn4DDataset 1.0, a massive dataset of over 254.4 million frames across egocentric human and robotic manipulation videos with high-quality pseudo-labels for depth and optical flow. We further propose RynnWorld-4D-Policy, an inverse dynamics head that consumes the internal 4D representations of RynnWorld-4D in a single forward pass, bypassing expensive multi-step denoising, to output robot actions in a closed-loop manner. Experiments show that RynnWorld-4D produces temporally and spatially coherent 4D predictions, and that RynnWorld-4D-Policy achieves state-of-the-art performance on real-world dexterous bimanual manipulation tasks, particularly excelling in tasks demanding spatial precision and temporal coordination.

## 1 Introduction

Robotic manipulation in the open world could greatly benefit from visual world models that predict how the environment would evolve given an agent’s interactions (zhao2026towards; zhao2025smap; li2026causal; agarwal2025cosmos; ali2025world). While recent generative video models (ha2018recurrent; xiang2024pandora; zheng2024open; wang2025wan) have shown encouraging progress in policy synthesis (du2023learning; liang2024dreamitate; zhen2025learning), data simulation and generation (zhu2024irasim), and long-horizon planning (du2023video; li2025novaflow), they remain limited by the 2D projective nature of pixels. This inherent limitation leads to a loss of critical spatial relationships, preventing precise 6-DoF pose estimation and depth-aware interaction (hu2024video; agarwal2025cosmos; li2026causal). Furthermore, 2D models often lack geometric grounding, leading to temporal inconsistencies such as fluctuating object scales and unphysical shape morphing, which hinders their utility in robust policy learning. Consequently, transitioning generative world modeling from 2D videos to geometry-integrated 4D scene evolution is an essential step toward a solid foundation for embodied intelligence.

Existing 4D scene-modeling approaches fall into two categories. The first builds on Neural Radiance Fields (NeRF) (mildenhall2021nerf) or 3D Gaussian Splatting (3DGS) (kerbl20233d), which can be further divided into optimization-based methods (zhao2024sg; zhao2024hfgs; yu20244real; bahmani20244d) that are computationally intensive and scene-specific, and feed-forward models (ren2024l4gm; wu2025cat4d) that prioritize speed but typically focus on object-centric generation. These approaches often require multi-view inputs or struggle to scale to complex scene-level environments. The second category comprises dynamic Structure-from-Motion (SfM) approaches (wang2025continuous; li2025megasam), which reconstruct time-varying point clouds but lack the generative capability to predict future states from a single image. Neither category readily provides a compact, scalable representation that integrates with the strong generative priors of pretrained video diffusion models.

![Image 1: Refer to caption](https://arxiv.org/html/2607.06559v1/x3.png)

Figure 1: Given an input RGB-D image and description, RynnWorld-4D generates RGB, depth, and optical flow videos synchronously, which can be further lifted into 3D scene flow (right).

To bridge this gap, we propose a lightweight projective 4D representation by predicting synchronized sequences of RGB, depth, and optical flow (RGB-DF): depth lifts each pixel to a 3D location, and depth together with optical flow can be back-projected into 3D scene flow under standard pinhole-camera assumptions, providing a per-point 3D motion cue (illustrated as the “3D Flow” in Fig. [1](https://arxiv.org/html/2607.06559#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). Compared to RGB-only sequences, this representation makes geometry and motion explicit; compared to explicit 3D volumes or 4D Gaussians, it stays in a 2D-aligned format and therefore inherits the scalability and the rich generative priors of large-scale video diffusion models.

Building on this representation, we present RynnWorld-4D, a 4D embodied world model that, conditioned on a single RGB-D image and a text instruction, synchronously generates RGB, depth, and optical-flow videos within one shared denoising loop (Fig. [1](https://arxiv.org/html/2607.06559#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). Specifically, we extend a pretrained video diffusion model (wang2025wan) into a tri-branch transformer, where each branch handles one modality with independent transformer with shared cross attention keys/values across modalities, while Joint Cross-Modal Attention modules enforce cross-modal consistency. This design preserves the strong generative priors of the pretrained backbone while allowing each modality to specialize, i.e., textures for RGB, spatial geometry for depth, and motion displacements for optical flow. A key challenge for training RynnWorld-4D is the absence of large-scale datasets with dense 4D annotations. To address this, we curate Rynn4DDataset 1.0, a large-scale hybrid dataset comprising over 254 million video frames drawn from egocentric human activity datasets (damen2020epic; wang2024egovid) and robotic manipulation datasets (wu2024robomind; liu2024rdt; jiang2025galaxea; wu2025robocoin; bu2025agibot), each enriched with high-quality pseudo-annotations for depth and optical flow.

The RGB-DF representation offers a critical advantage: it aligns more closely with a robot’s action space than raw 2D pixel changes. Consequently, a downstream policy trained on RynnWorld-4D’s internal 4D representations bypasses the heavy structural inference typically required when operating on 2D latents alone. Leveraging this synergy, we introduce RynnWorld-4D-Policy, an inverse dynamics head that extracts robot actions directly from RynnWorld-4D ’s predictive 4D features. By utilizing these internal latents in a single forward pass and bypassing the iterative denoising bottleneck, RynnWorld-4D-Policy enables high-frequency, closed-loop control suitable for real-time interaction. In summary, our work makes the following contributions:

*   •
We introduce a projective 4D representation that co-generates RGB, depth, and optical flow, and we show how it admits a natural 3D-scene-flow reading that makes geometry and motion explicit while staying compatible with large-scale video diffusion priors.

*   •
We develop RynnWorld-4D, a tri-branch 4D world model that co-generates physically coherent RGB-DF sequences through mutual cross-modal interactions.

*   •
We curate Rynn4DDataset 1.0, a large-scale 4D embodied video dataset with depth and optical flow annotations for training 4D embodied world model.

*   •
We propose RynnWorld-4D-Policy, which leverages the internal 4D representations to enable high-frequency, closed-loop robotic control.

## 2 Related Work

### 2.1 World Model

Learning a dynamics model of the world that supports downstream action generation has been a long-standing challenge (ha2018world; sutton1991dyna). Early work learns world models in low-dimensional state spaces (achille2018separation; lesort2018state), which are efficient to train but difficult to generalize across visually diverse environments. With advances in generative modeling, a growing body of recent work has explored video models as foundation world models (kong2024hunyuanvideo; wang2025wan; yang2024cogvideox). However, these models remain in the 2D pixel space, limiting their ability to capture 3D geometric structure and leaving a large representational gap between their predictions and the 3D actions a robot must produce.

3D world models attempt to close this gap by reasoning over meshes or explicit surfaces (wang2021neus; pfaff2020learning; jiang2025phystwin; zhao2025physsplat; xia2025drawer; xia2024video2game; zhen2025learning; guo2026articulat3d), radiance fields or Gaussians (mildenhall2021nerf; kerbl20233d; driess2023learning; xie2024physgaussian), or particle systems (sanchez2020learning; abou2024particlenerf; zhang2025particle; chen20254dnex). Hybrid approaches additionally reason over hierarchical structures (kaelbling2011hierarchical; wang2025enact; zhao2026high). While these representations offer richer geometric reasoning, they often require multi-view inputs, are scene-specific, or lack the scalability of pretrained video priors. Most closely related to our work are (zhen2025learning), which models the 4D scene from RGB-DN (RGB, Depth, and Normal) videos with language-conditioned control, and (chen20254dnex), which produces high-quality dynamic point clouds for novel-view video synthesis. Our work shares the goal of scalable 4D prediction but introduces a projective 4D representation that co-generates optical flow alongside RGB and depth, making inter-frame 3D motion explicit. Unlike methods relying on static geometry like surface normals (zhen2025learning), our inclusion of optical flow allows for back-projection into 3D scene flow, providing explicit dynamic cues essential for learning accurate inverse dynamics. This is particularly critical for dexterous manipulation, where the fine-grained trajectory of objects and end-effectors is what differentiates success from failure.

### 2.2 Future Prediction for Embodied Control

A growing line of work bridges generative modeling and control by using 2D future prediction to guide policy learning (bharadhwaj2024gen2act; ye2024latent; ye2026world; bi2025motus; hu2024video). Representative methods include SuSIE (black2023zero), which employs a goal-conditioned keyframe generator (brooks2023instructpix2pix), and UniPi (du2023learning), which learns inverse dynamics over generated sequences. Downstream actions are then derived via online planning (hu2024video; williams2017model; hafner2019learning; pineau2003point), offline policy synthesis (hafner2019dream; hansen2023td; chua2018deep; hafner2025training), or inverse-dynamics models (du2023learning; bi2025motus). However, because these pipelines operate entirely in 2D pixel space and often require repeated denoising for every action step, they face inherent limitations in both geometric accuracy and control reactivity. In contrast, RynnWorld-4D operates on a unified 4D representation that jointly encodes appearance, geometry, and motion. Building upon this, RynnWorld-4D-Policy directly consumes the internal predictive features of RynnWorld-4D in a single forward pass. By bypassing the need for per-step video decoding and denoising, our approach enables high-frequency, closed-loop robotic control, effectively translating imagined 4D trajectories into precise, real-time robotic actions.

## 3 Method

To address the data scarcity in 4D generative modeling, we first introduce Rynn4DDataset 1.0 in Sec. [3.1](https://arxiv.org/html/2607.06559#S3.SS1 "3.1 Rynn4DDataset 1.0 ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"), a large-scale hybrid dataset specifically curated for training feed-forward 4D generative models. Building upon this, Sec. [3.3](https://arxiv.org/html/2607.06559#S3.SS3 "3.3 RynnWorld-4D ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation") presents RynnWorld-4D, a framework capable of co-generating future sequences—including RGB frames, depth maps, and optical flow from a single RGB-D observation and a linguistic task description. Finally, in Sec. [3.4](https://arxiv.org/html/2607.06559#S3.SS4 "3.4 RynnWorld-4D-Policy ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"), we introduce RynnWorld-4D-Policy, which leverages the predictive 4D representations from RynnWorld-4D to derive final robotic actions.

### 3.1 Rynn4DDataset 1.0

To bridge the gap in large-scale 4D training data, we introduce Rynn4DDataset 1.0, a hybrid dataset comprising over 254.4 million video frames from human-centric (Epic-Kitchens (damen2020epic), EgoVid (wang2024egovid)) and robotic manipulation datasets (RoboMIND (wu2024robomind), RDT-1B (liu2024rdt), Galaxea (jiang2025galaxea), RoboCoin (wu2025robocoin), AgiBot (bu2025agibot)). Each frame is enriched with high-quality 4D pseudo-annotations: fine-grained instructions (bai2025qwen3), monocular depth (lin2025depth), and dense optical flow (morimitsu2025dpflow). The statistics and composition of Rynn4DDataset 1.0 are visualized in Fig. [2](https://arxiv.org/html/2607.06559#S3.F2 "Figure 2 ‣ 3.1 Rynn4DDataset 1.0 ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"). The pipeline for our multimodal annotation is illustrated in Fig. [3](https://arxiv.org/html/2607.06559#S3.F3 "Figure 3 ‣ 3.1 Rynn4DDataset 1.0 ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation").

![Image 2: Refer to caption](https://arxiv.org/html/2607.06559v1/x4.png)

Figure 2: Composition of the Rynn4DDataset 1.0 dataset. We provide a large-scale hybrid collection of 254.4M frames, balancing human egocentric videos with diverse robotic manipulation data. This diversity ensures that the world model learns both general object interaction priors and robot-specific execution traces.

Video captioning. We use Qwen3-VL (bai2025qwen3) to generate captions for the video data (damen2020epic; wang2024egovid; wu2024robomind; liu2024rdt; jiang2025galaxea). Specifically, we leverage the model’s strong video-language understanding capabilities to produce detailed, structured descriptions of each video clip. The videos are first sampled at a frame rate of 1 FPS and split into segments of 5 seconds. For each segment, we provide the following prompt to the model:

> Please describe this video in detail. Include the following aspects:
> 
> 
> 1. The main subject and action in the video.
> 
> 
> 2. The environment and background.
> 
> 
> 3. Any objects and their interactions.
> 
> 
> 4. The overall scene context and atmosphere.
> 
> 
> Provide a concise but comprehensive caption in one paragraph.

The generation is performed with a maximum output length of 512 tokens and a temperature of 0.7 to balance creativity and coherence. The generated captions are then collected and stored in JSON format for downstream tasks.

![Image 3: Refer to caption](https://arxiv.org/html/2607.06559v1/x5.png)

Figure 3: Data Curation Pipeline. The video data is collected from diverse sources and partitioned into short clips during data preprocessing. Each clip undergoes a multi-modal annotation process: (1) Video Captioning: Qwen3-VL (bai2025qwen3) generates detailed natural language descriptions of the video content; (2) Optical Flow Estimation: DPFlow (morimitsu2025dpflow) computes dense per-frame motion fields, which are visualized and saved as flow videos; (3) Depth Estimation: Depth Anything 3 (lin2025depth) produces monocular depth predictions, which are upsampled to the original resolution and saved as depth videos with a global depth range of [0.0,5.0] meters.

Optical flow annotation. We employ DPFlow (morimitsu2025dpflow), a state-of-the-art optical flow estimation model. For each video, frame pairs are processed sequentially at native resolution, and the estimated flow fields are visualized via color encoding and saved as MP4 videos at 25 FPS.

Depth annotation. We employ Depth Anything 3 (lin2025depth), specifically the DA3NESTED-GIANT-LARGE-1.1 checkpoint, which provides dense per-frame depth predictions along with camera pose estimation. Each video is sampled at 30 FPS and processed at a working resolution of 392 pixels (short side, upper-bound resize).

To convert the estimated depth maps into viewable depth videos, we load the compressed depth arrays and upsample each frame to the original video resolution using bilinear interpolation. Depth values are clipped to a global range of [0.0,5.0] meters and quantized to 8-bit grayscale via I=\lfloor d/d_{\max}\times 255\rfloor. The resulting frames are saved as RGB videos.

### 3.2 3D Scene Reconstruction from Multi-Modal Videos

A key advantage of the RGB-DF (i.e., RGB, depth, and optical flow) representation is its inherent geometric interpretability. By combining the co-generated depth and optical flow, we can reconstruct a temporally consistent 3D scene and derive metric scene flow.

Geometric Unprojection. Given the generated depth map D_{t} at frame t, each pixel \mathbf{p}_{t}=[u,v,1]^{\top} in homogeneous coordinates is unprojected into the 3D camera space as:

\mathbf{P}_{t}=D_{t}(u,v)\cdot\mathbf{K}^{-1}\mathbf{p}_{t},(1)

where \mathbf{K} is the camera intrinsic matrix. This process lifts the 2D projective sequence into a metric 3D point cloud \mathcal{C}_{t}=\{\mathbf{P}_{t}^{i}\}_{i=1}^{H\times W}.

Metric Scene Flow Derivation. To capture the underlying 4D dynamics, we leverage the co-generated dense optical flow \mathbf{f}_{opt}=[\Delta u,\Delta v]^{\top} to establish temporal correspondences. A 3D point \mathbf{P}_{t} is tracked to its position at t+1 by:

\mathbf{P}_{t+1}=D_{t+1}(u+\Delta u,v+\Delta v)\cdot\mathbf{K}^{-1}(\mathbf{p}_{t}+[\Delta u,\Delta v,0]^{\top}).(2)

The 3D scene flow is then defined as \mathbf{f}_{3D}=\mathbf{P}_{t+1}-\mathbf{P}_{t}, representing the per-point metric displacement. This explicit 4D mapping ensures that the generated trajectories are not merely visual hallucinations but correspond to physically plausible 3D movements.

Refinement and Visualization. To suppress artifacts at depth discontinuities, we apply a depth-gradient-based edge filter, masking out pixels where \|\nabla D\|>\tau. The resulting refined 3D trajectories are projected into a canonical bird’s-eye view (BEV). In our qualitative analysis (see the “3D Flow” panel in Fig. [1](https://arxiv.org/html/2607.06559#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")), these trajectories are rendered as colored trails over a depth-ordered point cloud backdrop, providing an intuitive verification of the model’s spatial-temporal coherence.

![Image 4: Refer to caption](https://arxiv.org/html/2607.06559v1/x6.png)

Figure 4: Overview of RynnWorld-4D. Our pipeline leverages the large-scale Rynn4DDataset 1.0 dataset to train a generative model capable of predicting future 4D sequences. Given a single RGB-D observation and a language instruction, RynnWorld-4D co-generates future RGB frames, depth maps, and optical flow. These predictive 4D representations are then aggregated by RynnWorld-4D-Policy to derive the final robot actions.

### 3.3 RynnWorld-4D

To achieve synchronized generation of RGB-DF sequences, we extend a pretrained video generative model into a tri-branch architecture (see the overview in Fig. [4](https://arxiv.org/html/2607.06559#S3.F4 "Figure 4 ‣ 3.2 3D Scene Reconstruction from Multi-Modal Videos ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). This representation is not merely a concatenation of channels; it admits a physically-grounded unprojection into 3D scene flow, as detailed in Sec. [3.2](https://arxiv.org/html/2607.06559#S3.SS2 "3.2 3D Scene Reconstruction from Multi-Modal Videos ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"). We denote the latents for modality m\in\{\text{rgb, depth, flow}\} as \bm{z}_{t}^{m}, where t\in[0,1] represents the flow-matching timestep. Each \bm{z}_{t}^{m}\in\mathbb{R}^{T\times C\times H\times W} encapsulates the entire temporal sequence of T frames.

Tri-branch Architecture. To inherit the powerful generative priors of the pretrained model while capturing the distinct characteristics of each modality, we expand the single-branch backbone of Wan (wang2025wan) into a tri-branch structure. This decoupled design allows each modality to model its unique feature distributions, such as complex textures for RGB, spatial geometry for depth, and motion displacements for flow to mitigate representation interference among divergent modalities.

Joint Cross-Modal Attention. To enforce cross-modal consistency, we introduce a Joint Cross-Modal Attention (JA) module that is inserted every _three_ transformer blocks across all 30 Wan-2.2 layers (at layers 0,3,6,\dots,27), yielding 10 JA modules in total. Each JA module is appended after the intra-modal self-attention of its host block.

Before cross-modal mixing, each branch m\in\{\text{rgb,depth,flow}\} receives a learnable modality embedding \bm{e}^{m}\in\mathbb{R}^{1\times 1\times d} (zero-initialized so the module starts as a pure residual) and is normalized by a per-modality LayerNorm \operatorname{LN}^{m} to align numerical scales across branches:

\tilde{\bm{z}}_{l}^{m}=\operatorname{LN}^{m}\bigl(\bm{z}_{l}^{m}+\bm{e}^{m}\bigr).(3)

Each branch m produces _one_ query and _one_ shared key/value pair that is reused by all other branches’ queries, reducing the parameter cost from 18d^{2} to 12d^{2} per block:

\bm{Q}_{l}^{m}=\operatorname{RMSNorm}_{q}\!\bigl(\operatorname{QProj}_{l}^{m}(\tilde{\bm{z}}_{l}^{m})\bigr),\quad[\bm{K}_{l}^{m},\bm{V}_{l}^{m}]=\operatorname{KVProj}_{l}^{m}(\tilde{\bm{z}}_{l}^{m}),\quad\bm{K}_{l}^{m}\!\leftarrow\!\operatorname{RMSNorm}_{k}(\bm{K}_{l}^{m}).(4)

Tokens are reshaped from [B,T{\cdot}S,d] to [B{\cdot}T,S,d] so that cross-modal attention is restricted to tokens of the _same temporal frame_ across modalities, and 3D Rotary Positional Embeddings are applied to \bm{Q}_{l}^{m} and \bm{K}_{l}^{m} to inject spatial position information consistently across branches. Each query attends only to the keys/values of the two complementary modalities:

\bm{A}_{l}^{m}=\operatorname{Attn}\bigl(\operatorname{RoPE}(\bm{Q}_{l}^{m}),\;\operatorname{RoPE}(\bm{K}_{l}^{\text{cross}}),\;\bm{V}_{l}^{\text{cross}}\bigr),(5)

with \bm{K}_{l}^{\text{cross}}=\operatorname{concat}(\{\bm{K}_{l}^{j}\}_{j\neq m}) and \bm{V}_{l}^{\text{cross}}=\operatorname{concat}(\{\bm{V}_{l}^{j}\}_{j\neq m}).

Instead of the double zero-initialization used in ControlNet—which we found to introduce a saddle-point deadlock—we combine a _zero-initialized_ output projection \operatorname{OutProj}_{l}^{m} with a learnable gate g_{l}^{m} initialized to 1:

\hat{\bm{z}}_{l}^{m}=\bm{z}_{l}^{m}+\tanh(g_{l}^{m})\cdot\operatorname{OutProj}_{l}^{m}(\bm{A}_{l}^{m}).(6)

At initialization \operatorname{OutProj}_{l}^{m}\equiv 0 guarantees a smooth warm start from the Stage-1 checkpoint, while \tanh(g_{l}^{m})=\tanh(1)\neq 0 ensures non-zero gradients flow into the gate so that it can decrease, increase, or change sign as training proceeds, preventing the joint pathway from being trapped at the origin.

Phased Training Strategy. To bridge the significant distribution gaps between modalities, we propose a phased training paradigm: Stage 1: Modality Adaptation. In this initial stage, we disable the Joint Cross-Modal Attention and train the three branches independently. This allows the depth and flow branches to effectively adapt to their respective geometric and kinetic distributions. Stage 2: Joint Attention Training. We insert Joint Cross-Modal Attention modules every three layers across all 30 transformer blocks. The entire backbone and per-branch self-attention/FFN are frozen; only the Joint Cross-Modal Attention projections, RMSNorms, per-modality LayerNorms, tanh gates, and the three modality embeddings are trainable. Joint Cross-Modal Attention uses 3D RoPE and a frame-wise mask so cross-modal attention stays within the same temporal frame. Stage 3: Full-Parameter Joint SFT. With the joint module already aligned, we unfreeze the entire model and continue on the full Rynn4DDataset 1.0.

Branch Dropout. In Stages 2 and 3, with probability p_{\text{drop}} we randomly select one of \{\text{depth},\text{flow}\} at each training step and replace its noisy latent (frames [1{:}]) with pure Gaussian noise, forcing the JA modules to reconstruct it from the visible modalities. The RGB branch is never dropped, since it serves as the appearance anchor: destroying it would leave the joint module with no consistent reference.

Training Objective. All three stages are optimized using the flow matching objective (lipman2022flow). For each modality m\in\mathcal{M}=\{\text{rgb},\text{depth},\text{flow}\}, we learn a velocity field \bm{v}_{\theta}^{m} that transports Gaussian noise \bm{\epsilon}^{m} to data \bm{z}_{0}^{m} along the path \bm{z}_{t}^{m}=(1-t)\bm{z}_{0}^{m}+t\bm{\epsilon}^{m}. The first frame of each modality is the clean image-to-video conditioning latent (a real RGB frame, a real depth frame, and a zero-flow frame, respectively) and is excluded from supervision; we use the slice [1{:}] to denote frames 1,\dots,T-1. The total loss is:

\mathcal{L}_{\text{total}}\;=\;\sum_{m\in\mathcal{M}}\lambda_{m}\,\mathbb{E}_{\bm{z}_{0}^{m},\bm{\epsilon}^{m},t,\bm{c}}\!\left[\bigl\|\bm{v}_{\theta}^{m}\!\bigl(\bm{z}_{t}^{m},t,\bm{c}\bigr)_{[1{:}]}-\bigl(\bm{\epsilon}^{m}-\bm{z}_{0}^{m}\bigr)_{[1{:}]}\bigr\|_{2}^{2}\right],(7)

where \bm{c} denotes the text prompt together with the initial RGB-D observation, and \bm{\epsilon}^{\text{rgb}}=\bm{\epsilon}^{\text{depth}}=\bm{\epsilon}^{\text{flow}} is a single Gaussian noise sample shared across the three modalities so that their denoising trajectories stay temporally aligned. The modality weights are \lambda_{\text{rgb}}=\lambda_{\text{depth}}=1 throughout, while \lambda_{\text{flow}}=0.5 in Stage 1 (the flow first frame carries no informative signal at warm-up) and \lambda_{\text{flow}}=1.0 in Stages 2 and 3.

### 3.4 RynnWorld-4D-Policy

We leverage RynnWorld-4D as a predictive 4D vision encoder. Given the current RGB-D observation and instruction, we perform a forward pass through the frozen RynnWorld-4D, which yields a latent trajectory encoding future dynamics—serving as a powerful representation for robotic manipulation. We extract intermediate hidden states across all branches and concatenate them along the channel dimension to form F_{p}\in\mathbb{R}^{B\times T\times 3C\times H\times W}, where C is the latent channel dimension per branch. By concatenating features from these decoupled branches, RynnWorld-4D-Policy benefits from specialized representations: the RGB branch provides rich visual context, while the independent depth and flow branches offer explicit geometric and kinetic cues, respectively.

Flow Former. To compress 4D features, we use a Flow Former with learnable queries \bm{Q}. It processes F_{p} via frame-wise spatial cross-attention followed by temporal self-attention:

\bm{Q}^{\prime}_{i}=\operatorname{Spat-CrossAttn}(\bm{Q}_{i},F_{p}[i]),\quad\bm{Q}^{\prime\prime}=\operatorname{FFN}(\operatorname{Temp-SelfAttn}(\bm{Q}^{\prime})),i\in\{1,\dots,T\}(8)

where i indexes the frame sequence, and \bm{Q}^{\prime\prime} encapsulate the predicted spatio-temporal dynamics.

Action Generation. We employ a flow matching (lipman2022flow) policy to generate actions, following the objective defined in Eq. [7](https://arxiv.org/html/2607.06559#S3.E7 "Equation 7 ‣ 3.3 RynnWorld-4D ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"). Here, the velocity field v_{\phi} operates on the action space \bm{a}, conditioned on the predictive 4D tokens \bm{Q}^{\prime\prime}, text embedding l_{\text{emb}}, and proprioception p_{0}. During inference, the action is derived via an ODE solver in N=4 steps, enabling high-frequency closed-loop control through parallel action chunking (see Sec. [3.5](https://arxiv.org/html/2607.06559#S3.SS5 "3.5 Inference Latency and Real-time Control ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation") for details).

### 3.5 Inference Latency and Real-time Control

To evaluate the feasibility of RynnWorld-4D-Policy in real-world scenarios, we conduct a detailed timing analysis of our inference pipeline. The model is deployed on a workstation equipped with an NVIDIA RTX 5090 GPU, leveraging FP8 quantization and FlashAttention 3 (FA3) kernels to accelerate the transformer-based 4D generation.

Note that the 4D visual features are extracted from the frozen RynnWorld-4D in a single forward pass (N=1). The subsequent 4-step ODE sampling for action generation occurs only within the lightweight RynnWorld-4D-policy head.

Latency Breakdown. The overall control frequency is determined by the total cycle time of the RynnWorld-4D forward pass and the action generation head. Given a sequence of K=10 actions generated per forward pass, a control frequency of approximately 9 Hz is achieved with a cycle time of \sim 1.1 s. Tab. [1](https://arxiv.org/html/2607.06559#S3.T1 "Table 1 ‣ 3.5 Inference Latency and Real-time Control ‣ 3 Method ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation") provides a granular breakdown of the time spent in each phase.

Table 1: Inference latency breakdown on NVIDIA RTX 5090 (FP8).

The primary bottleneck is the tri-branch Transformer, which accounts for 89.5% of the total latency.

Control Frequency. It is important to distinguish between the planning frequency (the rate at which the 4D world model refreshes its mental state) and the effective control frequency. Although a single forward pass takes \sim 1.1 s (yielding a planning frequency of \approx 0.9 Hz), the RynnWorld-4D-Policy employs action chunking by predicting K=10 future steps in a single inference. As these 10 actions are executed sequentially while the next planning cycle is computed in parallel, the system achieves an effective control frequency of \approx 9 Hz.

Closed-loop Robustness. While 9 Hz is lower than traditional low-level PID controllers (typically >500 Hz), RynnWorld-4D-Policy maintains high robustness through two mechanisms:

Instead of a single action, the policy outputs a sequence of K=10 future actions. During the \sim 1.1s inference cycle, the robot executes the previously planned action chunk at 50 Hz via a cached lookup. The 9 Hz update rate is sufficient to capture most human-scale manipulation dynamics.

Unlike 2D policies that suffer from visual aliasing or depth ambiguity, our policy’s internal latents are grounded in 3D scene flow. As shown in our ablation, the inclusion of explicit kinetic cues allows the policy to predict object movements. This anticipation compensates for the slight sensing-to-actuation lag, as the model is not just reacting to the current frame but is conditioned on a predicted 4D trajectory.

In real-world tests, we observe that even when objects are slightly bumped during the 1s execution window, the next 9 Hz update effectively re-plans the trajectory because the RynnWorld-4D latents encompass a spatial volume rather than just a pixel point, providing a wider capture range for recovery.

## 4 Experiments

### 4.1 Implementation Details

#### 4.1.1 RynnWorld-4D

Our RynnWorld-4D model is built upon the Wan 2.2-TI2V-5B diffusion transformer (wang2025wan), a 30-layer DiT with hidden dimension d=3072 and FFN dimension 14{,}336. We extend its native single-branch RGB backbone into a unified tri-branch architecture for the synchronous synthesis of RGB, depth, and optical flow sequences. The depth and flow branches are initialized by duplicating the pre-trained components—patch embeddings, self-attention, normalization layers, and FFNs—leveraging the robust spatial-temporal priors of the video backbone.

To ensure semantic alignment while minimizing overhead, we share the text cross-attention Key/Value projections across all three branches, as the linguistic task description provides a modality-agnostic semantic signal. We insert Joint Cross-Modal Attention (JA) modules every three transformer blocks across all 30 layers (at layers 0, 3, 6, …, 27), yielding 10 modules in total. For each branch m, the JA module queries the concatenated K/V pairs from the complementary modalities j\neq m:

\hat{\bm{z}}_{l}^{m}=\bm{z}_{l}^{m}+\tanh(g_{l}^{m})\cdot\operatorname{CrossBranchAttn}(\bm{Q}_{l}^{m},\bm{K}_{l}^{\text{cross}},\bm{V}_{l}^{\text{cross}}),(9)

where g_{l}^{m} is a learnable scalar gate initialized to 1. To ensure a smooth transition from independent branch training, we initialize the output projection to zero while keeping g_{l}^{m}=1. JA employs 3D RoPE and a frame-wise mask to restrict attention to tokens within the same temporal frame.

##### Phased Training Strategy.

We adopt a three-stage curriculum to bridge modality distribution gaps. Tab. [2](https://arxiv.org/html/2607.06559#S4.T2 "Table 2 ‣ Stage 3: Full-Parameter Joint SFT. ‣ 4.1.1 RynnWorld-4D ‣ 4.1 Implementation Details ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation") summarizes the stage-wise configuration. Each stage is initialized from the _model-only_ checkpoint of the previous stage (optimizer/scheduler reset) to ensure stability. We utilize the AdamW optimizer (\beta_{1}{=}0.9,\beta_{2}{=}0.95, weight decay 1{\times}10^{-4}) with cosine scheduling and linear warm-up. We track an Exponential Moving Average (EMA, decay 0.9999) with shadow weights on CPU for inference.

##### Stage 1: Modality Adaptation.

Branches are trained independently under modality-specific flow-matching supervision to repurpose RGB priors for geometric and kinetic modeling without gradient interference. We use a learning rate of 2{\times}10^{-5} with 500 warm-up steps and flow weight \lambda_{\text{flow}}{=}0.5.

##### Stage 2: Frozen-Backbone Joint Attention.

We freeze the backbone and instantiate the 10 JA modules. To preserve established representations, we employ zero-initialization for the output projections of the JA modules. The learning rate is set to 5{\times}10^{-5} with 200 warm-up steps. We apply Branch Dropout (p_{\text{drop}}{=}0.2) on \{\text{depth},\text{flow}\} to enhance cross-modal robustness.

##### Stage 3: Full-Parameter Joint SFT.

We unfreeze the entire model for joint fine-tuning on the Rynn4DDataset 1.0. We employ the learning rate as 1{\times}10^{-5} for all trainable parameters. Branch Dropout is reduced to p_{\text{drop}}{=}0.1.

Table 2: Stage-wise training configuration. Effective batch size is per-GPU batch (1) \times gradient accumulation (2-4) \times N_{\text{GPU}}.

##### Resource and Optimization.

All stages train at 81\times 480\times 640 (yielding T=21 latent frames under the causal VAE’s 4\times temporal compression, i.e., T_{\text{latent}}=(T_{\text{pixel}}-1)/4+1) with bf16 mixed precision and gradient checkpointing. Stages 2 and 3 leverage DeepSpeed ZeRO-2 with optimizer offload to manage memory for additional JA parameters.

![Image 5: Refer to caption](https://arxiv.org/html/2607.06559v1/x7.png)

Figure 5: Real-world Manipulation Benchmark. We establish a comprehensive evaluation suite comprising six diverse tasks to assess the model’s performance in open-world manipulation, providing a rigorous testbed for our 4D world model.

#### 4.1.2 RynnWorld-4D-Policy

We utilize RynnWorld-4D as a frozen 4D vision encoder. The model operates at 480\times 640 resolution, producing T=21 latent frames (after VAE temporal compression with a ratio of 4). At each inference step, we perform single-step feature extraction at diffusion timestep t=500, capturing intermediate hidden states from block 15 of the transformer. By concatenating the 3072-dimensional features from the RGB, depth, and optical flow branches, we obtain a comprehensive 4D representation F_{p}.

At each decision step, we take a single RGB observation frame as input. The condition frame is center-cropped to 480\times 640 and normalized to [-1,1]. The extracted spatiotemporal features are reshaped and fed into Flow Former. This compresses the high-dimensional RynnWorld-4D features into a fixed-size representation suitable for policy decoding.

We employ a flow matching policy head with 4-step Euler ODE sampling at inference time. Despite the multi-step sampling in the action space, the policy maintains high efficiency because the heavy 4D backbone features are only computed once. The policy predicts action chunks of length 10, where each action is 54-dimensional. During deployment, the model executes 10 actions open-loop before re-querying the visual backbone, yielding an effective control frequency of \sim 9 Hz.

We train with AdamW optimizer using a learning rate of 1\times 10^{-4}, \beta=(0.9,0.9), and weight decay 0.05. We employ a tri-stage learning rate schedule: 2% linear warmup, 8% constant hold, and 90% cosine decay to 10^{-6}\times the peak learning rate. The RynnWorld-4D backbone is kept frozen throughout training; only the Flow Former and flow matching policy head are optimized. Training uses mixed precision with a batch size of 1 per GPU for 100 epochs.

### 4.2 Setups and Baselines

World model metrics. To evaluate the generative performance and physical fidelity of our world model, we conduct benchmarks on a held-out test set of 50 video sequences, randomly sampled from (wu2024robomind; liu2024rdt; jiang2025galaxea). We assess the model across three axes:

*   •
(1) Visual Synthesis Quality: generative fidelity, following (zhou2025pai) (IQ, MS, SC, Subj.) and pixel-level alignment (PSNR, SSIM, LPIPS) with ground truth;

*   •
(2) Geometric Accuracy: evaluating structural integrity via depth estimation metrics (AbsRel, \delta_{1}<1.25);

*   •
(3) Temporal Motion Consistency: measuring dynamic precision through optical flow error (AEPE).

Detailed metric definitions are provided in Appendix [7](https://arxiv.org/html/2607.06559#S7 "7 Additional Details on Evaluation Metrics ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation").

Real-world task metric. The primary evaluation metric is the Success Rate, defined as the percentage of successful task completions over 35 consecutive real-world trials. A trial is considered successful if the robot completes the task within 120 seconds.

Hardware platform. For real-world data collection and policy evaluation, we utilize the TIANJI M6 robot equipped with a WUJI HAND. A RealSense D435i camera is integrated to capture first-person view (FPV) images. Please refer to our Appendix. [6](https://arxiv.org/html/2607.06559#S6 "6 Real Robot System Setup ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation") for more details.

Robot tasks. To demonstrate the generalization ability of our pipeline across diverse manipulation challenges, we evaluate our method on six distinct tasks that span varying levels of bimanual coordination, contact richness, and precision:

*   •
(1) Dual Picking: The robot uses its left arm to pick an apple and its right arm to pick a banana from a plate, sequentially placing both objects onto the tabletop.

*   •
(2) Block Pushing: A sequential pushing task where the left arm pushes a large block from the left zone to the center, after which the right arm takes over to push it to the designated right target zone.

*   •
(3) Hand-over: An intra-robot transfer task where the left gripper picks up a cabbage and hands it over to the right gripper, which then completes the placement.

*   •
(4) Bimanual Lifting: A heavy-load coordination task where both arms must synchronously lift a watermelon plush from the table and accurately place it into a tray.

*   •
(5) Lid Placement: A precision-oriented task requiring the robot to pick up a lid and accurately align it to cover a cardboard box.

*   •
(6) Bowl Stacking: There are two small bowls on the table; the task involves picking up one bowl and carefully stacking it on top of the other.

As shown in Fig. [5](https://arxiv.org/html/2607.06559#S4.F5 "Figure 5 ‣ Resource and Optimization. ‣ 4.1.1 RynnWorld-4D ‣ 4.1 Implementation Details ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"), these tasks collectively challenge the model’s proficiency in dual-arm synergy, temporal sequencing, and long-horizon interaction. Tab. [3](https://arxiv.org/html/2607.06559#S4.T3 "Table 3 ‣ 4.2 Setups and Baselines ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation") summarizes the per-task demonstration budget: the full corpus is used to train RynnWorld-4D, exposing it to a rich diversity of physical interactions, while a fixed budget of 200 episodes per task is allocated for training RynnWorld-4D-Policy. For each task, we evaluate the model’s generalization by applying significant randomization to the initial environment state. This includes varying the 6-DoF poses of task-relevant objects (e.g., fruits) in terms of both workspace coordinates and axial rotations.

Table 3: Statistics of the Real-World Dataset used for training.

![Image 6: Refer to caption](https://arxiv.org/html/2607.06559v1/x8.png)

Figure 6: Qualitative results of RynnWorld-4D. Starting from a single RGB-D image and a text prompt, RynnWorld-4D synchronously generates RGB video, depth maps, and optical flow, preserving sharp geometric structures and producing temporally consistent motion fields.

### 4.3 Results

4D World Modeling. As shown in Tab. [4](https://arxiv.org/html/2607.06559#S4.T4 "Table 4 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"), compared to state-of-the-art video generation models such as Wan (wang2025wan) and CogVideoX (yang2024cogvideox), RynnWorld-4D maintains highly competitive visual quality (IQ) while significantly outperforming them in reconstruction fidelity. This indicates that while general video models excel at creative synthesis, our model is better at preserving the structural and textural integrity of the scene during evolution, benefiting from the mutual regularization between depth and flow branches. When compared to existing 4D world models, RynnWorld-4D demonstrates a clear advantage in both spatial and temporal accuracy. In terms of geometry, our model achieves a \delta_{1} of 0.610, nearly doubling the performance of 4DNeX (chen20254dnex) (0.327) and TesserAct (zhen2025learning) (0.279). Regarding motion, RynnWorld-4D uniquely provides synchronized optical flow with a low AEPE of 0.170, whereas most baseline 4D models lack the capability to produce explicit motion fields.

As visualized in Fig. [6](https://arxiv.org/html/2607.06559#S4.F6 "Figure 6 ‣ 4.2 Setups and Baselines ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"), the generated depth and flow maps are not only internally consistent but also precisely aligned with the RGB texture changes. This result validates our core hypothesis: jointly modeling RGB, depth, and flow within a single diffusion loop acts as a powerful physical regularizer.

Table 4: Quantitative evaluation of 4D world modeling quality. Metrics span visual synthesis (RGB), geometric structure (Depth), and temporal dynamics (Optical Flow). N/A denotes that the baseline lacks the native capability to generate specific modalities.

Policy Learning. Tab. [5](https://arxiv.org/html/2607.06559#S4.T5 "Table 5 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation") summarizes the performance of RynnWorld-4D-Policy across various robotic manipulation tasks, where it consistently outperforms state-of-the-art baselines including Diffusion Policy (DP) (chi2025diffusion) and foundation models like \pi_{0}(black2024pi_0) and \pi_{0.5}(intelligence2025pi_).

Notably, in tasks requiring high spatial precision such as Lid Placement and Bowl Stacking, RynnWorld-4D-Policy achieves success rates of 65.71%, surpassing the next best baseline (DP) by 8.57%. Even more striking is the Hand-over task, a challenge involving dynamic object transfer where foundation models struggle. This performance gap stems from two fundamental limitations of current foundation models: first, their pre-training data is predominantly biased towards parallel-jaw grippers, lacking the inherent priors for the complex dexterous hand coordination. Second, in a hand-over scenario, 2D-based models struggle to reason about the relative 3D distance and potential self-occlusion between two high-DOF end-effectors. Furthermore, while these 2D policies must implicitly recover complex transfer dynamics from appearance residuals in the RGB stream, RynnWorld-4D-Policy benefits from explicit kinetic and geometric cues provided by the world model’s internal 4D latents, proving that predictive 4D representations are essential for tasks requiring precise temporal and spatial coordination where pure generative 2D priors fall short.

Table 5: Success rates (%) on robotic manipulation tasks. We compare RynnWorld-4D-Policy against state-of-the-art policy learning baselines and foundation models across six challenging tasks.

### 4.4 Ablation Study

We conduct extensive ablation studies to validate our architectural design and training strategies.

Effectiveness of Tri-branch Fusion. To verify the necessity of synchronized generation, we compare RynnWorld-4D with the Independent Branches baseline, where three diffusion branches are trained separately. As shown in Tab. [4](https://arxiv.org/html/2607.06559#S4.T4 "Table 4 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"), while independent branches can generate individual modalities, they suffer from significant performance degradation in depth (0.737 vs. 0.310 AbsRel) and flow (0.247 vs. 0.170 AEPE). This confirms that our mutual feature interaction mechanism is crucial for enforcing cross-modal consistency and physical fidelity in 4D sequences.

Necessity of Modality Adaptation. The comparison between the full model and the w/o MA baseline (which skips the initial modality-specific adaptation and proceeds directly to joint tri-branch training) highlights the efficacy of our phased training strategy. Without this first stage, the depth and flow branches struggle to bridge the gap between their inherited RGB priors and their specific modality distributions (Tab. [4](https://arxiv.org/html/2607.06559#S4.T4 "Table 4 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). This leads to a substantial drop in geometric accuracy (\delta_{1} decreases from 0.610 to 0.479) and compromised motion precision, proving that modality-specific adaptation is a prerequisite for effective multi-modal fusion.

Significance of Large-scale 4D Pre-training. We evaluate the necessity of Rynn4DDataset 1.0 by comparing our full model with the w/o 4D Pre-training variant, which is trained exclusively on a limited set of task-specific robotic manipulation data. Omitting the large-scale pre-training on Rynn4DDataset 1.0 leads to a severe performance collapse across all dimensions, with the AEPE surging from 0.170 to 0.729 (Tab. [4](https://arxiv.org/html/2607.06559#S4.T4 "Table 4 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). These results underscore that task-specific data alone lacks the diversity required to master complex spatio-temporal dynamics.

Effectiveness of Predictive Latents. To verify the importance of the RynnWorld-4D latent space, we compare our model against a baseline that replaces the predictive encoder with a standard ResNet-18 (he2016deep) image encoder (w/o RynnWorld-4D in Tab. [5](https://arxiv.org/html/2607.06559#S4.T5 "Table 5 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). The significant performance drop across all tasks—most notably in Dual Picking where success falls from 94.29% to 71.43%—highlights that static 2D features are insufficient for complex tasks. This confirms that the temporal and spatial dynamics captured in RynnWorld-4D’s predictive representations are crucial for robust policy learning.

Impact of 4D Modalities. We further analyze the individual contribution of each modality (Tab. [5](https://arxiv.org/html/2607.06559#S4.T5 "Table 5 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). Using only RGB latents (RGB) yields sub-optimal success rates as the policy lacks explicit structural grounding. Integrating depth (RGB + Depth) provides significant gains in tasks requiring spatial precision, such as Hand-over and Bimanual Lifting. Meanwhile, adding optical flow (RGB + Optical Flow) enhances motion-sensitive tasks. The full RynnWorld-4D-Policy, combining all three, achieves the best performance, confirming that the synergy of visual context, spatial geometry, and kinetic dynamics is essential for robust robotic manipulation.

Role of 3D RoPE in Joint Attention. We disable the 3D Rotary Positional Embedding (RoPE) inside the Joint Cross-Modal Attention modules (w/o RoPE in JA in Tab. [4](https://arxiv.org/html/2607.06559#S4.T4 "Table 4 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")) to decouple spatial coordinates from cross-modal interactions. Although intra-modal self-attention preserves local spatial structure, removing the shared positional bias in the cross-branch pathway disrupts the geometric correspondence between modality-specific features at identical (u,v) coordinates. This misalignment manifests as a significant decay in reconstruction fidelity, with \delta_{1} dropping from 0.610 to 0.450 and AEPE rising from 0.170 to 0.210. These results highlight that 3D RoPE serves as a critical alignment bridge, enabling the JA modules to achieve spatially-aware feature fusion at the pixel level rather than mere global semantic averaging.

Per-branch FFN vs. Shared FFN. By default, our architecture employs independent feed-forward networks (FFNs) for each branch, initialized from the pre-trained RGB backbone to preserve specialized representation manifolds. Replacing these with a single FFN shared across all modalities (Shared FFN in Tab. [4](https://arxiv.org/html/2607.06559#S4.T4 "Table 4 ‣ 4.3 Results ‣ 4 Experiments ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")) leads to a systemic performance collapse. This suggests that the latent spaces of RGB textures, depth manifolds, and motion fields are inherently heterogeneous; forcing them through a shared non-linear transformation induces catastrophic interference in feature representation. The resulting decline in geometric accuracy (AbsRel 0.580 / \delta_{1} 0.380) and motion stability (AEPE 0.280) empirically validates that modality-specific FFNs are essential to mitigate cross-modal distribution shifts and maintain high-fidelity 4D generation. Although this shared variant reduces parameter overhead, the substantial drop in generative fidelity underscores the necessity of modality-specific capacity in 4D world modeling.

## 5 Conclusion

In this paper, we presented RynnWorld-4D, a novel framework that shifts the paradigm of generative world modeling from 2D pixel sequences to consistent 4D scene evolution. By introducing a lightweight yet expressive representation consisting of synchronized RGB, depth, and optical flow (RGB-DF), we effectively bridge the gap between the scalability of video diffusion models and the geometric rigor required for robotic manipulation. Our core contributions include the development of a tri-branch architecture that ensures cross-modal consistency through mutual feature interactions, and the curation of Rynn4DDataset 1.0, the large-scale hybrid dataset specifically designed for training 4D generative models. Furthermore, we demonstrated that RynnWorld-4D-Policy can effectively leverage these predictive 4D representations as an implicit world model, enabling high-frequency, closed-loop robotic control that outperforms existing 2D-based baselines. Extensive experiments show that our approach not only generates high-fidelity 4D futures but also significantly enhances the success rate and precision of downstream manipulation tasks. We believe RynnWorld-4D provides a promising foundation for building general-purpose embodied intelligence capable of understanding and interacting with the complex 3D world.

Limitation. Despite the reactive capabilities of RynnWorld-4D-Policy, our framework has several limitations. First, the 4D sequence generation relies on a diffusion denoising process, which introduces computational overhead. Currently, our implementation achieves an effective control frequency of approximately 9 Hz on an NVIDIA RTX 5090 GPU; while sufficient for many tasks, this latency remains a bottleneck for ultra-high-frequency control. Second, our model is primarily optimized for egocentric perspectives. Extending 4D spatio-temporal consistency to multi-view systems or collaborative multi-robot setups remains an open challenge for future research.

## References

\beginappendix

## 6 Real Robot System Setup

Our real robot is built on the TIANJI M6 and WUJI Hand, as shown in Fig. [7](https://arxiv.org/html/2607.06559#S6.F7 "Figure 7 ‣ 6 Real Robot System Setup ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"). The policy’s inference frequency is set at 50 Hz. The commands are sent with a delay kept between 18 and 30 milliseconds. The low-level interface operates at a frequency of 500 Hz, ensuring smooth real-time control. The communication between the control policy and the low-level interface is realized through LCM (Lightweight Communications and Marshalling).

![Image 7: Refer to caption](https://arxiv.org/html/2607.06559v1/x9.png)

Figure 7: Standardized Experimental Platform. All real-world experiments are conducted on a unified hardware configuration consisting of the TIANJI M6 7-DOF robotic arm and the 20-DOF WUJI dexterous hand. This integrated system provides the high-precision control and high-dimensional actuation space required for complex manipulation tasks.

We collect real-world demonstration data through a teleoperation system, as shown in Fig. [8](https://arxiv.org/html/2607.06559#S6.F8 "Figure 8 ‣ 6 Real Robot System Setup ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation"). Our hardware setup consists of dual Tianji 7-DOF robotic arms and dual Wuji 20-DOF dexterous hands, yielding a total of 54 degrees of freedom.

For arm control, the human operator wears five HTC Vive trackers (mounted on the chest, both wrists, and both upper arms). The system computes wrist-to-chest relative transforms at 100-120 Hz and feeds them into a Pinocchio-based inverse kinematics solver running in a separate process. The resulting joint commands are further smoothed by a Ruckig trajectory generator with velocity, acceleration, and jerk constraints before being sent to the robot arms at 200 Hz.

For hand control, the operator wears Manus data gloves. The raw glove signals are converted to a 21-point MediaPipe hand skeleton format and retargeted to the 20-DOF Wuji hand joint space via a dedicated retargeting module, with an exponential moving average filter applied for motion smoothing.

![Image 8: Refer to caption](https://arxiv.org/html/2607.06559v1/x10.png)

Figure 8: Operator to collect real world data.

## 7 Additional Details on Evaluation Metrics

This section provides technical definitions and implementation details for all evaluation metrics used in our experiments.

### 7.1 Generative Quality

We adopt a subset of task-aligned metrics from PAI-Bench (zhou2025pai) to assess the visual and temporal quality of generated videos:

*   •
Imaging Quality (IQ): Evaluates low-level visual fidelity including clarity, noise levels, and compression artifacts. We employ the MUSIQ (ke2021musiq) predictor trained on the SPAQ dataset to compute a frame-level perceptual quality score.

*   •
Motion Smoothness (MS): Quantifies temporal coherence and physical plausibility of motion dynamics. It is computed as the reconstruction error between generated frames and those synthesized via the AMT-S (li2023amt) frame interpolation model, where lower interpolation error indicates smoother motion.

*   •
Subject Consistency (SC): Measures the identity stability of the primary subject across the video sequence. We compute the mean pairwise cosine similarity of DINO (zhang2022dino) ViT-B/16 features between the first frame and all subsequent frames.

*   •
I2V-Subject (Subj.): For image-to-video generation, this metric evaluates how faithfully the model preserves the identity of the input reference image throughout the generated sequence. It is computed as the DINO feature similarity between the conditioning image and each generated frame.

*   •
PSNR (Peak Signal-to-Noise Ratio): Measures pixel-wise reconstruction accuracy between generated and GT frames. Higher values indicate lower distortion. We report the mean PSNR across all frames (excluding the shared first frame) and all video samples.

*   •
SSIM (Structural Similarity Index): Evaluates the preservation of structural information by jointly considering luminance, contrast, and structural components between generated and GT frame pairs.

*   •
LPIPS (Learned Perceptual Image Patch Similarity): Measures perceptual distance between generated and GT frames using deep features from a pre-trained AlexNet. Lower values indicate higher perceptual similarity, and LPIPS is generally better aligned with human visual judgments than pixel-level metrics.

### 7.2 Geometric Accuracy

For methods that jointly predict depth maps, we evaluate structural fidelity of the estimated geometry using scale-invariant depth metrics:

*   •Absolute Relative Error (AbsRel \downarrow): Measures the mean relative deviation between the predicted depth \hat{d} and ground-truth depth d^{*}. Since different methods may produce depth in arbitrary scales, we first perform scale-invariant alignment via median scaling: s=\text{median}(d^{*})/\text{median}(\hat{d}), then compute:

\text{AbsRel}=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}\frac{|s\cdot\hat{d}_{p}-d^{*}_{p}|}{d^{*}_{p}},(10)

where \mathcal{V} denotes the set of valid pixels (d^{*}>0). 
*   •Threshold Accuracy (\boldsymbol{\delta_{1}<1.25}\uparrow): Reports the percentage of pixels whose depth ratio falls within a tolerance threshold:

\delta_{1}=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}\mathbb{1}\left[\max\left(\frac{s\cdot\hat{d}_{p}}{d^{*}_{p}},\;\frac{d^{*}_{p}}{s\cdot\hat{d}_{p}}\right)<1.25\right].(11)

Higher values indicate better geometric alignment with the ground truth. 

### 7.3 Motion Consistency

For methods that predict optical flow, we evaluate the temporal dynamics accuracy:

*   •Average End-Point Error (AEPE \downarrow): Measures the accuracy of predicted optical flow fields against ground-truth flow. For each pixel p with predicted flow \hat{\mathbf{f}}_{p}=(\hat{u}_{p},\hat{v}_{p}) and ground-truth flow \mathbf{f}^{*}_{p}=(u^{*}_{p},v^{*}_{p}), the end-point error is defined as:

\text{AEPE}=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}\left\|\hat{\mathbf{f}}_{p}-\mathbf{f}^{*}_{p}\right\|_{2}=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}\sqrt{(\hat{u}_{p}-u^{*}_{p})^{2}+(\hat{v}_{p}-v^{*}_{p})^{2}}.(12)

Since both predicted and ground-truth optical flow are stored as color-coded visualizations using the Middlebury color wheel encoding, we compute the AEPE as the per-pixel \ell_{2} distance in the normalized RGB color space between the predicted and ground-truth flow maps. The metric is averaged over all valid frames (excluding the first frame, which has no temporal reference) and all video samples. Lower AEPE indicates more accurate motion prediction. 

## 8 Preliminaries

Our study builds upon the Wan family of video generation models (wang2025wan), a latent video diffusion transformer capable of generating temporally coherent video from a single input image or text prompt. The model consists of a 3D variational autoencoder (\mathcal{E},\mathcal{D}) and a transformer-based diffusion model parameterized by \Theta. Given an input latent z_{0}=\mathcal{E}(V_{0}), the forward process follows the rectified flow formulation (esser2024scaling), where the noised latent is generated by linear interpolation:

z_{t}=(1-t)\,z_{0}+t\,\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)(13)

with timestep t\in[0,1]. The denoising process learns a velocity field v_{\Theta}(z_{t},t) that guides the transformation of noise back to data. The model is trained using a conditional flow matching (lipman2022flow), with objective:

\mathcal{L}_{\text{CFM}}=\mathbb{E}_{t,z_{0},\epsilon}\!\left[\left\lVert v_{\Theta}(z_{t},t)-u_{t}(z_{0}\mid\epsilon)\right\rVert_{2}^{2}\right](14)

where u_{t} is the target velocity derived analytically from the forward process. At inference, a sequence of latent frames is recovered by integrating v_{\Theta} over time.

In the image-to-video (I2V) setting, the model is conditioned on an initial image I_{0} encoded as z_{\text{img}}=\mathcal{E}(I_{0}). The transformer-based denoiser \mathcal{F}_{\Theta} autoregressively predicts video latents \{z^{(f)}\}_{f=1}^{F}, starting from z_{\text{img}} and producing temporally consistent sequences. In the text-to-video (T2V) setting, the model is instead conditioned on a text prompt p encoded as z_{\text{text}}=\mathcal{T}(p) and starts the autoregressive generation from noise. The final video is reconstructed as \hat{V}=\mathcal{D}(z^{(1)},\dots,z^{(F)}).

## 9 Baseline Implementation Details

Free4D. Free4D (liu2025free4d) lifts a single image into a dynamic 4D Gaussian Splatting (4DGS) representation. Given the first frame of each ground-truth video as input, we run the full Free4D pipeline: (1) the built-in ViewCrafter module with DUSt3R-based monocular depth estimation synthesizes 25 novel views from the input image via a video diffusion model at 576\times 1024 resolution; (2) COLMAP sparse reconstruction estimates camera poses and produces a sparse point cloud for Gaussian initialization; (3) the 4DGS model with HexPlane-based deformation fields is optimized for 10,000 iterations (3,000 static initialization + 7,000 joint optimization), with temporal resolution [64,64,64,150]; (4) RGB and depth videos are rendered from the original camera viewpoint across all timesteps. Since the rendered depth is in arbitrary scale, we apply per-frame median scaling alignment before computing depth metrics.

4DNeX. 4DNeX (chen20254dnex) is a feed-forward 4D scene generation framework that repurposes the Wan2.1-I2V-14B (wang2025wan), fine-tuned with learnable domain embeddings and LoRA adapters, to jointly produce RGB appearance and per-pixel XYZ point-cloud geometry from a single image and a text prompt. We adopt the official variant, using the provided 4dnex-lora weights (rank 64, fused at scale 0.5). For each sequence we extract the first frame of the ground-truth video as the conditioning image and use the corresponding caption, appending the official POINTMAP_STYLE. suffix as required by the released model. For each sample we run 50 denoising steps with classifier-free guidance scale 5.0 and seed 42, generating 49 frames at 24 fps. We obtain the depth video by taking the z-channel of the predicted pointmap, min–max normalized per sequence to [0,255].

TesserAct. TesserAct (zhen2025learning) is built upon CogVideoX-5b-I2V (yang2024cogvideox) and fine-tuned to jointly generate RGB, depth, and surface normal videos from a single initial frame and a language instruction. We adopt the official checkpoint. Since TesserAct requires depth and normal maps as additional conditioning inputs, we first extract the initial RGB frame from each ground-truth video, then apply network to estimate monocular depth and surface normals. The three modalities are concatenated along the channel dimension to form a 9-channel input. We generate 49 frames at 640\times 480 resolution using 50 DDPM denoising steps with guidance scale 7.5, image guidance scale 1.5. The model outputs RGB, depth, and normal videos concatenated along the width axis; we split along width to obtain separate RGB and depth predictions for evaluation.

## 10 Additional Qualitative Visualizations via RynnWorld-4D

To further showcase the generation quality of RynnWorld-4D, we provide extended paired visualizations of RGB, Depth, and Optical Flow generated via RynnWorld-4D (Fig. [9](https://arxiv.org/html/2607.06559#S10.F9 "Figure 9 ‣ 10 Additional Qualitative Visualizations via RynnWorld-4D ‣ RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation")). RynnWorld-4D synchronously predicts future RGB, depth maps, and optical flow from a single RGB-D observation. These results demonstrate:

(i) Cross-modal Consistency: The geometric structures in depth maps and motion boundaries in optical flow are precisely aligned with the RGB textures.

(ii) Physical Fidelity: The model accurately captures complex 4D dynamics, such as object displacements and multi-contact interactions, in both human-centric and robot-specific environments.

(iii) Temporal Coherence: The generated sequences maintain stability over time without significant flickering or structural morphing.

![Image 9: Refer to caption](https://arxiv.org/html/2607.06559v1/x11.png)

Figure 9: Extended qualitative results. Each row displays the generated RGB, depth, and optical flow sequences. The results highlight RynnWorld-4D’s ability to produce spatially and temporally coherent 4D predictions across various manipulation scenarios.

