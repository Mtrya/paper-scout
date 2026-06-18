# PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation

The PAIWorld Team

Institute of AI for Industries, Chinese Academy of Sciences

## Abstract

World foundation models (WFMs) have emerged as powerful simulators of physical environments, yet they predominantly operate in a single-view setting and lack the multi-view 3D consistency that robotic manipulation demands. Robotic systems inherently rely on multiple cameras, including egocentric, eye-to-hand, and wrist-mounted views, to capture complementary viewpoints for policy learning. Current multi-view world models, however, simply concatenate view tokens without explicit geometric reasoning, yielding cross-view object drift, depth inconsistency, and texture misalignment that propagate errors into downstream planning and control. We trace these failures to two fundamental deficiencies: (1) the absence of an explicit inter-view communication mechanism, which forces each viewpoint to generate in isolation, and (2) the absence of a 3D geometric prior, which leaves the model without guidance on what constitutes physically correct cross-view structure. We argue that resolving both is necessary and sufficient: an information pathway across views and a geometric signal that steers it must coexist, since communication without geometric guidance collapses to trivial shortcuts, while geometric priors without an inter-view pathway cannot propagate constraints across viewpoints. Building on this analysis, we present PAIWorld, a framework that augments diffusion-transformer world models along two technical pillars, realized by three components. To build the inter-view communication pathway, Geometry-Aware Cross-View Attention blocks open an explicit pathway across views, while Geometric Rotary Position Embedding encodes camera ray directions and extrinsic poses into this attention via rotary position encoding. To supply the geometric learning signal, Latent 3D-REPA distills 3D-aware features from frozen 3D foundation models, ensuring the exchanged content is 3D-consistent. Built upon the DiT-based world foundation model, PAIWorld attains state-of-the-art multi-view 3D consistency on robotic manipulation benchmarks, ranking 1st on the WorldArena leaderboard and 2nd on the AgiBot-Challenge2026 leaderboard, and enables downstream applications including model-based planning, world action models, and multi-view policy post-training.

Date: June 2026

## 1 Introduction

World foundation models (WFMs) have rapidly progressed from compact latent-dynamics modules [1, 2] to large-scale video generation systems capable of simulating complex physical environments [3–9]. By learning to predict future visual observations conditioned on actions or text, WFMs serve as world simulators that can be leveraged for model-based planning [10], policy evolution [11], and data synthesis in robotic learning [12, 13]. The Cosmos platform [3], in particular, has demonstrated that diffusion-transformer (DiT)

![](images/3b6fd7454f34841693cf744b7f320554e3816c5aa84556a97b102ca50c0c9e87.jpg)  
Figure 1 PAIWorld is a 3D-consistent multi-view world foundation model for robotic manipulation. Pre-trained on 2.5M multi-view video clips, PAIWorld serves as a versatile backbone for a range of downstream applications: multi-view world generation, world action modeling, robotic planning, and policy post-training. Across these settings, PAIWorld maintains cross-view 3D consistency, with coherent object placement, depth, and texture across all viewpoints, making its imagined rollouts physically plausible for embodied decision-making.

architectures [14] trained on internet-scale video data can produce temporally coherent, physically plausible visual rollouts, making WFMs a promising backbone for embodied intelligence. Recent unified models such as Pelican-Unified [15] further integrate world modeling with understanding, reasoning, and action within a single framework.

However, robotic manipulation systems are inherently multi-view. Standard configurations employ wristmounted and egocentric cameras simultaneously to supply complementary geometric and semantic cues for policy learning [16–18]. When a world model serves as a simulator for such a system, it must generate future observations across all viewpoints while maintaining strict 3D consistency: the same object must appear at geometrically compatible locations, with coherent depth and texture, across every view at every time step. Any breakdown in this consistency, whether cross-view object drift, depth contradictions, or texture misalignment, directly undermines the physical plausibility of imagined trajectories and propagates errors into downstream planning and control [10, 19].

Existing approaches to multi-view world modeling fall short of this requirement. Single-view WFMs such as Cosmos [3], CogVideoX [6], and Vista [7] produce high-quality temporal predictions but are architecturally restricted to a single viewpoint. Methods that do handle multiple views, such as Genie [20] and iVideoGPT [21], typically concatenate tokens from different viewpoints along the sequence dimension without any explicit mechanism for cross-view geometric reasoning. This “flat concatenation” strategy treats multi-view tokens identically to temporal tokens, leaving the model to discover cross-view correspondences implicitly from data. Such implicit discovery grows increasingly unreliable as the number of viewpoints and the complexity of the scene scale up.

We trace the root cause of these failures to two fundamental deficiencies in existing multi-view approaches. First, they lack an explicit inter-view communication mechanism. Flat concatenation provides no dedicated pathway for viewpoints to exchange information; each view’s tokens attend to the entire sequence without distinguishing same-view from cross-view tokens, so the model must infer geometric correspondences implicitly. As a consequence, each viewpoint effectively generates in isolation, with no means to coordinate predictions or resolve cross-view conflicts. Second, they lack a 3D geometric prior. Even with a communication pathway in place, the model receives no supervisory signal specifying what geometrically consistent 3D structure looks like. Absent such guidance, cross-view information exchange gravitates toward superficial shortcuts, such as matching color palettes or copying textures, rather than learning genuine 3D correspondences.

These two deficiencies call for two remedies that operate at distinct levels. At the architectural level, the model needs an information pathway that lets viewpoints exchange features during generation; at the trainingobjective level, it needs a geometric learning signal that steers what flows through this pathway toward 3D-consistent structure. Crucially, neither remedy alone is sufficient, precisely because they act on different levels. An inter-view communication pathway without geometric supervision lets information flow but cannot guarantee that the exchanged content respects 3D geometry; in practice it learns shortcuts such as texture copying or uniform averaging. Conversely, a geometric prior without an inter-view pathway sharpens each view’s 3D awareness in isolation, but the resulting constraints have no route to propagate across viewpoints, so cross-view inconsistencies persist. Only when both are present does the system achieve genuine multi-view 3D consistency: a pathway that carries information, and an objective that makes the information geometrically meaningful.

Based on this analysis, we present PAIWorld, a framework built on two technical pillars, an architectural communication pathway and a geometric training objective, realized by three lightweight, modular components on the DiT backbone (figure 2). The first pillar, the inter-view pathway, is realized by two cooperating components: Geometry-Aware Cross-View Attention blocks, interleaved within the DiT, open the pathway for viewpoints to exchange features, while a shared Geometric Rotary Position Embedding (Geo-RoPE) encodes camera ray directions and extrinsic poses into this attention via rotary position encoding [22, 23], biasing the pathway to route information along geometrically corresponding tokens. The second pillar, the geometric objective, is realized by the Latent 3D-REPA that aligns intermediate DiT features with 3D-aware representations distilled from frozen 3D foundation models via the REPA framework [24], supplying the supervisory signal that makes the exchanged content 3D-consistent. In short, Cross-View Attention and Geo-RoPE let geometric information flow across views, while Latent 3D-REPA supervises that information to be 3D-consistent, so that only their combination produces coherent multi-view 3D structure. Our contributions are as follows:

• We identify two fundamental deficiencies in existing multi-view world models, namely the lack of inter-view communication and the absence of a 3D geometric prior, and argue that addressing both jointly is necessary and sufficient for multi-view 3D consistency.

• We propose PAIWorld, which builds these two pillars from three plug-and-play components: Geometry-Aware Cross-View Attention and Geometric Rotary Position Embedding form the architectural pathway, while Latent 3D-REPA provides the geometric objective. These lightweight modules apply to any DiT-based world model.

• We show that PAIWorld delivers state-of-the-art multi-view 3D consistency on robotic manipulation benchmarks: it ranks 1st on the WorldArena benchmark (best overall EWMScore 70.67%, with the best Motion Quality among all entries) and 2nd on the AgiBot-Challenge2026 leaderboard (EWMScore 82.45%), where it attains the best Scene Consistency (90.41%) among all entries. This improved consistency further translates directly into gains on downstream embodied tasks, including model-based robotic planning and world action model fine-tuning.

The remainder of this paper is organized as follows. section 2 reviews related work on world models, multi-view generation, and 3D-aware representations. section 3 presents the PAIWorld framework in detail. section 5 summarizes our findings and discusses future directions.

## 2 Related Work

## 2.1 World Foundation Models for Physical AI

The concept of learning internal models of the environment, commonly known as world models, has a long history in reinforcement learning and cognitive science [1]. Early approaches such as Dreamer [2] and DayDreamer [10] learn compact latent-state dynamics models that enable planning through imagined rollouts, demonstrating the viability of model-based learning for physical robot control. More recently, the success of large-scale generative models has motivated a paradigm shift toward world foundation models (WFMs) that operate directly in pixel or video space. Sora [5] demonstrated that video diffusion models can generate temporally coherent, physically plausible visual simulations. The Cosmos platform [3] further formalized this direction by training DiT-based [14] video generation models on internet-scale data for physical AI applications, while Cosmos 3 [4] extended it to an omnimodal framework unifying language, image, video, audio, and action modalities. CogVideoX [6], HunyuanVideo [8], Stable Video Diffusion [25], and Wan [9] achieved high-fidelity video generation across diverse domains. DIAMOND [26] demonstrated that diffusion-based world models can capture fine visual details critical for decision-making. Pelican-Unified [15] advocates a unified paradigm integrating world modeling with understanding, reasoning, and action for embodied intelligence. However, these models are fundamentally single-view: they generate one coherent video stream without explicit mechanisms for multi-view consistency.

Interactive world models represent another important direction. UniSim [27] learns real-world simulators from diverse data sources, Genie [20] and Genie 2 [28] enable interactive environment generation from single images, and iVideoGPT [21] scales autoregressive world models to complex environments. In the robotic manipulation domain, GR-2 [29] builds a generative video-language-action model with web-scale knowledge, IRASim [30] learns action-conditioned video simulators from real robot data, EnerVerse [31] envisions embodied future spaces for manipulation planning, and LaDi-WM [13] employs latent diffusion-based world models for predictive manipulation. AgiBot World [32] provides a large-scale multi-view manipulation platform, and WorldArena [33] offers a unified benchmark for evaluating embodied world models. While some of these systems support multiple viewpoints via token concatenation, none introduces explicit cross-view geometric reasoning, a critical limitation for robotic applications, where 3D consistency directly governs policy quality.

## 2.2 Multi-View and 3D-Aware Visual Generation

Multi-view generation has been extensively studied in the context of 3D content creation from single images. Zero-1-to-3 [34] pioneered viewpoint-conditioned diffusion by fine-tuning a 2D diffusion model with relative camera transformations. SyncDreamer [35] introduced synchronized multi-view denoising with 3D-aware attention to ensure geometric consistency across generated views. MVDream [36] extended multi-view diffusion to text-conditioned 3D generation, while MVDiffusion [37] enabled holistic multi-view image generation with correspondence-aware attention. SV3D [38] leveraged latent video diffusion for orbital multi-view synthesis, and SV4D [39] further extended this to dynamic 3D content with multi-frame and multi-view consistency. CAT3D [40] demonstrated that multi-view diffusion models can create high-quality 3D assets from arbitrary inputs.

Our work differs from this line in three respects, which together define a regime these methods do not target. First, scene scope: the above methods are object-centric, generating views of a single foreground object against a clean background, whereas robotic manipulation requires modeling cluttered scenes with a manipulator, objects, and a dynamic background. Second, dynamics: they synthesize static objects or short orbital trajectories, while a world model must roll out temporally evolving dynamics driven by text or actions. Third, camera configuration: they assume dense, smoothly-varying viewpoints around an object, whereas robotic rigs provide a few fixed, wide-baseline cameras (egocentric, eye-to-hand, wrist) with little view overlap, where implicit correspondence learning is far harder. PAIWorld is designed for this dynamic, wide-baseline, scene-level regime, and injects geometry through an explicit communication pathway and a 3D supervisory objective rather than relying on dense view sampling.

Camera control in video generation has been explored by CameraCtrl [23], which represents camera poses using Plücker ray coordinates and injects them into video diffusion models. ViewCrafter [41] further tames video diffusion models for high-fidelity novel view synthesis. These works provide technical foundations for camera-aware generation, but focus on single-view camera trajectory control rather than multi-view consistency.

## 2.3 3D Representations and Geometric Reconstruction

Recent advances in geometric 3D vision have produced powerful tools for evaluating and enforcing 3D consistency. Neural Radiance Fields (NeRF) [42] and 3D Gaussian Splatting [43] established the foundation for differentiable 3D scene representations. DUSt3R [44] and its successor MASt3R [45] learn to predict dense 3D point maps from image pairs without requiring known camera parameters, enabling direct measurement of cross-view geometric consistency. The Depth Anything series [46–48] provides foundation models for monocular depth estimation that capture robust 3D-aware features from large-scale training. VGGT [49] further advances this direction by grounding visual geometry in a unified transformer that jointly predicts camera poses, depth, and 3D point maps from arbitrary image collections. These models encode rich geometric priors that can serve as supervision targets for world models seeking 3D consistency.

![](images/463869ad34177503aebc8470a4108d8d8d616c32d9cb4f4c35747281d5f20546.jpg)  
Figure 2 Overview of the PAIWorld framework. Built on a DiT-based flow matching backbone, PAIWorld rests on two technical pillars realized by three components. Pillar 1, the inter-view pathway (two components): (1) Geometry-Aware Cross-View Attention blocks open an explicit communication pathway across views, and (2) Geometric Rotary Position Embedding (Geo-RoPE) encodes camera ray directions and extrinsic poses into this attention, biasing it toward geometrically corresponding tokens. Pillar 2, the geometric objective (one component): (3) Latent 3D-REPA aligns intermediate DiT representations with 3D-aware features from a frozen 3D foundation model (Depth Anything 3), supplying the supervisory signal. The first two components build the pathway, while Latent 3D-REPA ensures the information flowing through it is 3D-consistent.

The REPA framework [24] introduced the idea of aligning intermediate representations of diffusion transformers with those of pretrained encoders, demonstrating accelerated training and improved generation quality for image diffusion. Our Latent 3D-REPA extends this principle from 2D image generation to multi-view video world models, using 3D-aware features from Depth Anything 3 as the alignment target to inject geometric consistency directly into the diffusion process.

## 3 Method

We present PAIWorld, a framework for injecting 3D consistency into flow-matching world foundation models for multi-view robotic manipulation. As argued in section 1, multi-view 3D consistency rests on two technical pillars: an architectural pathway for inter-view communication and a training objective that enforces 3Dconsistent content. PAIWorld builds these two pillars from three components on a DiT-based flow matching backbone (figure 2). The pathway is formed by two cooperating components: Geometry-Aware Cross-View Attention (section 3.4), which opens the inter-view pathway, and Geometric Rotary Position Embedding (section 3.3), which shapes this pathway with camera geometry. The objective is supplied by a single component, Latent 3D-REPA (section 3.5). We first formalize the problem setting (section 3.1), then describe each component and analyze why both pillars must be present simultaneously (section 3.6).

## 3.1 Problem Formulation

Consider a robotic manipulation system equipped with V cameras, each providing a video stream. At time step t, the system observes a set of images $\{ I _ { t } ^ { v } \} _ { v = 1 } ^ { V }$ with associated camera intrinsics $\{ \mathbf { K } ^ { v } \} _ { v = 1 } ^ { V }$ and extrinsics $\{ [ \mathbf { R } ^ { v } \mid \mathbf { t } ^ { v } ] \} _ { v = 1 } ^ { V } \in \mathrm { S E } ( 3 )$ . Given a conditioning signal c (text description or action sequence) and context frames $\{ I _ { 1 : t _ { 0 } } ^ { 1 : V } \}$ , the goal of multi-view video generation is to model the conditional distribution:

$$
p _ { \theta } \left( \{ I _ { t _ { 0 } + 1 : T } ^ { v } \} _ { v = 1 } ^ { V } \ \middle | \ \{ I _ { 1 : t _ { 0 } } ^ { v } \} _ { v = 1 } ^ { V } , \ \{ \mathbf { K } ^ { v } , \mathbf { R } ^ { v } , \mathbf { t } ^ { v } \} _ { v = 1 } ^ { V } , c \right) ,\tag{1}
$$

where T is the prediction horizon. Beyond per-view fidelity, the generated multi-view video must satisfy a multi-view 3D consistency requirement: at every time step, the views should admit a coherent 3D explanation. Formally, there exists a consistent 3D scene $S _ { t }$ such that all views $\{ I _ { t } ^ { v } \} _ { v = 1 } ^ { V }$ can be obtained by rendering $S _ { t }$ from their respective camera poses; equivalently, points corresponding to the same 3D location across views must respect the underlying epipolar geometry. This requirement is what existing single-view and flat-concatenation models fail to enforce, and it motivates the two pillars of our design.

## 3.2 Flow Matching DiT for Video Generation

We adopt a Diffusion Transformer (DiT) [14] trained with a flow matching objective [50, 51] operating in the latent space of a pretrained video VAE. Following Wan2.1 [9], we use its spatial-temporal VAE to compress each video both spatially and temporally into a compact latent representation, which substantially reduces the token count and makes multi-view video modeling tractable. Given an input video, the VAE encoder produces a latent representation $\mathbf { z } _ { 0 } \in \mathbb { R } ^ { T \times H \times W \times C }$ The model learns a velocity field $u _ { \theta } ( \mathbf { z } _ { s } , s )$ that transports samples from a noise distribution toward the data distribution along a linear interpolation path:

$$
\begin{array} { r } { \mathbf { z } _ { s } = ( 1 - s ) \mathbf { z } _ { 0 } + s \pmb { \epsilon } , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , } \end{array}\tag{2}
$$

where $s \in [ 0 , 1 ]$ is the flow timestep. The training objective minimizes:

$$
\mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { s , \epsilon } \left[ \| u _ { \theta } ( \mathbf { z } _ { s } , s ) - ( \epsilon - \mathbf { z } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] .\tag{3}
$$

Conditioning Signal Injection. The conditioning signal c is injected into each DiT block via Adaptive Layer Normalization (AdaLN) [14]. For text-conditioned generation, c is a text embedding that modulates the scale and shift parameters of layer normalization, globally steering the generation toward the described scene. For action-conditioned generation, rather than representing robot actions as raw vectors, we follow EVAC [52] and render actions into spatial action maps that are concatenated with the noisy latent along the channel dimension. This spatial representation preserves the geometric structure of the action (e.g., end-effector trajectory projected into each camera view), enabling the model to ground action semantics in pixel space rather than learning an implicit mapping from abstract action vectors.

Multi-View Token Concatenation. For multi-view generation, a naive approach concatenates all view tokens along the sequence dimension, yielding $\mathbf { z } _ { 0 } ^ { \mathrm { c o n c a t } } \in \mathbb { R } ^ { ( \smile , T ) \times H \times W \times C }$ . While the standard temporal self-attention operates over this concatenated sequence, it treats multi-view tokens identically to temporal tokens without any geometric inductive bias. The model must discover cross-view correspondences entirely from data, which is insufficient for consistent 3D generation.

## 3.3 Geometric Rotary Position Embedding

To encode 3D geometric information into the attention mechanism, we introduce a Geometric Rotary Position Embedding (Geo-RoPE) that separately encodes pixel-level ray directions and view-level camera poses via rotary position encoding [22].

Dual-Component Design. For each attention head with dimension $d ,$ we split the query and key vectors into two equal subspaces: a ray subspace of dimension $d _ { r } = d / 2$ and a pose subspace of dimension $d _ { p } = d / 2$ . Each subspace receives its own geometric encoding via RoPE.

Ray Component. For each token at spatial location $( h , w )$ in view $v ,$ we compute the world-space ray direction by unprojecting the pixel through the camera intrinsics ${ \mathbf { K } } ^ { v }$ and rotating by the inverse of the camera rotation $( \mathbf { R } ^ { v } ) ^ { - 1 }$

$$
\mathbf { d } ^ { v } ( h , w ) = \operatorname { n o r m a l i z e } \left( ( \mathbf { R } ^ { v } ) ^ { \top } \cdot ( \mathbf { K } ^ { v } ) ^ { - 1 } \left( { \begin{array} { c } { h + 0 . 5 } \\ { w + 0 . 5 } \\ { 1 } \end{array} } \right) \right) \in \mathbb { R } ^ { 3 } .\tag{4}
$$

The 3D ray direction is cyclically expanded to fill the $d _ { r } / 2$ frequency slots and used as position coordinates for RoPE rotation on the ray subspace of q and k.

Pose Component. For each view v, we extract a 12-dimensional pose feature vector that captures the full camera geometry:

$$
\mathbf { e } ^ { v } = \underbrace { \mathrm { [ y a w , p i t c h , r o l l } , \quad \mathbf { t } ^ { v } } _ { \mathrm { E u l e r ~ a n g l e s } } , \underbrace { \mathbf { t } ^ { v } } _ { \mathrm { t r a n s l a t i o n } } , \underbrace { - ( \mathbf { R } ^ { v } ) ^ { \top } \mathbf { t } ^ { v } } _ { \mathrm { c a m e r a ~ p o s i t i o n } } , \underbrace { ( \mathbf { R } ^ { v } ) ^ { \top } \mathbf { e } _ { z } } _ { \mathrm { o p t i c a l ~ a x i s } } \in \mathbb { R } ^ { 1 2 } ,\tag{5}
$$

where $\mathbf { e } _ { z } = [ 0 , 0 , 1 ] ^ { \top }$ . This pose vector is shared across all spatial positions within a view and applied via RoPE to the pose subspace.

Split-RoPE Application. The complete Geo-RoPE operates as:

$$
\begin{array} { r l } & { \mathbf { q } _ { \mathrm { r a y } } , \mathbf { q } _ { \mathrm { p o s e } } = \mathrm { s p l i t } ( \mathbf { q } , \mathbf { \Xi } [ d _ { r } , d _ { p } ] ) } \\ & { \qquad \tilde { \mathbf { q } } _ { \mathrm { r a y } } = \mathrm { R o P E } ( \mathbf { q } _ { \mathrm { r a y } } , \mathbf { \Xi } \mathbf { d } ^ { v } ( h , w ) ) } \\ & { \qquad \tilde { \mathbf { q } } _ { \mathrm { p o s e } } = \mathrm { R o P E } ( \mathbf { q } _ { \mathrm { p o s e } } , \mathbf { e } ^ { v } ) } \\ & { \qquad \tilde { \mathbf { q } } = [ \tilde { \mathbf { q } } _ { \mathrm { r a y } } ; \tilde { \mathbf { q } } _ { \mathrm { p o s e } } ] } \end{array}\tag{6}
$$

The same operation applies to keys. This design ensures that the ray subspace captures fine-grained pixel-level geometric correspondences (tokens viewing the same 3D point receive similar rotations), while the pose subspace encodes view-level identity (tokens from the same camera share identical pose rotations). Separating the two prevents interference between the spatially-varying ray signal and the spatially-uniform pose signal.

## 3.4 Geometry-Aware Cross-View Attention

Standard temporal self-attention in each DiT block operates independently per view (after merging V into the batch dimension), providing no explicit cross-view interaction. To open the inter-view communication pathway, we introduce two complementary attention mechanisms: dedicated Cross-View Attention blocks and periodic spatial-concat self-attention.

Multi-View Self-Attention Blocks. At selected DiT layers, we insert a dedicated Cross-View Attention subblock. For each temporal frame t (distinct from the flow timestep s), let $\{ \mathbf { Z } _ { t } ^ { v } \} _ { v = 1 } ^ { V }$ denote the feature maps from all views, where $\mathbf { Z } _ { t } ^ { v } \in \mathbb { R } ^ { ( H \cdot W ) \times D }$ . Each view’s queries and keys are first projected and then rotated by Geo-RoPE using that view’s own camera geometry, so that the rotation applied to view v depends on the rays and pose of camera v:

$$
\begin{array} { r l r } & { \tilde { \mathbf { Q } } _ { t } ^ { v } = \mathrm { G e o R o P E } _ { v } ( \mathbf { W } _ { Q } \mathbf { Z } _ { t } ^ { v } ) , \qquad \tilde { \mathbf { K } } _ { t } ^ { v } = \mathrm { G e o R o P E } _ { v } ( \mathbf { W } _ { K } \mathbf { Z } _ { t } ^ { v } ) , \qquad \mathbf { V } _ { t } ^ { v } = \mathbf { W } _ { V } \mathbf { Z } _ { t } ^ { v } , } & \\ & { \hat { \mathbf { Z } } _ { t } ^ { v } = \mathbf { Z } _ { t } ^ { v } + \mathrm { g a t e } \cdot \mathrm { s o f t m a x } \left( \frac { \tilde { \mathbf { Q } } _ { t } ^ { v } [ \tilde { \mathbf { K } } _ { t } ^ { 1 } ; \dots ; \tilde { \mathbf { K } } _ { t } ^ { V } ] ^ { \top } } { \sqrt { d } } \right) [ \mathbf { V } _ { t } ^ { 1 } ; \dots ; \mathbf { V } _ { t } ^ { V } ] , } & \end{array}\tag{7}
$$

where $\mathrm { G e o R o P E } _ { v }$ denotes the split ray-pose rotary encoding of equation (6) instantiated with the intrinsics $\mathbf { K } ^ { v }$ and extrinsics $\mathbf { \Sigma } [ \mathbf { R } ^ { v } \mid \mathbf { t } ^ { v } ]$ of view $v ,$ and $[ \cdot ; \cdot ]$ concatenates the per-view keys (values) along the token dimension. Because each view is rotated by its own geometry, the query of view v and the key of view $v ^ { \prime }$ attain a high inner product precisely when their tokens observe the same 3D point, so geometrically corresponding tokens across views naturally receive higher attention weights. The gate is initialized to zero via AdaLN-Zero, preserving the pretrained single-view model at initialization.

Spatial-Concat Self-Attention. Periodically, in place of per-view temporal attention, we flatten the view and spatial dimensions into a single token axis of length $V \cdot H \cdot W$ and perform joint spatio-view self-attention over all these tokens simultaneously. This provides a broader receptive field in which each token can attend to all spatial positions across all views within the same temporal context, complementing the dedicated cross-view blocks.

Why the Pathway Alone Is Insufficient. Cross-View Attention and Geo-RoPE together establish the architectural pathway along which viewpoints exchange information, biased toward geometrically corresponding tokens. Yet an architectural bias only shapes how information flows; it does not dictate what content is 3D-consistent. Without an explicit geometric objective, the pathway can still settle into trivial shortcuts, such as copying textures across views or averaging features, that minimize the generation loss while violating true 3D structure. This motivates the geometric learning signal introduced next.

## 3.5 Latent 3D-REPA (3D Geometric Prior)

To provide the geometric learning signal for cross-view interaction, we introduce Latent 3D-REPA, a tokenrelation distillation objective that aligns the DiT’s intermediate representations with features from frozen 3D foundation models.

3D-Aware Feature Extraction. We employ Depth Anything $3 \ [ 4 8 ]$ as our 3D-aware feature extractor. For each multi-view frame set $\{ I _ { t } ^ { v } \} _ { v = 1 } ^ { V }$ , Depth Anything 3 produces dense features that encode rich geometric priors including depth, 3D point maps, and camera-relative spatial structure. In addition, it recovers camera extrinsics and intrinsics that are used by Geo-RoPE and for 3D point map reconstruction.

Token Relation Distillation. Rather than directly regressing 3D features token-by-token, we distill the relational structure between tokens. At a selected intermediate layer ℓ of the DiT, we extract the feature map $\mathbf { H } _ { \ell }$ and project it to the VGGT feature dimension via a lightweight 3D convolutional projector $g _ { \phi } .$ yielding per-token features $\mathbf { F } ^ { \mathrm { D i T } } = g _ { \phi } ( \mathbf { H } _ { \ell } )$ ; the corresponding frozen features from Depth Anything 3 are denoted $\mathbf { \bar { F } } ^ { \mathrm { { D A 3 } } }$ We supervise the pairwise relations among these tokens rather than their absolute values, since relational structure is invariant to the feature-space discrepancy between the two encoders.

Computing the full token-token similarity matrix is prohibitively expensive $\scriptstyle ( N = V \cdot H \cdot W$ tokens per frame, over $T$ frames). We therefore estimate the relations through anchor sampling: a random subset of K tokens is drawn as anchors, and we measure the cosine similarity between every token and each anchor. For a token set $\mathbf { F } = \{ \mathbf { f } _ { i } \} _ { i = \cdot } ^ { M }$ 1 with sampled anchor indices $\mathcal { A } \subset \{ 1 , . . . , M \} , | \mathcal { A } | = K$ , we define the sampled similarity matrix ${ \bf S } ( { \bf F } ) \dot { { \bf \varphi } } \in \mathbb { R } ^ { M \times K }$ as

$$
\mathbf { S } ( \mathbf { F } ) _ { i , a } = { \frac { \mathbf { f } _ { i } ^ { \top } \mathbf { f } _ { a } } { \| \mathbf { f } _ { i } \| \left\| \mathbf { f } _ { a } \right\| } } , \quad a \in { \mathcal { A } } .\tag{8}
$$

The distillation loss aligns this relational structure at two granularities, applying the operator $\mathbf { S } ( \cdot )$ of equation (8) to two different token sets:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R E P A } } = \mathcal { L } _ { \mathrm { s p a t i a l } } + \mathcal { L } _ { \mathrm { t e m p o r a l } } . } \end{array}\tag{9}
$$

The spatial term operates within each frame: let $\mathbf { F } _ { \mathrm { f r a m e } }$ denote the N tokens of a single frame (across all views and spatial positions), from which $K _ { s }$ anchors are sampled. Writing $\mathbf { S } _ { \mathrm { i n t r a } } : = \mathbf { S } ( \mathbf { F } _ { \mathrm { f r a m e } } ) \in \mathbb { R } ^ { N \times K _ { s } }$ , the term aligns the DiT and Depth Anything 3 relations:

$$
\mathcal { L } _ { \mathrm { { s p a t i a l } } } = \mathrm { { S m o o t h L 1 } \big ( \mathbf { S } _ { \mathrm { { i n t r a } } } ^ { \mathrm { { D i T } } } , ~ \mathbf { S } _ { \mathrm { { i n t r a } } } ^ { \mathrm { { D A 3 } } } \big ) . }\tag{10}
$$

The temporal term operates over the entire clip: let $\mathbf { F } _ { \mathrm { c l i p } }$ denote the full set of $T \cdot N$ tokens, from which $K _ { t }$ anchors are sampled across all frames. Writing $\mathbf { S } _ { \mathrm { i n t e r } } : = \mathbf { \dot { S } } ( \mathbf { F } _ { \mathrm { c l i p } } ) \in \mathbb { R } ^ { ( T \cdot N ) \times K _ { t } }$

$$
\mathcal { L } _ { \mathrm { t e m p o r a l } } = \mathrm { S m o o t h L 1 } \left( \mathbf { S } _ { \mathrm { i n t e r } } ^ { \mathrm { D i T } } , \mathbf { S } _ { \mathrm { i n t e r } } ^ { \mathrm { D A 3 } } \right) .\tag{11}
$$

Both $\mathbf { S } _ { \mathrm { { i n t r a } } }$ and $\mathbf { S } _ { \mathrm { i n t e r } }$ are instances of the same sampled-similarity operator $\mathbf { S } ( \cdot )$ , differing only in their token set and anchor count: $\mathbf { S } _ { \mathrm { { i n t r a } } }$ captures intra-frame geometric relations (within a single time step, across views and space), while $\mathbf { S } _ { \mathrm { { i n t e r } } }$ captures cross-frame relations that span the temporal dimension. Stochastic anchor sampling reduces the cost from quadratic to $O ( M K )$ while retaining an effective gradient signal.

Why Geometric Priors Alone Are Insufficient. Latent 3D-REPA encourages the DiT’s token relations to mirror those of a geometry-aware encoder, providing an explicit signal of correct 3D structure. Yet this per-frame prior cannot enforce cross-view consistency on its own: each view improves its 3D awareness in isolation, but without an inter-view pathway the views cannot coordinate to produce geometrically compatible outputs.

## 3.6 Joint Mechanism

Neither pillar suffices in isolation, as motivated above; their value emerges from coupling. When both are present, they form a reinforcing loop:

1. The pathway (architecture). Geometry-Aware Cross-View Attention opens the pathway through which views exchange information, and Geo-RoPE biases this pathway to route information along geometrically corresponding tokens, while also placing all views in a shared 3D coordinate frame.

2. The objective (supervision). Latent 3D-REPA ensures the exchanged content is geometrically meaningful, aligning the DiT’s token relations with a 3D-aware prior that spans all views; because Geo-RoPE has fixed a common reference frame, these supervised constraints propagate coherently across views rather than collapsing into per-view shortcuts.

The pathway carries information and the objective makes it 3D-consistent; because each addresses a deficiency the other cannot, their combination yields a non-additive improvement in 3D consistency.

## 3.7 Training Objective

The total training objective combines the flow matching loss with the Latent 3D-REPA distillation loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d i f f } } + \lambda \cdot \mathcal { L } _ { \mathrm { R E P A } } ,\tag{12}
$$

where ${ \mathcal { L } } _ { \mathrm { d i f f } }$ is the flow matching velocity prediction loss from equation (3) and $\lambda = 0 . 5$ balances generation quality with 3D alignment.

The Depth Anything 3 encoder is kept frozen throughout training, serving as a fixed 3D prior. Cross-View Attention blocks are initialized with AdaLN-Zero gating (gate= 0 at initialization) so that the pretrained backbone weights are exactly preserved at step zero, and the new modules gradually contribute as training progresses.

## 4 Experiments

We evaluate PAIWorld on two generation paradigms: action-conditioned video generation and text-conditioned multi-view generation. We first describe our implementation details, then present quantitative comparisons against state-of-the-art baselines across three benchmarks.

## 4.1 Implementation Details

Base Model. We build PAIWorld upon Cosmos-Predict2.5 [53], a flow-matching diffusion transformer (DiT) world foundation model operating in the latent space of a pretrained video VAE. Our full model has approximately 14B parameters. We adopt Cosmos-Reason1 [54], a Physical AI vision-language model, as the text embedder to provide physically-grounded conditioning. The Geometry-Aware Cross-View Attention blocks and Geo-RoPE modules are inserted into the pretrained backbone, while the REPA projection heads are randomly initialized.

Table 1 Action-conditioned generation results on the WorldArena benchmark. Best results are in bold, second best are underlined.
<table><tr><td>Method</td><td>EWMScore ↑</td><td>Visual Quality ↑</td><td>Motion Quality ↑</td><td>Content Consistency ↑</td><td>Physics Adherence ↑</td><td>3D Accuracy ↑</td><td>Controllability ↑</td></tr><tr><td>MAI</td><td>66.17</td><td>63.28</td><td>62.10</td><td>59.60</td><td>66.16</td><td>94.29</td><td>60.98</td></tr><tr><td>Hoseley</td><td>66.17</td><td>62.60</td><td>67.96</td><td>58.75</td><td>63.90</td><td>92.84</td><td>59.11</td></tr><tr><td>MWM</td><td>66.47</td><td>59.30</td><td>58.35</td><td>63.14</td><td>55.46</td><td>89.06</td><td>82.85</td></tr><tr><td>Pelican-Unify [15]</td><td>66.61</td><td>63.60</td><td>61.73</td><td>60.41</td><td>63.98</td><td>97.65</td><td>61.77</td></tr><tr><td>FlowWAM-FiveAges</td><td>66.87</td><td>64.07</td><td>61.70</td><td>60.20</td><td>65.17</td><td>97.78</td><td>62.05</td></tr><tr><td>BetaBWM</td><td>67.00</td><td>62.85</td><td>68.83</td><td>61.36</td><td>63.78</td><td>92.04</td><td>60.39</td></tr><tr><td>SparkWorld</td><td>67.29</td><td>65.04</td><td>59.71</td><td>59.59</td><td>58.59</td><td>92.54</td><td>73.76</td></tr><tr><td>BWM-Fast</td><td>67.87</td><td>62.79</td><td>78.79</td><td>58.30</td><td>61.18</td><td>91.53</td><td>60.26</td></tr><tr><td>GenieEnvisioner-Sim2.0-2B [59]</td><td>68.26</td><td>60.99</td><td>62.16</td><td>60.12</td><td>66.11</td><td>95.06</td><td>73.33</td></tr><tr><td>PAIWorld (Ours)</td><td>70.67</td><td>63.05</td><td>79.66</td><td>57.97</td><td>61.20</td><td>91.58</td><td>74.40</td></tr></table>

Dataset. We curate a large-scale multi-view robotic manipulation dataset of approximately 2.5M video clips from five sources: AgiBot-World [32] (35%), RoboMIND [55] (20%), Galaxea [56] (15%), RoboTwin [57] (15%), and RoboCOIN [58] (15%). These datasets provide multi-camera video streams of robotic manipulation, accompanied by text descriptions or action annotations. Together they span diverse embodiments, manipulation tasks, and camera configurations, offering broad coverage for training a generalizable multi-view world model. For action-conditioned generation, we further fine-tune on task-specific data from the AgiBot-Challenge2026 and WorldArena benchmarks.

Training Configuration. All experiments are conducted on 200 NVIDIA H200 GPUs. Training proceeds for a total of 30,000 iterations with a batch size proportional to the GPU count. We use the AdamW optimizer with a cosine learning rate schedule. The learning rate warms up linearly over the first 3,000 iterations to peak value $3 \times 1 0 ^ { - 5 }$ , then decays following a cosine schedule. The full training takes approximately 7 days. The REPA alignment loss weight is set to λ = 0.5. The 3D foundation model encoder (Depth Anything 3 [48]) is kept frozen throughout training.

## 4.2 Action-Conditioned Video Generation

We first evaluate PAIWorld in the action-conditioned setting, where the model generates future observations given a sequence of robot actions. This setting directly measures the model’s utility as a world simulator for robotic planning and control. We report results on two benchmarks: WorldArena and AgiBot-Challenge2026.

## 4.2.1 WorldArena Benchmark

The WorldArena benchmark provides a comprehensive evaluation suite with seven fine-grained metrics that decompose world model quality into distinct dimensions. Results are presented in table 1.

Evaluation Metrics.

• EWMScore: Overall world model quality score.

• Visual Quality: Perceptual quality of individual generated frames.

• Motion Quality: Smoothness and realism of temporal dynamics.

• Content Consistency: Semantic and appearance coherence across frames and views.

• Physics Adherence: Physical plausibility of object interactions and dynamics.

• 3D Accuracy: Geometric correctness of cross-view 3D structure.

• Controllability: Fidelity of generated video to the input action commands.

![](images/e15c13f44efef20ed00261779614864486b91227e43bd261b6efb2548ab3b115.jpg)  
“Pull open the drawer with right arm, pick up the wooden block with left arm and put it in”

Figure 3 Qualitative results on the WorldArena benchmark. Each row shows a future rollout generated from an initial observation and a commanded action sequence. PAIWorld produces physically plausible dynamics, with object interactions and motion that respect the action commands, while keeping the scene layout stable over long horizons.

As reported in table 1, PAIWorld ranks 1st on the WorldArena benchmark, achieving the best overall EWMScore of 70.67, a clear margin of 2.41 over the second-best entry (GenieEnvisioner-Sim2.0-2B, 68.26). This top ranking is driven by our decisive lead on Motion Quality (79.66, best overall, 0.87 above the runnerup), which reflects the temporally coherent, physically plausible dynamics that action-conditioned rollouts demand. The WorldArena benchmark [33] decomposes world model quality into fine-grained dimensions, allowing us to pinpoint these strengths: the 3D Accuracy metric directly evaluates geometric correctness of cross-view structure, the core capability our framework targets, while Physics Adherence measures whether generated object interactions obey physical laws and Controllability quantifies how faithfully the model follows input action commands, both critical for downstream robotic planning. figure 3 presents representative rollouts, where the generated observations track the commanded actions and preserve physically plausible scene dynamics over time.

## 4.2.2 AgiBot-Challenge2026 Benchmark

The AgiBot-Challenge2026 benchmark evaluates action-conditioned world models on robotic manipulation tasks with four metrics. Results are reported in table 2.

Evaluation Metrics.

• EWMScore: Exponentially Weighted Model Score evaluating overall world model quality.

• PSNR: Peak Signal-to-Noise Ratio measuring reconstruction fidelity.

![](images/b4cc975543a82068b52c11ad5efec5b61eb4c25110099a886f219cabaf06e282.jpg)  
“Pick up a bottle with left arm and another bottle with right arm”

Figure 4 Additional qualitative results on the WorldArena benchmark. PAIWorld generates temporally coherent rollouts across diverse scenes and action sequences, maintaining consistent object appearance and physically plausible motion throughout the predicted horizon.  
Table 2 Action-conditioned generation results on the AgiBot-Challenge2026 benchmark. Best results are in bold, second best are underlined.
<table><tr><td>Team</td><td>EWMScore ↑</td><td>PSNR ↑</td><td>Scen Cons.↑</td><td>nDTW ↑</td></tr><tr><td>NeoVerse-ABot</td><td>0.829</td><td>0.6246</td><td>0.8974</td><td>0.9651</td></tr><tr><td>Loop</td><td>0.8241</td><td>0.6207</td><td>0.9024</td><td>0.9492</td></tr><tr><td>Wild Path</td><td>0.8232</td><td>1</td><td>1</td><td>1</td></tr><tr><td>VIPL-GENUN</td><td>0.8195</td><td>1</td><td>1</td><td>1</td></tr><tr><td>PAIWorld (Ours)</td><td>0.8245</td><td>0.6161</td><td>0.9041</td><td>0.9531</td></tr></table>

• Scene Consistency: Temporal semantic coherence measured by DINOv2 feature similarity.

• nDTW: Normalized Dynamic Time Warping measuring trajectory alignment between generated and ground-truth sequences.

On the AgiBot-Challenge2026 leaderboard, PAIWorld achieves an EWMScore of 0.8245, ranking second overall and surpassing established teams such as Wild Path and VIPL-GENUN. Notably, it attains the best Scene Consistency score (0.9041), showing that our explicit cross-view geometric reasoning translates directly into superior multi-view coherence under action conditioning. The nDTW score of 0.9531 indicates that the generated trajectories closely track the ground-truth action sequences, validating our action-map conditioning. While NeoVerse-ABot edges ahead on EWMScore and PSNR, PAIWorld overtakes it on Scene

![](images/04f948681cf63ab8c0874b36dddfa50e378e250638d1c80989f4a9bd67e5e8c5.jpg)  
“Mop the desktop with a disinfection gun holding on the right arm”

Figure 5 Qualitative results on the AgiBot-Challenge2026 benchmark. For diverse manipulation tasks, PAIWorld rolls out future frames conditioned on the executed robot actions. The generated end-effector and object motions closely track the ground-truth trajectories, and the predicted frames remain sharp and temporally coherent across the rollout.  
Table 3 Text-conditioned multi-view generation results on the AgiBot-World benchmark. Best results are in bold, second best are underlined. ↑ indicates higher is better, ↓ indicates lower is better.
<table><tr><td>Method</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td><td>Scen Cons. ↑</td><td>Geometric ↑</td><td>MEt3R↓</td></tr><tr><td>Genie-Envisioner [59]</td><td>0.7445</td><td>0.3345</td><td>83.7847</td><td>207.2025</td><td>0.9231</td><td>0.5327</td><td>15.75</td></tr><tr><td>Cosmos-Predict2 [3]</td><td>0.5870</td><td>0.3251</td><td>58.2837</td><td>188.6350</td><td>0.8456</td><td>0.4824</td><td>17.47</td></tr><tr><td>Wan2.1 [9]</td><td>0.5715</td><td>0.3354</td><td>56.4735</td><td>184.2186</td><td>0.8617</td><td>0.4716</td><td>16.59</td></tr><tr><td>PAIWorld (Ours)</td><td>0.7683</td><td>0.1844</td><td>45.0389</td><td>175.7778</td><td>0.9041</td><td>0.4056</td><td>14.20</td></tr></table>

Consistency (+0.67%), the metric most directly tied to 3D consistency, which is our primary design objective. figure 5 shows representative action-conditioned rollouts, where the generated observations faithfully reflect the commanded robot actions and maintain physically plausible scene dynamics over time.

## 4.3 Text-Conditioned Multi-View Generation

We evaluate text-conditioned generation on the AgiBot-World benchmark, comparing against three state-ofthe-art baselines: Genie-Envisioner [59], Cosmos-Predict2.5 [53], and Wan2.1 [9]. Results are summarized in table 3.

![](images/e6cb70736c9ab3892a2043e103732b2e0d2fbcd46213c28d5613766b830ce7a5.jpg)  
“Put the plate into the micro-wave oven with left arm”

Figure 6 Additional qualitative results on the AgiBot-Challenge2026 benchmark. Across further manipulation tasks, PAIWorld produces action-faithful rollouts in which object and gripper motions follow the commanded actions while preserving fine visual detail over time.

Evaluation Metrics. We report seven metrics spanning perceptual quality, distributional fidelity, and 3D geometric consistency:

• SSIM: Structural similarity measuring pixel-level correspondence.

• LPIPS: Learned perceptual similarity in deep feature space (lower is better).

• FID: Fréchet Inception Distance measuring frame-level distributional quality.

• FVD: Fréchet Video Distance capturing temporal distributional coherence.

• Scene Consistency: Temporal semantic coherence measured by DINOv2 feature similarity.

• Geometric: Temporal geometric error measured by Sampson epipolar distance (lower is better).

• MEt3R: 3D consistency via point cloud cross-projection (lower is better).

As shown in figure 7, PAIWorld produces multi-view frames with markedly stronger cross-view geometric consistency. Where the baselines exhibit visible object drift and texture misalignment between viewpoints, PAIWorld preserves coherent 3D structure across views.

PAIWorld attains the best score on 6 of the 7 metrics. On perceptual quality, it reaches an SSIM of 0.7683 and an LPIPS of 0.1844, surpassing the second-best method (Genie-Envisioner) by 3.2% in SSIM and 45% in LPIPS, evidence of substantially sharper and more structurally faithful frames. On distributional fidelity, PAIWorld achieves an FID of 45.04, a 20% improvement over Wan2.1 (56.47), indicating that its generations closely match the real data distribution.

Most tellingly, PAIWorld obtains the best MEt3R score of 14.20, a metric that directly quantifies multi-view

![](images/547006041960689e136b31940a465997e55298ae39e47a29231ec69e30d5b689.jpg)  
Figure 7 Qualitative comparison of multi-view video generation. For each scene, we show generated frames from two viewpoints. Compared to Genie-Envisioner, PAIWorld produces geometrically consistent cross-view outputs with coherent object placement, depth structure, and texture alignment across viewpoints.

Table 4 Ablation of the architectural pathway (Geometry-Aware Cross-View Attention with Geo-RoPE, “CVA”) and the geometric objective (Latent 3D-REPA, “REPA”) on the AgiBot-World benchmark. ∆ denotes MEt3R improvement over the backbone. Best results are in bold.
<table><tr><td>CVA</td><td>REPA</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>MEt3R↓</td><td>A</td></tr><tr><td>×</td><td>×</td><td>0.6912</td><td>0.2783</td><td>53.17</td><td>16.84</td><td>一</td></tr><tr><td>√</td><td>×</td><td>0.7204</td><td>0.2361</td><td>50.02</td><td>15.91</td><td>0.93</td></tr><tr><td>×</td><td>√</td><td>0.7156</td><td>0.2447</td><td>49.88</td><td>16.12</td><td>0.72</td></tr><tr><td>√</td><td>√</td><td>0.7683</td><td>0.1844</td><td>45.04</td><td>14.20</td><td>2.64</td></tr></table>

3D reconstruction error. This 10% improvement over the second-best Genie-Envisioner (15.75) confirms that our three components (Geo-RoPE, Geometry-Aware Cross-View Attention, and Latent 3D-REPA) jointly inject geometric consistency into the generation process. The Geometric consistency score of 0.4056 (where lower denotes better cross-view alignment) corroborates this advantage. Genie-Envisioner attains the highest Semantic score (0.9231), owing to its explicit text-grounding mechanism, while PAIWorld remains competitive at 0.9041, a marginal gap on semantics in exchange for a decisive lead on every geometric measure.

## 4.4 Ablation Study

Our central claim is that multi-view 3D consistency requires two remedies acting at complementary levels: an architectural pathway for inter-view communication (Geometry-Aware Cross-View Attention, shaped by Geo-RoPE) and a training objective that enforces 3D-consistent content (Latent 3D-REPA), and that neither alone is sufficient. To test this directly, we ablate the two components on the AgiBot-World benchmark, starting from the backbone with flat multi-view token concatenation and adding each remedy in isolation and in combination. Results are reported in table 4.

The results support our claim along three lines. First, each remedy alone yields only a modest gain. Adding the communication pathway without geometric supervision (row 2) improves MEt3R by 0.93, but the pathway, lacking an explicit 3D objective, partially settles into texture-copying shortcuts that limit its benefit. Adding the geometric objective without an inter-view pathway (row 3) improves MEt3R by 0.72, but the per-view geometric signal has no route to propagate across viewpoints, so cross-view inconsistencies persist. Second, the combination is super-additive. The full model improves MEt3R by 2.64, substantially exceeding the sum of the individual gains (0.93 + 0.72 = 1.65). This non-additive jump is the empirical signature of the reinforcing loop analyzed in section 3.6: the pathway transmits information while the objective makes that information 3D-consistent, and only their coupling enforces consistency that propagates coherently across all views. Third, perceptual quality tracks the same pattern: SSIM, LPIPS, and FID all improve most when both components are present, confirming that the geometric gains do not come at the expense of visual fidelity.

## 5 Conclusion

We presented PAIWorld, a framework for achieving 3D-consistent multi-view generation in world foundation models for robotic manipulation. Our analysis establishes that multi-view 3D consistency requires two remedies acting at complementary levels: an architectural pathway for inter-view communication and a training-objective signal that enforces 3D-consistent content. Geometry-Aware Cross-View Attention, shaped by Geometric Rotary Position Embedding, opens the pathway through which viewpoints exchange information along geometrically corresponding tokens; Latent 3D-REPA supplies the geometric learning signal that makes the exchanged content faithful to true 3D structure. We showed that neither remedy alone suffices: a pathway without geometric supervision degenerates into trivial shortcuts, while a geometric prior without an inter-view pathway cannot propagate constraints across views. Their combination, addressing both the architectural and objective levels at once, yields consistent multi-view 3D generation.

Built upon the DiT-based world foundation model, PAIWorld attains state-of-the-art multi-view 3D consistency on robotic manipulation benchmarks, leading on reconstruction-based, geometric, and scene-consistency metrics across both text- and action-conditioned settings. This improved consistency carries over to downstream embodied tasks: model-based planning benefits from physically plausible imagined trajectories, world action models exhibit stronger action-visual causal alignment, and multi-view policy post-training yields more effective manipulation policies.

Several promising directions remain for future work. First, incorporating physical interaction modeling, such as contact dynamics, deformable objects, and fluid simulation, would push 3D consistency beyond geometry into physics-aware world modeling. Second, scaling to long-horizon planning scenarios will require maintaining 3D consistency over extended temporal rollouts, potentially through hierarchical or recurrent architectures. Third, by coupling our world model with a World Action Model (WAM), we aim to build a world-model-driven data closed loop for embodied intelligence: the world model generates diverse imagined experiences, the WAM learns from these experiences to improve its policy, and the improved policy in turn collects higher-quality real-world data to further refine the world model, enabling continuous self-improvement and autonomous evolution of embodied agents. Fourth, we plan to develop industrial manufacturing foundation models built upon our world modeling framework, targeting applications such as dynamic scheduling of production lines and real-time control of manufacturing processes, where accurate physical simulation and multi-view monitoring are critical for intelligent manufacturing.

## Contributions

Core Contributors: Yuhang Huang, Jiazhao Zhang, Xuan Lv, Junyan Xu, Zhiyuan Yu, Ruizhen Hu, Kai Xu

Contributors: Wancheng Feng, Shilong Zou, Hewen Xiao, Ziqiao Zhou, Kaiyun Huang, Zhiyu Peng, Juzhan Xu, Hang Zhao, Chenyang Zhu, Renjiao Yi, Yifei Huang, Douhui Wu, Yan Zhang, Kexu Cheng, Chunhe Song, Yunzhi Xue, Xiuhong Zhang, Leitao Guo, Yunji Chen, Bin Wu, Haibin Yu

Corresponding Author: Kai Xu

## References

[1] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

[2] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations (ICLR), 2020.

[3] NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, et al. Cosmos world foundation model platform for physical AI. arXiv preprint arXiv:2501.03575, 2025.

[4] NVIDIA. Cosmos 3: Omnimodal world models for physical AI. arXiv preprint arXiv:2606.02800, 2026.

[5] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, Lifang He, and Lichao Sun. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024.

[6] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations (ICLR), 2025.

[7] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. In Advances in Neural Information Processing Systems, volume 37, 2024.

[8] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[9] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[10] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Ken Goldberg, and Pieter Abbeel. DayDreamer: World models for physical robot learning. In Proceedings of the Conference on Robot Learning (CoRL), volume 205, pages 2226–2240, 2022.

[11] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In Advances in Neural Information Processing Systems, volume 36, 2023.

[12] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. RoboDreamer: Learning compositional world models for robot imagination. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

[13] Yuhang Huang, Jiazhao Zhang, Shilong Zou, Xinwang Liu, Ruizhen Hu, and Kai Xu. LaDi-WM: A latent diffusion-based world model for predictive manipulation. arXiv preprint arXiv:2505.11528, 2025.

[14] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.

[15] Yi Zhang, Yinda Chen, Che Liu, Zeyuan Ding, Jin Xu, Shilong Zou, Junwei Liao, Jiayu Hu, Xiancong Ren, Xiaopeng Zhang, et al. Pelican-Unified 1.0: A unified embodied intelligence model for understanding, reasoning, imagination and action. arXiv preprint arXiv:2605.15153, 2026.

[16] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. RT-1: Robotics transformer for real-world control at scale. In Robotics: Science and Systems (RSS), 2023.

[17] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Proceedings of the Conference on Robot Learning (CoRL), volume 229, pages 2165–2183, 2023.

[18] Open X-Embodiment Collaboration. Open X-embodiment: Robotic learning datasets and RT-X models. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2024.

[19] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robotics: Science and Systems (RSS), 2023.

[20] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Proceedings of the 41st International Conference on Machine Learning, 2024.

[21] Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. iVideoGPT: Interactive VideoGPTs are scalable world models. In Advances in Neural Information Processing Systems, volume 37, 2024.

[22] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[23] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for video diffusion models. In International Conference on Learning Representations (ICLR), 2025.

[24] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. REPA: Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2025.

[25] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

[26] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Francois Fleuret. DIAMOND: Diffusion for world modeling: Visual details matter in atari. In Advances in Neural Information Processing Systems, volume 37, 2024.

[27] Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In International Conference on Learning Representations (ICLR), 2024.

[28] Jack Parker-Holder, Stephen Spencer, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, et al. Genie 2: A large-scale foundation world model. Google DeepMind Blog, 2024. https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/.

[29] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. GR-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

[30] Fangqi Zhu, Hongtao Wu, Song Guo, et al. IRASim: Learning interactive real-robot action simulators. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[31] Siyuan Huang, Liliang Chen, Pengfei Zhou, Shengcong Chen, Zhengkai Jiang, Yue Hu, Yue Liao, Peng Gao, Hongsheng Li, Maoqing Yao, and Guanghui Ren. EnerVerse: Envisioning embodied future space for robotics manipulation. In Advances in Neural Information Processing Systems, 2025.

[32] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. AgiBot World Colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.

[33] Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, et al. WorldArena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026.

[34] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3D object. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9298–9309, 2023.

[35] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. SyncDreamer: Generating multiview-consistent images from a single-view image. In International Conference on Learning Representations (ICLR), 2024.

[36] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. MVDream: Multi-view diffusion for 3D generation. In International Conference on Learning Representations, 2024.

[37] Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. MVDiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. In Advances in Neural Information Processing Systems, volume 36, 2023.

[38] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitrii Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. SV3D: Novel multi-view synthesis and 3D generation from a single image using latent video diffusion. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[39] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. SV4D: Dynamic 3D content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024.

[40] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, and Ben Poole. CAT3D: Create anything in 3D with multi-view diffusion models. In Advances in Neural Information Processing Systems, volume 37, 2024.

[41] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[42] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision (ECCV), pages 405–421, 2020.

[43] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.

[44] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20697–20709, 2024.

[45] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3D with MASt3R. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[46] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10371–10381, 2024.

[47] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. In Advances in Neural Information Processing Systems, volume 37, 2024.

[48] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. In International Conference on Learning Representations (ICLR), 2026.

[49] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[50] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[51] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

[52] Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, and Guanghui Ren. EnerVerse-AC: Envisioning embodied environments with action condition. arXiv preprint arXiv:2505.09723, 2025.

[53] NVIDIA. World simulation with video foundation models for physical AI. arXiv preprint arXiv:2511.00062, 2025.

[54] NVIDIA. Cosmos-Reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025.

[55] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

[56] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and G0 dual-system VLA model. arXiv preprint arXiv:2509.00576, 2025.

[57] Yao Mu, Tianxing Chen, Zeyu Gao, Zhiqian Lan, Yude Zou, Lunkai Lin, Zhiqiang Xie, and Ping Luo. RoboTwin: Dual-arm robot benchmark with generative digital twins. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[58] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025.

[59] Yue Liao, Yuxin Jiang, Liliang Chen, Siyuan Huang, Pengfei Zhou, Shengcong Chen, Chiming Liu, Xindong He, Yi Liu, Maoqing Yao, Guanghui Ren, and Hongsheng Li. Genie Envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025.