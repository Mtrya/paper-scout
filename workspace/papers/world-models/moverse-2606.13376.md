![](images/bffee17a3f80015a55bde23b7f4a55a54726dd20c8c31c6e8a7be7eb7c26bbee.jpg)

![](images/d2410f820faab1e4d54ba9c0ec252b1b9afabb1bb3325dc9c7710a943feb7c9c.jpg)

# MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold

Yang Zhou1∗†, Ziheng Wang2∗†, Yuqin Lu1∗†, Haofeng Liu3‡, Jun Liang3‡, Shengfeng He4, Jing Li3

1South China University of Technology 2Columbia University 3Orange Team, Youku Moku-Lab, HUJING Digital Media & Entertainment Group 4Singapore Management University

Abstract. We present MoVerse, a real-time video world model that creates an interactively navigable scene from a single narrow-field-of-view image. This setting is challenging because the input observes only a small fraction of the environment, while interactive roaming requires a complete surrounding world, persistent geometry, controllable camera motion, and temporally coherent high-fidelity observations. MoVerse addresses this problem by separating world construction from observation rendering. It first expands the input into a gravity-aligned 360◦ panorama with topology-aware diffusion, closing the missing field of view before 3D reasoning. It then lifts the panorama into a persistent 3D Gaussian scaffold using panoramic geometry-aware residual prediction, yielding a dense and directly renderable spatial memory. Finally, a Gaussian-conditioned video renderer translates scaffold renderings along user-specified camera trajectories into photorealistic video. To make this renderer practical for interaction, we train a bidirectional diffusion teacher for high-quality conditional rendering and distill it into a causal autoregressive student for bounded-latency streaming. This design combines the controllability and long-range consistency of explicit 3D representations with the perceptual quality of generative video models. MoVerse supports real-time scene roaming at 8 FPS on a single NVIDIA RTX 4090 GPU, demonstrating a practical path toward single-image world creation with interactive video output.

![](images/3df5b32a45c94f2dd3ad56fd4d76f95ecc6084b8d330afe51263ce4261ed6b85.jpg)

![](images/1177300f5ab84ec7a25f1ba3f204e603230e77dfd229e1735f64f4e6163d9bdf.jpg)

![](images/29cb50978e5ddaf9195ef9043cb6e3136c8a4a6cc807b1d7539ac4d0a24d8552.jpg)

## 1 Introduction

Generating a navigable world from a single narrow-field-of-view (NFOV) image is fundamentally underconstrained. The input observes only a small frustum, while interactive applications such as VR prototyping, digital twins, content creation, and embodied-agent simulation require a user to move through a complete, spatially persistent environment. A practical system must therefore solve three coupled problems: it must complete the missing field of view, convert that completion into a camera-controllable scene representation, and render high-quality temporally coherent video during interaction.

Existing approaches usually emphasize only part of this requirement. Explicit 3D methods build persistent scene assets, such as point clouds, meshes, or 3D Gaussian scenes, and then render novel views from them [1, 2, 3, 4, 5, 6, 7, 8, 9]. Their explicit state provides durable spatial memory, accurate camera control, and high frame-rate rendering. However, when the asset is lifted directly from a single NFOV image, most of the scene must be inferred from weak evidence; when additional views are synthesized before reconstruction, the resulting asset can inherit cross-view inconsistencies and generation cost. Direct rendering from such assets may expose holes, floaters, depth errors, or limited perceptual quality, especially under large viewpoint changes.

Implicit video and world models take the opposite route [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]. They generate observations during interaction and store history in attention windows, recurrent states, or key–value caches. These models can produce visually strong videos, but their long-range geometry is only as stable as the implicit memory retained by the model. As the user follows a long trajectory or revisits a previously seen region, the scene can drift, change identity, or develop boundary discontinuities that are difficult to correct online.

Hybrid systems use explicit geometry as a condition for generative rendering [20, 21, 22, 23, 24, 25]. They show that a geometric anchor can constrain camera motion and scene layout while a learned renderer improves appearance. However, the strength of the anchor and the deployability of the renderer remain in tension: sparse or coarse anchors give limited spatial guidance, whereas strong generative renderers are often too expensive or too bidirectional for real-time interaction. This motivates a design in which the explicit condition is dense, panoramic, directly renderable, and reusable, while the video model is distilled into a causal renderer for streaming use.

We present MoVerse, a three-stage system for real-time video world modeling with panoramic Gaussian guidance. Given a single NFOV image, MoVerse first generates a gravity-aligned 360◦ equirectangular panorama, then lifts the panorama into a persistent 3D Gaussian scaffold, and finally renders interactive video through a Gaussian-conditioned autoregressive video model. The key design choice is to separate world construction from observation rendering: Stages I and II build a reusable scaffold offline, while Stage III turns scaffold renderings along user-specified camera trajectories into high-fidelity video online.

MoVerse is organized around three technical components:

• Geometry-aware panoramic generation. Stage I closes the missing field of view before 3D lifting. It canonicalizes an unposed NFOV input into a gravity-aligned panoramic frame using differentiable autoleveling, then performs masked latent diffusion completion with circular latent encoding and shift-equivariant generation to respect the horizontal S1 topology of ERP panoramas. The model is trained on Horizon360, a curated set of canonicalized panoramas with yaw-centered perspective-view supervision.

• Panoramic Gaussian scaffold construction. Stage II converts the completed panorama into a directly renderable 3DGS scene using a feed-forward panoramic Gaussian generator. The scaffold is initialized by spherical back-projection from panorama depth, uses latitude-aware scale correction proportional to cos ϕ, and predicts residuals in angular–inverse-depth space so that Gaussian updates remain consistent with ERP geometry and horizontal closure.

• Gaussian-conditioned streaming video rendering. Stage III renders the scaffold along the requested camera trajectory and translates the resulting RGB conditioning stream into final video. A bidirectional teacher, initialized from Wan2.1-T2V, learns dense Gaussian-conditioned video rendering with shared positional coordinates between target and condition tokens. It is then distilled into a causal autoregressive student using self-forcing and distribution matching, enabling bounded-latency streaming with a local key–value cache while the explicit scaffold carries long-range spatial memory.

![](images/5e0c10fcc30b50b66690c90d811c6eb68488cb7b9336ebc8c1b111c015f2bf94.jpg)  
Figure 1 MoVerse pipeline overview. From a single narrow-field-of-view input image, Stage I synthesizes a gravity-aligned 360◦ panorama, Stage II lifts the panorama into a persistent 3D Gaussian scaffold, and Stage III renders Gaussianconditioned video along user-specified camera trajectories for real-time interactive roaming.

This factorization gives each stage a clear role. The panorama generator supplies complete omnidirectional evidence instead of asking the 3D module to hallucinate most of the world. The Gaussian predictor turns that evidence into a persistent renderable asset rather than leaving scene state inside a video model. The causal renderer improves perceptual quality and temporal coherence without replacing the camera motion and layout encoded by the scaffold. In our deployment configuration, this design supports real-time roaming at 8 FPS on a single NVIDIA RTX 4090 GPU.

Stage-wise positioning. The three stages connect several lines of prior work. Stage I builds on panorama outpainting and 360◦ generation [26, 27, 28, 29], but emphasizes gravity canonicalization and spherical topology. Stage II follows feed-forward 3DGS reconstruction [30, 31, 32, 33, 34, 35], but specializes residual Gaussian prediction to ERP panoramas. Stage III adopts the conditioning pattern of camera-controlled video and novel-view synthesis [23, 21, 22] on top of modern video generators [36, 37, 38], realizing causal streaming rather than offline clip refinement.

## 2 Method

## 2.1 Overview

Given a single NFOV image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , MoVerse produces a real-time navigable video stream through three sequential stages, as illustrated in Fig. 1. The pipeline first completes the missing omnidirectional context, then stores the completed world in a persistent Gaussian scaffold, and finally renders interactive observations by translating scaffold renderings into temporally coherent video.

1. Stage I: Panoramic Generation (I → P ). The input image is expanded into a gravity-aligned $3 6 0 ^ { \circ }$ equirectangular panorama P . This step closes the missing field of view before any 3D lifting, providing structurally complete observations for every viewing direction and eliminating the need for the subsequent 3D stage to hallucinate large unobserved regions.

2. Stage II: Gaussian Scene Generation $( P \to { \mathcal { G } } )$ The panorama is lifted into a panoramic 3D Gaussian scene $\mathcal { G } = \{ ( \mu _ { k } , \Sigma _ { k } , \alpha _ { k } , c _ { k } ) \} _ { k = 1 } ^ { K }$ , where each Gaussian is parameterized by its center $\mu _ { k }$ , covariance $\Sigma _ { k }$ opacity $\alpha _ { k }$ , and appearance $c _ { k }$ . Unlike point-cloud or mesh representations, 3D Gaussians are directly splatting-renderable at real-time frame rates, yielding a dense, camera-controllable RGB conditioning video $\hat { V } _ { 1 : T } = \{ \hat { V } _ { t } \} _ { t = 1 } ^ { T }$ that is substantially more informative than sparse point-cloud projections, but still

![](images/03d07c734955530acd7ab4397218284747e1426315839922e8f00b03c583b239.jpg)  
Figure 2 Stage I panoramic generation. The input NFOV image is auto-leveled into a gravity-aligned canonical viewing space and then completed as a topology-aware ERP panorama. Circular latent encoding and shift-equivariant generation preserve the horizontal $S ^ { 1 }$ boundary.

imperfect as final photorealistic observations.

3. Stage III: Gaussian-Conditioned Video Rendering $( \hat { V } _ { 1 : T }  V _ { 1 : T } )$ . A learned video renderer translates the Gaussian-rendered conditioning video $\hat { V } _ { 1 : T }$ into a high-fidelity output video $V _ { 1 : T } = \{ V _ { t } \} _ { t = 1 } ^ { T }$ . By generating sequences rather than independent frames, the renderer maintains temporal coherence across successive viewpoints while enhancing perceptual quality without changing the camera motion or scene layout implied by the scaffold.

Offline vs. online split. Stages I and II are executed once as offline scaffold construction: given a new input image, they produce a persistent 3DGS asset in seconds. Stage III operates online: at interaction time, the scaffold is rendered at the user-requested camera trajectory, and the video renderer streams enhanced frames in real time. This separation ensures that the computationally heavier generative steps (panorama synthesis, 3D lifting) do not bottleneck interactive exploration, while the explicit scaffold provides durable geometric memory that prevents drift across arbitrarily long trajectories.

## 2.2 Stage I: Panoramic Generation

Stage I expands the input NFOV image I into a complete equirectangular panorama $P \in \mathbb { R } ^ { H _ { e } \times W _ { e } \times 3 }$ . Image outpainting and completion have progressed from GAN-based extrapolation to diffusion-based inpainting and latent completion [39, 40, 41, 42, 43]. In parallel, recent panoramic generation methods have adapted diffusion priors to 360◦ content synthesis, including text-to-panorama generation, perspective-conditioned panorama outpainting, and immersive world generation [44, 45, 26, 28, 27, 5, 46]. Rather than treating this step as generic image outpainting, we formulate it as geometry-aware panoramic completion. The generated panorama should provide complete 360◦ scene context, maintain a gravity-aligned horizon, and respect the horizontal $S ^ { 1 }$ periodicity of ERP images. We denote this stage as

$$
P = G _ { \theta } ( I ) ,\tag{1}
$$

where $G _ { \theta }$ first canonicalizes the input view and then completes the surrounding omnidirectional scene. Fig. 2 summarizes the resulting Stage I pipeline: the input is stabilized into a canonical panoramic frame, used as masked visual context for diffusion-based completion, and decoded with topology-aware operations to produce a horizontally closed ERP panorama.

The generator is implemented as a conditional latent diffusion completion model, following the common practice of performing high-resolution synthesis in a compressed latent space [42]. Let E and D denote the latent encoder and decoder. During training, a canonical target panorama is encoded as $z _ { 0 } = \mathcal { E } ( P )$ , and a binary ERP mask M marks the known region induced by the sampled input view. The visible reference latent is

$$
z _ { \mathrm { r e f } } = z _ { 0 } \odot M .\tag{2}
$$

where ⊙ denotes element-wise multiplication, with M broadcast across latent channels. At inference time, the known latent context is instead formed from the input image after projection and canonical alignment. During denoising, the network predicts the injected noise from the noisy latent, the mask, and the known visual context:

$$
\hat { \epsilon } = \epsilon _ { \theta } ( z _ { t } \oplus M \oplus z _ { \mathrm { r e f } } , t , \tau ( y ) ) ,\tag{3}
$$

where $\oplus$ denotes channel-wise concatenation and $\tau ( y )$ is the conditioning embedding. The corresponding latent diffusion objective is

$$
\mathcal { L } _ { \mathrm { L D M } } = \mathbb { E } _ { z _ { 0 } , \epsilon , t } \left[ \left\| \epsilon - \epsilon _ { \theta } ( z _ { t } \oplus M \oplus z _ { \mathrm { r e f } } , t , \tau ( y ) ) \right\| _ { 2 } ^ { 2 } \right] .\tag{4}
$$

This masked latent formulation preserves the observed input while synthesizing the unobserved panoramic field of view.

A direct perspective-to-ERP outpainting pipeline is insufficient for this purpose because it ignores two domain gaps. The first is projective variance. Real input images are generally unposed: arbitrary pitch and roll distort the projected ERP observation, causing straight walls, floor boundaries, and horizons to become curved or tilted. Many panorama outpainting pipelines project the input view into ERP using predefined camera assumptions or known extrinsics [26, 28, 27, 5, 46], which limits robustness for unconstrained photographs. These non-linear distortions conflict with the structural priors that image diffusion models learn from perspective imagery. The second is topological severing. ERP panoramas are horizontally periodic: the left and right image boundaries correspond to adjacent azimuthal directions on the same physical sphere. Standard diffusion backbones, VAEs, and positional encodings are designed for bounded Euclidean images and can therefore introduce artificial seams at this boundary. Existing works mitigate this issue through circular blending, synchronized denoising, circular padding, or specialized panoramic operators [45, 47, 48, 49], but these treatments do not fully remove the underlying mismatch between Euclidean image grids and the closed azimuthal topology. We address these two gaps in order: first by defining a gravity-aligned canonical panoramic representation, and then by enforcing topology-aware completion in the latent generation process.

## 2.2.1 Canonical panoramic representation.

We complete the scene in a canonical viewing space C. This space removes pitch and roll ambiguity by aligning the environmental horizon with the equator of the ERP canvas. The output of Stage I is therefore not merely a visually plausible panorama, but a normalized panoramic observation with stable longitude–latitude coordinates. In this representation, vertical architectural structures remain plumb and the horizon remains level. A separate yaw-centered target convention, described below in the training data construction, places the input view at the center of the panorama rather than near the periodic boundary. We write the gravity canonicalization abstractly as

$$
z _ { \mathrm { c a n } } = \mathcal { T } _ { \mathrm { a l i g n } } ( z _ { \mathrm { r e f } } ) ,\tag{5}
$$

where $\tau _ { \mathrm { a l i g n } }$ maps the observed latent context into the gravity-aligned panoramic frame.

## 2.2.2 Differentiable auto-leveling.

We adopt differentiable auto-leveling to estimate $\tau _ { \mathrm { a l i g n } }$ for uncalibrated perspective inputs without requiring camera extrinsics at inference time. The auto-leveling module first predicts a dense 2D correspondence field

$$
P _ { \mathrm { d e n s e } } \in \mathbb { R } ^ { H _ { I } \times W _ { I } \times 2 } ,\tag{6}
$$

which represents pixel-wise shifts toward a gravity-leveled reference. Instead of directly using this dense field as a free-form spatial warp, the correspondences are passed through a rigid transformation bottleneck implemented by a differentiable soft-argmin solver. The solver searches for a low-dimensional rigid spherical

transformation $\hat { R }$ whose reprojection best explains the dense correspondences, including the gravity-alignment rotation needed to remove pitch and roll under the canonical yaw convention. The actual canonical warp is then performed by spherical grid sampling:

$$
z _ { \mathrm { c a n } } = \mathcal { W } ( z _ { \mathrm { r e f } } , \hat { R } ) ,\tag{7}
$$

where W denotes differentiable spherical sampling.

This constraint is important for stable diffusion-based training. Although spatial transformer networks provide a general mechanism for differentiable image warping [50], an unconstrained spatial transformer can absorb high-frequency reconstruction gradients as local non-rigid distortions, producing unstable jelly-like warps. The rigid bottleneck instead restricts alignment updates to globally coherent camera rotations. To stabilize early training, we also apply an auxiliary geometric anchor to the dense correspondence prediction:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \mathrm { S m o o t h } _ { L _ { 1 } } ( P _ { \mathrm { d e n s e } } , P _ { \mathrm { G T } } ) ,\tag{8}
$$

where $P _ { \mathrm { G T } }$ is analytically obtained from canonicalized panorama–view pairs. The final warp still passes through the rigid solver, so the dense field supervises alignment evidence rather than becoming a free-form deformation field.

## 2.2.3 Topology-aware panoramic completion.

After canonicalization, the generator completes the missing field of view while preserving the horizontal closure of the ERP domain. We use circular latent encoding so that horizontal convolutional padding in the latent autoencoder wraps around the azimuthal axis rather than introducing artificial image borders. This makes the latent compression and reconstruction process consistent with the periodic condition of ERP panoramas.

We further encourage shift-equivariant diffusion generation. Let $\operatorname { R o l l } _ { \delta } ( \cdot )$ denote a horizontal circular shift by offset δ. For the canonical diffusion input $X = z _ { t }$ ⊕ M ⊕ $z _ { \mathrm { c a n } } .$ the base and shifted branches are

$$
\hat { \epsilon } _ { \mathrm { b a s e } } = \epsilon _ { \theta } ( X , t , \tau ( y ) ) ,\tag{9}
$$

$$
\hat { \epsilon } _ { \mathrm { s h i f t e d } } = \epsilon _ { \theta } ( \operatorname { R o l l } _ { \delta } ( X ) , t , \tau ( y ) ) .\tag{10}
$$

The shift-consistency objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s h i f t } } = \mathbb { E } _ { z _ { 0 } , \epsilon , t , \delta } \left[ \| \mathrm { R o l l } _ { \delta } ( \hat { \epsilon } _ { \mathrm { b a s e } } ) - \hat { \epsilon } _ { \mathrm { s h i f t e d } } \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{11}
$$

For transformer-based diffusion backbones, this circular shift can be implemented efficiently by shifting positional coordinates rather than physically rolling all latent tokens. During inference, random horizontal shifts can also be applied across denoising steps and inverted at the end, so no fixed tensor boundary consistently acts as the seam location.

## 2.2.4 Canonical training data.

The panoramic generator is trained with Horizon360, a curated collection of gravity-aligned indoor and outdoor panoramas. Existing omnidirectional datasets and scene resources provide diverse panoramic content [51, 52, 53, 3], but raw panoramas often contain inconsistent pitch, roll, and horizon placement. The raw panoramas are therefore canonicalized by rotating the spherical image so that the true horizon aligns with the ERP equator and physical verticals remain plumb, following the geometric intuition behind panoramic layout and horizon estimation [54, 55, 56, 57, 58]. Fig. 3 visualizes this rectification process. This provides the diffusion model with a consistent structural prior instead of asking it to learn from panoramas with arbitrary tilt and roll.

To expose the model to realistic input photographs, perspective views are sampled from the canonical panoramas with variable yaw, pitch, roll, field of view, and aspect ratio. The sampling distribution mixes common handheld captures with more extreme camera poses, so the auto-leveling module learns both small everyday misalignments and challenging out-of-distribution tilts. We also circularly roll each target panorama by the input yaw so that the observed view is centered in the canonical canvas:

$$
P _ { \mathrm { c e n t e r e d } } = \mathrm { R o l l } ( P _ { \mathrm { E R P } } , - \psi _ { \mathrm { i n p u t } } ) .\tag{12}
$$

Canonicalization  
![](images/a7b185a95251764f7e4ed4cf9371c200b5b260b06fb9b5f53b5f74887f7d8310.jpg)  
Figure 3 Canonical panorama construction. Raw unaligned panoramas are rotated on the sphere so that the horizon coincides with the ERP equator and physical verticals become plumb, yielding the canonical training targets used by Stage I.

This yaw-centered convention keeps the known region away from the periodic boundary and makes completion more stable.

The Stage I training objective combines the standard latent diffusion loss with the geometric and topological regularizers:

$$
\mathcal { L } _ { \mathrm { s t a g e 1 } } = \mathcal { L } _ { \mathrm { L D M } } + \lambda _ { \mathrm { s h i f t } } \mathcal { L } _ { \mathrm { s h i f t } } + \lambda _ { \mathrm { f l o w } } \mathcal { L } _ { \mathrm { f l o w } } .\tag{13}
$$

The final output $P = G _ { \theta } ( I )$ is an equirectangular, gravity-aligned, horizontally periodic panorama centered around the input view, which serves as the canonical panoramic input for the following 3D construction stage.

## 2.3 Stage II: Gaussian Scene Generation

Stage II converts the gravity-aligned panorama P from Stage I into a persistent, directly renderable 3D Gaussian scaffold G, building a dense and stable geometry-and-appearance anchor for downstream synthesis. We therefore utilize a feed-forward 3DGS predictor: it runs once per input panorama, produces an explicit representation that can be reused during interaction, and yields informative RGB conditioning frames for the video renderer. Fig. 4 summarizes this panorama-to-Gaussian pathway: the model constructs an ERP-aware spherical Gaussian initialization and then predicts residual corrections to geometry and appearance before producing a standard splatting-renderable 3DGS asset.

## 2.3.1 Panoramic Gaussian initialization.

The input panorama is represented in equirectangular projection (ERP). Each pixel $( u , v )$ corresponds to a longitude–latitude pair

$$
\theta = \left( { \frac { u } { W } } - { \frac { 1 } { 2 } } \right) 2 \pi , \qquad \phi = \left( { \frac { 1 } { 2 } } - { \frac { v } { H } } \right) \pi ,\tag{14}
$$

and hence to a unit viewing direction

$$
{ \bf d } ( \theta , \phi ) = \bigl ( \cos \phi \sin \theta , - \sin \phi , \cos \phi \cos \theta \bigr ) .\tag{15}
$$

Given an estimated panorama depth map $D _ { \mathbf { \lambda } }$ we instantiate Gaussians on a strided ERP grid $\Omega ^ { \prime } = \{ ( u _ { k } , v _ { k } ) \} _ { k = 1 } ^ { K }$ with resolution $H ^ { \prime } \times W ^ { \prime }$ , so $K = H ^ { \prime } W ^ { \prime }$ . For each grid location, let

$$
\theta _ { k } = \theta ( u _ { k } ) , \qquad \phi _ { k } = \phi ( v _ { k } ) , \qquad \mathbf { d } _ { k } = \mathbf { d } ( \theta _ { k } , \phi _ { k } ) , \qquad D _ { k } = D ( u _ { k } , v _ { k } ) .\tag{16}
$$

![](images/08a1fb49b216464b563bd3b7ac0015947086cf0f15679f03847e70c780bf1d85.jpg)  
Figure 4 Stage II Gaussian scene generation. Given the completed gravity-aligned panorama from Stage I, MoVerse initializes Gaussian primitives in the panoramic domain with latitude-aware scaling, and predicts residual Gaussian attributes in angular–inverse-depth space to obtain a persistent 3DGS scaffold for downstream video rendering.

The initial Gaussian center is obtained by spherical back-projection,

$$
\begin{array} { r } { \mu _ { k } = D _ { k } \mathbf { d } _ { k } . } \end{array}\tag{17}
$$

We also initialize the Gaussian scale in an ERP-aware manner. Since the spherical area represented by an ERP pixel is proportional to cos $\phi _ { k }$ , the tangential footprint associated with longitude spacing should shrink toward high latitudes. We therefore set the initial scale as

$$
s _ { k } \propto D _ { k } \cos \phi _ { k } ,\tag{18}
$$

with a lower bound to avoid degenerate splats near the poles. Each primitive then carries a center, anisotropic scale, rotation, color, and opacity, yielding

$$
\mathcal { G } = \{ ( \mu _ { k } , s _ { k } , q _ { k } , c _ { k } , \alpha _ { k } ) \} _ { k = 1 } ^ { K } ,\tag{19}
$$

where $\mu _ { k }$ is the Gaussian center, $s _ { k }$ denotes the 3D scale vector, $q _ { k }$ a quaternion rotation, $c _ { k }$ RGB appearance, and $\alpha _ { k }$ opacity.

## 2.3.2 Depth-guided residual prediction.

Our panoramic Gaussian generator follows the depth-guided residual 3DGS prediction principle of SHARP [35], but adapts it from perspective images to equirectangular panoramas. This adaptation replaces planar camera assumptions with spherical geometry: pixel displacements correspond to angular changes, sampling density varies with latitude, and the horizontal boundary must remain topologically closed. Starting from the initialized scaffold, we extract multi-scale image features with a pretrained vision backbone and decode them with a DPT-style feature head into geometry and texture features. The feature head processes the encoder features from the panorama depth model and further fuses an explicit feature input formed by concatenating the RGB panorama with inverse depth, i.e., $[ P , D ^ { - 1 } ]$ . Let $\mathbf { f } _ { k }$ denote the decoded feature vector at grid location $k ,$ and let $h _ { \psi }$ denote the lightweight residual prediction head with learnable parameters $\psi .$ . The head predicts residual Gaussian parameters as

$$
\left( \Delta \theta _ { k } , \Delta \phi _ { k } , \Delta z _ { k } , \Delta s _ { k } , \Delta q _ { k } , \Delta c _ { k } , \Delta \alpha _ { k } \right) = h _ { \psi } ( \mathbf { f } _ { k } ) .\tag{20}
$$

The prediction head is zero-initialized, so before learning the network reproduces the initialized spherical scaffold. Training therefore starts from a physically meaningful initialization and only learns local corrections. This residual formulation is intended to keep the scaffold anchored to the panorama and the provided depth geometry, while still allowing the network to correct local errors needed for multi-view renderability.

## 2.3.3 Attribute composition.

We compose all predicted Gaussian quantities as residuals around the initialized scaffold. The most important case is the Gaussian center, because ERP panoramas use an angular image domain rather than a planar one. In a pinhole camera, image-plane offsets are naturally expressed in normalized device coordinates. In an ERP

panorama, horizontal motion changes longitude and vertical motion changes latitude. For each primitive $k ,$ we therefore predict position offsets $( \Delta \theta _ { k } , \Delta \phi _ { k } , \Delta z _ { k } )$ and apply them in angular–inverse-depth space before spherical back-projection:

$$
\theta _ { k } ^ { \prime } = \theta _ { k } + \lambda _ { x y } \Delta \theta _ { k } , \qquad \phi _ { k } ^ { \prime } = \mathrm { c l i p } ( \phi _ { k } + \lambda _ { x y } \Delta \phi _ { k } , - \pi / 2 + \epsilon , \pi / 2 - \epsilon ) , \qquad D _ { k } ^ { \prime } = { \frac { 1 } { \mathrm { s o f t p l u s } ( \rho _ { k } + \lambda _ { z } \Delta z _ { k } ) + \epsilon } } \mathrm { c l o s s } ( { \bf k } + { \bf k } ) ,\tag{21}
$$

where $\rho _ { k } = \mathrm { s o f t p l u s } ^ { - 1 } ( D _ { k } ^ { - 1 } )$ is the unconstrained parameter corresponding to the base inverse depth, and $\lambda _ { x y } , \lambda _ { z }$ control the residual magnitude. The corrected center is then

$$
\boldsymbol \mu _ { k } = D _ { k } ^ { \prime } \mathbf { d } ( \boldsymbol \theta _ { k } ^ { \prime } , \boldsymbol \phi _ { k } ^ { \prime } ) .\tag{22}
$$

This angular composition makes the learned displacement consistent with the ERP sampling grid and preserves the horizontal $S ^ { 1 }$ closure of the panorama.

Other attributes follow the same residual composition. Scale is applied as a bounded multiplicative update to the latitude-corrected base scale, quaternions are residual updates to the identity rotation, colors are initialized from the panorama, and opacities from a constant prior; the corresponding activations keep scales positive and colors and opacities in valid ranges. During training, these residuals provide local corrections while keeping scales positive, colors in range, and opacities valid. The final output is flattened into a standard 3DGS set and can be rendered by an off-the-shelf splatting renderer without special panoramic logic.

## 2.3.4 Training objective.

We train Stage II on HM3D dataset [59] using differentiable rendering. For each training scene, the model takes an ERP panorama and its depth as input, predicts a 3DGS scaffold, and renders it into the target view. Let $R ( { \mathcal { G } } )$ denote the rendering of the predicted Gaussians at the target camera, and let $I ^ { t g t }$ be the corresponding ground-truth image. The main reconstruction objective combines pixel, perceptual, and structural terms:

$$
\mathcal { L } _ { \mathrm { t g t } } = \lambda _ { 1 } \| R ( \mathcal { G } ) - I ^ { t g t } \| _ { 1 } + \lambda _ { \mathrm { l p i p s } } \mathrm { L P I P S } ( R ( \mathcal { G } ) , I ^ { t g t } ) + \lambda _ { \mathrm { s s i m } } \big ( 1 - \mathrm { S S I M } ( R ( \mathcal { G } ) , I ^ { t g t } ) \big ) .\tag{23}
$$

This term trains the scaffold to be directly renderable from target viewpoint. To keep the scaffold anchored to the observed input, we also render it back to the source panorama or its corresponding source views and apply a source reconstruction loss. For geometric stability, we include an inverse-depth loss when depth supervision is available:

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \| D _ { \mathrm { p r e d } } ^ { - 1 } - D _ { \mathrm { g t } } ^ { - 1 } \| _ { 1 } .\tag{24}
$$

We further regularize the angular–depth position residuals,

$$
\mathcal { L } _ { \Delta } = \mathrm { m e a n } \big ( \mathrm { m a x } ( | \Delta _ { \theta \phi z } | - \delta , 0 ) \big ) ,\tag{25}
$$

which allows small local corrections but discourages large deviations from the depth-initialized scaffold. The final objective is

$$
{ \mathcal { L } } _ { \mathrm { G S } } = { \mathcal { L } } _ { \mathrm { t g t } } + \lambda _ { \mathrm { s r c } } { \mathcal { L } } _ { \mathrm { s r c } } + \lambda _ { \mathrm { d e p t h } } { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { \Delta } { \mathcal { L } } _ { \Delta } .\tag{26}
$$

This objective matches the intended role of Stage II: the scaffold should be faithful to the input panorama, stable as geometry, and directly renderable from novel viewpoints.

## 2.4 Stage III: Gaussian-Conditioned Video Rendering

Stage III acts as a renderer over the Gaussian scaffold. Given a user-specified camera trajectory $\{ \pi _ { t } \} _ { t = 1 } ^ { T }$ , the Stage II scaffold is first rendered by 3D Gaussian splatting into a controllable conditioning stream,

$$
\hat { V } _ { t } = \operatorname { R e n d e r } ( \mathcal G , \pi _ { t } ) , \quad \quad \hat { V } _ { 1 : T } = \{ \hat { V } _ { t } \} _ { t = 1 } ^ { T } .\tag{27}
$$

The rendered frames inherit the key advantages of the explicit scaffold: they follow the requested camera motion, preserve a persistent scene layout, and remain spatially anchored over long trajectories. However, they are still imperfect as final observations. A feed-forward panoramic Gaussian scaffold may contain floaters in depth-ambiguous regions, grazing-angle aliasing on floors and ceilings, disocclusion holes when the camera moves away from the input pose, and residual temporal splatting artifacts under sub-pixel motion. Stage III converts this dense but imperfect RGB stream $\hat { V } _ { 1 : T }$ into the final video $V _ { 1 : T } = \{ V _ { t } \} _ { t = 1 } ^ { T }$

![](images/097250f9e3fea757f1d6c4cad7f4d91d8a768c69b9e1248070d76e14158e6996.jpg)  
Figure 5 Stage III Real Time Rendering. The rendered image latents are passed through a causal auto-rergessive DiT to generate clean images. We use MemRoPE to manage KV Cache for infinite exploration.

This design closes the three-stage factorization. Stage I has already completed the missing field of view, and Stage II has already anchored that completion into an explicit 3DGS scene. Stage III therefore does not need to hallucinate a world from scratch. Instead, it enhances appearance and repairs local rendering artifacts while respecting the camera motion and scene layout implied by the scaffold. In this sense, MoVerse differs from purely implicit video world models: long-range spatial memory is stored in ${ \mathcal { G } } _ { : }$ while the video model performs local, high-quality observation rendering.

## 2.4.1 Bidirectional conditional teacher.

We first adapt a modern text-to-video diffusion transformer, initialized from Wan2.1-T2V-1.3B [36], into a bidirectional Gaussian-conditioned video renderer. The teacher models

$$
p ( V _ { 1 : T } \mid \hat { V } _ { 1 : T } , \tau ) ,\tag{28}
$$

where $\tau$ is a text prompt. The prompt provides a weak semantic prior, while the Gaussian-rendered RGB stream is the dominant spatial condition. This prevents the video generator from behaving as a free textto-video model: its role is to translate the scaffold rendering into a cleaner observation, not to replace the scaffold.

Let $z \in \mathbb { R } ^ { C \times T \times H \times W }$ be the VAE latent of the target video $V _ { 1 : T }$ and $c \in \mathbb { R } ^ { C \times T \times H \times W }$ be the VAE latent of the Gaussian-rendered conditioning video $\hat { V } _ { 1 : T }$ . At diffusion time t, the noisy target latent is

$$
z _ { t } = ( 1 - \sigma _ { t } ) z + \sigma _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{29}
$$

We concatenate the noisy target tokens and conditioning tokens as

$$
\tilde { z } = [ z _ { t } \parallel c ] .\tag{30}
$$

The two halves use shared rotary positional indices: every conditioning token receives the same spatial– temporal position code as its aligned target token. This shared-RoPE design makes the condition and target

collide at the same location in the transformer coordinate system, encouraging the model to treat c as a dense aligned rendering condition rather than as a separate video placed at an offset position.

The teacher is trained with a flow-matching objective, evaluated only on the target half of the concatenated tokens:

$$
\mathcal { L } _ { \mathrm { t e a c h e r } } = \mathbb { E } _ { z , c , \tau , \sigma _ { t } , \epsilon } \left[ \| v _ { \theta } ( \tilde { z } , \tau , t ) - ( \epsilon - z ) \| _ { 2 } ^ { 2 } \right] ,\tag{31}
$$

where $v _ { \theta }$ denotes the diffusion transformer prediction. The bidirectional teacher can attend to the full clip, so it serves as a quality model for Gaussian-conditioned video rendering rather than as the final deployment model.

## 2.4.2 Causal student for streaming interaction.

The bidirectional teacher is unsuitable for interactive roaming because it requires access to the full video window and performs multi-step generation over all frames jointly. We therefore distill it into a causal autoregressive student. The student generates one latent block at a time, conditions on the current Gaussian-rendered frame block, and attends to recently generated history through a local key–value cache. This converts the teacher into a bounded-latency renderer that can stream frames as the user changes the camera trajectory.

The distillation follows the Self Forcing [60] and DMD [61] principle. During training, the student is rolled out autoregressively and is exposed to its own generated context, so it learns to remain stable under the same causal conditions used at inference. A short bidirectional-to-causal warm-up [16] initializes the student before distribution matching. Within distribution matching, we adopt RAVEN [62]: each self-rollout is repacked into a teacher-forcing sequence of clean endpoints and noisy intermediates, so the DMD gradient also flows through the clean half’s QKV projections and supervises how the student encodes its own past context into the KV cache used by future blocks. The resulting model trades global bidirectional refinement for low-latency local refinement. This trade-off is appropriate for MoVerse because the explicit scaffold, not the video model cache, carries long-range scene consistency.

## 2.4.3 Training data mixture.

Training pairs are organized as triplets $( V _ { 1 : T } ^ { \mathrm { g t } } , \hat { V } _ { 1 : T } , \tau )$ , where $V _ { 1 : T } ^ { \mathrm { g t } }$ is a target video, $\hat { V } _ { 1 : T }$ is a paired geometryconditioned rendering, and τ is a caption. The purpose of the mixture is not to simply aggregate datasets, but to expose the renderer to progressively more realistic versions of the signal it will receive at deployment: cameraaligned but incomplete geometric projections, genuine Gaussian splatting artifacts, and finally renderings from panorama-conditioned scaffolds.

The first family uses real videos with known or recoverable camera motion to synthesize trajectory-aligned proxy renderings from a single reference view. We estimate depth for the reference frame, lift it once into a world-space point set, and reproject it through the remaining cameras. The resulting conditioning videos preserve the target trajectory and coarse scene layout, but they naturally contain missing regions, broken boundaries, and many-to-one warp artifacts. These examples are useful precisely because they isolate the basic inpainting problem caused by camera translation: the renderer must complete newly exposed regions while respecting the motion and layout encoded by the geometric projection.

The second family replaces point reprojection with rendered Gaussian scaffolds constructed from sparse real-video keyframes. For each scene, several keyframe-level Gaussian predictions are brought into a common metric frame, merged, and rendered along the original camera path. These conditioning videos are no longer generic warps; they are actual splatting renderings, and therefore expose the model to the failure modes of Gaussian representations: depth-dependent floaters, imperfect opacity accumulation, anisotropic support errors, and aliasing on grazing surfaces. This family teaches the video model to act as a renderer for explicit Gaussian geometry rather than merely as a video inpainting model.

The third family matches the MoVerse interface most closely. Instead of starting from a NFOV image, we start from panoramic observations, construct a Gaussian scaffold with the same panorama-to-Gaussian pathway used by Stage II, and render that scaffold along immersive camera trajectories. We build such pairs from UE-rendered panoramas and additional panoramic sources, including LayerPano3D [3], Matterport3D [52], and Polyhaven HDRIs [53]. These samples provide the strongest supervision for the deployment-time interface because the condition $\ddot { V } _ { 1 : T }$ is produced by rendering a panorama-conditioned scaffold. They are complemented by the first two families, which provide broader scene diversity and denser coverage of generic disocclusion and Gaussian-rendering artifacts.

The mixture therefore follows a representation-driven progression:

trajectory-aligned geometric proxy → explicit Gaussian render → panorama-conditioned Gaussian render.

(32)

This curriculum first teaches robust artifact repair under camera motion, then specializes the renderer to Gaussian splatting artifacts, and finally adapts it to the Stage II-to-Stage III interface used by MoVerse at deployment.

## 2.4.4 Real-time inference.

At deployment, the distilled student runs causally with K = 1 latent frame per autoregressive block. Under the Wan VAE temporal stride, each latent block corresponds to four pixel-space frames. We use a two-step denoising schedule and a MemRoPE-style local key–value cache [63]: one sink frame, one long-term EMA memory token, and a sliding window of three local frames, with online RoPE indexing that keeps temporal positions inside the trained range over long rollouts. This cache maintains short-horizon temporal continuity, while the Stage II scaffold anchors long-horizon spatial consistency. The standard VAE decoder is replaced by a TAEHV decoder [64] for faster streaming.

These choices deliberately trade global bidirectional refinement for bounded-latency local refinement. Since every output block is conditioned on the current scaffold rendering, the model does not need to remember the entire world in its hidden state. In our deployment configuration, this causal renderer reaches 8 FPS end-to-end roaming of the scene on a single NVIDIA RTX 4090 GPU.

## 3 Results

In this section, we present qualitative results for MoVerse across the full pipeline and its three stages.

## 3.1 Full pipeline results

Fig. 6 shows end-to-end results produced from a single NFOV image. Each example contains the input image, the gravity-aligned ERP panorama from Stage I, Gaussian-rendered conditioning frames from the Stage II 3D Gaussian scaffold, and the final observations generated by the Stage III causal autoregressive renderer under the same camera trajectory. The results illustrate the complementary roles of the explicit scaffold and the learned renderer: the 3D Gaussian scaffold provides camera-controllable spatial structure, while the Gaussian-conditioned video renderer improves perceptual quality and temporal continuity.

## 3.2 Stage I: Panoramic Generation

Fig. 7 evaluates the Stage I panoramic generation module. Given a perspective NFOV input, Stage I synthesizes a gravity-aligned ERP panorama that can be reprojected into multiple perspective views. We include seam-crossing views to examine whether the generated panorama remains coherent under the horizontal S1 topology of the ERP domain.

## 3.3 Stage II: Gaussian Scene Generation

Fig. 8 visualizes trajectories rendered directly from the Stage II 3D Gaussian scaffold. The scaffold produces camera-controllable Gaussian-rendered observations that preserve the global layout of the completed panorama.

## 3.4 Stage III: Gaussian-Conditioned Video Rendering

Fig. 9 shows outputs from the autoregressive video renderer. The renderer translates Gaussian-rendered conditioning frames into final video observations while streaming along user-specified camera trajectories. Across the shown trajectories, it enhances the scaffold renderings into visually coherent observations while preserving the scene layout imposed by the explicit representation.

![](images/c00e32ec6799c0d51b6e4c90ef8627435bbad0b9fc622e384b1608433fd001bf.jpg)  
Figure 6 Full pipeline results. For each scene, the first column shows the input NFOV image and the Stage I gravity-aligned ERP panorama. The remaining columns show Stage II Gaussian scaffold renderings and the corresponding Stage III autoregressive video-rendering outputs along the same camera trajectory.

![](images/a9d6f6d9c1d18558bc38d753cc25d2aa0ed7a0a4db81804713d9feb210abf53d.jpg)  
Figure 7 Stage I panoramic generation results. Each example shows a perspective NFOV input, the completed gravityaligned ERP panorama, and perspective views rendered from the panorama, including seam-crossing views.

![](images/eabaece9eacdcb297dc8cfdacbc7a1392167d7f6ebe239940320b5e621fdf1ef.jpg)  
Figure 8 Stage II Gaussian scaffold rendering results. For each scene, the first column shows the input condition, and the remaining columns show novel-view frames rendered directly from the 3D Gaussian scaffold along two camera trajectories.

![](images/05972f7affec67019a2b0e22f4cda58e3a2c1ec2740e25621c81d9097881afb6.jpg)  
Figure 9 Stage III autoregressive rendering results. For each scene, the first column shows the Gaussian-rendered input condition, and the remaining columns show frames generated by the causal autoregressive renderer along two camera trajectories.

![](images/0cfbc64a007ee04abd00c8681bb494c5f9a27a6546dcf016cea32b043e212536.jpg)  
Figure 10 Bidirectional conditional teacher results. For each scene, the first column shows the Gaussian-rendered input condition, and the remaining columns show frames synthesized by the bidirectional Gaussian-conditioned video renderer along two camera trajectories.

## 3.5 Bidirectional Conditional Teacher Results

Fig. 10 shows results from the bidirectional conditional teacher. The teacher is implemented as a bidirectional Gaussian-conditioned video renderer and can attend to the full temporal window, serving as a quality model for Gaussian-conditioned video rendering. These results provide the visual target used to guide distillation of the causal autoregressive student.

## 4 Discussion

MoVerse explores a hybrid route between explicit 3D reconstruction and implicit video world modeling. Its central design choice is to store long-range spatial memory in a persistent panoramic 3D Gaussian scaffold, while using a causal video renderer only for local, high-fidelity observation synthesis. This separation is useful in practice: the scaffold fixes the camera trajectory and scene layout over long horizons, and the video model can focus on repairing splatting artifacts, filling small disocclusions, and improving perceptual realism. The three-stage factorization also makes the system modular. Improvements in panorama generation, feed-forward Gaussian prediction, or video distillation can be incorporated independently without changing the overall interface between stages.

At the same time, this factorization exposes several limitations. First, the final world is bounded by the quality of Stage I panoramic completion. If the generated panorama introduces semantically inconsistent rooms, incorrect horizon structure, or implausible content behind the camera, Stage II will faithfully lift those errors into the scaffold, and Stage III may render them more convincingly rather than correct them. Second, the Gaussian scaffold remains an approximate geometric representation. Depth ambiguity, thin structures, reflective surfaces, transparent objects, and regions near the poles of the ERP domain can still produce floaters, holes, or unstable opacity accumulation under large translations. Third, the causal video renderer trades global clip-level optimization for streaming latency. Although the explicit scaffold provides long-range spatial consistency, the renderer may still introduce short-term texture drift, over-smoothing, or delayed correction when the conditioning render is severely degraded. Finally, the current system depends on a staged training pipeline and multiple data sources, including canonical panoramas, depth-supervised Gaussian prediction, and paired scaffold-render/video data; simplifying this supervision remains an important direction for broader deployment.

Future work should strengthen the feedback between stages rather than treating them as a one-way pipeline. For example, uncertainty from the panoramic generator could guide Gaussian density allocation, scaffold rendering errors could trigger local panorama or geometry refinement, and the video renderer could expose consistency signals back to the scaffold. Another promising direction is to extend the static-scene assumption toward dynamic objects and editable worlds, where the persistent scaffold must support object-level manipulation as well as camera motion. Overall, MoVerse shows that real-time video world modeling need not choose between explicit geometry and generative video. A dense panoramic Gaussian scaffold can provide durable spatial memory and controllability, while a distilled autoregressive renderer supplies interactive visual quality, enabling single-image world creation with real-time roaming.

## Acknowledgements

We thank Jie Ma and Ben Hu for their valuable assistance with data collection, curation, and preprocessing.

## References

[1] H.-X. Yu, H. Duan, J. Hur, K. Sargent, M. Rubinstein, W. T. Freeman, F. Cole, D. Sun, N. Snavely, J. Wu et al., “Wonderjourney: Going from anywhere to everywhere,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6658–6667.

[2] H.-X. Yu, H. Duan, C. Herrmann, W. T. Freeman, and J. Wu, “Wonderworld: Interactive 3d scene generation from a single image,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5916–5926.

[3] S. Yang, J. Tan, M. Zhang, T. Wu, G. Wetzstein, Z. Liu, and D. Lin, “Layerpano3d: Layered 3d panorama for hyper-immersive scene generation,” in Proceedings of the special interest group on computer graphics and interactive techniques conference conference papers, 2025, pp. 1–10.

[4] K. Zheng, Y. Fan, J. Gu, Z. Xu, X. He, and X. E. Wang, “Self-evolving 3d scene generation from a single image,” arXiv preprint arXiv:2512.08905, 2025.

[5] H. Team, Z. Wang, Y. Liu, J. Wu, Z. Gu, H. Wang, X. Zuo, T. Huang, W. Li, S. Zhang et al., “Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels,” arXiv preprint arXiv:2507.21809, 2025.

[6] M.-A. Schneider, L. Höllein, and M. Nießner, “Worldexplorer: Towards generating fully navigable 3d scenes,” in Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 2025, pp. 1–11.

[7] Z. Yang, W. Ge, Y. Li, J. Chen, H. Li, M. An, F. Kang, H. Xue, B. Xu, Y. Yin et al., “Matrix-3d: Omnidirectional explorable 3d world generation,” arXiv preprint arXiv:2508.08086, 2025.

[8] T. Shen, S. Bahmani, K. He, S. G. Srinivasan, T. Cao, J. Ren, R. Li, Z. Wang, N. Sharp, Z. Gojcic et al., “Lyra 2.0: Explorable generative 3d worlds,” arXiv preprint arXiv:2604.13036, 2026.

[9] T. HY-World, C. Cao, X. Zuo, Z. Wang, Y. Zhang, J. Wu, Z. Liu, Y. Gong, Y. Liu, B. Yuan et al., “Hy-world 2.0: A multi-modal world model for reconstructing, generating, and simulating 3d worlds,” arXiv preprint arXiv:2604.14268, 2026.

[10] G. DeepMind, “Genie 3: A new frontier for world models,” https://deepmind.google/blog/ genie-3-a-new-frontier-for-world-models/, 2025.

[11] WorldLabs, “RTFM: A real-time frame model,” https://www.worldlabs.ai/blog/rtfm, 2025.

[12] J. Yu, J. Bai, Y. Qin, Q. Liu, X. Wang, P. Wan, D. Zhang, and X. Liu, “Context as memory: Scene-consistent interactive long video generation with memory retrieval,” in Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 2025, pp. 1–11.

[13] X. He, C. Peng, Z. Liu, B. Wang, Y. Zhang, Q. Cui, F. Kang, B. Jiang, M. An, Y. Ren et al., “Matrix-game 2.0: An open-source real-time and streaming interactive world model,” arXiv preprint arXiv:2508.13009, 2025.

[14] T. HunyuanWorld, “Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency,” arXiv preprint, 2025.

[15] X. Mao, S. Lin, Z. Li, C. Li, W. Peng, T. He, J. Pang, M. Chi, Y. Qiao, and K. Zhang, “Yume: An interactive world generation model,” arXiv preprint arXiv:2507.17744, 2025.

[16] Y. Hong, Y. Mei, C. Ge, Y. Xu, Y. Zhou, S. Bi, Y. Hold-Geoffroy, M. Roberts, M. Fisher, E. Shechtman et al., “Relic: Interactive video world model with long-horizon memory,” arXiv preprint arXiv:2512.04040, 2025.

[17] R. Team, Z. Gao, Q. Wang, Y. Zeng, J. Zhu, K. L. Cheng, Y. Li, H. Wang, Y. Xu, S. Ma et al., “Advancing open-source world models,” arXiv preprint arXiv:2601.20540, 2026.

[18] Z. Wang, Z. Liu, J. Li, K. Huang, B. Xu, F. Kang, M. An, P. Wang, B. Jiang, Y. Wei et al., “Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory,” arXiv preprint arXiv:2604.08995, 2026.

[19] H. Zhu, H. Liu, Y. Zhao, T. Ye, J. Chen, J. Yu, T. He, S. Han, and E. Xie, “Sana-wm: Efficient minute-scale world modeling with hybrid linear diffusion transformer,” arXiv preprint arXiv:2605.15178, 2026.

[20] J. Wang, L. Ye, T. Lu, J. Xiao, J. Zhang, Y. Guo, X. Liu, R. Chellappa, C. Peng, A. Yuille et al., “Evoworld: Evolving panoramic world generation with explicit 3d memory,” arXiv preprint arXiv:2510.01183, 2025.

[21] X. Ren, T. Shen, J. Huang, H. Ling, Y. Lu, M. Nimier-David, T. Müller, A. Keller, S. Fidler, and J. Gao, “Gen3c: 3d-informed world-consistent video generation with precise camera control,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 6121–6132.

[22] M. Yu, W. Hu, J. Xing, and Y. Shan, “Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2025, pp. 100–111.

[23] H. Liu, Y. Zhou, Z. Wang, Z. Xu, Z. Peng, J. Ma, J. Liang, S. He, and J. Li, “Mocam: Unified novel view synthesis via structured denoising dynamics,” arXiv preprint arXiv:2605.12119, 2026.

[24] P. Wang, L. Chen, Z. Ma, Y. Guo, G. Zhang, and L. Zhang, “One2scene: Geometric consistent explorable 3d scene generation from a single image,” arXiv preprint arXiv:2602.19766, 2026.

[25] I. Team, D. Shen, G. Zhang, H. Liu, H. Ji, J. Liu, J. Guo, N. Wang, S. Pan, W. Pan et al., “Inspatio-worldfm: An open-source real-time generative frame model,” arXiv preprint arXiv:2603.11911, 2026.

[26] T. Wu, C. Zheng, and T.-J. Cham, “Panodiffusion: 360-degree panorama outpainting via diffusion,” in ICLR, 2024.

[27] H. Feng, D. Zhang, X. Li, B. Du, and L. Qi, “Dit360: High-fidelity panoramic image generation via hybrid training,” arXiv preprint arXiv:2510.11712, 2025.

[28] D. Zheng, C. Zhang, X.-M. Wu, C. Li, C. Lv, J.-F. Hu, and W.-S. Zheng, “Panorama generation from nfov image done right,” in CVPR, 2025, pp. 21 610–21 619.

[29] X. Yuan, S. Tang, K. Li, and P. Wang, “Camfreediff: camera-free image to panorama generation with diffusion model,” in CVPR, 2025, pp. 16 408–16 417.

[30] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao et al., “Anysplat: Feed-forward 3d gaussian splatting from unconstrained views,” ACM Transactions on Graphics (TOG), vol. 44, no. 6, pp. 1–16, 2025.

[31] J. Kim and S. Lee, “Vg3t: Visual geometry grounded gaussian transformer,” arXiv preprint arXiv:2512.05988, 2025.

[32] Z. Chen, C. Wu, Z. Shen, C. Zhao, W. Ye, H. Feng, E. Ding, and S.-H. Zhang, “Splatter-360: Generalizable 360 gaussian splatting for wide-baseline panoramic images,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 590–21 599.

[33] C. Zhang, H. Xu, Q. Wu, C. C. Gambardella, D. Phung, and J. Cai, “Pansplat: 4k panorama synthesis with feed-forward gaussian splatting,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 11 437–11 447.

[34] J. Ren, M. Xiang, J. Zhu, and Y. Dai, “Panosplatt3r: Leveraging perspective pretraining for generalized unposed wide-baseline panorama reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 28 959–28 969.

[35] L. Mescheder, W. Dong, S. Li, X. Bai, M. Santos, P. Hu, B. Lecouat, M. Zhen, A. Delaunoy, T. Fang et al., “Sharp monocular view synthesis in less than a second,” arXiv preprint arXiv:2512.10685, 2025.

[36] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[37] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.

[38] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang et al., “Hunyuanvideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.

[39] C. Zheng, T.-J. Cham, and J. Cai, “Pluralistic image completion,” in CVPR, 2019, pp. 1438–1447.

[40] Y. Wang, X. Tao, X. Shen, and J. Jia, “Wide-context semantic image extrapolation,” in CVPR, 2019, pp. 1399–1408.

[41] S. Zhao, J. Cui, Y. Sheng, Y. Dong, X. Liang, E. I. Chang, and Y. Xu, “Large scale image completion via co-modulated generative adversarial networks,” arXiv preprint arXiv:2103.10428, 2021.

[42] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10 684–10 695.

[43] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in CVPR, 2022, pp. 11 461–11 471.

[44] J. Li and M. Bansal, “Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation,” NeurIPS, vol. 36, pp. 21 878–21 894, 2023.

[45] M. Feng, J. Liu, M. Cui, and X. Xie, “Diffusion360: Seamless 360 degree panoramic image generation based on diffusion models,” arXiv preprint arXiv:2311.13141, 2023.

[46] Y. Lu, J. Zhang, T. Fang, J.-D. Nahmias, Y. Tsin, L. Quan, X. Cao, Y. Yao, and S. Li, “Matrix3d: Large photogrammetry model all-in-one,” in CVPR, 2025, pp. 11 250–11 263.

[47] Y. Lee, K. Kim, H. Kim, and M. Sung, “Syncdiffusion: Coherent montage via synchronized joint diffusions,” NeurIPS, vol. 36, pp. 50 648–50 660, 2023.

[48] Q. Wang, W. Li, C. Mou, X. Cheng, and J. Zhang, “360dvd: Controllable panorama video generation with 360-degree video diffusion model,” in CVPR, 2024, pp. 6913–6923.

[49] K. Liao, X. Xu, C. Lin, W. Ren, Y. Wei, and Y. Zhao, “Cylin-painting: Seamless 360 panoramic image outpainting and beyond,” IEEE TIP, vol. 33, pp. 382–394, 2023.

[50] M. Jaderberg, K. Simonyan, A. Zisserman et al., “Spatial transformer networks,” NeurIPS, vol. 28, 2015.

[51] J. Xiao, K. A. Ehinger, A. Oliva, and A. Torralba, “Recognizing scene viewpoint using panoramic place representation,” in CVPR. IEEE, 2012, pp. 2695–2702.

[52] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niebner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3d: Learning from rgb-d data in indoor environments,” in 2017 International Conference on 3D Vision (3DV). IEEE Computer Society, 2017, pp. 667–676.

[53] Poly Haven, “Poly haven hdris,” https://polyhaven.com/hdris, accessed: December 2025.

[54] Y. Zhang, S. Song, P. Tan, and J. Xiao, “Panocontext: A whole-room 3d context model for panoramic scene understanding,” in ECCV. Springer, 2014, pp. 668–686.

[55] C. Zou, A. Colburn, Q. Shan, and D. Hoiem, “Layoutnet: Reconstructing the 3d room layout from a single rgb image,” in CVPR, 2018, pp. 2051–2059.

[56] C. Sun, C.-W. Hsiao, M. Sun, and H.-T. Chen, “Horizonnet: Learning room layout with 1d representation and pano stretch data augmentation,” in CVPR, 2019, pp. 1047–1056.

[57] Z. Jiang, Z. Xiang, J. Xu, and M. Zhao, “Lgt-net: Indoor panoramic room layout estimation with geometry-aware transformer network,” in CVPR, 2022, pp. 1654–1663.

[58] C. Sun, M. Sun, and H.-T. Chen, “Hohonet: 360 indoor holistic understanding with latent horizontal features,” in CVPR, 2021, pp. 2573–2582.

[59] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. M. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang, M. Savva, Y. Zhao, and D. Batra, “Habitat-matterport 3d dataset (HM3d): 1000 large-scale 3d environments for embodied AI,” in Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021. [Online]. Available: https://arxiv.org/abs/2109.08238

[60] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman, “Self forcing: Bridging the train-test gap in autoregressive video diffusion,” Advances in Neural Information Processing Systems, vol. 38, pp. 167 283–167 308, 2026.

[61] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park, “One-step diffusion with distribution matching distillation.” in CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6613–6623.

[62] Y. Lu, R. Zuo, and J. Deng, “Raven: Real-time autoregressive video extrapolation with consistency-model grpo,” arXiv preprint arXiv:2605.15190, 2026.

[63] Y. Kim, Q. Hu, C.-C. J. Kuo, and P. A. Beerel, “Memrope: Training-free infinite video generation via evolving memory tokens,” arXiv preprint arXiv:2603.12513, 2026.

[64] O. Boer Bohan, “Taehv: Tiny autoencoder for hunyuan video,” https://github.com/madebyollin/taehv, 2025.