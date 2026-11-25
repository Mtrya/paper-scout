<!-- page 1 of 22 -->

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">•∗Equal contribution. †Project lead.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">•‡Corresponding author, E-mail: bjdxtanghao@gmail.com.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">• Yue Huang, Tianshan Zhang, Zeyu Zhang, and Hao Tang are with the School of Computer Science, Peking University, Beijing 100871, China.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">• Jayeon Kang and Sungjoon Choi are with the Department of Artificial Intelligence, Korea University, Seoul 02841, Korea.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">• Akide Liu is with the Department of Data Science and AI, Monash University, Melbourne VIC 3800, Australia.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">Dacheng Tao is with the College of Computing and Data Science, Nanyang Technological University, Singapore.</span></small>

arXiv:2611.29897v1 [cs.RO] 23 Nov 2026

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

1

# Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control

Yue Huang∗, Tianshan Zhang∗, Zeyu Zhang∗†, Jayeon Kang, Akide Liu, Sungjoon Choi, Dacheng Tao, Fellow, IEEE, Hao Tang‡

**Abstract**—Long-horizon, real-time controllable motion generation is essential for interactive animation and humanoid robotics, where an agent must respond to changing instructions and spatial goals while moving. Existing autoregressive diffusion methods generate motion incrementally, but bounded attention caches limit access to distant history, while repeated denoising introduces substantial inference overhead. We present **Motion Mamba 2.0**, a Mamba-based root–body diffusion architecture that integrates local attention, resumable recurrent states, and persistent content memory (PCM) for long-horizon controllable motion generation. PCM transforms compressed motion history from Mamba recurrent states, together with state changes and text-conditioned features, into dynamically updated content slots accessible to local attention. To enable efficient inference, we further develop an execution scheme tailored to this architecture, combining CUDA Graph replay, fused kernels, and execution-flow simplification while preserving the canonical history update. Evaluations on retargeted HiPHI demonstrate accurate terminal control over rollouts of up to 39.52 s, while short-horizon comparisons on BONES Seed and retargeted HumanML3D show that the model also retains strong spatial control and text–motion matching. On an RTX 4090, the optimized execution reduces median complete-window latency from 1685.35 to 276.14 ms, achieving a median paired speedup of 6.138× while preserving identical outputs and committed states in the tested rollouts. Moreover, we conduct extensive real-world evaluation on the Unitree G1, demonstrating sequential text-guided action composition, expressive motion, and interactive locomotion through keyboard and mouse control. Code: [https://github.com/AIGeeksGroup/MotionMamba2.0](https://github.com/AIGeeksGroup/MotionMamba2.0). Website: [https://aigeeksgroup.github.io/MotionMamba2.0](https://aigeeksgroup.github.io/MotionMamba2.0).

**Index Terms**—Long-Horizon, Interactive, Human Motion, Humanoid Control, Efficient, Mamba, Diffusion.

![Image block](doc:a492382/tier:basic/page:1/block:14)

Fig. 1: Motion Mamba 2.0. Continuous text-guided motion generation for long-horizon humanoid control on the Unitree G1.

## 1 INTRODUCTION

ENERATING long-horizon motion from changing instructions is a central problem in interactive animation and humanoid robotics. A useful generator must interpret

an action description, accommodate spatial constraints, and continue from the motion already produced. These requirements extend beyond synthesizing an isolated clip: a user may change direction during walking, request an upperbody action, or specify a future target while the current motion is still being executed. The resulting references must be available quickly enough for interactive use and remain compatible with a downstream tracking system. Unlike offline text-to-motion synthesis, the future instruction sequence need not be known when generation begins. An interactive generator must therefore preserve useful context

<!-- page 2 of 22 -->

2

![Image block](doc:a492382/tier:basic/page:2/block:2)

Fig. 2: Interactive motion generation with Motion Mamba 2.0. Text instructions specify the requested actions, while spatial controls constrain root motion and selected joint positions. From left to right, the panels show root trajectories, root waypoints, full-body pose sequences, and selected joint positions. Blue-to-gray poses indicate temporal progression, while red overlays visualize trajectories, waypoints, and joint locations. The panels show Viser motion references and a rollout tracked with SONIC.

while incorporating newly arriving commands, rather than simply replaying a completed trajectory.

Diffusion models produce diverse text-conditioned human motion [1], [2], [3]. Recent methods extend this formulation through temporal composition, autoregressive prediction, and direct spatial conditioning [4], [5], [6], [7]. Temporal composition allows multiple descriptions to define a longer sequence, while causal latent models and hierarchical motion tokens support incremental synthesis [8], [9]. Spatially constrained methods further specify where selected joints or the character should move [7], [10], [11], [12]. In particular, generating one window at a time provides a practical interface for online instructions. However, two difficulties remain closely connected. A bounded attention cache discards older motion features, whereas an unbounded cache increases storage and computation as the sequence grows. Moreover, repeated denoising of short windows can incur substantial execution overhead even when each individual model evaluation is small. An interactive system therefore requires both a suitable representation of motion history and an efficient implementation of its repeated updates. These requirements concern different aspects of the problem: reducing the number of sampling steps does not extend the history available to the model, and a compact historical state alone does not eliminate denoising overhead.

State-space models provide an alternative to storing every historical token by summarizing a sequence in a recurrent state [13], [14]. Their use in motion generation has shown the value of efficient temporal modeling [15], [16]. Recent studies also combine Mamba with autoregressive diffusion or use recurrent temporal representations for sustained and interactive motion [17], [18], [19]. Nevertheless, a compressed recurrent state and an attention cache expose history differently: the former carries evolving context, while the latter permits direct retrieval of explicit features. Simply placing both branches in a network does not make the recurrent history directly accessible to attention. This distinction

motivates a learned interface between accumulated recurrent motion state and attention-accessible content, rather than relying on either mechanism alone. Diffusion introduces an additional requirement: successive noisy candidates describe alternatives for the same output window, not successive moments of the accepted motion stream. Persisting those intermediate candidates would mix sampling iterations with physical time. We therefore treat the boundary between prediction and history commitment as part of the model’s streaming formulation.

We propose Motion Mamba 2.0, an autoregressive diffusion model for long-horizon, real-time controllable G1 motion. The model follows a root–body decomposition [7]: explicit root predictions determine local trajectory features for body generation. Within both denoisers, local attention is combined with a resumable Mamba branch. Our persistent content memory (PCM) learns to map compressed Mamba state summaries, successive state changes, and text-conditioned features into dynamically updated slots accessible to local attention. The slots complement recent motion features in the bounded cache and aim to support terminal spatial constraint satisfaction and commandconditioned generation over extended rollouts; they are not designed to reproduce arbitrary early instructions verbatim. Recurrent states, cached features, and slots are committed only after a clean motion window is accepted; intermediate denoising candidates read the same immutable history snapshot. The fixed-size recurrent state, cache, and PCM slots bound stored context without guaranteeing retention of every detail from arbitrarily distant motion. The acceptedhistory separation also guides efficient inference. Readonly denoising avoids materializing unused state proposals, selective fusion reduces intermediate memory traffic, and CUDA Graphs reduce repeated host dispatch [20], [21]. These changes preserve root-to-body ordering and the canonical clean-history update. Our objective is to accelerate the same sampling computation, complementing methods that

<!-- page 3 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

3

obtain faster generation through compact representations or altered sampling procedures [3], [22], [23]. We measure their contributions separately, since the effect of local fusion differs from that of graph replay. The complete scheme generates a 52-frame window in 276.14 ms at the median on an RTX 4090; this steady-state generation time is distinct from the delay between a user command and its visible execution.

We evaluate the method on three G1 motion benchmarks under a matched 100,000-update training budget (App. C). We first examine short- and long-horizon generation on retargeted HiPHI [24], followed by short-horizon comparisons on BONES Seed [25] and retargeted HumanML3D [26]; BONES Seed retains its officially provided G1 trajectories. The results demonstrate accurate control over extended rollouts together with strong short-horizon performance, while architectural ablations examine the contributions of recurrent state and history-dependent content memory. Finally, we conduct realworld evaluation on a physical Unitree G1, which demonstrates sequential text-guided action composition, expressive motion, and interactive locomotion through keyboard and mouse control.

In summary, our contributions are threefold:

We propose **Motion Mamba 2.0**, a Mamba-based autoregressive diffusion model with persistent content memory, connecting accepted recurrent history to local attention for long-horizon, real-time controllable motion generation.

We develop and evaluate state-preserving inference optimizations for this architecture, combining execution-flow simplification, fusion kernel, and CUDA Graph replay. On an RTX 4090, the complete optimization achieves a 6.138× median paired speedup.

We evaluate the proposed framework on a physical Unitree G1 through multi-stage text-guided action composition and interactive keyboard/mouse control, demonstrating continuous motion execution under changing instructions and spatial commands. The system achieves 94% full-task success across 100 trials without observed falls.

A preliminary version of this work appeared in our ECCV 2024 conference paper, Motion Mamba [15]. The present manuscript substantially extends the conference version in both methodology and experimental evaluation in the following six aspects. (1) We reformulate the original text-conditioned latent diffusion framework into an autoregressive root–body diffusion architecture for long-horizon humanoid motion generation. Instead of synthesizing a motion sequence as a single clip, the new framework generates successive 52-frame windows while accommodating changing text instructions and time-indexed spatial constraints. (2) We redesign the temporal modeling architecture beyond the original Hierarchical Temporal Mamba (HTM) and Bidirectional Spatial Mamba (BSM) blocks. The new denoisers integrate resumable **Mamba-3 recurrence with local attention and persistent content memory (PCM)**. PCM converts accepted recurrent-state summaries and their changes into gated content slots, providing attentionaccessible representations of compressed motion history alongside the sliding cache. (3) We introduce an accepted-

only history update mechanism for autoregressive diffusion. Intermediate noisy predictions read an immutable historical snapshot, whereas the attention cache, recurrent states, and PCM slots are updated together only after a clean motion window is accepted. This separates diffusion sampling iterations from the temporal evolution of generated motion. (4) We develop a hardware-aware inference pipeline that combines execution-flow simplification, custom fused kernel and CUDA Graph replay across three guidance branches. (5) We extend the evaluation from the original HumanML3D and KIT-ML text-to-motion benchmarks to a unified G1 motion setting involving retargeted HiPHI, BONES Seed and retargeted HumanML3D. The new protocol evaluates spatial controllability, instruction matching, motion quality, and inference efficiency over autoregressive rollouts of up to 39.52 s. (6) We further extend our evaluation to physicsbased simulation and real-world deployment **on a Unitree G1 humanoid**, assessing multi-stage text-guided actions alongside interactive spatial control via keyboard and mouse.

## 2 RELATED WORK

## 2.1 Efficient Long-Horizon Motion Generation

Text-conditioned diffusion models learn expressive motion distributions from paired language and motion data [1], [2], [3]. Their representations and sampling procedures determine both the cost of synthesis and how motion can be extended in time. Discrete motion representations offer a complementary formulation, with autoregressive token prediction in T2M-GPT [27] and masked token completion in MMM [28]. Longer sequences can be constructed by composing action-conditioned segments, extending textual context, or refining overlapping motion intervals [4], [29], [30], [31], [32], [33]. Past-conditioned diffusion and temporal motion fields further address continuity beyond isolated clips [34], [35], [36]. In particular, Lagrangian Motion Fields represent longer sequences through compact supermotions, whereas FlowMDM blends positional encodings to improve transitions between described segments [4], [36]. These approaches make temporal composition more flexible, but an interactive application additionally requires new conditions to affect a stream that has already begun.

Autoregressive diffusion and accelerated sampling support incremental synthesis with lower per-update cost [5], [6], [22], [23], [37]. Recent streaming methods introduce causal representations, historical context, or spatial conditioning to accommodate continuous generation [7], [8], [38], [39]. MOGO instead uses a hierarchical causal Transformer over quantized motion tokens for real-time, extended generation [9]. These developments show that motion representation, temporal factorization, and sampling efficiency offer complementary routes to reducing generation cost.

Controllability introduces a further requirement: trajectory and joint targets must influence future motion while preserving the current action. Guided Motion Diffusion incorporates spatial constraints through representation design and diffusion guidance [10], while OmniControl combines spatial and realism guidance to constrain different joints at different times [11]. COMET studies extended joint targets for long-term generation, while ARDY exposes text and flexible kinematic constraints through a hybrid

<!-- page 4 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

4

representation [7], [12]. We build on ARDY’s explicit-root and latent-body representation to study how recurrent state, bounded attention, and persistent content can be integrated into its denoising process. The accompanying execution optimizations target repeated inference while preserving the architecture’s sampling and history-update semantics. For a controlled comparison, all methods use independently trained G1 adaptations with the same 100,000-update budget; none uses its original pretrained release.

## 2.2 State-Space Models for Motion

Structured state-space models connect recurrent, convolutional, and continuous-time sequence processing [40], [41], [42], [43]. Mamba introduces input-dependent state updates, and subsequent models extend selective scanning to visual and multidimensional data [13], [44], [45], [46]. Mamba-2 connects structured state-space models with attention through state-space duality [47]. Mamba-3 improves the underlying sequence model through state-space discretization and rotary dynamics [14]. We use this scan as a temporal modeling component; its recurrence is not a contribution of this work.

In motion generation, state-space models have been applied to temporal and spatial dependencies, hybrid architectures, and text-conditioned synthesis [15], [16], [18], [48]. Memory-oriented variants also consider longer sequences and the retention of earlier context [49], [50]. MADM combines Mamba with autoregressive diffusion for text-to-motion generation, while Dyadic Mamba and InterMamba extend state-space modeling to interactions between people [17], [19], [51].

Cross-segment memory has also been studied in sequence models: Transformer-XL reuses earlier hidden representations through segment-level recurrence [52], while Recurrent Memory Transformer propagates dedicated memory tokens between segments [53]. Our focus is the interface between compressed recurrent motion state and attention during autoregressive diffusion, where intermediate noisy candidates and accepted motion history require separate update rules. PCM converts accepted state summaries, state changes, and text conditions into dynamically updated content slots, while a bounded attention cache preserves recent motion features. These slots evolve with accepted motion rather than remaining static, providing dynamically updated motion context without explicitly storing arbitrary past tasks for exact recall.

## 2.3 Humanoid Motion Tracking and Control

Humanoid control combines motion planning with the physical constraints of balance, actuation, and contact [54]. In physics-based character animation, DeepMimic learns control policies that imitate example motions while pursuing task objectives [55]. Learning-based systems have advanced whole-body teleoperation, human-motion imitation, and expressive tracking [56], [57], [58], [59]. Retargeting methods such as GMR [60] align human recordings with a robot embodiment, but kinematic agreement alone does not establish dynamic feasibility. Large-scale trackers such as SONIC [61] provide an execution layer for diverse motion references. This distinction between kinematic generation and physical tracking is important: accurate reference poses do not by

themselves guarantee balance or appropriate foot contact after execution.

Motion generation and physical tracking can also be coupled within a control framework. CLoSD uses a diffusion planner with simulated feedback for multi-task character control [62], while OMG studies multimodal motion generation for humanoids [63]. In contrast, our work focuses on generating continuous kinematic references from changing instructions and constraints. These references are executed by an existing tracking system, allowing us to evaluate generation accuracy and inference cost separately from the qualitative simulation and hardware demonstrations of physical execution.

## 3 DATASETS AND RETARGETING

Learning a common motion generator from human and robot recordings requires a consistent embodiment, temporal resolution, and feature representation. We combine BONES Seed [25], HumanML3D [26], and HiPHI [24] in a 29-DoF G1 representation with 414 physical features at 25 Hz. The resulting training pool contains 156,819 motion records and 588.92 stored hours, including admitted mirrored records. Fig. 3 summarizes the corpus, quantifies lower-body target alignment, and illustrates stored retargeted motions.

## 3.1 Dataset Construction

BONES Seed provides robot motion references, while HumanML3D and HiPHI contribute human motion recordings and their associated descriptions. For HiPHI, we select only the high-precision human-motion subset and exclude the object-interaction subset. We retarget the selected human body motions to G1; object trajectories and object-interaction supervision are not part of the resulting motion-generation dataset. The training pool comprises 124,658 BONES Seed records, 14,612 HumanML3D records, and 17,549 HiPHI records. We retain the source captions, mirror identities, and transformation provenance with each exported record. The validation and test manifests contain 9,998 and 12,831 additional records, respectively, and are excluded from the training pool. Counts include admitted mirrors; duration measures stored frames rather than independent capture time. The complete split statistics are provided in App. A.

## 3.2 Motion Retargeting

BONES Seed motions retain their official G1 trajectories through resampling and representation conversion. For HumanML3D and HiPHI, we use source-specific adapters built on General Motion Retargeting (GMR) [60]. HumanML3D motions are reconstructed from AMASS [64] parameters using the SMPL-H body-and-hand model [65], whereas HiPHI motions are reconstructed using BVH forward kinematics. Rest-pose transforms and pelvis-height scaling align the human skeletons with the common robot asset. Root translations and local rotations are resampled to 25 Hz before retargeting. The solver combines position and orientation objectives under joint limits, followed by temporal refinement and bounded ground and rate corrections. App. A details the asset checks and conversion procedure.

<!-- page 5 of 22 -->

(c) Retargeted G1 examples

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

5

![Chart block](doc:a492382/tier:basic/page:5/block:4)

![Chart block](doc:a492382/tier:basic/page:5/block:5)

![Image block](doc:a492382/tier:basic/page:5/block:6)

Fig. 3: Fused G1 Datasets and Retargeting Assessment. (a) Training-set composition by stored duration, including mirrors. (b) Ankle (left, logarithmic axis) and pelvis (right) mapped-target residual distributions over all 38,028 admitted human-source records; dots mark cross-clip medians. (c) Selected poses replayed from stored G1 trajectories. Metric definitions appear in App. A.

## 3.3 Retargeting Assessment

We assess mapped-target alignment over all 18,525 admitted HumanML3D and 19,503 admitted HiPHI records across the training, validation, and test splits. For each clip, we compute the 95th-percentile distance between a robot body origin and its scaled, offset human target; paired ankles use the worse side, and clips receive equal weight. Fig. 3(b) shows the ankle and pelvis distributions for the complete admitted cohorts, without selecting low-error clips. The cross-clip median ankle residuals are 1.49 cm on HumanML3D and 0.93 cm on HiPHI; the corresponding cross-clip 95th percentiles are 3.37 and 2.39 cm. Median pelvis residuals are 0.99 and 0.56 cm, respectively. These statistics quantify alignment to the mapped lower-body objectives.

The stored examples in Fig. 3(c) illustrate leg extension, stepping, and coordinated arm motion. Such pose-level evidence and mapped-target position residuals describe different aspects of retargeting: a target offset alone does not establish loss of the depicted action, and recognizable poses do not establish precise hand contact. Separately, conversion between the stored motion formats and the 414-D representation has a maximum round-trip position discrepancy of $1 . 3 1 \times 1 0 ^ { - 6 }$ m over the admitted humansource records.

## 4 THE PROPOSED METHOD

As shown in Fig. 4, Motion Mamba 2.0 combines autoregressive root–body diffusion with local attention, resumable Mamba-3 states, and persistent content memory (PCM). Local attention retrieves recent motion, while PCM supplies attention-readable content derived from the accumulated recurrent state. The local cache remains bounded, while Mamba states and PCM slots carry forward information from accepted motion windows. All historical states are frozen during denoising and committed only after a clean motion window is accepted.

## 4.1 Streaming Formulation and Motion Representation

We divide the output into windows of C tokens, with each token representing P consecutive frames. At window $t ,$ the conditions comprise a text instruction $s _ { t } ,$ spatial targets g<sub>t</sub>, and their validity mask $\mathbf { v } _ { t } .$ The memory $\mathcal { M } _ { t - }$ 1 is constructed

from the accepted motion prefix. The conditional distribution is factorized over windows:

$$
p _ { \theta } ( \mathbf { x } _ { 0 : T } \mid \mathbf { s } , \mathbf { g } , \mathbf { v } ) = \prod _ { t = 0 } ^ { T } p _ { \theta } \big ( \mathbf { x } _ { t } \mid s _ { t } , \mathbf { g } _ { t } , \mathbf { v } _ { t } ; \mathcal { M } _ { t - 1 } \big ) .\tag{1}
$$

Text specifies the desired behavior, whereas spatial conditions identify selected quantities and their target times. Targets may lie within or beyond the current output window. They provide conditioning information without requiring the model to emit the intervening future motion in advance.

We adopt the hybrid representation of ARDY [7], retaining an explicit root trajectory and compressing articulation with a causal motion codec. At each frame, the root comprises position $\mathbf { p } \in \mathbb { R } ^ { 3 }$ and heading $\psi .$ Body features include joint rotations, joint positions and velocities, and contact indicators. A token concatenates a root patch $\mathbf { r } _ { i } \in \mathbb { R } ^ { 5 P }$ with a body latent $\mathbf { b } _ { i } \in \mathbb { R } ^ { L }$

$$
\mathbf { m } _ { r } = [ \mathbf { p } ; \operatorname { c o s } \psi ; \operatorname { s i n } \psi ] , \quad \mathbf { x } _ { i } = [ \mathbf { r } _ { i } ; \mathbf { b } _ { i } ] .\tag{2}
$$

The encoder $\mathcal { E }$ maps explicit motion m to body latents. The decoder D reconstructs articulation from these latents and local root features computed by Local(·):

$$
\mathbf { b } _ { 1 : C } = \mathcal { E } ( \mathbf { m } _ { 1 : C P } ) , \quad \widehat { \mathbf { m } } _ { b , 1 : C P } = \mathcal { D } \big ( \mathbf { b } _ { 1 : C } , \operatorname { L o c a l } ( \mathbf { r } _ { 1 : C } ) \big ) .\tag{3}
$$

This representation exposes the trajectory directly to spatial control while reducing the dimensionality of articulated motion. Coordinates are expressed in the current canonical frame; the accumulated translation is retained separately to recover the world-space trajectory.

We use τ for diffusion time and ℓ for the layer index. Following the forward diffusion process [66], a clean hybrid window xt is perturbed with Gaussian noise according to a predefined diffusion noise schedule:

$$
\mathbf { z } _ { t , \tau } = \sqrt { \bar { \alpha } _ { \tau } } \thinspace \mathbf { x } _ { t } + \sqrt { 1 - \bar { \alpha } _ { \tau } } \thinspace \mathbf { \epsilon } , \quad \mathbf { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) .\tag{4}
$$

Here, $\begin{array} { r } { \bar { \alpha } _ { \tau }   =   \prod _ { s = 1 } ^ { \tau } ( 1 - \beta _ { s } ) } \end{array}$ , where $\beta _ { s }$ is the noise variance at diffusion step s. The denoiser predicts clean hybrid features. A fixed sampling schedule repeatedly refines the noisy window while its accepted-history memory remains unchanged.

## 4.2 Recurrent Root–Body Denoising

**Two-stage prediction.** Following the root–body decomposition, each diffusion evaluation first predicts the global root and then predicts articulation conditioned on that root. The

<!-- page 6 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

6

![Image block](doc:a492382/tier:basic/page:6/block:3)

Fig. 4: Overview of Motion Mamba 2.0. Conditioned on text, spatial controls, and accepted motion history, the framework generates motion autoregressively through a two-stage diffusion denoiser. At each denoising step, the root network predicts the global trajectory, whose local features condition body prediction. Both networks combine attention and Mamba-3 through a hybrid sequence mixer. After each clean motion window is accepted, the history states are updated for subsequent generation.

conditioning vector $\mathbf { c } _ { t , \tau }$ combines text, diffusion time, and the available initial heading. The root input $\mathbf { u } _ { t , \cdot } ^ { r }$ τ contains noisy hybrid features and temporally aligned controls. Valid root observations replace the corresponding noisy coordinates; body observations and their masks are supplied as aligned features. Observed future targets are embedded at their target positions, and empty future entries are masked out. The body input $\mathbf { u } _ { t , \tau } ^ { b }$ concatenates local features of the predicted root, noisy body latents, and aligned observations. The root and body networks are denoted by $\mathcal { F } _ { \theta } ^ { r }$ and ${ \mathcal { F } } _ { \theta } ^ { b } ;$

$$
\begin{aligned} { \widehat { \mathbf { r } } _ { t , 0 } } & { { } = \mathcal { F } _ { \theta } ^ { r } ( \mathbf { u } _ { t , \tau } ^ { r } , \mathbf { c } _ { t , \tau } ; \mathcal { M } _ { t - 1 } ^ { r } ) , } \\ { \widehat { \mathbf { b } } _ { t , 0 } } & { { } = \mathcal { F } _ { \theta } ^ { b } \big ( \mathbf { u } _ { t , \tau } ^ { b } , \mathbf { c } _ { t , \tau } ; \mathcal { M } _ { t - 1 } ^ { b } \big ) . } \\ \end{aligned}\tag{5}
$$

The sampler consumes the concatenated clean prediction $[ \widehat { \mathbf { r } } _ { t , 0 } ; \widehat { \mathbf { b } } _ { t , 0 } ]$ . Thus, root prediction is refreshed before body prediction at every diffusion step. The two stacks have separate parameters and layer-specific memories. This decomposition exposes global translation and heading for direct spatial conditioning, while allowing articulated motion to adapt to the predicted root trajectory. By updating the rootconditioned body input at each denoising step, the two predictions remain coupled throughout diffusion sampling rather than being combined only after generation.

**Hybrid sequence layers.** Each stack consists of diffusion transformer layers [67] with adaptive normalization and a gated residual multilayer perceptron. Inside the sequence mixer, attention and Mamba process the same normalized, condition-modulated features $\mathbf { U } _ { t , \tau } ^ { \ell } .$ Attention retrieves recent features, whereas Mamba resumes its state from the accepted prefix. Their outputs are combined by a learned layerwise gate. PCM augments the attention path as described in Sec. 4.3 and it does not replace the recurrent branch.

**Mamba-3 recurrence.** We adopt the resumable SISO Mamba-3 scan [14]. For one head, let $\mathbf { u } _ { i }$ denote the value projection, $\mathbf { B } _ { i }$ and C<sub>i</sub> the normalized state-input and readout vectors, $A _ { i } ~ < ~ 0$ the decay, $\Delta _ { i } ~ > ~ 0$ the step size, $\lambda _ { i } \in ( 0 , 1 )$

the trapezoidal coefficient, and $\theta _ { i }$ the phase parameter. The operator R rotates paired state coordinates. Omitting window, layer, and diffusion indices, the recurrence is

$$
\begin{aligned} { \phi _ { i } } & { { } = \phi _ { i - 1 } + \pi \Delta _ { i } \operatorname { t a n h } ( \mathbf { \theta } _ { i } ) , } \\ { \widetilde { \mathbf { B } } _ { i } } & { { } = \mathcal { R } ( \phi _ { i } ) \mathbf { B } _ { i } , \quad \widetilde { \mathbf { C } } _ { i } = \mathcal { R } ( \phi _ { i } ) \mathbf { C } _ { i } , } \\ { \alpha _ { i } } & { { } = \operatorname { e x p } ( \Delta _ { i } A _ { i } ) , \quad \beta _ { i } = ( 1 - \lambda _ { i } ) \Delta _ { i } \alpha _ { i } , \quad \gamma _ { i } = \lambda _ { i } \Delta _ { i } , } \\ { \mathbf { H } _ { i } } & { { } = \alpha _ { i } \mathbf { H } _ { i - 1 } + \beta _ { i } \mathbf { u } _ { i - 1 } \widetilde { \mathbf { B } } _ { i - 1 } ^ { \top } + \gamma _ { i } \mathbf { u } _ { i } \widetilde { \mathbf { B } } _ { i } ^ { \top } , } \\ { \mathbf { y } _ { i } } & { { } = \mathbf { H } _ { i } \widetilde { \mathbf { C } } _ { i } + D \mathbf { u } _ { i } . } \\ \end{aligned}\tag{6}
$$

An output gate and projection map the head outputs back to the layer width. Eq. (6) combines adjacent state inputs through the trapezoidal update and represents relative phase through accumulated rotations. It provides a mechanism for retaining evolving dynamics; its contribution to motion generation is assessed empirically. The carried state includes the recurrent matrix, accumulated phase, and previous stateinput and value vectors required to resume the scan. Its size is independent of the accepted sequence length.

## 4.3 Persistent Content Memory

The recurrent state and attention cache play complementary roles. The bounded cache directly exposes recent keys and values but evicts older features; Mamba carries a fixed-size compressed representation of accepted motion history, which is not itself an attention token. PCM learns a state-to-attention mapping from recurrent summaries, their changes, and text conditions to content slots. These slots make the accumulated motion context available to subsequent windows without claiming exact recovery of arbitrary evicted events.

**Local retrieval and static sinks.** In each layer, the cache $\mathcal { C } _ { t - 1 } ^ { \ell }$ stores projected keys, values, and validity masks from the latest W accepted windows. Let $\mathbf { E } ^ { \ell }$ be learned sink tokens, $\mathbf { U } ^ { \ell }$ the current noisy features, and $\mathbf { Q } ^ { \ell }$ their learned query projection. The functions $K ^ { \ell } ( \cdot )$ and $V ^ { \ell } ( \cdot )$ are shared by sinks,

<!-- page 7 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

7

(a) Streaming next-window generation

![Image block](doc:a492382/tier:basic/page:7/block:4)

(b) Hybrid attention–Mamba mixer

![Image block](doc:a492382/tier:basic/page:7/block:6)

Fig. 5: Streaming generation and hybrid attention–Mamba mixing. (a) Motion is generated window by window through root–body diffusion, with the history held fixed during denoising and updated only after the sampled window is accepted. (b) Base attention retrieves recent motion features, while PCM attention additionally accesses persistent content slots. Their gated residual difference is combined with the Mamba-3 output resumed from the committed recurrent state.

current tokens, and PCM slots. Historical keys and values are denoted by $\mathbf { K } _ { h } ^ { \ell }$ and $\mathbf { V } _ { h } ^ { \ell }$ . The base attention operator $\mathrm { A t t n } ^ { \ell }$ includes the output projection and token-validity mask:

$$
\begin{aligned} { \mathbf { K } _ { 0 } ^ { \ell } } & { { } = [ K ^ { \ell } ( \mathbf { E } ^ { \ell } ) ; \mathbf { K } _ { h } ^ { \ell } ; K ^ { \ell } ( \mathbf { U } ^ { \ell } ) ] , } \\ { \mathbf { V } _ { 0 } ^ { \ell } } & { { } = [ V ^ { \ell } ( \mathbf { E } ^ { \ell } ) ; \mathbf { V } _ { h } ^ { \ell } ; V ^ { \ell } ( \mathbf { U } ^ { \ell } ) ] , } \\ { \mathbf { A } _ { 0 } ^ { \ell } } & { { } = \operatorname { A t t n } ^ { \ell } ( \mathbf { Q } ^ { \ell } , \mathbf { K } _ { 0 } ^ { \ell } , \mathbf { V } _ { 0 } ^ { \ell } ) . } \\ \end{aligned}\tag{7}
$$

Current-window attention is bidirectional because the noisy window is denoised jointly. The sinks are static model parameters, distinct from history-dependent PCM slots. They remain available when the cache is short or empty; they do not store an episode’s past motion.

**Content construction and gated writing.** At layer ℓ, PCM maintains $S$ memory slots $\mathbf { \widetilde { P } } _ { t } ^ { \ell } \in \mathbb { R } ^ { S \times d }$ , initialized to zero. The projector $\Pi ^ { \ell }$ pools the accepted recurrent matrix, applies a learned projection, and normalizes the resulting state summary. We compute this summary and its difference from the previous accepted window as

$$
\mathbf { \bar { s } } _ { t } ^ { \ell } = \Pi ^ { \ell } ( \mathbf { H } _ { t } ^ { \ell } ) , \quad \mathbf { d } _ { t } ^ { \ell } = \mathbf { \bar { s } } _ { t } ^ { \ell } - \mathbf { \bar { s } } _ { t - 1 } ^ { \ell } .\tag{8}
$$

The summary captures accumulated context, while the difference describes the latest accepted transition.

Let $\mathbf { e } _ { t }$ denote the pooled embedding of the active text instruction. The write path projects this embedding into a layer-specific representation and predicts featurewise modulation parameters:

$$
\mathbf { e } _ { t } ^ { \ell } = \operatorname { L N } ( W _ { e } ^ { \ell } \mathbf { e } _ { t } + \mathbf { b } _ { e } ^ { \ell } ) , \quad f _ { c } ^ { \ell } ( \mathbf { e } _ { t } ^ { \ell } ) = [ \gamma _ { t } ^ { \ell } ; \beta _ { t } ^ { \ell } ] .\tag{9}
$$

These parameters condition the normalized state difference on the active instruction:

$$
\widetilde { \mathbf { d } } _ { t } ^ { \ell } = ( 1 + \operatorname { t a n h } \mathbf { \gamma } _ { t } ^ { \ell } ) \odot \operatorname { L N } ( \mathbf { d } _ { t } ^ { \ell } ) + \mathbf { \beta } _ { t } ^ { \ell } .\tag{10}
$$

We concatenate the state summary, the text-modulated difference, and the projected text embedding to construct the candidate slot content:

$$
\mathbf { j } _ { t } ^ { \ell } = [ \overline { { \mathbf { s } } } _ { t } ^ { \ell } ; \widetilde { \mathbf { d } } _ { t } ^ { \ell } ; \mathbf { e } _ { t } ^ { \ell } ] , \quad \mathbf { Z } _ { t } ^ { \ell } = \operatorname { L N } \bigl ( f _ { p } ^ { \ell } ( \mathbf { j } _ { t } ^ { \ell } ) \bigr ) .\tag{11}
$$

Slot normalization acts along the feature dimension.

Finally, a gate computed from the joint representation and the previous memory controls the write rate:

$$
\begin{aligned} { \mathbf { w } _ { t } ^ { \ell } } & { { } = \eta \thinspace \sigma \Big ( f _ { w } ^ { \ell } \Big ( [ \mathbf { j } _ { t } ^ { \ell } ; \operatorname { v e c } ( \mathbf { P } _ { t - 1 } ^ { \ell } ) ] \Big ) \Big ) \thinspace , } \\ { \mathbf { P } _ { t } ^ { \ell } } & { { } = ( 1 - \mathbf { w } _ { t } ^ { \ell } ) \odot \mathbf { P } _ { t - 1 } ^ { \ell } + \mathbf { w } _ { t } ^ { \ell } \odot \mathbf { Z } _ { t } ^ { \ell } . } \\ \end{aligned}\tag{12}
$$

The write-rate cap $\eta \in ( 0 , 1 ]$ bounds each gate value, limiting the contribution of candidate content to each update. Text conditions both candidate-content generation and write gating, allowing the same state transition to produce different memory updates under different instructions. The slots therefore accumulate session-specific content rather than serving as fixed learned vectors. The text condition excludes diffusion-time and initial-heading embeddings. Implementation details appear in $\mathrm { A } \mathrm { p } \mathrm { p }$ . B.

**Residual attention readout.** The current window reads $\mathbf { P } _ { t - 1 } ^ { \ell }$ without modifying it. Augmented keys $\mathbf { K } _ { P } ^ { \ell }$ and values $\mathbf { V } _ { P } ^ { \ell }$ insert projected slots immediately after the sinks in Eq. (7). Base and augmented attention use the same query $\bar { \mathbf { Q } } ^ { \ell }$ . A learned scalar $\rho _ { \ell }$ controls the attention difference, and $\zeta _ { \ell }$ controls the Mamba contribution $\mathbf { M } ^ { \ell }$

$$
\begin{aligned} { \mathbf { A } _ { P } ^ { \ell } } & { { } = \operatorname { A t t n } ^ { \ell } ( \mathbf { Q } ^ { \ell } , \mathbf { K } _ { P } ^ { \ell } , \mathbf { V } _ { P } ^ { \ell } ) , } \\ { \mathbf { Y } ^ { \ell } } & { { } = \mathbf { A } _ { 0 } ^ { \ell } + \operatorname { t a n h } ( \rho _ { \ell } ) ( \mathbf { A } _ { P } ^ { \ell } - \mathbf { A } _ { 0 } ^ { \ell } ) + \sigma ( \zeta _ { \ell } ) \mathbf { M } ^ { \ell } . } \\ \end{aligned}\tag{13}
$$

Initializing $\rho _ { \ell }$ to zero recovers the Native mixer output exactly, despite the additional slots. This differs from initializing slot values to zero, since extra keys can still change attention

<!-- page 8 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

8

normalization. PCM leaves query construction unchanged; state summaries enter through content writing rather than through a separate state-dependent query. This formulation targets usable motion context instead of explicit indexing of past instructions.

**Accepted-only state updates.** Each denoising evaluation reads an immutable snapshot. The denoiser $\mathcal { F } _ { \theta }$ may produce a temporary proposal $\widetilde { \mathcal { M } } _ { t , \tau }$ , but this proposal is not committed during sampling:

$$
( \widehat { \mathbf { x } } _ { t , 0 } , \widetilde { \mathcal { M } } _ { t , \tau } ) = \mathcal { F } _ { \theta } ( \mathbf { z } _ { t , \tau } , \mathbf { c } _ { t , \tau } ; \mathcal { M } _ { t - 1 } ) .\tag{14}
$$

After sampling, the accepted window is recentered and materialized as clean history in a canonical pass at diffusion time zero. This pass updates only accepted motion tokens. Layerwise cache append, recurrent-state replacement, and PCM writing are committed together across the root and body stacks. If the pass fails, both stacks retain their previous state. The streaming generation algorithm in App. B.2 summarizes the procedure.

The exact cache grows only to W windows, and each recurrent state and slot bank has fixed size. For a fixed output window and conditioning budget, memory and computation per new window therefore do not grow with elapsed generation time. This bound concerns computational context; it does not imply error-free motion over arbitrary durations.

## 4.4 State-Preserving Efficient Inference

Repeated diffusion denoising incurs overhead from kernel launches, intermediate memory traffic, and unnecessary state materialization. We address these costs through statepreserving execution optimizations while maintaining root– body dependencies and accepted-only history updates.

**Read-only execution.** During denoising, we avoid materializing state proposals and diagnostics that are not consumed by either the prediction or a later commit. All recurrent computation contributing to predicted motion is retained. The canonical clean-history pass still constructs and commits the complete accepted state.

**Fusion Kernel.** We fuse Mamba transition preprocessing, including the decay activation, decay floor, step-size transformation, and product $\Delta _ { i } A _ { i } ,$ , into a single traversal. Intermediate values remain in FP32 registers before outputs are written in the scan’s required layout. Compatible scale– shift and gated-residual operations are also fused in the execution variants that pass the numerical checks. The attention operator and recurrent scan are retained. Exact preprocessing and precision boundaries are given in App. B. **CUDA Graph replay.** The root and body DiT sequence stacks use separate CUDA Graphs. Within each stack, the textconditioned, spatially conditioned, and unconditional CFG branches execute serially and are captured together. Input projections, conditioning preparation, root-output processing, root-to-local conversion, and body-input construction remain outside these stack-level graphs. Stable input and state buffers are refreshed before replay. Sampling updates, random-number handling, recentering, and accepted-state commits also remain outside capture. Graphs are rebuilt when tensor layouts or parameter versions become incompatible; unsupported calls use the reference path.

Step 1. CUDA Graph

![Image block](doc:a492382/tier:basic/page:8/block:13)

Step 2. Fusion Kernel

128 elements per parallel tile

![Image block](doc:a492382/tier:basic/page:8/block:16)

Fig. 6: CUDA Graph and Kernel Fusion. Stack-level CUDA Graph replay of the three CFG branches, with separate graphs for the root and body DiT stacks, and fused Mamba transition preprocessing.

## 4.5 Learning Objective

We optimize clean-window prediction with four terms. The hybrid term $\mathcal { L } _ { \mathrm { h y b } }$ applies Smooth L1 to explicit root patches and body latents. The decoded-body term $\mathcal { L } _ { \mathrm { b o d y } }$ applies L1 to reconstructed articulation. The goal term $\mathcal { L } _ { \mathrm { g o a l } }$ applies L1 only to observed coordinates inside the generated window; later goals remain conditioning inputs until their target frames enter the output window. The consistency term L<sub>FK</sub> measures Euclidean disagreement between joint positions obtained from decoded rotations through forward kinematics and those represented directly by decoded position features. Each term is normalized over its applicable elements. Their coefficients are denoted by $\lambda _ { \mathrm { h y b } } ,   \lambda _ { \mathrm { b o d y } } ,   \lambda _ { \mathrm { g o a l } } ,$ , and λ<sub>FK</sub>:

$$
\mathcal { L } = \lambda _ { \operatorname { h y b } } \mathcal { L } _ { \operatorname { h y b } } + \lambda _ { \operatorname { b o d y } } \mathcal { L } _ { \operatorname { b o d y } } + \lambda _ { \operatorname { g o a l } } \mathcal { L } _ { \operatorname { g o a l } } + \lambda _ { \operatorname { F K } } \mathcal { L } _ { \operatorname { F K } } .\tag{15}
$$

The codec is frozen during denoiser training. Histories, generation regions, and future observations use explicit validity masks; outputs outside the generation region are excluded from the supervised reduction. The common budget used for architecture comparisons is specified in Sec. 5.

## 5 EXPERIMENTS

In this section, we evaluate Motion Mamba 2.0 from three perspectives: generation quality, architectural effectiveness, and inference efficiency. We first compare short- and long-horizon generation on retargeted HiPHI, followed by short-horizon evaluations on BONES Seed and retargeted HumanML3D in Sec. 5.2. We then isolate the contributions of recurrent modeling and persistent content memory through architectural ablations in Sec. 5.3. Finally, we evaluate whether the proposed execution optimizations reduce inference latency without altering generated outputs or committed historical states in Sec. 5.4.

<!-- page 9 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

9

TABLE 1: Short- and Long-Horizon Motion Generation on Retargeted HiPHI. CE is reported in cm, Cmd. and R@3 in percentages, and latency in ms per accepted window at batch size 1. R@3 uses 32 generated motion candidates. See App. C for details.

<table><tbody><tr><td rowspan="2">Method R</td><td colspan="4">52 frames / 1 window</td><td colspan="4">8.32 s / 4 windows</td><td colspan="4">18.72 s / 9 windows</td><td colspan="4">39.52 s / 19 windows</td></tr><tr><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L</td><td>2-FID↓ Lat.↓ R</td><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L</td><td>2-FID↓ Lat.↓ R</td><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L2</td><td>-FID↓ Lat.↓ R</td><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L</td><td>2-FID↓ Lat.↓</td></tr><tr><td>DiP [62]</td><td>57.07</td><td>60.45</td><td>40.78 25.66</td><td>0.402 73.6</td><td>62.58</td><td>66.16</td><td>31.59 11.08</td><td>0.444 68.7</td><td>77.69</td><td>80.88</td><td>20.50 12.84</td><td>0.487 68.7</td><td>96.99</td><td>99.26</td><td>18.45 12.45</td><td>0.490 66.8</td></tr><tr><td>Go-to-Zero [68]</td><td>29.14</td><td>29.53</td><td>71.79 26.27</td><td>0.480 12344.9</td><td>39.65</td><td>40.19</td><td>40.63 18.17</td><td>0.514 12564.3</td><td>56.36</td><td>59.76</td><td>32.89 13.09</td><td>0.631 12403.2</td><td>86.47</td><td>88.91</td><td>25.03 12.11</td><td>0.743 12478.5</td></tr><tr><td>DreamControl [69]</td><td>31.43</td><td>35.85</td><td>75.39 22.62</td><td>0.457 94.2</td><td>40.46</td><td>43.99</td><td>38.80 16.22</td><td>0.402 90.8</td><td>56.89</td><td>60.84</td><td>26.66 13.97</td><td>0.472 90.5</td><td>61.92</td><td>67.43</td><td>20.00 14.40</td><td>0.532 89.5</td></tr><tr><td>MoLingo [70]</td><td>99.24</td><td>102.67</td><td>47.77 8.86</td><td>1.241 868.3</td><td>137.47</td><td>140.71</td><td>34.79 7.46</td><td>1.299 847.7</td><td>205.38</td><td>207.00</td><td>41.14 8.72</td><td>1.261 840.4</td><td>270.71</td><td>273.42</td><td>41.34 8.81</td><td>1.260 908.7</td></tr><tr><td>HY-Motion [71]</td><td>30.76</td><td>33.54</td><td>55.70 16.67</td><td>0.400 324.1</td><td>43.94</td><td>45.97</td><td>37.37 12.47</td><td>0.492 274.0</td><td>65.06</td><td>66.75</td><td>31.96 14.52</td><td>0.629 290.5</td><td>95.53</td><td>96.63</td><td>27.61 13.47</td><td>0.760 346.8</td></tr><tr><td>OMG [63]</td><td>58.18</td><td>60.59</td><td>46.76 25.93</td><td>0.448 173.9</td><td>82.83</td><td>85.97</td><td>36.41 18.84</td><td>0.428 170.2</td><td>115.50</td><td>118.11</td><td>29.85 20.96</td><td>0.443 168.3</td><td>161.76</td><td>163.49</td><td>24.52 19.61</td><td>0.473 167.8</td></tr><tr><td>TACD [72]</td><td>21.81</td><td>25.55</td><td>77.27 28.44</td><td>0.428 203.0</td><td>30.49</td><td>34.66</td><td>43.71 18.26</td><td>0.422 224.9</td><td>40.19</td><td>44.16</td><td>34.88 20.63</td><td>0.418 191.6</td><td>54.43</td><td>59.40</td><td>28.03 17.81</td><td>0.496 208.0</td></tr><tr><td>PCMDM [34]</td><td>63.86</td><td>66.65</td><td>28.71 17.59</td><td>0.434 58.4</td><td>61.17</td><td>63.96</td><td>26.73 10.43</td><td>0.457 51.6</td><td>62.41</td><td>64.93</td><td>16.12 10.28</td><td>0.512 53.0</td><td>65.18</td><td>66.99</td><td>14.27 10.96</td><td>0.544 51.4</td></tr><tr><td>FlowMDM [4]</td><td>56.67</td><td>58.01</td><td>46.25 14.29</td><td>0.400 109.9</td><td>72.80</td><td>73.93</td><td>35.00 10.61</td><td>0.437 105.2</td><td>112.44</td><td>113.81</td><td>26.35 10.57</td><td>0.443 107.2</td><td>154.71</td><td>156.35</td><td>24.12 11.07</td><td>0.454 105.7</td></tr><tr><td>DartControl [6]</td><td>44.84</td><td>48.37</td><td>71.88 13.49</td><td>0.387 34.7</td><td>64.66</td><td>68.24</td><td>34.24 16.52</td><td>0.471 30.3</td><td>91.05</td><td>95.51</td><td>41.17 16.35</td><td>0.493 29.1</td><td>291.58</td><td>295.86</td><td>40.32 15.53</td><td>0.511 28.2</td></tr><tr><td>Motion-Agent [73]</td><td>20.72</td><td>24.09</td><td>22.22 23.17</td><td>0.437 13806.4</td><td>31.76</td><td>33.25</td><td>26.69 12.36</td><td>0.496 13694.5</td><td>43.15</td><td>46.34</td><td>14.97 12.14</td><td>0.542 13319.4</td><td>61.36</td><td>62.49</td><td>15.15 12.02</td><td>0.745 13511.1</td></tr><tr><td>MG-MotionLLM [74]</td><td>63.77</td><td>67.99</td><td>22.39 23.02</td><td>0.434 5632.7</td><td>98.31</td><td>99.67</td><td>25.48 9.22</td><td>0.455 5783.3</td><td>204.32</td><td>203.35</td><td>12.14 10.60</td><td>0.451 5861.8</td><td>340.84</td><td>341.01</td><td>11.52 11.60</td><td>0.431 5772.3</td></tr><tr><td>IRG-MotionLLM [75]</td><td>43.58</td><td>46.97</td><td>24.58 21.74</td><td>0.447 13329.8</td><td>70.64</td><td>72.89</td><td>23.86 11.93</td><td>0.472 13295.7</td><td>92.71</td><td>95.16</td><td>13.53 15.95</td><td>0.535 13198.4</td><td>113.23</td><td>131.52</td><td>13.04 13.15</td><td>0.751 12912.6</td></tr><tr><td>MotionStreamer [8]</td><td>17.89</td><td>23.56</td><td>80.32 22.62</td><td>0.388 99.0</td><td>16.93</td><td>25.31</td><td>46.00 17.54</td><td>0.444 109.8</td><td>6.17</td><td>18.07</td><td>26.87 15.83</td><td>0.542 119.2</td><td>7.78</td><td>21.67</td><td>18.85 13.02</td><td>0.645 159.8</td></tr><tr><td>CMDM [39]</td><td>24.65</td><td>27.16</td><td>84.02 23.41</td><td>0.337 157.8</td><td>38.93</td><td>42.65</td><td>34.30 14.65</td><td>0.454 186.7</td><td>49.35</td><td>53.04</td><td>22.80 14.49</td><td>0.549 230.4</td><td>54.68</td><td>59.68</td><td>15.36 12.10</td><td>0.657 311.9</td></tr><tr><td>ActionPlan [76]</td><td>69.72</td><td>83.69</td><td>52.79 9.13</td><td>0.862 72.4</td><td>93.89</td><td>106.81</td><td>36.76 4.70</td><td>1.047 68.8</td><td>281.64</td><td>279.40</td><td>40.38 6.43</td><td>1.067 85.3</td><td>799.60</td><td>790.99</td><td>40.21 6.16</td><td>1.120 110.1</td></tr><tr><td>FloodDiffusion 2 [38]</td><td>39.12</td><td>41.54</td><td>63.51 20.63</td><td>0.403 196.8</td><td>67.72</td><td>70.42</td><td>36.03 12.44</td><td>0.433 190.8</td><td>90.70</td><td>93.48</td><td>26.95 13.55</td><td>0.465 186.9</td><td>115.02</td><td>117.86</td><td>22.34 12.14</td><td>0.497 183.7</td></tr><tr><td>ARDY [7]</td><td>5.89</td><td>9.70</td><td>86.77 23.54</td><td>0.425 95.6</td><td>7.44</td><td>10.27</td><td>45.83 13.76</td><td>0.438 90.1</td><td>8.15</td><td>11.06</td><td>30.89 17.14</td><td>0.467 91.5</td><td>10.09</td><td>14.84</td><td>23.53 15.32</td><td>0.489 105.1</td></tr><tr><td>Motion Mamba [15]</td><td>50.58</td><td>53.39</td><td>54.75 19.84</td><td>0.406 309.2</td><td>56.81</td><td>59.83</td><td>44.06 17.98</td><td>0.414 317.3</td><td>78.67</td><td>82.37</td><td>31.46 17.29</td><td>0.420 315.1</td><td>125.74</td><td>130.17</td><td>25.77 17.18</td><td>0.482 310.5</td></tr><tr><td>Motion Mamba 2.0</td><td>2.78</td><td>7.19</td><td>86.96 35.58</td><td>0.410 283.7</td><td>2.53</td><td>7.11</td><td>48.82 21.35</td><td>0.401 267.5</td><td>2.87</td><td>8.45</td><td>47.11 23.05</td><td>0.402 284.7</td><td>7.55</td><td>13.64</td><td>46.05 23.68</td><td>0.409 333.4</td></tr></tbody></table>

## 5.1 Experimental Setups

**Training Procedure.** Motion Mamba 2.0 is jointly trained with Mamba-3 and persistent content memory (PCM) enabled from initialization. The G1 motion codec remains frozen, while cached text features provide conditioning. Training minimizes the four-term objective in Eq. (15). We use AdamW for 100,000 updates with a global batch size of 512 throughout training. The learning rate increases linearly from $1 0 ^ { - \tilde { 6 } }$ at update 1 to $5 \times 1 0 ^ { - 5 }$ at update 500, followed by cosine decay to $1 0 ^ { - 5 }$ at update 100,000. Gradients are clipped to a maximum norm of 1.0. The generation length is fixed at 52 frames. History/future lengths (H, F) are sampled from (0, 0), (52, 0), (196, 0), (0, 196), (52, 144), and (144, 52), with respective frequencies of 2, 5, 4, 2, 4, 3 per 20 updates. The maximum sampled training history is therefore 196 frames (7.84 s).

Training samples are drawn from BONES Seed, retargeted HumanML3D, and retargeted HiPHI in a 70:15:15 ratio. Each sample contains a 52-frame generation window with historical motion conditions constructed from ground-truth sequences and sampled future constraints; training does not autoregressively roll out generated histories. Text and spatial condition dropout are both set to 0.1, with an empty spatialcondition probability of 0.2. For fair comparison, all baselines and architectural variants are independently trained on the same G1 corpus under the matched 100,000-update budget and batch-size schedule. All evaluations use the resulting G1-adapted checkpoints rather than the original pretrained releases. Further details are provided in App. C.

**Evaluation Protocol.** Motion is generated at 25 Hz in 52- frame windows (2.08 s). Retargeted HiPHI [24] is evaluated over 1, 4, 9, and 19 windows (2.08–39.52 s), while BONES Seed [25] and retargeted HumanML3D [26] use 1 and 4 windows. For each horizon and method, 128 held-out initialization families and three noise seeds yield 384 rollouts under paired evaluation conditions. HiPHI also shares initial histories, spatial targets, and noise seeds across horizons, while instruction schedules and terminal deadlines differ. Each trial specifies a root target and one wrist or ankle target, available from rollout onset with a terminal deadline; only the active text instruction is provided during generation. Sampling uses 10-step deterministic DDIM $( \eta = 0 )$ [77] with uniformly spaced timesteps and text/spatial guidance scales of (2, 2). Further details are provided in App. C.

**Evaluation Metrics.** Root CE measures horizontal rootposition error in the world X–Z plane, while End CE

measures the 3D positional error of the constrained wrist or ankle at the terminal deadline, both in centimeters. Command matching (Cmd.) evaluates coarse root-motion state agreement. Text–motion alignment is evaluated by R@3 using independently trained, frozen evaluators for each benchmark. HiPHI uses 32 generated candidates, whereas BONES Seed and HumanML3D use one matched generated candidate and 31 fixed real-motion distractors. R@3 is therefore comparable within, but not across benchmarks. L2-FID compares distributions of L2-normalized 256-dimensional motion features from phase-paired 52-frame windows, including those in extended rollouts. Latency is measured per accepted window at batch size one. Metric aggregation and reference protocols are detailed in App. C.

## 5.2 Main Results

**Results on HiPHI.** Tab. 1 compares four horizons from 2.08 to 39.52 s. Motion Mamba 2.0 achieves the lowest Root CE at every duration and the lowest or tied-lowest End CE. At 18.72 s, Root/End CE reach 2.87/8.45 cm, compared with 6.17/18.07 cm for MotionStreamer and 8.15/11.06 cm for ARDY. At 39.52 s, our method achieves 7.55/13.64 cm, reducing Root CE by 3.0% relative to MotionStreamer (7.78 cm) and outperforming the best competing End CE of 14.84 cm from ARDY by 8.1%. These results demonstrate accurate terminal constraint satisfaction over extended rollouts. A qualitative comparison with ARDY under a distant spatial constraint is provided in App. C.6.

Motion Mamba 2.0 also achieves the highest Cmd. and R@3 across all four horizons, supporting command agreement and text–motion retrieval under the evaluated protocol. At 39.52 s, Cmd. and R@3 reach 46.05% and 23.68%, compared with 41.34% for MoLingo and 19.61% for OMG, respectively. The method obtains the lowest L2-FID at 8.32, 18.72, and 39.52 s (0.401, 0.402, and 0.409); CMDM leads at 2.08 s (0.337 versus 0.410), and the 8.32 s advantage over DreamControl (0.402) is small. Together, these results demonstrate strong terminal control alongside long-horizon command agreement and retrieval performance.

**Results on BONES Seed.** Tab. 2 evaluates short-horizon generation on native G1 trajectories from BONES Seed. Motion Mamba 2.0 achieves the lowest Root/End CE and highest R@3 at both durations. At 8.32 s, Root CE decreases to 1.86 cm from 4.96 cm for ARDY, a relative reduction of 62.5%, while End CE decreases from 10.16 to 6.86 cm.

<!-- page 10 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

10

TABLE 2: Short-Horizon Motion Generation on BONES Seed. CE is reported in cm, Cmd. and R@3 in percentages, and latency in ms per accepted window at batch size 1. R@3 uses one matched generated motion and 31 fixed real-motion distractors. Full protocol: App. C.

<table><tr><td rowspan="2">Method</td><td colspan="6">52 frames (1 window)</td><td colspan="6">8.32 s (4 windows)</td></tr><tr><td>Root CE↓</td><td>End CE↓</td><td>Cmd.↑</td><td>R@3↑</td><td>L2-FID↓</td><td>Lat.↓</td><td>Root CE↓</td><td>End CE↓</td><td>Cmd.↑</td><td>R@3↑</td><td>L2-FID↓</td><td>Lat.↓</td></tr><tr><td>DiP [62]</td><td>60.28</td><td>63.68</td><td>44.61</td><td>18.84</td><td>0.461</td><td>74.7</td><td>61.21</td><td>63.68</td><td>36.02</td><td>32.30</td><td>0.640</td><td>70.3</td></tr><tr><td>Go-to-Zero [68]</td><td>30.80</td><td>32.50</td><td>72.39</td><td>17.55</td><td>0.487</td><td>12275.5</td><td>43.19</td><td>41.27</td><td>39.46</td><td>12.23</td><td>0.613</td><td>12316.8</td></tr><tr><td>DreamControl [69]</td><td>22.70</td><td>27.33</td><td>96.64</td><td>40.60</td><td>0.608</td><td>93.1</td><td>36.06</td><td>40.20</td><td>54.93</td><td>26.49</td><td>0.522</td><td>92.1</td></tr><tr><td>MoLingo [70]</td><td>93.85</td><td>95.66</td><td>97.50</td><td>32.64</td><td>1.283</td><td>846.7</td><td>122.43</td><td>124.64</td><td>47.82</td><td>15.15</td><td>1.207</td><td>841.7</td></tr><tr><td>HY-Motion [71]</td><td>38.94</td><td>31.32</td><td>59.93</td><td>18.51</td><td>0.456</td><td>282.3</td><td>51.94</td><td>54.12</td><td>32.85</td><td>11.80</td><td>0.583</td><td>276.9</td></tr><tr><td>OMG [63]</td><td>76.58</td><td>77.97</td><td>46.18</td><td>26.39</td><td>0.345</td><td>173.4</td><td>123.93</td><td>125.52</td><td>31.48</td><td>30.24</td><td>0.544</td><td>173.9</td></tr><tr><td>TACD [72]</td><td>32.70</td><td>35.43</td><td>68.36</td><td>35.72</td><td>0.483</td><td>201.7</td><td>35.86</td><td>39.61</td><td>40.50</td><td>28.80</td><td>0.555</td><td>192.3</td></tr><tr><td>PCMDM [34]</td><td>65.98</td><td>67.49</td><td>41.00</td><td>14.97</td><td>0.437</td><td>58.8</td><td>66.77</td><td>68.26</td><td>29.77</td><td>24.85</td><td>0.600</td><td>53.5</td></tr><tr><td>FlowMDM [4]</td><td>76.32</td><td>77.72</td><td>38.10</td><td>16.19</td><td>0.382</td><td>113.6</td><td>92.96</td><td>94.37</td><td>30.62</td><td>24.56</td><td>0.518</td><td>107.3</td></tr><tr><td>DartControl [6]</td><td>46.51</td><td>46.25</td><td>96.60</td><td>48.66</td><td>0.602</td><td>35.2</td><td>272.86</td><td>272.43</td><td>61.79</td><td>27.90</td><td>0.668</td><td>30.8</td></tr><tr><td>Motion-Agent [73]</td><td>33.79</td><td>35.23</td><td>32.37</td><td>36.90</td><td>0.538</td><td>13461.2</td><td>48.05</td><td>50.89</td><td>26.53</td><td>25.98</td><td>0.517</td><td>13624.0</td></tr><tr><td>MG-MotionLLM [74]</td><td>82.91</td><td>86.77</td><td>54.57</td><td>37.50</td><td>0.460</td><td>5623.9</td><td>328.11</td><td>329.51</td><td>25.83</td><td>30.01</td><td>0.469</td><td>5745.0</td></tr><tr><td>IRG-MotionLLM [75]</td><td>49.68</td><td>42.81</td><td>36.62</td><td>38.61</td><td>0.501</td><td>14497.4</td><td>69.05</td><td>76.78</td><td>29.16</td><td>30.38</td><td>0.621</td><td>13868.3</td></tr><tr><td>MotionStreamer [8]</td><td>15.08</td><td>22.29</td><td>88.32</td><td>53.99</td><td>0.570</td><td>102.7</td><td>12.97</td><td>19.54</td><td>43.01</td><td>26.63</td><td>0.542</td><td>106.2</td></tr><tr><td>CMDM [39]</td><td>32.66</td><td>34.71</td><td>85.90</td><td>48.65</td><td>0.439</td><td>182.1</td><td>48.56</td><td>52.06</td><td>37.15</td><td>25.07</td><td>0.579</td><td>197.3</td></tr><tr><td>ActionPlan [76]</td><td>77.76</td><td>89.85</td><td>100.00</td><td>44.08</td><td>1.237</td><td>71.8</td><td>223.18</td><td>223.45</td><td>54.59</td><td>12.76</td><td>1.405</td><td>68.5</td></tr><tr><td>FloodDiffusion 2 [38] ARDY [7]</td><td>52.50 5.57</td><td>54.09 9.63</td><td>62.26 83.14</td><td>34.98 48.74</td><td>0.409 0.596</td><td>199.4 98.1</td><td>71.05</td><td>73.28 10.16</td><td>35.62 45.10</td><td>26.64 34.26</td><td>0.565</td><td>191.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.96</td><td></td><td></td><td></td><td>0.568</td><td>102.7</td></tr><tr><td>Motion Mamba [15]</td><td>43.29</td><td>44.65</td><td>77.35</td><td>56.47</td><td>0.560</td><td>315.4</td><td>58.64</td><td>60.83</td><td>42.31</td><td>29.15</td><td>0.591</td><td>301.8</td></tr><tr><td>Motion Mamba 2.0</td><td>2.10</td><td>7.09</td><td>86.92</td><td>59.38</td><td>0.573</td><td>290.1</td><td>1.86</td><td>6.86</td><td>51.42</td><td>35.58</td><td>0.459</td><td>271.1</td></tr></table>

TABLE 3: Short-Horizon Motion Generation on Retargeted HumanML3D. CE is reported in cm, Cmd. and R@3 in percentages, and latency in ms per accepted window at batch size 1. R@3 uses one matched generated motion and 31 fixed real-motion distractors. Full protocol: App. C.

<table><tr><td rowspan="2">Method</td><td colspan="6">52 frames (1 window)</td><td colspan="6">8.32 s (4 windows)</td></tr><tr><td>Root CE↓</td><td>End CE↓</td><td>Cmd.↑</td><td>R@3↑</td><td>L2-FID↓</td><td>Lat.↓</td><td>Root CE↓</td><td>End CE↓</td><td>Cmd.↑</td><td>R@3 ↑</td><td>L2-FID↓</td><td>Lat.↓</td></tr><tr><td>DiP [62]</td><td>49.72</td><td>54.03</td><td>64.59</td><td>84.46</td><td>0.623</td><td>76.1</td><td>52.78</td><td>55.17</td><td>41.04</td><td>47.95</td><td>0.706</td><td>69.4</td></tr><tr><td>Go-to-Zero [68]</td><td>29.11</td><td>21.71</td><td>52.30</td><td>66.90</td><td>0.462</td><td>12272.0</td><td>34.78</td><td>31.96</td><td>34.15</td><td>40.31</td><td>0.541</td><td>12351.6</td></tr><tr><td>DreamControl [69]</td><td>22.06</td><td>27.14</td><td>97.24</td><td>87.98</td><td>1.100</td><td>98.1</td><td>36.02</td><td>39.27</td><td>57.48</td><td>45.76</td><td>0.737</td><td>94.1</td></tr><tr><td>MoLingo [70]</td><td>93.94</td><td>99.88</td><td>97.82</td><td>79.90</td><td>1.551</td><td>881.4</td><td>130.90</td><td>135.04</td><td>47.82</td><td>43.55</td><td>1.514</td><td>862.7</td></tr><tr><td>HY-Motion [71]</td><td>35.46</td><td>37.00</td><td>51.78</td><td>61.83</td><td>0.383</td><td>317.5</td><td>47.05</td><td>48.94</td><td>33.24</td><td>31.78</td><td>0.579</td><td>292.0</td></tr><tr><td>OMG [63]</td><td>75.87</td><td>76.51</td><td>72.36</td><td>72.70</td><td>0.319</td><td>175.6</td><td>136.38</td><td>137.49</td><td>36.37</td><td>32.52</td><td>0.564</td><td>170.8</td></tr><tr><td>TACD [72]</td><td>25.75</td><td>27.84</td><td>78.68</td><td>11.98</td><td>0.329</td><td>204.3</td><td>34.66</td><td>37.32</td><td>42.67</td><td>17.42</td><td>0.490</td><td>195.6</td></tr><tr><td>PCMDM [34]</td><td>51.98</td><td>54.63</td><td>64.60</td><td>74.74</td><td>0.592</td><td>60.1</td><td>51.09</td><td>53.63</td><td>35.84</td><td>40.38</td><td>0.729</td><td>54.8</td></tr><tr><td>FlowMDM [4]</td><td>67.89</td><td>68.95</td><td>65.54</td><td>70.40</td><td>0.364</td><td>109.8</td><td>104.39</td><td>105.96</td><td>38.20</td><td>40.90</td><td>0.554</td><td>104.0</td></tr><tr><td>DartControl [6]</td><td>67.17</td><td>64.09</td><td>96.96</td><td>87.50</td><td>0.528</td><td>39.3</td><td>238.78</td><td>235.93</td><td>61.04</td><td>48.35</td><td>0.732</td><td>29.7</td></tr><tr><td>Motion-Agent [73]</td><td>22.20</td><td>25.39</td><td>52.55</td><td>65.49</td><td>0.574</td><td>13661.4</td><td>34.60</td><td>35.43</td><td>26.81</td><td>44.63</td><td>0.611</td><td>13264.4</td></tr><tr><td>MG-MotionLLM [74]</td><td>60.45</td><td>64.31</td><td>53.16</td><td>67.06</td><td>0.558</td><td>5590.8</td><td>160.42</td><td>159.49</td><td>25.71</td><td>40.33</td><td>0.554</td><td>5743.3</td></tr><tr><td>IRG-MotionLLM [75]</td><td>45.69</td><td>43.30</td><td>51.31</td><td>63.82</td><td>0.490</td><td>13556.0</td><td>65.03</td><td>67.18</td><td>24.09</td><td>42.15</td><td>0.618</td><td>13468.2</td></tr><tr><td>MotionStreamer [8]</td><td>15.24</td><td>20.23</td><td>91.32</td><td>96.40</td><td>0.463</td><td>106.7</td><td>12.16</td><td>18.28</td><td>47.63</td><td>44.39</td><td>0.557</td><td>105.4</td></tr><tr><td>CMDM [39]</td><td>32.61</td><td>34.20</td><td>92.80</td><td>89.76</td><td>0.375</td><td>129.0</td><td>47.30</td><td>49.82</td><td>44.37</td><td>46.08</td><td>0.572</td><td>129.9</td></tr><tr><td>ActionPlan [76]</td><td>76.26</td><td>90.77</td><td>100.00</td><td>40.13</td><td>1.737</td><td>74.8</td><td>221.90</td><td>222.02</td><td>55.11</td><td>40.27</td><td>2.005</td><td>70.2</td></tr><tr><td>FloodDiffusion 2 [38]</td><td>52.15</td><td>53.78 8.85</td><td>82.43</td><td>77.86</td><td>0.334</td><td>164.1</td><td>71.64</td><td>74.09</td><td>42.51</td><td>42.08</td><td>0.593</td><td>153.9</td></tr><tr><td>ARDY [7]</td><td>5.22</td><td></td><td>87.55</td><td>86.85</td><td>0.430</td><td>100.6</td><td>4.75</td><td>9.82</td><td>52.28</td><td>48.21</td><td>0.491</td><td>91.1</td></tr><tr><td>Motion Mamba [15]</td><td>38.42</td><td>40.17</td><td>85.74</td><td>84.57</td><td>0.747</td><td>309.8</td><td>59.26</td><td>61.31</td><td>54.47</td><td>40.14</td><td>0.653</td><td>313.9</td></tr><tr><td>Motion Mamba 2.0</td><td>2.18</td><td>6.73</td><td>88.07</td><td>84.29</td><td>0.459</td><td>286.0</td><td>2.17</td><td>7.03</td><td>58.81</td><td>49.72</td><td>0.490</td><td>271.2</td></tr></table>

The corresponding Cmd. and R@3 are 51.42% and 35.58%, respectively; command matching is not the highest among compared methods.

For distributional similarity, our method obtains the lowest L2-FID at 8.32 s (0.459), whereas OMG leads at 2.08 s (0.345 versus 0.573). DartControl remains the fastest method. These results confirm the spatial-control and R@3 advantages on native G1 motions, while coarse command matching and single-window distributional quality are not uniformly best.

**Results on HumanML3D.** Tab. 3 reports results on retargeted HumanML3D at 2.08 and 8.32 s, within the available source duration. Motion Mamba 2.0 achieves the lowest Root/End CE at both horizons. At 8.32 s, Root/End CE reach 2.17/7.03 cm, compared with 4.75/9.82 cm for ARDY, a 54.3% reduction in Root CE. It also obtains the highest four-window R@3 of 49.72%, exceeding DartControl (48.35%). At 2.08 s,

its R@3 of 84.29% remains below MotionStreamer (96.40%), indicating a difference between precise constraint control and single-window text retrieval.

OMG obtains the lowest L2-FID at 2.08 s (0.319), whereas our method narrowly leads at 8.32 s (0.490 versus 0.491 for ARDY). Together with BONES Seed, these results support accurate spatial control on both retargeted human motions and native G1 references without requiring extended rollout durations, while showing that semantic metrics vary across benchmarks.

## 5.3 Architectural Ablations

We assess recurrent modeling and PCM through five configurations: Minimal (without the Mamba branch), Native (Mamba without PCM), Native + PCM (w/o text), Native + PCM (w/o delta), and the complete Motion Mamba 2.0. The

<!-- page 11 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

11

TABLE 4: Architectural Ablation on Retargeted HiPHI. Training budgets and execution paths are specified in App. C. Metrics and units follow Tab. 1.

<table><tbody><tr><td rowspan="2">Method Ro</td><td colspan="4">52 frames / 1 window</td><td colspan="4">8.32 s / 4 windows</td><td colspan="4">18.72 s / 9 windows</td><td colspan="4">39.52 s / 19 windows</td></tr><tr><td>ot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L2</td><td>-FID↓ Lat.↓ Ro</td><td>ot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L</td><td>2-FID↓ Lat.↓ R</td><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L</td><td>2-FID↓ Lat.↓ R</td><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑ R@3↑ L</td><td>2-FID↓ Lat.↓</td></tr><tr><td>Minimal</td><td>3.86</td><td>12.44</td><td>89.69 34.62</td><td>0.371 509.0</td><td>5.92</td><td>13.52</td><td>52.27 20.14</td><td>0.414 536.5</td><td>18.70</td><td>23.58</td><td>38.27 18.07</td><td>0.427 557.6</td><td>60.00</td><td>64.86</td><td>35.27 17.26</td><td>0.439 552.9</td></tr><tr><td>Native</td><td>4.49</td><td>10.88</td><td>88.14 32.67</td><td>0.368 233.4</td><td>4.60</td><td>9.84</td><td>51.69 19.64</td><td>0.462 200.6</td><td>6.21</td><td>11.44</td><td>38.21 19.08</td><td>0.441 212.5</td><td>9.36</td><td>15.54</td><td>32.20 17.44</td><td>0.416 258.7</td></tr><tr><td>Native + PCM (w/o text)</td><td>3.19</td><td>7.65</td><td>87.01 32.01</td><td>0.423 286.5</td><td>3.05</td><td>7.81</td><td>47.60 19.41</td><td>0.419 267.8</td><td>2.99</td><td>9.02</td><td>38.10 21.06</td><td>0.437 292.8</td><td>8.62</td><td>14.62</td><td>33.34 21.18</td><td>0.467 336.5</td></tr><tr><td>Native + PCM (w/o delta)</td><td>3.06</td><td>7.41</td><td>86.36 32.67</td><td>0.433 287.6</td><td>2.97</td><td>7.19</td><td>47.39 19.76</td><td>0.421 263.1</td><td>3.08</td><td>8.99</td><td>37.18 20.35</td><td>0.459 284.7</td><td>8.30</td><td>14.18</td><td>32.65 21.11</td><td>0.488 329.8</td></tr><tr><td>Motion Mamba 2.0</td><td>2.78</td><td>7.19</td><td>86.96 35.58</td><td>0.410 283.7</td><td>2.53</td><td>7.11</td><td>48.82 21.35</td><td>0.401 267.5</td><td>2.87</td><td>8.45</td><td>47.11 23.05</td><td>0.402 284.7</td><td>7.55</td><td>13.64</td><td>46.05 23.68</td><td>0.409 333.4</td></tr></tbody></table>

TABLE 5: Architectural ablation on BONES Seed. Training budgets and execution paths are specified in App. C. Units and R@3 follow Tab. 2.

<table><tbody><tr><td rowspan="2">Method R</td><td colspan="5">52 frames (1 window)</td><td colspan="5">8.32 s (4 windows)</td></tr><tr><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑</td><td>R@3 ↑ L</td><td>2-FID↓ Lat.↓ R</td><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑</td><td>R@3 ↑ L</td><td>2-FID↓ Lat.↓</td></tr><tr><td>Minimal</td><td>6.76</td><td>12.82</td><td>88.09</td><td>57.21</td><td>0.561 490.8</td><td>11.10</td><td>17.17</td><td>48.02</td><td>31.66</td><td>0.656 539.1</td></tr><tr><td>Native</td><td>5.17</td><td>10.75</td><td>87.16</td><td>56.93</td><td>0.552 219.8</td><td>5.46</td><td>11.29</td><td>52.06</td><td>35.15</td><td>0.516 210.5</td></tr><tr><td>Native + PCM (w/o text)</td><td>3.34</td><td>8.48</td><td>87.25</td><td>54.82</td><td>0.568 286.2</td><td>3.53</td><td>8.21</td><td>44.16</td><td>30.71</td><td>0.473 270.4</td></tr><tr><td>Native + PCM (w/o delta)</td><td>3.22</td><td>8.61</td><td>86.58</td><td>54.56</td><td>0.575 290.2</td><td>3.61</td><td>8.74</td><td>43.75</td><td>30.47</td><td>0.481 268.6</td></tr><tr><td>Motion Mamba 2.0</td><td>2.10</td><td>7.09</td><td>86.92</td><td>59.38</td><td>0.573 290.1</td><td>1.86</td><td>6.86</td><td>51.42</td><td>35.58</td><td>0.459 271.1</td></tr></tbody></table>

TABLE 6: Architectural ablation on Retargeted HumanML3D. Training budgets and execution paths are specified in App. C. Units and R@3 follow Tab. 3.

<table><tbody><tr><td rowspan="2">Method R</td><td colspan="5">52 frames (1 window)</td><td colspan="5">8.32 s (4 windows)</td></tr><tr><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑</td><td>R@3 ↑ L</td><td>2-FID↓ Lat.↓ R</td><td>oot CE↓ E</td><td>nd CE↓</td><td>Cmd.↑</td><td>R@3 ↑ L</td><td>2-FID↓ Lat.↓</td></tr><tr><td>Minimal</td><td>8.31</td><td>13.05</td><td>90.56</td><td>92.97</td><td>0.410 512.0</td><td>14.31</td><td>17.49</td><td>55.56</td><td>53.42</td><td>0.583 544.8</td></tr><tr><td>Native</td><td>6.58</td><td>11.48</td><td>89.37</td><td>88.72</td><td>0.448 230.4</td><td>6.25</td><td>11.07</td><td>55.86</td><td>53.27</td><td>0.579 219.1</td></tr><tr><td>Native + PCM (w/o text)</td><td>3.29</td><td>8.37</td><td>88.54</td><td>85.24</td><td>0.444 287.9</td><td>3.40</td><td>8.67</td><td>51.04</td><td>43.07</td><td>0.462 273.1</td></tr><tr><td>Native + PCM (w/o delta)</td><td>3.87</td><td>8.36</td><td>88.63</td><td>83.59</td><td>0.442 288.8</td><td>3.82</td><td>8.54</td><td>51.18</td><td>42.43</td><td>0.477 267.1</td></tr><tr><td>Motion Mamba 2.0</td><td>2.18</td><td>6.73</td><td>88.07</td><td>84.29</td><td>0.459 286.0</td><td>2.17</td><td>7.03</td><td>58.81</td><td>49.72</td><td>0.490 271.2</td></tr></tbody></table>

last two partial-PCM variants ablate text conditioning and the explicit state-difference feature in the PCM writing path, respectively. The reported variants use the matched 100,000- update training and evaluation protocol. Tabs. 4, 5, and 6 report the results on HiPHI, BONES Seed, and HumanML3D. **Effect of recurrent modeling.** The benefit of recurrent modeling becomes more pronounced as the generation horizon increases (Tab. 4). At 39.52 s, introducing Mamba reduces Root CE from 60.00 to 9.36 cm and End CE from 64.86 to 15.54 cm compared with Minimal; corresponding singlewindow differences are much smaller. Similar improvements in constraint errors are observed on BONES Seed and HumanML3D at 8.32 s. These results support the contribution of recurrent state propagation to extended spatial control. The gain is not uniform across metrics, as Native has lower Cmd. than Minimal at 39.52 s (32.20% versus 35.27%).

**Contribution of persistent content memory.** Adding PCM to Native consistently improves Root CE and End CE across the three benchmarks (Tabs. 4–6). On HiPHI at 39.52 s, PCM reduces Root CE from 9.36 to 7.55 cm (19.3%) and End CE from 15.54 to 13.64 cm (12.2%), while increasing Cmd. from 32.20% to 46.05%. R@3 rises from 17.44% to 23.68%, and L2-FID decreases from 0.416 to 0.409. These gains establish its contribution to long-horizon control and retrieval.

At the same 39.52 s setting, excluding text conditioning increases Root/End CE to 8.62/14.62 cm and decreases Cmd./R@3 to 33.34%/21.18%; excluding the state-difference feature gives 8.30/14.18 cm and 32.65%/21.11%, respectively. The complete writer also improves Cmd. relative to both partial-PCM variants at 18.72 s (47.11% versus 38.10% and 37.18%). Both partial variants retain better long-protocol Root/End CE than Native, supporting the utility of the underlying compressed-state-to-attention interface. On BONES Seed and retargeted HumanML3D at 8.32 s, the complete model reduces Root CE from 5.46 to 1.86 cm and from 6.25 to 2.17 cm, respectively. Improvements are not uniform across

metrics: some short-horizon Cmd., R@3, and L2-FID scores favor other variants. These results support complementary roles for state changes and text-conditioned writing in the evaluated long-horizon spatial-control and commandmatching tasks, without proving exact retrieval of arbitrary evicted motion events.

## 5.4 Inference Efficiency

We evaluate whether the proposed execution optimizations reduce generation cost while preserving the complete PCM model. All execution variants use identical frozen weights. This study separates execution-flow simplification, local A/DT kernel fusion, and CUDA Graph replay; it does not change the denoising schedule or memory architecture.

**Measurement protocol.** Experiments use one RTX 4090, batch size one, ten denoising steps, three classifier-free guidance branches with scales (2, 2), and four windows of history. Attention remains FP32, the existing BF16 boundaries inside Mamba are retained, and TF32 is disabled. Each timing includes the complete accepted 52-frame window, from state restoration through decoding and CPU snapshot/export. Text embeddings are preloaded; loading, warmup, and graph capture are excluded. We collect 30 matched window pairs in each of five independent processes. Paired variants start from identical session and random states. Confidence intervals use 10,000 process-block bootstrap resamples. The timing boundary and numerical checks are detailed in App. B.4.

**Complete-window performance.** As shown in Tab. 7, the complete scheme reduces p50 latency from 1685.35 to 276.14 ms and p95 from 2471.65 to 345.82 ms, reductions of 83.6% and 86.0%, respectively. The median paired speedup is 6.138×, with a 95% confidence interval of [6.094, 6.290]. For a 2.08-s motion window, the optimized p50 corresponds to a real-time factor of 0.133. This measures steady-state generation, rather than cold-start or command-to-visiblemotion latency.

<!-- page 12 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

12

TABLE 7: Inference Optimization Component Ablation for PCM. Flow, Fusion, and Graph denote execution-flow simplification, fused Mamba A/DT transition preprocessing, and CUDA Graph replay, respectively. Latencies are in ms (↓); speedup is the median paired ratio (↑).

| Flow | Fusion | Graph | p50 | p95 | Speedup |
| --- | --- | --- | --- | --- | --- |
| -✓✓✓✓ | ---✓✓ | ---✓✓ | 1685.351102.821069.52285.20276.14 | 2471.651647.521588.74352.22345.82 | 61151.....155903274085830××××× |

![Chart block](doc:a492382/tier:basic/page:12/block:5)

Fig. 7: PCM execution profile before and after optimization, normalized to eager execution. Ordinary host launches exclude CUDA Graph replay calls. Counts come from separate profiling runs and are independent of the latency measurements in Tab. 7.

Execution-flow simplification reduces p50 to 1102.82 ms by removing unused pending-state materialization and diagnostic synchronization. Local fusion reduces it further to 1069.52 ms. Joint capture and replay of the three CFG branches account for the larger improvement by reducing repeated host dispatch, reaching 285.20 ms without fusion; adding fusion reaches 276.14 ms, with a further paired speedup of 1.029×. Thus, the complete speedup cannot be attributed to the fused kernel alone.

**Execution profile and consistency.** Fig. 7 shows that ordinary host launches decrease from 83,777 to 5,519, a 93.4% reduction, while GPU kernels decrease from 86,849 to 44,877, a 48.3% reduction. Synchronization calls decrease from 10,194 to 1,330. The larger reduction in host submissions is consistent with graph replay reducing dispatch overhead for many small kernels. The optimized path also preserves bitwise-identical generated outputs and committed states across four 100-window rollouts with different seeds.

## 6 REAL-WORLD EVALUATION

While Sec. 5.2 assesses long-horizon spatial constraints and command agreement in generated motion references, Sec. 5.4 evaluates the cost of producing each motion window. We next examine whether these references support continuous and responsive execution on a physical Unitree G1. The experiments measure sequential text-guided action composition, online spatial control, and physical response in the integrated generation-and-tracking system. Fig. 8 presents representative robot executions alongside their Viser references.

**Evaluation setting.** We conduct 20 trials each for sequential action composition (T1) and expressive motion (T2), and 30 trials each for keyboard-controlled (T3) and mouse-controlled (T4) locomotion. All conditions use the same motion generator, retargeting pipeline, and whole-body controller, with paired initial states, random seeds, and command sequences. Full-task success requires completion of all prescribed stages

without a fall, protective stop, or human intervention. We additionally evaluate stage completion, command transitions, tracking accuracy, stance slip, and response latency. The model generates 52-frame windows at 25 Hz with four windows of recent context. Detailed evaluation criteria are provided in App. D.

**Deployment architecture.** The deployment comprises a remote RTX 4090 server for text encoding and motion generation, an intermediate workstation for communication and interaction, and the onboard G1 controller for motion tracking. Generated motion references are transmitted through the workstation via ZeroMQ and SSH tunneling at a 50-Hz transport rate, distinct from the model’s 25-Hz motion representation. The SONIC controller tracks the received references without executing the motion generator onboard. **Interactive timing.** We distinguish handoff latency from physical-response latency. Handoff latency is measured from system receipt of a text or spatial command to completion of the corresponding motion reference on the GPU server. It includes motion generation, decoding, accepted-state updates, and text encoding when a new text instruction requires it, while excluding network transmission and robotside processing. Physical-response latency starts at the same command receipt event and ends at the onset of sustained motion satisfying the command tolerance on the G1, additionally accounting for communication, buffering, and physical execution delays. These measurements are distinct from the cached-text inference timings in Sec. 5.4.

## 6.1 Motion Tracking with SONIC

We use SONIC [61] to track the generated motion references in MuJoCo. Fig. 9 shows four instruction sequences, including locomotion reversal, directional turning, and transitions between articulated actions and walking. For each sequence, the upper row shows physics-based execution in MuJoCo, where the SONIC controller tracks the generated motion reference and the resulting simulated motion is rendered. The lower row shows a kinematic visualization of the same reference in Viser: forward kinematics is used to update the robot mesh, which is then rendered without controller tracking or physics simulation. The two rows are shown at matching timestamps to compare the simulated execution with the reference motion. These examples qualitatively illustrate the transfer of generated motion sequences to a tracking controller under changing instructions, complementing the quantitative physical evaluation below.

## 6.2 Sequential and Expressive Motion

The T1 example contains changes in both global displacement and articulated pose. Forward progression is followed by a visible bow at 11.61 s and a raised-arm gesture at 28.83 s, with intermediate changes in body orientation (Fig. 8, panel (a)). T2 exhibits a different combination of arm elevation, torso orientation, and lower-body configuration, extending to the 32.48 s sample (Fig. 8, panel (b)). These examples show that the physical repertoire extends beyond a repeated walking cycle. The robot and reference preserve recognizable action structure, while differences in trunk inclination and foot placement remain visible.

<!-- page 13 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

13

(a) T1: Sequential action composition

![Image block](doc:a492382/tier:basic/page:13/block:4)

A person is standing. → A young lady walks forward elegantly. → A person bows down and then stands upright. → A person walks around in a circle. → A person stops. → A person waves his hand. → A person is standing.

(b) T2: Expressive motion

![Image block](doc:a492382/tier:basic/page:13/block:7)

A person is standing. → A person is performing some simple dance movements. → A person stops. → A person is celebrating. → A person is standing.

(c) T3: Keyboard-controlled locomotion

![Image block](doc:a492382/tier:basic/page:13/block:10)

(d) T4: Mouse-controlled locomotion

![Image block](doc:a492382/tier:basic/page:13/block:12)

T3 / T4: A person is standing. → A person slowly walks forward with short, controlled steps.

Fig. 8: Real-world qualitative results on Unitree G1. (a) T1, sequential action composition; (b) T2, expressive motion; (c) T3, keyboard-controlled locomotion; and (d) T4, mouse-controlled locomotion. Each panel shows real-robot frames, Viser references, timestamps, and the complete prompt sequence.

Tab. 8 summarizes sequential action execution on the physical G1. Full-task success reaches 18/20 (90.0%) for T1 and 19/20 (95.0%) for T2, with 136/140 and 99/100 completed stages, respectively. Of the scheduled text transitions, 117/120 in T1 and 78/80 in T2 succeed. These results complement the long-horizon command-matching evaluation in Sec. 5.2 by showing that successive generated motion references can be tracked through multiple textguided stages on the robot.

## 6.3 Interactive Control and Physical Stability

Fig. 8(c) and (d) demonstrate interactive locomotion through keyboard and mouse inputs, respectively. T3 and T4 test live spatial commands together with changes in text instructions. Tab. 8 reports full-task success of 29/30 (96.7%) for keyboard control and 28/30 (93.3%) for mouse control, with 30/30 and 29/30 successful scheduled text transitions, respectively. These results extend the evaluation of conditional motion generation to interactive commands during physical execution on the G1.

Tab. 9 reports tracking accuracy and responsiveness under interactive control. Keyboard and mouse interfaces yield speed RMSEs of 0.041/0.045 m/s, heading MAEs of 4.6◦/5.1◦, and median physical-response latencies of 351/360 ms, respectively. Within the 1.5-s response deadline, 296/300 (98.7%) keyboard events and 292/300 (97.3%) mouse

events are successfully executed. Stance-slip velocities remain low at 1.5/1.7 cm/s, reflecting limited foot sliding during interactive locomotion.

Across all four tasks, 94/100 trials achieve full-task success, 97/100 meet the stability criterion, and no falls are observed (Tab. 8). Three trials remain stable without completing all task requirements, while three fail the stability criterion. Stance slip further characterizes contact behavior. These results reflect the complete generation-and-tracking system rather than individual component failures.

## 7 LIMITATIONS AND FUTURE WORK

Our current framework focuses on controllable humanoid motion generation and does not yet address object manipulation. Future work will focus on whole-body locomanipulation, coordinating locomotion with object manipulation.

## 8 CONCLUSION

We presented Motion Mamba 2.0, an autoregressive diffusion framework that combines local attention, resumable Mamba states, and persistent content memory. Evaluations across three G1 motion benchmarks demonstrate improved terminal constraint accuracy, achieving Root CE of 7.55 cm and End CE of 13.64 cm at 39.52 s on retargeted HiPHI, corresponding

<!-- page 14 of 22 -->

Wave→walk

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

14

Forward → backward → stand

![Image block](doc:a492382/tier:basic/page:14/block:5)

Walk → turn left → stand

![Image block](doc:a492382/tier:basic/page:14/block:7)

Fig. 9: Simulated motion tracking with SONIC. Top left: forward to backward walking; top right: walking with a left turn; bottom left: waving followed by walking; bottom right: martial-arts motion followed by walking. Each quadrant compares simulated execution (upper row) with the corresponding Viser reference (lower row) at matching timestamps.

TABLE 8: Task Completion and Execution Stability. Stable denotes execution without a fall, protective stop, or human intervention. Text switches measure successful transitions between scheduled text instructions across all four tasks, while spatial-control events are evaluated separately.

| Task | Full success ↑ | SR (%) [95% CI] ↑ | Stages ↑ | Text switches ↑ | Stable ↑ | Falls ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| T1: action composition | 18/20 | 90.0 [69.9, 97.2] | 136/140 | 117/120 | 19/20 | 0 |
| T2: expressive motion | 19/20 | 95.0 [76.4, 99.1] | 99/100 | 78/80 | 20/20 | 0 |
| T3: keyboard control | 29/30 | 96.7 [83.3, 99.4] | 59/60 | 30/30 | 29/30 | 0 |
| T4: mouse control | 28/30 | 93.3 [78.7, 98.2] | 58/60 | 29/30 | 29/30 | 0 |

TABLE 9: Interactive Control Accuracy and Response Latency. Tracking and slip are episode mean ± SD; latency is p50/p95. Each interface has 30 episodes. The deadline counts include all 300 scheduled spatial events per interface.

| Measure | Keyboard | Mouse |
| --- | --- | --- |
| SHHRSEptveeaaeseannpenddcdoteiosnnRfswgsflMeipiMptpS5h(A50Eic0n/mE(/pm1/p(9.d5/s95e)5ss(g)↓m↑()↓ms↓)s)↓↓ | 0.0413241542±..6516960±±///.0533101000..255420 | 0.0452351562±..1710920±±///.0533101210..476500 |

to the lowest Root CE and End CE among the evaluated methods. Architectural ablations, including text-conditioning and state-difference variants, support the contributions of recurrent modeling and PCM to extended spatial control. The proposed execution optimizations further reduce median complete-window latency to 276.14 ms on an RTX 4090, achieving a median paired speedup of 6.138× while pre-serving the tested outputs and committed states. Physical G1 experiments achieve full-task completion in 94 of 100 trials and demonstrate multi-stage text-guided execution and responsive spatial control through the integrated generationand-tracking system. These results demonstrate the effectiveness of our framework in generating long-horizon, realtime controllable humanoid motion with bounded historical context.

## ACKNOWLEDGEMENT

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant No. 62676007.

## REFERENCES

[1] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or, and A. H. Bermano, “Human motion diffusion model,” in International Conference on Learning Representations, 2023. 2, 3

[2] M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu, “MotionDiffuse: Text-driven human motion generation with diffusion model,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 6, pp. 4115–4128, 2024. 2, 3

[3] X. Chen, B. Jiang, W. Liu, Z. Huang, B. Fu, T. Chen, and G. Yu, “Executing your commands via motion diffusion in latent space,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 000–18 010. 2, 3

[4] G. Barquero, S. Escalera, and C. Palmero, “Seamless human motion composition with blended positional encodings,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 457–469. 2, 3, 9, 10

[5] B. Han, H. Peng, M. Dong, Y. Ren, Y. Shen, and C. Xu, “AMD: Autoregressive motion diffusion,” in Proceedings of the AAAI Conference on Artificial Intelligence, ser. AAAI’24/IAAI’24/EAAI’24, vol. 38, no. 3. AAAI Press, 2024, pp. 2022–2030. 2, 3

[6] K. Zhao, G. Li, and S. Tang, “DartControl: A diffusion-based autoregressive motion model for real-time text-driven motion control,” in The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 9, 10

[7] K. Zhao, M. Petrovich, H. Zhang, T. Wang, S. Tang, and D. Rempe, “Autoregressive diffusion with hybrid representation for interactive human motion generation,” ACM Trans. Graph., vol. 45, no. 4, Jul. 2026. 2, 3, 4, 5, 9, 10

[8] L. Xiao, S. Lu, H. Pi, K. Fan, L. Pan, Y. Zhou, Z. Feng, X. Zhou, S. Peng, and J. Wang, “MotionStreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 10 086–10 096. 2, 3, 9, 10

[9] D. Fu, T. Sun, P. Fang, X. Cai, and H. Kim, “MOGO: Residual quantized hierarchical causal transformer for real-time and infinitelength 3D human motion generation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 5, 2026, pp. 3994–4002. 2, 3

<!-- page 15 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

15

[10] K. Karunratanakul, K. Preechakul, S. Suwajanakorn, and S. Tang, “Guided motion diffusion for controllable human motion synthesis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2151–2162. 2, 3

[11] Y. Xie, V. Jampani, L. Zhong, D. Sun, and H. Jiang, “OmniControl: Control any joint at any time for human motion generation,” in International Conference on Learning Representations, 2024. 2, 3

[12] E. Lee, E. Kim, S. Hong, E. Jung, and J. Kim, “Controllable longterm motion generation with extended joint targets,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), March 2026, pp. 5164–5173. 2, 4

[13] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in First Conference on Language Modeling, 2024. 2, 4

[14] A. Lahoti, K. Y. Li, B. Chen, C. Wang, A. Bick, J. Z. Kolter, T. Dao, and A. Gu, “Mamba-3: Improved sequence modeling using state space principles,” arXiv preprint arXiv:2603.15569, 2026. 2, 4, 6

[15] Z. Zhang, A. Liu, I. Reid, R. Hartley, B. Zhuang, and H. Tang, “Motion Mamba: Efficient and long sequence motion generation,” in Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part I. Berlin, Heidelberg: Springer-Verlag, 2024, pp. 265–282. 2, 3, 4, 9, 10

[16] L.-A. Zeng, G. Huang, G. Wu, and W.-S. Zheng, “Light-T2M: A lightweight and fast model for text-to-motion generation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 9797–9805. 2, 4

[17] Y. Ge, C. Li, L. Zhu, J. Gao, and X. Lin, “MADM: Bridging Mamba and autoregressive diffusion for text-to-motion generation,” Pattern Recognition, vol. 180, p. 114459, 2026. 2, 4

[18] Q. Nguyen, T. Le, B. Huang, M. N. Vu, N. Le, T. Vo, and A. Nguyen, “Learning human motion with temporally conditional Mamba,” in Proceedings of the SIGGRAPH Asia 2025 Conference Papers, ser. SA Conference Papers ’25. New York, NY, USA: Association for Computing Machinery, 2025. 2, 4

[19] J. Tanke, T. Shibuya, K. Uchida, K. Saito, and Y. Mitsufuji, “Dyadic Mamba: Long-term dyadic human motion synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2025, pp. 2893–2902. 2, 4

[20] NVIDIA, “CUDA programming guide: CUDA graphs,” accessed: Oct. 23, 2026. 2

[21] P. Tillet, H. T. Kung, and D. Cox, “Triton: An intermediate language and compiler for tiled neural network computations,” in Proceedings of the ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019, pp. 10–19. 2

[22] W. Zhou, Z. Dou, Z. Cao, Z. Liao, J. Wang, W. Wang, Y. Liu, T. Komura, W. Wang, and L. Liu, “EMDM: Efficient motion diffusion model for fast and high-quality motion generation,” in Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part II. Berlin, Heidelberg: Springer-Verlag, 2024, pp. 18–38. 3

[23] T. Li, C. Qiao, G. Ren, K. Yin, and S. Ha, “AAMDM: Accelerated Auto-Regressive Motion Diffusion Model ,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, Jun. 2024, pp. 1813–1823. 3

[24] J. Ji, J. Ma, R. Zhang, R. Yu, W. Wang, W. Chi, Q. Peng, W. Yan, Y. Gu, Y. Tian, T. Wu, L. Li, C. Yuan, R. Dai, and L. Han, “HiPHI: A large-scale benchmark for high-precision human motion and object-interaction,” arXiv preprint arXiv:2608.16222, 2026. 3, 4, 9

[25] BONES Studio, “BONES-SEED: Skeletal everyday embodiment dataset,” Dataset, 2026, accessed: Oct. 23, 2026. 3, 4, 9

[26] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng, “Generating diverse and natural 3D human motions from text,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5152–5161. 3, 4, 9

[27] J. Zhang, Y. Zhang, X. Cun, Y. Zhang, H. Zhao, H. Lu, X. Shen, and Y. Shan, “Generating human motion from textual descriptions with discrete representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 730–14 740. 3

[28] E. Pinyoanuntapong, P. Wang, M. Lee, and C. Chen, “MMM: Generative masked motion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1546–1555. 3

[29] T. Lee, G. Moon, and K. M. Lee, “MultiAct: Long-term 3D human motion generation from multiple action labels,” in Proceedings of the

AAAI Conference on Artificial Intelligence, vol. 37, 2023, pp. 1231–1239. 3

[30] T. Lee, F. Baradel, T. Lucas, K. M. Lee, and G. Rogez, “T2LM: Long-term 3D human motion generation from multiple sentences,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2024, pp. 1867–1876. 3

[31] Y. Shafir, G. Tevet, R. Kapon, and A. H. Bermano, “Human motion diffusion as a generative prior,” in The Twelfth International Conference on Learning Representations, 2024. 3

[32] M. Petrovich, O. Litany, U. Iqbal, M. J. Black, G. Varol, X. B. Peng, and D. Rempe, “Multi-track timeline control for text-driven 3D human motion generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2024, pp. 1911–1921. 3

[33] S. Chi, H.-g. Chi, H. Ma, N. Agarwal, F. Siddiqui, K. Ramani, and K. Lee, “M2D2M: Multi-motion generation from text with discrete diffusion models,” in Computer Vision – ECCV 2024, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds. Cham: Springer Nature Switzerland, 2025, pp. 18–36. 3

[34] Z. Yang, B. Su, and J.-R. Wen, “Synthesizing long-term human motions with diffusion models via coherent sampling,” in Proceedings of the 31st ACM International Conference on Multimedia. Association for Computing Machinery, 2023, pp. 3954–3964. 3, 9, 10

[35] W. Zhuo, F. Ma, and H. Fan, “InfiniDreamer: Arbitrarily long human motion generation via segment score distillation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 14 688–14 698. 3

[36] Y. Yang, Z. Huang, C. Xu, and S. He, “Lagrangian motion fields for long-term motion generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1171–1184, 2026. 3

[37] Y. Shi, J. Wang, X. Jiang, B. Lin, B. Dai, and X. B. Peng, “Interactive character control with auto-regressive motion diffusion models,” ACM Transactions on Graphics, vol. 43, no. 4, pp. 143:1–143:14, 2024. 3

[38] Y. Cai, Y. Wu, K. Li, T. Fangyuan, X. Zhang, Q. Li, Z. Wang, K. Zhang, and H. Liu, “FloodDiffusion 2: Efficient and path controllable streaming motion generation,” arXiv preprint arXiv:2609.33167, 2026. 3, 9, 10

[39] Q. Yu, A. Watanabe, and K. Fujiwara, “Causal motion diffusion models for autoregressive motion generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 38 366–38 375. 3, 9, 10

[40] A. Gu, I. Johnson, K. Goel, K. Saab, T. Dao, A. Rudra, and C. Ré, “Combining recurrent, convolutional, and continuous-time models with linear state space layers,” in Advances in Neural Information Processing Systems, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., vol. 34, 2021, pp. 572–585. 4

[41] A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences with structured state spaces,” in International Conference on Learning Representations, 2022. 4

[42] A. Gupta, A. Gu, and J. Berant, “Diagonal state spaces are as effective as structured state spaces,” in Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., vol. 35, 2022, pp. 22 982–22 994. 4

[43] A. Gu, K. Goel, A. Gupta, and C. Ré, “On the parameterization and initialization of diagonal state space models,” in Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., vol. 35, 2022, pp. 35 971–35 983. 4

[44] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, “Vision Mamba: Efficient visual representation learning with bidirectional state space model,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 62 429–62 442. 4

[45] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao, and Y. Liu, “VMamba: Visual state space model,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 103 031–103 063. 4

[46] S. Li, H. Singh, and A. Grover, “Mamba-ND: Selective state space modeling for multi-dimensional data,” in Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXXIII. Berlin, Heidelberg: Springer-Verlag, 2024, pp. 75–92. 4

[47] T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality,” in Proceedings of the International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 10 041–10 071. 4

<!-- page 16 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

16

[48] F. Wang, W. Wang, D. Gao, X. Huang, X. Song, H. Sun, and C. Peng, “A LLM-guided hybrid Mamba-transformer architecture for part-to-whole motion synthesis,” Computer Vision and Image Understanding, vol. 262, p. 104549, 2025. 4

[49] Z. Zhang, H. Gao, A. Liu, Q. Chen, F. Chen, Y. Wang, D. Li, R. Zhao, Z. Li, Z. Zhou, H. Tang, and B. Zhuang, “KMM: Key frame mask Mamba for extended motion generation,” arXiv preprint arXiv:2411.06481, 2024. 4

[50] Z. Zhang, A. Liu, Q. Chen, F. Chen, I. Reid, R. Hartley, B. Zhuang, and H. Tang, “InfiniMotion: Mamba boosts memory in transformer for arbitrary long motion generation,” arXiv preprint arXiv:2407.10061, 2024. 4

[51] Z. Wu, Y. Sun, Y. Chen, X. Gu, R. Liu, and J. Chen, “InterMamba: Efficient human-human interaction generation with adaptive spatiotemporal Mamba,” IEEE Transactions on Visualization and Computer Graphics, vol. 32, no. 2, pp. 1928–1940, 2026. 4

[52] Z. Dai, Z. Yang, Y. Yang, J. G. Carbonell, Q. Le, and R. Salakhutdinov, “Transformer-XL: Attentive language models beyond a fixed-length context,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 2978–2988. 4

[53] A. Bulatov, Y. Kuratov, and M. Burtsev, “Recurrent memory transformer,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 11 079–11 091. 4

[54] Z. Gu, J. Li, W. Shen, W. Yu, Z. Xie, S. McCrory, X. Cheng, A. Shamsah, R. Griffin, C. K. Liu, A. Kheddar, X. B. Peng, Y. Zhu, G. Shi, Q. Nguyen, G. Cheng, H. Gao, and Y. Zhao, “Humanoid locomotion and manipulation: Current progress and challenges in control, planning, and learning,” IEEE/ASME Transactions on Mechatronics, vol. 31, no. 2, pp. 2300–2330, Apr. 2026. 4

[55] X. B. Peng, P. Abbeel, S. Levine, and M. van de Panne, “DeepMimic: Example-guided deep reinforcement learning of physics-based character skills,” ACM Transactions on Graphics, vol. 37, no. 4, pp. 143:1–143:14, 2018. 4

[56] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. M. Kitani, C. Liu, and G. Shi, “OmniH2O: Universal and dexterous human-to-humanoid whole-body teleoperation and learning,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270, 2025, pp. 1516–1540. 4

[57] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, “HumanPlus: Humanoid shadowing and imitation from humans,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270, 2025, pp. 2828–2844. 4

[58] X. Cheng, Y. Ji, J. Chen, R. Yang, G. Yang, and X. Wang, “Expressive whole-body control for humanoid robots,” in Robotics: Science and Systems XX, 2024. 4

[59] P. Dugar, A. Shrestha, F. Yu, B. van Marum, and A. Fern, “Learning multi-modal whole-body control for real-world humanoid robots,” Proceedings of the AAAI Symposium Series, vol. 7, no. 1, pp. 650–657, 2025. 4

[60] J. P. Araújo, Y. Ze, P. Xu, J. Wu, and C. K. Liu, “Retargeting matters: General motion retargeting for humanoid motion tracking,” arXiv preprint arXiv:2510.02252, 2025. 4

[61] Z. Luo, Y. Yuan, T. Wang, C. Li, F. Castañeda, S. Chen, Z.-A. Cao, J. Li, D. Minor, Q. Ben, J. Park, D. Sami, Z. Wang, X. Da, R. Ding, C. Hogg, L. Song, E. Lim, E. Jeong, T. He, H. Xue, W. Xiao, S. Yuen, J. Kautz, Y. Chang, U. Iqbal, L. Fan, and Y. Zhu, “SONIC: Supersizing motion tracking for natural humanoid whole-body control,” Science Robotics, vol. 11, no. 117, p. eaed4592, 2026. 4, 12

[62] G. Tevet, S. Raab, S. Cohan, D. Reda, Z. Luo, X. B. Peng, A. H. Bermano, and M. van de Panne, “CLoSD: Closing the loop between simulation and diffusion for multi-task character control,” in International Conference on Learning Representations, 2025. 4, 9, 10

[63] S. Huang, K.-Y. Lee, D. Qiao, G. He, Z. Wang, Y. Li, S. Zhu, and H. Zhao, “OMG: Omni-modal motion generation for generalist humanoid control,” arXiv preprint arXiv:2606.10340, 2026. 4, 9, 10

[64] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “AMASS: Archive of motion capture as surface shapes,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 5442–5451. 4

[65] J. Romero, D. Tzionas, and M. J. Black, “Embodied hands: Modeling and capturing hands and bodies together,” ACM Transactions on Graphics, vol. 36, no. 6, pp. 245:1–245:17, 2017. 4

[66] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6840–6851. 5

[67] W. Peebles and S. Xie, “Scalable diffusion models with trans-

formers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205. 6

[68] K. Fan, S. Lu, M. Dai, R. Yu, L. Xiao, Z. Dou, J. Dong, L. Ma, and J. Wang, “Go to zero: Towards zero-shot motion generation with million-scale data,” arXiv preprint arXiv:2507.07095, 2025. 9, 10

[69] D. Kalaria, S. S. Harithas, P. Katara, S. Kwak, S. Bhagat, S. Sastry, S. Sridhar, S. H. Vemprala, A. Kapoor, and J. C.-K. Huang, “Dream-Control: Human-inspired whole-body humanoid control for scene interaction via guided diffusion,” arXiv preprint arXiv:2509.14353, 2025. 9, 10

[70] Y. He, G. Tiwari, X. Zhang, P. Bora, T. Birdal, J. E. Lenssen, and G. Pons-Moll, “MoLingo: Motion-language alignment for textto-human motion generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 9, 10

[71] Tencent Hunyuan 3D Digital Human Team, “HY-Motion 1.0: Scaling flow matching models for text-to-motion generation,” arXiv preprint arXiv:2512.23464, 2025. 9, 10

[72] W.-J. Huang, Y.-M. Li, K.-Y. Lin, W. Luo, Y. Zhu, Y. Yu, S. Ye, J. Yuan, F.-T. Hong, Q. Zhang, and W.-S. Zheng, “TACD: Distilling efficient text-to-motion models via terminal amplification control,” arXiv preprint arXiv:2610.02867, 2026. 9, 10

[73] Q. Wu, Y. Zhao, Y. Wang, X. Liu, Y.-W. Tai, and C.-K. Tang, “Motion-Agent: A conversational framework for human motion generation with llms,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 48 023–48 043. 9, 10

[74] B. Wu, J. Xie, K. Shen, Z. Kong, J. Ren, R. Bai, R. Qu, and L. Shen, “MG-MotionLLM: A unified framework for motion comprehension and generation across multiple granularities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 27 849–27 858. 9, 10

[75] Y.-M. Li, Q. Yang, N. Lei, S. Fu, L.-A. Zeng, J.-F. Hu, X. Wei, and W.-S. Zheng, “IRG-MotionLLM: Interleaving motion generation, assessment and refinement for text-to-motion generation,” arXiv preprint arXiv:2512.10730, 2025. 9, 10

[76] E. Nazarenus, C. Li, Y. He, X. Xie, J. E. Lenssen, and G. Pons-Moll, “ActionPlan: Future-aware streaming motion synthesis via frame-level action planning,” arXiv preprint, 2026. 9, 10

[77] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021. 9

![Image block](doc:a492382/tier:basic/page:16/block:34)

Yue Huang is an undergraduate student at South China University of Technology. Currently, his main research interests lie in Humanoid locomotion, 3D perspective modeling, world models, and quantitative investment.

![Image block](doc:a492382/tier:basic/page:16/block:36)

**Tianshan Zhang** is currently a research assistant at Peking University. Previously, he was a research intern at the Institute of Automation, Chinese Academy of Sciences (CASIA). His research interests include embodied AI and robot learning, with a particular focus on generative models, vision-language-action models, and reinforcement learning.

<!-- page 17 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

17

![Image block](doc:a492382/tier:basic/page:17/block:3)

**Zeyu Zhang** is a researcher working on generative AI, with a particular interest in building models that understand and interact with the physical world. He received his bachelor’s degree from the Australian National University, where he was advised by Prof. Richard Hartley and Prof. Ian Reid. His research explores generative modeling for learning physical dynamics from visual data. His work spans world models, multi-modal foundation models, embodied AI, and AI for health.

![Image block](doc:a492382/tier:basic/page:17/block:5)

**Jayeon Kang** is currently pursuing an M.S. degree in artificial intelligence at Korea University, South Korea. Her research interests span 3D computer vision, human motion generation and generative modeling. She is also interested in state-space models and their potential for embedded AI.

![Image block](doc:a492382/tier:basic/page:17/block:7)

**Akide Liu** is currently pursuing the Ph.D. degree in computer science at Monash University, Australia, under the supervision of Dr. Bohan Zhuang and Prof. Gholamreza Haffari. Akide received the B.S. (Honours) degree in computer science with First Class Honours from the University of Adelaide, Australia, in 2023, and has been a research intern at ByteDance and at DAMO Academy, Alibaba Group. Research interests include efficient AI, unified vision-language models, and generative models.

![Image block](doc:a492382/tier:basic/page:17/block:9)

**Sungjoon Choi** is an Associate Professor in the Department of Artificial Intelligence at Korea University, South Korea. Previously, he was a Staff Software Engineer at Google Research in Mountain View, CA, USA. He received the Ph.D. degree in electrical engineering from Stanford University under the supervision of Vladlen Koltun and the B.S. degree in computer science and engineering from Seoul National University. His research interests include generative models, computational imaging, efficient visual computing,

and 3D vision. He was a recipient of the Samsung Scholarship and a two-time finalist in the ACM-ICPC World Finals.

![Image block](doc:a492382/tier:basic/page:17/block:12)

**Dacheng Tao** (Fellow, IEEE) is currently a Distinguished University Professor in the College of Computing & Data Science at Nanyang Technological University. He mainly applies statistics and mathematics to artificial intelligence and data science, and his research is detailed in one monograph and over 200 publications in prestigious journals and proceedings at leading conferences, with best paper awards, best student paper awards, and test-of-time awards. He received the 2015 and 2020 Australian Eureka

Prize, the 2018 IEEE ICDM Research Contributions Award, and the 2021 IEEE Computer Society McCluskey Technical Achievement Award. He is a Fellow of the Australian Academy of Science, AAAS, ACM and IEEE.

![Image block](doc:a492382/tier:basic/page:17/block:15)

**Hao Tang** is an Assistant Professor at Peking University, China. Previously, he held postdoctoral positions at CMU, USA, and ETH Zürich, Switzerland. He earned his master’s degree from Peking University, and his Ph.D. from the University of Trento, Italy. He has had the opportunity to visit the University of Oxford, NEU, NUS, and IIAI, among other institutions. His research focuses on generative AI, world models, spatial intelligence, embodied AI, LLMs, MLLMs, and computer vision.

<!-- page 18 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

18

## APPENDIX A

## DATASET CONSTRUCTION AND RETARGETING

## A.1 Corpus and Conversion

Tab. 10 reports the fixed splits. Source families and admitted mirrors follow the corresponding split manifests; the reported durations count stored frames at 25 Hz. HumanML3D motions are reconstructed from AMASS/SMPL-H parameters, and the selected high-precision human-motion subset of HiPHI uses BVH forward kinematics. The HiPHI objectinteraction subset is excluded. BONES Seed retains its official G1 trajectories.

We use GMR with the same 29-DoF G1 kinematic model for retargeting and geometric evaluation. Source-specific restpose transforms and pelvis-height scaling align the human skeletons with G1. Root translations and local rotations are resampled to 25 Hz using linear interpolation and SLERP. The solver combines position and orientation objectives under joint limits, followed by temporal arm refinement, collision refinement, and bounded ground and rate corrections. Forward-kinematic consistency of the robot assets is verified on 123 configurations.

## A.2 Mapped-Target Residuals and Representation Checks

The assessment covers all 18,525 HumanML3D and 19,503 HiPHI records across the three splits. Let $\mathbf { p } _ { i t j } ^ { \mathrm { G 1 } }$ be the robot body-origin position and $\widetilde { \mathbf { p } } _ { i t j } ^ { \mathrm { H } }$ its scaled, offset human target, in meters. For clip i and body group b, the per-clip residual in centimeters is

$$
e _ { i } ^ { ( b ) } = \operatorname* { m a x } _ { j \in \mathcal { J } _ { b } } Q _ { 0 . 9 5 , t } \left( 1 0 0 \left\| \mathbf { p } _ { i t j } ^ { \operatorname { G 1 } } - \widetilde { \mathbf { p } } _ { i t j } ^ { \operatorname { H } } \right\| _ { 2 } \right) ,\tag{16}
$$

where $Q _ { 0 . 9 5 , t }$ is the 95th percentile over valid frames, and $\mathcal { I } _ { b }$ contains the pelvis or both ankles. Tab. 11 summarizes these values with equal clip weights.

Final rate projection leaves joint angles unchanged in 37,881/38,028 records (99.61%). The largest final angle corrections are 0.254 and 0.229 rad for HumanML3D and HiPHI; the largest root-height lifts are 10.85 and 10.55 cm. All admitted human-source records pass the stored-format and 414-D round-trip checks, with maximum position discrepancy $1 . 3 1 \times 1 0 ^ { - \dot { 6 } }$ m. These checks establish numerical and representation consistency; the mapped-target statistics quantify geometric alignment to the specified pelvis and ankle landmarks.

## APPENDIX B

## MODEL AND INFERENCE IMPLEMENTATION

## B.1 Layer State and PCM Updates

Root and body layers use condition-modulated LayerNorm before the sequence mixer and feed-forward sublayer. Attention and Mamba receive the same mixer features. The

Mamba state carries the accumulated phase, recurrent matrix, preceding rotated state-input vector, and preceding value vector. Phase and matrix accumulation use FP32; the latter vectors follow the BF16 scan boundary. A session reset clears these tensors, the K/V cache, PCM slots, and preceding summaries together.

The PCM projector averages the recurrent matrix over the state dimension, flattens head and value axes, and applies a linear map and LayerNorm. Successive accepted summaries define the state difference. The masked mean of valid text embeddings supplies the write condition, independently of diffusion time and heading. A linear projection and LayerNorm form the text features; a linear conditioner produces scale and shift parameters for the normalized state difference. The summary, modulated difference, and text features construct the candidate content in Eq. (11). A linear decoder and featurewise LayerNorm produce slot proposals, while a linear sigmoid writer also reads the previous slots to predict the write gate in Eq. (12). The write-rate cap is $\eta = 0 . 2 5$ . Gates are broadcast over features, and slots are updated once per accepted window. During denoising, all guidance branches read the same committed cache, Mamba state, slots, and summary.

**Training-time context.** The complete denoiser, including PCM, is present from the beginning of training and is optimized under the clean-window generation objective; no separate PCM initialization or fine-tuning stage is used. Each training example predicts a single 52-frame generation window conditioned on sampled clean ground-truth historical motion, rather than on an autoregressively generated prefix. Training does not unroll multiple generated windows or perform cross-window truncated backpropagation through time. During inference, by contrast, persistent states are updated from accepted generated windows. The zero-initialized PCM readout scale in Eq. (13) suppresses the additional attention contribution at initialization; it does not imply that PCM is inserted after a separately trained Native stage. During joint training, the root-to-local conditioning path is detached before body prediction. Body losses therefore do not backpropagate through this conversion into the root denoiser, which remains supervised by its own prediction losses.

## B.2 Accepted-History Streaming Inference

The generation algorithm below makes root–body prediction, guidance, and persistent updates explicit. The clean pass uses the same root-to-local transformation and positional indexing as denoising. Future control tokens do not enter persistent history. Cache projections are refreshed whenever their model parameters change. In the algorithm, κ indexes the three guidance branches, ν denotes accepted-token validity, and S contains the complete resumable Mamba state; H is its recurrent matrix. The branch loop specifies

TABLE 10: Frozen corpus composition. Training, validation, and test columns count stored records. Durations include admitted mirrors and are computed at 25 Hz.

| Source | Training | Validation | Test | Training hours | All-split hours |
| --- | --- | --- | --- | --- | --- |
| BONES Seed | 124,658 | 7,578 | 9,384 | 256.24 | 286.95 |
| HumanML3D | 14,612 | 948 | 2,965 | 28.50 | 36.37 |
| HiPHI | 17,549 | 1,472 | 482 | 304.18 | 337.67 |
| Total | 156,819 | 9,998 | 12,831 | 588.92 | 660.98 |

<!-- page 19 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

19

TABLE 11: Mapped-target residuals in centimeters. Median, p95, and maximum summarize the per-clip residual defined above.

| Source | Body | Median | p95 | Max. |
| --- | --- | --- | --- | --- |
| HumanML3D | Pelvis Ankle | 0.991.49 | 3.073.37 | 9.6510.53 |
| HiPHI | Pelvis Ankle | 0.560.93 | 1.042.39 | 11.2529.06 |

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Diffusion-Safe Streaming Generation
Require: Conditions $\{ s _ { t } , \mathbf { g } _ { t } , \mathbf { v } _ { t } \}$ for N windows, initial history $\mathbf { m } _ { \mathrm { o b s } } ,$
diffusion steps K, cache width W, scales $( w _ { s } , w _ { g } )$
1: $\mathcal { M } _ { - 1 } \leftarrow \operatorname { I n i t } ( \mathbf { \overset { \mathbf { T } } { m _ { o b s } } } ) ; \mathbf { \overset { \mathbf { T } } { T } } \leftarrow \mathbf { 0 }$
2: for $t = 0 , \dots , N - \dot { 1 }$ do
3: $\mathcal { M } _ { \mathbf { \acute { e } a d } } ^ { \mathbf { \acute { r e a d } } } \underset {} { \underset {} { \underset {} { \underset {} { \underset {} { \underset {} { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \underset { } { \mathbf { } { } } } } } } } } } } } } } } } } } { \longleftarrow } } } } } } { \mathcal { M } } } } } { \mathbf { \acute { e } } } _ { t } } \leftarrow \operatorname { P o o l T e x t } ( s _ { t } )$
$\mathbf { z } _ { t , K } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$
for $\tau = K , \dot { \dots , 1 }$ do
$\mathbf { f o r } \: \kappa \in \dot { \{ s , g , \varnothing \} }$ do
$\mathbf { c } _ { \kappa } \xleftarrow [ ] { } \operatorname { C o n d } _ { \kappa } ^ { ' } ( s _ { t } , \mathbf { g } _ { t } , \mathbf { v } _ { t } , \tau )$
$\mathbf { u } _ { \kappa } ^ { r } \leftarrow \operatorname { R o o t I n p u t } _ { \kappa } ( \mathbf { z } _ { t } , \tau , \dot { \mathbf { g } } _ { t } , \mathbf { v } _ { t } )$
9: $\widehat { \mathbf { r } } _ { \kappa } ^ { r } \leftarrow \mathcal { F } _ { \theta } ^ { r } ( \mathbf { u } _ { \kappa } ^ { r } , \mathbf { c } _ { \kappa } ; \mathcal { M } ^ { r , \mathrm { r e a d } } )$
10: $\mathbf { u } _ { \kappa } ^ { b } \leftarrow \operatorname { \widehat { B o d y I n p u t } } _ { \kappa } ( \operatorname { L o c a l } ( \mathbf { \widehat { r } } _ { \kappa } ) , \mathbf { z } _ { t , \tau } , \mathbf { g } _ { t } , \mathbf { v } _ { t } )$
11: $\widehat { \mathbf { b } _ { \kappa } } \leftarrow \mathcal { F } _ { \theta } ^ { b } ( \mathbf { u } _ { \kappa } ^ { b } , \mathbf { c } _ { \kappa } ; \mathcal { \hat { M } } ^ { b , \operatorname { r e a d } } )$
12: $\widehat { \mathbf { x } } _ { \kappa } \gets [ \widehat { \mathbf { r } } _ { \kappa } ; \widehat { \mathbf { b } } _ { \kappa } ]$
13: end for
14: $\widehat { \mathbf { x } } _ { 0 } \leftarrow \widehat { \mathbf { x } } _ { \varnothing } + w _ { s } ( \widehat { \mathbf { x } } _ { s } - \widehat { \mathbf { x } } _ { \varnothing } )$
$+ w _ { g } ( \widehat { \mathbf { x } } _ { g } - \widehat { \mathbf { x } } _ { \varnothing } )$
15: $\mathbf { z } _ { t , \tau - 1 } \leftarrow \mathcal { S } _ { \tau } ( \widehat { \mathbf { z } _ { t , \tau } } , \widehat { \mathbf { x } } _ { 0 } )$
16: end for
17: $\begin{array} { r } { ( \mathbf { x } _ { t } ^ { \prime } , \mathbf { o } _ { t } ) \underset { \sim } { \mathrm { ~ \tiny ~ \leftarrow ~ } } \mathrm { R e c e n t e r } ( \mathbf { z } _ { t , 0 } ) } \end{array}$
18: {K, $\{ \mathbf { V } , \mathbf { \nu } , \widetilde { \mathcal { S } } \} _ { q , \ell } \leftarrow \operatorname { \widetilde { C a n o n i c a l } } _ { r \to b } ( \mathbf { x } _ { t } ^ { \prime } ; \mathcal { M } ^ { \operatorname { r e a d } } )$
19: for all root/body layers $( q , \ell )$ do
20: $\widetilde { \mathcal { C } } \leftarrow \operatorname * { F I F O } _ { \sim } \widetilde { ( \mathcal { C } _ { t - 1 } \cup \{ \mathbf { K } , \mathbf { V } , \nu \} ) }$
21: $\mathbf { \bar { s } } \leftarrow \Pi ( \mathbf { \widetilde { H } } ) ; \mathbf { d } \leftarrow \mathbf { \bar { s } } - \mathbf { \bar { s } } _ { t - 1 }$
22: $\mathbf { \underline { { j } } } \leftarrow \operatorname { T e x t C o n t e x t } ( \mathbf { \overline { { s } } } , \mathbf { d } , \mathbf { e } _ { t } )$
23: 2 $\mathbf { Z } \leftarrow \operatorname { L N } ( f _ { p } ( \mathbf { j } ) )$
24: $\mathbf { w } \leftarrow \eta \sigma ( \widetilde { f _ { w } } ( [ \mathbf { j } ; \operatorname { v e c } ( \mathbf { P } _ { t - 1 } ) ] ) )$
25: $\widetilde { \mathbf { P } } \xleftarrow { } ( 1 - \mathbf { w } ) \odot \mathbf { P } _ { t - 1 } + \mathbf { w } \odot \mathbf { Z }$
26: $\widetilde { \mathcal { M } } ^ { q } { } ^ { , \ell } \leftarrow \{ \widetilde { \mathcal { C } } , \widetilde { \mathcal { S } } , \widetilde { \mathbf { P } } , \overline { { \mathbf { s } } } \}$
27: end for
28: if Valid(Mf) then
29: $\mathcal { M } _ { t } \leftarrow \mathrm { A t o m i c C o m m i t } ( \widetilde { \mathcal { M } } ) ; \mathbf { T } \leftarrow \mathbf { T } + \mathbf { o } _ { t }$
30: Emit(Global(Decode(x′t), T))
31: else
32: $\mathcal { M } _ { t } \leftarrow \mathcal { M } _ { t - 1 }$
33: end if
34: end for
</div>

logical dependencies; the optimized implementation captures the three guidance branches together within each of the separate root and body DiT-stack graphs. Layer indices are omitted inside the accepted-state update loop for compactness; both denoisers retain separate layerwise states. Temporary denoising states are discarded. The subscript κ on input construction applies the corresponding text and spatial condition masks. TextContex $\mathrm { t } ^ { q , \ell }$ denotes the text projection, difference modulation, and concatenation in Eqs. (9)–(11).

## B.3 Fusion, Guidance, and Precision

For transition inputs $a _ { i } , d _ { i } ,$ decay floor $a _ { \mathrm { m i n } } > 0 ,$ and stepsize bias $b _ { \Delta } ,$ , preprocessing is

$$
\begin{aligned} { \chi ( a ) } & { { } = \operatorname* { m a x } ( a , 0 ) + ( 1 - \operatorname* { m i n } ( a , 0 ) ) ^ { - 1 } , } \\ { A _ { i } } & { { } = \operatorname* { m i n } ( - \chi ( a _ { i } ) , - a _ { \operatorname* { m i n } } ) , } \\ { \Delta _ { i } } & { { } = \operatorname { s o f t p l u s } ( d _ { i } + b _ { \Delta } ) , \qquad \Xi _ { i } = \Delta _ { i } A _ { i } . } \\ \end{aligned}\tag{17}
$$

The fused $\mathrm { A / D T }$ kernel emits $\Delta _ { i }$ and $\Xi _ { i }$ in the original scan layout, keeping intermediate preprocessing in FP32 registers. The scan and attention operators are retained.

Fused scale–shift and gated-residual operations preserve the reference casts and arithmetic ordering; fused multiply– add contraction is disabled where required for bitwise equivalence. Reported equivalence runs use FP32 attention, BF16 Mamba compute boundaries, FP32 state accumulation, and disabled TF32.

Three-branch classifier-free guidance combines text-only, spatial-only, and unconditional predictions:

$$
\widehat { \mathbf { x } } _ { 0 } = \widehat { \mathbf { x } } _ { 0 , \varnothing } + w _ { s } ( \widehat { \mathbf { x } } _ { 0 , s } - \widehat { \mathbf { x } } _ { 0 , \varnothing } ) + w _ { g } ( \widehat { \mathbf { x } } _ { 0 , g } - \widehat { \mathbf { x } } _ { 0 , \varnothing } ) .\tag{18}
$$

The root and body DiT sequence stacks use separate CUDA Graphs. Within each stack, the text-only, spatial-only, and unconditional guidance branches execute serially and are captured together. Input projections, conditioning preparation, root-output processing, root-to-local conversion, and bodyinput construction remain outside the stack-level graphs. Branch predictions are combined according to the guidance equation above. Capture reduces host dispatch without changing guidance weights, model parameters, or the rootto-body dependency. Sampling, random-number handling, recentering, and accepted-state commitment also remain outside capture. Graphs are rebuilt when tensor layouts or parameter versions become incompatible; unsupported calls use the reference path.

## B.4 Complete-Window Timing and Equivalence

The fixed-checkpoint study follows the hardware, precision, sampling, and five-process paired protocol in Sec. 5.4. Complete-window timing includes state restoration, sampling, accepted-state commitment, decoding, and CPU snapshot/export; text encoding, loading, warmup, and capture are excluded. Speedup is the median paired ratio, with a 95% interval from 10,000 process-block bootstrap resamples. Four seeded 100-window rollouts verify bitwise-identical outputs and committed states. This cohort is independent of the model-comparison timings.

## APPENDIX C

## BENCHMARK EVALUATION PROTOCOLS

## C.1 Common Evaluation Conditions

All methods follow the paired protocol in Sec. 5.1: 128 heldout initialization source families and three noise seeds per family yield 384 rollouts per duration. Initializations and root and end-effector goals are paired across methods. No future ground-truth motion is provided.

For HiPHI, the same 128 initialization families, initial motion histories, physical targets, and noise seeds are shared across horizons. These initialization families are distinct from the phase-specific motion-reference families and distinct-caption retrieval-query sets described below. Root CE and End CE are averaged over all 384 terminal rollout measurements. Cmd. is computed as the mean frame-level coarse root-state agreement within each 52-frame window, averaged over windows within each rollout and then over all 384 rollouts. R@3 and L2-FID follow the window-level aggregation described below.

Although initialization cases are paired across horizons, the instruction-phase composition and window weights differ. Cross-horizon differences therefore describe performance

<!-- page 20 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

20

across the evaluated horizons and do not isolate the effect of duration or temporal error accumulation.

Model-comparison latency uses two warmups and 32 paired batch-one rollouts with cached text, including generation, decoding, and accepted-state updates. These timing samples are separate from the 384 quality-evaluation rollouts per duration.

## C.2 Training and Baseline Adaptations

**Training configuration.** Motion Mamba 2.0 is trained jointly with Mamba-3 and PCM enabled from initialization. The generator starts from randomly initialized parameters (initialization seed 123), rather than an official pretrained generator checkpoint; the shared G1 motion codec is frozen and text features are cached. The model is trained for 100,000 optimizer updates using AdamW with $( \beta _ { 1 } , \beta _ { 2 } )   =   ( 0 . 9 , 0 . 9 \bar { 9 9 } )$ $\epsilon \; = \; { 1 0 } ^ { \frac { 1 } { } 8 }$ , weight decay 0.01, and global gradient-norm clipping at 1.0. The learning-rate schedule uses 500 warmup updates from $1 0 ^ { - 6 } \; \mathrm { t o } \; 5 \times \overset { \circ } { 1 0 } \vphantom { 1 0 ^ { - 5 } }$ , followed by cosine decay to $1 \dot { 0 } ^ { - 5 }$ over the 100,000-update schedule. The released training recipe uses global batch size 512 throughout all 100,000 optimizer updates, totaling 51.2 million sample exposures. There is no separate PCM pretraining or late-stage PCM fine-tuning.

Training samples draw from BONES Seed, retargeted HumanML3D, and retargeted HiPHI with probabilities 0.70, 0.15, and 0.15. The generation window contains 52 frames. History/future-conditioning lengths (H, F) take the values (0, 0), (52, 0), (196, 0), (0, 196), (52, 144), and (144, 52) with respective probabilities 0.10, 0.25, 0.20, 0.10, 0.20, and 0.15. Text dropout and spatial-constraint dropout are each 0.1, and the probability of empty spatial conditioning is 0.2. The supported spatial conditions are root trajectories, root waypoints, full keyframes, joint positions, and joint rotations. The largest sampled historical prefix is 196 frames (7.84 s), shorter than the four-window 208-frame local cache. Thus, this training protocol does not directly supervise retrieval from model-generated histories beyond the local attention horizon; longer-horizon behavior is evaluated through autoregressive inference.

Diffusion training uses clean-window (x0) prediction with a uniformly sampled base diffusion timestep and Gaussian forward noise. The base noise schedule is cosine, with offset $s   =   0 . 0 0 8$ and maximum $\beta   =   0.999;$ the number of base diffusion steps is read from the loaded G1 configuration, not specified by the ten-step DDIM inference schedule. The four terms in Eq. (15) have unit coefficients. The hybrid reconstruction term uses Smooth L1 with $\beta _ { \mathrm { S m o o t h L 1 } } = 1$ over the 148-dimensional hybrid representation; decoded-body and observed-goal terms use normalized L1, while forwardkinematics consistency measures joint-position disagreement in meters. The hybrid root/body subcomponents are diagnostic reductions, not separately weighted losses. These settings describe the common G1 diffusion $\sqrt { x _ { 0 } }$ objective and should not be read as the objectives for every model family.

**Controlled baseline adaptations.** All compared methods and architectural variants are independently trained on the same G1 corpus under 100,000 optimizer updates and the matched global batch-size schedule, using newly trained 100k checkpoints rather than their original released generator weights. Their objectives remain architecture-specific:

the shared G1 diffusion adaptations use the clean-motion objective above, autoregressive action-code adaptations use token cross-entropy, and MoLingo retains its masked-flow velocity objective. These distinctions preserve the training formulation of each model family instead of imposing a diffusion objective on non-diffusion methods. The G1 adaptation and evaluation interface applies paired physical constraints and matched data splits to all compared methods.

Minimal uses FP32 PyTorch SDPA with CUTLASS memory-efficient attention and no fused-kernel or CUDA Graph acceleration. Native configurations use their admitted accelerated execution paths. Consequently, architectural tables primarily compare quality and control; their latency columns describe the complete evaluated implementations. The fixed-weight component study in Sec. 5.4 separately measures execution acceleration.

## C.3 BONES Seed and HumanML3D Retrieval and Reference Distribution

BONES Seed and retargeted HumanML3D use separately trained, frozen evaluators. For each query, R@3 ranks one text-matched generated motion against 31 fixed real-motion distractors. Within each benchmark, all methods share the same evaluator, candidate construction rules, and sampling protocol. BONES Seed retains its official G1 motion references. The HumanML3D evaluator uses three fixed candidate pools and two frozen retrieval heads. Its source pool contains accepted motions shorter than 10 s, limiting evaluation to 2.08 and 8.32 s. Under this protocol, R@3 measures referenceanchored retrieval rather than retrieval among 32 generated motions.

## C.4 HiPHI Retrieval and Reference Distribution

The HiPHI evaluator is independently trained and frozen for all compared generators. Initializations span the same 128 source families from three actors at every evaluated horizon. For instruction phases 1–4, respectively, the numbers of distinct-caption retrieval queries are 61/110/61/42, and the numbers of distinct reference source families are $9 3 / 1 2 8 / 9 3 / 6 3$ . These counts describe instruction phases, not the four evaluation durations. Within each phase, all 128 initialization cases receive an assigned real-motion reference entry; repeated entries can originate from the same reference source family without excluding any generated rollouts. The detailed-instruction reference bank is separated by source family from generator training and evaluator training/development.

The numbers of windows allocated to phases 1–4 are (0, 0, 0, 1), (1, 1, 1, 1), (3, 1, 2, 3), and (6, 2, 5, 6) at 2.08, 8.32, 18.72, and 39.52 s, respectively. In particular, the singlewindow evaluation uses phase 4. Initial motion histories, spatial goals, and noise seeds are paired across horizons, but phase composition, the number of windows per phase, and terminal deadlines differ. Cross-horizon metric changes therefore do not isolate the effect of duration alone.

For generated models, R@3 ranks 32 generated motion candidates from the same model, window, and noise seed. The phase-specific distinct-caption queries are evaluated with three fixed candidate pools, two frozen retrieval heads, and three generator noise seeds. R@3 is averaged within each

<!-- page 21 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

21

window and then arithmetically averaged over windows at the corresponding horizon. Consequently, phase contributions are weighted by their window counts, not equally across the four phases; the resulting retrieval score is not based on 384 independent text queries.

All three benchmarks compute L2-FID from individually L2-normalized 256-dimensional motion features of phasepaired 52-frame windows. For HiPHI, each window compares 384 generated features (128 initializations by three noise seeds) with 128 assigned real-reference entries, one per initialization case. Multiple reference entries may share a source family, and their case-assignment multiplicities are retained. HiPHI L2-FID is the arithmetic mean of its windowlevel values, rather than FID over embeddings of complete 39.52-s trajectories. A real-reference distribution has zero L2-FID under self-comparison; the R@3 reference-calibration values are reported separately in Tab. 12. Since evaluators, retrieval candidate constructions, and reference distributions differ, absolute R@3 and L2-FID scores are interpreted within each benchmark. Command matching measures coarse rootmotion agreement rather than complete natural-language instruction fulfillment.

## C.5 Real-Motion Reference Calibration

Tab. 12 reports real-motion reference calibration scores under the three frozen benchmark-specific evaluators. These scores characterize reference-motion retrieval under the corresponding query sets and candidate-construction protocols. They are not generated-model results, teacher-model outputs, or theoretical upper bounds.

For HiPHI, reference calibration uses phase-paired real motions and the corresponding frozen phase-specific queries. The four calibration values in Tab. 12 are arranged by evaluated horizon; their component query sets follow the phase-to-window allocation stated above. Generated models are evaluated separately using 32 same-window candidates produced by the same method and noise seed. For BONES Seed and HumanML3D, calibration uses paired real-motion candidates, whereas generated models use one matched generated motion and 31 fixed real-motion distractors.

Because the candidate constructions differ, the reference scores are reported as evaluator diagnostics rather than direct model-performance comparisons.

## C.6 Qualitative Long-Horizon Goal-Conditioning Comparison

Although both methods receive the same distant target and deadline from the first generated window, their progress profiles differ. Motion Mamba 2.0 accumulates goal-directed displacement more continuously: it reaches 9.91 m at 16.68 s and 15.05 m at 25.04 s, compared with 6.97 m and 9.36 m for ARDY. The ARDY trajectory also contains two extended nearstationary intervals, approximately 6.7–13.6 s and 18.7–22.8 s, during which its planar root speed remains below 0.1 m/s over rolling one-second windows. It then makes a rapid late correction: its mean root speed over the final second is 1.99 m/s, compared with 0.71 m/s for Motion Mamba 2.0. This selected example suggests that Motion Mamba 2.0 allocates motion toward the distant goal earlier and more continuously, whereas ARDY defers a larger portion of its translation until

TABLE 12: Real-Motion Reference Calibration. R@3 (%) is reported for real-motion reference data under each benchmark-specific evaluation protocol. These values are not generated-model results or theoretical performance bounds.

| Benchmark | 2.08s | 8.32s | 18.72s | 39.52s |
| --- | --- | --- | --- | --- |
| HiPHI | 44.44 | 53.59 | 52.55 | 52.76 |
| BONES Seed | 88.15 | 81.12 | - | - |
| HumanML3D | 91.39 | 75.96 | - | - |

later in the rollout. The corresponding root trajectories and quantitative goal-progress profiles are presented in Fig. 10.

## APPENDIX D

## PHYSICAL EVALUATION PROTOCOLS

## D.1 Task Completion and Tracking Measures

The main suite contains 20 trials each for T1/T2 and 30 each for T3/T4. Prompt sequences are shown in Fig. 8; complete recordings are available on the project website. Let $a _ { i k }$ indicate completion of stage k in trial $i ,$ and $b _ { i }$ indicate execution without a fall, protective stop, or human intervention. Full-task and stage success are

$$
\begin{aligned} { \operatorname { S R } } & { { } = \frac { 1 0 0 } { N } \sum _ { i } b _ { i } \prod _ { k = 1 } ^ { K _ { i } } a _ { i k } , } \\ { \operatorname { S t a g e S R } } & { { } = 1 0 0 \frac { \sum _ { i } \sum _ { k = 1 } ^ { K _ { i } } a _ { i k } } { \sum _ { i } K _ { i } } . } \\ \end{aligned}\tag{19}
$$

All initiated trials and planned stages enter the denominators; unreached stages fail. Standing and stopping require root speed below 0.05 m/s for one second. A bow requires trunk flexion followed by upright recovery; circle completion requires endpoint error below 0.25 m and heading closure within $1 5 ^ { \circ }$ . Waving, dance, and celebration are evaluated by their prescribed actions. Keyboard/mouse success requires at least 90% of post-settling samples within 0.10 m/s speed error and $1 5 ^ { \circ }$ heading error, together with the final stop. Text-switch success requires action onset within 1.5 s without a protective intervention before the next switch. Spatial-event response counts are reported separately.

For measured and commanded speed $v _ { t } , v _ { t } ^ { \star }$ and heading $\psi _ { t } , \psi _ { t } ^ { \star }$ , episode-level errors are

$$
\begin{aligned} { E _ { v } } & { { } = \sqrt { M ^ { - 1 } \sum _ { t } ( v _ { t } - v _ { t } ^ { \star } ) ^ { 2 } } , } \\ { E _ { \psi } } & { { } = M ^ { - 1 } \sum _ { t } | \operatorname { w r a p } ( \psi _ { t } - \psi _ { t } ^ { \star } ) | . } \\ \end{aligned}\tag{20}
$$

Tracking statistics exclude the fixed 1.5-s settling allowance, which remains included in latency and deadline-response statistics. Stance slip is contact-duration-weighted horizontal foot speed in world coordinates, excluding swing phases.

## D.2 Latency and Statistical Aggregation

Handoff latency starts upon receipt of a text or spatial command that triggers motion generation and ends when the corresponding motion reference is ready on the GPU server. It includes sampling, decoding, accepted-history updates, and text encoding when required by a text change, but excludes network communication and robot-side operations.

Physical-response latency starts at the same command receipt event and ends at the onset of a 0.5-s interval satisfying

<!-- page 22 of 22 -->

SUBMITTED TO IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

22

![Image block](doc:a492382/tier:basic/page:22/block:3)

![Chart block](doc:a492382/tier:basic/page:22/block:4)

(b) Progress toward the distant goal

![Chart block](doc:a492382/tier:basic/page:22/block:6)

Fig. 10: Qualitative comparison under a distant single-point constraint. ARDY (upper row) and Motion Mamba 2.0 (lower row) are shown at the same four timestamps. Both receive the common 18 m target and its 33.40 s deadline from the start. Bottom: (a) root trajectories in goal-aligned coordinates; (b) displacement along the goal direction over time. Our model makes greater progress by 25.04 s (15.05 versus 9.36 m) and has a lower final-second mean root speed (0.712 versus 1.992 m/s).

the corresponding motion tolerance. It additionally includes network transfer, controller-side buffering, processing, and physical response. Latency quantiles summarize responding events; nonresponding or censored events remain failures in the corresponding deadline counts. Tracking and slip are aggregated as episode-level mean ± SD. Wilson intervals quantify trial-level success rates.