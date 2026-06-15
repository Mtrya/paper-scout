# Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack

Tencent Robotics X × Tencent Hy Team

The past year has witnessed a rapid proliferation of Vision-Language-Action (VLA) models, with growing attention now turning to the next generation of embodied foundation models. However, a truly generalist robot is unlikely to emerge from any single model in isolation. Rather, it must be built on a full robot learning stack that remains robust from data collection to real-world deployment.

In this report, we present Hy-Embodied-0.5-VLA, abbreviated as HyVLA-0.5, an end-to-end system that spans the full robot learning stack: data collection, model design, continued pre-training and supervised fine-tuning, RL post-training, and real-world deployment. Each component serves a distinct role in this stack. For data, we develop a custom fingertip UMI device with a motion-capture cage to collect over 10,000 hours of egocentric, sub-millimeterprecision human demonstrations that can also directly serve as post-training trajectories. For modeling, we extend the Hy-Embodied-0.5 backbone with a flow-matching action expert, a compact memory encoder, and a delta-chunk action representation that decouples policy learning from embodiment-specific kinematics. For continued pre-training and fine-tuning, starting from the checkpoint continued-pre-trained on the UMI corpus, we introduce two real-robot SFT tracks: Track-A for target-robot adaptation and Track-B for UMI-only cross-embodiment transfer. For RL post-training, we introduce a Proximalized Preference Optimization (PRO)-based offline RL algorithm that turns failure cases into rapid policy improvement and drives performance toward near-ceiling success rates without requiring a learned reward model. For deployment, an asynchronous inference pipeline with lightweight trajectory smoothing enables high-frequency closed-loop control. Taken as a whole, the Hy-Embodied-0.5-VLA stack marks a meaningful step toward deployable generalist robots.

Website: tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla Github: github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA Model: huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI Dataset: huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data

## 1 Introduction

Recent advances in Vision-Language-Action (VLA) architectures have demonstrated promising capabilities in continuous robotic control [1–6]. Yet turning these model advances into deployable generalist robots requires more than stronger policies: the data, training, adaptation, and execution layers must be co-designed around real-hardware constraints.

These system-level requirements expose three coupled challenges on the data side. First, traditional teleoperation [7, 8] relies on master–slave interfaces that force operators to unnaturally adapt to the robot’s workspace, lacks direct haptic feedback, and therefore precludes delicate manipulation. Second, while leveraging human data [9, 10] or hand-held frameworks such as UMI [11] alleviates data scarcity, these alternatives introduce new limitations: raw human demonstrations greatly enrich behavioral diversity but provide overly coarse action labels, and existing UMI rigs improve localization through SLAM at the cost of cumbersome handheld devices that fail to capture fingertip-level force transmission. Third, bridging the cross-embodiment gap involves more than adapting kinematics: it requires addressing the embodiment gap between human and robot motion spaces, the control gap induced by different dynamics and actuation, and the perception gap between human egocentric views and robot-mounted camera observations.

Beyond data, the architectural design, training paradigms, and deployment stack of VLA models present equally critical bottlenecks. Early approaches largely relied on autoregressive modeling over discretized action tokens [12, 13], which inherently limits both execution speed and control precision. Recent frameworks mitigate this by coupling a Vision-Language Model with a flow-matching action expert that predicts continuous actions [1], yet their foundational visual backbones are not explicitly engineered for robotic control: a significant gap remains between generalist visual representations and the dense spatiotemporal reasoning required for physical interaction. On top of representational issues, standard imitation learning struggles to reach last-mile dexterity, while existing reinforcement learning recipes for continuous control typically depend on brittle reward models or value networks [14]. Finally, even a well-trained policy is of limited use unless it can be served at high frequency in a closed visual loop on real hardware—a deployment constraint that is rarely treated as a first-class design target. Addressing these combined bottlenecks therefore requires a unified pipeline that jointly tackles data, model, policy refinement, and deployment.

![](images/d76a75dc89730fbc7c7707bbc3db74c2866be050b8b7fe0ad675c08c86b85ba9.jpg)  
Fig. 1: Overview of Hy-Embodied-0.5-VLA. An end-to-end VLA system that pairs the Hy-Embodied-0.5-MoT backbone with a flow-matching action expert under a delta-chunk action representation, pre-trained on a 10K-hour egocentric UMI corpus and refined with a reward-free, Proximalized Preference Optimization (PRO)-based offline RL stage (FlowPRO). A single pre-trained checkpoint specializes along two parallel post-training tracks for cross-embodiment transfer to morphologically unseen robots.

To address these challenges, we present Hy-Embodied-0.5-VLA (Fig. 1), an end-to-end system that spans the full stack—from custom data-collection hardware to production-ready deployment. Rather than treating VLA modeling as an isolated problem, HyVLA-0.5 is organized as a complete pipeline in which data, modeling, RL post-training, and deployment each serve a distinct role.

For data, we build a custom fingertip UMI device paired with a motion-capture cage, and use it to collect over 10K hours of egocentric, sub-millimeter-precision human demonstrations. The fingertip form factor restores natural haptic perception that bulky handheld rigs cannot offer; the motion-capture cage produces high-fidelity action labels beyond the reach of SLAM-only pipelines; and the egocentric viewpoint supplies global semantic context rather than over-relying on local wrist cameras. Crucially, the same trajectories can also directly serve as post-training data, making them reusable for downstream adaptation and reducing the need for separate target-robot data collection (Sec. 3.1).

For modeling, we extend our Hy-Embodied-0.5 [15] backbone—a 4B Mixture-of-Transformers VLM pre-trained on embodied corpora—with a flow-matching action expert for continuous, high-frequency action prediction. Compared with adapting general-purpose VLMs [16–18], this embodied-native initialization yields stronger spatial priors and faster post-training convergence. We further introduce a compact memory encoder for spatiotemporal context, and adopt a delta-chunk action representation that predicts incremental end-effector motion between consecutive steps. The delta-chunk formulation decouples policy learning from embodiment-specific kinematics and substantially shrinks the optimization search space, providing a clean substrate for cross-embodiment post-training and deployment (Secs. 2 and 5.1).

For continued pre-training and fine-tuning, we first pre-train HyVLA-0.5 on the 10K-hour UMI corpus, then specialize the resulting checkpoint through task-specific supervised fine-tuning. Realrobot SFT is organized into two tracks: Track-A studies intra-embodiment adaptation with target-robot demonstrations and deployment on the same platform, while Track-B studies UMI-only crossembodiment transfer to morphologically different robots without target-robot teleoperation (Sec. 3.3).

For RL post-training, we introduce FlowPRO [19], a critic-free, reward-free Proximalized Preference Optimization (PRO)-based offline reinforcement learning algorithm. Through a teleoperated intervention-and-rollback pipeline, paired success/failure trajectories are harvested directly from policy rollouts. An RPRO loss then aligns these preferences with the continuous flow-matching objective, while a contrastive gradient-cancellation property suppresses catastrophic forgetting. FlowPRO turns failure cases into a rapid iteration loop for improving long-tail manipulation robustness and driving performance toward near-ceiling success rates, without training any reward or value network (Sec. 4).

For deployment, we implement an asynchronous inference framework that overlaps backbone forward passes with action execution, and stitches successive delta chunks via a simple yet effective cubic Bézier action smoother that guarantees C1-continuous transitions (Sec. 5). Together, these components enable high-frequency, closed-loop control on real hardware and complete the path from data collection to real-world operation on the factory floor. The rest of this report details how the full HyVLA-0.5 pipeline is built, trained, and validated across large-volume pre-training, cross-embodiment post-training, PRO-based refinement, and physical robot deployment.

## 2 Model Architecture

HyVLA-0.5 follows the vision-language-action (VLA) paradigm, in which a pre-trained visionlanguage model (VLM) supplies broad semantic perception and a dedicated action module translates the resulting multi-modal context into low-level robot control (Fig. 2). On top of this paradigm, HyVLA-0.5 comprises three components. Firstly, the backbone is the embodied VLM Hy-Embodied-0.5 [15], which adopts a Mixture-of-Transformers (MoT) architecture [20] with modality-adaptive computation and native-resolution image encoding. Secondly, an action expert generates continuous action chunks through conditional flow matching [1, 21], with the robotics-specific state and action streams kept separate from the VLM and coupled to it through shared attention. Finally, the image encoder is extended into a compact memory encoder that aggregates a multi-frame observation history through interleaved temporal-spatial attention [22]. We first formalize the problem in Sec. 2.1, and then detail the backbone (Sec. 2.2), the action expert (Sec. 2.3), and the compact memory encoder (Sec. 2.4).

![](images/b0f45546b0e03ee292c80dae7d6624a19fb9385eb87e00f9125a69d642f8ab7d.jpg)  
Fig. 2: Architectural overview of HyVLA-0.5. The framework adopts a MoT architecture to facilitate cross-modal interactions via a shared joint-attention mechanism. To effectively process K-frame multiview RGB sequences, the image encoder is extended into a compact memory encoder. Specifically, temporal attention blocks are interleaved every four layers to enforce causal masking across the temporal dimension and seamlessly incorporate historical visual context. As depicted on the right, the attention mask demonstrates our block-wise causal attention strategy. Following Hy-Embodied-0.5 [15], we apply local bidirectional attention to model the multi-view observations.

## 2.1 Problem Formulation

We formulate manipulation as a goal-conditioned, chunk-level control problem. At every decision step t, the policy consumes a multi-modal observation ot and predicts a chunk of future actions ${ \bf A } _ { t } ;$ that is, we model the conditional distribution $p ( \mathbf { A } _ { t } \mid \mathbf { o } _ { t } )$ . Formally,

$$
\mathbf { o } _ { t } = \bigl ( \mathbf { I } _ { t } , \ \ell , \ \mathbf { s } _ { t } \bigr ) , \quad \mathcal { T } _ { t } = \bigl \{ \mathbf { I } _ { t - k } ^ { v } \ \bigr \} _ { v = 1 : n } ^ { k = 0 : K - 1 } , \qquad \mathbf { A } _ { t } = \bigl ( \mathbf { a } _ { t } , \ \mathbf { a } _ { t + 1 } , \ \ldots , \ \mathbf { a } _ { t + H - 1 } \bigr ) ,\tag{1}
$$

where $\mathcal { T } _ { t }$ is the visual stream, ℓ the language instruction, $\mathbf { s } _ { t }$ the proprioceptive state, and $A _ { t }$ the predicted action chunk of horizon H. We describe each component below.

Visual Input. The visual stream $\mathcal { T } _ { t }$ is a multi-view, multi-frame RGB observation: at step t it comprises the K most recent frames from each of the n camera viewpoints $( e . g .$ . a head-mounted view together with a wrist-mounted view per arm), i.e. $n \times K$ images in total. The history length K is a configurable hyperparameter; its value at each training stage is specified in Sec. 3, with $K { = } 1$ recovering the single-frame case.

Language Input. A natural-language task instruction $\ell \ ( e . g .$ “hang the mug on the rack”) defines the goal. It is tokenized and jointly encoded with the visual stream by the VLM backbone, enabling the policy to ground its behaviors in the commanded semantics.

Proprioceptive Input. The robot state $\mathbf { s } _ { t }$ encodes the current pose of the controlled end-effector(s) and is projected into the backbone embedding space, providing the embodiment-grounded context that anchors action prediction to the robot’s present configuration.

Action Output. Instead of single-step execution, the policy predicts an entire action chunk [7] of horizon H per inference cycle. This ensures temporally smooth, high-frequency control while significantly reducing the inference latency, as the VLM backbone is evaluated only once to condition the entire $\dot { H } \cdot$ -step generation via flow matching.

End-effector-frame Representation. Both the proprioceptive state $s _ { t }$ and the action $\mathbf { a } _ { t ^ { \prime } }$ are formulated in the end-effector frame (EEF), an embodiment-agnostic representation that decouples the policy from robot-specific joint kinematics. For each controlled arm, a pose is parameterized by a 3-D Cartesian translation (xyz) and a 6-D continuous rotation representation [23], augmented by a 1-D normalized gripper command, $i . e . , \mathbf { s } _ { t } , \mathbf { a } _ { t ^ { \prime } } \in \mathbb { R } ^ { 1 0 }$ per arm. The proprioceptive state $s _ { t }$ is defined in the end-effector frame with respect to the embodiment root, while each future action $\mathbf { a } _ { t ^ { \prime } }$ is a delta-chunk defined in the relative $E E F$ that takes the current state $s _ { t }$ as its reference frame.

Optional Co-Training Tasks. Beyond learning from action-labeled trajectories, the unified VLA architecture integrates auxiliary next-token prediction tasks to preserve its foundational visionlanguage reasoning and spatial grounding capabilities. We denote this auxiliary data mixture as $\mathcal { D } _ { \mathrm { c t } } = \mathcal { D } _ { \mathrm { V Q A } } \cup \mathcal { D } _ { \mathrm { 2 D } } \cup \mathcal { D } _ { \mathrm { 3 D } }$ . Each training instance is formulated as a pair $( \mathbf { c } , y _ { 1 : M } )$ , where c represents the vision-language conditions, and $y _ { 1 : M }$ denotes a sequence of M serialized target tokens. Depending on the specific task, $y _ { 1 : M }$ consists of semantic answer tokens for VQA, normalized 2D spatial coordinates, or 3D geometric parameters formulated within the camera or scene frame. Crucially, this co-training objective directly optimizes the parameters of the shared VLM backbone, ensuring it maintains and enriches the vital semantic and spatial representations.

## 2.2 Hy-Embodied: Modality-Adaptive Computing Backbone

HyVLA-0.5 builds upon the embodied VLM $\mathbf { H y - E m b o d i e d - 0 . 5 - M o T \ [ 1 5 ] } _ { }$ , a compact model with 4 B parameters optimized for edge deployment. It instantiates the standard image-encoder-plus-languagemodel recipe, and we detail three key design choices adapted for manipulation.

Native-resolution visual encoding. The backbone encodes images with Hy-ViT $2 . 0 ,$ a nativeresolution Vision Transformer (ViT) [24, 25] that accepts arbitrary input resolutions and is distilled from a larger internal teacher. Each camera stream can therefore be processed at its native resolution rather than being down-sampled to a fixed size.

Modality-adaptive Computation via MoT. The backbone adopts a Mixture of Transformers (MoT) architecture [20], which is directly initialized with the pre-trained weights of HY-Embodied-0.5 [15]. This design maintains non-shared QKV and FFN parameters for the visual and textual streams. Specifically, during the forward pass, all visual tokens extracted by the ViT are computed using a duplicated, vision-specific parameter set, whereas textual tokens are processed using the original language parameters. Cross-modal interaction is strictly limited to the shared self-attention layers. Consequently, the visual and textual parameters are updated independently. Furthermore, following the original configuration of HY-Embodied-0.5 [15], the backbone applies bidirectional attention strictly among the visual tokens of each individual image, while maintaining standard causal attention for the language tokens.

Co-training Objective. For the auxiliary VQA and spatial grounding instances sampled from $\mathcal { D } _ { \mathrm { c t } } .$ , the VLM backbone employs its native language modeling head to autoregressively decode the serialized target tokens. We optimize this process via a standard next-token prediction objective:

$$
\mathcal { L } _ { \mathrm { n t p } } ( \theta ) = \mathbb { E } _ { ( \mathbf { c } , y ) \sim \mathcal { D } _ { \mathrm { c t } } } \left[ - \sum _ { j = 1 } ^ { M } \log p _ { \theta } \big ( y _ { j } \mid \mathbf { c } , y _ { < j } \big ) \right] ,\tag{2}
$$

where $y _ { j }$ denotes the j-th serialized target token.

## 2.3 Action Expert with Dual-Tower Flow Matching

Rather than discretizing actions into language-like tokens, HyVLA-0.5 equips the backbone with an action expert that models the continuous distribution $p ( \mathbf { A } _ { t } \mid \mathbf { o } _ { t } )$ directly via conditional flow matching [21].

Dual-tower Routing. On top of the MoT backbone, HyVLA-0.5 separates the joint transformer into an understanding-oriented VLM tower and a generation-oriented action-expert tower. The VLM tower processes visual and textual context with the modality-adaptive parameters described above, while the action expert consumes the projected robot state and noisy action tokens [st, $\mathbf { A } _ { t } ^ { \tau } ]$ to produce the continuous action velocity field. The two towers interact through shared self-attention, allowing grounded visual-language context to guide action generation.

Block-wise Causal Attention. We partition the token sequence into three blocks, $\left[ \mathscr { T } _ { t } , \ell \right] , \left[ \mathbf { s } _ { t } \right]$ , and $\big [ \mathbf { a } _ { t , 0 } ^ { \tau } , \ldots , \mathbf { a } _ { t , H - 1 } ^ { \tau } \big ]$ , and apply attention that is bidirectional within each block but strictly causal across blocks. The perception block is prevented from attending to the robotics-specific blocks, minimizing distribution shift from VLM pre-training; the state block is isolated so that its keys and values can be cached; and the noisy-action block attends to the full prefix.

Flow-matching Objective. Let $\mathbf { A } _ { t } ^ { \tau } = \tau \mathbf { A } _ { t } + ( 1 - \tau ) { \boldsymbol { \mathsf { \Phi } } }$ ϵ with $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ denote the noisy action chunk at flow timestep $\tau \in [ 0 , 1 ]$ . The action expert regresses the velocity field $v _ { \theta }$ that transports noise to the target actions, trained with

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f m } } ( \theta ) \ = \ \mathbb { E } _ { p ( \mathbf { A } _ { t } | \mathbf { o } _ { t } ) , q ( \mathbf { A } _ { t } ^ { \tau } | \mathbf { A } _ { t } ) } \big \| v _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } ) - ( \epsilon - \mathbf { A } _ { t } ) \big \| _ { 2 } ^ { 2 } , } \end{array}\tag{3}
$$

where ${ \bf A } _ { t }$ is the ground-truth chunk, $\mathbf { A } _ { t } ^ { \tau }$ its noised version, $\mathbf { A } _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } )$ the predicted velocity conditioned on the observation $o _ { t }$ , and $\epsilon - A _ { t }$ the target denoising direction. The flow timestep $\tau$ is sampled from a Beta distribution skewed toward high-noise regimes, which emphasizes the harder, more informative stages of action denoising. When auxiliary data is mixed with robot demonstrations, the total objective is $\mathbf { \bar { \mathcal { L } } } ( \theta ) = \mathcal { L } _ { \mathrm { f m } } ( \theta ) + \lambda _ { \mathrm { n t p } } ^ { - } \mathcal { L } _ { \mathrm { n t p } } ( \theta )$ , with $\dot { \lambda } _ { \mathrm { { n t p } } } { = } 0$ recovering action-only training.

Inference. At deployment, the policy generates an action chunk by integrating the learned velocity field from $\scriptstyle \tau = 0 \ t _ { 0 } \ \tau = 1$ via the forward Euler update $\mathbf { A } _ { t } ^ { \tau + \delta } = \mathbf { A } _ { t } ^ { \tau } + \delta \mathbf { v } _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } )$ over 10 integration steps $( \delta \mathrm { = } 0 . 1 )$ . Because the conditioning observation prefix ot remains constant across all solver iterations, its keys and values are cached during the initial forward pass. Consequently, subsequent steps exclusively recompute the action tokens, significantly reducing computational overhead.

## 2.4 Compact Memory Encoder with Temporal-Spatial Attention

HyVLA-0.5 conditions on the K-frame multi-view history $\mathcal { T } _ { t }$ of Eq. (1) to form a compact memory encoding. Encoding all $n \times K$ frames independently and forwarding them to the backbone would multiply the visual token count passed to the VLM. We instead extend the image encoder into a video encoder that compresses the temporal dimension before tokens reach the VLM backbone.

Factorized Temporal-spatial Attention. Following Pi-MEM [22], the video encoder preserves the patchify-then-attend structure of a standard ViT and inserts a temporal pass once every L layers. At such a layer, we add a fixed sinusoidal temporal encoding $e ( k )$ (with $e ( 0 ) = \mathbf { 0 } )$ and reuse the same QKV and output projection $W _ { O }$ of the underlying ViT block, then factorize the attention into two passes that share these projections:

$$
\begin{array} { r } { ( \mathrm { t e m p o r a l } ) \quad \tilde { \mathbf { V } } _ { p } = \mathrm { C a u s a l A t t n } \big ( \mathbf { Q } _ { p } , \mathbf { K } _ { p } , \mathbf { V } _ { p } \big ) , \quad \mathrm { o v e r ~ t h e ~ } K \mathrm { ~ f r a m e s ~ a t ~ e a c h ~ p a t c h ~ } p ; } \end{array}\tag{4}
$$

$$
\mathrm { ( s p a t i a l ) } \quad \tilde { \mathbf { X } } _ { k } = \mathbf { W } _ { O } \mathrm { A t t n } \big ( \mathbf { Q } _ { k } , \mathbf { K } _ { k } , \tilde { \mathbf { V } } _ { k } \big ) , \qquad \mathrm { o v e r ~ t h e ~ } n \mathrm { ~ p a t c h e s ~ w i t h i n ~ e a c h ~ f r a m e ~ } k ,\tag{5}
$$

where $\tilde { X } _ { k }$ is the attention output that the block feeds into its residual connection and MLP. The temporal pass is a causal attention, so each frame attends only to the present and past, matching the streaming nature of on-robot perception. The spatial pass is the original bidirectional self-attention within a frame, applied to the time-mixed values V˜ . This factorization avoids the $\mathcal { O } ( n ^ { 2 } K ^ { 2 } )$ cost of joint space-time attention and reduces the per-layer cost to $\mathcal { O } ( K n ^ { 2 } + n K ^ { 2 } )$

Token-count-preserving compression. In the upper layers of the video encoder we discard the patch representations of past frames and forward only the current-frame tokens to the backbone. Because the interleaved temporal attention has already baked the historical context into the current-frame representation, the number of visual tokens passed to the VLM matches that of a single-frame policy.

Parameter-free, Transfer-friendly Design. The video encoder introduces no new learnable parameters relative to the single-image Hy-ViT 2.0: both passes reuse the QKV and $W _ { O }$ projections of Eq. (4)–(5), and the temporal encoding $e ( k )$ is a fixed sinusoid with $e ( 0 ) = \mathbf { 0 }$ rather than a learned table. Consequently, when K=1 the causal temporal attention is the identity and $e ( 0 ) = \mathbf { 0 }$ leaves the input unchanged, so each augmented block reduces exactly to the pre-trained ViT block. The memory-augmented backbone is therefore initialized directly from the Hy-Embodied-0.5 weights and recovers the single-frame encoder as a special case.

## 3 Pre-training and Supervised Fine-tuning

This section focuses on the supervised stages of HyVLA-0.5 training: large-scale pre-training on the Hy-UMI-10K corpus to learn a generalist action prior, followed by supervised fine-tuning (SFT) on task-specific demonstrations from each target embodiment.

## 3.1 Hy-UMI-10K: High-Fidelity Manipulation Dataset

HyVLA-0.5 is pre-trained on Hy-UMI-10K, a hand-held Universal Manipulation Interface (UMI) dataset [11] of more than 10K hours collected in-house (Fig. 4), and it is the sole data source for pre-training. Unlike standard UMI pipelines that recover gripper poses from on-board visual SLAM, our capture rig tracks each gripper with an external optical motion-capture system, which labels every 6-DoF trajectory at sub-millimetre precision in a single, globally consistent world frame—hence high-fidelity. We describe its capture device, composition, and the pre-training recipe below.

Capture device. Demonstrations are acquired with custom-designed hand-held UMI grippers detached from kinematics of specific embodiments (Fig. 3). The gripper design follows that of a commonly adopted industrial gripper, Changingtek CTAG2F90, to help reduce deployment gap. The gripper-mounted camera is located close to the gripper surface, minimizing collisions of the protruding camera when operating in tight space. Gripper openness is measured by the rotary encoders at gripper joints producing sub-millimetre accuracy, without relying on visual identification of gripper openness. Gripper poses are tracked by an external optical motion-capture system which resolves each 6-DoF trajectory at sub-millimetre precision in a single global Cartesian frame and also synchronises the head RGB-D camera to avoid interference of IR emissions. This optical tracking replaces the on-board visual SLAM used by conventional UMI rigs, obtaining superior accuracy in pose trajectories with minimal operational risks of pose jitters and track losses due to temporary lack of visual features in SLAM-based pose estimation systems. This setup with optical tracking systems prioritizes high quality action labels for tasks involving fine motor skills, at the cost of inconvenient in-the-wild deployment. The grippers are designed with ergonomic finger-attached mechanisms that allow direct actuation with contact and force feedback onto human fingers rather than relying on indirect trigger-based actuation with less obvious force feedback. Some grippers are optionally instrumented with 6-dimensional force torque sensors located at the tips, and their fingers are attached to the operator’s own fingers rather than operated through mechanical triggers, giving a proprioception-aligned mapping between human intent and recorded action, and the tip-located sensors allow more direct measurements of force intent compared wrist-located sensors. Because the recording is anchored to the gripper rather than to any fixed base, the corpus is free of base-placement variance. The rigs capture RGB-D streams, though in the current version of HyVLA-0.5, only the RGB modality is consumed in training, while depth data remain available for future training stages.

![](images/fd728caf8cb8fd00eb34581fc94723a7663fb48f612dc8943ce4f0a1ea75b47e.jpg)  
(a) UMI workstation and grippers

![](images/dde417823f37329b9f579ff3e01900d749af3fff92ea4fb3d22015ff298325d9.jpg)  
(b) Camera views  
Fig. 3: UMI custom data collection workstation. The in-house designed hardware setup features an external optical motion-capture system delivering sub-millimeter high-precision tracking, an ego-centric visual perspective camera with native depth capture, a 6-dimensional force-sensing gripper on each hand.

Composition and distribution. The corpus spans more than 1M episodes and 10K hours of demonstrations across 70 distinct tasks, organised into six scene-based task families—Laundry Room (28.5%), Kitchen (19.2%), Personal Care & Miscellaneous (13.8%), Dexterous / Tool-use (10.4%), Storage & Organization (10.0%), and Cleaning (5.7%). These six families account for the bulk of the corpus, while the remaining tasks form a long tail spanning diverse object categories and environmental conditions. Manipulated objects cover a broad spectrum from rigid containers and tableware to precision instruments and deformable fabrics. A complete characterization of task families, object-category breakdown, and per-task hour distribution is provided in Fig. 4.

## 3.2 Pre-training

Setup. We initialize the VLM from the Hy-Embodied-0.5-MoT [15] checkpoint for pre-training. While the action expert shares the same architectural configuration as the VLM, it is instantiated and randomly initialized as an independent Transformer module. Furthermore, its hidden and intermediate sizes are scaled down from 2048 to 1024 and from 6144 to 2048, respectively, yielding an effective parameter count of 370M. All model parameters are trainable and are optimized under the flowmatching objective (Eq. 3). To accelerate large-scale pre-training, we set K=1, i.e., no historical image frames are used as input, so the video encoder (Sec. 2.4) reduces to the standard single-image encoder. The policy ingests 3 camera views at 224×320 resolution and predicts a future action chunk of horizon H=50 at 10 Hz.

![](images/75d36e2528d5b1007b389d0c8c74c587060cebcf7f9c35b98895e0e78c50af25.jpg)  
Fig. 4: UMI dataset distribution. Detailed characterization of our diverse, in-house collected 10K-hour UMI demonstration corpus. The distribution outlines broad scale, diverse skill categories, environmental conditions, and manipulated objects, ensuring generalist-level manipulation capacity.

Data and Pre-training recipe. We use the full 10K-hour UMI corpus for pre-training. The dataloader samples the dataset with replacement: it first samples an episode from the full corpus with probability proportional to episode length, then uniformly samples one frame from that episode as the current frame, and finally takes the future action sequence with chunk size H=50 at 10 Hz as the ground-truth action chunk. Both state and action inputs are normalized using their dataset-wide mean and standard deviation before being fed into the network. We train for 200K steps with a global batch size of 1,024 and a base learning rate of $5 \times 1 0 ^ { - 5 }$ . The learning rate is linearly warmed up to its maximum value over the first 1K steps, decayed to one tenth of the peak value over the subsequent 160K steps, and kept training for another 40K steps. We use AdamW optimizer [26] and perform training in bfloat16 mixed precision.

## 3.3 Supervised Fine-tuning

Setup. Initializing from the UMI pre-trained VLA checkpoint (Sec. 3.2), we run supervised fine-tuning (SFT) on task-specific demonstrations from each target embodiment under the flow-matching objective (Eq. 3). Both the VLM and action expert weights are loaded from the pre-trained model, and all parameters remain trainable. Unlike pre-training, SFT sets $K { = } 6 ,$ enabling the video encoder of Sec. 2.4 to condition on the current frame together with five historical frames.

Embodiments and Data. We fine-tune our model across one simulated embodiment and four realworld platforms. In simulation, we employ the Aloha-AgileX bimanual setup from the RoboTwin 2.0 benchmark [27], covering 50 manipulation tasks; each task provides 50 clean-environment episodes and 500 randomized-environment episodes, resulting in 2.75K episodes and more than 6M frames in total. For real-world SFT, we organize the data into two deployment tracks that separate intraembodiment adaptation from cross-embodiment transfer. Track A (intra-embodiment) collects demonstrations through tele-operation on the same robot platform used for evaluation; here, the Dobot X-Trainer covers four tasks with 300 demonstrations per task (18 hours in total). Track B (crossembodiment) fine-tunes only on task-specific UMI demonstrations and deploys to morphologically different target robots without target-robot teleoperation; this track covers one task on JAKA K1 (300 UMI demonstrations, 1.2 hours) and one task on Astribot S1 (200 UMI demonstrations, 1.5 hours). Separately, we use Unitree G1 (1 task, 400 UMI demonstrations, 2.2 hours) for the force-modality validation in Sec. 6.2.

Post-training recipe. For real-world deployment, actions are sampled at 50 Hz with an action-chunk horizon of H=50 and a history interval of 1 second. We train for 60K steps with a global batch size of 32 and a base learning rate of $2 . 5 \times 1 0 ^ { - 5 }$ , decayed over 40K steps. For RoboTwin 2.0, due to the larger data scale, we downsample future actions from the current frame with stride 3, use an action-chunk horizon of $H { = } 2 0$ and a history interval of 5×stride. The global batch size is set to 128 and the remaining optimization settings follow the pre-training recipe. More details are described in Appendix A.

![](images/cc10c3541a8a5b7e9a0528ac71acd59fabce958df9175dda22904aaaf7cffa6d.jpg)  
Fig. 5: FlowPRO data pipeline for collecting real-robot preference trajectories and converting them into dense per-state preference tuples. During policy rollouts, an operator triggers an intervention-androllback: the system rewinds to a prior state, logs the executed segment as a negative trajectory, and records a corrective teleoperation segment as the paired positive trajectory. A smooth-interpolation procedure then synthesizes the missing counterpart action on each branch to yield per-state tuples $\bar { ( } s , a ^ { w } , a ^ { l } )$ used for preference optimization.

## 4 Reinforcement Learning Post-Training

After supervised pre-training and SFT (Sec. 3.3), HyVLA-0.5 further improves real-robot deployment through failure-driven post-training. This stage follows the FlowPRO recipe [19], using a flowmatching-aware preference-optimization loss (RPRO) together with a teleoperated intervention-androllback data pipeline. In this way, HyVLA-0.5 converts a small number of real-robot corrections into measurable deployment gains without training any reward or value model.

## 4.1 Design Principles

As discussed in section 7, real-robot post-training generally falls into three families: SFT/DAgger, reward- or value-based RL, and preference-based RL. Their characteristic limitations motivate the three FlowPRO design principles below:

– (P1) Exploit failures directly. Negative trajectories are not discarded or merely flagged for re-labelling; they are fed back into the action-generation loss as per-state, per-chunk contrastive signals against their paired positive corrections.

– (P2) Avoid reward and critic models entirely. The training signal is computed in closed form from a frozen reference policy and the current policy, using a flow-matching log-likelihood proxy. This bypasses the dense-reward-design bottleneck that plagues contact-rich manipulation.

– (P3) Anchor the implicit reward. A symmetric proximal regularizer prevents the absolute magnitude of the implicit reward from exploding. This structurally forbids the plain-DPO reward-hacking failure mode in which the policy drifts away from both $a ^ { w }$ and $a ^ { l }$

The remainder of this section formalises the loss (Sec. 4.2) and the data pipeline that supplies the per-state preference tuples it consumes.

## 4.2 Method

FlowPRO proceeds as an iterative offline-RL loop on top of an SFT-pretrained HyVLA-0.5 base policy (Fig. 5). Each round contains three steps: (1) collect on-robot preference pairs via teleoperated intervention-and-rollback; (2) convert these sparse trajectory-level corrections into dense per-state preference tuples via Smooth Interpolation; (3) optimize the policy with the RPRO loss on a mixed batch of new pairs, historical pairs, and SFT data (Fig. 6). The previous round’s policy serves as the reference policy $\pi _ { \mathrm { r e f } }$ in the next round.

RPRO loss. The HyVLA-0.5 action head is a flow-matching model [21, 28]. Given a state $s = ( o , l )$ with visual observations o and a language instruction $l ,$ a velocity field $v _ { \theta } ( a _ { t } , t \ | \ s )$ transports

![](images/91bf85d6781bd1d7f001154146148ed0a58ecbe498bf36baa8f129749515854f.jpg)  
Fig. 6: RPRO optimization. The learnable policy $\pi _ { \theta }$ and frozen reference $\pi _ { \mathrm { r e f } }$ predict actions $a ^ { \theta }$ and $a ^ { \mathrm { r e f } }$ for the same state. The objective pulls $a ^ { \theta }$ toward the preferred action $\boldsymbol { a } ^ { w } \left( \boldsymbol { r } ^ { w } \uparrow \right)$ and pushes it from the dispreferred $a ^ { l } \left( r ^ { l } \downarrow \right)$ . A proximal regularizer (blue dashed) anchors both reward branches to $\pi _ { \mathrm { r e f } } .$ , preventing reward hacking. Batches mix $\mathcal { D } _ { \mathrm { p r e f } } ^ { k } , \mathcal { D } _ { \mathrm { p r e f } } ^ { < k }$ , and ${ \mathcal { D } } _ { \mathrm { S F T } }$

Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ to an action chunk a. This transport follows the linear interpolant $a _ { t } = ( 1 - t ) \epsilon +$ ta over flow time $t \in [ 0 , 1 ]$ , with conditional velocity $u ( a _ { t } \mid a ) : = a - \epsilon$ . Following Flow-DPO [29], we adopt the per-sample flow-matching regression loss as a tractable surrogate for the negative log-likelihood,

$$
\ell _ { \boldsymbol { \theta } } ( \boldsymbol { s } , \boldsymbol { a } ) = \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] , \boldsymbol { \epsilon } \sim \mathcal { N } ( 0 , I ) } \big [ \| \boldsymbol { v } _ { \boldsymbol { \theta } } ( \boldsymbol { a } _ { t } , t \mid \boldsymbol { s } ) - \boldsymbol { u } ( \boldsymbol { a } _ { t } \mid \boldsymbol { a } ) \| ^ { 2 } \big ] ,\tag{6}
$$

which yields the implicit-reward proxy used by RPRO,

$$
\begin{array} { r } { r _ { \theta } ( s , a ) = \frac { \beta } { 2 } \big ( \ell _ { \mathrm { r e f } } ( s , a ) - \ell _ { \theta } ( s , a ) \big ) , } \end{array}\tag{7}
$$

where $\ell _ { \mathrm { r e f } }$ and $\ell _ { \theta }$ denote the flow-matching losses under the reference and current policies. Substituting Eq. (7) into the PRO pairwise objective [30] gives the flow-matching-adapted PRO loss,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { P R O } } ( \theta ) = - \mathbb { E } _ { ( s , a ^ { w } , a ^ { l } ) \sim \mathcal { D } } \Big [ \underbrace { \log \sigma \big ( r _ { \theta } ( s , a ^ { w } ) - r _ { \theta } ( s , a ^ { l } ) \big ) } _ { \mathcal { L } _ { \mathrm { c o n } } : \mathrm { c o n t r a s t i v e ~ o p t i m i z e r } } } \\ & { \qquad + \underbrace { \sum _ { a \in \{ a ^ { w } , a ^ { l } \} } \frac { 1 } { 2 } \big [ \log \sigma \big ( r _ { \theta } ( s , a ) \big ) + \log \sigma \big ( - r _ { \theta } ( s , a ) \big ) \big ] } _ { \mathcal { L } _ { \mathrm { r e g } } : \mathrm { p r o x i m a l ~ r e g u l a r i z e r } } \Big ] , } \end{array}\tag{8}
$$

where $\mathcal { L } _ { \mathrm { r e g } }$ is minimized at $r _ { \theta } ( s , a ) = 0$ and grows symmetrically with $| r _ { \theta } ( s , a ) |$ , anchoring the absolute magnitude of the implicit reward and thereby preventing the reward-hacking pathology of plain Flow-DPO. To preserve base-policy performance and reinforce direct regression toward $a ^ { w }$ , we combine $\mathcal { L } _ { \mathrm { P R O } }$ with a supervised term:

$$
\begin{array} { r } { \mathcal { L } _ { \mathtt { R P R O } } ( \theta ) = \lambda _ { \mathtt { P R O } } \mathcal { L } _ { \mathtt { P R O } } ( \theta ) + \lambda _ { \mathtt { S F T } } \mathcal { L } _ { \mathtt { S F T } } ( \theta ) , \qquad \mathcal { L } _ { \mathtt { S F T } } ( \theta ) = \mathbb { E } _ { ( s , a ^ { w } ) \sim \mathcal { D } } [ \ell _ { \theta } ( s , a ^ { w } ) ] . } \end{array}\tag{9}
$$

A useful side-property of $\operatorname { E q . } \left( 8 \right)$ is contrastive gradient cancellation: when $a ^ { w } = a ^ { l } , \nabla _ { \theta } \mathcal { L } _ { \mathrm { c o n } } = \mathbf { 0 }$ leaving only $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { r e g } }$ and $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F I } }$ active. This makes it safe to route SFT-style demonstrations through the same RPRO loss, which we exploit in batch composition below.

Data collection: intervention-and-rollback. We collect preference trajectory pairs $( \tau ^ { w } , \tau ^ { l } )$ via a teleoperated intervention-and-rollback pipeline (Fig. 5). During rollouts of the current policy, the operator intervenes whenever an erroneous or dangerous action is observed. The system then (1) rewinds to an earlier state $s _ { t - \Delta }$ with operator-chosen horizon ∆ and records the executed segment as the negative trajectory $\tau ^ { l } ; ( 2 )$ retrieves the observation at t − ∆ as a visual reference in case the environment has changed and the physical scene needs to be reset; and (3) records the operator’s corrective demonstration from $s _ { t - \Delta }$ as the positive trajectory $\tau ^ { w }$ . A single operator action thus yields a naturally paired $( \tau ^ { w } , \tau ^ { l } )$ sharing the same initial state. Varying ∆ across interventions diversifies the per-pair starting state without recording separate positive and negative rollouts.

Smooth Interpolation and batch mixing. Because $\tau ^ { w }$ and $\tau ^ { l }$ diverge after $s _ { t - \Delta }$ , each subsequent state belongs to only one trajectory. To produce dense per-state tuples $( s , a ^ { w } , a ^ { l } )$ required by Eq. (8), we synthesize the missing counterpart with a Smooth Interpolation procedure (Fig. 5). For a state M on $\dot { \tau ^ { l } }$ , we locate its closest point $M ^ { \prime }$ on $\tau ^ { w }$ under a weighted distance metric. We then construct a synthetic positive action chunk that bridges from M to a transition point J on $\tau ^ { w }$ via a cubic Bézier for positions, Slerp for orientations, and linear interpolation for the gripper. The chunk then follows $\tau ^ { w }$ until it ends at $N ^ { \prime }$ , while the negative action is simply the next H steps along $\tau ^ { l }$ . For states already on $\tau ^ { w }$ or in ${ \mathcal { D } } _ { \mathrm { S F T } }$ , we set $a ^ { w } = a ^ { l }$ . The contrastive gradient cancellation above makes these samples act as regularized SFT samples. Across iterations, we keep the round-k pairs $\mathcal { D } _ { \mathrm { p r e f } } ^ { k } ,$ the historical pool $\begin{array} { r } { \mathcal { D } _ { \mathrm { p r e f } } ^ { < k } = \bigcup _ { j < k } \mathcal { D } _ { \mathrm { p r e f } } ^ { j } , } \end{array}$ and ${ \mathcal { D } } _ { \mathrm { S F T } }$ in separate buffers. We mix mini-batches at fixed proportions: 80%/20% for $k { = } 1 ( \mathcal { D } _ { \mathrm { p r e f } } ^ { k } / \mathcal { D } _ { \mathrm { S F T } } )$ and $7 0 \% / 1 5 \% / 1 5 \%$ for $k { \geq } 2 ( \mathcal { D } _ { \mathrm { p r e f } } ^ { k } / \mathcal { D } _ { \mathrm { p r e f } } ^ { < k } / \mathcal { D } _ { \mathrm { S F T } } )$ . This schedule up-weights the newest, most informative failure states, replays previously corrected ones to prevent regression, and retains a non-trivial SFT share to anchor base capabilities.

Experimental validation of FlowPRO on four real-robot bimanual tasks (Bottle, Cap, USB, Zip) is reported in 6.3 together with the rest of the empirical evaluation.

## 5 Deployment

Deployment mainly addresses three runtime issues: mapping end-effector delta chunks to heterogeneous robot platforms, serving VLA predictions at the robot control rate, and stitching independently predicted chunks into smooth motion. We handle them with three lightweight components: a platform mapper that keeps the learned action interface unchanged across embodiments (Sec. 5.1); an asynchronous inference–execution loop that overlaps backbone forward passes with servo execution (Sec. 5.2); and a latency-aware cubic-Bézier stitcher that removes stale prefixes and enforces smooth chunk transitions (Sec. 5.3). The same deployment stack is used across all real-robot evaluations.

## 5.1 Embodiment-Agnostic Platform Mapping

The role of platform mapping is to preserve the robot-agnostic contract established by the delta-chunk representation. The policy outputs a 20-dimensional dual-arm action chunk (10 dimensions per end-effector: a 3-D Cartesian translation and a 6-D rotation — the first two rows of an $S O ( 3 )$ rotation matrix — both expressed relative to the end-effector pose at the start of the chunk, together with a 1-D gripper opening command). Embodiment-specific kinematics are deferred to deployment, where the relative $S E ( 3 )$ prediction is composed with the initial end-effector pose to recover absolute world-frame targets and inverse kinematics (IK) is then solved on the target robot to produce joint commands.

For intra-embodiment deployment (Track A), the world frame remains the same in deployment and data collection. For cross-embodiment deployment (Track B), data collection and deployment use different embodiments, so we instantiate mappings for two embodiment types: fixed-base arms and the floating-base humanoid. In the equations below, $^ A T _ { B }$ denotes the pose of frame B in frame A; $W , G _ { t } , G _ { t + k } ^ { - }$ , and C are the world, current gripper, predicted future gripper k steps further, and the chassis frame.

For fixed-base arms such as JAKA K1, the rel-EE chunk is cast into the world frame using the current gripper pose $W _ { T _ { G _ { t } } }$ from forward kinematics,

$$
^ W T _ { G _ { t + k } } \ = \ ^ { W } T _ { G _ { t } } \cdot ^ { G _ { t } } T _ { G _ { t + k } } ,\tag{10}
$$

For the humanoid like Astribot S1, a deterministic heuristic infers a fixed chassis frame $w _ { T _ { C } }$ and a floating torso frame from the predicted gripper targets (Appendix B.1), yielding

$$
^ C T _ { G _ { t + k } } \ = \ \left( ^ { W } T _ { C } \right) ^ { - 1 } . ^ { W } T _ { G _ { t } } . ^ { G _ { t } } T _ { G _ { t + k } } .\tag{11}
$$

where $w _ { T _ { C } }$ is a constant transform cached after calculation. The additional 24 head/torso dimensions (12 each: 3 position + 9 rotation) are set by this heuristic rather than predicted by the policy. This keeps the learned action interface unchanged across Track A intra-embodiment deployment and Track B cross-embodiment deployment.

![](images/ad8d87fdeb933389919e082ce93ce482c7969c4f2f45d1b864c691df8dc054fb.jpg)  
Fig. 7: Asynchronous execution timeline. Policy inference, Bézier smoothing, buffer overwrite, and servo-rate action execution are overlapped; executed actions are recorded in H to estimate tangents for the next chunk stitch.

## 5.2 Asynchronous Execution for Real-Time Control

A high-capacity VLA policy runs slower than the robot servo loop, so synchronous execution would leave the robot idle between forward passes. We therefore decouple inference from command dispatch using a producer–consumer runtime with a thread-safe action buffer B (Fig. 7). The inference thread queries the policy from the latest observation and overwrites B with a smoothed action sequence, while the execution thread pops commands from B at the control frequency and records recent poses for tangent estimation. Overlapping these two loops hides much of the backbone latency behind continuous execution.

## 5.3 Latency-Aware Bézier Chunk Stitching

Chunk stitching is critical in asynchronous execution: delayed chunks must be reconnected to the robot’s current state without introducing motion discontinuities. We use a cubic Bézier segment to form a compact $C ^ { 1 }$ -continuous connector with controllable endpoint positions and tangents.

The first design choice is to select the connection point between the Bézier connector and the retained chunk. We set γ as a lightweight deployment hyperparameter, chosen according to hardware response such as acceleration limits and servo rate. A smaller $\gamma$ selects an earlier point in the retained chunk and preserves more policy-predicted actions, but leaves less room to correct the delayed boundary. A larger γ provides a smoother landing target but skips more predicted actions. We clip the resulting index away from both ends so that the future tangent can be estimated from neighboring waypoints.

The control points are chosen with the same intuition. The Bézier curve should leave the robot’s current trajectory in the direction it was already moving, and enter the future chunk in the direction that the policy predicts next. We therefore place the two inner control points along the historical motion direction and the local direction of the future chunk, with their distance scaled by the gap between the current robot state and the reconnection point. This gives a smooth transition without introducing an additional learned controller.

Based on this design, the runtime procedure is as follows. Given an original chunk of length $N$ , we first discard the stale prefix

$$
K = \lceil N / \alpha \rceil , \qquad K \leq N - 3 ,\tag{12}
$$

where $\alpha > 1$ is the truncation ratio, and retain $\mathcal { F } = \{ \mathbf { f } _ { 0 } , \dots , \mathbf { f } _ { M - 1 } \}$ with $M = N - K$

Let $\mathbf { h } _ { 0 }$ be the last executed EE position. We choose an interior connection point $\mathbf { f } _ { c }$ with $c =$ $\mathrm { c l i p } ( \lfloor \gamma M \rfloor , 1 , M - 2 )$ , where the clipping keeps the two-sided future tangent well-defined, and construct a cubic Bézier segment $\mathbf { B } ( t ) $ satisfying

$$
\mathbf { B } ( 0 ) = \mathbf { h } _ { 0 } , \quad \mathbf { B } ( 1 ) = \mathbf { f } _ { c } , \quad \dot { \mathbf { B } } ( 0 ) \parallel \hat { \mathbf { d } } _ { \mathrm { h i s t } } , \quad \dot { \mathbf { B } } ( 1 ) \parallel \hat { \mathbf { d } } _ { \mathrm { f u t } } ,\tag{13}
$$

![](images/cef625fca04e7cb4110a1c540bbd1bf5eebe60e564093fc564acc7ef2a086ab5.jpg)  
Fig. 8: Trajectory comparison between raw action chunks (orange) and the asynchronous Béziersmoothed trajectory (blue). Smoothing reduces visible discontinuities at chunk boundaries for both arms across x, y, and z dimensions.

with tangents

$$
\hat { \mathbf { d } } _ { \mathrm { h i s t } } = \frac { \mathbf { h } _ { 0 } - \mathbf { h } _ { - 1 } } { \left\| \mathbf { h } _ { 0 } - \mathbf { h } _ { - 1 } \right\| } , \qquad \hat { \mathbf { d } } _ { \mathrm { f u t } } = \frac { \mathbf { f } _ { c + 1 } - \mathbf { f } _ { c - 1 } } { \left\| \mathbf { f } _ { c + 1 } - \mathbf { f } _ { c - 1 } \right\| } ,\tag{14}
$$

when the corresponding norm is non-zero. The endpoint control points anchor the transition at the current state and the reconnection point, while the two inner control points encode the historical and future tangents:

$$
\mathbf { P } _ { 0 } = \mathbf { h } _ { 0 } ,
$$

$$
\mathbf { P } _ { 1 } = \mathbf { P } _ { 0 } + \lambda \hat { \mathbf { d } } _ { \mathrm { h i s t } } ,\tag{15}
$$

$$
\mathbf { P } _ { 2 } = \mathbf { P } _ { 3 } - \lambda \hat { \mathbf { d } } _ { \mathrm { f u t } } ,
$$

$$
\mathbf { P } _ { 3 } = \mathbf { f } _ { c } ,\tag{16}
$$

where $\lambda = \sigma \| \mathbf { P } _ { 3 } - \mathbf { P } _ { 0 } \|$ controls the tangent length. The transition curve is

$$
\mathbf { B } ( t ) = ( 1 - t ) ^ { 3 } \mathbf { P } _ { 0 } + 3 ( 1 - t ) ^ { 2 } t \mathbf { P } _ { 1 } + 3 ( 1 - t ) t ^ { 2 } \mathbf { P } _ { 2 } + t ^ { 3 } \mathbf { P } _ { 3 } ,\tag{17}
$$

and is uniformly sampled to replace the discontinuous boundary segment. Position is smoothed in $\mathbb { R } ^ { 3 }$ , orientation uses SLERP, gripper commands are linearly interpolated, and each arm is processed independently. The resulting transition is $C ^ { 1 } .$ -continuous, policy-agnostic, and controlled by the embodiment-dependent parameters $\alpha , \gamma ,$ and σ.

Fig. 8 compares raw chunked actions with the asynchronously Bézier-smoothed trajectory.

## 6 Evaluation

The empirical validation addresses two parallel questions: how well HyVLA-0.5 performs after standard downstream supervised fine-tuning in simulation and on real hardware (Secs. 6.1 and 6.2), and how much FlowPRO post-training further improves a deployed policy (Sec. 6.3).

For real hardware, we organize the SFT evaluation into two deployment tracks: Track A fine-tunes and evaluates on the same tele-operated robot platform, while Track B fine-tunes only on UMI demonstrations and deploys on morphologically different robots without target-robot teleoperation. We evaluate four Track-A tasks and two Track-B tasks. Foundational baselines $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ are identically parameterized and trained with matched data and iteration budgets.

## 6.1 Simulated Tasks

On RoboTwin 2.0 [27], we report task success rates averaged over 100 stochastic rollouts per task and then over the full 50-task suite. Results are evaluated under both Clean and Randomized settings, with aggregate comparisons and ablations shown in Table 1; the complete per-task breakdown is deferred to Appendix A (Table 3).

<table><tr><td>Method</td><td>Clean</td><td>Randomized</td></tr><tr><td>Other methods</td><td></td><td></td></tr><tr><td>π0[1]</td><td>65.9</td><td>58.4</td></tr><tr><td>ABot-M0 [31]</td><td>81.2</td><td>80.4</td></tr><tr><td>T0.5 [2]</td><td>82.7</td><td>76.8</td></tr><tr><td>Qwen-VLA [32]</td><td>86.1</td><td>87.2</td></tr><tr><td>LingBot-VLA [33]</td><td>86.5</td><td>85.3</td></tr><tr><td>starVLA [34]</td><td>88.2</td><td>88.3</td></tr><tr><td>Motus [35]</td><td>88.7</td><td>87.0</td></tr><tr><td>JoyAI-RA [36]</td><td>90.5</td><td>89.3</td></tr><tr><td>Ablations</td><td></td><td></td></tr><tr><td>HyVLA-0.5</td><td>90.9</td><td>90.1</td></tr><tr><td>w/o compact memory encoder</td><td>88.8</td><td>88.6</td></tr><tr><td>w/o compact memory encoder and UMI pre-training</td><td>88.1</td><td>87.9</td></tr></table>

Table 1: Evaluation results on the RoboTwin 2.0 benchmark [27]. Success rate (%) under the Clean and Randomized settings, averaged over 100 runs per task and then over the 50-task suite. The upper block lists competing methods; the lower block reports removal-based ablations from the full HyVLA-0.5 model. Per column, the best result is in bold.

Baselines and main results. We benchmark against eight contemporary VLA systems—π0 [1], π0.5 [2], ABot-M0 [31], LingBot-VLA [33], starVLA [34], Motus [35], JoyAI-RA [36], and Qwen-VLA [32]—using each method’s officially reported success rates under the same Clean and Randomized protocol. As shown in the upper block of Table 1, HyVLA-0.5 attains the best success rate in both settings, reaching 90.9% on Clean and 90.1% on Randomized. It outperforms π0 by 25.0 points on Clean (vs. 65.9%) and by 31.7 points on Randomized (vs. 58.4%), and remains clearly ahead of π0.5 by 8.2 and 13.3 points (vs. 82.7% and 76.8%). Even against the strongest competing method, JoyAI-RA, HyVLA-0.5 still leads by 0.4 and 0.8 points (vs. 90.5% and 89.3%).

Ablation. The lower block of Table 1 conducts a removal-based ablation starting from the full HyVLA-0.5 model. Removing the compact memory encoder reduces performance from 90.9% / 90.1% to 88.8% / 88.6% on Clean / Randomized; further removing the large-scale UMI pre-training stage lowers the scores to 88.1% / 87.9%. Together, these ablations show that both UMI pre-training and short-horizon visual memory contribute consistent gains. Although the egocentric real-world UMI corpus is visually distant from the synthetic RoboTwin 2.0 renderings, UMI pre-training still provides a modest gain in simulation. The limited magnitude is expected, given the large gaps in task distribution, action trajectories, and visual appearance. In contrast, Sec. 6.2 shows its significant benefit on real-robot tasks, where the domain gap to UMI demonstrations is smaller.

## 6.2 Real-World Tasks

We evaluate HyVLA-0.5 on real-robot bimanual manipulation through the two deployment tracks introduced above, spanning three platforms and six benchmark tasks, plus a qualitative forcediscrimination task on a Unitree G1. All per-task snapshots and success rates are reported in Figure 9. Track A tests intra-embodiment fine-tuning, while Track B probes whether UMI-only post-training can transfer task semantics across embodiments.

![](images/26da6a6093ba18b8746723e88ec358818144a95cd2a87deaa217f333ed33222d.jpg)  
Fig. 9: Real-robot evaluation on six bimanual manipulation tasks. Left panel: Snapshots of representative task executions captured during rollout. Right panel: Per-task success rates (%) after supervised fine-tuning on tele-operated or UMI demonstrations.

Track A — Intra-Embodiment Fine-Tuning (Dobot X-Trainer). Data are collected via teleoperation on a Dobot X-Trainer and the same platform is used for evaluation. We benchmark four bimanual tasks: Insert Bottles, where the robot grasps two cylindrical bottles and inserts each into a dedicated holder under tight geometric tolerances; Fold and Store Glasses, where it picks up a pair of eyeglasses, folds the temples inward through coordinated bimanual motion, and places them into a protective case; Set the Table, where it arranges a plate, a fork, and a knife at canonical positions on a dining surface, requiring long-horizon spatial planning and precise 6-DoF placement; and Zip Up the Pen Case, where it opens the zipper, inserts a pen, and closes the zipper along the full track under deformable-object dynamics.

Effect of UMI Pre-training on Track-A Tasks. The per-task results in Figure 9 reveal a consistent pattern on the precision-critical tasks. For Fold and Store Glasses and Zip Up the Pen Case, success hinges on a few decisive sub-steps rather than on the trajectory as a whole, e.g. folding the temples without slipping, or pinching the zipper slider before pulling. Without Hy-UMI-10K pre-training, the policy is visibly less accurate at exactly these moments, where sub-centimetre positioning and stable bimanual force coupling are required. The resulting local errors then propagate downstream and dominate the failure modes. Pre-training reverses this pattern: predictions sharpen at the same critical moments, end-to-end success rates rise accordingly, and coarser segments of the trajectory remain essentially unchanged. This task-level evidence corroborates the simulation ablation in Section 6.1. It suggests that the principal value of large-scale, high-precision UMI pre-training is to sharpen the action distribution at the precision-critical bottlenecks of downstream manipulation, and that this benefit transfers from human demonstrations to real-robot post-training.

Track B — Cross-Embodiment Transfer (JAKA K1, Astribot S1). For each target robot, we post-train the UMI pre-trained checkpoint on task-specific UMI demonstrations only—without any target-robot teleoperation—and deploy the resulting policy on the corresponding robot. We benchmark two tasks: Put Away the Accessory on JAKA K1, where the robot picks up a sub-centimetre hair tie and places it into the centre cell of a compartment box whose cell size nearly matches the tie’s diameter; and Clean Up the Table on Astribot S1, where the humanoid locates scattered paper cups on a tabletop and deposits them sequentially into a waste bin.

Effect of UMI Pre-training on Track-B Tasks. Track B isolates the contribution of UMI pre-training to cross-embodiment deployment: since no target-robot data is ever seen during fine-tuning, any gain over an identically configured baseline must come from the prior learned during pre-training. Figure 9 shows that this gain is substantial on both robots: HyVLA-0.5 achieves markedly higher success rates than π0 and π0.5 on Put Away the Accessory and Clean Up the Table, despite all three policies being post-trained on the same UMI data. The improvement indicates that large-scale, high-fidelity UMI pre-training equips the model with embodiment-agnostic action priors that survive a deployment shift to morphologically unseen robots, and that these priors make the small UMI fine-tuning set sufficient on its own to recover deployable performance on a new platform.

Force-Modality Validation (Unitree G1). Because our handheld UMI gripper records tip force signals during demonstration collection, the resulting data directly contains the physical cues needed for force-aware, and potentially force-controlled, manipulation. We show this capability on a Unitree G1 equipped with our end effector, where the policy performs a force-discrimination task: it sequentially grasps two boxes and places the lighter one into a front basket. For this task, we augment the action expert with two lightweight TCN encoders [37] and an MLP projector, which together encode a 50-step F/T window for each hand (∼2M added parameters). The augmented policy is then post-trained on a small set of UMI demonstrations recorded with the workstation’s tip force/torque signals (Sec. 3.1). Since the lighter-object position is randomized across trials, spatial memory alone cannot solve the task; the policy must compare the grasp-phase force profiles before deciding which box to place. HyVLA-0.5 reliably selects the lighter box across trials (Fig. 10), showing that the tactile signals captured by the UMI workstation provide actionable non-visual cues for downstream policy learning.

![](images/a32786e7f5005e3379282bba9b86394e53b46abbd73a93be49af3a4898384f83.jpg)  
Fig. 10: Force-guided object discrimination on a Unitree G1. The robot sequentially grasps two boxes of differing mass and places the lighter one into the front basket, confirming that the in-house UMI workstation captures actionable tactile information.

## 6.3 Real-World Reinforcement

![](images/0d72db41e285cdbcb824e5bb472ab9637f62475899e7125be05cc0746481edda.jpg)  
Fig. 11: Additional fine-grained real-robot tasks for FlowPRO post-training. Beyond Insert Bottles (Bottle, sub-cm insertion) and Zip Up the Pen Case, which are shown in Fig. 9, we further evaluate FlowPRO on two fine-grained tasks: USB insertion (USB, sub-mm precision) and Pen-Cap Assembly (Cap, in-air bimanual coordination). This figure illustrates these two additional tasks.

Setup. All FlowPRO experiments are conducted on a Dobot X-Trainer bimanual platform. We evaluate on four long-horizon bimanual tasks (Fig. 11): Bottle, Cap, USB, and Zip. Starting from the same HyVLA-0.5 SFT checkpoint $\pi _ { \mathrm { r e f } } .$ , every method runs K=3 rounds of iterative post-training under an identical data-collection budget. Each entry in Table 2 is averaged over 3 training seeds; per-seed success rate (SR) is computed from n=100 rollouts with randomized initial placements, and completion time (CT) is averaged over the same rollouts.

Baselines. We compare RPRO against two representative comparators that cover both regimes of the design space: DAgger [38] (positive-only dataset aggregation) and $\pi _ { 0 . 6 } { } ^ { * }$ [14] (advantage-conditioned regression that uses the same positive-and-negative pairs as RPRO but injects the preference signal as a conditioning token rather than through a contrastive loss). All methods share the same HyVLA-0.5 SFT backbone and the same iterative data-collection protocol.

<table><tr><td rowspan="2">Fine-tune</td><td colspan="2">BOTTLE</td><td colspan="2"> $\mathbf { C A P }$ </td><td colspan="2">USB</td><td colspan="2"> $\mathrm { Z m }$ </td></tr><tr><td>SR</td><td>CT</td><td>SR</td><td>CT</td><td>SR</td><td>CT</td><td>SR</td><td>CT</td></tr><tr><td>DAgger</td><td> $9 3 \pm 2 . 1 \%$ </td><td>27s</td><td> $8 8 \pm 1 . 8 \%$ </td><td> $2 9 \mathrm { s }$ </td><td> $8 6 \pm 2 . 4 \%$ </td><td>25s</td><td> $8 3 \pm 2 . 0 \%$ </td><td>55s</td></tr><tr><td> $\pi _ { 0 . 6 } { } ^ { * }$ </td><td> $9 5 \pm 1 . 5 \%$ </td><td>24s</td><td> $9 5 \pm 1 . 2 \%$ </td><td>27s</td><td> $9 5 \pm 1 . 4 \%$ </td><td>23s</td><td> $8 9 \pm 1 . 6 \%$ </td><td>45s</td></tr><tr><td>RPRO</td><td> $\mathbf { 9 9 \pm 0 . 6 \% }$ </td><td>16s</td><td> $\mathbf { 9 9 2 0 . 7 \% }$ </td><td>21s</td><td> $\mathbf { 9 8 \pm 0 . 9 \% }$ </td><td>22s</td><td> $\mathbf { 9 4 \pm 1 . 1 \% }$ </td><td>37 s</td></tr></table>

Table 2: Final success rate and completion time after $K { = } 3$ rounds of post-training on four real-robot bimanual tasks, with HyVLA-0.5 as the base policy. SR (↑, %) is reported as mean ± std (in points) across 3 training seeds, with each per-seed SR computed over $n { = } 1 0 0$ randomized rollouts; $\operatorname { C T } \left( { \downarrow , \mathrm { { s } } } \right)$ is the cross-rollout mean. Best per column in bold.

Results. Table 2 and Fig. 12 summarize the comparison. RPRO vs. DAgger. DAgger relies on positive samples only, while RPRO additionally exploits negative trajectories through a contrastive loss; the resulting per-state push-away gradient from $a ^ { l }$ pulls the policy back from nearby failure modes, yielding a consistent gain across all four tasks. RPRO vs. $\pi _ { 0 . 6 } { } ^ { * }$ . On identical preference data, RPRO still outperforms the advantage-conditioned $\pi _ { 0 . 6 } { } ^ { * }$ baseline. $\pi _ { 0 . 6 } { } ^ { * }$ relies on the model to discover the “improved”/“unimproved” partition from a single conditioning token under a pure regression objective—an indirect pressure that can be diluted by the rest of the VLM context—whereas RPRO (4.2) injects the preference signal directly into the action-generation loss, pushing $\pi _ { \theta }$ toward $a ^ { w }$ and away from $a ^ { l }$ per state and per chunk. Across all four tasks, RPRO attains the highest SR with the shortest CT, indicating both more reliable and more efficient task execution.

![](images/62d93e4199a7fb539761f01a2faa7d26d34403c3967694af21b6c4b019451ec9.jpg)

![](images/bc7f1551ccac78c0508500fdf0180b43dec7b40983c57851d2a6ed8dd93b3d73.jpg)

![](images/a31cf4e4e59d1bb3b7f9ca486a7db23e5c781811e7217b604353d76565cf6c6c.jpg)

![](images/1991744298fda58b0fa657e79a2a2f5a15e77a0021f098c577741ad791635dfa.jpg)  
Fig. 12: Per-iteration success rate on the four real-robot tasks with HyVLA-0.5 as the base policy. Iteration 0 corresponds to the shared SFT checkpoint; iterations 1–3 correspond to successive rounds of post-training. RPRO consistently dominates DAgger and $\pi _ { 0 . 6 } { } ^ { * }$ throughout the iterative process.

## 7 Related Work

Generalist VLA Models Early VLAs abstracted robotic control into discrete tokens processed by autoregressive heads atop pre-trained VLMs, as exemplified by RT-2 [12] and OpenVLA [13]. While effectively transferring semantic priors, this discretisation inherently constrained control frequency and spatial precision. π0 [1] supplanted discrete action spaces with flow-matching velocity fields, restoring continuous, high-frequency (e.g., 50 Hz) execution capabilities. Concurrently, DeepMind introduced Gemini Robotics [3], bringing Gemini-level reasoning to physical control, and NVIDIA released GR00T N1 [4], an open foundation model for generalist humanoid control pre-trained on teleoperation, human video, and synthetic data. $\pi _ { 0 . 5 }$ [2] subsequently advanced the flow-matching paradigm with open-world generalization, while Gemini Robotics 1.5 [39] extended the approach with advanced embodied reasoning and cross-embodiment motion transfer. LingBot-VLA [33] takes a pragmatic approach, scaling to 20 K hours of real-world dual-arm data across 100 tasks with a throughput-optimised open-source codebase. Unlike autoregressive VLAs, HyVLA-0.5 operates entirely within a continuous flow-matching paradigm; unlike $\pi _ { 0 }$ and $\pi _ { 0 . 5 } ,$ it adopts an MoT-based embodied-native backbone, relies on a 10 K-hour UMI pre-training corpus, and features a specialised deployment protocol for zero-shot cross-embodiment transfer.

Embodied VLM Backbones The majority of contemporary VLAs depend on general-purpose vision-language models such as PaliGemma [16] or Qwen-VL [40]. Recently, domain-specific backbones such as RoboBrain [41], RynnBrain [42], and Hy-Embodied-0.5 [15] have emerged to better address the fine-grained visual acuity required for manipulation. The internal Hy-Embodied report introduced a prototype VLA fine-tuned from 5 k hours of UMI data, achieving promising baseline success rates on X-Trainer tasks [15]. Building strictly upon this foundation, HyVLA-0.5 doubles the UMI scale to 10 k hours, implements the rel-EE representation to facilitate humanoid deployment, and introduces the FlowPRO RL post-training stage, thereby extending efficacy to unseen platforms including JAKA and Astribot S1.

Pre-training and Post-training Recipes for VLAs Current multi-embodiment pre-training paradigms, such as TRI’s Large Behaviour Models [43] and the $\pi _ { 0 . 5 }$ methodology, primarily leverage aggregated teleoperation datasets (e.g., Open-X-Embodiment [44], DROID [45]). By contrast, the foundational pre-training signal of HyVLA-0.5 is predominantly sourced from human-centric UMI data, optimising the action expert under a singular flow-matching loss.

Hand-Held Demonstrations and UMI The Universal Manipulation Interface (UMI) [11] pioneered the capture of robot-agnostic demonstration data via hand-held gripper rigs. Subsequent efforts, such as DexUMI [46], expanded the morphological applicability of such rigs. Recently, several frameworks have addressed the feasibility of migrating UMI-style hand-held data to humanoid and mobile systems, such as EgoMI [47] (which captures synchronized head-hand tracking for whole-body and active vision manipulation) and HoMMI [48] (which learns whole-body mobile manipulation directly from robot-free egocentric human demonstrations). HyVLA-0.5 scales UMI data to over 10k hours, and demonstrates UMI-based cross-embodiment transfer to a humanoid under a Stage-2 protocol entirely devoid of target-robot teleoperation.

Preference Post-Training in Continuous Control Real-robot post-training pipelines for VLA models broadly fall into three families, each with a characteristic limitation that re-emerges in the flow-matching setting. (i) SFT and its interactive extensions—vanilla SFT [1, 13] and DAgger-style human correction [38]—scale to real hardware but only weakly exploit the failure signals from autonomous rollouts: vanilla SFT discards them, while DAgger uses them merely to trigger expert correction rather than as a direct optimization signal. (ii) Reward- or value-based RL [14, 49–51] requires training a reliable reward, value, or advantage model, which itself becomes a key obstacle for contact-rich manipulation where dense reward signals are difficult to obtain; HIL-SERL [51] and $\pi _ { 0 . 6 ^ { * } }$ [14] additionally introduce significant engineering overheads such as advantage values and intricate reward shaping. (iii) Preference-based RL bypasses reward design via preference data: Direct Preference Optimization (DPO) [52] operates without critics by optimising likelihood ratios but was originally designed for discrete text, while recent extensions to continuous flow-matching policies, such as Flow-DPO [29] and the trajectory-level GRAPE [53], restore preference learning to flow-based VLAs but inherit the reward-hacking failure mode of plain DPO and dilute the per-state learning signal. Unlike $\pi _ { 0 . 6 } *$ , our FlowPRO recipe (4) is entirely critic- and reward-free; unlike Flow-DPO and GRAPE, the underlying RPRO loss anchors the implicit reward via a proximal regularizer that explicitly forbids the plain-DPO reward-hacking pathology, and exploits a contrastive-gradient-cancellation property to safely co-train on SFT samples through the same objective.

Asynchronous Inference and Action-Chunk Smoothing Action chunking [7] has become the de-facto deployment recipe for VLA policies but introduces intra-chunk jitter, chunk-boundary discontinuities, and idle gaps when the backbone latency exceeds the servo period. Inference-Time RTC [54] introduces a lightweight flow-matching action server that refines coarse action chunks at high frequency, decoupling the slow backbone from fast control; Training-Time RTC [55] further co-trains this refinement module with the policy. VLASH [56] learns an adaptive halting mechanism that determines chunk size based on task complexity, reducing inter-chunk gaps. By contrast, our deployment recipe (5) is training-free and plug-and-play for arbitrary policies, explicitly guarantees $C ^ { \mathrm { { f } } }$ continuity at chunk boundaries via tangent-aligned cubic Bézier curves, and is applicable to both Cartesian and joint-space control.

## 8 Discussion

HyVLA-0.5 Pipeline HyVLA-0.5 co-designs data, representation, policy refinement, and deployment execution for deployable generalist robots, rather than treating the VLA as a standalone policy. Cross-embodiment deployment relies on a complete set of components beyond model scale alone: high-fidelity UMI data provides reusable supervision for learning precise manipulation priors; the compact memory encoder and rel-EE/delta-chunk representation give the policy temporal context while keeping the action interface independent of platform-specific kinematics; FlowPRO converts real failure cases into compact offline refinement without requiring large-scale online exploration; and asynchronous chunk stitching makes the same checkpoint executable under real hardware latency. These components address different bottlenecks—data quality, action representation, failure correction, and deployment timing—but they share the same goal: preserving a stable policy interface while absorbing embodiment-specific differences outside the learned core. Together, they turn HyVLA-0.5 from a single model into a practical robot-learning stack for cross-embodiment deployment.

Future Work HyVLA-0.5 opens several questions that we are eager to explore, especially around data, model generalization, and real-world deployment. On the data side, an important direction is to move beyond motion capture while preserving high-precision supervision; exoskeleton-based collection is a promising route toward this goal. Since Hy-UMI-10K already provides high-accuracy action labels, it also offers a simple way to study the marginal value of precision for pre-training, for example by injecting controlled noise into the labels. In addition, the egocentric UMI camera still differs from robot-mounted deployment cameras, leaving room for systematic visual augmentation studies. To support these explorations, we will release a 2,000-hour self-collected UMI subset and invite the community to study these questions and beyond.

Another key direction is real-world execution efficiency. In deployment, success is not only whether the robot can complete a task, but also whether it can execute at a practical task cadence. A key next step is therefore to improve deployment-time execution speed while maintaining safety and precision. This likely requires combining deployment-time adaptation with reinforcement learning.

Finally, the emergence of embodied intelligence remains an important open direction. HyVLA-0.5 does not study zero-shot generalization, as we believe the current data scale is still insufficient for making such claims. At the same time, recent systems such as π0.7 [57] have begun to show early signs of zero-shot behavior, suggesting that larger-scale data and stronger pipelines may lead to qualitatively new capabilities. How to evaluate these capabilities rigorously, and how to use evaluation itself to drive the iteration of embodied models and deployment pipelines, remains an open problem.

## References

[1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[2] Physical Intelligence. π0.5: A VLA with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[3] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[4] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[5] Haitao Lin, Hanyang Yu, Jingshun Huang, He Zhang, Yonggen Ling, Ping Tan, Xiangyang Xue, and Yanwei Fu. Universal pose pretraining for generalizable vision-language-action policies. RSS 2026, 2026.

[6] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[7] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023.

[8] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[9] Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, et al. Egovla: Learning vision-language-action models from egocentric human videos. arXiv preprint arXiv:2507.12440, 2025.

[10] Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, et al. Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos. arXiv preprint arXiv:2510.21571, 2025.

[11] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Robotics: Science and Systems (RSS), 2024.

[12] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023.

[13] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024.

[14] Physical Intelligence. π∗0.6: A VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[15] Tencent Robotics X and Tencent HY Vision Team. Hy-Embodied-0.5: Embodied foundation models for real-world agents. Tencent hy technical report, Tencent, 2025. URL https: //github.com/Tencent-Hunyuan/HY-Embodied.

[16] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726, 2024.

[17] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

[18] Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin Zhao, Subhashree Radhakrishnan, et al. Eagle 2: Building post-training data strategies from scratch for frontier vision-language models. arXiv preprint arXiv:2501.14818, 2025.

[19] Yihao Wu, He Zhang, Junbo Tan, Xueqian Wang, and Zhengyou Zhang. Flowpro: Reward-free reinforced fine-tuning of flow-matching vlas via proximalized preference optimization. In arXiv preprint arXiv:2606.05468, 2026. Under review.

[20] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. Mixture-oftransformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

[21] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[22] Physical Intelligence. Multi-scale embodied memory for vision-language-action models. arXiv preprint arXiv:2603.03596, 2026.

[23] He Zhang, Sebastian Starke, Taku Komura, and Jun Saito. Mode-adaptive neural networks for quadruped motion control. ACM Transactions on Graphics (ToG), 37(4):1–11, 2018.

[24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[25] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023.

[26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https://arxiv.org/abs/1711.05101.

[27] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, and Yubin Guo. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[28] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. International Conference on Learning Representations (ICLR), 2023.

[29] Jie Liu, Gongye Liu, Jiajun Liang, Zhihao Yuan, Xingchao Liu, Mengnan Zheng, Xuewei Wu, Qian Wang, Wei Qin, Min Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

[30] Kuan Guo, Yifan Li, and Zhengyang Chen. Proximalized preference optimization for diverse feedback types: A decomposed perspective on DPO. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 94533–94576, 2026.

[31] Fan Yang et al. ABot-M0: VLA foundation model for robotic manipulation with action manifold learning. arXiv preprint arXiv:2602.11236, 2026.

[32] Qwen Team. Qwen-vla: Unifying vision-language-action modeling across tasks, environments, and robot embodiments. 2026. URL https://arxiv.org/abs/2605.30280.

[33] Wei Wu et al. LingBot-VLA: A pragmatic VLA foundation model. arXiv preprint arXiv:2601.18692, 2026.

[34] StarVLA Community. starVLA: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

[35] Dong Bi et al. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025.

[36] Tianle Zhang et al. JoyAI-RA 0.1: A foundation model for robotic autonomy. arXiv preprint arXiv:2604.20100, 2026.

[37] Colin Lea, René Vidal, Austin Reiter, and Gregory D. Hager. Temporal convolutional networks: A unified approach to action segmentation. In Computer Vision – ECCV 2016 Workshops, pages 47–54. Springer, 2016.

[38] Stéphane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.

[39] Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025.

[40] Shuai Bai, Yuanbo Cai, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[41] Huajie Tan et al. RoboBrain 2.5: Depth in sight, time in mind. arXiv preprint arXiv:2601.14352, 2026.

[42] Ronghao Dang, Jiayan Guo, Zixuan Zeng, Kang Yan, Jinpeng Wu, Chenrui Shi, Haifeng Wang, Le Liu, Shiyang Chen, Jin Huang, Ziming Huang, and Deli Zhao. RynnBrain: Open embodied foundation models, 2026. URL https://arxiv.org/abs/2602.14979.

[43] Jose Barreiros, Aditya Bhat, Eric Cousineau, et al. A careful examination of large behavior models for multitask robot manipulation. arXiv preprint arXiv:2507.05331, 2025.

[44] Open X-Embodiment Collaboration. Open X-Embodiment: Robotic learning datasets and RT-X models. In IEEE International Conference on Robotics and Automation (ICRA), 2024.

[45] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems (RSS), 2024.

[46] Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Goldie Xu, Linxi Fan, Manuela Veloso, and Shuran Song. DexUMI: Using human hand as the universal manipulation interface for dexterous manipulation. arXiv preprint arXiv:2505.21864, 2025.

[47] Justin Yu, Yide Shentu, Di Wu, Pieter Abbeel, and Ken Goldberg. EgoMI: Learning active vision and whole-body manipulation from egocentric human demonstrations. arXiv preprint arXiv:2511.00153, 2025.

[48] Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Shuran Song, and Cheng Chi. HoMMI: Learning whole-body mobile manipulation from human demonstrations. arXiv preprint arXiv:2603.03243, 2026.

[49] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[50] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[51] Jianlan Luo, Charles Xu, Jeffrey Wu, and Sergey Levine. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. Science Robotics, 10(105):eads5033, 2025.

[52] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[53] Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, and Siwei Lyu. GRAPE: Generalizing robot policy via preference alignment. arXiv preprint arXiv:2411.19309, 2024.

[54] Physical Intelligence. Real-time action chunking with large models. arXiv preprint arXiv:2503.07206, 2025.

[55] Physical Intelligence. Training-time real-time chunking: Co-training high-frequency action refinement with policies, 2025. Physical Intelligence Blog Post.

[56] Jiaming Tang, Yufei Sun, Yilong Zhao, et al. VLASH: Real-time VLAs via future-state-aware asynchronous inference. arXiv preprint arXiv:2512.01031, 2025.

[57] Physical Intelligence, Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, et al. π0.7: a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

## Appendix

## A RoboTwin 2.0 Evaluation Details

Per-task results. Table 3 reports the per-task success rates of HyVLA-0.5 on the 50-task RoboTwin 2.0 suite. We evaluate each task under the Clean and Randomized settings and include this breakdown as a complement to the aggregate comparison in Table 1 (§6.1).

Data filtering. We apply an offline cleaning step because a small subset of RoboTwin 2.0 demostrations contains implausible inverse-kinematics solutions, which often manifest as abnormal episode lengths. For each task, we cluster the episode-length distribution using HDBSCAN with cluster-selection radius 5 and all other settings kept at defaults. This identifies stable length modes. An episode is flagged as dirty if it satisfies any of the following conditions: (i) it is assigned as an HDBSCAN noise point; (ii) it belongs to an under-populated length mode with estimated size below 100 episodes; or (iii) it lies in the top 5% length tail of the longest well-populated mode. Episodes passing all three checks form the clean subset used for training.

Action decoding. We let the policy predict actions under two complementary frames: (i) relative-EEF, which captures smooth local motion; and (ii) EEF, which anchors the target globally and avoids drift accumulation. The EEF based actions are concatenated after the relative ones along the chunk axis, resulting a doubled chunk size. At inference these two predictions are fused, with quaternion orientations interpolated via SLERP, combining the local precision of relative motion with the global stability of absolute targets.

<table><tr><td>Task</td><td>Clean</td><td>Rand.</td><td>Task</td><td>Clean</td><td>Rand.</td></tr><tr><td>adjust bottle</td><td>99</td><td>99</td><td>place can basket</td><td>91</td><td>81</td></tr><tr><td>beat block hammer</td><td>99</td><td>99</td><td>place cans plasticbox</td><td>100</td><td>100</td></tr><tr><td>blocks ranking rgb</td><td>99</td><td>99</td><td>place container plate</td><td>98</td><td>99</td></tr><tr><td>blocks ranking size</td><td>93</td><td>94</td><td>place dual shoes</td><td>94</td><td>95</td></tr><tr><td>click alarmclock</td><td>100</td><td>100</td><td> place empty cup</td><td>100</td><td>100</td></tr><tr><td>click bell</td><td>100</td><td>99</td><td>place fan</td><td>96</td><td>98</td></tr><tr><td>dump bin bigbin</td><td>96</td><td>98</td><td> place mouse pad</td><td>88</td><td>93</td></tr><tr><td>grab roller</td><td>100</td><td>100</td><td>place object basket</td><td>86</td><td>85</td></tr><tr><td>handover block</td><td>93</td><td>73</td><td> place object scale</td><td>90</td><td>89</td></tr><tr><td>handover mic</td><td>87</td><td>98</td><td>place object stand</td><td>94</td><td>98</td></tr><tr><td>hanging mug</td><td>37</td><td>33</td><td> place phone stand</td><td>91</td><td>94</td></tr><tr><td>lift pot</td><td>99</td><td>99</td><td>place shoe</td><td>98</td><td>100</td></tr><tr><td>move can pot</td><td>94</td><td>99</td><td>press stapler</td><td>88</td><td>80</td></tr><tr><td>move pillbottle pad</td><td>96</td><td>96</td><td>put bottles dustbin</td><td>89</td><td>85</td></tr><tr><td>move playingcard away</td><td>98</td><td>96</td><td>put object cabinet</td><td>78</td><td>83</td></tr><tr><td>move stapler pad</td><td>94</td><td>95</td><td>rotate qrcode</td><td>93</td><td>98</td></tr><tr><td>open laptop</td><td>99</td><td>98</td><td>scan object</td><td>91</td><td>94</td></tr><tr><td>open microwave</td><td>72</td><td>64</td><td>shake bottle</td><td>100</td><td>99</td></tr><tr><td>pick diverse bottles</td><td>88</td><td>72</td><td>shake bottle horizontally</td><td>100</td><td>99</td></tr><tr><td>pick dual bottles</td><td>82</td><td>82</td><td>stack blocks three</td><td>98</td><td>97</td></tr><tr><td>place a2b left</td><td>77</td><td>75</td><td>stack blocks two</td><td>98</td><td>99</td></tr><tr><td>place a2b right</td><td>74</td><td>78</td><td>stack bowls three</td><td>87</td><td>83</td></tr><tr><td>place bread basket</td><td>95</td><td>93</td><td>stack bowls two</td><td>98</td><td>97</td></tr><tr><td>place bread skillet</td><td>94</td><td>90</td><td>stamp seal</td><td>85</td><td>81</td></tr><tr><td>place burger fries</td><td>99</td><td>98</td><td>turn switch</td><td>50</td><td>49</td></tr><tr><td>Average (50 tasks)</td><td>90.9</td><td>90.1</td><td></td><td></td><td></td></tr></table>

Table 3: Per-task evaluation results of HyVLA-0.5 on the RoboTwin 2.0 benchmark [27].

## B Supplementary Deployment

## B.1 UMI-to-Robot Deployment Derivation

Humanoid-specific derivation. UMI demonstrations are recorded in its own world frame and lack a torso pose. As end effector poses on Astribot S1 are defined in its own chassis frame, as mentioned in Eq. (11), and a torso pose in the chassis frame is crucial for reasonable upper-body poses and efficient IK solving, we need to find a mapping from UMI world frame to S1 chassis frame and figure out a torso pose as well. Two methodologies have been proved feasible either by our experiments or by related works:

1. Heuristic torso/head pose inference (used in our experiments). A lightweight rule-based estimator consumes bimanual gripper poses $\{ ^ { W } T _ { G _ { \epsilon } ^ { \mathrm { L } } } , ^ { W ^ { * } } T _ { G _ { \epsilon } ^ { \mathrm { R } } } \}$ and infers the world-to-chassis transform, the torso pose, and the head pose such that (i) the torso forward axis aligns with the centroid of the two gripper positions and (ii) the torso height places both grippers within an empirically established comfortable reach shell of the upper body; see Algorithm 1 for details. It assumes that the UMI world frame and the robot chassis frame are related by a pure translation (identical orientation), which holds on Astribot S1. For robot platforms whose chassis frame has a different orientation, an additional fixed rotation can be applied without further altering the algorithm.

2. Whole-body IK solvers (alternative). HoMMI-style whole-body IK [48] jointly resolves torso and arm configurations from EE targets and could replace the heuristic above. We document this compatibility for completeness; our Astribot S1 results in §6.2 use the heuristic exclusively.

## B.2 Track-B Reachability and Data Hygiene

Because UMI demonstrations are captured without a robot in the loop, they intrinsically lack any guarantee of reachability for arbitrary target morphologies. We deploy two standardised pre-deployment hygiene protocols, both executed offline with zero runtime overhead:

– Unitree G1 & Astribot S1. A pre-deployment reachability verification bounds the planned task envelope to the platform; tasks exceeding the humanoid’s reachable shell are excluded from this report.

– JAKA K1. The post-training UMI corpus for a given JAKA task is filtered via a single-pass IK feasibility check on the target arm; trajectories that violate JAKA’s arm kinematics are removed from the post-training set.

Neither mechanism alters the policy or its action representation; they merely enforce distributional alignment between the post-training set and the physical deployment frontier.

## C FlowPRO Hyperparameters

Confirmed implementations:

– Iterations: $k \in \{ 1 , 2 , 3 \}$ ; each round runs 25 000 optimizer steps (75 000 total).

– Batch size: Global batch size of 20 (5 samples per GPU).

– Optimizer: AdamW, initial learning rate $1 \times 1 0 ^ { - 5 }$ , linear warmup over 1,000 steps, cosine decay over the next 15,000 steps to a floor of $2 . 5 \times 1 0 ^ { - 6 }$

– Batch composition:

$$
\bullet k = 1 \dot { \ ! } \mathcal { D } _ { \mathrm { p r e f } } ^ { 1 } / \mathcal { D } _ { \mathrm { S F T } } = 8 0 / 2 0 .
$$

$$
\bullet k \geq 2       : \mathcal { D } _ { \mathrm { p r e f } } ^ { k } / \mathcal { D } _ { \mathrm { p r e f } } ^ { < k } / \mathcal { D } _ { \mathrm { S F T } } = 7 0 / 1 5 / 1 5 .
$$

– Distance metric: To find the closest point M′ on $\tau ^ { w }$ for a given state M on $\tau _ { \cdot } ^ { l }$ , we use $d ( M , M ^ { \prime } ) = \lVert p _ { M } - p _ { M ^ { \prime } } \rVert _ { 2 } + 0 . 5 \cdot d _ { \mathrm { g e o } } ( R _ { M } , R _ { M ^ { \prime } } ) + 0 . 2 \cdot | \bar { g _ { M } } - g _ { M ^ { \prime } } |$ , where $\pmb { p } \in ^ { 3 }$ is the end-effector position, $d _ { \mathrm { g e o } }$ is the geodesic distance between rotation matrices in $S O ( 3 ) , g \in [ 0 , 1 ]$ is the normalized gripper width, and the weights are set empirically.

– Initialization: The Stage-2 checkpoint serves as both initialisation θ and frozen reference policy $\theta _ { \mathrm { r e f } }$

– Data scale: $\leq \mathcal { O } ( 1 0 ^ { 2 } )$ preference pairs per task; X-Trainer rollouts only.

Algorithm 1: Heuristic Mapping from UMI Gripper Poses to Whole-Body Targets in the Chassis   
Frame   
Input $\overline { { : T _ { L } ^ { W } , T _ { R } ^ { W } \in S E ( 3 ) } }$ $^ { \prime * }$ UMI gripper poses in the world frame W \*/   
1 $L$ $^ { \prime * }$ full nominal arm reach (meters) $^ { * / }$   
2 $h _ { 0 }$ $^ { \prime * }$ nominal standing height of chassis–shoulder line \*/   
3 $\alpha \in [ 0 , 1 ]$ $^ { \prime * }$ horizontal back-shift as a fraction of $L ~ * /$   
4 $\Delta z _ { C }$ $^ { \prime * }$ net vertical offset for chassis localization \*/   
5 $\theta _ { 0 }$ $^ { \prime * }$ constant forward torso pitch $^ { * / }$   
6 $\delta \in [ 0 , 1 ]$ $^ { \prime * }$ blend factor between hand height and $h _ { 0 }$ \*/   
$R _ { \mathrm { a l i g n } }$ $^ { \prime * }$ fixed UMI→robot gripper-axis rotation \*/   
8 $T _ { H } ^ { T }$ /\* fixed torso-to-head calibration transform $^ { * / }$   
Output Chassis-frame targets $T _ { L } ^ { C } , T _ { R } ^ { C } , T _ { T } ^ { C } , T _ { H } ^ { C }$   
:   
$^ { \prime * }$ Step 1: align UMI gripper axes to robot gripper axes $^ { * / }$   
9 $T _ { L } ^ { W } \gets T _ { L } ^ { W } \ : R _ { \mathrm { a l i g n } }$   
10 $T _ { R } ^ { \overline { { { W } } } }  T _ { R } ^ { \overline { { { W } } } } \ : R _ { \mathrm { a l i g n } }$   
/\* Step 2: hand midpoint and horizontal facing direction $^ { * / }$   
11 $m ^ { W }  \textstyle { \frac { 1 } { 2 } } \bigl ( t ( T _ { L } ^ { W } ) + t ( T _ { R } ^ { W } ) \bigr )$ $^ { \prime * }$ mean of hand translations $^ { * / }$   
12 $f ^ { W } \gets \tilde { \Pi } _ { x y } ( m ^ { W } ) / \left. \Pi _ { x y } ( m ^ { W } ) \right.$ $^ { \prime * }$ unit vector in world XY plane $^ { * / }$   
13 $\mathbf { i f } \parallel \Pi _ { x y } ( m ^ { W } ) \parallel < \varepsilon$ then   
14 $\begin{array} { r l } { \mathbf { \eta } } & { { } \mathbf { \Delta } f ^ { W }  \mathbf { e } _ { x } } \end{array}$ $^ { \prime * }$ degenerate fallback $^ { * / }$   
15 end   
$^ { \prime * }$ Step 3: one-shot chassis localization (cached per episode) $^ { * / }$   
16 if $T _ { C _ { - } } ^ { W }$ is not cached then   
17 $p _ { C } ^ { W }  m ^ { W } - \alpha L f ^ { W } + \Delta z _ { C } \mathbf { e } _ { z }$ $^ { \prime * }$ back-shift + vertical drop $^ { * / }$   
18 $T _ { C } ^ { W } \gets ( \mathbf { I } , p _ { C } ^ { W } )$   
19 cache $T _ { C } ^ { W }$   
20 end   
/\* Step 4: re-express grippers and helpers in the chassis frame $^ { * / }$   
21 $T _ { L } ^ { C }  \overline { { ( } } T _ { C } ^ { W } ) ^ { - 1 } T _ { L } ^ { W }$   
22 $T _ { R } ^ { C } \gets ( T _ { C } ^ { W } ) ^ { - 1 } T _ { R } ^ { W }$   
23 $m ^ { \mathrm { { \tilde { C } } } } \gets ( T _ { C } ^ { W } ) ^ { - 1 } m ^ { W }$   
24 $f ^ { C } \gets \dot { R } ( \breve { T } _ { C } ^ { W } ) ^ { \top } f ^ { W }$   
/\* Step 5: heuristic torso pose $^ { * / }$   
25 $\psi  \mathrm { a t a n 2 } ( f _ { y } ^ { C } , f _ { x } ^ { C } )$ $^ { \prime * }$ yaw aligned with the hands $^ { * / }$   
26 $R _ { T } ^ { C }  R _ { z } ( \psi ) \backslash R _ { y } ( \theta _ { 0 } )$ $^ { \prime * }$ yaw, then constant forward pitch $^ { * / }$   
27 $p _ { T } ^ { \bar { C } } \gets \left( 0 , 0 , \left( 1 - \delta \right) m _ { z } ^ { C } + \delta h _ { 0 } \right) ^ { \top }$ /\* height = convex blend of hand and standing   
\*/   
28 $\dot { T _ { T } ^ { C } } \gets ( R _ { T } ^ { C } , p _ { T } ^ { C } )$   
/\* Step 6: head by fixed torso-to-head transform \*/   
29 $T _ { H } ^ { C } \gets T _ { T } ^ { C } T _ { H } ^ { T }$   
30 return $( T _ { L } ^ { C } , T _ { R } ^ { C } , T _ { T } ^ { C } , T _ { H } ^ { C } )$

## D Author Contributions

Project supervisors. Han Hu and Zhengyou Zhang.

Project leaders. He Zhang and Lingzhu Xiang.

Core contributors. Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, and Yongming Rao.

Contributors. Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang and Zisheng Lu.