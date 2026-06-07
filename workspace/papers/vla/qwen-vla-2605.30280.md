# Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments

Qwen Team

https://qwen.ai/blog?id=qwenvla

https://github.com/QwenLM/Qwen-VLA

## Abstract

Embodied intelligence is often studied through specialized models, each designed for a single scenario or task, such as manipulation and navigation, leading to fragmented capabilities and limited generalization across diverse tasks, environments, and robot embodiments. In this work, we investigate whether these heterogeneous embodied decision-making problems can be unified within a single vision-language-action model. We present Qwen-VLA, a unified embodied foundation model that extends Qwen’s vision-language modeling stack from perception, understanding and reasoning to continuous action and trajectory generation through a DiT-based action decoder. Our approach adopts a large-scale joint pretraining recipe over diverse data sources, including robotics manipulation trajectories, human egocentric demonstrations, synthetic simulation data, vision-and-language navigation data, trajectory-centric supervision, and auxiliary visionlanguage data. To support multiple robot platforms within a shared model, we introduce embodiment-aware prompt conditioning, where robot-specific textual descriptions are prepended to specify the current embodiment and control convention. We further cast manipulation, navigation, and trajectory prediction into a unified action-and-trajectory prediction framework, enabling transferable visual grounding, spatial reasoning, and continuous action generation across robot morphologies, task families, and environments. Experiments on manipulation, navigation, and trajectory-centric benchmarks show that Qwen-VLA supports embodied control across task families and robot embodiments, with consistent multi-task performance and out-of-distribution generalization across variations in scene layout, background, lighting, object configuration, and robot embodiment. As a unified generalist policy, Qwen-VLA-Instruct simultaneously achieves 97.9% on LIBERO, 73.7% on Simpler-WidowX, 86.1/87.2% on RoboTwin-Easy/Hard, 69.0% OSR on R2R, and 59.6% SR on RxR, while further attaining 76.9% average OOD success in real-world ALOHA experiments and 26.6% zero-shot success rate on DOMINO dynamic manipulation.

![](images/04a32bf2af40e3c757d57e04bdffe78644c14873b5576bb3b7b13b429920e935.jpg)  
Figure 1: Overview of Qwen-VLA, a unified embodied model trained on mixed manipulation, navigation, and vision-language understanding data to generate both robot actions and textual responses.

## 1 Introduction

Embodied intelligence (Zitkovich et al., 2023; O’Neill et al., 2024) aims to build agents that can perceive the physical world, understand natural-language instructions, reason over spatial and temporal context, and execute actions to accomplish goals. Recent progress in vision-language models (VLMs) (Wang et al., 2024; Bai et al., 2025b; Team et al., 2025; Bai et al., 2025a; Team, 2026) has substantially improved open-world visual understanding and language grounding, while diffusion- or flow-based policies (Black et al., 2024; 2025; Intelligence et al., 2026) have shown strong potential for modeling continuous, highdimensional robot actions. However, most existing embodied systems (NVIDIA et al., 2025; Luo et al., 2025; Yang et al., 2026) remain specialized to a narrow task family, robot embodiment, or evaluation setting. Manipulation models (Intelligence et al., 2026) are often trained for tabletop or dexterous control, and navigation models (Zhang et al., 2024; 2025b; Cheng et al., 2025) are designed around waypoint or action prediction in indoor environments. This fragmentation limits transfer across tasks, environments, and embodiments, and makes it difficult to scale embodied learning in the same way as general-purpose vision-language pretraining.

A central challenge is that embodied decision-making problems appear heterogeneous on the surface. Robot manipulation may require predicting end-effector poses (Black et al., 2024; Zheng et al., 2025; Zhang et al., 2026), joint positions (Fu et al., 2024; NVIDIA et al., 2025), gripper states, or dexterous-hand configurations (Luo et al., 2025); navigation may require predicting waypoints or discrete movement decisions (Zhang et al., 2025a; Wang et al., 2025a; Wei et al., 2025); egocentric human demonstrations may provide wrist and hand trajectories rather than robot control signals. These tasks also differ in observation format, control frequency, prediction horizon, action dimensionality, and evaluation protocol. Nevertheless, they share a common computational structure: an embodied agent must condition on visual observations, language instructions, and embodiment-specific constraints, then predict future actions or trajectories that are physically and semantically aligned with the task. This observation motivates a unified formulation for embodied modeling. We leverage this insight to design a joint pretraining framework that absorbs manipulation, navigation, egocentric human demonstrations, and vision-language reasoning into a single vision-language-action model. The resulting model generalizes across embodiments and task settings.

In this work, we present Qwen-VLA, a unified vision-language-action model for heterogeneous embodied decision-making. Qwen-VLA is built upon Qwen3.5-4B, a natively multimodal backbone from Qwen families that has demonstrated state-of-the-art performance across a wide range of vision-language benchmarks. The backbone brings strong fine-grained visual perception, robust referential grounding, multilingual instruction following, and structured reasoning capabilities, which are critical for embodied tasks that require interpreting spatial relations, identifying referred objects, and following complex multi-step instructions. On top of the backbone, we attach a DiT-based flow-matching action decoder that specializes in fine-grained continuous action generation. Instead of designing separate output heads or task-specific architectures for different embodiments, Qwen-VLA represents manipulation actions, navigation waypoints, and human egocentric motions in a shared action-and-trajectory prediction space. This design allows a single model to absorb supervision from diverse embodied datasets while preserving a unified inference interface.

A key component of Qwen-VLA is large-scale joint pretraining over diverse embodied and visionlanguage data. Our pretraining mixture includes robot manipulation trajectories, human egocentric demonstrations, synthetic simulation trajectories, vision-and-language navigation data, spatial grounding data, autonomous-driving VQA, fine-grained embodied action captions, and general vision-language data. This mixture is designed to cover both low-level motor priors and high-level semantic reasoning. Robot trajectories provide direct action supervision, egocentric human data supplies scalable real-world manipulation priors, navigation data introduces long-horizon instruction following and exploration, synthetic simulation improves controllability and out-of-distribution coverage, and vision-language data, i.e., selectively sampled for embodied-relevant capabilities such as spatial reasoning, referential grounding, and instruction following, preserves and strengthens the backbone’s perception and reasoning foundations for downstream embodied tasks.

To enable cross-embodiment learning, we introduce embodiment-aware prompt conditioning. Each training example is prepended with a textual description of the current embodiment, including the robot platform, arm configuration, control convention, control frequency, and prediction horizon. The prompt serves as the sole interface through which the model is informed of embodiment-specific control semantics. Combined with a unified action representation, this allows the same action decoder to handle different control modes, action dimensions, and temporal horizons without changing model architecture. As a result, Qwen-VLA can learn shared visual grounding and spatial reasoning capabilities across heterogeneous embodiments while still respecting platform-specific action conventions.

Training such a model is non-trivial because the VLM backbone and action decoder enter optimization in asymmetric states: the backbone already possesses general-purpose vision-language representations from large-scale pretraining, while the DiT action decoder is randomly initialized. Naively launching multimodal joint training from this point can waste computation on vision-independent decoder learning and destabilize the pretrained representations. We therefore adopt a staged training recipe grounded in a compression view of action learning. A language instruction such as “pick up the red cup” together with an embodiment prompt compactly encodes the task intent in a handful of tokens, yet the corresponding action trajectory may span hundreds of high-dimensional joint-position values. Bridging this dimensionality gap is a structured decompression problem, and our first training stage, i.e., text-to-action DiT pretraining (T2A), teaches the decoder to serve as a language-conditioned action decompressor before any visual input is introduced. Once this compressed action prior is in place, continued pretraining grounds it in visual observations, supervised fine-tuning specializes it for downstream tasks, and reinforcement learning optimizes closed-loop task success. This recipe separates action-prior compression, visual grounding, task specialization, and success-driven refinement into distinct stages.

We evaluate Qwen-VLA across multiple embodied settings, including robot manipulation, vision-andlanguage navigation, out-of-distribution generalization, and cross-embodiment transfer. The evaluation is designed not only to measure in-domain task success, but also to test whether a unified model can generalize under changes in object layout, lighting, background, language instruction, sensor noise, and robot embodiment. Empirically, our results show that large-scale joint pretraining over heterogeneous embodied data yields strong multi-task performance and improves robustness beyond narrow specialist training. These findings support the view that manipulation, navigation, and trajectory-centric embodied tasks can be treated as different manifestations of a shared action-and-trajectory prediction problem.

Our contributions are summarized as follows:

• We introduce Qwen-VLA, a unified vision-language-action model that formulates manipulation, navigation, and egocentric action modeling within a shared action-and-trajectory space. Built on the Qwen3.5-4B vision-language backbone with a DiT-based flow-matching policy head, Qwen-VLA supports embodied control across multiple robot platforms and task families.

• We construct a large-scale joint pretraining mixture over heterogeneous data sources, including manipulation trajectories from multiple robots, egocentric human demonstrations, synthetic simulation, navigation, and curated vision-language data. We further propose embodiment-aware prompt conditioning that unifies diverse robot platforms, control conventions, and prediction horizons in one model without separate per-embodiment policies.

• We design a progressive training recipe that includes action pretraining, multimodal continued pretraining, supervised fine-tuning, and reinforcement learning to build a strong policy model that bridges the gap between discrete vision-language tokens and continuous action trajectories, improving both training stability and downstream transfer.

• We evaluate Qwen-VLA across manipulation, navigation, out-of-distribution robustness, and cross-embodiment generalization benchmarks. The results show that co-training and progressive learning improve multi-task performance under scene, object, lighting, and embodiment shifts.

## 2 Unified Embodied Model

## 2.1 Problem Formulation

We study a broad family of embodied decision-making tasks, including robot manipulation, visionand-language navigation, trajectory prediction, and human egocentric action modeling. We propose to address these tasks within a unified embodied model, motivated by their shared computational structure: despite differences in output format and evaluation protocol, all tasks require an agent to ground language in visual observations, reason over spatial and temporal context, and predict future actions or trajectories.

We formulate all tasks in a unified conditional prediction framework. At time step t, the model receives a visual context $o _ { t } ,$ a language instruction x, an embodiment description e, and an optional task identifier z. Here, ot may consist of one or multiple image frames, video observations, or history windows; x specifies the task instruction; e is a textual prompt describing the current robot platform and control convention; and z identifies the task family when needed. The model is trained to predict a target sequence $y _ { t : t + H - 1 }$ over a prediction horizon H:

$$
p _ { \theta } \big ( y _ { t : t + H - 1 } \mid o _ { t } , x , e , z \big ) .
$$

The target sequence $y _ { t : t + H - 1 }$ is task-dependent but represented in a unified action-and-trajectory space.

For manipulation tasks, it corresponds to future robot actions like end-effector positions; for navigation tasks, it represents navigation decisions or waypoints; for trajectory-centric tasks such as autonomous driving or motion forecasting, it denotes the precise future spatial trajectory of the agent or surrounding entities in continuous coordinate space; and for egocentric embodied data, it captures human body or hand motion trajectories represented in structured pose spaces such as MANO (Romero et al., 2017) or skeletal joint sequences. This unified formulation enables joint optimization across heterogeneous embodied datasets within a single model, facilitating transferable visual grounding, spatial reasoning, and action generation across task families. The formulation is also extensible along both the input and output axes: on the input side, augmenting the conditioning context ot with episodic memory or persistent state would enable long-horizon planning and failure recovery; on the output side, co-predicting future visual states alongside actions would unify action generation with world modeling, allowing the agent to anticipate the consequences of its actions.

## 2.2 Model Architecture

Our model comprises a vision-language backbone for high-level understanding and reasoning and a flow-matching action expert for fine-grained action generation.

Vision-language backbone. We adopt Qwen3.5 (Team, 2026) as the backbone. Qwen3.5 is a natively multimodal model trained with early vision-language fusion: visual tokens produced by a ViT with spatial merging are interleaved directly into the text token stream, enabling unified processing of images, videos, and language within a single transformer. Its hybrid attention design combines gated linear attention in the majority of layers with grouped-query softmax attention at regular intervals, allowing efficient encoding of long multimodal sequences while retaining full-precision global reasoning where needed.

Action expert. We attach a single-stream DiT-style (Esser et al., 2024) flow matching policy as an action expert for predicting precise actions across both robot and human embodied data (Janner et al., 2022; Chi et al., 2023; Liang et al., 2023; Black et al., 2024). The action expert concatenates VLM hidden states with a noisy action chunk into one sequence and processes them through joint self-attention with AdaLN timestep conditioning (Peebles &Xie, 2023) and multi-section RoPE aligned with the backbone. This decoupled design lets the action expert specialize in fine-grained action generation and naturally handle the multimodality and high-frequency dynamics of embodied action distributions, while preserving the backbone’s pretrained capabilities. The expert is trained with a flow-matching objective (Lipman et al., 2023) and produces action sequences via a few Euler integration steps at inference, enabling low-latency real-time control. In total, our action expert contains approximately 1.15B parameters: 16 DiT blocks account for the bulk (70.8M each, 1.13B combined), with the remaining parameters distributed among action projection MLPs that map between the raw action dimension and the DiT latent space (4.9M), a linear layer that transforms VLM hidden states to the DiT channel dimension (3.9M), timestep embedding (2.8M), and output AdaLN modulation (4.7M).

## 2.3 Embodiment-aware Prompt Conditioning

To support multiple robot embodiments within one shared model, we prepend a robot-specific textual prompt to each training example that describes the current platform, arm configuration, and control convention. The prompt follows the template:

The robot is {robot\_tag} with {single arm / dual arms}[, waist][, and mobile base]. The control frequency is {FPS} Hz. Please predict the next {chunk\_size} control actions to execute the following task: {ori\_instruction}.

The robot tag and optional modifiers (waist, mobile base) are set per embodiment; FPS and chunk\_size reflect the dataset’s original control frequency and prediction horizon. Table 2 in Section 3.2 summarises the representative robot platforms covered by our pretraining corpus together with their arm configurations and action types.

## 2.4 Unified Action and Trajectory Representation

We unify the tensor interface and masking scheme, but do not force all embodiments into a single physical action semantic space. Each dataset preserves its native control convention, specified through the embodiment prompt and dataset-specific normalization. Concretely, each training sample contributes a target tensor $\mathbf { Y } \in \mathbb { R } ^ { H \times K }$ , where H is a fixed prediction horizon and K is a fixed channel dimension shared across all control modes.

Control signal types. We cover two families of continuous control signals. Manipulation signals include delta end-effector position $( \Delta x , \Delta y , \Delta z )$ , end-effector rotation expressed as Euler angles or quaternions, absolute joint positions, gripper aperture, and dexterous-hand joint angles. Navigation trajectory signals follow the VLN convention and are represented as $\left( \Delta x , \Delta y , \Delta \theta \right)$ per waypoint, encoding relative displacement and heading change in the ground plane. Despite their different physical semantics, both families are sequences of real-valued vectors predicted over a horizon, and are therefore treated identically by the action expert.

Channel layout. A given control mode uses $c \leq K$ channels. These c task-relevant values are placed in the leading c dimensions of Y, and the remaining $K - c$ dimensions are zero-padded. A per-channel binary mask $\mathbf { M } \in \{ 0 , 1 \} ^ { H \times K }$ records which channels carry valid signals: $M _ { h , k } = 1$ if and only if channel $k < c$ and time step h falls within the task’s chunk length $H _ { \mathrm { t a s k } } \leq H$ . This scheme requires no embodimentspecific output heads; a single set of DiT parameters handles all control modes, with the mask preventing padded entries from influencing the gradient.

Task-aware conditioning. Each training sample is prefixed with the embodiment-aware prompt described in Section 2.3, which specifies the robot platform, arm configuration, control frequency, and prediction horizon. For VLN samples the prompt analogously states the navigation convention and waypoint horizon. These prompt tokens are processed by the VLM backbone and their hidden states are concatenated with the noisy action chunk as input to the DiT, so the action expert is always conditioned on the precise control specification of the current sample without any architectural changes.

## 2.5 Training Objectives

We train the entire model end-to-end with a weighted sum of two objectives that cover continuous action generation and vision-language understanding.

Flow-matching action loss. For all samples with continuous control targets (manipulation, VLN trajectory waypoints, and human egocentric data after action alignment), we supervise the action expert with a conditional flow-matching objective (Lipman et al., 2023). Given a clean target $\mathbf { Y } _ { 0 } \in \mathbb { R } ^ { H \times K }$ and noise $\mathbf { Y } _ { 1 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ , we form the linear interpolant ${ \bf Y } _ { \tau } = ( 1 - \tau ) { \bf Y } _ { 0 } + \tau { \bf Y } _ { 1 }$ with $\tau \in \mathsf { [ 0 , 1 ] }$ , and train the expert $v _ { \theta }$ to predict the conditional velocity field.

To avoid the gradient being dominated by padding, we apply a per-channel, per-step loss with two levels of averaging. For a sample whose control mode activates c channels and $H _ { \mathrm { t a s k } }$ time steps, let $\mathbf { M } \in \{ 0 , 1 \} ^ { H \times K }$ be the validity mask defined in Section 2.4. We first compute the mean squared error for each active channel $k < c \mathrm { : }$

$$
\ell _ { k } = \frac { \displaystyle \sum _ { h = 1 } ^ { H } { M _ { h , k } \left\| \left( v _ { \theta } ( \mathbf { Y } _ { \tau } , \tau \mid o _ { 1 : t } , x , e , z ) - ( \mathbf { Y } _ { 1 } - \mathbf { Y } _ { 0 } ) \right) _ { h , k } \right\| _ { 2 } ^ { 2 } } } { \displaystyle \sum _ { h = 1 } ^ { H } { M _ { h , k } } } ,\tag{1}
$$

and then average uniformly over the c active channels:

$$
\mathcal { L } _ { \mathrm { a c t } } = \mathbb { E } _ { \tau , \Upsilon _ { 0 } , \Upsilon _ { 1 } } \left[ \frac { 1 } { c } \sum _ { k = 0 } ^ { c - 1 } \ell _ { k } \right] .\tag{2}
$$

This two-level averaging ensures that each control dimension contributes equally to the gradient regardless of how many channels a given embodiment uses, and that padded positions are fully excluded. At inference, action chunks are produced by a few Euler integration steps from $\tau = 1 \mathrm { t o } \tau = \dot { 0 }$

Vision-language loss. To preserve and strengthen the multimodal capabilities of the backbone, we keep a standard next-token prediction loss on auxiliary vision-language data, fine-grained embodied action captions, autonomous driving VQA, and general VL pretraining corpora:

$$
\mathcal { L } _ { \mathrm { v l } } = - \sum _ { i } \log p _ { \theta } ( w _ { i } \mid w _ { < i } , o _ { 1 : t } ) ,\tag{3}
$$

where wi are text tokens. This objective stabilizes language grounding under heavy embodied co-training and prevents catastrophic forgetting of perception and reasoning skills.

![](images/5f466c533306c3c8de94a4842048f07ca8911973334d0add677bca28daf9bbd2.jpg)  
Figure 2: Training recipe of Qwen-VLA. Stage I (T2A) trains the DiT action decoder to reconstruct actions from text alone, building a structured action prior without visual input. Stage II (CPT) unfreezes both modules to ground this prior in visual observations. Stage III (SFT) branches into multi-task and ❄"real-robot tracks, and Stage IV (RL) optimizes closed-loop task success via environment rewards.

Joint objective. The overall training loss is a weighted combination

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { a c t } } \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { v l } } \mathcal { L } _ { \mathrm { v l } } , } \end{array}\tag{4}
$$

where the weights $\lambda _ { \mathrm { { a c t } } } , \lambda _ { \mathrm { { v l } } }$ are tuned to balance the gradient magnitudes of the two objectives. Within each mini-batch we mix samples from all task families according to a fixed sampling ratio, so that every optimization step jointly updates the backbone and the action expert with manipulation, VLN trajectory, and vision-language signals.

## 3 Large-Scale Joint Pretraining

## 3.1 Training Recipe

A usable VLA model requires co-training a cognitive backbone and a motor decoder, a division of labor analogous to the complementary roles of the cerebrum and cerebellum in biological motor control. In practice, however, these two modules enter training in deeply asymmetric states: the VLM backbone is already strongly pretrained, while the DiT action decoder is randomly initialized. Naive joint training from this starting point is inefficient and unstable. The decoder has to learn several things at once: the shape of the action distribution, how to condition on language and embodiment, the flow-matching dynamics of its own parameterization, and how to ground actions in vision. At the same time, every step pays the cost of encoding images. Meanwhile, noisy gradients from a fresh decoder may perturb the pretrained backbone before the decoder has learned useful action structure.

Our staged recipe is motivated by a compression perspective on action learning. Raw action trajectories are dense, high-frequency, and embodiment-dependent: a single manipulation episode may contain thousands of joint-position values spread across dozens of degrees of freedom. Yet the underlying task intent is compactly captured by a language instruction (“pick up the red $\mathrm { c u p ^ { \prime \prime } } )$ and an embodiment prompt specifying the robot platform and control convention. This description fits in a handful of tokens. There exists a vast dimensionality gap between this compressed task description and the full action signal; bridging it is a structured decompression problem.

We cast T2A as learning this decompression map. By withholding images and training the DiT solely on language-conditioned action prediction, we force the decoder to acquire a structured prior over the action space that is indexed entirely by language. This is more than a warm-start: the decoder learns how different linguistic descriptions select different regions of the action distribution, how embodiment prompts modulate the same task intent into platform-specific motor programs, and the temporal coherence and compositionality of full action trajectories at the sequence level, using only the compressed description. With this language-indexed action prior in place, subsequent multimodal training can focus its capacity on grounding the prior in concrete visual observations rather than learning action generation from scratch.

Based on this principle, we adopt a four-stage training recipe on top of a pretrained Qwen3.5 VLM backbone: (I) text-to-action DiT pretraining (T2A), (II) continued pretraining (CPT), (III) supervised fine-tuning (SFT) in two parallel branches, and (IV) reinforcement learning (RL). Each stage is defined by the gap it closes in the one before it.

Stage I: Text-to-action DiT pretraining (T2A). We freeze the VLM and train only the DiT, conditioning it on text and the embodiment prompt e (Section 2.3), but deliberately withholding images. The decoder thus operates as a pure language-to-action decompressor: it must reconstruct high-dimensional action distributions from the compact linguistic encoding alone, without any visual shortcut. In doing so, T2A installs a structured action prior in which language selects the region of action space, the embodiment prompt specifies the platform-specific motor parameterisation, and the flow-matching dynamics govern the generative process before any image is introduced.

Stage II: Continued pretraining (CPT). With the decoder warm-started, CPT focuses its multimodal capacity on exactly the problem that T2A could not address: grounding actions in visual observations and adapting the backbone to embodied perception. We unfreeze both modules and train on the heterogeneous mixture of Section 3.2, which deliberately combines simulation and real-robot trajectories so that the resulting checkpoint has seen both domains. This property is important for the post-training stages described below: SFT and RL can specialise toward either domain.

Stage III: Supervised fine-tuning (SFT). While CPT provides broad cross-embodiment and cross-task coverage, downstream deployment benefits from aligning the model with curated, high-quality demonstrations drawn from the target task distribution. We branch SFT into two parallel tracks from the CPT checkpoint. The first track performs multi-task SFT that jointly fine-tunes the model on heterogeneous tasks, including visual question answering, spatial grounding, manipulation, and navigation, under embodiment-balanced and task-balanced sampling. The second track fine-tunes from the CPT checkpoint on in-house teleoperation data for real-world robot deployment, testing whether CPT’s cross-domain priors transfer to physical hardware.

Stage IV: Reinforcement learning (RL). SFT optimises a likelihood over demonstrations; what ultimately matters is closed-loop task success, a property of executed trajectories that no imitation objective can directly optimise. Starting from the multi-task SFT checkpoint, RL fine-tunes the policy with sparse binary success rewards collected exclusively in a single simulation environment (SimplerEnv), producing the final model Qwen-VLA-Instruct. This deliberately narrow RL setup tests whether task-success improvements obtained in one simulated environment can transfer to other out-of-distribution environments.

## 3.2 Pretraining Data

The quality and diversity of the pretraining corpus directly determine how well the cognitive backbone and motor decoder co-adapt across embodiments and task families. We construct a large and heterogeneous pretraining mixture to equip the model with broad embodied perception, spatial reasoning, and action generation capabilities. The mixture spans five data families: robot manipulation trajectories, human egocentric demonstrations, synthetic simulation data, navigation and trajectory-centric data, and auxiliary vision-language data. Table 1 summarizes the composition and sampling weights of each source.

Table 1: Pretraining data mixture composition.
<table><tr><td>Data Source</td><td>Proportion (%)</td></tr><tr><td>Robot Manipulation Trajectories</td><td>74.2</td></tr><tr><td>Human Egocentric Trajectories</td><td>6.0</td></tr><tr><td>Navigation Trajectories</td><td>7.5</td></tr><tr><td> Synthetic Simulation Trajectories (ours)</td><td>3.7</td></tr><tr><td>General Vision-Language Data</td><td>3.4</td></tr><tr><td>Spatial Grounding (2D)</td><td>2.5</td></tr><tr><td> Autonomous Driving VQA</td><td>2.4</td></tr><tr><td>Fine-Grained Embodied Action Caption</td><td>0.2</td></tr><tr><td>Total</td><td>100.0</td></tr></table>

## 3.2.1 Robotics Manipulation Trajectories

Real and simulated robot manipulation trajectories form the core of our pretraining corpus, accounting for approximately 74.2% of the total pretraining mixture. The data spans tabletop manipulation, mobile manipulation, bimanual tasks, and dexterous hand control across a diverse set of robot embodiments.

Public datasets. To improve data scale, diversity, and cross-embodiment generalization, we train on a broad mixture of publicly available real-robot datasets, including RobotSet (Bharadhwaj et al., 2023), Galaxea (Jiang et al., 2025), AgiBot World (AgiBot-World-Contributors et al., 2025), RoboCOIN (Wu et al., 2025b), RoboMIND V1/V2 (Wu et al., 2025a; Hou et al., 2025), RDT-1B (Liu et al., 2025a), DROID (Khazatsky et al., 2024), BridgeData V2 (Walke et al., 2024), RH20T (Fang et al., 2023), RT-1 (Brohan et al., 2022), and BC-Z (Jang et al., 2022). These datasets cover diverse robotic settings, including tabletop manipulation, mobile manipulation, bimanual manipulation, dexterous hand control, and in-the-wild task execution, amounting to over 10,000 hours of interaction data across heterogeneous embodiments and task scenarios. This large-scale mixture provides complementary supervision over robot morphologies, camera viewpoints, object categories, backgrounds, language instructions, and action distributions, thereby reducing overfitting to any single embodiment or environment and improving robustness under distribution shifts.

In addition to real-robot data, we include simulation-based manipulation trajectories from InternData-A1 (Tian et al., 2025) and GR00T-X-Embodiment-Sim (NVIDIA et al., 2025), which are generated through motion planning in diverse virtual environments. These simulated trajectories augment scene and object diversity, especially for long-tail object configurations and task layouts. Before training, all real and simulated datasets are curated and converted into a unified observation-action format while preserving their original task semantics, enabling scalable training over heterogeneous robot experience.

Proprietary datasets. We supplement with over 1,000 hours of in-house collected real-robot trajectories covering a variety of manipulation tasks and robot platforms, together contributing approximately 20% of the total pretraining mixture. In addition, we generate over 8M synthetic simulation trajectories using our scalable simulation pipeline illustrated in Section 3.2.3, contributing a further 3.7% of the mixture and substantially expanding the diversity of scenes, objects, and task configurations beyond what real-robot teleoperation alone can cover.

Embodiment-aware prompt conditioning. As described in Section 2.3, each training example is prepended with a robot-specific prompt specifying the current platform, arm configuration, and control convention. Table 2 lists the representative embodiments in our pretraining corpus with their arm configurations and action types.

Table 2: Representative robot embodiments in the pretraining corpus. “EEF” = end-effector pose; “Joint” = joint angles; “∆” = delta (relative) commands; “Abs” = absolute commands; $^ { \prime \prime } { \bf G } ^ { \prime \prime } = \mathrm { g r i p p e r } ; \ ^ { \prime \prime } \bar { \bf D } \bar { \bf H } ^ { \prime \prime } =$ dexterous hand.
<table><tr><td>Robot</td><td>Arms</td><td>Action type</td></tr><tr><td>WidowX Google Robot Franka Panda</td><td>Single Single Single /Dual</td><td>△EEF +G △EEF+G △EEF +G; Abs Joint + G</td></tr><tr><td>ARX5 Fourier GR-1</td><td>Dual Dual</td><td>△EEF+ G △EEF+G</td></tr><tr><td>Mobile ALOHA</td><td>Dual</td><td>△EEF +G;Abs Joint + G</td></tr><tr><td>AgiBot A2-D Galaxea R1</td><td>Dual</td><td>Abs Joint + G; Abs Joint + DH</td></tr><tr><td></td><td>Dual</td><td>Abs Joint + G</td></tr><tr><td>AIRBOTMMK2 TienKung Real Human</td><td>Dual Dual</td><td>Abs Joint + DH Abs Joint + G; Abs Joint + DH</td></tr></table>

Action representation. Different datasets adopt different action conventions: some provide absolute end-effector poses in Cartesian space (Xie et al., 2026), others provide delta end-effector commands, and others use absolute or relative joint-space control signals. We preserve each dataset’s original action format rather than converting to a shared representation, relying on embodiment-aware prompt conditioning to inform the model of the current control convention.

All action dimensions are normalized per dataset using quantile statistics. For each action dimension d in dataset $k ,$ we compute the 1st and 99th percentiles $q _ { 0 1 } ^ { k }$ and $q _ { 9 9 } ^ { k }$ over all trajectories and apply a linear mapping:

$$
\tilde { a } _ { d } = 2 \cdot \frac { a _ { d } - q _ { 0 1 } ^ { k } } { q _ { 9 9 } ^ { k } - q _ { 0 1 } ^ { k } } - 1 ,\tag{5}
$$

clipping the result to [−1, 1]. This per-dataset quantile normalization removes scale differences across embodiments and action spaces while preserving relative motion structure within each source.

Language instructions. Language instructions are sourced from a combination of original dataset annotations and model-generated captions provided alongside the data. All instructions undergo a quality filtering and consistency checking pipeline: trajectories whose language annotations are inconsistent with the observed motion are discarded, ensuring reliable language–action alignment across diverse sources. Beyond action-labelled data, we incorporate fine-grained embodied video–caption pairs (described in Section 3.2.5) as auxiliary vision-language supervision. This dense language signal bridges the gap between coarse task labels and rich semantic descriptions of robot behaviour, enabling the model to generalise to a broader range of natural-language instructions at inference time.

Camera view representation. Many robot datasets provide observations from multiple camera viewpoints simultaneously, e.g. a head-mounted ego-centric camera alongside one or two wrist-mounted cameras. To make the view provenance explicit to the model, we wrap each image in the token stream with a pair of view-specific boundary tokens:

$$
< | t a g _ { - } s t a r t | > \langle \mathrm { i m a g e } \rangle < | t a g _ { - } \mathsf { e n d } | >
$$

where tag identifies the camera source, e.g. ego, cam\_left\_wrist, or cam\_right\_wrist. The explicit view labels allow the VLM backbone to form view-aware representations and let the action expert attend selectively to each viewpoint when producing actions, without requiring any architectural changes or additional input channels.

Data cleaning. We apply a multi-stage filtering pipeline to remove trajectories with corrupted or missing frames, near-zero-variance action sequences indicative of static recordings, and episodes with anomalous lengths. For datasets without explicit action labels, pseudo-actions are recovered via finite differences on proprioceptive state sequences.

## 3.2.2 Egocentric Human Data

Compared with teleoperated robot trajectories, egocentric human demonstrations offer a more abundant and scalable source of real-world manipulation experience. Humans interact with diverse objects in open-world environments every day, naturally producing dexterous manipulation behaviours that span a much broader range of scenes, objects, and task semantics than robot teleoperation. Recent studies demonstrate that training on large-scale egocentric human videos endows vision-language-action models with richer manipulation priors and improves generalization to downstream robot tasks (Kareer et al., 2025; Luo et al., 2025; Li et al., 2026b; Luo et al., 2026; Zheng et al., 2026; Hu et al., 2026). Motivated by these findings, we incorporate a diverse collection of egocentric human manipulation datasets (6.0% of the total pretraining mixture) to provide the model with broad manipulation priors that complement the robot trajectory data.

Data sources. Our egocentric corpus is drawn from four publicly available datasets, each contributing complementary coverage in terms of task diversity, scene variety, and annotation granularity. (1) We adopt the Ego4D (Grauman et al., 2022) and EPIC-KITCHENS (Damen et al., 2022) subsets processed by VITRA (Li et al., 2026b), which applies an automated pipeline to segment egocentric human videos into atomic manipulation trajectories and generate fine-grained language annotations, together with framewise 3D hand and camera motion trajectories. (2) EgoDex (Hoque et al., 2026) is a large-scale dexterous manipulation dataset captured with Apple Vision Pro, comprising 829 hours of egocentric video with paired 3D hand and finger tracking across 194 diverse tabletop tasks. (3) EgoVerse (Punamiya et al., 2026) is a collaborative egocentric demonstration platform spanning over 1,300 hours across 1,965 tasks and 240 scenes, with standardized formats and manipulation-relevant annotations. (4) Xperience (Ropedia, 2026) is a large-scale egocentric multimodal dataset providing synchronized first-person recordings with depth, hand and body motion capture, and hierarchical language annotations.

Action representation. The model receives an egocentric image at an arbitrary time step together with a language instruction and predicts an action chunk comprising future bimanual wrist motion and hand articulation over a fixed prediction horizon. For each hand, the wrist motion is represented as the SE(3) transformation of the wrist coordinate frame at a future time step relative to the initial frame. During training, this relative action is represented with a translation vector and an axis-angle rotation, yielding 6 wrist action dimensions per hand. To represent hand articulation compactly, we perform principal component analysis (PCA) on the 45-dimensional axis-angle joint pose of the hand across all human datasets and retain the weights of the first 10 principal components. These low-dimensional coefficients, dubbed eigengrasps (Ciocarlie et al., 2007; Yuan et al., 2025), capture the dominant modes of human hand-pose variation while discarding per-joint redundancy. The model predicts relative wrist actions and 10 eigengrasp coefficients per hand, yielding a total of 32 action dimensions per time step for egocentric human data.

## 3.2.3 Synthetic Simulation Data

To improve the coverage, controllability, and robustness of embodied supervision, we construct a large-scale synthetic simulation data pipeline with two complementary components: (1) vision-languageaction data, where the model predicts actions from both task instructions and image observations, and (2) language-action data, where the model predicts actions from language alone. These two sources of supervision play different but synergistic roles. The text-only component encourages the model to acquire high-level task abstractions and language-action regularities without depending on visual appearance, while the vision-conditioned component grounds those abstractions in realistic perceptual observations, scene variations, and embodied interaction dynamics. We developed our pipeline based on IsaacLab (Mittal et al., 2025) for our simulation environment and cuRobo (Sundaralingam et al., 2023) for collision-avoidance motion planning.

Vision-language-action data. For the vision-conditioned synthetic data, we use an internal early version of ROBOINF (XLANG Lab, 2026) to generate standard VLA-style supervision, where the model receives a language instruction and image observations as input and predicts robot actions as output. ROBOINF constructs manipulation scenes, generates scene-conditioned tasks, synthesizes executable success checks, produces robot motion programs through simulator feedback, and rolls out successful trajectories under domain randomization. Since the full technical details of the pipeline are provided in our ROBOINF blog post, we only summarize the dataset composition and diversity here.

In this release, we focus on the random-placement scene generation setting for simplicity. We construct 20 tabletop scenes, each augmented with 10 different object initial-pose configurations, resulting in 200 base scene configurations. These scenes contain diverse everyday objects arranged in physically valid tabletop layouts, providing varied visual and spatial contexts for manipulation.

On top of these scenes, we generate 450 manipulation tasks spanning both short- and long-horizon behaviors. Short-horizon tasks typically involve a small number of primitive manipulation steps, such as “Place the two green staplers side by side”. Long-horizon tasks require multiple sequential interactions or more compositional reasoning, such as “Group the drinks together and leave the cleaning sponge by itself”, where the robot needs to manipulate several objects in sequence. As illustrated in Figure 3, this task mixture exposes the model to both atomic manipulation skills and higher-level instruction-following patterns over extended temporal horizons.

For each task, we generate 300 successful trajectories with different environment and execution augmentations. During rollout, we randomize visual, geometric, and control-related factors, including lighting, camera poses, backgrounds, table textures, robot initial states, object initial poses, and controller dynamics such as stiffness and damping. For visual randomization, we sample from a large pool of approximately 3K backgrounds and 1K table textures. For other factors, such as camera pose, robot state, lighting, and controller parameters, we define reasonable ranges manually and sample variations within those ranges. These augmentations are designed to improve robustness to distribution shifts in appearance, viewpoint, layout, and low-level execution dynamics.

A useful property of this generated data is that the trajectories are produced from motion-planning programs that are naturally decomposed into intermediate stages. We therefore additionally segment each full trajectory into several subtask trajectories, as shown in Figure 3. This provides supervision at multiple temporal granularities: full trajectories teach the model to follow high-level instructions over longer horizons, while subtask trajectories expose the model to shorter atomic behaviors and intermediate goals. We expect this mixture to help the model better connect high-level language instructions with the lower-level action segments needed to complete them.

Overall, the vision-conditioned synthetic data contains 359,848 full successful trajectories including subtask segments. The resulting dataset provides a large-scale source of diverse, controllable, and automatically verified VLA supervision, complementing real-world demonstrations and improving the

![](images/441886844c0c9d03ee7583323ad32701ef7fe708e5f4a1ad4b727fa4c2f836e4.jpg)

Figure 3: Examples of data generated through ROBOINF. The top row shows a short-horizon task, “Place the two green staplers side by side,” which consists of a compact sequence of reaching, grasping, transporting, and placing. The bottom row shows a long-horizon task, “Group the drinks together and leave the cleaning sponge by itself,” which requires multiple object manipulations and can be decomposed into subtask segments such as picking and placing each drink.

model’s generalization across task types, visual conditions, and execution variations.

Language-action data. We additionally construct a text-only action dataset as a complementary source of supervision. The goal of this component is to expose the model to a broad space of manipulation intents and action patterns before introducing visual observations, thereby encouraging generalization in the language-action joint space. Concretely, we brainstorm a diverse set of atomic and compositional manipulation tasks that can be executed by the robot in simulation, and generate corresponding trajectories using robot-state-centric pipelines without image inputs. This data primarily serves as a form of semantic and behavioral pretraining, helping the model capture common task structures such as moving, grouping, separating, aligning, rotating, stacking, and reordering objects from language instructions alone.

Specifically, we define six task template families that span the core single-arm manipulation primitives: pick-and-place, linear pushing, linear pulling, rotation with repositioning, rotation toward a viewpoint direction, and positional swapping of two objects. Each template encodes a canonical motion structure while randomizing the spatial parameters and natural language description across instances. Instructions are procedurally composed from template-specific vocabularies of verbs, prepositions, object names, and spatial references to produce varied descriptions of the same underlying behavior (e.g., “Place the apple into the container,” “Move the sponge toward the target location”). This design provides precise control over task coverage and difficulty while remaining simple enough to scale to millions of trajectories.

To ensure broad coverage of embodiment-specific kinematics and workspace geometry, we instantiate all six task templates across six single-arm robot configurations (Franka Panda, UR10e, UR5e, Kinova Gen3, TM12, and xArm7) spanning different degrees of freedom, arm morphologies, and reachable workspaces. For each robot–task pair, we generate approximately 200k trajectories, yielding roughly 7.2M trajectories and over 14,000 hours of simulated robot trajectory data in total. Trajectory synthesis proceeds without physics simulation or scene rendering: for each instance, we sample end-effector goal poses within the robot’s reachable workspace bounds, randomizing positions, orientations, and inter-pose distances, and compute collision-free joint-space paths using a GPU-accelerated batch motion planner. The generation pipeline is parallelized across multiple GPUs, with each worker independently sampling target configurations, planning motions, and writing successful trajectories. Only kinematically valid trajectories for which the planner produces feasible, collision-free solutions across all required motion segments are retained.

Each trajectory records joint positions, joint velocities, end-effector poses (position and orientation), and gripper states at 50 Hz, providing a rich and flexible supervision signal that can support a variety of action representations and control modes during training.

During training, this dataset serves as the primary corpus for Stage I T2A pretraining (Section 3.1), where the DiT action decoder is initialized on language–action correspondences before visual grounding is introduced. By exposing the decoder to a large and diverse set of language–action pairs without visual conditioning, this component pre-empts visual shortcuts and establishes a strong action prior that transfers to and complements the downstream vision-conditioned training stages.

## 3.2.4 Navigation data.

We incorporate Navigation data (7.5%), which stands out for its long-horizon trajectories coupled with rich visual information. The mobile robot is assumed to have 3 degrees of freedom: translation in the plane and rotation about the vertical axis (heading angle). Navigation data comprises navigation videos (sampled at 2 FPS), instructions, and trajectory information, jointly equipping the model with mobile capabilities such as instruction following, object searching, and target tracking.

Instruction following. Instruction-following navigation data (4.3%) covers a diverse range of navigation horizons and instruction granularities. In these episodes, the mobile robot must execute commands that intermix motion primitives (e.g., move forward, turn right/left, stop) with textual landmarks. This data encourages the model to align navigation history with the given instruction and predict the subsequent trajectory.

Object searching. Object-searching data (2.3%) derives from navigation histories where the robot seeks a text-described object within large-scale indoor environments. The robot is challenged to remember previously navigated areas while exploring efficiently to locate the target. Such episodes foster the model’s long-horizon exploration and identification capabilities.

Target tracking. Target-tracking data (1.0%) consists of episodes in which the robot follows a moving target (e.g., a described person) through indoor spaces, maintaining a safe following distance and keeping the target in view. The resulting trajectories blend reactive motion adjustments with longer-horizon path decisions triggered by turns or occlusions.

## 3.2.5 Vision-language data.

We further incorporate auxiliary vision-language supervision (8.5% combined) to strengthen semantic grounding, fine-grained instruction following, and general visual reasoning.

Fine-grained embodied action caption. Most existing robot datasets provide only coarse task-level labels (e.g., “pick up the ceramic bowl”), which are insufficient for precise action prediction: the same label may correspond to vastly different execution strategies (e.g., grasping from the left vs. the right, rotating clockwise vs. counter-clockwise), creating ambiguity during policy learning. Drawing inspiration from fine-grained text control in video generation, we construct dense action descriptions that tightly couple language to each individual episode. To build a diverse yet manageable annotation set, we draw episodes from major open-source manipulation datasets across all task categories and apply a clustering-based selection strategy: episodes within each task are grouped by their action information, and representative ones are sampled to ensure broad coverage. Each selected episode is annotated along 13 dimensions: action primitive, actor identity, object recognition and disambiguation, contact region, source and target location, trajectory and orientation, gripper state, and body motion. We use a two-stage pipeline for annotation. In the first stage, a vision-language model (Qwen3.6-plus) watches the video and extracts a coarse action sequence along with the manipulated object. In the second stage, the video is densely sampled into frames to preserve subtle motion details; the model then re-watches these frames conditioned on the first-stage output and produces fine-grained, step-by-step descriptions enriched with contact points, spatial relationships, and motion trajectories. For multi-view datasets, an additional refinement pass incorporates wrist- or side-camera footage to correct details only visible from close-up or alternative perspectives. All generated captions are finally reviewed and corrected by human annotators. This pipeline yields approximately 48,000 fine-grained video–caption pairs (0.2% of the pretraining mixture). Table 3 contrasts a coarse label with our fine-grained caption for the same episode.

Table 3: Coarse task label vs. fine-grained action caption for the same episode.
<table><tr><td>Coarse label</td><td>&quot;Pick up, rotate,and place the ceramic bowl.&quot;</td></tr><tr><td>Fine-grained</td><td> Step 1: Pick up the ceramic bowl from the right far edge.</td></tr><tr><td></td><td>Step 2: Rotate the bowl clockwise for two full circles.</td></tr><tr><td></td><td>Step 3: Place the bowl at the center of the table.</td></tr></table>

Building on the annotated data, we further fine-tune a dedicated VLM for embodied action captioning. We also design a companion evaluation benchmark consisting of QA pairs derived from the ground-truth annotations together with a scoring rubric for assessing caption quality.

Autonomous driving VQA. We incorporate autonomous-driving visual question-answering data (2.4%) as a complementary source of visually grounded, decision-oriented supervision. This mixture focuses on four transferable capabilities: (1) Temporal scene understanding: driving scenes evolve under ego-motion, dynamic agents, and changing hazards. We include LingoQA (Marcu et al., 2024), DriveAction (Hao et al., 2025), and MMAU (Fang et al., 2024) to strengthen event understanding and future-state prediction. (2) Surround-view spatial reasoning: multi-camera observations provide wide-field geometric context. We include Impromptu-VLA (Chi et al., 2025), nuScenes-QA (Qian et al., 2024), nuScenes-MQA (Inoue et al., 2024), MapLM (Cao et al., 2024), and WaymoQA (Yu et al., 2025b) to improve viewpoint-robust localization and spatial-relation understanding across diverse vehicles, sensors, and road layouts. (3) Language-grounded localization: embodied agents must bind language to objects, regions, and spatial relations before acting. We include CODA-LM (Chen et al., 2025a), Talk2Car (Deruyttere et al., 2019), DrivingVQA Corbiere et al. (2025), DriveLM (Sima et al., 2024), W3DA (Zhou et al., 2025), and GRAID (Elmaaroufi et al., 2025) for object-level grounding and region-aware question answering. (4) Planning-aware reasoning: safe decision making requires linking perception to route, lane, and trajectory constraints. We include Bench2Drive-VL (Jia et al., 2026), DriveGPT4 (Xu et al., 2024), OmniDrive (Wang et al., 2025b), Senna (Jiang et al., 2024), NAVSIM-RecogDrive (Li et al., 2026c), and NAVSIM-Traj to provide supervision for maneuver selection and future trajectory prediction. Prior to pretraining, all data sources undergo standardization into a unified conversational format. We convert multiple-choice annotations into freeform QA supervision and normalize bounding boxes to a 1000-scale coordinate system. Additionally, multi-frame samples receive frame tags to maintain temporal order, while surround-view samples are assigned view tags to explicitly denote camera identity. Collectively, these structured sources offer transferable supervision for object-centric grounding, temporal state estimation, intent prediction, and goal-conditioned decision-making. This comprehensively complements the manipulation and navigation data utilized in our embodied pretraining.

Spatial grounding. We include 2D bounding box grounding data (2.5%) to reinforce object-level spatial understanding. Accurate spatial grounding is a prerequisite for language-conditioned manipulation, where the model must localize task-relevant objects from instruction-specified descriptions.

General vision-language data. To preserve robust visual perception and language grounding during embodied VLA pretraining, we supplement the primary action-centric dataset with a curated mixture of general vision-language data (3.4% of total tokens). This auxiliary mixture explicitly comprises: (1) captioning data for image–text alignment; (2) knowledge-oriented VQA and visual reasoning for world knowledge retention; (3) OCR and text-grounding samples for fine-grained textual perception; (4) interleaved instruction-following data for stable cross-modal alignment; and (5) common grounding tasks (e.g., referring expression, spatial relation prediction) for object-level language grounding. Critically, to better support embodied reasoning, we strategically up-weight video-centric, spatial-relation, and 3D-aware supervision within this mixture. Collectively, these general vision-language objectives mitigate catastrophic forgetting, retain foundational world-knowledge priors, and enhance robustness in languageconditioned embodied perception.

## 4 Post-Training

The large-scale pretraining of T2A and CPT produces Qwen-VLA-Base, a generalist vision-languageaction model that exhibits broad cross-task and cross-embodiment generalization. Although this broad coverage equips the model with versatile knowledge, it does not yet yield the precision needed for

reliable closed-loop control on specific downstream tasks. To bridge this gap, we introduce a two-stage post-training procedure that specializes base model for accurate task execution:

(i) A multi-task supervised fine-tuning (SFT) stage that jointly fine-tunes both the VLM backbone and the action expert on heterogeneous tasks, including visual question answering, spatial grounding, manipulation, and navigation, under embodiment-balanced and task-balanced sampling.

(ii) A reinforcement learning (RL) stage, initialized from the SFT checkpoint, that further refines the policy by optimizing directly against task-success-driven rewards obtained from on-policy rollouts in simulation, producing the final model Qwen-VLA-Instruct.

Across both stages, we use cosine-decayed learning rates with separate group-wise schedules for the vision-language backbone and the action decoder, along with gradient clipping consistent with pretraining. The specific data and methods for each stage are detailed in the subsequent sections.

## 4.1 Multi-Task Supervised Fine-Tuning

Data. The SFT data mixture comprises three categories: general vision-language samples, visionlanguage navigation episodes, and robotic manipulation demonstrations. First, we curate a set of vision-language samples—covering visual question answering, spatial grounding, and action captioning— to maintain the backbone’s visual understanding throughout fine-tuning. Second, we collect robotic manipulation demonstrations from simulation, including trajectories across a diverse set of platforms spanning single-arm, dual-arm, and humanoid robots, all using a prediction horizon of 16 action steps per chunk (Community, 2026). Following the same embodiment-aware prompt design used in pretraining (Section 2.3), every training sample is prefixed with a textual prompt that specifies the robot platform, arm configuration, control frequency, and prediction horizon. Third, we collect vision-language navigation episodes from continuous control environments, covering varied indoor scenes and instruction styles. We retain only successfully completed episodes, with a prediction horizon of 8 waypoints per chunk.

Objective. The SFT stage jointly optimizes two losses: next-token prediction on vision-language tokens and flow matching on action tokens. The loss weights are set to 0.1 for vision-language next-token prediction and 1.0 for both manipulation and navigation action prediction, ensuring that the model retains its language and visual understanding while focusing gradient capacity on action generation.

## 4.2 Reinforcement Learning

Likelihood-based SFT optimizes a surrogate that measures how well the policy imitates demonstrations, but the quantity that ultimately matters in embodied control is closed-loop task success: whether the executed trajectory achieves the goal. These two objectives can diverge: a policy may assign high likelihood to plausible actions yet fail when its own outputs shift the state distribution away from the demonstration support. The RL stage closes this gap by directly optimizing a task-success-driven reward on trajectories rolled out by the policy itself, producing the final model. We implement this stage using the RLinf framework (Yu et al., 2025a).

Policy optimization objective. We adopt Proximal Policy Optimization (PPO) (Schulman et al., 2017) with Generalized Advantage Estimation (GAE) (Schulman et al., 2015) as the RL algorithm. Let $\pi _ { \theta }$ denote the current policy and $\pi _ { \theta _ { \mathrm { o l d } } }$ the policy that generated the rollout batch. The clipped surrogate objective is

$$
\begin{array} { r } { \mathcal { L } ^ { \mathrm { a c t o r } } ( \theta ) = - \mathbb { E } _ { t } \Big [ \operatorname* { m i n } \Big ( r _ { t } ( \theta ) \hat { A } _ { t } , \ \mathrm { c l i p } \big ( r _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon \big ) \hat { A } _ { t } \Big ) \Big ] , } \end{array}\tag{6}
$$

where $r _ { t } ( \theta ) = \pi _ { \theta } ( a _ { t } \mid s _ { t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { t } \mid s _ { t } )$ is the importance ratio, with $s _ { t } = \left( o _ { t } , x , e \right)$ the current state (visual observation, task instruction, and embodiment prompt following Section 2.3) and $a _ { t }$ the predicted action chunk of horizon $H , { \hat { A } } _ { t }$ is the GAE advantage estimate with discount $\gamma { = } 0 . 9 9$ and trace-decay $\lambda { = } 0 . 9 5$ and $\epsilon { = } 0 . 2$ is the clipping threshold applied symmetrically. The total loss combines the policy surrogate with a value-function regression term:

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathcal { L } ^ { \mathrm { a c t o r } } ( \boldsymbol { \theta } ) + c _ { v } \mathcal { L } ^ { \mathrm { v a l u e } } ( \boldsymbol { \theta } ) ,\tag{7}
$$

where $\mathcal { L } ^ { \mathrm { v a l u e } }$ is a loss on value predictions and $c _ { v } = 1$ . We perform four optimization epochs per rollout batch.

Value estimation. Rather than training a separate critic network, we attach a lightweight value head directly to the vision-language backbone. The value head mean-pools all VLM hidden states and maps them to a scalar value estimate through a linear projection. We apply stop-gradient on the VLM hidden states before they enter the value head, so that value-function gradients do not propagate back through the pretrained backbone. The value head is trained with a clipped MSE loss and a separate learning rate $( 1 0 ^ { - 4 }$ , approximately 20× the actor learning rate of $5 \times 1 0 ^ { - 6 } )$ , allowing it to converge rapidly while the policy updates remain conservative.

Log-probability estimation under flow matching. A key challenge in applying PPO to our flowmatching action decoder is computing the log-probability log $\pi _ { \theta } { \big ( } a _ { t } { \big ) } \ s _ { t } )$ required for the importance ratio $r _ { t } (  { \breve { \theta } } )$ . Unlike autoregressive token-based policies where log-probabilities are directly available from the softmax output, flow-matching models define an implicit density through a learned velocity field and iterative denoising. We convert the deterministic probability-flow ODE into a corresponding SDE by injecting controlled noise at each Euler denoising step (Song et al., 2021), so that each transition becomes an explicit Gaussian whose log-probability can be computed analytically without numerical ODE integration. During rollout we store the intermediate denoising states; at the PPO update we re-evaluate the velocity field under the current parameters and recompute the Gaussian log-probability, yielding the importance ratio at negligible additional cost. By default we randomly select a single denoising step per rollout for the log-probability estimate, requiring only one extra DiT forward pass during recomputation. Both log-probabilities and advantages are computed at the action-chunk level: each chunk of $H { = } \dot { 1 } 6$ action steps receives a single scalar reward and a single advantage estimate, matching the temporal granularity of the flow-matching decoder’s output.

Reward design. We use a sparse binary reward from the simulator: R=1 if the task goal is achieved at episode end and R=0 otherwise. Credit assignment across action chunks is handled by GAE (eq. (6)), which propagates the episode-level signal back through the value baseline. No learned reward model is used; all signals come from the simulator’s ground-truth task-completion semantics.

Rollout infrastructure. On-policy rollouts are collected in simulation environments that mirror the downstream evaluation benchmarks. We adopt a decoupled client-server architecture in which a remote benchmark server hosts the simulation environments and the training process communicates with it via a network interface. This separation allows the simulation workload to scale independently of the GPU-intensive policy optimization. We instantiate N=128 parallel environment instances distributed across multiple manipulation task suites with non-uniform allocation proportional to task difficulty. Each training iteration collects 8 rollout epochs of 128 environment steps each, yielding $N \times \lfloor 1 2 8 / H \rfloor \times 8 =$ $1 2 8 \times \tilde { 8 } \times 8 = 8 , 1 9 2$ transition chunks per iteration (where $H { = } 1 \dot { 6 }$ is the action chunk length). During rollout, actions are sampled from the policy with temperature $\tau { = } 1 . 0 ;$ at evaluation time, we reduce the temperature to $\tau { = } 0 . 6$ to sharpen the action distribution. The embodiment prompts used during rollout are identical to those in SFT, ensuring that the RL stage does not reintroduce a distribution shift in the prompt conditioning.

Out-of-domain generalization. Because the embodiment prompt is the sole platform-specific interface, the RL-refined policy transfers to out-of-domain environments, tasks, and embodiments without an additional adaptation head or domain-specific fine-tuning. At deployment, we simply replace the embodiment prompt with the textual description of the physical robot platform, including its arm configuration, control frequency, and action space, while keeping the VLM backbone and DiT action decoder unchanged. This zero-shot generalization is enabled by two properties of our training recipe: (1) the CPT stage has already exposed the model to both simulated and real-world visual distributions (Section 3.1), so the backbone’s visual representations are not specific to the simulation renderer; and (2) the RL stage optimizes task-success metrics that are invariant to the visual domain, encouraging the policy to rely on semantically meaningful visual features rather than renderer-specific textures. We quantify the cumulative effect of each post-training stage in Section 5.2.3.

## 5 Experiments

We conduct extensive experiments to evaluate the performance of Qwen-VLA in both simulation and real-world settings, covering two core embodied AI domains: robotic manipulation and visual navigation. We evaluate two model variants throughout: Qwen-VLA-Base, trained with large-scale pretraining on diverse embodied data, and Qwen-VLA-Instruct, further fine-tuned with instruction-following data collected across diverse simulated environments and tasks, capable of executing fine-grained manipulation and navigation with greater precision and reliability in one unified model.

## 5.1 Main Results

## 5.1.1 Manipulation Results in Simulation

To comprehensively evaluate the manipulation capabilities of Qwen-VLA, we benchmark on four simulation environments covering single-arm and dual-arm settings: LIBERO (Liu et al., 2023), Simpler (Li et al., 2024), RoboCasa-GR1 Tabletop Tasks (Nasiriany et al., 2024), and RoboTwin 2.0 (Mu et al., 2024).

• LIBERO is a single-arm tabletop benchmark comprising diverse tasks across four splits: Libero-Spatial, Libero-Object, Libero-Goal, and Libero-Long.

• Simpler-WidowX is a real-to-sim evaluation suite using the WidowX robot arm.

• RoboCasa-GR1 is a bimanual humanoid benchmark with 24 atomic kitchen tasks.

• RoboTwin 2.0 is a dual-arm benchmark with 50 bimanual tasks divided into Easy and Hard difficulty tiers.

Together, these benchmarks test a wide range of manipulation skills from tabletop pick-and-place to long-horizon bimanual kitchen tasks. For all benchmarks, we set the action chunk length H = 16 and report the average success rate following the standard evaluation protocol in StarVLA (Community, 2026).

Table 4: Robot manipulation results across benchmarks: specialists vs. a single generalist. Specialist models are fine-tuned separately on each benchmark, whereas our generalist (Qwen-VLA) is trained once on all embodiments jointly and evaluated across all platforms without per-benchmark adaptation. Bold: best; underline: second best.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="5">Benchmark</td></tr><tr><td>LIBERO</td><td>RoboCasa-GR1</td><td>Simpler-WidowX</td><td>RoboTwin-Easy</td><td>RoboTwin-Hard</td></tr><tr><td>π0 (Black et al., 2024)</td><td rowspan="5">Specialist</td><td>94.4</td><td>1</td><td>1</td><td>65.9</td><td>58.4</td></tr><tr><td>StarVLA-OFT (Community, 2026)</td><td>96.6</td><td>48.8</td><td>64.6</td><td>50.4</td><td>1</td></tr><tr><td>GR00T N1.6 (NVIDIA et al.,225)</td><td>97.2</td><td>49.9</td><td>63.2</td><td>47.6</td><td>1</td></tr><tr><td>T0.5 (Black et al.,025)</td><td>97.6</td><td>37.0</td><td>46.9</td><td>82.7</td><td>76.8</td></tr><tr><td>ABot-M0 (Yang et al.,2026)</td><td>98.6</td><td>58.3</td><td>1</td><td>86.0</td><td>85.0</td></tr><tr><td>Being-H0.5 (Luo et al.,026)</td><td rowspan="2"></td><td>97.6</td><td>53.3</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Qwen-VLA-Base</td><td>90.8</td><td>40.4</td><td>64.3</td><td>64.3</td><td>66.4</td></tr><tr><td>Qwen-VLA-Instruct</td><td>Generalist</td><td>97.9</td><td>56.7</td><td>73.7</td><td>86.1</td><td>87.2</td></tr></table>

Table 4 compares our generalist model against state-of-the-art specialist policies. Specialist models are fine-tuned separately on each benchmark and remain embodiment-specific, while our single generalist is trained jointly on diverse, large-scale data spanning multiple scenes and embodiments, enabling deployment across all four platforms through embodiment-aware prompting alone.

A single generalist outperforms most specialists. Despite being an all-in-one model, Qwen-VLA-Instruct surpasses the majority of specialist baselines. On LIBERO, it achieves 97.9%, on par with the best specialists. On RoboCasa-GR1, it reaches 56.7%, outperforming π0.5 (37.0%), GR00T N1.6 (49.9%), and Being-H0.5 (53.3%). On Simpler-WidowX, it attains 73.7%, surpassing specialist baselines including StarVLA-OFT (64.6%). On RoboTwin-Easy/Hard, it scores 86.1%/87.2%, exceeding the previous best specialist ABot-M0 (86.0%/85.0%). These results confirm that joint multi-embodiment training does not sacrifice task-specific performance; in several benchmarks, the generalist even outperforms dedicated models.

Pretraining provides a strong foundation, and instruction tuning yields substantial gains. Qwen-VLA-Base already achieves reasonable performance (e.g., 90.8% on LIBERO, 64.3% on Simpler-WidowX), indicating that large-scale pretraining learns transferable manipulation primitives across embodiments. With instruction tuning, Qwen-VLA-Instruct improves consistently across all benchmarks (+7.1% on LIBERO, +16.3% on RoboCasa-GR1, +9.4% on Simpler-WidowX, +21.8% on RoboTwin-Easy, +20.8% on RoboTwin-Hard), confirming that a modest amount of task-specific alignment is sufficient to convert general pretrained representations into precise, deployable policies.

## 5.1.2 Manipulation Results in the Real World

We collect the real-world dataset on ALOHA, a representative bimanual robot embodiment. The platform consists of two 6-DoF robotic arms equipped with parallel-jaw grippers, forming a bimanual manipulation system. It is instrumented with three RGB cameras, including two wrist-mounted cameras and one first-person-view camera. Unlike Qwen-VLA-Instruct, which is fine-tuned on simulated environments, we further validate Qwen-VLA on real-world tasks by fine-tuning on demonstrations collected through ALOHA across multiple tasks. We prepare two variants sharing the same architecture: Qwen-VLA-alohaw/o pretrain trained from scratch, and Qwen-VLA-alohaw/ pretrain fine-tuned from Qwen-VLA-Base, to evaluate the transferability of large-scale embodied pretraining to real-world settings.

![](images/40bd4e93d96624994989b48387c0e4f09d19cd20c1c4881e32b515b424464d0e.jpg)  
Figure 4: Overview of real-world evaluation tasks on the ALOHA bimanual platform.

We evaluate the model on a diverse set of real-world manipulation tasks, covering both in-domain and out-of-distribution (OOD) scenarios. The in-domain evaluation contains six task categories: pick and place, table cleaning, bowl stacking, bowl picking and object placement, towel folding, fine-grained manipulation. The OOD evaluation further tests generalization to unseen colors, instances, positions, background, language instructions.

In-domain tasks. The in-domain evaluation covers six task categories of increasing complexity:

• Task 1: Pick and Place. The robot picks up basic objects, including a block and a pen, and places them at specified target locations.

• Task 2: Table Cleaning. The robot cleans up objects from the table and places them into a target container.

• Task 3: Bowl Stacking. The robot stacks bowls in the color order specified by the language instruction.

• Task 4: Bowl Pick & Place. The robot picks up a bowl and places target objects into it according to the language instruction.

• Task 5: Towel Folding. The robot folds a towel on the table, testing its ability to manipulate deformable objects.

• Task 6: Fine-grained Manipulation. This category includes several precise manipulation tasks, such as opening a drawer to place an object inside and then closing it, inserting an object into a slot, and stacking two blocks.

OOD tasks. The OOD evaluation includes five settings that assess different types of generalization:

• Color Generalization. This setting evaluates color generalization on Task 1 and Task 4, where the robot needs to handle unseen color specifications, such as picking up a bowl with a specified color (unseen) and placing an object into it, or putting a pen with a specified color (unseen) into the target container.

• Instance Generalization. This setting evaluates instance generalization on Task 1 and Task 2, where the robot encounters novel objects, such as grapes (unseen) and a lemon (unseen) during table cleaning,

or a red block (unseen) during object placement.

• Position Generalization. This setting evaluates position generalization on Task 3 and Task 4, where bowls and target objects appear at spatial configurations (unseen).

• Background Generalization. This setting evaluates background and lighting generalization on Task 2 and Task 3, including lighting conditions (unseen) for bowl stacking and a green background (unseen) for table cleaning.

• Instruction Generalization. This setting evaluates instruction generalization on Task 3 and Task 6, where the robot is required to execute different actions according to the given language instruction. Examples include stacking bowls in a specified order (unseen) or stacking blocks according to color relations, e.g., placing the green block (unseen) on the purple block (unseen).

Table 5: In-domain performance across short-horizon and long-horizon task categories. $\mathrm { { Q w e n – V L A - a l o h a _ { w / o p r e t r a i n } } }$ and Qwen-VLA-aloha $\mathbf { \bar { w } } / \mathbf { \tau } _ { \mathrm { p r e t r a i n } }$ share the same Qwen-VLA model architecture; the former is trained from scratch, and the latter is fine-tuned from Qwen-VLA-Base.
<table><tr><td rowspan="2">Model</td><td colspan="3">Short-horizon Tasks</td><td colspan="3">Long-horizon Tasks</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td></td><td>Pick and PlaceTable Cleaning</td><td>BowlStacking</td><td>Bowl Pick &amp; Place</td><td>Towel Folding</td><td>Fine-grained Manipulation</td></tr><tr><td> $\overline { { \mathrm { G R 0 0 T N 1 . 6 ( N V I D I A e t a l . } , 2 0 2 5 ) } }$ </td><td>30.8</td><td>38.5</td><td>53.8</td><td>19.2</td><td>19.2</td><td>10.3</td><td>28.6</td></tr><tr><td>70.5 (Black et al.,025)</td><td>73.1</td><td>84.6</td><td>88.5</td><td>69.2</td><td>80.8</td><td>33.3</td><td>71.6</td></tr><tr><td>Qwen-VLA-alohaw/o pretrain</td><td>30.8</td><td>53.8</td><td>61.5</td><td>64.1</td><td>50.0</td><td>30.8</td><td>48.5</td></tr><tr><td>Qwen-VLA-alohaw/pretrain</td><td>96.2</td><td>92.3</td><td>98.7</td><td>87.2</td><td>65.4</td><td>61.5</td><td>83.6</td></tr></table>

Results on in-domain tasks. Table 5 summarises in-domain results. $\mathrm { { Q w e n - V L A - a l o h a _ { w } / \ p r e t r a i n } }$ achieves the best average success rate of 83.6%, outperforming strong baselines including GR00T N1.6 (NVIDIA et al., 2025) and $\pi _ { 0 . 5 }$ (Black et al., 2025) on most task categories. The benefit of large-scale pretraining is evident from the comparison with Qwen-VLA-aloha $^ { \mathrm { l } } \mathbf { w } / \mathrm { o } \ p \mathrm { r e t r a i n } \mathrm { : \ }$ fine-tuning from Qwen-VLA-Base raises the average success rate from 48.5% to 83.6%, with particularly large gains on Pick and Place, Table Cleaning, Bowl Stacking, Bowl Pick & Place, and Fine-grained Manipulation. These results confirm that pretraining provides a strong foundation for real-world manipulation that transfers effectively across diverse in-domain tasks.

Results on OOD tasks. Table 6 reports the OOD generalization performance. Qwen-VLA-aloha ${ \bf \nabla } \cdot { \bf w } /$ pretrain consistently achieves the best results across all five generalization settings, including color, instance, position, background, and instruction generalization. Its average OOD success rate reaches $7 6 . 9 \% ,$ substantially outperforming $\pi _ { 0 . 5 }$ (Black et al., 2025) by 35.4 percentage points and Qwen-VLA-aloha ${ \bf { \sigma } } \cdot { \bf { w } } / { \bf { o } }$ pretrain by 40.7 percentage points. The improvement is particularly significant under background and instruction generalization, where Qwen-VLA-alohaw/ pretrain achieves 80.8% and 84.6%, respectively. These results suggest that pretraining not only improves in-domain performance but also provides stronger robustness to unseen visual conditions, object instances, positions, and instruction variations.

Overall analysis. Overall, the results demonstrate that pretraining plays a critical role in real-world manipulation learning. While Qwen-VLA-alohaw/o pretrain and Qwen-VLA-aloha $w / \mathrm { p r e t r a i n }$ share the same architecture, training from scratch leads to much lower performance, especially under OOD settings. This comparison indicates that the performance gain does not simply come from model architecture, but mainly from the pretrained Qwen-VLA-Base model. Moreover, the strong OOD performance of Qwen-VLA-aloh $^ { 1 } \mathrm { w } / \mathrm { p r e t r a i n }$ shows that pretrained vision-language-action representations can effectively support generalization across both visual variations and instruction variations.

## 5.1.3 Navigation Results

We evaluate the navigation capabilities of Qwen-VLA on the vision-and-language navigation in continuous environments (VLN-CE (Krantz et al., 2020; Ku et al., 2020)). Specifically, we test Qwen-VLA Base and Instruct versions on the Val-Unseen split of the R2R and RxR benchmarks, and compare them with the open-source baselines. We use the default setting of VLN-CE and, in particular, implement a sliding-window waypoint action compatible with the predicted trajectory of Qwen-VLA.

As shown in Table 7, Qwen-VLA-Instruct achieves the best performance on most metrics across both R2R and RxR benchmarks. On R2R Val-Unseen, it attains the highest Oracle Success rate (69.0) and Success Rate (57.5), surpassing StreamVLN by 4.8 and 0.6 points respectively, while maintaining competitive SPL. On the more challenging RxR Val-Unseen split, Qwen-VLA-Instruct leads in both SR (59.6) and SPL (47.8), outperforming all baselines by a notable margin. Joint training on VLA and VLN data maintains reasonable performance on both benchmarks.

Table 6: OOD performance across generalization categories. Qwen-VLA-alohaw/o pretrain and Qwen-VLA-alohaw/ pretrain share the same Qwen-VLA model architecture; the former is trained from scratch, and the latter is fine-tuned from pretrained Qwen-VLA-Base.
<table><tr><td>Model</td><td>Color</td><td>Instance</td><td>Position</td><td>Background</td><td>Instruction</td><td>Avg</td></tr><tr><td>GR0OT N1.6 (NVIDIA et al., 2025)</td><td>46.2</td><td>38.5</td><td>3.8</td><td>19.2</td><td>19.2</td><td>25.4</td></tr><tr><td>π0.5 (Black et al.,2025)</td><td>57.7</td><td>61.5</td><td>19.2</td><td>26.9</td><td>42.3</td><td>41.5</td></tr><tr><td>Qwen-VLA-alohaw/o pretrain</td><td>42.3</td><td>30.8</td><td>34.6</td><td>30.8</td><td>42.3</td><td>36.2</td></tr><tr><td>Qwen-VLA-alohaw/pretrain</td><td>88.5</td><td>76.9</td><td>53.8</td><td>80.8</td><td>84.6</td><td>76.9</td></tr></table>

Table 7: Comparison with open-source baselines on VLN-CE. We compare the Qwen-VLA Base and Instruct versions with widely regarded open-source baselines on the Val-Unseen split of the R2R and RxR benchmarks. Bold: best; underline: second best.
<table><tr><td rowspan="2">Method</td><td colspan="4">R2R Val-Unseen</td><td colspan="4">RxR Val-Unseen</td></tr><tr><td>NE↓</td><td>OS↑</td><td>SR个</td><td>SPL个</td><td>NE↓</td><td>SR个</td><td>SPL个</td><td>nDTW↑</td></tr><tr><td>NaVid (Zhang et al., 2024)</td><td>5.7</td><td>49.2</td><td>41.9</td><td>36.5</td><td>5.7</td><td>45.7</td><td>38.2</td><td>1</td></tr><tr><td>Uni-NaVid (Zhang et al., 2025b)</td><td>5.6</td><td>53.3</td><td>47.0</td><td>42.7</td><td>6.2</td><td>48.7</td><td>40.9</td><td>1</td></tr><tr><td>NaVILA (Cheng et al., 2025)</td><td>5.2</td><td>62.5</td><td>54.0</td><td>49.0</td><td>6.8</td><td>49.3</td><td>44.0</td><td>58.8</td></tr><tr><td>StreamVLN (Wei et al., 2025)</td><td>5.0</td><td>64.2</td><td>56.9</td><td>51.9</td><td>6.2</td><td>52.9</td><td>46.0</td><td>61.9</td></tr><tr><td>Qwen-VLA-Base</td><td>5.2</td><td>61.7</td><td>53.8</td><td>49.4</td><td>6.4</td><td>55.1</td><td>45.8</td><td>56.2</td></tr><tr><td>Qwen-VLA-Instruct</td><td>5.1</td><td>69.0</td><td>57.5</td><td>51.2</td><td>5.8</td><td>59.6</td><td>47.8</td><td>57.1</td></tr></table>

## 5.1.4 Out-of-Distribution Evaluation on Static Manipulation

To test whether pre-training priors generalise beyond the fine-tuning distribution, we construct a controlled OOD benchmark where fine-tuning covers only simple pick-and-place yet evaluation demands unseen task types. We develop SimplerEnv-OOD, a suite of 6 out-of-distribution tasks spanning 3 tabletop scenes built upon the SimplerEnv framework with a WidowX robotic arm. All models are fine-tuned solely on the Bridge training split, which contains only simple pick-and-place demonstrations with fixed object pairings. None of the OOD task instructions, spatial relations, or manipulation primitives appear in the training data. The six tasks are:

• MoveAway: Move the spoon farther from the cloth such that the final distance exceeds the initial distance. This tests positional generalization through distance-increasing placement.

• MoveRight: Move the spoon to the right side of the cloth. This tests positional generalization through directional placement with orthogonal-axis constraints.

• PlaceNear: Move the carrot to a position adjacent to but not on top of the plate. This tests positional generalization through proximity-based placement.

• PlaceRight: Place the spoon onto the right half region of the cloth. This tests positional generalization with region-constrained placement, requiring the policy to interpret a sub-region specifier (“right half”) and achieve both correct target contact and spatial precision within the designated area.

• PutFront: Place the carrot in front of the plate. This tests positional generalization along the front/back directional axis.

• StackYellow: Pick up the yellow block and stack it on the green block. The training data only contains stacking green on yellow; this reversed order tests visual generalization by requiring color-based reasoning and generalization to novel object-color bindings not seen during training.

As shown in Table 8, Qwen-VLA-Instruct achieves the highest average success rate of 32.0%, substantially outperforming π0.5 (12.6%) across most tasks. The advantage is strongest on positional generalization: π0.5 fails completely on MoveRight and PlaceNear, while Qwen-VLA-Instruct reaches 33.3% and 39.6%, respectively. Consistent gains also appear on MoveAway (43.8% vs. 26.1%) and PlaceRight (47.9% vs. 32.1%), indicating stronger generalization to unseen spatial instructions. On StackYellow (22.9% vs. 4.2%), where the training data only contains stacking green on yellow but the task requires the reversed order, Qwen-VLA-Instruct demonstrates stronger generalization to novel color-object bindings.

## 5.1.5 Out-of-Distribution Evaluation on Dynamic Manipulation

Deploying robots in complex environments requires mastering dynamic manipulation and necessitating adaptation to independent object motion (Liang et al., 2026). To systematically evaluate this critical capability, the recently introduced DOMINO benchmark (Fang et al., 2026) pioneers a comprehensive evaluation paradigm. It innovatively introduces a hierarchical motion design and a continuous Manipulation Score to capture execution quality under unpredictable constraints. These unique features make DOMINO an exceptional stress test for determining whether a vision-language-action model possesses generalizable spatial-to-kinematic mapping capabilities rather than merely executing memorized static routines. We evaluated Qwen-VLA in a zero-shot setting across all 35 DOMINO suites and report the Success Rate (SR) alongside the Manipulation Score (MS).

Table 8: OOD generalization results on SimplerEnv-OOD. Bold: best; underline: second best.
<table><tr><td rowspan="2">Method</td><td colspan="7">OOD Task</td></tr><tr><td>MoveAway</td><td>MoveRight</td><td>PlaceNear</td><td>PlaceRight</td><td>PutFront</td><td>StackYellow</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>(Black et al., 2025)  $\pi _ { 0 . 5 }$ </td><td>26.1</td><td>0.0</td><td>0.0</td><td>32.1</td><td>13.0</td><td>4.2</td><td>12.6</td></tr><tr><td>Qwen-VLA-Base</td><td>31.3</td><td>31.6</td><td>16.7</td><td>47.1</td><td>63</td><td>18.8</td><td>25.3</td></tr><tr><td>Qwen-VLA-Instruct</td><td>43.8</td><td>33.3</td><td>39.6</td><td>47.9</td><td>4.2</td><td>22.9</td><td>32.0</td></tr></table>

Table 9: Comparison with state-of-the-art methods on DOMINO. Bold: best; underline: second best.
<table><tr><td>Method</td><td>SR (%)↑</td><td>MS↑</td></tr><tr><td>Fine-tuned on dynamic manipulation data</td><td>1.5</td><td>6.1</td></tr><tr><td>OpenVLA (Kim et al., 2024) RDT-1B (Liu et al., 2025b)  $\pi _ { 0 }$  (Black et al., 2024) π0.5 (Black et al., 2025) InternVLA-M1 (Chen et al.,2025b) VLA-Adapter (Wang et al., 2026) π0-FAST (Pertsch et al., 2025)</td><td>5.3 8.2 9.6 5.4 4.4 3.5</td><td>17.7 24.0 26.2 27.6 24.3 20.9</td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025) StarVLA-OFT (Community,2026) PUMA (Fang et al., 2026)</td><td>9.1 10.9 17.2</td><td>24.1 30.5 35.0</td></tr><tr><td>Zero-shot to dynamic manipulation OpenVLA-OFT (Kim et al., 2025)</td><td>6.7</td><td>20.0</td></tr><tr><td> $\pi _ { 0 . 5 }$  (Black et al., 2025) LingBot-VLA w/ depth (Wu et al., 2026) LingBot-VA (Li et al.,2026a)</td><td>7.5 11.8 24.1</td><td>20.4 26.7</td></tr></table>

As shown in Table 9, Qwen-VLA establishes the best overall performance on the DOMINO benchmark. Qwen-VLA-Instruct achieves the highest SR (26.6%) and MS (39.5). In the zero-shot category, our model demonstrates a substantial advantage over standard vision-language-action architectures by surpassing both OpenVLA-OFT (Kim et al., 2025) and $\pi _ { 0 . 5 }$ (Black et al., 2025) by over 19 percentage points in SR. Furthermore, it edges out the representative WAM-style method LingBot-VA (Li et al., 2026a) while demonstrating stronger continuous execution quality under moving-object dynamics. Crucially, Qwen-VLA outperforms the entire suite of baselines explicitly fine-tuned for dynamic manipulation. For example, the advanced method PUMA (Fang et al., 2026) relies on DOMINO-specific fine-tuning and temporal motion inputs to maintain dynamic awareness. Despite lacking these tailored adaptations and relying solely on current-frame observations, Qwen-VLA-Instruct surpasses PUMA by 9.4 percentage points in SR and 4.5 points in MS. Our model achieves these impressive gains at inference using only current-frame observations, without any dynamic-manipulation fine-tuning. This result highlights the advantage of our unified action-and-trajectory pretraining strategy in learning transferable spatial-tokinematic priors.

We attribute this out-of-distribution generalization to the combination of decisive action generation and broad embodied pretraining. The flow-matching action decoder produces coherent action chunks. This mechanism reduces hesitation and helps the policy act precisely within narrow temporal windows. Additionally, large-scale joint pretraining across manipulation, navigation, trajectory prediction, and vision-language data provides transferable priors over visual grounding, spatial reasoning, and continuous control. These robust priors enable Qwen-VLA to generalize effectively to moving-object manipulation despite the complete absence of dynamic training data.

## 5.1.6 Out-of-Distribution Generalization in Real World

![](images/4be97a3704f8b63f2010d37e9deaa73ffcfe33963aee062cf053f690a57aad24.jpg)  
Figure 5: Qualitative out-of-distribution generalization of Qwen-VLA-Base on the ALOHA dual-arm robot. Top-left: color-conditioned grasping of green, blue, red, and yellow balls. Top-right: upper two panels show grasping of novel objects (green broccoli, toy duck); lower two panels show a compositional “clean up the table” task with sequential pick-and-place into a bin (blue umbrella, toy duck, bottled yogurt). Bottom-left: interaction with completely unseen objects (sunglasses, plush doll, toy duck). Bottom-right: atomic action transfer through uncapping a pen and placing the cap, robust to unseen yellow backgrounds.

To qualitatively examine continual pre-training, we evaluate Qwen-VLA-Base on the ALOHA dual-arm robot under four out-of-distribution settings: color grounding, novel object manipulation, unseen object recognition, and background robustness. Representative rollouts are shown in Figure 5.

Color grounding (top-left). We test whether Qwen-VLA-Base can discriminate between visually similar objects that differ only in color. Across four trials the robot is prompted with “grasp {color} ball” for green, blue, red, and yellow, respectively. In all cases the model correctly identifies the target color and produces accurate reaching-and-grasping trajectories, indicating color-conditioned semantic grounding after continual pre-training.

Novel object grasping and compositional clean-up (top-right). We next test whether the model can zero-shot grasp objects never seen in manipulation pretraining data. The top two panels show successful grasping prompted by “grasp {object}” for a green broccoli and a toy duck, both absent from the manipulation training set. The model successfully localizes and grasps the broccoli; for the toy duck, it correctly identifies the object and reaches toward it, though the irregular shape prevents a stable grasp. The bottom two panels further demonstrate a compositional task prompted by “clean up the table,” where the robot sequentially picks up a blue umbrella, a toy duck, and a bottled yogurt and places them into a bin. All three are everyday objects absent from the manipulation set. Qwen-VLA-Base clears all objects in sequence, showing zero-shot visual recognition of novel objects and re-planning after each pick-and-place cycle. This behavior is consistent with the mixed VL–VLA–VLN training setup: visual-language data exposes the model to a broader object vocabulary, which can transfer to manipulation even without in-domain demonstrations.

Unseen object interaction (bottom-left). We further probe zero-shot generalization by issuing “approach {object}” instructions for object categories entirely absent from pretraining: sunglasses, a plush doll, and a toy duck. Notably, the “approach” action itself is almost never present in our manipulation data, yet the model correctly interprets the instruction and navigates the end-effector toward each target object. For the toy duck, the model even succeeds in grasping it by the head. For sunglasses, the model accurately localizes the object and approaches it, though the thin and flat geometry prevents a stable grasp. This reflects the physical boundary of shape generalization rather than a failure in visual recognition or instruction following. This result suggests instruction-following transfer from mixed VL–VLA–VLN continual pre-training: the model can follow novel action verbs and recognize unseen objects using linguistic and visual coverage from the general-purpose pre-training data.

![](images/f8722fe4f376241572c6240c0e8ad249c25c25294afa18be2fc80a763dcd54a6.jpg)  
Synthetic Data Ratio in T2A Corpus (a) Data Composition & Prediction Mode

![](images/91fbb4a60a106c48d3a53ed6cb978ea8b9ef85f72f25e94e69c2ae6ae7e9fbf4.jpg)  
(b) Timestep Distribution

![](images/b41b7d15f7881dffa243f2badedb5bd5b5f2c32891698f04d02483c6a8413aeb.jpg)  
(c) T2A Training Duration  
Figure 6: T2A pretraining ablations. (a) Data composition and prediction mode. SFT success rate (%) vs. synthetic data ratio in the T2A corpus. The dashed line marks the baseline without T2A (60.9%). Fullsequence prediction with ∼20% synthetic + 80% real data achieves the best result (71.1%, +10.2 pp over no T2A). Chunk prediction consistently underperforms full-sequence $( \mathrm { e . g . + 4 . 9 }$ pp at 10% synthetic), and including image tokens during T2A further hurts chunk-mode performance $( { \dot { - } } { \dot { 2 } } . 9 \mathrm { p p } )$ , confirming that images should be fully suppressed at T2A. (b) Flow-matching timestep distribution. The combination of Sigmoid-Normal p(τ) at T2A and Beta p(τ) at SFT achieves the highest success rate (71.1%); reversing the choice at either stage degrades performance. (c) T2A training duration. Performance peaks at 2,000 steps (71.1%) and plateaus through 10,000 steps; 40,000 steps causes a slight drop due to overfitting to the T2A corpus.

Background robustness (bottom-right). Finally, we probe robustness to visual domain shift. Qwen-VLA-Base successfully uncaps a pen and places the cap on the table, which is a dexterous two-stage action requiring precise force control. While this task may have been encountered during pre-training, the yellow background used here was never seen during training. Despite this unseen visual context, the policy completes the task without any degradation. This invariance to background perturbation is consistent with the visual diversity of the general-purpose VL understanding data used during continual pre-training. Exposure to natural images with varied scenes, lighting, and backgrounds may encourage the visual encoder to focus on task-relevant objects rather than spurious background cues.

## 5.2 Ablation Studies

## 5.2.1 Text-to-Action Pretraining Ablations

As established in Section 3.1, T2A trains the decoder to reconstruct action distributions from language and embodiment prompts alone, building a structured action prior before visual grounding begins. We now ablate five design choices that shape this prior: data composition, sequence prediction mode, visual input, flow-matching timestep distribution, and training duration. All ablations share the same downstream training recipe; results are task-success rate (%) after SFT from the T2A checkpoint on Simpler-WidowX.

Data composition. T2A acquires action priors from language alone, so its data composition directly shapes what the decoder learns. We consider two sources. Syn (language-only synthetic trajectories from our simulation pipeline) is cheap to scale and broad in task coverage, but its trajectories are kinematically idealised. Real (real-robot teleoperation data, vision dropped) captures authentic physical dynamics but is expensive to collect and narrower in task scope.

As shown in Fig. 6(a), purely real T2A data yields 51.0%, while purely synthetic data reaches 64.1%—better, but still suboptimal. Mixing ∼20% synthetic with 80% real achieves the best result (71.1%), a +10.2 pp gain over the no-T2A baseline (60.9%). Syn broadens the coverage of language–action correspondences that the decoder can learn, while Real anchors the prior in physically plausible dynamics.

Sequence prediction mode: full-length vs. chunk. Full-sequence prediction supervises the entire trajectory in one forward pass, exposing the decoder to long-horizon temporal structure and natural start/termination patterns. Chunk prediction instead splits each trajectory into fixed-length windows and trains the decoder to produce only the next window from a task instruction, which truncates long-horizon dependencies and creates positional ambiguity at chunk boundaries.

Fig. 6(a) shows that full-sequence prediction consistently outperforms chunk prediction across data mixes. At 10% synthetic data, the gap is +4.9 pp (65.4% vs. 60.4%); at 0% synthetic (real-only) the gap is +2.9 pp (51.0% vs. 48.2%). Full trajectories let the decoder learn how language maps to complete action sequences, including trajectory-level coherence and compositionality, whereas chunks provide only local fragments that leave this mapping incomplete. Importantly, the action representation is identical between T2A and downstream stages: actions are encoded as delta end-effector displacements relative to the first frame of each chunk, and the embodiment prompt specifies the robot platform, normalisation convention, and prediction horizon, allowing the decoder to handle heterogeneous embodiments within the same T2A objective.

Visual input during T2A. T2A is deliberately a vision-free stage. Including image observations would raise per-step cost by roughly an order of magnitude and, more importantly, would let the decoder shortcut through visual cues instead of learning to ground actions in language.

Fig. 6(a) confirms this: at 10% synthetic data, chunk prediction with images scores 57.6%, while the same setting without images reaches 60.4%—a −2.9 pp penalty for including vision. When images are present, the decoder can rely on visual correlations rather than building a reliable language–action mapping, diluting exactly the prior that T2A is designed to establish. Image tokens should be fully suppressed during T2A; visual grounding is deferred to CPT, by which time the decoder already possesses a well-formed action prior.

Flow-matching timestep distribution. Flow-matching policies condition on a scalar timestep $\tau \in [ 0 , 1 ]$ that interpolates between the clean target (τ=0) and pure noise (τ=1). The sampling distribution $p ( \tau )$ determines which noise levels receive the most gradient signal during training. The standard practice is to use a Beta distribution, which concentrates density toward the clean end; we follow this convention at the CPT and SFT stages. However, since T2A operates without visual conditioning, the decoder has no backbone features to guide denoising, and we hypothesise that shifting gradient toward intermediate timesteps—where the signal-to-noise ratio is most informative for learning the language–action mapping— may be more effective. We therefore compare Beta against Sigmoid-Normal, which peaks at intermediate noise levels (Fig. 6(b)).

As shown in Fig. 6(b), Sigmoid-Normal p(τ) at T2A combined with Beta $p ( \tau )$ at SFT achieves the highest success rate (71.1%). Swapping to Beta at T2A drops performance to 65.4% (−5.7 pp), while using Sigmoid-Normal at SFT instead of Beta yields 62.8% (−8.3 pp). The worst combination—Beta at both stages—reaches only 59.4%. Without visual conditioning to guide denoising, intermediate timesteps carry the most learning signal for the language–action prior; once the VLM backbone supplies rich conditioning at CPT/SFT, a Beta-shaped $p ( \tau )$ that spreads gradient more uniformly becomes more sample-efficient. We therefore use Sigmoid-Normal at T2A and Beta at CPT/SFT throughout our final pipeline.

T2A training duration. Longer T2A training lets the decoder more fully explore the action distribution, but excessive training on a fixed T2A corpus risks overfitting to its idiosyncrasies and reducing the plasticity available to CPT. We sweep T2A duration over a wide range to locate the sweet spot.

As shown in Fig. 6(c), performance peaks at 2,000 steps (71.1%) and remains comparable at 4,000 steps (67.5%) and 10,000 steps (67.2%). At 40,000 steps we observe a notable degradation to 60.4%. The rapid convergence is expected: the language–action mapping that T2A learns is structurally lower-dimensional than the full vision–language–action space, so a modest number of steps suffices to capture it; prolonged training begins to memorise specific trajectory instances rather than refining the structural prior. We adopt 2,000 T2A steps as the default in our final pipeline, offering a favourable cost–quality tradeoff.

## 5.2.2 Multi-Embodiment Co-training

We study two design axes that are critical for multi-embodiment co-training: the role of vision-language data during action learning and the projection policy for heterogeneous action spaces.

Impact of vision-language data on action learning. Although vision-language supervision operates in discrete token space while action prediction targets continuous trajectories, the two objectives share a common perceptual and linguistic backbone. A natural question is whether retaining VL data during embodied fine-tuning helps or hurts action quality. We compare three configurations in Fig. 7: (a) Actiononly: the backbone is fine-tuned exclusively on embodied trajectories with no VL supervision; (b) Action + VL data: a fraction of VL data (VQA, captioning, spatial grounding) is mixed into the training corpus.

![](images/d7c82f93f9ae10cf1692216f7cf2ab583b6fc794a0e2ba5d2748a9dd4e56ccb6.jpg)  
(a)

![](images/5b7aef5130ae16a830f34a793ec9d185e18832b6a0738e8582055ef8c06a01ee.jpg)  
(b)  
Figure 7: Vision-language co-training ablations. (a) Impact of VL data on action learning. Task-average success rate (%) across four benchmarks for VLA-Only (action data only) vs. VL+VLA (action + visionlanguage data co-training). On simpler benchmarks (Libero, Simpler-WidowX) the two configurations perform comparably, confirming that VL co-training introduces no interference. On benchmarks that demand fine-grained object recognition and compositional instruction parsing, VL+VLA yields clear gains: +4.9 pp on RoboCasa-GR1 $( 5 1 . 1 \%  5 6 . 0 \% )$ and +4.6 pp on RoboTwin-2.0 (81.8% → 86.4%). (b) Transferability of pretrained DiT. SFT success rate (%) on RoboCasa-GR1 vs. training steps (k) when pairing the original Qwen3.5 VLM backbone with a DiT decoder initialised from scratch vs. a pretrained DiT. The pretrained DiT converges faster and reaches a higher peak.

As shown in $\mathrm { F i g . } 7 ( \mathrm { a } ) .$ , mixing VL data into training yields clear benefits on manipulation benchmarks. On Libero and Simpler-WidowX, VLA-Only and VL+VLA achieve nearly identical success rates. On RoboCasa-GR1, which requires whole-body control in cluttered kitchen scenes with diverse objects and spatial configurations, VL+VLA improves over VLA-Only by +4.9% absolute $( 5 1 . 1 \%  5 6 . 0 \% )$ . Moreover, we attach the pretrained DiT to a fresh Qwen3.5-4B VLM and compare against a from-scratch baseline where the DiT is randomly initialised. As shown in Fig. 7 (b), the pretrained DiT consistently outperforms the from-scratch baseline throughout training, converging faster in early stages and reaching a higher peak success.

Projection design for heterogeneous embodiments. Different robot platforms have different action dimensionalities and semantics (e.g. 7-DoF arm vs. 29-DoF whole-body). To learn a shared representation space across heterogeneous action formats, the projection between the DiT latent space and per-embodiment action vectors must reconcile this mismatch. To this end, we compare three representative designs.

Multi-MLP. Each embodiment owns a private encoder MLP that maps its native action dimension to the DiT hidden size, and a private decoder MLP that maps back. Let ${ \bar { d } } _ { i }$ denote the action dimensionality of embodiment i, h the DiT hidden size, and N the number of embodiments. Each embodiment owns a private encoder $\dot { f } _ { i } ^ { \mathrm { e n c } } : \mathbb { R } ^ { d _ { i } } \xrightarrow { } \mathbb { R } ^ { h }$ and decoder $f _ { i } ^ { \mathrm { d e c } } : \mathbb { R } ^ { h } \xrightarrow { } \mathbb { R } ^ { d _ { i } }$ , totalling $2 h \textstyle \sum _ { i = 1 } ^ { N } d _ { i }$ parameters.

Concatenation. Actions from all N embodiments are concatenated into a single vector of dimensionality $\begin{array} { r } { D = \sum _ { i = 1 } ^ { N } d _ { i } ; } \end{array}$ each embodiment occupies a dedicated, non-overlapping segment. A shared encoder $f ^ { \mathrm { e n c } } : \mathbb { R } ^ { D }  \mathbb { R } ^ { h }$ and decoder $f ^ { \mathrm { d e c } } : \mathbb { R } ^ { \bar { h } } \xrightarrow [ ] { } \mathbb { R } ^ { D }$ process the full vector $( 2 h \textstyle \sum _ { i = 1 } ^ { N } d _ { i }$ parameters), and each embodiment retrieves its predictions from the corresponding segment.

Zero-Padding. A single shared MLP encodes all embodiments; shorter action vectors are right-padded with zeros to $d _ { \operatorname* { m a x } { } } = \operatorname* { m a x } _ { i } d _ { i }$ before encoding. A shared encoder $f ^ { \mathrm { e n c } } : \mathbb { R } ^ { d _ { \mathrm { m a x } } }  \mathbb { R } ^ { h }$ and decoder $f ^ { \mathrm { d e c } } : \mathbb { R } ^ { h } \to \mathbb { R } ^ { d _ { \operatorname* { m a x } } }$ process the padded vector $( 2 h d _ { \operatorname* { m a x } }$ parameters); each embodiment reads only its di-dimensional prefix. Since $\begin{array} { r } { d _ { \mathrm { m a x } } ^ { \bullet } \leq \sum _ { i } d _ { i } , } \end{array}$ Zero-Padding uses fewer projection parameters than both Multi-MLP and Concatenation.

As shown in Table 10, all co-training projection designs achieve performance comparable to the singleembodiment baselines on both benchmarks (Bridge single-train 62.8% vs. co-train ∼63%; Robocasa singletrain 53.4% vs. co-train 52–53%), confirming that multi-embodiment co-training introduces no significant penalty relative to training on each dataset independently. Among the three designs, performance differences are marginal (<1.2% absolute on both benchmarks), indicating that the choice of projection architecture has limited impact on task success once a shared latent space is established. However, Zero-Padding is architecturally the lightest: it requires only $2 h d _ { \mathrm { { m a x } } }$ projection parameters compared with $2 h \textstyle \sum _ { i } d _ { i }$ for Multi-MLP and Concatenation. We therefore adopt Zero-Padding as the default projection design.

Table 10: Projection design ablation. Results are task-average success rates (%). All co-training variants share the same training recipe.
<table><tr><td rowspan="2"></td><td colspan="2">Single-Embodiment Training</td><td colspan="3">Multi-Embodiment Co-training</td></tr><tr><td>Bridge Only</td><td>Robocasa Only</td><td>Multi-MLP</td><td>Concat.</td><td>Zero-Pad</td></tr><tr><td>Bridge</td><td>62.8</td><td>丨</td><td>63.3</td><td>63.0</td><td>63.0</td></tr><tr><td>Robocasa</td><td>1</td><td>53.4</td><td>52.1</td><td>52.8</td><td>53.2</td></tr></table>

## 5.2.3 Effect of RL Post-Training

We study whether RL on top of SFT provides additional gains, and whether those gains transfer to benchmarks not seen during RL training. To isolate the contribution of each post-training stage, we evaluate three checkpoints along the training pipeline: the CPT base model (Qwen-VLA-Base), the model after multi-task SFT (+SFT), and the model after RL refinement (+RL, i.e., Qwen-VLA-Instruct). RL rollouts are collected exclusively in SimplerEnv with sparse binary success rewards; no other benchmark data is used during the RL stage.

Table 11: Cumulative effect of post-training stages. Each row adds one stage on top of the previous checkpoint. RL rollouts are collected only in SimplerEnv. All numbers are success rates (%) except the DOMINO Manipulation Score (MS).
<table><tr><td rowspan="2"> Stage</td><td colspan="5">In-Distribution</td><td>OOD-static</td><td colspan="2">0OD-dynamic (DOMINO)</td></tr><tr><td>Simpler</td><td>RoboCasa</td><td>RoboTwin-E</td><td>RoboTwin-H</td><td>LIBERO</td><td>SimplerOOD</td><td>SR</td><td>MS</td></tr><tr><td>CPT</td><td>64.3</td><td>40.4</td><td>64.3</td><td>66.4</td><td>90.8</td><td>25.3</td><td>21.1</td><td>37.4</td></tr><tr><td>+SFT</td><td>70.8</td><td>56.0</td><td>86.3</td><td>87.1</td><td>97.8</td><td>31.6</td><td>25.7</td><td>39.1</td></tr><tr><td>+RL</td><td>73.7</td><td>56.7</td><td>86.1</td><td>87.2</td><td>97.9</td><td>32.0</td><td>26.6</td><td>39.5</td></tr></table>

As shown in Table 11, each post-training stage provides complementary gains. SFT delivers large improvements across all benchmarks by specializing the general-purpose CPT representations to downstream task distributions. RL then provides a further uplift on top of SFT, with the largest gain on SimplerEnv (+2.9 pp, from 70.8% to 73.7%), the environment in which RL rollouts are collected. This confirms that directly optimizing closed-loop task success captures policy improvements that the imitation objective alone cannot.

More importantly, the RL gains are not confined to the training environment. On benchmarks that are not part of the RL rollout distribution, performance is preserved or mildly improved: RoboCasa increases by 0.7 pp, RoboTwin-Hard by 0.1 pp, LIBERO by 0.1 pp, and the OOD evaluation on SimplerEnv-OOD by 0.4 pp. RoboTwin-Easy exhibits negligible changes (<0.6 pp), confirming that RL does not cause catastrophic forgetting on held-out tasks. On the DOMINO dynamic manipulation benchmark, which tests zero-shot generalization to moving objects, both SR (25.7% → 26.6%) and MS (39.1 → 39.5) improve despite the complete absence of dynamic manipulation data during RL training. These results indicate that task-success optimization in simulation preserves performance with mild positive transfer: the policy learns to execute more decisively and recover from compounding errors, benefits that generalize across evaluation settings.

## 5.2.4 State Conditioning

Many robot learning methods condition the policy on explicit proprioceptive state (e.g., joint angles) in addition to visual observations, under the assumption that knowing the current robot configuration helps predict the next action. However, proprioceptive inputs also introduce embodiment-specific dependencies that may hinder cross-embodiment generalization, since different platforms have different joint spaces and kinematic structures. We examine whether explicit state conditioning benefits action learning in Qwen-VLA and, if so, where in the architecture the state should be injected. We compare three strategies on RoboTwin-2.0:

• No State. The model receives only visual observations and the language instruction. No proprioceptive information is provided to either the VLM backbone or the DiT action decoder.

• State in VLM Prompt. Joint angles are discretized into 256 bins per dimension and serialized into the language prompt, e.g., "state: [34, 128, 67, 200, 55, 12, 88] pick up the bowl". The VLM processes state tokens together with images and the instruction; the DiT receives no additional state input.

• State in DiT. Joint angles are passed as continuous-valued vectors directly to the DiT action decoder as an additional conditioning input, bypassing the VLM entirely.

Table 12: State conditioning ablation on RoboTwin-2.0 (success rates, %).
<table><tr><td>Conditioning</td><td>RoboTwin-Easy</td><td>RoboTwin-Hard</td></tr><tr><td>No State</td><td>88.7</td><td>87.4</td></tr><tr><td>State in VLM Prompt</td><td>89.3</td><td>88.7</td></tr><tr><td>State in DiT</td><td>89.4</td><td>88.3</td></tr></table>

As shown in Table 12, incorporating proprioceptive state yields only marginal gains regardless of the injection strategy: at most +0.7 pp on Easy and +1.3 pp on Hard relative to the vision-only baseline. The two injection strategies perform comparably (≤0.4 pp difference), suggesting that neither the VLM’s semantic processing nor the DiT’s continuous conditioning offers a clear advantage for encoding jointangle information. We attribute the limited impact of state conditioning to two factors. First, the multiview visual observations already provide sufficient information about the robot’s current configuration, especially for tasks where the end-effector is visible in at least one camera view. Second, the flow-matching action decoder predicts relative action displacements rather than absolute poses, reducing the need for an explicit current-state reference. Given the limited benefits and the additional complexity of maintaining embodiment-specific state interfaces across diverse platforms, we opt not to include state conditioning in our default framework, keeping the embodiment-aware text prompt as the sole platform-specific input.

## 6 Conclusion

We presented Qwen-VLA, a unified vision-language-action model that extends the Qwen vision-language backbone from perception and reasoning to embodied action generation. By formulating manipulation, navigation, egocentric action modeling, and trajectory prediction within a shared action-and-trajectory prediction framework, Qwen-VLA enables a single model to learn from heterogeneous embodied data across tasks, environments, and robot embodiments. Through embodiment-aware prompt conditioning, a DiT-based flow-matching action decoder, large-scale joint pretraining, and SFT/RL post-training, Qwen-VLA achieves strong generalist performance while retaining flexibility across different control conventions. Our results suggest that language-grounded VLA models provide a practical path toward embodied intelligence by aligning multimodal perception, instruction understanding, embodiment constraints, and executable actions. Compared with visual prediction-centric world models, Qwen-VLA emphasizes grounding multimodal understanding into actions that can be executed by embodied agents. Together, these results indicate that VLA models can serve as an actionable interface between multimodal foundation models and embodied agents, bridging high-level language reasoning with low-level physical control.

## 7 Limitations and Future Work

Despite its generalist capabilities, Qwen-VLA still has several limitations. First, embodied action data remains far smaller and less diverse than vision-language pretraining data, limiting robustness to long-tail objects, environments, embodiments, and contact-rich interactions. Second, joint training across visionlanguage understanding, navigation, and action generation introduces optimization trade-offs. While action-oriented training improves policy learning, it can modestly regress some pure vision-language and navigation evaluations, suggesting the need for better objective balancing, data curricula, and modular specialization. Third, current evaluations are still largely short-horizon and benchmark-driven, leaving long-duration, failure-prone real-world deployment as an open challenge.

These limitations motivate several future directions. Scaling real-world interaction data through autonomous collection, simulation, and sim-to-real transfer will be critical for improving robustness. Large-scale human video, from both egocentric and third-person views, can provide physical priors and temporal abstractions beyond robot demonstrations. Future models should incorporate long-horizon planning, episodic memory, and world modeling, enabling agents to track state, decompose tasks, recover from failures, and anticipate action consequences. Finally, richer physical feedback such as force, tactile, and proprioceptive signals, combined with large-scale reinforcement learning in simulation and the real world, may further bridge the gap between language-grounded embodied understanding and reliable control.

## 8 Contributions and Acknowledgments

## Core Contributors

<table><tr><td>Qiuyue Wang* Yitao Liu Xuhong Huang Jingren Zhou</td><td>Mingsheng Li* Junhao Chen Pei Lin</td><td>Jian Guan* Zhixuan Liang Junyang Lin</td><td>Jinhui Ye Jie Zhang Dayiheng Liu</td><td>Sicheng Xie Xintong Hu Shuai Bai+</td></tr><tr><td>Contributors</td><td></td><td></td><td></td><td></td></tr><tr><td> Jiazhao Zhang</td><td>Haoqi Yuan</td><td>Gengze Zhou</td><td>Hang Yin</td><td>Ye Wang</td></tr><tr><td>Yiyang Huang</td><td>Zixing Lei</td><td>Wujian Peng Xin Zhou</td><td>Delin Chen Haoyang Li</td><td>Yingming Zheng Anzhe Chen</td></tr><tr><td>Jingyang Fan</td><td> Xianwei Zhuang</td><td>Yuchong Sun</td><td>Ruizhe Chen</td><td>Zhaohai Li</td></tr><tr><td>Tong Zhang</td><td> Xuejing Liu</td><td></td><td></td><td></td></tr><tr><td>Chenxu Lu</td><td>Zhibo Yang</td><td>Tao Yu</td><td>Xionghui Chen</td><td></td></tr></table>

## Acknowledgements

We thank Jun Huang, Jianlou Si, Chao Wang, and Nana Jiang for their strong support on simulation environments and reinforcement learning.

## References

AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, Li Chen, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems, 2025. URL https://arxiv.org/ abs/2503.06669.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025b.

Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking, 2023. URL https://arxiv.org/abs/2309.01918.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: A vision-language-action flow model for general robot control. Technical report, Physical Intelligence, 2025.

Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, et al. RT-1: Robotics transformer for real-world control at scale, 2022. URL https://arxiv.org/abs/2212.06817.

Xu Cao, Tong Zhou, Yunsheng Ma, Wenqian Ye, Can Cui, Kun Tang, Zhipeng Cao, Kaizhao Liang, Ziran Wang, James M Rehg, et al. Maplm: A real-world large-scale vision-language benchmark for map and traffic scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21819–21830, 2024.

Kai Chen, Yanze Li, Wenhua Zhang, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, et al. Automated evaluation of large vision-language models on self-driving corner cases. In IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 7817–7826, 2025a.

Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, et al. Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy. arXiv preprint arXiv:2510.13778, 2025b.

An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Xueyan Zou, Jan Kautz, Erdem Biyik, Hongxu Yin, Sifei Liu, and Xiaolong Wang. NaVILA: Legged robot vision-language-action model for navigation. In Robotics: Science and Systems (RSS), 2025.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robotics: Science and Systems (RSS), 2023.

Haohan Chi, Huan-ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, et al. Impromptu vla: Open weights and open data for driving vision-languageaction models. In Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

Matei Ciocarlie, Corey Goldfeder, and Peter Allen. Dexterous grasping via eigengrasps: A lowdimensional approach to a high-complexity problem. In Robotics: Science and systems manipulation workshop-sensing and adapting to the real world, 2007.

StarVLA Community. Starvla: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

Charles Corbiere, Simon Roburin, Syrielle Montariol, Antoine Bosselut, and Alexandre Alahi. Retrieval-based interleaved visual chain-of-thought in real-world driving scenarios. arXiv preprint arXiv:2501.04671, 2025.

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision, 130:33–55, 2022.

Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, and Marie Francine Moens. Talk2car: Taking control of your self-driving car. In Proceedings of the conference on empirical methods in natural language processing, pp. 2088–2098, 2019.

Karim Elmaaroufi, Liheng Lai, Justin Svegliato, Yutong Bai, Sanjit A Seshia, and Matei Zaharia. Graid: Enhancing spatial reasoning of vlms through high-fidelity data generation. arXiv preprint arXiv:2510.22118, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for highresolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning (ICML), pp. 12606–12633, 2024.

Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Chenxi Wang, Junbo Wang, Haoyi Zhu, and Cewu Lu. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot, 2023. URL https://arxiv.org/abs/2307.00595.

Heng Fang, Shangru Li, Shuhan Wang, Xuanyang Xi, Dingkang Liang, and Xiang Bai. Towards generalizable robotic manipulation in dynamic environments. arXiv preprint arXiv:2603.15620, 2026.

Jianwu Fang, Lei-lei Li, Junfei Zhou, Junbin Xiao, Hongkai Yu, Chen Lv, Jianru Xue, and Tat-Seng Chua. Abductive ego-view accident video understanding for safe driving perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22030–22040, 2024.

Zipeng Fu, Tony Z Zhao, and Chelsea Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117, 2024.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18995–19012, 2022.

Yuhan Hao, Zhengning Li, Lei Sun, Weilong Wang, Naixin Yi, Sheng Song, Caihong Qin, Mofan Zhou, Yifei Zhan, and Xianpeng Lang. Driveaction: A benchmark for exploring human-like driving decisions in vla models. arXiv preprint arXiv:2506.05667, 2025.

Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. In International Conference on Learning Representations (ICLR), 2026.

Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, et al. RoboMIND 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence, 2025. URL https://arxiv.org/ abs/2512.24653.

Xintong Hu, Xuhong Huang, Jinyu Zhang, Yutong Yao, Yuchong Sun, Qiuyue Wang, Mingsheng Li, Sicheng Xie, Yitao Liu, Junhao Chen, Yixuan Chen, Yingming Zheng, Shuai Bai, and Tao Yu. Finevla: Fine-grained instruction alignment for steerable vision-language-action policies, 2026. URL https: //arxiv.org/abs/2605.27284.

Yuichi Inoue, Yuki Yada, Kotaro Tanahashi, and Yu Yamaguchi. Nuscenes-mqa: Integrated evaluation of captions and qa for autonomous driving datasets using markup annotations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 930–938, 2024.

Physical Intelligence, Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, et al. π0.7: a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning, 2022. URL https://arxiv.org/abs/2202.02005.

Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, pp. 9902–9915. PMLR, 2022.

Xiaosong Jia, Yuqian Shao, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2drivevl: Benchmarks for closed-loop autonomous driving with vision-language models. arXiv preprint arXiv:2604.01259, 2026.

Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024.

Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, et al. Galaxea open-world dataset and G0 dualsystem VLA model, 2025. URL https://arxiv.org/abs/2509.00576.

Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. Egomimic: Scaling imitation learning via egocentric video. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 13226–13233. IEEE, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset, 2024. URL https://arxiv.org/abs/2403.12945.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, et al. Openvla: An open-source visionlanguage-action model. In 8th Annual Conference on Robot Learning, 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

Jacob Krantz, Erik Wijmans, Arjun Majundar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision and language navigation in continuous environments. In European Conference on Computer Vision (ECCV), 2020.

Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4392–4412, 2020.

Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026a.

Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, Yizhong Zhang, Xi Chen, Hao Chen, Lily Sun, Dong Chen, Jiaolong Yang, and Baining Guo. Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos. In IEEE International Conference on Robotics and Automation (ICRA), 2026b.

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.

Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, 2026c.

Dingkang Liang, Cheng Zhang, Xiaopeng Xu, Jianzhong Ju, Zhenbo Luo, and Xiang Bai. Cook and clean together: Teaching embodied agents for parallel task execution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 18415–18424, 2026.

Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, and Ping Luo. Adaptdiffuser: diffusion models as adaptive self-evolving planners. In Proceedings of the 40th International Conference on Machine Learning, pp. 20725–20745, 2023.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations (ICLR), 2023.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yifeng Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation, 2025a. URL https://arxiv.org/abs/2410.07864.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, 2025b.

Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: Vision-language-action pretraining from large-scale human videos. arXiv preprint arXiv:2507.15597, 2025.

Hao Luo, Ye Wang, Wanpeng Zhang, Sipeng Zheng, Ziheng Xi, Chaoyi Xu, Haiweng Xu, Haoqi Yuan, Chi Zhang, Yiqing Wang, Yicheng Feng, and Zongqing Lu. Being-h0.5: Scaling human-centric robot learning for cross-embodiment generalization. arXiv preprint arXiv:2601.12993, 2026.

Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, et al. Lingoqa: Visual question answering for autonomous driving. In European Conference on Computer Vision, pp. 252–269, 2024.

Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Muñoz, Xinjie Yao, René Zurbrügg, Nikita Rudin, Lukasz Wawrzyniak, Milad Rakhsha, Alain Denzler, Eric Heiden, Ales Borovicka, Ossama Ahmed, Iretiayo Akinola, Abrar Anwar, Mark T. Carlson, Ji Yuan Feng, Animesh Garg, Renato Gasoto, Lionel Gulich, Yijie Guo, M. Gussert, Alex Hansen, Mihir Kulkarni, Chenran Li, Wei Liu, Viktor Makoviychuk, Grzegorz Malczyk, Hammad Mazhar, Masoud Moghani, Adithyavairavan Murali, Michael Noseworthy, Alexander Poddubny, Nathan Ratliff, Welf Rehberg, Clemens Schwarke, Ritvik Singh, James Latham Smith, Bingjie Tang, Ruchik Thaker, Matthew Trepte, Karl Van Wyk, Fangzhou Yu, Alex Millane, Vikram Ramasamy, Remo Steiner, Sangeeta Subramanian, Clemens Volk, CY Chen, Neel Jawale, Ashwin Varghese Kuruttukulam, Michael A. Lin, Ajay Mandlekar, Karsten Patzwaldt, John Welsh, Huihua Zhao, Fatima Anes, Jean-Francois Lafleche, Nicolas Moënne-Loccoz, Soowan Park, Rob Stepinski, Dirk Van Gelder, Chris Amevor, Jan Carius, Jumyung Chang, Anka He Chen, Pablo de Heras Ciechomski, Gilles Daviet, Mohammad Mohajerani, Julia von Muralt, Viktor Reutskyy, Michael Sauter, Simon Schirm, Eric L. Shi, Pierre Terdiman, Kenny Vilella, Tobias Widmer, Gordon Yeoman, Tiffany Chen, Sergey Grizan, Cathy Li, Lotus Li, Connor Smith, Rafael Wiltz, Kostas Alexis, Yan Chang, David Chu, Linxi "Jim" Fan, Farbod Farshidian, Ankur Handa, Spencer Huang, Marco Hutter, Yashraj Narang, Soha Pouya, Shiwei Sheng, Yuke Zhu, Miles Macklin, Adam Moravanszky, Philipp Reist, Yunrong Guo, David Hoeller, and Gavriel State. Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025. URL https://arxiv.org/abs/2511.04831.

Yao Mu, Tianxing Chen, Shijia Peng, Zanxin Chen, Zeyu Gao, Yude Zou, Lunkai Lin, Zhiqiang Xie, and Ping Luo. Robotwin: Dual-arm robot benchmark with generative digital twins (early version). In European Conference on Computer Vision, pp. 264–273. Springer, 2024.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. RoboCasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems (RSS), 2024.

NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4195–4205, 2023.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Ryan Punamiya, Simar Kareer, Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Jiaqi Chen, Davide Liconti, Lawrence Y. Zhu, et al. Egoverse: An egocentric human dataset for robot learning from around the world. arXiv preprint arXiv:2604.07607, 2026.

Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 4542–4550, 2024.

Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: modeling and capturing hands and bodies together. ACM Transactions on Graphics (TOG), 36(6):1–17, 2017.

Ropedia. Xperience-10m: A large-scale egocentric multimodal dataset of human experience. https: //huggingface.co/datasets/ropedia-ai/xperience-10m, 2026.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017.

Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European conference on computer vision, pp. 256–274, 2024.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Balakumar Sundaralingam, Siva Kumar Sastry Hari, Adam Fishman, Caelan Garrett, Karl Van Wyk, Valts Blukis, Alexander Millane, Helen Oleynikova, Ankur Handa, Fabio Ramos, Nathan Ratliff, and Dieter Fox. curobo: Parallelized collision-free minimum-jerk robot motion generation, 2023. URL https://arxiv.org/abs/2310.17274.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.

Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, and Jiangmiao Pang. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy, 2025. URL https://arxiv. org/abs/2511.16651.

Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale, 2024. URL https://arxiv.org/abs/2308.12952.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv:2409.12191, 2024.

Shaoan Wang, Jiazhao Zhang, Minghan Li, Jiahang Liu, Anqi Li, Kui Wu, Fangwei Zhong, Junzhi Yu, Zhizheng Zhang, and He Wang. TrackVLA: Embodied visual tracking in the wild. arXiv preprint arXiv:2505.23189, 2025a.

Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the computer vision and pattern recognition conference, pp. 22442–22452, 2025b.

Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale vision-language-action model. In Proceedings of the AAAI conference on artificial intelligence, volume 40, pp. 18638–18646, 2026.

Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, et al. Streamvln: Streaming vision-and-language navigation via slowfast context modeling. arXiv preprint arXiv:2507.05240, 2025.

Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, Shichao Fan, Xinhua Wang, Fei Liao, Zhen Zhao, Guangyu Li, Zhao Jin, Lecheng Wang, Jilei Mao, Ning Liu, Pei Ren, Qiang Zhang, Yaoxu Lyu, Mengzhen Liu, He Jingyang, Yulin Luo, Zeyu Gao, Chenxuan Li, Chenyang Gu, Yankai Fu, Di Wu, Xingyu Wang, Sixiang Chen, Zhenyu Wang, Pengju An, Siyuan Qian, Shanghang Zhang, and Jian Tang. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation. In Robotics: Science and Systems XXI, RSS2025. Robotics: Science and Systems Foundation, June 2025a. doi: 10.15607/rss.2025.xxi.152. URL http://dx.doi.org/10.15607/RSS.2025.XXI.152.

Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation, 2025b. URL https://arxiv.org/abs/2511.17441.

Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Shuailei Ma, He Sun, Yong Wang, Zhenqi Qiu, Houlong Xiong, Ziyu Wang, Shuai Zhou, Yiyu Ren, Kejia Zhang, Hui Yu, Jingmei Zhao, Qian Zhu, Ran Cheng, Yong-Lu Li, Yongtao Huang, Xing Zhu, Yujun Shen, and Kecheng Zheng. A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692v1, 2026.

Sicheng Xie, Lingchen Meng, Zijie Diao, Haidong Cao, Zhiying Du, Shuyuan Tu, Jiaqi Leng, Qiuyue Wang, Mingsheng Li, Shuai Bai, Zuxuan Wu, and Yu-Gang Jiang. Unify robot actions in camera frame, 2026. URL https://arxiv.org/abs/2511.17001.

XLANG Lab. Introducing roboinf: A scalable data engine for generalist robot manipulation. https: //xlang.ai/blog/roboinf, May 2026. Blog post.

Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 9(10):8186–8193, 2024.

Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, Feng Xiong, Xing Wei, Zhiheng Ma, and Mu Xu. ABot-M0: Vla foundation model for robotic manipulation with action manifold learning. arXiv preprint arXiv:2602.11236, 2026.

Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, Chunyang Zhu, Junhao Hu, et al. Rlinf: Flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation. arXiv preprint arXiv:2509.15965, 2025a.

Seungjun Yu, Seonho Lee, Namho Kim, Jaeyo Shin, Junsung Park, Wonjeong Ryu, Raehyuk Jung, and Hyunjung Shim. Waymoqa: A multi-view visual question answering dataset for safety-critical reasoning in autonomous driving. arXiv preprint arXiv:2511.20022, 2025b.

Haoqi Yuan, Bohan Zhou, Yuhui Fu, and Zongqing Lu. Cross-embodiment dexterous grasping with reinforcement learning. In International Conference on Learning Representations (ICLR), 2025.

Jianke Zhang, Xiaoyu Chen, Qiuyue Wang, Mingsheng Li, Yanjiang Guo, Yucheng Hu, Jiajun Zhang, Shuai Bai, Junyang Lin, and Jianyu Chen. Vlm4vla: Revisiting vision-language-models in visionlanguage-action models. arXiv preprint arXiv:2601.03309, 2026.

Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Navid: Video-based vlm plans the next step for vision-and-language navigation. arXiv preprint arXiv:2402.15852, 2024.

Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, and He Wang. Embodied navigation foundation model, 2025a. URL https://arxiv.org/abs/2509.12129.

Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, and He Wang. Uni-NaVid: A video-based vision-language-action model for unifying embodied navigation tasks. Robotics: Science and Systems, 2025b.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025.

Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, Trevor Darrell, Furong Huang, Yuke Zhu, Danfei Xu, and Linxi Fan. Egoscale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710, 2026.

Yuchen Zhou, Jiayu Tang, Xiaoyan Xiao, Yueyao Lin, Linkai Liu, Zipeng Guo, Hao Fei, Xiaobo Xia, and Chao Gou. Where, what, why: Towards explainable driver attention prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2675–2685, 2025.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, 2023.