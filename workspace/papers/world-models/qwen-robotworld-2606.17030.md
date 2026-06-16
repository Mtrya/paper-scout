# Qwen-RobotWorld Technical Report: Unifying Embodied World Modeling through Language-Conditioned Video Generation

Qwen Team

https://pre.qwen.ai/blog?id=qwen-robotworld

## Abstract

We introduce QWEN-ROBOTWORLD, a language-conditioned video world model for embodied intelligence. With natural language as a unified action interface, it predicts physically grounded future visual trajectories from current observations across robotic manipulation, autonomous driving, indoor navigation, and human-to-robot transfer. This unified formulation provides three promising application directions: synthetic data generation for policy training augmentation, scalable virtual environments for policy evaluation, and language-guided planning signals for downstream robot control. This is achieved through a three-part design: a) Double-Stream MMDiT with MLLM Action Encoding, where a 60-layer double-stream diffusion transformer couples frozen Qwen2.5-VL semantics with video-VAE latents through layer-wise joint attention; b) Embodied World Knowledge (EWK), an 8.6M video-text corpus (200M+ frames) with action-language mapping over 20+ embodiments and 500+ action categories; and c) General+Expert Progressive Curriculum, a two-stage training strategy that first learns general visual priors and then injects embodied specialization under a shared language interface. Extensive results show strong competitiveness: ranks 1st overall on EWM-Bench and DreamGen Bench, outperforms all open-source models on WorldModelBench and PBench. Additional zero-shot analyses on RoboTwin-IF benchmark further support robust generalization and multi-view consistency.

![](images/bb8ca39776331c55d00a303db3888ec83e4b2dea2f36270ec2f9e9c35428498c.jpg)

## 1 Introduction

Embodied intelligence requires agents to perceive, reason, and act within physical environments—spanning robotic manipulation at tabletop scale, autonomous navigation through urban traffic, and wayfinding across indoor spaces. Training such systems directly in the real world is costly, inefficient, and fraught with safety risks. World models offer a scalable alternative: by learning environment dynamics from observational data, they serve as interactive training platforms that allow embodied agents to acquire and refine behaviors without physical deployment.

A world model can be formalized as a state transition function: given a current state st and an action $a _ { t } ,$ it predicts the resulting state $s _ { t + 1 } = f ( s _ { t } , a _ { t } )$ Ye et al. (2026). In video-based world models, states are visual observations (video frames or their latent representations), and the model generates future visual trajectories conditioned on the current observation and an action signal. The action at can take various forms—low-level motor commands, high-level waypoint trajectories, or natural language instructions. Among these, natural language is the most general and accessible action representation Ye et al. (2026): a single instruction such as “pick up the red cup and place it on the shelf” implicitly encodes the complete action sequence, goal state, and physical constraints, without requiring robot-specific control interfaces. Language actions can furthermore be utilized in two complementary directions: as an explicit input fused into the model’s condition signal to govern state transitions, or as an output inferred post-hoc from generated video to serve as an action label. This flexibility positions language-conditioned world models as universal simulation backbones that generalize across embodied platforms without interface redesign.

However, a fundamental tension currently limits world model effectiveness. General video generation models OpenAI (2024); Google DeepMind (2025) learn rich visual priors from internet-scale data but fail to accurately model embodied physics—contact dynamics, rigid-body structural constraints, and action-consequence relationships that are critical for physically plausible state transitions. Domain-specific embodied models Agarwal et al. (2025); Chen et al. (2025a); Team et al. (2025), conversely, are tailored to individual scenarios (e.g., tabletop manipulation or driving); they rely on structured, robot-specific action representations such as joint angles or waypoints, which cannot generalize across embodiment types or task categories, fundamentally limiting their utility as cross-platform simulation environments.

Bridging this gap requires grounding diverse embodied experiences in general visual priors, with natural language as the unified action interface that enables cross-scenario and cross-task integration. Different embodied domains provide complementary physical knowledge that collectively enriches the world model’s state transition function: manipulation teaches fine-grained contact physics and object-state transformations within confined workspaces; autonomous driving teaches large-scale multi-agent dynamics and 3D scene geometry through ego-motion parallax and scene-scale transitions; indoor navigation teaches room-scale spatial reasoning, where language instructions must be grounded into spatially coherent visual trajectories over extended horizons. Because these domains share a common language interface, they can be trained jointly—with each domain’s physical knowledge reinforcing the others rather than conflicting. Furthermore, translating human demonstrations into robot executions through video editing opens a practical pathway to scale embodied training data beyond the limits of physical robot collection.

We present QWEN-ROBOTWORLD, a language-conditioned video world model in the Qwen series that realizes this vision through tightly coupled innovations in architecture, data, and training. Beyond high-fidelity action-conditioned prediction, the model serves as a unified backbone that, with task-specific adaptation, can support three representative embodied world model applications: a synthetic data engine, a policy evaluation environment, and an action planner.

Architecture: Double-Stream MMDiT with MLLM Action Encoding (§3). To implement languageconditioned state transitions, we adopt a double-stream Multimodal Diffusion Transformer (MMDiT) backbone. An understanding stream processes rich semantic features extracted by a frozen Qwen2.5-VL encoder, representing the action at; a generation stream processes visual latents from a video-compatible VAE, representing the visual state st. The two streams interact via joint attention at every layer, enabling bidirectional cross-modal fusion throughout the denoising process. Using an MLLM as the action encoder—rather than lightweight encoders such as T5 Raffel et al. (2020) or CLIP Radford et al. (2021)—yields two key advantages: (1) its deep language understanding accurately parses complex, compositional instructions into precise condition signals that govern fine-grained state transitions; (2) its internalized world knowledge (e.g., that robot arms are rigid bodies with fixed link lengths and joint constraints) implicitly constrains the space of physically plausible transitions, and—combined with T2I co-training—prevents object deformation across video frames without requiring explicit geometric prompts, a common failure mode in models lacking such semantic grounding.

Data: Embodied World Knowledge Dataset (§2). To train a state transition function that generalizes across embodied domains, we construct the Embodied World Knowledge (EWK) dataset—approximately 8.6M video-text pairs comprising over 200M observation frames. The corpus spans four embodied domains alongside general video data (30% of the total): manipulation (∼5.9M samples, 20+ robot morphologies, 1300+ skills) provides the core embodied foundation; autonomous driving (∼200K samples from Waymo, NVIDIA PhysicalAI-AD, Bench2Drive, and Sekai) contributes large-scale ego-motion and multi-agent dynamics; indoor navigation (6K+ language-guided episodes from VLNVerse) provides room-scale spatial reasoning grounded in continuous trajectories; and human-to-robot transfer data—generated via an automated MANO Romero et al. (2017)-to-robot pipeline across 14 robot morphologies—enables cross-embodiment video editing. A central methodological contribution is our action-language mapping framework, which standardizes actions across 20+ robot embodiment types and 500+ action categories into a unified natural language interface, yielding approximately 8.6M high-quality cross-scenario, cross-task embodied video-text pairs. This is complemented by task-aware temporal segmentation (ensuring each sample captures a complete, well-defined state transition) and a hierarchical five-layer viewpoint-aware annotation pipeline that substantially improves caption specificity and downstream instruction-following.

Training: From General Priors to Embodied Specialization (§4). We adopt a two-stage progressive training curriculum. In pretraining, joint training across T2I, T2V, and TI2V tasks over general-domain data builds foundational visual priors, with T2I specifically anchoring geometrically correct object morphology that transfers to video generation. In the SFT stage, embodied data is introduced progressively (70% embodied, 30% general) through a four-phase mixing schedule: single-view manipulation → multi-view expansion → multi-view concatenated generation → complex tasks and cross-domain data. Within the embodied portion, manipulation dominates at ∼90% sampling weight to ensure depth of physical grounding, while multi-view concatenation and navigation/driving data each receive ∼5% to provide breadth. This general + expert joint training paradigm—unified under the natural language action interface—enables stable co-training across diverse scenarios and tasks, with each domain’s physical knowledge mutually reinforcing the others. Asymmetric 3D RoPE positional encoding and multi-view concatenation training enable geometrically consistent synthesis across synchronized camera views without architectural modification.

Evaluated on four established benchmarks, QWEN-ROBOTWORLD achieves competitive performance across cross-scenario and cross-task settings. It outperforms all open-source models on WorldModelBench (8.99, 3rd overall), attaining perfect physics adherence scores across Newton’s laws, mass conservation, fluid dynamics, and gravity—on par with leading closed-source models—while achieving strong instruction following (2.33/3.0). It ranks 1st overall on EWMBench (4.60), with substantially leading motion fidelity in HSD (0.566, +33% over the runner-up) and top scene consistency (0.914). On DreamGen Bench, the model ranks 1st overall (4.952) across three robotic embodiment subsets, excelling in object-level compositional generalization. On PBench, it outperforms all open-source models (0.804), with domain understanding placing 3rd overall (0.857) and motion smoothness ranking 2nd among open-source models (0.990). Qualitative results further showcase generalization across cross-task video editing—including human-to-robot transfer, where the model synthesizes realistic robot execution from a human demonstration video without robot-specific prompting—as well as autonomous driving scene synthesis and room-scale indoor navigation generation; additional zero-shot performance on RoboTwin-IF benchmark further support robust transfer under complex instructions.

## Our contributions are summarized as follows:

• Framework. We propose QWEN-ROBOTWORLD, a language-conditioned video world model that treats natural language as a universal action interface to unify cross-scenario and cross-task embodied capabilities. By jointly training manipulation, driving, navigation, and human-to-robot transfer under a shared language interface, the model achieves complementary physical generalization that no single-domain model can match.

• Data. We propose an action-language mapping framework that standardizes 20+ robot embodiment types and 500+ action categories into a unified natural language interface, and construct approximately 8.6M high-quality, cross-scenario, cross-task embodied video-text pairs constituting the EWK dataset.

• Training. We propose a general + expert joint training paradigm that, under the unified natural language interface, equips the model with both broad world modeling capability and deep embodied domain expertise, enabling stable and scalable co-training across diverse scenarios and tasks.

• Performance. QWEN-ROBOTWORLD achieves comprehensive improvements on cross-scenario and cross-task embodied evaluation metrics, ranking 1st overall on EWMBench and DreamGen Bench and outperforming all open-source models on WorldModelBench and PBench.

## 2 Data

The central challenge in training a universal embodied world model is not data scale alone, but representational heterogeneity: robotic manipulation actions are expressed as joint angles or end-effector waypoints, driving as steering commands and velocity profiles, and navigation as heading vectors—each requiring a separate model or interface per domain. We resolve this through an action-language mapping framework that converts heterogeneous actions from 20+ robot embodiment types and 500+ action categories into a unified natural language interface. Under this unified interface, videos from a Franka gripper, an autonomous vehicle, and an indoor navigation agent all become instances of the same language-conditioned video generation task, enabling cross-scenario and cross-task joint training under a single model without any domain-specific control interface. As shown in Figure 1, this framework produces approximately 6M high-quality, cross-scenario, cross-task embodied video–text pairs, which we further augment with general video data (30% of the total) to construct the Embodied World Knowledge (EWK) dataset: a corpus of 8.6M video–text pairs comprising over 200M observation frames.

## 2.1 Action-Language Mapping

The action-language mapping framework addresses a fundamental asymmetry in embodied data: the visual states (video frames) are already in a common pixel space, but the action representations are fragmented across incompatible modalities. Our framework resolves this by projecting all action signals onto a shared natural language space, so that the same diffusion transformer can learn $s _ { t + 1 } = f ( \bar { s } _ { t } , a _ { t } )$ regardless of the underlying physical domain.

Why Language as the Unified Action Interface. Unlike low-level action representations—joint angles, end-effector waypoints, force-torque commands—which are hardware-specific and require a separate control interface per embodiment, natural language offers a universal, embodiment-agnostic action interface. A single instruction such as “grasp the red cup and lift it vertically” implicitly encodes the full action sequence, goal state, and physical constraints, without any knowledge of the underlying kinematic chain. By training the model to predict the next visual state $s _ { t + 1 }$ from a language action at alone, we obtain a simulation backbone that generalizes across embodiments—whether a Franka gripper, an Aloha dual-arm system, or a humanoid—without retraining or re-engineering robot-specific interfaces. This generality, however, places demanding requirements on annotation quality: each caption must function as a complete, self-contained action specification, precise enough that the model can predict $s _ { t + 1 }$ from $a _ { t }$ and st alone, without access to any robot metadata or proprioceptive signals.

![](images/bb58c2662d5d905004ca0ee165365d927cade911d88ca9f967fc94ff39efe4ea.jpg)  
Figure 1: Overview of the Embodied World Knowledge (EWK) training corpus. General world data (top) supplies foundational priors on appearance, geometry, and dynamics from internet-scale video and image collections. Structured embodied data (middle) is organized along four complementary axes, each targeting a distinct source of physical variation: Multi-Embodiment (human hands, diverse robot manipulators, mobile agents); Multi-Task (short-horizon atomic skills, long-horizon compositional planning, specific skills such as locomotion and HRI); Multi-Scenario, a reality-first, sim-augmented design that bridges real captures and the simulators where downstream VLA policies are trained and evaluated; and Multi-View (main, wrist, and synchronized multi-view streams covering both global planning and fine-grained effector–object interaction). Jointly, these signals supply the semantics, geometry, physical alignment, and causal relationships (bottom) required for language-conditioned action understanding and future-state generation.

Hierarchical Five-Layer Annotation. To consistently produce such action-rich captions across 20+ robot embodiment types and 500+ action categories, we design a hierarchical annotation framework with five progressive layers. The first three form a structured chain-of-thought that decomposes each visual state transition into interpretable components:

1. Task Goal Layer—infer the high-level intent of the transition (what should change between st and $s _ { t + 1 } )$ , integrating external instructions with observed video content;

2. Action Detail Layer—decompose the action at into spatio-temporal trajectories, micro-actions, speed, and force, with mandatory explicit declaration of viewpoint information (egocentric main view, wrist view, external view, or concatenated multi-view combinations);

3. Physical Feedback Layer—describe the observable consequences of the action on the environment (object displacement, deformation, contact state changes), grounding each transition in verifiable physical outcomes.

Based on this analysis, two granularities of action descriptions are generated:

4. Comprehensive Description (50–100 words)—fully specifies the viewpoint–agent–action–feedback quadruple, providing a rich action signal for precise state transition prediction;

5. Concise Description (15–30 words)—retains only the essential viewpoint–agent–key action elements, enabling the model to handle brief, high-level commands at inference time.

Table 1: Detailed inventory of the Embodied World Knowledge (EWK) training data mixture, organized by domain.
<table><tr><td>Dataset</td><td>Embodiment</td><td>Views</td><td>Tasks</td><td>Contribution</td></tr><tr><td colspan="5">A. Manipulation</td></tr><tr><td>EgoHOD Pei et al. (2025), EPIC-Kitchens Damen et al. (2018), Egocentric-10k Build</td><td>Human hands</td><td>Egocentric</td><td>Daily grasping &amp; kitchen</td><td>Dexterity &amp; coordination prior</td></tr><tr><td>AI (2025) Bridge V2 Walke et al. (2023), RH20T Fang et al. (2024), Droid Khazatsky</td><td>Single-arm grippers</td><td>external + wrist</td><td>Tabletop pick-and-place</td><td>Interaction primitives</td></tr><tr><td>et al. (2024) Robomind Wu et al. (2025a), RoboCoin Wu et al. (2025b) Agibot-World AgiBot-</td><td>Single/dual-arm, humanoids</td><td>Ego + external</td><td>Rigid&amp; deformable objects</td><td>Cross-embodiment generalization Temporal &amp; multi-view</td></tr><tr><td>World-Contributors (2025), Galaxea Galaxea AI (2025)</td><td>Single-arm (gripper + dexterous hand)</td><td>Synced ego + wrist + external</td><td>Long-horizon sequential</td><td>consistency</td></tr><tr><td>Qwen-Aloha (internal) ActionNet Fourier</td><td>Dual-arm grippers Dexterous hands</td><td>Head + dual wrist Wrist + external</td><td>Diverse grasping Tool use &amp; in-hand</td><td>Multi-view grasping prior Fine-grained dexterity</td></tr><tr><td>ActionNet Team &amp;Mu (2025), OpenLoong OpenLoong Baihu Team (2025)</td><td></td><td></td><td></td><td></td></tr><tr><td>InternData-A1 Tian et al. (2025), Robotwin Chen et al. (2025b), Groot-XE Bjorck et al. (2025), RT1 Brohan et al. (2023)</td><td>Mixed arms (simulated)</td><td>Variable</td><td>Fluids &amp; deformables</td><td>Sim-to-real alignment</td></tr><tr><td colspan="5">B.Autonomous Driving Waymo E2E Waymo Team</td></tr><tr><td>(2024),NVIDIA PhysicalAI-AD NVIDIA</td><td>Ego vehicle</td><td>5-8 surround-view</td><td>Urban driving &amp; traffic</td><td>Large-scale motion &amp; 3D geometry</td></tr><tr><td>(2025b) Bench2Drive Jia et al. (2024)</td><td>Ego vehicle (sim)</td><td>6 surround-view</td><td>9.8K traffic scenarios</td><td>Sim diversity &amp; GT</td></tr><tr><td>Sekai Sekai Team (2025)</td><td>Pedestrian /drone</td><td>Egocentric</td><td> Urban walking</td><td>annotations Pedestrian-scale locomotion</td></tr><tr><td colspan="5">C. Indoor Navigation</td></tr><tr><td>VLNVerse Lin et al. (2025)</td><td>Mobile agent</td><td>Egocentric</td><td>134 indoor scenes, lang-guided</td><td>3D reasoning &amp; lang-trajectory align</td></tr><tr><td colspan="5">D. Human-to-Robot Transfer</td></tr><tr><td>Paired H2R dataset</td><td>Human →14 robot arms</td><td>Egocentric bimanual</td><td>Cross-embodiment manipulation</td><td>Video editing supervision</td></tr></table>

We enforce four quality control principles: operation focus (only agent actions and object interactions), viewpoint definition (explicit viewpoint type and semantic role), objectivity (only visible dynamics), and physical verifiability (only visually verifiable outcomes). In training, we sample from comprehensive and concise descriptions with equal probability (50% each), so the model learns to execute both detailed trajectory specifications and brief task-level commands.

Coverage: 20+ Robot Embodiments, 500+ Action Categories. The framework is applied across all data domains. On the embodiment axis, it covers human hands, seven robot arm configurations (single-arm gripper, dual-arm gripper, single-arm dexterous hand, dual-arm dexterous hand, mobile dual-arm, half-humanoid, and full humanoid), ego vehicle (surround-view cameras), pedestrian/drone, and mobile navigation agent—representing 20+ distinct robot embodiments in total, as sourced from RoboCoin (15 robot models across three structural categories), Robomind (4 morphologies), InternData-A1 (4 robot models), Groot-XE, and various other datasets. On the action axis, it spans 500+ action categories derived from the explicit motion primitive vocabularies across our training datasets—Agibot-World alone defines 84 distinct manipulation primitives (grasp, push, pour, fold, wipe, cut, etc.)—supplemented by unique primitives from other manipulation datasets and locomotion/navigation actions (turning, lane-changing, waypoint following, obstacle avoidance, etc.), organized into four tiers: (1) manipulation primitives, (2) long-horizon compositions, (3) locomotion and navigation, and (4) dynamic and deformable interactions. This systematic coverage ensures that the resulting embodied video-text pairs span a semantically rich and physically diverse action space that no single domain could provide.

## 2.2 Data Collection

## 2.2.1 General Data

General world data lays the foundation for the model to grasp basic physical laws and form accurate visual representations. This category encompasses diverse videos and still images from the internet. Video data are standardized to 24 FPS and support multiple resolutions and aspect ratios (1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, etc.). Image data integrates high-quality static photographs, serving as visual quality anchors that establish precise representations of object appearance, material texture, and spatial composition. All general data is annotated with natural language descriptions generated by Qwen2.5-VL Bai et al. (2025); annotations omit viewpoint-specific information to maintain flexibility and generality. Notably, we adopt a conservative stance on AI-generated content (AIGC): general data excludes AI-produced images and videos, as these often introduce visual artifacts, physical inconsistencies, and implicit biases that could undermine the model’s generalization capabilities.

## 2.2.2 Embodied Manipulation Data

To enable the world model to acquire grounded physical understanding across scenarios and tasks, we build a structured data mixture spanning manipulation, driving, navigation, and cross-embodiment transfer domains, as summarized in Table 1. For the core manipulation domain, we organize the data around four dimensions: Multi-Embodiment, Multi-Task, Multi-Scenario, and Multi-View.

Multi-Embodiment. Manipulation data spans a spectrum of embodiments—human hands, single-arm grippers, dual-arm dexterous systems, mobile manipulators, and full-body humanoids—so the model learns to separate task-level intent from embodiment-specific kinematics. Human manipulation data (EgoHOD Pei et al. (2025), EPIC-Kitchens Damen et al. (2018)) provides a dexterity ceiling: the model observes what physically capable interaction looks like, acquiring priors for fluid hand–eye coordination and tool use. Robot data then teaches the model how those same intents map onto diverse mechanical morphologies. By exposing the model to both human demonstrations and robot executions of overlapping tasks (e.g., Robomind Wu et al. (2025a), RoboCoin Wu et al. (2025b)), it learns embodiment-invariant action semantics—the ability to predict “what should happen next” regardless of whether the actor is a two-finger gripper or a seven-finger dexterous hand.

Multi-Task. The manipulation corpus covers a skill hierarchy from atomic contact-level actions to extended multi-step procedures, teaching the model to operate at multiple temporal granularities. Shorthorizon datasets (Bridge V2 Walke et al. (2023), RH20T Fang et al. (2024)) provide dense coverage of fundamental interaction primitives—grasping, pushing, inserting—that ground the model’s understanding of contact physics and object affordances. Long-horizon datasets (Agibot-World AgiBot-World-Contributors (2025), Galaxea Galaxea AI (2025)) chain these primitives into coherent sequences, forcing the model to maintain state tracking and causal reasoning across dozens of steps. Additionally, dynamic-interaction datasets (Humanoid Everyday Zhao et al. (2025)) introduce high-velocity, whole-body motions that test the model’s ability to predict outcomes under significant momentum and balance constraints. Together, this range ensures the model can reason about both “what happens when you press here” and “what happens after ten sequential decisions.”

Multi-Scenario. Multi-scenario coverage advances along two complementary axes: breadth across real environments, and extension to simulator-rendered scenarios. Along the first axis, physical interaction manifests differently depending on context—a kitchen counter presents different lighting, clutter density, and surface properties than a factory floor or an outdoor worksite. Our manipulation data is therefore predominantly real-world, spanning domestic kitchens, workshops, laboratories, and unstructured outdoor settings, exposing the model to genuine variation in illumination, occlusion, material appearance, and background complexity—so it does not brittly overfit to any single environment. Along the second axis, we incorporate photorealistic simulation data (InternData-A1 Tian et al. (2025)) as a first-class complement. This is motivated by the VLA landscape: a substantial portion of policy models are trained in simulators, and virtually all are evaluated there using standardized benchmarks such as LIBERO Liu et al. (2023), SimplerEnv Li et al. (2024), and RLBench James et al. (2020). A world model intended as a general simulation backbone must therefore generate faithfully under simulator-style appearances and physics, bridging the visual domain gap between real and synthetic observations so it can serve both sim-to-real transfer and closed-loop evaluation pipelines. The simulation portion additionally supplies precisely controlled variations in lighting, object pose, and camera placement that further strengthen visual robustness.

Multi-View. Single-view data teaches the model to predict plausible futures from a fixed perspective, but many physically critical events are partially or fully occluded from any single camera. Synchronized multi-view recordings (Agibot-World AgiBot-World-Contributors (2025), Robomind Wu et al. (2025a)) expose the model to the same event from head-mounted, wrist-mounted, and external viewpoints simultaneously. This serves two purposes: during training, cross-view correspondence acts as a geometric regularizer, implicitly teaching the model about object shape, depth, and spatial relationships; at inference, the model can generate from any individual viewpoint or compose multi-view outputs that remain mutually consistent. Approximately 1.6M of our 6M embodied samples include synchronized 2–4 view concatenations, providing substantial multi-view supervision without dominating the corpus.

## 2.2.3 Autonomous Driving Data

While manipulation data captures fine-grained object interactions within a confined workspace, autonomous driving data exposes the model to a substantially larger motion space with diverse maneuvers (turning, lane changing, acceleration) spanning a much wider range of velocities and trajectories. Driving scenes also contain rich multi-agent dynamics—surrounding vehicles, pedestrians, and cyclists interacting under traffic rules—requiring the world model to learn how multiple objects move, occlude, and influence each other over time. Furthermore, the large camera displacement provides dense supervisory signal for 3D scene geometry through parallax and perspective changes, strengthening the model’s capacity for view synthesis and spatial reasoning.

We curate multi-view driving videos from four large-scale datasets: Waymo E2E Waymo Team (2024) (real-world driving, 8 surround-view cameras, 7,044 clips / 11.3h), NVIDIA PhysicalAI-AD NVIDIA (2025b) (real-world driving, 5 cameras with 30◦–120◦ FoV, 1,342,418 clips / 1,715.9h), Bench2Drive Jia et al. (2024) (CARLA-simulated driving under 9,881 diverse traffic scenarios, 6 cameras, 384,948 clips / 511.2h), and Sekai Sekai Team (2025) (egocentric pedestrian walking and drone videos, 9,995 clips / 166.6h with scene and weather annotations). In total, the driving data comprises 1,744,405 clips spanning 2,405 hours. We apply a unified three-stage processing pipeline: (1) frame extraction with trajectory unification into a common waypoint format, (2) action-based clip segmentation (2–8 s) according to ego maneuver transitions, and (3) caption generation combining structured trajectory descriptions with optional VLM augmentation.

## 2.2.4 Egocentric Indoor Navigation Data

Egocentric indoor navigation data provides a complementary perspective to both manipulation and driving data. Unlike manipulation which focuses on fine-grained object interactions within a confined workspace, and driving which operates in large-scale outdoor environments, indoor navigation requires the model to understand room-scale spatial layouts, obstacle-aware path planning, and the mapping from textual navigation commands to spatially coherent visual trajectories.

Following VLNVerse Lin et al. (2025), we collect physically grounded egocentric navigation data using NVIDIA Isaac Sim NVIDIA (2022) with photorealistic rendering and continuous control. We gather 6,064 successful navigation episodes across 134 indoor scenes, each consisting of an egocentric RGB video (256 × 256 resolution at 10 FPS) paired with natural language navigation instructions. The trajectories average approximately 8.2 m in length (ranging from 4 to 17.5 m), accumulating a total traversal distance of roughly 49.8 km and approximately 5.8 hours of continuous first-person navigation video. The instructions are provided in two formats: single-string step-by-step directives (3,031 episodes, averaging 67.2 words) and multi-granularity descriptions at formal, natural, and casual registers (3,033 episodes).

Video generation models trained on such traversal data can acquire emergent 3D consistency and spatial coherence across frames Gao et al. (2026); Bar et al. (2025), while the physically grounded, actionconditioned nature of each sequence encourages the model to internalize depth reasoning, geometric consistency, and obstacle-aware planning Shang et al. (2025); Han et al. (2025); Zhen et al. (2025). By grounding language instructions in continuous egocentric traversals, this data enables the world model to jointly learn language understanding, 3D spatial reasoning, and embodied action prediction within indoor environments.

## 2.2.5 Human-to-Robot Transfer Data

To train the model on cross-embodiment visual correspondence without physical robot collection, we curate two complementary sources of human-to-robot transfer data. The first is a large-scale human-robot paired dataset constructed from egocentric bimanual manipulation recordings via an automated pipeline: 3D hand keypoints are extracted through MANO Romero et al. (2017) reconstruction and retargeted to robot end-effector trajectories, human hands are removed via video inpainting, and 14 robot arm models

![](images/84e6bde397c9f505aa6a6191f12c83840fe2d44b834a4fd3c74d65f52687082c.jpg)

Figure 2: Overview of the unified data processing pipeline. Stage 1 (Raw Data Collection) collects heterogeneous data from five source categories spanning general and embodied domains. Stage 2 (Video Preprocessing) applies domain-adaptive operations—frame extraction, frame interpolation, sub-task splitting, main-view selection, and multi-view concatenation—to produce uniformly structured clips. Stage 3 (Hierarchical Annotation) generates viewpoint-aware captions through a five-layer framework: task goal, action detail, physical feedback, comprehensive caption, and concise caption. Stage 4 (Caption Quality Filtering) combines an automated LLM-based judge with human evaluation; underperforming captions are routed back for scenario-, task-, or embodiment-specific iterative prompt refinement.

are rendered into the inpainted scene using MuJoCo Todorov et al. (2012) inverse kinematics, yielding four aligned video streams per episode (original human video, hand-removed scene, pure simulation, and robot-overlaid scene). The diversity of 14 embodiments within shared scenes ensures the editing capability generalizes across robot morphologies.

The second source addresses a fundamental limitation of direct rendering: simplified renderers ignore scene illumination, cast shadows, and material-dependent specular reflections, creating a photometric gap between rendered and real observations. To bridge this, we build upon the open-sourced InternA1 dataset Tian et al. (2025), which uses NVIDIA Isaac Sim NVIDIA (2022) to provide photorealistic RGB observations with environment lighting and accurate shadows. Using the same dynamics parameters and robot URDFs, we render matched egocentric views in MuJoCo Todorov et al. (2012)—without lighting or shadow effects—producing paired samples that share identical geometry and viewpoint while differing in photometric realism. This paired data enables the model to learn the visual mapping between simplified rendering and photorealistic observations, covering Franka Emika Panda, AgileX Split Aloha, ARX Lift2, and AgiBot Genie1 across single-arm, dual-arm, mobile dual-arm, and humanoid configurations, with approximately 80K episodes spanning pick-and-place, articulated object manipulation, and multi-object rearrangement tasks.

## 2.3 Data Processing

We design a unified data processing pipeline that transforms heterogeneous raw data from diverse embodied and general video sources into high-quality, consistently formatted training samples. As illustrated in Figure 2, the pipeline consists of four stages: (1) Raw Data Collection, (2) Video Preprocessing, (3) Hierarchical Annotation, and (4) Caption Quality Filtering with Iterative Prompt Refinement. Stages 2 and 3 apply domain-adaptive operations depending on source data characteristics, while Stage 4 forms a closed feedback loop that routes underperforming captions back for targeted re-annotation.

## 2.3.1 Stage 1: Raw Data Collection

The pipeline begins by ingesting raw video data from five source categories spanning both general and embodied domains. General Video provides internet-scale visual diversity from documentaries, professional stock libraries, and curated web clips. Manipulation data covers a broad spectrum of robot embodiments—single-arm grippers, dual-arm systems, dexterous hands, mobile platforms, and humanoids—from datasets including EgoHOD, Bridge V2, DROID, RoboMind, Agibot-World, and others. Autonomous Driving contributes large-scale ego-motion and multi-agent dynamics from Waymo, Bench2Drive, NVIDIA PhysicalAI-AD, and Sekai. Indoor Navigation supplies language-guided spatial reasoning episodes from VLNVerse across 134 indoor scenes. Human-to-Robot Transfer provides paired human demonstration and robot execution data constructed via our automated MANO-to-robot pipeline across 14 robot types.

## 2.3.2 Stage 2: Video Preprocessing

Raw videos undergo domain-adaptive preprocessing to produce uniformly structured clips suitable for training. We apply five complementary operations depending on the source data characteristics:

Frame Extraction. For short-horizon task videos (typically single-step manipulations lasting 2–8 s), we extract frames at a target rate that captures the essential phases of the interaction—approach, contact, manipulation, and result—ensuring each sample contains the complete causal chain of the atomic action.

Frame Interpolation. When source videos have insufficient frame rates for smooth motion learning, we apply temporal interpolation to increase frame density, preserving continuous motion trajectories critical for modeling fine-grained contact dynamics and object-state transitions.

Sub-task Splitting. For long-horizon episodes involving multi-step procedures (e.g., sequential pick-andplace, complex assembly), we decompose the video into semantically coherent sub-task segments. Each segment captures a complete atomic action with clear start and end states, preventing the partial-execution artifacts that arise from naive uniform truncation.

Main-View Selection. For multi-camera recordings where only a primary viewpoint is needed (e.g., single-view manipulation training), we select the most informative camera stream—typically the egocentric or external view that best captures the interaction region—discarding redundant angles.

Multi-View Concatenation. Conversely, for multi-view co-training, we concatenate synchronized clips from 2–4 camera viewpoints into a single horizontal layout, preserving temporal alignment across views. This enables the model to learn cross-view geometric consistency and synchronized state transitions without architectural modifications.

## 2.3.3 Stage 3: Hierarchical Annotation

## Hierarchical Annotation Prompt Template

You are an expert embodied-AI annotator. Given a video clip of a {{embodiment\_type}} performing a manipulation task captured from a {{viewpoint\_type}} viewpoint, produce the following five-layer annotation.

Layer 1 – Task Goal: Identify the high-level intent of this interaction. What is the agent trying to achieve? Describe the desired state transition from the current observation to the goal state in one sentence.

Layer 2 – Action Detail: Decompose the agent’s actions into a step-by-step sequence. For each step, specify: (a) the motion trajectory and direction, (b) micro-actions (approach, grasp, lift, rotate, release, etc.), (c) estimated speed and force level. You must explicitly state the viewpoint (egocentric / wrist / external / multi-view concatenation).

Layer 3 – Physical Feedback: Describe the observable physical consequences of each action on the environment: object displacement, deformation, contact state changes, and any secondary effects (e.g., liquid sloshing, cloth folding). Only include visually verifiable outcomes.

— Generation Phase —

Layer 4 – Comprehensive Caption (50–100 words): Synthesize Layers 1–3 into a cohesive paragraph that fully specifies the viewpoint–agent–action–feedback quadruple. Include the camera perspective, embodiment identity, complete action sequence, and physical outcomes.

Layer 5 – Concise Caption (15–30 words): Condense to an instruction-style summary retaining only the essential viewpoint–agent–key action elements, suitable as a direct language command for the world model.

Quality Constraints:

• Operation focus: describe only agent actions and object interactions; omit background narration.

• Viewpoint definition: explicitly name the viewpoint type and its semantic role.

• Objectivity: report only visible dynamics; do not infer hidden states.

• Physical verifiability: every claimed outcome must be visually confirmable from the video.

Preprocessed videos pass through our five-layer hierarchical annotation framework (Section 2.1), which generates viewpoint-aware captions at two granularities—comprehensive (50–100 words) and concise (15–30 words)—sampled with equal probability during training. The prompt template used by the annotation model is shown above.

## 2.3.4 Stage 4: Caption Quality Filtering

To ensure annotation quality across the diverse range of scenarios, tasks, and embodiments in our corpus, we implement a closed-loop quality filtering system combining automated assessment with human oversight. Captions that fail quality checks are routed back to Stage 3 for targeted re-annotation, forming an iterative refinement loop.

Judge Pipeline. An automated LLM-based judge assesses each caption along several dimensions, including factual accuracy, specificity, instruction clarity, and viewpoint consistency. Specifically, it evaluates whether the caption correctly describes the video content, provides sufficient detail beyond generic descriptions, can function as an actionable command, and maintains spatial references consistent with the camera perspective. Captions that do not satisfy any of these criteria are flagged for further review.

Human Evaluation. A subset of captions—particularly those near judgment thresholds or from underrepresented domains—undergoes manual review by human annotators who validate correctness, identify systematic failure patterns, and provide ground-truth corrections that inform subsequent prompt refinements.

Iterative Prompt Refinement. When the judge pipeline identifies consistent underperformance in specific categories, we trigger targeted prompt redesign along three axes: scenario-specific retries (e.g., outdoor lighting conditions, kitchen environments), task-specific retries (e.g., articulated object manipulation, fluid pouring), and embodiment-specific retries (e.g., humanoid bimanual coordination, dexterous hand manipulation). Each retry employs a specialized prompt template tailored to the failure mode, and the refined captions are re-evaluated through the judge pipeline until they meet quality standards. This iterative loop ensures that no scenario, task, or embodiment category suffers from systematically poor annotations due to one-size-fits-all prompting.

Final Corpus Statistics. After the complete four-stage pipeline, the final training corpus comprises approximately 8.6M video-text pairs (over 200M observation frames), with embodied data accounting for 70% and general data for 30%. Within the embodied portion, single-view manipulation data constitutes the majority at ∼4.3M samples, followed by ∼1.6M multi-view concatenated samples with synchronized 2–4 camera views, and ∼200K navigation and driving samples.

## 3 Model

## 3.1 Model Architecture

![](images/a78675c3ec431d77d38caed53dc2d4635107760fcef134b3c96671029c6cfae1.jpg)  
Figure 3: Overview of our video generation architecture with 60-layer double-stream MMDiT backbone.  
As shown in Figure 3, the model consists of three components: an MLLM as the action encoder, a VAE as the state encoder/decoder, and an MMDiT Esser et al. (2024) as the transition function, organized in a

![](images/0214136e08114bef4f11c1a457b262213aa543c9aeb3c0c139d15194fbfb39ba.jpg)  
Figure 4: Scene2Robot: multi-segment conditioning for cross-embodiment video synthesis. The input sequence is organized as three contiguous segments — scene condition (F frames), robot reference (F frames), and generation (F frames). An index-based mechanism assigns condition tokens to timestep t = 0 and excludes them from loss computation, so only the generation segment is trainable. Joint attention at every MMDiT block enables the generation segment to simultaneously attend to scene appearance and robot motion trajectory, producing semantically coherent cross-embodiment synthesis.

double-stream design.

MLLM — Action Encoder. We employ a frozen Qwen2.5-VL Bai et al. (2025) to encode user inputs into condition signals. For a given input text S, it extracts last-layer hidden states h = ϕ(S), serving as the action condition.

VAE — State Encoder/Decoder. The VAE encodes video frames into latent representations z = E(x) and decodes predicted latents back into visual observations. We adopt the Wan-VAE Wan et al. (2025) architecture, which handles both image and video modalities.

MMDiT — Transition Function. The MMDiT adopts a double-stream architecture: the understanding stream receives the MLLM encoding h (projected via a trainable connector), and the generation stream receives noisy state latents from the VAE. At each block, the two streams interact via joint attention. The backbone comprises 60 double-stream blocks with 24 attention heads (head dimension 128), hidden size 3,072, and patch size 2×2. Total parameters: MLLM 7B, VAE 127M (encoder 54M + decoder 73M), MMDiT 20B. The context length supports up to 48,360 video tokens.

## 3.2 3D Rotary Position Encoding

We employ 3D RoPE Su et al. (2024); Heo et al. (2024) to independently encode the temporal, spatial height, and spatial width dimensions. Rather than allocating dimensions uniformly, we use an asymmetric split: 16 dimensions for the temporal axis and 56 dimensions each for height and width, totaling 128 dimensions (pe\_axes\_dim = [16, 56, 56]). The temporal axis receives fewer dimensions as adjacent frames are strongly correlated; the spatial axes receive more to capture the greater diversity of object positions and scene layouts. We also apply Scalable RoPE Wan et al. (2025) to support generalization to varying resolutions and durations at inference.

## 3.3 Scene2Robot

Building upon the double-stream MMDiT architecture (§3.1) and the asymmetric 3D RoPE encoding (§3.2), we design SCENE2ROBOT, a multi-segment conditioning mechanism that repurposes the same backbone for cross-embodiment video synthesis, as illustrated in Figure 4.

First-Frame Conditioning (TI2V Baseline). For standard text-image-to-video tasks, the first frame serves as a fixed visual condition: its VAE latents are assigned timestep t=0 in the generation stream and excluded from the denoising loss, while the frozen Qwen2.5-VL encodes the text instruction into the understanding stream. Because the double-stream joint attention (§3.1) fuses both signals at every layer, the generation tokens can simultaneously attend to the visual anchor and the semantic action specification, producing temporally coherent continuations grounded in the language command.

Multi-Segment Extension for Human-to-Robot Transfer. Human-to-robot transfer poses a video editing problem: the model must reference both the scene context (background, object layout, lighting) and the target robot’s motion trajectory from a simulated demonstration. We address this by extending first-frame conditioning to a three-segment input sequence, all processed within the same VAE–MMDiT pipeline without any architectural modification:

1. Scene condition (F frames): the original human demonstration video, with human hands masked out, encoded by the VAE to provide appearance, spatial layout, and object state information.

2. Robot reference (F frames): a simulated robot execution rendered via MuJoCo, encoded by the VAE, supplying the target embodiment’s kinematic trajectory and morphology.

3. Generation (F frames): noisy latents to be denoised into the final photorealistic robot execution video.

Segments (1) and (2) share the same t=0 assignment as first-frame conditioning and are excluded from loss computation; only segment (3) receives gradient updates during training. The 3D RoPE encoding (§3.2) assigns each segment its own temporal index range, allowing the model to distinguish temporal positions across segments. Joint attention in every MMDiT block then enables the generation tokens to simultaneously attend to scene appearance from segment (1), robot motion from segment (2), and the MLLM action semantics from the understanding stream. This tripartite conditioning enables the model to synthesize photorealistic robot executions that faithfully preserve both the scene context and the instructed manipulation behavior.

## 4 Training

## 4.1 Training Strategy

We propose a joint training paradigm in which general scene generation and robot manipulation prediction are unified under a single natural language interface as the same conditional video generation task, with the model continuously receiving gradient updates from both data regimes throughout training. This shared formulation allows general world priors and embodied action priors to reinforce each other through a common backbone, enabling stable cross-scenario and cross-task co-training. The curriculum proceeds in two progressive stages: pretraining establishes broad world foundations, and SFT deepens embodied specialization while preserving the general-expert balance.

## 4.1.1 Pretraining Stage: Establishing General World Foundation

General World Priors. We curate over 200M real-world observation samples from 14 high-quality video platforms, covering natural scenes, daily life, and sports. This breadth allows the model to internalize domain-agnostic world priors—object motion, lighting variation, collision dynamics—that form the general backbone for later embodied generalization. We further incorporate multi-camera synchronized observations with 3D RoPE spatial encoding, establishing preliminary cross-view geometric consistency as a spatial foundation for multi-view embodied generation.

Human Interaction Priors. We introduce large-scale first-person hand manipulation data (Ego4D Grauman et al. (2022), EPIC-Kitchen Damen et al. (2018), etc.). Human demonstration serves as a natural bridge between general and embodied: by learning grasping, tool use, and object manipulation from everyday human behavior, the model builds action priors and affordance understanding that transfer directly to robot operation in later stages.

Multi-Task Joint Training. T2I, T2V, and TI2V tasks are trained jointly on a shared backbone, serving as the core mechanism through which general and embodied capabilities coexist in one model. The T2I task learns sharp visual representations from general image data, acting as a visual quality anchor whose object morphology knowledge automatically transfers to video generation tasks through the shared backbone, preventing deformation and identity inconsistency. Task ratios gradually shift from pure T2I toward full three-task joint training, so the model operates stably across multiple generation modes by the end of pretraining.

## 4.1.2 SFT Stage: Embodied Specialization

The SFT stage progressively deepens embodied expertise while keeping general world data in every training batch, ensuring that embodied specialization and general world modeling capability advance together rather than trade off.

Progressive Embodied Knowledge Injection. We adopt a four-phase data mixing schedule. In early training, multi-embodiment robot data and human hand manipulation data co-dominate: human action priors guide the learning of cross-embodiment operation commonalities, while robot data strengthens concrete execution representations. We then gradually increase wrist-view and third-person view data to broaden viewpoint coverage. Building on this, we introduce multi-view concatenated training: synchronized first frames from multiple cameras are spatially concatenated as a single input, requiring the model to jointly generate subsequent frames for all views simultaneously, forcing the attention layers to establish cross-view spatial correspondences and achieve geometrically consistent multi-view generation. In the final phase, scarce high-complexity tasks (pouring, folding, bimanual coordination, multi-material interaction) and long-horizon reasoning data are targeted for supplementation to push the frontier of embodied capability. Throughout this process, general world data continuously participates in every training batch, jointly acting on the same backbone alongside embodied data to ensure that embodied specialization and general world modeling capability advance together.

## 4.2 Training Objective and Infrastructure

We adopt the flow matching objective Lipman et al. (2023); Liu et al. (2022), where input videos are encoded into latent space via the VAE encoder and noise is sampled from a standard normal distribution. Qwen2.5-VL encodes text inputs as guidance signal. Timesteps are sampled from a log-normal distribution with adaptive shifting based on video sequence length Esser et al. (2024). For TI2V tasks, the first-frame timestep is fixed at 0 to ensure that the generation process is conditioned on the given observation frame. Training is conducted with Megatron-LM Shoeybi et al. (2019) using a hybrid parallelism strategy, with selective activation recomputation Korthikanti et al. (2023) applied to a subset of dual-stream blocks to balance memory usage and training throughput.

## 5 Experiments

We conduct comprehensive evaluations on four benchmarks spanning embodied manipulation, physical reasoning, and general video quality. Across these benchmarks, our model delivers consistently strong results, achieving state-of-the-art performance on EWMBench for embodied world modeling (Overall 4.60, +0.55 over LVP), ranking 1st overall on DreamGen Bench (Total 4.952), and 1st among open-source models on WorldModelBench (Total 8.99).

Quantitative Evaluation (§5.1). We evaluate against two categories of baselines: (1) general video generation models—Sora2 OpenAI (2024), Veo3 Google DeepMind (2025), Wan2.6 Wan Team (2025), Kling Kuaishou Technology (2024), and LTX-2 Lightricks (2025); and (2) embodied world models—Cosmos Agarwal et al. (2025), WoW Chi et al. (2025), LVP Chen et al. (2025a), Vidar Feng et al. (2025), and GigaWorld Team et al. (2025).

Qualitative Analysis (§5.2). We evaluate manipulation capabilities along three progressive dimensions: fine-grained language grounding, generalization across embodiments, tasks, and viewpoints, and zeroshot robustness against strong baselines.

Cross-Domain Generalization (§5.3) further covers human-to-robot transfer, autonomous driving, and indoor navigation as supplementary tasks.

## 5.1 Quantitative Evaluation

Unless noted otherwise, quantitative tables use boldface for the best value in each column and underline for the second best.

## 5.1.1 EWMBench: Embodied Motion Fidelity

Benchmark. EWMBench Yue et al. (2025) evaluates embodied world models on three dimensions: scene consistency (SceneC), motion correctness (HSD, Dyn, nDTW), and semantic alignment (Diversity, BLEU, CLIP, Logics). The benchmark contains 21 samples across 7 tasks with clear action-ordering constraints.

Table 2: Performance comparison on EWMBench.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td>Scene</td><td colspan="3">Motion</td><td colspan="4">Semantics</td><td rowspan="2">Overall</td></tr><tr><td>SceneC</td><td>HSD</td><td>Dyn</td><td>nDTW</td><td>Diversity</td><td>BLEU</td><td>CLIP</td><td>Logics</td></tr><tr><td rowspan="5">General</td><td>Veo3</td><td>0.8415</td><td>0.2130</td><td>0.1932</td><td>0.1613</td><td>0.0221</td><td>0.2139</td><td>0.8965</td><td>0.9474</td><td>3.49</td></tr><tr><td>Wan2.6</td><td>0.6712</td><td>0.2034</td><td>0.0900</td><td>0.1715</td><td>0.0502</td><td>0.1616</td><td>0.8743</td><td>1.0000</td><td>3.22</td></tr><tr><td>Kling26</td><td>0.8211</td><td>0.3272</td><td>0.1822</td><td>0.3423</td><td>0.0173</td><td>0.2591</td><td>0.9014</td><td>1.0000</td><td>3.85</td></tr><tr><td>LTX-2</td><td>0.7850</td><td>0.2076</td><td>0.1283</td><td>0.2443</td><td>0.0120</td><td>0.1425</td><td>0.8869</td><td>0.5000</td><td>3.01</td></tr><tr><td>Sora2</td><td>0.8526</td><td>0.2807</td><td>0.3494</td><td>0.2754</td><td>0.0314</td><td>0.2466</td><td>0.9100</td><td>0.9474</td><td>3.89</td></tr><tr><td rowspan="5">Embodied</td><td>Cosmos</td><td>0.7963</td><td>0.2500</td><td>0.2052</td><td>0.2533</td><td>0.0803</td><td>0.1230</td><td>0.8458</td><td>0.7333</td><td>3.29</td></tr><tr><td>GigaWorld</td><td>0.8707</td><td>0.3050</td><td>0.0849</td><td>0.2783</td><td>0.0278</td><td>0.2048</td><td>0.8873</td><td>0.9000</td><td>3.56</td></tr><tr><td>LVP</td><td>0.8795</td><td>0.4248</td><td>0.0433</td><td>0.6226</td><td>0.0093</td><td>0.2179</td><td>0.8995</td><td>0.9524</td><td>4.05</td></tr><tr><td>Vidar</td><td>0.7341</td><td>0.1877</td><td>0.1520</td><td>0.1769</td><td>0.0653</td><td>0.1607</td><td>0.8821</td><td>0.9411</td><td>3.30</td></tr><tr><td>Wow</td><td>0.8866</td><td>0.2494</td><td>0.0529</td><td>0.2566</td><td>0.0266</td><td>0.1932</td><td>0.9001</td><td>0.9524</td><td>3.52</td></tr><tr><td></td><td>Ours</td><td>0.9142</td><td>0.5660</td><td>0.3429</td><td>0.6708</td><td>0.0114</td><td>0.2079</td><td>0.8834</td><td>1.0000</td><td>4.60</td></tr></table>

Results. Table 2 shows our model ranks 1st overall with a score of 4.60, outperforming the runner-up LVP (4.05) by +0.55. We lead in motion fidelity—HSD (0.566) surpasses LVP (0.425) by 33%—and achieve top performance in scene consistency (SceneC: 0.914) and logic constraint satisfaction (Logics: 1.00).

## 5.1.2 DreamGen Bench

Benchmark. DreamGen Bench Zhou et al. (2025) evaluates the quality of robot videos generated by video world models, measuring instruction following (IF) and physics alignment (PA) across three subsets of the GR1 robot embodiment: environment generalization (GR1-Env), object generalization (GR1-Object), and behavior generalization (GR1-Behavior). IF is assessed using Qwen2.5-VL Bai et al. (2025) as the evaluator.

Table 3: Performance comparison on DreamGen Bench.
<table><tr><td rowspan="2">Model</td><td colspan="2">GR1-Env</td><td colspan="2">GR1-Object</td><td colspan="2">GR1-Behavior</td><td rowspan="2">Total</td></tr><tr><td>PA</td><td>IF</td><td>PA</td><td>IF</td><td>PA</td><td>IF</td></tr><tr><td>Cosmos-sft</td><td>0.709</td><td>0.655</td><td>0.775</td><td>0.720</td><td>0.649</td><td>0.621</td><td>4.129</td></tr><tr><td>LVP</td><td>0.810</td><td>0.772</td><td>0.745</td><td>0.829</td><td>0.713</td><td>0.889</td><td>4.758</td></tr><tr><td>Vidar</td><td>0.445</td><td>0.647</td><td>0.478</td><td>0.726</td><td>0.394</td><td>0.651</td><td>3.341</td></tr><tr><td>GigaWorld</td><td>0.621</td><td>0.933</td><td>0.500</td><td>0.852</td><td>0.426</td><td>0.884</td><td>4.216</td></tr><tr><td>Wow</td><td>0.793</td><td>0.826</td><td>0.755</td><td>0.849</td><td>0.809</td><td>0.696</td><td>4.728</td></tr><tr><td>Ours</td><td>0.828</td><td>0.793</td><td>0.840</td><td>0.878</td><td>0.781</td><td>0.832</td><td>4.952</td></tr></table>

Results. Table 3 shows our model achieves the highest total score of 4.952, ranking 1st overall. We lead in GR1-Object IF (0.878, 1st), demonstrating strong object-level compositional generalization, and physics alignment is consistent across all subsets (PA: 0.828/0.840/0.781). GR1-Behavior IF (0.832) slightly trails LVP (0.889) and GigaWorld (0.884), indicating long-horizon behavior generalization as a direction for further improvement.

## 5.1.3 PBench: Physical Behavior Evaluation

Benchmark. PBench NVIDIA (2025a) evaluates models on two complementary aspects: (1) Domain Score, which measures physical behavior understanding via QA pairs assessed by Qwen2.5-VL across six domains (AV, Robot, Industry, Physics, Human, Common Sense); and (2) Quality Score, which measures visual quality via eight VBench Huang et al. (2024) metrics including image-to-video consistency, aesthetic quality, motion smoothness, and subject consistency. The Overall Score is the average of the two.

Results. As shown in Table 4, our model outperforms all among open-source models with an overall score of 0.804. Domain understanding is our strongest dimension (0.857, 3rd overall), surpassing most closed-source models. Motion smoothness also stands out (0.990, 2nd among open-source models), reflecting consistent temporal coherence in generation. Aesthetic quality (0.455) and imaging quality (0.649) are relatively lower, primarily because our model is purpose-built for embodied tasks and operates at a lower output resolution than general-purpose video generators, which reduces VBench’s pixel-level quality scores; nonetheless, this resolution is fully sufficient for downstream robot control tasks.

Table 4: Performance comparison on PBench.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="8">Quality Metrics (VBench)</td><td rowspan="2">Qual.</td><td rowspan="2">Domain</td><td rowspan="2">Overall</td></tr><tr><td>12V-Bg</td><td>I2V-S</td><td>Aes</td><td>Img</td><td>Bg-Con</td><td>Mot</td><td>Sub-Con</td><td>O-Con</td></tr><tr><td rowspan="5">General</td><td>Veo3</td><td>0.975</td><td>0.980</td><td>0.526</td><td>0.698</td><td>0.938</td><td>0.994</td><td>0.927</td><td>0.128</td><td>0.771</td><td>0.882</td><td>0.827</td></tr><tr><td>Wan2.6</td><td>0.856</td><td>0.843</td><td>0.514</td><td>0.719</td><td>0.906</td><td>0.978</td><td>0.843</td><td>0.136</td><td>0.724</td><td>0.832</td><td>0.778</td></tr><tr><td>Sora2</td><td>0.981</td><td>0.973</td><td>0.487</td><td>0.672</td><td>0.961</td><td>0.994</td><td>0.954</td><td>0.129</td><td>0.769</td><td>0.841</td><td>0.805</td></tr><tr><td>Kling26</td><td>0.982</td><td>0.979</td><td>0.521</td><td>0.699</td><td>0.920</td><td>0.990</td><td>0.927</td><td>0.124</td><td>0.768</td><td>0.874</td><td>0.821</td></tr><tr><td>LTX-2</td><td>0.948</td><td>0.955</td><td>0.506</td><td>0.622</td><td>0.932</td><td>0.986</td><td>0.904</td><td>0.118</td><td>0.746</td><td>0.845</td><td>0.796</td></tr><tr><td rowspan="5">Embodied</td><td>Cosmos</td><td>0.974</td><td>0.973</td><td>0.470</td><td>0.663</td><td>0.940</td><td>0.989</td><td>0.931</td><td>0.160</td><td>0.763</td><td>0.840</td><td>0.802</td></tr><tr><td>LVP</td><td>0.979</td><td>0.981</td><td>0.515</td><td>0.679</td><td>0.954</td><td>0.991</td><td>0.962</td><td>0.116</td><td>0.772</td><td>0.812</td><td>0.792</td></tr><tr><td>GigaWorld</td><td>0.957</td><td>0.944</td><td>0.495</td><td>0.641</td><td>0.925</td><td>0.984</td><td>0.892</td><td>0.128</td><td>0.746</td><td>0.841</td><td>0.794</td></tr><tr><td>Vidar</td><td>0.935</td><td>0.922</td><td>0.501</td><td>0.573</td><td>0.912</td><td>0.982</td><td>0.863</td><td>0.120</td><td>0.726</td><td>0.810</td><td>0.768</td></tr><tr><td>Wow</td><td>0.967</td><td>0.957</td><td>0.517</td><td>0.689</td><td>0.941</td><td>0.980</td><td>0.929</td><td>0.111</td><td>0.761</td><td>0.786</td><td>0.774</td></tr><tr><td></td><td>Ours</td><td>0.956</td><td>0.943</td><td>0.455</td><td>0.649</td><td>0.956</td><td>0.990</td><td>0.933</td><td>0.124</td><td>0.751</td><td>0.857</td><td>0.804</td></tr></table>

## 5.1.4 WorldModelBench: Physical Reasoning and Instruction Following

Benchmark. WorldModelBench Li et al. (2025) evaluates models on three dimensions: instruction following (0–3 scale), common sense (frame and temporal quality), and physics adherence (5 violation types: Newton’s laws, mass conservation, fluid dynamics, penetration, gravity). The benchmark contains 350 instances across 7 domains with 56 subdomains.

Table 5: Performance comparison on WorldModelBench.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td>Instr.</td><td colspan="3">Common Sense</td><td colspan="5">Physics Adherence</td><td>Phys.</td><td rowspan="2">Total</td></tr><tr><td>（0-3）</td><td>Frame</td><td>Temp</td><td>Overall</td><td>Newton</td><td>Mass</td><td>Fluid</td><td>Penetr.</td><td>Grav.</td><td>Overall</td></tr><tr><td rowspan="4">General</td><td>Veo3 Wan2.6</td><td>2.52 2.50</td><td>0.98 0.99</td><td>0.95 0.95</td><td>1.93 1.94</td><td>1.00</td><td>0.89 0.89</td><td>0.99</td><td>0.91</td><td>1.00</td><td>4.80</td><td>9.25</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>1.00</td><td></td><td>0.99</td><td>0.94</td><td>1.00</td><td>4.83</td><td>9.27</td></tr><tr><td>Sora2</td><td>2.21</td><td>0.96</td><td>0.93</td><td>1.88</td><td>1.00</td><td>0.91</td><td>0.99</td><td>0.95</td><td>1.00</td><td>4.84</td><td>8.93</td></tr><tr><td>Kling26</td><td>1.59</td><td>0.97</td><td>1.00</td><td>1.97</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>5.00</td><td>8.55</td></tr><tr><td rowspan="5">Embodied</td><td>LTX-2</td><td>1.97</td><td>0.69</td><td>0.62</td><td>1.32</td><td>0.99</td><td>0.60</td><td>1.00</td><td>0.73</td><td>1.00</td><td>4.32</td><td>7.61</td></tr><tr><td>Cosmos</td><td>2.14</td><td>1.00</td><td>0.94</td><td>1.94</td><td>1.00</td><td>0.92</td><td>1.00</td><td>0.94</td><td>1.00</td><td>4.86</td><td>8.94</td></tr><tr><td>LVP</td><td>2.01</td><td>0.89</td><td>0.91</td><td>1.80</td><td>1.00</td><td>0.93</td><td>0.99</td><td>0.95</td><td>1.00</td><td>4.87</td><td>8.67</td></tr><tr><td>GigaWorld</td><td>2.13</td><td>0.59</td><td>0.46</td><td>1.05</td><td>1.00</td><td>0.48</td><td>0.99</td><td>0.69</td><td>0.98</td><td>4.13</td><td>7.31</td></tr><tr><td>Vidar</td><td>1.62</td><td>0.54</td><td>0.45</td><td>0.99</td><td>1.00</td><td>0.56</td><td>1.00</td><td>0.85</td><td>1.00</td><td>4.40</td><td>7.01</td></tr><tr><td rowspan="2"></td><td>Wow</td><td>2.05</td><td>0.76</td><td>0.65</td><td>1.41</td><td>1.00</td><td>0.65</td><td>0.99</td><td>0.81</td><td>1.00</td><td>4.45</td><td>7.91</td></tr><tr><td>Ours</td><td>2.33</td><td>0.87</td><td>0.85</td><td>1.72</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.94</td><td>1.00</td><td>4.94</td><td>8.99</td></tr></table>

Results. Table 5 shows our model outperforms all open-source models (8.99, 3rd overall), trailing only closed-source Wan2.6 and Veo3. We achieve perfect physics adherence (1.00) across all four categories and strong instruction following (2.33/3.0), with the common-sense gap attributable to our lower output resolution.

## 5.2 Qualitative Analysis

## 5.2.1 Fine-Grained Language Grounding

Precise grounding of language in visual actions is foundational to QWEN-ROBOTWORLD’s design as a language-conditioned world model. Figure 5 evaluates this capability at two levels. (a) Contrastive pairs: given identical initial frames, the model produces qualitatively distinct videos when a single keyword differs—target object identity, action type, or spatial placement—demonstrating fine-grained semantic discrimination beyond generic manipulation priors. (b) Complex instructions: the model handles longhorizon sequential tasks with multi-step dependencies and abstract goal instructions that require inferring the manipulation sequence from context, decomposing each into a temporally coherent execution without explicit sub-task prompts.

## 5.2.2 Generalization across Embodiments, Tasks, and Viewpoints

Figure 6 demonstrates three complementary dimensions of QWEN-ROBOTWORLD’s manipulation capability. (A) Cross-embodiment: a single instruction drives four distinct robot morphologies—single-arm gripper, dual-arm system, humanoid, and dexterous hand—without embodiment-specific adaptation, validating natural language as a universal action interface; each cell shows three key frames (initial, mid,

Pickupthepen fromthe tableand placeit onthewooden tray.

## (a) Contrastive Instruction Following

Pick up the yellow potato from the table.

![](images/7b0c7cd368fc71424a93203b1b993db3048da2eae195f47b0aed05135804caaa.jpg)  
Targetobject

![](images/83e3c57b5c29d431affb6adc21ba038530ac7f77983a2221922c969d83d9aaa6.jpg)

![](images/20785b0e55397bd50054ef608ea66def0815003fddf9ef30d03cc470cbca4af0.jpg)  
Take the glue stick and hand it to the person.  
Destination

![](images/082222cbc22c467a051c4b4933e3cbd23eb5180e40d7eef9cbc7485c64f3add2.jpg)  
Take the glue stick andput it in the penholder.  
Action type

![](images/2eb016f4c0dbc6a062f081c319820d32fdec2eca22e55a3097e1d83f6dc0aafb.jpg)

## (b) Complex Instruction Following

![](images/1c9fb2933f16e1edfbb6e967310d9d2495ca0ed8509fc588b3445bf8b558dc58.jpg)  
Pickuptheredandyelowbellpeppersinsequenceandplacethemonthetablefromleftorightfromthecamera'sperspective.

![](images/1f483bc3781aad94a63c9d2d4998b8799834c785be2c36b3418d357ce14ed02b.jpg)  
Figure 5: Fine-grained language grounding. (a) Contrastive: each pair of columns shares an identical initial frame (colored border); only the highlighted keyword differs between the two instructions. Pair 1: target object identity. Pair 2: destination. Pair 3: action type. In every case the generated motion is precisely grounded to the discriminating keyword. (b) Complex: two examples requiring multi-step execution or abstract goal inference. Colored labels mark key action milestones within each generated sequence.

final). (B) Cross-task × cross-environment: generations across fruit pick-and-place, bowl retrieval, cloth folding, and human–robot handover each exhibit task-appropriate contact dynamics, reflecting grounded physical knowledge across diverse real-world environments; each row shows an initial frame followed by four evenly-spaced generated frames. (C) Multi-view consistency: three synchronized camera streams (main, wrist-left, wrist-right) are jointly generated from the same supermarket pick-and-place episode as (B, row 1), with object identity and motion trajectory remaining geometrically consistent across all viewpoints.

## 5.2.3 Zero-Shot Robustness on RoboTwin-IF

Building on the single-model capabilities demonstrated above, we next examine whether these gains persist under controlled model-to-model comparisons. Aggregate embodied-world-model scores can entangle three different failure sources: instruction mismatch, cross-view inconsistency, and generic visual degradation. To isolate these factors, we perform a zero-shot side-by-side comparison on four Unitree G1 tasks against two strong embodied baselines, LVP and Cosmos2.5-14B. Figure 7 shows that

## (A) Cross-Embodiment

Pick up the object and place it in the container.Same instruction across four morphologies.

![](images/cdb8ada430c8724778b5e1e2556c492eb3830d26212d5b4499f65976ccfd5b29.jpg)

(B)Cross-Task X Cross-Environment  
![](images/603a62f070480ae1149b437c1aa96be8650eafec9bc5efd6f2ab37d514044f54.jpg)

![](images/9b759de7c8c8ecf3f07855a97d2dba51117b91aeb69cd9be546bae6aae77060b.jpg)  
Figure 6: Generalization across embodiments, tasks, and viewpoints. (A) Cross-embodiment: one instruction drives four morphologies (single-arm, dual-arm, humanoid, dexterous hand); three key frames per cell. (B) Cross-task × cross-environment: initial frame (orange border) followed by four generated frames across four tasks. (C) Multi-view: main and wrist cameras jointly generated from the same episode as (B, row 1).

QWEN-ROBOTWORLD more consistently preserves language-grounded execution (correct object/action correspondence and cleaner goal completion) while maintaining coherent multi-view trajectories. The two baselines show different failure patterns. LVP more often produces incomplete task execution, while Cosmos2.5-14B tends to exhibit weaker alignment between the instruction and the generated manipulation outcomes in more complex cases.

To validate this behavior under a benchmark setting, we evaluate zero-shot performance on RoboTwin-IF (Instruction Following), a newly proposed benchmark built on the RoboTwin simulator with many newly constructed complex tasks. Notably, although QWEN-ROBOTWORLD mixes only a small amount of open-source RoboTwin data during training, it still shows strong zero-shot performance on RoboTwin-IF together with stable multi-view consistency across synchronized camera streams. These results suggest that the model’s gains are not limited to a few qualitative examples, but generalize to more challenging unseen embodied tasks. Overall, QWEN-ROBOTWORLD demonstrates stronger zero-shot robustness than prior baselines by better aligning instruction following, action realism, and cross-view coherence in a unified generation framework.

Task 04  
Task 01  
Assemble a smallelectronic device by inserting a black rectangular component into a matching black housing. Ours (multi-view)  
![](images/02023f9f8d2c07daecfa2bf7f418f89689d2c3edbf4d1570e86a738974f1be49.jpg)  
Wipe yellowish liquid stains from a light table using a gray rag until clean and dry. Ours (multi-view)

![](images/b52407bb6325b9fb3eac231b025ff449f6792e45c91d66104f00f104e62a8b0e.jpg)  
Task 03

Place banana, watermelon slice, and avocado onto color-matched plates from the cuting board.  
![](images/d2e6df8ef86458b8a7e3ca039f05400f531b5b87a9c8aee0c5cad8c0bd47006b.jpg)

Clean the whiteboard using a blue eraser and remove the red wavy line/markings. Ours (single-view)  
![](images/b60b9a0e10922ffdd5ef94feab2dc288a6e23dbc548ee6d88ff2ae6904ffe0bc.jpg)  
Figure 7: Zero-shot qualitative comparison on language–action alignment and multi-view coherence. Side-by-side grids under identical conditioning (same initial frame(s), prompt, and camera layout), comparing QWEN-ROBOTWORLD against LVP and Cosmos2.5-14B.

Figure 8 provides representative RoboTwin-IF zero-shot cases as qualitative evidence for this benchmark result. Each task is visualized with ten uniformly sampled frames anchored by the first and last frame, making intermediate progress and final completion directly visible. Across these newly constructed complex tasks, QWEN-ROBOTWORLD preserves coherent execution and cross-view consistency, which is consistent with the quantitative RoboTwin-IF finding.

![](images/cb349a152b6fb7c5edc500661ca4c09431188f67bc9c7186de3f0ba83548673e.jpg)  
Figure 8: RoboTwin-IF zero-shot qualitative cases. The benchmark is built on the RoboTwin simulator with newly constructed complex tasks.

## 5.3 Cross-Domain Generalization

Beyond manipulation-centric evaluation, we assess the model’s generalization to supplementary task families beyond the core manipulation domain. Figure 9 shows human-to-robot transfer across eight target embodiments, where the model preserves task intent from human demonstrations while adapting motion to embodiment-specific kinematic constraints. Figure 10 covers mobility scenarios, including autonomous driving episodes from Bench2Drive, NVIDIA PhysicalAI-AD, Sekai, and Waymo, and egocentric indoor navigation episodes from VLNVerse. Together, these results indicate that the learned language-conditioned transition model generalizes beyond a single embodiment or scenario family.

![](images/8d1bc07dd6223845232e07dfd6ef48c02258d008e6c9c83c5dafebf72dfcd4a1.jpg)

![](images/55c7015573e1b5d5ae6e8b59bd828c3e969cccb366fb01eab499c55e0767632d.jpg)

Figure 9: Human-to-robot transfer. Across eight target embodiments, each row compares a human demonstration (left) with the synthesized robot execution (right) for the same task, using five uniformly sampled frames per video. The generated trajectories preserve task intent while adapting motion to embodiment-specific kinematic constraints.  
![](images/0a3df5d0e90b79fbd4270615aac3a5bec06868cfe90e0c4e31d439d367b17463.jpg)

![](images/65003685fec6e525a4191a25fbfd99d38508650c8f522909adcb4bc58a82fbb0.jpg)  
Figure 10: Mobility generation. Paired columns with five rows: (left) Autonomous driving episodes from Bench2Drive, NVIDIA PhysicalAI-AD, Sekai, and Waymo; (right) Egocentric indoor navigation from VLNVerse with language-guided first-person traversal. Each episode uses five uniformly sampled frames.

## 6 Conclusion

In this report, we present QWEN-ROBOTWORLD, a language-conditioned world model framework for embodied intelligence that unifies robotic manipulation, autonomous driving, indoor navigation, and human-to-robot transfer under a shared natural language action interface. To realize this objective, we develop a three-part system: a double-stream MMDiT architecture with MLLM action encoding for semantically precise and physically grounded generation, the Embodied World Knowledge (EWK) dataset with large-scale cross-embodiment action-language alignment, and a general+expert progressive curriculum that couples broad visual priors with embodied specialization. This design enables one common backbone that can be adapted toward three representative embodied world model applications— synthetic data generation, policy evaluation, and action planning. Across both benchmark evaluations and zero-shot analyses, QWEN-ROBOTWORLD demonstrates strong, consistent performance and robust multi-view instruction-following generalization. We hope this work provides a practical foundation for building embodied world models that are not only perceptually strong, but also functionally useful for downstream robotic learning and control.

## Authors

Jie Zhang\*, Xiaoyue Chen\*, Anzhe Chen, Chenxu Lv, Deqing Li, Gengze Zhou, Hang Yin, Haoqi Yuan, Haoyang Li, Jiahao Li, Jiazhao Zhang, Jingren Zhou, Kaiyuan Gao, Kun Yan, Lihan Jiang, Ningyuan Tang, Pei Lin, Qihang Peng, Shengming Yin, Tianhe Wu, Tianyi Yan, Xiao Xu, Yan Shu, Yanran Zhang, Ye Wang, Yi Wang, Yilei Chen, Yixian Xu, Yiyang Huang, Yuxiang Chen, Zekai Zhang, Zhendong Wang, Zhixing Lei, Zhixuan Liang, Zihao Liu, Zikai Zhou, Xiong-Hui Chen†, Chenfei Wu†

## References

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical AI. arXiv preprint arXiv:2501.03575, 2025.

AgiBot-World-Contributors. AgiBot World Colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15791– 15801, 2025.

Johan Bjorck, Fernando Castaneda, Linxi Fan, Dieter Fox, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, et al. RT-1: Robotics transformer for real-world control at scale. In Robotics: Science and Systems (RSS), 2023.

Build AI. Egocentric-10k. Hugging Face Datasets, 2025. URL https://huggingface.co/datasets/ builddotai/Egocentric-10K.

Boyuan Chen, Tianyuan Zhang, Haoran Geng, Kiwhan Song, Caiyi Zhang, Peihao Li, William T. Freeman, Jitendra Malik, Pieter Abbeel, Russ Tedrake, Vincent Sitzmann, and Yilun Du. Large video planner enables generalizable robot control, 2025a. URL https://arxiv.org/abs/2512.15840.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025b.

Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, and Jian Tang. Wow: Towards a world omniscient world model through embodied interaction, 2025. URL https://arxiv.org/abs/2509.22642.

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Scaling egocentric vision: The EPIC-KITCHENS dataset. In European Conference on Computer Vision (ECCV), 2018.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for highresolution image synthesis. In ICML, 2024.

Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Chenxi Wang, Junbo Wang, Haoyi Zhu, and Cewu Lu. RH20T: A comprehensive robotic dataset for learning diverse skills in one-shot. In IEEE International Conference on Robotics and Automation (ICRA), 2024.

Yao Feng, Hengkai Tan, Xinyi Mao, Chendong Xiang, Guodong Liu, Shuhe Huang, Hang Su, and Jun Zhu. Vidar: Embodied video diffusion model for generalist manipulation, 2025. URL https: //arxiv.org/abs/2507.12898.

Fourier ActionNet Team and Yao Mu. Actionnet: A dataset for dexterous bimanual manipulation. 2025.

Galaxea AI. Galaxea open-world dataset and G0 dual-system VLA model. arXiv preprint arXiv:2509.00576, 2025.

Zelin Gao, Qiuyu Wang, Yanhong Zeng, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

Google DeepMind. Veo 3. https://deepmind.google/technologies/veo/veo-3/, 2025. URL https: //deepmind.google/technologies/veo/veo-3/.

Kristen Grauman, Andrew Westbury, Eugene Byrne, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Mingfei Han, Liang Ma, Kamila Zhumakhanova, Ekaterina Radionova, Jingyi Zhang, Xiaojun Chang, Xiaodan Liang, and Ivan Laptev. RoomTour3D: Geometry-aware video-instruction tuning for embodied navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In European Conference on Computer Vision, pp. 289–305. Springer, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. RLBench: The robot learning benchmark. IEEE Robotics and Automation Letters, 5(2):3019–3026, 2020.

Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Li. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS Datasets and Benchmarks Track, 2024.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems (RSS), 2024.

Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5:341–353, 2023.

Kuaishou Technology. Kling: A progressive framework for video generation. https://klingai.com, 2024. URL https://klingai.com.

Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E. Gonzalez, Ion Stoica, Song Han, and Yao Lu. Worldmodelbench: Judging video generation models as world models, 2025. URL https://arxiv.org/abs/2502.20694.

Xuanlin Li, Kyle Hsu, Jiayuan Liu, Ken Goldberg, and Sergey Levine. Evaluating real-world robot manipulation policies in simulation. In Conference on Robot Learning (CoRL), 2024.

Lightricks. LTX-Video: Realtime video latent diffusion. https://github.com/Lightricks/LTX-Video, 2025. URL https://github.com/Lightricks/LTX-Video.

Sihao Lin, Zerui Li, Xunyi Zhao, Gengze Zhou, Liuyi Wang, Rong Wei, Rui Tang, Juncheng Li, Hanqing Wang, Jiangmiao Pang, Anton van den Hengel, Jiajun Liu, and Qi Wu. VLNVerse: A benchmark for vision-language navigation with versatile, embodied, realistic simulation and evaluation. arXiv preprint arXiv:2512.19021, 2025.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

NVIDIA. NVIDIA Isaac Sim. https://developer.nvidia.com/isaac-sim, 2022.

NVIDIA. PBench: A physical ai benchmark for world models, 2025a. URL https://huggingface.co/ datasets/nvidia/PBench.

NVIDIA. Nvidia physicalai autonomous driving dataset. 2025b. https://developer.nvidia.com/ physicalai.

OpenAI. Sora: Creating video from text. https://openai.com/sora, 2024. URL https://openai.com/ sora.

OpenLoong Baihu Team. OpenLoongData-v1.0. https://www.openloong.org.cn/en/datasets/baihu, 2025.

Baoqi Pei, Yifei Huang, Jilan Xu, Guo Chen, Yuping He, Lijin Yang, Yali Wang, Weidi Xie, Yu Qiao, Fei Wu, and Limin Wang. Modeling fine-grained hand-object dynamics for egocentric video representation learning. In International Conference on Learning Representations (ICLR), 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020.

Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 36(6), 2017.

Sekai Team. Sekai: Real-world egocentric walking videos for world model training. 2025. https: //huggingface.co/datasets/sekai.

Yu Shang, Xin Zhang, Yinzhou Tang, Lei Jin, Chen Gao, Wei Wu, and Yong Li. RoboScape: Physicsinformed embodied world model. arXiv preprint arXiv:2506.23135, 2025.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

GigaWorld Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jiagang Zhu, Kerui Li, Mengyuan Xu, Qiuping Deng, Siting Wang, Wenkang Qin, Xinze Chen, Xiaofeng Wang, Yankai Wang, Yu Cao, Yifan Chang, Yuan Xu, Yun Ye, Yang Wang, Yukun Zhou, Zhengyuan Zhang, Zhehao Dong, and Zheng Zhu. Gigaworld-0: World models as data engine to empower embodied ai, 2025. URL https://arxiv.org/abs/2511.19861.

Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, and Jiangmiao Pang. InternData-A1: Pioneering high-fidelity synthetic data for pre-training generalist policy. arXiv preprint arXiv:2511.16651, 2025.

Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5026–5033, 2012.

Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. BridgeData V2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wan Team. Wan: Open and advanced large-scale video generative models. https://wanxai.com, 2025. URL https://wanxai.com.

Waymo Team. Waymo open dataset: End-to-end driving. 2024. https://waymo.com/open/.

Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, et al. RoboMIND: Benchmark on multiembodiment intelligence normative data for robot manipulation. In Robotics: Science and Systems (RSS), 2025a.

Shihan Wu et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025b.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, and Joel Jang. World action models are zero-shot policies, 2026. URL https://arxiv.org/abs/2602.15922.

Hu Yue, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, and Guanghui Ren. EWMBench: Evaluating scene, motion, and semantic quality in embodied world models, 2025. URL https://arxiv.org/abs/2505.09694.

Zhenyu Zhao, Hongyi Jing, Xiawei Liu, Jiageng Mao, Abha Jha, Hanwen Yang, Rong Xue, Sergey Zakharov, Vitor Guizilini, and Yue Wang. Humanoid everyday: A comprehensive robotic dataset for open-world humanoid manipulation. arXiv preprint arXiv:2510.08807, 2025.

Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, and Chuang Gan. TesserAct: Learning 4D embodied world models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Joel Zhou, Jordan Juravsky, Sanja Fidler, Umar Bhatt, and Nima Fazeli. DreamGen: Unlocking generalization in robot learning through neural trajectories. arXiv preprint arXiv:2505.12705, 2025.