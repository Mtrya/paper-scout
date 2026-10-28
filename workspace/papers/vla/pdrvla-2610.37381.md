<!-- page 1 of 25 -->

arXiv:2610.37381v1 [cs.RO] 27 Oct 2026

Xiaomi-Robotics-0 PDRVLA

![Chart block](doc:252b6f6/tier:basic/page:1/block:3)

# PDRVLA: A VISION–LANGUAGE–ACTION MODEL WITH PHYSICAL DYNAMICS REPRESENTATIONS

Hanqing Wang<sup>2,\*</sup> Zhaotian Cai<sup>2,\*</sup> Ruichuan An<sup>1,\*</sup> Bohan Zeng<sup>1,\*</sup> Tao Zhang<sup>2,\*</sup> Yilin Chen<sup>2,\*</sup> Xu Jiang<sup>2</sup> Kaiquan Chen<sup>2</sup> Chongyang Wu<sup>2</sup> Haoran Li<sup>2</sup> Fan Yang<sup>2</sup> Mingan Lin<sup>2</sup> Wentao Zhang<sup>1,†</sup> Zenan Zhou<sup>2,†</sup> Youzheng Wu<sup>2</sup>

<sup>1</sup>Peking University <sup>2</sup>JD.com

## ABSTRACT

Vision–Language–Action (VLA) models generate robot actions from visual observations and language instructions, but their understanding of physical dynamics remains limited. Video world models can capture scene dynamics, yet generating future videos introduces high latency into the control loop. We introduce PDRVLA, whose lightweight Physical Dynamics Module extracts physical tokens from vision–language features to condition action generation without online video generation. The module learns through direct prediction of future RGB, depth, and optical-flow features, with auxiliary supervision through a frozen pretrained video world model. Training relies entirely on publicly available data, expanded through human-to-robot synthesis, subtask annotation, and embodied chainof-thought (ECoT) annotation. Across four simulation benchmarks, PDRVLA achieves success rates of 99.3% on LIBERO, 89.9% on LIBERO-Plus, 93.6% on RoboTwin 2.0, and 81.5% on SimplerEnv. Ablations show improved future prediction with world-model supervision and evaluate how subtask conditioning, future object-pose supervision, and ECoT training affect action success.

## (a) PDRVLA architecture

![Image block](doc:252b6f6/tier:basic/page:1/block:10)

(b) Benchmark performance

Figure 1: PDRVLA overview. (a) Simplified architecture. (b) Success rates on four simulation benchmarks.

## 1 Introduction

Vision–Language–Action (VLA) models connect visual observations and language instructions to robot control, drawing on pretrained vision–language models (VLMs) and demonstrations from heterogeneous robots [Black et al., 2024, Physical Intelligence, 2025]. Their understanding of the physical world, however, remains limited. For example, a recent

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>\*</sup>Equal contribution.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>†</sup>Corresponding authors: [zhouzenan@jd.com,](mailto:zhouzenan@jd.com) [wentao.zhang@pku.edu.cn.](mailto:wentao.zhang@pku.edu.cn)</span></small>

<!-- page 2 of 25 -->

analysis reports weak encoding of object physical properties across several VLA policies and unreliable adaptation of their motions to changes in object mass [Xu, 2026]. For manipulation, recognizing the current scene is only part of the problem: grasping, moving, and placing objects also change their configurations and spatial relationships. Anticipating how a scene evolves during interaction can provide useful context for action selection, motivating representations that connect current observations with future appearance, geometry, and motion.

Pretrained video world models offer a complementary way to learn physical dynamics by predicting how scenes change over time [Team Wan et al., 2025]. Incorporating these models into robot policies can connect future prediction with action generation [Kim et al., 2026]. However, generating future videos within the control loop adds latency to action generation. Recent approaches instead use video prediction during training to shape compact representations that can condition a policy without online video generation [InternRobotics, 2026, Tang et al., 2026].

Following this direction, we introduce PDRVLA, a VLA model with a lightweight Physical Dynamics Module (Figure 1). The module extracts a fixed set of physical tokens from intermediate VLM features. The action expert uses these tokens alongside the final-layer VLM features to generate actions. We train the module with two complementary objectives. Direct supervision uses future RGB, depth, and optical-flow latent features as targets, encouraging the tokens to encode future appearance, geometry, and motion. A frozen pretrained video world model conditioned on the physical tokens provides auxiliary supervision for temporal prediction. Joint policy training then combines action and vision–language (VL) objectives with continued direct future prediction. The resulting policy uses physical tokens for action generation without running the video model during control.

Our training corpus is built entirely from publicly available datasets. It combines real-robot, handheld UMI, and simulated trajectories with general and robotics-oriented VL data, and is expanded through egocentric human-to-robot synthesis. Subtask and embodied chain-of-thought (ECoT) annotations provide additional supervision for task stages, scene understanding, progress assessment, and next motion. Together, these sources support training across heterogeneous embodiments and complementary forms of action and language supervision.

We evaluate PDRVLA on LIBERO [Liu et al., 2023], LIBERO-Plus [Fei et al., 2025], RoboTwin 2.0 [Chen et al., 2025a], and SimplerEnv [Li et al., 2024], achieving success rates of 99.3%, 89.9%, 93.6%, and 81.5%, respectively. Ablation studies show that world-model-assisted supervision improves future prediction. Additional experiments examine the effects of subtask conditioning, future object-pose supervision, and ECoT annotations on policy performance.

The report makes three contributions:

• We introduce PDRVLA, whose lightweight Physical Dynamics Module extracts compact physical tokens from intermediate VLM features to complement the final-layer features used for action generation. We train these representations through direct prediction of future RGB, depth, and optical-flow features, with auxiliary supervision from a frozen video world model. Direct future prediction continues during joint policy training, while deployment uses the learned physical tokens without running the video generator.

• We assemble a cross-embodiment training corpus entirely from publicly available datasets, integrating realrobot, handheld UMI, and simulated demonstrations with general and robotics-oriented VL data. Our data pipeline standardizes heterogeneous trajectories, expands embodiment coverage through egocentric humanto-robot synthesis, and adds subtask and ECoT annotations, supporting joint learning of robot actions and language responses about task stages, scene context, progress, and next motion.

• We evaluate PDRVLA on four simulation benchmarks and conduct targeted analyses of physical-representation learning and policy performance. The results show improved future prediction with world-model assistance and characterize the effects of subtask conditioning, future object-pose supervision, and ECoT annotations on action success across tasks and inference settings.

## 2 Data

## 2.1 Corpus Overview

Our training corpus combines VLA data with VL data. The VLA portion pairs observations and task instructions with control targets, grounding the model in robot behavior across diverse embodiments and environments. The VL portion extends perceptual, spatial, and task-level reasoning beyond the coverage of action trajectories.

We organize the corpus into five categories. The VLA portion comprises real-robot trajectories, UMI demonstrations, and simulated interactions, which provide complementary trade-offs in physical fidelity, collection cost, embodiment coverage, and scale. The VL portion comprises general image- and video-language data together with spatial and

2

<!-- page 3 of 25 -->

embodied examples more closely aligned with manipulation. Table 1 summarizes the composition and scale of each category; individual sources are described in Sections 2.2 and 2.3.

Table 1: Core training corpus by category.

| Category | Statistics |
| --- | --- |
| VLA Data |  |
| Real-Robot | 1.26M trajectories / 14,893 hours |
| UMI | 1.22M trajectories / 10,259 hours |
| Simulation | 1.87M trajectories / 8,868 hours |
| VL Data |  |
| General VL | 8.05M VL examples |
| Spatial &amp; Embodied VL 23.63M VL examples |  |

In addition to aggregating existing datasets, we derive new training data through synthesis and annotation. Egocentric human-to-robot synthesis converts EgoDex demonstrations [Hoque et al., 2026] into visually and kinematically adapted trajectories for multiple robot embodiments. Subtask annotation assigns a stage label to each frame and produces subtask-prediction VL examples, while ECoT annotation produces keyframe-level VL examples describing the scene, task progress, and next motion. Table 2 summarizes these derived data.

Table 2: Statistics of synthesized and annotated training data.

| Augmentation | Statistics |
| --- | --- |
| Egocentric Conversion | 0.90M robot trajectories / 2,646 hours |
| Subtask Annotation | 1.79B annotated frames / 1.1M VL examples |
| ECoT | 2.56M annotated keyframes |

## 2.2 VLA Data

## 2.2.1 Real-Robot Data

Real-robot trajectories expose the model to real-world dynamics, sensor variation, and contact-rich interactions. We aggregate public datasets spanning different robots, environments, and task families.

Our single-arm tabletop and in-the-wild manipulation data come from RoboSet [Bharadhwaj et al., 2024], DROID [Khaz atsky et al., 2024], BridgeData V2 [Walke et al., 2023], BC-Z [Jang et al., 2022], and RH20T [Fang et al., 2024]. Together, they cover diverse scenes, objects, operators, and manipulation skills, including pick-and-place, articulatedobject interaction, and tool use.

We extend the embodiment coverage with bimanual and multi-platform data from RoboMIND V1/V2 [Wu et al., 2025a, Hou et al., 2025], RoboCOIN [Wu et al., 2025b], and ABC-130K [Allshire et al., 2026]; humanoid, arm, and dexterous-hand data from Humanoid Everyday [Zhao et al., 2026a] and the Unitree UniFoLM releases [Unitree Robotics, 2026a,b,c,d,e,f]; and large-scale multi-robot data from AgiBot World Colosseo [Bu et al., 2025, AgiBot World Colosseum contributors, 2024, AgiBot World Team, 2026].

REASSEMBLE [Sliwowski et al., 2025] contributes contact-rich assembly and disassembly trajectories, while FuSe [Jones et al., 2025] and MolmoAct [Lee et al., 2025] provide additional manipulation trajectories.

Selected embodiment-specific subsets from the LeRobot community collections [HuggingFaceVLA Community, $\mathrm { 2 0 2 5 a , } \mathrm { b , } \mathrm { c ] }$ add further cross-institution and cross-embodiment coverage.

## 2.2.2 UMI Data

UMI data are collected with a handheld gripper and wrist-mounted camera rather than by teleoperating a particular robot [Chi et al., 2024]. The resulting demonstrations record real-world interaction as camera observations and end-effector pose trajectories without tying the motion to a robot’s joint configuration.

We use six UMI datasets with different scales, capture systems, and task coverage: UMI [Chi et al., 2024], FastUMI [Zhaxizhuoma et al., 2025], FastUMI-100K [Liu et al., 2025], Hy-Embodied-0.5-VLA-Data [Zhang et al., 2026],

3

<!-- page 4 of 25 -->

HiFi-UMI-2K [Simple AI et al., 2026], and RealOmni-Open Data [GenRobot.AI (Jianzhi Robotics), 2025]. After source-specific filtering, they contribute approximately 1.22 million episodes and 10,259 hours.

Because UMI demonstrations do not provide robot joint-space commands, we use their temporally aligned end-effector poses directly to construct the unified Cartesian action representation in Section 2.4.

## 2.2.3 Simulation Data

Simulation complements real-world data with controllable, reproducible interaction and access to ground-truth states and success signals. It also makes it practical to vary tasks and embodiments without additional hardware collection.

The simulation corpus covers three complementary settings: standardized skills and long-horizon tasks from ManiSkill2 [Gu et al., 2023], Meta-World [Yu et al., 2019], and RoboCasa [Nasiriany et al., 2024]; large-scale cross-embodiment trajectories from InternData-A1 [Tian et al., 2026], GR00T-X-Embodiment-Sim [NVIDIA, 2025a], and InternVLA-M1 [Chen et al., 2025b]; and bimanual or dexterous manipulation from RoboTwin 2.0 [Chen et al., 2025a] and DexMimicGen [Jiang et al., 2025].

## 2.3 VL Data

## 2.3.1 General VL Data

The general VL mixture includes vision-centric instruction data from Cambrian-1 [Tong et al., 2024]; captioning and open-ended visual question answering from COCO-Caption [LMMs-Lab, 2026] and VQAv2 [HuggingFaceM4, 2026]; and image- and video-reasoning examples from OneThinker [Feng et al., 2025].

## 2.3.2 Spatial and Embodied VL Data

Spatial and embodied VL data bring the supervision closer to manipulation. We divide them into spatial reasoning over general imagery and embodied or procedural reasoning drawn from robot data, simulation, and web video.

The spatial-reasoning group covers spatial question answering, geometric relations, coordinate-aware grounding, referring-expression localization, and multi-view or 3D reasoning. It includes SpaceThinker [Remyx AI, 2026a], OpenSpaces [Remyx AI, 2026b], SpaceOm [Remyx AI, 2026c], RefSpatial [An, 2026], data released with Scaling Spatial Intelligence with Multimodal Foundation Models [Cai et al., 2026a], and cross-view correspondence examples linking egocentric, third-person, and robot-camera observations [Wang et al., 2025].

The embodied and procedural group includes point supervision for object, free-space, and affordance localization from RoboPoint [Yuan et al., 2024]; visual question answering and long-horizon reasoning over manipulation videos from Robo2VLM [Chen et al., 2025c] and RoboVQA [Sermanet et al., 2024]; interleaved perception and reasoning from EO-1 [Qu et al., 2025]; physical-commonsense supervision from Cosmos-Reason1 [NVIDIA, 2025b]; and web-derived procedural supervision for step decomposition and progress reasoning from ProcCorpus-60M [Feng et al., 2026].

## 2.4 Data Processing

Trajectories from different datasets vary in action format, coordinate conventions, hardware names, and video configuration. We standardize action representations, filter invalid trajectories and frames, unify embodiment metadata, and align and encode visual observations. Processing rules account for differences in hardware and collection procedures.

**Action Representation.** To standardize end-effector (EEF) poses across datasets, we convert positions to meters and orientations to rotation matrices, and normalize the coordinate axes to a right-handed convention with +x forward, +y left, and +z up. Each manipulator’s poses remain relative to the arm base or the world frame, as specified by the source dataset. We express its current and target poses in the same reference frame and record that frame in the metadata.

We represent EEF actions as changes from the current pose to a target pose. The target is the dataset-provided absolute EEF target pose when available, or the next observed pose after resampling otherwise. For a given manipulator, let r denote its reference frame, e the EEF frame, and t a timestep after resampling. The current and target positions are $\mathbf { p } _ { r , t } , \mathbf { p } _ { r , t } ^ { * } \in \mathbb { R } ^ { 3 }$ , and their orientations are $R _ { r e , t } , R _ { r e , t } ^ { * } \in S O ( \tilde { 3 } )$ . The superscript ∗ denotes the target. Here, $R _ { r e , t }$ maps vectors from the EEF frame e into reference frame r. The action consists of a translation delta and a rotation delta, both expressed in r:

$$
\Delta a _ { t } ^ { \operatorname { e e f } } = \left( \Delta \mathbf { p } _ { r , t } , \Delta R _ { r , t } \right) , \qquad \Delta \mathbf { p } _ { r , t } = \mathbf { p } _ { r , t } ^ { * } - \mathbf { p } _ { r , t } , \quad \Delta R _ { r , t } = R _ { r e , t } ^ { * } R _ { r e , t } ^ { \mathsf { T } } ,\tag{1}
$$

4

<!-- page 5 of 25 -->

For grippers with opening measurements, we normalize the current and target openings to [0, 1] using the gripper’s physical travel range, with 0 denoting fully closed and 1 fully open. The gripper action is the normalized absolute target opening.

**Trajectory Filtering.** We reject entire episodes that are too short, lack usable action targets or state sequences, contain values whose units cannot be recovered, or have image and state streams that cannot be aligned. We also reject episodes whose task descriptions conflict with the source metadata. Filtering thresholds are set separately for each dataset or subset to account for differences in hardware and collection procedures.

For otherwise usable episodes, frames containing non-finite values, abnormal repeated observations, abrupt spikes, or out-of-range values are excluded from supervision but retained at their original temporal positions to preserve image–state–action alignment. If the corruption cannot be isolated to individual frames, we reject the entire episode.

**Embodiment Metadata Standardization.** Robot hardware names vary across datasets, and the same arm can be paired with different grippers or dexterous hands. We record robot, arm, and end-effector types separately in robot\_type, arm\_type, and hand\_type. We standardize aliases such as panda and franka\_panda to a common arm name while retaining the identity of the attached end effector. Missing or inconsistent metadata are checked against source documentation and recorded images or videos.

**Visual Processing.** Camera observations and robot states may be recorded at different rates. We align them at the lowest rate among the streams, capped at 30 Hz. At most four camera views are retained per episode, prioritizing wrist and third-person views, with view selection configured for each dataset. The selected videos are resized to 640 pixels on the longest side and encoded with a maximum group-of-pictures (GOP) size of 2 frames to accelerate random frame access. Post-processing checks verify that video frames remain aligned with robot states and actions.

## 2.5 Data Synthesis and Annotation

We expand the training corpus through human-to-robot synthesis and language annotation. The former adapts egocentric demonstrations into robot observations and actions; the latter adds subtask labels and ECoT descriptions of the scene, task progress, and next motion to existing trajectories.

## 2.5.1 Egocentric Human-to-Robot Synthesis

Human egocentric data cover diverse real-world scenes and manipulation tasks, but human hands and arms differ from robot manipulators in both appearance and kinematics. We therefore adapt both the recorded motion and the visual observations before using these demonstrations for VLA training.

We apply the human-to-robot synthesis pipeline of Qwen-RobotManip [Yuan et al., 2026] to EgoDex [Hoque et al., 2026], generating demonstrations for 12 robot-arm models: Panda, UR5e, UR10e, FR3, Kinova Gen3, IIWA, ARX-L5, xArm7, ViperX, WidowX, Piper, and YAM.

**Motion Retargeting.** We map EgoDex’s per-frame 3D hand keypoints to an end-effector (EEF) pose trajectory and gripper commands. The EEF trajectory is mapped into each target robot’s coordinate system, and inverse kinematics (IK) produces the corresponding arm-joint trajectory. We discard demonstrations with too few high-confidence hand keypoints and retargeted trajectories whose EEF poses, computed through forward kinematics, deviate excessively from the desired poses.

**Visual Adaptation.** We segment the human hands and arms with SAM3 [Carion et al., 2026] and use ProPainter [Zhou et al., 2023] to reconstruct the background in the masked regions across the video. We render the target robot arm and gripper using the retargeted motion, then composite them into the reconstructed video. DA3 depth estimates [Lin et al., 2025] determine which scene surfaces occlude the rendered robot during compositing. The resulting videos, arm-joint trajectories, and gripper commands form the synthetic VLA examples.

After filtering and standardization, the synthesized corpus contains 896,832 bimanual robot trajectories, totaling approximately 2,646 hours. These trajectories derive from 74,813 distinct EgoDex demonstrations, each of which may contribute variants for multiple target robots. Because the robot variants share a common source pool, the resulting datasets have similar sizes across robot models, with approximately 74,000–75,000 trajectories and 219–221 hours per target arm model.

5

<!-- page 6 of 25 -->

## 2.5.2 Subtask Annotation

Long-horizon manipulation tasks contain stages with distinct intermediate goals. Following prior work on hierarchical task structure [Zhang et al., 2024, Chen et al., 2023], we segment trajectories into ordered subtasks and assign a subtask label to each frame. These annotations yield both subtask-annotated VLA trajectories and VL examples about task stages and transitions.

**Subtask Segmentation.** Source-provided subtask annotations are reused when available; the remaining trajectories are automatically annotated with Qwen3.8-27B [Qwen Team, 2026a] on sparsely sampled observations. We sample one frame per second and increase the sampling rate for short trajectories to provide more observations. Observations at each sampled timestamp come from a selected camera view, with multiple views spatially concatenated when needed, such as two wrist-camera views for bimanual interactions.

The sampled observations are divided into sliding windows of at most 128 frames with a 16-frame overlap. For each window, the model predicts subtask descriptions and segment boundaries. For each sampled frame in an overlap, we use the prediction from the window whose midpoint is closest to that frame. We merge these predictions into an episode-level segmentation and propagate the segment labels to every frame, producing subtask-annotated VLA trajectories. Frames after the last predicted boundary receive a done label indicating the end of the annotated sequence.

**Subtask VL Examples.** Following $\pi _ { 0 . 5 }$ [Physical Intelligence, 2025], we further convert these annotations into VL examples that ask the model to identify the current subtask from a visual observation and the episode-level instruction. Additional queries ask for the next subtask, the remaining subtask sequence, or whether the current segment has reached its annotated end. Figure 2 summarizes six variants that differ in query time and in whether prior subtask history is included.

![Chart block](doc:252b6f6/tier:basic/page:6/block:6)

Figure 2: Six types of subtask VL examples, differing in query timing, answer content, and subtask-history context. No observations after the query frame are provided.

Overall, subtask annotation covers approximately 1.88 million episodes, 1.79 billion frames, and 20,101 hours. Automatic subtask annotation processes 78 million sampled frames (4.4% of the total). From approximately 294,000 of the annotated episodes, we construct 1.1 million single-turn VL examples across 22 data families. Table 3 reports the source composition of the VL corpus.

6

<!-- page 7 of 25 -->

Table 3: Subtask VL examples by source group. Native denotes source-provided subtask labels.

<table><tr><td>Source Group</td><td>Labels</td><td>Examples</td><td>Source Group</td><td>Labels</td><td>Examples</td></tr><tr><td>EgoDex</td><td>VLM</td><td>125,000</td><td>RoboCasa-pretrain</td><td>VLM</td><td>40,000</td></tr><tr><td>InternVLA-sim</td><td>VLM</td><td>103,415</td><td>UniFoLM</td><td>VLM</td><td>35,789</td></tr><tr><td>ABC-130K</td><td>VLM</td><td>93,621</td><td>Table30</td><td>VLM</td><td>34,884</td></tr><tr><td>RoboCOIN</td><td>Native</td><td>91,642</td><td>RW-RL</td><td>VLM</td><td>26,397</td></tr><tr><td>OpenNeoData</td><td>VLM</td><td>83,917</td><td>Humanoid Everyday</td><td>VLM</td><td>24,866</td></tr><tr><td>GALAXEA</td><td>Native</td><td>75,022</td><td>community-v3</td><td>VLM</td><td>20,838</td></tr><tr><td>FastUMI</td><td>VLM</td><td>70,597</td><td>BC-Z</td><td>VLM</td><td>20,000</td></tr><tr><td>RealOmin</td><td>VLM</td><td>57,864</td><td>RoboCasa-target</td><td>Native</td><td>18,000</td></tr><tr><td>RoboTwin2</td><td>VLM</td><td>52,691</td><td>E-Bench</td><td>VLM</td><td>10,983</td></tr><tr><td>H-EmbodVis</td><td>VLM</td><td>48,376</td><td>community-v1</td><td>VLM</td><td>8,675</td></tr><tr><td>Table30-v2</td><td>VLM</td><td>43,862</td><td>community-v2</td><td>VLM</td><td>8,506</td></tr><tr><td colspan="6">Total</td></tr></table>

## 2.5.3 ECoT Annotation

Subtask labels identify the current intermediate goal but provide limited detail about the scene, task progress, and the motion that follows. We therefore augment the segmented trajectories in Section 2.5.2 with ECoT annotations [Zawalski et al., 2024, Feng et al., 2026, Sun et al., 2026] that connect each subtask to its visual context and subsequent motion. Following Qwen-RobotManip [Yuan et al., 2026], we annotate selected keyframes with scene descriptions, progress assessments, and next-motion descriptions derived from recorded poses.

**Keyframe Selection.** Keyframes are drawn only from frames marked valid during data processing (Section 2.4). Within each subtask, we retain its first and last valid frame and select additional frames when their visual similarity to the previous keyframe falls below a threshold. We compute image hashes with the Python ImageHash library<sup>1</sup>to measure this similarity.

**Scene and Progress Annotation.** For each keyframe, Qwen3.8-27B describes task-relevant objects, spatial relations, and gripper state, and assesses whether the active subtask is complete. To account for earlier events, we uniformly sample frames between the episode’s first frame and the current keyframe. Qwen3.8-27B uses these observations to generate a textual memory summary of gripper motion, object motion, and state changes. The model then uses the summary and the current frame to assess progress and explain its assessment. The summary is used during annotation and omitted from the training input.

**Next-Motion Annotation.** Where recorded EEF poses are available, we derive the next-motion description by comparing the EEF poses and gripper openings at the current keyframe and 0.5 seconds later. We ignore translation, rotation, and opening changes below their respective thresholds and map the remaining changes to directional phrases describing EEF motion and gripper opening or closing. These phrases are combined into a sentence such as “The gripper moves forward, tilts down, and rotates clockwise.”

We apply this pipeline to selected real-robot and UMI datasets, producing 2,564,851 annotated keyframes (Table 4). The annotations yield both ECoT-annotated VLA trajectories and VL examples whose targets contain the scene description, progress assessment, and, where available, next-motion description. We additionally use the released annotations from ProcCorpus-60M [Feng et al., 2026].

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>[https://github.com/JohannesBuchner/imagehash](https://github.com/JohannesBuchner/imagehash)</span></small>

7

<!-- page 8 of 25 -->

Table 4: ECoT keyframes annotated by our pipeline.

| Dataset | Data Type | Annotated Keyframes |
| --- | --- | --- |
| AgiBot World [Bu et al., 2025] | Real | 530,301 |
| ABC-130K [Allshire et al., 2026] | Real | 530,289 |
| RoboCOIN [Wu et al., 2025b] | Real | 691,908 |
| RoboMIND [Wu et al., 2025a] | Real | 153,508 |
| HiFi-UMI-2K [Simple AI et al., 2026] | UMI | 575,197 |
| FastUMI-100K [Liu et al., 2025] | UMI | 83,648 |
| Total |  | 2,564,851 |

## 3 Model Architecture

As shown in Figure 3, PDRVLA consists of a Qwen3.5-4B VLM backbone [Qwen Team, 2026b], the Physical Dynamics Module, and an action expert. Given visual observations and a task instruction, the backbone provides intermediate features to the physical module, which produces a fixed-length sequence of physical tokens. The action expert combines these tokens with the final-layer VLM features and the current proprioceptive state to generate a chunk of robot commands.

![Image block](doc:252b6f6/tier:basic/page:8/block:5)

Figure 3: PDRVLA architecture. (a) Physical tokens complement final-layer VLM features as inputs to the action expert. Starting from Gaussian noise, Euler updates integrate predicted velocities into an action chunk. (b) Learnable queries attend to shared VLM features at every physical-module block before projection to physical tokens. Camera views and token sequences are schematic.

## 3.1 Input Construction

The VLM input is a single user message containing the current camera views followed by the task text, formatted with Qwen3.5’s native chat template. Views follow the order specified in each dataset’s camera configuration. We prefix the task instruction with the robot embodiment, arm and end-effector types, and control frequency when available, using the format embodiment &lt;robot&gt;, arm &lt;arm&gt;, hand &lt;hand&gt;. control &lt;rate&gt; Hz. &lt;instruction&gt;. For action generation with or without subtask conditioning, we use the VLM features directly without appending an assistant-generation prefix. The current proprioceptive state is supplied separately to the action expert.

8

<!-- page 9 of 25 -->

Subtask descriptions can be appended to the task instruction as Current step: &lt;subtask&gt;. During training, this text comes from the trajectory annotations. At inference, PDRVLA can first generate a subtask description from the current observations and task instruction, then append it to the instruction for the action-conditioning pass. Without subtask conditioning, the policy generates actions directly from the current observations and original task instruction, together with the robot state.

For ECoT-conditioned action generation, the VLM first generates a response describing the scene, task progress, and next motion, which provides additional context to the action expert.

## 3.2 Physical Dynamics Module

Later VLM layers may suppress visual information useful for predicting physical change. Layer-wise studies have found stronger perceptual representations at intermediate layers of large language models and a prominent role for these layers in accessing visual information [Jain et al., 2025, Kaduri et al., 2025]. Motivated by these findings, at timestep t, we use the VLM’s layer-l features $H _ { t } ^ { ( l ) }$ for the current multi-view observations and task instruction as input to the Physical Dynamics Module. In our implementation, l = 8 for a backbone with 32 decoder blocks. The action expert also uses the final-layer features from the same backbone pass.

As shown in Figure 3(b), the module uses 50 learnable queries to extract a fixed set of physical tokens $C _ { t }$ from $H _ { t } ^ { ( l ) }$ Across 4 Transformer blocks, the queries retrieve information from the VLM features through cross-attention, exchange information through self-attention, and pass through a feed-forward layer. Each block has a hidden width of 1024, 4 attention heads, and a feed-forward expansion ratio of 1. A final projection maps the query outputs to the VLM hidden width, producing 50 physical tokens of width 2560. Future-prediction supervision trains the module to encode dynamics-related information in these tokens (Section 4.1), which are subsequently supplied to the action expert alongside the final-layer VLM features. The module adds approximately 57.3 M parameters (1.4% of the 4B backbone).

Appendix B compares the VLM input layer, cross-attention placement, direct-supervision weighting, and number of learnable queries.

## 3.3 Action Generation

At timestep t, the action policy receives the final-layer VLM features $H _ { t } ^ { \mathrm { l a s t } }$ , the physical tokens $C _ { t } ,$ , and a proprioceptive vector $q _ { t }$ encoding the current robot state. For Cartesian control, the EEF pose in $q _ { t }$ comprises three position coordinates and a continuous 6D orientation representation obtained by flattening the first two columns of its rotation matrix. The policy predicts a chunk $\mathbf { A } _ { t } = [ \mathbf { a } _ { t , 0 } , \ldots , \mathbf { a } _ { t , K - 1 } ]$ of $K = 8$ robot commands, where k in $\mathbf { a } _ { t , k }$ indexes a step within the chunk predicted at t. The complete chunk is executed before the policy replans from a new observation.

**Cross-Embodiment Action Representation.** Training a single policy across heterogeneous robots requires a common representation that accommodates different controller interfaces. We use a fixed-dimensional action vector with separate left and right manipulator slots. Each slot reserves dimensions for four component types:

• arm-joint positions: 7 dimensions;

• end-effector (EEF) pose: 9 reserved dimensions, of which 6 encode the action;

• dexterous-hand joint positions: 12 dimensions;

• gripper opening: 1 dimension.

The two slots together form a 58-dimensional vector. The EEF slot shares its reserved width with the 9D proprioceptive pose representation; its action occupies the first 6 dimensions, with the remaining 3 padded and masked. EEF actions are expressed relative to the pose at the beginning of the chunk, while arm-joint positions, dexterous-hand joint positions, and gripper openings are absolute targets. Dimensions unsupported by an embodiment’s controller are padded and excluded from the training loss; execution uses only the supported components.

For Cartesian control, all EEF targets within a chunk are expressed relative to the EEF pose at the chunk’s start. Let $\mathbf { p } _ { t } \in \mathbb { R } ^ { 3 }$ and $R _ { t } \in S O ( 3 )$ denote this starting position and orientation, and let $\mathbf { p } _ { t + k } ^ { * }$ and $R _ { t + k } ^ { * }$ denote the target position and orientation for step $k = 0 , \ldots , K - 1$ . The superscript ∗ denotes a target. Omitting the left/right manipulator index, each 6D EEF action comprises a translation delta and a rotation delta:

$$
\mathbf { a } _ { t , k } ^ { \mathrm { e e f } } = \left[ \mathbf { p } _ { t + k } ^ { * } - \mathbf { p } _ { t } , \; \mathrm { L o g } \big ( R _ { t + k } ^ { * } R _ { t } ^ { \mathsf { T } } \big ) \right] \in \mathbb { R } ^ { 6 } ,\tag{2}
$$

where L $\mathrm { b g } : S O ( 3 ) \rightarrow \mathbb { R } ^ { 3 }$ maps the relative rotation to an axis–angle vector. Target poses are obtained as described in Section 2.4.

9

<!-- page 10 of 25 -->

**Action Expert.** We train a Diffusion Transformer (DiT)-based action expert with conditional flow matching. It receives a noisy action chunk, whose action vectors are mapped to action tokens by an input projection shared across embodiments. We concatenate the final-layer VLM features and physical tokens along the token dimension to form the conditioning sequence:

$$
X _ { t } = \left[ H _ { t } ^ { \mathrm { l a s t } } ;   C _ { t } \right] ,\tag{3}
$$

The expert alternates self-attention over its input tokens with cross-attention to $X _ { t }$ . The current proprioceptive state $q _ { t }$ is encoded as a prefix token, while flow time conditions the action-token encoding and the Transformer’s adaptive layer normalization. A shared output projection maps the resulting action-token features to a velocity field over the chunk. The action expert contains approximately 150 M parameters. At inference, we initialize the chunk with Gaussian noise and integrate the predicted velocity field from flow time 0 to 1 using four Euler steps. Section 4 defines the interpolation path and masked velocity-matching loss.

## 4 Training

Figure 4 summarizes physical-module pretraining and joint policy training. The resulting policy is then adapted to each benchmark. Physical pretraining combines direct prediction of future RGB, depth, and optical-flow latent features with auxiliary supervision through a frozen pretrained video model. We then jointly train the VLM, Physical Dynamics Module, and action expert on VLA and VL data, retaining the three direct prediction losses alongside action and VL supervision. The video world model and its output heads are used only during physical pretraining. Additional architecture settings, optimization parameters, and sampling details are provided in Appendix A.

## 4.1 Physical Dynamics Pretraining

We pretrain the Physical Dynamics Module (Section 3.2) to encode future appearance, geometry, and motion in its physical tokens. A direct prediction objective supervises these tokens using future RGB, depth, and optical-flow latent features. Because the module and its prediction head are trained from scratch, we also use a frozen pretrained video model to provide auxiliary supervision for representation learning.

**Future Prediction Targets.** For physical pretraining, each example pairs the current observations and task instruction with a five-frame video sampled from a recorded clip spanning the current timestep and the following 30 action steps. The RGB frames provide appearance targets, while DA3 [Lin et al., 2025] and DPFlow [Morimitsu et al., 2025] generate depth and optical-flow pseudo-labels for geometry and motion supervision. Following RynnWorld-4D [Zhao et al., 2026b], depth values are clipped to a fixed range and mapped to grayscale intensities, while optical-flow fields are color-encoded to form the corresponding depth and flow videos. We encode the three video targets into compact latent features rather than predicting them in pixel space. LingBot-Video’s frozen Wan video variational autoencoder (VAE) [Team Wan et al., 2025] provides these representations, and the same encoded targets are reused for direct prediction.

**Direct Future Prediction.** The direct prediction head applies layer normalization and a linear projection to the physical tokens $C _ { t }$ , then uses learned spatial queries for scaled dot-product attention pooling. Three modality-specific linear readouts predict future RGB, depth, and optical-flow features. Let $z _ { \mathrm { f u t u r e } } ^ { m }$ denote the future portion of the VAE-encoded video for modality m $\in \bar { \{ } r , d , f \}$ , corresponding to RGB, depth, and optical flow. We apply adaptive average pooling to obtain $y ^ { m }   \stackrel { \circ } { = }   \mathrm { P o o l } \widetilde { ( z _ { \mathrm { f u t u r e } } ^ { m } ) }$ This reduces the spatial resolution of the prediction target while retaining a coarse spatial layout, so the lightweight head need not reconstruct the full-resolution latent features.

To reduce differences in target scale across modalities and feature dimensions, we standardize the pooled targets as

$$
\widetilde { y } ^ { m } = \frac { y ^ { m } - \mu ^ { m } } { s ^ { m } } ,\tag{4}
$$

where $\mu ^ { m }$ and $s ^ { m }$ are estimated from training samples separately for each modality, channel, and spatial position. A lower bound on the standard deviation prevents excessive amplification of low-variance features. Denoting the head’s prediction for modality m by $g _ { \eta } ^ { m } ( C _ { t } )$ , the direct prediction loss is

$$
\mathcal { L } _ { \operatorname { d i r e c t } } = \frac { 1 } { 3 } \sum _ { m \in \{ r , d , f \} } \operatorname { M S E } \left( g _ { \eta } ^ { m } ( C _ { t } ) , \: \widetilde { y } ^ { m } \right) .\tag{5}
$$

10

<!-- page 11 of 25 -->

(a) Physical pretraining

![Image block](doc:252b6f6/tier:basic/page:11/block:2)

![Image block](doc:252b6f6/tier:basic/page:11/block:3)

Figure 4: Training stages. (a) Physical pretraining combines direct future-feature prediction with supervision through a frozen video model. (b) Joint policy training removes the video model, retains direct prediction, and adds action and VL objectives. Grid stacks represent feature tensors; sg marks the detached intermediate-feature path. Camera views and video frames are schematic.

**World-Model-Assisted Prediction.** Beyond direct feature regression, we use a pretrained video model to guide physical representation learning, following related video-conditioning approaches [InternRobotics, 2026, Tang et al., 2026]. The frozen LingBot-Video-Dense-1.3B model [Ma et al., 2026] receives an RGB latent clip, the physical tokens $C _ { t }$ as conditioning inputs, and a noise timestep. Future-prediction losses train the Physical Dynamics Module to provide context for this video predictor. In addition to its frozen RGB output head, two trainable heads read LingBot’s final hidden features to predict depth and optical-flow velocities, extending the auxiliary supervision to future geometry and motion.

Within the RGB latent clip, the current features remain clean and the future features are mixed with Gaussian noise. Let $z _ { \mathrm { c u r r e n t } } ^ { r }$ and $z _ { \mathrm { f u t u r e } } ^ { r }$ denote these two parts of the encoded clip. We sample a noise level $\sigma \sim \mathcal { U } ( 0 , 1 )$ and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ with the same shape as the future latent. The RGB input is constructed as

$$
z _ { \sigma , \mathrm { c u r r e n t } } ^ { r } = z _ { \mathrm { c u r r e n t } } ^ { r } , \qquad z _ { \sigma , \mathrm { f u t u r e } } ^ { r } = ( 1 - \sigma ) z _ { \mathrm { f u t u r e } } ^ { r } + \sigma \epsilon .\tag{6}
$$

The three output heads predict future latent velocities $\widehat { v } _ { \mathrm { f u t u r e } } ^ { m }$ for $m \in \{ r , d , f \}$ . For the linear data-to-noise path in Eq. 6, the target velocity is the difference between the noise and the clean latent. We use the same noise tensor ϵ for all three modalities, yielding targets $\epsilon - z _ { \mathrm { f u t u r e } } ^ { m }$ . The auxiliary loss sums the three equally weighted velocity errors:

$$
\mathcal { L } _ { \operatorname { W M } } = \sum _ { m \in \{ r , d , f \} } \operatorname { M S E } ( \widehat { v } _ { \operatorname { f u t u r e } } ^ { m } , \epsilon - z _ { \operatorname { f u t u r e } } ^ { m } ) \thinspace .\tag{7}
$$

11

<!-- page 12 of 25 -->

Gradients from $\mathcal { L } _ { \mathrm { W M } }$ train the depth and optical-flow heads and pass through the frozen LingBot backbone to train the Physical Dynamics Module. The RGB head and LingBot backbone are not updated.

**Pretraining Objective.** The complete physical-pretraining objective combines direct future prediction with the auxiliary flow-matching loss:

$$
\mathcal { L } _ { \operatorname { p r e } } = \mathcal { L } _ { \operatorname { W M } } + \lambda ( k ) \mathcal { L } _ { \operatorname { d i r e c t } } ,\tag{8}
$$

where k denotes the optimization step and $\lambda ( k ) = 0 . 5 \operatorname* { m i n } ( k / 2 0 0 0 , 1 )$ increases the direct prediction weight linearly from 0 to 0.5 over the first 2000 steps, after which it remains constant.

Physical pretraining updates the Physical Dynamics Module, the depth and optical-flow output heads, and the direct prediction head. The VLM and the pretrained LingBot-Video components remain frozen.

## 4.2 Action Policy Training

After physical pretraining, we attach the action expert and jointly train the complete policy on VLA examples and the general, spatial, embodied, ECoT, and subtask VL data. The direct prediction head is retained to supervise future RGB, depth, and optical-flow features during action learning. The video world model and its RGB, depth, and optical-flow output heads are no longer used.

**Joint Policy Training.** The VLM, pretrained Physical Dynamics Module, action expert, and direct prediction head are all trainable. Physical tokens are concatenated with the final-layer VLM features to form $X _ { t } = [ \bar { H } _ { t } ^ { \mathrm { l a s t } } ; C _ { t } ]$ for the action expert (Section 3.3). The intermediate VLM features entering the physical module are detached: action and direct prediction losses update the physical module but do not propagate into the VLM through this input. The action loss updates the VLM through the final-layer conditioning path, while the direct losses also update their prediction head. VL supervision updates only the VLM.

**Training Objectives.** Conditional flow matching trains the action expert to transform Gaussian noise into an action chunk. Given a target chunk $\mathbf { A } _ { t }$ containing $K = 8 ^ { \tilde { } }$ actions, we sample noise $\xi \sim \mathcal { N } ( 0 , I )$ of the same shape and a flow time $\tau \in [ 0 , 1 ]$ . The interpolated chunk and target velocity are

$$
x _ { \tau } = ( 1 - \tau ) \xi + \tau \mathbf { A } _ { t } , \qquad v ^ { * } = \mathbf { A } _ { t } - \xi .\tag{9}
$$

The action expert $v _ { \psi } ( x _ { \tau } , \tau , X _ { t } , q _ { t } )$ predicts the velocity from the interpolated chunk, flow time, conditioning features $X _ { t } = [ H _ { t } ^ { \mathrm { l a s t } } ; C _ { t } ]$ , and current proprioceptive state.

Frames marked invalid during data processing (Section 2.4) are excluded as starting observations for training samples. Within a retained chunk, action steps whose target frames are invalid are excluded from the loss, as are padded steps beyond the episode boundary. For each sample i, a two-dimensional binary supervision mask $M _ { i }$ has rows corresponding to chunk steps and columns to action-vector dimensions. An entry is 1 only if the step is valid and the dimension is supported by the robot’s controller; otherwise, it is 0. The masked velocity-matching loss over a minibatch B is

$$
\mathcal { L } _ { \operatorname { a c t } } = \frac { \sum _ { i \in \mathcal { B } } \| M _ { i } \odot ( \widehat { v } _ { i } - v _ { i } ^ { * } ) \| _ { F } ^ { 2 } } { \sum _ { i \in \mathcal { B } } \| M _ { i } \| _ { 1 } + \delta } ,\tag{10}
$$

where $\widehat { v } _ { i }$ and $v _ { i } ^ { * }$ are the action expert’s predicted velocity and the target velocity for sample $i ,$ respectively, and ⊙ denotes element-wise multiplication. The loss is averaged over all supervised action elements in the minibatch, with $\delta = 1 0 ^ { - 6 }$ preventing division by zero.

Alongside action prediction, VL examples supervise the VLM through next-token prediction of target responses. The VL loss $\mathcal { L } _ { \mathrm { V L } }$ is the cross-entropy on response tokens only, averaged first within each example and then across examples. It updates only the VLM, without passing through the action expert or Physical Dynamics Module.

For VLA examples with future targets, $\mathcal { L } _ { \mathrm { d i r e c t } }$ averages the retained direct prediction head’s RGB, depth, and opticalflow prediction errors, as defined in Eq. 5. Target construction follows Section 4.1, including pooling and the standardization in Eq. 4. Each optimization step combines VLA and VL supervision with these direct prediction losses:

$$
\mathcal { L } _ { \operatorname { p o l i c y } } = \mathcal { L } _ { \operatorname { a c t } } + 0 . 1 \mathcal { L } _ { \operatorname { V L } } + 0 . 1 \mathcal { L } _ { \operatorname { d i r e c t } } .\tag{11}
$$

**VLA Frame Sampling.** To reduce closely spaced training examples while retaining changes in end-effector control, we select chunk starting frames from a regular temporal grid at approximately 8 Hz and from frames with changes in gripper or dexterous-hand target commands. Selection is performed when building the training index. Each selected frame anchors an eight-step action chunk read from the original, temporally aligned stream without further subsampling.

12

<!-- page 13 of 25 -->

## 5 Experiments

We evaluate PDRVLA on four simulation benchmarks covering single-arm and bimanual manipulation, robustness to observation and instruction perturbations, and WidowX manipulation under visual-matching settings. Performance is reported as task success rates and compared with published VLA baselines. Section 5.3 examines the effect of world-model-assisted pretraining on future prediction. Additional studies examine subtask conditioning and its update frequency, auxiliary future object-pose supervision, and ECoT supervision during training versus generation at inference.

## 5.1 Experimental Setup

LIBERO Liu et al. [2023] evaluates single-arm manipulation across four suites: Spatial, Object, Goal, and Long, each containing 10 tasks. We fine-tune the pretrained policy jointly on LIBERO demonstrations from all four suites for 40,000 updates with a global batch size of 128. Adaptation uses dense valid starting frames rather than the frame-selection procedure used during policy pretraining. We evaluate 500 episodes per suite (2,000 in total) and report success rates for each suite and their average.

LIBERO-Plus Fei et al. [2025] evaluates the LIBERO tasks under camera, robot, language, lighting, background, noise, and layout perturbations. We evaluate the LIBERO checkpoint and report success rates for each perturbation category and across all 10,030 evaluation cases.

RoboTwin 2.0 Chen et al. [2025a] evaluates bimanual manipulation on 50 Aloha-AgileX tasks. We fine-tune the pretrained policy for 150,000 updates with a global batch size of 1024, using the RoboTwin 2.0 Aloha-AgileX clean50 and randomized500 demonstration sets. Valid starting frames are sampled densely. We evaluate the resulting checkpoint for 100 episodes per task in each setting and report clean and randomized success rates together with their average.

SimplerEnv Li et al. [2024] evaluates WidowX manipulation on four Bridge visual-matching tasks. We fine-tune the pretrained policy on the full Bridge training set for 50,000 updates with a global batch size of 128, using dense valid starting frames. We report per-task success rates and their average.

## 5.2 Main Results

Across the four simulation benchmarks, PDRVLA achieves the highest aggregate success rates among the compared methods, demonstrating strong manipulation performance and robustness under benchmark perturbations. Baseline results are drawn from the cited papers and official repositories under their respective training and evaluation protocols, including comparisons reported by InternVLA-A1.5 [InternRobotics, 2026] and Qwen-RobotManip [Yuan et al., 2026]. Table 5 summarizes the aggregate results, followed by detailed comparisons for each benchmark.

Table 5: Overview of simulation results (success rate, %). – denotes unavailable results.

| Method | LIBERO | LIBERO-Plus | RoboTwin 2.0 | SimplerEnv |
| --- | --- | --- | --- | --- |
| π<sub>0</sub> Black et al. [2024] | 94.4 | 53.6 | 62.2 | 27.1 |
| π<sub>0</sub>.5 <sup>Physical</sup> <sup>Intelligence</sup> [2025] | 96.9 | 84.4 | 79.8 | 57.1 |
| OpenVLA-OFT Kim et al. [2025] | 97.1 | 69.6 | - | - |
| GR00T-N1.7 Bjorck et al. [2025], NVIDIA [2026] | 97.0 | - | - | 59.3 |
| Cosmos-Policy Kim et al. [2026] | 98.5 | 82.2 | - | - |
| LingBot-VLA 1.0 Wu et al. [2026a] | - | - | 87.6 | - |
| LingBot-VLA 2.0 Wu et al. [2026b] | - | - | 93.2 | - |
| LingBot-VA Li et al. [2026] | 98.5 | - | 92.3 | - |
| Xiaomi-Robotics-0 Cai et al. [2026b] | 98.7 | - | - | 79.2 |
| Qwen-VLA-Instruct Wang et al. [2026a] | 97.9 | - | 86.7 | 73.7 |
| Qwen-RobotManip Yuan et al. [2026] | 99.1 | 89.0 | 93.0 | - |
| InternVLA-A1.5 InternRobotics [2026] | 98.9 | 84.8 | 93.2 | 80.8 |
| PDRVLA | 99.3 | 89.9 | 93.6 | 81.5 |

**LIBERO.** On LIBERO, PDRVLA achieves the highest average success rate among the compared methods at 99.3%, compared with 99.1% for Qwen-RobotManip and 98.9% for InternVLA-A1.5 (Table 6). It achieves 100.0% success on Spatial and 99.8% on both Object and Goal. On Long, its success rate is 97.6%, below LingBot-VA’s 98.5%.

13

<!-- page 14 of 25 -->

Table 6: LIBERO success rates (%).

| Method | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- |
| π<sub>0</sub> Black et al. [2024] | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| π<sub>0</sub>.5 <sup>Physical</sup> <sup>Intelligence</sup> [2025] | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT Kim et al. [2025] | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| GR00T-N1.7 Bjorck et al. [2025] | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| Cosmos-Policy Kim et al. [2026] | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| LingBot-VA Li et al. [2026] | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Xiaomi-Robotics-0 Cai et al. [2026b] | 98.8 | 100.0 | 98.8 | 97.2 | 98.7 |
| Qwen-VLA-Instruct Wang et al. [2026a] | - | - | - | - | 97.9 |
| Qwen-RobotManip Yuan et al. [2026] | - | - | - | - | 99.1 |
| InternVLA-A1.5 InternRobotics [2026] | 98.6 | 99.8 | 98.6 | 98.4 | 98.9 |
| PDRVLA | 100.0 | 99.8 | 99.8 | 97.6 | 99.3 |

**LIBERO-Plus.** On LIBERO-Plus, PDRVLA achieves the highest aggregate success rate among the compared methods at 89.9%, compared with 89.0% for Qwen-RobotManip and 84.8% for InternVLA-A1.5 (Table 7). It leads in four of the seven perturbation categories, while Qwen-RobotManip performs better under robot-initial-state, sensor-noise, and object-layout changes. Robot-initial-state perturbations are the most challenging category for PDRVLA, with a success rate of 75.2%.

Table 7: LIBERO-Plus success rates by perturbation category (%).

| Method | Sensor noise | Camera viewpoints | Robot initial states | Light conditions | Language instructions | Background textures | Object layout | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| π<sub>0</sub> Black et al. [2024] | 79.0 | 13.8 | 6.0 | 85.0 | 58.8 | 81.4 | 68.9 | 53.6 |
| π<sub>0</sub>.5 <sup>Physical</sup> <sup>Intelligence</sup> [2025] | 89.0 | 78.4 | 73.6 | 96.2 | 80.8 | 94.1 | 84.5 | 84.4 |
| OpenVLA-OFT Kim et al. [2025] | 75.8 | 56.4 | 31.9 | 88.7 | 79.5 | 93.3 | 74.2 | 69.6 |
| Cosmos-Policy Kim et al. [2026] | 92.7 | 75.8 | 63.3 | 96.5 | 81.7 | 88.9 | 82.2 | 82.2 |
| Qwen-RobotManip Yuan et al. [2026] | 97.7 | 87.2 | 75.5 | 96.6 | 85.6 | 97.7 | 87.3 | 89.0 |
| InternVLA-A1.5 InternRobotics [2026] | 95.6 | 83.1 | 55.1 | 96.4 | 86.9 | 98.2 | 85.2 | 84.8 |
| PDRVLA | 95.6 | 88.2 | 75.2 | 99.1 | 90.7 | 99.4 | 86.5 | 89.9 |

RoboTwin 2.0. On RoboTwin 2.0, PDRVLA achieves the highest average success rate among the compared methods at 93.6%, compared with 93.2% for both LingBot-VLA 2.0 and InternVLA-A1.5 and 93.0% for Qwen-RobotManip (Table 8). It achieves 93.9% success in the clean setting and 93.3% in the randomized setting, leading the compared methods in both settings.

Table 8: RoboTwin 2.0 success rates (%) in clean and randomized settings and their average.

| Method | Clean | Rand. | Avg. |
| --- | --- | --- | --- |
| π<sub>0</sub> Black et al. [2024] | 65.9 | 58.4 | 62.2 |
| π<sub>0</sub>.5 <sup>Physical</sup> <sup>Intelligence</sup> [2025] | 82.7 | 76.8 | 79.8 |
| LingBot-VLA 1.0 Wu et al. [2026a] | 88.6 | 86.7 | 87.6 |
| LingBot-VLA 2.0 Wu et al. [2026b] | 93.5 | 92.8 | 93.2 |
| LingBot-VA Li et al. [2026] | 92.9 | 91.6 | 92.3 |
| Qwen-VLA-Instruct Wang et al. [2026a] | 86.1 | 87.2 | 86.7 |
| Qwen-RobotManip Yuan et al. [2026] | 93.4 | 92.5 | 93.0 |
| InternVLA-A1.5 InternRobotics [2026] | 93.3 | 93.0 | 93.2 |
| PDRVLA | 93.9 | 93.3 | 93.6 |

**SimplerEnv.** On SimplerEnv, PDRVLA achieves the highest average success rate among the compared methods at 81.5%, compared with 80.8% for InternVLA-A1.5 and 79.2% for Xiaomi-Robotics-0 (Table 9). Its strongest result is on Eggplant→Basket, where it achieves 97.6% success, compared with 83.3% for Xiaomi-Robotics-0 and 75.7% for InternVLA-A1.5. On Carrot→Plate, its success rate is 64.6%, below InternVLA-A1.5’s 85.4%.

14

<!-- page 15 of 25 -->

Table 9: SimplerEnv (WidowX / Bridge, visual matching) success rates (%).

| Method | Spoon →Towel | Carrot →Plate | Stack Blocks | Eggplant →Basket | Avg. |
| --- | --- | --- | --- | --- | --- |
| π<sub>0</sub> Black et al. [2024] | 29.1 | 0.0 | 16.6 | 62.5 | 27.1 |
| π<sub>0</sub>.5 <sup>Physical</sup> <sup>Intelligence</sup> [2025] | 49.3 | 64.7 | 44.7 | 69.7 | 57.1 |
| GR00T-N1.7 NVIDIA [2026] | 78.0 | 58.0 | 48.0 | 53.0 | 59.3 |
| Xiaomi-Robotics-0 Cai et al. [2026b] | 95.8 | 62.5 | 75.0 | 83.3 | 79.2 |
| Qwen-VLA-Instruct Wang et al. [2026a] | - | - | - | - | 73.7 |
| InternVLA-A1.5 InternRobotics [2026] | 92.4 | 85.4 | 69.5 | 75.7 | 80.8 |
| PDRVLA | 92.4 | 64.6 | 71.2 | 97.6 | 81.5 |

## 5.3 Component Analysis

We first examine whether world-model supervision improves future-feature prediction, then study the use of subtask and ECoT annotations in action policies. An additional exploratory experiment tests object-pose prediction as auxiliary supervision. The policy studies use independent training setups and baselines.

## 5.3.1 Physical Representation Learning

We examine whether supervision from a video world model improves the physical module’s ability to predict future features beyond direct supervision alone. We compare direct future-feature prediction with the same objective augmented by world-model-assisted flow matching. Both runs use the original Qwen3.5-4B backbone, equal training-sample budgets, and identical learning-rate and direct-loss-weight schedules. We compare their normalized RGB, depth, and optical-flow prediction errors before loss weighting, excluding the auxiliary world-model loss. Figure 5 shows that RGB and depth training errors decrease more rapidly early in training with world-model assistance. The two runs show a smaller gap in optical-flow error, and the gaps narrow later in training.

(a) RGB

![Chart block](doc:252b6f6/tier:basic/page:15/block:8)

(b) Depth

![Chart block](doc:252b6f6/tier:basic/page:15/block:10)

(c) Optical flow

![Chart block](doc:252b6f6/tier:basic/page:15/block:12)

Figure 5: Normalized training MSE with direct prediction alone or additional world-model supervision. Each curve shows sample-weighted means over the preceding 1,000 updates; insets show updates 1,000–6,000 in greater detail.

To evaluate prediction on unseen data, we assess both runs on the same 828 RoboCOIN clips from 100 episodes excluded from training, using their original trained direct heads without refitting. Table 10 reports the final-checkpoint errors. The mean gives equal weight to the three modalities. Mean error decreases from 0.6439 to 0.6354, a relative reduction of 1.32%, with the largest gains in RGB (4.06%) and depth (2.12%). The flow reduction is smaller (0.62%), and its 95% paired episode-bootstrap interval includes zero. Together, the training curves and held-out results show improved future-feature prediction, with the clearest benefits in RGB and depth.

15

<!-- page 16 of 25 -->

Table 10: Final normalized prediction MSE on held-out RoboCOIN clips. Mean denotes the equally weighted average across the three modalities. Relative reductions are measured against Direct only; intervals are 95% paired episodebootstrap intervals.

| Target D | irect only Di | rect + WM Red | uction (%) | 95% interval (%) |
| --- | --- | --- | --- | --- |
| RGB | 0.2558 | 0.2454 | 4.06 | [3.04, 4.99] |
| Depth | 0.3154 | 0.3087 | 2.12 | [0.93, 3.31] |
| Flow | 1.3606 | 1.3522 | 0.62 | [-0.07, 1.28] |
| Mean | 0.6439 | 0.6354 | 1.32 | [0.71, 1.89] |

## 5.3.2 Auxiliary Supervision for Action Policies

**Subtask Conditioning.** We first test whether self-generated subtask descriptions improve action success on LIBERO-Plus [Fei et al., 2025]. At inference, the policy’s own VLM predicts the current subtask from the observations and task instruction, then appends the description to the instruction for action generation. Table 11 compares this setting with the same SFT checkpoint evaluated without preliminary subtask prediction.

Table 11: Self-generated subtask conditioning on LIBERO-Plus by perturbation category (success rate, %).

| Variant | Sensor noise | Camera viewpoints | Robot initial states | Light conditions | Language instructions | Background textures | Object layout | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT baseline | 95.57 | 88.18 | 75.16 | 99.12 | 90.70 | 99.44 | 86.49 | 89.93 |
| + Subtask conditioning | 96.50 | 89.18 | 76.32 | 98.51 | 89.92 | 99.07 | 86.75 | 90.23 |

Subtask conditioning raises overall success from 89.93% to 90.23%. Its largest gains occur under robot-initial-state and camera-viewpoint perturbations (+1.16 and +1.00 percentage points), with additional improvements under sensor noise and object-layout changes. Success decreases under lighting, language, and background-texture perturbations.

**Subtask Update Frequency.** We examine how often self-generated subtask descriptions need to be refreshed during execution on eight RoboCasa365 [Nasiriany et al., 2026] tasks, with 100 evaluation episodes per task under a fixed-seed setting. The policy’s VLM predicts the current subtask from the observations and task instruction. We compare refresh intervals of 1, 8, and 32 action chunks using the same subtask-conditioned checkpoint. Between refreshes, the description is reused while actions are generated from updated observations. A separately trained policy without subtask conditioning serves as the baseline (Table 12).

Table 12: RoboCasa365 success rates (%) with different subtask refresh intervals.

<table><tr><td colspan="5">Subtask refresh interval (chunks)</td></tr><tr><td>Task</td><td>Baseline</td><td>1</td><td>8</td><td>32</td></tr><tr><td>CuttingToolSelection</td><td>46</td><td>60</td><td>50</td><td>33</td></tr><tr><td>RinseSinkBasin</td><td>39</td><td>49</td><td>44</td><td>37</td></tr><tr><td>PrepareCoffee</td><td>7</td><td>13</td><td>7</td><td>7</td></tr><tr><td>PanTransfer</td><td>22</td><td>25</td><td>21</td><td>15</td></tr><tr><td>LoadDishwasher</td><td>48</td><td>55</td><td>58</td><td>33</td></tr><tr><td>ArrangeTea</td><td>7</td><td>15</td><td>22</td><td>16</td></tr><tr><td>PreSoakPan</td><td>31</td><td>29</td><td>38</td><td>21</td></tr><tr><td>WashFruitColander</td><td>10</td><td>12</td><td>17</td><td>3</td></tr><tr><td>Average</td><td>26.25</td><td>32.25</td><td>32.13</td><td>20.63</td></tr></table>

Using refresh intervals of 1 or 8 chunks yields average success rates of 32.25% and 32.13%, respectively, compared with 26.25% for the baseline. These similar averages mask task-level differences: changing the interval from 1 to 8 reduces CuttingToolSelection success from 60% to 50%, but increases ArrangeTea success from 15% to 22%. Extending the interval to 32 reduces average success to 20.63%, below the baseline. An eight-chunk interval therefore gives similar aggregate success to per-chunk refreshes in this evaluation.

16

<!-- page 17 of 25 -->

To characterize these updates, we measure the subtask change rate, defined as the percentage of successive predictions that differ. The rates are 11.6%, 26.5%, and 56.1% for refresh intervals of 1, 8, and 32 chunks, respectively. Most per-chunk predictions therefore repeat the preceding subtask.

**ECoT Training and Generation.** We examine whether ECoT supervision improves action success and whether its benefits require ECoT generation at inference. We compare a baseline trained without ECoT with a policy whose VLM is trained to generate ECoT responses that provide context to the action expert. The ECoT-trained checkpoint is evaluated both with and without generation at inference (Table 13).

Table 13: Effects of ECoT training and inference-time generation on success rates (%).

<table><tr><td></td><td colspan="2">LIBERO</td><td colspan="2">LIBERO-Plus</td></tr><tr><td>Setting</td><td>Long</td><td>Average</td><td>Long</td><td>Overall</td></tr><tr><td>No ECoT training</td><td>92.8</td><td>97.1</td><td>65.18</td><td>79.59</td></tr><tr><td>ECoT-trained, no generation</td><td>96.0</td><td>98.2</td><td>69.87</td><td>79.25</td></tr><tr><td>ECoT-trained, with generation</td><td>95.6</td><td>98.3</td><td>68.72</td><td>79.12</td></tr></table>

Without ECoT generation at inference, ECoT training raises Long-suite success from 92.8% to 96.0% on LIBERO and from 65.18% to 69.87% on LIBERO-Plus, while LIBERO average success increases from 97.1% to 98.2%. Generating ECoT at inference provides no consistent additional gain: LIBERO average success increases slightly to 98.3%, but Long-suite success decreases to 95.6% and 68.72%, respectively. Aggregate LIBERO-Plus success is 79.25% without generation and 79.12% with it, both below the baseline’s 79.59%.

**Exploratory Object-Pose Supervision.** Object-centric predictive methods use anticipated object configurations to guide manipulation: Goal-VLA recovers target object poses from generated goal images [Chen et al., 2025d], while P2P-T learns future tool-pose representations to condition action generation [Wang et al., 2026b]. Motivated by these approaches, we examine whether future object-pose prediction as an auxiliary training objective improves action success on LIBERO-Plus. Two auxiliary heads share a pooled representation of VLM-token hidden states: a classification head predicts whether the object moves, and a regression head predicts its translation and rotation increments over the next eight action steps. Both heads are used only during training. Table 14 compares this variant with the SFT baseline; neither uses subtask conditioning at inference.

Table 14: Auxiliary object-pose supervision on LIBERO-Plus by perturbation category (success rate, %).

| Variant | Sensor noise | Camera viewpoints | Robot initial states | Light conditions | Language instructions | Background textures | Object layout | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT baseline | 95.57 | 88.18 | 75.16 | 99.12 | 90.70 | 99.44 | 86.49 | 89.93 |
| + Object-pose supervision | 96.56 | 89.56 | 74.52 | 98.25 | 90.50 | 98.88 | 87.15 | 90.12 |

Object-pose supervision increases overall success from 89.93% to 90.12%, a gain of 0.19 percentage points. Success improves under camera-viewpoint, sensor-noise, and object-layout perturbations, with the largest gain of 1.38 percentage points under viewpoint changes. It decreases in the other four categories, including drops of 0.87 percentage points under lighting changes and 0.64 under robot-initial-state changes.

## 6 Conclusion

We presented PDRVLA, a VLA model that uses future prediction to learn compact physical representations for action generation. Direct supervision and a pretrained video world model shape these representations during training; at deployment, the policy uses the learned physical tokens without running the video model or prediction heads. Results across four simulation benchmarks demonstrate strong manipulation performance, while the prediction study shows that world-model assistance improves accuracy over direct supervision alone.

Future work will explore predictive targets beyond RGB, depth, and optical flow, including object-centric geometry, contact states, and material properties. We will investigate how these cues complement one another and contribute to action learning while retaining a lightweight policy.

17

<!-- page 18 of 25 -->

## References

Kevin Black et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Physical Intelligence. π<sub>0</sub>.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Xinnuo Xu. Do VLAs understand and adapt to the objects they handle, or simply replay learned behaviors? arXiv preprint arXiv:2610.06078, 2026. URL [https://arxiv.org/abs/2610.06078.](https://arxiv.org/abs/2610.06078)

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. URL [https://arxiv.org/abs/2503.20314.](https://arxiv.org/abs/2503.20314)

Moo Jin Kim et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

InternRobotics. InternVLA-A1.5: Unifying understanding, latent foresight, and action for compositional generalization. arXiv preprint arXiv:2607.04988, 2026.

Qu Tang, Benhui Zhuang, Bo Yuan, Xue Yu, Longteng Guo, and Junlan Feng. World tokens: Enhancing embodied policies with training-time world modeling. arXiv preprint arXiv:2608.09730, 2026. URL [https://arxiv.org/abs/2608.09730](https://arxiv.org/abs/2608.09730).

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, 2023.

Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. LIBERO-Plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025. URL [https://arxiv.org/abs/2510.13626.](https://arxiv.org/abs/2510.13626)

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025a. URL [https://arxiv.org/abs/2506.18088.](https://arxiv.org/abs/2506.18088)

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024. URL [https://arxiv.org/abs/2405.05941](https://arxiv.org/abs/2405.05941).

Ryan Hoque, Peide Huang, David Yoon, Jian Zhang, et al. Egodex: Learning dexterous manipulation from large-scale egocentric video. In International Conference on Learning Representations, volume 2026, pages 4218–4237, 2026.

Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar. RoboAgent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. In IEEE International Conference on Robotics and Automation, 2024. URL [https://arxiv.org/abs/2309.01918.](https://arxiv.org/abs/2309.01918)

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems, 2024. URL [https://arxiv.org/abs/2403.12945](https://arxiv.org/abs/2403.12945).

Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. BridgeData v2: A dataset for robot learning at scale. In Conference on Robot Learning, 2023. URL [https://arxiv.org/abs/2308.12952.](https://arxiv.org/abs/2308.12952)

Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. BC-Z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, 2022. URL [https://arxiv.org/abs/2202.02005.](https://arxiv.org/abs/2202.02005)

Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Chenxi Wang, Junbo Wang, Haoyi Zhu, and Cewu Lu. RH20T: A comprehensive robotic dataset for learning diverse skills in one-shot. In IEEE International Conference on Robotics and Automation, 2024. URL [https://arxiv.org/abs/2307.00595.](https://arxiv.org/abs/2307.00595)

Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, et al. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation. In Robotics: Science and Systems, 2025a. URL [https://doi.org/10.48550/arXiv.2412.13877.](https://doi.org/10.48550/arXiv.2412.13877)

18

<!-- page 19 of 25 -->

Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, Qiuxuan Feng, Zhao Jin, et al. RoboMind 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence. arXiv preprint arXiv:2512.24653, 2025. URL [https://arxiv.org/abs/2512.24653.](https://arxiv.org/abs/2512.24653)

Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. RoboCoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025b. URL [https://arxiv.org/abs/2511.17441.](https://arxiv.org/abs/2511.17441)

Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, Xi Chen, Rocky Duan, Phillip Isola, Jitendra Malik, Fred Shentu, Guanya Shi, Philipp Wu, and Angjoo Kanazawa. Scalable behavior cloning with open data, training, and evaluation, 2026. URL [https://arxiv.org/abs/2606.27375.](https://arxiv.org/abs/2606.27375)

Zhenyu Zhao, Hongyi Jing, Xiawei Liu, Jiageng Mao, Abha Jha, Hanwen Yang, Rong Xue, Sergey Zakharov, Vitor Guizilini, and Yue Wang. Humanoid everyday: A comprehensive robotic dataset for open-world humanoid manipulation. In IEEE International Conference on Robotics and Automation, 2026a. URL [https://arxiv.org/abs/2510.08807](https://arxiv.org/abs/2510.08807).

Unitree Robotics. UniFoLM-WBT-Dataset. [https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset), 2026a.

Unitree Robotics. UniFoLM-G1-Dex1-DiverseManip-Dataset. [https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-diversemanip-dataset,](https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-diversemanip-dataset) 2026b.

Unitree Robotics. UniFoLM-G1-Dex1-Dataset. [https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-dataset](https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-dataset), 2026c.

Unitree Robotics. UniFoLM-G1-BrainCo-Dataset. [https://huggingface.co/collections/unitreerobotics/unifolm-g1-brainco-dataset](https://huggingface.co/collections/unitreerobotics/unifolm-g1-brainco-dataset), 2026d.

Unitree Robotics. UniFoLM-Z1-Arm-Dataset. [https://huggingface.co/collections/unitreerobotics/unifolm-z1-arm-dataset](https://huggingface.co/collections/unitreerobotics/unifolm-z1-arm-dataset), 2026e.

Unitree Robotics. UniFoLM-G1-Dex3-Dataset. [https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex3-dataset](https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex3-dataset), 2026f.

Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. AgiBot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2025. URL [https://arxiv.org/abs/2503.06669](https://arxiv.org/abs/2503.06669).

AgiBot World Colosseum contributors. AgiBot World Colosseum. [https://github.com/OpenDriveLab/AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World), 2024.

AgiBot World Team. AgiBot world 2026. [https://huggingface.co/datasets/agibot-world/AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026), 2026. URL [https://huggingface.co/datasets/agibot-world/AgiBotWorld2026.](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)

Daniel Sliwowski, Shail Jadav, Sergej Stanovcic, Jedrzej Orbik, Johannes Heidersberger, and Dongheui Lee. Reassemble: A multimodal dataset for contact-rich robotic assembly and disassembly. In Robotics: Science and Systems, 2025. URL [https://arxiv.org/abs/2502.05086.](https://arxiv.org/abs/2502.05086)

Joshua Jones, Oier Mees, Carmelo Sferrazza, Kyle Stachowicz, Pieter Abbeel, and Sergey Levine. Beyond sight: Finetuning generalist robot policies with heterogeneous sensors via language grounding. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), Atlanta, USA, 2025.

Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact: Action reasoning models that can reason in space, 2025. URL [https://arxiv.org/abs/2508.07917.](https://arxiv.org/abs/2508.07917)

HuggingFaceVLA Community. Lerobot community dataset v1: A curated cross-embodiment pretraining dataset for vision-language-action models. [https://huggingface.co/datasets/HuggingFaceVLA/community\_dataset\_v1](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v1), 2025a. 128 datasets from 55 contributors, SO-100 arm; pretraining data for SmolVLA.

HuggingFaceVLA Community. Lerobot community dataset v2: A large-scale community-contributed robotics dataset for vision-language-action learning. [https://huggingface.co/datasets/HuggingFaceVLA/community\_dataset\_v2](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2), 2025b. 340 datasets from 117 contributors, 59 GB, LeRobot v2.0/v2.1 format.

HuggingFaceVLA Community. Lerobot community dataset v3: A cross-embodiment pretraining dataset for visionlanguage-action models. [https://huggingface.co/datasets/HuggingFaceVLA/community\_dataset\_v3](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v3),2025c. 791 datasets from 235 contributors, 46 robot embodiments, 25.97M frames; builds on SmolVLA pretraining data.

19

<!-- page 20 of 25 -->

Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024. doi:[10.15607/RSS.2024.XX.045](https://doi.org/10.15607/RSS.2024.XX.045). URL [https://www.roboticsproceedings.org/rss20/p045.pdf.](https://www.roboticsproceedings.org/rss20/p045.pdf)

Zhaxizhuoma, Kehui Liu, Chuyue Guan, Zhongjie Jia, Ziniu Wu, Xin Liu, Tianyu Wang, Shuai Liang, et al. FastUMI: A scalable and hardware-independent universal manipulation interface with dataset. In Conference on Robot Learning, 2025. URL [https://arxiv.org/abs/2409.19499.](https://arxiv.org/abs/2409.19499)

Kehui Liu, Zhongjie Jia, Yang Li, Zhaxizhuoma, Pengan Chen, Song Liu, Xin Liu, Pingrui Zhang, Haoming Song, Xinyi Ye, et al. FastUMI-100k: Advancing data-driven robotic manipulation with a large-scale UMI-style dataset. arXiv preprint arXiv:2510.08022, 2025. URL [https://arxiv.org/abs/2510.08022.](https://arxiv.org/abs/2510.08022)

He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, et al. Hy-Embodied-0.5-VLA: From vision-language-action models to a real-world robot learning stack. arXiv preprint arXiv:2606.14409, 2026. URL [https://arxiv.org/abs/2606.14409.](https://arxiv.org/abs/2606.14409)

Simple AI, Yuteng Wei, Jinming Ma, Jiawei Wang, Weitao Zhou, Yushen Zuo, Ke Rui, Minglei Li, Jinhao Zhang, Zhikang Pan, Xiang Wang, Haoran Jia, Huan Du, Zicheng Zeng, Jun Ma, Guiyu Qin, Di Zhang, and Xiaofei Li. Hifiumi: Learning deployable manipulation policies from high-fidelity umi data alone. arXiv preprint arXiv:2607.25895, 2026.

GenRobot.AI (Jianzhi Robotics). 10Kh RealOmni-Open Data: A Large-Scale Bimanual Embodied Intelligence Dataset. [https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData), 2025. 13,000+ hours, 5M+ clips, dual-arm household manipulation. License: CC BY-SA 4.0.

Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, et al. ManiSkill2: A unified benchmark for generalizable manipulation skills. In International Conference on Learning Representations, 2023. URL [https://arxiv.org/abs/2302.04659.](https://arxiv.org/abs/2302.04659)

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Avnish Narayan, Hayden Shively, Adithya Bellathur, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, 2019. URL [https://arxiv.org/abs/1910.10897.](https://arxiv.org/abs/1910.10897)

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. RoboCasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems, 2024. URL [https://arxiv.org/abs/2406.02523.](https://arxiv.org/abs/2406.02523)

Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, et al. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 976–985, 2026.

NVIDIA. Physicalai-robotics-GR00T-x-embodiment-sim. Hugging Face dataset, 2025a. URL [https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim). Synthetic cross-embodiment trajectory datasets used for GR00T N1 post-training.

Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, et al. InternVLA-M1: A spatially guided vision-language-action framework for generalist robot policy. arXiv preprint arXiv:2510.13778, 2025b. URL [https://arxiv.org/abs/2510.13778.](https://arxiv.org/abs/2510.13778)

Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Fan, and Yuke Zhu. DexMimic-Gen: Automated data generation for bimanual dexterous manipulation via imitation learning. In IEEE International Conference on Robotics and Automation, 2025. URL [https://arxiv.org/abs/2410.24185.](https://arxiv.org/abs/2410.24185)

Shengbang Tong et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. URL [https://arxiv.org/abs/2406.16860.](https://arxiv.org/abs/2406.16860)

LMMs-Lab. COCO-Caption dataset. [https://huggingface.co/datasets/lmms-lab/COCO-Caption,](https://huggingface.co/datasets/lmms-lab/COCO-Caption) 2026.

HuggingFaceM4. VQAv2 dataset. [https://huggingface.co/datasets/HuggingFaceM4/VQAv2,](https://huggingface.co/datasets/HuggingFaceM4/VQAv2) 2026.

Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, Yan Feng, Peng Pei, Xunliang Cai, and Xiangyu Yue. OneThinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025. URL [https://arxiv.org/abs/2512.03043.](https://arxiv.org/abs/2512.03043)

Remyx AI. SpaceThinker dataset. [https://huggingface.co/datasets/remyxai/SpaceThinker,](https://huggingface.co/datasets/remyxai/SpaceThinker) 2026a.

Remyx AI. OpenSpaces dataset. [https://huggingface.co/datasets/remyxai/OpenSpaces,](https://huggingface.co/datasets/remyxai/OpenSpaces) 2026b.

Remyx AI. SpaceOm dataset. [https://huggingface.co/datasets/remyxai/SpaceOm,](https://huggingface.co/datasets/remyxai/SpaceOm) 2026c.

Jingkun An. RefSpatial dataset. [https://huggingface.co/datasets/JingkunAn/RefSpatial,](https://huggingface.co/datasets/JingkunAn/RefSpatial) 2026.

20

<!-- page 21 of 25 -->

Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, et al. Scaling spatial intelligence with multimodal foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026a. URL [https://arxiv.org/abs/2511.13719.](https://arxiv.org/abs/2511.13719)

Yipu Wang, Yuheng Ji, Yuyang Liu, Enshen Zhou, Ziqiang Yang, Yuxuan Tian, Ziheng Qin, Yue Liu, Huajie Tan, Cheng Chi, Zhiyuan Ma, Daniel Dajun Zeng, and Xiaolong Zheng. Towards cross-view point correspondence in vision-language models. arXiv preprint arXiv:2512.04686, 2025. URL [https://arxiv.org/abs/2512.04686.](https://arxiv.org/abs/2512.04686)

Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. RoboPoint: A vision-language model for spatial affordance prediction in robotics. In Conference on Robot Learning, 2024. URL [https://arxiv.org/abs/2406.10721.](https://arxiv.org/abs/2406.10721)

Kaiyuan Chen, Shuangyu Xie, Zehan Ma, Pannag R Sanketi, and Ken Goldberg. Robo2VLM: Visual question answering from large-scale in-the-wild robot manipulation datasets. arXiv preprint arXiv:2505.15517, 2025c. URL [https://arxiv.org/abs/2505.15517.](https://arxiv.org/abs/2505.15517)

Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, et al. RoboVQA: Multimodal long-horizon reasoning for robotics. In IEEE International Conference on Robotics and Automation, 2024. URL [https://arxiv.org/abs/2311.00899.](https://arxiv.org/abs/2311.00899)

Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Dong Wang, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Maoqing Yao, Bin Zhao, et al. EO-1: An open unified embodied foundation model for general robot control. arXiv preprint arXiv:2508.21112, 2025. URL [https://arxiv.org/abs/2508.21112.](https://arxiv.org/abs/2508.21112)

NVIDIA. Cosmos-Reason1: From physical common sense to embodied reasoning. Technical report, 2025b.

Youhe Feng, Hansen Shi, Haoyang Li, Xinlei Guo, Yang Wang, Chengyang Zhang, Jinkai Zhang, Xiaohan Zhang, Jie Tang, and Jing Zhang. ProcVLM: Learning procedure-grounded progress rewards for robotic manipulation, 2026. URL [https://arxiv.org/abs/2605.08774.](https://arxiv.org/abs/2605.08774)

Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, et al. Qwen-RobotManip technical report: Alignment unlocks scale for robotic manipulation foundation models. arXiv preprint arXiv:2606.17846, 2026. URL [https://arxiv.org/abs/2606.17846.](https://arxiv.org/abs/2606.17846)

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, volume 2026, pages 138846–138923, 2026.

Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10477–10486, 2023.

Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Zichen Zhang, Yunshuang Li, Osbert Bastani, Abhishek Gupta, Dinesh Jayaraman, Yecheng Jason Ma, and Luca Weihs. Universal visual decomposer: Long-horizon manipulation made easy. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6973–6980. IEEE, 2024.

Yuanpei Chen, Chen Wang, Li Fei-Fei, and C Karen Liu. Sequential dexterity: Chaining dexterous policies for long-horizon manipulation. arXiv preprint arXiv:2309.00987, 2023.

Qwen Team. Qwen3.8-27B, 2026a. URL [https://huggingface.co/Qwen/Qwen3.8-27B.](https://huggingface.co/Qwen/Qwen3.8-27B) Official model card.

Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. In Conference on Robot Learning (CoRL), 2024.

Nan Sun, Yuan Zhang, Yongkun Yang, Wentao Zhao, Peiyan Li, Jun Guo, Wenxuan Song, Pengxiang Ding, Runze Suo, Yifei Su, Xin Xiao, Xinghang Li, and Huaping Liu. Revisiting embodied chain-of-thought for generalizable robot manipulation. arXiv preprint arXiv:2606.03784, 2026.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026b. URL [https://qwen.ai/blog?id=qwen3.5](https://qwen.ai/blog?id=qwen3.5).

Jitesh Jain, Zhengyuan Yang, Humphrey Shi, Jianfeng Gao, and Jianwei Yang. Elevating visual perception in multimodal LLMs with visual embedding distillation. In Advances in Neural Information Processing Systems, 2025. URL [https://arxiv.org/abs/2412.09585.](https://arxiv.org/abs/2412.09585)

Omri Kaduri, Shai Bagon, and Tali Dekel. What’s in the image? a deep-dive into the vision of vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. URL [https://openaccess.thecvf.com/content/CVPR2025/papers/Kaduri\_Whats\_in\_the\_Image\_A\_Deep-Dive\_into\_the\_Vision\_of\_CVPR\_2025\_paper.pdf.](https://openaccess.thecvf.com/content/CVPR2025/papers/Kaduri_Whats_in_the_Image_A_Deep-Dive_into_the_Vision_of_CVPR_2025_paper.pdf)

21

<!-- page 22 of 25 -->

Henrique Morimitsu, Xiaobin Zhu, Roberto M. Cesar, Xiangyang Ji, and Xu-Cheng Yin. DPFlow: Adaptive optical flow estimation with a dual-pyramid framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17810–17820, 2025. URL [https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu\_DPFlow\_Adaptive\_Optical\_Flow\_Estimation\_with\_a\_Dual-Pyramid\_Framework\_CVPR\_2025\_paper.html.](https://openaccess.thecvf.com/content/CVPR2025/html/Morimitsu_DPFlow_Adaptive_Optical_Flow_Estimation_with_a_Dual-Pyramid_Framework_CVPR_2025_paper.html)

Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, and Zhongyu Li. RynnWorld-4D: 4d embodied world models for robotic manipulation. arXiv preprint arXiv:2607.06559, 2026b. URL [https://arxiv.org/abs/2607.06559](https://arxiv.org/abs/2607.06559).

Shuailei Ma, Jiaqi Liao, Xinyang Wang, Jingjing Wang, Chaoran Feng, Zijing Hu, Chong Bao, Zichen Xi, Yuqi Gan, Weisen Wang, et al. Scaling mixture-of-experts video pretraining for embodied intelligence. arXiv preprint arXiv:2607.07675, 2026. URL [https://arxiv.org/abs/2607.07675.](https://arxiv.org/abs/2607.07675)

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. URL [https://arxiv.org/abs/2502.19645.](https://arxiv.org/abs/2502.19645)

Johan Bjorck et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

NVIDIA. Isaac GR00T N1.7: SimplerEnv benchmark results. [https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md,](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md) 2026. Accessed October 22, 2026.

Wei Wu, Fan Lu, Yunnan Wang, et al. A pragmatic VLA foundation model. arXiv preprint arXiv:2601.18692, 2026a. URL [https://arxiv.org/abs/2601.18692.](https://arxiv.org/abs/2601.18692)

Wei Wu, Fangjing Wang, Fan Lu, et al. From foundation to application: Improving VLA models in practice. arXiv preprint arXiv:2607.06403, 2026b. URL [https://arxiv.org/abs/2607.06403.](https://arxiv.org/abs/2607.06403)

Lin Li, Qihang Zhang, Yiming Luo, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026. URL [https://arxiv.org/abs/2601.21998.](https://arxiv.org/abs/2601.21998)

R. Cai et al. Xiaomi-Robotics-0: An open-sourced vision-language-action model with real-time execution. arXiv preprint arXiv:2602.12684, 2026b.

Qiuyue Wang et al. Qwen-VLA: Unifying vision-language-action modeling across tasks, environments, and robot embodiments. arXiv preprint arXiv:2605.30280, 2026a. URL [https://arxiv.org/abs/2605.30280.](https://arxiv.org/abs/2605.30280)

Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. RoboCasa365: A large-scale simulation framework for training and benchmarking generalist robots. In International Conference on Learning Representations, 2026. URL [https://robocasa.ai/assets/robocasa365\_iclr26.pdf.](https://robocasa.ai/assets/robocasa365_iclr26.pdf)

Haonan Chen, Jingxiang Guo, Bangjun Wang, Tianrui Zhang, Xuchuan Huang, Boren Zheng, Yiwen Hou, Chenrui Tie, Jiajun Deng, and Lin Shao. Goal-VLA: Image-generative VLMs as object-centric world models empowering zeroshot robot manipulation. arXiv preprint arXiv:2506.23919, 2025d. URL [https://arxiv.org/abs/2506.23919](https://arxiv.org/abs/2506.23919).

Bangjun Wang, Longyan Wu, Yukun Wei, Shenghe Shao, Chaoyi Huang, Wenze Cui, Zetong Xu, Hanlin Wu, Long Chen, Yi Ma, and Hongyang Li. From pixel to poses: Object-centric tool manipulation learning from human demonstrations. arXiv preprint arXiv:2609.35375, 2026b. URL [https://arxiv.org/abs/2609.35375.](https://arxiv.org/abs/2609.35375)

22

<!-- page 23 of 25 -->

## Appendix

## A Implementation and Training Details

## A.1 Action-Expert Configuration

The action expert has 16 Transformer blocks, alternating cross-attention and self-attention, with a token width of 768 and 12 attention heads of width 64. A final token projection maps features to 1024 dimensions before they are decoded into action velocities. The expert’s input sequence consists of a proprioceptive prefix token, 32 learnable tokens, and 8 action tokens with learned positional embeddings. Attention and feed-forward dropout is 0.2, and the proprioceptive input is dropped with probability 0.2 during training.

During action flow-matching training, we randomly sample the flow time τ to vary the mixture of Gaussian noise and target actions in Eq. 9. Smaller values give more weight to noise, while larger values give more weight to the target actions. Specifically, we draw $b \sim \mathrm { B e t a } \bar { ( } 1 . 5 , 1 )$ and set $\tau = 1 - \operatorname* { m i n } ( b , s ) / \overline { { s } } ,$ with $s = 0 . 9 9 9$ . Flow time is discretized into 1000 buckets and embedded both in the action encoder and in the adaptive normalization of the Transformer blocks and final output projection. Each example is repeated 8 times with independently sampled flow times and noise for the flow-matching objective. Inference uses four Euler steps of size 0.25 over [0, 1].

## A.2 Frame Selection

We select starting frames for VLA training at regular intervals of approximately 0.125 s (8 Hz). We also add frames where a gripper or dexterous-hand target command changes by more than a threshold between consecutive frames. Each manipulator’s available gripper and hand channels are checked separately. We combine the two sets of candidates and remove frames that are too close together, keeping frames with command changes in preference to regularly sampled frames. The minimum spacing is 0.01 s, converted to a frame interval using the episode’s frame rate. The remaining candidates must pass the sample-validity checks described in Section 4.2.

## A.3 World-Model-Assisted Pretraining Comparison

**Training Setup.** The comparison in Section 5.3 tests whether adding world-model supervision improves future prediction. Both runs use direct future-feature supervision; Direct + WM additionally includes the auxiliary worldmodel objective. The two runs share the following settings. Training clips contain 5 frames at a resolution of 480 × 640. We optimize the physical module and trainable prediction heads with AdamW, a peak learning rate of $3 \times 1 0 ^ { - 4 }$ , and weight decay 0.01. The learning rate increases linearly over the first 5% of updates and then follows cosine decay. Both runs use 8 GPUs with a global batch size of 128 and random seed 42; trainable parameters remain in FP32 with BF16 autocasting. Each run consists of 35,082 updates and 4,490,400 sample exposures, with 32 examples in the final update. The direct-loss weight increases linearly from zero to 0.5 over the first 2,000 updates in both runs.

**Evaluation Protocol.** We evaluate the trained physical modules and their original direct prediction heads without refitting or updating parameters. Both runs use the same 828 RoboCOIN clips from 100 held-out episodes and identical target-normalization statistics. Checkpoints are evaluated at steps 0, 1,000, 2,000, 5,000, 10,000, 20,000, 30,000, and 35,082; Table 10 reports the final checkpoint. Modality errors are averaged over clips, and the overall mean gives equal weight to RGB, depth, and optical flow. Relative reductions use Direct-only error as the denominator. We estimate 95% percentile intervals from 10,000 paired episode-bootstrap resamples using seed 42, retaining all clips of each sampled episode and computing clip-weighted means within each resample. These intervals quantify episode-sampling uncertainty for the fixed trained checkpoints.

## B Physical-Module Design Choices

We select four physical-module settings in sequence: the VLM input layer, cross-attention placement, direct-supervision weighting, and query count. At each stage, we retain the settings selected in the preceding stages and compare candidates for the next choice. For each candidate, we train the physical module with the original Qwen3.5-4B backbone, then freeze both the VLM and physical module and fit a new linear readout to predict future features. The readout measures how readily future information can be recovered from the learned representations. We use validation scores to select configurations, with the single-seed results summarized in Table 15.

**Training Setup and Comparison Protocol.** The design study uses a 500k-example training pool, 6,000 optimization steps, seed 42, and learning rate $3 \times 1 0 ^ { - 4 }$ . Physical-module hidden width, block count, head count, output width,

23

<!-- page 24 of 25 -->

and feed-forward expansion ratio are fixed at 1024, 4, 4, 2560, and 1. The world-model-assisted loss has weight 1 throughout these design comparisons.

We first compare VLM input layers using 256 queries, cross-attention at every physical-module block, and a constant direct-loss weight of 0.1. We then fix the selected VLM Block 8 and compare cross-attention placements, retaining 256 queries and weight 0.1; parameter counts vary with the number of cross-attention blocks. Next, we retain Block 8, every-block cross-attention, and 256 queries while comparing direct-loss weights and warm-up schedules. Finally, we compare query counts using Block 8, every-block cross-attention, and the selected direct-loss schedule, which warms up to 0.5 over 2,000 steps.

**Frozen-Representation Probe.** During readout fitting and evaluation, the VLM and physical module remain frozen. Masked mean pooling produces a 2560-dimensional representation, from which an affine ridge readout predicts pooled future RGB, depth, and color-encoded flow latent targets. The probe data span five source families, with 2,048 clips for readout fitting and a separate set of 1,024 validation clips for scoring each configuration. Within the fitting pool, an episode-separated split of 1,665 fitting clips and 383 validation clips is used to select the regularization strength. We then refit the readout on all 2,048 fitting clips before scoring it on the separate validation set. Readout fitting uses FP64 standardized spectral ridge, with standardization statistics estimated from fitting data only.

We report Mean Prediction Skill, which measures the relative MSE reduction over a baseline that always predicts the fitting-set mean for each source family and modality. The reductions are averaged equally across source families and modalities. A score of zero matches this baseline, and higher scores indicate better prediction. Let S denote the source families, $\mathcal { M } = \{ r , d , f \}$ the RGB, depth, and flow modalities, yb the readout prediction, and y the target. For source s and modality m, let $\bar { y } _ { s , m } ^ { \mathrm { f i t } }$ be the mean target estimated from fitting labels. The score is

$$
\frac { 1 } { | \mathcal { S } | | \mathcal { M } | } \sum _ { s \in \mathcal { S } } \sum _ { m \in \mathcal { M } } \left[ 1 - \frac { \operatorname { M S E } _ { s , m } ( \widehat { y } , y ) } { \operatorname { M S E } _ { s , m } ( \bar { y } _ { s , m } ^ { \operatorname { f i t } } , y ) } \right] .\tag{12}
$$

Table 15: Physical-module design comparisons using frozen-representation probes.

| Design choice | Mean Prediction Skill ↑ |
| --- | --- |
| VLM input layer |  |
| Block 8 | 0.3617 |
| Block 16 | 0.3413 |
| Block 24 | 0.3314 |
| Final block | 0.2432 |
| Cross-attention placement within the physical module |  |
| First block | 0.3383 |
| Final block | 0.3590 |
| Every block | 0.3617 |
| Direct prediction supervision |  |
| No direct supervision | 0.2683 |
| Constant weight 0.1 | 0.3617 |
| Constant weight 0.5 | 0.3769 |
| Constant weight 1.0 | 0.3723 |
| Linear warm-up to 0.5 over 2,000 steps | 0.3834 |
| Number of learned queries |  |
| 50 |  |
|  | 0.3844 |
| 128 | 0.3815 |
| 256 | 0.3834 |

**Results and Interpretation.** Block 8 gives the highest score among the tested VLM layers (0.3617 versus 0.2432 for the final block), supporting intermediate-layer input. Cross-attention at every physical-module block scores slightly higher than final-block-only access (0.3617 versus 0.3590). With world-model supervision retained, adding direct supervision with the selected weight warm-up raises the score from 0.2683 to 0.3834, the highest among the tested weighting schemes. Under this schedule, 50 queries achieve 0.3844, close to 0.3834 for 256 queries, supporting the compact token sequence.

24

<!-- page 25 of 25 -->

## C Additional LIBERO-Plus Results

We additionally summarize the LIBERO-Plus results by the originating LIBERO task suite (Table 16). These are the same evaluations reported by perturbation category in Tables 11 and 14, grouped to show how the effects vary across task suites.

Table 16: Subtask conditioning and object-pose supervision on LIBERO-Plus by source task suite (success rate, %).

| Source task suite | SFT baseline | + Subtask conditioning | + Object-pose supervision |
| --- | --- | --- | --- |
| Spatial | 96.50 | 96.59 | 95.71 |
| Object | 92.85 | 93.21 | 93.05 |
| Goal | 83.94 | 84.06 | 85.18 |
| Long | 86.90 | 87.53 | 86.94 |
| Overall | 89.93 | 90.23 | 90.12 |

Subtask conditioning improves success across all four source suites, with its largest gain on Long (+0.63 percentage points). Object-pose supervision gives its largest gain on Goal (+1.24 percentage points) but reduces Spatial success by 0.79 percentage points.

## D Franka End-Effector Pose Distribution Diagnostics

We examine absolute end-effector (EEF) height distributions in the Franka data from the pretraining mixture and LIBERO, and assess whether the normalization range from pretraining covers the LIBERO observations. The comparison focuses on distribution differences associated with scene geometry, even when the robot embodiment is unchanged.

**Audit Protocol.** We inspect 75 Franka configuration entries from the pretraining mixture and the four LIBERO suites. For each entry, we sample up to eight shards and eight row groups per selected shard, targeting approximately 2,048 frames with seed 42. The audit yields 143,507 valid arm-level poses from 162,081 sampled frames; seven RobotSet entries lack EEF poses and are recorded as missing. Positions are interpreted in meters according to source metadata and retained in their original coordinate frames.

**Scene Geometry and Normalization Ranges.** The pretraining right-arm position statistics examined in the audit have a first-to-99th-percentile z interval of approximately [−0.096, 0.772] m. We use its upper bound of 0.772 m as a reference to examine the coverage of LIBERO EEF heights.

LIBERO Spatial and Goal use tabletops at 0.90 m, Object uses a floor-level workspace, and Long combines living room, kitchen, and study scenes with work surfaces at 0.41, 0.90, and 0.867 m, respectively. Checks of the original NVIDIA LIBERO position fields confirm that the suite-level height differences are present before pose conversion. The suite-dependent height ranges are consistent with these scene geometries. Table 17 compares the sampled heights with the examined pretraining upper bound.

Table 17: Sampled absolute EEF-height statistics in LIBERO. The pretraining upper bound is 0.772 m.

| Suite Me | dian z (m) | z p01-p99 (m) Above u | pper bound (%) |
| --- | --- | --- | --- |
| Spatial | 1.041 | 0.915-1.285 | 100.0 |
| Object | 0.225 | 0.010-0.336 | 0.0 |
| Goal | 1.076 | 0.911-1.290 | 100.0 |
| Long | 0.712 | 0.447-1.267 | 49.1 |

All sampled Spatial and Goal positions and 49.1% of sampled Long positions exceed the reference upper bound. The differences across suites illustrate how a shared robot embodiment can still have different absolute pose distributions because of scene geometry, affecting the coverage of a normalization range transferred between datasets.

25