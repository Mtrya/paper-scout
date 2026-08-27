# GigaBrain-0.7: Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture

GigaBrain Team

Project: https://gigaai.cc/blog/gigabrain07 Code: https://github.com/open-gigaai/giga-brain-0

GigaBrain Team (alphabetical order): Angen Ye, Axiang Sun, Can Jin, Chenxi Cheng, Chong Shi, Dengke Shang, Dingqian Zhang, Guan Huang, Guangqiang Wang, Guangqing Ding, Guo Li, Hangcong Li, Hengyu Zhong, Hongtao Lu, Jianbo Qin, Jiming Mao, Jing Zhu, Jindi Lv, Jingzhi Cui, Junjie Xie, Junyi Bao, Kai Liu, Lei Yuan, Limin Long, Lv Feng, Mingming Yu, Peng Li, Pengfei Yi, Qi Li, Qianli Zhang, Qingfang Li, Qitang Hu, Rui Zhang, Shaoyan Sun, Shibo Sun, Shiying Duan, Tenghui Chen, Tianze Liu, Weijie Ke, Wenyao Xue, Xiaofeng Wang, Xiaoyu Tian, Xinyu Liu, Xinze Chen, Yang Wang, Yankai Wang, Yejun Zeng, Yifan Li, Yifei Nie, Yilong Li, Yilong Liu, Yongchao Feng, Yumeng Wang, Yun Ye, Zhichao Liu, Ziheng He, Zonghai Yang, Zheng Zhu

![](images/4ba7b906c1bcff94cf0ab565b60b992367c6e078ca52736f4814ff958e3b926e.jpg)  
Figure 1: Overview of GigaBrain-0.7. GigaBrain-0.7 is an embodied foundation-model system that coordinates understanding and planning, prediction and evaluation, and action and control through a three-system architecture. It learns from heterogeneous pretraining data and on-robot experience to support generalization and continual improvement across diverse robot embodiments and real-world scenarios.

## Abstract

Vision-language-action (VLA) models have become a dominant paradigm for generalist embodied agents, demonstrating strong complex and long-horizon task completion in structured settings. Yet it remains an open question whether current VLA systems can benefit from more effective architectural design, scale to substantially larger and more heterogeneous data regimes, and achieve broader generalization across tasks and embodiments. To this end, we present GigaBrain-0.7, an embodied foundation model with substantially improved generalization across diverse robot embodiments. Specifically, GigaBrain-0.7 unifies understanding, prediction, and action through a three-system architecture, scales pretraining to over 37,000 hours of heterogeneous embodied data, and introduces one-stage alignment training that jointly optimizes vision-language understanding and multi-embodiment action generation. Compared with the preceding GigaBrain-0 series and prior state-of-the-art models including $\pi _ { 0 . 5 } ,$ GigaBrain-0.7 achieves substantial improvements in foundation zero-shot capabilities, language-conditioned instruction following, and post-training task success rates. In particular, on our in-house Maker H01 platform and mainstream robot embodiments, GigaBrain-0.7 demonstrates strong task adaptability and completion ability across both home and industrial scenarios. All training code and pretrained model weights will be released.

## 1. Introduction

Vision-Language-Action (VLA) models have emerged as a compelling paradigm for general-purpose robot control by adapting pretrained Vision-Language Models (VLMs) to translate visual observations and language instructions into executable actions [5, 6, 27, 39]. As pretraining data and model capacity scale, recent VLAs have demonstrated increasingly strong out-of-the-box real-robot execution, downstream adaptation, and generalization across tasks, environments, and embodiments [62, 63, 78, 79, 86, 87]. However, embodied scaling presents challenges that extend well beyond collecting more trajectories or enlarging the policy network. Robot data vary substantially across embodiments, action spaces, and execution conditions; without appropriate alignment and contextualization, greater data diversity may introduce interference rather than transferable knowledge [63, 87]. At the model level, most VLAs remain centered on reactive observation-to-action prediction, with limited mechanisms for anticipating future states or evaluating behavioral progress. Recent systems have begun to address this through generated visual subgoals, world-model-conditioned policy learning, and predictive training objectives [28, 41, 63, 78], yet understanding and planning, predictive judgment, action execution, and experience-driven improvement remain only partially coordinated across the full learning lifecycle. Building capable embodied foundation models therefore requires more than scaling isolated policies— it calls for a unified learning system that integrates heterogeneous experience and multi-embodiment pretraining with understanding, prediction, action, and continual improvement from rollout and corrective feedback.

To this end, we present GigaBrain-0.7, an embodied foundation-model system designed to scale heterogeneous experience and coordinate understanding, prediction, action, and experience-driven improvement. At its foundation is a one-stage, multi-embodiment VLA pretraining framework that jointly trains a pretrained VLM backbone, a continuous-action expert, and multimodal understanding objectives in a single pass, eliminating the optimization fragmentation typical of multi-stage training pipelines. The model is organized around a three-system architecture. System 1 (Action and Control) generates continuous action chunks via flow matching [48] from multimodal observations, language instructions, proprioceptive states, and higher-level contextual signals, with an embodiment-aware action expert that accommodates different degrees of freedom and control modes across robot morphologies. System 2 (Understanding and Planning) interprets current and historical visual observations through temporal context modeling, tracks task progress, and decomposes long-horizon instructions into executable subtasks via hierarchical prompts. System 3 (Prediction and Evaluation) employs a world model to predict future visual states (subgoal images) and estimate state values, providing two complementary conditioning interfaces for System 1. The three systems communicate through structured semantic, visual, and value-based interfaces while retaining specialized objectives. On the data side, GigaBrain-0.7 is pretrained on over 37,000 hours of curated embodied trajectory data spanning 16 robot morphologies and approximately 270 million vision-language samples. All trajectory data is standardized through a unified processing pipeline including format conversion to LeRobot v3.0 [7], cross-embodiment state-action normalization, LLM-assisted language instruction rewriting and subtask annotation, and multistage quality control. Building on Knowledge Insulation [20], we employ Soft Knowledge Insulation (Soft KI), which attenuates rather than completely blocks action gradients entering the VLM backbone, allowing controlled adaptation to embodied control while preserving general vision-language capabilities. Building on GigaBrain-0 [27], which introduced world models as scalable data engines and embodied reasoning supervision, and GigaBrain-0.5M\* [28], which developed world-model-conditioned policy learning via RAMP, GigaBrain-0.7 extends the series toward a coordinated learning system spanning data curation, pretraining, planning, prediction, execution, and continual improvement.

In our experiments, we first conduct systematic scaling studies to evaluate the effects of robot-data scale, data-source composition, VLM backbone capacity, and VLA coupling architecture. Results show a clear positive trend with increasing data scale: larger-scale pretraining consistently reduces validation loss and makes challenging real-robot behaviors increasingly reliable. Together with the System 3 ablations and out-of-distribution evaluations, these results show signs of emergent embodied capabilities arising from large-scale heterogeneous pretraining. We then assess the foundation capabilities of GigaBrain-0.7 on multimodal understanding benchmarks and on simulation environments spanning tabletop, mobile, and household manipulation. Furthermore, we conduct extensive real-world evaluations on the AgileX PiPER/PiPER-X and our in-house Maker H01 platforms, covering zero-shot and multi-task execution, language following, out-of-distribution generalization, and long-horizon manipulation. We additionally perform controlled ablation studies to analyze the individual and joint contributions of System 3's future-state predictions and value-based evaluation signals. Finally, we compare GigaBrain-0.7 against prior GigaBrain models [27, 28] and representative VLA baselines including π0.5 [62] after task-specific post-training, and evaluate experience-driven policy improvement using humanin-the-loop rollout data and corrective feedback. GigaBrain-0.7 achieves substantial improvements over the preceding GigaBrain-0 series in foundation zero-shot capabilities, language-conditioned instruction following, and post-training task success rates. All training code and pretrained model weights will be publicly released.

## 2. Related Work

## 2.1. Vision-Language-Action Architectures

Recent progress in pretrained multimodal models has catalyzed the development of vision-language-action (VLA) models for general-purpose robot control [5, 6, 9, 23, 27, 39, 40, 56, 62, 78, 79, 86, 87, 89]. These models extend large-scale vision-language representations to physical action, but differ substantially in how actions are represented and how action generation is coupled to the pretrained VLM. From this perspective, existing architectures can be broadly organized into three families: autoregressive VLM-as-Actor models, serial (cascaded) VLM-action-expert models, and parallel Mixture-of-Transformers (MoT) models.

## Autoregressive action generation.

The first family extends the VLM's native autoregressive interface to robot control by representing continuous actions as discrete token sequences. RT-2 [6] and OpenVLA [39] predict robot-action tokens through next-token prediction, keeping the VLM itself as the action generator. Subsequent work improves the efficiency and expressiveness of action tokenization. FAST [60] exploits temporal redundancy through frequency-domain compression, while learned tokenization methods [19, 44] learn compact discrete representations of continuous action sequences. Galaxea G0.5 [23] further develops this direction with a learned cross-embodiment action tokenizer, allowing reasoning and action tokens to be generated within a single autoregressive Transformer stream. These approaches preserve a direct interface between language modeling and physical action, but sequential token decoding becomes increasingly costly as control frequency, prediction horizon, and action dimensionality grow. OpenVLA-OFT [40] consequently explores parallel decoding, action chunking, and continuous action representations for more efficient downstream adaptation of autoregressive VLA backbones.

## Serial (cascaded) VLM-action-expert coupling.

A second family separates multimodal reasoning from continuous action generation. Rather than requiring the VLM itself to decode low-level control, these architectures use the pretrained VLM as a multimodal context encoder and attach a separately parameterized action decoder or Action Expert. Representative cascaded or decoupled designs include GR00T N1 [56], Gemini Robotics [25], and Qwen-RobotManip [87]. In this paradigm, vision-language representations are first constructed by the VLM and subsequently consumed by the motor generator through an explicit feature interface. Qwen-RobotManip, for example, employs a flow-matching DiT whose blocks cross-attend to visual and language representations extracted from the final VLM layer. Such modular designs allow the action module to specialize in continuous control, while their effectiveness depends strongly on the information exposed through the VLM-action interface.

## Parallel Mixture-of-Transformers.

A third and increasingly common direction retains a dedicated continuous Action Expert while coupling it more deeply with the vision-language backbone through Mixture-of-Transformers architectures [5, 9, 27, 62, 63, 79, 86, 89]. $\pi _ { 0 }$ [5] established the influential VLM-Action Expert formulation with block-wise causal interaction and flow-matching action generation. The same architectural lineage is extended in $\pi _ { 0 . 5 }$ [62] and $\pi _ { 0 . 7 } \ [ 6 3 ]$ with heterogeneous pretraining, hierarchical task conditioning, observation history, and richer context conditioning. GigaBrain-0 [27] similarly combines a pretrained VLM with a continuous Action Expert in an MoT formulation, together with RGB-D perception, Embodied Chain-of-Thought supervision, and Knowledge Insulation [20]. Wall-OSS-0.5 [86] introduces layer-wise VL and Action Experts with joint attention and end-to-end gradient flow, while HyVLA-0.5 [89] combines an embodied-native MoT backbone with a flow-matching Action Expert and compact temporal memory. Xiaomi-Robotics-0 [9] and Xiaomi-Robotics-1 [79] also adopt VLM-DiT MoT architectures, conditioning flow-matching action generation on VLM representations and KV caches.

Although these models differ in attention routing and gradient coupling, they share a common design principle: a pretrained VLM is paired with a dedicated continuous Action Expert, with different architectures exposing different degrees of interaction between the two components. Several recent systems additionally incorporate visual history into the policy context. HyVLA-0.5 [89] introduces a compact memory encoder for spatiotemporal context, while $\pi _ { 0 . 7 }$ [63] incorporates multi-frame observation history into context-conditioned policy learning. GigaBrain-0.7 follows the parallel VLM-Action Expert direction for continuous action generation, while introducing separate functional interfaces for understanding and planning, and for future-state prediction and evaluation.

## 2.2. VLA Training Paradigms

Beyond architecture, VLA systems differ substantially in how vision-language understanding and robot action generation are introduced and optimized over the model lifecycle. Training recipes range from staged adaptation, where representation learning and continuous action optimization are separated, to increasingly joint formulations that optimize vision-language and action objectives within the same VLA pretraining stage.

## Multi-stage vs. one-stage pretraining.

Most VLAs initialize from pretrained vision-language models and subsequently introduce embodied and action supervision, but differ in how strongly these objectives are separated during optimization. Early autoregressive approaches such as RT-2 [6] and OpenVLA [39] adapt pretrained multimodal representations to robot control through action-token prediction. $\pi _ { 0 . 5 }$ [62] adopts a staged training recipe in which robot actions are represented with discrete tokens during broad pretraining, while a flow-matching Action Expert is introduced during posttraining for continuous action generation. Xiaomi-Robotics-0 [9] further decomposes VLA pretraining into two optimization steps: it first trains the VLM jointly on vision-language and robot-trajectory supervision and subsequently freezes the VLM while training the flow-matching DiT. Xiaomi-Robotics-1 [79] similarly adopts an explicit pretraining-post-training recipe, pretraining on over 100,000 hours of real-world UMI trajectories with automatically generated state-transition language and subsequently aligning the learned capabilities to

robot embodiments and imperative task instructions.

A complementary direction increasingly unifies vision-language and action objectives within the same VLA pretraining stage. GigaBrain-0 [27] jointly optimizes embodied reasoning, discrete-action prediction, and continuous flow-matching action generation under a unified objective. Wall-OSS-0.5 [86] develops a single-stage gradient-bridged co-training recipe that jointly optimizes multimodal cross-entropy, action-token prediction, and continuous flow matching. Qwen-RobotManip [87] emphasizes cross-source alignment as a prerequisite for scaling heterogeneous manipulation data across different representations, motions, and behaviors. LingBot-VLA 2.0 [78] further scales pretraining to approximately 60,000 hours of embodied data spanning diverse robot configurations and egocentric human experience. GigaBrain-0.7 follows this joint-training direction and scales it to heterogeneous multi-embodiment data, jointly optimizing general vision-language supervision, hierarchical task prediction, discrete action representations, and continuous action generation within the same VLA pretraining stage.

## Task-specific post-training.

Large-scale pretraining establishes broadly transferable representations and manipulation priors, while posttraining specializes these capabilities to target embodiments, tasks, and deployment conditions. $\pi _ { 0 . 5 }$ [62] introduces continuous flow-matching action generation during post-training and specializes the pretrained model on high-quality task-relevant robot data. OpenVLA-OFT [40] studies efficient downstream fine-tuning of autoregressive VLA backbones through parallel decoding, action chunking, and continuous action prediction. Xiaomi-Robotics-1 [79] uses cross-embodiment robot data during post-training to align UMI-pretrained capabilities with robot embodiments and imperative instructions. HyVLA-0.5 [89] similarly specializes its pretrained model to target robot embodiments through supervised fine-tuning before subsequent reinforcement-learning refinement. Together, these approaches establish post-training as an important bridge between general-purpose pretraining and high-precision downstream robot behavior.

## Reinforcement learning.

Supervised post-training remains bounded by the state distribution and behavioral quality represented in demonstration data, motivating the use of policy-generated experience for further improvement. Recent approaches therefore extend VLA learning with reinforcement learning over simulated or real-robot rollouts. VLA-RL [52] and SimpleVLA-RL [46] investigate direct reinforcement-learning updates of pretrained VLAs from environment feedback, while Z-1 [10] develops a GRPO-based post-training framework for flow-based VLAs with prefix-based rollout construction and success-aware reward design. RL Token [80] instead exposes a compact representation from a pretrained VLA as an interface for lightweight online actor-critic optimization.

Other approaches make greater use of offline experience, advantage information, or preference supervision. AWR [59] performs advantage-weighted regression over off-policy experience, while $\pi _ { 0 . 6 } ^ { * }$ [61] introduces RECAP (RL with Experience and Corrections via Advantage-conditioned Policies), which learns from demonstrations, on-policy experience, and expert corrections through advantage conditioning. GigaBrain-0.5M\* [28] introduces RAMP, in which world-model-predicted future states and values condition policy fine-tuning and are iteratively combined with human-in-the-loop rollout experience. HyVLA-0.5 [89] introduces FlowPRO, a critic-free and reward-free PRO-based offline RL method that learns from paired failure and corrective trajectories. Together, these approaches move VLA adaptation beyond static imitation toward closed-loop learning from policy-generated experience and corrective feedback.

## 2.3. World Model-Powered VLA

World models provide a complementary route for improving embodied foundation models by learning predictive structure from large-scale visual experience. Existing work connects world modeling with robot learning through several distinct roles, ranging from scalable data generation to unified world-action modeling and predictive or evaluative signals for policy learning.

## Data generation and augmentation.

One line of work treats world models as scalable data engines that expand the distribution available for policy training. GigaBrain-0 [27] uses world models to synthesize diverse embodied experience through video generation, Real2Real and Sim2Real transfer, viewpoint variation, and human-to-robot transfer. DreamGen [38] similarly uses video world models to generate synthetic interaction trajectories and recovers corresponding robot actions through latent-action or inverse-dynamics models. These approaches primarily use predictive models to increase the diversity and coverage of training experience beyond what can be collected efficiently on physical robots.

## Unified World-Action Models.

A second direction integrates visual dynamics and action prediction within a common generative model. GR-2 [12] combines future video generation with robot-action prediction. DreamZero [83] builds a World-Action Model on a pretrained video diffusion backbone that jointly models future world evolution and robot actions. Unified World Models (UWM) [95] couples video and action diffusion within a unified architecture that supports policy, forward-dynamics, inverse-dynamics, and video-generation modes. GigaWorld-Policy [82] develops an action-centered World-Action Model, using future visual dynamics to provide additional supervision for action learning while allowing explicit future-video generation to be omitted during deployment. Cosmos Policy [41] fine-tunes pretrained video models for visuomotor control and supports robot-action prediction together with future-state and value prediction for planning. Cosmos 3 [57] further extends world modeling toward an omnimodal foundation model for physical AI.

Beyond unified World-Action Models, World-VLA-Loop [50] studies closed-loop refinement between a video world model and a VLA policy. Policy-generated failure rollouts are used to improve the world model, which subsequently provides an updated learned environment for further VLA reinforcement learning. This formulation allows the world model and policy to improve iteratively rather than treating either component as fixed.

## Policy conditioning and value estimation.

A complementary line retains a VLA-oriented policy while using predictive representations, visual futures, or task-progress estimates to improve policy learning and execution. JEPA-VLA [55] integrates video-predictive visual embeddings into existing VLA models to strengthen representations of temporal dynamics and actionrelevant visual information. π0.7 [63] conditions policy behavior on generated visual subgoals together with language, observation history, and episode-level context. GigaBrain-0.5M\* [28] introduces RAMP, where a separately pretrained world model predicts future states and task values that condition VLA policy learning and support iterative refinement from human-in-the-loop rollout experience. Feat2Go [69] derives continuous task-progress targets from a pretrained visual world model, trains a value model to estimate this progress, and uses the resulting values for reward shaping during policy optimization.

Together, these works establish complementary roles for predictive models in robot learning, including data generation, dynamics representation, World-Action modeling, policy conditioning, value estimation, and closed-loop policy improvement. GigaBrain-0.7 builds on these directions through a dedicated System 3 for prediction and evaluation. System 3 is pretrained separately for robot-centric future-state prediction and task-progress estimation. Its predicted subgoal images and value-derived progress signals are subsequently provided as additional conditions during task-specific VLA post-training. At inference time, the policy is conditioned on positive progress to guide continuous action generation, without explicit candidate-action scoring or selection. The resulting design coordinates predictive modeling, conditional action generation, and subsequent experience-driven refinement while retaining specialized objectives for understanding, prediction, and action.

Table 1: Composition of the embodied trajectory corpus after cleaning.
<table><tr><td>Data Type</td><td>Duration (h)</td><td>Proportion</td></tr><tr><td>Real Robot</td><td>20,535.65</td><td>55.12%</td></tr><tr><td>UMI</td><td>8,251.83</td><td>22.15%</td></tr><tr><td>EGO</td><td>2,862.36</td><td>7.68%</td></tr><tr><td>Simulation</td><td>1,453.92</td><td>3.90%</td></tr><tr><td>WM</td><td>4,153.22</td><td>11.15%</td></tr><tr><td>Total</td><td>37,256.98</td><td>100.00%</td></tr></table>

## 3. Data Curation

To support instruction understanding, visual perception, and continuous control of VLA models in openworld environments, we construct a multi-source training corpus composed of embodied trajectory data and vision-language data. Embodied trajectory data provide state-action sequences, interaction processes, and robotcontrol supervision, while vision-language data preserve the model's general visual-semantic capabilities and further strengthen spatial-relation reasoning, object localization, affordance prediction, and task understanding. These two data families provide complementary supervision during training, mitigating the degradation of vision-language capabilities that can arise from training solely on action data.

## 3.1. Dataset Composition

The training corpus consists of two major components: embodied trajectory data and VLM image-text/questionanswering data. After cleaning, the embodied trajectory corpus contains 37,256.98 hours of data from real robots, Universal Manipulation Interface (UMI) demonstrations [16], egocentric (EGO) human demonstrations, simulation, and world-model-generated (WM) data. These trajectories primarily provide state-action sequences, interaction processes, and robot-control supervision. In parallel, we construct 271,976,674 VLM imagetext/question-answering samples covering image captioning, general visual question answering, spatial-relation reasoning, region grounding, point prediction, affordance understanding, and robotic task understanding. This multimodal corpus preserves broad vision-language capabilities while providing semantic and spatial supervision for embodied tasks. Real-robot trajectories constitute the primary source of continuous-action supervision; UMI and EGO data supplement near-manipulation viewpoints and first-person human-interaction priors; and simulation and WM data provide complementary samples for controlled environments, long-horizon tasks, and specific target distributions. Tab. 1 summarizes the embodied trajectory corpus, and Fig. 2 provides an overview of the complete training-data composition.

## Real-robot data.

The real-robot corpus is assembled from in-house robot deployments and large-scale real-world robot datasets, including [2, 3, 22, 36, 58, 77]. After cleaning and standardization, the real-robot corpus spans 16 robot types, 1,810,101 episodes, 2,077,837,071 frames, and a total of 20,535.65 hours. Maker H01, Maker M01, Agibot-G1, AgileX, Ark, and Galaxea R1 Lite constitute the major data sources, while Franka, UR5, Realman RMC-AIDA-L, and other platforms further extend cross-embodiment and cross-hardware coverage. This distribution provides the basis for learning action patterns across different degrees of freedom, end-effectors, and locomotion modalities. Tab. 2 reports the detailed statistics by robot type.

## UMI and EGO data.

The UMI corpus is composed of in-house Maker U01 demonstrations and the Jianzhi 10K dataset [26]. After cleaning, it contains 8,251.83 hours, including 6,876.25 hours of in-house data and 1,375.58 hours from Jianzhi 10K. This data source provides large-scale human manipulation demonstrations from near-manipulation viewpoints that visually resemble local observations from wrist-mounted or end-effector-proximal robot cameras. It therefore complements real-robot trajectories with supervision for hand-eye coordination, local spatial

4.2kh / 11.15%

Maker H01

Real Robot

## Dataset Composition Overview

![](images/534dec919c6e822d560e84d46b567273f8b2896740d85b1ed2902a919f741e17.jpg)  
20.5kh / 55.12%  
Physical robot trajectories collected from deployed platforms

![](images/c96cb5d9ed0c7dde8b0faea66be769b25f17419162edad68d205193c233beeeb.jpg)  
2.9kh / 7.68%  
First-person human operation videos aligned with robot views

![](images/23842f18826f7b3acba1758542465b4960ec810f66092f80a3be001eddad977e.jpg)  
8.3kh /  22.15%  
Near-hand demonstrations supporting fine-grained manipulation

## Operating-hours allocation

Generated trajectories that expand temporal and scene diversity

<table><tr><td rowspan=2 colspan=1>Real Robot</td><td></td></tr><tr><td rowspan=1 colspan=1>20.5kh |  55.12%</td></tr><tr><td rowspan=1 colspan=1>UMI</td><td rowspan=1 colspan=1>8.3kh | 22.15%</td></tr><tr><td rowspan=1 colspan=1>EGO</td><td rowspan=1 colspan=1>2.9kh | 7.68%</td></tr><tr><td rowspan=1 colspan=1>Simulation</td><td rowspan=1 colspan=1>1.5kh | 3.90%</td></tr><tr><td rowspan=1 colspan=2>World Model         4.2kh |11.15%</td></tr></table>

![](images/97c4da418c0daa8ab01d7aa942cafb044a246d0ffa9b920b81de89b5f05a7ddd.jpg)

![](images/dfe0b576e4d6f96259e8b2dde9120d38b61ca0a4d204994d3191c8e06dce9079.jpg)  
1.5kh / 3.90%  
Virtual environments and task rollouts for scalable coverage

![](images/570fe8eec9f4ad92956805f1e61aff5b90b5d177f66067c5762ee7559b54534a.jpg)  
For VLM image-text question answering

A person lying down and a black dog.

Q: What color hat is the person wearing?

Q: What is in the person's hand? A: frisbee

## Atomic Actions

![](images/eefe3bc1397dca6d5d0848260c38e048acee46124b7306f00c0a43bb50461b31.jpg)

![](images/2167bcf5a8f7089a2f6a87bb9e460022cfb4fc3cd3a4186857e04b5b353617a7.jpg)

## Platform and Data-Capture Device Examples

## Environment & Task Distribution

Representative robot embodiments and data-capture devices; H01, M01, U01 and E01 are emphasized.

![](images/166ef8c845a3e357b876bbdd8063254c201e4dd36ec1479fe41ec6a83458e814.jpg)

![](images/e5c6db1c9c09cc2821ba15ec6ae86c26bb46bcda6c8065798d743eac8cffe763.jpg)  
Maker M01

![](images/831dc3f4449492d7a2b6abf266856dbb9a1826245736677ce7c9ae8538c52c16.jpg)  
Maker U01

![](images/cb87fe33bbc5c9e17b7360b1cb461b57d35f4d5346ba2721dc2cfd81f6ead489.jpg)  
Maker E01

![](images/598fc71e255eb66e1da5384b8ee1f90c7234ce90771bfd51284f450b60a570b7.jpg)  
Figure 2: Composition of the GigaBrain-0.7 training corpus. The embodied trajectory corpus contains 37.3K hours of real-robot, UMI, EGO, simulation, and world-model-generated data, while the VLM corpus contains 271,976,674 image-text/question-answering samples. The figure further summarizes representative robot embodiments, data-capture devices, atomic actions, and task/environment distributions.

Table 2: Statistics of the real-robot data by robot type.
<table><tr><td>Robot Type</td><td>Episodes</td><td>Frames</td><td>Frame Prop.</td></tr><tr><td>Agibot-G1</td><td>226,675</td><td>380,282,116</td><td>18.30%</td></tr><tr><td>AgileX</td><td>133,101</td><td>179,727,199</td><td>8.65%</td></tr><tr><td>Maker H01</td><td>866,864</td><td>910,496,160</td><td>43.82%</td></tr><tr><td>Ark</td><td>30,559</td><td>35,960,476</td><td>1.73%</td></tr><tr><td>Galaxea R1 Lite</td><td>27,793</td><td>35,058,414</td><td>1.69%</td></tr><tr><td>Franka</td><td>87,148</td><td>25,149,233</td><td>1.21%</td></tr><tr><td>UR5</td><td>57,210</td><td>25,007,639</td><td>1.20%</td></tr><tr><td>Maker M01</td><td>319,964</td><td>418,534,560</td><td>20.14%</td></tr><tr><td>Realman RMC-AIDA-L</td><td>19,419</td><td>17,087,665</td><td>0.82%</td></tr><tr><td>Dexmal DOS-W1</td><td>12,043</td><td>16,122,952</td><td>0.78%</td></tr><tr><td>Agibot-G2</td><td>6,092</td><td>15,641,535</td><td>0.75%</td></tr><tr><td>TienKung</td><td>11,218</td><td>7,598,429</td><td>0.37%</td></tr><tr><td>Galbot G1</td><td>5,452</td><td>5,175,924</td><td>0.25%</td></tr><tr><td>Ark-mobile</td><td>4,295</td><td>4,318,879</td><td>0.21%</td></tr><tr><td>Unitree G1edu</td><td>1,411</td><td>910,983</td><td>0.04%</td></tr><tr><td>AI2 Alphabot 2</td><td>857</td><td>764,907</td><td>0.04%</td></tr></table>

Table 3: Composition of the UMI and EGO datasets after cleaning.
<table><tr><td>Data Type</td><td>Dataset</td><td>Duration (h)</td></tr><tr><td>UMI</td><td>In-house Maker U01</td><td>6,876.25</td></tr><tr><td>UMI</td><td>Jianzhi 10K</td><td>1,375.58</td></tr><tr><td>UMI Total</td><td>一</td><td>8,251.83</td></tr><tr><td>EGO</td><td>EgoDex</td><td>724.84</td></tr><tr><td>EGO</td><td>EgoVerse</td><td>835.77</td></tr><tr><td>EGO</td><td>WiYH</td><td>165.38</td></tr><tr><td>EGO</td><td>In-house Maker E01</td><td>1,136.37</td></tr><tr><td>EGO Total</td><td>一</td><td>2,862.36</td></tr></table>

geometry, and object-contact relationships.

The EGO corpus contains 2,862.36 hours after cleaning and combines in-house Maker E01 data with EgoDex [35], EgoVerse [64], and World in Your Hands (WiYH) [91]. Because first-person human videos share similar observation viewpoints with robot-mounted cameras, they provide broader object categories, manipulation scenes, and human-interaction priors, strengthening semantic understanding and temporal modeling of real manipulation processes. Tab. 3 summarizes the UMI and EGO data composition

## Simulation and generated data.

To complement physically collected trajectories, we further construct 5,607.14 hours of simulation and generated data through two complementary pipelines. The simulation subset contains 1453.92 hours of trajectories collected from configurable physics-based environments, including high-quality trajectories curated from [15, 24, 31, 33, 54], while the world-model-generated subset contributes 4,153.22 hours of data derived from real interaction samples. The two sources target complementary regimes: simulation provides controllable coverage of task and scene configurations that are costly or difficult to reproduce on physical robots, whereas generation expands the visual and contextual diversity of existing real-world interaction data. Together, they increase coverage of long-tail interaction conditions while reducing the need to repeatedly collect rare or safety-sensitive scenarios on physical hardware.

## VLM image-text/question-answering data.

In addition to embodied trajectory data, we construct 271,976,674 VLM image-text/question-answering samples, including 15,234,327 self-collected samples, to preserve foundational vision-language capabilities and strengthen the spatial understanding, object localization, and affordance reasoning required by robotic tasks. The corpus covers image captioning, general visual question answering, multi-image reasoning, region grounding, point prediction, spatial-relation reasoning, robotic task understanding, and affordance learning.

The captioning data include [11, 47, 67, 68, 84, 85], and image-text samples converted from [4], providing supervision for open-vocabulary recognition and image-level semantic description. The general VQA data include [18, 21, 30, 43, 49, 72, 76]. Embodied-specific data include [13, 42, 45, 73, 81, 88, 93], strengthening spatial referring, target localization, manipulation-feasibility judgment, and trajectory-related reasoning in robotic scenes.

## 3.2. Embodied Trajectory Data Processing Pipeline

Robot trajectories and human manipulation data from heterogeneous sources differ substantially in storage format, robot degrees of freedom, coordinate definitions, action frequency, camera naming, language-annotation granularity, and quality distribution. To enable stable joint training across these sources, we develop a unified embodied-trajectory processing pipeline that converts raw data into standardized representations and applies multi-stage quality control before export.

![](images/cce26b77484718607f23f30e4b8c5a8e33c54ff3c774252351bfd9a4218e29f9.jpg)  
Figure 3: Real-robot data processing pipeline. Heterogeneous robot data are converted into the LeRobot v3.0 format, mapped to a unified robot representation, standardized through language-instruction rewriting and subtask annotation, and filtered through multi-stage quality control before training.

## Data format conversion and metadata organization.

We first convert raw data into a unified training-storage format. For LeRobot datasets, all samples are standardized to the LeRobot v3.0 format, improving data-loading efficiency and unifying the organization of episode, frame, observation, action, and language instruction fields. During conversion, we also organize dataset provenance, robot type, task description, camera list, sampling frequency, state dimensionality; action dimensionality, and cleaning flags, enabling downstream training to perform precise filtering and sampling by data source, robot morphology, and task type.

## Unified robot representation.

To handle mismatched degrees of freedom and sensor configurations across robot embodiments, we construct a unified robot state and action representation. The dimension ordering follows left arm, right arm, head, waist, and base or legs, while both joint-space and end-effector-space fields are retained to support different supervision signals. Under this representation, joint states, end-effector poses, gripper states, base velocities, and camera observations are organized into a reusable standard format. Missing motion components are represented with masks or placeholder dimensions, avoiding embodiment-specific data formats for individual robot morphologies. Because camera naming and mounting conventions also vary across platforms, preprocessing maps camera fields from each source to standardized camera definitions while retaining the original camera names for traceability.

## Language instruction rewriting and subtask annotation.

Raw task instructions often contain inconsistent naming, mixed languages, heterogeneous description granularity, or weak correspondence with the actual task. For example, the same manipulator or gripper may be referred to as left/right/both gripper or hand across different data sources; some Chinese instructions must be aligned with English training corpora; and some samples contain only scene-level words without executable task descriptions. To address these inconsistencies, we adopt an LLM-assisted instruction-standardization pipeline. We first extract instructions using rule-based procedures, rewrite them into canonical task instructions with GLM-5.1 [29], and write the normalized instructions back to the dataset, followed by human spot checks. To improve modeling of long-horizon task structure, we further construct subtask-level annotations. For robot manipulation videos, a multimodal large model temporally segments the manipulation process and generates atomic subtask descriptions; samples with ambiguous task boundaries or unstable model annotations are further corrected through human annotation and review.

## Multi-stage data cleaning.

The cleaning stage filters or corrects numerical anomalies, prolonged stationary segments, kinematic inconsistencies, and training-anomalous samples.

First, we perform q01/q99 extreme-value filtering. Because quantile normalization is used during training, extreme outliers can substantially stretch the normalization interval and compress normal action signals into a narrow range, thereby destabilizing optimization. We therefore compute the 1st and 99th percentiles independently for each robot type and action dimension and remove anomalous frames or trajectories outside the valid range.

Second, we remove prolonged stationary segments. We compute inter-frame state differences, normalize each dimension by the corresponding joint range of motion, and then calculate the overall $L _ { 2 }$ change magnitude. When the magnitude falls below a predefined threshold, the corresponding frame pair is classified as stationary; consecutive stationary pairs are merged into stationary segments. This procedure reduces the proportion of long, uninformative idle periods and focuses training on effective interaction behaviors.

Third, following Qwen-RobotManip [87], we perform end-effector-pose consistency correction. For each robot type in the training corpus, we collect the corresponding URDF files, align the base coordinate systems, and manually configure the end-effector reference point, thereby standardizing end-effector pose definitions across embodiments.

Finally, we use anomalous training losses as an additional filtering signal. During early pretraining, we record sample-level and data-source-level loss distributions. If particular trajectories persistently produce abnormally high losses or exhibit large loss spikes, we trace them back to the original video, action sequence, and language annotation to identify temporal misalignment, action discontinuities, incorrect instructions, or video corruption. Confirmed anomalous samples are then added to the filtering list.

## UMI data processing.

For UMI demonstrations, we first synchronize multi-camera images, handle poses, gripper states, and host-side timestamps to ensure temporal consistency between visual observations and manipulation trajectories. The acquisition system relies on synchronized multi-camera exposure and triggering, while handle-pose and gripperopening signals are recorded through wired connections, providing a common temporal basis for subsequent cross-modal alignment, trajectory retargeting, and executability validation.

![](images/cf9ff58d117b55a77d778f5b1f23151f797bf1053d34d43f196c9cb459ff6b25.jpg)  
Figure 4: UMI data processing pipeline. Multi-camera observations, 6-DoF handle poses, and gripper states are temporally aligned, filtered for trajectory and modality quality, and retargeted to the Maker H01 end-effector frame before simulation and sampled real-robot replay validation.

![](images/13054674ff2b9a38c9e6d5c1a50ee9005a32d7d1655c8b0e9805e8c6e1e6106a.jpg)  
Figure 5: EGO data processing pipeline. First-person manipulation videos are filtered for interaction and video quality, supplemented with hand-pose and hand-object information when necessary, and standardized across EgoDex, EgoVerse, WiYH, and in-house data sources.

For quality control, we screen raw clips according to trajectory smoothness, information density, cross-modal consistency, and embodiment executability. We remove samples with pose loss, rapid jitter, prolonged stationary periods, or clear inconsistencies between projected handle motion and visual observations. The retained trajectories are then mapped to the Maker H01 end-effector Tool Center Point (TCP) frame and validated through inverse-kinematics solving, simulation-based executability checks, and sampled real-robot replay. After processing, UMI data are organized into unified training samples containing visual observations, end-effector actions, timestamps, and task-semantic information, and are jointly used with real-robot data for policy learning.

## EGO data processing.

EGO data supplement first-person examples of human manipulation, allowing the model to observe how human hands approach target objects, establish contact, and complete actions such as moving, placing, or opening objects.

We first apply video-level quality filtering and remove clips without clear hand-object interaction, with corrupted frames, prolonged stationary periods, severe occlusion, or viewpoints unsuitable for interpreting the manipulation process. This ensures that retained clips clearly expose the hand, target object, and resulting

## manipulation outcome.

We then standardize video frames, camera parameters, hand trajectories, and existing action annotations. Timestamps, coordinate systems, and field formats are normalized across data sources so that video observations, hand motion, and action labels can be chronologically organized into training samples consistent with robot trajectories. For data with hand keypoints or hand trajectories, we organize the temporal evolution of hand positions and associate them with contacted or manipulated target objects. For samples without hand- or object-position annotations, we use camera-trajectory estimation, hand-pose estimation, or existing annotations to supplement interaction locations, target objects, and their relative spatial relationships. After processing EGO samples enter training as current observations paired with language descriptions and future actions or hand-motion trajectories, strengthening the model's understanding of first-person manipulation processes, hand-object contact relationships, and near-field spatial interactions.

## 3.3. Simulation and Generated Data

Real-robot collection provides physically grounded interaction data, but scaling coverage to rare, long-tail, or safety-sensitive conditions through physical collection alone remains costly. We therefore complement realworld data with two scalable production pipelines: world-model-based video generalization and physics-based simulation. Together, these pipelines contribute 5,607.14 hours to the embodied pretraining corpus.

## World-model-based data generation.

The generation pipeline starts from curated real-world interaction samples and expands their scene and visual conditions through prompt-conditioned video generation. Generation tasks are constructed from the source interaction together with configurable scene and generation conditions, producing additional samples that preserve the underlying interaction context while broadening the visual and environmental coverage of the training distribution. After generation, the resulting samples are organized under the same data management and quality-control framework as the remaining embodied corpus.

## Physics-based simulation.

In parallel, we construct simulation tasks from configurable robot, object, and scene assets. Each task is instantiated in a physics-based environment and executed to produce temporally consistent simulated interaction data. Compared with physical collection, this pipeline provides a controllable and repeatable source of interaction experience for configurations that are difficult to reproduce frequently or safely on real hardware.

## Complementary long-tail coverage.

The two pipelines serve complementary purposes. World-model-based generation expands the diversity of real interaction samples while retaining their visual and behavioral context, whereas physics-based simulation provides explicit control over task and scene configurations. Their combination broadens the coverage of long-tail interaction conditions and provides additional experience beyond what can be collected efficiently from physical robots alone.

## 3.4. VLM Data Processing and Annotation

## VLM data processing pipeline.

The VLM corpus combines public image-text/question-answering datasets, converted video-text resources, embodied-specific VLM datasets, and self-collected embodied annotations derived from our trajectory data. We process all sources into a unified, training-ready multimodal dialogue format. The pipeline consists of annotation-format unification, image quality control, and sample-structure validation, ensuring consistency in text fields, image files, visual placeholders, dialogue-role ordering, and task metadata before training.

During annotation-format unification, image descriptions, single-turn question answering, multi-turn question answering, and embodied-specific supervision are mapped into a common dialogue representation according to task type, with appropriate image placeholders retained for samples that require visual input. This step reduces formatting discrepancies across data sources and enables direct integration with the downstream training framework. During image quality control, we check image readability, resolution, aspect ratio, color mode, and content validity, standardizing repairable samples and removing samples that are missing, corrupted, or insufficiently informative. Before final export, we further validate field completeness, image-text path consistency, dialogue-role ordering, and non-empty text constraints, producing structurally consistent and quality-controlled VLM training data.

## Embodied VLM annotation pipeline.

In addition to aggregating public VLM datasets, we generate self-collected embodied VLM annotations from the cleaned real-robot, UMI, and EGO data. We extract keyframes, short video clips, historical observation windows, and the corresponding task instructions or trajectory context, and organize them into annotation units that can be processed by multimodal models. We then use Qwen3.6-27B [66] for large-scale batch annotation. Following the annotation strategy in [9], the model extracts intermediate structured states including object instances, interactable parts, spatial relations, viewpoint information, task stages, and action intent, from which VLM supervision samples are subsequently derived.

## Affordance prediction.

For affordance-prediction tasks, the annotation pipeline focuses on identifying the interactable properties of objects and object parts, including graspable regions, pressable or pullable components, placeable regions, and manipulable objects relevant to the task objective. The multimodal model generates candidate interaction regions and corresponding language descriptions from object states in images or videos. These annotations are further transformed into supervision for target localization, component identification, affordance judgment, and placement-region selection, enabling the model to learn correspondences among object function, manipulation constraints, and locally reachable space.

## Spatial understanding.

For spatial-understanding tasks, the annotation pipeline constructs spatial relationships among objects and between objects and the observer. For data with depth, multi-view observations, or camera parameters, geometric information is preferentially used to generate supervision for relative position, proximity, occlusion, containment, and free space. For RGB-only images or first-person videos, candidate spatial relationships are generated by combining visual cues with model reasoning and are subsequently filtered through rule-based consistency checks. This data is primarily used to construct spatial-relation judgment, referring-expression localization, region-grounding, and spatial-feasibility question-answering samples.

## High-level task planning.

For high-level task-planning tasks, annotations are built from real-robot trajectories, UMI near-manipulation demonstrations, and EGO first-person human-manipulation clips, with emphasis on task stages, current subtasks, next-step actions, and task-completion states. The multimodal model performs temporal understanding over continuous clips and generates current-state summaries, subtask labels, action pre/post relationships, and plausible next operations. These structured annotations are further converted into next-step action prediction, task-progress judgment, action-consequence prediction, and goal-consistency question-answering samples, providing supervision for long-horizon task understanding and high-level planning.

## Quality control.

To reduce noise from automatic generation, we apply multi-level quality control after data production. Samples are first automatically filtered for geometric consistency, target uniqueness, temporal consistency, and answerformat compliance. Low-confidence samples, non-unique answers, and semantically ambiguous cases are further selected for human spot-check review. Retained samples are then converted into a unified multimodal dialogue format, with data provenance, task type, generation rules, and quality status recorded for downstream sampling, error tracing, and dataset version management

![](images/779f8ca99fa7b311e0458d5917a9656ddccc4d59a9364cb759633a796bb0826d.jpg)  
Figure 6: Architecture of GigaBrain-0.7. GigaBrain-0.7 organizes embodied intelligence into three interacting systems. System 2 performs visual-language understanding and task planning. System 3 introduces explicit prediction and evaluation by predicting future observations and estimating task-related values. System 1 integrates the current observation and short-term memory, subtask instructions, robot state, and predictive signals to generate executable actions through parallel discrete and continuous action pathways. The architecture further exposes interfaces for offline and online experience reinforcement, connecting perception, prediction, action, and subsequent policy improvement.

## 4. Model Architecture

Vision-language-action models provide a scalable interface between visual-language understanding and continuous robot control. A common design couples a pretrained vision-language model with a dedicated action generator, allowing semantic representations to condition low-level motor commands. For long-horizon physical interaction, however, understanding the current scene and predicting the next action are not sufficient on their own. The policy must also maintain temporal context, anticipate how the scene may evolve, and assess whether its current behavior is making useful progress.

GigaBrain-0.7 addresses these requirements through a three-system architecture, as illustrated in Fig. 6. System 2 is responsible for understanding and planning, System 3 for prediction and evaluation, and System 1 for action and control. System 2 is a PaliGemma2 (3B) [71] vision-language model that interprets the scene and decomposes tasks. System 1 extends a PaliGemma2 (3B) backbone with a dedicated Action Expert (0.5B) for continuous action generation. System 3 is an independent GigaWorld-1 [75] based world value model (5B) that provides System 1 with predictive signals about future task evolution.

We organize the architecture into three progressively coupled layers. World simulation provides semantic understanding, task planning, future prediction, and evaluation. Action alignment maps these multimodal signals to continuous control across heterogeneous robot embodiments. Experience reinforcement connects policy execution back to offline and online policy improvement.

## 4.1. World Simulation: Understanding, Prediction, and Evaluation

The world-simulation layer provides System 1 with context beyond the current visual observation. It combines System 2, which interprets the current scene and decomposes the task, with System 3, which explicitly models future observations and task-related values. System 2 describes what should be done next, while System 3 provides prospective information about what may happen next and how the current execution state is evaluated.

## System 2: Understanding and Planning.

System 2 receives the current visual observation together with the task instruction. It produces chain-of-thought reasoning that describes the current visual context—identifying relevant objects and their spatial arrangement followed by a subtask instruction for the immediate manipulation objective. For example, given the instruction “hang the item on the hanger," System 2 may observe “there are hats, bags..."and decompose the task into the subtask "pick up the red hat" (Fig. 6).

This decomposition provides an intermediate interface between high-level task understanding and low-level action generation. The resulting subtask information is passed to System 1 and also provides task context for System 3, allowing prediction and control to operate on a more explicit representation of the current task stage.

## System 3: Prediction and Evaluation.

System 3 is the predictive component of GigaBrain-0.7. Built upon GigaWorld-1 [75], it receives the current embodied context—including the current observation, proprioceptive state, and subtask description from System 2 (Fig. 6). Unlike System 2, which reasons over observations that are already available, System 3 explicitly models information about future task evolution. System 3 exposes two complementary signals to System 1.

Future observation. Conditioned on the current embodied context and subtask instruction, the world model generates a short video whose temporal extent is aligned with the subtask horizon of System 1. The last frame of this predicted sequence is extracted as the subgoal image gt—a compact visual representation of the physical state expected after short-horizon task progress. It is encoded and provided to System 1 as an additional visual condition, allowing action generation to incorporate an explicit representation of the anticipated scene evolution.

Value-based evaluation. System 3 additionally estimates a scalar value Vt representing the current task progress—how far the current execution state has advanced toward subtask completion. While sharing the same world-model backbone as the video pathway, the value pathway attends primarily to visual observations. The value estimate is converted into a binary advantage condition $A _ { t } \in \{ 0 , 1 \}$ , indicating whether the predicted task progress is increasing or not over the current action segment. As shown in Fig. 6, this value-derived condition is injected into System 1's prompt, providing a compact evaluation of the current execution state. The derivation of $A _ { t }$ from $V _ { t }$ is described in Section 5.3.

Together, the subgoal image captures anticipated physical outcomes in visual space, while the advantage provides a scalar evaluation signal for action generation and subsequent experience-based learning.

## 4.2. Action Alignment: Multi-Embodiment Action Generation

The action-alignment layer converts multimodal context into executable robot control. This role is performed by System 1, a vision-language-action model that integrates semantic planning, predictive information, temporal context, and embodiment-specific robot states within a shared action-generation framework.

## System 1: Action and Control.

System 1 receives the current observation and short-term visual memory, the chain-of-thought output and subtask from System 2, proprioceptive state and Robot ID, and predictive signals from System 3 (Fig. 6). These inputs are fused by the VLA to produce action chunks for closed-loop robot execution.

System 1 combines its PaliGemma2 (3B) vision-language backbone with the Action Expert (0.5B) in a Mixture-of-Transformers (MoT) design. Let $\mathbf { H } _ { l } ^ { \mathrm { V L } }$ and $\mathbf { H } _ { l } ^ { \mathrm { A E } }$ denote the hidden states of the vision-language and action-expert streams at layer l. The two streams are coupled through joint attention but maintain separate feed-forward parameters:

$$
{ \bf O } _ { l } ^ { \mathrm { V L } } = \mathrm { C a u s a l A t t n } _ { l } \left( { \bf H } _ { l } ^ { \mathrm { V L } } ; { \bf H } _ { l } ^ { \mathrm { V L } } \right) ,\tag{1}
$$

$$
\mathbf { O } _ { l } ^ { \mathrm { A E } } = \mathrm { A t t n } _ { l } \left( \mathbf { H } _ { l } ^ { \mathrm { A E } } ; \ \left[ \mathbf { H } _ { l } ^ { \mathrm { V L } } , \mathbf { H } _ { l } ^ { \mathrm { A E } } \right] \right) ,\tag{2}
$$

$$
\mathbf { H } _ { l + 1 } ^ { \mathrm { V L } } = \mathrm { F F N } _ { l } ^ { \mathrm { V L } } \big ( \mathbf { O } _ { l } ^ { \mathrm { V L } } \big ) , \quad \mathbf { H } _ { l + 1 } ^ { \mathrm { A E } } = \mathrm { F F N } _ { l } ^ { \mathrm { A E } } \big ( \mathbf { O } _ { l } ^ { \mathrm { A E } } \big ) ,\tag{3}
$$

where $\operatorname { A t t n } ( \mathbf { Q } ; \mathbf { C } )$ denotes attention with queries from Q and keys/values from context C, [·, ·] denotes sequence concatenation, and residual connections and layer normalizations are omitted for clarity. The vision-language stream uses causal self-attention over its own tokens (Eq. 1), preserving the pretrained VLM's autoregressive interface. The action expert attends bidirectionally to the union of both streams (Eq. 2), enabling grounded action generation informed by the full semantic context without injecting action-specific representations into the VLM attention computation.

## Short-term visual memory.

Physical interaction is inherently temporal. The same instantaneous observation may correspond to different task states depending on preceding interactions, while temporary occlusions and repeated visual configurations make single-frame perception insufficient for many long-horizon tasks.

GigaBrain-0.7 therefore incorporates short-term visual memory through Temporal-Spatial Blocks inserted into the visual encoder. Given the current observation together with a short history of preceding frames, each block factorizes visual aggregation into temporal and spatial processing. The temporal operation causally aggregates information across recent frames, allowing the current representation to capture how task-relevant regions evolve over time. The subsequent spatial operation models interactions among visual regions within the current observation, integrating the temporally enriched features into a coherent scene representation.

Rather than forwarding all historical visual tokens to the VLA backbone, the Temporal-Spatial Blocks progressively fuse historical information into the current-frame representation. After temporal-spatial processing, past-frame tokens are discarded and only the enriched current-frame tokens are retained. The number of visual tokens entering the VLA backbone therefore remains comparable to the single-frame setting, providing temporal context without naively increasing the high-level context length with the number of history frames.

## Discrete and continuous action pathways.

System 1 supports two complementary action representations. The NTP-based discrete pathway generates discrete action tokens autoregressively through the vision-language head, providing a symbolic action representation compatible with next-token prediction. The FM-based continuous pathway generates action chunks through the Action Expert via conditional flow matching, taking noise-corrupted action representations as input and predicting continuous trajectories for precise robot control.

Both pathways share the same visual-language context through the MoT joint attention but serve different roles: the discrete pathway bridges language understanding with motor control during training, while the continuous pathway provides the executable actions used at deployment.

## Soft Knowledge Insulation.

A central difficulty in VLA learning is balancing pretrained vision-language representations with robot-specific action learning. The VLM must remain sensitive to language, objects, spatial relations, and task semantics, while adapting sufficiently to support physical control.

GigaBrain-0.7 introduces Soft Knowledge Insulation (Soft KI) between the vision-language and continuous-

![](images/e89e152c254563a1a0c091df9ef96c4a47f23ede3f8653133a3d6591a6adfb31.jpg)  
Figure 7: Experience Reinforcement Pipeline. The experience reinforcement pipeline consists of three stages: supervised fine-tuning, offline reinforcement, and online reinforcement. Supervised fine-tuning establishes basic competence on complex, fine-grained out-of-domain tasks. Offline reinforcement develops progress awareness and error recovery, while online reinforcement further improves fine-grained manipulation and task success.

action pathways. Rather than completely isolating semantic representations from action learning [5], Soft KI provides a controlled interface that allows the backbone to develop embodied-aware visual representations while preserving its general-purpose vision-language capabilities.

## Embodiment-aware state interface.

Scaling a single VLA across heterogeneous robots requires alignment between different state spaces and kinematic configurations. Following the unified robot representation in Section 3, GigaBrain-0.7 provides System 1 with both the proprioceptive state and a Robot ID through a dedicated projection module. The proprioceptive state is element-wise masked by a per-embodiment validity vector—distinguishing absent dimensions from zero-valued states—and projected together with the Robot ID into the shared VLA representation. This corresponds to the "Projection" module in Fig. 6.

The main Action Expert parameters are shared across embodiments, while per-embodiment input and output projections handle differences in native action definitions. This allows different robot platforms to share the core manipulation knowledge while preserving embodiment-specific precision. Rotations are represented using the continuous 6D format [94] throughout the action space.

## 4.3. Experience Reinforcement

As illustrated on the right side of Fig. 6, GigaBrain-0.7 builds an experience-reinforcement interface around System 1 that connects the post-trained policy to both offline and online learning stages. First, human-in-theloop (HIL) intervention data generated by System 1 are used for offline learning, yielding an error-aware progress discriminator and a policy with a certain degree of error-recovery capability. During online interaction, subtask success labels, together with the progress discriminator, are used to construct dense rewards, while the policy obtained from offline learning is connected to a lightweight actor-critic loop. The same System 1 policy is retained throughout pretraining, post-training, and deployment, allowing supervised learning, offline experience learning, and online interaction to form a continuous model lifecycle without introducing a separate controller.

## 5. Model Training

Training proceeds in four stages. We first pre-train Systems 1 and 2 jointly as a generalist vision-language-action policy on the heterogeneous corpus described in Section 3. We then extend GigaWorld-1 into a World Value Model through robot-centric video pre-training followed by value learning, producing System 3. During task-

specific VLA post-training, System 3 is frozen and its outputs condition System 1 on downstream demonstrations while Systems 1 and 2 are jointly optimized. Finally, the post-trained policy is refined through offline and online reinforcement learning on its own deployment experience.

## 5.1. VLA Pre-Training

We jointly train Systems 1 and 2 on the full heterogeneous corpus—real robot, UMI, EGO, simulation, worldmodel-generated trajectories, and vision-language data—using a unified pre-training recipe that jointly optimizes semantic understanding, hierarchical task prediction, and continuous action generation.

## Joint vision-language-action learning.

GigaBrain-0.7 supports two complementary prediction pathways (Section 4.2): an autoregressive next-token prediction (NTP) pathway for language, subtask, and discrete-action outputs, and a flow-matching (FM) pathway for continuous action chunks. Unlike staged training recipes that introduce continuous action generation only in a later training phase, both pathways are optimized throughout GigaBrain-0.7 pre-training.

For an autoregressive target sequence $y _ { 1 : M }$ conditioned on multimodal context $c ,$ the NTP objective is

$$
\mathcal { L } _ { \mathrm { N T P } } = - \mathbb { E } \left[ \sum _ { j = 1 } ^ { M } m _ { j } \log p _ { \theta } \left( y _ { j } \mid c , y _ { < j } \right) \right] ,\tag{4}
$$

where $m _ { j }$ masks positions without autoregressive supervision. Depending on the data source, the targets include subtask descriptions, discrete action representations, and general vision-language outputs such as captioning, VQA, spatial reasoning, grounding, and affordance understanding.

For continuous action generation, let $a _ { t : t + H }$ denote a ground-truth action chunk and let $\epsilon \sim \mathcal { N } ( 0 , I )$ denote Gaussian noise of the same dimensionality. Following the flow-matching formulation used in prior GigaBrain models [27], we construct

$$
a _ { t : t + H } ^ { \tau } = \tau a _ { t : t + H } + ( 1 - \tau ) \epsilon , \qquad \tau \in [ 0 , 1 ] .\tag{5}
$$

The Action Expert predicts the corresponding flow field and is optimized with

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { a , \epsilon , \tau } \left[ \left| \left| v _ { \theta } \left( a _ { t : t + H } ^ { \tau } , c , \tau \right) - \left( a _ { t : t + H } - \epsilon \right) \right| \right| _ { 2 } ^ { 2 } \right] .\tag{6}
$$

The two pathways are jointly optimized during pre-training:

$$
\mathcal { L } _ { \mathrm { V L A } } = \mathcal { L } _ { \mathrm { N T P } } + \mathcal { L } _ { \mathrm { F M } } .\tag{7}
$$

To reduce interference between continuous-action learning and the pretrained vision-language representation, we apply Soft Knowledge Insulation [20]. Specifically, FM gradients are propagated fully within the Action Expert but attenuated by a coefficient $\alpha _ { \mathrm { K I } }$ before entering the vision-language backbone:

$$
\nabla _ { \theta _ { \mathrm { V L } } } \mathcal { L } _ { \mathrm { V L A } } = \nabla _ { \theta _ { \mathrm { V L } } } \mathcal { L } _ { \mathrm { N T P } } + \alpha _ { \mathrm { K I } } \nabla _ { \theta _ { \mathrm { V L } } } \mathcal { L } _ { \mathrm { F M } } , \qquad 0 < \alpha _ { \mathrm { K I } } < 1 .\tag{8}
$$

This preserves full autoregressive supervision of the vision-language backbone while still allowing continuousaction learning to adapt its representations toward embodied control.

## Temporal context supervision.

To train the Temporal-Spatial Blocks described in Section 4.2, we provide historical observations alongside the current frame. Historical observations are randomly dropped during training, exposing the model to varying amounts of temporal context—including single-frame inputs—and reducing dependence on a fixed history length.

## Hierarchical task supervision.

Each trajectory carries both a task-level instruction l and a temporally aligned subtask instruction $\boldsymbol { \hat { \ell } } _ { t } .$ We mix two supervision modes. In the first, both l and $\hat { \ell } _ { t }$ are provided and the model directly predicts the corresponding action. In the second, only the task-level instruction is provided, requiring System 2 to autoregressively infer the current subtask before System 1 generates the corresponding action. This couples high-level task reasoning and low-level control within the same pre-training framework. Causal attention masks prevent subtask targets and future actions from leaking into the representations used to predict them.

## Multi-embodiment unified optimization.

The pre-training corpus spans embodiments with substantially different morphologies, degrees of freedom, and control conventions. Following the unified representation in Section 3, heterogeneous states and actions are aligned to a common semantic structure while retaining embodiment-specific validity information.

Each embodiment's proprioceptive state is projected into the shared model space through the embodiment-aware interface described in Section 4.2. Validity masks distinguish unavailable dimensions from valid zero-valued states, and invalid action dimensions are excluded from both flow-matching supervision and noise injection. The Action Expert shares its main Transformer parameters across embodiments, while embodiment-specific input and output projections accommodate differences in action dimensionality and control convention.

For egocentric data, future actions are re-expressed in the coordinate frame of the current observation to provide a stable reference throughout the action chunk. Rotations use the continuous 6D representation defined in Section 4.2.

## 5.2. World Model Pre-Training

System 3 provides two signals to System 1 (Section 4.1): a predicted future observation and a task-progress value estimate. We obtain both capabilities by extending GigaWorld-1 [75] through two successive training stages.

## Stage I: Robot-centric video pre-training.

Starting from GigaWorld-1, we continue video pre-training on robot manipulation data drawn from the realrobot and simulation tiers of the data pyramid (Section 3). The objective is to adapt general visual dynamics toward the state transitions and contact interactions that characterize robotic manipulation.

Conditioned on the current embodied context and subtask instruction, the model generates a future video whose temporal extent matches the action-chunk horizon of System 1. The final predicted frame serves as the subgoal image $g _ { t } - \mathbf { a }$ compact visual representation of the physical state expected after short-horizon task progress.

## Stage II: Value learning.

Future-state prediction tells the policy what may happen, but not whether the current state is favorable for task completion. In a second stage, we equip the model with task-progress estimation.

We extend the video-pre-trained checkpoint into a Mixture-of-Transformers World Value Model that jointly supports future prediction and value estimation. Similar to Viva [53], the value pathway is supervised to estimate task progress—a scalar reflecting how far the current execution state has advanced toward subtask completion, derived from trajectory-level completion annotations in the training data. The value pathway attends primarily to current and historical visual observations when producing the progress estimate Vt.

System 3 parameters are frozen after this stage. During subsequent VLA post-training, System 3 generates fresh outputs $g _ { t }$ and $V _ { t }$ for each training sample but receives no gradient updates.

## 5.3. World-Model-Powered VLA Post-Training

We combine the pre-trained VLA with the frozen System 3 during task-specific post-training. System 3's outputs are provided directly to System 1 as additional input conditions; only Systems 1 and 2 are optimized on downstream demonstrations. Freezing the World Value Model preserves its predictive representations while allowing the VLA to learn how these signals should influence task-specific action generation.

## Subgoal-image conditioning.

For each post-training sample, System 3 generates future video predictions spanning the same time horizon as the target subtask. The subgoal image $g _ { t }$ is encoded through the visual pathway and appended to System 1's multimodal context, providing the policy with an explicit visual prediction of the near-future state alongside the current observation.

## Value-derived conditioning.

The task-progress estimate from System 3 is converted into a binary advantage condition. For two temporally separated states within the action-chunk interval, we compute the progress difference and discretize it:

$$
A _ { t } = \mathcal { k } [ V _ { t + \delta _ { t } } - V _ { t } > 0 ] \ \in \ \{ 0 , 1 \} ,\tag{9}
$$

where $A _ { t } ~ = ~ 1$ marks an action segment associated with increasing task progress and $A _ { t } = 0$ marks nonimproving or regressing segments. The advantage is embedded as a conditioning token in the VLA prompt, allowing System 1 to distinguish productive from unproductive behaviors in the training data. At inference time, the policy is conditioned on positive progress, biasing action generation toward behaviors associated with advancing task state—analogous to the advantage-conditioned approach in RAMP [28], but with the advantage derived from a dedicated world value model.

## Condition dropout.

We apply stochastic condition dropout to prevent over-dependence on System 3 outputs. The subgoal image and the value-derived condition are dropped independently, with probabilities 0.50 and 0.15 respectively. The VLA is consequently exposed to four conditioning regimes: neither signal, subgoal only, value only, and both.

## Post-training optimization.

Task-specific post-training optimizes the policy through continuous action generation under the augmented context

$$
c _ { t } ^ { \mathrm { p o s t } } = \left( c _ { t } ^ { \mathrm { b a s e } } , g _ { t } , A _ { t } \right) ,\tag{10}
$$

where $c _ { t } ^ { \mathrm { b a s e } }$ contains the current observation, temporal context, language instruction, proprioceptive state, and embodiment information; $g _ { t }$ denotes the predicted subgoal image; and $A _ { t }$ denotes the value-derived progress condition.

We optimize the same flow-matching objective defined in $\mathrm { E q . } 6 ,$ using $c _ { t } ^ { \mathrm { p o s t } }$ in place of the pre-training context c. System 3 remains frozen throughout this stage, while Systems 1 and 2 are optimized on task-specific demonstrations.

## 5.4. Experience-Driven Reinforcement Learning

Demonstration learning is bounded by the states present in the offline dataset. Deploying the policy on real robots reveals failure states, recovery opportunities, and corrective behaviors that arise from the policy's own execution distribution. Starting online RL directly at this point is sample-inefficient—the policy may repeatedly visit the same failure states—so we adopt a staged offline-to-online pipeline.

## Offline experience reinforcement.

We deploy the world-model-powered post-trained VLA to collect rollout trajectories and organize them by execution outcome. The resulting dataset contains both successful and failed behavior, providing direct evidence

about which actions lead to desirable and undesirable outcomes.

We apply an Advantage-Weighted Regression (AWR) [59] style objective for offline policy refinement. In this part, reward $r _ { t }$ is calculated from the progress discriminator and predefined rules, advantage ${ \tilde { A } } _ { t }$ is defined as

$$
\tilde { A } _ { t } = G _ { t } - \tilde { V } _ { t }\tag{11}
$$

Where $G _ { t }$ and $\tilde { V } _ { t }$ denote the cumulative return and value estimate, respectively.

Rollout segments are weighted according to their estimated advantage: successful and higher-progress segments receive greater training weight, while failed or lower-progress behaviors are down-weighted. This allows the policy to learn preferentially from better experience without requiring additional environment interaction during optimization.

The offline stage converts already-collected experience into improved behavior and provides a stronger initialization for online learning, substantially reducing the costly exploration required in the next stage.

## Online reinforcement with human correction.

After offline refinement, we redeploy the policy and collect on-policy trajectories. When the robot reaches a difficult state, a human operator intervenes and provides corrective actions. The corrective demonstration is recorded as a labeled action segment at the precise failure state and enters the training stream, providing direct supervision on the hardest states of the current policy distribution.

We use an actor-critic framework for online optimization, retaining System 1 as the deployment actor and training a separate critic for value estimation. Rewards combine chunk-level progress differences with terminal sparse signals. Training alternates between rollout collection and model updates:

$$
\pi ^ { ( k ) } \longrightarrow \mathcal { D } _ { \mathrm { r o l l o u t } } ^ { ( k ) } \longrightarrow \mathrm { U p d a t e } \Big ( \pi ^ { ( k ) } , \mathcal { D } _ { \mathrm { r o l l o u t } } ^ { ( k ) } \Big ) \longrightarrow \pi ^ { ( k + 1 ) } .\tag{12}
$$

Each improved policy generates the experience for subsequent iterations, progressively shifting training toward the state distribution encountered during deployment.

## Continual improvement.

The four training stages form a progressive lifecycle: pre-training establishes broad embodied priors; worldmodel pre-training adds future prediction and progress estimation; post-training specializes these signals for target tasks; and experience-driven reinforcement improves the policy on states produced by its own execution. The online stage can be repeated as new rollout data becomes available, forming a continual rollout-correction-update cycle. A detailed treatment of the human-intervention protocol and the full online RL formulation will be provided in a dedicated future GigaBrain report.

## 6. Experiments

We evaluate GigaBrain-0.7 through a series of experiments designed to answer the following questions: (1) Which backbone and VLM-action expert architecture provide a strong basis for large-scale embodied pretraining? (2) How do data scale and data composition affect pretrained robot capabilities?(3) How does temporal context help resolve ambiguous repeated states during manipulation?(4) What are the individual and joint contributions of System 3's future-state and value-derived conditioning signals?(5) Does heterogeneous pretraining produce executable out-of-the-box behavior and generalization beyond task configurations observed during data collection? (6) How much can task-specific post-training and subsequent experience-driven reinforcement learning further improve performance?

## 6.1. Experimental Setup

## Evaluation platforms.

We conduct real-world evaluations on two robot platforms: the AgileX PiPER/PiPER-X, a low-cost dual-arm manipulator widely used for tabletop manipulation benchmarking, and our in-house Maker H01, an embodiednative humanoid platform with a head, waist, and bimanual arms. The two platforms differ substantially in degrees of freedom, control interface, camera configuration, and workspace geometry, providing a rigorous test of cross-embodiment generalization.

## Baselines.

For real-world policy evaluation, we compare GigaBrain-0.7 against the following baselines: (i) $\pi _ { 0 . 5 }$ [62], a state-of-the-art VLA model with heterogeneous co-training and hybrid discrete-continuous supervision; (ii) GigaBrain-0.1, the predecessor model in the GigaBrain series with PaliGemma2 backbone; (iii) Xiaomi-Robotics-1 [79]; (iv) Galaxea G0.5 [23]. These real-world policy baselines are evaluated under the same protocol and task definitions. Separately, the embodied vision-language benchmark evaluation includes Xiaomi-Robotics-0 and G0.5-base, as well as Spirit-v1.5 [70], Wall-OSS-0.5 [86], and Hy-Embodied-0.5-VLA-UMI [89].

## Evaluation protocol.

Each task is evaluated over a fixed number of trials (typically 10–20 per configuration). We report success rate as the primary metric, defined as the fraction of trials in which the robot completes the full task objective. For language-following evaluations, each task is conditioned on a specific natural-language prompt that specifies attributes such as color, target identity, or spatial direction (e.g., "pick up the red spoon"), and a trial is considered successful only if the correct target is manipulated. We distinguish between in-distribution (ID) tasks whose objects and configurations are represented in the training set and out-of-distribution (OOD) tasks that require generalization to unseen objects, layouts, or instructions.

## Post-training.

For post-training evaluations, all models are fine-tuned on task-specific demonstration data collected on the corresponding target platform. The resulting policies are evaluated using the same task definitions and evaluation protocol described above.

## 6.2. Backbone and Architecture Comparison

Before studying data scaling, we first establish the System 1 configuration used in the subsequent experiments. We compare candidate VLM backbones and alternative mechanisms for coupling the vision-language model with the continuous action expert. These experiments are intended to identify a robust model configuration for large-scale pretraining

## VLM backbone comparison.

We first compare PaliGemma2[71], Qwen3.5[65], and Gemma 4[74] while keeping the dual-stream VLA formulation fixed. The models are evaluated on representative real-robot tasks spanning structured manipulation, language-conditioned object interaction, and long-horizon deformable-object manipulation. As shown in Tab. 4, increasing backbone size does not lead to uniformly stronger robot performance in the evaluated setting. Gemma 4 performs strongly on the more structured tasks, whereas PaliGemma2 is the only backbone that achieves nonzero success on shirt folding while remaining competitive on fruit manipulation. We therefore use PaliGemma2 as the backbone for the following architecture comparison and data-scaling experiments.

## VLM-action expert coupling.

With the backbone fixed to PaliGemma2, we next compare three ways of connecting the vision-language representation to the continuous action expert: Dual Stream, Last-Layer Cross Attention, and Multi-Layer Cross Attention. A dual-stream expert interleaves with the VLM at every layer, attending to concatenated VLM-expert keys and values within a single attention computation, without any separate cross-attention module. A last-layer cross-attention expert instead decouples the action head entirely from the backbone, with every expert block cross-attending only to the VLM's final-layer hidden states. A multi-layer cross-attention expert maintains a per-layer interface, with expert block i cross-attending to the hidden states of the corresponding VLM layer i. The three variants thus span a spectrum of coupling strength: from per-layer interleaving, through a per-layer interface, to a single terminal representation, progressively decoupling the expert from the backbone at the cost of a narrower information pathway.

Table 4: Comparison of VLM backbones for System 1. The evaluated configurations differ in model size and image resolution; the comparison is used to determine the backbone for subsequent experiments rather than as a controlled model-scaling study. Model size denotes the total parameter count of the full model, including both the VLM backbone and the action expert. \*Gemma 4 Clean Desk was evaluated under a limited-range position-generalization setting.
<table><tr><td>Backbone</td><td>Model Size</td><td>Clean Desk</td><td>Fruit Picking</td><td>Shirt Folding</td></tr><tr><td>PaliGemma2</td><td>3.5B</td><td>50%</td><td>88%</td><td>30%</td></tr><tr><td>Qwen3.5</td><td>5B</td><td>60%</td><td>12%</td><td>0%</td></tr><tr><td>Gemma 4</td><td>8.5B</td><td>100%* ☆</td><td>92%</td><td>0%</td></tr></table>

Table 5: Comparison of VLM-action expert coupling architectures. All variants use PaliGemma2 as the VLM backbone. Success rates are measured on real robots; training and inference costs are reported for the corresponding configurations.
<table><tr><td>Architecture</td><td>Clean Desk</td><td>Fruit Picking</td><td>Shirt Folding</td><td>Train s/step</td><td>Infer. S</td></tr><tr><td>Dual Stream</td><td>50%</td><td>88%</td><td>30%</td><td>4.93</td><td>0.221</td></tr><tr><td>Last-Layer Cross Attention</td><td>20%</td><td>40%</td><td>0%</td><td>4.65</td><td>0.073</td></tr><tr><td>Multi-Layer Cross Attention</td><td>20%</td><td>36%</td><td>0%</td><td>6.59</td><td>0.108</td></tr></table>

Tab. 5 shows a clear trade-off between action-generation capability and computational efficiency. Restricting the action expert to the final VLM representation yields the lowest inference latency, whereas the multi-layer cross-attention variant incurs additional computational cost without improving task success in the evaluated settings. In contrast, the dual-stream architecture achieves the strongest real-robot performance across all three tasks, including nonzero success on the more challenging shirt-folding task, although at a higher inference cost.

Based on these comparisons, we adopt the PaliGemma2-based dual-stream architecture as the default System 1 configuration in the remainder of the experiments. This fixes the model architecture before we study the effect of increasing the scale and diversity of the pretraining data.

## 6.3. Data Scaling

With the model architecture fixed, we next study how GigaBrain-0.7 benefits from scaling heterogeneous embodied data. We consider three complementary questions: whether increasing the overall amount of pretraining data improves optimization, whether additional robot experience translates into stronger real-robot capability, and whether human demonstrations provide complementary supervision beyond robot trajectories alone.

## Scaling the pretraining data.

We first vary the amount of pretraining data while keeping the model configuration unchanged. As shown in Fig. 8a, larger training sets consistently reach lower validation loss, with the separation between data scales remaining visible throughout optimization. Within the evaluated range, we therefore observe continued improvement as the available pretraining data increases.

Pre-training Data Type and Scale vs. Real-Robot Success Rate  
Scaling Performance with Pretraining Data  
![](images/0e580aaa50b09ae8cb9f6853168f678c59cc00055395e39694864be6740042f8.jpg)  
(a) Validation loss under different pretraining-data scales.

Pre-training Data Scale vs. Real-Robot Success Rate Pre-train: Fold Chaotic Clothes — Pre-train: Pick and Place Fruits  
![](images/1adfcc1ce95e2fd33dab81a358d9d775a86f001e0ff2d36dbf69b27ecb19f5b4.jpg)  
(b) Real-robot performance as robot-data scale increases.

![](images/8da8bb9b103b7c7e4ee045d403aa0478c659f3ea5d3fcb75dd979992cd4be17a.jpg)  
(c) Effect of UMI and EGO data in the pretraining mixture.  
Figure 8: Scaling GigaBrain-0.7 with heterogeneous embodied data. (a) Increasing the amount of pretraining data consistently reduces validation loss under a fixed model configuration. (b) Increasing robot-data scale improves real-robot performance, with a stronger effect on the more challenging deformable-object task. (c) UMI and EGO provide complementary supervision to robot trajectories, and their combination yields the strongest overall performance.

## Scaling robot experience.

We next examine whether the same trend translates to physical execution. Fig. 8b evaluates models trained with progressively larger subsets of the robot data on fruit manipulation and chaotic clothes folding.

Both tasks benefit from additional robot experience, but the scaling behavior differs substantially with task complexity. Fruit manipulation improves relatively smoothly, whereas clothes folding shows a much sharper dependence on data scale. The latter requires sustained interaction with deformable objects and a broader range of intermediate states, suggesting that complex long-horizon skills place substantially greater demands on the coverage of the pretraining distribution. Together with the consistently decreasing validation loss, this trend suggests that scaling does more than improve the training objective: manipulation behaviors that are weak or unreliable at smaller scales become increasingly reliable as embodied experience grows. We view this progression as a direct manifestation of capability emergence with scale.

## Scaling across data sources.

Finally, we investigate whether scaling data diversity provides benefits beyond increasing robot trajectories alone. Starting from 3,000 hours of robot data, we add equal amounts of UMI or EGO demonstrations, and

![](images/6cbfde6afbf2aceb33a6536e13553807cd25a9bdb68fec9aa21c75607202aadc.jpg)

(a) Without temporal context. The policy repeatedly revisits similar interaction states and falls into a repeated-action cycle.  
![](images/1423da4ec11cb592a16ea8aa83789cd4cf58c02d750894cd075a85e725556679.jpg)  
(b) With temporal context. Historical observations help disambiguate the revisited state, allowing the policy to exit the repeated interaction and continue task execution.

Figure 9: Qualitative analysis of temporal context in repeated-state manipulation. Each rollout is represented by ten temporally ordered frames. Without temporal context, the policy repeatedly returns to visually similar interaction states and produces a recurring action pattern. With temporal context, recent interaction history provides additional information about the current stage of execution, enabling the policy to break the repeated-action cycle and continue toward the task goal.

further evaluate their combination.

As shown in Fig. 8c, both human-data sources improve performance over robot-only pretraining, and the combination of robot, UMI, and EGO data provides the strongest results across the evaluated settings. The improvement persists after task-specific post-training, indicating that the additional human experience changes the quality of the learned pretraining prior rather than providing only an immediate improvement to the base policy.

The different sources also exhibit complementary effects. Robot trajectories provide direct embodiment-specific control supervision, UMI contributes manipulation experience from a near-hand observation and motion interface, and EGO broadens the diversity of first-person human interaction. Their combined benefit supports the central motivation of our data pyramid: effective scaling depends not only on the number of trajectories, but also on bringing heterogeneous forms of physical experience into a common training representation.

## 6.4. Temporal Context Analysis

Many manipulation tasks contain visually similar states that recur at different stages of execution. When action generation relies only on the current observation, these states can become ambiguous because similar visual configurations may require different subsequent behaviors depending on the preceding interaction history. We qualitatively examine this effect by comparing otherwise matched policies with and without temporal visual context.

As illustrated in Fig. 9, the single-frame policy can become trapped in a repeated-action cycle when the current observation closely resembles states encountered earlier in the episode. Although these instantaneous visual observations are similar, they occur at different stages of the interaction and do not necessarily require the

Prediction and Evaluation in System 3  
![](images/3ec345ea84577faf1c3159169e295d5e4529b6b02e8dcd25862fad2c4fdcca35.jpg)  
Figure 10: Prediction and evaluation with the World Value Model. The model predicts both future visual observations and future state values. The inferred advantage decreases sharply when the imagined trajectory leads to failure and increases when the trajectory recovers toward successful task execution.

## same subsequent behavior.

Temporal context provides information about the preceding interaction trajectory and helps distinguish such visually similar but temporally distinct states. In the representative example, the policy with temporal context eventually departs from the recurring interaction pattern and proceeds to the next stage of the task, whereas the single-frame policy continues to repeat similar actions. This qualitative comparison illustrates that temporal modeling can support closed-loop manipulation not only by retaining longer-horizon context, but also by resolving local ambiguity when similar visual states recur during execution.

## 6.5. System 3 Prediction and Evaluation Analysis

## 6.5.1. Prediction and Evaluation

System 3 jointly predicts future visual observations and future state values. The predicted future state values can then be used to estimate the advantage of a candidate action trajectory. As shown in Fig. 10, when the imagined future trajectory results in task failure, the estimated advantage exhibits a pronounced decrease. In contrast, once the trajectory recovers and returns to a successful execution mode, the advantage increases substantially. These results suggest that the model captures value-relevant changes in predicted future outcomes and can use them to distinguish unfavorable failure trajectories from recovered task-progressing behaviors.

Compared to $\pi _ { 0 . 7 } [ 6 3 ]$ and $\pi _ { 0 . 6 } ^ { * }$ [37], our assessment of action quality does not rely directly on human annotations or VLM-based reasoning. Instead, it is derived from a more fine-grained and dense action evaluation produced by the world model through future imagination. This approach yields action-value estimates that are more closely aligned with the resulting rollout outcomes and exhibits greater robustness in generalization across scenarios. Compared with GigaBrain-0.5M\*, we extend the future prediction horizon beyond the length of the action chunk. We find that this longer horizon improves the success rate on complex tasks and enhances the robustness of the model.

![](images/5a93cb9cdfaa07cbf4ca986aed308ad18fc68cfa8b6a96f43921fd0312b64e1e.jpg)  
(a) Task success rate.

![](images/7cae3754951c49bc722f8eefe97b27e893bb8948b0ffbf2ef562432ae062478a.jpg)  
(b) Average task score.

![](images/222dae1a6690bd8dc189c987ec556a1202a3ded17611b3f917d39d9a73632f26.jpg)  
(c) Average completion time.  
Figure 11: Ablation of the prediction and evaluation interfaces in System 3. The base VLA is compared with image conditioning (+ SuBImAGE), value conditioning (+VALUE), and their combination. Completion time is reported only when successful task completion is observed.

## 6.5.2. Ablation in System 3

To isolate future image and value effects, we compare four configurations: BAsE, + SuBIMAGE, +VALUE, and +SUBIMAGE+VALUE.

The four variants are evaluated on clothes folding with AgileX PiPER, gift wrapping with AgileX PiPER-X, and cube sorting with Maker H01. We jointly report task success, average task score, and completion time, since success alone can become saturated on easier tasks.

On clothes folding, all four configurations achieve 100% task success. The additional System 3 signals nevertheless improve both task score and execution efficiency. The average score increases from 68.3% for BAsE to 81.7%, 85.0%, and 88.3% for + SUBIMAGE, +VALUE, and + SUBIMAGE +VALUE, respectively. Average completion time decreases from 107 s to 83 s, 79 s, and 75 s.

The effect is more pronounced on gift wrapping. The base model fails to complete the task, while + SuBImAGE, +VALUE, and + SuBIMAGE +VALUE achieve 20%, 60%, and 80% success. The corresponding task scores increase from 61.1% to 71.4%, 93.3%, and 96.7%. Among variants with successful episodes, completion time decreases from 65 s to 60 s and finally 55 s.

Cube sorting exhibits a more mixed pattern. Success increases from 40% for the base model to 50% with image conditioning, 45% with value conditioning, and 55% when both are enabled. The combined model obtains the highest average task score, increasing from 55.0% to 87.5%. Value conditioning alone yields the shortest observed completion time (80 s), whereas the full configuration provides the strongest overall success and progress metrics.

Taken together, these experiments show that System 3 provides effective predictive and progress-aware guidance for task execution. Future-state and value conditioning improve not only whether a task is completed, but also how effectively the policy progresses through imperfect intermediate states, as reflected by higher task scores and shorter completion times. This effect remains visible even when binary success is already saturated, and becomes more pronounced on the more challenging gift-wrapping task. These results indicate that predictive and evaluative context can improve the robustness of behaviors acquired by the underlying policy, rather than merely increasing task success.

## 6.6. Foundation Model Evaluation

We next evaluate GigaBrain-0.7 directly after pretraining, without task-specific adaptation. Rather than measuring specialization on a small set of downstream tasks, this evaluation asks whether large-scale heterogeneous pretraining produces a policy that can already execute diverse instructions and manipulation behaviors out of

GigaBrain-0.7: One Model, Many Tasks (Language Following)  
![](images/442633bf255fb7fc3e0d1c7c022383cd47e977d775385cec28354e427938a5d9.jpg)  
Figure 12: Out-of-the-box multi-task language following of GigaBrain-0.7. We evaluate the same pretrained policy on AgileX PiPER and Maker H01 under both in-distribution and out-of-distribution settings.

## the box.

We evaluate the model on both AgileX PiPER and Maker H01, covering language-conditioned multi-task execution and longer-horizon manipulation. For each platform, we additionally introduce out-of-distribution settings that alter task configurations, objects, or interaction requirements beyond those represented by the corresponding in-distribution evaluation.

## 6.6.1. Out-of-the-Box Language Following

As shown in Fig. 12, the same pretrained GigaBrain-0.7 policy can follow multiple language instructions on both AgileX PiPER and Maker H01 without task-specific adaptation. Across the in-distribution settings, the policy executes diverse instruction-conditioned behaviors involving object selection, scene rearrangement, and multi-step manipulation. These results show that heterogeneous pretraining produces an executable generalist policy before downstream adaptation, rather than merely providing a stronger initialization for task-specific post-training.

We next evaluate the same pretrained policy under controlled out-of-distribution settings that probe several distinct axes of generalization. Rather than changing only visual appearance, these evaluations vary taskrelevant object instances, spatial arrangements, scenes, and compositions of previously acquired manipulation skills. This follows the broader principle that foundation-model quality is better revealed when the evaluation distribution departs from the configurations observed during data collection.

Representative OOD executions are shown in Fig. 13. The two embodiments probe complementary forms of generalization. On AgileX PiPER, the evaluated pepper instance is absent from the corresponding training

![](images/97e5b965dfdb406032182369aa0c09e2bf39b0795c91417f11727f8a1f845a27.jpg)

(a) AgileX PiPER: the target pepper instance is absent from the training data, while the two containers were not collected together in the same task configuration.  
![](images/27f0c10d13a7f03c97a042ca13de9a68db847bcd70efa282bfa9f7eba882a1d1.jpg)  
(b) Maker H01: grapes are absent from the H01 training data, and neither the evaluation scene nor the task composition was observed during data collection.

Figure 13: Object, spatial, and compositional generalization of the pretrained GigaBrain-0.7 policy. The same pretrained GigaBrain-0.7 policy is deployed on AgileX PiPER and Maker H01 without task-specific adaptation. Panel (a) evaluates a pepper instance that is absent from the corresponding training data; moreover, the two target containers were not collected together in the same task configuration and their relative placement is randomized at evaluation time. Panel (b) introduces a broader compound distributional shift on Maker H01: grapes are absent from the H01 training data, and neither the evaluation scene nor the corresponding task composition was observed during H01 data collection.

## GigaBrain-0.7: One Model, Many Tasks (Complex Manipulation)

AgileX PiPER (In-Domain)

![](images/b71c67eabb8d6adb340786b1fae5d54321443498f57ff676510d62fd3f4343c7.jpg)  
Maker H01 (In-Domain)  
AgileX PiPER (Out-of-Domain)

![](images/f0795faccee32541e98ef05c5d658b7b94e0b14f8f0d064be953e28d9f68a5bc.jpg)

![](images/88e0c1c1d937de81d4b0b328bf6b4f4266119cfd06946c1b555aac715270f5fc.jpg)  
Maker H01 (Out-of-Domain)

![](images/4ea93d46c7f21b2013550be3ff40b496b1a73dca052fe213e1b216f7827ebe2e.jpg)  
Figure 14: Out-of-the-box complex manipulation with GigaBrain-0.7. The evaluation covers longer-horizon manipulation under ID and OOD settings on AgileX PiPER and Maker H01.

data. More importantly, the two target containers were not collected together in the same task configuration and their relative placement is randomized during evaluation. Successful execution therefore requires more than recognizing a familiar scene template: the policy must identify the language-specified target, localize the relevant container in the current arrangement, and compose the appropriate pick-and-place behavior for a previously unobserved object-container configuration. Maker H01 introduces a broader compound distributional shift. The H01 training data contain neither the evaluated grapes nor the test scene or the corresponding task composition. Nevertheless, the pretrained policy follows the human instruction, selects the relevant object, and completes the requested manipulation without task-specific adaptation. Together, these examples provide qualitative evidence that heterogeneous pretraining gives rise to target understanding, spatial grounding, and compositional generalization beyond the task configurations explicitly observed during data collection.

The ID-OOD gap also highlights the current limits of foundation transfer. Generalization remains stronger when the underlying interaction structure is preserved and becomes more challenging when multiple factors—such as object identity, scene configuration, and task composition—shift simultaneously. Both AgileX PiPER and Maker H01 exhibit degradation under the OOD settings, indicating that robust generalization across task distributions and heterogeneous embodiments remains an important direction for continued scaling.

## 6.6.2. Out-of-the-Box Complex Manipulation

We further evaluate whether the pretrained policy can perform manipulation tasks that require sustained interaction over multiple control stages. As shown in Fig. 14, GigaBrain-0.7 exhibits non-trivial out-of-the-box performance on both AgileX PiPER and Maker H01, including deformable-object manipulation and longer-

![](images/23735557151548bca4aecd8d8fb4fcf68f25ecd89b24f0049165eee35bb7f463.jpg)

(a) AgileX PiPER: folding unseen garment instances from diverse unstructured initial states.  
![](images/7871ede2d8b42d2c75b39bc0b281f26151646f3ae497e6acf726fdf971fb0d3e.jpg)  
(b) Maker H01: folding an unseen garment instance from an unstructured initial state.

Figure 15: Out-of-the-box garment-instance and deformable-state generalization. The pretrained GigaBrain-0.7 policy is deployed on garment-folding tasks on AgileX PiPER and Maker H01 without task-specific adaptation. The evaluated garment instances are absent from the corresponding training data and are initialized in diverse, highly unstructured configurations. Panel (a) shows two representative AgileX PiPER rollouts with different unseen garments and initial states, while Panel (b) shows a Maker H01 rollout under the same evaluation principle. Each row contains five temporally ordered stages from a single rollout, illustrating how the policy maintains sustained interaction as the deformable garment geometry evolves throughout execution.

horizon task execution, without task-specific adaptation.

Representative out-of-the-box executions are visualized in Fig. 15. These evaluations probe generalization along two coupled dimensions: garment identity and deformable-object state. The evaluated garments are absent from the corresponding training data and are presented in diverse, highly unstructured initial configurations. This setting differs from conventional appearance-only generalization. Because cloth geometry changes continuously during manipulation, the policy cannot rely on a fixed object shape or replay a memorized trajectory. Successful execution requires repeatedly establishing appropriate contact, responding to the evolving garment state, and maintaining progress across successive action chunks. Despite these variations, the pretrained policy can reorganize the garment and sustain the folding behavior across both AgileX PiPER and Maker H01 without task-specific adaptation. These results suggest that heterogeneous pretraining transfers reusable interaction structure across unseen garment instances and substantial variation in deformable-object configuration, rather than only memorizing specific object appearances or initial states.

Out-of-distribution complex manipulation remains substantially more challenging, particularly when changes in the task configuration require interaction patterns that differ from those more strongly represented in pretraining. Nevertheless, the policy retains executable behavior on multiple OOD settings, suggesting that interaction patterns acquired from heterogeneous experience can be reused and recombined as task conditions change.

Taken together with the language-following results, these experiments show that large-scale pretraining provides more than task-specific motor patterns: it yields a policy that can execute diverse instructions, sustain longer-horizon interaction, and retain useful behavior beyond familiar task configurations. We view this progression as one manifestation of the emergent embodied capabilities enabled by scaling heterogeneous pretraining.

## 6.7. Post-Training Evaluation

The pretrained policy provides a generalist starting point, while practical deployment often requires specialization to a particular embodiment or task distribution. We therefore evaluate GigaBrain-0.7 after task-specific post-training on two complementary settings: multi-task language following, which emphasizes instruction grounding and target discrimination, and complex manipulation, which requires sustained physical interaction and multi-stage execution.

## 6.7.1. Multi-Task Language Following

We first evaluate six language-conditioned manipulation tasks covering color discrimination, target selection, and directional reasoning. Fig. 16 provides an overview of performance on AgileX PiPER and Maker H01, while Tab. 6 reports the complete quantitative results.

Across the PiPER evaluation, GigaBrain-0.7 consistently improves over the preceding GigaBrain model and matches or exceeds πo.5 across the reported tasks. The gains are particularly visible when language must be grounded into target-specific or directional physical behavior, indicating that the pretrained vision-languageaction representation remains effective after specialization.

The comparison on Maker H01 is more challenging but exhibits the same overall trend. GigaBrain-0.7 improves the average success rate from 69.6% for GigaBrain-0.1 to 84.2%, with particularly large gains on button push and spoon grasping while maintaining strong performance on target- and direction-conditioned tasks. Together, the two embodiments indicate that the post-training gains are not restricted to a single robot configuration, although performance remains sensitive to the underlying embodiment and task. Beyond aggregate success rates, we further visualize representative post-training rollouts to examine whether the learned policies remain responsive to fine-grained language conditions. Figures 17-21 cover three complementary forms of instruction grounding—color attributes, object identity, and relative spatial relations—across AgileX PiPER and Maker H01.

SFT Model Language Following (AgileX PiPER)  
![](images/1bb9d68c89527f3b5fbd04e0a99cefaa9c612ed9a17b55b25ff6ae5cf6ebbacc.jpg)

SFT Model Language Following (Maker H01)  
![](images/acd895ac9a355f8bc65bdd8a6d2ee324b2f192c75f380a67418177772d6a5192.jpg)  
Figure 16: Post-training multi-task language following. We compare GigaBrain-0.7 with representative VLA baselines on six manipulation tasks that probe color, target, and directional grounding across two robot embodiments.

![](images/7f49a314c26d49f59b893eab4a4687f305e25decf918d603c79e03e8107ff745.jpg)  
Figure 17: Post-training language following on AgileX PiPER: color grounding. Given the instruction "Grab the orange spoon and put it on the plate.", GigaBrain-0.7 identifies the language-specified object by color and executes the corresponding pick-and-place behavior. The ten frames show temporally ordered stages of a representative rollout.

![](images/24bb2865c170e8bdd1fba21551d9ecd99596bb8b6a8dcb258d489a906407fc04.jpg)  
Figure 18: Post-training language following on AgileX PiPER: spatial grounding. Given the instruction "Grab the block on the left side of the plate and put it in the bowl.", the policy resolves the relative spatial reference, selects the corresponding block, and executes the instructed transfer. The ten frames show temporally ordered stages of a representative rollout.

![](images/65a38d82453848b249d9ac866d43b2de1f3db871554941d47a8069d6be325705.jpg)  
Figure 19: Post-training language following on Maker H01: color grounding. In a scene containing multiple colored alternatives, GigaBrain-0.7 identifies the language-specified blue target and executes the corresponding manipulation behavior. The rollout illustrates sensitivity to fine-grained visual attributes on the humanoid embodiment.

![](images/1e7cae20a3f8fd386ade6993dbc5bb3072e68217d2f35b43ae37275b12c3d4cb.jpg)  
Figure 20: Post-training language following on Maker H01: target grounding. Given the instruction "Pick up the fork and move it to the basket.", the policy identifies the requested utensil and maintains the languageconditioned target throughout the subsequent manipulation sequence.

Table 6: Post-training multi-task language-following evaluation. We evaluate six instruction-conditioned manipulation tasks on AgileX PiPER and Maker H01, covering three complementary language-following capabilities: color discrimination (button push and spoon grasping), target discrimination (bowl and tableware pick-and-place), and direction discrimination (fork and wood-block pick-and-place). Each cell reports success rate (%). Avg. denotes the mean across all six tasks and is reported only for models with complete evaluations. Best available results in each column are shown in bold.
<table><tr><td rowspan="2">Model</td><td colspan="6">AgileX PiPER</td><td rowspan="2">Avg.</td></tr><tr><td>Color</td><td></td><td>Target</td><td></td><td>Direction</td><td></td></tr><tr><td></td><td>Button Push</td><td>Spoon Grasping</td><td>Bowl Pick-Place</td><td>Tableware Pick-Place</td><td>Fork Pick-Place</td><td>Wood Block Pick-Place</td><td></td></tr><tr><td>π0.5</td><td>100.0</td><td>100.0</td><td>90.0</td><td>87.5</td><td>75.0</td><td>80.0</td><td>88.8</td></tr><tr><td>GigaBrain-0.1</td><td>66.7</td><td>77.8</td><td>85.0</td><td>72.2</td><td>80.0</td><td>75.0</td><td>76.1</td></tr><tr><td>G0.5</td><td>88.9</td><td>94.4</td><td>50.0</td><td>100.0</td><td>85.0</td><td>70.0</td><td>81.4</td></tr><tr><td>Xiaomi-Robotics-1</td><td>66.7</td><td>83.3</td><td>50.0</td><td>98.6</td><td>75.0</td><td>60.0</td><td>72.3</td></tr><tr><td>GigaBrain-0.7</td><td>100.0</td><td>100.0</td><td>95.0</td><td>88.9</td><td>85.0</td><td>80.0</td><td>91.5</td></tr></table>

<table><tr><td rowspan="2">Model</td><td colspan="7">Maker H01</td></tr><tr><td colspan="2">Color</td><td colspan="2">Target</td><td colspan="2">Direction</td><td>Avg.</td></tr><tr><td></td><td>Button Push</td><td>Spoon Grasping</td><td>Bowl Pick-Place</td><td>Tableware Pick-Place</td><td>Fork Pick-Place</td><td>Wood Block Pick-Place</td><td></td></tr><tr><td>π0.5</td><td>66.7</td><td>34.4</td><td>100.0</td><td>66.7</td><td>83.3</td><td>100.0</td><td>75.2</td></tr><tr><td>GigaBrain-0.1</td><td>44.4</td><td>34.4</td><td>100.0</td><td>55.6</td><td>83.3</td><td>100.0</td><td>69.6</td></tr><tr><td>G0.5</td><td>72.2</td><td>12.5</td><td>25.0</td><td>31.9</td><td>12.5</td><td>0.0</td><td>25.7</td></tr><tr><td>Xiaomi-Robotics-1</td><td>100.0</td><td>31.3</td><td>100.0</td><td>50.0</td><td>70.8</td><td>0.0</td><td>58.7</td></tr><tr><td>GigaBrain-0.7</td><td>100.0</td><td>56.3</td><td>100.0</td><td>61.1</td><td>87.5</td><td>100.0</td><td>84.2</td></tr></table>

![](images/9ebb2fbd5ddf196d25c6068c72597a1363d4a033549fcf37151e4daa7e401076.jpg)  
Figure 21: Post-training language following on Maker H01: spatial grounding. Given the instruction "Grab the fork on the left side of the plate and put it in the basket.", the policy grounds the relative spatial relation before executing the instructed pick-and-place sequence.

Across these examples, the post-trained policy conditions its behavior on distinctions expressed in the instruction rather than reproducing a fixed manipulation trajectory. Color attributes, object identity, and relative spatial relations all affect target selection and subsequent execution, and the same qualitative behavior is observed across both AgileX PiPER and Maker H01. These rollouts demonstrate that post-training preserves fine-grained instruction grounding while adapting the policy to heterogeneous robot embodiments.

## 6.7.2. Complex Manipulation

We next evaluate tasks requiring sustained contact, multi-stage execution, deformable-object manipulation, or tool-mediated interaction. Fig. 22 visualizes the comparison across platforms, and Tab. 7 provides the exact results.

Across both embodiments, GigaBrain-0.7 provides consistent improvements over the preceding GigaBrain model and remains competitive with or stronger than π0.5 across the completed complex-manipulation evaluations.

On PiPER, differences are relatively modest on several structured tasks where the strongest baselines already perform well, while clearer gains appear on interactions requiring more sustained contact or multi-stage execution. On Maker H01, the separation is substantially larger and extends across a broader range of skills, including deformable-object manipulation, food-related interaction, and multi-stage cleanup. We further visualize representative post-training executions on complex manipulation tasks in Figures 23-26. The selected tasks span deformable-object interaction, tool-mediated manipulation, household-item organization, and sequential object sorting, placing qualitatively different demands on contact, temporal coordination, and task progress.

SFT Model Complex Manipulation (AgileX PiPER)  
![](images/58047a8634cce395e3b4f483740527000a02d23307ceefd62e5df9f612787324.jpg)  
Complex Manipulation Tasks

SFT Model Complex Manipulation (Maker H01)  
![](images/35fc0d02fc4279e8167ce4a9cfa9cf3d955f5cac45d65f47e87098831015be0e.jpg)  
Complex Man ipulation Tasks

Figure 22: Post-training complex manipulation. The evaluation covers manipulation tasks requiring sustained interaction, multi-stage sequencing, deformable-object handling, and tool use on AgileX PiPER and Maker H01.  
![](images/caec26fd19b66fe7c185057747e8bce88d380231b1f0761b09b0960dbae2e8e5.jpg)  
Figure 23: Post-training complex manipulation on AgileX PiPER: play-dough manipulation. The policy performs repeated contact-rich interactions with deformable material, requiring subsequent actions to adapt as the object geometry changes throughout execution. The ten frames show temporally ordered stages of a representative rollout.

![](images/ee6e6ed6fe5e33317ea8e836255be433174bc5f81e6d00acdb31d781501cc19d.jpg)  
Figure 24: Post-training complex manipulation on AgileX PiPER: tabletop sweeping. GigaBrain-0.7 uses a brush to manipulate distributed material across the tabletop, requiring sustained tool-object contact and coordinated motion over multiple stages of execution.

![](images/eb8a378673cc7b9d48e94f373e414b3acb36c7399604b86883104230e2b654cf.jpg)  
Figure 25: Post-training complex manipulation on Maker H01: household-item organization. The humanoid policy performs a sequence of object selection, grasping, transport, and placement behaviors to organize multiple items across the workspace. The rollout illustrates sustained multi-stage execution on a substantially different robot embodiment.

![](images/a416f740e7d6902ebcac534f0a98408f237bcb10e6c61a73da0dcc3d6d2ec040.jpg)  
Figure 26: Post-training complex manipulation on Maker H01: block sorting. GigaBrain-0.7 performs repeated object selection and placement across successive manipulation stages, requiring the policy to maintain task progress as the scene changes after each interaction.

Table 7: Post-training single-task complex manipulation evaluation. We evaluate post-trained policies on a diverse set of complex manipulation tasks across AgileX PiPER and Maker H01. The benchmark covers object sorting, deformable-object manipulation, tableware arrangement, household-item storage, sweeping, food preparation and heating, and dish cleanup. Each cell reports task success rate (%). Avg. denotes the mean over all evaluated tasks on the corresponding embodiment and is reported only for models with complete evaluations. Best available results in each column are shown in bold.
<table><tr><td rowspan="2">Model</td><td colspan="6">AgileX PiPER</td></tr><tr><td>Cube Sorting</td><td>Play-Dough Kneading</td><td>Tableware Arranging</td><td>Household Item Storage</td><td>Rice Sweeping</td><td>Avg.</td></tr><tr><td>π0.5</td><td>95.0</td><td>95.0</td><td>77.8</td><td>95.0</td><td>20.0</td><td>76.6</td></tr><tr><td>GigaBrain-0.1</td><td>90.0</td><td>95.0</td><td>38.9</td><td>75.0</td><td>25.0</td><td>64.8</td></tr><tr><td>G0.5</td><td>45.0</td><td>20.0</td><td>38.9</td><td>85.0</td><td>0.0</td><td>37.8</td></tr><tr><td>Xiaomi-Robotics-1</td><td>35.0</td><td>85.0</td><td>44.4</td><td>75.0</td><td>5.0</td><td>48.9</td></tr><tr><td>GigaBrain-0.7</td><td>95.0</td><td>100.0</td><td>94.4</td><td>95.0</td><td>40.0</td><td>84.9</td></tr></table>

<table><tr><td rowspan="2">Model</td><td colspan="8">Maker H01</td></tr><tr><td>Cube Sorting</td><td>Play-Dough Kneading</td><td>Tableware Arranging</td><td>Household Item Storage</td><td>Food Prep. &amp; Heating</td><td>Dish Cleanup &amp; Washing</td><td>Rice Sweeping</td><td>Avg.</td></tr><tr><td>π0.5</td><td>8.3</td><td>50.0</td><td>51.1</td><td>91.7</td><td>50.0</td><td>40.0</td><td>25.0</td><td>45.2</td></tr><tr><td>GigaBrain-0.1</td><td>20.8</td><td>41.7</td><td>45.6</td><td>79.2</td><td>40.0</td><td>50.0</td><td>25.0</td><td>43.2</td></tr><tr><td>G0.5</td><td>0.0</td><td>0.0</td><td>27.8</td><td>0.0</td><td>20.0</td><td>10.0</td><td>0.0</td><td>8.3</td></tr><tr><td>Xiaomi-Robotics-1</td><td>8.3</td><td>20.8</td><td>34.4</td><td>54.2</td><td>30.0</td><td>40.0</td><td>41.7</td><td>32.8</td></tr><tr><td>GigaBrain-0.7</td><td>41.7</td><td>79.2</td><td>75.6</td><td>91.7</td><td>85.0</td><td>100.0</td><td>45.8</td><td>74.1</td></tr></table>

The consistency of these gains across task categories suggests that the benefit of GigaBrain-0.7 is not confined to a single manipulation primitive. At the same time, several contact-rich tasks remain far from saturation, indicating that complex physical interaction remains an important direction for additional data scaling and experience-based post-training.

## 6.8. Benchmark Evaluation

Beyond real-robot evaluation, we further assess GigaBrain-0.7 on standardized benchmarks that probe complementary capabilities of an embodied foundation model. We evaluate embodied vision-language understanding through MiMo-Embodied [34], and policy execution through three complementary simulation benchmarks: RoboTwin 2.0 [15], EBench [24], and RoboColiseum [1].

## 6.8.1. Embodied Vision-Language Evaluation

Beyond action generation, an embodied foundation model should retain the visual-semantic and spatial reasoning capabilities required to interpret physical scenes and support downstream interaction. We therefore evaluate the vision-language component of GigaBrain-0.7 on MiMo-Embodied [34], a suite of embodied vision-language benchmarks covering spatial understanding and affordance reasoning. We exclude VSI-Bench because it requires video input, whereas the VLM interface evaluated here operates on image observations.

Following the benchmark taxonomy, we organize the evaluated tasks into two capability groups. Spatial Understanding contains eight benchmarks covering spatial reasoning, visual grounding, referring expressions, and scene-level understanding. Affordance contains five benchmarks covering interaction-oriented referring, placement reasoning, point-level affordance prediction, part-level affordance understanding, and robotic affordance reasoning. For a fair comparison with existing models, Tab. 8 reports GigaBrain-0.7 before MiMo data are introduced into the subsequent pretraining mixture.

Table 8: Summary of embodied vision-language evaluation on MiMo-Embodied [34]. Spatial Avg. is averaged over eight Spatial Understanding benchmarks, and Affordance Avg. over five Affordance benchmarks. Overall denotes the mean over all 13 evaluated benchmarks. For GigaBrain-0.7, the reported comparison is obtained before MiMo data are incorporated into subsequent continued pretraining. Bold indicates the best result among the compared models.
<table><tr><td>Model</td><td>Spatial Avg.</td><td>Affordance Avg.</td><td>Overall</td></tr><tr><td>Xiaomi-Robotics-0 [9]</td><td>.1372</td><td>.0687</td><td>.1108</td></tr><tr><td>Spirit-v1.5 [70]</td><td>.3524</td><td>.2995</td><td>.3320</td></tr><tr><td>Wall-OSS-0.5 [86]</td><td>.2273</td><td>.0367</td><td>.1540</td></tr><tr><td>G0.5-base [23]</td><td>.3890</td><td>.3956</td><td>.3916</td></tr><tr><td>Hy-Embodied-0.5-VLA-UMI [89]</td><td>.2532</td><td>.0000</td><td>.1558</td></tr><tr><td>GigaBrain-0.7</td><td>.5215</td><td>.3669</td><td>.4621</td></tr></table>

As shown in Tab. 8, GigaBrain-0.7 already exhibits strong embodied vision-language capability before MiMo data are introduced into the training mixture, achieving the strongest overall and Spatial Understanding averages among the compared models.

We subsequently incorporate MiMo data into the same one-stage VLA pretraining mixture and continue joint optimization of vision-language and action objectives, without introducing a separate VLM adaptation stage. The resulting final checkpoint further reaches an overall MiMo score of 0.5704, indicating that additional embodied vision-language supervision can be absorbed directly within the unified VLA pretraining framework.

Since the later training mixture is no longer disjoint from the MiMo evaluation set, we treat this final score as a continued-pretraining diagnostic rather than a held-out benchmark result. The released GigaBrain-0.7 checkpoint corresponds to this final MiMo-enhanced pretraining stage.

## 6.8.2. Simulation Benchmark Evaluation

We further evaluate GigaBrain-0.7 on three simulation benchmarks with complementary task distributions and evaluation settings: RoboTwin 2.0 [15], EBench [24], and RoboColiseum [1]. Together, these benchmarks cover tabletop bimanual manipulation, mobile and long-horizon interaction, and reconstruction-based Real2Sim2Real evaluation.

For readability and reproducibility, the main text focuses on representative publicly available methods, while complete leaderboard snapshots, including additional submissions are provided in Appendix A together with their source URLs and snapshot dates.

## Physics-based simulation benchmarks.

We first evaluate on RoboTwin 2.0 [15], a large-scale benchmark for bimanual manipulation under both clean and randomized simulation conditions. Following the official Co-Train protocol, a single GigaBrain-0.7 policy is post-trained jointly on all 50 benchmark tasks using 50 clean demonstrations per task, and is subsequently evaluated under both the clean Easy setting and the domain-randomized Hard setting. This differs from the Single protocol, where a separate checkpoint is fine-tuned for each individual task.

As shown in Tab. 9, GigaBrain-0.7 achieves the strongest overall performance among the evaluated methods and ranks first under the challenging Hard setting. While several models obtain higher success in the clean environment, GigaBrain-0.7 retains substantially stronger performance after domain randomization. This result highlights the robustness of the learned policy to visual and environmental variations while operating as a single multi-task policy.

We next evaluate on EBench [24], which extends simulation evaluation to mobile manipulation, long-horizon execution, and dexterous and precise interaction. Among the publicly available VLA models compared in Tab. 10, GigaBrain-0.7 achieves the strongest performance in both overall success rate and aggregate task score, outperforming representative generalist VLA baselines including $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ . This result complements RoboTwin 2.0 by showing that the model's advantage extends beyond fixed-base bimanual manipulation to tasks involving mobility, extended execution horizons, and precise interaction. The complete official EBench leaderboard, including additional submissions not included in the main-text comparison, is provided in Appendix A.2.

Table 9: Evaluation on RoboTwin 2.0 [15]. Under the official Co-Train protocol, one policy is jointly posttrained on all 50 tasks using 50 clean demonstrations per task. Single denotes task-specific fine-tuning with a separate checkpoint for each task. Success rates (%) are reported under the Easy (clean) and Hard (domainrandomized) settings. Overall denotes the mean of Easy and Hard. Best results in each column are shown in bold.
<table><tr><td>Model</td><td>Method</td><td>Easy</td><td>Hard</td><td>Overall</td></tr><tr><td>π0 [5]</td><td>Single</td><td>46.42</td><td>16.34</td><td>31.38</td></tr><tr><td>Xiaomi-Robotics-0 [9]</td><td>Co-Train</td><td>62.9</td><td>18.2</td><td>40.55</td></tr><tr><td>X-VLA [90]</td><td>Co-Train</td><td>68.0</td><td>20.9</td><td>44.45</td></tr><tr><td>X-WAM [32]</td><td>Co-Train</td><td>70.0</td><td>25.8</td><td>47.90</td></tr><tr><td> $\pi _ { 0 . 5 } \ [ 6 2 ]$ </td><td>Co-Train</td><td>70.7</td><td>46.0</td><td>58.35</td></tr><tr><td>GigaBrain-0.7</td><td>Co-Train</td><td>66.8</td><td>67.9</td><td>67.35</td></tr></table>

Table 10: Evaluation on EBench [24] among publicly available VLA models. SR denotes the overall task success rate, and Score measures task progress. Results for $\pi _ { 0 } , \pi _ { 0 . 5 } ,$ , X-VLA, and GigaBrain-0.7 follow the corresponding leaderboard evaluations. \*The InternVLA-A1 result is reproduced from the EBench evaluation reported in Qwen-RobotManip [87], rather than from the current EBench leaderboard. Best results among the compared models are shown in bold. The complete official leaderboard, including additional submissions not shown here, is provided in Appendix A.2.
<table><tr><td>Model</td><td>SR (%)</td><td>Score</td></tr><tr><td>π0 [5]</td><td>23.59</td><td>37</td></tr><tr><td>X-VLA [90]</td><td>23.72</td><td>35</td></tr><tr><td>InternVLA-A1* [8]</td><td>23.90</td><td>36</td></tr><tr><td> $\pi _ { 0 . 5 }$  [62]</td><td>28.08</td><td>42</td></tr><tr><td>GigaBrain-0.7</td><td>33.30</td><td>46.1</td></tr></table>

## Reconstruction-based Real2Sim2Real evaluation.

Physics-based simulation provides scalable and reproducible evaluation, but its synthetic visual distribution inevitably introduces a simulation-to-reality gap. We therefore complement RoboTwin 2.0 and EBench with RoboColiseum [1], which adopts a reconstruction-based Real2Sim2Real evaluation pipeline to more closely align simulated observations with real-world environments.

RoboColiseum evaluates policies along four complementary dimensions: instruction following, spatial reasoning, robustness, and general manipulation. As shown in Tab. 11, GigaBrain-0.7 achieves the strongest performance across all four dimensions among the evaluated open-source models. The advantage is particularly evident in spatial reasoning, while strong performance is retained across instruction-conditioned execution, robustness, and general manipulation. Together with the real-robot evaluations in the preceding sections, these results provide complementary evidence that the capabilities acquired by GigaBrain-0.7 remain effective across both simulated and real-world manipulation settings.

Overall, GigaBrain-0.7 exhibits consistently strong performance across three complementary simulation regimes. It achieves the best overall and Hard-setting performance on RoboTwin 2.0 under the multi-task Co-Train protocol, obtains the strongest EBench result among the publicly available models compared in the main text, and leads all four evaluated capability dimensions on RoboColiseum. These results demonstrate that the benefits of large-scale heterogeneous pretraining extend across different robot embodiments, task structures, and evaluation domains. To facilitate reproducible evaluation, our implementation will also be integrated into XPolicyLab [17], providing a unified workflow for training, inference, and deployment across robotic learning benchmarks such as RoboTwin 2.0 and RoboDojo[14].

Table 11: Evaluation on RoboColiseum [1]. The benchmark evaluates four complementary dimensions of robot capability. We report representative open-source models with complete evaluations. Best results in each column are shown in bold.
<table><tr><td>Model</td><td>Instruction Following</td><td>Spatial Reasoning</td><td>Robustness</td><td>General Manipulation</td></tr><tr><td>π0 [5]</td><td>.3680</td><td>.1300</td><td>.3130</td><td>.3470</td></tr><tr><td>Xiaomi-Robotics-0 [9]</td><td>.6460</td><td>.2230</td><td>.5460</td><td>.3070</td></tr><tr><td>GR00T N1.7 [51]</td><td>.6460</td><td>.2490</td><td>.5380</td><td>.4380</td></tr><tr><td>π0.5 [62]</td><td>.7460</td><td>.3560</td><td>.6130</td><td>.5820</td></tr><tr><td>ACoT-VLA [92]</td><td>.7570</td><td>.3970</td><td>.6220</td><td>.4770</td></tr><tr><td>GigaBrain-0.7</td><td>.8166</td><td>.4729</td><td>.6800</td><td>.6092</td></tr></table>

## 6.9. Experience-Driven Reinforcement Learning

We finally examine whether GigaBrain-0.7 can continue improving through experience generated during its own real-robot execution, complementing the preceding evaluations of large-scale pretraining and task-specific post-training. Starting from the task-specific SFT policy, we apply the staged experience-reinforcement pipeline described in Section 5.4: rollout experience first supports offline policy refinement, after which the refined policy is redeployed for online reinforcement with human correction. This evaluation measures the incremental gains from demonstration-based specialization to offline experience learning and then to learning from the state distribution induced by the updated policy.

Four representative real-robot tasks expose complementary challenges. Two emphasize long-horizon, multistage execution. In Gift Box Packing on AgileX PiPER-X, the policy must open the box, place the target object inside, and then manipulate and align the lid; an early error changes the configuration encountered in later stages. Bearing Installation on Maker H01 involves an even longer sequence: the humanoid retrieves a workpiece from the parts bin, transfers it to the assembly station, picks up a bearing, aligns and installs it at the designated location, and returns the assembled workpiece. Two additional tasks emphasize precision and sustained contact at critical stages. In Link Installation on AgileX PiPER, the policy must adjust the link pose, align one end with a constrained mounting interface, and perform the insertion. In Cable Tie Insertion and Tying on AgileX PiPER-X, the robot must control a slender cable tie while routing it around the target component, threading its free end through the locking head, and pulling it through to tighten the connection. Fig. 27 shows representative executions. Together, these tasks evaluate experience-driven improvement in sequential progress and fine-grained physical interaction across PiPER-family manipulators and the Maker H01 humanoid.

Three successive training stages are compared under identical task definitions and success criteria. The SFT Baseline is the task-specific supervised policy before experience reinforcement. For Offline RL, the deployed policy collects real-robot rollouts containing successful, failed, and corrective experience. As described in Section 5.4, rollout segments are evaluated using progress-derived returns and value estimates, and the policy is refined with an advantage-weighted objective that assigns greater weight to higher-progress experience. Failures and partial progress can therefore provide training signal without new interaction during optimization. The refined policy then initializes Online RL: it is redeployed to collect on-policy trajectories, with human corrections introduced at difficult states. These corrections enter the training stream at the states that elicited them, while the actor-critic update combines progress-based feedback with terminal task outcomes. All stages refine the same System 1 deployment policy rather than replacing it with a separate task-specific controller.

![](images/40e5b9996f1329e35445d1365ee882961c5bc29d2c3900643d4af15417065ea2.jpg)  
Figure 27: Representative real-robot tasks for experience-driven reinforcement learning. (A) Gift Box Packing (PiPER-X). (B) Bearing Installation (H01). (C) Link Installation (PiPER). (D) Cable Tie Insertion and Tying (PiPER-X).

Table 12: Task success rates at different training stages.
<table><tr><td>Task</td><td>SFT Baseline</td><td>Offline RL</td><td>Online RL</td></tr><tr><td>Link Installation (PiPER)</td><td>20%</td><td>40%</td><td>100%</td></tr><tr><td>Gift Box Packing (PiPER-X)</td><td>80%</td><td>90%</td><td>100%</td></tr><tr><td>Cable Tie Insertion and Tying (PiPER-X)</td><td>0%</td><td>40%</td><td>100%</td></tr><tr><td>Bearing Installation (H01)</td><td>20%</td><td>60%</td><td>100%</td></tr></table>

Results in Tab. 12 show consistent stage-wise improvement on all four tasks. Average success rises from 30.0% for the SFT baseline to 57.5% after offline RL and 100% after online RL. Offline gains depend strongly on the capability of the starting policy. Gift Box Packing, already at 80% after SFT, improves to 90%. Bearing Installation and Cable Tie Insertion and Tying each gain 40 percentage points, rising from 20% to 60% and from 0% to 40%, respectively, while Link Installation improves from 20% to 40%. Offline reinforcement thus yields its largest gains where the SFT policy encounters substantial real-robot failures, while still benefiting a task with a strong supervised initialization.

The transition to online reinforcement produces a second substantial improvement. Link Installation and Cable Tie Insertion and Tying rise from 40% to 100%, Bearing Installation from 60% to 100%, and Gift Box Packing from 90% to 100%. The largest gains occur on the two tasks that require resolving precision-critical states during alignment, insertion, or threading. Redeployment is especially valuable in such cases: after offline refinement changes the policy, online rollouts expose the remaining difficult states under the updated policy distribution, enabling human corrections to provide targeted action supervision precisely where needed.

Taken together, the three stages demonstrate complementary roles for supervised demonstrations and deployment experience. SFT provides the task-specific initialization required for physical execution; offline reinforcement learns from differences in the quality of resulting rollouts; and online reinforcement continues this process using experience and corrections collected from the improved policy itself. This pattern holds for both the long-horizon Gift Box Packing and Bearing Installation tasks and the precision-critical Link Installation and Cable Tie Insertion and Tying tasks. Success increases at every stage across all four real-robot evaluations, with the final online-RL policies achieving 100% success on every task in the reported evaluations.

## 7. Conclusion and Future Work

In this work, we present GigaBrain-0.7, an embodied foundation model that scales heterogeneous embodied experience and coordinates understanding, prediction, and action through a three-system architecture. GigaBrain-0.7 is pretrained on over 37,000 hours of embodied trajectory data spanning 16 robot morphologies, together with large-scale vision-language supervision. Its one-stage VLA pretraining jointly optimizes multimodal understanding, hierarchical task prediction, discrete action supervision, and continuous action generation across heterogeneous embodiments. System 3 is separately pretrained for future-state prediction and task-progress estimation, and its outputs provide additional conditioning during task-specific post-training and positive-progress guidance at inference. Across real-robot, embodied vision-language, and simulation evaluations, GigaBrain-0.7 demonstrates strong out-of-the-box generalization, post-training performance, and robustness across diverse tasks and embodiments. Experience-driven offline and online reinforcement learning further improves the policy using rollout experience and corrective feedback.

Looking ahead, we will continue to scale heterogeneous embodied data and improve cross-source alignment, strengthen long-horizon predictive modeling and task-progress estimation, and develop more scalable closedloop learning from autonomous experience and human correction. We hope these efforts will move embodied foundation models toward more general, robust, and continually improving real-world robot intelligence.

## A. Simulation Benchmark Leaderboard Snapshots

For completeness and reproducibility, we provide snapshots of the benchmark leaderboards associated with the simulation comparisons in Section 6.8.2. The main text focuses on representative publicly available methods for concise and reproducible comparison, whereas this appendix preserves the complete leaderboard status at the time of manuscript preparation, including additional submissions not shown in the main-text tables.

Online leaderboards may change as new submissions are added. We therefore provide the source URL and snapshot access date for each benchmark below.

## A.1. RoboTwin 2.0

Leaderboard URL: https://robotwin-platform.github.io/leaderboard

Snapshot accessed: August 15, 2026.

<table><tr><td rowspan=1 colspan=1>Rank</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Contributor</td><td rowspan=1 colspan=1>Date</td><td rowspan=1 colspan=1>clean2random (hard)</td><td rowspan=1 colspan=1>clean2clean (easy)</td></tr><tr><td rowspan=1 colspan=1>#1</td><td rowspan=1 colspan=1>GigaBrain-0.7Co-train</td><td rowspan=1 colspan=1>GigaAl</td><td rowspan=1 colspan=1>26.08.15</td><td rowspan=1 colspan=1>67.9%6</td><td rowspan=1 colspan=1>66.8%</td></tr><tr><td rowspan=1 colspan=1>#2</td><td rowspan=1 colspan=1>π0.5Co-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>46.0%</td><td rowspan=1 colspan=1>70.7%</td></tr><tr><td rowspan=1 colspan=1>#3</td><td rowspan=1 colspan=1>X-WAMCo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>25.8%</td><td rowspan=1 colspan=1>70.0%</td></tr><tr><td rowspan=1 colspan=1>#4</td><td rowspan=1 colspan=1>Abot-M0Co-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>22.9%</td><td rowspan=1 colspan=1>57.4%</td></tr><tr><td rowspan=1 colspan=1>#5</td><td rowspan=1 colspan=1>X-VLACo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>20.9%</td><td rowspan=1 colspan=1>68.0%</td></tr><tr><td rowspan=1 colspan=1>#6</td><td rowspan=1 colspan=1>Xiaomi Robotics-0Co-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>18.2%</td><td rowspan=1 colspan=1>62.996</td></tr><tr><td rowspan=1 colspan=1>#7</td><td rowspan=1 colspan=1>π0Single</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>25.08.04</td><td rowspan=1 colspan=1>16.34%</td><td rowspan=1 colspan=1>46.42%</td></tr><tr><td rowspan=1 colspan=1>#8</td><td rowspan=1 colspan=1>EventVLACo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>15.7%</td><td rowspan=1 colspan=1>65.6%</td></tr><tr><td rowspan=1 colspan=1>#9</td><td rowspan=1 colspan=1>RDTSingle</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>25.08.04</td><td rowspan=1 colspan=1>13.72%</td><td rowspan=1 colspan=1>34.50%</td></tr><tr><td rowspan=1 colspan=1>#10</td><td rowspan=1 colspan=1>Spatial ForcingCo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>9.5%</td><td rowspan=1 colspan=1>77.2%</td></tr><tr><td rowspan=1 colspan=1>#11</td><td rowspan=1 colspan=1>GalaxeaVLACo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>9.1%</td><td rowspan=1 colspan=1>62.7%</td></tr><tr><td rowspan=1 colspan=1>#12</td><td rowspan=1 colspan=1>DP3Single</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>25.08.04</td><td rowspan=1 colspan=1>4.96%</td><td rowspan=1 colspan=1>55.24%</td></tr><tr><td rowspan=1 colspan=1>#13</td><td rowspan=1 colspan=1>starVLACo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>3.3%</td><td rowspan=1 colspan=1>44.3%</td></tr><tr><td rowspan=1 colspan=1>#14</td><td rowspan=1 colspan=1>AHA-WAMCo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>3.2%</td><td rowspan=1 colspan=1>64.3%</td></tr><tr><td rowspan=1 colspan=1>#15</td><td rowspan=1 colspan=1>FastWAMCo-train</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>26.08.10</td><td rowspan=1 colspan=1>1.9%</td><td rowspan=1 colspan=1>77.8%</td></tr><tr><td rowspan=1 colspan=1>#16</td><td rowspan=1 colspan=1>ACTSingle</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>25.08.04</td><td rowspan=1 colspan=1>1.74%</td><td rowspan=1 colspan=1>29.74%</td></tr><tr><td rowspan=1 colspan=1>#17</td><td rowspan=1 colspan=1>DPSingle</td><td rowspan=1 colspan=1>RoboTwin Team</td><td rowspan=1 colspan=1>25.08.04</td><td rowspan=1 colspan=1>0.64%</td><td rowspan=1 colspan=1>28.04%</td></tr></table>

Figure 28: Snapshot of the official RoboTwin 2.0 leaderboard. The leaderboard reports the evaluation protocol together with performance under the clean Easy and domain-randomized Hard settings. GigaBrain-0.7 is evaluated under the official Co-Train protocol used for the comparison in Tab. 9.

## A.2. EBench

Leaderboard URL: https://internrobotics.shlab.org.cn/eval/

Snapshot accessed: August 15, 2026.

The complete leaderboard snapshot above includes submissions that are not part of the representative publicly available model comparison in the main text. In particular, Qwen-RobotManip is retained here as part of the complete official leaderboard record.

The InternVLA-A1 result reported in Tab. 10 is not taken from the online leaderboard snapshot above. It is

<table><tr><td>VLA 模型 EBench 评测榜单</td></tr><tr><td>排名 模型名称 提交者 成功率 分数 操作</td></tr><tr><td>Qwen-RobotManip Qwen-Team 45.58% 60 查看详情</td></tr><tr><td>2 GigaBrain-0.7 GigaAl 33.27% 46 查看详情</td></tr><tr><td>3 pi05 InternRobotics_baseline 28.08% 42 查看详情</td></tr><tr><td>4 X-VLA InternRobotics_baseline 23.72% 35 查看详情 5 pi0 InternRobotics_baseline 23.59% 37 查看详情</td></tr><tr><td></td></tr></table>

Figure 29: Snapshot of the complete official EBench leaderboard. The leaderboard reports overall task success rate and aggregate task score for all listed submissions at the time of manuscript preparation.

reproduced from the EBench evaluation reported by Qwen-RobotManip [87] and is marked with a superscript asterisk in the main-text table.

## A.3. RoboColiseum

Leaderboard URL: https://robocoliseum.ai/

Snapshot accessed: August 15, 2026.

RoboColiseum reports separate leaderboards for four complementary capability dimensions: instruction following, spatial reasoning, robustness, and general manipulation. Figures 30–33 provide separate snapshots of the four leaderboard dimensions associated with the comparison in Tab. 11.

<table><tr><td>Instruction Following Spatial Reasoning Robustness General Manipulation 1</td></tr><tr><td>Rank Account Model Score Submitted At</td></tr><tr><td>GigaAl Open GigaBrain-0.7 0.8166 2026-08-12 14:15:49 View</td></tr><tr><td>2 Z0-AE Z0-AE-Instruction 0.8122 2026-08-07 09:26:39</td></tr><tr><td>3 Junkrat retrieval_vla_pi05_v3_b01_control 0.7833 2026-08-11 12:08:17</td></tr><tr><td>ACoT-VLA Open ACoT-VLA 4 0.757 2026-06-23 18:11:41 View</td></tr><tr><td>口 Mosson L_ins_v2 0.7545 2026-07-24 11:23:07</td></tr><tr><td>6 cns pi05_instruction_baseline 0.7541 2026-07-28 15:18:14</td></tr><tr><td>π₀.5 Open π₀.5 0.746 2026-06-26 10:07:04 View</td></tr><tr><td>8 Roboticsx ACoT-VLA 0.7459 2026-08-07 15:04:18</td></tr><tr><td>9 kp-robot instruction_base_v1.0-1 0.7344 2026-07-16 17:38:16</td></tr><tr><td>Seraph baseline-instruction-v1 0.7315 2026-07-02 12:40:44</td></tr><tr><td>dqq pi05_instruction_and_robust 0.723 2026-07-07 11:22:04</td></tr><tr><td>12 dai_zai 51模 0.7181 2026-07-06 17:04:25</td></tr><tr><td>13 GR00T N1.7 Open GR00T N1.7 0.646 2026-06-23 11:34:04 View</td></tr><tr><td>Xiaomi-Robotics-0 Open Xiaomi-Robotics-0 0.646 2026-08-11 11:11:04 View</td></tr><tr><td>15 π₀ Open π₀ 0.368 2026-06-22 15:18:22 View</td></tr><tr><td>Instruction Following Spatial Reasoning Robustness General Manipulation 11////////11</td></tr><tr><td>Rank Account Model Score Submitted At</td></tr><tr><td>GigaAl Open GigaBrain-0.7 0.4729 2026-08-09 13:38:55 View</td></tr><tr><td>2 USTB pi05-spatial-test 0.4062 2026-08-14 20:39:33</td></tr><tr><td>3 ACoT-VLA Open ACoT-VLA 0.397 2026-06-23 18:11:41 View</td></tr><tr><td>Mosson Spa_v1 0.3645 2026-07-14 14:50:14</td></tr><tr><td>π0.5 Open π0.5 0.356 2026-06-26 10:07:04 View</td></tr><tr><td>6 kp-robot spatial_base_v1.0 0.3166 2026-07-20 08:53:44</td></tr><tr><td>Awesomeman666 pi05 0.3145 2026-07-16 20:20:00</td></tr><tr><td>8 dqq pi05_spatial 0.3041 2026-07-07 14:33:27</td></tr><tr><td>GR00T N1.7 Open GR00T N1.7 0.249 2026-06-23 11:34:04 View</td></tr><tr><td>10 Xiaomi-Robotics-0 COpen Xiaomi-Robotics-0 0.223 2026-08-11 11:11:04 View</td></tr><tr><td>11 Open π₀ 0.13 2026-06-22 15:18:22 View</td></tr><tr><td>πo</td></tr></table>

Figure 30: Snapshot of the RoboColiseum Instruction Following leaderboard. The leaderboard evaluates the ability of robot policies to follow language-conditioned instructions at the time of manuscript preparation.

Figure 31: Snapshot of the RoboColiseum Spatial Reasoning leaderboard. The leaderboard evaluates spatial reasoning capability at the time of manuscript preparation.

<table><tr><td>Instruction Following Spatial Reasoning Robustness General Manipulation 1111/1111111</td></tr><tr><td>Rank Account Model Score Submitted At Open Source</td></tr><tr><td>GigaAl Open GigaBrain-0.7 0.68 2026-08-12 20:24:54 View</td></tr><tr><td>2 Z0-AE Z0-AE-Robustness 0.6399 2026-08-06 16:21:49</td></tr><tr><td>3 Junkrat retrieval_vla_pi05_v5_r50_robust 0.6313 2026-08-13 19:49:15</td></tr><tr><td>4 ACoT-VLA Open ACoT-VLA 0.622 2026-06-23 18:11:41 View</td></tr><tr><td>5 cns pi05_robust_baseline 0.6217 2026-07-29 09:54:09</td></tr><tr><td>6 π0.5 Open π₀.5 0.613 2026-06-26 10:07:04 View</td></tr><tr><td>kp-robot robustness_v1.0-1 0.5949 2026-07-16 20:40:24</td></tr><tr><td>Mosson Rob_v1 0.5925 2026-07-16 09:31:53</td></tr><tr><td>9 Xiaomi-Robotics-0 Open Xiaomi-Robotics-0 0.546 2026-08-11 11:11:04 View</td></tr><tr><td>10 GR00T N1.7 Open GR00T N1.7 0.538 2026-06-23 11:34:04 View 11 Open 0.313 2026-06-22 15:18:22 View</td></tr><tr><td>π₀ π₀</td></tr><tr><td>Instruction Following Spatial Reasoning Robustness General Manipulation 11//////////</td></tr><tr><td>Rank Account</td></tr><tr><td>Model Score Submitted At Open Source GigaAl Open GigaBrain-0.7 0.6092 2026-08-11 23:15:04 View</td></tr><tr><td>2 Z0-AE Z0-AE-Manipulation 0.5982 2026-08-03 18:27:15</td></tr><tr><td>π0.5 Open π0.5 0.582 3 2026-06-26 10:07:04 View</td></tr><tr><td>4 Mosson Manip_v1 0.5694 2026-07-10 10:35:17</td></tr><tr><td>5 Seraph baseline-manip-v1 0.5472 2026-07-02 10:42:31</td></tr><tr><td>5 kp-robot manip_base_combine_v1.0 0.5465 2026-08-03 09:12:24</td></tr><tr><td>ACoT-VLA Open ACoT-VLA 0.477 2026-06-23 18:11:41 View</td></tr><tr><td>8 Harmony harmony 0.4638 2026-07-28 19:28:41</td></tr><tr><td>GR00T N1.7 Open GR00T N1.7 0.438 2026-06-23 11:34:04 View</td></tr><tr><td>10 π₀ Open π₀ 0.347 2026-06-22 15:18:22 View</td></tr><tr><td>11 Xiaomi-Robotics-0 Open Xiaomi-Robotics-0 0.307 2026-08-11 11:11:04 View</td></tr></table>

Figure 32: Snapshot of the RoboColiseum Robustness leaderboard. The leaderboard evaluates policy robustness under the corresponding RoboColiseum protocol at the time of manuscript preparation.

Figure 33: Snapshot of the RoboColiseum General Manipulation leaderboard. The leaderboard evaluates general manipulation capability at the time of manuscript preparation.

## References

[1] Robocoliseum: A comprehensive, multi-dimensional benchmark for evaluating core capabilities of robotic models. https://robocoliseum.ai/, 2026. Accessed: 2026-08-14. 39, 40, 41, 42

[2] AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025. URL https://arxiv.org/ abs/2503.06669.7

[3] AgiBot World Team. Agibot world 2026. https://huggingface.co/datasets/agibot-world/ AgiBotWorld2026, 2026. 7

[4] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728-1738, 2021. URL https://openaccess.thecvf.com/content/ICCV2021/html/ Bain\_Frozen\_in\_Time\_A\_Joint\_Video\_and\_Image\_Encoder\_for\_ICCV\_2021\_paper.html. 10

[5] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A Vision-Language-Action Flow Model for General Robot Control, 2024. URL https://arxiv.org/abs/2410.24164. arXiv preprint arXiv:2410.24164. 2, 3, 4, 18, 41, 42

[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, 2023. URL https://arxiv.org/abs/2307.15818. arXiv preprint arXiv:2307.15818. 2, 3, 4

[7] Remi Cadene, Simon Aliberts, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, Mustafa Shukor, Jess Moss, Alexander Soare, Dana Aubakirova, Quentin Lhoest, Quentin Gallouedec, and Thomas Wolf. LeRobot: An opensource library for end-to-end robot learning. arXiv preprint arXiv:2602.22818, 2026. URL https: //arxiv.org/abs/2602.22818.3

[8] Junhao Cai, Zetao Cai, Jiafei Cao, Yilun Chen, Zeyu He, Lei Jiang, Hang Li, Hengjie Li, Yang Li, Yufei Liu, Yanan Lu, Qi Lv, Haoxiang Ma, Jiangmiao Pang, Yu Qiao, Zherui Qiu, Yanqing Shen, Xu Shi, Yang Tian, Bolun Wang, Hanqing Wang, Jiaheng Wang, Tai Wang, Xueyuan Wei, Chao Wu, Yiman Xie, Boyang Xing, Yuqiang Yang, Yuyin Yang, Qiaojun Yu, Feng Yuan, Jia Zeng, Jingjing Zhang, Shenghan Zhang, Shi Zhang, Zhuoma Zhaxi, Bowen Zhou, Yuanzhen Zhou, Yunsong Zhou, Hongrui Zhu, Yangkun Zhu, and Yuchen Zhu. Internvla-a1: Unifying understanding, generation and action for robotic manipulation, 2026. URL https://arxiv.org/abs/2601.02456. 41

[9] Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, Feng Qiu, Heng Qu, Yifei Su, Qiao Sun, Dong Wang, Donghao Wang, Yunhong Wang, Rujie Wu, Diyun Xiang,

Yu Yang, Hangjun Ye, Yuan Zhang, and Quanyun Zhou. Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution, 2026. URL https://arxiv.org/abs/2602.12684. arXiv preprint arXiv:2602.12684. 3, 4, 14, 40, 41, 42

[10] Lang Cao, Renhong Chen, Luyi Li, Peng Wang, Mofan Peng, and Yitong Li. Z-1: Efficient reinforcement learning for vision-language-action models, 2026. URL https://arxiv. org/abs/2606.31846. 5

[11] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021. URL https: //openaccess.thecvf.com/content/CVPR2021/html/Changpinyo\_Conceptual\_12M\_Pushing-Web-Scale\_Image-Text\_Pre-Training\_To\_Recognize\_Long-Tail\_Visual\_CVPR\_2021\_paper.html. 10

[12] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation, 2024. URL https://arxiv. org/abs/2410.06158. arXiv preprint arXiv:2410.06158. 6

[13] Kaiyuan Eric Chen, Shuangyu Xie, Zehan Ma, Pannag Sanketi, and Ken Goldberg. Robo2VLM: Improving visual question answering using large-scale robot manipulation data. In Advances in Neural Information Processing Systems: Datasets and Benchmarks Track, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 1f467c3e37abf9f86c78f44c6a27ee7c-Abstract-Datasets\_and\_Benchmarks\_Track.html.10

[14] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Weijie Wan, Baijun Chen, Haoran Lu, Haowen Yan, Honghao Su, et al. RoboDojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies. arXiv preprint arXiv:2607.04434, 2026. 42

[15] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=itonej9GIV. 9, 39, 40,41

[16] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024. doi: 10.15607/RSS.2024.XX.045. URL https://www.roboticsproceedings.org/rss20/p045.html. 7

[17] XPolicyLab Community, Tianxing Chen, Yue Chen, Tian Nian, Zijian Cai, Guangyu Chen, Wenwei Lin, Qiwei Liang, Peicheng Xiang, Kailun Su, et al. XPolicyLab: A unified standard and open ecosystem for robot policy evaluation and deployment. arXiv preprint arXiv:2608.09892, 2026. 42

[18] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh

Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and PixMo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 91–104, 2025. doi: 10.1109/CVPR52734.2025. 00018. URL https://openaccess.thecvf.com/content/CVPR2025/html/Deitke\_Molmo\_and\_PixMo\_ Open\_Weights\_and\_Open\_Data\_for\_State-of-the-Art\_CVPR\_2025\_paper.html.10

[19] Zibin Dong, Yicheng Liu, Shiduo Zhang, Baijun Ye, Yifu Yuan, Fei Ni, Jingjing Gong, Xipeng Qiu, Hang Zhao, Yinchuan Li, and Jianye Hao. ActionCodec: What Makes for Good Action Tokenizers, 2026. URL https://arxiv.org/abs/2602.15397. arXiv preprint arXiv:2602.15397. 3

[20] Danny Driess, Jost Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, and Sergey Levine. Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better. In Advances in Neural Information Processing Systems, volume 38, pages 102867–102888. Curran Associates, Inc., 2025. doi: 10.52202/085713-3439. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 94e936034d12bcd04834ec2773f02aff-Paper-Conference.pdf.3,4, 19

[21] Fhrozen. Open Images Narratives v2. Hugging Face dataset, 2025. URL https://huggingface.co/ datasets/Fhrozen/openimages-narratives-v2. Accessed 2026-08-13. 10

[22] Galaxea Team. Galaxea open-world dataset and g0 dual-system vla model. arXiv preprint arXiv:2509.00576, 2025. URL https://arxiv.org/abs/2509.00576.7

[23] Galaxea Team. Galaxea G0.5 Technical Report. Technical report, Galaxea AI, 2026. URL https: //opengalaxea.github.io/G05/Galaxea\_G0\_5.pdf. 3,23, 40

[24] Ning Gao, Jinliang Zheng, Xing Gao, Haoxiang Ma, Hanqing Wang, Yukai Wang, Jiantong Chen, Zanxin Chen, Shujie Zhang, Mingda Jia, Xuekun Jiang, Zihou Zhu, Xinyu Li, Shuai Wang, Hao Li, Wenzhe Cai, Yuqiang Yang, Xudong Xu, Zhaoyang Lyu, Yao Mu, Tai Wang, Jiangmiao Pang, Jia Zeng, Weinan Zhang, and Chunhua Shen. Ebench: Elemental diagnosis of generalist mobile manipulation policies, 2026. URL https://arxiv.org/abs/2606.18239. 9,39,40, 41

[25] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D'Ambrosio, Sudeep Dasari, Todor Davchev, Coline Devin, Norman Di Palo, Tianli Ding, Adil Dostmohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Humplik Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Jost Tobias Springenberg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Yuxiang Yang, Rui Yao, Sergey Yaroshenko, Wenhao Yu, Wentao Yuan, Jingwei Zhang, Tingnan Zhang, Allan Zhou, and Yuxiang Zhou. Gemini Robotics: Bringing AI into the Physical World, 2025. URL https://arxiv.org/abs/2503.20020. arXiv preprint arXiv:2503.20020. 4

[26] GenRobot. 10kh-realomin-opendata: Jianzhi 10k umi data source. Hugging Face dataset, 2026. URL https://huggingface.co/datasets/genrobot2025/10Kh-Real0min-0penData. 7

[27] GigaBrain Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jie Li Jiagang Zhu, Lv Feng, Peng Li, Qiuping Deng, Runqi Ouyang, Wenkang Qin, Xinze Chen, Xiaofeng Wang, Yang Wang, Yifan Li, Yilong Li, Yiran Ding, Yuan Xu, Yun Ye, Yukun Zhou, Zhehao Dong, Zhenan Wang, Zhichao Liu, and Zheng Zhu. GigaBrain-0: A World Model-Powered Vision-Language-Action Model, 2025. URL https://arxiv.org/abs/2510.19430. arXiv preprint arXiv:2510.19430. 2, 3, 4, 5, 6, 19

[28] GigaBrain Team, Boyuan Wang, Bohan Li, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Jie Li, Jindi Lv, Jingyu Liu, Lv Feng, Mingming Yu, Peng Li, Qiuping Deng, Tianze Liu, Xinyu Zhou, Xinze Chen, Xiaofeng Wang, Yang Wang, Yifan Li, Yifei Nie, Yilong Li, Yukun Zhou, Yun Ye, Zhichao Liu, and Zheng Zhu. GigaBrain-0.5M\*: a VLA That Learns From World Model-Based Reinforcement Learning, 2026. URL https://arxiv.org/abs/2602.12099. arXiv preprint arXiv:2602.12099. 2, 3, 5, 6, 21

[29] GLM-5-Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, et al. GLM-5: From vibe coding to agentic engineering, 2026. URL https: //arxiv.org/abs/2602.15763.11

[30] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Confer ence on Computer Vision and Pattern Recognition, pages 6904–6913, 2017. URL https://openaccess. thecvf.com/content\_cvpr\_2017/html/Goyal\_Making\_the\_v\_CVPR\_2017\_paper.html. 10

[31] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In International Conference on Learning Representations, 2023.9

[32] Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li and Huaping Liu. Unified 4d world action modeling from video priors with asynchronous denoising, 2026. URL https://arxiv.org/abs/2604.26694. 41

[33] Songhao Han, Boxiang Qiu, Yue Liao, Siyuan Huang, Chen Gao, Shuicheng Yan, and Si Liu. Robocerebra: A large-scale benchmark for long-horizon robotic manipulation evaluation. arXiv preprint arXiv:2506.06677, 2025.9

[34] Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, Mingliang Zhou, Yinan Zheng, Zihao Yue, Shuhao Gu, Hao Tian, Yuannan Shen, Jianwei Cui, Wen Zhang, Shaoqing Xu, Bing Wang, Haiyang Sun, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Chaofan Zhang, Wenbo Ding, Kun Ma, Guang Chen, Rui Cai, Diyun Xiang, Heng Qu, Fuli Luo, Hangjun Ye, and Long Chen. Mimo-embodied: X-embodied foundation model technical report, 2026. URL https://arxiv.org/abs/2511.16518. 39, 40

[35] Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video. In The Fourteenth International Conference on Learning Representations, 2026. doi: 10.48550/arXiv.2505.11709. URL https://openreview.net/forum?id= FFxkFMU89E. 9

[36] Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, Qiuxuan Feng, Zhao Jin, et al. RoboMIND 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence. arXiv preprint arXiv:2512.24653, 2025. URL https://arxiv.org/ abs/2512.24653.7

[37] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, and Zhiyuan Zhou. π0.6: a vla that learns from experience, 2025. URL https: //arxiv.org/abs/2511.14759.27

[38] Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, Loic Magne, Ajay Mandlekar, Avnish Narayan, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Yinzhen Xu, Xiaohui Zeng, Kaiyuan Zheng, Ruijie Zheng, Ming-Yu Liu, Luke Zettlemoyer, Dieter Fox, Jan Kautz, Scott Reed, Yuke Zhu, and Linxi Fan. DreamGen: Unlocking Generalization in Robot Learning through Video World Models, 2025. URL https://arxiv. org/abs/ 2505.12705. arXiv preprint arXiv:2505.12705. 6

[39] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-Language-Action Model, 2024. URL https://arxiv. org/abs/2406.09246. arXiv preprint arXiv:2406.09246. 2, 3, 4

[40] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success, 2025. URL https://arxiv.org/abs/2502.19645. arXiv preprint arXiv:2502.19645. 3,5

[41] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning, 2026. URL https://arxiv.org/abs/2601.16163. arXiv preprint arXiv:2601.16163. 2,6

[42] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, Aniruddha Kembhavi, Abhinav Gupta, and Ali Farhadi. AI2- THOR: An Interactive 3D Environment for Visual AI, 2017. URL https://arxiv.org/abs/1712.05474. 10

[43] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The open images dataset V4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. doi: 10.1007/s11263-020-01316-z. URL https://doi.org/10.1007/s11263-020-01316-z. 10

[44] Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. Behavior Generation with Latent Actions, 2024. URL https://arxiv. org/abs/2403.03181. arXiv preprint arXiv:2403.03181. 3

[45] Hao Li, Ziqin Wang, Zi-han Ding, Shuai Yang, Yilun Chen, Yang Tian, Xiaolin Hu, Tai Wang, Dahua Lin Feng Zhao, Si Liu, and Jiangmiao Pang. RoboInter: A holistic intermediate representation suite towards robotic manipulation, 2026. URL https://arxiv.org/abs/2602.09973. Accepted at ICLR 2026. 10

[46] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning, 2025. URL https://arxiv.org/abs/2509.09674. arXiv preprint arXiv:2509.09674. 5

[47] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In Computer Vision – ECCV 2014, pages 740–755. Springer, 2014. doi: 10.1007/978-3-319-10602-1\_48. URL https://arxiv.org/abs/ 1405.0312.10

[48] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling, 2022. URL https://arxiv.org/abs/2210.02747. arXiv preprint arXiv:2210.02747. 2

[49] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. doi: 10.1109/CVPR52733.2024. 02484. URL https://openaccess.thecvf.com/content/CVPR2024/html/Liu\_Improved\_Baselines\_ with\_Visual\_Instruction\_Tuning\_CVPR\_2024\_paper.html. 10

[50] Xiaokang Liu, Zechen Bai, Hai Ci, Kevin Yuchen Ma, and Mike Zheng Shou. World-vla-loop: Closed-loop learning of video world model and vla policy, 2026. URL https://arxiv. org/abs/2602.06508. 6

[51] Edith Llontop and Kalyan Vadrevu. NVIDIA Isaac GR00T N1.7: Open Reasoning VLA Model for Humanoid Robots. Hugging Face Blog, April 2026. URL https://huggingface.co/blog/nvidia/gr00t-n1-7. 42

[52] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning, 2025. URL https://arxiv.org/abs/2505.18719. arXiv preprint arXiv:2505.18719. 5

[53] Jindi Lv, Hao Li, Jie Li, Fankun Kong, Yang Wang, Pengfei Yi, Yifei Nie, Xiaofeng Wang, Zheng Zhu Chaojun Ni, Qiuping Deng, Hengtao Li, Jiancheng Lv, and Guan Huang. Viva: A video-generative value model for robot reinforcement learning. https://arxiv.org/abs/2604.08168, 2026. Accessed: 2026-08-16. 20

[54] Michael Matthews, Michael Beukman, Chris Lu, and Jakob Foerster. Kinetix: Investigating the training of general agents through open-ended physics-based control tasks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2410.23208. 9

[55] Shangchen Miao, Ningya Feng, Jialong Wu, Ye Lin, Xu He, Dong Li, and Mingsheng Long. JEPA-VLA: Video Predictive Embedding is Needed for VLA Models, 2026. URL https://arxiv.org/abs/2602.11832. arXiv preprint arXiv:2602.11832. 6

[56] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots, 2025. URL https://arxiv.org/abs/2503.14734. arXiv preprint arXiv:2503.14734. 3,4

[57] NVIDIA, :, Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini. Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, Aarti Basant, Mukesh Beladiya, Mohammad Qazim Bhat Zaid Pervaiz Bhat, Dan Blick, Vanni Brighella, Han Cai, Tiffany Cai, Eric Cameracci, Jiaxin Cao, Yulong Cao, Mark Carlson, Carlos Casanova, Ting-Yun Chang, Yan Chang, Yu-Wei Chao, Prithvijit Chattopadhyay, Roshan Chaudhari, Chieh-Yun Chen, Junyu Chen, Ke Chen, Qizhi Chen, Wenkai Chen, Xiaotong Chen, Yu Chen, An-Chieh Cheng, Click Cheng, Xiu Chia, Jeana Choi, Chaeyeon Chung, Wenyan Cong, Yin Cui, Magdalena Dadela, Nalin Dadhich, Wenliang Dai, Joyjit Daw, Alperen Degirmenci, Rodrigo Vieira Del Monte, Robert Denomme, Sameer Dharur, Marco Di Lucca, Ke Ding, Wenhao Ding, Yifan Ding, Yuzhu Dong, Nicole Drumheller, Yilun Du, Aigul Dzhumamuratova, Aleksandr Efitorov, Hamid Eghbalzadeh. Naomi Eigbe, Imad El Hanafi, Hassan Eslami, Benedikt Falk, Jiaojiao Fan, Jim Fan, Amol Fasale, Sergiy Fefilatyev, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Vikram Fugro, Prashant Gaikwad, TJ Galda. Katelyn Gao, Yihuai Gao, Wenhang Ge, Sreyan Ghosh, Arushi Goel, Vivek Goel, Akash Gokul, Rama Govindaraju, Jinwei Gu, Miguel Guerrero, Elfie Guo, Aryaman Gupta, Siddharth Gururani, Hugo Hadfield, Song Han, Ankur Handa, Zekun Hao, Mohammad Harrim, Ali Hassani, Nathan Hayes-Roth, Yufan He Chris Helvig, Cyrus Hogg, Madison Huang, Michael Huang, Sophia Huang, Yufan Huang, Jacob Huffman, DeLesley Hutchins, Suneel Indupuru, Boris Ivanovic, Arihant Jain, Joel Jang, Ryan Ji, Yanan Jian, Dongfu Jiang, Jingyi Jin, Atharva Joshi, Nikhilesh Joshi, Pranjali Joshi, Andy Ju, Jaehun Jung, Weiwei Kang, Scott Kassekert, Jan Kautz, Ashna Khetan, Julia Kiczka, Slawek Kierat, Gwanghyun Kim, Kuno Kim, Sunny Kim, Kezhi Kong, Xin Kong, Zhifeng Kong, Tomasz Kornuta, Egor Krivov, Hui Kuang, Saurav Kumar Chia-Wen Kuo, George Kurian, Wojciech Kutak, JF Lafleche, Himangshu Lahkar, Omar Laymoun, Jayjun Lee, Sanggil Lee, Gabriele Leone, Boyi Li, Freya Li, Jiajun Li, Jinfeng Li, Ling Li, Pengcheng Li, Shangru Li, Tingle Li, Xiaolong Li, Xuan Li, Zhaoshuo Li, Zhiqi Li, Hao Liang, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Ming-Yu Liu, Sifei Liu, Zihan Liu, Hai Loc Lu, Xiangyu Lu, Alice Luo, Ruipu Luo, Wenjie Luo, Jiangran Lyu, Martin Ding Ma, Nic Ma, Qianli Ma, Dawid Majchrowski, Louis Marcoux, Miguel Martin, Qing Miao, Ashkan Mirzaei, Shreyas Misra, Kaichun Mo, Durra Mohsin, Hyejin Moon, Pawel Morkisz, Saeid Motiian, Kirill Motkov, Seungjun Nah, Yashraj Narang, Deepak Narayanan, Thabang Ngazimbi, Julian Ouyang, Shubham Pachori, David Page, Yatian Pang, Sehwi Park, Mahesh Patekar, Mostofa Patwary, Marco Pavone, Trung Pham, Wei Ping, Soha Pouya, Shrimai Prabhumoye, Varun Praveen, Delin Qu, Hesam Rabeti, Morteza Ramezanali, Marilyn Reeb, Xuanchi Ren, Kristen Rumley, Wojciech Rymer, Jun Saito, Yeongho Seol, John Shao, Piyush Shekdar, Tianwei Shen, Humphrey Shi, Min Shi, Stella Shi, Kevin Shih, Mohammad Shoeybi, Mateusz Sieniawski, Shuran Song, Alexander Sotelo, Amir Sotoodeh, Sunil Srinivasa, Vignesh Srinivasakumar, Bartosz Stefaniak, Rahul Heinrich Steiger, Shangkun Sun, Jiaxiang Tang, Shitao Tang, Yangyang Tang, Yue Tang, Tolou Tavakkoli, Kayley Ting, Krzysztof Tomala, Wei-Cheng Tseng, Jibin Varghese, Sergei Vasilev, Thomas Volk, Raju Wagwani, Roger Waleffe Andrew Z. Wang, Boxiang Wang, Haoxiang Wang, Qiao Wang, Shihao Wang, Shijie Wang, Ting-Chun Wang, Yan Wang, Yu Wang, Rohit Watve, David Wehr, Fangyin Wei, Xinshuo Weng, Jay Zhangjie Wu, Kedi Wu, Hongchi Xia, Summer Xiao, Tianjun Xiao, Kevin Xie, Daguang Xu, Jiashu Xu, Mengyao Xu, Ruqing Xu, Xingqian Xu, Yao Xu, Dinghao Yang, Dong Yang, Hans Yang, Xiaodong Yang, Xuning Yang Yichu Yang, Yurong You, Zhiding Yu, Hao Yuan, Simon Yuen, Xiaohui Zeng, Pengcuo Zeren, Cindy Zha Haotian Zhang, Jenny Zhang, Jing Zhang, Liangkai Zhang, Paris Zhang, Shun Zhang, Xuanmeng Zhang Zhizheng Zhang, Ann Zhao, Yilin Zhao, Yuliya Zhautouskaya, Charles Zhou, Fengzhe Zhou, Shilin Zhu, Yuke Zhu, Dima Zhylko, and Artur Zolkowski. Cosmos 3: Omnimodal World Models for Physical AI, 2026 URL https://arxiv.org/abs/2606.02800. arXiv preprint arXiv:2606.02800. 6

[58] Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903, May 2024. doi: 10.1109/ICRA57147.2024.10611477. URL https://ieeexplore.ieee.org/document/10611477. 7

[59] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-Weighted Regression: Simple

and Scalable Off-Policy Reinforcement Learning, 2019. URL https://arxiv.org/abs/1910.00177. arXiv preprint arXiv:1910.00177. 5, 22

[60] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient Action Tokenization for Vision-Language-Action Models, 2025. URL https://arxiv.org/abs/2501.09747. arXiv preprint arXiv:2501.09747. 3

[61] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail. Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, and Zhiyuan Zhou. $\pi _ { 0 . 6 } ^ { * } \colon$ a VLA That Learns From Experience, 2025. URL https: //arxiv.org/abs/2511.14759. arXiv preprint arXiv:2511.14759. 5

[62] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a Vision-Language-Action Model with Open-World Generalization, 2025. URL https://arxiv.org/abs/2504.16054. arXiv preprint arXiv:2504.16054. 2, 3, 4, 5, 23, 41, 42

[63] Physical Intelligence, Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black. George Bokinsky, Shihao Cao, Thomas Charbonnier, Vedant Choudhary, Foster Collins, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Maitrayee Dhaka, Jared DiCarlo, Danny Driess, Michael Equi Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachlan Groom, Haroun Habeeb, Hunter Hancock, Karol Hausman, Gashon Hussein, Victor Hwang, Brian Ichter, Connor Jacobsen, Szymon Jakubczak, Rowan Jen, Tim Jones, Gregg Kammerer, Ben Katz, Liyiming Ke, Mairbek Khadikov, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Brendon LeCount, Sergey Levine, Xinyu Li, Adrian Li-Bell, Vladislav Lialin, Zhonglin Liang, Wallace Lim, Yao Lu, Enyu Luo, Vishnu Mano, Nandan Marwaha, Aikys Mongush, Liam Murphy, Suraj Nair, Tyler Patterson, Karl Pertsch, Allen Z. Ren, Gavin Schelske, Charvi Sharma, Baifeng Shi, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg Kyle Stachowicz, Will Stoeckle, Jiaming Tang, Jimmy Tanner, Shalom Tekeste, Marcel Torne, Kyle Vedder, Quan Vuong, Anna Walling, Haohuan Wang, Jason Wang, XuDong Wang, Chris Whalen, Samuel Whitmore, Blake Williams, Charles Xu, Sukwon Yoo, Lili Yu, Wuming Zhang, Zhuoyang Zhang, and Ury Zhilinsky. π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities, 2026. URL https://arxiv.org/abs/2604.15483. arXiv preprint arXiv:2604.15483. 2, 4, 6, 27

[64] Ryan Punamiya, Simar Kareer, Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Jiaqi Chen, Davide Liconti, Lawrence Y. Zhu, Patcharapong Aphiwetsa, Baoyu Li, Aniketh Cheluva, Pranav Kuppili, Yangcen Liu, Dhruv Patel, Aidan Gao, Hye-Young Chung, Ryan Co, Renee Zbizika, Jeff Liu Xiaomeng Xu, Haoyu Xiong, Geng Chen, Sebastiano Oliani, Wenkai Xuan, Chenyu Yang, Xi Wang, James Fort, Richard Newcombe, Josh Gao, Jason Chong, Garrett Matsuda, Aseem Doriwala, Marc Pollefeys Robert Katzschmann, Xiaolong Wang, Shuran Song, Judy Hoffman, and Danfei Xu. EgoVerse: An Egocentric Human Dataset for Robot Learning from Around the World, 2026. URL https://arxiv. org/ abs/2604.07607. arXiv preprint arXiv:2604.07607. 9

[65] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5. 23

[66] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL https: //qwen.ai/blog?id=qwen3.6-27b. 14

[67] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: Open dataset of CLIP-filtered 400 million image-text pairs. In NeurIPS Workshop on Data-Centric AI, 2021. URL https: //arxiv. org/ abs/2111.02114. 10

[68] Hengyu Shen, Tiancheng Gu, Bin Qin, Lan Wu, Yuling Wu, Shuo Tan, Zelong Sun, Jun Wang, Nan Wu Xiang An, Weidong Cai, Ziyong Feng, and Kaicheng Yang. DanQing: An up-to-date large-scale chinese vision–language pre-training dataset, 2026. URL https://arxiv.org/abs/2601.10305. 10

[69] Junyang Shu, Zhiwei Lin, Bingqing Wei, and Yongtao Wang. Feat2go: Visual feature-grounded value estimation for embodied reinforcement learning, 2026. URL https://arxiv.org/abs/2605.30795. 6

[70] Spirit AI Team. Spirit-v1.5: Clean Data Is the Enemy of Great Robot Foundation Models. Spirit AI Blog, 2026. URL https://www.spirit-ai.com/en/blog/spirit-v1-5. 23,40

[71] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, Siyang Qin, Reeve Ingle, Emanuele Bugliarello, Sahar Kazemzadeh, Thomas Mesnard, Ibrahim Alabdulmohsin, Lucas Beyer, and Xiaohua Zhai. PaliGemma 2: A Family of Versatile VLMs for Transfer, 2024. URL https: //arxiv. org/ abs/2412.03555. arXiv preprint arXiv:2412.03555. 15, 23

[72] Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for reasoning about natural language grounded in photographs. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6418-6428. Association for Computational Linguistics 2019. doi: 10.18653/v1/P19-1644. URL https://aclanthology.org/P19-1644/. 10

[73] Yingbo Tang, Lingfeng Zhang, Shuyi Zhang, Yinuo Zhao, and Xiaoshuai Hao. RoboAfford: A dataset and benchmark for enhancing object and spatial affordance learning in robot manipulation. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12706–12713. Association for Computing Machinery, 2025. doi: 10.1145/3746027.3758209. URL https://doi.org/10.1145/3746027.3758209. 10

[74] Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon, et al. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026. 23

[75] GigaWorld Team, Angyuan Ma, Boyuan Wang, Bohan Li, Chaojun Ni, Guo Li, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jingyu Liu, Jiwen Lu, Qiuping Deng, Tingdong Yu, Xuancheng Xu, Xinyu Zhou, Xiuwei Xu, Xinze Chen, Xiaofeng Wang, Xiaoyu Tian, Yang Wang, Yifan Chang, Yukun Zhou, Yun Ye, Zhenyu Wu, Zhanqian Wu, and Zheng Zhu. Gigaworld-1: A roadmap to build world models for robot policy evaluation, 2026. URL https://arxiv.org/abs/2607.02642. 15, 16, 20

[76] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In Advances in Neural Information Processing Systems, volume 37, pages 87310–87356, 2024. doi: 10.52202/079017-2771. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html. 10

[77] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu. Hongyu Wu, Yiheng Liu, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025. URL https://arxiv.org/abs/2511.17441. 7

[78] Wei Wu, Fangjing Wang, Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang, Shuailei Ma Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su, Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, and Kecheng Zheng. From Foundation to Application: Improving VLA Models in Practice, 2026. URL https://arxiv.org/abs/2607.06403. arXiv preprint arXiv:2607.06403. 2, 3, 5

[79] Xiaomi Robotics Team, Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, Optimus Qin, Yifei Su, Nan Sun, Qiao Sun, Runze Suo, Heyun Wang, Yunhong Wang, Rujie Wu, Caoyu Xia, Lina Zhang, Jack Zhao, Guoliang Chen, Wenlong Chen, Xinze He, Bin Li, Qing Li, Zhuorong Li, Heng Qu, Wenxuan Song, Diyun Xiang, Yifan Xie, Peiran Xu, Hangjun Ye, Wen Ye, Han Zhao, and Quanyun Zhou. Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories, 2026. URL https://arxiv.org/abs/2607.15330. arXiv preprint arXiv:2607.15330. 2, 3, 4,5,23

[80] Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, and Liyiming Ke. RL Token: Bootstrapping Online RL with Vision-Language-Action Models, 2026. URL https://arxiv.org/abs/2604.23073. arXiv preprint arXiv:2604.23073. 5

[81] Siyuan Yang, Linzheng Guo, Ouyang Lu, Zhaxizhuoma, Daoran Zhang, Xinmiao Wang, Ting Xiao, Fangzheng Yan, Zhijun Chen, Yan Ding, Chao Yu, Chenjia Bai, and Xuelong Li. VISTA: Vision-grounded and physics-validated adaptation of UMI data for VLA training, 2026. URL https://arxiv. org/abs/ 2606.04708.10

[82] Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, Min Cao, Peng Li, Qiuping Deng, Wenjun Mei, Xiaofeng Wang, Xinze Chen, Xinyu Zhou, Yang Wang, Yifan Chang, Yifan Li, Yukun Zhou, Yun Ye, Zhichao Liu, and Zheng Zhu. GigaWorld-Policy: An Efficient Action-Centered World-Action Model, 2026. URL https://arxiv.org/abs/2603.17240. arXiv preprint arXiv:2603.17240. 6

[83] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, and Joel Jang. World Action Models are Zero-shot Policies, 2026. URL https://arxiv.org/abs/2602.15922. arXiv preprint arXiv:2602.15922.6

[84] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. doi: 10.1162/tacl\_a\_00166. URL https: //aclanthology.org/Q14-1006/. 10

[85] Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Yue Cao, Xinlong Wang, and Jingjing Liu. CapsFusion: Rethinking image-text data at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14022–14032, 2024. URLhttps://openaccess.thecvf.com/content/CVPR2024/html/Yu\_CapsFusion\_Rethinking-Image-Text\_Data\_at\_Scale\_CVPR\_2024\_paper.html.10

[86] Ryan Yu, Pushi Zhang, Starrick Liu, Brae Liu, Miracle Kang, Shalfun Li, Lights Shi, Ellie Ma, Ping Yang, Chris Pan, Jerry Chen, Dongxiu Liu, Rain Sun, Miles Guo, Byron Zhang, Hugo Zhou, Zach Xu, Vincent Chen, Harrison Huang, James Wang, Dance Kuzi, Andy Zhai, Hang Su, Roy Gan, Lucy Liang, Hao Wang, and Qian Wang. Wall-OSS-0.5 Technical Report, 2026. URL https://arxiv.org/abs/2605.30877. arXiv preprint arXiv:2605.30877. 2, 3, 4, 5, 23, 40

[87] Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, Jie Zhang, Jingyang Fan, Gengze Zhou, Qihang Peng, Chenxu Lv, Xiaoyue Chen, An Yang, Fei Huang, Junyang Lin, Dayiheng Liu, Jingren Zhou, Chenfei Wu, and Xiong-Hui Chen. Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models, 2026. URL https://arxiv.org/abs/2606.17846. arXiv preprint arXiv:2606.17846. 2, 3, 4, 5, 11, 41, 46

[88] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. RoboPoint: A vision-language model for spatial affordance prediction in robotics. In Proceedings of the 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 4005–4020. PMLR, 2025. URL https://proceedings .mlr.press/v270/ yuan25c.html. 10

[89] He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Hu, and Zhengyou Zhang. Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack, 2026. URL https://arxiv. org/abs/2606.14409. arXiv preprint arXiv:2606.14409. 3, 4, 5, 23, 40

[90] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model, 2025. URL https://arxiv.org/abs/2510.10274.41

[91] Yupeng Zheng, Jichao Peng, Weize Li, Yuhang Zheng, Xiang Li, Yujie Jin, Julong Wei, Guanhua Zhang, Ruiling Zheng, Ming Cao, Songen Gu, Zhenhong Zou, Kaige Li, Ke Wu, Mingmin Yang, Jiahao Liu, Pengfei Li, Hengjie Si, Feiyu Zhu, Wang Fu, Likun Wang, Ruiwen Yao, Jieru Zhao, Yilun Chen, and Wenchao Ding. World in your hands: A large-scale and open-source ecosystem for learning human-centric manipulation in the wild, 2026. URL https://arxiv.org/abs/2512.24310. 9

[92] Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, and Guanghui Ren. Acot-vla: Action chain-of-thought for vision-language-action models, 2026. URL https://arxiv. org/abs/2601.11404. 42

[93] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, and Shanghang Zhang. RoboRefer: Towards spatial referring with reasoning in vision-language models for robotics. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 29416b66c2149872b9d1415a3fd2c5e0-Abstract-Conference.html. 10

[94] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5738–5746, 2019. doi: 10.1109/CVPR.2019.00589. 18

[95] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets. In

Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, June 2025. doi: 10.15607/RSS.2025. XXI.015. URL https://www.roboticsproceedings.org/rss21/p015.html. 6