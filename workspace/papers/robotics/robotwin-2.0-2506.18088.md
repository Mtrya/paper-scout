# RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation

Tianxing Chen<sup>2,16\*†</sup>, Zanxin Chen<sup>3,5\*</sup>, Baijun Chen<sup>15\*</sup>, Zijian Cai<sup>3,5\*</sup>, Yibin Liu<sup>13\*</sup>,   
Zixuan Li<sup>5\*</sup>, Qiwei Liang<sup>5</sup>, Xianliang Lin<sup>5</sup>, Yiheng Ge<sup>1</sup>, Zhenyu Gu<sup>7,8</sup>, Weiliang Deng<sup>3,11</sup>,   
Yubin Guo<sup>7,9</sup>, Tian Nian<sup>3,5</sup>, Xuanbing Xie<sup>12</sup>, Qiangyu Chen<sup>5</sup>, Kailun Su<sup>5</sup>, Tianling Xu<sup>10</sup>, Guodong Liu<sup>6,7</sup>, Mengkang Hu<sup>2</sup>, Huan-ang Gao<sup>6,16</sup>, Kaixuan Wang<sup>2,16</sup>, Zhixuan Liang<sup>2,3†</sup>, Yusen Qin<sup>4,6</sup>, Xiaokang Yang<sup>1</sup>, Ping Luo<sup>2,14B</sup>, Yao Mu<sup>1,3B†</sup>

<sup>1</sup> MoE key Lab of Artificial Intelligence, AI Institute, SJTU<sup>‡</sup>, <sup>2</sup> HKU MMLab<sup>‡</sup>, <sup>3</sup> Shanghai AI Lab, <sup>4</sup>D-Robotics, <sup>5</sup>SZU, <sup>6</sup>THU, <sup>7</sup>TeleAI, <sup>8</sup>FDU, <sup>9</sup>USTC, <sup>10</sup>SUSTech, <sup>11</sup>SYSU, <sup>12</sup>CSU, <sup>13</sup>NEU,<sup>14</sup>HKU-SH ICRC, <sup>15</sup>NJU, <sup>16</sup>Lumina EAI

Equal contribution  Corresponding authors <sup>†</sup> Co-project leads <sup>‡</sup> Equally leading organizations

Webpage:https://robotwin-platform.github.io Doc: https://robotwin-platform.github.io/doc

![](images/18c5bc5c950a274f231b7ef2326ef8ddd238fffb189ee1b1172db319886515bf.jpg)  
Figure 1: Overview of RoboTwin 2.0. RoboTwin 2.0 is a scalable framework for bimanual manipulation, integrating an expert data generation pipeline with a 50-task benchmark built on the RoboTwin Object Dataset (731 objects, 147 categories). A multimodal language agent automates task program synthesis, while flexible dual-arm configurations enable large-scale, diverse data collection. Policies trained on RoboTwin 2.0 exhibit improved robustness and generalization to unseen environments.

## Abstract

Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated,

large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a largescale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments. Empirical evaluation shows a 10.9% gain in code generation success rate. Building on this, we evaluate downstream policy learning. With a mix of large-scale synthetic data and only 10 real demonstrations, a vision–language–action (VLA) model achieves a 367% relative improvement over the 10-demo baseline. Even without real data, zero-shot models trained solely on synthetic data obtain a 228% relative gain, highlighting the effectiveness of our dataset in strengthening sim-to-real transfer and robustness to environmental variations. We release the data generator, benchmark, pre-collected dataset, and code to support scalable research in robust bimanual manipulation.

## 1 Introduction

Bimanual robotic manipulation is critical for enabling robots to perform complex real-world task such as collaborative assembly, tool use, and object handovers. Developing generalizable bimanual policies—particularly vision–language–action (VLA) foundation models—requires datasets that are simultaneously high-quality, diverse, and large-scale. In the absence of sufficient variability in object geometry, scene clutter, lighting conditions, instruction language, and robot embodiments, learned policies often overfit to narrow distributions and fail to generalize to novel environments or hardware platforms. Yet collecting real-world demonstrations at scale remains prohibitively expensive, timeconsuming, and logistically challenging, especially when aiming to cover a broad range of tasks, objects, and embodiments.

Simulation-based data generation provides a scalable alternative for collecting large-scale multimodal datasets and has shown promise in enabling sim-to-real transfer [34, 11]. However, existing pipelines fall short in three critical aspects. First, they lack automated quality control: without an expert-level validation loop, many generated trajectories include execution failures or suboptimal grasps, which degrade policy learning. Second, their domain randomization is often superficial, yielding overly clean and homogeneous scenes that omit essential real-world factors such as clutter, lighting variation, and ambiguous language instructions—elements crucial for robust sim-to-real transfer. Third, they overlook cross-embodiment variation: different bimanual platforms can differ substantially in their kinematic capabilities and grasp strategies. For example, a low-degree-of-freedom (DoF) platform like the Piper often relies on lateral grasps due to its limited dexterity, whereas a high-DoF arm such as the Franka is capable of top-down precision grasps. Yet, current synthetic datasets rarely encode such embodiment-specific affordances or task constraints, limiting their generality.

To address these challenges, we introduce RoboTwin 2.0, a scalable simulation-based data generation framework designed to produce high-quality, diverse, realistic, and interaction-rich datasets for bimanual manipulation. RoboTwin 2.0 integrates three key components: (1) an automated expert data generation pipeline that leverages multimodal large language models (MLLMs) and simulationin-the-loop feedback to iteratively validate and refine task execution code; (2) comprehensive domain randomization over language instructions, object clutter, background textures, lighting conditions, and tabletop configurations, aimed at closing the sim-to-real gap and enhancing policy generalization; and (3) embodiment-aware adaptation, in which object affordances are annotated and robot-specific action candidates are generated to account for heterogeneous dual-arm kinematics.

Building on these components, we introduce three new resources to support scalable research in bimanual manipulation: (1) the RoboTwin-OD asset library, comprising 731 annotated object instances across 147 categories; (2) an automated data generation pipeline with comprehensive domain randomization and a collection of over 100,000 expert trajectories spanning 50 tasks across five dual-arm robot platforms; and (3) a benchmark for evaluating policy generalization to cluttered environments and open-ended language goals. Together, these resources enable the community to train and evaluate robust bimanual manipulation policies under conditions that closely reflect real-world complexity and diversity.

In summary, our main contributions are as follows: (1) We develop an automated expert data generation framework that integrates multimodal large language models with simulation-in-theloop feedback to ensure high-quality, expert-level trajectories; (2) We propose a systematic domain randomization strategy that enhances policy robustness by increasing data diversity and sim-to-real generalization; (3) We introduce an embodiment-aware adaptation mechanism that generates robotspecific manipulation candidates based on object affordances; (4) We release the RoboTwin-OD asset library, a large-scale pre-collected multi-embodiment domain-randomized trajectory dataset, a scalable bimanual data generator, and a standardized evaluation benchmark to support scalable training and evaluation of generalizable policies across different robot embodiments, scene configurations, and language instructions.

## 2 Method

![](images/cec647b59b7754a0c5607b506d4fbddb0ce8cdf803b3e386dbf6c67f2254f7c1.jpg)  
Figure 2: RoboTwin 2.0 Pipeline. Built on RoboTwin-OD and a skill API, the framework uses MLLM-based code generation with simulation feedback to produce expert task programs and domainrandomized trajectories for policy training and evaluation.

We illustrate the overall RoboTwin 2.0 pipeline in Fig. 2. The framework begins with a task code generation module that leverages multimodal large language models (MLLMs) and simulation-in-theloop feedback to automatically synthesize executable task plans from natural language instructions. This module is grounded on a large-scale object asset library (RoboTwin-OD) and a predefined skill library, enabling scalable task instantiation across a broad range of object categories and manipulation scenarios. To ensure high-quality expert demonstrations, we integrate this automated generation pipeline with RoboTwin 2.0’s comprehensive domain randomization scheme, which diversifies observations along language, visual, and spatial axes. This pipeline supports the synthesis of diverse and realistic training data, facilitating the development of manipulation policies that are robust to real-world environmental variability.

## 2.1 Expert Code Generation via MLLMs and Simulation-in-the-Loop Feedback

Recent advances in language models show strong ability to generate intermediate task representations—such as textual plans [18], API calls, or executable code [33, 6, 19]—for complex robotic tasks. Multimodal large language models (MLLMs) further extend this capability by incorporating visual and proprioceptive inputs, enabling more grounded reasoning over real-world sensory data. However, prior systems often depend on strong manual priors or lack closed-loop feedback during program synthesis, which limits their robustness in diverse or dynamic environments.

To address these limitations, we propose an automated expert data generation pipeline that integrates programmatic code synthesis with multimodal execution feedback (Fig.3). The system adopts a closed-loop architecture with two agents: a code-generation agent and a vision–language model (VLM) observer. The code agent synthesizes task programs from instructions, while the observer monitors execution in simulation, detects failures, and suggests corrections. This iterative feedback loop enables the code agent to refine programs automatically, producing robust, self-improving expert data with minimal human supervision. Unlike prior MLLM-based pipelines such as GenSim2[20] and RoboGen [43], our system supports zero-shot generation of complex dual-arm behaviors beyond primitive pick-and-place actions.

![](images/6960c2561c20317b221d16ea2e01223ed307f34de655ac3a129c430fbc8c2953.jpg)  
Figure 3: Expert Code Generation Pipeline.

Input Specification. Each task is defined by a task name (e.g., Handover Block) and a natural language description of the objective. The code-generation agent is conditioned on three key inputs: a general API list, a set of example function calls, and a hierarchical constraint specification. These components collectively guide the synthesis of Python code to execute the task. Additionally, each task may include task-specific function call examples to further ground code generation in context.

Initial Code Generation. The code-generation agent synthesizes an initial Python program conditioned on the provided task inputs. It models the program synthesis process as a structured prediction problem over the space of available API calls, leveraging natural language understanding and few-shot prompting from task-specific examples. The generated code specifies a stepwise sequence of robot actions designed to accomplish the target manipulation objective.

Simulated Execution and Logging. The generated program is executed ten times per iteration within a simulated robotic environment. Multiple trials are used to account for stochastic variations in simulation dynamics, robot controllers, and sensor noise. After each execution batch, the system generates a structured execution log that records the success or failure of each trial and annotates failure cases with their corresponding causes—such as unexecutable code, left/right grasp failure, or incorrect object placement.

Multimodal Observation and Error Localization. In parallel with execution, a vision-language model (VLM) agent observes the robot’s behavior across all ten trials. The VLM performs frameby-frame inspection to evaluate the success of each program step and localize the point of failure when errors occur. Beyond temporal localization, the VLM also diagnoses failure modes by inferring whether the underlying cause stems from flawed logic, incorrect API usage, or other systemic issues. This diagnostic capability enables the system to address root causes rather than merely responding to superficial execution errors. The detail of VLM observation is shown in G.4.

Code Repair and Iterative Refinement. The code-generation agent receives two complementary feedback signals: (i) a quantitative execution log and (ii) a qualitative, localized diagnostic from the VLM. It integrates these inputs to revise the program by modifying or replacing instructions identified as failure-prone. The updated program is then re-evaluated in the next iteration, and the process continues until either the program achieves the setting success rate across ten simulated runs in one iteration or fails to do so after five consecutive refinements. This loop yields expert-level task code with minimal human supervision while avoiding indefinite refinement.

The outcome of this pipeline is a collection of robust, automatically synthesized programs that generate high-quality expert trajectories for downstream training and evaluation. By integrating multimodal reasoning with execution-level feedback, the system produces code that is not only syntactically correct but also semantically aligned with task objectives. This closed-loop generation framework substantially reduces human supervision while enabling scalable and self-improving expert data creation for complex robotic manipulation tasks.

## 2.2 Domain Randomization for Robust Robotic Manipulation

To enhance policy robustness to real-world variability, we apply domain randomization along five dimensions: (1) cluttered distractor objects, (2) background textures, (3) lighting conditions, (4) tabletop heights, and (5) diverse language instructions. This systematic augmentation broadens the training distribution and markedly improves generalization to unseen scenarios. The effects of these randomizations are visualized in Fig. 4a.

![](images/8344201f8d46e31444a68651ef8bd17429d36ea3b1d3f4d9348d966ee32d97bd.jpg)  
(a) Visualization of Domain Randomization

![](images/a0fba910e961cbf9afbf06aedaf60b15130cb635e38279c538e0ce38ae0e8031.jpg)  
(b) Texture Library  
Figure 4: Visualization of domain randomization and our texture library.

Scene Clutter. To enhance robustness to environmental variation, we augment tabletop scenes with task-irrelevant distractors drawn from RoboTwin-OD (731 objects across 147 categories; see Section 3.1). Each object includes placement annotations, enabling a generic API for semantically valid insertion. We ensure physical plausibility via collision-aware placement and precomputed volumes. To avoid policy confusion, distractors visually or semantically similar to task-relevant objects are excluded during sampling. This yields diverse yet unambiguous cluttered scenes for training.

Diverse Background Textures. We randomize tabletop surfaces and backgrounds using a large curated texture library. To build it, we first collected 1,000 diverse surface descriptions via LLM prompting and web crawling, then used Stable Diffusion v2 to generate 20 samples per description (20,000 total). After human-in-the-loop filtering, we obtained 11,000 high-quality textures. This library is applied in simulation to enrich visual diversity and reduce overfitting to clean synthetic environments (see Fig. 4b).

Lighting Variation. Real-world environments exhibit diverse illumination conditions, with variations in color temperature, source type, number, and placement. These factors alter object appearance and reflections, challenging vision-based manipulation. To enhance robustness, we randomize light color, type, intensity, and position within physically plausible bounds. As shown in Fig. 4a (second row), changes in color temperature can drastically shift object appearance (e.g., a shoe under warm vs. cool light). Training under such randomized conditions improves policy robustness to real-world illumination shifts.

Tabletop Heights. In practice, table heights vary across workspaces, affecting robot perception, kinematics, and interaction. To improve generalization, we uniformly randomize table height within a plausible range during simulation, introducing variability in viewpoints and spatial relations between robot and objects.

Trajectory-Level Diverse Language Instructions. To improve robustness to natural language variation, we use a multimodal LLM to generate diverse task templates and multiple object descriptions capturing geometry, appearance, and part-level attributes. Each task and object thus has several alternative phrasings, which can be flexibly combined. For every trajectory, we sample from these pools to compose instructions. For example, in Move Can Pot, the template “Use a to place A to the left of B” may yield diverse instructions such as “Use left arm to place sauce can to the left of gray kitchenpot” or “Use left arm to place white plastic lid sauce can to the left of kitchenpot for boiling and cooking.” This combinatorial augmentation produces a large set of linguistically varied instructions and significantly improves generalization to unseen language and scene configurations (see Appendix H, I).

## 2.3 Embodiment-Aware Grasp Adaptation

Due to differences in DoF and kinematic structures, robotic arms exhibit varying reachable workspaces and preferred manipulation strategies for the same task. For example, when grasping a can, the Franka arm typically favors a top-down approach, while the lower-DoF Piper arm is better suited to side grasps. As a result, a task successfully completed by Franka using a top-down grasp may require a side approach when executed with Piper, as shown in Fig. 6.

![](images/58beede98387ff0af055f7a3d5f301a45d8cee441e9287f598eb7a1c04e93c84.jpg)  
Aloha-AgileX

![](images/cd2907b0528f9ee9fed7f726e5aca114ecebee9fe974ae168356e0938509b29f.jpg)  
ARX-X5

![](images/5174fd6c1518173aed4eef48c6185e56ab543feed79a5d9e5464943cc3fa00a1.jpg)  
Piper

Figure 5: Five RoboTwin 2.0 Embodiments.  
![](images/25a8a7a33fb547987163d8c4f3ec80dff1e941bf558d5583a2eb524a418a6eb3.jpg)  
Franka

![](images/3e9a7c6927810152836e525778402b87ea517f8b14d0e521a895cd99c001d8ef.jpg)  
UR5

![](images/42fc4c0017272dc4972f1fe3f6bb86de28ab67a08dd6c927c188f5401cb2837f.jpg)  
Figure 6: Different Grasping Behavior.

To address embodiment-specific variations, we annotate each object with a rich set of candidate manipulation poses that cover multiple grasp axes and approach directions. This design captures both manipulation diversity and robot-specific preferences. To further expand the feasible space, we apply angular perturbations biased toward directions with higher arm reachability. Concretely, for each object we generate candidate grasps by combining preferred operation directions, randomized pose perturbations, and parallelized motion planning attempts.

## 3 RoboTwin 2.0 Data Generator, Benchmark and Large Scale Dataset

## 3.1 RoboTwin-OD: RoboTwin Object Dataset

![](images/a88acff40299fb9f3efccc1201f1da5837a2c1bdd1ee7795775bf9c4bc83b9ac.jpg)  
Figure 7: RoboTwin-OD. A large-scale object dataset for robotic manipulation with 147 categories and 731 objects, annotated with rich interaction labels and diverse language descriptions.

To enhance both manipulation capability and visual understanding, we construct a large-scale object dataset with rich semantic annotations, called RoboTwin-OD, covering 147 categories and 731 diverse objects. Specifically, this includes 534 instances across 111 categories that we generated in-house using RGB-to-3D reconstruction via the Rodin platform<sup>1</sup>, followed by convex decomposition and mesh merging to ensure physically accurate collision models. In addition, RoboTwin-OD incorporates 153 objects from 27 categories in Objaverse [10], and 44 articulated object instances from 9 categories in SAPIEN PartNet-Mobility [48]. Objects from all sources, including Objaverse, are used to construct cluttered scenes, with Objaverse specifically enhancing the visual and semantic diversity of distractor objects. We also develop a comprehensive texture library for surfaces and backgrounds using generative AI and human-in-the-loop filtering to ensure visual realism and diversity.

For robust manipulation, policies must generalize across diverse objects, which requires datasets with broad category coverage and varied intra-class instances. To facilitate language grounding, we developed an automated object description generator with human verification, producing 15 annotations per object that vary in shape, texture, functionality, part structure, and granularity.

To further support object-centric interaction, we annotate each object with key point–axis information, including placement points, functional points, grasp points, and grasp axes, explicitly encoding affordances. Together with our manipulation API library, these annotations enable generalizable grasp execution in simulation. All object information is available at http://robotwinplatform.github.io/doc/objects/ .

## 3.2 50 Tasks for Data Generation and Benchmarking

Building on our automated task generation framework, embodiment-adaptive behavior synthesis, and the large-scale RoboTwin-OD asset library, we construct a suite of 50+ dual-arm collaborative manipulation tasks. We further support data collection and evaluation on five distinct robot platforms, enabling comprehensive cross-embodiment benchmarking. Keyframes from representative tasks are shown in Fig. 8, and the complete task descriptions are available at http://robotwin-platform.github.io/doc/tasks/. We also pre-collected over 100,000 dualarm manipulation trajectories across 50 tasks in RoboTwin 2.0, which are available at https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset.

![](images/7edba150d31b3e5ecdd6dfacaa4d3bbea867ce27ceb62827a51df9be5f181279.jpg)  
Figure 8: 50 RoboTwin 2.0 Bimanual Manipulation Tasks.

## 4 Experiment

We design experiments to evaluate the effectiveness of RoboTwin 2.0 in three key aspects: (1) automating the generation of high-quality expert code for manipulation tasks; (2) improving policy robustness to environmental variation via diversified training data; and (3) demonstrating the utility and diversity of RoboTwin 2.0 as a standardized benchmark for evaluating policy generalization across tasks, scenes, and embodiments.

## 4.1 Evaluation of Automated Expert Code Generation

We evaluate our closed-loop expert data generation system on a suite of 10 robotic manipulation tasks, each specified with a natural language instruction. For each configuration, the code-generation agent produces multiple candidate programs, which are executed in simulation to account for stochasticity in dynamics, control, and perception. Task-level success is defined as the average success rate across all executions, as described in Section 2.1.

Table 1: Overall performance comparison across RoboTwin variants. Evaluated on the subset of tasks supported by both RoboTwin 1.0 and RoboTwin 2.0. Pertask success rate comparison is provided in Appendix 8.
<table><tr><td>Method</td><td>ASR</td><td>Top5-ASR</td><td>CR-Iter</td><td>Token</td></tr><tr><td>R1.0 Vanilla</td><td>47.4%</td><td>57.6%</td><td>1.00</td><td>1236.6</td></tr><tr><td>R1.0 + FB</td><td>60.4%</td><td>71.4%</td><td>2.46</td><td>1190.4</td></tr><tr><td>R1.0 + MM FB</td><td>63.9%</td><td>74.2%</td><td>2.42</td><td>1465.0</td></tr><tr><td>R2.0 Vanilla</td><td>62.1%</td><td>68.0%</td><td>1.00</td><td>569.4</td></tr><tr><td>R2.0 + FB</td><td>66.7%</td><td>73.6%</td><td>1.89</td><td>581.6</td></tr><tr><td>R2.0 + MM FB</td><td>71.3%</td><td>78.6%</td><td>1.76</td><td>839.7</td></tr></table>

![](images/6e3d2a74406fd3bbc2a771caf6e20c56a72e887bdb1be251fb6364c75c9e9678.jpg)  
Figure 9: RoboTwin Success Rate Distribution.

We evaluate performance with four metrics: ASR (Average Success Rate), Top5-ASR (success over the top-5 candidates per task), CR-Iter (average refinement iterations before termination), and Token (average number of tokens in generated policy code). Results on RoboTwin 1.0 and 2.0 are reported in Table 1 under three configurations: Vanilla (one-shot code generation), FB (feedback-based repair via execution logs), and MM FB (multimodal feedback with vision–language diagnostics). Per-task success rates are provided in Appendix 8.

Across all settings, multimodal feedback yields consistent gains. In RoboTwin 1.0, ASR improves from 47.4% (Vanilla) to 63.9% (MM FB); in RoboTwin 2.0, it rises from 62.1% to 71.3%. Improvements are also evident in Top5-ASR, suggesting that perceptual feedback disproportionately benefits the best candidate programs. RoboTwin 2.0 converges faster than 1.0 (e.g., 1.76 vs. 2.42 CR-Iter in MM FB), indicating stronger priors and more efficient refinement. Token cost is also substantially reduced, especially in Vanilla (569.4 vs. 1236.6), reflecting more concise initial code.

Figure 9 further shows that feedback narrows the success-rate distribution and raises the median. RoboTwin 2.0 with multimodal feedback achieves compact distributions centered above 80%, highlighting robustness and reliability.

Overall, three findings emerge: (1) vision–language feedback not only detects failures but also guides precise repairs; (2) architectural improvements in RoboTwin 2.0 accelerate convergence and reduce token usage; and (3) combining symbolic execution logs with perceptual diagnostics yields more reliable, semantically aligned expert data. Together, these results validate the effectiveness of our closed-loop, self-improving code generation architecture. Detailed setups, metric definitions, and additional analyses are provided in Appendix G.

## 4.2 Evaluating Efficiency with and without Adaptive Grasping

Table 2: Overall Performance Comparison between RoboTwin 1.0 and RoboTwin 2.0.
<table><tr><td>Method</td><td>Aloha-AgileX</td><td>Piper</td><td>Franka</td><td>UR5</td><td>ARX-X5</td><td>Average</td></tr><tr><td>RoboTwin 1.0</td><td>65.1%</td><td>2.4%</td><td>67.3%</td><td>57.6%</td><td>68.6%</td><td>52.2%</td></tr><tr><td>RoboTwin 2.0</td><td>78.8%</td><td>25.1%</td><td>67.2%</td><td>57.1%</td><td>74.2%</td><td>60.5%</td></tr><tr><td>Difference</td><td>+13.7%</td><td>+22.7%</td><td>-0.1%</td><td>-0.5%</td><td>+5.6%</td><td>+8.3%</td></tr></table>

To evaluate the effectiveness of our embodiment-aware grasp augmentation strategy, we measure the task success rate of automated data collection across 50 RoboTwin 2.0 tasks on five different robot embodiments. As shown in Table 2, we compare our RoboTwin 2.0 pipeline against the RoboTwin 1.0 baseline, which lacks diverse grasping and candidate augmentation. Results show that our method improves success rates, particularly for robots with constrained planning spaces, achieving an average improvement of 8.3% across all embodiments. Specifically, for high-DoF arms with large reachable workspaces, such as Franka and UR5 (7-DoF), success rates remain largely unchanged, indicating limited benefit when the robot already has sufficient kinematic flexibility. However, for lower-DoF platforms such as Aloha-AgileX, Piper, and ARX-X5 (6-DoF), our method leads to substantial gains of 13.5%, 22.7%, and 5.6%, respectively. These results demonstrate that our approach provides additional feasible grasp options that effectively mitigate the planning limitations of low-DoF manipulators. Success rates for all tasks can be found in Appendix L.

## 4.3 Assessing the Impact of RoboTwin 2.0 on Policy Robustness

Our goal is to evaluate whether the domain-randomized data in RoboTwin 2.0 can endow models with robustness to environmental perturbations. To this end, we first pre-train RDT and Pi0 on 9,600 expert trajectories collected from 32 tasks (300 per task) under two settings: clean (non-randomized) and domain-randomized.

For comparison, we also evaluate the released pretrained weights of RDT and Pi0 without additional fine-tuning. To further study generalization, we select five unseen tasks and collect 50 clean demonstrations per task for single-task training and fine-tuning. Finally, all policies—including ACT, DP, RDT, and Pi0—are evaluated under domain-randomized conditions to measure robustness in previously unseen environments. Detailed configurations are provided in Appendix C and D.

Table 3: Evaluating the Impact of RoboTwin 2.0 on Policy Robustness.
<table><tr><td>Simulation Tasks</td><td>ACT</td><td>DP</td><td>RDT</td><td>Pi0</td><td>RDT +Clean</td><td>Pi0 +Clean</td><td>RDT +Rand.</td><td>Pi0 +Rand.</td></tr><tr><td>Stack Bowls Two</td><td>0.0%</td><td>0.0%</td><td>30.0%</td><td>41.0%</td><td>8.0%</td><td>55.0%</td><td>49.0%</td><td>62.0%</td></tr><tr><td>Pick Dual Bottles</td><td>0.0%</td><td>0.0%</td><td>13.0%</td><td>12.0%</td><td>12.0%</td><td>15.0%</td><td>17.0%</td><td>7.0%</td></tr><tr><td>Move Can Pot</td><td>4.0%</td><td>0.0%</td><td>12.0%</td><td>21.0%</td><td>13.0%</td><td>35.0%</td><td>18.0%</td><td>22.0%</td></tr><tr><td>Place Object Basket</td><td>0.0%</td><td>0.0%</td><td>17.0%</td><td>2.0%</td><td>9.0%</td><td>8.0%</td><td>6.0%</td><td>22.0%</td></tr><tr><td>Place Shoe</td><td>0.0%</td><td>0.0%</td><td>7.0%</td><td>6.0%</td><td>9.0%</td><td>6.0%</td><td>30.0%</td><td>18.0%</td></tr><tr><td>Open Laptop</td><td>0.0%</td><td>0.0%</td><td>32.0%</td><td>46.0%</td><td>21.0%</td><td>33.0%</td><td>35.0%</td><td>50.0%</td></tr><tr><td>Press Stapler</td><td>6.0%</td><td>0.0%</td><td>24.0%</td><td>29.0%</td><td>21.0%</td><td>26.0%</td><td>27.0%</td><td>31.0%</td></tr><tr><td>Turn Switch</td><td>2.0%</td><td>1.0%</td><td>15.0%</td><td>23.0%</td><td>24.0%</td><td>21.0%</td><td>16.0%</td><td>21.0%</td></tr><tr><td>Average</td><td>2.0%</td><td>0.0%</td><td>18.8%</td><td>22.5%</td><td>14.6%</td><td>24.9%</td><td>24.8%</td><td>29.1%</td></tr></table>

As shown in Table 3, we observe that models fine-tuned with clean data show negligible improvements in average success rate compared to their pretrained counterparts, indicating that data without domain randomization does not help the model handle environmental variations. This also suggests that the low success rate of pretrained VLAs in simulation is not due to a Real-to-Sim gap, since we provide clean simulation data yet observe no clear improvement. In contrast, models pretrained with RoboTwin 2.0 data exhibit significantly improved generalization. Specifically, RDT and Pi0 achieve relative improvements of 31.9% and 29.3%. Notably, this performance gain persists even though the downstream tasks were trained using only clean, non-randomized data. This demonstrates that domain-randomized pretraining with RoboTwin 2.0 effectively equips models with robustness to visual and spatial variations. As a result, models pretrained with RoboTwin 2.0 can adapt to new tasks without requiring additional data augmentation or complex scene variations.

## 4.4 Evaluation on Sim-to-Real Performance

![](images/57ee7a2d6ce5879198104d1db80e89f37faf85a5c375d50a268da49be4ab2285.jpg)  
Seen Bg + not Cluttered

![](images/ff1d867aaa137b378af255d40444c7b7d5c0fd66e5faa86b0fb441391729790a.jpg)  
Unseen Bg + not Cluttered

![](images/58093af709ab1deb5331b11b24b3b323eb1ff43395fa59f8a30cd1901c095c9d.jpg)  
Seen Bg + Cluttered

![](images/a767b90dc654ade38553e26a693c8bef58f0c09ab99886182014770f5fe0857f.jpg)  
Unseen Bg + Cluttered  
Figure 10: Real-World Evaluation across Four Configurations.

To evaluate RoboTwin 2.0’s effectiveness in enhancing real-world policy robustness, we conduct experiments on four bimanual tasks: Stack Bowls, Handover Block, Pick Bottle, and Click Bell. All experiments use RDT as the policy backbone and are executed on the COBOT-Magic dual-arm platform. We compare three training settings: (1) 10 real-world demonstrations in clean tabletop environments; (2) the same demonstrations augmented with 1,000 domain-randomized synthetic trajectories generated under cluttered scenes with varied lighting and backgrounds; and (3) a synthetic only setting trained solely on the 1,000 domain-randomized trajectories. To further improve robustness to camera jitter and calibration errors, we apply random 3D perturbations to simulated camera poses (position and orientation), with the displacement magnitude bounded by 1 cm.

Evaluation is conducted under four test configurations: (i) clean tabletop with seen backgrounds, (ii) clean tabletop with unseen backgrounds, (iii) cluttered tabletop with seen backgrounds, and (iv) cluttered tabletop with unseen backgrounds (Fig.10). Since the synthetic-only setting excludes seen backgrounds during training, the corresponding entries in Table 4 are omitted. This setup directly tests whether RoboTwin 2.0 enables robust policy generalization without additional real-world data from visually complex environments.

Table 4: Real-World Experiment Results. We conduct controlled experiments on 4 dual-arm tasks: Stack Bowls, Handover Block, Pick Bottle, and Click Bell, each evaluated under 4 different settings.
<table><tr><td>Real World Task</td><td>Background Type</td><td>Cluttered or Not</td><td>10 Clean Real</td><td>1k RoboTwin 2.0</td><td>10 Clean Real + 1k RoboTwin 2.0 (Zero-shot)</td></tr><tr><td rowspan="3">Stack Bowls</td><td>Seen</td><td>False True</td><td>22.0% 12.0%</td><td>64.0% 58.0%</td><td>1 1</td></tr><tr><td>Unseen</td><td>False</td><td>10.0%</td><td>50.0%</td><td>60.0%</td></tr><tr><td rowspan="2"></td><td>True</td><td>12.0%</td><td>56.0%</td><td>52.0%</td></tr><tr><td>Seen</td><td>False</td><td>40.0%</td><td>48.0%</td><td>1</td></tr><tr><td rowspan="3">Handover Block</td><td></td><td>True</td><td>16.0%</td><td>12.0%</td><td>1</td></tr><tr><td rowspan="2">Unseen</td><td>False</td><td>36.0%</td><td>56.0%</td><td>56.0%</td></tr><tr><td>True</td><td>0.0%</td><td>36.0%</td><td>20.0%</td></tr><tr><td rowspan="3">Pick Bottle</td><td>Seen</td><td>False True</td><td>20.0% 8.0%</td><td>36.0% 40.0%</td><td>1 1</td></tr><tr><td rowspan="2">Unseen</td><td>False</td><td>4.0%</td><td>26.0%</td><td>10.0%</td></tr><tr><td>True</td><td>8.0%</td><td>28.0%</td><td>32.0%</td></tr><tr><td rowspan="4">Click Bell</td><td rowspan="2">Seen</td><td>False</td><td>36.0%</td><td>24.0%</td><td>1</td></tr><tr><td>True</td><td>20.0%</td><td>56.0%</td><td>1</td></tr><tr><td rowspan="2">Unseen</td><td></td><td>12.0%</td><td>24.0%</td><td></td></tr><tr><td>False True</td><td>16.0%</td><td>48.0%</td><td>20.0% 14.0%</td></tr><tr><td rowspan="4">Average</td><td rowspan="2">Seen</td><td></td><td></td><td></td><td></td></tr><tr><td>False</td><td>29.5%</td><td> $4 3 . 0 \% _ { + 1 3 . 5 \% }$  </td><td>1</td></tr><tr><td rowspan="2">Unseen</td><td>True</td><td>14.0%</td><td> $4 1 . 5 \% _ { + 2 7 . 5 \% }$ </td><td>1</td></tr><tr><td>False True</td><td>15.5% 9.0%</td><td>_  $3 9 . 0 \% _ { + 2 3 . 5 \% }$   $4 2 . 0 \% _ { + 3 3 . 0 \% }$ </td><td> $3 6 . 5 \% _ { + 2 1 . 0 \% }$   $2 9 . 5 \% _ { + 2 0 . 5 \% }$ </td></tr></table>

The experimental results show that real-world bimanual policies augmented with RoboTwin 2.0 achieve clear gains in robustness. In the few-shot setting—where 1,000 domain-randomized synthetic trajectories are combined with just 10 real-world demonstrations—the average success rate across all evaluation settings improves by 24.4%, with per-configuration gains of 13.5%, 27.5%, 23.5%, and 33.0%, respectively. In the zero-shot setting trained solely on synthetic data, we still observe notable improvements of 21.0% and 20.5% on the two unseen-background scenarios. Notably, performance gains become larger in visually complex scenes, indicating that RoboTwin 2.0 is especially effective under challenging conditions.

These improvements stem from two factors: (1) the high visual and physical fidelity of RoboTwin 2.0, enabling direct sim-to-real transfer, and (2) the ability of domain-randomized synthetic data to prepare policies for environmental variations absent from clean real-world demonstrations. Importantly, the strong performance of the few-shot setting suggests that only minimal real-world data is needed to effectively bridge the sim-to-real gap.

## 4.5 RoboTwin 2.0 Benchmark

To evaluate the benchmarking utility and generalization challenges of RoboTwin 2.0, we assess five policy models: ACT, DP, RDT, Pi0, and DP3. All VLAs are fine-tuned from their released pretrained weights in the single-task setting. Evaluations are conducted on all 50 benchmark tasks using the Aloha AgileX dual-arm embodiment. For each task, 50 clean expert demonstrations are used for training, and policies are tested with 100 rollouts under two conditions: Easy (clean) and Hard (domain-randomized with clutter, lighting, textures, and height variations). We provide a visualization of the benchmark setting in Appendix J. We report success rates as indicators of few-shot adaptation and robustness. Detailed setups are provided in Appendix C and D, and full results are available in Appendix K and on the Leaderboard.

Table 5: Subset of RoboTwin 2.0 benchmark. Full results in Appendix K and Leaderboard.
<table><tr><td>Simulation Task</td><td colspan="2">RDT</td><td colspan="2">Pi0</td><td colspan="2">ACT</td><td colspan="2">DP</td><td colspan="2">DP3</td></tr><tr><td></td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td></tr><tr><td>Adjust Bottle</td><td>81%</td><td>75%</td><td>90%</td><td>56%</td><td>97%</td><td>23%</td><td>97%</td><td>0%</td><td>99%</td><td>3%</td></tr><tr><td>Beat Block Hammer</td><td>77%</td><td>37%</td><td>43%</td><td>21%</td><td>56%</td><td>3%</td><td>42%</td><td>0%</td><td>72%</td><td>8%</td></tr><tr><td>Blocks Ranking RGB</td><td>3%</td><td>0%</td><td>19%</td><td>5%</td><td>1%</td><td>0%</td><td>0%</td><td>0%</td><td>3%</td><td>0%</td></tr><tr><td>Blocks Ranking Size</td><td>0%</td><td>0%</td><td>7%</td><td>1%</td><td>0%</td><td>0%</td><td>1%</td><td>0%</td><td>2%</td><td>0%</td></tr><tr><td>Click Alarmclock</td><td>61%</td><td>12%</td><td>63%</td><td>11%</td><td>32%</td><td>4%</td><td>61%</td><td>5%</td><td>77%</td><td>14%</td></tr><tr><td>Click Bell</td><td>80%</td><td>9%</td><td>44%</td><td>3%</td><td>58%</td><td>3%</td><td>54%</td><td>0%</td><td>90%</td><td>0%</td></tr><tr><td>Dump Bin Bigbin Grab Roller</td><td>64% 74%</td><td>32% 43%</td><td>83%</td><td>24%</td><td>68%</td><td>1%</td><td>49%</td><td>0%</td><td>85%</td><td>53%</td></tr><tr><td>Handover Block</td><td>45%</td><td>14%</td><td>96%</td><td>80%</td><td>94% 42%</td><td>25%</td><td>98%</td><td>0%</td><td>98%</td><td>2% 0%</td></tr><tr><td>Handover Mic</td><td></td><td>31%</td><td>45%</td><td>8%</td><td></td><td>0%</td><td>10%</td><td>0%</td><td>70%</td><td></td></tr><tr><td></td><td>90%</td><td></td><td>98%</td><td>13%</td><td>85%</td><td>0%</td><td>53%</td><td>0%</td><td>100%</td><td>3%</td></tr><tr><td>Hanging Mug Lift Pot</td><td>23% 72%</td><td>16% 9%</td><td>11% 84%</td><td>3% 36%</td><td>7% 88%</td><td>0% 0%</td><td>8% 39%</td><td>0% 0%</td><td>17%</td><td>1%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>97%</td><td>0%</td></tr><tr><td>...</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Move Pillbottle Pad</td><td>8%</td><td>0%</td><td>21%</td><td>1%</td><td>0%</td><td>0%</td><td>1%</td><td>0%</td><td>41%</td><td>0%</td></tr><tr><td>Average (in %)</td><td>34.5</td><td>13.7</td><td>46.4</td><td>16.3</td><td>29.7</td><td>1.7</td><td>28.0</td><td>0.6</td><td>55.2</td><td>5.0</td></tr></table>

Fig.5 and Appendix K report results on 50 tasks. Non-pretrained models (ACT, DP, DP3) perform poorly under Hard conditions, while pretrained models (RDT, Pi0) show stronger resilience, suggesting that vision–language–action pretraining provides useful priors for generalization. Still, success rates drop by 20.8% (RDT) and 30.1% (Pi0) from clean to randomized settings, underscoring robustness under domain shifts as a key challenge, likely due to limited diversity in pretraining data. DP3 achieves the best few-shot performance with limited samples, highlighting the role of 3D information, though its strong results partly stem from perfect point clouds and clean background segmentation in simulation. Together with Sections 4.3 and 4.4, these findings show RoboTwin 2.0’s value in complementing existing datasets with diverse, domain-randomized trajectories for improved generalization and robustness.

## 5 Related Work

## 5.1 Datasets and Benchmarks for Robotic Manipulation

Physics-based simulators underpin modern manipulation research. Existing platforms provide complementary strengths: SAPIEN [48] enables dynamic interaction with 2,300+ articulated objects; ManiSkill2 [16] supplies millions of demonstrations; Meta-World [50], CALVIN [32], LIBERO [30], and RoboVerse [15] target multi-task, language-conditioned, lifelong, and domain-randomized settings; RoboCasa [35] offers large-scale human demonstrations but lacks automation and dual-arm focus.

Large-scale real-world datasets further bridge sim-to-real: AgiBot World [4], RoboMIND [47], Open X-Embodiment [36], and Bridge [12] contribute millions of trajectories across diverse tasks, robots, and environments.

RoboTwin-1.0 [34] mirrored real demonstrations with simulated replicas for dual-arm benchmarking. In this work, RoboTwin 2.0 integrates LLM-driven feedback and systematic domain randomization across visual, physical, and task dimensions, producing richer corpora that improve policy robustness and generalization. A detailed comparison with prior benchmarks is provided in Appendix B.

## 5.2 Robot Learning in Manipulation

Many task-specific policy architectures [42, 21, 51, 9, 14, 7, 29, 42, 27, 28, 45, 44, 8] achieve strong single-task performance but struggle to transfer across embodiments. In contrast, foundation models trained on million-scale, multi-robot corpora have enabled robust zero-shot generalization: RT-1 [3] unifies vision, language and actions in a single transformer for real-time kitchen tasks; RT-2 [2] co-fine-tunes large vision–language models on web and robot data to unlock semantic planning and object reasoning; diffusion-based RDT-1B [31] and the π<sub>0</sub>[1] capture diverse bimanual dynamics from over a million episodes. Vision–language–action (VLA) frameworks like OpenVLA [23] and CogACT [26], together with adaptations like Octo [40], LAPA [49], and OpenVLA-OFT [22] demonstrate efficient fine-tuning to novel robots and sensor modalities.

To further advance this direction, our work introduces digital-twin data collection paired with extensive domain randomization, yielding datasets that closely mirror real robot dynamics and support the training of robust and generalizable bi-manual manipulation policies.

## 5.3 Domain Randomization in Imitation Learning

Prior works have shown that randomizing visual and physical parameters, including but not limited to textures, lighting, camera pose, mass, friction and control latency combined with noise injection in expert demonstrations, enables sim-to-real transfer and robust visuomotor policies [41, 37, 5, 7, 28], and optimizing over worst-case ensembles further improves resilience to extreme domain shifts [38, 25, 27]. However, these approaches apply randomization in isolation and lack bidirectional digital-twin feedback; our method integrates interactive simulation feedback with systematic domain randomization to generate higher-fidelity imitation data.

## 6 Conclusion

This paper presented RoboTwin 2.0, a scalable simulation framework for generating diverse, highfidelity expert data to support robust bimanual manipulation. Our system integrates MLLM-based task generation, embodiment-adaptive behavior synthesis, and comprehensive domain randomization to address key limitations in prior synthetic data generator.

By leveraging an annotated object library and automating trajectory generation, RoboTwin 2.0 produces data with rich visual, linguistic, and physical diversity while minimizing manual engineering effort. Experiments demonstrate its effectiveness in improving policy robustness to cluttered environments, generalization to unseen tasks, and cross-embodiment manipulation.

These findings highlight the importance of scalable, automated generation of semantically rich, domain-randomized data for learning robust manipulation policies. RoboTwin 2.0 provides a foundation for unified benchmarks and scalable sim-to-real pipelines, with future work focusing on real-world deployment and multi-object task complexity.

## 7 Acknowledgments

This paper is partially supported by AgileX Robotics, D-Robotics, and the Jockey Club STEM Lab of Autonomous Intelligent Systems funded by The Hong Kong Jockey Club Charities Trust.

## References

[1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi\_0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[2] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[3] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[4] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[5] Yevgen Chebotar, Ankur Handa, Viktor Makoviychuk, Miles Macklin, Jan Issac, Nathan Ratliff, and Dieter Fox. Closing the sim-to-real loop: Adapting simulation randomization with real world experience. In 2019 International Conference on Robotics and Automation (ICRA), pages 8973–8979. IEEE, 2019.

[6] Junting Chen, Yao Mu, Qiaojun Yu, Tianming Wei, Silang Wu, Zhecheng Yuan, Zhixuan Liang, Chao Yang, Kaipeng Zhang, Wenqi Shao, Yu Qiao, Huazhe Xu, Mingyu Ding, and Ping Luo. Roboscript: Code generation for free-form manipulation tasks across real and simulation, 2024.

[7] Tianxing Chen, Yao Mu, Zhixuan Liang, Zanxin Chen, Shijia Peng, Qiangyu Chen, Mingkun Xu, Ruizhen Hu, Hongyuan Zhang, Xuelong Li, et al. G3flow: Generative 3d semantic flow for pose-aware and generalizable object manipulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1735–1744, 2025.

[8] Tianxing Chen, Kaixuan Wang, Zhaohui Yang, Yuhao Zhang, Zanxin Chen, Baijun Chen, Wanxi Dong, Ziyuan Liu, Dong Chen, Tianshuo Yang, et al. Benchmarking generalizable bimanual manipulation: Robotwin dual-arm collaboration challenge at cvpr 2025 meis workshop. arXiv preprint arXiv:2506.23351, 2025.

[9] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023.

[10] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023.

[11] Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Heming Cui, et al. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. arXiv preprint arXiv:2505.03233, 2025.

[12] Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396, 2021.

[13] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, et al. Codebert: A pre-trained model for programming and natural languages. arXiv preprint arXiv:2002.08155, 2020.

[14] Zipeng Fu, Tony Z Zhao, and Chelsea Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117, 2024.

[15] Haoran Geng, Feishi Wang, Songlin Wei, Yuyang Li, Bangjun Wang, Boshi An, Charlie Tianyue Cheng, Haozhe Lou, Peihao Li, Yen-Jen Wang, et al. Roboverse: Towards a unified platform, dataset and benchmark for scalable and generalizable robot learning. arXiv preprint arXiv:2504.18904, 2025.

[16] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, et al. Maniskill2: A unified benchmark for generalizable manipulation skills. In The Eleventh International Conference on Learning Representations, 2023.

[17] Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. Unixcoder: Unified cross-modal pre-training for code representation. arXiv preprint arXiv:2203.03850, 2022.

[18] Mengkang Hu, Tianxing Chen, Qiguang Chen, Yao Mu, Wenqi Shao, and Ping Luo. Hiagent: Hierarchical working memory management for solving long-horizon agent tasks with large language model. arXiv preprint arXiv:2408.09559, 2024.

[19] Mengkang Hu, Tianxing Chen, Yude Zou, Yuheng Lei, Qiguang Chen, Ming Li, Yao Mu, Hongyuan Zhang, Wenqi Shao, and Ping Luo. Text2world: Benchmarking large language models for symbolic world model generation. arXiv preprint arXiv:2502.13092, 2025.

[20] Pu Hua, Minghuan Liu, Annabella Macaluso, Yunfeng Lin, Weinan Zhang, Huazhe Xu, and Lirui Wang. Gensim2: Scaling robot data generation with multi-modal and reasoning llms. In 8th Annual Conference on Robot Learning.

[21] Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. arXiv preprint arXiv:2402.10885, 2024.

[22] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[23] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, et al. Openvla: An open-source vision-languageaction model. In 8th Annual Conference on Robot Learning.

[24] Zhiqian Lan, Yuxuan Jiang, Ruiqi Wang, Xuanbing Xie, Rongkui Zhang, Yicheng Zhu, Peihang Li, Tianshuo Yang, Tianxing Chen, Haoyu Gao, et al. Autobio: A simulation and benchmark for robotic automation in digital biology laboratory. arXiv preprint arXiv:2505.14030, 2025.

[25] Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. Dart: Noise injection for robust imitation learning. In Conference on robot learning, pages 143–156. PMLR, 2017.

[26] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.

[27] Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, and Ping Luo. Adaptdiffuser: Diffusion models as adaptive self-evolving planners. In International Conference on Machine Learning, pages 20725–20745. PMLR, 2023.

[28] Zhixuan Liang, Yao Mu, Hengbo Ma, Masayoshi Tomizuka, Mingyu Ding, and Ping Luo. Skilldiffuser: Interpretable hierarchical planning via skill abstractions in diffusion-based task execution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16467–16476, 2024.

[29] Zhixuan Liang, Yao Mu, Yixiao Wang, Tianxing Chen, Wenqi Shao, Wei Zhan, Masayoshi Tomizuka, Ping Luo, and Mingyu Ding. Dexhanddiff: Interaction-aware diffusion planning for adaptive dexterous manipulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1745– 1755, 2025.

[30] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

[31] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

[32] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

[33] Yao Mu, Junting Chen, Qing-Long Zhang, Shoufa Chen, Qiaojun Yu, Chongjian Ge, Runjian Chen, Zhixuan Liang, Mengkang Hu, Chaofan Tao, et al. Robocodex: Multimodal code generation for robotic behavior synthesis. In International Conference on Machine Learning, pages 36434–36454. PMLR, 2024.

[34] Yao Mu, Tianxing Chen, Zanxin Chen, Shijia Peng, Zhiqian Lan, Zeyu Gao, Zhixuan Liang, Qiaojun Yu, Yude Zou, Mingkun Xu, et al. Robotwin: Dual-arm robot benchmark with generative digital twins. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025.

[35] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems (RSS), 2024.

[36] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[37] Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE international conference on robotics and automation (ICRA), pages 3803–3810. IEEE, 2018.

[38] Aravind Rajeswaran, Sarvjeet Ghotra, Balaraman Ravindran, and Sergey Levine. Epopt: Learning robust neural network policies using model ensembles. In International Conference on Learning Representations, 2017.

[39] Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. Codebleu: a method for automatic evaluation of code synthesis. arXiv preprint arXiv:2009.10297, 2020.

[40] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[41] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 23–30. IEEE, 2017.

[42] Chenxi Wang, Hongjie Fang, Hao-Shu Fang, and Cewu Lu. Rise: 3d perception makes real-world robot imitation simple and effective. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2870–2877. IEEE, 2024.

[43] Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation, 2023.

[44] Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. Dexvla: Visionlanguage model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025.

[45] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, and Jian Tang. Tinyvla: Toward fast, data-efficient visionlanguage-action models for robotic manipulation. IEEE Robotics and Automation Letters, 10(4):3988–3995, 2025.

[46] Wu Wen, Xiaobo Xue, Ya Li, Peng Gu, and Jianfeng Xu. Code similarity detection using ast and textual information. International Journal of Performability Engineering, 15(10):2683, 2019.

[47] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

[48] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, et al. Sapien: A simulated part-based interactive environment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11097–11107, 2020.

[49] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Se June Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. In CoRL 2024 Workshop on Whole-body Control and Bimanual Manipulation: Applications in Humanoids and Beyond.

[50] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pages 1094–1100. PMLR, 2020.

[51] Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy. arXiv e-prints, pages arXiv–2403, 2024.

[52] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Soroush Nasiriany, and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for robot learning. arXiv preprint arXiv:2009.12293, 2020.

## A Contributions

<table><tr><td>Project Leaders</td><td rowspan="3">RoboTwin-OD Baijun Chen, Qiangyu Chen, Kailun Su, Xuan- bing Xie, Zanxin Chen</td></tr><tr><td>Tianxing Chen, Yao Mu, Zhixuan Liang</td></tr><tr><td>Roadmap &amp; Methodology Yao Mu, Tianxing Chen, Ping Luo, Yusen Qin, Policies Training &amp; Evaluation</td></tr><tr><td>Xiaokang Yang, Kaixuan Wang</td><td rowspan="3">Tianxing Chen, Zijian Cai, Tian Nian, Huan-ang Gao, Tianling Xu</td></tr><tr><td>Data Generator &amp; Benchmark</td></tr><tr><td>Tianxing Chen, Zanxin Chen, Baijun Chen, Qi- Real-World Deployment wei Liang, Zixuan Li, Xianliang Lin</td></tr><tr><td>CodeGen Agent</td><td>Tianxing Chen, Tian Nian, Weiliang Deng Domain Randomization</td></tr><tr><td></td><td></td></tr><tr><td>Chen, Mengkang Hu</td><td>Yibin Liu, Zanxin Chen, Yiheng Ge, Tianxing Baijun Chen, Yubin Guo, Qiwei Liang, Zhenyu Gu, Guodong Liu, Zanxin Chen, Tianxing Chen</td></tr></table>

## B Benchmarking RoboTwin 2.0 Against Existing Datasets

We compare RoboTwin 2.0 against existing benchmarks and datasets across several key dimensions, including the number of supported tasks, the presence of domain randomization, support for automatic data generation, and compatibility with vision-language-action (VLA) model training and evaluation. The comparison is summarized in Table 6.

Table 6: Comparison of RoboTwin 2.0 with previous manipulation benchmarks and datasets.
<table><tr><td>Benchmark &amp; Dataset</td><td>#Tasks</td><td>Domain Randomization</td><td>Auto Data Generation</td><td>VLA Model Train &amp; Eval</td></tr><tr><td>Meta-world [50]</td><td>50</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Robosuite [52]</td><td>9</td><td>X</td><td>X</td><td>X</td></tr><tr><td>RoboCasa [50]</td><td>25</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Maniskill2 [16]</td><td>20</td><td>X</td><td>√</td><td>X</td></tr><tr><td>AutoBio [24]</td><td>16</td><td>X</td><td>√</td><td>√</td></tr><tr><td>RoboTwin 1.0 [34]</td><td>14</td><td>X</td><td>√</td><td>√</td></tr><tr><td>RoboTwin 2.0 (ours)</td><td>50</td><td>√</td><td>√</td><td>√</td></tr></table>

## C Domain Randomization Setting

Domain randomization in all experiments includes cluttered scenes, random lighting, table height variation (up to 3 cm), unseen language instructions and randomized background textures.

## D Policies Training Details

RDT in experiment 4.3 was pretrained for 100,000 steps with a batch size of 16 per GPU on 8 GPUs, and all single-task fine-tuning was conducted for 10,000 steps with a batch size of 16 per GPU on 4 GPUs.

Pi0 in experiment 4.3 was pretrained for 100,000 steps with a batch size of 32, and all fine-tuning was performed for 30,000 steps using the same batch size.

ACT was trained under a unified setup with a chunk size of 50, batch size of 8, and single-GPU training for 6,000 epochs. During deployment, we applied temporal\_agg for temporal aggregation to improve execution stability.

DP was trained for 600 epochs with a batch size of 128 and a planning horizon of 8.

DP3 was trained for 3,000 epochs with a batch size of 256, using a planning horizon of 8 and a point cloud resolution of 1,024, with precise segmentation of the background and tabletop.

## E Support for Flexible Embodiment Combinations

Our object-centric, embodiment-agnostic data generation framework enables seamless deployment across a wide range of dual-arm robotic systems. The pipeline supports flexible embodiment configu rations, allowing arbitrary combinations of heterogeneous manipulators and relative arm placements. This design ensures compatibility with diverse hardware setups and facilitates extensibility to future robotic platforms.

![](images/bac96ba245ae048711b1eb248e4669473e86f930e07d46709974259e3cad44d0.jpg)  
Figure 11: Heterogeneous Dual-Arm Control via Object-Centric Manipulation.

To execute high-success-rate manipulation trajectories across different embodiments (see Section 2.3), we integrate Curobo, a high-performance, GPU-accelerated motion planner that enables efficient and reliable planning under varied kinematic constraints.

Currently, our framework supports five robotic arms—Franka, Piper, UR5, ARX-X5, and Aloha-AgileX—along with multiple gripper types, including the Panda gripper and WSG gripper. As shown in Fig. 11, we demonstrate successful task executions across a variety of dual-arm pairings, highlighting RoboTwin 2.0’s ability to scale to heterogeneous robot configurations and its readiness for future real-world deployment.

## F Improvements of RoboTwin 2.0 over RoboTwin 1.0 Policy Codebase

<table><tr><td>Metric</td><td>RoboTwin 1.0</td><td>RoboTwin 2.0</td></tr><tr><td>Prompt Token Length ↓</td><td>5901.0</td><td>4719.1</td></tr><tr><td>Code Token Length ↓</td><td>1236.6</td><td>569.4</td></tr><tr><td>Parallelism Control ↑</td><td>x</td><td>√</td></tr><tr><td>AST Similarity [46] ↑</td><td>23.72%</td><td>44.78%</td></tr><tr><td>CodeBLEU Similarity [39] ↑</td><td>17.18%</td><td>18.53%</td></tr><tr><td>CodeBERT Similarity [13] ↑</td><td>97.72%</td><td>98.80%</td></tr><tr><td>Unixcoder Similarity [17] ↑</td><td>76.24%</td><td>82.21%</td></tr><tr><td>Avg. VLM Token Cost (per observation)</td><td></td><td>6894</td></tr></table>

Table 7: Code Generation Efficiency and Quality Comparison. Evaluation of prompt and generated code characteristics, along with code similarity metrics (AST Structural Similarity, CodeBERT, Unixcoder cosine similarity) against expert-written code, for RoboTwin 1.0 and RoboTwin 2.0 in zero-shot generation. The VLM observer cost is also reported for RoboTwin 2.0.

We first quantify the architectural impact of RoboTwin 2.0 in a one-shot generation without code repair and iterative refinement. Table 7 shows that RoboTwin 2.0 yields significantly shorter programs (569.4 vs. 1236.6 tokens), with reduced prompt length and higher structural similarity to humanwritten code. Crucially, it enables dual-arm parallelism via a unified API abstraction, which is absent in RoboTwin 1.0.

These improvements stem from the structured prompting and geometric API modularization designed into RoboTwin 2.0. Higher AST similarity (+21.06%), CodeBERT similarity (+1.08%), and Unixcoder alignment (+5.97%) indicate that RoboTwin 2.0 not only reduces code size but also improves semantic clarity and functional alignment.

In addition, RoboTwin 2.0 integrates a VLM observer, a plug-and-play module triggered only when execution fails. To quantify its overhead, we estimated VLM usage via the Kimi API (assuming each image = 1,024 tokens) over three representative tasks: the average cost was 6,295 input tokens and 599 output tokens, totaling 6,894 tokens. While this introduces moderate overhead, the VLM enables

RoboTwin 2.0 to catch and correct errors invisible to execution logging, significantly enhancing robustness and overall task success. Importantly, the observer remains optional and can be disabled when prioritizing token efficiency.

## G Experimental Details and Metric Definitions for Code Generation

We use the DeepSeek-V3 model for program synthesis and the moonshot-v1-32k-vision-preview model for multimodal error localization and verification. These models were selected for their strong performance in language reasoning and visual understanding while maintaining efficiency suitable for large-scale iterative refinement. The success rate of the i-th program is computed as $R _ { i } \ =$ $\textstyle { \frac { 1 } { M } } \sum _ { j = 1 } ^ { \bar { M } } s _ { i , j } ,$ , and the final success rate for a given task under a specific system variant is then defined as $\begin{array} { r } { R _ { \mathrm { t a s k } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R _ { i } } \end{array}$ . For detailed usage instructions, refer to https://robotwin-platform. github.io/doc/usage/expert-code-gen.html.

## G.1 Metric Definitions

We report the following metrics across all tasks:

ASR (Average Success Rate) is the average of R\_task across all 10 tasks. It reflects overall task performance across all generated programs.

Top5-ASR is the mean success rate computed using only the top 5 highest-performing programs per task. This metric estimates system potential under a best-of-selection strategy.

CR-Iter indicates the average number of feedback iterations required per task before reaching a success rate above 50% or exhausting the iteration budget.

Token denotes the average number of tokens of policy code generated by the language model per task. It serves as a proxy for computational cost and LLM inference budget.

These metrics jointly evaluate both the reliability and efficiency of the expert data generation pipeline under varying conditions of feedback, model capability, and refinement strategy.

## G.2 Task-Specific Performance Comparison on Code Generation

We compare the code generation success rates of RoboTwin 2.0 and RoboTwin 1.0 across all tasks. As shown, RoboTwin 2.0 consistently matches or outperforms the baseline on the majority of tasks, demonstrating the effectiveness of our multimodal feedback and refinement pipeline.

<table><tr><td>Task</td><td>R1.0 Vanilla</td><td>R1.0 + FB</td><td>R1.0 + MM FB</td><td>R2.0 Vanilla</td><td>R2.0 + FB</td><td>R2.0 + MMFB</td></tr><tr><td>beat_block_hammer</td><td>16%</td><td>48%</td><td>56%</td><td>23%</td><td>34%</td><td>53%</td></tr><tr><td>handover_block</td><td>2%</td><td>41%</td><td>45%</td><td>17%</td><td>50%</td><td>27%</td></tr><tr><td>pick_diverse_bottles</td><td>65%</td><td>65%</td><td>64%</td><td>60%</td><td>60%</td><td>62%</td></tr><tr><td>pick_dual_bottles</td><td>99%</td><td>99%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>place_container_plate</td><td>66%</td><td>79%</td><td>91%</td><td>84%</td><td>84%</td><td>82%</td></tr><tr><td>place_dual_shoes</td><td>19%</td><td>22%</td><td>25%</td><td>0%</td><td>2%</td><td>22%</td></tr><tr><td>place_empty_cup</td><td>90%</td><td>90%</td><td>100%</td><td>61%</td><td>61%</td><td>85%</td></tr><tr><td>place_shoe</td><td>72%</td><td>90%</td><td>90%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>stack_blocks_three</td><td>1%</td><td>2%</td><td>4%</td><td>76%</td><td>76%</td><td>82%</td></tr><tr><td>stack_blocks_two</td><td>44%</td><td>68%</td><td>64%</td><td>100%</td><td>100%</td><td>100%</td></tr></table>

Table 8: Task-Specific Performance Comparison between RoboTwin 2.0 and RoboTwin 1.0. R1.0/R2.0: RoboTwin 1.0 / 2.0. Bold numbers indicate the best result for each task.

## G.3 Per-task Success Rates of Code Generation

We report the success rates of all tasks in Tab. 9.

Table 9: Per-task success rates of our proposed R2.0 + MM FB algorithm on all RoboTwin 2.0-supported tasks.
<table><tr><td>Task</td><td>Rate</td><td>Task</td><td>Rate</td><td>Task</td><td>Rate</td><td>Task</td><td>Rate</td></tr><tr><td>Adjust Bottle</td><td>100%</td><td>Beat Block Hammer</td><td>53%</td><td>Blocks Ranking Rgb</td><td>80%</td><td>Blocks Ranking Size</td><td>80%</td></tr><tr><td>Click Alarmclock</td><td>0%</td><td>Click Bell</td><td>10%</td><td>Dump Bin Bigbin</td><td>0%</td><td>Grab Roller</td><td>74%</td></tr><tr><td>Handover Block</td><td>27%</td><td>Handover Mic</td><td>0%</td><td>Hanging Mug</td><td>0%</td><td>Lift Pot</td><td>40%</td></tr><tr><td>Move Can Pot</td><td>30%</td><td>Move Pillbottle Pad</td><td>50%</td><td>Move Playingcard Away</td><td>90%</td><td>Move Stapler Pad</td><td>100%</td></tr><tr><td>Open Laptop</td><td>0%</td><td>Open Microwave</td><td>0%</td><td>Pick Diverse Bottles</td><td>62%</td><td>Pick Dual Bottles</td><td>100%</td></tr><tr><td>Place A2B Left</td><td>50%</td><td>Place A2B Right</td><td>60%</td><td>Place Bread Basket</td><td>0%</td><td>Place Bread Skillet</td><td>0%</td></tr><tr><td>Place Can Basket</td><td>0%</td><td>Place Cans Plasticbox</td><td>100%</td><td>Place Container Plate</td><td>82%</td><td>Place Dual Shoes</td><td>22%</td></tr><tr><td>Place Empty Cup</td><td>85%</td><td>Place Fan</td><td>70%</td><td>Place Burger Fries</td><td>100%</td><td>Place Mouse Pad</td><td>100%</td></tr><tr><td>Place Object Basket</td><td>0%</td><td>Place Object Scale</td><td>80%</td><td>Place Object Stand</td><td>90%</td><td>Place Phone Stand</td><td>0%</td></tr><tr><td>Place Shoe</td><td>100%</td><td>Press Stapler</td><td>0%</td><td>Put Bottles Dustbin</td><td>0%</td><td>Put Object Cabinet</td><td>0%</td></tr><tr><td>Rotate Qrcode</td><td>80%</td><td>Scan Object</td><td>0%</td><td>Shake Bottle</td><td>0%</td><td>Shake Bottle Horizontally</td><td>0%</td></tr><tr><td>Stack Blocks Three</td><td>82%</td><td>Stack Blocks Two</td><td>100%</td><td>Stack Bowls Three</td><td>20%</td><td>Stack Bowls Two</td><td>30%</td></tr><tr><td>Stamp Seal</td><td>20%</td><td>Turn Switch</td><td>0%</td><td>Avg Success Rate</td><td>43.34%</td><td></td><td></td></tr></table>

## G.4 Multimodal Observation and Error Localization

To further investigate the capability of the VLM observer, we manually curated a dataset of 130 execution sequences, including 101 failed trials and 29 successful trials. Each sequence consists of the natural language task instruction, a series of visual observations, and policy code. This dataset enables us to evaluate both binary error detection and fine-grained error localization.

Error Detection. The VLM observer was first tasked with evaluating whether a robotic execution successfully completed the instructed task. The confusion matrix is as follows: TP = 16, FP = 61, TN = 40, FN = 13. The derived performance metrics are: Accuracy = 0.431, Precision = 0.208, Recall = 0.552, and F1-score = 0.302. These results suggest that while the VLM observer is reasonably sensitive in identifying execution failures (recall), its tendency to over-predict errors leads to a high false positive rate.

Error Localization. For failed executions, we further tested whether the VLM observer could correctly localize the program step or code snippet that led to failure. Among the 40 true negative samples (correctly identified failed trials), the observer successfully pinpointed the failure step in 12 cases, yielding a localization accuracy of 30%. This indicates that, while the observer shows promising diagnostic ability, there remains significant room for improvement in reliable error attribution.

Case Study. We illustrate one successful case of VLM observer reasoning. Given a task instruction and five visual observations, the observer produced a step-by-step analysis:

Step 1: The left arm successfully picks up the mug from the left side.

Step 2: The left arm successfully places the mug at the middle position.

Step 3: The right arm successfully picks up the mug from the middle position.

Step 4: The right arm attempts to hang the mug on the rack but fails.

Step 5: The right arm is in the process of moving away after attempting to hang the mug.

The overall task was not successfully completed. The failure occurred in Step 4, where the right arm attempted to hang the mug on the rack but did not succeed. The mug is not properly positioned on the rack. The error message in the code indicates an AttributeError: ’list’ object has no attribute ’p’. This error occurs in the line where the target\_point is being set to middle\_target\_pose.p. The issue is likely due to the fact that middle\_target\_pose is a list, and lists do not have an attribute ’p’. The code should be modified to correctly access the position data from the middle\_target\_pose list.

To fix the error, the code should be updated to access the position data correctly. For example, if middle\_target\_pose is a list containing the position and orientation, the position data should be accessed using middle\_target\_pose[0] instead of middle\_target\_pose.p. Additionally, the failure in Step 4 could be due to the incorrect position or orientation of the mug when attempting to hang it on the rack. The code should be reviewed to ensure that the mug is positioned and oriented correctly before executing the hanging motion.

Failure Modes. Despite these successes, we also observed cases where the VLM observer fails. Misclassifications primarily occur in failed executions: the observer may hallucinate correct outcomes or overlook subtle orientation errors. For instance, in a stacking task where boxes are misaligned, the observer sometimes marks the execution as successful if all boxes are placed on top of each other. Similarly, in tasks requiring fine-grained pose accuracy (e.g., placing shoes in the correct orientation), small but critical errors may be ignored. Furthermore, failures stemming from invisible factors—such as incorrect grasp axis parameters—remain challenging for purely vision-based observers to diagnose.

## G.5 LLM-Generated Code and Human-Written Code Case Study

LLM-Generated Code (gpt\_place\_shoe)

```python
class gpt_place_shoe(place_shoe):
def play_once(self):
# Initial observation
self.save_camera_images(task_name="place_shoe", step_name="
step1_initial_scene_state", generate_num_id="generate_num_0")
# Get the shoe’s position to determine which arm to use
shoe_pose = self.shoe.get_pose()
shoe_position = shoe_pose.p
arm_tag = ArmTag("left" if shoe_position[0] < 0 else "right")
# Grasp the shoe
self.move(self.grasp_actor(actor=self.shoe, arm_tag=arm_tag, pre_grasp_dis
=0.1, grasp_dis=0))
self.save_camera_images(task_name="place_shoe", step_name="
step2_shoe_grasped", generate_num_id="generate_num_0")
# Lift the shoe up to avoid collision
self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis=’
world’))
self.save_camera_images(task_name="place_shoe", step_name="step3_shoe_lifted
", generate_num_id="generate_num_0")
# Get target pose from the top functional point of the target block
target_pose = self.target_block.get_functional_point(1, "pose")
# Place the shoe on the target block with head towards left side
self.move(self.place_actor(actor=self.shoe, arm_tag=arm_tag, target_pose=
target_pose, functional_point_id=0, pre_dis=0.1, dis=0.02, is_open=True,
constrain="align", pre_dis_axis=’fp’))
self.save_camera_images(task_name="place_shoe", step_name="step4_shoe_placed
", generate_num_id="generate_num_0")
# Lift the gripper up after placing
self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis=’
world’))
# Return arm to origin
self.move(self.back_to_origin(arm_tag=arm_tag))
# Final observation
self.move(self.save_camera_images(task_name="place_shoe", step_name="
step5_final_scene_state", generate_num_id="generate_num_0"))
```  
Listing 1: LLM-Generated Code for gpt\_place\_shoe Task

Human-Written Code (place\_shoe)

```python
class place_shoe(base_task):
def play_once(self):
# Get the shoe’s position to determine which arm to use
shoe_pose = self.shoe.get_pose().p
```

```python
arm_tag = ArmTag("left" if shoe_pose[0] < 0 else "right")
# Grasp the shoe with specified pre-grasp distance and gripper position
self.move(self.grasp_actor(self.shoe, arm_tag=arm_tag, pre_grasp_dis=0.1,
gripper_pos=0))
# Lift the shoe up by 0.07 meters in z-direction
self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
# Get target_block’s functional point as target pose
target_pose = self.target_block.get_functional_point(0)
# Place the shoe on the target_block with alignment constraint and specified
pre-placement distance
self.move(self.place_actor(self.shoe, arm_tag=arm_tag, target_pose=
target_pose, functional_point_id=0, pre_dis=0.12, constrain="align"))
# Open the gripper to release the shoe
self.move(self.open_gripper(arm_tag=arm_tag))
```  
Listing 2: Human-Written Code for place\_shoe Task

The LLM generated code tends to be more verbose, explicitly logging intermediate visual states and detailing parameters (e.g., pre\_dis\_axis=’fp’, is\_open=True), while human-written scripts are more minimal, omitting intermediate steps and favoring compact execution. Despite functional similarity, the structural differences illustrate that MLLM-generated programs are not only executable but emphasize step-by-step clarity, contributing to more robust feedback and repair.

## H Task Instruction and Object Description Example

Instruction Templates (task: ‘Pick Dual Bottles’)   
"Use {a} to place {A} left of {B}.", "Set {A} to the left of {B}.", "Move {A} beside {B} using   
{a}.", "Place {A} on {B}’s left side.", "Using {a}, position {A} next to {B}.", "Stick {A} on the   
left of {B}.", "Use {a} and place {A} on {B}’s left.", etc

Object Description   
# object id - ‘001\_bottle/0’:   
"red bottle", "red soda bottle", "plastic red bottle", "red bottle with yellow label", "red   
plastic bottle with smooth surface", "yellow text printed on red bottle surface", "red bottle with   
white label design and markings", "red bottle with white sealing and brown top screw cap", etc   
# object id - ‘039\_mug/0’:   
"black mug", "dark coffee mug", "sleek black mug", "black ceramic mug", "single-handle mug",   
"smooth black surface mug", "medium-sized drinking mug", "round mug with curved side", "dark mug   
with sturdy handle", "solid black mug with smooth finish", etc

## I Prompts for Generating Task Instructions and Object Descriptions

```markdown
# Task Instruction Template
- Goal: Generate task instruction template
- Requirements:
- Generate 60 items. Vary in sentence length and structure
- Use natural action verbs (grab, slide, place)
split
- 50 items for training
- 10 items for evaluation
## Schema Requirements
```

![](images/58aa7b52520210d0e9d87349eace8c20868876f2b9d1073bd572d1b9a9ac4324.jpg)  
Listing 3: Prompts for Generating Task Instructions and Object Descriptions

## J RoboTwin 2.0 Benchmark Setting Visualization

We visualize the simulation settings of the RoboTwin 2.0 benchmark in Fig. 12. All models are trained on 50 clean (non-randomized) demonstrations per task (blue). For evaluation, the Easy setting also uses clean environments, while the Hard setting employs domain-randomized environments (green).

![](images/e2eab5cebcd64dc7b5b803cbc97935f58db746ae8666ee2b3644d67c8b49dbc1.jpg)  
Figure 12: Heterogeneous Dual-Arm Control via Object-Centric Manipulation.

## K Full RoboTwin 2.0 Benchmark

We report the evaluation results of five policies on the RoboTwin 2.0 benchmark under the Easy and Hard settings. Note that these two settings differ only in evaluation conditions, while the training setup remains identical. A continuously maintained online leaderboard is available at https://robotwinplatform.github.io/leaderboard.

Table 10: RoboTwin 2.0 Simulation Benchmark (clean vs randomized, 50+ tasks).
<table><tr><td rowspan=1 colspan=12>Simulation Task          RDT          Pi0          ACT          DP          DP3Easy HardEasy HardEasy HardEasy HardEasy Hard</td></tr><tr><td rowspan=1 colspan=4>Adjust Bottle         81% 75%</td><td rowspan=1 colspan=8>90%  56%  97%  23%  97%  0%  99% 3%</td></tr><tr><td rowspan=1 colspan=4>Beat Block Hammer     77%  37%</td><td rowspan=1 colspan=2>43%  21%</td><td rowspan=1 colspan=4>56%  3%   42%  0%</td><td rowspan=1 colspan=2>72%  8%</td></tr><tr><td rowspan=1 colspan=4>Blocks Ranking RGB     3%   0%</td><td rowspan=1 colspan=2>19% 5%</td><td rowspan=1 colspan=4>1%   0%   0%   0%</td><td rowspan=1 colspan=2>3%   0%</td></tr><tr><td rowspan=1 colspan=3>Blocks Ranking Size     0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=2>0%   0%</td><td rowspan=1 colspan=2>1%   0%</td><td rowspan=1 colspan=2>2%   0%</td></tr><tr><td rowspan=1 colspan=3>Click Alarmclock       61%</td><td rowspan=1 colspan=1>12%</td><td rowspan=1 colspan=1>63%</td><td rowspan=1 colspan=1>11%</td><td rowspan=1 colspan=2>32%  4%</td><td rowspan=1 colspan=2>61%  5%</td><td rowspan=1 colspan=2>77%  14%</td></tr><tr><td rowspan=1 colspan=3>Click Bell          80%</td><td rowspan=1 colspan=1>9%</td><td rowspan=1 colspan=1>44%</td><td rowspan=1 colspan=1>3%</td><td rowspan=1 colspan=2>58%  3%</td><td rowspan=1 colspan=2>54%  0%</td><td rowspan=1 colspan=2>90%  0%</td></tr><tr><td rowspan=1 colspan=3>Dump Bin Bigbin       64%</td><td rowspan=1 colspan=1>32%</td><td rowspan=1 colspan=1>83%</td><td rowspan=1 colspan=1>24%</td><td rowspan=1 colspan=1>68%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>49%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=2>85%  53%</td></tr><tr><td rowspan=1 colspan=3>Grab Roller         74%</td><td rowspan=1 colspan=1>43%</td><td rowspan=1 colspan=1>96%</td><td rowspan=1 colspan=1>80%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>25%</td><td rowspan=1 colspan=1>98%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>98%</td><td rowspan=1 colspan=1>2%</td></tr><tr><td rowspan=1 colspan=3>Handover Block       45%</td><td rowspan=1 colspan=1>14%</td><td rowspan=1 colspan=1>45%</td><td rowspan=1 colspan=1>8%</td><td rowspan=1 colspan=1>42%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>10%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=3>Handover Mic        90%</td><td rowspan=1 colspan=1>31%</td><td rowspan=1 colspan=1>98%</td><td rowspan=1 colspan=1>13%</td><td rowspan=1 colspan=1>85%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>53%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>3%</td></tr><tr><td rowspan=1 colspan=3>Hanging Mug        23%</td><td rowspan=1 colspan=1>16%</td><td rowspan=1 colspan=1>11%</td><td rowspan=1 colspan=1>3%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>8%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>17%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=3>Lift Pot           72%</td><td rowspan=1 colspan=1>9%</td><td rowspan=1 colspan=1>84%</td><td rowspan=1 colspan=1>36%</td><td rowspan=1 colspan=1>88%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>39%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=3>Move Can Pot        25%</td><td rowspan=1 colspan=1>12%</td><td rowspan=1 colspan=1>58%</td><td rowspan=1 colspan=1>21%</td><td rowspan=1 colspan=1>22%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>39%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>6%</td></tr><tr><td rowspan=1 colspan=3>Move Pillbottle Pad     8%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>21%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>41%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=3>Move Playingcard Away   43%</td><td rowspan=1 colspan=1>11%</td><td rowspan=1 colspan=1>53%</td><td rowspan=1 colspan=1>22%</td><td rowspan=1 colspan=1>36%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>47%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>68%</td><td rowspan=1 colspan=1>3%</td></tr><tr><td rowspan=1 colspan=3>Move Stapler Pad      2%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>12%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=3>Open Laptop         59%</td><td rowspan=1 colspan=1>32%</td><td rowspan=1 colspan=1>85%</td><td rowspan=1 colspan=1>46%</td><td rowspan=1 colspan=1>56%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>49%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>82%</td><td rowspan=1 colspan=1>7%</td></tr><tr><td rowspan=1 colspan=3>Open Microwave       37%</td><td rowspan=1 colspan=1>20%</td><td rowspan=1 colspan=1>80%</td><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=1>86%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>61%</td><td rowspan=1 colspan=1>22%</td></tr><tr><td rowspan=1 colspan=3>Pick Diverse Bottles     2%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>27%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>52%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=3>Pick Dual Bottles      42%</td><td rowspan=1 colspan=1>13%</td><td rowspan=1 colspan=1>57%</td><td rowspan=1 colspan=1>12%</td><td rowspan=1 colspan=1>31%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>24%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=3>Place A2B Left        3%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>31%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>46%</td><td rowspan=1 colspan=1>2%</td></tr><tr><td rowspan=1 colspan=3>Place A2B Right       1%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>27%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>13%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>49%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=3>Place Bread Basket      10%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>17%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>14%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>26%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=3>Place Bread Skillet      5%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>23%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>11%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>19%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=2>Place Burger Fries</td><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=1>27%</td><td rowspan=1 colspan=1>80%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>49%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>18%</td></tr><tr><td rowspan=1 colspan=2>Place Can Basket</td><td rowspan=1 colspan=1>19%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>41%</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>18%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>67%</td><td rowspan=1 colspan=1>2%</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Place Cans Plasticbox</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=1>34%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>16%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>40%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>48%</td><td rowspan=1 colspan=1>3%</td></tr><tr><td rowspan=1 colspan=2>Place Container Plate</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>17%</td><td rowspan=1 colspan=1>88%</td><td rowspan=1 colspan=1>45%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>41%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>86%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=2>Place Dual Shoes</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>9%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>8%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>13%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=2>Place Empty Cup</td><td rowspan=1 colspan=1>56%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>37%</td><td rowspan=1 colspan=1>11%</td><td rowspan=1 colspan=1>61%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>37%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>65%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=2>Place Fan</td><td rowspan=1 colspan=1>12%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>20%</td><td rowspan=1 colspan=1>10%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>3%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>36%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=2>Place Mouse Pad</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=2>Place Object Basket</td><td rowspan=1 colspan=1>33%</td><td rowspan=1 colspan=1>17%</td><td rowspan=1 colspan=1>16%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>65%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=2>Place Object Scale</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>10%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=2>Place Object Stand</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=1>36%</td><td rowspan=1 colspan=1>11%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>22%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=2>Place Phone Stand</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>35%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>13%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>44%</td><td rowspan=1 colspan=1>2%</td></tr><tr><td rowspan=1 colspan=2>Place Shoe</td><td rowspan=1 colspan=1>35%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>28%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>23%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>58%</td><td rowspan=1 colspan=1>2%</td></tr><tr><td rowspan=1 colspan=2>Press Stapler</td><td rowspan=1 colspan=1>41%</td><td rowspan=1 colspan=1>24%</td><td rowspan=1 colspan=1>62%</td><td rowspan=1 colspan=1>29%</td><td rowspan=1 colspan=1>31%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>6%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>69%</td><td rowspan=1 colspan=1>3%</td></tr><tr><td rowspan=1 colspan=2>Put Bottles Dustbin</td><td rowspan=1 colspan=1>21%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>54%</td><td rowspan=1 colspan=1>13%</td><td rowspan=1 colspan=1>27%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>22%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=1>21%</td></tr><tr><td rowspan=1 colspan=2>Put Object Cabinet</td><td rowspan=1 colspan=1>33%</td><td rowspan=1 colspan=1>18%</td><td rowspan=1 colspan=1>68%</td><td rowspan=1 colspan=1>18%</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>42%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=2>Rotate QRcode</td><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=1>68%</td><td rowspan=1 colspan=1>15%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=2>0%   13%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>74%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=2>Scan Öbject</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>18%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=2>0%   9%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>31%</td><td rowspan=1 colspan=1>1%</td></tr><tr><td rowspan=1 colspan=2>Shake Bottle Horizontally</td><td rowspan=1 colspan=1>84%</td><td rowspan=1 colspan=1>51%</td><td rowspan=1 colspan=1>99%</td><td rowspan=1 colspan=1>51%</td><td rowspan=1 colspan=1>63%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>59%</td><td rowspan=1 colspan=1>18%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>25%</td></tr><tr><td rowspan=1 colspan=2>Shake Bottle</td><td rowspan=1 colspan=1>74%</td><td rowspan=1 colspan=1>45%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=1>74%</td><td rowspan=1 colspan=1>10%</td><td rowspan=1 colspan=1>65%</td><td rowspan=1 colspan=1>8%</td><td rowspan=1 colspan=1>98%</td><td rowspan=1 colspan=1>19%</td></tr><tr><td rowspan=1 colspan=2>Stack Blocks Three</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>17%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=2>Stack Blocks Two</td><td rowspan=1 colspan=1>21%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>42%</td><td rowspan=1 colspan=1>1%</td><td rowspan=1 colspan=1>25%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>7%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>24%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=2>Stack Bowls Three</td><td rowspan=1 colspan=1>51%</td><td rowspan=1 colspan=1>17%</td><td rowspan=1 colspan=1>66%</td><td rowspan=1 colspan=1>24%</td><td rowspan=1 colspan=1>48%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>63%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>57%</td><td rowspan=1 colspan=1>5%</td></tr><tr><td rowspan=1 colspan=3>Stack Bowls Two       76%</td><td rowspan=1 colspan=1>30%</td><td rowspan=1 colspan=1>91%</td><td rowspan=1 colspan=1>41%</td><td rowspan=1 colspan=1>82%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>61%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>83%</td><td rowspan=1 colspan=1>6%</td></tr><tr><td rowspan=1 colspan=3>Stamp Seal          1%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=1>3%</td><td rowspan=1 colspan=1>4%</td><td rowspan=1 colspan=1>2%</td><td rowspan=1 colspan=1>0%</td><td rowspan=1 colspan=2>2%  0%</td><td rowspan=1 colspan=1>18%</td><td rowspan=1 colspan=1>0%</td></tr><tr><td rowspan=1 colspan=4>Turn Switch         35%  15%</td><td rowspan=1 colspan=1>27%</td><td rowspan=1 colspan=1>23%</td><td rowspan=1 colspan=1>5%</td><td rowspan=1 colspan=3>2%  36%  1%</td><td rowspan=1 colspan=1>46%</td><td rowspan=1 colspan=1>8%</td></tr><tr><td rowspan=1 colspan=6>Average (%)         34.5  13.7  46.4 16.3</td><td rowspan=1 colspan=6>29.7  1.7   28.0  0.6  55.2  5.0</td></tr></table>

## L Success Rates of Different Embodiments on RoboTwin 2.0 Tasks

Table 11 reports the success rates of five robot embodiments across the 50 RoboTwin 2.0 tasks, using the same set of expert programs for data generation.

Table 11: Success Rates of Different Embodiments on RoboTwin 2.0 Tasks.
<table><tr><td></td><td colspan="5">RoboTwin1.0</td><td colspan="5">RoboTwin2.0</td></tr><tr><td>Task Name</td><td>Aloha</td><td>ARX</td><td>Franka</td><td>Piper</td><td>UR5</td><td>Aloha</td><td>ARX</td><td>Franka</td><td>Piper</td><td>UR5</td></tr><tr><td>Adjust Bottle</td><td>92%</td><td>88%</td><td>39%</td><td>0%</td><td>7%</td><td>93%</td><td>94%</td><td>34%</td><td>0%</td><td>12%</td></tr><tr><td>Beat Block Hammer</td><td>68%</td><td>86%</td><td>95%</td><td>0%</td><td>86%</td><td>64%</td><td>93%</td><td>98%</td><td>15%</td><td>90%</td></tr><tr><td>Blocks Ranking Rgb</td><td>92%</td><td>98%</td><td>96%</td><td>0%</td><td>82%</td><td>96%</td><td>97%</td><td>99%</td><td>13%</td><td>53%</td></tr><tr><td>Blocks Ranking Size</td><td>90%</td><td>95%</td><td>92%</td><td>0%</td><td>60%</td><td>96%</td><td>97%</td><td>89%</td><td>7%</td><td>38%</td></tr><tr><td>Click Alarmclock</td><td>89%</td><td>99%</td><td>100%</td><td>0%</td><td>95%</td><td>92%</td><td>99%</td><td>100%</td><td>0%</td><td>95%</td></tr><tr><td>Click Bell</td><td>100%</td><td>100%</td><td>100%</td><td>9%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>91%</td><td>100%</td></tr><tr><td>Dump Bin Bigbin</td><td>85%</td><td>98%</td><td>90%</td><td>0%</td><td>82%</td><td>84%</td><td>100%</td><td>84%</td><td>9%</td><td>80%</td></tr><tr><td>Grab Roller</td><td>95%</td><td>69%</td><td>99%</td><td>0%</td><td>80%</td><td>95%</td><td>69%</td><td>99%</td><td>7%</td><td>81%</td></tr><tr><td>Handover Block</td><td>1%</td><td>3%</td><td>0%</td><td>0%</td><td>4%</td><td>83%</td><td>81%</td><td>0%</td><td>44%</td><td>0%</td></tr><tr><td>Handover Mic</td><td>62%</td><td>80%</td><td>92%</td><td>28%</td><td>0%</td><td>87%</td><td>98%</td><td>84%</td><td>65%</td><td>14%</td></tr><tr><td>Hanging Mug</td><td>68%</td><td>76%</td><td>5%</td><td>0%</td><td>12%</td><td>63%</td><td>73%</td><td>11%</td><td>0%</td><td>11%</td></tr><tr><td>Lift Pot</td><td>27%</td><td>50%</td><td>24%</td><td>5%</td><td>40%</td><td>27%</td><td>50%</td><td>36%</td><td>31%</td><td>40%</td></tr><tr><td>Move Can Pot</td><td>18%</td><td>0%</td><td>37%</td><td>2%</td><td>4%</td><td>93%</td><td>65%</td><td>92%</td><td>96%</td><td>99%</td></tr><tr><td>Move Pillbottle Pad</td><td>30%</td><td>52%</td><td>15%</td><td>0%</td><td>35%</td><td>67%</td><td>90%</td><td>69%</td><td>47%</td><td>86%</td></tr><tr><td>Move Playingcard Away</td><td>93%</td><td>100%</td><td>100%</td><td>0%</td><td>87%</td><td>99%</td><td>100%</td><td>100%</td><td>63%</td><td>66%</td></tr><tr><td>Move Stapler Pad</td><td>94%</td><td>92%</td><td>88%</td><td>0%</td><td>95%</td><td>92%</td><td>96%</td><td>89%</td><td>13%</td><td>75%</td></tr><tr><td>Open Laptop</td><td>76%</td><td>91%</td><td>78%</td><td>14%</td><td>55%</td><td>82%</td><td>92%</td><td>77%</td><td>23%</td><td>51%</td></tr><tr><td>Open Microwave</td><td>65%</td><td>85%</td><td>75%</td><td>5%</td><td>33%</td><td>96%</td><td>80%</td><td>59%</td><td>2%</td><td>23%</td></tr><tr><td>Pick Diverse Bottles</td><td>11%</td><td>1%</td><td>0%</td><td>0%</td><td>0%</td><td>51%</td><td>2%</td><td>0%</td><td>27%</td><td>4%</td></tr><tr><td>Pick Dual Bottles</td><td>8%</td><td>3%</td><td>0%</td><td>0%</td><td>0%</td><td>92%</td><td>6%</td><td>0%</td><td>81%</td><td>7%</td></tr><tr><td>Place A2B Left</td><td>65%</td><td>75%</td><td>70%</td><td>0%</td><td>72%</td><td>80%</td><td>88%</td><td>64%</td><td>29%</td><td>76%</td></tr><tr><td>Place A2B Right</td><td>70%</td><td>68%</td><td>68%</td><td>0%</td><td>69%</td><td>81%</td><td>82%</td><td>64%</td><td>31%</td><td>66%</td></tr><tr><td>Place Bread Basket</td><td>91%</td><td>91%</td><td>69%</td><td>0%</td><td>78%</td><td>89%</td><td>88%</td><td>62%</td><td>1%</td><td>67%</td></tr><tr><td>Place Bread Skillet</td><td>31%</td><td>28%</td><td>42%</td><td>0%</td><td>42%</td><td>34%</td><td>26%</td><td>42%</td><td>0%</td><td>37%</td></tr><tr><td>Place Can Basket</td><td>47%</td><td>1%</td><td>38%</td><td>0%</td><td>11%</td><td>70%</td><td>28%</td><td>61%</td><td>0%</td><td>3%</td></tr><tr><td>Place Cans Plasticbox</td><td>96%</td><td>93%</td><td>98%</td><td>0%</td><td>11%</td><td>100%</td><td>96%</td><td>85%</td><td>0%</td><td>82%</td></tr><tr><td>Place Container Plate</td><td>86%</td><td>85%</td><td>83%</td><td>0%</td><td>82%</td><td>89%</td><td>86%</td><td>86%</td><td>37%</td><td>81%</td></tr><tr><td>Place Dual Shoes</td><td>73%</td><td>28%</td><td>36%</td><td>0%</td><td>40%</td><td>77%</td><td>31%</td><td>41%</td><td>1%</td><td>32%</td></tr><tr><td>Place Empty Cup</td><td>92%</td><td>100%</td><td>100%</td><td>0%</td><td>100%</td><td>92%</td><td>100%</td><td>100%</td><td>4%</td><td>100%</td></tr><tr><td>Place Fan</td><td>93%</td><td>96%</td><td>75%</td><td>0%</td><td>85%</td><td>95%</td><td>93%</td><td>83%</td><td>0%</td><td>65%</td></tr><tr><td>Place Burger Fries</td><td>96%</td><td>95%</td><td>85%</td><td>0%</td><td>78%</td><td>97%</td><td>98%</td><td>80%</td><td>36%</td><td>74%</td></tr><tr><td>Place Mouse Pad</td><td>100%</td><td>80%</td><td>99%</td><td>2%</td><td>96%</td><td>99%</td><td>89%</td><td>100%</td><td>23%</td><td>73%</td></tr><tr><td>Place Object Basket</td><td>68%</td><td>13%</td><td>68%</td><td>0%</td><td>30%</td><td>74%</td><td>14%</td><td>61%</td><td>0%</td><td>7%</td></tr><tr><td>Place Object Scale</td><td>77%</td><td>93%</td><td>94%</td><td>0%</td><td>87%</td><td>78%</td><td>92%</td><td>82%</td><td>2%</td><td>76%</td></tr><tr><td>Place Object Stand</td><td>90%</td><td>92%</td><td>81%</td><td>0%</td><td>90%</td><td>97%</td><td>99%</td><td>81%</td><td>9%</td><td>92%</td></tr><tr><td>Place Phone Stand</td><td>66%</td><td>78%</td><td>52%</td><td>22%</td><td>44% 97%</td><td>66% 84%</td><td>78% 85%</td><td>45% 74%</td><td>53% 7%</td><td>49% 91%</td></tr><tr><td>Place Shoe Press Stapler</td><td>87% 87%</td><td>85% 96%</td><td>70% 99%</td><td>0% 0%</td></table>