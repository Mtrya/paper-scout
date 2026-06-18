# Guava: An Effective and Universal Harness for Embodied Manipulation

Haowen Liu1,∗, Xirui Li1,∗, Shaoxiong Yao2, Peng Shi3, Tianyi Zhou4, Jia-Bin Huang1, Furong Huang1, Jiayuan Mao5,6

1University of Maryland College Park

2University of Illinois Urbana-Champaign

3University of Waterloo

4Mohamed bin Zayed University of Artificial Intelligence

5University of Pennsylvania

6Amazon FAR

∗Co-first Author

Language models trained on large-scale vision-language data have demonstrated strong potential for embodied agents. Harnessing models through embodied tools use offers a promising alternative to end-to-end vision-language-action systems by combining high-level reasoning with external modules for perception, planning, and control. However, it remains unclear what makes an effective harness for embodied manipulation, and to what extent such a harness can unlock embodied capabilities in a wide range of reasoning models. In this work, we present Guava, a harness framework for embodied tool use developed through systematic exploration of the design space of agent workflows, action spaces, and observation spaces. Our study identifies three key ingredients for effective embodied agents: iterative perception-reasoning-action loops, semantic action abstractions, and multimodal observations. To understand whether these design principles are universal even to small models, we develop an end-to-end training pipeline that distills embodied manipulation capabilities into a 4B open-source model using fewer than 2K trajectories collected entirely in simulation. Experimental results in both simulation and real-world environments show performance comparable to frontier proprietary models while exhibiting strong generalization to unseen objects, novel instructions, and long-horizon tasks. Results suggest that a well-designed harness can serve as a scalable, model-agnostic interface for embodied manipulation, enabling strong emergent embodied capabilities in compact open-source models with minimal training data.

Date: June 18, 2026 Author E-mails: hwl@umd.edu, xiruili@umd.edu Project Page: https://guava-harness.github.io

## 1 Introduction

Language models trained on large-scale vision-language data have shown strong potential for building generalizable embodied agents (Driess et al., 2023; Zitkovich et al., 2023). Their semantic understanding (Kojima et al., 2023), visual grounding (Liu et al., 2023), and visual reasoning abilities (Thawakar et al., 2025; Xu et al., 2025; Lu et al., 2024) make them feasible backbones for robotic manipulation (Fu et al., 2026). One line of work addresses this problem by finetuning vision-language models into vision-language-action (VLA) policies that directly generate robot actions from visual observations and language instructions (?Black et al., 2025b,a; Lee et al., 2025; Hancock et al., 2026). However, it usually requires large amounts of robot demonstration data, which is expensive to collect, embodiment-dependent, and difficult to scale to diverse objects, scenes, and long-horizon tasks encountered in the real world.

Recent progress in harness engineering (OpenAI, 2026b) has enabled foundation models to operate in increasingly complex domains, including personalized workflows (Steinberger and The OpenClaw Community, 2026), software development (Anthropic, 2026; OpenAI, 2026c), and scientific discovery (Karpathy, 2026).

![](images/007c92887f2a5af04fc73f0018a2e9d0487ec89af71641359eb7ab4977304bb6.jpg)  
Figure 1 Guava Overview. Guava defines structured interaction strategies between an embodied agent and its environment, encouraging the embodied reasoning and tool-calling for manipulation. Built on this harness, we train a small agent in simulation that can be directly deployed in the real world across diverse evaluation scenarios.

Harnessing models without extensive fine-tuning provides a promising direction for building agentic manipulation systems (Shi et al., 2025; Fu et al., 2026). Rather than requiring models to internalize all low-level perception, planning, and control capabilities (Sapkota et al., 2026), harness-based systems enable language models to invoke external modules for robot manipulation. This modular design is particularly well-suited for embodied manipulation: specialized low-level tools encapsulate robot skills, while the language model focuses on high-level reasoning, tool selection, and task decomposition. Maestro (Shi et al., 2025) and concurrent work Cap-X (Fu et al., 2026) represent early efforts in this direction. Despite recent progress, it remains unclear what makes an effective harness for embodied manipulation. Existing systems often rely on one-shot code generation, domain-specific pipelines coupled with powerful frontier models, making it difficult to achieve robust long-horizon behavior and failure recovery at low inference latency and cost. We therefore ask a fundamental question: what are the key ingredients of an effective and general harness for embodied agents?

To answer this question, we explore the design space of embodied agents and identify three principles that are critical for effective manipulation. First, iterative ReAct (Yao et al., 2023) loops are essential for adapting to execution outcomes and recovering from failures. Second, semantic action abstractions allow language models to focus on task decomposition and planning instead of low-level robot control. Third, rich multimodal observations provide the environmental context necessary for embodied reasoning. Guided by these findings, we develop Guava, a harness framework for embodied tool use that combines ReAct-style interaction, semantic manipulation tools, and multimodal observations into a unified agent architecture.

Building upon Guava, we further investigate whether an effective harness can serve as an universal interface for embodied manipulation across models, enabling even small open-source models to acquire strong embodied capabilities. To answer this question, we develop an end-to-end training pipeline that distills embodied tool-use behaviors into a 4B model using fewer than 2K trajectories collected entirely in simulation. Experimental results show that the resulting model achieves performance comparable to frontier proprietary models across a diverse suite of manipulation tasks, while exhibiting strong generalization to unseen objects, instructions, and long-horizon scenarios. Moreover, the learned model transfers zero-shot from simulation to real-world and demonstrates robust recovery behaviors under execution failures.

Together, these results suggest that a well-designed harness can act as a scalable and model-agnostic interface for embodied manipulation. By combining effective harness design with data-efficient post-training, Guava enables compact open-source models to acquire strong manipulation capabilities, robust failure recovery, and real-world transfer from fewer than 2K simulation trajectories.

## 2 Related Work

Foundation Models for Robotic Manipulation. Large vision-language and vision-language-action models have recently become a central paradigm for building generalizable robotic manipulation systems. One line of work turns multimodal foundation models into robot policies by adding action-generation modules and training on large-scale robot trajectories, often using diffusion or flow-matching objectives for continuous action prediction (?Kim et al., 2025; Black et al., 2025a). These models acquire broad manipulation skills, but their adaptation often depends on substantial robot data, and their internal action representations make explicit constraint checking and plan repair difficult. Another line of work uses VLMs as high-level reasoning modules and grounds their outputs through structured spatial representations, such as affordance map and keypoints relations (Huang et al., 2023; Fang et al., 2024; Huang et al., 2024; Yuan et al., 2024). These methods improve spatial grounding and compositionality, but often rely on carefully designed perception-action interfaces or hand-specified planning primitives.

Recent work further explores explicit action reasoning before execution. MolmoAct and MolmoAct2 reason about actions through interpretable spatial or language-conditioned representations (Lee et al., 2025; Fang et al., 2026), while ThinkAct compresses embodied reasoning into a visual plan latent that conditions a downstream action model (Huang et al., 2026). Hierarchical methods such as HAMSTER (Li et al., 2025) separate high-level reasoning from low-level control policies. These approaches improve the reasoning ability of robot policies, but the execution still typically depends on learned action heads. This makes it difficult to explicitly inspect a plan and iteratively repair it according to the task specification.

Harnessing Agents for Robotic Manipulation. Pioneering works such as Code-as-Policies (Liang et al., 2022) and ProgPrompt (Singh et al., 2023) show that language models can compose perception outputs, control primitives, and task-specific APIs into executable policies or situated task plans. These harnessing frameworks inherit the modularity of classical robotic systems: perception, planning, and control can be represented as callable tools, while the language model composes them into task-specific behavior. Recent work has extended this paradigm to multimodal and multi-agent settings. RoboCodeX (Mu et al., 2024) translates high-level instructions and scene understanding into executable robotic programs through multimodal code generation, while RoCo and related works (Mandi et al., 2024; Chen et al., 2025) integrate generated code with multi-robot coordination and motion planning. Recent work Maestro (Shi et al., 2025) and concurrent work Cap-X (Fu et al., 2026) further scale this paradigm by enabling robots to call diverse perception and control tools through writing programs. However, most existing harness frameworks rely on one-shot program generation and execution, providing limited opportunities for agents to react to execution outcomes or recover from failures. In contrast, we study how harness design can enable effective embodied manipulation through a ReAct-style workflow (Yao et al., 2023) that continuously interleaves perception, reasoning, and action execution. We further investigate whether such a harness can act as a general interface for transferring embodied capabilities to small open models.

## 3 Gauva: Harnessing VLM for Embodied Manipulation

Effective manipulation harnesses should provide robustness, grounded decision making, and recovery from execution failures under stochastic interaction with the physical world. In this paper, we start with evaluating different design choices through controlled ablations on six long-horizon manipulation tasks in Robosuite (Zhu et al., 2020) based on frontier models.

## 3.1 Designing Effective Harness

Robot manipulation requires continual grounding under stochastic execution: grasps may fail, objects can shift unexpectedly, and the environment often deviates from the model’s initial prediction. We find that effective and robust harnesses share three key properties. First, iterative workflows such as ReAct (Yao et al., 2023) substantially improve robustness over single-turn planning by enabling the model to re-plan after execution failures and incorporate updated observations throughout task execution. Rather than predicting an entire trajectory from a single observation, the VLM operates in a closed-loop reasoning process that supports recovery from grasp failures and state deviations. Second, semantic-level action spaces reduce the low-level geometric and physical reasoning burden (Tong et al., 2024; Guan et al., 2024) placed on the VLM. Instead of directly producing joint-space controls, the model issues task and object-oriented manipulation skills while motion planning is delegated to lower-level controllers. The abstraction allows the VLM to focus on semantic task decomposition rather than execution feasibility. We provide the full list of available tools in Table 1 and additional implementation details in Supplementary. These tools define actions with clear semantic meanings, with a combination of highly abstracted tools e.g., grasp() and lower level tools e.g., move() to cover for fine-grained actions when necessary. Third, multimodal observations provide complementary information for embodied reasoning. Visual observations capture spatial relationships and object configurations, while textual state representations provide compact symbolic descriptions of robot states and task progress. Combining both modalities improves grounding and reduces ambiguity during sequential decision making. Together, these design choices transform manipulation from an open-loop prediction problem into a grounded closed-loop interaction process, significantly improving the reliability of frontier VLMs in embodied environments. Figure 2 validates our design choices by demonstrating the performance of GPT-5.4 (OpenAI, 2026a) under different harness configurations across six long-horizon tasks implemented in Robosuite (Zhu et al., 2020) where multimodal setting in an iterative workflow demonstrate a consistent higher performance across tasks.

Table 1 List of embodied tools in Guava. Each tool has clear semantic meaning, some taking in semantic only input while others allow fine-grained actions via numerical parameters.
<table><tr><td>Tool</td><td>Functionality</td></tr><tr><td> grasp(object)</td><td> Pick up an object using perception-guided grasping.</td></tr><tr><td>align(object,direction,clearance)</td><td>Align the gripper to specified direction around a target object at clearance distance.</td></tr><tr><td> get_position(object)</td><td> Query the 3D position of an object.</td></tr><tr><td> get_position_size(object)</td><td>Query both object position and bounding box size.</td></tr><tr><td> move(x,y,z)</td><td> Move the end effector to a target position.</td></tr><tr><td> rotate(angle, axis)</td><td> Rotate the gripper by angle around axis.</td></tr><tr><td> close_gripper()</td><td> Close the gripper.</td></tr><tr><td>release()</td><td>Open gripper fully.</td></tr><tr><td>home_pose()</td><td> Return the robot to home configuration.</td></tr></table>

![](images/a907e052a9d667850ac72f94600314f301072282cc46748478966b4c1f0140fe.jpg)

![](images/570b44822b4aee277379a0446cf30b898389f2268a204b08d6b0f34724023961.jpg)

![](images/cb824f50b26ca968c63cdf1442be83318c58d67145e760c7bbe98ac12bda9dc5.jpg)  
Figure 2 Impact of harness design on embodied manipulation. We evaluate alternative workflow strategies (left), action-space abstractions (middle), and observation modalities (right). The semantic action space achieves higher performance than a low-level interface requiring explicit geometric reasoning over object poses, grasp configurations, and motion trajectories, demonstrating the benefit of semantic tool abstractions for embodied agents. The results further indicate that iterative agent workflows and multimodal observations are both essential for achieving strong performance and robust failure recovery in manipulation tasks.

## 3.2 Learning Efficient and Generalizable Agentic Embodied Reasoning

While Guava enables strong manipulation performance with frontier VLMs, directly deploying such models remains prohibitively expensive due to the latency and cost of repeated multimodal API calls during closedloop execution. We therefore ask whether the embodied capabilities enabled by Guava can be distilled into a compact open-source model. To answer this question, we develop a data-efficient training pipeline that transfers embodied tool-use behaviors from frontier VLMs using fewer than 2K trajectories collected entirely in simulation.

Guava as a data engine. A key challenge in transferring embodied capabilities to compact models is obtaining diverse and high-quality demonstrations. To this end, we develop a data generation engine that collects interaction trajectories from frontier VLMs operating under the Guava harness (Figure 3). The generated data covers a broad range of manipulation skills and embodied reasoning behaviors, including grasping, pushing, spatial reasoning, and task planning. In particular, we find that explicitly incorporating recovery trajectories demonstrate how to recover from execution failures can significantly improve robustness. Concretely, we augment successful demonstrations with recovery trajectories generated from perturbed execution states, exposing the model to failures and off-trajectory scenarios during training. After data cleaning and balancing, the resulting dataset provides diverse supervision for learning generalizable embodied policies. Remarkably, fewer than 2K trajectories collected entirely in simulation are sufficient to transfer embodied capabilities from frontier VLMs to a compact 4B model.

![](images/d4cb03a8c0684dfb31d7a770e829d482ff9c60cf70bef77e37b704fb3074ddbf.jpg)  
Figure 3 Data generation engine for policy distillation. (Left) Frontier VLMs interact with simulation environments through the Guava harness to generate diverse interaction trajectories, where scene randomization and targeted perturbations produce both successful and recovery trajectories. (Right) Distribution of task categories in collected dataset.

Training pipeline. We post-train the model with a two-stage pipeline. First, we perform supervised finetuning (SFT) on trajectories collected by the embodied data engine, including both successful and recovery trajectories. This enables the policy to learn manipulation skills as well as corrective behaviors for execution failures. We then apply Group Relative Policy Optimization (GRPO) with a sparse task-success reward, following recent RL post-training approaches for reasoning models (Shao et al., 2024; Zhou et al., 2025). RL training is applied to more challenging long-horizon tasks that require iterative planning, tool use, and adaptation to execution errors. See Supplementary for more details.

## 4 Experiments

To investigate whether Guava can act as a general interface for embodied manipulation, we distill embodied tool-use behaviors into a 4B-parameter VLM using fewer than 2K trajectories collected entirely in simulation, resulting in Guava-Agent-4B. We evaluate the resulting agent in both simulation and real-world environments to assess its ability to generalize across tasks, recover from failures, and transfer embodied capabilities from frontier models to compact open-source models.

## 4.1 Setup

We instantiate Guava with a 4B-parameter VLM, Qwen3.5-4B (Qwen Team, 2026), and train it using the two-stage optimization pipeline while freezing the vision encoder and aligner. All training is conducted on 8 NVIDIA H100 80GB GPUs using bfloat16 precision. See Supplementary for detailed hyperparameters and implementation details. Our model is evaluated in both simulation and the real world. For simulation, we use Robosuite (Zhu et al., 2020); for real-world experiments, we deploy on a Franka Research 3 robot arm (Franka Robotics GmbH, 2026) with a calibrated Intel RealSense D435 RGB-D camera. The evaluation suite covers diverse object geometries, spatial arrangements, and manipulation strategies, including non-prehensile and long-horizon tasks. We divide the tasks into four categories: In-distribution (ID) tasks, which share task types with training but differ in scene configurations; Out-of-distribution (OOD) object tasks, which involve unseen objects or object names; OOD prompt tasks, which require following novel language instructions or task specifications; and OOD long-horizon tasks, which require composing multiple manipulation skills over extended interaction sequences. We report task success rate as the primary evaluation metric. A trial is considered successful if the agent completes the prompt-specified task within the execution horizon. We compare against three representative baselines. Qwen3.5-4B evaluates the base foundation model operating under the Guava harness without embodied post-training. GPT-5.4 serves as a strong proprietary VLM baseline equipped with the same observation space, tool set, and agent harness. Finally, CaP-Agent0 is a concurrent harness-based manipulation agent that performs one-shot code generation and execution. All methods are evaluated in the same environments. To ensure a fair comparison, all agentic methods are provided with the same observation inputs and tool APIs whenever possible; CaP-Agent0 uses its native execution interface.

Table 2 Quantitative comparison on simulation tasks. Values are success rates (%) evaluated over 15 episodes. Best results are marked bold.
<table><tr><td rowspan="2">Task Name</td><td colspan="4">Success Rate (%)</td><td rowspan="2">Best</td></tr><tr><td>CaP-Agento (Fu et al., 2026)</td><td>Qwen3.5-4B (Qwen Team, 2026)</td><td>GPT-5.4 (OpenAI, 2026a）</td><td>Guava-Agent-4B</td></tr><tr><td colspan="6">ID</td></tr><tr><td> place can in box</td><td>86.7</td><td>20.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>arrange by size</td><td>46.7</td><td>0.0</td><td>40.0</td><td>66.7</td><td>66.7</td></tr><tr><td>remove cube from tray</td><td>73.3</td><td>40.0</td><td>86.7</td><td>100.0</td><td>100.0</td></tr><tr><td>push basket close drawer</td><td>20.0 67.7</td><td>0.0</td><td>26.7</td><td>60.0</td><td>60.0</td></tr><tr><td></td><td></td><td>20.0</td><td>53.3</td><td>86.7</td><td>86.7</td></tr><tr><td colspan="6">OOD Object</td></tr><tr><td> pick up carrot</td><td>100.0</td><td>66.7</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>tomato near potato</td><td>86.7</td><td>40.0</td><td>86.7</td><td>86.7</td><td>86.7</td></tr><tr><td>lemon in bin</td><td>100.0</td><td>20.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>push pot</td><td>20.0</td><td>33.3</td><td>26.7</td><td>73.3</td><td>73.3</td></tr><tr><td colspan="6">OOD Prompt</td></tr><tr><td> stack cube reverse order</td><td>60.0</td><td>20.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>arrange by size reverse direction</td><td>46.7</td><td>13.3</td><td>26.7</td><td>66.7</td><td>66.7</td></tr><tr><td colspan="6">OOD Long Horizon</td></tr><tr><td> separate food and utensils</td><td>86.7</td><td>0.0</td><td>86.7</td><td>93.3</td><td>93.3</td></tr><tr><td>set table</td><td>93.3</td><td>20.0</td><td>93.3</td><td>93.3</td><td>93.3</td></tr><tr><td> shell game</td><td>0.0</td><td>53.3</td><td>46.7</td><td>6.7</td><td>53.3</td></tr><tr><td>place all red objetsin basket</td><td>53.3</td><td>0.0</td><td>80.0</td><td>0.0</td><td>80.0</td></tr><tr><td> Overall </td><td>62.7</td><td>23.1</td><td>70.2</td><td>75.6</td><td>75.6</td></tr></table>

## 4.2 Small VLM Achieves High Performance with Guava

Our experiments demonstrate that Guava can effectively transfer embodied manipulation capabilities to compact open-source models. Using fewer than 2K simulation trajectories, Guava-Agent-4B achieves performance comparable to frontier proprietary systems in both simulation and real-world environments. Key findings are summarized below.

Finding 1: Guava-Agent-4B achieves the strongest overall performance across ID and OOD tasks. As shown in Table 2, Guava-Agent-4B achieves the highest overall success rate of 75.6%, outperforming GPT-5.4 (70.2%) and CaP-Agent0 (62.7%). On ID tasks, it consistently achieves the best performance, including 100% on place can in box and remove cube from tray. The model also generalizes well to unseen objects and prompts, achieving 100% on pick up carrot, lemon in bin, and stack cube reverse order, while improving push pot from 26.7% (GPT-5.4) to 73.3%. On long-horizon tasks, it achieves 93.3% on both separate food and utensils and set table. Failures are primarily concentrated on particularly challenging tasks such as shell game (6.7%) and place all red objects in basket (0.0%).

Finding 2: Guava-Agent-4B transfers effectively from simulation to the real world. As shown in Figure 4, Guava-Agent-4B achieves the highest overall success rates on both ID (86%) and OOD (92%) real-world tasks, outperforming other baselines. On ID tasks, our method achieves perfect success on pick up orange, place can in box, and remove cube from tray, while substantially improving performance on the more challenging push basket task (60% vs. 40% for GPT-5.4). On OOD tasks, Guava-Agent-4B maintains strong performance, achieving 100% success on move object away and 90% on set table. These results demonstrate that embodied tool-use policies learned from fewer than 2K simulation trajectories can transfer effectively to real-world manipulation without additional real-world fine-tuning.

![](images/b80f46312eda5620359bac875087f5195ec86a3371c17e5a1669657483c43d9e.jpg)

![](images/0d40efc15e8e356d01859560c61228d2f0541431cb20d1fedd11c8cc4ac766b5.jpg)

![](images/9aa74719669421fbdbed3c07df0fd5a1f0ac96b9ed11589a09d9a2836ee24f4e.jpg)  
Figure 4 Real-world performance. Guava-Agent-4B achieves comparable performance to SOTA proprietary model in zero-shot real world setting. Success rates are evaluated over 10 episodes.

“set table”  
![](images/eb835e6a8977b8ab5f1a274ba48d9168e99ce852fca33790c492c7b0c0915b85.jpg)  
“place all red objects in basket”

“separate the objects”  
![](images/0436219dd5f9c292e10e9142b6316fecec70772f7e8b3f654b7bda0412fb1ee2.jpg)  
“remove cube from tray”  
“sort by size”  
“push basket to the right”  
Figure 5 Example real-world rollouts. Our model identifies the task-relevant objects and selects appropriate manipulation tools across diverse tasks and configurations.

Finding 3: RL post-training substantially improves long-horizon reasoning and recovery behaviors. Figure 6 compares the SFT and RL versions of Guava-Agent-4B on two challenging long-horizon tasks. While the SFT policy struggles on both shell game 6.7%) and place all red objects in basket (0.0%), RL post-training improves performance to 60.0% and 93.3%, respectively. These tasks require the agent to execute extended action sequences, recover from intermediate failures, and reason over task progress. The large gains suggest that training on challenging long-horizon tasks with sparse success rewards effectively strengthens recovery behaviors and enables the policy to better handle off-trajectory states.

![](images/1545a2eb1ce9548667399119539b9100241c1cb4b1ecf815b286ccb0ce255093.jpg)  
Figure 6 RL improves performance significantly for long-horizon tasks.

## Finding 4: Closed-loop execution is critical for long-horizon

manipulation. Compared with one-shot planning approaches such as CaP-Agent0, Guava-Agent-4B continuously interleaves observation, reasoning, and action execution on all trajectories. This enables the agent to detect failures, revise plans, and recover from execution errors, resulting in substantially stronger performance on long-horizon tasks. We provide some reasoning examples in Figure 7.

![](images/ae2edb0758a82b1240e086d84e6ae7108dbf4020e44064a3f1012fd928045ebf.jpg)

![](images/f7fbe2d2c82576654400f23482af6a2105fb8dc76f52986bacf7db8288d8e1e1.jpg)  
Think: The orange is on the table near the front-left, and the gripper is centered above the back side of the table. I should first move above the orange with some clearance.

![](images/18de25fa60685e7e6bcdfa26e780734881ce9871094f3292b9b523943a878ab9.jpg)  
Think: the gripper is now directly above the orange with adequate clearance and fully open. The orange is small and isolated, so a standard grasp should work. After grasping, I can lift if successful.

![](images/05165ee4c33e8e5c832e285fae9d1b60d6c1e713965aad8e0019f3719086e84d.jpg)  
Tool: align(orange,top,medium)  
Think: The orange is grasped successfully, but the gripper is still down at table height, so there is risk of dragging. Best next action is a straight upward move from the current pose.  
Think: I grasped the orange and lifted it slightly off the table, which satisfies picking it up.  
Tool: grasp(orange)  
Tool: move(0.555, -0.107, 0.0)  
Task completed.

Figure 7 Example reasoning & tool-call trajectory. At each step, the agent produces grounded reasoning trace to call appropriate embodied tools.  
![](images/da9db0bf5b4d5d2091ad0b12c01210fb82e532402270c9e1f56d0756b035aa07.jpg)  
Think: The grasp missed: the gripper is fully closed on air. I should release() and try again

![](images/970716d6e5d51bb894097eafe7bbec150b1915583c617757d80f6579ade1472f.jpg)  
Think: The last motion was aborted, likely because I lowered too close to the box rim while holding a tall object. I need to move() before placing to avoid clipping the box walls

Figure 8 Example recovery actions under execution failures. During real-world execution, our model can identify errors, reason about their causes, and call appropriate recovery actions.

Finding 5: Guava-Agent-4B exhibits recovery behaviors beyond those seen during training. As shown in Figure 8, we observe successful recovery from previously unseen execution failures, including real-world control issues caused by joint limits and unreachable poses. The agent often generates plausible corrective actions, such as replanning target poses or returning to a home configuration before retrying, suggesting that recovery emerges from reasoning over execution feedback rather than memorizing predefined correction patterns.

## 5 Conclusion

We present Guava, a harness framework for embodied manipulation that identifies three key ingredients for effective embodied agents: iterative reasoning, semantic action abstractions, and multimodal observations. Using fewer than 2K simulation trajectories, we transfer embodied capabilities into a 4B open-source model that achieves strong generalization, robust recovery, and competitive real-world performance. These results suggest that effective harnesses can act as transferable interfaces for embodied manipulation, enabling compact open-source models to acquire strong embodied capabilities with minimal training data.

Limitations. Our method has several limitations. It cannot handle dexterous manipulation due to current choices of action primitives. Our system cannot directly correct tool-level errors, such as invalid grasp proposals or incorrect SAM3 segmentations. However, it can detect such failures and attempt recovery through multiple retries or alternative actions. The current setup also assumes a single-view image from a fixed camera, which can be limiting due to occlusion or perspective effects; future work may explore a multi-view setup with dynamic views, e.g., an additional wrist camera. We also plan to scale our current setup to a larger dataset, a larger model size, and a broader toolset to handle more diverse tasks and achieve better performance.

## References

Anthropic. Claude Code by Anthropic: An agentic coding system. https://www.anthropic.com/product/claude-code, 2026. Accessed: 2026-05-28.

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, brian ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization. In Joseph Lim, Shuran Song, and Hae-Won Park, editors, Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 17–40. PMLR, 27–30 Sep 2025a. https://proceedings.mlr.press/v305/black25a.html.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, Laura Smith, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A Vision-Language-Action Flow Model for General Robot Control. In Proceedings of Robotics: Science and Systems, LosAngeles, CA, USA, June 2025b. doi: 10.15607/RSS.2025.XXI.010.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14455–14465, June 2024.

Junting Chen, Checheng Yu, Xunzhe Zhou, Tianqi Xu, Yao Mu, Mengkang Hu, Wenqi Shao, Yikai Wang, Guohao Li, and Lin Shao. EMOS: Embodiment-aware heterogeneous multi-robot operating system with LLM agents. In The Thirteenth International Conference on Learning Representations, 2025. https://openreview.net/forum?id= Ey8KcabBpB.

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In arXiv preprint arXiv:2303.03378, 2023.

Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, Shanli Xing, Jaemin Cho, Jae Sung Park, Ainaz Eftekhar, Peter Sushko, Karen Farley, Angad Wadhwa, Cole Harrison, Winson Han, Ying-Chun Lee, Eli VanderBilt, Rose Hendrix, Suveen Ellawela, Lucas Ngoo, Joyce Chai, Zhongzheng Ren, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact2: Action reasoning models for real-world deployment, 2026. https://arxiv.org/abs/2605.02881.

Kuan Fang, Fangchen Liu, Pieter Abbeel, and Sergey Levine. Moka: Open-world robotic manipulation through mark-based visual prompting. Robotics: Science and Systems (RSS), 2024.

Franka Robotics GmbH. Franka Research 3. https://franka.de/franka-research-3, 2026. Accessed: 2026-05-28.

Max Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, and Linxi "Jim" Fan. Cap-x: A framework for benchmarking and improving coding agents for robot manipulation, 2026. https://arxiv.org/abs/2603.22435.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models, 2024. https://arxiv.org/abs/2310.14566.

Asher James Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, and Anirudha Majumdar. Actions as language: Fine-tuning VLMs into VLAs without catastrophic forgetting. In The Fourteenth International Conference on Learning Representations, 2026. https://openreview.net/forum?id=sFO9d6XSlf.

Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. Thinkact: Vision-languageaction reasoning via reinforced visual latent planning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. https://openreview.net/forum?id=72UR53jN7T.

Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. In 7th Annual Conference on Robot Learning, 2023. https://openreview.net/forum?id=9\_8LF30mOC.

Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. In 8th Annual Conference on Robot Learning, 2024. https://openreview.net/forum?id=9iG3SEbMnL.

Andrej Karpathy. autoresearch: AI agents running research on single-GPU nanochat training automatically. https: //github.com/karpathy/autoresearch, March 2026. Accessed: 2026-05-28.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. In Pulkit Agrawal, Oliver Kroemer, and Wolfram Burgard, editors, Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 2679–2713. PMLR, 06–09 Nov 2025. https://proceedings.mlr.press/v270/kim25c.html.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners, 2023. https://arxiv.org/abs/2205.11916.

Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.

Yi Li, Yuquan Deng, Jesse Zhang, Joel Jang, Marius Memmel, Caelan Reed Garrett, Fabio Ramos, Dieter Fox, Anqi Li, Abhishek Gupta, and Ankit Goyal. HAMSTER: Hierarchical action models for open-world robot manipulation. In The Thirteenth International Conference on Learning Representations, 2025. https://openreview.net/forum? id=h7aQxzKbq6.

Jacky Liang, Wenlong Huang, F. Xia, Peng Xu, Karol Hausman, Brian Ichter, Peter R. Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500, 2022. https://api.semanticscholar.org/CorpusID:252355542.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. https://openreview.net/forum?id=w0H2xGHlkw.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. https://openreview.net/ forum?id=KUNzEQMWU7.

Zhao Mandi, Shreeya Jain, and Shuran Song. Roco: Dialectic multi-robot collaboration with large language models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 286–299. IEEE, 2024.

ModelScope Community. Ms-swift: Scalable lightweight infrastructure for fine-tuning. https://github.com/modelscope/ ms-swift, 2025. GitHub repository, accessed 2026-06-04.

Yao Mu, Junting Chen, Qinglong Zhang, Shoufa Chen, Qiaojun Yu, Chongjian GE, Runjian Chen, Zhixuan Liang, Mengkang Hu, Chaofan Tao, Peize Sun, Haibao Yu, Chao Yang, Wenqi Shao, Wenhai Wang, Jifeng Dai, Yu Qiao, Mingyu Ding, and Ping Luo. Robocodex: multimodal code generation for robotic behavior synthesis. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

OpenAI. Introducing gpt-5.4. https://openai.com/index/introducing-gpt-5-4/, 2026a. Accessed: 2026-05-28.

OpenAI. Harness engineering: Leveraging Codex in an agent-first world, February 2026b. https://openai.com/index/ harness-engineering/.

OpenAI. Codex: AI coding partner from OpenAI. https://openai.com/codex/, 2026c. Accessed: 2026-05-28.

Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, 2026. Accessed: 2026-05-28.

Ranjan Sapkota, Yang Cao, Konstantinos I. Roumeliotis, and Manoj Karkee. Vision-language-action (vla) models: Concepts, progress, applications and challenges, 2026. https://arxiv.org/abs/2505.04769.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. https://arxiv.org/abs/2402.03300.

Junyao Shi, Rujia Yang, Kaitian Chao, Bingqing Selina Wan, Yifei Simon Shao, Jiahui Lei, Jianing Qian, Long Le, Pratik Chaudhari, Kostas Daniilidis, et al. Maestro: Orchestrating robotics modules with vision-language models for zero-shot generalist robots. In NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI, 2025.

Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. Progprompt: program generation for situated robot task planning using large language models. Auton. Robots, 47(8):999–1012, August 2023. ISSN 0929-5593. doi: 10.1007/s10514-023-10135-3. https://doi.org/10.1007/s10514-023-10135-3.

Peter Steinberger and The OpenClaw Community. OpenClaw: Your own personal ai assistant. https://github.com/ openclaw/openclaw, 2026. Accessed: 2026-05-28.

Omkar Thawakar, Dinura Dissanayake, Ketan Pravin More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Ilmuz Zaman Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, Hisham Cholakkal, Ivan Laptev, Mubarak Shah, Fahad Shahbaz Khan, and Salman Khan. LlamaV-o1: Rethinking step-by-step visual reasoning in LLMs. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 24290–24315, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1247. https://aclanthology.org/2025.findings-acl.1247/.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024. https://arxiv.org/abs/2406.16860.

Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2087–2098, October 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. https://arxiv.org/abs/2210.03629.

Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction in robotics. In 8th Annual Conference on Robot Learning, 2024. https://openreview.net/forum?id=GVX6jpZOhU.

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s "aha moment" in visual reasoning on a 2b non-sft model, 2025. https://arxiv.org/abs/2503.05132.

Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Soroush Nasiriany, Yifeng Zhu, and Kevin Lin. robosuite: A modular simulation framework and benchmark for robot learning. In arXiv preprint arXiv:2009.12293, 2020.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

## Contents

1 Introduction 1   
2 Related Work 2   
3 Gauva: Harnessing VLM for Embodied Manipulation 3   
3.1 Designing Effective Harness 3   
3.2 Learning Efficient and Generalizable Agentic Embodied Reasoning 4   
4 Experiments 5   
4.1 Setup 5   
4.2 Small VLM Achieves High Performance with Guava 6   
5 Conclusion 8   
A Dataset for Fine-tuning Guava-Agent-4B 14   
A.1 Construction of Trajectories 14   
A.2 Data Processing 14   
A.3 Recovery Behavior Generation 14   
A.4 Summary 15   
B Tools 15   
C Training Details 16   
C.1 Hyperparameters 16   
C.2 RL Training . . 16   
D Additional Results 16   
E Additional Discussions 17   
F Examples 18

## Appendix

## A Dataset for Fine-tuning Guava-Agent-4B

We construct the dataset by deploying our harness framework with GPT-5.4 OpenAI (2026a) in RoboSuite (Zhu et al., 2020). The tool interface exposes environment observations, action execution, and episode-level feedback through a standardized API, enabling the model to interact closed-looped in simulation.

## A.1 Construction of Trajectories

Starting from a collection of task prompts, GPT-5.4 is allowed to execute actions within RoboSuite and generate complete trajectories consisting of observations, tool calls, model reasoning traces, and environment interactions by the system prompt provided in Prompt A.1. This process yields an initial pool of candidate trajectories spanning diverse manipulation tasks. We randomize parameters such as pose, lighting, camera views to improve diversity and generalization.

System Prompt for Data Generation   
You are an intelligent robot arm controller. You will be shown an image of the scene and given a task to   
complete using the available tools.   
Interaction Loop   
In every response, think step-by-step inside <think></think> tags first, then call exactly one tool. When the   
task is fully complete or irrecoverable, output your thinking and end with Task complete or Task failed.   
Reasoning Guidelines   
Inside <think>, analyze current scene state - including object and gripper poses, progress of task - what has   
been done and what remains, result of last action - was it a success, failure, or unexpected outcome, and propose   
the best next action based on your reasoning...   
Tool Calling   
<tool\_call>{"name": "tool\_name", "arguments": {"param": value}}</tool\_call>   
Gripper State   
The gripper’s current position [x,y,z], rotation [roll,pitch,yaw], and gripper opening % are provided at every   
turn. Use these values directly in your reasoning when necessary.   
Tool Definition   
...

## A.2 Data Processing

To improve dataset quality, we perform several stages of filtering and curation. First, trajectories are automatically categorized according to their execution outcomes, and we only retain episodes where tasks are completed successfully. Second, we filter out trajectories involving errors such as invalid tool parameters and bad simulation initialization. Third, we conduct manual inspection on a subset of trajectories to identify and remove low-quality samples exhibiting unrelated conversational behavior, excessive self-reflection, off-task dialogue, or other artifacts that do not contribute to task completion.

To reduce dataset bias, we additionally de-duplicate highly similar trajectories and remove repeated interaction patterns arising from near-identical task prompts or execution histories. This curation process helps prevent over-representation of specific task instances and encourages greater behavioral diversity within the dataset.

## A.3 Recovery Behavior Generation

From the success trajectories, we also generate recovery data by manually adding error perturbations by sampling from a set of predefined common set of errors, including missed grasp, object dropping during transport, wrong alignment. We then continue the interactive data generation procedure starting from the perturbed state. We also additionally generate trajectories starting from a randomly sampled state in the original trajectory to reduce overfitting to starting condition. Only valid counterfactuals are added, e.g., a pushing task will never encounter a missing grasp. We apply the same data processing pipeline to the recovery data, filtering out unsuccessful and low-quality rollouts.

## A.4 Summary

After filtering, Guava-Agent-4B’s fine-tuning dataset contains 1,934 trajectories corresponding to 237 unique task prompts. Among them, 1,191 trajectories (62%) are successful executions, while 743 trajectories (38%) correspond to recovery trajectories.

## B Tools

These tools constitute the complete action space available to the language model.

• grasp(object): Grasps the specified object object. The implementation first segments the target object from RGB-D observations using SAM3 (Carion et al., 2025) and estimates a grasp pose. The API supports any learned 6-DoF grasp planner, or a simple baseline PCA-based top-down grasp for flexibility. The robot then approaches the grasp pose, closes the gripper, and returns either grasped when the gripper cannot fully close or closed if the gripper closes completely.

• align(object, position, clearance): Moves the gripper to a specified relative position around the target object. The parameter position ∈ {top, left, right, front, back} defines the approach direction, while clearance ∈ {small, medium, large} controls the standoff distance which can be mapped to user-defined values. This design keeps VLM reasoning simple while grounding execution in 3D geometry. To compute the clearance distance, we use object geometry estimated from the segmented point cloud.

• get\_position(object): Returns the estimated 3D position of object\_name in the robot base frame. The position is computed as the centroid of the segmented object point cloud after outlier removal.

• get\_position\_and\_size(object): Returns both the estimated object position and axis-aligned boundingbox dimensions, enabling the agent to reason about object size and spatial constraints.

• move(position): Moves the robot end-effector to the Cartesian position position=[x,y,z] via a positionbased controller.

• rotate(angle\_deg, axis): Rotates the gripper in place by angle\_deg degrees about the specified body-frame axis axis ∈ {x, y, z}.

• close\_gripper(): Closes the gripper.

• release(): Opens the gripper to release a grasped object. The gripper optionally performs a short retraction motion to avoid post-release collisions.

• home\_pose(): Moves the robot to a predefined home configuration and serves as a recovery action when the current pose is unsuitable for further manipulation.

High-Level vs. Low-Level Tool Definitions To further understand the impact of tool abstraction on agent performance, we compare our semantic action space against a low-level geometric action interface. In this alternative setup, the agent is provided with only four primitive tools:

• move(x, y, z, roll, pitch, yaw, width): a fused low-level control primitive that moves the end effector to an absolute Cartesian pose and gripper opening. In this case, the agent must explicitly reason about object poses, grasp configurations, end-effector orientations, and manipulation trajectories.

• get\_position(object): unchanged.

• get\_position\_and\_size(object): unchanged.

• home\_pose(): unchanged.

Compared to our semantic action space, this formulation requires the VLM agent to directly perform low-level geometric and physical reasoning. The model must determine precise grasp poses, end-effector orientations, object clearances, and motion sequences before issuing manipulation commands. In contrast, our semantic action space abstracts these details behind task-oriented manipulation skills, allowing motion planning and execution to be handled by lower-level controllers.

As shown in Figure 2, the low-level tool interface consistently results in worse performance than the highlevel, semantic action interface. This observation supports the design principle discussed in Section 3: that semantic-level action spaces reduce the low-level geometric and physical reasoning burden placed on the VLM and enables the VLM to focus on semantic task decomposition and decision making. Our results therefore provide empirical evidence that appropriately designed semantic tool abstractions are a key ingredient for effective embodied agent performance.

## C Training Details

## C.1 Hyperparameters

The SFT stage uses a learning rate of $1 \times 1 0 ^ { - 5 }$ , an effective batch size of 32, and 3 epochs; the GRPO stage uses a learning rate of $5 \times 1 0 ^ { - 6 }$ , an effective batch size of 12, and 3 epochs. For GRPO, we sample $K = 4$ rollouts per prompt and optimize with a KL penalty of $\beta = 0 . 0 4$ . All training is conducted on 8 NVIDIA H100 80GB GPUs using bfloat16 precision, FlashAttention-2, and DeepSpeed ZeRO-3. We utilize ms-swift (ModelScope Community, 2025) as our training pipeline.

## C.2 RL Training

Unlike the standard post-training recipe (Zhou et al., 2025; Shao et al., 2024) that applies reinforcement learning across the entire supervised fine-tuning dataset, we perform GRPO exclusively on the two most challenging long-horizon manipulation tasks. These tasks require substantially more multi-step reasoning, error recovery, and action planning than the remaining tasks, making them the primary bottleneck for overall policy performance.

Our decision is motivated by both computational efficiency and training effectiveness. Long-horizon embodied reasoning introduces significantly higher reinforcement learning costs than conventional language-only RL settings, as each rollout requires repeated multimodal inference together with environment interaction. Consequently, the cost of collecting on-policy trajectories grows rapidly with episode length. Applying RL to the full task suite would therefore require substantially more training resources while providing diminishing returns on simpler tasks that are already well learned during supervised fine-tuning.

Instead, we adopt a targeted RL strategy that focuses optimization on the most difficult long-horizon scenarios. This design allows the policy to improve sequential planning, recovery from intermediate failures, and longrange task completion while remaining computationally tractable under our training budget. Empirically, we find that concentrating RL updates on these challenging tasks yields more favorable performance gains than uniformly allocating reinforcement learning compute across all tasks.

## D Additional Results

Guava Harness on Additional Models. We further tested several other frontier and open-source models with Guava’s harness framework in simulation as shown in Table 3. Our framework proves to work well consistently for frontier models like Gemini-3.1-Pro (Fu et al., 2026) and Claude-Sonnet-4.6 (Qwen Team, 2026), while Qwen3.5-2B (OpenAI, 2026a), due to its small size, has poor performance due to poor instruction following and tool calling ability, as it frequently hallucinates wrong reasoning and invalid tool calls.

Efficiency. Compared to Guava’s harness framework with GPT-5.4 (OpenAI, 2026a), Guava-Agent-4B use less tokens as shown in Figure 9.

Table 3 Results of Guava harness with additional base models. Values are success rates (%) evaluated over 15 episodes. Best results are marked bold.
<table><tr><td> Task Name</td><td>Gemini-3.1-Pro (Fu et al., 2026)(</td><td>Success Rate (%) Claude-Sonnet-4.6 (Qwen Team, 2026)</td><td>Qwen3.5-2B (OpenAI, 2026a)</td></tr><tr><td colspan="4"></td></tr><tr><td> place can in box</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td> arrange by size</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>remove cube from tray</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td> push basket</td><td>66.7</td><td>66.7</td><td>0.0</td></tr><tr><td>close drawer</td><td>80.0</td><td>73.3</td><td>0.0</td></tr><tr><td colspan="4">OOD Object</td></tr><tr><td> pick up carrot</td><td>100.0</td><td>100.0</td><td>26.7</td></tr><tr><td>tomato near potato</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>lemon in bin</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>push pot</td><td>60.0</td><td>60.0</td><td>0.0</td></tr><tr><td colspan="4">OOD Prompt</td></tr><tr><td> stack cube reverse order</td><td>100.0</td><td>86.7</td><td>6.7</td></tr><tr><td>arrange by size reverse direction</td><td>86.7</td><td>100.0</td><td>0.0</td></tr><tr><td colspan="4">OOD Long Horizon</td></tr><tr><td> separate food and utensils</td><td>100.0</td><td>86.7</td><td>0.0</td></tr><tr><td>set table</td><td>73.3</td><td>73.3</td><td>0088</td></tr><tr><td> shell game</td><td>93.3</td><td>86.7</td><td></td></tr><tr><td>place all red objects in basket</td><td>86.7</td><td>100.0</td><td></td></tr><tr><td>Overall</td><td>89.8</td><td>88.9</td><td>2.2</td></tr></table>

![](images/2ad656bd7860e435e1fa6fc32c2c7d78d7e12a66205ed5926e19d3254119c44f.jpg)  
Figure 9 Guava-Agent-4B is more token efficient than GPT-5.4 (OpenAI, 2026a).

## E Additional Discussions

Spatial Understanding in VLMs. As shown in Table 2, tasks requiring precise spatial reasoning, such as pushing and object arrangement, consistently exhibit the lowest success rates across all models in both simulation and the real world. This reveals a key limitation of current VLMs in their spatial reasoning abilities, consistent with prior observations (Chen et al., 2024) that VLMs often struggle with spatial understanding. For pushing or arrange by order, the model need to understand important spatial concepts such as direction, orientation and relative positions from the image. For example, to perform a push, the model needs to infer the pre-contact gripper pose, the target end position of object, and also the direction and magnitude of the push. We find that instead of inferring these spatial information from the image, the model often relies only on the world coordinates of the robot end effector and the objects given to the VLM as textual inputs. The VLM model utilizes a reasoning shortcut to naively associate directions with certain coordinates based on heuristics, e.g., left/right is equivalent to negative/positive x-axis without utilizing the image observation. This leads to common failure modes and remains a major bottleneck to the use of VLM in embodied tasks.

Awareness of Execution State. In read world experiments, we found out that when execution is interrupted and later resumed using the same instruction, the agent can infer which subtasks have already been completed and adapt its actions accordingly. For example, if an object is already grasped, the agent proceeds directly to the next step instead of attempting to grasp it again, indicating an internal representation of task progress rather than simple prompt imitation. This can be seen in Figure 10, where the agent directly insert the peg into the holder without generating a grasp action to pick up the peg first.

Task: Insert the white stick into the green holder.

![](images/d1d4afbcfb9974f0b42b4bf93197acafac19a1f3de6c966a8676d5f91acfba6d.jpg)

Think: The white stick is already in the gripper at the top center, hanging vertically. The green holder is on the table near the middle. To place the stick inside, I should first move above the holder with enough clearance, then descend and release. Since I'm holding an object, I should avoid clipping the holder rim.

Think: The white stick is inside the green holder from the camera view. The safest next step is to open the gripper to release it.

Tool: release()

Think: The white stick is visibly inside the green holder, and the gripper is open, so the insertion succeeded. No further action is needed.

Task completed.

Tool: align(green holder, top, medium)

Figure 10 State awareness. Our model enables awareness of task progress such that it does not repeat action that has already been completed.

Sim2Real for Embodied Agent. One important feature of our method is its ability to transfer from simulation to the real world. Unlike robotics RL or IL policies, which are often sensitive to visual sim-to-real gaps, our agentic formulation separates high-level semantic planning from low-level perception and control. This abstraction allows the system to leverage the general visual understanding capability of pretrained VLMs and generalize to new scenes without task-specific robot data.

We also find that simulation is not always an easier evaluation setting than the real world. In particular, imperfect contact dynamics in simulation may introduce failure modes that do not directly reflect real-world performance. This suggests that real-world evaluation remains essential for assessing embodied agentic systems. Future work should further study how to evaluate such systems accurately and efficiently across both simulation and real-world settings.

## F Examples

We present example reasoning and tool calling trajectories generated by our method for some tasks in Figure 11,12 and 13. These examples illustrate our method’s ability to perform long-horizon planning, recover from execution failures, and reason about spatial relationships in the scene.

![](images/227da042a1db07d8965442b89157104f1be4aa4097565c6f10ff6f44afc96f85.jpg)

“Place utensils onto the tray and food items into the basket”

Think: The fork is near the middle-right, the banana is lower-right, the tray is left-center, and the basket is far left. To complete the task, I should first place the fork onto the tray, then the banana into the basket. I should align above the fork first. The fork is small and flat, so a small clearance is appropriate.

Tool: align(fork, top, small)

Think: The gripper is now directly above the fork with a small top clearance. Nothing has been picked up yet. Best next step is to grasp the fork.

Tool: grasp(fork)

Think: The fork is grasped successfully, but the gripper is still very low near the table. Before moving sideways toward the tray, I should lift to avoid dragging the fork or colliding with the tray edge.

Tool: move(0.373, -0.017, 0.05)

Think: The fork is grasped and lifted clear of the table. Next, I need to place it onto the tray. A safe next step is to align above the tray before descending and releasing. Since the fork is thin, medium top clearance over the tray is enough to avoid clipping the tray edges.

Tool: align(tray, top, medium)

Think: The fork is grasped and positioned above the tray. The tray is open and unobstructed, and releasing here should place the fork onto it.

Tool: release()

Think: The fork is now on the tray, so that part is done. The banana is still on the table and needs to go into the basket. The gripper is open and above the workspace, so the next step is to position above the banana for a top-down grasp.

Tool: align(banana, top, medium)

Think: The gripper is aligned above the banana and open, so the next step is to grasp it.

Tool: grasp(banana)

Think: The banana is grasped successfully, but the gripper is still low near the table. Before moving sideways toward the basket, I should lift to avoid dragging or colliding with the table and basket rim.

Tool: move(0.506, 0.006, 0.06)

Think: The banana is lifted and clear of the table. Next, I need to place it into the basket. Since I’m holding an object and the basket has raised sides, aligning above the basket with large clearance is safest before descending for a gentle release. The basket is on the right with enough open top access.

Tool: align(basket, top, large)

Think: To place the banana into the basket gently, I should descend closer to the basket interior before releasing. Current z is 0.114 while the basket top is around 0.017 with height 0.09, so moving down to just above the basket bottom is appropriate and avoids dropping from height.

Tool: move(0.449, -0.372, 0.03)

Think: The banana is lowered into the basket area and appears safely inside the rim. This is the right height to let it settle gently onto the basket lining. I should release now.

Tool: release()

Think: The banana is in the basket and the fork is on the tray. The gripper is open and clear of objects. The task goal is satisfied.

Task completed.

Figure 11 Example task requiring the robot to pick up and place multiple objects in the scene.

“Pick up the carrot”

Think: The carrot is on the table near the lower center-right, unobstructed, and the gripper is high above the table near the center. To pick it up reliably, I should first move directly above the carrot. The gripper is open, which is fine. There are no nearby obstacles, so medium clearance is safe.

Tool: align(carrot, top, medium)

Think: The gripper is now centered above the carrot with a safe top clearance and open wide. Since I already aligned above it, grasping now should descend and close onto the carrot. Risk is low; after success I would need to lift, but first I need to secure it.

Tool: grasp(carrot)

<Human hand manually moves the carrot away>

![](images/2486c4bd6875bb15bda8991a1fc6d931e7ca61e84dd42b576e6aa832ce03b9f3.jpg)

Think: The grasp missed: the gripper is fully closed, so it closed on air or beside the carrot. The carrot is still on the table, slightly left of the gripper in the image. Since the gripper is closed and near table height, I should open it before trying again. After opening, I can re-align and retry.

Tool: release()

Think: The carrot is still on the table, slightly left of the gripper in the image. The gripper is now low near the table, so I should re-align above the carrot before retrying. Medium top clearance is appropriate for the carrot’s height and gives room to descend cleanly.

Tool: align(carrot, top, medium)

Think: The carrot has shifted and is now near the gripper, with the fingers already close above it. Since the gripper is open and already positioned above the carrot, the best next step is to call grasp again on the carrot.

Tool: grasp(carrot)

Think: The carrot is grasped successfully, but the gripper is still down at table height. To finish the pick-up safely, I should lift straight up before any other motion. There are no nearby obstacles above, and a modest vertical lift will clearly separate the carrot from the table.

Tool: move(0.466, -0.142, 0.08)

Think: The carrot is securely grasped and lifted clear of the table, so the pick-up task is complete. There was one failed grasp attempt, but I recovered by opening, re-aligning, re-grasping, and lifting successfully. No further action is needed.

Task complete.

Figure 12 Our system detects grasping failures and retries execution to successfully complete the task.

![](images/981432bed2e583d88d5800fbb9deef3eba2351cd9821331c29b352a3831192a4.jpg)

## “Push basket to the right”

The basket is centered under the gripper. To push it to the right from the camera view, I should approach from its left side. For a stable push I first need the gripper closed, since it is currently fully open. The gripper is already above the basket, but closing first is the next step before aligning and lowering for the push.

Tool: close\_gripper()

The gripper is now closed and not holding anything, which is fine for pushing. The basket remains centered below. To push right, I need to get to the basket’s left side in camera view. The next step is to align to the basket’s left side with a large clearance because the basket is wide and I want room to avoid clipping while positioning.

Tool: align(basket, left, large)

I am now to the left of the basket in camera view, which is the correct side to push it right. I need to lower slightly to contact near the upper sidewall/rim without crashing into the table. A modest downward move should put the closed fingers in good pushing contact before the lateral push.

Tool: move(0.42, -0.151, 0.15)

The gripper is now lowered to a good pushing height and remains on the basket’s left side. The basket center y is about -0.06 while the gripper y is -0.151, consistent with being to the left in camera view. To push the basket right, I should move across toward bigger y, keeping x and z roughly fixed. A substantial rightward move should contact and slide the basket without needing another setup step.

Tool: move(0.42, 0.14, 0.151)

The basket has clearly shifted right relative to the frame, and the gripper is now beside its left wall after the push. No further action is needed.

Task completed.

Figure 13 Example pushing task where understanding of spatial relationships and push in the right direction is required.