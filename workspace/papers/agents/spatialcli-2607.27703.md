# SpatialCLI: Learning to Reason With Spatial Tools, Then Without Them

Yang Zhou<sup>1,2,†</sup>, Zixuan Huang<sup>2,3,†</sup>, Sunzhu Li<sup>2</sup>, Zhuo Yang<sup>4</sup>, Chen Zhang<sup>1</sup>, Shunian Chen<sup>2,6</sup>, Caijun Yan<sup>1</sup>, Jianyao Xu<sup>2</sup>, Shunyu Liu<sup>5</sup>, Weijie Fu<sup>2</sup>, Peiliang Li<sup>2</sup>, Xiaozhi Chen<sup>2</sup>, Yuxiang Cai<sup>1,∗</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>Zhuoyu Technology, Shenzhen, China, <sup>3</sup>Beihang University

<sup>4</sup>University of Electronic Science and Technology of China, <sup>5</sup>Nanyang Technological University

<sup>6</sup>The Chinese University of Hong Kong, Shenzhen

<sup>†</sup>Equal contribution, <sup>∗</sup>Corresponding author

Vision-language models (VLMs) are increasingly used in embodied agents to interpret visual inputs, reason about spatial relationships, and make task-level decisions based on that reasoning. However, a fundamental capability mismatch remains: general VLMs can reason about the overall task but often miss the visual details that determine success, while specialist vision models can capture those details but cannot translate them into task-level decisions. In this work, we propose SpatialCLI, a framework that teaches VLMs to reason with spatial tools and progressively internalize the specialist perceptual capabilities they provide. SpatialCLI proceeds in three stages: (1) Call exposes specialist vision models as spatial tools to augment the VLM’s perception; (2) Learn uses Cold-Start SFT and agentic RL to improve tool use; and (3) Internalize verbalizes successful tool-use trajectories to internalize specialist perceptual capabilities. We further introduce SpatialCLI-Bench, a 516-example benchmark for compositional perception across localization, segmentation, depth, and pose. On MindCube, SpatialCLI raises Qwen3-VL-8B-Instruct from 29.3% to 84.6% with tools, surpassing GPT-5.6 Sol with tools (72.1%), while retaining 73.8% without tools after internalization.

Date: 2026.7.31 Code: https://github.com/IANNXANG/SpatialCLI Model: https://huggingface.co/ZYT-MFM/SpatialCLI-8B Dataset: https://huggingface.co/datasets/ZYT-MFM/SpatialCLI-Data Correspondence: imzhouyang@zju.edu.cn, caiyuxiang@zju.edu.cn

云ZYT

## 1 Introduction

As physical AI moves into open-ended real-world environments, task success increasingly depends on composing multiple spatial perceptual capabilities on demand rather than improving any single capability in isolation [21, 63]. As shown in Figure 1, finding the farthest teddy bear requires first using segmentation to identify all candidate bears, then comparing their distances using depth information, and finally localizing the selected bear. Diferent real-world tasks require diferent combinations of spatial perceptual capabilities, requiring general physical intelligence to select and coordinate complementary capabilities according to the task objective.

However, general VLMs and specialist vision models exhibit a fundamental capability mismatch under this requirement. General VLMs can interpret instructions, decompose tasks, and organize multi-step reasoning, but become unreliable when an answer hinges on precise localization, object boundaries, metric depth, or pose. Specialist vision models such as SAM 3 [4], DA3 [25], and VGGT [45] provide more reliable local perceptual evidence, but each primarily handles a specific perceptual operation and cannot understand which capabilities a task requires, compose outputs from multiple specialists, or translate those outputs into task level decisions. Recent systems such as SpaceTools [7], AlloSpatial [37], and S-Agent [10] take important steps by organizing external tools and spatial priors into agentic reasoning processes and training VLMs on the resulting interactions. An open question is whether the perceptual evidence accumulated during tool interaction can be converted into supervision that enables VLMs to internalize specialist perceptual capabilities and reason without invoking spatial tools. This would bring VLMs closer to becoming foundation models native to the physical world.

![](images/a1bf76512ef385183c9183fc2fceda349129218feb864168852c27941fb5aead.jpg)  
Figure 1 A conceptual comparison of a general VLM with and without SpatialCLI Tools.

In this work, we introduce SpatialCLI, a three-stage framework that enables VLMs to reason with spatial tools and then internalize the specialist perceptual capabilities they provide: (1) Call: Inference-Time Tool Augmentation. The Call stage exposes specialist vision models for localization, segmentation, depth, and pose as spatial tools, augmenting the VLM’s perception with fine-grained visual evidence. (2) Learn: Agentic Fine-Tuning. The Learn stage uses Cold-Start SFT to establish basic tool-use behaviors and agentic RL to improve tool planning, selection, and result utilization through task feedback. (3) Internalize: Trajectory-Guided Capability Internalization. SpatialCLI converts successful tool-use trajectories collected from SpatialCLI-RL into evidence-grounded perceptual reasoning chains, and then applies Dual-View Capability Internalization to train the final SpatialCLI model to internalize specialist perceptual capabilities while preserving its learned tool-use policy. Existing benchmarks predominantly evaluate spatial abilities in isolation, leaving it unclear whether models can coordinate multiple complementary capabilities to solve complex spatial problems. To address this evaluation gap, we construct SpatialCLI-Bench, a 516-example six-choice benchmark for compositional perceptual reasoning across localization, segmentation, depth, and pose. Our main contributions are summarized as follows:

• We propose SpatialCLI, a Call–Learn–Internalize framework that equips VLMs with specialist vision models as spatial tools. Through agentic fine-tuning, SpatialCLI learns to coordinate these tools and turns successful tool-use trajectories into evidence-grounded supervision for internalizing specialist perceptual capabilities.

• We construct SpatialCLI-Bench, a 516-example benchmark that evaluates compositional perceptual reasoning across localization, segmentation, depth, and pose. The benchmark exposes a substantial limitation of current frontier models: GPT-5.6 Sol achieves only 48.8%.

• Extensive experiments show that SpatialCLI efectively improves both tool-enabled and tool-free reasoning. On SpatialCLI-Bench, SpatialCLI-8B reaches 91.3% with tools and 72.7% without tools, with consistent gains on other embodied and spatial benchmarks. These results show that external tool use and internalized direct reasoning can coexist in one model, providing a path toward foundation models native to the physical world.

## 2 Related Work

Learning Spatial Capabilities in Vision-Language Models. Embodied VLMs and VLAs acquire grounded visuallanguage knowledge through pretraining and robot demonstrations [21, 43, 63]. However, current multimodal models still struggle with spatial relations, viewpoint transformation, and 3D understanding [51, 54, 57]. SpatialVLM, SpatialRGPT, and RoboRefer directly train spatial capabilities with spatial question answering, depth cues, 3D annotations, and task-specific supervision [5, 9, 62]. VLM3 further shows that standard VLM architectures can learn depth, pixel correspondence, camera pose, and object-level 3D understanding from text-based supervision and scaled data mixtures [3]. SpatialCLI follows a diferent route: the VLM invokes external specialist perception tools, then learns direct reasoning from supervision derived from its own successful tool-use trajectories.

![](images/25ccc6da793771f93c5dc8d6e2a91b2af514fdccc171b9a7ac233d2973288867.jpg)  
Figure 2 Overview of SpatialCLI. Its Call, Learn, and Internalize stages progressively transform specialist perceptual capabilities into callable spatial tools, learnable tool-use policies, and internalized model capabilities.

Tool-Augmented Agents and Spatial Reasoning. Building on advances in LLM reasoning [13, 19, 48], agents can plan, interact with environments, and invoke tools [14, 38, 40, 42, 56], enabling applications in search and information seeking [20, 29, 32], coding [52, 59], and GUI interaction [34, 61]. In robotics, hierarchical agents use VLMs as high-level planners and VLAs as low-level executors [6, 17, 24, 27, 55, 60], but their reliability remains constrained by the spatial perception of the VLM planner. Spatial reasoning systems use 3D priors or constraints [8, 31], geometric computation or multi-tool coordination [7, 15], and allocentric representations or spatiotemporal evidence [10, 37]. SpatialCLI not only learns to use multiple specialist perception tools through interaction, but also transforms the VLM’s own successful tool-use trajectories into supervision for direct reasoning, enabling reasoning both with external tools and directly without them.

## 3 Method

SpatialCLI aims to turn external specialist perceptual capabilities into abilities that VLMs can call, learn, and internalize. As shown in Figure 2, it proceeds in three stages. (1) inference-time tool augmentation equips VLMs with specialist vision models as spatial tools. (2) agentic fine-tuning fine-tunes the model to learn efective tool-use policies. (3) trajectory-guided capability internalization verbalizes tool-use trajectories to internalize specialist perceptual capabilities while preserving tool-use ability. The complete end-to-end procedure is summarized in Algorithm 1 of Appendix A. In the following, we first introduce the three stages of SpatialCLI and then describe the construction of SpatialCLI-Bench.

## 3.1 Inference-Time Tool Augmentation

Neither VLMs nor specialist vision models can independently solve complex embodied tasks, but they ofer complementary strengths. (1) VLMs excel at semantic understanding and task-level reasoning, but remain unreliable for fine-grained localization, segmentation, depth, and pose perception. (2) Specialist vision models provide precise perceptual outputs, but lack the task-level understanding needed to determine when they should be invoked or how their outputs should be combined.

To combine these complementary strengths, SpatialCLI builds an inference-time tool-augmented agent framework that equips the VLM with specialist vision models as spatial tools. Within this framework, the VLM understands the task, selects tools, composes evidence, and completes reasoning, while specialist vision models provide reliable local perceptual evidence. Following ReAct [56], at each interaction step, the VLM first reasons internally and then either produces a final answer or makes a tool call. When the VLM makes a tool call, SpatialCLI executes the corresponding spatial tool and returns the tool result to the VLM. The

VLM then continues reasoning with the updated interaction history until it produces a final answer or exhausts the tool-call budget. Following budget-aware tool-use agents [30], SpatialCLI exposes the remaining tool-call budget after each tool response, allowing the VLM to decide whether to gather additional evidence or terminate the interaction. We observe that discarding reasoning content across interaction turns causes redundant reasoning and token ineficiency [26]; we therefore retain the VLM’s reasoning throughout the entire interaction.

SpatialCLI provides four spatial tools. (1) Locate, backed by Locate Anything [46] and Grounding DINO [28], takes a language query and returns object bounding boxes. (2) Segment, backed by SAM 3 [4], takes a language query and returns polygonal object boundaries. (3) Depth, backed by Depth Anything 3 [25], takes queried image points and returns metric depth. (4) Pose, backed by Orient Anything V2 [47] and VGGT [45], takes an object or camera-motion query and returns object orientation or cross-view camera motion. The complete interfaces and registrations are provided in Appendix D.2 and Boxes D.1–D.4, while the shared agentic tool-use prompt is shown in Box G.1. SpatialCLI records the resulting tool calls, returned results, and final answer. We represent the executed tool-interaction trace and the corresponding complete interaction sample as

$$
\tau = \left( ( z _ { t } , a _ { t } , o _ { t } ) _ { t = 1 } ^ { T } \right) , \qquad \xi = ( I , q , \tau , y ) ,
$$

where $z _ { t }$ is the reasoning preceding the t-th executed tool call, $a _ { t }$ is that tool call, $o _ { t }$ is its returned result, $T$ is the number of executed tool calls, and y is the final answer produced after the trace.

## 3.2 Agentic Fine-Tuning

Within this agent framework, the VLM can interact with spatial tools to produce complete interaction trajectories, but it cannot yet reliably determine when to use which tool, how to specify appropriate arguments, or how to use the returned results to guide subsequent reasoning. Directly applying multi-tool RL to the initial model requires exploration over a large hybrid action space, where the model is prone to several failure patterns: incorrect tool-call formats, invalid arguments, unnecessary repeated calls, and ignoring tool results when they conflict with its prior beliefs. SpatialCLI therefore first uses Cold-Start SFT to establish basic tool-interaction behaviors and then applies agentic RL to improve tool-use planning, argument generation, result utilization, and termination through task-level feedback.

Cold-Start SFT. Efective RL exploration requires an initial policy capable of selecting appropriate tools and specifying valid arguments. SpatialCLI therefore uses Qwen3.5-397B-A17B [35] as the teacher model, equips it with the inference-time agent framework, and records the complete multi-turn trajectories it produces while solving the training tasks. As illustrated by the cases in Appendix H, these trajectories provide reusable reasoning and common tool-use patterns, such as planning before execution. We discard samples with invalid tool-call formats, failed tool execution, or incorrect final answers, yielding the SFT dataset $\mathcal { D } _ { \mathrm { S F T } }$ . During training, returned tool results serve only as context for subsequent generation, while the loss is computed over model-generated reasoning, tool calls, and final-answer tokens. This stage transfers these patterns to the initial model and yields the SpatialCLI-SFT checkpoint π<sub>SFT</sub> for RL.

Agentic RL. SFT can only imitate filtered teacher trajectories and cannot use task outcomes to improve tool-use decisions beyond the demonstrations. Starting from the SpatialCLI-SFT checkpoint $\pi _ { \mathrm { S F T } }$ , the model follows the same interaction loop as inference-time tool augmentation during rollout, with tool results dynamically appended to the interaction history to guide subsequent decisions. For each task $x = ( I , q , y ^ { * } ) \sim \mathcal { D } _ { \mathrm { R L } }$ GRPO [39] samples a group of G complete interactions $\{ \xi _ { i } \} _ { i = 1 } ^ { G }$ from the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ . Each interaction receives an outcome reward $R _ { i } = V ( y _ { i } , y ^ { * } )$ based on its final answer. Optimizing these rewards yields the SpatialCLI-RL checkpoint $\pi _ { \mathrm { R L } } ;$ the complete objective and training configuration are provided in $\mathrm { A p } \cdot$ pendix B.1.

## 3.3 Trajectory-Guided Capability Internalization

Moving toward foundation models native to the physical world requires perceptual capabilities to reside in the model itself rather than depend entirely on external tools. Inspired by learning from visual-program execution traces [18], SpatialCLI therefore uses successful SpatialCLI-RL trajectories as the source of supervision for internalizing the specialist perceptual capabilities supplied by spatial tools while preserving the learned tool-use policy. This process consists of two steps: (1) Progressive Evidence-Grounded Trajectory Verbalization converts successful tool-use trajectories into explicit perceptual reasoning chains, and (2) Dual-View Capability Internalization jointly trains on capability-internalization and tool-use views to internalize specialist perceptual capabilities without sacrificing tool use.

Progressive Evidence-Grounded Trajectory Verbalization. Raw SpatialCLI-RL trajectories contain lengthy model reasoning, tool-call syntax, heterogeneous tool returns, and repeated interaction history, making them unsuitable as direct natural-language supervision. Verbalizing an entire trajectory in a single pass requires processing a long context and can obscure dependencies between evidence collected across turns. SpatialCLI therefore processes each successful trajectory $( y = y ^ { * } )$ in two successive steps. (1) Turn-wise evidence consolidation. To preserve cross-turn dependencies without repeatedly processing the full raw history, SpatialCLI consolidates the newly collected evidence after each tool interaction. Let $e _ { t }$ denote the resulting evidence– reasoning unit at turn t, and let $e _ { < t } = ( e _ { 1 } , \ldots , e _ { t - 1 } )$ , with $e _ { < 1 } = \emptyset$ . The extractor combines the visual input and task instruction with the previously consolidated units and the current interaction:

$$
e _ { t } = \Psi _ { \mathrm { e x t } } \left( I , q , e _ { < t } , z _ { t } , a _ { t } , o _ { t } \right) , \qquad t = 1 , . . . , T .
$$

Each $e _ { t }$ exhaustively verbalizes the current tool result, preserves explicit visual observations from $z _ { t } ,$ and records how they update the accumulated evidence without repeating unchanged content. Each perceptual statement must be traceable to the current tool result, an explicit visual observation in $z _ { t } ,$ or a previous unit; the correct answer $y ^ { * }$ is withheld to prevent answer-conditioned evidence reconstruction. (2) Global trajectory verbalization. To integrate evidence distributed across turns into a coherent task-level reasoning chain, SpatialCLI passes the visual input, task instruction, consolidated evidence trajectory $E _ { \tau } = ( e _ { 1 } , . . . , e _ { T } )$ and correct answer to a global verbalizer:

$$
c = \Phi _ { \mathrm { v e r b } } \left( I , q , E _ { \tau } , y ^ { * } \right) .
$$

The verbalizer removes redundancy and organizes the dependencies from perceptual observations to the correct answer into a concise reasoning chain c, without introducing entities, attributes, values, or relations absent from $E _ { \tau }$ . The complete turn-wise consolidation and global verbalization prompts are provided in Appendix G, Boxes G.2 and G.3.

Dual-View Capability Internalization. To internalize specialist perceptual capabilities while preserving the model’s learned tool-use policy, SpatialCLI constructs two training views from each successful trajectory. (1) The Capability-Internalization View uses a direct-answer prompt and forms each training sample from the image, task instruction, explicit perceptual reasoning chain, and final answer. The model learns to generate the reasoning chain and answer without accessing external tools, thereby internalizing the specialist perceptual capabilities originally supplied by spatial tools. (2) The Tool-Use View uses a prompt containing tool descriptions and preserves the original interaction structure, with returned tool results serving as context for subsequent generation. Training supervises model-generated reasoning, tool calls, and final-answer tokens, preserving the model’s ability to decide when to invoke tools and how to use returned results. The two views are jointly optimized as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C I } } = \mathcal { L } _ { \mathrm { i n t e r n a l } } + \lambda \mathcal { L } _ { \mathrm { a g e n t i c } } . } \end{array}
$$

Here, $\mathcal { L } _ { \mathrm { i n t e r n a l } }$ supervises tool-free explicit perceptual reasoning and the final answer, while $\mathcal { L } _ { \mathrm { a g e n t i c } }$ supervises model-generated tokens in the tool-interaction trajectory. Both terms are mean negative log-likelihoods over their respective supervised tokens, and λ controls the relative weight of the Tool-Use View.

## 3.4 SpatialCLI-Bench Construction

Existing benchmarks predominantly evaluate spatial capabilities in isolation, making it dificult to assess whether models can coordinate multiple perceptual capabilities. We therefore construct SpatialCLI-Bench, a 516-example English six-choice visual question answering benchmark for compositional reasoning across localization, segmentation, depth, and pose.

<table><tr><td rowspan="2">Model</td><td rowspan="2">SpatialCLI Bench</td><td rowspan="2">MindCube</td><td colspan="2">MMSI</td><td rowspan="2">DA-2K</td><td colspan="2">BOPASK</td><td rowspan="2">Avg.</td></tr><tr><td>Motion-Cam Pos-Cam-Cam</td><td></td><td>Traj.</td><td>ObjRrr</td></tr><tr><td colspan="10">Frontier Models</td></tr><tr><td>GPT-5.6 Sol</td><td>48.8</td><td>70.3</td><td>52.7</td><td>62.4</td><td>79.2</td><td>51.5</td><td>56.4</td><td>62.0</td></tr><tr><td>+ SpatialCLI Tools</td><td>72.9 (↑24.1)</td><td>72.1 (↑1.8)</td><td>54.1 (↑1.4)</td><td>62.4(↑0.0)</td><td>86.1 (↑6.9)</td><td>52.8 (↑1.3)</td><td>49.5 (↓6.9)</td><td>68.1 (↑6.1)</td></tr><tr><td>Gemini 3.1 Pro</td><td>52.9</td><td>74.2</td><td>59.5</td><td>51.6</td><td>67.7</td><td>48.6</td><td>14.2</td><td>56.4</td></tr><tr><td>Qwen3.7-Plus</td><td>43.8</td><td>62.8</td><td>43.2</td><td>45.2</td><td>76.3</td><td>61.3</td><td>42.4</td><td>55.8</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>40.9</td><td>49.3</td><td>43.2</td><td>46.2</td><td>70.3</td><td>56.4</td><td>31.0</td><td>49.8</td></tr><tr><td>+ SpatialCLI Tools</td><td>79.8 (↑39.0)</td><td>67.6 (↑18.3)</td><td>46.0 (↑2.8)</td><td>54.8 (↑8.6)</td><td>91.9 (↑21.6)</td><td>57.7 (↑1.3)</td><td>44.7(↑13.7)</td><td>68.2 (↑18.4)</td></tr><tr><td colspan="9">Our Models</td></tr><tr><td>Qwen3.6-27B</td><td>46.3</td><td>56.6</td><td>40.5</td><td>40.9</td><td>75.0</td><td>50.9</td><td>36.8</td><td>52.5</td></tr><tr><td>+ SpatialCLI Tools</td><td>82.2 (↑35.9)</td><td>62.4 (↑5.8)</td><td>51.4 (↑10.9)</td><td>46.2 (↑5.3)</td><td>91.5 (↑16.5)</td><td>60.6 (↑9.7)</td><td>50.6 (↑13.8)</td><td>68.1 (↑15.6)</td></tr><tr><td>SpatialCLI-27B</td><td>76.3 (↑30.0)</td><td>80.4 (↑23.8)</td><td>41.9 (↑1.4)</td><td>43.0 (↑2.1)</td><td>85.5 (↑10.5)</td><td>57.3 (↑6.4)</td><td>53.1 (↑16.3)</td><td>68.0 (↑15.5)</td></tr><tr><td>+ SpatialCLI Tools</td><td>91.7 (↑45.4)</td><td>85.5 (↑28.9)</td><td>52.7(↑12.2)</td><td>53.8 (↑12.9)</td><td>91.9 (↑16.9)</td><td>58.2 (↑7.3)</td><td>55.6 (↑18.8)</td><td>75.9 (↑23.4)</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>44.4</td><td>54.2</td><td>36.5</td><td>44.1</td><td>71.9</td><td>52.3</td><td>39.2</td><td>51.3</td></tr><tr><td>+ SpatialCLI Tools</td><td>80.2 (↑35.8)</td><td>63.3 (↑9.1)</td><td>48.7 (↑12.2)</td><td>45.2(↑1.1)</td><td>91.3 (↑19.4)</td><td>58.5 (↑6.2)</td><td>43.5 (↑4.3)</td><td>66.6 (↑15.2)</td></tr><tr><td>SpatialCLI-35B-A3B</td><td>75.0 (↑30.6)</td><td>78.8 (↑24.6)</td><td>43.2 (↑6.7)</td><td>46.2 (↑2.1)</td><td>83.9 (↑12.0)</td><td>57.6 (↑5.3)</td><td>54.7(↑15.5)</td><td>67.7 (↑16.4)</td></tr><tr><td>+ SpatialCLI Tools</td><td>91.9 (↑47.5)</td><td>85.6 (↑31.4)</td><td>50.0 (↑13.5)</td><td>53.8(↑9.7)</td><td>91.8 (↑19.9)</td><td>57.9 (↑5.6)</td><td>53.7 (↑14.5)</td><td>75.4 (↑24.1)</td></tr><tr><td>Qwen3-VL-8B-Instruct 35.3</td><td></td><td>29.3</td><td>27.0</td><td>25.8</td><td>68.1</td><td>25.8</td><td>13.3</td><td>35.7</td></tr><tr><td>+ AlloSpatial Tools</td><td>43.2 (↑7.9)</td><td>35.2 (↑5.9)</td><td>24.3 (↓2.7)</td><td>35.5 (↑9.7)</td><td>47.4(↓20.7)</td><td>25.9 (↑0.1)</td><td>9.7 (↓3.6)</td><td>34.7 (↓1.0)</td></tr><tr><td>+ SpaceTools Tools</td><td>39.3 (↑4.0)</td><td>30.5 (↑1.2)</td><td>21.6 (↓5.4)</td><td>20.4 (↓5.4)</td><td>95.9 (↑27.8)</td><td>44.0 (↑18.2)</td><td>22.3 (↑9.0)</td><td>44.0 (↑8.2)</td></tr><tr><td>+ SpatialCLI Tools</td><td>66.5 (↑31.2)</td><td>47.2 (↑17.9)</td><td>41.9 (↑14.9)</td><td>39.8 (↑14.0)</td><td>91.6 (↑23.5)</td><td>54.3 (↑28.5)</td><td>20.6 (↑7.3)</td><td>56.7 (↑21.0)</td></tr><tr><td>SpatialCLI-8B</td><td>72.7(↑37.4)</td><td>73.8 (↑44.5)</td><td>35.1 (↑8.1)</td><td>32.3 (↑6.5)</td><td>80.8 (↑12.7)</td><td>54.6 (↑28.8)</td><td>52.3 (↑39.0)</td><td>62.9 (↑27.2)</td></tr><tr><td>+ SpatialCLI Tools</td><td>91.3 (↑56.0)</td><td>84.6 (↑55.3)</td><td>39.2 (↑12.2)</td><td>53.8 (↑28.0)</td><td>91.8 (↑23.7)</td><td>55.2(↑29.4)</td><td>48.5 (↑35.2)</td><td>73.2(↑37.5)</td></tr></table>

Table 1 Overall performance on embodied and spatial benchmarks. Scores in model-name rows are obtained under inference $\mathrm { w } / \mathrm { o }$ Tools, while + rows are obtained under inference w/ the indicated tools. Arrows denote score changes from the corresponding base model $\mathrm { w } / \mathrm { o }$ Tools. Avg. macro-averages benchmarks after first averaging subsets within each benchmark. The best result in each column is highlighted in bold.

The construction follows four stages: (1) Gemini 3.1 Pro [11] inventories reliable entities and scene relations; (2) specialist vision models provide localization, segmentation, metric-depth, and pose evidence, which is used to filter candidates with missing, inconsistent, or ambiguous evidence; (3) conditioned on the verified evidence, Gemini 3.1 Pro generates the question, correct answer, and five plausible distractors; and (4) human experts independently answer the questions without seeing Gemini’s answers, and only examples with matching answers are retained. Further details are provided in Appendix C.

## 4 Experiments

## 4.1 Experimental Setup

Models and Training Settings. We instantiate SpatialCLI with three base VLMs: Qwen3-VL-8B-Instruct [1], Qwen3.6-35B-A3B [35], and Qwen3.6-27B [35]. All SFT and RL experiments are conducted using verl [41]. Detailed training configurations are provided in Appendix B.1.

Evaluation Benchmarks. We evaluate SpatialCLI on SpatialCLI-Bench, MindCube [57], MMSI [54], DA-2K [53], and BOPASK [2]. We evaluate each model both w/o Tools and $\mathrm { w } / $ Tools. Detailed evaluation settings, including decoding parameters, tool-call budgets, and repeated evaluations, are provided in Appendix B.2.

Baselines. We compare SpatialCLI with general-purpose models, including GPT-5.6 Sol [33], Gemini 3.1 Pro [11], Qwen3.7-Plus [36], and Qwen3.5-397B-A17B [35]. We further compare with SpaceTools [7] and AlloSpatial [37], two agentic methods for improving model spatial capabilities.

![](images/92d45ceefc171bc366dd9a704ae4c6ba6e86c8bec89f6dd5f14165a1d04e47d7.jpg)  
(a) SpatialCLI-Bench Score

![](images/6ec1eef0fc2045fe369caece631ac0e9c06d40792cb55241528672c3a83976e5.jpg)  
(b) MindCube Score

![](images/6d21a0ada0a80932fa6b6bb96f8c6d231854fcfc4392c518cf4e749da70354ec.jpg)  
(c) Training Reward

![](images/97d2ebbef8b0c841e765c659372fa889bf2008871330c44eb1a92f368a827309.jpg)  
(d) Num. of Tool Calls

Figure 3 Training dynamics of Qwen3-VL-8B-Instruct under diferent RL strategies. (a,b) Inference w/ Tools (solid) and Inference w/o Tools (dashed) scores on SpatialCLI-Bench and MindCube. (c) Training reward over RL training. (d) Number of executed tool calls per trajectory over RL training. We compare RL w/ Tools + SFT, RL w/ Tools w/o SFT, and RL w/o Tools.  
![](images/9a27ff08c2285eeb9a3755af48d30917056b05862b543cbf772725bb8b74b32f.jpg)  
(a) Performance vs. Steps

![](images/9a7044df6cf9664e550e23939cecd7c519e34f12d7e386d297d7ddb96cdf626f.jpg)  
(b) CII vs. Steps

![](images/66f8282e5952a49f0fabc9d1fc9a94a224a76810b84bf50974464feffd90f160.jpg)  
(c) CII vs. Capacity

![](images/458724770552bc4aefe9b6e43ed5aa9e7e0a2e2d966f533bdae059f6cff8d8e6.jpg)  
(d) Performance vs. Capacity  
Figure 4 Capability internalization across training-data exposure and model capacity. (a) SpatialCLI-Bench scores under inference $\mathrm { w } / $ Tools and $\mathrm { w } / \mathrm { o }$ Tools over internalization-stage training steps. (b) Capability-specific CII over the same process. (c) Capability-specific CII across model capacities. (d) SpatialCLI-Bench, MindCube, and DA-2K scores under inference w/o Tools (dashed) and w/ Tools (solid) across model capacities.

## 4.2 Overall Performance

SpatialCLI Tools provide effective runtime spatial augmentation. Table 1 shows that SpatialCLI Tools provide broad gains across both frontier and smaller models. The gains are generally larger for less capable base models: averaged over benchmarks, SpatialCLI Tools improve GPT-5.6 Sol by 6.1 points, compared with 21.0 points for Qwen3-VL-8B-Instruct. This suggests that external specialist perception is particularly valuable when the base model’s native spatial capabilities are limited. Under the controlled Qwen3-VL-8B-Instruct comparison, SpatialCLI Tools outperform the other tool frameworks on most subsets and AlloSpatial on every subset, showing that efective tool interfaces and tool-use policies are critical for translating specialist perception into task performance.

Training jointly improves direct answering and practical tool use. After training, every SpatialCLI variant outperforms its corresponding initial model $\mathrm { w / o }$ Tools on every reported evaluation subset, demonstrating consistent capability internalization. Training also strengthens practical tool use: compared with applying SpatialCLI Tools directly to the corresponding initial models, the trained variants achieve higher w/ Tools performance on most evaluation subsets. For example, SpatialCLI-8B improves from 35.3 to 72.7 on SpatialCLI-Bench $\mathrm { w / o }$ Tools and further reaches 91.3 $\mathrm { w } / $ Tools. These results indicate that capability internalization need not trade of against tool use: training strengthens native spatial reasoning while preserving, and often further improving, the model’s ability to benefit from external tools.

## 4.3 Analysis

In this subsection, we analyze RL training dynamics, the progression of capability internalization with increasing training data, and its scaling behavior across model capacities. To quantify internalization, we use the Capability Internalization Index (CII), which measures how closely a model can reproduce the corresponding spatial-tool outputs without invoking the tools; higher values indicate stronger internalization, and the complete definition is provided in Appendix E.

Training Dynamics. Figure 3 compares RL $\mathrm { w } / $ Tools + SFT, RL $\mathrm { w } / $ Tools w/o SFT, and RL $\mathrm { w / o }$ Tools. RL $\mathrm { w / \ T o o l s \ + \ S F T }$ starts from a strong tool-use policy, remains stable on both evaluation benchmarks, and finishes with higher training reward and substantially better MindCube performance than RL $\mathrm { w } / $ Tools $\mathrm { w / o }$ SFT. RL $\mathrm { w } / $ Tools $\mathrm { w } / \mathrm { o } \ \mathrm { S F T }$ expands from 3.36 to 6.74 executed tool calls per trajectory, whereas RL w/ Tools + SFT remains near 2.56 throughout training, indicating that SFT provides a stable tool-use policy before RL. RL $\mathrm { w / o }$ Tools generally improves direct-answer performance, but its $\mathrm { w } / $ Tools performance begins to decline after approximately 100 training steps, indicating degraded tool-use competence. Moreover, its direct-answer performance reaches only 52.7, well below the 72.7 achieved through capability internalization. This gap indicates that SpatialCLI uses the available training data substantially more efectively than direct SFT or RL. Appendix F.1 provides the complete comparison.

Data Scaling of Capability Internalization. Figures 4a and 4b show that, as the model is exposed to more internalization data, its $\mathrm { w / o }$ Tools score rises from 40.1% to 74.0%, while the four-capability macro-average CII increases from 45.6 to 61.6; the two curves exhibit closely aligned upward trends. This synchronized improvement establishes a clear data-scaling trend: as internalization data increases, both task performance and CII continue to rise, indicating genuine transfer of specialist perceptual capabilities rather than finalanswer memorization. Although the $\mathrm { w } / $ Tools score temporarily declines, it recovers close to its starting level, indicating that Dual-View Capability Internalization restores the learned tool-use policy after transient interference and avoids catastrophic forgetting. Finally, the narrowing performance gap shows that, with increasing internalization data, spatial tools shift from an external capability source required for solving the tasks to an optional augmentation of the model’s native capabilities.

Internalization continues to scale with model capacity after tool-use performance saturates. Figure 4d reveals a clear separation between tool-use and internalization scaling: across the three model variants, $\mathrm { w } / $ Tools scores vary by at most 1.0 point on each benchmark, whereas both larger variants substantially outperform SpatialCLI-8B $\mathrm { w / o }$ Tools. The saturation of $\mathrm { w } / $ Tools performance is consistent with tool use requiring a relatively compact invocation-and-integration policy; once this policy is learned, the shared external specialists supply most of the required perceptual capability and thereby compress capacity-dependent diferences. Internalization is less prone to saturation because the model must instead encode and reproduce multiple specialist capabilities in its own parameters; accordingly, Figure 4c shows that SpatialCLI-35B-A3B and SpatialCLI-27B substantially outperform SpatialCLI-8B across all four capability-specific CII values. These results show that external tools reduce performance diferences across model sizes, whereas capability internalization continues to scale along two complementary dimensions: data exposure determines how fully a given model learns from the specialists, and model capacity afects how much of their capability it can absorb.

## 4.4 Ablation Studies

Using Qwen3-VL-8B-Instruct, we ablate how tools expose perceptual evidence and how tool-use trajectories are converted into capability-internalization supervision.

Structured returns provide more effective tool evidence. Table 2 compares the current structured tool returns with two red-border visual-return variants for Locate and Segment. The structured-plus-visual variant retains the structured coordinates and polygons while appending each visualization as a new image, whereas the visual-only variant returns the visualization without the corresponding structured result; Depth and Pose returns remain unchanged.

Visual Only improves over the initial model on all three evaluations, showing that tool-provided perceptual evidence remains useful even without structured Locate and Segment outputs. Adding structured coordinates and polygons yields substantially stronger performance, including a 20.1-point gain on SpatialCLI-Bench over Visual Only and consistent improvements on BOPASK. Structured Only performs comparably to Struc tured + Visual across all evaluations. Thus, explicit coordinates and polygons are the primary source of the improvement, while rendering the same evidence as additional images introduces extra visual context without a consistent score benefit. Structured returns can also be directly converted into textual capabilityinternalization supervision, making them easier for current VLMs to internalize than multimodal tool outputs such as annotated images.

<table><tr><td>Tool Return</td><td>SpatialCLI-Bench</td><td>BOPASK</td></tr><tr><td>No Tools (Initial Model)</td><td>35.3</td><td>Traj. ObjRrr 25.8 13.3</td></tr><tr><td>Visual Only</td><td>45.9</td><td>45.6 18.3</td></tr><tr><td>Structured + Visual</td><td>66.0</td><td>54.5 520.1</td></tr><tr><td>Structured Only (Ours)</td><td>66.5</td><td>54.3 20.6</td></tr></table>

Table2 Tool-return format ablation with Qwen3-VL-8B-Instruct. Locate and Segment return red-border visualizations, structured coordinates or polygons, or both.

Progressive verbalization and dual-view training are both necessary. Table 3 compares diferent forms of capability-internalization supervision. One-Pass Dual-View replaces Progressive Evidence-Grounded Trajectory Verbalization with a single pass over the complete trajectory while retaining the same dual-view training.

<table><tr><td rowspan="2">Internalization Variant</td><td colspan="2">w/o Tools</td><td colspan="2">w/ Tools</td></tr><tr><td>Score Tokens</td><td></td><td></td><td>Score Tokens</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>35.3</td><td>2738</td><td>66.5</td><td>949</td></tr><tr><td>Final Answer Only</td><td>52.7</td><td>19</td><td>52.1</td><td>28</td></tr><tr><td>CoT + Answer</td><td>45.0</td><td>5646</td><td>42.2</td><td>4401</td></tr><tr><td>Internalization View Only</td><td>71.1</td><td>227</td><td>62.6</td><td>235</td></tr><tr><td>Tool-Use View Only</td><td>42.2</td><td>6049</td><td>89.0</td><td>2155</td></tr><tr><td>One-Pass Dual-View</td><td>64.5</td><td>451</td><td>90.2</td><td>2599</td></tr><tr><td>Full Dual-View (Ours)</td><td>72.7</td><td>274</td><td>91.3</td><td>2480</td></tr></table>

Table 3 Capability-internalization ablation on SpatialCLI-Bench. One-Pass and Full Dual-View use single-pass and progressive trajectory verbalization, respectively. Tokens include model outputs and, $\mathrm { w } / $ Tools, tool returns; the best scores are bold.

For the untrained Qwen3-VL-8B-Instruct, external tools improve the score while reducing trajectory length from 2,738 to 949 tokens, indicating that tool use can also lower decoding cost. The Qwen3.5-397B-A17B case studies in Appendix H exhibit the same qualitative trend, with tools enabling more direct and reliable problem solving. Final Answer Only yields concise but limited direct answers and loses tool use, while CoT + Answer is verbose and weak, showing that neither outcome supervision nor ungrounded reasoning transfers the perceptual process. Internalization View Only produces strong and compact $\mathrm { w / o }$ Tools reasoning but weakens tool use, whereas Tool-Use View Only preserves $\mathrm { w } / $ Tools performance without internalizing the capability, demonstrating the complementary roles of the two views. One-Pass Dual-View supports both modes, but Full Dual-View reaches the best $\mathrm { w / o }$ Tools and $\mathrm { w } / $ Tools scores of 72.7 and 91.3 with shorter outputs, validating progressive evidence consolidation. Appendix F.3 further confirms that Full Dual-View retains the controlled calling behavior learned from the tool-use view.

## 5 Conclusion

General VLMs lack the fine-grained perceptual capabilities required for embodied and spatial reasoning and cannot readily learn them from specialist vision models. SpatialCLI addresses this gap by connecting specialist models as runtime tools, learning agentic tool use, and converting successful trajectories into dualview supervision for capability internalization. Across embodied and spatial benchmarks and model scales,

SpatialCLI consistently improves both $\mathrm { w } / $ Tools and $\mathrm { w / o }$ Tools reasoning, showing that external specialist capabilities can augment inference and be transferred into the VLM without sacrificing tool use.

SpatialCLI remains limited by specialist-tool coverage and reliability, with its current scope restricted to structured perceptual outputs and perception-centric tasks. Future work will extend internalization to multimodal outputs using unified multimodal models capable of both understanding and generation, and integrate VLA tools under VLM planning for joint perception and action.

## Appendix Table of Contents

A Algorithm Pseudocode 13   
B Detailed Experimental Settings 14   
B.1 Detailed Training Settings . 14   
B.2 Detailed Evaluation Settings 15   
B.3 Hardware and Software Environment . 16   
C SpatialCLI-Bench 16   
C.1 Scope and Sample Format 16   
C.2 Data Provenance and Composition 16   
C.3 Capability Composition and Oracle Plans 16   
C.4 Multi-Stage Construction Pipeline 17   
C.5 Independent Human Review and Filtering Yield 17   
D Spatial Tool Interfaces and Implementation 17   
D.1 Coordinate and Serialization Convention . 17   
D.2 Spatial Tool Interfaces 18   
E Capability Internalization Metric 23   
E.1 Evaluation Data and Capability Coverage 23   
E.2 Reference Construction and Evaluation Protocol 23   
E.3 Independent Human Verification 23   
E.4 Similarity Functions and Aggregation 23   
F Additional Experiments 25   
F.1 Comparison with Direct Fine-Tuning . 25   
F.2 Tool-Set Ablation . 26   
F.3 Tool-Use Behavior across Internalization Variants 27   
F.4 Sensitivity to the Tool-Use View Loss Weight 27   
G Prompt Templates 28   
G.1 Agentic Tool-Use Prompt Template . 28   
G.2 Turn-Wise Evidence Consolidation Prompt 28   
G.3 Global Trajectory Verbalization Prompt 30   
H Case Studies 31   
H.1 Case 1: Two-Image Camera Rotation and Depth 31   
H.2 Case 2: Two-Image Camera Translation and Instance Segmentation . 34   
H.3 Case 3: SpatialCLI-8B after Capability Internalization 39

## A Algorithm Pseudocode

Algorithm 1 summarizes the complete Call–Learn–Internalize procedure. Here, Interact $( \pi , x , \mathcal { U } , B )$ runs the ReAct agent with tools U and call budget B, returning $\xi = ( I , q , \tau , y ) ; \tau$ records each call’s reasoning, action, and observation, while ξ retains complete assistant responses, including terminal reasoning. Tool outputs are context only, and SFT minimizes mean negative log-likelihood over model-generated tokens.

Algorithm 1: SpatialCLI: Call–Learn–Internalize   
Require: Initial policy $\pi _ { 0 } ;$ teacher policy π<sub>teach</sub>; shared task pool $\mathcal { D } ;$ RL subset $\mathcal { D } _ { \mathrm { { R L } } } ;$ spatial tools $u ;$ executed  
call budget $B ;$ Tool-Use View weight λ   
Ensure: Trained SpatialCLI policy $\pi _ { \theta }$   
1: Call: Register U = {Locate, Segment, Depth, Pose} in the live agent loop.   
2: Define Interact $( \pi , x , \mathcal { U } , B )$ for $x = ( I , q , y ^ { * } )$ as follows; $y ^ { * }$ is not exposed to π:   
3: Initialize the complete history $\mathcal { H }  ( \mathcal { P } _ { \mathcal { U } } , I , q )$ , trace $\tau  ( )$ , and executed-call count $k  0 ,$ , where $\mathcal { P } _ { \mathcal { U } }$   
is the shared agentic prompt with the registrations of U.   
4: while $k < B$ do   
5: Sample the next complete assistant response $r = ( z , u ) \sim \pi ( \cdot \mid \mathcal { H } )$   
6: if u is a terminal answer y then   
7: Append r to H.   
8: return The serialized interaction $\xi = ( I , q , \tau , y )$   
9: end if   
10: Parse u as the next tool call $a _ { k + 1 } .$ , execute $o _ { k + 1 }  \mathcal { U } ( a _ { k + 1 } )$ , and set $k \gets k + 1 .$   
11: Append $\left( z , a _ { k } , o _ { k } \right)$ to τ and append the complete response r, observation $o _ { k } .$ , and budget notice for   
$B - k$ remaining calls to H; use the budget-exhausted instruction when $B - k = 0$   
12: end while   
13: Sample a terminal response $r = ( z _ { \mathrm { t e r m } } , y ) \sim \pi ( \cdot \mid \mathcal { H } )$ with tool calls disabled and append r to $\mathcal { H } .$   
14: return The serialized interaction $\xi = ( I , q , \tau , y )$   
15: Learn—Cold-Start SFT: $\widetilde { \mathcal { D } } _ { \mathrm { S F T } } \gets \{ \mathrm { I n t e r a c t } ( \pi _ { \mathrm { t e a c h } } , x , \mathcal { U } , B ) : x = ( I , q , y ^ { * } )$ is sampled from ${ \mathcal { D } } \}$   
16: $\mathcal { D } _ { \mathrm { S F T } }  \{ \xi \in \widetilde { \mathcal { D } } _ { \mathrm { S F T } } :$ valid call format, successful tool execution, and $y = y ^ { * } \}$   
17: π<sub>SFT</sub> $ \mathrm { S F T } ( \pi _ { 0 } , { \cal D } _ { \mathrm { S F T } } )$ on reasoning, tool-call, and final-answer tokens.   
18: Learn—Agentic RL: Initialize $\pi _ { \theta }  \pi _ { \mathrm { S F T } } .$   
19: for each agentic RL update do   
20: Sample $x = ( I , q , y ^ { * } ) \sim \mathcal { D } _ { \mathrm { R L } }$   
21: Set $\theta _ { \mathrm { o l d } }  \theta .$   
22: Sample $\{ \xi _ { i } \} _ { i = 1 } ^ { G } \sim \mu _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid I , q )$ through Interact $( \pi _ { \theta _ { \mathrm { o l d } } } , x , \mathcal { U } , B )$   
23: for $i = 1 , \dots , G$ do   
24: Set $R _ { i } \gets V ( y _ { i } , y ^ { * } )$ from the final answer of $\xi _ { i } .$   
25: end for   
26: Update θ by maximizing J<sub>GRPO</sub>(θ) in Equation (1).   
27: end for   
28: Set $\pi _ { \mathrm { R L } }  \pi _ { \theta } .$   
29: Internalize: Collect successful ξ = Interac $\mathbf { \chi } _ {  } ( \pi _ { \mathrm { R L } } , x , \mathcal { U } , B )$ with $y = y ^ { * }$ for tasks $x \in { \mathcal { D } } .$   
30: Initialize $\mathcal { D } _ { \mathrm { i n t e r n a l } }  \emptyset$ and $\mathcal { D } _ { \mathrm { a g e n t i c } }  \emptyset .$   
31: for each collected $\xi = ( I , q , \tau , y ^ { \ast } )$ , where $\tau = ( ( z _ { t } , a _ { t } , o _ { t } ) _ { t = 1 } ^ { T } )$ do   
32: Set $e _ { < 1 }  \emptyset .$   
33: for $t = 1 , \dots , T$ do   
34: $e _ { t } \gets \Psi _ { \mathrm { e x t } } ( I , q , e _ { < t } , z _ { t } , a _ { t } , o _ { t } ) ; \mathrm { s e t } e _ { < t + 1 } \gets ( e _ { 1 } , \dots , e _ { t } ) .$   
35: end for   
36: Set $E _ { \tau } \gets ( e _ { 1 } , \ldots , e _ { T } )$ and $c \gets \Phi _ { \mathrm { v e r b } } ( I , q , E _ { \tau } , y ^ { * } )$   
37: Add the direct-answer sample $( ( I , q ) , ( c , y ^ { * } ) )$ to $\mathcal { D } _ { \mathrm { i n t e r n a l } } .$   
38: Add the original interaction ξ to $\mathcal { D } _ { \mathrm { a g e n t i c } } ,$ masking all returned tool outputs from the loss.   
39: end for   
40: Initialize π ← π and minimize $\mathcal { L } _ { \mathrm { C I } } = \mathcal { L } _ { \mathrm { i n t e r n a l } } + \lambda \mathcal { L } _ { \mathrm { a g e n t i c } }$ on $\mathcal { D } _ { \mathrm { i n t e r n a l } }$ and $\mathcal { D } _ { \mathrm { a g e n t i c } }$   
41: return π<sub>θ</sub>

## B Detailed Experimental Settings

## B.1 Detailed Training Settings

InitialModels. We conduct training on Qwen3.6-27B [35], Qwen3.6-35B-A3B [35], and Qwen3-VL-8B-Instruct [1] as the initial models. All final models in the SpatialCLI series are obtained from their respective initial models through the same sequence of Cold-Start SFT, agentic RL, and trajectory-guided capability internalization.

Training Data. All three stages draw their stage-specific samples from a shared pool of 37,000 training tasks: 5,000 from Vlaser [50], 10,000 from MindCube-Train [57], 10,000 from BOPASK-Trajectory, 10,000 from BOPASK-Object-Rearrangement [2], and 2,000 from RefSpatial [62]. We select these sources for their complementary spatial supervision: Vlaser contributes embodied question answering and grounding, MindCube-Train contributes multi-view spatial reasoning, the two BOPASK subsets contribute interaction-centric trajectory planning and object rearrangement, and RefSpatial contributes diverse 2D and 3D spatial referring. Together, they cover single- and multi-image inputs as well as localization, depth, pose, trajectory, and rearrangement reasoning. For Cold-Start SFT, we use Qwen3.5-397B-A17B [35] as the teacher model, equip it with the live agent loop described above, and sample approximately 5,000 tasks from this pool for toolinteraction annotation. After filtering invalid tool-call formats, failed tool executions, and incorrect final answers, we retain approximately 40% as high-quality correct trajectories, yielding approximately 2,000 SFT trajectories. Agentic RL samples approximately 10,000 tasks from the same shared pool while explicitly excluding all tasks used for Cold-Start SFT. Capability internalization uses approximately 42,000 successful tool-use trajectories collected by rolling out SpatialCLI-RL on tasks from this pool. Both the turn-wise evidence consolidator and the global trajectory verbalizer use Qwen3.5-397B-A17B, with the same sampling and maximum-generation-length parameters as the evaluation configuration in Section B.2. Because trajectory verbalization succeeds for nearly all retained trajectories, each trajectory produces one Capability-Internalization View sample and one Tool-Use View sample, yielding approximately 84,000 view-specific training samples. We ensure that the training data have no textual or visual overlap with any benchmark evaluated in this work and therefore pose no risk of text or image leakage into evaluation.

Agentic RL Objective. For each task $x = ( I , q , y ^ { * } ) \sim \mathcal { D } _ { \mathrm { R L } }$ , the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ samples a group of G complete interactions $\{ \xi _ { i } \} _ { i = } ^ { G }$ through live interaction with the spatial tools, and a task verifier assigns each interaction the outcome reward $R _ { i } = V ( y _ { i } , y ^ { * } )$ . Most training tasks are multiple-choice questions, for which a deterministic parser extracts the final answer choice and assigns reward 1 if it matches the ground truth and 0 otherwise. For BOPASK-Trajectory and BOPASK-Object-Rearrangement, we directly use the per-sample point-set score defined in Equation (3) as the trajectory reward. We maximize the GRPO [39] objective using the asymmetric Clip-Higher bounds and token-level policy-gradient reduction from DAPO [58]:

$$
\begin{array} { l } { \mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { { x } \sim \mathcal { D } _ { \mathrm { R L } } , \{ \xi _ { i } \} _ { i = 1 } ^ { G } \sim \mu _ { \mathrm { o l d } } } ( \cdot | I , q ) } \\ { \displaystyle \left[ \frac { 1 } { \sum _ { i = 1 } ^ { G } | \mathcal { T } _ { i } | } \sum _ { i = 1 } ^ { G } \sum _ { \ell \in \mathcal { T } _ { i } } \operatorname* { m i n } \Bigl ( \rho _ { i , \ell } ( \theta ) \hat { A } _ { i , \ell } , \mathrm { c l i p } ( \rho _ { i , \ell } ( \theta ) , 1 - \varepsilon _ { \mathrm { l o w } } , 1 + \varepsilon _ { \mathrm { h i g h } } ) \hat { A } _ { i , \ell } \Bigr ) \right] . } \end{array}\tag{1}
$$

Here, $\mu _ { \theta _ { \mathrm { o l d } } }$ is the rollout distribution induced by the old policy and tool execution, and $\tau _ { i }$ contains only model-generated token positions in ξ —reasoning, tool calls, and the final answer—excluding returned tool outputs. For model-generated token $w _ { i , \ell }$ with interaction history $h _ { i , \ell } .$ , the token-level importance ratio is $\rho _ { i , \ell } ( \theta ) = \pi _ { \theta } ( w _ { i , \ell } \mid I , q , h _ { i , \ell } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( w _ { i , \ell } \mid I , q , h _ { i , \ell } )$ . The trajectory-level reward is broadcast to every modelgenerated token in the same interaction. The resulting group-relative advantage is

$$
\hat { A } _ { i , \ell } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } , \qquad \ell \in \mathcal { T } _ { i } ,\tag{2}
$$

Training Configurations. Tables 4 and 5 summarize the configurations shared by Cold-Start SFT and capability internalization, and the agentic RL configuration, respectively. The three initial models use the same hyperparameters. Cold-Start SFT and capability internalization both use full-parameter fine-tuning. Capability internalization uses the same settings as Cold-Start SFT, with the Tool-Use View loss weight set to $\lambda = 0 . 5$ . Training uses a fixed random seed of 42.

<table><tr><td>Category</td><td>Configuration</td></tr><tr><td rowspan="5">Optimization</td><td>Optimizer: AdamW</td></tr><tr><td>Weight Decay: 0.1</td></tr><tr><td>Learning Rate:  $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR Scheduler: Cosine</td></tr><tr><td>Warmup Steps: 10</td></tr><tr><td rowspan="4">Training</td><td>Fine-Tuning Method: Full-Parameter</td></tr><tr><td>Batch Size: 64</td></tr><tr><td>Context Length: 32768</td></tr><tr><td>Epochs: 2</td></tr></table>

Table 4 Training configuration shared by Cold-Start SFT and capability internalization.
<table><tr><td>Category</td><td>Configuration</td></tr><tr><td>Algorithm</td><td>Group Size: 8 Clip Ratio Low: 0.20 Clip Ratio High: 0.28 Dual-Clip C: 3.0 Entropy Coef: 0</td></tr><tr><td>Rollout</td><td>Maximum Input Length: 32768 Maximum Output Length: 32768 Tool-Call Budget: 10 Temperature: 1.0 Top-P: 1.0 Top-K: -1</td></tr><tr><td>Training</td><td>Training Batch Size: 128 Mini Batch Size: 32 Optimizer: AdamW Learning Rate:  $1 \times 1 0 ^ { - 6 }$  Weight Decay: 0.01 Gradient Clip: 1.0 Training Precision: FP16</td></tr></table>

Table 5 Agentic RL training configuration.

## B.2 Detailed Evaluation Settings

Evaluation Benchmarks. We evaluate SpatialCLI on SpatialCLI-Bench, MindCube [57], selected subsets of MMSI [54], DA-2K [53], and selected subsets of BOPASK [2]. Specifically, we use Motion-Cam and Pos-Cam-Cam from MMSI and Trajectory and Object-Rearrangement from BOPASK. For the two BOPASK subsets, we use the same continuous point-set score during agentic RL and evaluation. Model outputs and references express 2D points on a 0–999 image-coordinate scale; before scoring, we divide both coordinates by 1000. Let $\hat { \mathcal { P } }$ and ${ \mathcal { P } } ^ { * }$ be the resulting predicted and reference point sets. Their symmetric Chamfer distance and per-sample score are

$$
\begin{array} { l } { \displaystyle d _ { \mathrm { C D } } ( \hat { \mathcal { P } } , \mathcal { P } ^ { * } ) = \frac { 1 } { 2 } \left[ \frac { 1 } { | \hat { \mathcal { P } } | } \sum _ { \hat { p } \in \hat { \mathcal { P } } } \operatorname* { m i n } _ { \hat { p } ^ { * } \in \mathcal { P } ^ { * } } \| \hat { p } - p ^ { * } \| _ { 2 } + \frac { 1 } { | \mathcal { P } ^ { * } | } \sum _ { p ^ { * } \in \mathcal { P } ^ { * } } \operatorname* { m i n } _ { \hat { p } \in \hat { \mathcal { P } } } \| p ^ { * } - \hat { p } \| _ { 2 } \right] , } \\ { \displaystyle s _ { \mathrm { B O P A S K } } ( \hat { \mathcal { P } } , \mathcal { P } ^ { * } ) = \operatorname* { m a x } \left( 0 , 1 - \frac { d _ { \mathrm { C D } } ( \hat { \mathcal { P } } , \mathcal { P } ^ { * } ) } { 0 . 1 5 } \right) . } \end{array}\tag{3}
$$

If an output contains no valid predicted points, its per-sample score is zero. For each BOPASK subset, we report the mean per-sample score as a percentage. Under $\mathrm { w / o }$ Tools inference, no tools are registered and no system prompt is inserted; under $\mathrm { w } / $ Tools inference, all evaluated models are provided with the same four spatial tools used to train SpatialCLI. We parse final answers with benchmark-specific deterministic parsers and treat malformed or missing final answers as incorrect.

Evaluation Configurations. Unless a benchmark requires a diferent oficial setting, we use temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, a presence penalty of 1.5, a repetition penalty of 1.0, and a maximum generation length of 40,960 tokens. All task-performance evaluations permit at most ten tool calls per sample. We report the mean over three evaluation runs, without fixing evaluation seeds.

## B.3 Hardware and Software Environment

All experiments were conducted on a cluster whose nodes were equipped with 184 Intel Xeon CPU cores, 1.8 TiB of system memory, and 16 PPU-ZW810E accelerators, each with approximately 96 GB of device memory. Both SFT and RL used four nodes. The software environment used Ubuntu 24.04.2 LTS, a CUDA 12.9-compatible toolchain, Python 3.12.3, PyTorch 2.9.0, Transformers 5.5.4, vLLM 0.18.0+ppu2.0.0, verl 0.9.0.dev0, and Ray 2.49.2.

## C SpatialCLI-Bench

## C.1 Scope and Sample Format

SpatialCLI-Bench is a 516-example English six-choice visual question answering benchmark for evaluating tool-grounded compositional spatial reasoning. Each example contains one or two images, one question, and six answer choices with a unique correct answer. Each example has a private executable oracle tool plan of at most five steps, which is used for construction and auditing but is not exposed in the public question. Every example jointly evaluates at least two of three capability groups: grounding and 2D region relations (G), metric depth (D), and object pose or cross-view camera motion (P).

## C.2 Data Provenance and Composition

Every example has a unique candidate ID and a unique upstream visual-source ID. No complete visual unit is duplicated, every example is associated with exactly one visual source, and both images of a two-image example come from the same source. The visual sources are MindCube [57], HOPE [44], LINEMOD [16], YCB-V [49], HANDAL [12], CA-1M [23], OpenImages [22], and RefSpatial-Blender [62]. The 195 MindCubederived examples reuse image data from the existing MindCubeBench evaluation set. Table 6 gives the complete composition by visual source.

<table><tr><td>Visual Source</td><td>Count</td><td>Share</td></tr><tr><td>MindCube</td><td>195</td><td>37.79%</td></tr><tr><td>HOPE</td><td>110</td><td>21.32%</td></tr><tr><td>LINEMOD</td><td>46</td><td>8.91%</td></tr><tr><td>YCB-V</td><td>5</td><td>0.97%</td></tr><tr><td>HANDAL</td><td>1</td><td>0.19%</td></tr><tr><td>CA-1M</td><td>116</td><td>22.48%</td></tr><tr><td>OpenImages</td><td>29</td><td>5.62%</td></tr><tr><td>RefSpatial-Blender</td><td>14</td><td>2.71%</td></tr></table>

Table 6 Visual-source provenance of the 516 SpatialCLI-Bench examples.

The benchmark contains 243 single-image examples (47.09%) and 273 two-image examples (52.91%).

## C.3 Capability Composition and Oracle Plans

Table 7 summarizes the joint capability composition. Grounding appears in 414 examples (80.23%), depth in 403 (78.10%), and pose in 402 (77.91%), giving nearly balanced marginal coverage. All two-image examples include P through cross-view camera-motion or pose evidence.

<table><tr><td>Capability Combination</td><td>Count</td><td>Share</td></tr><tr><td>GD</td><td>114</td><td>22.09%</td></tr><tr><td>GP</td><td>113</td><td>21.90%</td></tr><tr><td>DP</td><td>102</td><td>19.77%</td></tr><tr><td>GDP</td><td>187</td><td>36.24%</td></tr></table>

Table 7 Joint capability composition of SpatialCLI-Bench.

The 516 private oracle plans contain 1,964 tool steps, averaging 3.81 steps per example. There are 12 twostep, 156 three-step, 268 four-step, and 80 five-step plans. Across all plans, query\_segment, query\_depth, query\_pose, and query\_locate are invoked 862, 471, 404, and 227 times, respectively.

## C.4 Multi-Stage Construction Pipeline

FrontierMultimodalModelPre-Annotation. A frontier multimodal model first builds a conservative structured inventory of uniquely referable visible entities for each candidate image or image pair. The inventory records each entity’s image index, visible description and attributes, a short tool query, and whether the entity is uncertain; rejected entities and their rejection reasons are recorded separately. We use Gemini 3.1 Pro [11] for this stage and retain only inventories with at least two reliable entities, including reliable coverage of both images for a two-image candidate.

Specialist-Vision-Model Fine Annotation. Specialist vision models then collect capability-aligned evidence for up to four reliable entities. Locate and Segment establish entity positions and regions; Depth is queried at a representative point derived from localization or segmentation; Pose estimates cross-view camera motion for two-image candidates or object orientation for up to two entities in single-image candidates. Each response is stored with its exact tool arguments, and a candidate is rejected when a required service returns missing, explicit-error, or empty evidence.

Frontier Multimodal Model Fine Annotation. Conditioned on the entity inventory and collected specialist evidence, Gemini 3.1 Pro generates the English question, one correct answer, five plausible distractors, a private evidence-grounded rationale, and a minimal reproducible oracle plan. The generation prompt requires every factual clause in the correct answer to be supported by specialist evidence and each distractor to be false for a specific evidence-grounded reason. Deterministic validation then enforces six unique choices, valid image indices and step dependencies, the five-step budget, exact capability–plan correspondence, and an exact match between every oracle step and a collected tool call; the choices are shufled deterministically only after validation.

## C.5 Independent Human Review and Filtering Yield

The automated construction pipeline described above initially generates 720 candidate questions. Human experts then answer each question independently without access to the answer generated by Gemini 3.1 Pro. The independent review is conducted by two graduate students with research backgrounds in embodied intelligence. The 516 candidates for which all independent human answers agree with the generated answer are retained in SpatialCLI-Bench, for an overall retention rate of 71.67%; the remaining 204 candidates are rejected because of answer disagreement. The supporting tool evidence and minimal executable tool plan are retained only as private audit metadata.

## D Spatial Tool Interfaces and Implementation

## D.1 Coordinate and Serialization Convention

Following the JSON grounding convention used by Qwen vision-language models [1], SpatialCLI represents points and boxes with point\_2d and bbox\_2d. All coordinates are integers in [0, 999], with the origin at the upper-left corner. We introduce polygon\_2d as a SpatialCLI-specific extension for segmentation masks.

A bounding box is serialized as $[ x _ { \mathrm { m i n } } , y _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } , y _ { \mathrm { m a x } } ]$ . A point is serialized as [x, y]. A polygon is represented by an ordered list of boundary vertices, and multiple connected components are represented as a list of polygons. The Segment tool extracts external mask contours and represents each connected component with at most 16 vertices, targeting a rasterized mask IoU of 0.97 whenever feasible within this vertex budget. The polygon is rasterized at the original resolution whenever mask IoU is required.

## D.2 Spatial Tool Interfaces

The agent sees only four model-level interfaces: query\_locate, query\_segment, query\_depth, and query\_- pose; it does not select the underlying specialist vision models. The service layer encapsulates six specialist vision models: Locate Anything [46] and Grounding DINO [28] for fused localization, SAM 3 [4] for segmentation, Depth Anything 3 [25] for metric depth, Orient Anything V2 [47] for object orientation, and VGGT [45] for multi-view camera motion. All four interfaces share the XML-wrapped JSON calling protocol and accept an optional image\_indices field with one-based image or frame indices. At runtime, only responses matched by the fixed regular expression for this protocol are treated as tool calls; all unmatched responses, including those with malformed tool-call syntax, are treated as final answers. Their complete registrations are provided in Boxes D.1–D.4.

Locate. Locate fuses detections from Locate Anything and Grounding DINO, while exposing only the fused boxes and center points to the agent. Both backends process the same image–query pair in parallel. Grounding DINO detections are filtered at a confidence threshold of 0.30 and suppressed using box NMS with an IoU threshold of 0.50. For a Locate Anything box $b _ { i } ^ { L }$ , we select the unmatched Grounding DINO box $b _ { j ^ { * } ( i ) } ^ { D }$ with the largest IoU. If their IoU is at least 0.50, the cross-confirmed box is

$$
b _ { i } ^ { F } = \mathrm { r o u n d } \left( \frac { 2 b _ { i } ^ { L } + b _ { j ^ { * } ( i ) } ^ { D } } { 3 } \right) ,\tag{4}
$$

which gives Locate Anything twice the coordinate weight of Grounding DINO. Unmatched predictions from either backend are retained to preserve recall. The combined candidates are greedily deduplicated: a candidate $b _ { i }$ is removed if a retained box $b _ { j }$ satisfies

$$
\mathrm { I o U } ( b _ { i } , b _ { j } ) \geq 0 . 7 5 \quad \mathrm { o r } \quad \frac { | b _ { i } \cap b _ { j } | } { \operatorname* { m i n } ( | b _ { i } | , | b _ { j } | ) } \geq 0 . 9 0 .\tag{5}
$$

Here, |b| denotes the area of box b. The second criterion removes near-contained duplicates that may have a modest box IoU because of their diferent areas. Backend identities, confidence scores, and cross-model IoUs remain internal.

Box D.1: Locate Tool Registration   
1 {   
2 "name": "query\_locate",   
3 "description": "Locate every visible instance matching a short language description. Use this when exact object   
positions, centers, boxes, counts, or left/right order depend on reliable grounding. Query a concrete visual category   
and attributes only. For ’closest’, ’farthest’, ’leftmost’, or ’second’ questions, do not put that relation in the query:   
locate all instances of the base category, then compare their returned centers or query their depths. All benchmark   
geometry and tool coordinates use the same normalized 0-999 image space. The tool returns bbox\_2d and point\_2d   
in that space; pass point\_2d directly to query\_depth when needed. It may return zero, one, or several instances; do   
not assume the first instance is the only one. Prefer this tool for answers that require explicit path points, object   
markers, or boxes. Answer simple categorical left/right yes-no questions directly when visually clear. It cannot infer   
five-point grasp-finger geometry or projected 3D cuboid corners from a 2D box, so do not treat its center or box   
corners as those answers. Tools are optional. In a multi-image or sampled-video task, omit image\_indices to use all   
images, or provide one or more 1-based image/frame numbers to select a subset.",   
"parameters": {   
"type": "object",   
"properties": {   
"query": {   
"type": "string",   
9 "description": "Concrete visual category/attributes, without positional or depth superlatives."   
10 },

```jsonl
11 "image_indices": {
12 "type": "array",
13 "items": {"type": "integer", "minimum": 1},
14 "minItems": 1,
15 "uniqueItems": true,
16 "description": "Optional 1-based image/frame numbers. Omit to use the only image in a single-image task or
all images jointly in a multi-image task."
17 }
18 },
19 "required": ["query"]
20 }
21 }
22
23 Example:
24 <tool_call>
25 {"name": "query_locate", "arguments": {"query": "red mug", "image_indices": [1]}}
26 </tool_call>
27
28 Possible return:
29 {"count": 1, "result": [{"bbox_2d": [532, 470, 628, 675], "point_2d": [580, 572]}]}
```

Segment. Segment uses a short text query to obtain SAM 3 masks and returns each reliable instance as a box, center point, and one or more polygonal connected components. Predictions with confidence at most 0.30 are removed, after which confidence-ordered mask suppression removes a candidate whose mask IoU with any retained instance is at least 0.90. For each retained mask, we extract pixel-level external contours, discard connected components smaller than 16 $\mathrm { \ p i x e l s } ^ { 2 }$ , and order the remaining components by area. Because only external contours are serialized, holes inside a component are not represented explicitly.

Contour vertices are quantized into the shared integer coordinate space; adjacent duplicate vertices and an explicit closing vertex are removed. For a closed contour $\Gamma _ { k }$ with perimeter $L _ { k }$ , we apply Douglas–Peucker simplification with tolerance $\varepsilon _ { k } ( r ) = r L _ { k }$ over

$$
r \in \{ 0 \} \cup \mathrm { L o g S p a c e } ( 1 0 ^ { - 5 } , 0 . 2 5 , 7 2 ) .\tag{6}
$$

Each candidate polygon P is first quantized into the final integer coordinate space, mapped back to the original resolution, and rasterized as $\mathcal { R } ( P )$ . Its fidelity to the corresponding mask component $M _ { k }$ is

$$
Q ( P ; M _ { k } ) = \mathrm { I o U } \left( M _ { k } , \mathcal { R } ( P ) \right) = \frac { \left| M _ { k } \cap \mathcal { R } ( P ) \right| } { \left| M _ { k } \cup \mathcal { R } ( P ) \right| } .\tag{7}
$$

We choose the candidate with the fewest vertices subject to

$$
Q ( P ; M _ { k } ) \ge 0 . 9 7 , \qquad 3 \le | P | \le 1 6 ,\tag{8}
$$

breaking ties in favor of higher IoU. We then greedily attempt to remove low-contribution vertices in ascending order of

$$
a _ { v } = \left| \left( p _ { v } - p _ { v - 1 } \right) \times \left( p _ { v + 1 } - p _ { v - 1 } \right) \right| ,\tag{9}
$$

accepting a removal only if the rasterized IoU remains at least 0.97. If no polygon reaches the target within the 16-vertex budget, the bounded candidate with the highest IoU is returned instead; thus 0.97 is a target rather than an unconditional guarantee. The returned point\_2d is the center of the SAM 3 predicted box, not the mask centroid, and need not lie inside the mask.

## Box D.2: Segment Tool Registration

1 {   
2 "name": "query\_segment",   
3 "description": "Segment every visible instance matching a short language description. Use this for exact object   
boundaries, occupied regions, shapes, containment, overlap, or placement areas; use query\_locate instead when   
centers, counts, or simple left/right order are suficient. Pass a concrete visual category/attributes only, without   
closest/leftmost/second relations, and do not pass a box. All benchmark geometry and tool coordinates use the same   
normalized 0-999 image space. Each returned instance contains bbox\_2d, point\_2d, and polygon\_2d in that space.

```jsonl
The point is the mask bounding-box center and is easiest to use for paths or object markers. polygon_2d is a list
because one mask may have disconnected parts. Prefer this tool for object markers, occupied regions, and
boundary-sensitive coordinate tasks. Do not use a mask center or polygon as five-point grasp-finger geometry or
projected 3D cuboid corners. Answer categorical yes-no and relative-depth questions directly unless an exact
boundary is genuinely required. Tools are optional. In a multi-image or sampled-video task, omit image_indices to
use all images, or provide one or more 1-based image/frame numbers to select a subset.",
4 "parameters": {
"type": "object",
"properties": {
"query": {
"type": "string",
9 "description": "Short description of the target instances."
10 },
11 "image_indices": {
12 "type": "array",
13 "items": {"type": "integer", "minimum": 1},
14 "minItems": 1,
15 "uniqueItems": true,
16 "description": "Optional 1-based image/frame numbers. Omit to use the only image in a single-image task or
all images jointly in a multi-image task."
17 }
18 },
19 "required": ["query"]
20 }
21 }
22
23 Example:
24 <tool_call>
25 {"name": "query_segment", "arguments": {"query": "red mug"}}
26 </tool_call>
27
28 Possible return:
29 {"count": 1, "result": [{"bbox_2d": [532, 470, 628, 675], "point_2d": [580, 572], "polygon_2d": [[[548, 475], [605,
472], [623, 516], [609, 670]]]}]}
```

Depth. Depth uses the DA3NESTED-GIANT-LARGE-1.1 metric backend to return camera-axis distance in meters at one or more selected image points. Inference uses a processing resolution of 504 pixels, and each query reads its corresponding depth-map pixel directly without spatial smoothing. Metric-scale alignment is made deterministic across workers. The returned depth\_m is a monocular estimate of camera-axis Z-depth, rather than Euclidean range from the camera center or a physical sensor measurement.

Box D.3: Depth Tool Registration   
1 {   
2 "name": "query\_depth",   
3 "description": "Query depth at one or more image points. Pass every point needed for a closer/farther,   
front/behind, or distance comparison in one call. Point coordinates are normalized to 0-999 and the number of points   
is unrestricted. For named objects, first use query\_locate to obtain their center points, then query all centers   
together. Each result contains depth\_m, an estimated camera-axis distance in meters; larger means farther.   
Monocular depth estimates can be noisy: answer directly when the relative depth is already visually clear, and call   
this tool only when depth is essential and genuinely ambiguous. In a multi-image or sampled-video task, omit   
image\_indices to use all images, or provide one or more 1-based image/frame numbers to select a subset.",   
"parameters": {   
"type": "object",   
"properties": {   
"points": {   
"type": "array",   
"minItems": 1,   
10 "description": "One or more [x, y] points in normalized 0-999 coordinates.",   
11 "items": {   
12 "type": "array",   
13 "items": {"type": "number"},   
14 "minItems": 2,   
15 "maxItems": 2   
16 }

17 },   
18 "image\_indices": {   
19 "type": "array",   
20 "items": {"type": "integer", "minimum": 1},   
21 "minItems": 1,   
22 "uniqueItems": true,   
23 "description": "Optional 1-based image/frame numbers. Omit to use the only image in a single-image task or   
all images jointly in a multi-image task."   
24 }   
25 },   
26 "required": ["points"]   
27 }   
28 }   
29   
30 Example:   
31 <tool\_call>   
32 {"name": "query\_depth", "arguments": {"points": [[320, 480], [700, 510]]}}   
33 </tool\_call>   
34   
35 Possible return:   
36 {"result": [{"point\_2d": [320, 480], "depth\_m": 2.41}, {"point\_2d": [700, 510], "depth\_m": 4.12}]}

Pose. Pose routes a named-object query to Orient Anything V2 and the exact query camera motion to the multi-view VGGT camera-pose backend. For object orientation, Pose first invokes the fused Locate interface and enlarges every detected box by 10% of its width and height on each side, with a minimum padding of two pixels. The crop is resized with its aspect ratio preserved, padded to $5 1 8 \times 5 1 8$ , and processed by Orient Anything V2, which predicts azimuth, elevation, and roll using 360, 180, and 360 discrete bins, respectively. Only the horizontal orientation is exposed to the agent. For azimuth $\varphi _ { \mathrm { a z } }$ , its eight-direction sector is

$$
\nu _ { \mathrm { a z } } = \left\lfloor \frac { \left( \varphi _ { \mathrm { a z } } + 2 2 . 5 ^ { \circ } \right) \mathrm { m o d } 3 6 0 ^ { \circ } } { 4 5 ^ { \circ } } \right\rfloor .\tag{10}
$$

The sector is converted into the complementary fields visible\_side and facing\_direction\_camera; in the latter, front points into the image and back points toward the camera.

For camera motion, VGGT jointly estimates world-to-camera extrinsics $E _ { i } = \left[ R _ { i } \mid t _ { i } \right]$ for the selected views in their given order. The camera center and the adjacent-view translation expressed in the source-camera frame are

$$
C _ { i } = - R _ { i } ^ { \top } t _ { i } , \qquad \Delta _ { i } = R _ { i } ( C _ { i + 1 } - C _ { i } ) , \qquad \hat { \Delta } _ { i } = \frac { \Delta _ { i } } { \| \Delta _ { i } \| _ { 2 } } \mathrm { ~ i f ~ } \| \Delta _ { i } \| _ { 2 } \geq 0 . 0 0 2 .\tag{11}
$$

Before serializing $\hat { \Delta } _ { i }$ as signed axes, we negate its Y and Z components to obtain the exposed convention of +X right, +Y up, and −Z forward. The relative view rotation is

$$
R _ { i \to i + 1 } = R _ { i + 1 } R _ { i } ^ { \top } .\tag{12}
$$

Translation and view rotation are summarized separately using these right-handed camera axes. A translation norm below 0.002 is treated as the same position, while rotations below $2 ^ { \circ }$ are suppressed. For $n _ { \mathrm { v i e w } }$ selected images, the tool returns $n _ { \mathrm { v i e w } } \mathrm { ~ - ~ } 1$ summaries for adjacent pairs rather than all pairwise relations. Only scale-independent motion directions and signed axes are exposed; metric translation magnitudes, continuous angles, and camera matrices remain internal.

Box D.4: Pose Tool Registration   
1 {   
2 "name": "query\_pose",   
3 "description": "Query a named object’s facing direction in one image, or camera motion across images. For an   
object, pass only a short visible description. The result returns visible\_side (the object’s side facing the camera) and   
facing\_direction\_camera, where front means deeper into the image, back means toward the camera, and left/right   
are the viewer’s image sides. For viewpoint change, pass exactly query=’camera motion’. Each result describes   
to\_image relative to from\_image. Use position only for questions about the camera’s shooting location or movement;   
use view\_rotation only for questions about where the view turns. direction directly matches ordinary words such as   
forward-right, left, up, or clockwise. position.axes uses signed right-handed coordinates where +X=right, +Y=up,

```json
and -Z=forward. view_rotation.axes is the signed camera-pose rotation and should be used directly only when
answer choices explicitly mention positive/negative X/Y/Z axes; for ordinary turn words use view_rotation.direction.
same-position means no reliable translation. Omit image_indices for image 2 relative to image 1; reverse them only
when the question explicitly asks for image 1 relative to image 2. Use object mode only for object orientation and
camera-motion mode only for actual viewpoint change; otherwise answer directly. In a multi-image or sampled-video
task, omit image_indices to use all images, or provide one or more 1-based image/frame numbers to select a subset.",
4 "parameters": {
5 "type": "object",
6 "properties": {
7 "query": {
8 "type": "string",
9 "description": "Object description, or ’camera motion’."
10 },
11 "image_indices": {
12 "type": "array",
13 "items": {"type": "integer", "minimum": 1},
14 "minItems": 1,
15 "uniqueItems": true,
16 "description": "Optional 1-based image/frame numbers. Omit to use the only image in a single-image task or
all images jointly in a multi-image task."
17 }
18 },
19 "required": ["query"]
20 }
21 }
22
23 Object-orientation example:
24 <tool_call>
25 {"name": "query_pose", "arguments": {"query": "red mug", "image_indices": [1]}}
26 </tool_call>
27
28 Possible return:
29 {"result": [{"image_index": 1, "bbox_2d": [532, 470, 628, 675], "visible_side": "front-right",
"facing_direction_camera": "back-left"}]}
30
31 Camera-motion example:
32 <tool_call>
33 {"name": "query_pose", "arguments": {"query": "camera motion", "image_indices": [1, 2]}}
34 </tool_call>
35
36 Possible return:
37 {"result": [{"from_image": 1, "to_image": 2, "position": {"direction": "forward-right", "axes": ["+X", "-Z"]},
"view_rotation": {"direction": "right", "axes": ["+Y"], "dominant_axis": "+Y"}}]}
```

DeploymentandRuntime. All six specialist-model services are jointly deployed on two GPUs. Compared with the central VLM, these specialist backends are generally compact and introduce modest runtime overhead. During long-running RL training, the observed mean latency per spatial-tool call was 2.916 seconds.

Cache Design. Each specialist backend uses a content-addressed two-tier cache: a bounded in-process leastrecently-used (LRU) cache backed by a persistent disk cache under outputs/cache/. Each key is a SHA-256 digest derived from the input-image content hashes rather than file paths, together with the request text or mode, the relevant model or checkpoint fingerprint, and all inference and post-processing settings that can afect the returned result. Consequently, identical image content can be reused across path aliases, whereas a changed image, model checkpoint, query, backend, or output-afecting parameter produces a diferent key. Lookup checks memory before disk and promotes a disk hit into the in-memory LRU. On a miss, the service executes the specialist model and atomically persists the successful result; failed requests are not cached, and per-process locks prevent concurrent duplicate inference.

The cache granularity follows each backend’s computation. Depth caches the full metric-depth map, intrinsics, confidence, image size, and scale factor per image, allowing diferent queried points on the same image to reuse one forward pass. Segment maintains both a two-entry image-state cache, which reuses the SAM 3 image encoding across text queries, and 128-entry result caches keyed by image, query, and segmentation settings. Locate and Pose each maintain 128-entry result LRUs; their disk entries store the corresponding structured outputs, while Pose additionally persists raw camera matrices for camera-motion requests. Pose keys distinguish the ordered image sequence, object-or-camera mode, selected camera backend, resolution, model fingerprints, and, for object orientation, the localization service. The equal-weight macro-average of the reported Depth, Segment, Pose, and Locate rates is approximately 58.8%; a request-weighted rate would additionally depend on the tool-call composition. The bounded LRU capacities limit memory growth, while cache hits bypass repeated specialist forward passes and therefore add negligible accelerator compute beyond lookup and output serialization. Table 8 reports the observed cache hit rates for each spatial tool.

<table><tr><td>Tool or Mode</td><td>Specialist Backend / Scope</td><td>Hit Rate (%)</td></tr><tr><td>Depth</td><td>Depth Anything 3</td><td>≈ 99.2</td></tr><tr><td>Segment</td><td>SAM 3</td><td>≈ 49.4</td></tr><tr><td>Pose</td><td>All Pose requests</td><td>≈ 58.2</td></tr><tr><td>Camera motion</td><td>VGGT, successful requests</td><td>100.0</td></tr><tr><td>Object orientation</td><td>Orient Anything V2</td><td>≈ 43.8</td></tr><tr><td>Locate</td><td>All Locate requests</td><td>≈ 28.5</td></tr><tr><td>Macro average</td><td>Four reported top-level paths</td><td>≈ 58.8</td></tr></table>

Table 8 Observed cache hit rates for spatial-tool requests. Pose is also broken down by its camera-motion and objectorientation backends. The macro average uses the reported Depth, Segment, overall Pose, and Locate rates.

## E Capability Internalization Metric

## E.1 Evaluation Data and Capability Coverage

The CII validation suite contains 1,000 held-out examples, balanced across five query types: 200 each for Locate, Segment, Depth, object orientation, and camera motion. The last two query types correspond to the two modes of the Pose interface. Each example contains the required image or image pair, a capability-specific request, and a structured reference output.

## E.2 Reference Construction and Evaluation Protocol

Candidate examples are converted into capability-specific requests, and the corresponding registered specialist tool is executed to produce each structured reference output. After the independent human verification described below, each retained tool output is serialized in the registered tool’s JSON result format. At evaluation time, the model receives only the image input and an instruction to return the requested tool format; no spatial tool is registered or executed. We use temperature 0.0, with a maximum of 512 output tokens for Locate, Depth, and both Pose modes, and 4,096 tokens for Segment because polygon serialization is substantially longer. A deterministic parser extracts the structured JSON response; a response from which no valid JSON object can be recovered receives zero similarity.

## E.3 Independent Human Verification

Two human experts independently verify each capability-specific request and its tool-produced reference against the image input. Neither expert can see the other’s judgment, and each checks that the request is unambiguous and the specialist-tool output is valid and consistent with the visual input. We retain a candidate only when both experts approve it; all other candidates are discarded. This independent agreement rule filters ambiguous requests and unreliable tool outputs before CII evaluation.

## E.4 Similarity Functions and Aggregation

We evaluate whether a model can reproduce spatial-tool outputs without executing external tools. For a capability type κ ∈ {Locate, Segment, Depth}, let $\hat { o } _ { i }$ be the model prediction and $o _ { i } ^ { * }$ the corresponding

spatial-tool output. For each capability, samples are indexed locally by $i = 1 , \ldots , N _ { \kappa }$ . We define

$$
\mathrm { C I I } _ { \kappa } = \frac { 1 0 0 } { N _ { \kappa } } \sum _ { i = 1 } ^ { N _ { \kappa } } { s _ { \kappa } ( \hat { o } _ { i } , o _ { i } ^ { * } ) } ,
$$

where $N _ { \kappa }$ is the number of samples for capability κ and $s _ { \kappa } \in [ 0 , 1 ]$ is a capability-specific similarity function. Because the Pose interface supports both object orientation and camera motion, we evaluate these two modes separately and macro-average them as defined below. Unless stated otherwise, a malformed prediction or one missing a required field receives zero similarity.

Locate. Let $\hat { B _ { i } }$ and $B _ { i } ^ { * }$ be the predicted and reference box sets. We obtain a one-to-one Hungarian matching $\mathcal { M } _ { i } ^ { \mathrm { b o x } }$ that maximizes total pairwise IoU and define

$$
s _ { \mathrm { L o c a t e } } \big ( \hat { o } _ { i } , o _ { i } ^ { * } \big ) = \frac { \sum _ { ( m , n ) \in { \mathcal { M } _ { i } ^ { \mathrm { b o x } } } } \mathrm { I o U } \big ( \hat { b } _ { i , m } , b _ { i , n } ^ { * } \big ) } { \operatorname* { m a x } \ ( | \hat { { \mathcal { B } _ { i } } } | , | { \mathcal { B } _ { i } ^ { * } } | ) } .
$$

This denominator assigns zero contribution to every unmatched prediction or reference box. If both sets are empty, the similarity is defined as one.

Segment. Let $\hat { S } _ { i }$ and $\boldsymbol { S } _ { i } ^ { * }$ be the predicted and reference instance-mask sets after rasterizing each polygon\_2d at the original image resolution. Using the IoU-maximizing Hungarian matching ${ \mathcal { M } } _ { i } ^ { \mathrm { m a s k } }$ , we define

$$
s _ { \mathrm { S e g m e n t } } ( \hat { o } _ { i } , o _ { i } ^ { * } ) = \frac { \sum _ { ( j , k ) \in \mathcal { M } _ { i } ^ { \mathrm { m a s k } } } \mathrm { I o U } ( \hat { m } _ { i , j } , m _ { i , k } ^ { * } ) } { \operatorname* { m a x } ( | \hat { S } _ { i } | , | S _ { i } ^ { * } | ) } .
$$

As in Locate, unmatched instances contribute zero, and two empty sets receive similarity one.

Depth. Let $\mathcal { P } _ { i }$ be the nonempty set of reference query points with valid metric-depth outputs, and let $\hat { d } _ { i , p }$ and $d _ { i , p } ^ { * }$ be the predicted and reference depths at point $p .$ Predicted entries are matched to reference queries by point\_2d. If a required point is missing or duplicated, or if the prediction contains an unrequested point, the sample receives zero similarity; otherwise, using $\varepsilon _ { \mathrm { d } } = 1 0 ^ { - 6 }$ for numerical stability, we compute

$$
\mathrm { A b s R e l } _ { i } = \frac { 1 } { \vert \mathcal { P } _ { i } \vert } \sum _ { p \in \mathcal { P } _ { i } } \frac { \vert \hat { d } _ { i , p } -  { d _ { i , p } ^ { * } } \vert } { \operatorname* { m a x } (  { d _ { i , p } ^ { * } } , \varepsilon _ { \mathrm { d } } ) } .
$$

Because lower AbsRel indicates better depth prediction, we convert it to a bounded similarity with

$$
s _ { \mathrm { D e p t h } } ( \hat { o } _ { i } , o _ { i } ^ { * } ) = \mathrm { e x p } ( - \mathrm { A b s R e l } _ { i } ) .
$$

Metric depth is evaluated directly without median or scale alignment.

Pose. The Pose interface has two modes whose outputs are discrete spatial directions rather than rotation matrices. For an object-orientation query, we evaluate the canonical facing\_direction\_camera field. We use the following circular angle mapping:

$$
\begin{array} { l l } { { \mathrm { f r o n t : ~ 0 ~ } } } & { { \mathrm { f r o n t \mathrm { - } r i g h t : ~ } \pi / 4 } } \\ { { \mathrm { r i g h t : ~ } \pi / 2 } } & { { \mathsf { b a c k \mathrm { - } r i g h t : ~ } 3 \pi / 4 } } \\ { { \mathsf { b a c k : ~ } \pi } } & { { \mathsf { b a c k \mathrm { - } l e f t : ~ } 5 \pi / 4 } } \\ { { \mathsf { l e f t : ~ } 3 \pi / 2 } } & { { \mathrm { f r o n t \mathrm { - } l e f t : ~ } 7 \pi / 4 } } \end{array}
$$

Let $\hat { \phi } _ { i }$ and $\phi _ { i } ^ { * }$ be the predicted and reference angles, and define their circular distance as

$$
\Delta \phi _ { i } = \operatorname* { m i n } \left( \lvert \hat { \phi } _ { i } - \phi _ { i } ^ { * } \rvert , 2 \pi - \lvert \hat { \phi } _ { i } - \phi _ { i } ^ { * } \rvert \right) .
$$

The object-orientation similarity is

$$
s _ { \mathrm { o b j } , i } = \operatorname* { m a x } ( 0 , \cos \Delta \phi _ { i } ) .
$$

The auxiliary visible\_side field and the returned box are not included in this score; localization is evaluated separately by Locate CII.

For a camera-motion query, we evaluate translation and view rotation separately from the signed axes returned by position.axes and view\_rotation.dominant\_axis. Let vec(α) be the unit vector denoted by signed axis $\alpha ,$ where, for example, vec $( + X ) = ( 1 , 0 , 0 )$ and $\operatorname { v e c } ( - Z ) = ( 0 , 0 , - 1 )$ ), and define

$$
\operatorname { d i r } ( A ) = { \frac { \sum _ { \alpha \in { \mathcal { A } } } \operatorname { v e c } ( \alpha ) } { \left\| \sum _ { \alpha \in { \mathcal { A } } } \operatorname { v e c } ( \alpha ) \right\| _ { 2 } } } .
$$

This mapping is defined only for a nonempty signed-axis set with a nonzero vector sum. Using $A _ { i } ^ { \mathrm { t r a n s } }$ for the translation axes and the singleton ${ \mathcal { A } } _ { i } ^ { \mathrm { r o t } }$ containing the dominant rotation axis, we define

$$
\begin{array} { r c l } { s _ { \mathrm { t r a n s } , i } } & { = } & { \operatorname* { m a x } \left( 0 , \mathrm { d i r } ( \hat { \mathcal { A } } _ { i } ^ { \mathrm { t r a n s } } ) ^ { \top } \mathrm { d i r } ( \mathcal { A } _ { i } ^ { \mathrm { t r a n s } , * } ) \right) , } \\ { s _ { \mathrm { r o t } , i } } & { = } & { \operatorname* { m a x } \left( 0 , \mathrm { d i r } ( \hat { \mathcal { A } } _ { i } ^ { \mathrm { r o t } } ) ^ { \top } \mathrm { d i r } ( \mathcal { A } _ { i } ^ { \mathrm { r o t } , * } ) \right) , } \end{array}
$$

and

$$
s _ { \mathrm { c a m } , i } = \frac { s _ { \mathrm { t r a n s } , i } + s _ { \mathrm { r o t } , i } } { 2 } .
$$

For Pose, an unrecognized direction label or an axis set for which dir(A) is undefined receives zero similarity. Finally, to prevent the more frequent query mode from dominating the metric, we compute

$$
\begin{array} { r c l } { { \displaystyle \mathrm { C I I } _ { \mathrm { o b j } } } } & { { = } } & { { \displaystyle \frac { 1 0 0 } { N _ { \mathrm { o b j } } } \sum _ { i = 1 } ^ { N _ { \mathrm { o b j } } } { s _ { \mathrm { o b j } , i } } , } } \\ { { \displaystyle \mathrm { C I I } _ { \mathrm { c a m } } } } & { { = } } & { { \displaystyle \frac { 1 0 0 } { N _ { \mathrm { c a m } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c a m } } } { s _ { \mathrm { c a m } , i } } . } } \end{array}
$$

$$
\mathrm { C I I _ { P o s e } = \frac { C I I _ { o b j } + C I I _ { c a m } } { 2 } . }
$$

Here, $N _ { \mathrm { o b j } } > 0$ and $N _ { \mathrm { { c a m } } } > 0$ are the numbers of object-orientation and camera-motion samples, respectively. We additionally report the two components separately. The four-capability macro-average reported in the main paper is

$$
\mathrm { C I I } _ { \mathrm { m a c r o } } = \frac { \mathrm { C I I } _ { \mathrm { L o c a t e } } + \mathrm { C I I } _ { \mathrm { S e g m e n t } } + \mathrm { C I I } _ { \mathrm { D e p t h } } + \mathrm { C I I } _ { \mathrm { P o s e } } } { 4 } .
$$

All capability-specific similarities lie in [0, 1] and equal one for an exact match; lower values indicate greater divergence from the spatial-tool output. Every capability-level CII and $\mathrm { C I I } _ { \mathrm { m a c r o } }$ therefore lie in [0, 100], with a higher value indicating stronger internalization.

## F Additional Experiments

## F.1 Comparison with Direct Fine-Tuning

Table 9 compares conventional Direct fine-tuning with inference-time tool augmentation, agentic fine-tuning, and capability internalization on SpatialCLI-Bench using Qwen3-VL-8B-Instruct. Every variant is evaluated both $\mathrm { w / o }$ Tools and $\mathrm { w } / $ Tools. For Direct variants, $\mathrm { w } / $ Tools evaluation exposes the same spatial-tool interfaces at inference time, even though their training does not include explicit tool-interaction supervision.

<table><tr><td rowspan="2">Stage</td><td colspan="3">Conventional Direct Fine-Tuning</td><td colspan="3">Agentic / SpatialCLI</td></tr><tr><td>Variant</td><td>w/o Tools</td><td>w/ Tools</td><td>Variant</td><td>w/o Tools</td><td>w/Tools</td></tr><tr><td>Inference Only</td><td>Initial Model</td><td>35.3</td><td>66.5</td><td>Initial Model</td><td>35.3</td><td>66.5</td></tr><tr><td>SFT</td><td>SFT  $\mathrm { w } / \mathrm { o }$  Tools</td><td>41.3</td><td>40.3</td><td>SFT w/ Tools</td><td>41.1</td><td>86.4</td></tr><tr><td> $\mathrm { R L ~ w } / \mathrm { o } ~ \mathrm { S F T }$ </td><td>RL  $\mathrm { w } / \mathrm { o }$  Tools  $\mathrm { w } / \mathrm { o } \ \mathrm { S F T }$ </td><td>52.7</td><td>68.2</td><td> $\mathrm { R L } \mathrm { w } / \mathrm { T o o l s } \mathrm { w } / \mathrm { o } \mathrm { S F T }$ </td><td>37.2</td><td>90.5</td></tr><tr><td> $\mathrm { R L } + \mathrm { S F T }$ </td><td> $\mathrm { R L } \mathrm { w } / \mathrm { o } \mathrm { T o o l s } + \mathrm { S F T }$ </td><td>51.6</td><td>48.1</td><td> $\mathrm { R L } \mathrm { w } / \mathrm { T o o l s } + \mathrm { S F T }$ </td><td>40.1</td><td>91.3</td></tr><tr><td>Internalization</td><td></td><td></td><td></td><td> $\mathrm { S p a t i a l C L I { - } 8 B }$ </td><td>72.7</td><td>91.3</td></tr></table>

Table 9 Comparison with conventional Direct fine-tuning on SpatialCLI-Bench. Variant names specify training configurations, while $\mathrm { w / o }$ Tools and $\mathrm { w } / $ Tools denote inference settings. SpatialCLI-8B is obtained by Dual-View Capability Internalization after RL $\mathrm { w } / $ Tool $+ \operatorname { S F T }$

The Inference Only row uses the same initial checkpoint on both sides; its Inference $\mathrm { w } / $ Tools result denotes enabling the spatial tools without fine-tuning. Inference-time tool augmentation alone yields a clear improvement $\mathrm { w } / $ Tools, showing that spatial tools efectively compensate for the fine-grained perceptual limitations of the initial model. SFT $\mathrm { w } / $ Tools learns the basic tool-interaction protocol, while RL $\mathrm { w } / $ Tools $\mathrm { w / o }$ SFT optimizes tool selection and termination from task feedback without Cold-Start SFT. SFT $\mathrm { w } / $ Tools raises the w/ Tools score from 66.5 to 86.4, and RL $\mathrm { w } / $ Tools + SFT further raises it to 91.3. Although RL $\mathrm { w } / $ Tools $\mathrm { w / o }$ SFT eventually reaches 90.5, the training dynamics in Figure 3 show that it does so with substantially longer and more tool-intensive trajectories. Capability internalization then raises the $\mathrm { w / o }$ Tools score from 40.1 to 72.7 while preserving the 91.3 w/ Tools score reached by RL $\mathrm { w / \ T o o l s + S F T }$ . In contrast, SFT w/o Tools, RL $\mathrm { w / o }$ Tools $\mathrm { w / o }$ SFT, and RL $\mathrm { w / o }$ Tools + SFT improve the $\mathrm { w / o }$ Tools score but do not acquire a comparably strong tool-use policy. The results therefore isolate complementary roles for the three stages: immediate perceptual support, stable tool-policy learning, and transfer of tool-supplied capabilities into $\mathrm { w / o }$ Tools inference.

## F.2 Tool-Set Ablation

To isolate the contribution of each spatial-tool group, we evaluate the initial Qwen3-VL-8B-Instruct checkpoint while varying only the registered tool set. The task inputs, shared agentic prompt template, decoding configuration, and evaluation protocol remain identical across variants. Each single-tool variant exposes only one of the Locate, Segment, Depth, or Pose interfaces, whereas All Tools exposes all four interfaces. The evaluated benchmarks, subsets, metrics, and reporting order exactly match those in the main-paper overall results (Table 1). Table 10 reports the resulting benchmark scores.

<table><tr><td rowspan="2">Model / Tool Set</td><td rowspan="2">SpatialCLI Bench</td><td rowspan="2">MindCube</td><td colspan="2">MMSI</td><td rowspan="2">DA-2K</td><td colspan="2">BOPASK</td><td rowspan="2">Avg.</td></tr><tr><td>Motion-Cam</td><td>Pos-Cam-Cam</td><td>Traj.</td><td>ObjRrr</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>35.3</td><td>29.3</td><td>27.0</td><td>25.8</td><td>68.1</td><td>25.8</td><td>13.3</td><td>35.7</td></tr><tr><td>+ Locate Only</td><td>31.8</td><td>32.4</td><td>24.3</td><td>30.1</td><td>46.6</td><td>53.7</td><td>19.3</td><td>34.9</td></tr><tr><td>+ Segment Only</td><td>39.0</td><td>26.3</td><td>27.0</td><td>28.0</td><td>57.0</td><td>43.0</td><td>21.0</td><td>36.3</td></tr><tr><td>+ Depth Only</td><td>40.1</td><td>27.3</td><td>24.3</td><td>26.9</td><td>91.6</td><td>22.7</td><td>14.1</td><td>40.6</td></tr><tr><td>+ Pose Only</td><td>50.4</td><td>44.5</td><td>37.8</td><td>38.7</td><td>52.0</td><td>4.1</td><td>1.4</td><td>37.6</td></tr><tr><td>+ All Tools</td><td>66.5</td><td>47.2</td><td>41.9</td><td>39.8</td><td>91.6</td><td>54.3</td><td>20.6</td><td>56.7</td></tr></table>

Table 10 Tool-set ablation of the initial Qwen3-VL-8B-Instruct model. Single-tool variants expose only the named interface, while All Tools exposes Locate, Segment, Depth, and Pose. Avg. follows the benchmark-level macroaveraging used in main-paper Table 1. The best result in each column is bolded, including ties.

The single-tool variants exhibit clear capability-aligned specialization. Depth alone matches All Tools on DA-2K (91.6); among the single-tool settings, Pose performs best on SpatialCLI-Bench, MindCube, and both MMSI subsets, while Locate and Segment are most efective on BOPASK-Trajectory and BOPASK-Object-Rearrangement, respectively. No single tool is uniformly beneficial across benchmarks. Jointly exposing all four tools raises the macro-average from 35.7 to 56.7 and achieves the best or tied-best result on six of the seven reported evaluations, supporting the complementarity of the four spatial interfaces.

## F.3 Tool-Use Behavior across Internalization Variants

To complement the score and output-length ablation in main-paper Table 3, Table 11 reports the actual tool invocations made by each capability-internalization variant $\mathrm { w } / $ Tools across all evaluated benchmarks. We report the mean number of tool calls over all samples, the percentage of samples that invoke at least one tool, and the mean number of calls among those samples. To reveal each variant’s unconstrained calling tendency, this diagnostic retains the maximum generation length of 40,960 tokens but does not apply the ten-call cap used in the task-performance evaluations.

<table><tr><td>Internalization Variant</td><td>Mean Tool Calls</td><td>Calling Samples (%)</td><td>Calls per Calling Sample</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>2.00</td><td>99.3</td><td>2.02</td></tr><tr><td>Final Answer Only</td><td>0.00</td><td>0.0</td><td>0.00</td></tr><tr><td>CoT + Answer</td><td>0.01</td><td>0.4</td><td>1.95</td></tr><tr><td>Internalization View Only</td><td>20.70</td><td>40.3</td><td>51.35</td></tr><tr><td>Tool-Use View Only</td><td>1.28</td><td>81.0</td><td>1.58</td></tr><tr><td>Full Dual-View</td><td>1.27</td><td>81.5</td><td>1.56</td></tr></table>

Table 11 Tool-use behavior of capability-internalization variants averaged across all benchmarks evaluated in this paper w/ Tools.

The initial model calls tools on 99.3% of samples, averaging 2.00 calls per sample. Final Answer Only never calls a tool, while CoT + Answer does so on only 0.4% of samples. Internalization View Only calls tools on 40.3% of samples, but averages 51.35 calls within those samples and 20.70 calls overall. This excessive number of calls indicates unstable and repetitive tool use rather than efective interaction. By contrast, Tool-Use View Only and Full Dual-View exhibit nearly identical controlled behavior: they call tools on 81.0% and 81.5% of samples and average 1.28 and 1.27 calls overall, respectively. Thus, dual-view training preserves the tool-use behavior learned from the tool-use view while obtaining the score and compact $\mathrm { w / o }$ Tools reasoning benefits reported in main-paper Table 3.

## F.4 Sensitivity to the Tool-Use View Loss Weight

To study the trade-of controlled by the Tool-Use View loss weight in $\mathcal { L } _ { \mathrm { C I } } = \mathcal { L } _ { \mathrm { i n t e r n a l } } + \lambda \mathcal { L } _ { \mathrm { a g e n t i c } }$ , we compare $\lambda \in \{ 0 . 2 , 0 . 5 , 1 . 0 , 1 . 5 \}$ using Qwen3-VL-8B-Instruct. All variants use the same training data, optimization configuration, training steps, and checkpoint-selection protocol. We evaluate the four-capability macro-average CII together with $\mathrm { w / o }$ Tools and $\mathrm { w } / $ Tools performance on SpatialCLI-Bench, thereby measuring both capability internalization and tool-use retention. Table 12 reports the resulting sensitivity analysis.

<table><tr><td rowspan="2">λ</td><td rowspan="2">CII Macro</td><td colspan="2">SpatialCLI-Bench</td></tr><tr><td>w/o Tools</td><td>w/ Tools</td></tr><tr><td>0.2</td><td>60.8</td><td>72.5</td><td>89.2</td></tr><tr><td>0.5 (Ours)</td><td>60.8</td><td>72.7</td><td>91.3</td></tr><tr><td>1.0</td><td>60.5</td><td>72.3</td><td>91.4</td></tr><tr><td>1.5</td><td>55.6</td><td>62.9</td><td>91.4</td></tr></table>

Table 12 Sensitivity to the Tool-Use View loss weight λ with Qwen3-VL-8B-Instruct. CII Macro averages Locate, Segment, Depth, and Pose, where Pose macro-averages object orientation and camera motion.

The results reveal a clear trade-of between capability internalization and tool-policy retention. Reducing λ to 0.2 leaves CII and $\mathrm { w / o }$ Tools performance comparable to $\lambda = 0 . 5$ , but lowers $\mathrm { w } / $ Tools performance from 91.3 to 89.2, suggesting that underweighting the Tool-Use View weakens retention of the learned tool-use policy. Increasing λ to 1.0 maintains $\mathrm { w } / $ Tools performance at 91.4 while slightly reducing CII and $\mathrm { w / o }$ Tools performance; further increasing it to 1.5 substantially lowers CII and $\mathrm { w / o }$ Tools performance to 55.6 and 62.9 without an additional $\mathrm { w } / $ Tools gain. Therefore, $\lambda = 0 . 5$ provides the most balanced trade-of rather than maximizing either training view alone.

## G Prompt Templates

## G.1 Agentic Tool-Use Prompt Template

The prompt injects the applicable tool registrations and examples into a shared calling protocol. The interaction permits at most ten executed tool calls per sample. After every tool result, SpatialCLI appends [Tool-call budget: at most N more tool calls may be made.]; when the budget reaches zero, it instead appends [Tool-call budget exhausted: no more tool calls are allowed. Answer the question immediately using the available information.]. In thinking mode, the complete thinking content is retained in the multi-turn history and passed back to the model in subsequent turns. The complete template is shown in Box G.1.

Box G.1: Agentic Tool-Use Prompt Template   
1 You have the following tools available.   
2 Tool call format:   
4 For each function call, return a JSON object with function name and arguments within <tool\_call></tool\_call>   
XML tags:   
5   
6 <tool\_call>   
7 {"name": <function-name>, "arguments": <args-json-object>}   
8 </tool\_call>   
9   
10 Tool list:   
11 <tools>   
12 [   
13 {tool\_definitions}   
14 ]   
15 </tools>   
16   
17 Examples:   
18 {tool\_examples}

## G.2 Turn-Wise Evidence Consolidation Prompt

This prompt implements the first stage of Progressive Evidence-Grounded Trajectory Verbalization by converting one newly completed tool interaction into an evidence–reasoning unit conditioned on the visual input, task instruction, and previously consolidated units, while withholding the correct answer. Its rules are designed to preserve every exposed specialist output and explicit visual observation without promoting plans, uncertain guesses, or answer-option analysis to observed facts. The tool catalog standardizes the interpretation of heterogeneous return fields, while calibrated wording and exact-value preservation retain the uncertainty and numerical fidelity of perceptual estimates. The complete template is shown in Box G.2.

## Box G.2: Turn-Wise Evidence Consolidation Prompt

You are preparing progressive evidence-grounded capability-internalization training data. Consolidate evidence from   
one step of a successful visual tool-use trajectory. Given the visual task, the previously consolidated   
evidence-reasoning units, and exactly one newly completed interaction, verbalize all information exposed by the   
current tool result and record its logical update.   
2   
3 Be literal, detailed, and evidence-grounded. No correct answer is provided at this stage. Do not solve from an answer   
key. Put the structured result in the requested XML sections.   
4   
5 Rules:   
6 1. Every perceptual claim must be traceable to the current tool result, an explicit visual observation written in the   
current original reasoning, or a previous unit. Do not turn a plan, a proposed tool action, answer-option analysis, or   
an uncertain guess into an observed fact.   
2. Exhaustively verbalize every item and every field exposed by the current tool result. Preserve the returned count,   
every instance, image index, label, bounding box, point, polygon component and vertex, depth value, pose field,   
motion field, direction, axis, error, and other returned value. Do not shorten, sample, merge, or omit returned   
information merely because it is lengthy or irrelevant to the task. Express machine structures as readable

natural-language statements rather than copying an unexplained machine serialization.

8 3. Preserve every explicit visual observation and absolute or relative visual-spatial relation from the current original reasoning, even when it is unrelated to the current task. This includes left/right, above/below, front/behind, near/far, overlap, containment, adjacency, occlusion, orientation, size, appearance, and count descriptions.

9 4. Use calibrated perceptual wording such as "is estimated at", "appears", "approximately", or "the result indicates" where appropriate, because tool outputs are perception estimates. Keep every returned numeric value exactly as given: do not round, alter, normalize again, or invent precision.

10 5. Explain how the new evidence confirms, revises, or extends prior evidence. Clearly resolve a prior visual estimate when stronger tool evidence supersedes it.

11 6. Do not repeat unchanged evidence already preserved in a previous unit, but never discard newly returned or newly stated visual information merely because it is task-irrelevant.

12 7. Empty evidence is valid only when the interaction failed or truly returned no information; preserve the failure or empty-result details themselves.

14 Return the structured result in this form:   
15 <evidence\_unit>   
16 <new\_evidence>   
17 Complete readable natural-language statements verbalizing all returned information.   
18 </new\_evidence>   
19 <logical\_update>   
20 Natural-language updates to the accumulated evidence and reasoning state.   
21 </logical\_update>   
22 </evidence\_unit>   
24 Tool catalog (field semantics only; not sample evidence):   
25 Perception-tool semantics

27 The catalog below explains how to interpret fields. It is not evidence for the current sample. A claim is supported only when the corresponding value actually appears in the current tool result or a previous consolidated unit.

29 1. query\_locate — open-vocabulary 2D localization

30 - Input: a short visible object/category description and optional 1-based image\_indices.

31 - Output: count and zero or more matched instances. Each instance may contain bbox\_2d=[x1,y1,x2,y2] and point\_2d=[x,y].

32 - Coordinates use the normalized 0–999 image plane with origin at the top-left: x increases to the right and y increases downward. Smaller y is higher; larger x is farther right.

33 - point\_2d is the center used for geometric comparisons or as input to query\_depth. bbox\_2d describes 2D extent only.

34 - This tool does not establish metric depth, segmentation boundaries, object orientation, or camera motion.

36 2. query\_segment — open-vocabulary instance segmentation

37 - Input: a short visible object/category description and optional 1-based image\_indices.

38 - The exposed segmentation result contains bbox\_2d, point\_2d, and polygon\_2d. point\_2d is derived from bbox\_2d rather than independently estimated from the mask area.

39 - bbox\_2d is the bounding box associated with that segmentation instance/mask and normally encloses all returned polygon components. It is emitted as a separate result field; it is not recomputed as the mathematically tight bounding box of the simplified polygon vertices, so small boundary diferences are expected.

40 - point\_2d is the rounded, clamped center of bbox\_2d: [(x1+x2)/2,(y1+y2)/2]. It is not the area centroid of the mask or polygon, and for a non-convex or disconnected mask it may lie outside the occupied mask region.

41 - bbox\_2d, point\_2d, and polygon\_2d all use the normalized 0–999 image coordinates. polygon\_2d is a list of simplified boundary components tracing the occupied image region; multiple disconnected components may belong to one mask.

42 - polygon\_2d supports boundary, shape, overlap, containment, free-space, and placement reasoning. Neither the bbox center nor a polygon is a grasp pose, 3D cuboid, metric depth, or object orientation.

44 3. query\_depth — metric camera-axis depth at image points

45 - Input: one or more normalized 0–999 points and optional 1-based image\_indices.

46 - Output: one result per point with point\_2d and depth\_m.

47 - depth\_m is estimated camera-axis distance in meters. Smaller depth\_m means closer to the camera; larger means farther.

48 - Compare the returned numeric values explicitly when relative depth is relevant. Do not confuse depth\_m with image-plane y, Euclidean object-to-object distance, or object size.

49 - Depth is an estimate and only supports claims at the queried points/images.

51 4. query\_pose — object orientation or cross-view camera motion

52 - Object mode input: a short object description, optionally with image\_indices. Output may include image\_index, bbox\_2d, visible\_side, and facing\_direction\_camera.

53 - visible\_side names the object’s side seen by the camera. facing\_direction\_camera describes where the object’s   
front points in camera coordinates; these are related but not interchangeable. For facing direction, front means deeper   
into the image, back means toward the camera, and left/right are the viewer’s image sides.   
54 - Camera-motion mode input uses query=’camera motion’. Each result is directional from from\_image to to\_image.   
55 - position.direction describes translation of the camera position. view\_rotation.direction describes where the camera   
view turns. Never substitute one for the other.   
56 - Signed camera axes are +X right, +Y up, and -Z forward. Use ordinary direction strings for ordinary-language   
questions and signed axes only when the task explicitly asks about axes.   
57   
58 For every tool: respect image\_index/image\_indices, zero-result and failure states. Never treat a tool description, an   
input query, or the model’s pre-call guess as observed evidence.   
59   
60 <visual inputs>   
61 {VISUAL\_INPUTS}   
62 </visual\_inputs>   
63   
64 <task\_instruction>   
65 {TASK\_INSTRUCTION}   
66 </task\_instruction>   
67   
68 <previous\_evidence\_units>   
69 {PREVIOUS\_EVIDENCE\_UNITS}   
70 </previous\_evidence\_units>   
71   
72 <current\_reasoning>   
73 {CURRENT\_REASONING}   
74 </current\_reasoning>   
75   
76 <current\_tool\_call>   
77 {CURRENT\_TOOL\_CALL}   
78 </current\_tool\_call>   
79   
80 <current\_tool\_result>   
81 {CURRENT\_TOOL\_RESULT}   
82 </current\_tool\_result>

## G.3 Global Trajectory Verbalization Prompt

This prompt implements the second stage by converting the ordered evidence–reasoning units into a single tool-free perceptual reasoning target conditioned on the correct final answer. The answer is introduced only after turn-wise evidence consolidation, so it can guide the organization of already extracted evidence without influencing its collection. The rules merge repeated or superseded statements in dependency order, prohibit unsupported additions and construction-process leakage, and require exact preservation of the final answer. The complete template is shown in Box G.3.

## Box G.3: Global Trajectory Verbalization Prompt

1 You are preparing the final trajectory-verbalization target for capability internalization. Convert ordered, grounded   
visual evidence into a tool-free reasoning target for vision-language SFT. Given the visual task, ordered   
evidence-reasoning units, and the correct final answer, construct a perceptual reasoning chain explaining why the   
evidence supports that answer.   
2   
3 Put the structured result in the requested XML sections. The target must read as direct visual reasoning and must   
not mention tools, calls, observations, evidence-unit boundaries, confidence scores, prompts, or this conversion   
process.   
4   
5 Rules:   
6 1. Use only entities, attributes, values, and relations present in the evidence units.   
7 2. Merge evidence in dependency order and remove repeated or superseded statements, but do not discard an explicit   
visual or spatial fact merely because it is unnecessary for choosing the answer.   
8 3. Preserve the concrete visual-spatial descriptions carried by the evidence units and include every essential inference   
needed to support all clauses of the answer.   
9 4. Write as direct visual reasoning. Never mention tools, calls, results, evidence units, confidence scores, or this   
instruction.   
10 5. Do not invent entities, attributes, measurements, or relations.

```xml
11 6. Put reasoning and answer in separate XML sections. Copy the provided final answer exactly.
12
13 Return the structured result in this form:
14 <internalization_target>
15 <reasoning_chain>
16 Complete, detailed, tool-free perceptual reasoning.
17 </reasoning_chain>
18 <final_answer>
19 Exact provided final answer.
20 </final_answer>
21 </internalization_target>
22
23 <visual_inputs>
24 {VISUAL_INPUTS}
25 </visual_inputs>
26
27 <task_instruction>
28 {TASK_INSTRUCTION}
29 </task_instruction>
30
31 <evidence_reasoning_units>
32 {EVIDENCE_REASONING_UNITS}
33 </evidence_reasoning_units>
34
35 <correct_final_answer>
36 {CORRECT_FINAL_ANSWER}
37 </correct_final_answer>
```

## H Case Studies

We present 2 w/o Tools versus $\mathrm { w } / $ Tools comparisons for Qwen3.5-397B-A17B. The first examines depth comparison and camera view rotation, whereas the second examines instance-level bounding boxes and camera translation. Together, they show how the agent decomposes a question into grounding, measurement, and cross-checking steps and uses tool evidence to revise unreliable visual judgments. Section H.3 separately com pares two correct SpatialCLI-8B results after capability internalization: direct answering and tool-augmented answering.

## H.1 Case 1: Two-Image Camera Rotation and Depth

The task input and ground truth are shown in Box H.1. In the comparison below, the direct answer infers that the beige receptacle is closer from apparent size and image position. It therefore selects C despite correctly identifying the rightward view rotation. The tool-augmented agent first grounds the two receptacles, queries two-image camera motion, and then passes the grounded centers to Depth. The measured depths, 0.928 m for the beige receptacle and 0.776 m for the blue one, overturn the initial visual guess. Combining the blue receptacle’s right-side location with the rightward rotation returned by Pose yields D. The trajectory follows a ground–measure–cross-check–verify logic.

## Box H.1: Task (SpatialCLI-Bench ID 269)

![](images/0b1dfab487ec4e9dc08a0ffc8c121bb56958c02dabaa3322bd6ce740b812f5b1.jpg)  
In the first image, there are two large plastic waste receptacles in the foreground: a beige one with a black bag and a blue one with a clear bag. Compare their distances from the camera to find the closer one. In the first image, is this closer receptacle located on the left or the right? Finally, when moving from the first image to the second image, what is the dominant direction of the camera’s view rotation?

## Options:

A: The blue receptacle is closer; it is located on the right; the camera rotated to the left.

B: The beige receptacle is closer; it is located on the left; the camera rotated to the left.

C: The beige receptacle is closer; it is located on the left; the camera rotated to the right.

D: The blue receptacle is closer; it is located on the right; the camera rotated to the right.

E: The blue receptacle is closer; it is located on the left; the camera rotated to the right.

F: The beige receptacle is closer; it is located on the right; the camera rotated to the left.

Before choosing, briefly state the relevant facts and verify every clause of the selected option against all evidence available during solving. For any depth observations, smaller values mean closer; do not retain an earlier visual guess after receiving stronger evidence. Please show your choice in the answer field with only the choice letter, e.g., {"answer": "C"}.

## Ground Truth

## Correct option: D.

## w/o Tools (Incorrect)

## <think>

The user wants me to analyze two images and answer a multi-part question.

\*\*Part 1: Compare distances of the two large plastic waste receptacles in the first image.\*\*

\- \*\*Receptacle 1:\*\* Beige/tan bin with a black bag. It’s on the left side of the foreground.

\- \*\*Receptacle 2:\*\* Blue bin with a clear/white bag. It’s on the right side of the foreground.

\- \*\*Comparison:\*\* Looking at the perspective and size, the beige bin appears larger and lower down in the frame relative to the blue bin next to it (though they are similar in height, the base of the beige bin is lower). Wait, let’s look closer. The beige bin is clearly in front of the blue bin behind it (the tall blue one with the lid). But there is another blue bin to the right. Let’s re-evaluate.

\- Actually, looking at the first image:

\- There is a beige bin on the left.

\- Behind it is a tall blue recycling bin with a lid.

\- To the right of the beige bin is a small black cart.

\- To the right of the black cart is another blue bin with a clear liner.

\- The prompt asks about "two large plastic waste receptacles in the foreground: a beige one with a black bag and a blue one with a clear bag."

\- Beige bin: Left foreground.

\- Blue bin (with clear bag): Right foreground.

\- Which is closer? The beige bin’s base is lower in the frame than the blue bin’s base. The beige bin obscures part of the view towards the back left. The blue bin on the right is also quite close. Let’s look at the perspective. The beige bin seems slightly closer to the camera lens because it feels more "in your face" and its bottom edge is lower. However, let’s look at the second image.

\- The second image shows \*only\* the blue bin with the recycling symbol. This is the bin that was on the right in the first image (blue bin, clear bag). Wait, let me check the details.

\- Image 1 Blue Bin (right): Blue, rectangular, clear/white bag.

## w/ Tools (Correct)

## <think>

The user wants me to analyze two images of an ofice setting, focusing on waste receptacles and camera movement.

\*\*Step 1: Identify the objects in the first image.\*\* - There is a beige trash can with a black bag on the left side of the foreground.

\- There is a blue recycling bin with a clear bag on the right side of the foreground.

\- Behind the beige can, there is another blue bin (taller).

\*\*Step 2: Compare distances of the beige and blue receptacles in the first image.\*\*

\- I need to determine which one is closer to the camera. - Visually, the beige can appears larger and lower in the frame, suggesting it might be closer. However, the blue bin on the right is also quite large.

\- Let’s use ‘query\_locate‘ to get their bounding boxes and centers, then ‘query\_depth‘ to check their actual distances. \*\*Step 3: Determine the location (left/right) of the closer receptacle.\*\*

\- Once I know which is closer, I need to state if it’s on the left or right.

\- Based on visual inspection:

\- Beige can: Left side.

\- Blue bin: Right side.

\*\*Step 4: Determine the camera’s view rotation from image 1 to image 2.\*\*

\- Image 1 shows the beige can and blue bin side-by-side with a cabinet behind them.

\- Image 2 shows primarily the blue bin (with the recycling symbol visible now) and white cabinets to its right. The beige can is no longer visible or is out of frame to the left.

\- This suggests the camera turned to the right to focus on the blue bin and the cabinets next to it. Or did it turn left?

\- Let’s look at the background. In image 1, to the right of the blue bin, there are desks and shelves. In image 2, to the right of the blue bin, there are white cabinets. Wait, let’s look closer.

\- In image 1, the blue bin is on the right. To its right, further back, is a desk area.

```prolog
- Image 2 Blue Bin: Blue, rectangular, white logo (recycling
symbol), clear/white bag. This matches the bin on the right
in Image 1.
- In Image 2, this blue bin is now on the left side of the frame
and very prominent. The camera has moved.
- Let’s re-evaluate depth in Image 1. The beige bin is on the
left. The blue bin is on the right. The beige bin looks slightly
closer because it is larger in apparent scale relative to what a
standard bin looks like, and its bottom is lower. BUT, let’s
look at the options.
- Option A: Blue closer, right, rotate left.
Option B: Beige closer, left, rotate left.
Option C: Beige closer, left, rotate right.
- Option D: Blue closer, right, rotate right.
Option E: Blue closer, left... (Blue is on the right in img 1)
- Option F: Beige closer, right... (Beige is on the left in img
1).
Let’s look really closely at Image 1.
- The beige bin is clearly the closest object. It is large and in
the immediate foreground.
- The blue bin on the right is also in the foreground but set
slightly further back or just to the side. Wait, actually,
looking at the floor tiles, the beige bin is positioned further
"down" (closer to bottom edge) than the blue bin on the right.
Therefore, the beige bin is closer.
- So, "The beige receptacle is closer". And "it is located on
the left". This eliminates A, D, E, F.
- We are left with B and C.
- B: Camera rotated to the left.
C: Camera rotated to the right.
**Part 2: Determine camera rotation.**
- **Image 1 View:** Shows beige bin (left), black cart
(center), blue bin (right). Background shows desks and
cabinets.
- **Image 2 View:** Shows the blue bin (from the right side
of Image 1) now on the *left* side of the frame. To its right
are white cabinets.
- In Image 1, the blue bin is on the right. In Image 2, we are
looking directly at the front/side of that same blue bin, and
the objects that were to its right (the white cabinets/shelves)
are now visible.
- Wait, let’s look at the background of Image 1. To the right
of the blue bin is a desk area.
- In Image 2, to the right of the blue bin are white cabinets.
Where were these white cabinets in Image 1? They are likely
the white structure behind/next to the blue bin. In Image 1,
to the right of the blue bin, we see a desk. The white cabinets
in Image 2 look like storage units.
- Let’s re-orient.
- Image 1: Left = Beige bin. Center = Black cart. Right =
Blue bin. Far Right = Desk area.
- Image 2: Left = Blue bin. Right = White cabinets.
- If I am looking at Image 1, and I want to see what is to the
*right* of the scene (where the blue bin is), I would turn the
camera to the **right**.
- Let’s trace the objects.
- Object X: Blue bin with clear bag. In Img 1, it’s on the
right. In Img 2, it’s on the left/center. This implies the
camera moved to the right, causing the object to shift left in
the frame. Or the camera rotated right.
- If I rotate right, objects on the right move towards the
center/left. Yes.
- Also, new objects appear on the right side. In Image 2, we
see white cabinets on the right. In Image 1, looking past the
blue bin, we see a desk area. Wait, let’s look at the white
cabinets in Image 1. Behind the black cart and blue bin is a
white wall/pillar. To the right of that is a desk.
- Let’s look at Image 2 again. The white cabinets are tall,
with handles. In Image 1, can we see these? Ah, look at the
top right of Image 1. There is a desk. Underneath/behind it
might be cabinets.
- Actually, look at the transition.
- Image 1: Beige bin (Left), Black Cart (Middle), Blue Bin
(Right).
- Image 2: Blue Bin (Left), White Cabinets (Right).
- The camera has shifted focus from the group (Beige, Cart,
Blue) to just the Blue bin and what is next to it.
- The Blue bin was on the right. Now it is on the left. This
means the camera panned/rotated to the **right**. By
turning right, the rightmost object (Blue bin) moves towards
the center/left of the view, and we see what is further to the
```

- In image 2, the blue bin is on the left side of the frame. To   
its right are white cabinets.   
- Let’s re-evaluate the scene.   
- Image 1: Beige bin (left), Blue bin (right). Behind them is a   
partition/cabinet edge.   
- Image 2: Blue bin is now on the left side of the image.   
White cabinets are to its right.   
- If the blue bin was on the right in image 1 and is on the left   
in image 2, the camera must have panned/rotated to the   
\*right\*. Imagine standing facing the bins. If you turn right,   
objects on your right move towards the center/left of your   
view.   
- Let’s verify with ‘query\_pose‘.   
\*\*Plan:\*\*   
1. Locate the beige trash can and the blue trash can in the   
first image.   
2. Get depth for both to confirm which is closer.   
3. Confirm left/right position based on location.   
4. Use ‘query\_pose‘ with "camera motion" to determine the   
rotation direction from image 1 to image 2.   
\*\*Execution:\*\*   
1. ‘query\_locate‘ for "beige trash can" and "blue trash can"   
in image 1.   
2. ‘query\_depth‘ on their centers.   
3. ‘query\_pose‘ for "camera motion" between image 1 and 2.   
Let’s refine the queries.   
- Query 1: "beige trash can with black bag" in image 1.   
- Query 2: "blue trash can with clear bag" in image 1.   
- Query 3: "camera motion" from image 1 to 2.   
Wait, looking at image 2, the blue bin has a recycling symbol.   
In image 1, the blue bin on the right has a clear bag. Is it the   
same bin? Yes, likely. The bin on the left in image 2 looks   
like the bin on the right in image 1.   
In image 1, the blue bin is to the right of the beige bin.   
In image 2, the blue bin is to the left of the white cabinets.   
This implies a rotation to the right. If I look straight at the   
gap between them, then turn right, the blue bin moves to the   
left of my field of view.   
Let’s get the data to be sure.

<tool\_call>   
"name": "query\_locate",   
"arguments": {   
"query": "beige trash can with black bag",   
"image\_indices": [   
1   
]   
}   
}   
</tool\_call>   
<tool\_call>   
"name": "query\_locate",   
"arguments": {   
"query": "blue trash can with clear bag",   
"image\_indices": [   
1   
]   
}   
</tool\_call>   
<tool\_call>   
"name": "query\_pose",   
"arguments": {   
"query": "camera motion",   
"image\_indices": [   
1,   
2   
]   
}   
}   
</tool\_call>   
<tool\_response>   
{"count":1,"result":[{"bbox\_2d":[203,176,418,872],"point\_2d":   
[310,524]}]}   
[Tool-call budget: at most 4 more tool calls may be made.]

![](images/6866cfd145bc41a8f58f4de67b3345b6448e0119214cdf681bd6e55e3c50d2f8.jpg)  
H.2 Case 2: Two-Image Camera Translation and Instance Segmentation

The task input and ground truth are shown in Box H.2. In the comparison below, the direct answer approximately identifies the target sandal and its maximum vertical coordinate, but interprets the scale change in the second image as backward camera motion. It consequently predicts backward-left and selects C. The toolaugmented agent first obtains a coarse location and the camera motion from Locate and Pose. When Locate

Image 1

merges the pair into one box, the agent switches to Segment to recover two instances and compares their maximum horizontal and vertical coordinates. The instance-level result supports 856, while Pose returns forward-left, yielding E. This trajectory follows a coarse-grounding–granularity-check–instance-refinement– joint-decision logic.

Box H.2: Task (SpatialCLI-Bench ID 323)  
![](images/20b07ff09fb319eee6260bcc952bfebfc09cdf354d87d015ee0efc4e935a42f4.jpg)

Image 2  
![](images/987105dcfd9588c811491c8aed5e8575f4dc047478c53ac748a1e9175a5d2c52.jpg)

Consider the two white sandals with pearl straps in the first image. First, determine which of these two sandals has a bounding box that extends further to the right (has the larger maximum x-coordinate). What is the maximum y-coordinate of that same sandal’s bounding box, and what is the direction of the camera’s translation from the first image to the second? Options:

A: Maximum y-coordinate: 747; Camera translation: backward-left

B: Maximum y-coordinate: 747; Camera translation: forward-left

C: Maximum y-coordinate: 856; Camera translation: backward-left

D: Maximum y-coordinate: 747; Camera translation: forward-right

E: Maximum y-coordinate: 856; Camera translation: forward-left

F: Maximum y-coordinate: 856; Camera translation: forward-right

Before choosing, briefly state the relevant facts and verify every clause of the selected option against all evidence available during solving. For any depth observations, smaller values mean closer; do not retain an earlier visual guess after receiving stronger evidence. Please show your choice in the answer field with only the choice letter, e.g., {"answer": "C"}.

## Ground Truth

Correct option: E.

## w/o Tools (Incorrect)

## <think>

The user wants me to identify a specific sandal based on its bounding box and then determine camera movement.

\*\*1. Identify the target sandal:\*\*

\- The prompt asks to consider the "two white sandals with pearl straps in the first image".

\- I need to find the bounding boxes for these two sandals.

\- Sandal 1 (left/back): This is the one further back, slightly to the left. Its strap has pearls.

\- Sandal 2 (right/front): This is the one closer to the front right. Its strap also has pearls. Wait, let’s look closer. - In Image 1:

\- There is a sandal in the foreground [143, 508, 765, 856].

This is the main white sandal facing somewhat left. It has a

## w/ Tools (Correct)

## <think>

The user wants me to perform two tasks based on the provided images.

\*\*Task 1: Analyze the first image.\*\*

\- Locate the "two white sandals with pearl straps".

\- Determine which one has a bounding box extending further to the right (larger max x-coordinate).

\- Find the maximum y-coordinate of that specific sandal’s bounding box.

\*\*Task 2: Analyze the camera motion.\*\* - Determine the direction of the camera’s translation from the first image to the second image.

\*\*Step-by-step Plan:\*\*

\- There is another sandal behind it/to the left [159, 492, 715, 747]. This one is partially visible, angled diferently. Wait, let’s re-evaluate.

\- Let’s look at the structure.

\- One sandal is clearly in the front: [143, 508, 765, 856]. This is the right-most sandal of the pair. It has a square toe and a pearl strap.

\- The other sandal is behind it: [159, 492, 715, 747]. This one is to the left and slightly behind.

- The question asks: "determine which of these two sandals has a bounding box that extends further to the right (has the larger maximum x-coordinate)."

- Sandal 1 (front/right): Box is roughly [143, 508, 765, 856]. Max x is \~765.

- Sandal 2 (back/left): Box is roughly [159, 492, 715, 747].   
Max x is \~715.

\- Therefore, the sandal with the larger maximum x-coordinate is the front/right one: [143, 508, 765, 856].

\*\*2. Find the maximum y-coordinate of that sandal:\*\* - The bounding box for the identified sandal is [143, 508, 765, 856].

\- The coordinates are usually [y\_min, x\_min, y\_max,

x\_max] or [x\_min, y\_min, x\_max, y\_max]. Let’s assume standard object detection format [ymin, xmin, ymax, xmax] or similar.

\- Let’s look at the coordinates provided in the thought process.

\- If the box is [143, 508, 765, 856], let’s interpret.

\- y\_min = 143? No, that’s too high up. The shoe is lower down.

\- Let’s look at the image dimensions. It’s a vertical phone photo.

\- The shoes are at the bottom. So y-coordinates should be large (near the bottom).

\- x-coordinates span the width.

\- Let’s re-examine the boxes.

\- Sandal 1 (front): ymin \~ 508, xmin \~ 143, ymax \~ 856, xmax \~ 765.

\- ymin=508 (mid-height), xmin=143 (left side), ymax=856 (bottom), xmax=765 (right side). This makes sense. The heel is around y=500-600. The toe is at the bottom y=856.

\- Sandal 2 (back): ymin \~ 492, xmin \~ 159, ymax \~ 747, xmax \~ 715.

\- ymin=492, xmin=159, ymax=747, xmax=715.

\- Comparing max x: Sandal 1 (765) > Sandal 2 (715). So the target sandal is the front one.

\- The question asks for the "maximum y-coordinate of that same sandal’s bounding box".

\- For the front sandal [143, 508, 765, 856], the max y is 856.

Wait, let me check the coordinate order. Usually it’s [ymin,

xmin, ymax, xmax] or [xmin, ymin, xmax, ymax].

\- Let’s look at the options. Options have max y as 747 or 856.

\- If the box is [143, 508, 765, 856], and assuming [ymin, xmin, ymax, xmax], then max y is 856.

\- If the box is [159, 492, 715, 747], max y is 747.

\- Since I identified the sandal extending further right as the front one, its max y should be 856.

\- Let’s double check which one extends further right.

\- Front sandal: The heel strap goes back. The toe is forward. The rightmost part is likely the heel/strap area or the side of the foot. In image 1, the sandal in the foreground (closest to camera) has its heel towards the right side of the frame. The other sandal is behind it.

\- There is a sandal lying flat: [159, 492, 715, 747]. This looks like the left shoe of the pair.

\- There is a sandal standing/upright: [143, 508, 765, 856].

This looks like the right shoe of the pair.

\- The upright sandal (right shoe) is closer to the right edge.

Its bounding box x\_max would be larger.

\- The flat sandal (left shoe) is further left.

\- So the target is the upright sandal.

\- Its max y-coordinate is the bottom of the shoe. In image 1, the bottom of the sole is quite low. 856 seems correct for the bottom. 747 would be higher up, corresponding to the other sandal.

\- So, Max y-coordinate = 856.

\*\*3. Determine camera translation:\*\*

\- Compare Image 1 and Image 2.

1. \*\*Locate the sandals in Image 1:\*\* I need to find the two white sandals. Looking at Image 1, there are two distinct white heeled sandals sitting on a pink cloth. They have pearl-like embellishments on the toe strap. One is slightly behind/left of the other. Wait, let’s look closer.

\- There is a pair of white strappy heels.

\- One is more towards the left/back (further away).

\- One is more towards the front/right (closer).

\- Actually, looking at the layout: There is a sandal whose heel is visible on the right side and toe pointing left. And another sandal behind it. Let’s use ‘query\_locate‘ to get precise bounding boxes for "white sandals with pearl straps" or similar. The prompt says "two white sandals with pearl straps". In Image 1, I see two white heels. One is clearly in the foreground, angled slightly right. The other is behind it. Let’s get their locations.

2. \*\*Determine the right-most sandal:\*\* Compare the ‘bbox\_2d‘ values. Specifically, look at ‘x\_max‘ (the third value in ‘[x\_min, y\_min, x\_max, y\_max]‘).

\- Sandal A (foreground): likely has a larger x\_max because it’s more to the right.

\- Sandal B (background): likely has a smaller x\_max.

3. \*\*Find max y-coordinate:\*\* Once the right-most sandal is identified, take its ‘y\_max‘ (the fourth value in the bbox). 4. \*\*Determine camera motion:\*\* Use ‘query\_pose‘ with

\- Image 1 shows a close-up of the shoes on a bed/surface with a headboard behind.

\- Image 2 shows the same shoes but from a diferent angle. The background has changed significantly. It looks like we moved back and possibly rotated. Wait, looking at Image 2, the shoes are in the foreground, but the background is a room with clothes hanging. This suggests a significant change in viewpoint or perhaps even a diferent setup, but the shoes look identical. Let’s assume it’s the same scene or a very similar setup.

\- Actually, looking closely at Image 1 vs Image 2:

\- Image 1: Shoes are on a pink "Hello Kitty" cloth. Behind is a white headboard.

\- Image 2: Shoes are on the same pink cloth. Behind them are black and white sneakers. Further back is a room with clothes racks.

\- This implies the camera moved. In Image 1, the camera is close, looking somewhat down/forward at the shoes against a wall/headboard. In Image 2, the camera is further back,

showing more of the room. The shoes are facing the camera more directly in Image 2.

\- Let’s use the tool ‘query\_pose‘ to get the exact vector. \*\*Execution:\*\*

\* \*\*Step 1:\*\* ‘query\_locate‘ for "white sandals with pearl straps" in image 1.

\*\*Step 2:\*\* ‘query\_pose‘ for "camera motion" from image 1 to image 2.

Let’s refine the locate query. "white high heel sandals" or "white shoes with pearls".

In Image 1, there are two white sandals.

One is at roughly [150, 500, 700, 850] (guessing).

Let’s just call ‘query\_locate‘ on "white sandals" for image 1.

Then I will compare the x-coordinates.

Then I will read the y-max.

Then I will interpret the camera motion.

Image 1 is a close up. Image 2 is further away. So "backward".

In Image 1, the shoes are angled. In Image 2, they are facing front.

Let’s rely on the tool for the specific direction (e.g.,

backward-left, forward-right).

Let’s start the tool calls.

</think>

<tool\_call>   
{ "name": "query\_locate", "arguments": { "query": "white sandals with pearl straps", "image\_indices": [ 1 ] }   
}

- Image 1: Close-up of the shoes. We see the headboard   
clearly. The shoes are large in the frame.   
- Image 2: Wide shot. The shoes are further away. We see the   
whole room, clothes rack, towels.   
- The camera has moved \*backward\* (away from the shoes) to   
capture more of the scene.   
- Also, the angle changes. In Image 1, we are looking slightly   
down and from the front-right. In Image 2, we are further   
back.   
- Let’s look at the position of the shoes relative to the frame.   
- Image 1: Shoes are central/bottom.   
Image 2: Shoes are central/bottom.   
- Wait, let’s look at the options.   
Options involve "backward-left", "forward-left",   
"forward-right".   
- "Backward" means moving away from the subject. Image 2   
is definitely further away than Image 1. The shoes are much   
smaller. So "backward" is the primary movement.   
- Is there a lateral movement?   
- In Image 1, the black/white sneaker is on the far right. The   
white sandals are to its left.   
- In Image 2, the white sandals are in the front center. The   
black/white sneaker is behind them.   
- This suggests a significant change in arrangement or camera   
angle.   
- Let’s look at the background.   
- Image 1: White headboard behind the shoes.   
- Image 2: A room with a clothes rack and towels. The   
headboard is not immediately visible behind the shoes in the   
same way. Wait, looking at Image 2, the shoes are on a pink   
surface (Hello Kitty towel?). In Image 1, they are on the   
same pink surface.   
- In Image 1, the background is a white padded headboard.   
- In Image 2, the background is a room with a rack. The   
headboard is gone or further back? No, the surface they are   
on seems to be a bed or table.   
- Actually, looking at Image 2, the shoes are on a pink   
towel/sheet. Behind them is a chair with towels draped over   
it.   
- In Image 1, behind the shoes is a white curved object   
(headboard?).   
- This implies the camera moved significantly.   
- Let’s reconsider the "translation".   
- Image 1 is a close up. Image 2 is a wide shot. This is a   
"backward" movement (zoom out or physically move back).   
- Let’s look at the options again.   
- A: backward-left   
- B: forward-left   
- C: backward-left   
- D: forward-right   
- E: forward-left   
- F: forward-right   
- Only A and C have "backward". Since Image 2 is clearly   
further away (objects are smaller, more context visible), the   
camera must have moved backward.   
- So the answer must be A or C.   
- Both A and C say "backward-left"   
- Wait, let’s check the Max Y coordinate again.   
- If the answer is A, Max Y is 747.   
- If the answer is C, Max Y is 856.   
- I determined earlier that the sandal extending further right   
is the front one. with Max Y = 856.   
- Let’s re-verify the "further right" condition.   
- Sandal 1 (flat/left): Box [159, 492, 715, 747]. Max X = 715.   
- Sandal 2 (upright/right): Box [143, 508, 765, 856]. Max X =   
765.   
- 765 > 715. So Sandal 2 is the one.   
- Max Y of Sandal 2 is 856.   
- So the answer should be C.   
Let’s double check the "backward-left" translation.   
- In Image 1, the camera is close.   
- In Image 2, the camera is further back. So "backward" is   
correct.   
- Why "left"?   
- In Image 1, the view is centered on the white sandals. The   
sneaker is cut of on the right.   
- In Image 2, the view includes the sneaker fully behind the   
sandals. The camera seems to have moved back and perhaps   
slightly left to center the composition diferently? Or maybe   
the objects were rearranged.

```html
</tool_call>
<tool_call>
"name": "query_pose",
"arguments": {
"query": "camera motion",
"image_indices": [
1,
2
}
}
</tool_call>
<tool_response>
{"count":1,"result":[{"bbox_2d":[143,499,857,857],"point_2d":
[500,678]}]}
[Tool-call budget: at most 4 more tool calls may be made.]
</tool_response>
<tool_response>
{"result":[{"from image":1."to image":2."position":{"direction"
"forward-left","axes":["-X"."+Y"."-Z"]}."view rotation"
{"direction":"left"."axes":["+X"."-Y"]."dominant axis":"-Y"}}]}
[Tool-call budget: at most 3 more tool calls may be made.]
</tool_response>
<think>
The query_locate returned only one bounding box for both
sandals combined: [143, 499, 857, 857]. This suggests the tool
grouped them. I need to distinguish between the two
individual sandals to find which has the larger maximum
x-coordinate.
Let me segment each sandal individually to get their separate
bounding boxes.
</think>
<tool_call>
"name": "query_segment",
"arguments": {
"query": "left white sandal with pearl strap",
"image_indices": [
1
}
}
</tool_call>
<tool_call>
"name": "query_segment",
"arguments": {
"query": "right white sandal with pearl strap",
"image_indices": [
1
}
}
</tool_call>
<tool_response>
{"count":2,"result":[{"bbox_2d":[145,507,784,855],"point_2d":
[464,681],"polygon_2d":[[[755,521],[651,525],[582,647],[471,718],
[348,732],[292,699],[279,743],[156,771],[167,833],[428,840],[680,697],
[680,744],[726,755],[782,630],[726,614]]]},{"bbox_2d":[156,493,759,
746],"point_2d":[458,620],"polygon_2d":[[[713,499],[538,539],[515,
593],[544,561],[549,596],[359,663],[302,633],[161,735],[269,741],[292,
693],[348,727],[415,725],[498,661],[521,679],[578,644],[588,574]],
[[703,574],[697,569],[663,571],[647,574],[636,585],[628,607],[622,610],
[626,619],[634,622],[667,607],[682,589],[699,582]],[[734,610],[734,613]
,[751,613],[753,610],[751,608],[745,608],[740,607]]]}]}
[Tool-call budget: at most 2 more tool calls may be made.]
</tool_response>
<tool_response>
{"count":0."result":[}
[Tool-call budget: at most 1 more tool call may be made.]
</tool_response>
<think>
The query_locate returned only one bounding box
[143,499,857,857] for "white sandals with pearl straps", which
```

\- Assuming the objects didn’t move (which is a big assumption, but usually implied in these "camera translation" questions unless obvious), let’s trace the perspective.

\- In Image 1, the white headboard is directly behind.

\- In Image 2, the background is a room. This suggests the "bed" in Image 1 might be diferent from the surface in Image 2, or the angle changed drastically.

\- Actually, looking closely at Image 2, the pink towel has "Relaxy" and "Love" written on it. In Image 1, we see "Relaxy" and "Love" on the pink towel under the shoes. So it’s the same surface.

\- In Image 1, the white object behind is a headboard.

\- In Image 2, the background is diferent. Wait, looking at the top of Image 2, there are lights on the ceiling. In Image 1, there are lights at the top too.

- It looks like Image 1 is a crop or a close-up taken on the bed.   
Image 2 is taken standing up further back.

\- If I move backward from the position in Image 1, I would see more of the room.

\- In Image 1, the sneaker is to the right. In Image 2, the sneaker is behind the sandals. This implies the sandals were moved or the camera angle changed significantly.

\- However, usually these questions assume a rigid scene. If the scene is rigid:

\- Image 1: Sandals in front, sneaker to the right.

\- Image 2: Sandals in front, sneaker behind.

\- This implies the camera moved from a position where the

sneaker was to the right, to a position where it is behind.

\- If the sneaker is to the right in Image 1, and behind in

Image 2, the camera must have moved to the \*left\*.

\- Imagine the setup: Sandals at (0,0), Sneaker at (1, 0).

\- Camera 1 (Image 1): Looking from front-right? No, if

sneaker is to the right, camera is looking from front-leftish? sneaker is to the right, camera is looking from front-leftish?

\- Let’s assume standard coordinates. X increases right. Y increases down.

\- Image 1: Sandals center. Sneaker right edge.

\- Image 2: Sandals center. Sneaker behind.

\- This implies a rotation or a significant move.

\- But let’s look at the options again. "backward-left".

\- If I am close (Image 1) and I move backward and left...

\- Moving backward makes things smaller. Correct.

\- Moving left shifts the view to the right. Objects on the right would move towards the center.

\- In Image 1, the sneaker is on the far right.

\- In Image 2, the sneaker is behind the sandals (more central).

\- So the sneaker moved from right-edge to center. This

happens if the camera pans/moves left.

\- So "backward-left" fits perfectly.

So, the logic holds:

1. Target sandal: The one with larger max x. That’s the front/right sandal [143, 508, 765, 856]. Max x is 765. The other is [159, 492, 715, 747], max x 715.

2. Max y of target sandal: 856.

3. Camera translation: Backward (zoom out/move away) and Left (sneaker moves from right edge to center).

Therefore, Option C: Maximum y-coordinate: 856; Camera translation: backward-left.

Let’s double check the bounding box values provided in the

prompt’s hidden ground truth or just infer them.

The prompt asks me to determine the box.

Box 1 (left sandal): Looks like ‘[159, 492, 715, 747]‘.

\- ymin: 492 (top of strap/heel area)

\- xmin: 159 (toe area left) – wait, the toe is left.

\- ymax: 747 (bottom of heel/sole)

\- xmax: 715 (right side of heel strap)

Box 2 (right sandal): Looks like ‘[143, 508, 765, 856]‘.

\- ymin: 508 (top of strap)

\- xmin: 143 (toe left)

\- ymax: 856 (bottom of sole/toe)

\- xmax: 765 (right side of heel)

Wait, let’s look at the image carefully.

\- There is a sandal lying somewhat flat/angled left: The toe is

pointing towards bottom-left. The heel is towards top-right.

\- There is a sandal standing more upright: The toe is pointing

towards bottom-left. The heel is towards top-right.

Actually, they look like a pair.

\- Left/Back sandal: Toe is near x=160, y=700. Heel is near x=700, y=500.

\- Right/Front sandal: Toe is near x=140, y=850. Heel is near x=760, y=500.

likely encompasses both sandals together. The

query\_segment returned two instances, but looking at the bboxes:

First instance: bbox\_2d=[145,507,784,855], so max\_x=784, max\_y=855

Second instance: bbox\_2d=[156,493,759,746], so max\_x=759, max\_y=746

Wait, let me re-read the segment results more carefully: Result 1: bbox\_2d=[145,507,784,855], point\_2d=[464,681] Result 2: bbox\_2d=[156,493,759,746], point\_2d=[458,620] So for Result 1: min\_x=145, min\_y=507, max\_x=784, max\_y=855

For Result 2: min\_x=156, min\_y=493, max\_x=759, max\_y=746

The sandal with the larger maximum x-coordinate is Result 1 (max\_x=784 vs 759).

The maximum y-coordinate of that sandal (Result 1) is 855. But wait, the options mention 747 or 856. Let me check if there’s a rounding or of-by-one issue. The bbox format might be [x\_min, y\_min, x\_max, y\_max] or the values might need adjustment.

Looking at the original query\_locate result:

bbox\_2d=[143,499,857,857], which has max\_y=857. But

that was for both sandals combined.

From the segment results:

\- First sandal: max\_y=855

\- Second sandal: max\_y=746

The first sandal has max\_x=784, second has max\_x=759. So the first sandal extends further right.

Its max\_y is 855, which is closest to 856 in the options.

Camera motion from image 1 to image 2: "forward-left"

according to query\_pose.

So the answer should be: max\_y=856 (approximately),

camera translation=forward-left.

That matches option E.

</think>

{"answer": "E"}

![](images/2a11407138ab98f7c1b0e61f4599768c713ce8bc7538efc627e443977706e3da.jpg)

## H.3 Case 3: SpatialCLI-8B after Capability Internalization

The task input and ground truth are shown in Box H.3. The full-width traces below present, in order, the tool-free Qwen3-VL-8B-Instruct baseline, SpatialCLI-8B with tools, and SpatialCLI-8B without tools. The baseline correctly identifies the yellow ball as farther right and the camera translation as forward-left, but confuses the first-image depth relation while repeatedly reasoning from cross-image scale changes; it therefore predicts that the yellow ball is closer and selects F. SpatialCLI-8B with tools first grounds the objects and then queries depth and camera motion. The returned horizontal centers are 329 and 644, the depths are 0.351 m and 0.637 m, and the translation is forward-left, so every clause supports E. Its final cross-image Locate call is a redundant verification after the answer is already determined and does not change the conclusion. Without tools, SpatialCLI-8B produces internalized estimates of 0.362 m and 0.615 m and the same forwardleft motion, yielding E directly. This example qualitatively illustrates agreement between internalized direct answering and runtime tool verification on the same task.

Image 1  
![](images/16df2c8e50e513fac60dd7ef395fcbced997f885fa939f56c5fba5002b41fb39.jpg)

Image 2  
![](images/ea20bd823eefed026149c34ae1ec23147522eb3f5e56b2923acaff2f2c2024a0.jpg)

In the first image, which object is located further to the right between the clear plastic bottle and the yellow ball, which of the two is closer to the camera, and in which direction did the camera translate to capture the second image? Options:

A: The yellow ball is further to the right, the clear plastic bottle is closer to the camera, and the camera translated backward-left. B: The clear plastic bottle is further to the right, the clear plastic bottle is closer to the camera, and the camera translated forward-left.

C: The clear plastic bottle is further to the right, the yellow ball is closer to the camera, and the camera translated forward-right.   
D: The yellow ball is further to the right, the clear plastic bottle is closer to the camera, and the camera translated forward-right.   
E: The yellow ball is further to the right, the clear plastic bottle is closer to the camera, and the camera translated forward-left.   
F: The yellow ball is further to the right, the yellow ball is closer to the camera, and the camera translated forward-left.

Before choosing, briefly state the relevant facts and verify every clause of the selected option against all evidence available during solving. For any depth observations, smaller values mean closer; do not retain an earlier visual guess after receiving stronger evidence. Please show your choice in the answer field with only the choice letter, e.g., {"answer": "C"}.

Ground Truth

Correct option: E.

## Qwen3-VL-8B-Instruct w/o Tools (Incorrect)

To determine the correct option, I will analyze the spatial relationships and camera movement between the two images. \*\*Step 1: Analyze relative positions in the first image.\*\*

In the first image, the clear plastic bottle is on the left side of the frame, and the yellow ball is to its right. The bottle is positioned closer to the foreground (and thus the camera) than the ball, as it appears larger and more detailed. Therefore: - The yellow ball is further to the right.

\- The clear plastic bottle is closer to the camera.

\*\*Step 2: Analyze camera translation from the first to the second image.\*\*

In the first image, the bottle is prominent in the foreground. In the second image, the bottle is now on the far right edge of the frame and is much smaller, indicating it has been moved farther away from the camera’s viewpoint. The yellow ball, which was to the right of the bottle in the first image, is now more centrally located and appears larger than before, suggesting it has also moved closer to the camera and/or the camera has moved toward it.

The key observation for camera movement is that the bottle has moved from a dominant foreground position to a background position on the right. This implies the camera has moved backward (away from the bottle) and to the left (to reposition the bottle to the right edge of the frame). The yellow ball, being to the right of the bottle in the first image, would then appear to have moved to the left relative to the bottle’s new position, but since the camera moved left, the ball appears to shift to the right relative to the scene. However, the ball is now more central and larger, so the camera must have translated forward (toward the ball) and to the left (to bring the ball into the center).

Actually, looking at the second image, the yellow ball is now more central, and the bottle is on the far right. The bottle is smaller in the second image, meaning the camera is farther from it. The ball is larger in the second image, meaning the camera is closer to

<table><tr><td>it. The ball was to the right of the bottle in the first image, and now it is more central. So, the camera has moved to the left (so the bottle moves to the right edge) and forward (so the ball becomes larger). Therefore, the camera translated forward-left. This matches option E: &quot;The yellow ball is further to the right, the clear plastic bottle is closer to the camera, and the camera translated forward-left.&quot; Wait – I need to verify the &quot;further to the right&quot; clause. In the first image, the bottle is on the left, the ball is on the right. So the ball is further to the right. That part is correct. The bottle is closer to the camera – correct. Camera translated forward-left – correct, because the bottle moves to the right edge (camera moved left) and gets smaller (camera moved back), but the ball gets larger (camera moved forward). Wait, if the camera moves forward, the ball should get larger, and if it moves left, the bottle should move to the right edge. But in the second image, the bottle is on the right edge and smaller, and the ball is larger and more central. So yes, the camera moved forward and to the left. Therefore, option E is correct.</td></tr><tr><td>Let me check other options: A: Says camera translated backward-left – but if camera moved backward, the bottle should appear even smaller, which it does, but the ball should appear smaller too, which it doesn&#x27;t. Also, if camera moved backward-left, the ball would be moving to the left, not becoming more central. So A is wrong. B: Says clear plastic bottle is further to the right – no, in the first image, the bottle is on the left. C: Says clear plastic bottle is further to the right – again, no. D: Says camera translated forward-right – if camera moved forward-right, the bottle would be more to the right, which it is, but the ball would be to the left, which it isn&#x27;t; it&#x27;s more central. Also, if camera moved forward-right, the bottle should appear smaller, which it does, but the ball should appear smaller or stay same size, not larger. So D is wrong.</td></tr><tr><td>F: Says yellow ball is closer to the camera – yes, it appears larger – but the camera translated forward-left, not forward-left with the ball being closer? No, F says &quot;the yellow ball is closer to the camera&quot; which is true, but then says &quot;the camera translated forward-left&quot; – which is also true, but it doesn&#x27;t specify the bottle&#x27;s position correctly? Wait, F says &quot;the yellow ball is further to the right&quot; – which is true, and &quot;the yellow ball is closer to the camera&quot; – which is true, and &quot;the camera translated forward-left&quot; which is true. But F doesn&#x27;t say anything about the bottle being closer – which is false. The bottle is not closer, it&#x27;s farther. So F is incorrect because it incorrectly states that the yellow ball is closer to the camera? No, it says &quot;the yellow ball is closer to the</td></tr><tr><td>camera&quot; – which is true – but then it doesn&#x27;t contradict the bottle being farther. Wait, F says: &quot;The yellow ball is further to the right, the yellow ball is closer to the camera, and the camera translated forward-left.&quot; This is factually correct based on the evidence. But wait – the question asks for the object that is further to the right and which is closer to the camera. The bottle is not closer, the ball is. So F is partially correct, but it doesn&#x27;t address the bottle&#x27;s position correctly? No, F doesn&#x27;t say the bottle is closer, it only says the yellow ball is closer. But in the question, we are asked to identify which of the two is closer. The answer must include both parts. Option F omits the bottle&#x27;s position, but the question requires us to state which of the two is closer. Since F says &quot;the yellow ball is closer to the camera,&quot; that&#x27;s correct, and the other clauses are also correct. But let&#x27;s compare F with E.</td></tr><tr><td>E says: &quot;The yellow ball is further to the right, the clear plastic bottle is closer to the camera, and the camera translated forward-left.&quot; But in reality, the clear plastic bottle is NOT closer to the camera in the second image – it is farther away. So E is incorrect. F says: &quot;The yellow ball is further to the right, the yellow ball is closer to the camera, and the camera translated forward-left.&quot;</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>This is correct. The yellow ball is further to the right (true), the yellow ball is closer to the camera (true), and the camera translated forward-left (true). But wait – the question asks: &quot;which of the two is closer to the camera&quot;. It doesn&#x27;t specify which one to pick, but in context, it&#x27;s asking for the correct statement. Both E and F have the same first two clauses? No, E says the bottle is closer, F says the ball is closer. Since the bottle is farther, F is correct.</td></tr><tr><td>But let&#x27;s double-check the camera movement. In the first image, the bottle is large and in the foreground. In the second image, the bottle is small and on the right edge. So the camera moved backward (away from the bottle) and to the left (so the bottle moves to the right edge). The ball is larger and more</td></tr></table>

## SpatialCLI-8B w/ Tools (Correct)

<think>   
The user wants me to analyze two images and answer three specific questions:   
1. \*\*Which object is further to the right in the first image?\*\* (Clear plastic bottle vs. Yellow ball)   
2. \*\*Which object is closer to the camera in the first image?\*\* (Clear plastic bottle vs. Yellow ball)   
3. \*\*In which direction did the camera translate to capture the second image?\*\*   
Let’s break this down step-by-step.   
\*\*Step 1: Analyze the first image for horizontal position (left/right).\*\*   
- I need to locate the "clear plastic bottle" and the "yellow ball". - Looking at Image 1, the clear plastic bottle is on the left side, very prominent in the foreground. - The yellow ball is further back and towards the right side of the image relative to the bottle. - Let’s use ‘query\_locate‘ to get precise coordinates if needed, but visually it seems quite clear. The bottle is central-left, the ball   
is right-ish. Wait, let’s look closer.

- In Image 1:   
- Bottle: Occupies the left/center foreground.   
- Ball: Located to the right of the bottle.   
- Therefore, the yellow ball is further to the right.   
\*\*Step 2: Analyze the first image for depth (closer/further).\*\*   
- I need to determine which is closer: the bottle or the ball.   
- Visually, the bottle is much larger and obscures the background. It is clearly in the immediate foreground.   
- The ball is smaller and sits further back on the patterned mat.   
- So, the clear plastic bottle is closer to the camera.   
- To be rigorous, I can use ‘query\_depth‘. I will locate both objects first to get their centers, then query depth.   
\*\*Step 3: Analyze camera motion between Image 1 and Image 2.\*\*   
- I need to determine the translation direction.   
- Image 1 shows the bottle prominently on the left.   
- Image 2 shows the bottle cut of on the far right edge (partially visible) and a toothbrush on a cloth on the left. The yellow ball   
is now more central/left.   
- Let’s look at the background features.   
- Image 1: Person standing in the distance center-right. White cabinet/shelf unit on the right.   
- Image 2: Glass door/window structure visible. Red box on the left. The person is still visible in the background but perspective   
has changed.   
- Let’s use ‘query\_pose‘ with "camera motion" to get the exact translation vector/direction.   
\*\*Plan:\*\*   
1. Locate "clear plastic bottle" and "yellow ball" in Image 1 to confirm positions and get points for depth.   
2. Query depth for the centers of these two objects in Image 1 to confirm which is closer.   
3. Query "camera motion" between Image 1 and Image 2.   
\*\*Execution:\*\*   
1. ‘query\_locate‘ for "clear plastic bottle" and "yellow ball" in image 1.   
2. ‘query\_depth‘ using the points obtained.   
3. ‘query\_pose‘ for "camera motion".   
Let’s refine the queries.   
- Query 1: ‘query\_locate‘ with query="clear plastic bottle", image\_indices=[1]   
- Query 2: ‘query\_locate‘ with query="yellow ball", image\_indices=[1]   
- Query 3: ‘query\_depth‘ using points from above.   
- Query 4: ‘query\_pose‘ with query="camera motion", image\_indices=[1, 2] (Wait, the tool description says "Omit image\_indices   
for image 2 relative to image 1". Since there are two images, I should probably just pass the query. Or maybe specify indices if   
needed. The description says "In a multi-image... omit image\_indices to use all images". So I’ll omit it.)   
Let’s start.   
</think>   
<tool\_call>   
"name": "query\_locate",   
"arguments": {   
"query": "clear plastic bottle",   
"image\_indices": [   
1   
]   
}   
</tool\_call>   
<tool\_call>   
"name": "query\_locate",   
"arguments": {   
"query": "yellow ball",   
"image\_indices": [   
1   
3   
</tool\_call>   
<tool\_response>   
{"count":1,"result":[{"bbox\_2d":[127,0,531,834],"point\_2d":[329,417]}]}   
</tool\_response>   
<tool\_response>   
{"count":1,"result":[{"bbox\_2d":[560,283,729,414],"point\_2d":[644,348]}]}   
</tool\_response>   
<think>   
The clear plastic bottle center is at [329, 417] and the yellow ball center is at [644, 348]. Comparing x-coordinates: 644 > 329, so   
the yellow ball is further to the right.   
Now I need to determine which object is closer to the camera in image 1. Let me query the depth of both points.   
</think>   
<tool\_call>   
  
"name": "query\_depth",   
"arguments": {   
"points": [   
329,

```diff
417
],
644,
348
]
],
"image_indices": [
1
]
}
}
</tool_call>
<tool_response>
{"result":[{"point_2d":[329.0,417.0],"depth_m":0.351},{"point_2d":[644.0,348.0],"depth_m":0.637}]}
</tool_response>
<think>
From the tool calls:
- In image 1:
- The clear plastic bottle center is at [329, 417].
The yellow ball center is at [644, 348].
Comparing x-coordinates: 644 > 329, so the **yellow ball is further to the right**.
- Depth values (smaller = closer): bottle depth_m = 0.351, ball depth_m = 0.637. So the **clear plastic bottle is closer to the
camera**.
- Camera motion from image 1 to image 2:
We need to determine the translation direction.
- Looking at the images visually:
- Image 1 shows the bottle very large in the foreground left, and the ball mid-ground right.
- Image 2 shows the bottle mostly out of frame or just the edge on the far right (actually, looking closely at image 2, the bottle is
barely visible on the right edge), and the ball is now more central-left. The red box which was not clearly visible or was further
back/left is now visible on the left.
- Actually, let’s re-examine the scene layout.
- Image 1: Bottle is close, bottom-left/center. Ball is behind it, to the right.
- Image 2: The view has shifted. The ball is now in the upper-left quadrant relative to the new frame. A red object (looks like a
toy bus) is visible on the far left. The white cloth with green dots is in the foreground left.
- Let’s look at the background floor/wall features.
- In image 1, there is a person standing in the background hallway.
- In image 2, the same person is visible but lower in the frame, suggesting the camera moved forward and tilted down or moved
position such that the perspective changed.
- Wait, let’s look at the options provided. They mention "forward-left", "backward-left", "forward-right".
- Let’s use the ‘query_pose‘ tool for camera motion between image 1 and image 2.
Let me call query_pose with ’camera motion’ for images 1 and 2.
</think>
<tool_call>
"name": "query_pose",
"arguments": {
"query": "camera motion",
"image_indices": [
1,
2
]
}

</tool call>
<tool_response>
{"result":[{"from_image":1,"to_image":2,"position":{"direction":"forward-left","axes":["-X","+Y","-Z"]},"view_rotation":{"direction":
"left","axes":["+X","-Y","-Z"],"dominant_axis":"-Y"}}]}
</tool_response>
<think>
Let me verify the facts:
1. **Which object is further to the right?**
- Clear plastic bottle center: [329, 417]
- Yellow ball center: [644, 348]
- The x-coordinate of the yellow ball (644) is greater than the x-coordinate of the bottle (329), so the yellow ball is further to the
right.
2. **Which object is closer to the camera?**
- Bottle depth: 0.351 m
- Ball depth: 0.637 m
- Smaller depth means closer, so the clear plastic bottle (0.351 m) is closer to the camera than the yellow ball (0.637 m).
3. **Camera translation direction:**
- The query_pose result shows position.direction = "forward-left" when going from image 1 to image 2.
So the correct option should state:
- The yellow ball is further to the right ✓
- The clear plastic bottle is closer to the camera ✓
Camera translated forward-left ✓
This matches option E.
</think>
```

<tool\_call>   
{   
"name": "query\_locate",   
"arguments": {   
"query": "clear plastic bottle"   
}   
}   
</tool\_call>   
<tool\_response>   
{"count":2,"result":[{"image\_index":1,"bbox\_2d":[127,0,531,834],"point\_2d":[329,417]},{"image\_index":2,"bbox\_2d":[946,249,998,519],   
"point\_2d":[972,384]}]}   
</tool\_response>   
<think>   
In the first image:   
- The clear plastic bottle is located at point [329, 417] and the yellow ball is at point [644, 348]. Since x=644 > x=329, the yellow   
ball is further to the right.   
- Depth values: bottle depth\_m = 0.351, ball depth\_m = 0.637. Smaller depth means closer, so the bottle (0.351) is closer to the   
camera than the ball (0.637).   
- Camera motion from image 1 to image 2 is "forward-left" according to query\_pose result.   
Now verify each clause of option E:   
- "The yellow ball is further to the right": True (644 > 329 in x-coordinate).   
"the clear plastic bottle is closer to the camera": True (0.351 < 0.637).   
- "and the camera translated forward-left": True (query\_pose confirms this direction).   
All clauses match evidence.   
</think>   
{"answer": "E"}

## SpatialCLI-8B w/o Tools (Correct)

The clear plastic bottle is located at horizontal center x=331, while the yellow ball is at horizontal center x=644. Since 644 > 331, the yellow ball is positioned further to the right in the first image. Depth measurements show the bottle is at 0.362 meters from the camera and the yellow ball is at 0.615 meters; since a smaller depth value indicates closer proximity, the clear plastic bottle is closer to the camera than the yellow ball. Finally, analysis of the camera motion between the two images determines that the camera translated forward-left to capture the second image. These facts confirm that the yellow ball is further to the right, the clear plastic bottle is closer to the camera, and the camera moved forward-left.   
</think>

{"answer": "E"}

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[2] Vineet Bhat, Sungsu Kim, Valts Blukis, Greg Heinrich, Prashanth Krishnamurthy, Ramesh Karri, Stan Birchfield, Farshad Khorrami, and Jonathan Tremblay. Bop-ask: Object-interaction reasoning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16746–16757, 2026.

[3] Zhipeng Cai, Zhuang Liu, Yunyang Xiong, Zechun Liu, Vikas Chandra, and Yangyang Shi. Vlm3: Vision language models are native 3d learners. arXiv preprint arXiv:2605.30561, 2026.

[4] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[5] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.

[6] Siyi Chen, Hugo Hadfield, Alex Zook, Mikaela Angelina Uy, Chan Hee Song, Erwin Coumans, Xuning Yang, Faisal Ladhak, Qing Qu, Stan Birchfield, et al. Volo: A physical orchestrator for open-vocabulary long-horizon manipulation. arXiv preprint arXiv:2606.07723, 2026.

[7] Siyi Chen, Mikaela Angelina Uy, Chan Hee Song, Faisal Ladhak, Adithyavairavan Murali, Qing Qu, Stan Birchfield, Valts Blukis, and Jonathan Tremblay. Spacetools: Tool-augmented spatial reasoning via double interactive rl. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 37109–37120, 2026.

[8] Zeren Chen, Xiaoya Lu, Zhijie Zheng, Pengrui Li, Lehan He, Yijin Zhou, Jing Shao, Bohan Zhuang, and Lu Sheng. Geometrically-constrained agent for spatial reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 38689–38699, 2026.

[9] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In Advances in Neural Information Processing Systems, volume 37, pages 135062–135093, 2024.

[10] Yalun Dai, Hao Li, Shulin Tian, Runmao Yao, Yuhao Dong, Fangzhou Hong, Zhaoxi Chen, Fangfu Liu, Baoliang Tian, Dingwen Zhang, et al. S-agent: Spatial tool-use elicits reasoning for spatial intelligence. arXiv preprint arXiv:2606.20515, 2026.

[11] Google DeepMind. Gemini 3.1 Pro model card, February 2026. https://deepmind.google/models/model-cards/ gemini-3-1-pro/.

[12] Andrew Guo, Bowen Wen, Jianhe Yuan, Jonathan Tremblay, Stephen Tyree, Jefrey Smith, and Stan Birchfield. Handal: A dataset of real-world manipulable object categories with pose annotations, afordances, and reconstructions. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 11428–11435. IEEE, 2023.

[13] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[14] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14953–14962, 2023.

[15] Yi Han, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Cheng Chi, Lu Sheng, and Shanghang Zhang. Tiger: Tool-integrated geometric reasoning in vision-language models for robotics. arXiv preprint arXiv:2510.07181, 2025.

[16] Stefan Hinterstoisser, Vincent Lepetit, Slobodan Ilic, Stefan Holzer, Gary Bradski, Kurt Konolige, and Nassir Navab. Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. In Asian conference on computer vision, pages 548–562. Springer, 2012.

[17] Jiaheng Hu, Mohit Shridhar, Caden Lu, Dhruv Shah, Hao-Tien Lewis Chiang, Jie Tan, and Annie Xie. What matters in orchestrating robot policies: A systematic study of hierarchical vla agents. arXiv preprint arXiv:2606.10267, 2026.

[18] Yushi Hu, Otilia Stretcu, Chun-Ta Lu, Krishnamurthy Viswanathan, Kenji Hata, Enming Luo, Ranjay Krishna, and Ariel Fuxman. Visual program distillation: Distilling tools and programmatic reasoning into vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9590– 9601, 2024.

[19] Zixuan Huang, Xin Xia, Yuxi Ren, Jianbin Zheng, Xuanda Wang, Zhixia Zhang, Hongyan Xie, Songshi Liang, Zehao Chen, Xuefeng Xiao, et al. Does your reasoning model implicitly know when to stop thinking? arXiv preprint arXiv:2602.08354, 2026.

[20] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

[21] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[22] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. arXiv preprint arXiv:1811.00982, 2018.

[23] Justin Lazarow, David Grifiths, Gefen Kohavi, Francisco Crespo, and Afshin Dehghan. Cubify anything: Scaling indoor 3d object detection. arXiv preprint arXiv:2412.04458, 2024.

[24] Zixing Lei, Changxing Liu, Yichen Xiong, Minhao Xiong, Yuanzhuo Ding, Zhipeng Zhang, Weixin Li, and Siheng Chen. Towards long-horizon embodied agents with tool-aligned vision-language-action models. arXiv preprint arXiv:2605.13119, 2026.

[25] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

[26] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

[27] Haowen Liu, Xirui Li, Shaoxiong Yao, Peng Shi, Tianyi Zhou, Jia-Bin Huang, Furong Huang, and Jiayuan Mao. Guava: An efective and universal harness for embodied manipulation. arXiv preprint arXiv:2606.18363, 2026.

[28] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

[29] Shunyu Liu, Minghao Liu, Huichi Zhou, Zhenyu Cui, Yang Zhou, Yuhao Zhou, Jialiang Gao, Heng Zhou, Yunhao Yang, Wendong Fan, et al. Veriweb: Verifiable long-chain web benchmark for agentic information-seeking. arXiv preprint arXiv:2508.04026, 2025.

[30] Tengxiao Liu, Zifeng Wang, Jin Miao, I Hsu, Jun Yan, Jiefeng Chen, Rujun Han, Fangyuan Xu, Yanfei Chen, Ke Jiang, et al. Budget-aware tool-use enables efective agent scaling. arXiv preprint arXiv:2511.17006, 2025.

[31] Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. In Advances in neural information processing systems, volume 37, pages 68803–68832, 2024.

[32] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jef Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

[33] OpenAI. Gpt-5.6: Frontier intelligence that scales with your ambition, July 2026. https://openai.com/index/ gpt-5-6/.

[34] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

[35] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. https://qwen.ai/blog?id=qwen3.5.

[36] Qwen Team. Qwen3.7-Plus: Multimodal agent intelligence, June 2026. https://qwen.ai/blog?id=qwen3.7-plus.

[37] Shouwei Ruan, Bin Wang, Zhenyu Wu, Qihui Zhu, Yuxiang Zhang, Jingzhi Li, Yubin Wang, and Xingxing Wei. Allospatial: Agentic harness framework for spatial reasoning in foundation models. arXiv preprint arXiv:2606.08952, 2026.

[38] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in neural information processing systems, volume 36, pages 68539–68551, 2023.

[39] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[40] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. In Advances in Neural Information Processing Systems, volume 36, pages 38154–38180, 2023.

[41] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.

[42] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11888–11898, 2023.

[43] HY Team, Xumin Yu, Zuyan Liu, Ziyi Wang, He Zhang, Yongming Rao, Fangfu Liu, Yani Zhang, Ruowen Zhao, Oran Wang, et al. Hy-embodied-0.5: Embodied foundation models for real-world agents. arXiv preprint arXiv:2604.07430, 2026.

[44] Stephen Tyree, Jonathan Tremblay, Thang To, Jia Cheng, Terry Mosier, Jefrey Smith, and Stan Birchfield. 6-dof pose estimation of household objects for robotic manipulation: An accessible dataset and benchmark. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13081–13088. IEEE, 2022.

[45] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

[46] Shihao Wang, Shilong Liu, Yuanguo Kuang, Xinyu Wei, Yangzhou Liu, Zhiqi Li, Yunze Man, Guo Chen, Andrew Tao, Guilin Liu, et al. Locateanything: Fast and high-quality vision-language grounding with parallel box decoding. arXiv preprint arXiv:2605.27365, 2026.

[47] Zehan Wang, Ziang Zhang, Jiayang Xu, Jialei Wang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Orient anything v2: Unifying orientation and rotation understanding. arXiv preprint arXiv:2601.05573, 2026.

[48] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances in neural information processing systems, volume 35, pages 24824–24837, 2022.

[49] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes. arXiv preprint arXiv:1711.00199, 2017.

[50] Ganlin Yang, Tianyi Zhang, Haoran Hao, Weiyun Wang, Yibin Liu, Dehui Wang, Guanzhou Chen, Zijian Cai, Junting Chen, Weijie Su, et al. Vlaser: Vision-language-action model with synergistic embodied reasoning. arXiv preprint arXiv:2510.11027, 2025.

[51] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.

[52] John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, volume 37, pages 50528–50652, 2024.

[53] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Advances in Neural Information Processing Systems, volume 37, pages 21875–21911, 2024.

[54] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025.

[55] Zhejian Yang, Yongchao Chen, Xueyang Zhou, Jiangyue Yan, Dingjie Song, Yinuo Liu, Yuting Li, Yu Zhang, Pan Zhou, Hechang Chen, et al. Agentic robot: A brain-inspired framework for vision-language-action models in embodied agents. arXiv preprint arXiv:2505.23450, 2025.

[56] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

[57] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025.

[58] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pages 113222–113244, 2025.

[59] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

[60] Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, Wenhao Tang, et al. Harness vla: Steering frozen vlas into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026.

[61] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.

[62] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. In Advances in Neural Information Processing Systems, volume 38, pages 28404–28481, 2025.

[63] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.