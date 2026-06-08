# Thinking with Imagination: Agentic Visual Spatial Reasoning with World Simulators

Chenming Zhu1,2∗, Jingli Lin2,3∗, Yilin Long2,4, Peizhou Cao2,5, Tai Wang2, Jiangmiao Pang2, Xihui Liu1,‡

1The University of Hong Kong, 2Shanghai AI Laboratory, 3Shanghai Jiao Tong University, 4Fudan University, 5Beihang University

∗Equal contribution, ‡Corresponding author

While Vision-Language Models (VLMs) have shown strong visual reasoning capabilities, their spatial reasoning abilities remain largely constrained to the observed images and text-oriented chain-ofthought. They often struggle to infer unobserved layouts, maintain cross-view consistency, and reason from alternative viewpoints when only limited egocentric observations are available. In this work, we study this problem as thinking with imagination, where a VLM actively acquires imagined visual evidence by interacting with a world simulator during reasoning. We propose Astra, an agentic spatial reasoning framework that empowers VLMs with action-conditioned visual imagination. Specifically, Astra couples Astra-VL, an RL-trained VLM policy, with Astra-WM, a Bagel-based world simulator that generates novel-view observations from context images and natural-language camera motions. To provide reliable imagined evidence, Astra-WM is trained with view consistency tuning to improve pose and content consistency across views. In the RL stage, we propose a world-simulator-in-theloop two-phase RL curriculum to stabilize tool-use exploration and advance the model’s ability to invoke the simulator only when imagined observations improve over direct answering. Experiments demonstrate that both the world simulator and the agentic policy are necessary: Astra-WM improves simulator-augmented Gemini-3-Flash on MMSI-Bench from 45.1 to 49.5, while Astra-VL improves the Qwen3-VL backbone from 29.8 to 38.8 on MMSI-Bench and from 36.8 to 42.7 on MindCube. These results show that imagined observations can provide useful spatial evidence, but effective world-model-augmented reasoning requires learning when, where, and how to imagine.

Project Page: https://zcmax.github.io/projects/Thinking-With-Imagination

## 1 Introduction

Spatial reasoning from multi-view images requires more than recognizing visible objects [1–7]. When only a few egocentric observations are available, an agent must infer unobserved layout, maintain cross-view consistency, and reason from alternative perspectives. A spatial relation may be ambiguous from the current views but become evident after a small viewpoint change, such as moving forward to inspect a camera-object relation, rotating to check viewpoint alignment, or looking beyond the current field of view to infer scene layout. Humans naturally handle such uncertainty by maintaining spatial mental models that can be updated and mentally manipulated beyond the observed pixels. In contrast, current vision-language models (VLMs) are largely tied to the images they are given, and often struggle to reason about spatial evidence that is missing from the input observations.

Recent studies have explored spatial reasoning scaffolds such as reasoning chains, intermediate views, and cognitive maps [1, 8]. These approaches suggest that VLMs may benefit from explicit spatial reasoning processes, but most of them still reason over a fixed visual context or rely on pre-defined intermediate representations. They do not allow the model to actively decide which missing viewpoint would be useful and acquire corresponding visual evidence during reasoning. This motivates an interactive formulation of spatial reasoning, where the model can seek additional evidence instead of only interpreting the observations it already has. In this paper, we study this problem as thinking with imagination: a VLM issues camera-motion queries to a world simulator, obtains imagined novel-view observations, and continues reasoning over both observed and imagined evidence. This transforms spatial mental modeling from a purely internal reconstruction problem

![](images/b9b254578a149f300534038a7db458fde5dedd8df76bffdcf4890a997f5fa5df.jpg)  
Figure 1 Reasoning trajectory of Astra. Astra tackles the challenging visual spatial reasoning task by agentic leveraging the world simulator within the reasoning process.

## into an interactive evidence-acquisition problem.

However, turning this interactive formulation into an effective system cannot be achieved by simply chaining an existing VLM with a generation model. First, the world simulator must provide reliable imagined evidence: its outputs should be not only visually plausible, but also spatially consistent with the requested camera motion and the original scene. Otherwise, the generated view may mislead the reasoning policy. Second, the VLM policy must learn to govern simulator use, including when additional evidence is needed, what viewpoint to request, and how to ground the returned observation in the original context. Our preliminary analysis confirms both bottlenecks. Off-the-shelf generation models can produce visual plausible images, but often lack spatial consistency to serve as reliable world simulators. Meanwhile, naively connecting Qwen3-VL to a world simulator can degrade performance, because the model is not trained to decide when to invoke the simulator, which camera motion to request, or how to incorporate the returned observation.

To address these challenges, we propose Astra, an agentic spatial reasoning framework that integrates an agentic VLM policy with an action-conditioned world simulator. The framework consists of two components: Astra-VL, a Qwen3-VL-based agentic reasoning model, and Astra-WM, a Bagel-based world simulator for generating action-conditioned novel views. During reasoning, Astra-VL adaptively decides when additional visual evidence is needed, plans a camera-motion query, invokes Astra-WM, and integrates the returned imagined observation into subsequent reasoning. This enables Astra to dynamically acquire alternative-view evidence through iterative reasoning and planning, rather than relying only on the initially observed images. We train both components to support reliable agentic imagination. For Astra-WM, we construct qualityverified SFT data for world simulation and fine-tune Bagel with View Consistency Tuning, enabling it to generate spatially consistent novel views conditioned on natural-language camera-motion instructions. To verify the simulator, we design pose-consistency and content-consistency evaluations, which measure whether generated views follow the requested motion and preserve scene content and spatial layout. For Astra-VL, we construct the Spatial QA data corpus and train the policy with a world-simulator-in-the-loop two-phase RL curriculum. The first phase teaches valid simulator invocation and keeps tool-use trajectories in the on-policy distribution, while the second phase encourages selective imagination by comparing tool-augmented reasoning with direct no-tool answering.

Our experiments show that world-simulator-augmented spatial reasoning requires both reliable imagined evidence and a policy that can govern its use. First, simulator-quality analysis shows that directly using off-the-shelf Bagel brings limited benefit, whereas Astra-WM achieves stronger pose and content consistency and improves simulator-augmented Gemini-3.0-Flash on MMSI-Bench from 45.1 to 49.5. This validates the need for spatially consistent world-simulator training rather than generic image generation. Second, simulator access alone is not sufficient for open-source VLMs: simply connecting Qwen3-VL to the world simulator can degrade performance, as the model has not learned when to invoke the tool, what camera motion to request, or how to ground the returned observation. In contrast, Astra-VL learns world-simulator interaction through RL, and the full Astra framework improves the Qwen3-VL backbone from 29.8 to 38.8 on MMSI-Bench and from 36.8 to 42.7 on MindCube. Together, these results show that useful imagination depends on both the quality of the world simulator and the agentic policy that decides when, where, and how to use it. Effective imagination is therefore not merely access to a generator, but a learned interaction policy for acquiring and using spatial evidence.

## 2 Related Work

VLMs for Spatial Intelligence. Spatial reasoning, which requires understanding and manipulating 3D relationships from visual observations, remains a fundamental challenge for vision-language models (VLMs). Existing efforts mainly improve spatial capability through large-scale training on specialized spatial datasets or by injecting geometric priors into model architectures, such as explicit 3D representations, depth cues, or structure-aware visual features [9–14]. While effective, these approaches often rely on static visual inputs and supervised spatial annotations, making it difficult for models to actively resolve ambiguity through interaction. Another line of work equips VLMs with external tools to perform deterministic geometric computation or visual analysis [15–17]. For instance, tool-augmented agents can decompose a spatial query into a sequence of module calls and use the returned results for answer prediction. However, these methods typically treat tools as passive solvers for predefined geometric operations, rather than as interactive environments that can provide new observations conditioned on the model’s own actions. In contrast, our work studies agentic spatial reasoning with the simulator, where the VLM learns to decide when and how to act, query a world simulator for imagined observations, and integrate these observations into multi-turn reasoning to reduce spatial uncertainty.

Thinking with Images. Recent works on thinking with images [18–22] have advanced agentic visual reasoning by enabling VLMs to interleave textual reasoning with iterative visual operations. Representative systems such as DeepEyes [23] show that reinforcement learning can induce image-text reasoning behaviors, while Pixel Reasoner [24] and Mini-o3 [25] reveal that effective tool use often requires staged training to overcome learning traps and discover deep multi-turn trajectories. SenseNova-MARS [9] further expands the tool space from perceptual image operations to open-web access and external knowledge fetching. Beyond operating on existing images, recent studies on multimodal world models suggest that visual generation can serve as an intermediate reasoning representation for physical and spatial inference [26]. While these works demonstrate the promise of image-space tools and generated visuals for reasoning, they primarily study whether visual operations or visual generation can improve reasoning. In contrast, our work formulates world simulation as a general agentic interface for visual spatial reasoning, where a VLM can actively query an action-conditioned world simulator through explicit camera-motion actions and integrate the returned imagined observations into subsequent reasoning. This formulation decouples the reasoner and planner from the simulator and allows any suitable VLM and world simulator to be composed under a shared action-observation protocol. It also enables a more detailed study of the interaction between reasoning and simulation, including when to invoke simulation, what actions to request, whether forced simulator use is sufficient, how simulator quality affects downstream performance, and how selective imagination emerges through simulator-in-the-loop reinforcement learning.

## 3 Method

In this section, we describe how Astra couples a spatially consistent world simulator with an agentic VLM policy for interactive spatial reasoning. We first formulate visual spatial reasoning as an interactive decision process in Sec. 3.1. We then introduce Astra-WM, a Bagel-based world simulator trained with view consistency tuning to produce spatially consistent imagined observations, in Sec. 3.2. Finally, we present the worldsimulator-in-the-loop two-phase RL curriculum for Astra-VL in Sec. 3.3, followed by the data construction pipeline in Sec. 3.4.

## 3.1 Problem Formulation

Task and Objective. Given a spatial question q and an initial set of context images $O _ { 0 } = \{ I _ { 1 } , \ldots , I _ { n } \}$ , the agent must produce the correct answer by reasoning over the available visual evidence. At each turn, it can either answer directly or query a world simulator for an imagined observation from an alternative viewpoint.

Observation Space. The interaction trajectory Tt records all information available to the agent up to turn t, including the question, original context images, previous reasoning steps, simulator actions, and simulator outputs. We therefore represent the agent state directly as the trajectory:

$$
s _ { t } = \tau _ { t } .
$$

After t simulator calls, the trajectory contains t imagined observations $\{ \hat { I } _ { 1 } , \ldots , \hat { I } _ { t } \}$ . If the agent invokes the simulator at state $s _ { t } ,$ the simulator returns a new tool observation

$$
o _ { t + 1 } = ( m _ { t + 1 } , \hat { I } _ { t + 1 } ) ,
$$

where $\hat { I } _ { t + 1 }$ is the generated novel view and $m _ { t + 1 }$ describes the reference image and executed camera motion. The trajectory is then updated by appending the new observation:

$$
\mathcal T _ { t + 1 } = \mathcal T _ { t } \circ o _ { t + 1 } ,
$$

where ◦ denotes ordered concatenation. This motion provenance is important because the policy must distinguish original context images from imagined views and reason about which reference viewpoint and camera motion each generated observation corresponds to.

Action Space. At each turn, the agent first produces its reasoning process and then selects one of two high-level actions: Invoke, which queries the world simulator for an imagined observation, or Answer, which outputs the final answer and terminates the trajectory. Thus, the action space is

$$
a _ { t } \in \{ \mathrm { I N V O K E } ( \rho _ { t } ) , \mathrm { A N S W E R } ( y _ { t } ) \} ,
$$

where $\rho _ { t }$ denotes the parameters of a simulator query and $y _ { t }$ denotes the final answer.

An Invoke action is parameterized by a reference image, a motion type, and a motion magnitude. The motion type is drawn from a compact camera-control vocabulary, including lateral movement, forward/backward movement, yaw rotation, vertical movement, and pitch rotation. These parameters are converted into a natural-language camera-motion instruction and sent to the world simulator.

An Answer action produces the final response and ends the trajectory. Trajectories with malformed simulator queries, missing reasoning steps in a single turn are treated as invalid.

## 3.2 View Consistency Tuning

Astra-WM is the world-simulation component of Astra, serving as the external visual imagination tool for Astra-VL. Given the context images, a reference image, and a camera-motion query from the agent, Astra-WM generates a novel-view observation that approximates what the scene would look like after executing the requested motion. Formally, when Astra-VL issues an Invoke action with query $\rho _ { t } .$ , the query specifies a reference image index $r _ { t } .$ , a motion type, and a motion magnitude. We convert $\rho _ { t }$ into a natural-language camera-motion instruction $u _ { t }$ and generate the imagined view as

![](images/318a0e36db1b1c604cdc4c72b5396adb5ad05555cdae0fc3b24a42784363a90d.jpg)  
Figure 2 Astra consists of two components: Astra-VL and Astra-WM. The overview illustrates the input-output details of both models during training and inference, as well as the two-phase reinforcement learning training pipeline.

$$
\hat { I } _ { t + 1 } = \mathcal { W } ( \mathcal { T } _ { 1 : t } , r _ { t } , u _ { t } ) ,
$$

where $\mathcal { T } _ { 1 : t } = \{ I _ { 1 } , \ldots , I _ { t } \}$ denotes the context images available to the agent, $r _ { t }$ indexes the reference image selected by the agent, and W denotes the Bagel-based Astra-WM.

A useful world simulator must provide spatially consistent evidence rather than merely visually plausible images: the generated view should preserve scene identity, follow the requested camera motion, and maintain the relative layout of visible objects across viewpoints. However, off-the-shelf Bagel is not explicitly optimized for this role, and our consistency evaluation shows that its generated views often fail to follow the requested motion or preserve scene content reliably, which can mislead Astra-VL when simulator outputs are treated as visual evidence.

To address this limitation, we introduce view consistency tuning, where Bagel is fine-tuned on quality-verified world-simulator SFT data to generate spatially consistent novel views conditioned on context images and natural-language camera-motion instructions. This converts Bagel from a generic multimodal generation model into Astra-WM, an action-conditioned world simulator that better aligns imagined observations with the requested viewpoint change and provides more reliable evidence for downstream spatial reasoning.

## 3.3 Two-Phase Reinforcement Learning

We train Astra-VL with a world-simulator-in-the-loop two-phase RL pipeline: the first phase keeps simulatorinvoking trajectories in the on-policy distribution and teaches valid tool interaction, while the second phase encourages the policy to use the simulator only when the imagined observation improves over direct answering. For rollout i, let $r _ { i } ^ { \mathrm { e m } }$ denote the final-answer exact-match reward, $r _ { i } ^ { \mathrm { f m t } }$ denote the format-validity reward, and $n _ { i } ^ { \mathrm { t o o l } }$ denote the number of simulator calls in the trajectory. Let $\lambda _ { \mathrm { f m t } }$ and $\lambda _ { \mathrm { u s e } }$ be the weights for the format reward and simulator-use reward, respectively, and let c denote the maximum number of tool calls eligible for the usage bonus.

Phase 1: Exploration and Tool Acquisition. The first phase trains Astra-VL to acquire the mechanics of valid interaction with Astra-WM. At this stage, the goal is not yet perfectly selective imagination, but to prevent the policy from collapsing to short direct answers before it has learned how to invoke the simulator. We therefore combine the answer reward and format reward with a small capped simulator-use bonus:

$$
r _ { i } ^ { ( 1 ) } = r _ { i } ^ { \mathrm { e m } } + \lambda _ { \mathrm { f m t } } r _ { i } ^ { \mathrm { f m t } } + \lambda _ { \mathrm { u s e } } \operatorname* { m i n } ( n _ { i } ^ { \mathrm { t o o l } } , c ) .
$$

The cap prevents the model from being rewarded for repeatedly calling the simulator, while still encouraging valid simulator interaction.

Phase 2: Selective Imagination. The second phase shifts the objective from invoking the simulator to using it only when it improves spatial reasoning. For each rollout group g, we additionally evaluate a direct no-tool baseline using a prompt that forbids simulator calls. This baseline is used only to estimate the benefit of simulator use and is not included in the policy update. For a normal rollout i in group $^ { g , }$ let $e _ { i }$ be its exact-match score after possible simulator use, and let $e _ { g } ^ { \mathrm { d i r e c t } }$ be the exact-match score of the corresponding no-tool baseline. We define the tool-induced gain as

$$
\Delta _ { i } = e _ { i } - e _ { g } ^ { \mathrm { d i r e c t } } .
$$

The phase-two reward combines answer correctness, format validity, optional simulator-use shaping, and the relative gain over direct answering:

$$
r _ { i } ^ { ( 2 ) } = r _ { i } ^ { \mathrm { { e m } } } + \lambda _ { \mathrm { { f m t } } } r _ { i } ^ { \mathrm { { f m t } } } + \lambda _ { \mathrm { { u s e } } } \operatorname* { m i n } ( n _ { i } ^ { \mathrm { { t o o l } } } , c ) + \alpha \operatorname* { m a x } ( 0 , \Delta _ { i } ) - \beta \operatorname* { m a x } ( 0 , - \Delta _ { i } ) .
$$

Here, α rewards trajectories whose simulator interaction improves over direct answering, while $\beta$ penalizes trajectories whose simulator use hurts the answer.

## 3.4 Data Construction

World Simulator SFT Data. Training Astra-WM requires action-conditioned novel-view supervision in the form of context images, a reference image, a camera-motion instruction, and the corresponding target observation. We construct 544k quality-verified SFT samples from posed multi-view indoor scenes, including IsaacSim [27], ScanNet++ [28], ScanNet [29], Matterport3D [30], DL3DV [31] and ARKitScenes [32]. These samples train Astra-WM to synthesize target views conditioned on scene context and explicit camera-motion queries, matching the interface used by Astra-VL during reasoning. More details are provided in our Appendix.

Agentic RL Data. To enhance Qwen3-VL’s ability to invoke tools for improved spatial reasoning, we construct an agentic reinforcement learning dataset from multiple spatial reasoning sources. Specifically, we downsample raw samples from different spatial reasoning categories in SenseNova-800K [9], VST-500K [10], and the training splits of our self-constructed Hard-UMMQA dataset. For each sample, we query Qwen3-VL-8B-Instruct, the model we aim to improve, five times under a high sampling temperature of 1.5. We then retain only the samples for which Qwen3-VL-8B-Instruct produces correct answers in no more than one out of the five trials. These retained examples are regarded as challenging samples and are used for RL training, resulting in 6k final training samples.

## 4 Experiments

## 4.1 Implementation Details

Model and Training Setup. Our Astra-VL is initialized from a Qwen3-VL-8B checkpoint and trained only with the RL stage. We implement RL training with veRL [33] and optimize the policy using GRPO. During training, we deploy the Astra-WM through vLLM-Omni [34] as the online world simulator. For RL training, we use a prompt batch size of 128, a rollout group size of 4, a PPO mini-batch size of 128, and a learning rate of $1 \times 1 0 ^ { - 5 }$ for 5 epochs. The vision tower is frozen, while the policy is trained with bfloat16 FSDP and gradient checkpointing. We adopt the Clip-Higher strategy with $\epsilon _ { \mathrm { l o w } } = 0 . 2$ and $\epsilon _ { \mathrm { h i g h } } = 0 . 2 8$ , and disable the explicit KL penalty. Each trajectory supports up to 3 tool rounds and 10 assistant turns. We set $\lambda _ { \mathrm { f m t } } = 0 . 5$ $\alpha = 0 . 1 , \beta = 0 . 0 3 , c = 1$ 1, and $\lambda _ { \mathrm { u s e } } = 0 . 0 2$ . Empirically, this exposes a key tradeoff in agentic spatial RL: sparse gain rewards can cause tool use to collapse, whereas dense usage rewards preserve exploration but may encourage over-imagination.

Table 1 Experimental Results on Spatial Reasoning Benchmarks. We compare Direct Answer, Forced Tool-Use, and Agentic Tool-Use settings. Values in parentheses denote absolute changes over the corresponding Direct Answer result of the same model. All denotes the average accuracy for each benchmark. More details on the benchmark subcategories, such as PR., are provided in the Appendix.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="5">MMSI-Bench</td><td colspan="4">MindCube-tiny</td></tr><tr><td></td><td>Attr.</td><td>Mot.</td><td>MSR</td><td>All</td><td>Rot.</td><td>Ard.</td><td>Amg.</td><td>Al</td></tr><tr><td colspan="10">Direct Answer</td></tr><tr><td rowspan="10"></td><td>Qwen3-VL-8B-Instruct [35] Qwen3-VL-30B-Instruct [</td><td>30.8</td><td>30.1</td><td>27.7 25.9</td><td>28.1 29.1</td><td>29.8</td><td>53.6</td><td>38.0</td><td>31.1</td><td>36.8</td></tr><tr><td>[35]</td><td>31.2</td><td>35.8</td><td></td><td></td><td>30.6</td><td>39.9</td><td>47.5</td><td>38.5</td><td>41.8</td></tr><tr><td>Bagel-7B-MoT[36]</td><td>33.5</td><td>27.7 23.1</td><td>25.3 22.7</td><td>30.8 30.8</td><td>31.0 25.3</td><td>34.5 34.0</td><td>31.4 26.8</td><td>42.8 33.0</td><td>34.7 31.1</td></tr><tr><td>SpatialLLM[37] Spatial-MLLM [38]</td><td>24.5 28.5</td><td>25.4</td><td>18.0</td><td>26.3</td><td>26.1</td><td>33.8</td><td>34.5</td><td>28.3</td><td>32.1</td></tr><tr><td>SpatialLadder [39]</td><td>30.3</td><td>23.3</td><td>16.0</td><td>21.2</td><td>25.4</td><td>30.5</td><td>39.8</td><td>47.8</td><td>42.3</td></tr><tr><td>SpaceR [40]</td><td>29.1</td><td>29.4</td><td>21.9</td><td>22.5</td><td>26.9</td><td>29.8</td><td>30.0</td><td>26.8</td><td>28.3</td></tr><tr><td>Video-R1 [41]</td><td>30.5</td><td>25.4</td><td>22.0</td><td>26.8</td><td>27.8</td><td>30.0</td><td>30.5</td><td>41.3</td><td>35.8</td></tr><tr><td>RoboBrain-2.0 [42]</td><td>28.9</td><td>28.8</td><td>22.5</td><td>28.0</td><td>28.9</td><td>29.7</td><td>35.8</td><td>45.2</td><td>39.6</td></tr><tr><td>VILASR [43]</td><td>35.9</td><td>26.0</td><td>21.0</td><td>23.2</td><td>29.8</td><td>34.4</td><td>25.7</td><td>29.4</td><td>29.1</td></tr><tr><td>VLaser [44]</td><td>29.8</td><td>26.9</td><td>26.0</td><td>18.9</td><td>27.3</td><td>31.5</td><td>24.8</td><td>38.2</td><td>32.6</td></tr><tr><td rowspan="4">Proprietary</td><td>GLM-4.5V 45]</td><td>35.6</td><td>36.9</td><td>29.3</td><td>30.3</td><td>33.8</td><td>60.0</td><td>25.5</td><td>42.2</td><td>39.6</td></tr><tr><td>GPT-4o [46]</td><td>28.0</td><td>32.3</td><td>36.0</td><td>30.8</td><td>30.3</td><td>33.5</td><td>35.0</td><td>37.2</td><td>35.8</td></tr><tr><td>Gemini-2.5-Pro [47]</td><td>39.0</td><td>36.2</td><td>33.3</td><td>34.3</td><td>36.9</td><td>89.5</td><td>54.5</td><td>48.8</td><td>57.5</td></tr><tr><td>Gemini-3-Flash [48]</td><td>45.6</td><td>45.4</td><td>44.0</td><td>46.0</td><td>45.4</td><td>93.0</td><td>72.0</td><td>61.7</td><td>70.5</td></tr><tr><td colspan="10">Forced Tool-Use (zero-shot)</td></tr><tr><td rowspan="3">Open-source</td><td>Qwen3-VL-8B-Instruct [ [35] [35]</td><td>30.4(-0.4)</td><td>29.5 (-0.6) 28.7 (-7.1)</td><td>19.6 (-8.1) 21.6 (-4.3)</td><td>30.8(+2.7)</td><td>28.6 (-1.2)</td><td>31.1 (-22.5)</td><td>23.7(-14.3)</td><td>26.8 (-4.3)</td><td>27.6 (-9.2)</td></tr><tr><td>Qwen3-VL-30B-Instruct</td><td>31.5(+0.3)</td><td>25.6 (-2.1)</td><td>24.7(-0.6)</td><td>28.1(-1.0)</td><td>28.9 (-1.7)</td><td>34.7 (-5.2)</td><td>32.7 (-14.8)</td><td>38.1 (-0.4)</td><td>35.7 (-6.1)</td></tr><tr><td>Bagel-7B-MoT [36] Gemini-3-Flash [48]</td><td>31.3 (-2.2)</td><td>51.5(+6.1)</td><td>43.4(-0.6)</td><td>28.7(-2.1)</td><td>29.7(-1.3)</td><td>33.9 (-0.6) 93.0(+0.0)</td><td>26.8(-4.6) 70.3(-1.7)</td><td>31.8 (-11.0) 65.0 (+3.3)</td><td>29.2(-5.5) 72.7(+2.2)</td></tr><tr><td colspan="10">50.4(+4.8) 50.3(+4.3） 49.5(+4.1)</td></tr><tr><td></td><td></td><td></td><td></td><td>Agentic Tool-Use</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-source</td><td> Astra Qwen3-VL-8B-Instruct</td><td>42.3(+11.5) 41.0(+10.9) 32.1(+4.4) 33.6(+5.5) 38.8(+9.0)</td><td></td><td></td><td></td><td></td><td>60.1(+6.5)</td><td>43.5(+5.5)</td><td>36.8 (+5.7)</td><td>42.7(+5.9)</td></tr></table>

Evaluation setup. We evaluate spatial reasoning on two held-out benchmarks: MMSI-Bench [2] and Mind-Cube [1]. MMSI-Bench contains 1,000 multi-view spatial reasoning examples across diverse spatial-relation categories, while MindCube evaluates multi-view spatial reasoning in structured 3D environments. We report exact-match answer accuracy (EM) as the main task metric. For agent behavior, we report the tool-call rate and the average number of tool calls. For world-simulator quality, we report pose consistency, which measures whether generated views follow the requested camera motion, and content consistency, which measures object-level precision, recall, and topology consistency to assess scene-content and layout preservation. More details about benchmark subcategories and simulator-quality metrics are provided in the Appendix.

Workflow Modes. We evaluate our model against several strong baselines. These include proprietary models, such GLM-4.5V, Gemini-3-Flash, as well as open-souce general VLMs and spatial VLMs. We test the models under the following three workflow settings, which control how a model is allowed to access the world simulator:

• Direct Answer : The model answers directly from the original context images without invoking the world simulator.

• Forced Tool-Use: The model is required by an explicit system prompt to invoke the world simulator for a predefined number of maximum interaction steps.

• Agentic Tool-Use: The model autonomously decides when and how to interact with the world simulator in the rollout reasoning process.

## 4.2 Main Results

Effectiveness of Astra-WM. We first evaluate whether generated observations from the world simulator provide useful evidence for spatial reasoning under the Forced Tool-Use setting. In this setting, the reasoning model is required to invoke the world simulator for a predefined number of steps before producing the final answer. We evaluate Gemini-3-Flash with different world simulators under the forced two-step tool-use setting. As shown in Tab. 2, off-the-shelf Bagel provides only limited gains, whereas our trained Astra-WM substantially improves simulator-augmented Gemini-3-Flash on MMSI-Bench, increasing the overall accuracy from 0.451 to 0.495 and the spatial-relation accuracy from 0.458 to 0.504. This shows that imagined observations can provide useful evidence for downstream spatial reasoning, but their benefit depends on the spatial consistency of the simulator. However, a reliable simulator alone is not sufficient. Prompted forced tool-use requires the model to interact with the simulator mechanically, but does not teach it whether additional observations are needed, which viewpoint would be most informative, or how the returned observation should be grounded in the original context. As shown in Tab. 1, open-source VLMs such as Qwen3-VL and Bagel can underperform their direct-answer counterparts under forced tool-use, indicating that they often fail to generate useful simulator actions or integrate imagined observations effectively.

Table 2 Simulator Quality and Zero-Shot Simulator-Augmented Reasoning on MMSI-Bench.
<table><tr><td>Model</td><td>Pose Cons.</td><td>Content Cons.</td><td>Al</td><td>PR.</td><td>Cam.-Obj.</td><td>Cam.-Reg.</td><td>Cam.-Cam.</td><td>Obj.-Obj.</td><td>Obj.-Reg.</td><td>Reg.-Reg.</td></tr><tr><td>Gemini-3-Flash</td><td></td><td></td><td>0.451</td><td>0.458</td><td>0.442</td><td>0.602</td><td>0.430</td><td>0.436</td><td>0.447</td><td>0.395</td></tr><tr><td>+ Bagel</td><td>9.0/3.0</td><td>0.356/0.396/0.102</td><td>0.458</td><td>0.469</td><td>0.523</td><td>0.434</td><td>0.443</td><td>0.483</td><td>0.459</td><td>0.468</td></tr><tr><td>+ Astra-WM30k</td><td>72.5/70.5</td><td>0.532/0.560/0.230</td><td>0.463</td><td>0.471</td><td>0.510</td><td>0.470</td><td>0.411</td><td>0.511</td><td>0.458</td><td>0.470</td></tr><tr><td> + Astra-WM60k</td><td>69.0/75.0</td><td>0.534/0.561/0.234</td><td>0.495</td><td>0.504</td><td>0.500</td><td>0.554</td><td>0.554</td><td>0.522</td><td>0.470</td><td>0.494</td></tr></table>

Table 3 Effectiveness of the Two-Phase RL Curriculum on MMSI-Bench.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="2">Tool Behavior</td><td colspan="5">MMSI-Bench</td></tr><tr><td>Tool rate</td><td>Calls/row</td><td>PR.</td><td>Attr.</td><td>Mot.</td><td>MSR</td><td>Al</td></tr><tr><td rowspan="2"> Single-stage</td><td>Tool-gain only  $( \lambda _ { \mathrm { u s e } } = 0 )$ </td><td>4.9</td><td>0.049</td><td>36.5</td><td>36.8</td><td>28.4</td><td>31.2</td><td>34.3</td></tr><tr><td>Usage bonus only  $( \lambda _ { \mathrm { u s e } } = 0 . 0 2 )$ </td><td>98.1</td><td>1.400</td><td>39.4</td><td>38.7</td><td>30.2</td><td>32.5</td><td>36.1</td></tr><tr><td rowspan="2"> Two-phase</td><td> Phase 1 only</td><td>98.0</td><td>1.120</td><td>40.1</td><td>39.2</td><td>30.8</td><td>32.9</td><td>36.8</td></tr><tr><td> Phase 1 -&gt; Phase 2 (full)</td><td>61.5</td><td>0.780</td><td> 42.3</td><td> 41.0</td><td> 32.1</td><td>33.6</td><td> 38.8</td></tr></table>

Effectiveness of Astra-VL. We next study whether an open-source VLM can be trained to use the world simulator in an agentic manner. Unlike forced tool-use, which prescribes a fixed number of simulator interactions, Astra makes tool-use decisions conditioned on the current reasoning state. This requires the policy to solve three coupled problems: deciding whether additional evidence is needed, selecting an informative camera-motion query, and grounding the returned imagined observation in the original context before answering. As shown in Tab. 1, simply prompting Qwen3-VL to use the simulator does not reliably improve performance and can even underperform direct answering, suggesting that simulator access alone is insufficient for open-source VLMs. In contrast, after world-simulator-in-the-loop RL training, Astra-VL learns to interact with the simulator effectively, and the full Astra framework improves the Qwen3-VL-8B backbone from 29.8 to 38.8 on MMSI-Bench and from 36.8 to 42.7 on MindCube. These gains indicate that the improvement comes not from tool access alone, but from learning an agentic policy that can decide when to imagine, where to query, and how to use imagined evidence.

## 4.3 Ablation Study

We ablate three factors that determine the effectiveness of Astra: the spatial consistency of the world simulator, the reward design for maintaining useful tool exploration, and the inference-time control over simulator interaction. Together, these ablations show that effective imagination requires more than simulator access alone: the imagined observations must be spatially reliable, and the policy must learn when and how to use them.

Effectiveness of View Consistency Tuning. The goal of training the world simulator is not merely photorealistic generation, but spatially reliable evidence for reasoning. As shown in Tab. 2, off-the-shelf Bagel has low pose and content consistency, and therefore brings only limited and uneven gains when used as a zero-shot simulator. After view consistency tuning, Astra-WM produces more reliable imagined observations, with substantially improved motion following and scene-content preservation. This improvement in simulator quality leads to stronger downstream spatial reasoning, yielding the best overall and spatial-relation performance among the simulator variants. The gains are especially evident on relation types that benefit from alternative viewpoints, such as Cam.–Cam., Obj.–Obj., Obj.–Reg., and Reg.–Reg. These results show that generic image generation is insufficient for world-simulator-augmented reasoning; useful imagined observations require both accurate camera-motion following and consistent scene-layout preservation.

![](images/0974a1a89429773a1516765efd003adf0d3b13256a28a061ac24e59d0acacc56.jpg)  
Figure 3 Inference-time workflow mode ablation of our Astra on MMSI-Bench.

Effectiveness of the Two-Phase RL Curriculum. Tab. 3 studies how reward design affects both tool-use behavior and downstream spatial reasoning. A single-stage tool-gain reward provides a direct comparison against no-tool answering, but it fails to maintain exploration. The policy quickly collapses toward direct answering, reaching only a 4.9% tool-call rate and 0.049 calls per rollout. This confirms that sparse relative-gain rewards are too weak early in training, when valid and useful simulator trajectories are still rare. Adding a dense simulator-use bonus prevents this collapse, but leads to the opposite failure mode. The policy invokes the simulator on 98.1% of rollouts with 1.400 calls per rollout, indicating near-universal imagination rather than selective tool use. Although this improves over the collapsed setting, it does not teach the model when the simulator is actually necessary. Similarly, the Phase-1-only model learns the mechanics of simulator interaction, but still relies on frequent tool use and lacks a reliable stopping criterion. Our full two-phase curriculum achieves the best balance between exploration and selective imagination. It obtains the strongest MMSI-Bench performance, improving the overall score to 38.8 and achieving the best results. These results show that the two phases play complementary roles: Phase 1 keeps simulator-invoking trajectories within the on-policy distribution and teaches valid tool interaction, while Phase 2 shifts the objective from merely calling the simulator to using it only when imagined observations improve over direct answering. Thus, the gain of Astra does not come from encouraging more tool calls, but from learning a more effective policy for deciding when simulator interaction is useful.

Inference-Time Workflow Mode Ablation. We further ablate the inference-time workflow mode of the same trained Astra policy on MMSI-Bench positional-relation tasks. This experiment isolates the effect of agentic control at inference time: the model parameters and world simulator are fixed, while only the workflow mode is changed among Direct Answer, Forced Tool-Use, and Agentic Tool-Use, as defined above. As shown in Fig. 3, simulator access affects different relation types unevenly. Forced tool-use substantially improves camera-centric relations, increasing Cam.–Cam. from 39.1 to 47.9, Cam.–Obj. from 39.2 to 56.8, and Cam.–Reg. from 42.2 to 48.1. This suggests that imagined viewpoints are especially useful when the question depends on camera pose, egocentric viewpoint changes, or missing visual evidence from an alternative view. However, forced tool-use hurts several object- or region-centric relations: Obj.–Obj. drops from 28.7 to 24.3, Reg.–Reg. drops from 33.1 to 31.7, and Obj.–Reg. drops from 36.1 to 31.9. These categories often require stable reasoning over the existing layout, where unnecessary generated views may introduce noise or distract the policy from reliable original observations. Agentic tool-use achieves the best overall performance, improving from 36.4 in direct-answer mode and 40.1 in forced tool-use mode to 42.3. It preserves the large gains on camera-centric categories while recovering performance on object- and region-centric categories compared with forced tool-use. This shows that the benefit of Astra comes not only from access to imagined observations, but also from the learned policy’s ability to decide whether to invoke the simulator, which action to issue, and how to ground the final answer in the returned observation.

## 5 Conclusion

We introduced Astra, an agentic spatial reasoning framework that couples a spatially consistent world simulator with an RL-trained VLM policy. By querying imagined observations from alternative viewpoints, Astra enables VLMs to acquire missing spatial evidence rather than reasoning only from the given images. Our experiments show that both reliable simulation and selective tool-use policy learning are necessary: Astra-WM improves the usefulness of generated views through view consistency tuning, while Astra-VL learns when and how to invoke the simulator through a two-phase RL curriculum. These results highlight that effective imagination is not simply access to a generator, but a learned interaction process for acquiring, grounding, and using spatial evidence.

## References

[1] Qineng Wang, Baiqiao Yin, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Jiajun Wu, Li Fei-Fei, and Manling Li. Mindcube: Spatial mental modeling from limited views, 2026. URL https://arxiv.org/abs/2506.21458.

[2] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, and Jiangmiao Pang. Mmsi-bench: A benchmark for multi-image spatial intelligence. In ICLR, 2025.

[3] JingLi Lin, Chenming Zhu, Runsen Xu, Xiaohan Mao, Xihui Liu, Tai Wang, and Jiangmiao Pang. Ostbench: Evaluating the capabilities of mllms in online spatio-temporal scene understanding. arXiv preprint arXiv:2507.07984, 2025.

[4] Jingli Lin, Runsen Xu, Shaohao Zhu, Sihan Yang, Peizhou Cao, Yunlong Ran, Miao Hu, Chenming Zhu, Yiman Xie, Yilin Long, Wenbo Hu, Dahua Lin, Tai Wang, and Jiangmiao Pang. Mmsi-video-bench: A holistic benchmark for video-based spatial intelligence, 2025. URL https://arxiv.org/abs/2512.10863.

[5] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models, 2025. URL https://arxiv.org/abs/2505.21500.

[6] Wufei Ma, Haoyu Chen, Guofeng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. arXiv preprint arXiv:2412.07825, 2024.

[7] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. arXiv preprint arXiv:2412.14171, 2024.

[8] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv:2412.14171, 2024.

[9] Yong Xien Chng, Tao Hu, Wenwen Tong, Xueheng Li, Jiandong Chen, Haojia Yu, Jiefan Lu, Hewei Guo, Hanming Deng, Chengjun Xie, et al. Sensenova-mars: Empowering multimodal agentic reasoning and search via reinforcement learning. arXiv preprint arXiv:2512.24330, 2025.

[10] Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, Yi Lin, and Hengshuang Zhao. Visual spatial tuning, 2025. URL https://arxiv.org/abs/2511.05491.

[11] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024.

[12] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. arXiv preprint arXiv:2504.01901, 2025.

[13] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Shijie Zhou, Dilin Wang, Zhicheng Yan, Hongyu Xu, Justin Theiss, Tianlong Chen, Jiachen Li, Zhengzhong Tu, Zhangyang Wang, and Rakesh Ranjan. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction, 2025. URL https://arxiv.org/abs/2505.20279.

[14] Wenbo Hu, Jingli Lin, Yilin Long, Yunlong Ran, Lihan Jiang, Yifan Wang, Chenming Zhu, Runsen Xu, Tai Wang, and Jiangmiao Pang. G2vlm: Geometry grounded vision language model with unified 3d reconstruction and spatial reasoning. arXiv preprint arXiv:2511.21688, 2025. URL https://arxiv.org/abs/2511.21688.

[15] Zeren Chen, Xiaoya Lu, Zhijie Zheng, Pengrui Li, Lehan He, Yijin Zhou, Jing Shao, Bohan Zhuang, and Lu Sheng. Geometrically-constrained agent for spatial reasoning, 2025. URL https://arxiv.org/abs/2511.22659.

[16] Yi Han, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Cheng Chi, Lu Sheng, and Shanghang Zhang. Tiger: Tool-integrated geometric reasoning in vision-language models for robotics, 2026. URL https://arxiv.org/abs/2510.07181.

[17] Zefeng Zhang, Xiangzhao Hao, Hengzhu Tang, Zhenyu Zhang, Jiawei Sheng, Xiaodong Li, Zhenyang Li, Li Gao, Daiting Shi, Dawei Yin, et al. Cooper: A unified model for cooperative perception and reasoning in spatial intelligence. arXiv preprint arXiv:2512.04563, 2025.

[18] OpenAI. Introducing o3 and o4 mini. https://openai.com/index/introducing-o3-and-o4-mini/, 2025.

[19] Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, and Xing Yu. Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.

[20] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025.

[21] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025.

[22] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025.

[23] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

[24] Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025.

[25] Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025.

[26] Jialong Wu, Xiaoying Zhang, Hongyi Yuan, Xiangcheng Zhang, Tianhao Huang, Changjing He, Chaoyi Deng, Renrui Zhang, Youbin Wu, and Mingsheng Long. Visual generation unlocks human-like reasoning through multimodal world models. arXiv preprint arXiv:2601.19834, 2026.

[27] NVIDIA. Isaac sim: Robotics simulation and synthetic data generation. https://developer.nvidia.com/isaac/sim, 2025.

[28] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.

[29] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.

[30] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017.

[31] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.

[32] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. URL https://openreview.net/forum?id=tjZjv\_qh\_CE.

[33] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297. ACM, March 2025. doi: 10.1145/3689031.3696075. URL http://dx.doi.org/10.1145/3689031.3696075.

[34] Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, et al. vllm-omni: Fully disaggregated serving for any-to-any multimodal models. arXiv preprint arXiv:2602.02204, 2026.

[35] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[36] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[37] Wufei Ma, Luoxin Ye, Celso M de Melo, Alan Yuille, and Jieneng Chen. Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17249–17260, 2025.

[38] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025.

[39] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531, 2025.

[40] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

[41] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

[42] BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, et al. Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029, 2025.

[43] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965, 2025.

[44] Ganlin Yang, Tianyi Zhang, Haoran Hao, Weiyun Wang, Yibin Liu, Dehui Wang, Guanzhou Chen, Zijian Cai, Junting Chen, Weijie Su, et al. Vlaser: Vision-language-action model with synergistic embodied reasoning. arXiv preprint arXiv:2510.11027, 2025.

[45] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025.

[46] OpenAI. Gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024.

[47] Google DeepMind. Gemini 2.5: Our most intelligent ai model. https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/, 2025.

[48] Google DeepMind. Gemini 3 flash. https://deepmind.google/models/gemini/flash/, December 2025. Model card and product documentation.

[49] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection, 2024. URL https://arxiv.org/abs/2303.05499.

[50] Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Qingping Sun, Tongxi Zhou, Jiaqi Li, Hui En Pang, Oscar Qian, Yukun Wei, Zhiqian Lin, Xuanke Shi, Kewang Deng, Xiaoyang Han, Zukai Chen, Xiangyu Fan, Hanming Deng, Lewei Lu, Liang Pan, Bo Li, Ziwei Liu, Quan Wang, Dahua Lin, and Lei Yang. Scaling spatial intelligence with multimodal foundation models, 2026. URL https://arxiv.org/abs/2511.13719.

[51] Scannet license. https://kaldir.vc.in.tum.de/scannet/ScanNet\_TOS.pdf.

[52] Matterport3d license. https://kaldir.vc.in.tum.de/matterport/MP\_TOS.pdf.

## A Additional Details on Training Data

## A.1 World Simulator SFT Data

To equip the world simulator (i.e., Bagel) with strong novel view synthesis capabilities, we construct a large-scale training dataset consisting of tuples $\left( \mathcal { T } _ { c t x } , ~ p , ~ I _ { t g t } \right)$ , where $\mathcal { T } _ { c t x }$ denotes a set of context images, p is a camera motion instruction (e.g., “move 2.5 meters to the left”), and $I _ { t g t }$ is the target image corresponding to the new viewpoint. Given $\mathcal { T } _ { c t x }$ and $p ,$ the model is required to infer the scene geometry and synthesize the observation $I _ { t g t }$ under the transformed camera pose. An overview is illustrated in Fig. 4:

Source Data Collection. Our training scenes are collected from ScanNet, ScanNet++, Matterport3D, ARKitScenes, and DL3DV (training splits), covering both indoor and outdoor environments, with a total of 11,292 scenes. Each scene is represented as a scanned RGB-D video, consisting of per-camera tuples $\{ ( I _ { i } , D _ { i } , { \bf T } _ { i } ) \} _ { i = 1 } ^ { M } ,$ where $I _ { i }$ is the RGB image, $D _ { i }$ is the depth image, and $\mathbf { T } _ { i } \in S E ( 3 )$ is the camera pose.

Camera Pairs Selection. For each scene, we sample camera pairs to construct training samples $( \mathcal { C } _ { c t x } , \ C _ { t g t } )$ where $\mathcal { C } _ { c t x }$ contains 2–3 context cameras and $C _ { t g t }$ is the target camera. To ensure the validity and effectiveness of the constructed samples, each camera pair must satisfy the following two constraints:

• View Coverage Constraint. To guarantee the completeness of background information in the target view, we require that most of the visible scene in the target camera can be observed from the context cameras. The visible point cloud of a camera is obtained via back-projection:

$$
\mathcal { P } ( C ) = \left\{ \mathbf { X } = \mathbf { T } \cdot \left( D ( u , v ) \cdot K ^ { - 1 } [ u , v , 1 ] ^ { T } \right) \right\} .\tag{1}
$$

The coverage ratio is defined as:

$$
\mathrm { C o v e r a g e } = \frac { | \mathcal { P } ( C _ { t g t } ) \cap \bigcup _ { C \in \mathcal { C } _ { c t x } } \mathcal { P } ( C ) | } { | \mathcal { P } ( C _ { t g t } ) | } .\tag{2}
$$

We require Coverage $\geq 0 . 8 5$

• Viewpoint Diversity Constraint. To avoid redundancy in visual information, we require sufficient differences between cameras in the same pair. For any pair of cameras $C _ { i } , C _ { j } \in \mathcal { C } _ { c t x } .$ , at least one of the following conditions must hold:

$$
\| \mathbf { t } _ { i } - \mathbf { t } _ { j } \| _ { \mathrm { x y } } \geq 1 . 0 \mathrm { m o r } \quad | z _ { i } - z _ { j } | \geq 1 . 0 \mathrm { m } ,\tag{3}
$$

$$
| \Delta \theta _ { i j } | \geq 3 0 ^ { \circ } \quad \mathrm { o r } \quad | \Delta \phi _ { i j } | \geq 3 0 ^ { \circ } ,\tag{4}
$$

where $\Delta \theta _ { i j }$ and $\Delta \phi _ { i j }$ denote the yaw (left-right rotation) and pitch (up-down rotation) differences, respectively.

Sample Construction. Given a camera pair $( \mathcal { C } _ { c t x } , C _ { t g t } )$ , we select one context camera as the source camera $C _ { s r c }$ . We compute the relative camera transformation:

$$
\Delta \mathbf { T } = \mathbf { T } _ { { t g t } } \mathbf { T } _ { { s r c } } ^ { - 1 } ,\tag{5}
$$

and decompose it into interpretable motion components $( d _ { x } , d _ { y } , d _ { z } , d \theta , d \phi )$ , where $( d _ { x } , d _ { y } )$ denote horizontalplane translation, $d _ { z }$ denotes vertical motion, and (dθ, dϕ) denote yaw and pitch rotations. These motion parameters are converted into natural language prompts p. (To simplify the problem into a one-step setting, we filter the data and retain only samples where a single degree of freedom dominates, i.e., only one of the five motion components changes significantly). Based on the above pipeline, we construct a total of 544,197 training samples.

## B World Simulator Evaluation

To evaluate the world simulator’s novel-view generation capability, we construct a test set using the same pipeline as for the World Simulator SFT data. Specifically, we sample from the test splits of five datasets, including DL3DV [31], ScanNet [29], ScanNet++ [28], Matterport3D [30], and ARKitScenes [32]. For each dataset, we collect 200 samples, resulting in 1,000 evaluation samples in total. We evaluate the generated views from two complementary perspectives: pose consistency and content consistency.

![](images/d56f73e6355586340fc24f5bce6661b80d0eab2fc09e4f35e2e3afaaab7b32b4.jpg)  
Figure 4 Pipeline for Constructing Training Data for the World Simulator

Pose Consistency. The pose metric evaluates whether the generated image reflects the intended camera motion. Since the world simulator only produces an RGB image, we first estimate the depth of the generated image using Depth Anything, and then recover its corresponding camera pose by aligning it with the source RGB-D observation and the known source camera pose. Based on the predicted target pose and the ground-truth target pose, we compute their relative transformations with respect to the source camera.

Following the motion representation used in World Simulator SFT data construction, each relative transformation is decomposed into five interpretable components:

$$
( d _ { x } , d _ { y } , d _ { z } , d _ { \theta } , d _ { \phi } ) ,
$$

where $( d _ { x } , d _ { y } , d _ { z } )$ denote translational changes and $( d _ { \theta } , d _ { \phi } )$ denote yaw and pitch rotations. We compare the predicted motion with the ground-truth motion along each degree of freedom. A generated sample is considered pose-consistent if it correctly preserves unchanged dimensions and predicts the correct direction and a sufficiently close magnitude for the dominant changed dimension. The final pose score is computed as the average success rate over all evaluation samples.

Content Consistency. The content metric evaluates whether the generated image preserves the visual information of the ground-truth target image, including object categories, object counts, object locations, and pairwise spatial relations. We first use a VLM to identify the union of key object categories appearing in either the generated image or the target image, using the following prompt:

Please list ALL distinct key object categories that appear in EITHER image. Use exactly the same category name for the same type of object across both images.

Given the extracted object categories, we then use GroundingDINO [49] to detect the corresponding objects in both images. Detected objects are matched between the generated image and the target image according to category consistency and bounding-box overlap. Based on the matched objects, we compute object-level recall and precision, which respectively measure how many target objects are successfully generated and how many generated objects are consistent with the target image.

In addition to object-level consistency, we further evaluate spatial topology consistency. For each pair of objects in the target image, we represent their spatial relation using the direction between their bounding-box centers. If both objects can be matched in the generated image, we compare the corresponding pairwise relation with that in the target image; otherwise, the relation is treated as missing. The topology score is computed by averaging this consistency over all object pairs.

Together, the pose and content metrics provide a holistic evaluation of whether the world simulator can follow the specified camera motion while generating a target view with consistent object-level content and spatial structure.

## C Case Study and Error Analysis

We provide qualitative case studies to better understand how the visual-CoT trajectory succeeds or fails when interacting with the world simulator. Rather than assigning each failure to a single category, we analyze representative cases along the full reasoning chain: the initial spatial hypothesis, the tool-use decision, the generated action, the simulator observation, and the final answer. These cases further illustrate why effective world-model-augmented reasoning requires more than tool access.

Case 1: Imagination Provides Missing Viewpoint Evidence. In successful cases, the original context images leave an important spatial relation ambiguous, but the model identifies the uncertainty and queries a viewpoint that directly resolves it. The returned observation provides new visual evidence that was not available in the original context, allowing the model to revise or confirm its spatial hypothesis before answering, as illustrated in Fig. 5. This case shows the intended behavior of agentic imagination: the tool is not used as a generic extra image generator, but as a targeted evidence-acquisition mechanism.

Case 2: Correct Tool Access But Uninformative Action. In some failures, the model correctly decides that additional evidence is needed, but issues an action that does not reduce the relevant uncertainty. For example, it may rotate in a direction that keeps the target object out of view, move from an unhelpful reference image, or request a viewpoint change that is unrelated to the queried relation. The resulting observation may be spatially consistent, but it does not help answer the question. This suggests that the bottleneck is not only whether to call the simulator, but whether the model can formulate informative camera-motion queries.

Case 3: Spatially Inconsistent Simulator Observation. Some failures originate from the world simulator itself. The model may request a camera motion, but the generated observation does not faithfully follow the instruction or fails to preserve the scene layout. Typical issues include incorrect viewpoint change, missing or hallucinated objects, object identity drift, or layout inconsistency across views. Because the reasoning policy treats the generated image as evidence, such simulator errors can mislead the final answer. As illustrated in Fig. 6, the model makes an incorrect spatial analysis during its initial attempt to reason about the spatial relation. The subsequently obtained simulated observation is spatially inconsistent, which further reinforces the model’s earlier mistaken spatial interpretation rather than correcting it.

Case 4: Useful observation is ignored or misused. In other cases, the simulator returns a potentially useful observation, but the model fails to incorporate it correctly. The model may continue relying on its initial hypothesis, confuse original and generated image indices, or interpret the generated view without accounting for the executed camera motion. It may also over-trust a generated view and discard reliable evidence from the original context. These cases show that imagined observations are not self-interpreting: the policy must ground each new view relative to its reference image and motion before using it in the final answer.

## D Prompt Templates and Tool Schemas

We illustrate our full prompt used during training and inference for the Agentic Workflow in Fig. 7.

## E Limitations and Future Work

The current system still falls short of robust selective imagination. First, without appropriate exploration, the open-source policy can collapse to direct answering; with a simple usage bonus, it can instead overuse the simulator. Second, the simulator may generate views that are visually plausible but not useful for the queried relation. Third, the policy may confuse original and generated image indices, over-trust a generated observation, or fail to continue exploring when the first generated view is uninformative. Finally, our current reward uses exact-match differences, which are sparse and may not capture partially useful observations.

Future work should improve each component of imagination governance: learning a stronger router for whether to imagine, training action policies that optimize expected information gain, adding verifier-style reasoning after each tool observation, and constructing preference data that contrasts helpful and harmful tool calls for the same question. A stronger version of the system should learn that tool use is neither a free reward nor a failure mode: it is an action whose value depends on the unresolved spatial uncertainty.

## F License Information for the Public Datasets Used

The raw scene data used for our World Simulator SFT data are derived from ScanNet, Matterport3D, ARKitScenes, DL3DV, and ScanNet++ [28–32]. The samples used for RL training are sourced from SenseNova-SI-800K and VST-500K [10, 50]. ARKitScenes is distributed under the Apple license, SenseNova-SI-800K is released under the Apache-2.0 license, and VST-500K is for research use only. ScanNet, ScanNet++, Matterport3D and DL3DV are governed by their respective custom licenses [51, 52]. In using these datasets, we strictly comply with their corresponding licenses and terms of use.

![](images/99650eae73450e5c580974289a72c379bef9497243c464611bb196496df7d4bc.jpg)

![](images/1ec562c07ff731bbe4672eeef518336d6774d4af64b312fcdfa84d43594e4975.jpg)  
Question: You are given 2 images from one scene. Use all images as context. From the viewpoint of Image-1, where is Image-2 taken? Options: A. front B. right C. back D.left

Response: <think> I need to determine the spatial relationship between Image-1 and Image-2 - specifically, from the viewpoint of Image-1, where Image-2 iscaptured. Image-1 shows the back/side of a beige sofa, with the camera positioned low and behind it,looking toward the room's far wall (which includes a tiled fireplace or hearth area). Image-2 shows the same sofa froma higher, more frontal angle一revealing its seat cushionand armrest,along with surrounding furniture such as a bookshelf and tripod stand.

This implies that Image-2 is taken from the front of the sofa,relative to the perspective in Image-1. But is this unambiguous?

The key ambiguity: Is Image-2 actually "in front"of Image-1(i.e., directly facing the sofa)一 or could it be rotated around the sofa in some other way (e.g., the photo was repositioned to showthe sofa "head-on" despite being originallycaptured from a side/back angle)?

To resolve whether “front" is truly the correct answer 一 or if it's misleading due to camera rotation 一 I should simulate a novel view from the camera position of Image-1 and compute what Image-2 \*would look like\* under the exact same viewpoint. This willconfirm if Image-2 corresponds to a front-facing shot from Image-1.

Therefore,Icall'world\_simulator\_tool' with Image-1 as reference,applying a forward motion to generate a simulated view from behind the sofa.If the output matches Image-2 (i.e., reveals the same angle and objects),then Image-2 is indeed taken from the front 一 butif not, we'll know for sure.</think> <action> Based on the viewpoint of the image 1, Move forward by 1.5 meters parallel to the ground.</action>

![](images/00f469f5ad198bd97b33924ca95c10c1614a0f352756fafd8c19ec534cf5c276.jpg)  
Tool: World Simulator

Action: Based on the viewpoint of the image 1, Move forward by 1.5 meters parallel to the ground.

![](images/efceb7191d0c2374e4389392cff86541a7843c657a242ef9c4527fed4e153c35.jpg)

<think> The simulated view from Image-1(moving forward by 1.5 meters) perfectly matches Image-2 一 it reveals the same camera angle, the sofa's front cushion and armrest, the bookshelf,and tripod stand.This confirms that Image-2 is captured from the front of the sofa relative to Image-1's perspective.</think> <answer>A. front</answer>

Figure 5 Case 1 of Astra.

![](images/d31eecb4d8cdce2ff57c509a5796639aa02a9a6552ef22e92c042555a61e8aee.jpg)

![](images/96299df42078a5ebabee37b4c9a486e77a7cf7d42c90228a98cd25bf77fe7875.jpg)  
Question: There are 2 egocentric images of the same place. From the second image viewpoint, how should one move to reach the camera position of the first image? Options: A. front B. right C.back D. left

Response: <think> The question asks how to move from the camera position in Image 2 to that of Image 1-essentially what camera motion would transform Image 2into Image 1.

Observing both images:

\- Image 1 shows a door on the left,an exit sign,and a shiny trash can near the entrance. - Image 2 shows a row of washing machines on the right, with a countertop on the left and a green bucket in the background.The floor drain is centered in both views.

Spatially, Image 1 is taken from a viewpoint closer to the door, looking inward toward the space. Image 2 appears to be taken from a position further down the room, looking more toward the back wall (where the machines line up).

To go from Image 2 → Image 1, the camera must move forward (toward the door)- i.e., closer to the front of the room. This corresponds to moving"front"in the options - meaning,toward the viewer's face if they were standing at the camera's position in Image 2.

However, since this is a spatial reasoning problem and involves viewing directions between two staticimages,it's safer tosimulate the scene to verify.

Let me use world\_simulator\_tool’ to generate a view from Image 2 as if moving forward to confirm whether it matches Image 1.</think> <action> Based on the viewpoint of the image 2, Move forward by 1.5 meters parallel to the ground.</action>

![](images/1a4cdee2144ea73c97b4479a64b5b2c92e7eeec19233c93186957d5bd26f85e1.jpg)  
Tool: World Simulator  
Action: Based on the viewpoint of the image 2, Move forward by 1.5 meters parallel to the ground.

![](images/287755d0eb39cc5e8d8e17d2a46a7bc43c0c79d05b7b470f78ba95d247ea641e.jpg)

<think> The simulated view confirms: moving forward from the Image 2 perspective brings the camera into alignment with Image 1-the shiny trash can, door, and exit sign come into frame,and the washing machines recede into the background.This means to reach the camera position of Image 1 from Image 2, you must move forward.</think> <answer>A. front</answer>

Figure 6 Case 2 of Astra.

![](images/37ad6c6611ba84a76cc16ab2c15af55236aa803fa58a5ca9cef2bb68e850adec.jpg)  
Figure 7 Full prompt used during training and inference for the Agentic Workflow.