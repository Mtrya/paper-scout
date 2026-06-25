# EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies

Ganlin Yang<sup>1,2∗</sup>, Zhangzheng Tu<sup>4,3∗</sup>, Yuqiang Yang<sup>2∗</sup>, Sitong Mao<sup>5</sup>, Junyi Dong<sup>5</sup> Tianxing Chen<sup>6</sup>, Jiaqi Peng<sup>7,2</sup>, Jing Xiong<sup>8,2</sup>, Jiafei Cao<sup>2</sup>, Jifeng Dai<sup>7</sup>, Wengang Zhou<sup>1</sup> Yao Mu<sup>3,2†</sup>, Tai Wang<sup>2†</sup>

<sup>1</sup>University of Science and Technology of China <sup>2</sup>Shanghai AI Laboratory <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Dalian University of Technology <sup>5</sup>Huawei Technologies Co., Ltd. <sup>6</sup>The University of Hong Kong <sup>7</sup>Tsinghua University <sup>8</sup>Peking University

Project Page: EventVLA

Abstract: Memory remains a critical bottleneck for long-horizon robotic manipulation, as standard Vision-Language-Action (VLA) policies often fail when task-relevant cues become occluded or unobservable over time. While existing memory-augmented methods utilize historical context, they either suffer from severe information bottlenecks, incur high latency via decoupled dual systems, or rely on unselective buffers that accumulate massive visual redundancies. To address these limitations, we introduce EventVLA, an end-to-end framework founded on the concept of sparse visual evidence memory that comprises two core components: foundational visual anchors to retain initial and short-term contexts, and a dynamic Keyframe Evidence Memory (KEM) module. Specifically, KEM directly predicts future keyframe probabilities from the VLA’s latent embeddings to autonomously capture and store sparse, task-critical visual events. This foresight-driven mechanism empowers the policy to dynamically evaluate the future causal utility of current observations, preserving transient visual evidence before it becomes unobservable. Furthermore, we propose RoboTwin-MeM, a diagnostic benchmark specifically designed to evaluate non-Markovian manipulation tasks with interactive visual evidence. Extensive evaluations show that across 17 memory-requiring simulation tasks and 4 real-world bimanual tasks, EventVLA achieves an average success rate improvement of +40% over state-ofthe-art memory-augmented VLAs.

Keywords: Memory, Robotic Manipulation, Robotic Benchmark

## 1 Introduction

Recent Vision-Language-Action (VLA) policies excel in generalizable and fine-grained manipulation [1, 2, 3, 4, 5], yet they predominantly operate under a strict Markovian assumption. This implicitly assumes all task-relevant information remains persistently visible. In reality, physical workspaces change dynamically, and agents must constantly retain intermediate states, such as the original location of a displaced item to guide subsequent actions. To address this non-Markovian challenge, memory-aware VLAs have emerged across three paradigms [6]. First, Dual-system Memory-VLAs [7, 8, 9] decouple cognition from control but suffer from high latency and severe error propagation. Second, Recurrent architectures [10, 11] compress history into hidden states, creating an information bottleneck that discards fine-grained visual details. Third, Memory Buffers [12, 13] preserve visual fidelity but blindly accumulate redundant frames without a selective mechanism. Consequently, a critical question remains: exactly when and what visual evidence should a VLA preserve to maximize execution success without overwhelming computational limits?

![](images/3294c82c22dad5686aec225f8d5973b05149d25b666753c6b76e0505bfc3e7d1.jpg)  
Figure 1: Overview of EventVLA. EventVLA tackles long-horizon, memory-requiring manipulation tasks by storing sparse, task-critical visual evidence. The figure illustrates the (a) non-Markovian challenge, (b) our proposed and evaluated benchmarks, (c) event-driven memory design, and (d) strong gains across simulation and real-world tasks.

To avoid the massive redundancy of standard memory buffers [12, 14], we identify that a sparse set of historical keyframes provides sufficient context for many long-horizon tasks. We define these as foundational visual anchors: the initial frame (capturing the invariant global layout) and a shortterm history window (providing local motion cues). While these heuristic anchors efficiently solve structurally simple memory-requiring tasks, they fundamentally fail in complex interactive scenarios where task-critical evidence emerges unexpectedly and subsequently disappears. For example, a robot may briefly observe an object’s color when lifting an opaque cover, or need to track a designated target that later becomes occluded. Such transient visual evidence cannot be recovered from initial or recent frames; it manifests as a interactive sparse event that must be actively captured and preserved.

Building upon this insight, we introduce EventVLA, an end-to-end framework rooted in sparse visual evidence memory. EventVLA eliminates historical redundancy by seamlessly combining foundational visual anchors with a dynamic Keyframe Evidence Memory (KEM) module. Unlike rigid, rule-based heuristics, KEM establishes an autonomous, data-driven mechanism designed to actively capture transient, interaction-driven events. Specifically, by performing foresight-driven keyframe predictions over the upcoming execution horizon, KEM empowers the VLA policy to proactively schedule sparse memory writes for critical intermediate states. This predictive strategy ensures that transient visual evidence is captured long before it becomes explicitly required by the task, seamlessly bridging the temporal gap between its brief appearance and its eventual use in downstream execution. To learn this capability without prohibitive manual annotation, we develop an offline, Qwen3-VL-based [15] automatic labeling pipeline that extracts precise keyframe supervision from demonstrations.

Beyond algorithm design, evaluating memory-augmented policies requires benchmarks that accurately capture the non-Markovian dynamics in real-world manipulation, where task-critical evidence often manifests only transiently during intermediate interactions. Because existing benchmarks like RMBench [8] can largely be solved by basic visual anchors alone, we introduce RoboTwin-MeM, a diagnostic simulation benchmark explicitly featuring such genuinely non-Markovian scenarios. It comprises 8 challenging tasks, where the required intermediate keyframes systematically scale from 1 to 5. Extensive evaluations demonstrate EventVLA’s superiority across diverse domains.

It sets a new state-of-the-art on conventional memory-oriented tasks (67.8% on RMBench) and achieves a 75.2% average success rate on the newly transient-memory-required RoboTwin-MeM, vastly outperforming existing memory-based VLAs. Furthermore, in demanding real-world bimanual tasks, EventVLA significantly surpasses both reactive $\left( \pi _ { 0 . 5 } \left[ 1 \right] \right)$ and memory-augmented $( \pi _ { M E M }$ [9]) baselines with up to 80% success rates, confirming its robust non-Markovian situational awareness.

## 2 Related Work

## 2.1 Memory-Augmented Policies for Long-Horizon Manipulation

Recent Vision-Language-Action (VLA) foundation models [1, 2, 16, 17, 5, 3, 4, 18, 19, 20, 21, 22, 23, 24, 25] achieve remarkable generalizability but are fundamentally memoryless. Operating under a strict Markovian assumption, they struggle with non-Markovian tasks where critical visual information is transient or occluded. To address this, memory-augmented VLAs have emerged across three paradigms. First, dual-system Memory-VLAs [7, 8, 9, 26, 27] use a high-level VLM for planning but suffer from error propagation and high inference latency. Second, recurrent memory architectures [10, 11, 28, 29, 30] compress histories into hidden states, creating an information bottleneck that discards fine-grained visual details. Third, Memory Buffers [31, 12, 13, 32, 33, 34, 14] retain historical frames to bypass compression; however, existing methods blindly accumulate redundant frames, drowning out sparse key evidence and incurring heavy overhead. EventVLA optimizes this paradigm by preserving only sparse visual evidence. By combining static visual anchors with a dynamic Keyframe Evidence Memory (KEM), EventVLA selectively captures transient states, balancing robust task execution with real-time computational efficiency.

## 2.2 Memory-Oriented Manipulation Benchmarks

Standard simulation suites [35, 36, 37, 38, 39] emphasize long-horizon execution rather than explicit memory reasoning, as task-relevant information typically remains persistently visible. While recent memory-centric benchmarks [40, 41, 42, 43, 44] attempt to address this, they are often limited in scale, tailored exclusively for reinforcement learning, or still feature observable states. The closest suites to ours, RMBench [8] and RoboMME [6], systematically stratify memory demands but can largely be solved by static visual anchors alone, leaving strictly non-Markovian intermediate states under-explored. To bridge this methodological gap, we introduce RoboTwin-MeM. Distinct from existing benchmarks, RoboTwin-MeM isolates genuinely non-Markovian manipulation tasks where critical visual evidence transiently emerges during interaction and subsequently disappears, providing a rigorous diagnostic platform to evaluate a VLA policy’s capacity for intermediate state retention.

## 3 EventVLA Framework

## 3.1 Problem Formulation and Foundational Visual Anchors

We formalize long-horizon robotic manipulation as a non-Markovian decision process. Standard reactive VLA policies map the current observation $o _ { t }$ and language instruction l directly to an action, i.e., $a _ { t } = \pi ( o _ { t } , l )$ , which fundamentally fails when critical information becomes occluded or unobservable over time. To address this, EventVLA incorporates an explicit, external sparse visual evidence memory buffer $M _ { t }$ to condition action generation along with the immediate observation:

$$
a _ { t } = \pi ( o _ { t } , M _ { t - 1 } , l )\tag{1}
$$

where $M _ { t - 1 }$ selectively stores key historical frames to preserve essential visual evidence while minimizing informational and computational redundancy.

The memory buffer is structured as $M _ { t } = A _ { t } \cup E _ { t }$ , seamlessly uniting foundational visual anchors $A _ { t }$ and interaction-driven event keyframes $E _ { t }$ . The visual anchors $A _ { t }$ represent a deterministic, rule-based baseline designed to capture the permanent scene layout and immediate temporal context.

Specifically, the visual anchors at timestep t consist of the initial workspace configuration $o _ { 0 }$ and a short-term history sliding window of size $K \colon$

$$
A _ { t } = o _ { 0 } \cup o _ { t - K } , . . . , o _ { t - 1 }\tag{2}
$$

Here, the initial frame $o _ { 0 }$ serves as a permanent spatial anchor, allowing the VLA model to preserve an invariant memory of the original scene arrangement before any displacements occur. Meanwhile, the short-term history $o _ { t - i }$ supplies the model with critical motion and task progression cues, enabling smooth and continuous action generation. However, since these rigid anchors cannot capture unpredictable, transient evidence arising midway through complex interactions, they are dynamically augmented by $E _ { t }$ produced by the Keyframe Evidence Memory (KEM) module, as detailed in Section 3.2 and the overall framework is shown in Fig. 2.

## 3.2 Keyframe Evidence Memory (KEM) Module

To actively capture transient, interaction-driven events that foundational visual anchors inherently miss, such as the brief exposure of an occluded object, we introduce the Keyframe Evidence Memory (KEM) module. To implement this mechanism efficiently, KEM is designed as a lightweight, parallel prediction head operating directly alongside the primary action heads. Rather than utilizing isolated features, the keyframe prediction head ingests the exact hidden states $h _ { t } \in \bar { \mathbb { R } ^ { H \times d } }$ extracted from the final layer of the VLA’s autoregressive transformer, for action horizon H. Because $h _ { t }$ naturally encapsulates the joint embedding of visual observations and action-conditioned query tokens, the keyframe head inherits a proactive awareness of the model’s future execution plan. Specifically, the keyframe head projects these shared hidden states $h _ { t }$ to a vector of

![](images/4cfe78979b193c3c19b29921415570050b7bb1b5532e7f0f6394d1fabdd8b011.jpg)  
Figure 2: EventVLA framework. EventVLA maintains a sparse visual evidence memory composed of foundational visual anchors and interaction-driven event keyframes, and uses the KEM module to proactively commit task-critical future key observations into memory.

keyframe probabilities pˆ<sub>t</sub> spanning the future chunk horizon $H \colon$

$$
\hat { \mathbf { p } } _ { t } = \sigma ( \mathbf { K E M } _ { \mathrm { m l p } } ( h _ { t } ) ) = [ \hat { p } _ { t } ^ { 1 } , \hat { p } _ { t } ^ { 2 } , \dots , \hat { p } _ { t } ^ { H } ] ^ { T } \in [ 0 , 1 ] ^ { H }\tag{3}
$$

where $\sigma ( \cdot )$ denotes the element-wise sigmoid function, and each scalar $\hat { p } _ { t } ^ { i } \in [ 0 , 1 ]$ explicitly represents the predicted probability of the i-th future execution step being a task-critical keyframe. The rationale for this chunk-wise prediction is straightforward: a purely step-wise classifier would completely miss task-critical events that transiently manifest and vanish midway through the execution window (e.g., at step $t + i$ where $0 < i < H )$ . This limitation motivates KEM to adopt a foresight-driven, chunk-wise paradigm $\hat { \mathbf { p } _ { t } }$ , empowering the VLA policy to proactively map out a “memory schedule” across the entire upcoming execution horizon.

Driven by this predictive vector $\hat { \mathbf { p } } _ { t }$ , EventVLA triggers a sparse memory write event whenever a predicted probability crosses a threshold $( \hat { p } _ { t } ^ { i } \ge \tau _ { \mathrm { c o m m i t } } )$ , dynamically committing the raw image at $t + i$ to the event buffer $E _ { t }$ . To satisfy real-time constraints, $E _ { t }$ is bounded by a maximum capacity $N _ { \mathrm { m a x } }$ , managed via a First-In-First-Out (FIFO) eviction policy. At any execution step t, these dynamically accumulated event keyframes $E _ { t - 1 }$ are seamlessly combined with the foundational visual anchors $A _ { t }$ and the immediate observation $o _ { t }$ into a single, temporally ordered sequence:

![](images/01678dd23a5838a8ac5380ac8c2a481fa9563e3bcb095d09c9ab56ba861a4de2.jpg)  
Figure 3: Overview of the 8 evaluation tasks in the RoboTwin-MeM benchmark. To rigorously evaluate the capacity for intermediate visual evidence retention, each task is explicitly parameterized by n (ranges from 1 to 5), denoting the exact number of transient, interaction-driven keyframes that must be memorized to succeed. These task-critical intermediate events are highlighted with blue borders.

$$
I _ { \mathrm { i n p u t } } = c o n c a t e n a t e ( [ A _ { t } , E _ { t - 1 } , o _ { t } ] )\tag{4}
$$

Feeding this unified sequence directly into the VLM’s vision encoder allows the self-attention layers to dynamically extract complex temporal correlations across sparse historical frames, natively endowing the model with robust situational awareness for long-horizon manipulation.

## 3.3 End-to-End Training and Inference Details

To train the KEM module without prohibitive manual annotation costs, we employ an offline Qwen3- VL [15] automated pipeline to extract ground-truth timestamps of task-critical events. To mitigate the inherent temporal ambiguity of physical interactions, we supervise chunk-wise keyframe predictions using temporally smoothed soft labels via a sequence-averaged BCE objective $\left( L _ { \mathrm { k e m } } \right)$ . The framework is optimized end-to-end alongside the standard action generation loss $\left( L _ { \mathrm { a c t i o n } } \right)$

$$
L = L _ { \mathrm { a c t i o n } } + \lambda L _ { \mathrm { k e m } }\tag{5}
$$

To bridge the train-test distribution shift while maintaining early training stability, we apply a scheduled teacher-to-student curriculum that gradually transitions memory construction from groundtruth to autonomous predictions. During online inference, continuous keyframe probabilities naturally cluster around unfolding semantic events. To enforce strict memory sparsity and prevent redundant buffer flooding, we distill these dense predictions into discrete write events using a 1D Non-Maximum Suppression (NMS) and temporal cooldown pipeline. Comprehensive mathematical formulations regarding the soft labels, curriculum, NMS algorithm, and the automated labeling pipeline are deferred to Appendix A. Additionally, complete network structures and training configurations are detailed in Appendix B.3.

## 4 RoboTwin-MeM Benchmark

To systematically evaluate the capability of VLA policies to capture and retain transient visual evidence, we introduce RoboTwin-MeM, a diagnostic simulation benchmark. Developed within the RoboTwin 2.0 [35] simulation platform and built on top of the SAPIEN [45] physics engine, RoboTwin-MeM supports both automated data synthesis and integrated policy evaluation within a unified pipeline. This infrastructure ensures scalable data generation alongside consistent, reproducible benchmarking for robotic manipulation. Furthermore, we provide fine-grained language annotations that align strictly with each action-observation pair. These annotations assign explicit linguistic descriptions to low-level interactions and state transitions, offering structured and dense supervision signals that are highly beneficial for training downstream memory modules.

The core distinction between RoboTwin-MeM and existing memory-centric suites is its explicit isolation and quantification of intermediate memory demands. While previous benchmarks often permit policies to succeed by relying merely on initial static anchors or short-term histories, RoboTwin-MeM forces the model to actively memorize unpredictable visual evidence generated midway through execution. To rigorously diagnose this capability, we explicitly parameterize task complexity using n: the exact number of intermediate event keyframes that must be dynamically preserved. As illustrated in Fig. 3, RoboTwin-MeM comprises 8 genuinely non-Markovian tasks featuring extremely long execution horizons, averaging between 430 and 1544 steps per episode. Across the benchmark, the required intermediate keyframe count n systematically ranges from 1 to 5, establishing a tiered difficulty hierarchy for non-Markovian control. The detailed task statistics and language instructions can be found in Table 4.

Crucially, this n-parameterized design allows RoboTwin-MeM to evaluate a diverse spectrum of memory capabilities beyond trivial history concatenation. First, tasks like Pick the Unhidden Block (n = 3) and Cover Blocks Hard (n = 4) demand transient memory; essential visual evidence is briefly exposed when a cover is lifted and completely disappears once closed, requiring the policy to instantly anchor this fleeting information. Second, the benchmark evaluates sequence tracking and counting logic via tasks like Press Button Keyframe $( n \in [ 2 , 5 ] )$ , where each button press represents an execution-critical event that must be sequentially registered to dictate task success. Finally, the Reproduce Route task (n = 4) tests the model’s in-context learning capacity, requiring the agent to observe a demonstration, extract randomized spatial keypoints, and leverage these cues in-context to duplicate the route. This coverage of transient recognition, event counting, and in-context imitation makes RoboTwin-MeM a rigorous benchmark for evaluating memory-augmented robotic policies.

## 5 Experiments

## 5.1 Performance on Simulation Benchmarks

To thoroughly assess the efficacy of EventVLA, we benchmark our framework against a comprehensive suite of state-of-the-art baselines, categorized into two major paradigms. For standard, reactive (non-memory-based) VLA policies, we select DP [3], ACT [5], π<sub>0.5</sub> [1], X-VLA [17], and QwenOFT [46]. For memory-augmented methods, we evaluate dual-system architectures, including MemER [7] and Mem-0 [8], as well as the end-to-end MemoryVLA [12] framework, where we reproduce its variants based on both the official OpenVLA-OFT [47] and QwenOFT [46] implementations. Our proposed EventVLA is also constructed upon the identical open-source QwenOFT backbone as its foundational base model.

Evaluation on RMBench and the Efficacy of Visual Anchors: First, we evaluate our method on RMBench [8], as shown in Table 1. Because tasks in this suite primarily rely on persistent spatial layouts and fixed motion style rather than hidden intermediate states, we deploy a streamlined version of EventVLA utilizing solely foundational visual anchors. Ex-

Table 1: Overall average success rates on RMBench. Detailed pertask breakdowns are in Appendix Table 8.
<table><tr><td>Methods</td><td>Average (%)</td></tr><tr><td>Non Memory-based VLAs DP</td><td>5.8</td></tr><tr><td>ACT</td><td>5.9</td></tr><tr><td>π0.5</td><td>10.4</td></tr><tr><td>X-VLA</td><td>9.8</td></tr><tr><td>QwenOFT</td><td>5.6</td></tr><tr><td>Dual-system Memory-VLAs</td><td></td></tr><tr><td>MemER</td><td>8.7</td></tr><tr><td>Mem-0</td><td>42.0</td></tr><tr><td>End-to-end Memory-VLAs</td><td></td></tr><tr><td>MemoryVLA (OpenVLA)</td><td>19.4</td></tr><tr><td>MemoryVLA (QwenOFT)</td><td>41.7</td></tr><tr><td>EventVLA&amp; Ablations</td><td></td></tr><tr><td>EventVLA (w/o initial)</td><td>33.7</td></tr><tr><td>EventVLA (w/o short-term)</td><td>23.8</td></tr><tr><td>EventVLA (VA only)</td><td>67.8</td></tr></table>

perimental results demonstrate that this configuration achieves an average success rate of 67.8%, securing state-of-the-art performance and proving that rule-based anchors provide sufficient context for simple memory-required long-horizon manipulation. To validate the structural necessity of these components, we conduct two ablation studies. Removing the initial frame (EventVLA w/o initial) or discarding the short-term history (EventVLA w/o short-term) causes the overall success rate to plummet to 33.7% and 23.8%, respectively. This confirms that both the initial global spatial reference and the short-term motion cues are indispensable for effective visual anchoring.

Table 2: RoboTwin-MeM benchmark results. ( Bold : best; Underlined: second-best).
<table><tr><td rowspan="3">Tasks</td><td>n=1</td><td>n=2</td><td colspan="2">n=3</td><td colspan="3">n=4</td><td>n=5</td><td rowspan="3">Total average</td></tr><tr><td>Rearrange Blocks Hard</td><td>Put Back Block Hard</td><td>Pick Objects in Order</td><td>Pick the Unhidden Block Blocks Hard</td><td>Cover</td><td>Find Seal Stamp</td><td>Reproduce Route</td><td>Press Button Keyframe</td></tr><tr><td colspan="9">▼ Non Memory-based Vision-language-action Models:</td></tr><tr><td>π0.5</td><td>20%</td><td>19%</td><td>1%</td><td>14%</td><td>0%</td><td>8%</td><td>0%</td><td>0%</td><td>7.8%</td></tr><tr><td>QwenOFT</td><td>3%</td><td>26%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>1%</td><td>3.8%</td></tr><tr><td colspan="9">▼ Dual-system Memory-based Vision-language-action Models:</td></tr><tr><td>MemER</td><td>32%</td><td>4%</td><td>12%</td><td>2%</td><td>0%</td><td>26%</td><td>3%</td><td>5%</td><td>10.5%</td></tr><tr><td>Mem-0</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0.0%</td></tr><tr><td colspan="9">▼ End-to-end Memory-based Vision-language-action Models:</td></tr><tr><td>MemoryVLA (OpenVLA)</td><td>12%</td><td></td><td></td><td></td><td>0%</td><td>10%</td><td></td><td>14%</td><td>4.9%</td></tr><tr><td>MemoryVLA (QwenOFT)</td><td>39%</td><td>0%</td><td>1%</td><td>1%</td><td>1%</td><td>11%</td><td>2%</td><td>25%</td><td>10.8%</td></tr><tr><td>EventVLA (VA only)</td><td>62%</td><td>13%</td><td>5%</td><td>20%</td><td>0%</td><td>26%</td><td>0%</td><td>18%</td><td>18.0%</td></tr><tr><td>EventVLA (VA+KEM)</td><td>62%</td><td>93%</td><td>90%</td><td>54%</td><td>94%</td><td>63%</td><td>98%</td><td>48%</td><td>75.2%</td></tr><tr><td>EventVLA (implicit memory bank)</td><td>51%</td><td>9%</td><td>16%</td><td>37%</td><td>1%</td><td>68%</td><td>2%</td><td>15%</td><td>24.9%</td></tr><tr><td>EventVLA (hard label)</td><td>59%</td><td>77%</td><td>28%</td><td>62%</td><td>85%</td><td>36%</td><td>6%</td><td>37%</td><td>48.8%</td></tr><tr><td>EventVLA (w/o NMS)</td><td>62%</td><td>93%</td><td>49% 28%</td><td>36%</td><td>10% 39%</td><td>35%</td><td>97%</td><td>45% 17%</td><td>53.4%</td></tr><tr><td>EventVLA  $( N _ { \mathrm { m a x } } = 2 )$ </td><td>51%</td><td>35%</td><td>18%</td><td>33% 28%</td><td>16%</td><td>53% 29%</td><td>0% 0%</td><td>38%</td><td>32.0%</td></tr><tr><td>EventVLA (chunk size=30)</td><td>22%</td><td>98%</td><td>2%</td><td>17%</td><td>6%</td><td>16%</td><td>10%</td><td>12%</td><td>31.1%</td></tr><tr><td>EventVLA (chunk size=15)</td><td>16%</td><td>30%</td><td></td><td></td><td></td><td></td><td></td><td></td><td>13.6%</td></tr></table>

Evaluation on RoboTwin-MeM and the Necessity of KEM: While foundational visual anchors excel on RMBench, their limitations become starkly apparent when evaluated on RoboTwin-MeM, our diagnostic suite explicitly designed to test intermediate state memory. As detailed in Table 2, relying solely on rule-driven visual anchors (VA only) yields a mere 18.0% average success rate. This sharp drop indicates that fixed historical windows are fundamentally inadequate for tasks requiring VLA policies to retain transient visual evidence generated mid-execution. To overcome this non-Markovian bottleneck, the full EventVLA framework augments these visual anchors with the dynamic Keyframe Evidence Memory (KEM) module (VA+KEM). Experimental results reveal a qualitative leap: the complete EventVLA achieves a 75.2% success rate, outperforming all baseline models by a substantial margin. This striking performance delta (from 18.0% to 75.2%) compellingly demonstrates that KEM’s dynamic event capture and foresight-driven writing mechanisms are indispensable for solving complex, long-horizon tasks that hinge on transient intermediate memory.

## Evaluation on Standard Markovian Benchmarks:

To verify that EventVLA preserves fundamental reactive control, we evaluate it on standard Markovian tasks in RoboTwin-2.0 [35] (Table 3). Rather than degrading performance, our memory mechanism slightly improves success rates over the memoryless QwenOFT baseline (80.0% to 83.8% on Easy; 78.0% to 81.6% on Hard), seamlessly complementing standard closed-loop execution.

Table 3: RoboTwin2.0 benchmark results. EventVLA outperforms its baseline foundation model QwenOFT on Markovian tasks.
<table><tr><td>Tasks</td><td>π0</td><td> $\pi _ { 0 . 5 }$ </td><td>X-VLA</td><td>QwenFast</td><td>QwenOFT EventVLA</td><td></td></tr><tr><td>Easy</td><td>65.9%</td><td>82.7%</td><td>72.8%</td><td>72.5%</td><td>80.0%</td><td>83.8%</td></tr><tr><td>Hard</td><td>58.4%</td><td>76.8%</td><td>72.8%</td><td>83.2%</td><td>78.0%</td><td>81.6%</td></tr></table>

## 5.2 Ablation Analysis of EventVLA

To systematically validate the structural design of the Keyframe Evidence Memory (KEM) module, we conduct ablation studies on the challenging RoboTwin-MeM suite (Table 2, bottom in gray).

Core Mechanisms: We observe that replacing explicit raw image concatenation with an implicit latent memory bank drastically drops the success rate from 75.2% to 24.9%, creating a severe information bottleneck. Similarly, substituting temporally smoothed soft labels with rigid binary targets destabilizes the predictive head, reducing performance to 48.8%.

Buffer and Horizon Management: Removing the NMS post-processing or restricting the buffer capacity $( N _ { \mathrm { m a x } } = 2 )$ leads to redundant frame flooding and premature FIFO eviction of critical early evidence, degrading success rates to 53.4% and 32.0%, respectively. Finally, shrinking the execution chunk size (from 50 to 30 or 15) severely truncates KEM’s foresight window, preventing proactive event scheduling and plummeting performance to 31.1% and 13.6%.

![](images/d9d9a0ddbe11ca3d9ad28d798f64427170cfeba6cd76db5ddb674d7be9e03110.jpg)  
Figure 4: Real-world experimental setups and results on the ARX ACONE bimanual robot. We evaluate four memory-intensive manipulation tasks: Find Block Easy, Find Block Hard, Pick-X-Times, and Pick in Order.

Comprehensive in-depth analyses of these ablation modes, along with real-time inference speed profiling, are deferred to Appendix C.2.

## 5.3 Real-World Robot Evaluation

To evaluate EventVLA in physical environments, we deploy our framework on the ARX ACONE bimanual robot across four non-Markovian manipulation tasks, each tested over 20 independent trials. These tasks explicitly evaluate diverse cognitive memory capabilities under real-world settings: 1) Find Block Easy and Find Block Hard require the model to remember the spatial location of a hidden block after only transient visual exposure. 2) Pick-X-Times tests counting logic, requiring the robot to read a randomized number and manipulate a block accordingly. 3) Pick in Order evaluates in-context memory by asking the robot to reproduce a randomized sequence initially pointed out by a stick. We benchmark EventVLA against a state-of-the-art non-memory model, π<sub>0.5</sub> [1], and a reproduced memory-augmented baseline, π<sub>MEM</sub> [9].

As illustrated in Fig. 4, the purely reactive $\pi _ { 0 . 5 }$ policy almost entirely fails across all tasks (achieving only 0% to 10% success rates) as it fundamentally lacks the historical context required to infer occluded states. While the memory-augmented $\pi _ { M E M }$ baseline demonstrates partial improvements (e.g., 50% on Find Block Easy), its performance degrades significantly on more complex multi event-requiring tasks like Pick-X-Times (30%) and Pick in Order (40%) due to the lossy compression of long-term history. In stark contrast, EventVLA achieves commanding success rates of 90%, 60%, 90%, and 75% across the four tasks, respectively. This robust physical performance validates that the KEM module can effectively extract and retain critical transient visual cues, empowering the VLA model with long-horizon memory awareness in the real world.

## 6 Limitations

While EventVLA effectively captures transient visual evidence, its bounded event buffer limits scalability in exceptionally long-horizon tasks $( \mathrm { e . g . , > 1 0 }$ minutes) with high event densities. Such scenarios risk buffer saturation and premature eviction of early historical cues. Future work will explore hierarchical memory or compressed representations to manage massive event sequences.

## 7 Conclusion

We introduced EventVLA, an end-to-end framework tackling non-Markovian long-horizon manipulation via sparse visual evidence memory. By uniting rule-based visual anchors with a foresight-driven Keyframe Evidence Memory (KEM) module, EventVLA proactively captures task-critical transient events, completely avoiding the redundancy of dense memory. Furthermore, we proposed RoboTwin-MeM, a diagnostic benchmark for evaluating intermediate memory capabilities. Extensive evaluations across 17 simulation and 4 real-world tasks demonstrate that EventVLA significantly outperforms state-of-the-art memory-augmented VLAs, ensuring robust memory-requiring long-horizon physical execution.

## Acknowledgments

This work is supported by Shanghai Artificial Intelligence Laboratory.

## References

[1] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[2] P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, et al. $\pi _ { 0 . 6 } ^ { * } \colon$ a vla that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[3] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[4] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv preprint arXiv:2403.03954, 2024.

[5] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[6] Y. Dai, H. Fu, J. Lee, Y. Liu, H. Zhang, J. Yang, C. Finn, N. Fazeli, and J. Chai. Robomme: Benchmarking and understanding memory for robotic generalist policies. arXiv preprint arXiv:2603.04639, 2026.

[7] A. Sridhar, J. Pan, S. Sharma, and C. Finn. Memer: Scaling up memory for robot control via experience retrieval. arXiv preprint arXiv:2510.20328, 2025.

[8] T. Chen, Y. Wang, M. Li, Y. Qin, H. Shi, Z. Li, Y. Hu, Y. Zhang, K. Wang, Y. Chen, et al. Rmbench: Memory-dependent robotic manipulation benchmark with insights into policy design. arXiv preprint arXiv:2603.01229, 2026.

[9] M. Torne, K. Pertsch, H. Walke, K. Vedder, S. Nair, B. Ichter, A. Z. Ren, H. Wang, J. Tang, K. Stachowicz, et al. Mem: Multi-scale embodied memory for vision language action models. arXiv preprint arXiv:2603.03596, 2026.

[10] L. Xiao, J. Li, J. Gao, F. Ye, Y. Jin, J. Qian, J. Zhang, Y. Wu, and X. Yu. Ava-vla: Improving vision-language-action models with active visual attention. arXiv preprint arXiv:2511.18960, 2025.

[11] A. Bulatov, Y. Kuratov, and M. Burtsev. Recurrent memory transformer. Advances in Neural Information Processing Systems, 35:11079–11091, 2022.

[12] H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang. Memoryvla: Perceptual-cognitive memory in vision-language-action models for robotic manipulation. arXiv preprint arXiv:2508.19236, 2025.

[13] H. Li, S. Yang, Y. Chen, Y. Tian, X. Yang, X. Chen, H. Wang, T. Wang, F. Zhao, D. Lin, et al. Cronusvla: Transferring latent motion across time for multi-frame prediction in manipulation. arXiv e-prints, pages arXiv–2506, 2025.

[14] X. Wang, X. Gao, J. Fu, Z. Li, D. Fortier, G. Mullins, A. Kolobov, and B. Guo. Lola: Long horizon latent action learning for general robot manipulation. arXiv preprint arXiv:2512.20166, 2025.

[15] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[16] S. Liu, B. Li, K. Ma, L. Wu, H. Tan, X. Ouyang, H. Su, and J. Zhu. Rdt2: Exploring the scaling limit of umi data towards zero-shot cross-embodiment generalization. arXiv preprint arXiv:2602.03310, 2026.

[17] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, et al. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025.

[18] T. Chen, Y. Mu, Z. Liang, Z. Chen, S. Peng, Q. Chen, M. Xu, R. Hu, H. Zhang, X. Li, et al. G3flow: Generative 3d semantic flow for pose-aware and generalizable object manipulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1735–1744, 2025.

[19] J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025.

[20] M. Lin, P. Ding, S. Wang, Z. Zhuang, Y. Liu, X. Tong, W. Song, S. Lyu, S. Huang, and D. Wang. Hif-vla: Hindsight, insight and foresight through motion representation for vision-languageaction models. arXiv preprint arXiv:2512.09928, 2025.

[21] Z. Liang, Y. Li, T. Yang, C. Wu, S. Mao, T. Nian, L. Pei, S. Zhou, X. Yang, J. Pang, et al. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. arXiv preprint arXiv:2508.20072, 2025.

[22] G. Yang, T. Zhang, H. Hao, W. Wang, Y. Liu, D. Wang, G. Chen, Z. Cai, J. Chen, W. Su, et al. Vlaser: Vision-language-action model with synergistic embodied reasoning. arXiv preprint arXiv:2510.11027, 2025.

[23] W. Shen, Y. Liu, Y. Wu, Z. Liang, S. Gu, D. Wang, T. Nian, L. Xu, Y. Qin, J. Pang, et al. Expertise need not monopolize: Action-specialized mixture of experts for vision-language-action learning. arXiv preprint arXiv:2510.14300, 2025.

[24] J. Wen, Y. Zhu, M. Zhu, Z. Tang, J. Li, Z. Zhou, X. Liu, C. Shen, Y. Peng, and F. Feng. Diffusionvla: Scaling robot foundation models via unified diffusion and autoregression. In Forty-second International Conference on Machine Learning, 2025.

[25] J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.

[26] R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daume III, A. Kolobov, F. Huang, and J. Yang.´ Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. In International Conference on Learning Representations, volume 2025, pages 54277– 54296, 2025.

[27] H. Tan, P. Co, Y. Xu, S. Rong, Y. Ji, C. Chi, X. Chen, Q. Zhang, Z. Zhao, P. Wang, et al. Actionsketcher: From reasoning to action via visual sketches for long-horizon robotic manipulation. arXiv preprint arXiv:2601.01618, 2026.

[28] H. Wang, Z. Jing, J. Ao, S. Song, X. Li, G. Huang, and C. Bai. Beyond short-horizon: Vqmemory for robust long-horizon manipulation in non-markovian simulation benchmarks. arXiv preprint arXiv:2603.09513, 2026.

[29] M. Torne, A. Tang, Y. Liu, and C. Finn. Learning long-context diffusion policies via past-token prediction. arXiv preprint arXiv:2505.09561, 2025.

[30] Y.-L. Wei, H. Liao, Y. Lin, P. Wang, Z. Liang, G. Liu, and W.-S. Zheng. Cyclemanip: Enabling cyclic task manipulation via effective historical perception and understanding. arXiv preprint arXiv:2512.01022, 2025.

[31] H. Jang, S. Yu, H. Kwon, H. Jeon, Y. Seo, and J. Shin. Contextvla: Vision-language-action model with amortized multi-frame context. arXiv preprint arXiv:2510.04246, 2025.

[32] M. Lin, X. Liang, B. Lin, L. Jingzhi, Z. Jiao, K. Li, Y. Ma, Y. Liu, S. Zhao, Y. Zhuang, et al. Echovla: Robotic vision-language-action model with synergistic declarative memory for mobile manipulation. arXiv preprint arXiv:2511.18112, 2025.

[33] Y. Lei, Z. Liang, H. Zhang, and P. Luo. Vpwem: Non-markovian visuomotor policy with working and episodic memory. arXiv preprint arXiv:2603.04910, 2026.

[34] L. Tan, J. Li, and G. Jing. Memoact: Atkinson-shiffrin-inspired memory-augmented visuomotor policy for robotic manipulation. arXiv preprint arXiv:2603.18494, 2026.

[35] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[36] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

[37] S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T.-k. Chan, et al. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. arXiv preprint arXiv:2410.00425, 2024.

[38] C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Mart´ın-Mart´ın, C. Wang, G. Levine, M. Lingelbach, J. Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pages 80–93. PMLR, 2023.

[39] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.

[40] H. Fang, M. Grotz, W. Pumacay, Y. R. Wang, D. Fox, R. Krishna, and J. Duan. Sam2act: Integrating visual foundation model with a memory architecture for robotic manipulation. arXiv preprint arXiv:2501.18564, 2025.

[41] E. Cherepanov, N. Kachaev, A. K. Kovalev, and A. I. Panov. Memory, benchmark & robots: A benchmark for solving complex tasks with reinforcement learning. arXiv preprint arXiv:2502.10550, 2025.

[42] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 44776–44791, 2023.

[43] S. Han, B. Qiu, Y. Liao, S. Huang, C. Gao, S. Yan, and S. Liu. Robocerebra: A large-scale benchmark for long-horizon robotic manipulation evaluation. Advances in Neural Information Processing Systems, 38, 2026.

[44] H. Lei, W. Song, H. Zhang, J. Pei, J. Chen, H. Yan, H. Zhao, P. Ding, Z. Zhang, L. Huang, et al. Robomemarena: A comprehensive and challenging robotic memory benchmark. arXiv preprint arXiv:2605.10921, 2026.

[45] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, et al. Sapien: A simulated part-based interactive environment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11097–11107, 2020.

[46] S. Community. Starvla: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

[47] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[48] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

## Appendix

## A Implementation Details of EventVLA

## A.1 Training Formulations and Curriculum Strategy

To obtain ground-truth (GT) keyframe supervisions, we leverage an offline automated labeling pipeline powered by Qwen3-VL [15]. By parsing raw demonstration videos alongside task descriptions, the VLM extracts the exact timestamps of task-critical intermediate events, denoted as $t ^ { * }$ . However, in physical robot execution, keyframe semantics inherently exhibit temporal ambiguity—frames immediately preceding or succeeding $t ^ { * }$ are often equally valid for capturing the visual evidence. To prevent noisy gradients caused by rigid binary supervision, we smooth the annotations into a soft target vector $\bar { \mathbf { y } _ { t } } \in [ 0 , 1 ] ^ { H }$ utilizing a raised cosine kernel. Specifically, for a future step i within a dilation radius R of a GT event $t ^ { * }$ , the soft target is defined as $\begin{array} { r } { y _ { t } ^ { i } = 0 . 5 ( 1 + \cos ( \pi \frac { | t + i - t ^ { * } | } { R } ) ) } \end{array}$ .

To supervise the chunk-wise keyframe predictions against these temporally smoothed annotations, we formulate the Keyframe Evidence Memory loss $L _ { \mathrm { k e m } }$ as a sequence-averaged Binary Cross-Entropy (BCE) objective. This explicitly aligns each predicted scalar probability $\hat { p } _ { t } ^ { i } \in [ 0 , 1 ]$ ] with its corresponding soft target $y _ { t } ^ { i } \in [ 0 , 1 ]$ across the entire future action horizon $H \colon$

$$
L _ { \mathrm { k e m } } = - \frac { 1 } { H } \sum _ { i = 1 } ^ { H } \left[ y _ { t } ^ { i } \log ( \hat { p } _ { t } ^ { i } ) + ( 1 - y _ { t } ^ { i } ) \log ( 1 - \hat { p } _ { t } ^ { i } ) \right]\tag{6}
$$

The entire framework is then optimized end-to-end via a joint objective that couples memory awareness with precise motor control:

$$
L = L _ { \mathrm { a c t i o n } } + \lambda L _ { \mathrm { k e m } }\tag{7}
$$

where $L _ { \mathrm { a c t i o n } }$ denotes the standard continuous action generation loss $( \mathrm { e . g . }$ , regression or flowmatching), and λ serves as a balancing coefficient to appropriately scale the memory supervision.

During training, constructing the event buffer $E _ { t }$ dynamically from the model’s own predictions in the early stages causes severe training instability, whereas relying exclusively on GT keyframes introduces a critical train-test distribution shift since GT keyframes are unavailable at test time. To bridge this gap, we implement a scheduled teacher-to-student curriculum. We introduce an annealing parameter α that linearly decays from 1 to 0 over the training duration. At each step, the framework decides whether to commit an observation to $E _ { t }$ using the GT keyframes with probability α (teacherforcing), or relying on its own thresholded predictions $( \hat { p } _ { t } ^ { i } \ge \tau _ { \mathrm { c o m m i t } } )$ with probability $1 - \alpha$ . This gradual transition ensures stable initial convergence while forcing the VLA policy to eventually adapt to its own autonomous memory updating cadence.

## A.2 Online Inference and Post-Processing

During online inference, the chunk-wise prediction $\hat { { \bf p } } _ { t }$ naturally yields clustered, temporally continuous high-probability scores around an unfolding semantic event. To prevent redundant frames of the same visual event from flooding the bounded buffer $E _ { t }$ , we compress the H-dimensional probability vector into discrete, sparse write events via a rigorous post-processing extraction pipeline.

First, we identify a set of local probability peaks $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ by applying the confidence threshold $\tau _ { \mathrm { c o m m i t } }$ coupled with a 1D Non-Maximum Suppression (NMS) algorithm. Specifically, a future step index i is selected as a candidate peak if its probability exceeds the threshold and represents the local maximum within a sliding temporal window of radius w:

$$
\mathcal { K } _ { t } = \left\{ i \in \{ 1 , \dots , H \} \biggm | \hat { p } _ { t } ^ { i } \geq \tau _ { \mathrm { c o m m i t } } \wedge \hat { p } _ { t } ^ { i } = \operatorname* { m a x } _ { j \in \mathcal { N } _ { w } ( i ) } \hat { p } _ { t } ^ { j } \right\}\tag{8}
$$

where $\mathcal { N } _ { w } ( i ) = [ \operatorname* { m a x } ( 1 , i - w )$ , min $( H , i + w ) ]$ denotes the NMS neighborhood.

While NMS effectively isolates local peaks, rapid consecutive events might still trigger excessive memory writes. To strictly enforce operational sparsity, a temporal cooldown period C is evaluated sequentially over the candidates. A candidate peak $i \in \mathcal { K } _ { t }$ is officially validated and committed to $E _ { t }$ if and only if $( t + i ) - t _ { \mathrm { l a s t } } > C$ , where $t _ { \mathrm { l a s t } }$ denotes the absolute physical timestamp of the most recently committed keyframe. Through this cascading extraction mechanism, the framework mathematically distills the dense predictive landscape into an optimal, highly sparse subset. This guarantees that memory allocation remains strictly tied to novel interactive evidence, maximizing information retention while seamlessly adhering to real-time execution constraints and bounded memory buffer size $N _ { \mathrm { m a x } }$

## A.3 Automated Keyframe Annotation Pipeline

To circumvent the prohibitive costs associated with dense manual frame annotation for long-horizon tasks, we develop an automated, highly scalable keyframe labeling pipeline powered by Large Vision-Language Models (VLMs). Specifically, we deploy the state-of-the-art Qwen3-VL-235B-A22B-Instruct-FP8 [15] model on a local server equipped with 8 NVIDIA A800 GPUs using the vLLM [48]s framework.

Data Pre-processing and In-Context Learning. Rather than feeding an unmanageably long continuous video stream directly into the VLM, we uniformly sample the temporal horizon into a discrete set of frames (e.g., 128 frames per episode). To ensure robust spatial awareness, particularly in scenarios involving severe occlusions, we extract and concatenate multi-view observations (e.g., global head camera and wrist camera) for each sampled timestep. Crucially, to align the VLM’s outputs with our specific definition of transient visual evidence, we employ an In-Context Learning (ICL) strategy. The prompt includes a few-shot demonstration from identical or similar tasks, containing the sampled frames alongside their ground-truth keyframe steps, establishing a rigorous template for temporal alignment and JSON-formatted outputs.

The exact system prompt utilized for the automated pipeline is presented below:

System Prompt for VLM-based Keyframe Annotation   
System Prompt:   
You are an expert robot-video keyframe annotator.   
CRITICAL REQUIREMENT:   
- The annotation MUST strictly follow task instruction.   
- Treat task instruction as the primary objective definition; if visuals are ambiguous, prioritize consistency with it.   
Episode metadata:   
- episode id: <episode id>   
- task instruction: <task instruction>   
- total frames: <total frames>   
- required keyframe count: <num keyframes>   
- provided views: <selected views>   
What to annotate:   
1. Find key state transitions for this task (e.g., stable grasp acquired, object placed, cycle transition).   
2. Keep keyframes representative and temporally ordered across the full task progress.   
3. For repeated pick/place cycles, pick the most stable and recognizable moments per cycle.   
Output format constraints:   
1. Return JSON only. No markdown, no explanations.   
2. Format: { ”keyframe steps”: [int, int, ...] }   
3. keyframe steps must:   
- have length exactly <num keyframes>   
- be strictly increasing   
- be in [0, <total frames - 1>]   
- contain no duplicates

Annotation Reliability and Error Analysis. To rigorously validate the reliability of this automated pipeline, we conducted a comprehensive cross-validation study. In the simulation environments, we compared the keyframes automatically annotated by the Qwen3-VL-235B model against the precise algorithmic ground-truth (GT) states acquired directly from the RoboTwin 2.0 [35] physics engine. The results demonstrate that the VLM’s predictions exhibit an average absolute temporal error of less than 10 timesteps. Furthermore, when deployed on the four complex real-world bimanual tasks, the prediction error remained within 50 timesteps compared to human-annotated ground truth. Given that our evaluation episodes feature extremely long horizons (often exceeding 1500 to 2000 steps), this negligible temporal variance, which is naturally accommodated by our temporally smoothed soft labels, strongly confirms that our VLM-powered automated annotation pipeline is highly reliable, precise, and ready for scalable deployment.

## B Experimental Setups and Benchmarks

## B.1 RoboTwin-MeM Benchmark Details

Table 4: Detailed statistics of our proposed RoboTwin-MeM Benchmark.
<table><tr><td>Task Name</td><td>Episodes</td><td> $\mathbf { A v } \mathbf { g } .$  #Steps</td><td>Intermediate Keyframes</td><td>Task Instruction</td></tr><tr><td>Press Button Keyframe</td><td>50</td><td>430</td><td>[2,5]</td><td>Read the two number cards, press the left button as many times as the left card shows, press the middle button as many times as the right card shows, then press the right button once.</td></tr><tr><td>Pick the Unhidden Block</td><td>50</td><td>699</td><td>3</td><td>Open the covers one by one to identify the hidden colors, close them after inspec- tion, then pick up the visible block whose color is not hidden.</td></tr><tr><td>Rearrange Blocks Hard</td><td>50</td><td>879</td><td>1</td><td>Move a chosen block from its mat to the center and press the button, return the same block to its mat and press again, then move the other block to the center and press once more.</td></tr><tr><td>Pick Objects in Order</td><td>50</td><td>1124</td><td>3</td><td>Open the covers one by one to observe the objects inside, close them after in- spection, then pick up the objects in the observed order.</td></tr><tr><td>Find Seal and Seal Stamp</td><td>50</td><td>1338</td><td>[1,4]</td><td>Open the covers one by one to find and take out the seal, close the cover after inspection, stamp with it, then return it under its original cover.</td></tr><tr><td>Reproduce Route</td><td>50</td><td>1417</td><td>4</td><td>Move the center red block to the four blue pads in a random order, returning it to the center. Then use the outside red block to repeat the same pad order.</td></tr><tr><td>Put Back Block Hard</td><td>50</td><td>1468</td><td>2</td><td>For each row, move the center block to a randomly selected outer pad in the same row, move the arm back, return the block to the center, move the arm back again, and press the button. Finally, move both blocks back to the same outer pads they first visited, then press the button.</td></tr><tr><td>Cover Blocks Hard</td><td>50</td><td>1544</td><td>4</td><td>Open the covers one by one, close them after inspection, then reopen them in the order: red, green, blue, yellow.</td></tr></table>

As introduced in Sec. 4 of the main text, RoboTwin-MeM is a diagnostic simulation suite specifically engineered to isolate and evaluate genuinely non-Markovian robotic manipulation. Unlike conventional long-horizon environments where the workspace state remains persistently visible, RoboTwin-MeM enforces strict visual occlusions and temporal delays. In these tasks, critical visual evidence, such as the hidden color of a block, the identity of an object under a cover, or a randomly generated spatial sequence, manifests only transiently during intermediate interactions before becoming completely unobservable.

To systematically quantify memory capacity, RoboTwin-MeM spans 8 complex bimanual manipulation tasks with exceptionally long execution horizons, ranging from an average of 430 to 1,544 steps per episode. The difficulty of each task is explicitly parameterized by $n \in [ 1 , 5 ]$ , which defines the exact number of intermediate keyframe events the robot must autonomously capture and retain to successfully complete the instruction.

![](images/80470ab7f53d88865d2568b59a7398a44d4dd21ac5d3b4274a3ec9dd49f18c75.jpg)  
Figure 5: Expanded real-world execution sequences of EventVLA across the four manipulation tasks. The specific task-critical intermediate keyframes, which the policy autonomously captures and commits to memory, are highlighted with blue borders.

For instance, memory-intensive tasks like Cover Blocks Hard (n = 4) and Pick Objects in Order (n = 3) require the robot to lift opaque covers to inspect hidden attributes, remember them after the covers are closed, and execute subsequent pick-and-place actions based on that stored memory. Similarly, Press Button Keyframe requires the robot to read randomized number cards and translate them into a sequential counting and pressing logic. As visualized in Fig. 3, these transient, interactiondriven keyframes (highlighted with blue borders) serve as the critical informational bridge between past observations and future actions.

The comprehensive task statistics, including the average number of steps, the required intermediate keyframe count n, and the specific language instructions for all 8 evaluation tasks, are detailed in Table 4.

## B.2 Real-world Tasks Details

To supplement the single-frame task overviews provided in Fig. 4, Fig. 5 presents the expanded, step-by-step temporal sequences for the four real-world manipulation tasks. These full execution rollouts illustrate exactly when transient visual evidence emerges during physical interaction. The task-critical intermediate keyframes that the policy must autonomously capture and commit to its dynamic event buffer, such as briefly exposing a hidden block, reading a randomized number, or observing a specific sequence pointed out by a stick, are explicitly highlighted with blue borders. By successfully isolating and retaining these sparse states, EventVLA effectively bridges the temporal gap required for non-Markovian control.

## B.3 Network Architecture and Hyper-parameters

For RMBench and RoboTwin-MeM, our EventVLA framework is built upon the open-source QwenOFT [46] architecture, which utilizes Qwen3-VL-4B-Instruct as the foundational Vision-Language Model (VLM). The visual observations are resized to $2 2 4 \times 2 2 4$ before being processed by the vision encoder.

During the training phase, the entire framework is optimized end-to-end using the AdamW optimizer for 80,000 training steps. We apply a differential learning rate strategy to ensure stable convergence: the pre-trained VLM backbone is fine-tuned with a lower learning rate of $1 \times 1 0 ^ { - 5 }$ , while the newly initialized components (the action head and the Keyframe Evidence Memory prediction head) are trained with a higher learning rate of $1 \times 1 0 ^ { - 4 }$ . The action prediction horizon (H) is set to 50 steps for all tasks.

To explicitly reflect the different memory demands of our evaluated benchmarks, we configure the memory modules differently. For RMBench, which primarily evaluates foundational visual anchoring without the need for intermediate transient memory, the policy is trained exclusively with initial and short-term visual anchors. The detailed network architecture and training hyperparameters for RMBench are summarized in Table 5.

Conversely, for the strictly non-Markovian RoboTwin-MeM, the full Keyframe Evidence Memory (KEM) module is activated. To ensure early training stability and bridge the train-test distribution shift, we apply a scheduled teacher-to-student curriculum, where the teacher-forcing probability α decays linearly from 1.0 to 0.0 over the train-

Table 5: Network Architecture and Training Hyperparameters for RMBench.
<table><tr><td>Configurations</td><td>Values</td></tr><tr><td>Network Architecture Base VLM Action Model Type Action Dimension</td><td>Qwen3-VL-4B-Instruct Optimized Fine-Tuning (OFT) 14</td></tr><tr><td>Action Horizon (H) Image Resolution</td><td>50 224 × 224</td></tr><tr><td>Training Hyper-parameters</td><td></td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Training Steps</td><td></td></tr><tr><td></td><td>80,000</td></tr><tr><td>Base VLM Learning Rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td></td><td></td></tr><tr><td>Action Head Learning Rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Per-Device Batch Size</td><td>4</td></tr><tr><td></td><td></td></tr><tr><td>Gradient Accumulation Steps</td><td>1</td></tr><tr><td>Memory Module Status</td><td>Visual Anchors Only</td></tr><tr><td>Visual Anchors At</td><td> $o _ { 0 } , o _ { t - 3 0 } , o _ { t - 1 5 }$ </td></tr></table>

ing duration. As detailed in Table 6, we also introduce specific hyper-parameters to govern the online memory extraction pipeline. The chunk-wise keyframe predictions are filtered using a commit confidence threshold of $\tau _ { \mathrm { c o m m i t } } = 0 . 5 5$ . To enforce rigorous memory sparsity, we apply a 1D Non-Maximum Suppression (NMS) sliding window with a radius of $w = 8$ , followed by a temporal cooldown period of $C = 1 0$ steps between consecutive memory writes. Finally, the dynamic event buffer is bounded by a maximum capacity of $N _ { \mathrm { m a x } } = 5$ , managed by a FIFO eviction policy to satisfy real-time computational constraints.

For physical deployment on the real-world robot platform, we adapt our framework utilize $\pi _ { 0 . 5 } \ [ 1 ]$ as the foundational Vision-Language-Action Model. The action head is configured to predict a 32-dimensional continuous action over a horizon of H = 50 steps. During fine-tuning on realworld demonstrations, the entire framework is jointly optimized for 60,000 steps using the AdamW optimizer with a global batch size of 32 in bfloat16 precision. We apply a uniform peak learning rate of $5 \times 1 0 ^ { - 5 }$ for both the base VLM and the newly initialized heads, following a cosine decay schedule with 2,000 warm-up steps. To manage the Keyframe Evidence Memory (KEM) module during physical execution, we maintain a commit confidence threshold of $\tau _ { \mathrm { c o m m i t } } = 0 . 5 5$ , a maximum event buffer capacity of $N _ { \mathrm { m a x } } = 5$ , an NMS temporal window radius of $w = 8 ,$ , and a commit cooldown period of $C = 1 0$ . The keyframe loss weight λ is set to 0.1, alongside a scheduled teacher-to-student curriculum where α decays linearly from 1.0 to 0.0. The comprehensive network architecture and training details for the real-world tasks are summarized in Table 7.

Table 6: Network Architecture and KEM Hyper-parameters for RoboTwin-MeM.
<table><tr><td>Configurations</td><td>Values</td></tr><tr><td>Base VLM Action Model Type Action Horizon (H) Optimizer Training Steps Base VLM Learning Rate KEM &amp; Action Head Learning Rate Per-Device Batch Size Gradient Accumulation Steps</td><td>Qwen3-VL-4B-Instruct Optimized Fine-Tuning (OFT) 50 AdamW  $8 0 { , } 0 0 0$   $1 \times 1 0 ^ { - 5 }$   $1 \times 1 0 ^ { - 4 }$  4 1</td></tr><tr><td>Keyframe Evidence Memory (KEM) Settings Memory Module Status Visual Anchors At Teacher-Forcing Annealing (α) Commit Confidence Threshold (τcommit)</td><td>Full EventVLA (VA + KEM)  $o _ { 0 } , o _ { t - 3 0 } , o _ { t - 1 5 }$  Linear decay (1.0 → 0.0) 0.55 5</td></tr></table>

Table 7: Network Architecture and KEM Hyper-parameters for real-robot.
<table><tr><td>Configurations</td><td>Values</td></tr><tr><td>Network Architecture &amp; Basic Training</td><td></td></tr><tr><td>Base VLM</td><td>PaliGemma  $\left( \pi _ { 0 . 5 } \right)$ </td></tr><tr><td>Image Resolution</td><td> $2 2 4 \times 2 2 4$ </td></tr><tr><td>Text Sequence Length</td><td>200</td></tr><tr><td>Action Horizon (H)</td><td>50</td></tr><tr><td>Action Dimension</td><td>32</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Optimizer Hyper-parameters</td><td> $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , e p s = 1 e - 8$ </td></tr><tr><td>Weight Decay</td><td>0.01</td></tr><tr><td>Training Steps</td><td>60,000</td></tr><tr><td>Warm-up Steps</td><td>2,000</td></tr><tr><td>Base VLM Learning Rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>KEM &amp; Action Head Learning Rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Minimum Learning Rate</td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Learning Rate Schedule</td><td>cosine decay with minimum LR</td></tr><tr><td>Global Batch Size</td><td> $^ { 3 2 }$ </td></tr><tr><td>Numerical Precision</td><td>bfloat16</td></tr><tr><td>Keyframe Evidence Memory (KEM) Settings</td><td></td></tr><tr><td>Memory Module Status</td><td>Full  $\mathrm { E v e n t V L A ( V A + K E M ) }$ </td></tr><tr><td>Visual Anchors At</td><td> $O _ { 0 } , O _ { t - 6 0 } , O _ { t - 4 0 } , O _ { t - 2 0 }$ </td></tr><tr><td>Teacher-Forcing Annealing (α)</td><td>lienar decay (1.0 → 0.0)</td></tr><tr><td>Commit Confidence Threshold (τcommit)</td><td>0.55</td></tr><tr><td>Max Event Buffer Size  $( N _ { \operatorname* { m a x } } )$ </td><td>5</td></tr><tr><td>NMS Temporal Window Radius (w)</td><td>8</td></tr><tr><td>Commit Cooldown Period (C)</td><td>10</td></tr><tr><td>Keyframe Loss Weight (λ)</td><td>0.1</td></tr><tr><td></td><td></td></tr></table>

## C Extended Experimental Results and Analysis

Table 8: RMBench benchmark results. ( Bold : best; Underlined: second-best).
<table><tr><td>Tasks</td><td>Observe and Pick Up</td><td>Rearrange Blocks</td><td>Put Back Block</td><td>Swap Blocks</td><td>Swap T</td><td>Battery Try</td><td>Blocks Ranking Try</td><td>Cover Blocks</td><td>Press Button</td><td>Total average</td></tr><tr><td colspan="9">▼ Non Memory-based Vision-language-action Models:</td><td></td></tr><tr><td>DP</td><td>1%</td><td>0%</td><td>0%</td><td>11%</td><td>20%</td><td>10%</td><td>10%</td><td>0%</td><td>0%</td><td>5.8%</td></tr><tr><td>ACT</td><td>1%</td><td>29%</td><td>0%</td><td>2%</td><td>2%</td><td>19%</td><td>0%</td><td>0%</td><td>0%</td><td>5.9%</td></tr><tr><td>π0.5</td><td>9%</td><td>13%</td><td>11%</td><td>24%</td><td>15%</td><td>16%</td><td>6%</td><td>0%</td><td>0%</td><td>10.4%</td></tr><tr><td>X-VLA</td><td>9%</td><td>13%</td><td>18%</td><td>16%</td><td>3%</td><td>26%</td><td>1%</td><td>2%</td><td>0%</td><td>9.8%</td></tr><tr><td>QwenOFT</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>14%</td><td>37%</td><td>0%</td><td>0%</td><td>5.6%</td></tr><tr><td colspan="9">▼ Dual-system Memory-based Vision-language-action Models:</td><td></td></tr><tr><td>MemER</td><td>7%</td><td>17%</td><td>0%</td><td>14%</td><td>7%</td><td>27%</td><td>0%</td><td>6%</td><td>0%</td><td>8.7%</td></tr><tr><td>Mem-0</td><td>4%</td><td>89%</td><td>90%</td><td>67%</td><td>14%</td><td>28%</td><td>18%</td><td>68%</td><td>0%</td><td>42.0%</td></tr><tr><td colspan="9">▼ End-to-end Memory-based Vision-language-action Models:</td><td></td></tr><tr><td>MemoryVLA (OpenVLA)</td><td>0%</td><td>22%</td><td>50%</td><td>17%</td><td>9%</td><td>25%</td><td>12%</td><td>40%</td><td>0%</td><td>19.4%</td></tr><tr><td>MemoryVLA (QwenOFT)</td><td>2%</td><td>53%</td><td>81%</td><td>76%</td><td>9%</td><td>33%</td><td>53%</td><td>69%</td><td>0%</td><td>41.7%</td></tr><tr><td>EventVLA (w/o initial)</td><td>10%</td><td>64%</td><td>63%</td><td>16%</td><td>8%</td><td>39%</td><td>87%</td><td>15%</td><td>2%</td><td>33.7%</td></tr><tr><td>EventVLA (w/o short-term)</td><td>15%</td><td>34%</td><td>20%</td><td>18%</td><td>94%</td><td>16%</td><td>14%</td><td>4%</td><td>0%</td><td>23.8%</td></tr><tr><td>EventVLA (visual anchors only)</td><td>21%</td><td>96%</td><td>95%</td><td>96%</td><td>87%</td><td>35%</td><td>81%</td><td>97%</td><td>3%</td><td>67.8%</td></tr></table>

## C.1 Detailed Per-Task Breakdown on RMBench

Due to space limits in the main text, we present the comprehensive task-level breakdown for all baseline and ablation models on the RMBench suite in Table 8, which is the task-specific expanded version of Table 1.

The detailed breakdown clearly illustrates that our streamlined configuration, EventVLA (visual anchors only), significantly outperforms both non-memory and prior memory-augmented baselines across the vast majority of tasks. Notably, on memory-intensive structural manipulation tasks such as Rearrange Blocks (96%), Put Back Block (95%), Swap Blocks (96%), and Cover Blocks (97%), EventVLA achieves near-perfect success rates. This demonstrates that our rule-based visual anchor ing mechanism effectively captures the persistent spatial layouts and fixed motion styles required for conventional memory-oriented scenarios without the need for complex state compression. Furthermore, the ablation variants (w/o initial and w/o short-term) show severe performance degradation across almost all tasks, confirming that both the initial global spatial reference and short-term motion cues are indispensable components of the visual anchors.

## C.2 Extended Ablation Analysis and Inference Efficiency

Extended Ablation Analysis. To deeply understand the contributions of individual design choices within the Keyframe Evidence Memory (KEM) module, we expand upon the ablation studies conducted on the RoboTwin-MeM suite (summarized in the main text Sec. 5.2 and Table 2).

• Memory Representation (Explicit Images vs. Implicit Bank): In the implicit memory bank variant, captured keyframes are aggregated into a compressed latent embedding rather than appended as explicit raw images. When handling complex tasks that demand the retention of multiple distinct events $( \mathbf { e } . \mathbf { g } . , n \geq 3 )$ , squeezing disparate historical features into a single latent vector creates a severe information bottleneck. Explicit raw image concatenation avoids this lossy compression, providing complete, lossless contextual evidence for the VLA’s multi-frame attention mechanism.

• Supervision Strategy (Soft Labels vs. Hard Labels): Physical keyframe events naturally span continuous temporal windows. Replacing our raised cosine soft labels with strict binary targets induces extreme label sparsity and heavily penalizes valid adjacent frames. This rigid supervision destabilizes the predictive head, ultimately causing it to fail in triggering essential memory writes. Soft labels provide the necessary temporal tolerance for robust event capture in environments with execution variance.

• Buffer Management (The Necessity of NMS and Capacity): Without the 1D Non-Maximum Suppression (NMS) post-processing algorithm, redundant adjacent frames rapidly flood the bounded dynamic event buffer. Conversely, a strictly minimal buffer $( \mathrm { e } . \mathrm { g } . , N _ { \mathrm { m a x } } = 2 )$ inherently lacks the structural capacity required for complex, multi-stage tasks. Both scenarios lead to premature buffer saturation and trigger early FIFO eviction, which mistakenly discards foundational historical evidence (such as the first observed hidden color) before it can be utilized. This underscores that NMS-driven event sparsity and adequate memory capacity are both vital.

Table 9: Ablation Study on EventVLA’s Inference Speed. Latency denotes the average time in seconds required for generating each chunk (s/chunk), while Throughput denotes the average number of chunks generated per second (chunks/s).
<table><tr><td rowspan="2" colspan="2">Tasks</td><td>n=1</td><td> $\mathrm { n } { = } 2$ </td><td colspan="2"> $\mathrm { n } { = } 3$ </td><td colspan="3">n=4</td><td> $\mathrm { n } { = } 5$ </td><td rowspan="2">Total</td></tr><tr><td>Rearrange Blocks Hard</td><td>Put Back Block Hard</td><td>Pick Objects in Order</td><td>Pick the Unhidden Block Blocks Hard</td><td>Cover</td><td>Find Seal Stamp</td><td>Reproduce Route</td><td>Press Button Keyframe</td></tr><tr><td rowspan="2">QwenOFT</td><td>Latency</td><td>0.31</td><td>0.36</td><td>0.36</td><td>0.39</td><td>0.41</td><td>0.39</td><td>0.30</td><td>0.32</td><td>0.36</td></tr><tr><td>Throughput</td><td>3.21</td><td>2.82</td><td>2.82</td><td>2.56</td><td>2.57</td><td>2.62</td><td>3.46</td><td>3.20</td><td>2.91</td></tr><tr><td rowspan="2">EventVLA (visual anchors only)</td><td>Latency</td><td>0.92</td><td>0.78</td><td>1.08</td><td>0.83</td><td>1.05</td><td>1.02</td><td>0.95</td><td>1.08</td><td>0.96</td></tr><tr><td>Throughput</td><td>1.11</td><td>1.35</td><td>0.93</td><td>1.21</td><td>0.96</td><td>1.02</td><td>1.07</td><td>0.94</td><td>1.07</td></tr><tr><td rowspan="2">EventVLA (VA+KEM)</td><td>Latency</td><td>0.90</td><td>0.88</td><td>1.20</td><td>0.97</td><td>1.22</td><td>1.11</td><td>1.25</td><td>1.22</td><td>1.09</td></tr><tr><td>Throughput</td><td>1.13</td><td>1.16</td><td>0.84</td><td>1.03</td><td>0.83</td><td>0.92</td><td>0.81</td><td>0.83</td><td>0.94</td></tr></table>

• Foresight Horizon (The Impact of Action Chunk Size): The execution chunk size governs KEM’s look-ahead window. Shrinking this horizon truncates the model’s predictive capacity, preventing the keyframe head from effectively anticipating and scheduling upcoming transient events, thus neutralizing KEM’s proactive memory commitment capability.

Inference Efficiency. To verify that EventVLA can be effectively deployed on physical robots, we meticulously evaluate its real-time inference speed. Table 9 details the latency and throughput of our framework across the RoboTwin-MeM benchmark.

The purely reactive QwenOFT baseline achieves an average throughput of 2.91 Hz with a latency of 0.36 seconds. Incorporating external visual anchors slightly increases the computational footprint due to the extended multi-frame input sequence, resulting in an average throughput of 1.07 Hz. When the full EventVLA framework (incorporating dynamic KEM) is deployed, it maintains an average throughput of 0.94 Hz (1.09 seconds latency). Given that VLA policies typically operate as high-level planners alongside low-level, high-frequency controllers, this throughput comfortably meets the operational constraints for real-world robotic deployment. This confirms that our sparse memory commitment strategy strikes an optimal balance between robust non-Markovian reasoning and practical real-time execution efficiency.

## D Qualitative Visualizations

## D.1 Simulation Rollouts in RoboTwin-MeM

To provide an intuitive understanding of EventVLA’s dynamic memory scheduling and execution process, we visualize the qualitative rollouts across all 8 strictly non-Markovian tasks in the RoboTwin MeM benchmark. Figure 6 illustrates the successful execution sequences for four memory-intensive tasks: Rearrange Blocks Hard, Pick the Unhidden Block, Put Back Block Hard, and Cover Blocks Hard. Figure 7 demonstrates the execution pipelines for the remaining four tasks: Press Button Keyframe, Pick Objects in Order, Find Seal and Seal Stamp, and Reproduce Route.

Across these diverse scenarios, the visualizations clearly highlight how the KEM module proactively triggers sparse memory writes the exact moment transient visual evidence emerges (e.g., observing the hidden color of a block immediately after lifting an opaque cover, or reading a randomized number). By locking these critical intermediate states into the event buffer before they become unobservable, EventVLA effectively bridges the temporal gap and seamlessly guides the subsequent long-horizon manipulation.

![](images/ba1a25f0af89b37b117ea32fcc410b6906b5912cdbb0b8ba5cb8926283d09397.jpg)  
Figure 6: Qualitative rollouts of EventVLA on four RoboTwin-MeM simulation tasks: Rearrange Blocks Hard, Pick the Unhidden Block, Put Back Block Hard, and Cover Blocks Hard.

## D.2 Real-World Robot Execution Sequences

To further validate the practical efficacy of our framework, we provide qualitative execution sequences of EventVLA deployed on the real-world ARX ACONE bimanual robot. Figure 8 showcases the successful completion of four memory-intensive manipulation tasks: Find Block Easy, Pick-X-Times, Find Block Hard, and Pick in Order.

The visualizations demonstrate that EventVLA can robustly capture and retain critical intermediate visual cues despite real-world occlusions and randomized spatial placements. Whether reading a randomized number from a paper to dictate counting logic, or observing a stick pointing at bottles to memorize an in-context sequence, the policy successfully utilizes its sparse visual evidence memory to execute complex, multi-stage physical tasks, exhibiting both strong non-Markovian remembering and spatial generalization capabilities.

![](images/689856b7fc1bd0b6d2ec27e3456478816aaeb905343696916a2b0b5a6c3b1bde.jpg)  
Figure 7: Qualitative rollouts of EventVLA on the remaining four RoboTwin-MeM simulation tasks: Press Button Keyframe, Pick Objects in Order, Find Seal and Seal Stamp, and Reproduce Route.

Find Block Easy: Cover the block with the nearest cup, then lift the cup covering the block and pick up the block.

![](images/8165199e905e757e0cae9c14100c9aa4e1edf87cf183a67caad6f07a2a2d6159.jpg)  
Pick-X-Times: Pick up and put down the block the number of times as shown on the paper.

![](images/97ca5e8b2eb728bd24c6d47e07f73f05787d21313c1594e8f9cec5af29376459.jpg)

Find Block Hard: Lift the cups on the table one by one from left to right, checking if there is a hidden cube underneath, then put the cups down. Finally, open the cup containing the cube and pick the cube up.  
![](images/03b1c737e04eb055fb4d4f468b40c723befdb5bd3ff17f10cc322942139a4198.jpg)

Pick in Order: Pick up and put down the bottles on the table in the order which they are pointed by the stick one by one at the beginning.  
![](images/dec3a6f39866ad5d6e4fd0bdb974c262d9de59b4f4cdbb7ee8eb57aa9952d734.jpg)  
Figure 8: Qualitative real-world robot execution sequences of EventVLA on four tasks: Find Block Easy, Find Block Hard, Pick-X-Times, and Pick in Order.