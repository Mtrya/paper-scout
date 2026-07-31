# TURBOVLA: REAL-TIME VISION-LANGUAGE-ACTION MODEL AT 32 HZ ON AN RTX 4090 WITH <1 GB VRAM

Hengyi Xie<sup>1∗</sup>, Chenfei Yao<sup>1∗</sup>, Xianjin Wu<sup>1</sup>, Xuanyang Xi<sup>2</sup>, Yiping Tang<sup>2</sup>, Di Xu<sup>2</sup>, Yingying Zhu<sup>1</sup>, Dingkang Liang<sup>1†</sup>, Xiang Bai<sup>1</sup>, Han Ding<sup>1</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Huawei Technologies Co. Ltd, China   
{hengyi xie, yaochenfei, dkliang, xbai}@hust.edu.cn   
{xixuanyang, tangyiping, xudi21}@huawei.com   
§ htt<sub>p</sub>s://<sub>g</sub>ithub.com/H-EmbodVis/TurboVLA   
5 https://H-EmbodVis.github.io/TurboVLA

![](images/9217f4686f086ba2250489e7476afd66fe603ac716cd778cf15792ae54482a95.jpg)  
Figure 1: TurboVLA enables compact local deployment and real-time language-conditioned manipulation with 0.2B parameters, 0.9 GB inference VRAM and 31.2 ms policy latency.

## ABSTRACT

Vision-language-action (VLA) models commonly adopt an LLM-centric $V  { \pmb { L } }  A$ pathway, where visual observations are projected into the representation space of a large language model before being decoded into robot actions. Although effective, this design incurs substantial computation and memory overhead at every policy invocation. In this work, we introduce TurboVLA, a new VLA paradigm that reformulates the conventional V → L → A pathway as a direct $\bar { V } + { \cal L } \stackrel { - } {  } A$ mapping. Instead of using a large language model as the central interface between perception and action, TurboVLA independently encodes visual observations and language instructions, directly exchanges information between them through lightweight bidirectional vision-language interaction, and predicts continuous action chunks with a compact decoder. This simple design constructs task-conditioned representations directly from visual and linguistic features, sig nificantly reducing the computational and memory costs of VLA inference. On LIBERO, TurboVLA achieves 97.7% average success with only 0.2B parameters, 31.2 ms inference latency, and 0.9 GB inference VRAM on a consumer-grade RTX 4090, matching or outperforming substantially larger VLA policies. These results establish TurboVLA as a simple and effective alternative to the prevailing LLMcentric VLA paradigm, offering a new perspective on how vision, language, and action can be connected for efficient robotic manipulation.

![](images/83b60441f5d26f6398330c43afb5ecd2ff8d660a72e81a0f0e53eb7b4fa22c8a.jpg)  
(a) The comparison between existing LLM-centric VLA (left) and our method (right)  
(b) Performance Improvement  
Figure 2: From LLM-centric VLA to TurboVLA. (a) LLM-centric VLA predicts actions from largelanguage-model representations, whereas TurboVLA directly fuses visual and instruction features for continuous control. (b) TurboVLA achieves highly competitive LIBERO performance with substantially lower latency and model scale.

## 1 INTRODUCTION

Vision-language-action (VLA) models have become a powerful framework for language-conditioned robotic manipulation, connecting visual observations, natural-language instructions, and robot actions within a unified policy (Brohan et al., 2023; Zitkovich et al., 2023; Kim et al., 2024; Octo Model Team et al., 2024; Black et al., 2025a; Physical Intelligence et al., 2025; Liu et al., 2025; Li et al., 2024; Fu et al., 2025). A common design is to place a large language model at the center of this process. Such systems effectively follow an indirect $V  \bar { L } $ A pathway: visual observations are converted into language-aligned representations, combined with the task instruction, processed by the large language model, and subsequently decoded into actions (Driess et al., 2023; Zitkovich et al., 2023; Kim et al., 2024). This design transfers broad semantic knowledge from large-scale pretraining to robot control and supports open-vocabulary understanding, semantic generalization, and high-level reasoning.

LLM-centric VLA models, however, introduce a substantial bottleneck for real-time robotic execution, which is critical for responsive interaction, high-throughput manipulation, and deployment on resource-constrained robotic platforms. As summarized in Fig. 2(a), existing LLM-centric VLA models mainly follow two action-generation designs. Autoregressive VLA models, such as Open-VLA (Kim et al., 2024) and RT-2 (Zitkovich et al., 2023), represent actions as tokens and therefore inherit the sequential decoding cost of language generation. Recent methods alleviate this cost through parallel action decoding, continuous action heads, or dedicated action experts (Black et al., 2025a; Physical Intelligence et al., 2025; Kim et al., 2025; Li et al., 2024; Shukor et al., 2025). Although these action-expert designs avoid token-by-token action generation, visual observations and instructions are still processed through language models with billions of parameters before actions are predicted. These large language-model cores impose substantial computation and memory overhead, resulting in high inference latency and limiting control frequency. This raises a more fundamental question: how to design a simple, elegant and efficient VLA that directly maps vision and language to actionsfor execution-level manipulation, without centering on a large language model?

Our key observation is that language is necessary for instruction-conditioned manipulation, but execution-level control need not be centered on a large language model. Once the instruction already specifies the intended manipulation skill, the execution policy does not need to perform open-ended language generation or autonomous task decomposition. Instead, it primarily needs to use the instruction to determine how the current visual evidence should guide action. In current LLM-centric VLA models, this interaction is mediated through a general-purpose language-model representation whose broad reasoning and generative capacity exceeds the requirements of many execution-level tasks. A lightweight text encoder, such as BERT (Devlin et al., 2019), can provide the executionrelevant semantics of the instruction, while compact cross-modal interaction allows language and vision to jointly construct a control-oriented representation (Lynch & Sermanet, 2020; Shridhar et al., 2022; Jang et al., 2022; Mees et al., 2022b). This suggests a different VLA paradigm: rather than organizing perception and control around an LLM-centered latent space, vision and language can interact directly to form representations specialized for continuous action prediction.

Therefore, we propose TurboVLA, a simple yet efficient $V + L \to A$ model for real-time languageconditioned manipulation. As shown in Fig. 2(a), TurboVLA separately processes visual observations and task instructions using a visual encoder and a lightweight text encoder. Inspired by the efficient cross-modal interaction used in advanced visual grounding models such as Grounding DINO (Liu et al., 2024b), TurboVLA replaces the large-language-model-centered execution pathway with direct vision-language interaction, avoiding the computation and memory overhead of processing multimodal inputs through a billion-parameter language model. A compact cross-attention module efficiently fuses instruction and visual features, which are then decoded into continuous action chunks in a single forward pass (Zhao et al., 2023), without autoregressive action-token generation. Thi lightweight design significantly reduces inference latency and GPU memory usage.

Extensive experiments show that TurboVLA achieves real-time execution while preserving strong manipulation performance. On a consumer-grade RTX 4090, it requires only 31.2 ms of end-to-end policy latency, measured from receiving the current multimodal observation to producing an action chunk, corresponding to more than 30 action chunk predictions per second (32 Hz). TurboVLA contains only 0.2B parameters, approximately 6% the parameter count of $\pi _ { 0 . 5 }$ (Physical Intelligence et al., 2025), while using less than 1 GB VRAM during inference. Despite this lightweight design, TurboVLA achieves 97.7% average success on LIBERO (Liu et al., 2023), matching the performance of substantially larger VLA systems. As summarized in Fig. 2(b), TurboVLA provides a favorable trade-off among manipulation performance, inference latency and model scale, thereby lowering the hardware barrier to deploying language-conditioned manipulation policies in latency-sensitive and resource-constrained robotic systems. More broadly, these results motivate the VLA community to examine whether execution-level control must remain centered on large language models and to evaluate future systems beyond task success alone. Our contributions are summarized as follows:

• We revisit the LLM-centric design of existing VLA models and identify the large language model core as a major bottleneck for real-time action execution. Based on this analysis, we introduce a real-time VLA paradigm that retains language conditioning while removing the large language model from execution-level control.

• We propose TurboVLA, a simple and efficient $V + L $ A model that combines lightweight instruction encoding, direct vision-language interaction, robot-state conditioning, and nonautoregressive continuous action chunk prediction.

• Experiments on LIBERO show that TurboVLA achieves 97.7% average success while running at over 30 online policy inferences per second on a consumer-grade RTX 4090 with only 0.2B parameters and less than 1 GB of inference VRAM. Beyond LIBERO, TurboVLA remains effective in challenging bimanual and real-world settings.

## 2 RELATED WORK

Vision-language-action models. Vision-language-action (VLA) models integrate visual observations, task instructions, and action prediction within a unified policy, often leveraging large-scale visionlanguage pretraining for semantic generalization. RT-1 (Brohan et al., 2023) demonstrated scalable transformer-based robot control, while RT-2 (Zitkovich et al., 2023) and OpenVLA (Kim et al., 2024) adapted pretrained vision-language models to robot trajectories through an action-token interface. Continuous-control models such as $\pi _ { 0 }$ (Black et al., 2025a) and $\pi _ { 0 . 5 }$ (Physical Intelligence et al., 2025) instead attach dedicated action experts to pretrained multimodal backbones. This generalist VLA direction has also been advanced through cross-embodiment datasets (O’Neill et al., 2024), reusable policy representations (Octo Model Team et al., 2024), diffusion-based robot policies (Liu et al., 2025; Li et al., 2024), and foundation models for diverse embodiments (Bjorck et al., 2025; Bu et al., 2025; Wang et al., 2026c). Recent approaches augment VLA learning with visual foresight (Yang et al., 2026), predictive world knowledge (Zhang et al., 2025c), and latent reasoning (Bai et al., 2026).

Other methods incorporate geometry-aware control representations through pose-centric pretraining or point-action interaction (Lin et al., 2026a; Chen et al., 2026a). Recent work further extends VLA policies to dynamic manipulation by incorporating temporal motion cues and short-horizon future prediction (Fang et al., 2026). Together, these works demonstrate the benefits of large pretrained multimodal representations and increasingly expressive intermediate representations. TurboVLA focuses on a different architectural choice: rather than routing every control step through a large generative multimodal backbone, it encodes task text separately and integrates it directly with visual observations and robot state for execution-level action prediction.

Efficient execution in VLA policies. Recent work improves VLA efficiency through both action-side redesign and backbone-side optimization. Action-as-token policies inherit the sequential decoding process of language models, motivating continuous action experts (Black et al., 2025a; Physical Intelligence et al., 2025; Kim et al., 2025), compact and structured action tokenizers (Pertsch et al., 2025; Liu et al., 2026), and action chunking with parallel decoding (Liu et al., 2025; Liang et al., 2026c). Compact VLA architectures, including TinyVLA (Wen et al., 2025), RoboMamba (Liu et al., 2024a), SmolVLA (Shukor et al., 2025), and Evo-1 (Lin et al., 2026b), reduce model scale or inference cost while retaining pretrained multimodal representations. A complementary line of work reduces redundant backbone computation through quantization (Xu et al., 2026; Wang et al., 2025), token reuse or pruning (Xu et al., 2025; Jiang et al., 2025), dynamic depth (Yang et al., 2025), structural pruning (Wang et al., 2026a; Zhang et al., 2026), and distillation (Chen et al., 2026c; Jeon et al., 2026). Other methods improve responsiveness without changing the base policy, including asynchronous action-chunk execution (Black et al., 2025b), streaming inference and horizon-aware flow sampling (Lu et al., 2026), and speculative inference (Niu et al., 2026). These approaches accelerate action generation or reduce computation while largely retaining a large multimodal backbone as the execution representation. In contrast, TurboVLA removes the large generative language backbone from the low-level control pathway and constructs the action representation directly from compact visual, textual, and proprioceptive features.

Language interfaces for robot control. Textual instructions can serve as task specifications that condition perception and control rather than as prompts for generative language modeling. Early imitation-learning methods demonstrated that a shared policy can map visual observations and naturallanguage commands directly to different manipulation behaviors (Lynch & Sermanet, 2020; Stepputtis et al., 2020). CLIPort (Shridhar et al., 2022) combines pretrained vision-language semantics with a spatial manipulation pathway, while BC-Z (Jang et al., 2022) conditions a multi-task policy on pretrained text or human-video embeddings. CALVIN (Mees et al., 2022b) and HULC (Mees et al., 2022a) extend textual task conditioning to long-horizon control from unstructured demonstrations. PerAct (Shridhar et al., 2023) incorporates textual goals into a voxel-based transformer policy, whereas VIMA (Jiang et al., 2023) represents tasks through interleaved textual and visual prompts. Beyond execution-level policies, embodied multimodal language models combine language understanding, 3D grounding, and task scheduling to generate grounded action plans (Liang et al., 2026a). Such planning-oriented capabilities are complementary to the efficient execution-level control studied in this work. These works establish textual representations as effective task inputs for robot control. TurboVLA studies this interface under the current VLA paradigm, examining whether a compact text encoder and direct vision-text interaction are sufficient for high-performance, real-time continuous control.

## 3 PRELIMINARIES

LLM-centric vision-language-action models. Most existing VLA models (Kim et al., 2024; Black et al., 2025a; Physical Intelligence et al., 2025; Zhou et al., 2025; Fu et al., 2025) place a large language model at the center of the vision-to-action pathway. Given visual observations ${ \mathcal { O } } _ { n } .$ , a visual encoder first extracts visual features and projects them into the token space of the language model. The projected visual tokens are then concatenated with the tokenized task instruction and jointly processed by the large language model:

$$
\widetilde { Z } _ { n } ^ { v } = P _ { v } ( E _ { v } ( \mathcal { O } _ { n } ) ) , \qquad H _ { n } ^ { L } = F _ { L } \left( \left[ \widetilde { Z } _ { n } ^ { v } ; \mathrm { T o k } ( x ) \right] \right) ,\tag{1}
$$

where $E _ { v }$ denotes the visual encoder, $P _ { v }$ maps visual features into the language model embedding space, Tok(x) denotes the instruction tokens, and $F _ { L }$ is the large language model. Importantly, the stage L is not merely responsible for encoding language. It serves as the central representational bridge between visual perception and robot action: visual information is aligned with the languagemodel space, integrated with the task instruction, and transformed into the multimodal representation from which actions are predicted. We therefore summarize this prevailing computation pathway as $V  { \pmb { L } }  A$ , where L denotes the LLM-centered multimodal interface.

![](images/c6baaf69e9f3466a1bb57c7563afb20d7f733a00d1eb543fe2e161d24a6d0a3d.jpg)  
(a) Overall Architecture  
Figure 3: Overview of TurboVLA. (a) TurboVLA simply yet efficiently maps visual observations and language instructions to continuous action chunks through compact modality encoders, visionlanguage interaction, and an action chunk decoder. (b) The interaction module is designed as simple as possible. It uses stacked bidirectional cross-attention to produce vision-aware instruction features and instruction-conditioned visual features.

Existing LLM-centric VLA models mainly differ in how actions are generated from $H _ { n } ^ { L } .$ . Autoregressive models discretize robot actions and predict them sequentially from the language model representation (Zitkovich et al., 2023; Kim et al., 2024), whereas action-expert models use a separate continuous decoder,

$$
\hat { \bf A } _ { n } = D _ { \mathrm { a c t } } \left( H _ { n } ^ { L } , s _ { n } \right) ,\tag{2}
$$

to generate actions in parallel (Black et al., 2025a; Physical Intelligence et al., 2025; Kim et al., 2025; Li et al., 2024). Although action-expert models avoid token-by-token action generation, they preserve the same representational dependency, as the action decoder operates on features produced by the large language model. Thus, despite using different action-generation mechanisms, both designs retain L as the central bridge from visual perception to action prediction.

Direct vision-language interaction. Cross-attention provides a simple and efficient mechanism for directly exchanging information between visual and language features. Given visual features $Z ^ { v }$ and instruction features $Z ^ { l }$ , language-conditioned visual features can be obtained by

$$
\widetilde { Z } ^ { v } = Z ^ { v } + \mathrm { A t t n } \left( Q _ { v } , K _ { l } , V _ { l } \right) ,\tag{3}
$$

while vision-aware instruction features are produced by exchanging the query and context modalities. Such bidirectional interaction allows task language to shape visual processing while visual context refines the instruction representation. Vision-language grounding models such as Grounding DINO (Liu et al., 2024b) employ this type of direct cross-modal interaction to establish fine-grained correspondence between textual concepts and visual content. While these models use the resulting features for object localization, we use direct vision-language interaction to construct control-oriented representations for continuous action prediction.

## 4 TURBOVLA

We introduce TurboVLA, a direct and simple $V + L \to A$ paradigm for execution-level languageconditioned manipulation. As illustrated in Fig. 3(a), TurboVLA first encodes visual observations and the task instruction using a vision encoder and a lightweight text encoder. A simple and compact vision-language interaction module then directly exchanges information between the two modalities to construct action-ready features. Finally, an action chunk decoder combines these features with the current robot state and predicts a complete sequence of continuous actions in a single forward pass. Unlike LLM-centric VLA models, our method does not route visual and textual inputs through a large language model before action prediction.

## 4.1 MULTIMODAL FEATURE ENCODING

To reduce the overhead of an LLM-centered execution pathway while retaining simple yet sufficient instruction understanding, TurboVLA uses compact modality-specific encoders. Execution-level instructions typically specify manipulation skills through objects, attributes, and spatial relations, without requiring open-ended generation or task-level planning. We therefore encode instructions with a lightweight encoder such as BERT (Devlin et al., 2019) and process visual observations with a vision encoder. As shown in Fig. 3(a), the resulting features are projected into a shared hidden dimension d for subsequent vision-language interaction and action prediction. Given a task instruction $x ,$ the text encoder extracts token-level instruction features:

$$
Z ^ { l } = P _ { l } \left( f _ { \mathrm { t e x t } } ( x ) \right) \in \mathbb { R } ^ { N _ { l } \times d } ,\tag{4}
$$

where P<sub>l</sub> projects the encoder output into the policy dimension and $N _ { l }$ is the number of instruction tokens. We retain the complete token sequence rather than a pooled embedding so that objects, attributes, and spatial relations remain available for fine-grained visual conditioning.

For each camera observation $I _ { n } ^ { ( i ) }$ , the image encoder extracts spatial visual features, which are projected and augmented with positional and camera-view embeddings:

$$
Z _ { n } ^ { v , ( i ) } = P _ { v } \left( f _ { \mathrm { i m g } } \left( I _ { n } ^ { ( i ) } \right) \right) + E _ { \mathrm { p o s } } ^ { ( i ) } + e _ { \mathrm { v i e w } } ^ { ( i ) } , \qquad Z _ { n } ^ { v } = \left[ Z _ { n } ^ { v , ( 1 ) } ; \dots ; Z _ { n } ^ { v , ( K ) } \right] .\tag{5}
$$

Here, $E _ { \mathrm { p o s } } ^ { ( i ) }$ preserves within-view spatial structure and $e _ { \mathrm { v i e w } } ^ { ( i ) }$ identifies the camera source. Concatenating the K streams retains complementary cues from multiple viewpoints.

The robot state is required for translating task-conditioned scene features into executable actions but not necessary for visual-language correspondence. We encode it separately as

$$
Z _ { n } ^ { s } = f _ { \mathrm { s t a t e } } ( s _ { n } ) \in \mathbb { R } ^ { N _ { s } \times d } ,\tag{6}
$$

where $f _ { \mathrm { s t a t e } }$ is a lightweight projection network. State features are introduced directly to the action decoder, keeping cross-modal interaction focused on task-conditioned scene understanding. These modality-specific encoders replace the high-dimensional LLM interface with compact feature sequences tailored to execution-level manipulation, reducing intermediate activation memory and downstream attention cost while preserving the information required for control.

## 4.2 VISION-LANGUAGE INTERACTION MODULE

Independently encoded visual and textual features do not yet identify which visual content is relevant to the current instruction. While LLM-centric VLAs perform this alignment within a large language backbone, TurboVLA instead uses the simple yet efficient vision-language interaction module in Fig. 3(b) to directly exchange information between the two streams.

Let $V _ { n } ^ { 0 } = Z _ { n } ^ { v }$ and $L _ { n } ^ { 0 } = Z ^ { l }$ denote the initial visual and instruction features. The interaction module progressively updates both streams through N bidirectional cross-modal layers:

$$
\left( V _ { n } ^ { \ell } , L _ { n } ^ { \ell } \right) = \mathrm { F u s i o n L a y e r } _ { \ell } \left( V _ { n } ^ { \ell - 1 } , L _ { n } ^ { \ell - 1 } \right) , \qquad \ell = 1 , \ldots , N .\tag{7}
$$

Each layer consists of layer normalization, bidirectional cross-attention, and modality-specific feedforward networks with residual connections. Visual-to-instruction attention injects scene context

into the instruction stream, while instruction-to-visual attention conditions visual features on task semantics. After the final layer, the updated streams are concatenated as

$$
Z _ { n } ^ { v l } = \bigl [ V _ { n } ^ { N } ; L _ { n } ^ { N } \bigr ] .\tag{8}
$$

Through this compact interaction module, information including target objects, attributes, and spatial relations can modulate the relevant visual features, while the instruction representation is simultaneously adapted to the current scene. This simple interaction design efficiently provides task-specific multimodal information for action prediction without relying on the broad generative and reasoning capacity of a large language model.

## 4.3 CONTINUOUS ACTION CHUNK PREDICTION

We use a ACT-style (Zhao et al., 2023) lightweight transformer decoder to map the fused multimodal representation and robot-state features to a sequence of continuous actions:

$$
\begin{array} { r } { \hat { { \bf A } } _ { n } = D _ { \theta } \left( Q _ { a } , \left[ Z _ { n } ^ { \mathrm { v l } } ; Z _ { n } ^ { s } \right] \right) \in \mathbb { R } ^ { H \times d _ { a } } , } \end{array}\tag{9}
$$

where $Q _ { a } = [ q _ { 1 } , \dots , q _ { H } ]$ contains H learnable action queries and $D _ { \theta }$ denotes the action chunk decoder. Introducing the robot state at this stage provides the current embodiment configuration while leaving the preceding interaction module focused on task-conditioned scene understanding.

All action queries are decoded in parallel, allowing the policy to predict the complete H-step action chunk in a single forward pass without action tokenization or sequential generation. We train TurboVLA through behavior cloning on expert action chunks. Given a target sequence ${ \bf A } _ { n } ^ { * } =$ $[ a _ { n , 1 } ^ { * } , \ldots , a _ { n , H } ^ { * } ]$ , the training objective is $\ell _ { 1 }$ loss, and no auxiliary language-modeling objective is required. Together, compact feature encoding, direct vision-language interaction, and parallel action decoding form the efficient $V + L \to A$ execution pathway shown in Fig. 3(a).

## 5 EXPERIMENTS

We evaluate whether TurboVLA can retain strong language-conditioned manipulation performance while substantially reducing the model scale, inference latency, and memory overhead of LLM-centric VLA policies. We first describe the implementation details and evaluation protocols for LIBERO (Liu et al., 2023), RoboTwin 2.0 (Chen et al., 2026b), and real-world deployment. We then evaluate the performance–efficiency trade-off on single-arm manipulation, examine the scalability of the proposed architecture to bimanual multi-task control, validate its effectiveness in real-world deployment, and ablate the major components of our direct vision-language interaction design.

## 5.1 IMPLEMENTATION DETAILS

We use DINOv3 (Simeoni et al.´ , 2025) as the visual backbone and BERT (Devlin et al., 2019) as the lightweight instruction encoder. Visual and textual features are projected into a shared space with $d = 2 5 6$ and processed by N = 6 bidirectional vision-language interaction layers initialized from grounding-pretrained feature-enhancement weights (Liu et al., 2024b). An ACT-style transformer decoder (Zhao et al., 2023) maps the resulting multimodal features and robot state to continuous action chunks. Across all benchmarks, we train through behavior cloning with the $\ell _ { 1 }$ loss, using a learning rate of $5 \times 1 0 ^ { - 5 }$ on four RTX 4090 GPUs. Benchmark-specific settings are described below.

## 5.2 BENCHMARKS AND METRICS

We evaluate TurboVLA across three complementary settings: single-arm manipulation on LIBERO (Liu et al., 2023), bimanual manipulation on RoboTwin 2.0 (Chen et al., 2026b), and deployment on a real robotic platform.

LIBERO contains four suites—LIBERO-Object, LIBERO-Spatial, LIBERO-Goal, and LIBERO-Long—each comprising ten language-conditioned manipulation tasks. We use the modified no noops RLDS datasets released with OpenVLA (Kim et al., 2024) and jointly train one mixedsuite model with a DINOv3 ViT-B backbone. The model predicts 12-step chunks of continuous 7-DoF actions and is trained for 80k steps with 10k warm-up steps and an effective batch size of

256. Following the VLA-Adapter rollout protocol (Wang et al., 2026b), without introducing any additional techniques, we conduct 50 rollouts per task and report suite-level and average success rates over 2,000 trials.

RoboTwin 2.0 comprises 50 language-conditioned bimanual manipulation tasks requiring coordinated dual-arm control. Given our available compute budget, we restrict training to the official clean demonstrations and do not include randomized-scene data. We jointly train one multi-task model with a DINOv3 ViT-L backbone to predict 50-step chunks of 14-dimensional absolute joint-position actions. Training lasts 55k steps with 1k warm-up steps and an effective batch size of 192. Following the StarVLA (StarVLA Community, 2026) training and evaluation framework, without introducing any additional techniques, we conduct 100 clean-setting rollouts per task and report the average success rate across all 50 tasks.

Real-world evaluation. We conducted real-world experiments using an AgileX Piper platform illustrated in Fig. 4. We consider four representative language-conditioned manipulation tasks: grab roller, move playing card away, press stapler, and stack three bowls. These tasks require accurate object grounding, viewpoint robustness, and stable closed-loop execution under real sensory noise. We initialize the policy from the TurboVLA checkpoint pretrained on LIBERO and fine-tune it on 4 × 65 teleoperated real-world demonstrations for 12.5k steps. Each task is evaluated over 40 trials, and we report the success rate. We compare against $\pi _ { 0 . 5 }$ under the same platform, training data, and evaluation protocol.

Metrics and comparisons. We use task success rate as the primary metric across all benchmarks. We also report total parameter count, inference latency, and inference VRAM. For all other runnable methods included in the comparison, these efficiency metrics are measured using official architectures, implementations, and checkpoints on an RTX 4090 with batch size one. Latency is measured from multimodal input to producing an action chunk or an equivalent number of autoregressive action tokens, while inference VRAM denotes the peak GPU memory usage of the complete online policy.

## 5.3 MAIN RESULTS

Tab. 1 and Tab. 2 present complementary evaluations of TurboVLA on simulation benchmarks. From these results, we draw the following observations.

1) Moving beyond an LLM-centered execution pathway improves the performance–efficiency frontier. As summarized in Tab. 1, TurboVLA matches the manipulation capability of large Capability-oriented VLAs at a substantially lower cost. It achieves 97.7% average success, compared with 96.9% for $\pi _ { 0 . 5 }$ (Physical Intelligence et al., 2025) while using only about 6% of its parameters and significantly reducing inference latency from 93.6 ms to 31.2 ms. Our method also outperforms the recent VLA-JEPA (Sun et al., 2026) in average success while being over 3× faster and using only about 7% of its parameters. This comparison indicates that strong execution-level manipulation is not inherently tied to using a multi-billion-parameter LLM as the central interface between perception and action. The advantage remains clear over Acceleration-oriented VLAs: OpenVLA-OFT (Kim et al., 2025) and Discrete Diffusion VLA (Liang et al., 2026c) optimize action generation and achieve inference latencies of 112.2 ms and 60.8 ms respectively, yet both remain slower and yield lower average success than our method as their large language backbones are still retained in the center of execution. Compared with Lightweight VLAs, TurboVLA further improves both sides of the trade-off. It outperforms recent Evo-1 (Lin et al., 2026b) and VLA-Adapter (Wang et al., 2026b) in average success while being substantially smaller and faster. This performance–efficiency advantage also extends to the RoboTwin 2.0 benchmark. As shown in Tab. 2, TurboVLA achieves 60.2% average success across 50 bimanual tasks with 43.4 ms inference latency, outperforming both $\pi _ { 0 . 5 }$ at 57.0% and 95.6 ms and StarVLA-α (Ye et al., 2026) at 50.3% and 74.9 ms.

These results show that neither accelerating action generation nor reducing model scale alone is sufficient. By redesigning the multimodal execution pathway, TurboVLA validates the simple and direct $V + L \to A$ paradigm as a more effective way to jointly achieve strong manipulation performance, low latency, and compact model scale across both single-arm and bimanual control settings.

2) Architectural efficiency translates into practical deployability. Practical robot deployment is jointly constrained by policy accuracy, response latency, and resident memory, rather than by any single efficiency metric. As summarized in Tab. 1, most high-performing VLA policies operate in the multi-billion-parameter regime and require several gigabytes of inference VRAM, while their inference latency is generally substantially higher than that of TurboVLA. Such resource requirements can restrict deployment to platforms with high-memory GPUs or require additional compression and system-level optimization. In contrast, the complete TurboVLA policy combines 97.7% average success with 31.2 ms action-chunk inference and only 0.9 GB of inference VRAM. This favorable efficiency profile also translates well to real robotic deployment. As shown in Fig. 4, TurboVLA achieves 92.5%, 80%, 90%, and 87.5% success on four real-world AgileX Piper tasks, consistently outperforming $\pi _ { 0 . 5 }$ . These results show that the proposed direct $V + \bar { L } $ A pathway is sufficient and effective in real-world execution-level manipulation.

Table 1: Comparison on LIBERO. “Emb. PT.” denotes additional embodied pretraining on robot data beyond LIBERO. Params denotes total parameter. Latency denotes time from multimodal input to producing an action chunk or an equivalent number of autoregressive action tokens. Both latency and Inference VRAM are measured on a single RTX 4090 with batch size one. For TurboVLA, the reported parameter count corresponds to the DINOv3 ViT-B configuration.
<table><tr><td rowspan="2">Method</td><td rowspan="2">PT.</td><td colspan="3">Emb. Deployment Efficiency</td><td colspan="5">LIBERO Success Rate (%)</td></tr><tr><td colspan="3">Params VRAM Latency (B)↓ (GB)↓</td><td colspan="5">Spa.</td></tr><tr><td>Non-VLA policy baseline</td><td></td><td></td><td></td><td>(ms)↓</td><td></td><td>Obj.</td><td></td><td></td><td>Goal Long Avg.↑</td></tr><tr><td>Diffusion Policy (Chi et al., 2023) (Rss&#x27;23)</td><td>X</td><td>0.3</td><td>1.1</td><td>924.8</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Capability-oriented VLAs OpenVLA (Kim et al., 2024) (CoRL&#x27;24)</td><td></td><td>7.5</td><td>14.9</td><td>202.9</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>π0 (Black et al., 2025a) (RSS&#x27;25)</td><td></td><td>3.2</td><td>12.3</td><td>84.2</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td></tr><tr><td>UniVLA (Bu et al., 2025) (RSs*25)</td><td></td><td>7.6</td><td>15.0</td><td>173.8</td><td>96.5</td><td>96.8</td><td>95.6</td><td>92.0</td><td>95.2</td></tr><tr><td>π0.5 (Physical Intelligence et al., 2025) (CoRL&#x27;25)</td><td></td><td>3.4</td><td>12.8</td><td>93.6</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>CogVLA (Li et al., 2025) (NeurIPS’25)</td><td></td><td>8.3</td><td>16.1</td><td>115.5</td><td>98.6</td><td>98.8</td><td>96.6</td><td>95.4</td><td>97.4</td></tr><tr><td>Mantis (Yang et al., 2026) (CVPR&#x27;26)</td><td></td><td>4.9</td><td>7.9</td><td>198.7</td><td>98.8</td><td>99.2</td><td>94.4</td><td>94.2</td><td>96.7</td></tr><tr><td>MM-ACT (Liang et al., 2026b) (CVPR&#x27;26)</td><td>X</td><td>8.2</td><td>16.3</td><td>723.2</td><td>97.8</td><td>99.4</td><td>94.8</td><td>93.0</td><td>96.3</td></tr><tr><td>VLA-JEPA (Sun et al., 2026) (ECCV’26)</td><td></td><td>2.8</td><td>5.3</td><td>108.7</td><td>96.2</td><td>99.6</td><td>97.2</td><td>95.8</td><td>97.2</td></tr><tr><td>VEGA-3D (Wu et al., 2026) (ECCV&#x27;26)</td><td></td><td>9.0</td><td>16.0</td><td>546.4</td><td>97.4</td><td>99.4</td><td>97.0</td><td>95.2</td><td>97.3</td></tr><tr><td>Acceleration-oriented VLAs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025) (Rss&#x27;25)</td><td></td><td>7.7</td><td>15.7</td><td>112.2</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td></tr><tr><td>DDVLA (Liang et al., 2026c) (1CML&#x27;26)</td><td></td><td>7.5</td><td>14.5</td><td>60.8</td><td>97.2</td><td>99.4</td><td>96.8</td><td>92.2</td><td>96.4</td></tr><tr><td>Lightweight VLAs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SmolVLA (Shukor et al., 2025) (ArXiv&#x27;25)</td><td>X</td><td>2.3</td><td>7.1</td><td>203.1</td><td>93.0</td><td>94.0</td><td>91.0</td><td>77.0</td><td>88.8</td></tr><tr><td>DreamVLA (Zhang et al., 2025c) (NeurIPS&#x27;25)</td><td>X</td><td>0.7</td><td>1.5</td><td>128.0</td><td>97.5</td><td>94.0</td><td>89.5</td><td>89.5</td><td>92.6</td></tr><tr><td>VLA-Adapter (Wang et al., 2026b) (AAAr&#x27;26)</td><td>X</td><td>1.5</td><td>4.3</td><td>87.3</td><td>97.8</td><td>99.2</td><td>97.2</td><td>95.0</td><td>97.3</td></tr><tr><td>Evo-1 (Lin et al., 2026b) (CVPR&#x27;26)</td><td>X</td><td>0.8</td><td>1.7</td><td>137.2</td><td>92.7</td><td>97.7</td><td>96.3</td><td>92.3</td><td>94.8</td></tr><tr><td>TurboVLA (Ours)</td><td>X</td><td>0.2</td><td>0.9</td><td>31.2</td><td>99.2</td><td>99.8</td><td>97.4</td><td>94.2</td><td>97.7</td></tr></table>

Table 2: Comparison on RoboTwin 2.0, with all methods trained and evaluated exclusively on the clean setting. “Emb. PT.” denotes additional embodied pretraining on robot data beyond RoboTwin 2.0, and Params denotes total parameter. Per-task methods train a separate policy for each of the 50 tasks, whereas multi-task methods jointly train a single policy across all tasks. For TurboVLA, the reported parameter count corresponds to the DINOv3 ViT-L configuration.
<table><tr><td>Method</td><td>Emb. PT.</td><td></td><td></td><td>Params (B) ↓ Lat. (ms) ↓ Avg. Success (%) ↑</td></tr><tr><td>Per-task training</td><td></td><td></td><td></td><td></td></tr><tr><td>Diffusion Policy (Chi et al., 2023) (Rss&#x27;23)</td><td>X</td><td>0.1</td><td>794.1</td><td>28.0</td></tr><tr><td>ACT (Zhao et al., 2023) (RSS&#x27;23)</td><td>x</td><td>0.1</td><td>20.4</td><td>29.7</td></tr><tr><td>DP3 (Ze et al., 2024) (RSS&#x27;24)</td><td>X</td><td>0.3</td><td>78.4</td><td>55.2</td></tr><tr><td>π0 (Black et al., 2025a) (RSs&#x27;25)</td><td>√</td><td>3.2</td><td>87.6</td><td>46.4</td></tr><tr><td>FlowPolicy (Zhang et al., 2025b) (AAAr&#x27;25)</td><td>x</td><td>0.3</td><td></td><td>41.0</td></tr><tr><td>RDT (Liu et al., 2025) (ICLR&#x27;25)</td><td>√</td><td>1.7</td><td>204.8</td><td>34.5</td></tr><tr><td>SeedPolicy (Gui et al., 2026) (ArXiv’26)</td><td>X</td><td>0.2</td><td>823.9</td><td>42.8</td></tr><tr><td>Multi-task training</td><td></td><td></td><td></td><td></td></tr><tr><td>UP-VLA (Zhang et al., 2025a) (ICML&#x27;25)</td><td></td><td>1.6</td><td>74.3</td><td>52.9</td></tr><tr><td>π0.5 (Physical Intelligence et al., 2025) (CoRL&#x27;25)</td><td>√</td><td>3.4</td><td>95.6</td><td>57.0</td></tr><tr><td>StarVLA-α (Ye et al., 2026) (ECCV&#x27;26)</td><td>X</td><td>3.8</td><td>74.9</td><td>50.3</td></tr><tr><td>TurboVLA (Ours)</td><td>X</td><td>0.4</td><td>43.4</td><td>60.2</td></tr></table>

![](images/16d72a6abdc3034d5a31a34fc080a87846c38e5027d4ff470e3116f2945cfe45.jpg)

![](images/da959ef6fe7866e337055510580ad8f111c839ef47c98ae5ce30ec324136988e.jpg)  
Figure 4: Real-world evaluation on the AgileX Piper platform. Left: our single-arm setup with a wrist-view RGB-D camera and a third-view RGB-D camera, together with the objects used in the four tasks. Right top: success-rate comparison between TurboVLA and $\pi _ { 0 . 5 }$ on four real-world manipulation tasks. Right bottom: qualitative execution examples of TurboVLA.

## 5.4 ABLATION STUDY

We conduct ablations on LIBERO to study four questions: whether semantic language conditioning is necessary, how the instruction should be encoded, which vision-language interaction design is most effective, and how sensitive the method is to the interaction depth N and action horizon H.

Semantic language conditioning and instruction encoding. We first study the role of language itself. Tab. 3 shows that removing language reduces the average success rate from 97.7% to 70.8%, with the largest drop on LIBERO-Goal $( 9 7 . 4 \%  1 1 . 6 \% )$ . This confirms that the policy cannot rely on visual priors alone when multiple behaviors are compatible with the same scene. Replacing semantic instructions with a learned task-ID embedding recovers part of the performance, but still remains 2.3% below the full model, indicating that natural-language instructions provide more than closed-set task identity. Then, we examine whether the proposed architecture depends on a specific text backbone. As shown in Tab. 4, T5-small (Raffel et al., 2020) achieves a competitive 97.1% average success rate, while the SigLIP (Zhai et al., 2023) text encoder reaches 95.5%, suggesting that execution-level instructions can be effectively handled by lightweight text encoders without a large generative language model, and that the proposed architecture is not tied to a particular text representation.

Vision-language interaction design. Having established the importance of semantic instruction features, we next study how visual and language features should interact before action decoding. As illustrated in Fig. 5, we compare no interaction, two asymmetric cross-attention variants, and the proposed bidirectional interaction, while keeping all other architecture and training settings unchanged. As shown in Tab. 5, direct concatenation achieves 95.2% average success, while the two one-way cross-attention variants improve it to 96.1% and 96.5%. Bidirectional interaction performs best at 97.7%, indicating that scene-aware instruction features and instruction-conditioned visual features provide complementary information for action prediction.

Interaction depth and action horizon. Finally, we explore two practical hyperparameters of the policy. Tab. 6 shows that increasing the number of interaction layers from N = 2 to N = 6 steadily improves the average success rate from 93.5% to 97.7%, while a deeper model with $N = 8$ slightly degrades to 96.6%. We therefore use N = 6 as a good balance between capacity and efficiency. We also vary the action horizon H while keeping the rest of the architecture unchanged. As shown in Fig. 6, performance improves from 96.4% at H = 8 to 97.7% at $H = 1 2$ , then drops to 95.6% at $H = 1 5$ . This suggests that too short a horizon limits temporal expressiveness, while too long a horizon makes chunk prediction more difficult. We therefore use $H = 1 2$ in all main experiments.

![](images/b612b5f7950b470bae300567913c34b1cca276f45d7bf8d6868a80aaea00a3a1.jpg)  
(d) Bidirectional Interaction (ours)

(a) w/o Interaction  
![](images/082987a4a6b62e311187b149a8234b275f87356cd8ac2dc217b3eab9554883ba.jpg)

Table 3: Effect of language conditioning.
<table><tr><td>Condition</td><td>Spa.</td><td>Obj.</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>w/o Language</td><td>87.0</td><td>99.4</td><td>11.6</td><td>85.0</td><td>70.8</td></tr><tr><td>Task-ID Embedding</td><td>95.6</td><td>98.6</td><td>95.8</td><td>91.6</td><td>95.4</td></tr><tr><td>Semantic Instruction</td><td>99.2</td><td>99.8</td><td>97.4</td><td>94.2</td><td>97.7</td></tr></table>

Table 4: Effect of the instruction encoder.
<table><tr><td>Text Encoder</td><td>Overall Params (M)</td><td>Spa. Obj.</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>SigLIP-Base</td><td>216.9</td><td>98.6</td><td>99.6 94.8</td><td>89.0</td><td>95.5</td></tr><tr><td>T5-Small</td><td>141.9</td><td>98.8</td><td>99.8 96.8</td><td>92.8</td><td>97.1</td></tr><tr><td>BERT</td><td>216.1</td><td>99.2 99.8</td><td>97.4</td><td>94.2</td><td>97.7</td></tr></table>

Table 5: Effect of vision-language interaction.
<table><tr><td>Interaction Design</td><td>Spa.</td><td>Obj.</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>w/o Interaction</td><td>97.4</td><td>99.8</td><td>90.8</td><td>92.8</td><td>95.2</td></tr><tr><td>Language Queries Visual</td><td>98.4</td><td>99.4</td><td>94.2</td><td>92.4</td><td>96.1</td></tr><tr><td>Visual Queries Language</td><td>98.6</td><td>100.0</td><td>94.4</td><td>93.0</td><td>96.5</td></tr><tr><td>Bidirectional Interaction</td><td>99.2</td><td>99.8</td><td>97.4</td><td>94.2</td><td>97.7</td></tr></table>

Table 6: Effect of interaction depth N.
<table><tr><td>N</td><td>Overall Params (M) Spa. Obj. Goal Long Avg.</td></tr><tr><td>2</td><td>206.6 96.6 99.6 88.4</td></tr><tr><td>4</td><td>89.4 211.3 98.0 99.4 93.2</td></tr><tr><td>6 216.1 99.2 99.8 97.4</td><td>92.2 95.7 94.2 97.7 92.8 96.6</td></tr><tr><td>8 220.8</td><td>98.2 99.6 95.8</td></tr></table>

![](images/00a4998f6201f5efcc2038f990cb716cd30248efa652a5272b5e2a16c6d04739.jpg)

![](images/90263c36eb3aad263dfb064c0b95eb695cc8fd6611132d4aaabb4e8b629e6b2b.jpg)  
(b) Language Queries Visual

![](images/efb6fbd7782ff0c915abf80bec0fde34fdf2f44a7220750aa5ae9d72e6b99c2f.jpg)  
(c) Visual Queries Language  
Figure 5: Vision-language interaction variants. (a) Directly concatenating visual and instruction features. (b) Updating only the instruction features using visual features. (c) Updating only the visual features using instruction features. (d) Bidirectional interaction jointly updates both feature streams.

Overall, these ablations show that TurboVLA achieves efficiency without discarding semantic language information or explicit cross-modal modeling. Its performance is enabled by lightweight instruction encoding together with sufficiently bidirectional vision-language interaction, supporting the proposed direct execution pathway as an effective alternative to an LLM-centered VLA architecture.

## 6 CONCLUSION

In this paper, we propose TurboVLA, a simple yet efficient ${ \dot { V + } } { \pmb { L } }  { \dot { A } }$ paradigm that moves beyond the conventional LLM-centered execution pathway for visionlanguage-action learning. By combining lightweight instruction encoding, compact visual representations, bidirectional vision-language interaction, and action-chunk decoding, TurboVLA preserves task-conditioned manipulation capability while significantly reducing model size, inference latency, and memory consumption. Our results suggest that execution-level control does not necessarily require a general-purpose LLM as the central interface between perception and action, and we hope this architecture provides a new insight for the community to further examine the role of large language models in VLA systems.

![](images/1018fba806b438504549840fcff16f9856a88bccb0a58930f15733a022287f8a.jpg)  
Figure 6: Effect of action horizon H on LIBERO.

Nevertheless, TurboVLA is designed primarily for concrete execution-level instructions and may not provide the complex semantic understanding and reasoning required for high-level task planning. Future work will explore combining the high-level planning capability of LLMs with the efficient execution pathway of TurboVLA to build hierarchical systems that are both intelligent and efficient.

## REFERENCES

Shuanghao Bai, Jing Lyu, Wanqi Zhou, Zhe Li, Dakai Wang, Lei Xing, Xiaoguang Zhao, Pengwei Wang, Zhongyuan Wang, Cheng Chi, et al. Latent reasoning vla: Latent thinking and prediction for vision-language-action models. In Proc. ofIntl. Conf. on Machine Learning, 2026.

Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang,˜ Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π : A vision-language-action flow model for general robot control. In Proc. of Robotics: Science and Systems, 2025a.

Kevin Black, Manuel Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. In Proc. of Advances in Neural Information Processing Systems, volume 38, pp. 33383– 33407, 2025b.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. In Proc. ofRobotics: Science and Systems, 2023.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. In Proc. of Robotics: Science and Systems, 2025.

Shizhe Chen, Paul Pacaud, and Cordelia Schmid. Pointact: Vision-language-action models with multi-scale point-action interaction. In Proc. ofRobotics: Science and Systems, 2026a.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In Proc. ofIntl. Conf. on Machine Learning, 2026b.

Yuxuan Chen, Yixin Han, Yize Huang, and Xiao Li. Rlrc: Reinforcement learning-based recovery for compressed vision-language-action models. IEEE Robotics and Automation Letters, 2026c.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proc. of Robotics: Science and Systems, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In Proc. of Intl. Conf. on Machine Learning, 2023.

Heng Fang, Shangru Li, Shuhan Wang, Xuanyang Xi, Dingkang Liang, and Xiang Bai. Towards generalizable robotic manipulation in dynamic environments. In Proc. of European Conference on Computer Vision, 2026.

Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proc. ofIEEE Intl. Conf. on Computer Vision, pp. 24823–24834, 2025.

Youqiang Gui, Yuxuan Zhou, Shen Cheng, Xinyang Yuan, Haoqiang Fan, Peng Cheng, and Shuaicheng Liu. Seedpolicy: Horizon scaling via self-evolving diffusion policy for robot manipulation. arXiv preprint arXiv:2603.05117, 2026.

Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Proc. of the Conference on Robot Learning, pp. 991–1002. PMLR, 2022.

Boseong Jeon, Yunho Choi, and Taehan Kim. Shallow-π: Knowledge distillation for flow-based vlas. In Proc. of the IEEE Int. Conf. on Intelligent Robots and Systems, 2026.

Titong Jiang, Xuefeng Jiang, Yuan Ma, Xin Wen, Bailin Li, Kun Zhan, Peng Jia, Yahui Liu, Sheng Sun, and Xianpeng Lang. The better you learn, the smarter you prune: Towards efficient visionlanguage-action models via differentiable token pruning. arXiv preprint arXiv:2509.12594, 2025.

Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. In Proc. of Intl. Conf. on Machine Learning, 2023.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. In Proc. of the Conference on Robot Learning, 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In Proc. ofRobotics: Science and Systems, 2025.

Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.

Wei Li, Renshan Zhang, Rui Shao, Jie He, and Liqiang Nie. Cogvla: Cognition-aligned visionlanguage-action models via instruction-driven routing & sparsification. In Proc. ofAdvances in Neural Information Processing Systems, volume 38, pp. 137646–137675, 2025.

Dingkang Liang, Cheng Zhang, Xiaopeng Xu, Jianzhong Ju, Zhenbo Luo, and Xiang Bai. Cook and clean together: Teaching embodied agents for parallel task execution. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 18415–18424, 2026a.

Haotian Liang, Xinyi Chen, Bin Wang, Mingkang Chen, Yitian Liu, Yuhao Zhang, Zanxin Chen, Tianshuo Yang, Yilun Chen, Jiangmiao Pang, et al. Mm-act: Learn from multimodal parallel generation to act. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pp. 35080–35090, 2026b.

Zhixuan Liang, Yizhuo Li, Tianshuo Yang, Chengyue Wu, Sitong Mao, Liuao Pei, Tian Nian, Shunbo Zhou, Xiaokang Yang, Jiangmiao Pang, et al. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. In Proc. ofIntl. Conf. on Machine Learning, 2026c.

Haitao Lin, Hanyang Yu, Jingshun Huang, He Zhang, Yonggen Ling, Ping Tan, Xiangyang Xue, and Yanwei Fu. Posevla: Universal pose pretraining for generalizable vision-language-action policies. In Proc. ofRobotics: Science and Systems, 2026a.

Tao Lin, Yilei Zhong, Yuxin Du, Jingjing Zhang, Jiting Liu, Yinxinyu Chen, Encheng Gu, Ziyan Liu, Hongyi Cai, Yanwen Zou, et al. Evo-1: Lightweight vision-language-action model with preserved semantic alignment. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pp. 13397–13406, 2026b.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. In Proc. ofAdvances in Neural Information Processing Systems, volume 36, pp. 44776–44791, 2023.

Chaoqi Liu, Xiaoshen Han, Jiawei Gao, Yue Zhao, Haonan Chen, and Yilun Du. Oat: Ordered action tokenization. In Proc. ofRobotics: Science and Systems, 2026.

Jiaming Liu, Mengzhen Liu, Zhenyu Wang, Pengju An, Xiaoqi Li, Kaichen Zhou, Senqiao Yang, Renrui Zhang, Yandong Guo, and Shanghang Zhang. Robomamba: Efficient vision-languageaction model for robotic reasoning and manipulation. In Proc. of Advances in Neural Information Processing Systems, volume 37, pp. 40085–40110, 2024a.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In Proc. ofEuropean Conference on Computer Vision, pp. 38–55. Springer, 2024b.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In Proc. ofIntl. Conf. on Learning Representations, 2025.

Yuxiang Lu, Zhe Liu, Xianzhe Fan, Zhenya Yang, Jinghua Hou, Junyi Li, Kaixin Ding, and Hengshuang Zhao. Faster: Rethinking real-time flow vlas. arXiv preprint arXiv:2603.19199, 2026.

Corey Lynch and Pierre Sermanet. Language conditioned imitation learning over unstructured data. In Proc. ofRobotics: Science and Systems, 2020.

Oier Mees, Lukas Hermann, and Wolfram Burgard. What matters in language conditioned robotic imitation learning over unstructured data. IEEE Robotics and Automation Letters, 7(4):11205– 11212, 2022a.

Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022b.

Jiahui Niu, Kefan Gu, Yucheng Zhao, Shengwen Liang, Tiancai Wang, Xing Hu, Ying Wang, and Huawei Li. Realtime-vla flash: Speculative inference framework for diffusion-based vlas. arXiv preprint arXiv:2605.13778, 2026.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. In Proc. ofRobotics: Science and Systems, 2024.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models. In Proc. of the IEEE Int. Conf. on Robotics and Automation, pp. 6892–6903. IEEE, 2024.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. In Proc. ofRobotics: Science and Systems, 2025.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. In Proc. ofthe Conference on Robot Learning, 2025.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In Proc. ofthe Conference on Robot Learning, pp. 894–906. PMLR, 2022.

Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Perceiver-actor: A multi-task transformer for robotic manipulation. In Proc. ofthe Conference on Robot Learning, pp. 785–799. PMLR, 2023.

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A visionlanguage-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, et al. Dinov3.¨ arXiv preprint arXiv:2508.10104, 2025.

StarVLA Community. Starvla: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

Simon Stepputtis, Joseph Campbell, Mariano Phielipp, Stefan Lee, Chitta Baral, and Heni Ben Amor. Language-conditioned imitation learning for robot manipulation tasks. In Proc. of Advances in Neural Information Processing Systems, volume 33, pp. 13139–13150, 2020.

Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. Vla-jepa: Enhancing vision-language-action model with latent world model. In Proc. ofEuropean Conference on Computer Vision, 2026.

Hanzhen Wang, Jiaming Xu, Yushun Xiang, Jiayi Pan, Yongkang Zhou, Yong-Lu Li, and Guohao Dai. Specprune-vla: Accelerating vision-language-action models via action-aware self-speculative pruning. In Proc. of Intl. Conf. on Machine Learning, 2026a.

Hongyu Wang, Chuyan Xiong, Ruiping Wang, and Xilin Chen. Bitvla: 1-bit vision-language-action models for robotics manipulation. arXiv preprint arXiv:2506.07530, 2025.

Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale visionlanguage-action model. In Proc. of the AAAI Conf. on Artificial Intelligence, pp. 18638–18646, 2026b.

Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. In Proc. of Intl. Conf. on Learning Representations, 2026c.

Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.

Xianjin Wu, Dingkang Liang, Tianrui Feng, Kui Xia, Yumeng Zhang, Xiaofan Li, Xiao Tan, and Xiang Bai. Generation models know space: Unleashing implicit 3d priors for scene understanding. In Proc. ofEuropean Conference on Computer Vision, 2026.

Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, and Chang Xu. Vla-cache: Efficient vision-language-action manipulation via adaptive token caching. In Proc. of Advances in Neural Information Processing Systems, volume 38, pp. 164448–164473, 2025.

Yuhao Xu, Yantai Yang, Zhenyang Fan, Yufan Liu, Yuming Li, Bing Li, and Zhipeng Zhang. Qvla: Not all channels are equal in vision-language-action model’s quantization. In Proc. of Intl. Conf. on Learning Representations, 2026.

Yantai Yang, Yuhao Wang, Zichen Wen, Luo Zhongwei, Chang Zou, Zhipeng Zhang, Chuan Wen, and Linfeng Zhang. Efficientvla: Training-free acceleration and compression for vision-languageaction models. In Proc. of Advances in Neural Information Processing Systems, volume 38, pp. 40891–40914, 2025.

Yi Yang, Xueqi Li, Yiyang Chen, Jin Song, Yihan Wang, Zipeng Xiao, Jiadi Su, You Qiaoben, Pengfei Liu, and Zhijie Deng. Mantis: A versatile vision-language-action model with disentangled visual foresight. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pp. 42505–42515, 2026.

Jinhui Ye, Ning Gao, Senqiao Yang, Jinliang Zheng, Zixuan Wang, Yuxin Chen, Pengguang Chen, Yilun Chen, Shu Liu, and Jiaya Jia. Starvla-alpha: Reducing complexity in vision-language-action systems. In Proc. ofEuropean Conference on Computer Vision, 2026.

Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In Proc. of Robotics: Science and Systems, 2024.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proc. of IEEE Intl. Conf. on Computer Vision, pp. 11975–11986, 2023.

Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, and Jianyu Chen. Up-vla: A unified understanding and prediction model for embodied agent. In Proc. of Intl. Conf. on Machine Learning, 2025a.

Qinglun Zhang, Zhen Liu, Haoqiang Fan, Guanghui Liu, Bing Zeng, and Shuaicheng Liu. Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation. In Proc. ofthe AAAI Conf. on Artificial Intelligence, volume 39, pp. 14754–14762, 2025b.

Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Dan Wang, Yuan Du, and Shanghang Zhang. Mole-vla: Dynamic layer-skipping vision language action model via mixture-of-layers for efficient robot manipulation. In Proc. ofthe AAAI Conf. on Artificial Intelligence, volume 40, pp. 18764–18772, 2026.

Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. In Proc. of Advances in Neural Information Processing Systems, volume 38, pp. 24195–24228, 2025c.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Proc. ofRobotics: Science and Systems, 2023.

Xin Zhou, Dingkang Liang, Sifan Tu, Xiwu Chen, Yikang Ding, Dingyuan Zhang, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation. In Proc. of IEEE Intl. Conf. on Computer Vision, pp. 27817–27827, 2025.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Proc. of the Conference on Robot Learning, pp. 2165–2183. PMLR, 2023.