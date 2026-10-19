# Natura-1: The Will to Act

- 来源:Nature Will 官方博客(即技术报告),2026-10-13 发布
- URL: https://www.naturewill.com/blog/natura-1
- 引用格式:@misc{naturewill2026natura1, title={{Natura-1: The Will to Act}}, author={{Nature Will Team}}, year={2026}, howpublished={\url{https://www.naturewill.com/blog/natura-1}}}
- 说明:本文件为博客正文存档(FetchURL 抽取,图片/视频未存档;视频位于 media.naturewill.com)。发布方自述:除六个长程任务(标 †)有额外微调外,页面所有视频与结果来自同一个基础模型权重。

## Introducing Natura-1

At Nature Will, our mission is to solve general physical intelligence, building foundation models that learn directly from human experience, master general dexterous manipulation, and generalize across the physical world. That physical world was shaped by and for the human hand: we made tools to fit our hands, then used those tools to make everything else. In such a world, no machine can be truly general without being dexterous.

Yet enabling a general foundation model to control high-degree-of-freedom (DoF) dexterous hands remains a major challenge in physical intelligence. Contact dynamics leave little room for error, and the action space grows exponentially with the number of hand joints. As a result, prior work has largely focused on either general-purpose policies for simple grippers or task-specific systems for dexterous manipulation.

Today, we are introducing Natura-1, our first general foundation model for the physical world. Natura-1 is a 50-billion-parameter mixture-of-experts (MoE) model pretrained on more than one million hours of egocentric human experience. It takes in language and vision and drives 20-DoF dexterous hands within a 100-millisecond control loop. On top of this foundation, we conduct efficient post-training on a limited amount of robot data to adapt pretrained world knowledge to dexterous manipulation, yielding a model that generalizes to a wide range of unseen scenarios. Natura-1 follows complex natural-language instructions zero-shot, reasoning through spatial constraints, multi-object relationships, and negative conditions. It can also autonomously carry out long, continuous workflows, including folding delicate, deformable materials, rendering calligraphy with compliant brushstrokes, and switching tools on the fly across multi-stage tasks.

Together, these results offer early evidence that a single foundation model can learn dexterous skills from human experience and apply them to unfamiliar tasks and environments. Rather than engineering a separate solution for every task, we aim to expand these capabilities by learning from human experience at greater scale.

## Everything Has to Scale

A general model is only as general as its experience. The physical world is too varied to be covered by designed tasks or hand-built skills, so the goal is simple to state: train a very large model, using substantial computing resources, with data that spans as many environments and tasks as possible. Larger models and more compute are not enough. To generalize across the physical world, a model needs experience as varied as the world itself.

Before building anything, we therefore asked where physical experience exists at this scale. At present, robots cannot provide it. Robot data collection is bounded by robot capability: a robot can only gather data in settings where it already works, and it can only work in settings where data already exists. Humans, by contrast, already operate across virtually every environment and task, with their hands central to much of that activity. This is why Natura-1 learns from human experience, and why it is designed for dexterous hands.

Training on human data at this scale required building the full stack in-house. Our data infrastructure manages tens of petabytes in a unified lakehouse, with a throughput of hundreds of thousands of hours of egocentric video per week. It deduplicates footage at both the video and clip level across all sources, and screens for visual and sensor anomalies. It also enforces per-task quotas that keep the distribution close to real human activity. We then recover dense 3D hand poses from monocular and stereo video. Each activity is annotated hierarchically, from its overall goal down to subtasks and individual steps. While no robot hand maps exactly onto the human hand, anthropomorphic dexterous hands offer the closest correspondence available today. After filtering, the curated corpus comprises one million hours of language-conditioned action trajectories.

We train on this data with an in-house distributed engine co-designed with a 50-billion-parameter mixture-of-experts (MoE) architecture, developed to unify visual world modeling and action generation. Batches stream directly from object storage without relying on a shared file system, and custom attention kernels considerably outperform general baselines such as FlexAttention on our workloads. The stack sustains over 50% Model FLOPs Utilization (MFU) across long-running jobs on hundreds of GPUs. Following pretraining, a limited amount of robot manipulation data is sufficient to adapt the model to a physical robot hand. At inference, a Rust orchestrator layer overlaps observation streaming, computation, and action transfer, while fused CUDA kernels trim latency on the critical path. Together, they maintain a strict 100-millisecond observation-to-action loop on physical hardware. Natura-1 is among the first models trained on this stack, and each of its components has been designed to scale further.

> Every video and every result that follows comes from a single foundation model: Natura-1.†
> † Six long-horizon tasks, marked †, use additional tuning on top of Natura-1; all other results share the same weights.

## Real-World Capabilities

We evaluate Natura-1 on physical dual-arm robots with dexterous hands. We begin with a benchmark that measures dexterous manipulation along twelve capability dimensions. We then test how precisely, and for how long, a single policy can carry out multi-step tasks without human intervention. These tasks last several minutes and require both fine hand control and coordinated action from start to finish. Next we test zero-shot generalization: whether Natura-1 can follow open-ended language instructions on tasks it was never trained to perform. The last part covers something we did not plan to measure at all: actions the model came up with on its own when a task did not go as expected.

Taken together, these results show that a single foundation model can be both dexterous and general. They also give an early glimpse at physical capabilities that emerge at scale without being designed in.

### 3.1 Benchmarking dexterous manipulation

The absence of a publicly recognized dexterous-hand foundation model makes a strictly like-for-like baseline unavailable, so we select two representative baselines spanning different paradigms: π0.5, a well-established vision-language-action (VLA) foundation model, and GPT-6-Astra, a frontier large multimodal model that we use directly as a coding-based policy. On our dual-arm platform (ROKAE AR5 with Wuji hands), neither baseline produces a scored result: π0.5 fails even after being fine-tuned on our dexterous-hand robot data, and GPT-6-Astra fails to score within the 60-minute per-task time limit. We also evaluate both baselines on the ARX AC One dual-arm gripper platform, first fine-tuning π0.5 on gripper demonstrations covering exactly the same general tasks as Natura-1's dexterous-hand training set. Under this setting, π0.5 and GPT-6-Astra score 13.5 and 16.7 with grippers, respectively, whereas Natura-1 scores **53.4 (Base)** and **59.1 (Agentic)** with 20-DoF dexterous hands.

The four tasks below illustrate the principle behind the benchmark: none of them appears in Natura-1's post-training data, and neither do their scenes, objects, or action combinations. All evaluations use a single fixed training set of real-robot trajectories, with no task-specific tuning for these tasks; every result therefore reflects generalization rather than recall. Each execution is shown alongside the closest demonstrations retrieved from the training set, allowing readers to understand the benchmark's design and judge how much each task differs from the model's training experience.

示例任务 01/04:Tray loading & transport(组合泛化/工具与物体泛化/空间与场景泛化)。指令:Place the red apple in the blue bowl, set the pink bottle on the tray, and carry the loaded tray to the blue placemat. Natura-1 把物品放上中间托盘、双臂搬运且不碰旁边托盘;训练集中只有分离的动作(捡水果、搬已装好的托盘),模型泛化到抓瓶子并在不同桌面布局中组合这些动作。

### 3.2 Long-horizon execution with dexterous hands

While the benchmark evaluates capabilities separately, real-world tasks require them to work together in one continuous workflow. Everyday work involves more than picking and placing rigid objects: soft materials change shape, elastic parts spring back, and tools demand steady, carefully controlled pressure. Grippers can handle some of these tasks with specialized fixtures or attachments, but dexterous hands offer greater flexibility. The palm can support soft materials, the web between the thumb and index finger stabilizes tool handles, and the fingers can adjust their grip, hold elastic parts in place, and coordinate handoffs. As these actions change an object's shape, how it is supported, and its contact conditions, the model must continually adjust its grip and motion based on feedback rather than follow a fixed trajectory.

With task-specific tuning, Natura-1 demonstrates sustained autonomous execution across workflows that combine precise manipulation with multi-step coordination. The six demonstrations below show continuous, single-take autonomous workflows, several lasting multiple minutes without human intervention. Each requires a single policy to handle diverse materials and changing contact conditions, including pouring liquids, scooping items, handling flexible objects, using various tools, and precisely aligning parts.

六个长程工作流(标 †,有任务微调):包馄饨(支撑软面皮、填料、捏合)等,连续单镜头自主执行,多个长达数分钟。

### 3.3 Zero-shot generalization and instruction following

In this section, we set aside the structure of the benchmark and test Natura-1 zero-shot on open-ended instructions. Every trial below was recorded during our researchers' day-to-day interactions with the robot: spontaneous, unscripted prompts, many of them deliberately extreme, with no reference trajectories to guide execution. The actions, manipulation strategies, objects, and scene backgrounds all differ from anything in Natura-1's robot training data. Many of the tasks are unusual even by everyday standards: objects used and arranged in unconventional ways, instructions that exclude certain choices, and physical disturbances throughout execution.

The two preceding sections test complementary properties. The long-horizon workflows demonstrate the depth of a trained skill: how precise and how sustained execution can become after task-specific tuning. The benchmark establishes zero-shot physical generalization under controlled conditions. Real-world instruction following depends on the latter. When people give a robot instructions in their own words, they will naturally ask it to handle unfamiliar objects, arrangements, and physical challenges. A policy that cannot generalize physically to such conditions fails before instruction following even begins. The question in this section is whether that generalization holds without the guardrails of a structured benchmark, when the language is open and the task unconstrained.

In these trials, language defines the goal, and real-world feedback guides the motion. When objects move, contact conditions change, or execution is disrupted, Natura-1 adjusts its actions and continues to pursue the instructed goal rather than abandoning it. This demonstrates closed-loop instruction following beyond fixed scenes and familiar routines.

示例:同一场景三条指令对比——"左手抓牙刷盒、右手抓水瓶,同时放进垃圾桶" / "先左手…再右手…" / "先右手…再左手…"。Natura-1 纯靠语言调节双臂时序:要求顺序时严格串行,要求同时时并行双手执行。

另有一段近 30 分钟连续单镜头日常交互录像。

### 3.4 Emergent behaviors beyond predefined capabilities

This final section covers behaviors we did not set out to test. Dexterous control, instruction following, and long-horizon planning are the capabilities we deliberately built and measured. In daily interactions with Natura-1, however, we observe novel manipulation behaviors that our evaluations were not designed to capture. These behaviors emerge when the model encounters unfamiliar situations or difficulties during a task. Even without explicit guidance from robot demonstrations, it develops new approaches to continue working toward the goal.

When a grasp fails, a tool is unfamiliar, or the direct path is blocked, Natura-1 can find an alternative strategy: sliding a card over the edge of a yellow box to create a graspable overhang, using a rod to pull an object within reach, or turning its wrist mid-air to steady a toppling stack. None of these intermediate steps is specified in the prompt, and none appears in the robot demonstrations. The model devises them in response to the physical situation at hand.

We regard these behaviors as an early sign of more general physical intelligence. Learning from diverse human experience at scale appears to equip the model with an implicit understanding of physical mechanics, sufficient to invent solutions to unfamiliar problems rather than merely interpolate between previously seen trajectories.

示例:杆子拉篮——纱线篮放在超出臂展的高台上,一根长杆穿过篮柄;指令只是"抓纱线"。Natura-1 自发两步计划:抓住杆的近端向后拉,把篮子拖到工作区,一手稳杆、另一手取纱线。

## Built to Scale

### 4.1 Data

Physical intelligence requires learning from the breadth and structure of human experience. We build an end-to-end data system that continuously transforms human experience into training-ready video-language-action trajectories.

The foundation is a large-scale and diverse human experience dataset: **1 million hours of egocentric human data**, **61,702,471 episodes**, **317,842 tasks**, **14,286 scenes**. Primarily controlled data acquisition, complemented by large-scale internet ego videos. Includes monocular, stereo, multi-camera, and wrist-mounted views.

**Scaling collection.** An agentic workflow backed by a database uses semantic retrieval to map data vendor proposals to a structured task taxonomy, analyze coverage, and adjust acquisition priorities toward underrepresented areas. Internet ego videos expand coverage beyond enumerable tasks.

**Cleaning.** Unified pipeline: visual-feature dedup (duplicate videos, overlapping segments, similar clips across vendors and internet), sensor completeness/readability/continuity/temporal-consistency validation; for internet video: shot segmentation, hand visibility checks, model-based quality scoring, VLM-based evaluation.

**Motion reconstruction.** In-house motion labeling pipeline for monocular and stereo ego videos reconstructs dense hand and body motion into a unified representation. Motion labels achieve **41% reduction in MPJPE** vs existing open-source approaches on an internal benchmark with motion-capture ground truth.

**Language annotation.** Agentic annotation workflow sharing episode context; hierarchical: task goals → intermediate steps → individual actions, tracking how interactions evolve.

**Distribution steering.** Semantic embeddings + clustering + VLM analysis organize data into task/scene trees, revealing over/under-represented regions; sampling strategies and acquisition priorities refined continuously.

**Infrastructure.** Kubernetes-based parallel pipeline; Lakehouse managing **tens of PB**; processes **350,000 hours of ego data per week**, scaling linearly with compute.

### 4.2 Infrastructure

Two computational demands: high-throughput training on large-scale multimodal trajectories; real-time inference within a strict **100-millisecond control-loop budget**.

Training, inference, and GPU kernel frameworks built in-house from scratch. Inference: Rust control plane (connections, validation, preprocessing, orchestration) + GPU workers; GPU-resident state, reusable caches, captured execution graphs; separate paths for control messages and bulk payloads.

Training: lightweight modular architecture, native PyTorch flexibility + distributed/precision/compilation/eval/recovery support; data loader sustains high throughput from remote object storage; **MFU > 50% and GPU utilization 98%** in measured production configuration.

Kernel: shared GPU kernel layer across training and inference; custom attention kernels deliver ~**1,200 TFLOP/s** average across forward and backward benchmarks — **over 6× a tuned FlexAttention baseline**; **>1.7× end-to-end training speedup**, **2× inference speedup** in evaluated configurations.

End-to-end inference latency within 100 ms; serves multiple physical robots concurrently with persistent per-robot context.

### 4.3 Model

Natura-1 is a single 50B mixture-of-experts transformer trained jointly on language, images, video, and action. Instruction understanding, world modeling, and action generation are handled by a single sequence model rather than separate subsystems. Three principles: how much the model learns per unit of compute, how much it learns per trajectory, and what it learns about.

**Whole Trajectories on a Standard Transformer.** Natura-1 models an entire episode autoregressively as one sequence of language, image, video, and action tokens, so every forward and backward pass supervises every step of a trajectory, not a single sampled observation–action window. The backbone is a standard MoE transformer; modality-specific inductive bias lives only in the sequence layout and the attention pattern.

**Video and Action as One State.** At any moment, what the camera sees and what the hand does are two projections of the same physical state, so learning to predict video is also learning to act. Video is the abundant modality: action labels are scarce and costly, while video dominates the tokens of every trajectory. Natura-1 uses past video as memory and learns to predict future video as part of its world model. It trains video and action together using flow matching and joint denoising, with a separate loss formulation for each modality. A trajectory yields far more supervision than its action labels alone — the basis for the robustness, memory, and recovery visible in the sustained workflows above.

**Prediction Is Not Understanding.** A model can fit the joint distribution of video and action and still ignore the instruction: in most clips the scene alone predicts what happens next, so language is the easiest signal to discard. Natura-1's pretraining makes language load-bearing: using the fine-grained captions from the data pipeline, it is shaped so that a difference in stated intent yields a different predicted future, in video and action alike. In parallel, the model draws more fully on visual priors to read the scene an instruction refers to. Together, these approaches ground zero-shot instruction following in understanding rather than imitation.

**Scaling curves.** As training advances from ~44,000 to 1,000,000 hours of experience: held-out human data Action MSE −47.7%, Action MAE −37.3%; same weights evaluated on held-out real-robot validation: MSE −67.7%, MAE −49.3%, without further training for this evaluation. Both evaluations use teacher forcing, conditioning each prediction on past observations.

## The Work Ahead

Natura-1 provides early evidence that generalization can coexist with the complexity of dexterous hands. Further progress depends on the full stack: more capable and stable effector and sensing systems, pretraining data at greater scale and coverage, and models that transfer to robots with less adaptation.
