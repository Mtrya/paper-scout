# WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity

Yuxue Yang <sup>1,∗,†</sup>, Shuyao Shang <sup>1,∗</sup>, Jiahe Wang <sup>1</sup>, Zitong Zhou <sup>1</sup>, Liang Tan <sup>1</sup> Junhan Zeng <sup>1</sup>, Ruizhi Li <sup>1</sup>, Junyan Li <sup>1</sup>, Yu Liu <sup>4</sup>, Xiao Yang <sup>5</sup>, Yong Li <sup>5</sup> Jun Zhu <sup>5</sup>, Hongsheng Li <sup>2,3</sup>, Tieniu Tan <sup>1</sup>, Lue Fan <sup>1,†,</sup> <sup></sup>, Zhaoxiang Zhang <sup>1,</sup> <sup></sup>

<sup>1</sup> CASIA <sup>2</sup> SLAI <sup>3</sup> CUHK <sup>4</sup> AMAP <sup>5</sup> THU

Equal Contribution <sup>†</sup> Project Leaders <sup></sup> Corresponding Authors

Controllable video generation models are increasingly being developed as world models. Accordingly, evaluating them in this role extends beyond the apparent appearance of generated videos to the inherent reactivity of the worlds they depict: the ability to infer from the scene state how the world should react and to generate plausible consequences not explicitly described in the input. Yet existing benchmarks mainly assess visual quality or explicit instruction fulfillment by checking whether requested actions and interaction outcomes are realized, leaving inherent reactivity underexamined. We introduce WorldExam, a hierarchical diagnostic benchmark spanning four levels: Visual Quality, Control Adherence, Spatial Consistency, and World Reactivity. It comprises 1,474 cases across eight dedicated tasks and supports unified evaluation of camera-, action-, and language-driven model paradigms. The World Reactivity level evaluates scene-conditioned reactions and goal-directed behaviors beyond what is explicitly specified in the input. Evaluation of 20 representative models reveals a clear capability split. Camera-driven models excel at camera control, but their interfaces do not support dynamic interaction; action-driven models control subjects more precisely but often leave the world unresponsive; and language-driven models perform better on interaction but follow complex controls less faithfully. No model combines broad task coverage with consistently strong performance, showing that high visual quality and explicit instruction fulfillment do not guarantee inherent reactivity.

Project Website: https://WorldExam.github.io

## 1 Introduction

Controllable video generation models are increasingly being developed as world models rather than standalone clip generators [3, 5–7, 12, 38, 40, 54, 56]. Such models are expected to predict future visual states from an initial observation and control instructions, including camera trajectories, action sequences, and language prompts. Evaluating them in this role extends beyond the apparent appearance of generated videos to the inherent reactivity of the worlds they depict [19, 55]. When a subject moves onto stairs, its motion should adapt to the terrain; when it approaches an obstacle, the world should show contact, avoidance, or blockage; when it enters another agent’s personal space, that agent should respond plausibly. These efects are scene-conditioned consequences rather than direct depictions of the input. Together, they reveal a model’s inherent reactivity: its ability to infer from the scene state how the world should react and to generate such consequences plausibly.

Recent benchmarks have advanced world-model evaluation beyond perceptual quality to structured layout control [10], unified action interfaces [11, 51, 52, 57, 58], prompt-specified interaction efects [49, 62], and embodied-AI and autonomous-driving applications [22, 36]. As summarized in Tab. 1, they span camera-, action-, and language-driven model paradigms and increasingly cover camera and subject control, scene revisiting, and interaction outcomes. Complementary benchmarks probe implicit rules, future-state reasoning, and law-specific physical consistency in specialized settings [24, 26, 43, 48]. Yet most benchmarks still assess explicit instruction fulfillment: a desired layout, camera trajectory, action sequence, or interaction consequence is specified in advance, and the model is evaluated on whether the specified outcome is realized. This evaluation is necessary, but it leaves underexamined a model’s ability to infer additional consequences implied by the initial state but not described in the instruction.

![](images/12e6569888d4616833687940358e86f3b3dfe5e7b772a2b8bf84763f9a6c7eab.jpg)  
Figure 1 Overview of WorldExam. WorldExam is a hierarchical diagnostic benchmark from apparent appearance to inherent reactivity. It evaluates 1,474 test cases across camera-, action-, and language-driven interfaces, four diagnostic levels, and eight tasks under a unified evaluation pipeline.

We introduce WorldExam, a hierarchical diagnostic benchmark designed around this distinction, as summarized in Fig. 1. WorldExam represents each controllable behavior as a composition of atomic control units and adapts these units to each model’s native interface: SE(3) camera trajectories for camera-driven models, discrete action sequences for action-driven models, and natural-language prompts for language-driven models. For World Reactivity cases, the model-facing instruction specifies only the explicit control or goal, leaving the expected scene-conditioned reactions unstated. This design distinguishes direct fulfillment of a requested outcome from behavior beyond what the input explicitly specifies.

WorldExam organizes evaluation into the four diagnostic levels: Visual Quality, Control Adherence, Spatial Consistency, and World Reactivity. We instantiate this hierarchy with eight evaluation tasks: Camera Control, Subject Control, Scene Revisit, Terrain Interaction, Object Interaction, Social Interaction, Physical Reaction, and Goal Completion. Visual Quality is measured with task-agnostic metrics; Control Adherence is evaluated through Camera Control and Subject Control; and Spatial Consistency is evaluated through Scene Revisit. The World Reactivity level covers scene-conditioned reactions and goal-directed behaviors. Within this level, four reaction-oriented tasks use control units as triggers while leaving the induced scene-conditioned reactions unstated. Goal Completion extends the same principle to goal-directed behavior: it specifies a high-level goal while leaving the detailed execution steps unstated. For example, a goal to arrange three bolts by height specifies the target layout, but not which object to move first or how to realize the motion frame by frame.

For model-interface compatibility, WorldExam uses two tracks rather than one global ranking. The staticscene track controls only the camera and is available to all three paradigms. The dynamic-interaction track requires observable subject–environment interaction and is therefore evaluated only on compatible actionand language-driven models. Separating the tracks avoids treating unsupported capabilities as failures or averaging scores obtained under diferent scene assumptions and task sets.

Table 1 Comparison with representative world-model benchmarks. The table compares supported model paradigms, viewpoints, task coverage, case counts, and evaluated models. C, A, and L denote camera-, action-, and language-driven model paradigms, respectively. <sup>†</sup> indicates that the instruction specifies the expected interaction consequence; the corresponding WorldExam tasks leave the evaluated reaction unstated.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Model Paradigm</td><td colspan="2">Viewpoint</td><td colspan="8">WorldExam Evaluation Tasks</td><td rowspan="2"></td><td rowspan="2">#Cases #Models</td></tr><tr><td>First Person</td><td>Third Person</td><td>Camera Control</td><td>Subject Control</td><td>Scene Revisit</td><td>Terrain Inter.</td><td>Object Inter.</td><td>Social Inter.</td><td>Physical React.</td><td>Goal Compl.</td></tr><tr><td>WorldScore [10]</td><td>C/L</td><td>V</td><td>x</td><td>L</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>3,000</td><td>20</td></tr><tr><td>MIND [57]</td><td>A</td><td></td><td></td><td></td><td>√</td><td></td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td><td>250</td><td>2</td></tr><tr><td>Omni-WorldBench [49]</td><td>C/L</td><td></td><td>V</td><td></td><td>V</td><td>V</td><td>x</td><td>√t</td><td>x</td><td>√†</td><td>x</td><td>1,068</td><td>18</td></tr><tr><td>WorldMark [52]</td><td>C/A/L</td><td></td><td></td><td>V</td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>500</td><td>6</td></tr><tr><td>iWorld-Bench [11]</td><td>C/A/L</td><td></td><td>x</td><td>V</td><td>X</td><td>√</td><td>X</td><td>x</td><td>X</td><td>x</td><td>x</td><td>4,900</td><td>14</td></tr><tr><td>WBench [58]</td><td>C/A/L</td><td></td><td>1</td><td></td><td>J</td><td></td><td>x</td><td></td><td>x</td><td></td><td>x</td><td>289</td><td>20</td></tr><tr><td>WorldOlympiad [62]</td><td>A/L</td><td></td><td>x</td><td></td><td>x</td><td>x</td><td>x</td><td>√t</td><td>x</td><td>√†</td><td>X</td><td>1,000</td><td>8</td></tr><tr><td>WorldRoamBench [51]</td><td>A</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td><td>600</td><td>10</td></tr><tr><td>WorldExam (Ours)</td><td>C/A/L</td><td>√</td><td></td><td>√</td><td>J</td><td>J</td><td>V</td><td>J</td><td>√</td><td></td><td></td><td>1,474</td><td>20</td></tr></table>

Our evaluation of 20 representative models reveals clear trade-ofs across the four levels and three paradigms. Camera-driven models excel at camera control, but their interfaces do not support dynamic interaction. Action-driven models control subjects more precisely but often leave the world unresponsive. Language-driven models perform better on interaction tasks but follow complex controls less faithfully. These capability splits are obscured by aggregate scores, motivating separate reporting at both level and task granularity.

Our contributions are summarized as follows.

• We extend world model evaluation beyond apparent appearance to inherent reactivity: inferring from the scene state how the world should react and generating plausible consequences absent from the input.

• We propose WorldExam, a benchmark of 1,474 cases across eight tasks that supports unified evaluation of camera-, action-, and language-driven model paradigms.

• We evaluate 20 representative models, revealing paradigm-dependent capability splits. No model combines broad task coverage with consistently strong performance, showing that high visual quality and explicit instruction fulfillment do not guarantee inherent reactivity.

• We will publicly release the benchmark data and evaluation toolkit to facilitate systematic evaluation and foster continued progress in the video world model community.

## 2 Related Work

## 2.1 Video World Models

Recent video world models increasingly support controllable video generation for gaming, robotics, embodied AI, and open-world simulation. Based on their primary control interfaces, they can be broadly grouped into camera-, action-, and language-driven paradigms. Camera-driven models [3, 9, 13, 39, 56, 59] condition generation on camera trajectories, represented in two main ways. Some approaches, such as ReCamMaster [3] and FantasyWorld [9], inject camera trajectories through learned camera encoders or embeddings, whereas others [13, 39, 56, 59] reconstruct 3D priors from the input and reproject them to target viewpoints; representative methods include NeoVerse [56] and InSpatio-World [39]. Action-driven models [20, 30, 38, 40, 46, 50, 65] generate future frames conditioned on discrete action sequences through keyboard-like interfaces. Among them, WorldPlay [38] and LingBot-World [40] focus on real-time interaction and consistent generation under direct action control. Language-driven models [1, 5, 12, 17, 32, 35, 44] generate videos from text or image-text prompts, demonstrating advances in semantically complex video generation. Across paradigms, video world models are evolving from short open-loop synthesis toward controllable, persistent, and interactive environment simulation. Heterogeneous interfaces complicate direct comparison, while controllability, longterm memory, and inherent reactivity remain key challenges.

## 2.2 Video World Model Benchmarks

A growing body of benchmarks evaluates complementary aspects of video world modeling. Some emphasize perceptual and temporal quality [14, 15, 27, 28, 63]; others target compositionality, world knowledge, implicit rules, and future-state reasoning [8, 26, 37, 48]. Physics-oriented benchmarks [4, 18, 24, 31, 43, 47, 53] diagnose law-specific dynamics, geometric consistency, and generalization under physical interactions; embodied benchmarks [16, 19, 21, 29, 33, 36, 55, 60] evaluate action fidelity, physical executability, planning utility, reliability, and trustworthiness; and autonomous-driving benchmarks [2, 22, 64] emphasize ego-action control, trajectory plausibility, safety, and downstream driving utility. Beyond these settings, general benchmarks [10, 11, 49, 51, 52, 57, 58, 61, 62] evaluate interactive world models across varied scenes and interfaces.

Among these general benchmarks, WorldScore [10] evaluates camera-trajectory-based layout control and geometric consistency, while MIND [57] focuses on action control and closed-loop revisit consistency. World-Mark [52] and iWorld-Bench [11] improve cross-model comparison through standardized or unified action representations. Omni-WorldBench [49] evaluates prompt-specified interaction outcomes, afected and unafected entities, and intermediate causal state transitions; WBench [58] extends evaluation to multi-turn navigation, subject actions, event editing, and perspective switching; and WorldOlympiad [62] probes long horizon interaction and physics. WorldRoamBench [51] further couples long-horizon action-conditioned generation with diagnostics of controllability, visual drift, mechanics, optics, 3D consistency, and memory. Collectively, these benchmarks substantially broaden interactive evaluation, but most still center on explicit instruction fulfillment by checking whether specified controls or interaction outcomes are realized. In contrast, WorldExam adapts atomic control units to each model’s native interface and evaluates inherent reactivity through scene-conditioned reactions and goal-directed behaviors beyond what the input explicitly specifies.

## 3 WorldExam

WorldExam supports unified evaluation of video world models with diferent control interfaces. We formulate a video world model as a function f : I × C → V, where I is the initial image, C is the model-facing input instruction and V is the generated video. We consider three common paradigms: camera-driven models take camera trajectories in SE(3), action-driven models take discrete action sequences over {W (move forward), S (move backward), A (move left), D (move right), ↑ (tilt up), ↓ (tilt down), ← (pan left), → (pan right), ∅ (stop)}, and language-driven models take natural-language prompts.

To compare these paradigms, WorldExam uses interface adaptation to map a shared case to each model’s native interface. WorldExam represents controllable behavior as an ordered composition of atomic control units, such as moving forward (W) and then panning right (→), and adapts this control intent into an SE(3) camera trajectory, a discrete action sequence, or a natural-language prompt. Under this setup, WorldExam first defines a four-level diagnostic hierarchy (Sec. 3.1) and then instantiates it through eight evaluation tasks (Sec. 3.2). We next describe the test case curation pipeline (Sec. 3.3) and the benchmark statistics (Sec. 3.4). Metric definitions and scoring protocols are described in Sec. 4.

## 3.1 Toward World Reactivity: A Four-Level Diagnostic Hierarchy

WorldExam organizes world-model evaluation into four diagnostic levels of increasing scope: Visual Quality, Control Adherence, Spatial Consistency, and World Reactivity. Visual Quality measures the video’s apparent appearance, including perceptual plausibility, temporal stability, and aesthetic quality. Control Adherence measures whether the controlled camera or subject follows the input control. Spatial Consistency measures whether the model preserves a coherent world when the camera revisits a previously observed viewpoint.

![](images/0df044f4ae6c70596aee5139c528bc06738fe76914c0a62c51209c56037b71ee.jpg)  
Figure 2 WorldExam taxonomy, tracks, and metrics. Four diagnostic levels map to eight evaluation tasks, which are assigned to static-scene or dynamic-interaction tracks according to scene assumptions and model applicability. Each track reports task-specific and general metrics. Representative examples illustrate the geometry-based evaluations.

By contrast, the World Reactivity level evaluates scene-conditioned reactions and goal-directed behaviors beyond what is explicitly specified in the input. In reaction-oriented cases, a control specifies the initiating behavior but leaves its scene-conditioned consequences unstated, which the model must infer from the scene. For example, a move-forward control specifies the subject’s direction but not how its motion should adapt to the terrain. If there is an obstacle or a nearby agent in its motion path, the subject may stop or avoid it, another agent may yield, or an object may move on contact. In goal-directed cases, a high-level goal specifies the desired target, and the model needs to infer from the initial scene how to realize it frame by frame.

Although the four levels form a diagnostic progression, strong Visual Quality, Control Adherence, and Spatial Consistency do not guarantee successful scene-conditioned reactions or goal execution. Conversely, success on World Reactivity does not compensate for visual artifacts, control errors, or spatial drift. WorldExam therefore reports the four levels separately to localize failures in generation quality, explicit control, spatial persistence, and behavior that must be inferred from the scene.

## 3.2 From Diagnostic Levels to Evaluation Tasks

Fig. 2 shows how the four diagnostic levels are instantiated. Visual Quality uses task-agnostic metrics across all videos, while the other three levels are instantiated by eight tasks. Control Adherence includes Camera Control and Subject Control, while Spatial Consistency uses Scene Revisit. World Reactivity includes four reaction-oriented tasks, Terrain Interaction, Object Interaction, Social Interaction, and Physical Reaction, plus Goal Completion for goal-directed execution.

The tasks are reported through two tracks rather than a single global score to avoid penalizing models for tasks their interfaces do not support. The static-scene track contains Camera Control and Scene Revisit and is available to all three model paradigms because each interface can express camera motion. The dynamicinteraction track contains Subject Control and the five World Reactivity tasks and applies only to compatible action- and language-driven models; Goal Completion is language-only.

Control Adherence. Camera Control tests whether the generated camera motion follows the prescribed controls. Each case composes one to three atomic camera controls from {W, S, A, D, ↑, ↓, ←, →}, assigns each an execution-time fraction, and executes them in order over the assigned intervals. Subject Control applies the same construction to a designated third-person subject using {W, S, A, D}.

Spatial Consistency. Scene Revisit tests the model’s spatial memory of the initial observation. Each case uses a round-trip camera trajectory formed by an outgoing control and its inverse, such as “move left” followed by “move right”, or “tilt up” followed by “tilt down”. After moving away, the camera should return to the initial viewpoint while the returned view preserves the scene’s geometry, appearance, and content.

![](images/28fdf987089fb6a6765dd4a8d713a438a009cd483de1e52c53d9dd9fb4ea44d2.jpg)  
Figure 3 Test case curation pipeline. For the six dynamic-interaction tasks, a task pattern is expanded into a structured draft, candidate initial images are generated and human-filtered, and the scene description, text prompt, and optional checklist are refined against the selected image before the case is finalized.

World Reactivity. The four reaction-oriented tasks pair an initial scene with a single atomic subject control. Terrain Interaction places stairs, slopes, bridges, trenches, or other structured terrain along the controlled subject’s path. The input specifies only the horizontal motion direction, while the model must infer how the subject should adapt its height and maintain contact with the terrain. Object Interaction places a movable, flexible, or rigid target along the subject’s path so that the subject, one of its body parts, or a carried tool is expected to make contact with it. It evaluates whether the target produces an immediate type-appropriate response, such as motion when loose, deformation when flexible, or blockage when rigid, without interpenetration. Social Interaction places other agents along the subject’s path or within its social distance, creating an imminent local conflict. It evaluates whether the afected agents respond plausibl through avoidance, yielding, stopping, or changing path. Physical Reaction tests whether a dynamic process unfolds over time according to physical regularities, including gravity, friction, momentum transfer, constrained motion, fluid response, and pendulum-like swinging. Each case uses one control from {W, S, A, D, ∅ (stop)}. A motion control may trigger the process, whereas ∅ is used when the initial scene is expected to evolve autonomously without subject motion. Although Object Interaction and Physical Reaction may both involve contact, the former targets the immediate type-conditioned response of a designated object, whereas the latter targets the temporal evolution of a physical process. Goal Completion is language-only and provides a high-level goal together with an initial scene containing relevant entities, distractors, preconditions, and constraints. Unlike the four reaction-oriented tasks, it uses no atomic control sequence or execution-time fractions. The input may state necessary subgoals or ordering constraints. The model should ground the goal in the initial scene, select the correct entities, ignore distractors, and produce coherent execution steps toward the desired target frame by frame. Sec. C provides examples of all eight tasks and representative checklists.

## 3.3 Test Case Curation Pipeline

WorldExam constructs cases diferently for the two tracks. For static-scene Camera Control and Scene Revisit, we pair suitable first-person scenes from existing datasets [25, 41, 57] with compositions of atomic control units. For dynamic-interaction Subject Control and the five World Reactivity tasks, the pipeline in Fig. 3 constructs initial scenes supporting the intended behavior and evaluation.

For each dynamic-interaction task, a task-specific pattern library defines the intended semantic coverage over subject motion, structured terrain, object contact, social conflict, physical processes, or goal-directed situations. After a pattern is sampled, a schema-guided LLM case composer expands it into a structured draft containing a detailed scene description, an initial-image generation prompt, a control intent or high-level goal, and a draft text prompt for language-driven models. For Object Interaction, Social Interaction, Physical Reaction, and Goal Completion, the draft also includes a case-specific checklist of observable evaluation criteria. In each World Reactivity case, the model-facing input specifies only the explicit control or high-level goal; the scene-conditioned reaction or detailed execution process remains unstated.

The initial-image generation prompt is used only to synthesize N candidate initial images. Human filtering retains candidates in which the relevant entities are visible, the spatial layout supports the intended behavior or event, the image is consistent with the draft, and suficient motion space remains for the continuation.

![](images/676ed43237a5d33505f29ce357b823897a5c74af84c381c48afac0458fce4af0.jpg)  
(b) World Reactivity composition.  
Figure 4 Benchmark statistics. The top panel summarizes distributions by viewpoint, subject type, visual style, and scene content. The bottom panel shows terrain types, social-interaction scenarios, object types, physical-reaction types, and Goal Completion domains and task types.

Candidates that already depict the evaluated event or desired target, hide relevant entities, or make the intended behavior physically infeasible are discarded. The selected initial image $I ^ { 0 }$ therefore provides a concrete pre-event state from which the intended behavior, reaction, or goal-directed execution can unfold.

An image-conditioned case refiner then revises the draft to match $I ^ { 0 }$ while preserving the sampled pattern and intended control or goal. It updates entity references, spatial relations, the scene description, the text prompt, and the optional checklist so that all referenced entities and preconditions are grounded in the selected image. For tasks evaluated using checklists, the final checklist $L = \{ \ell _ { k } \} _ { k = 1 } ^ { K }$ is fixed at this stage. The finalized case consists of $I ^ { 0 }$ , the control intent or high-level goal, the grounded text prompt, the optional checklist.

## 3.4 Benchmark Statistics

As shown in Fig. 2, WorldExam contains 1,474 cases across eight evaluation tasks. Fig. 4 summarizes both the overall dataset composition and the task-specific composition of the five World Reactivity tasks.

At the dataset level, the cases span first- and third-person viewpoints with first-person viewpoints accounting for 31.4% of the benchmark and providing substantial egocentric coverage. The subject taxonomy spans humans, animals, vehicles, and robots, while the visual-style taxonomy mixes outdoor and indoor real scenes with 3D renderings, cinematic footage, close-up views, animation, and dashcam videos. Scene content is also deliberately broad: no single scene type dominates the benchmark, and the largest category, trafic scenes,

accounts for only 14.7% of the cases.

Within the five World Reactivity tasks, the cases are further distributed across task-specific semantic subcategories, including terrain types, social-interaction scenarios, object types, physical-reaction types, and Goal Completion domains and task types. No single subcategory accounts for more than 35% of its corresponding task. This coverage reduces dependence on any one visual or semantic template and supports task-specific analysis across diverse scene-conditioned reactions and goal-directed situations. Representative cases across these dimensions are shown in Sec. A.

## 4 Evaluation Protocol and Metrics

For Camera Control, Scene Revisit, Subject Control, and Terrain Interaction, we lift each generated video into 3D with a geometry reconstruction model and evaluate it in the reconstructed space. Camera Control and Scene Revisit use the recovered camera trajectories, whereas Subject Control and Terrain Interaction use the recovered 3D subject trajectories and terrain geometry. For the remaining four tasks, we use GPT-5.5 as the vision-language model (VLM) judge to score generated videos against predefined case-specific checklists. Each track reports task-specific metrics together with task-agnostic general metrics for visual quality.

## 4.1 Static-Scene Track

Given generated frames $V = \{ I _ { t } \} _ { t = 1 } ^ { T }$ , we use VGGT-Ω [45] to estimate camera poses, intrinsics, and depths. Camera Control. Each case specifies an ordered sequence $\{ ( a _ { i } , \rho _ { i } ) \} _ { i = 1 } ^ { N } ( 1 \leq N \leq 3 )$ . Here, $a _ { i } \in \{ \mathrm { W } , \mathrm { S } , \mathrm { A } , \mathrm { D }$ $\uparrow , \downarrow , \left. , \right. \}$ is an atomic control unit, and $\rho _ { i }$ is its execution-time fraction, with $\textstyle \sum _ { i } \rho _ { i } = 1$ . For cameraand action-driven interfaces, we allocate $n _ { i } = [ \rho _ { i } T ]$ frames to the i-th control using nearest-integer rounding, and adjust the allocation to ensure $\textstyle \sum _ { i } n _ { i } = T$ . From each generated video, we recover a frame-wise camera trajectory $\{ ( \mathbf { R } _ { t } , \mathbf { t } _ { t } ) \} _ { t = 1 } ^ { T }$ . Because the three interfaces specify camera motion diferently, we construct the model-facing input and evaluation reference separately for each interface.

For camera-driven models, controls are composed sequentially starting from the initial camera pose, with each subsequent control applied relative to the endpoint pose of the previous control. These endpoints serve as keyframes, which we interpolate over the allocated frame intervals to obtain a frame-wise input trajectory in SE(3). This input trajectory also serves directly as the frame-wise reference trajectory $\{ ( \mathbf { R } _ { t } ^ { * } , \mathbf { t } _ { t } ^ { * } ) \} _ { t = 1 } ^ { T }$ . Before comparison, we express both the recovered and reference trajectories relative to their respective first-frame poses. We then compute the translation and rotation errors $( e _ { t } , e _ { r } )$ between the two trajectories as

$$
e _ { t } = \operatorname* { m i n } _ { s \geq 0 } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| s \mathbf { t } _ { t } - \mathbf { t } _ { t } ^ { * } \| _ { 2 } , \qquad e _ { r } = \frac { 1 8 0 } { \pi } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname { a r c c o s } \left( \frac { \mathrm { t r } \big ( \mathbf { R } _ { t } ( \mathbf { R } _ { t } ^ { * } ) ^ { \top } \big ) - 1 } { 2 } \right) .\tag{1}
$$

The nonnegative scale s resolves the translation-scale ambiguity of monocular camera reconstruction, making $e _ { t }$ scale-invariant, while $1 8 0 / \pi$ converts $e _ { r }$ from radians to degrees. In implementation, the argument of arccos is clipped to [−1, 1] for numerical stability.

Action-driven models map each control to the model’s native discrete action and assign the corresponding $n _ { i }$ frames to the i-th action. Language-driven models instead verbalize each control, join the resulting motion phrases in order with “then,” and prepend the instruction to the scene description; for example, “W” followed $\mathrm { b y } \ { } ^ { 6 6 } \to \ ^ { 5 }$ becomes “The camera moves forward, then pans right. [Scene description].” This prompt preserves the control order but does not specify the duration of each control.

Unlike camera-driven interfaces, action- and language-driven interfaces do not specify an exact camera trajectory in SE(3), so we evaluate their recovered trajectories against control-level references segment by segment. For action-driven models, the frame ranges assigned to the discrete actions directly define the segmen boundaries. For language-driven models, we instead partition the recovered trajectory into N segments by applying dynamic-programming-based change-point detection [42] to frame-to-frame changes in translation and rotation. The resulting segments are matched in temporal order to the N atomic controls.

Within each segment, we express the recovered camera poses relative to the first frame, so that the segment starts from the identity pose. The assigned control determines whether the reference motion is a translation or a rotation. For a translation control, we linearly interpolate the reference translation from 0 to a unit vector $\mathbf { u } _ { i }$ in the prescribed direction, while keeping $\mathbf { R } ^ { * } = \mathbf { I }$ throughout. The unit displacement is suficient because $e _ { t }$ is invariant to translation scale. For a rotation control, we set $\mathbf { t } ^ { * } = \mathbf { 0 }$ and construct $\mathbf { R } ^ { * }$ using the prescribed axis and direction; its translation error is computed as the mean per-frame $\| \mathbf { t } _ { t } \| _ { 2 }$ without scale alignment. Because the interface does not specify a rotation angle, the reference angle is linearly interpolated from zero to the total angle recovered within the segment. The segment-level references therefore evaluate the prescribed direction and motion progression without imposing a fixed magnitude.

For each segment $i ,$ we compute $( e _ { t , i } , e _ { r , i } )$ using the error definitions above. A translation segment is assigned the maximum translation error $e _ { t , i } = 0 . 5$ if its displacement is below 5% of the largest segment displacement in the same video or if its net motion is not aligned with the prescribed direction. A rotation segment is assigned the maximum rotation error $e _ { r , i } = 1 5 ^ { \circ }$ if its total rotation angle is below $5 ^ { \circ }$ or is opposite to the prescribed direction. Finally, we obtain the video-level errors $( e _ { t } , e _ { r } )$ by averaging the corresponding segment-level errors using the number of frames in each segment as weights.

Across all interfaces, we normalize the resulting errors as $s _ { t } = \operatorname* { m a x } ( 0 , 1 - e _ { t } / 0 . 5 )$ and $s _ { r } = \operatorname* { m a x } ( 0 , 1 - e _ { r } / 1 5 )$ where $e _ { r }$ is measured in degrees, and report their geometric mean, $S _ { \mathrm { c a m } } = 1 0 0 \sqrt { s _ { t } s _ { r } }$ . Thus, a high Camera Control score requires the recovered camera motion to follow the prescribed directions and temporal progression.

Scene Revisit. Scene Revisit evaluates two requirements after a round-trip camera motion: returning the camera to its initial pose and preserving the initial scene in the returned view. Each case pairs an outgoing control $a _ { 1 }$ with its inverse $a _ { 2 } = a _ { 1 } ^ { - 1 }$ . We set their execution-time fractions to $\rho _ { 1 } = 0 . 4$ and $\rho _ { 2 } = 0 . 6$ , reserving a longer temporal window for the return motion so that the model has suficient opportunity to reach the initial viewpoint. We adapt this pair to the three interfaces as in Camera Control.

Let $P _ { t } = ( \mathbf { R } _ { t } , \mathbf { t } _ { t } )$ denote the recovered camera pose. For camera- and action-driven models, $\mathcal { T } _ { \mathrm { r e t u r n } }$ is the frame range allocated to $_ { a _ { 2 } ; }$ for language-driven models, whose prompt does not specify control duration, it begins at 40% of the video. Because the camera may return before the video ends, we search this entire segment and select the frame whose recovered pose is closest to the initial pose:

$$
t _ { \mathrm { r e v } } = \underset { t \in \mathcal { T } _ { \mathrm { r e t u r n } } } { \arg \operatorname* { m i n } } d ( P _ { t } , P _ { 1 } ) .\tag{2}
$$

For translation round trips, $d ( P _ { t } , P _ { 1 } ) = \| \mathbf { t } _ { t } - \mathbf { t } _ { 1 } \| _ { 2 } ;$ for rotation round trips, d is the relative rotation angle between $\mathbf { R } _ { t }$ and $\mathbf { R } _ { 1 } . \mathrm { ~ A ~ }$ translation revisit succeeds when this minimum distance is within 10% of the maximum displacement reached during the outgoing segment; a rotation revisit succeeds when its minimum angular distance is below $5 ^ { \circ }$ . Averaging this binary result over all cases gives Revisit Success $S _ { \mathrm { s u c c } } \in [ 0 , 1 ]$

We then compare the input image with the selected revisit frame using PSNR, LPIPS, and SSIM. Selecting the frame by recovered pose rather than using the final frame makes this appearance comparison insensitive to small diferences in return timing. After averaging over cases, we normalize the three appearance metrics as $s _ { \mathrm { P } } = \mathrm { m i n } ( \mathrm { P S N R } / 2 5 , 1 ) , s _ { \mathrm { L } } = 1 - \mathrm { L P I P S }$ , and $s _ { \mathrm { S } } = \mathrm { S S I M }$ , and combine them with Revisit Success:

$$
S _ { \mathrm { r e v } } = 1 0 0 \sqrt { S _ { \mathrm { s u c c } } \cdot \frac { s _ { \mathrm { P } } + s _ { \mathrm { L } } + s _ { \mathrm { S } } } { 3 } } .\tag{3}
$$

The geometric aggregation gives a high Scene Revisit score only when the camera both returns to the initial viewpoint and recovers a consistent view.

General metrics. Across all static-scene videos, we additionally report five task-agnostic general metrics. 3D Consistency adapts the metric of WorldScore [10] to VGGT-Ω outputs. Using the recovered geometry, we project valid pixels from a source frame to a nearby target and back, then measure the cycle reprojection error. Photometric Consistency measures the forward-backward optical-flow cycle error between neighboring frames using average endpoint error (AEPE). Temporal Flickering, Aesthetic Quality, and Imaging Quality are adapted from VBench [14]. These metrics summarize whether the static-scene generations are geometrically stable, temporally coherent, and visually plausible.

## 4.2 Image-Space Displacement Alignment for Camera-Driven Models

Identical translation values in an input SE(3) trajectory can induce diferent image-space displacements across camera-driven models because their pose-conditioning interfaces interpret the translation magnitude diferently. Larger displacements expose more novel-view content and increase the dificulty of generating controlled, spatially consistent videos. By increasing the input translation multiplier to induce progressively larger image-space displacements, the results in Tab. 3 and Fig. 7 show corresponding declines in Camera Control, Scene Revisit, and general-metric performance.

![](images/468272ce02d0ad333314e21393175f0a8c81476b3df3f1a02bc43abc08b4419c.jpg)  
Figure 5 Image-space displacement alignment for camera-driven models. Given an initial image and anchor mask for case $^ { c , }$ WorldExam measures model m’s image-space displacement $d _ { m , c }$ in a default-input calibration pass and scales its input translations by $k _ { m , c } = W / ( 2 d _ { m , c } )$ to align displacement across camera-driven models.

The scale-invariant translation error in Sec. 4.1 does not address this diference. Its scalar s is fitted after generation only to resolve the coordinate-scale mismatch between the recovered and reference trajectories; it neither changes the input trajectory nor normalizes the image-space displacement in the generated video. We therefore calibrate input translations before generation to align image-space displacement (Fig. 5).

For each model m and case c, we estimate a translation calibration factor using the anchor mask provided on the initial frame. We first generate a calibration video with the model’s default input translation magnitude, apply a horizontal camera control, either “move left” or “move right”, and track the anchor through the video using SAM2 [34]. Let $d _ { m , c }$ denote the horizontal image-space displacement of the tracked mask center, and let W denote the frame width. We set the target displacement to $d ^ { * } = W / 2$ and compute $k _ { m , c } = d ^ { * } / d _ { m , c } = W / ( 2 d _ { m , c } )$ . For the final generation used in evaluation, we multiply the translation components of the default input trajectory by $k _ { m , c }$ while leaving its rotations unchanged.

## 4.3 Dynamic-Interaction Track

Subject Control. Subject Control uses the same ordered-control construction as Camera Control, with $a _ { i } \in$ {W, S, A, D} applied to a designated subject. Action-driven models receive native discrete subject actions over the allocated frame ranges. For language-driven models, we verbalize the ordered controls together with the designated subject and prepend the resulting instruction to the scene description; for example, “W” followed by “A” becomes “The [subject] moves forward, then the [subject] moves left. [Scene description].”

As illustrated in Fig. 6, SAM2 [34] tracks the designated subject from a first-frame mask, and VGGT-Ω [45] lifts the tracked pixels into 3D to recover the subject trajectory. The horizontal-region mask estimates gravity and the horizontal plane; projecting the initial camera’s viewing direction onto this plane defines the forward reference direction, with the other directions derived analogously.

We evaluate the recovered subject trajectory segment by segment. The action-driven segment boundaries follow the frame ranges assigned to the discrete subject actions, whereas the N language-driven segments are inferred by applying the same change-point procedure as in Camera Control to frame-to-frame subject displacement. Within each segment, we translate the recovered trajectory so that its first-frame subject position is the origin. The associated atomic control unit selects a horizontal unit direction ${ \bf { u } } _ { i } ,$ and the reference trajectory is linearly interpolated from the origin to $\mathbf { u } _ { i }$ . We fit a post-generation translation scale s between the recovered and reference trajectories, as in Camera Control, and compute the segment-level translation error $e _ { t , i } .$ . A segment whose net displacement is below 0.5% of the reconstructed scene scale or whose motion is misaligned with the prescribed direction receives the maximum error $e _ { t , i } = 0 . 5$ . Finally, the error $e _ { t }$ is obtained by frame-count-weighted averaging, and the Subject Control score is $S _ { \mathrm { s u b } } = 1 0 0 \operatorname* { m a x } ( 0 , 1 - e _ { t } / 0 . 5 )$

![](images/4e87caf9f6800461eecf29a7fd32a7e00af2b1c134723e088bb8fc7d81b5c0d8.jpg)  
Figure 6 Geometry-based evaluation of Subject Control and Terrain Interaction. (a) SAM2 and VGGT-Ω recover the subject trajectory, while the horizontal-region mask estimates gravity and the horizontal plane used to define the control directions. (b) A subject-free image restores the complete terrain geometry, onto which the trajectory is projected along gravity to obtain corresponding terrain trajectory.

Terrain Interaction. Unlike Subject Control, Terrain Interaction uses a single atomic control unit to induce a subject-terrain interaction. For evaluation, we additionally provide a subject-free terrain image to recover the complete terrain geometry. We then project the recovered 3D subject trajectory along gravity onto this geometry to obtain the corresponding 3D terrain trajectory.

During evaluation, we first compute the Subject Control score from the horizontal component of the subject trajectory and use it as a gating check; cases that fail this check are considered not to follow the control and receive a Terrain Interaction score of zero. For cases that pass this check, we extract local extrema and the endpoint from the height of the terrain trajectory as evaluation points. The Terrain Interaction score is the ratio of the number of evaluation points at which the subject and terrain trajectories exhibit consistent height changes to the total number of evaluation points.

Checklist-Based World Reactivity Evaluation. Considering that the remaining four tasks require semantic and causal judgments that cannot be captured by recovered trajectories, we evaluate them with a VLM judge against the case-specific checklist $L = \{ \ell _ { k } \} _ { k = \pm } ^ { K }$ constructed in Sec. 3.3. Each checklist covers the initiating condition, the resulting reaction or goal execution progress, and invalid outcomes. At evaluation time, the VLM receives 10 temporally ordered frames uniformly sampled from the generated video, together with the checklist. Using a task-specific prompt, the judge returns one binary decision per item; contradicted, missing, ambiguous, of-screen, or otherwise unverifiable evidence is counted as unsatisfied. The case score is

$$
S _ { \mathrm { c h e c k } } ( V , L ) = \frac { 1 0 0 } { K } \sum _ { k = 1 } ^ { K } \mathbb { I } [ \ell _ { k } \mathrm { ~ i s ~ s a t i s f i e d ~ i n ~ } V ] .\tag{4}
$$

Object Interaction. The checklist verifies that contact occurs with the designated object, precedes and causes the reaction, and produces a type-consistent outcome without interpenetration. It also checks that the direction and extent of the reaction remain consistent with the contact.

Social Interaction. The checklist verifies that the controlled motion creates the intended conflict and that at least one visible afected agent makes a timely adjustment attributable to the controlled subject. Unchanged, delayed, unrelated, or physically implausible responses are counted as failures.

Table 2 Static-scene track evaluation. Task averages the Camera Control and Scene Revisit scores, General averages the five general metrics, and Overall averages Task and General scores. Down ↓ and up ↑ arrows indicate that lower and higher values are better, respectively. The best and second-best results per paradigm are bold and underlined.
<table><tr><td rowspan="2">Model</td><td colspan="3">Camera Control</td><td colspan="5">Scene Revisit</td><td rowspan="2">3D Cons.</td><td rowspan="2">Photo. Cons.</td><td rowspan="2">Temp. Flick.</td><td rowspan="2">Aesth. Imag. Quality Quality</td><td colspan="3"></td></tr><tr><td>T. Err.↓</td><td>R. Err.↓</td><td>Score↑</td><td>Success↑</td><td>LPIPS↓ PSNR↑</td><td>SSIM↑</td><td>Score↑</td><td></td><td>Task</td><td>General</td><td>Overall</td></tr><tr><td>Camera-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TrajectoryCrafter [59]</td><td>0.14</td><td>1.01</td><td>80.32</td><td>1.000</td><td>19.08</td><td>0.240</td><td>0.545</td><td>83.03</td><td>96.98</td><td>62.83</td><td>92.56</td><td>52.17</td><td>67.03</td><td>81.68</td><td>74.31</td><td>78.00</td></tr><tr><td>ReCamMaster [3]</td><td>0.47</td><td>3.91</td><td>38.64</td><td>0.815</td><td>16.02</td><td>0.353</td><td>0.415</td><td>68.01</td><td>99.33</td><td>81.97</td><td>95.52</td><td>54.58</td><td>73.45</td><td>53.33</td><td>80.97</td><td>67.15</td></tr><tr><td>Voyager [13]</td><td>0.31</td><td>4.29</td><td>56.19</td><td>0.995</td><td>16.88</td><td>0.388</td><td>0.466</td><td>76.27</td><td>89.54</td><td>27.43</td><td>93.04</td><td>53.59</td><td>63.69</td><td>66.23</td><td>65.46</td><td>65.84</td></tr><tr><td>FantasyWorld [9]</td><td>1.04</td><td>7.82</td><td>18.46</td><td>0.560</td><td>15.41</td><td>0.323</td><td>0.374</td><td>55.79</td><td>98.57</td><td>75.60</td><td>95.91</td><td>57.11</td><td>73.98</td><td>37.12</td><td>80.23</td><td>58.68</td></tr><tr><td>NeoVerse [56]</td><td>0.01</td><td>0.60</td><td>97.33</td><td>1.000</td><td>21.72</td><td>0.141</td><td>0.662</td><td>89.25</td><td>98.19</td><td>70.83</td><td>93.53</td><td>52.72</td><td>72.19</td><td>93.29</td><td>77.49</td><td>85.39</td></tr><tr><td>InSpatio-World (1.3B) [39]</td><td>0.09</td><td>1.24</td><td>85.94</td><td>1.000</td><td>20.79</td><td>0.224</td><td>0.606</td><td>85.90</td><td>98.44</td><td>64.88</td><td>93.68</td><td>53.78</td><td>73.60</td><td>85.92</td><td>76.88</td><td>81.40</td></tr><tr><td>Action-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hunyuan-GameCraft [20]</td><td>0.09</td><td>11.95</td><td>41.33</td><td>0.385</td><td>13.36</td><td>0.542</td><td>0.367</td><td>41.77</td><td>93.61</td><td>53.93</td><td>93.91</td><td>54.97</td><td>71.69</td><td>41.55</td><td>73.62</td><td>57.59</td></tr><tr><td>Astra [65]</td><td>0.31</td><td>7.92</td><td>32.59</td><td>0.365</td><td>12.20</td><td>0.601</td><td>0.309</td><td>38.15</td><td>96.46</td><td>78.41</td><td>96.38</td><td>51.71</td><td>71.78</td><td>35.37</td><td>78.95</td><td>57.16</td></tr><tr><td>WorldPlay [38]</td><td>0.04</td><td>1.04</td><td>92.74</td><td>0.790</td><td>18.42</td><td>0.271</td><td>0.531</td><td>72.51</td><td>98.69</td><td>81.06</td><td>95.90</td><td>53.92</td><td>73.31</td><td>82.63</td><td>80.58</td><td>81.61</td></tr><tr><td>Yume 1.5 [30]</td><td>0.12</td><td>3.23</td><td>75.67</td><td>0.255</td><td>12.37</td><td>0.615</td><td>0.359</td><td>32.45</td><td>98.44</td><td>67.53</td><td>95.18</td><td>53.42</td><td>75.28</td><td>54.06</td><td>77.97</td><td>66.02</td></tr><tr><td>LingBot-World [40]</td><td>0.11</td><td>6.22</td><td>58.19</td><td>0.605</td><td>15.58</td><td>0.341</td><td>0.406</td><td>58.35</td><td>99.59</td><td>81.52</td><td>96.53</td><td>59.87</td><td>75.00</td><td>58.27</td><td>82.50</td><td>70.39</td></tr><tr><td>Infinite-World [50]</td><td>0.17</td><td>2.35</td><td>71.70</td><td>0.645</td><td>14.31</td><td>0.388</td><td>0.345</td><td>57.34</td><td>99.91</td><td>92.09</td><td>95.99</td><td>56.27</td><td>76.41</td><td>64.52</td><td>84.13</td><td>74.33</td></tr><tr><td>Matrix-Game 3.0 [46]</td><td>0.06</td><td>4.36</td><td>76.36</td><td>0.860</td><td>13.82</td><td>0.485</td><td>0.372</td><td>64.25</td><td>95.59</td><td>30.14</td><td>93.40</td><td>48.80</td><td>73.39</td><td>70.30</td><td>68.26</td><td>69.28</td></tr><tr><td>Language-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kling 2.5 [17]</td><td>0.19</td><td>6.90</td><td>50.18</td><td>0.325</td><td>13.54</td><td>0.532</td><td>0.381</td><td>38.81</td><td>99.87</td><td>87.87</td><td>97.56</td><td>56.28</td><td>75.70</td><td>44.50</td><td>83.46</td><td>63.98</td></tr><tr><td>Veo 3.1 [12]</td><td>0.27</td><td>8.22</td><td>40.83</td><td>0.447</td><td>12.30</td><td>0.594</td><td>0.345</td><td>43.05</td><td>99.06</td><td>73.61</td><td>95.43</td><td>55.49</td><td>76.57</td><td>41.94</td><td>80.03</td><td>60.99</td></tr><tr><td>Hailuo 2.3 [32]</td><td>0.15</td><td>5.20</td><td>63.29</td><td>0.505</td><td>13.52</td><td>0.519</td><td>0.387</td><td>48.70</td><td>99.38</td><td>79.88</td><td>95.33</td><td>56.30</td><td>75.52</td><td>55.99</td><td>81.28</td><td>68.64</td></tr><tr><td>Wan 2.6 I2V [44] Seedance 1.5 [35]</td><td>0.17 0.20</td><td>6.12 7.69</td><td>57.72 49.18</td><td>0.495 0.375</td><td>13.28 12.28</td><td>0.527</td><td>0.353</td><td>47.32</td><td>99.49</td><td>69.42</td><td>94.13 94.81</td><td>53.93</td><td>77.53</td><td>52.52</td><td>78.90</td><td>65.71</td></tr><tr><td>Vidu Q3 [5]</td><td>0.24</td><td>8.47</td><td>42.75</td><td>0.360</td><td>12.65</td><td>0.602</td><td>0.342 0.351</td><td>39.23</td><td>97.31</td><td>56.60 71.28</td><td>95.01</td><td>54.45</td><td>74.88</td><td>44.21 40.90</td><td>75.61</td><td>59.91</td></tr><tr><td>HappyHorse 1.0 [1]</td><td>0.18</td><td>5.62</td><td>58.29</td><td>0.420</td><td>12.48</td><td>0.586</td><td>0.352</td><td>39.05</td><td>99.43</td><td>74.17</td><td>95.02</td><td>55.38 55.56</td><td>76.73</td><td>50.37</td><td>79.57 80.27</td><td>60.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.564</td><td></td><td>42.45</td><td>99.62</td><td></td><td></td><td></td><td>77.00</td><td></td><td></td><td>65.32</td></tr></table>

Physical Reaction. The checklist verifies the timing and cause of the process, its evolution under the relevant physical regularity, and the preservation of required supports, attachments, contacts, and constraints. Freezing, premature onset, interpenetration, broken attachments, or unexplained energy are counted as failures.

Goal Completion. The checklist separately evaluates correct grounding, intermediate progress, compliance with stated ordering and scene-dependent constraints, and final completion. Scores credit partial progress and accept alternative executions that reach the desired target under the same observable requirements.

General metrics. The dynamic-interaction track separately reports four VBench metrics [14]: Subject Consistency and Motion Smoothness for feature and temporal consistency, and Aesthetic Quality and Imaging Quality for visual appeal and frame-level quality. We omit 3D Consistency, Photometric Consistency, and Temporal Flickering because valid motion and state changes disrupt their geometric and optical-flow correspondences.

## 5 Experiments

## 5.1 Experimental Setup

We evaluate 20 representative video world models: 6 camera-driven, 7 action-driven, and 7 language-driven models. For each case, we construct the model-facing input in the model’s native format using the interface adaptation described in Sec. 3 and evaluate the generated video with the protocols in Sec. 4. All 20 models are evaluated on the static-scene track. The dynamic-interaction track requires observable third-person subjectscene interaction and therefore includes WorldPlay [38], LingBot-World [40], and all seven language-driven models. The other five action-driven models either do not support third-person subject control or cannot control a visible third-person subject reliably, and are therefore excluded from the dynamic-interaction track. Camera-driven models are excluded because their interfaces control only the camera, and Goal Completion is evaluated only on language-driven models. To avoid compromising model performance, we use each model’s default resolution, video length, and other inference settings whenever applicable. For streaming models with flexible generation lengths, we constrain the output to 100–200 frames to prevent quality drift in excessively long generations from biasing the evaluation. Some closed-source commercial language-driven systems apply proprietary prompt enhancement before video generation. Consistent with the functional formulation in Sec. 3, we retain such default preprocessing as part of the native end-to-end pipeline. Detailed inference settings and task eligibility are provided in Sec. B; unsupported tasks are marked with dashes in the result tables.

Table 3 Effect of the input translation multiplier on NeoVerse. The multiplier is applied only to the translation components of the input SE(3) trajectory; rotations remain unchanged. Metrics and the Task, General, and Overall aggregates follow Tab. 2.
<table><tr><td rowspan="2">Translation Multiplier</td><td colspan="3">Camera Control</td><td colspan="5">Scene Revisit</td><td rowspan="2">3D Cons. Score↑</td><td rowspan="2">Photo. Cons.</td><td rowspan="2">Temp. Flick.</td><td rowspan="2">Aesth. Quality</td><td rowspan="2">Imag. Quality</td><td colspan="3">Average</td></tr><tr><td>T. Err.↓</td><td>R. Err.↓</td><td>Score↑</td><td>Success↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td></td><td>Task</td><td>General</td><td>Overall</td></tr><tr><td>0.10×</td><td>0.002</td><td>0.43</td><td>98.25</td><td>1.000</td><td>22.68</td><td>0.128</td><td>0.704</td><td>90.98</td><td>99.91</td><td>80.17</td><td>95.41</td><td>54.06</td><td>72.53</td><td>94.62</td><td>80.42</td><td>87.52</td></tr><tr><td>0.50×</td><td>0.003</td><td>0.51</td><td>97.84</td><td>1.000</td><td>22.17</td><td>0.134</td><td>0.683</td><td>90.11</td><td>99.54</td><td>76.33</td><td>94.21</td><td>53.39</td><td>72.47</td><td>93.98</td><td>79.19</td><td>86.59</td></tr><tr><td>0.75×</td><td>0.004</td><td>0.56</td><td>97.57</td><td>1.000</td><td>22.00</td><td>0.137</td><td>0.676</td><td>89.80</td><td>99.07</td><td>73.86</td><td>93.81</td><td>53.03</td><td>72.30</td><td>93.69</td><td>78.41</td><td>86.05</td></tr><tr><td>1.00×</td><td>0.005</td><td>0.60</td><td>97.33</td><td>1.000</td><td>21.72</td><td>0.141</td><td>0.662</td><td>89.25</td><td>98.19</td><td>70.83</td><td>93.53</td><td>52.72</td><td>72.19</td><td>93.29</td><td>77.49</td><td>85.39</td></tr><tr><td>2.00×</td><td>0.015</td><td>0.91</td><td>95.32</td><td>1.000</td><td>21.32</td><td>0.151</td><td>0.641</td><td>88.37</td><td>96.32</td><td>62.45</td><td>92.93</td><td>52.00</td><td>71.54</td><td>91.85</td><td>75.05</td><td>83.45</td></tr></table>

## 5.2 Static-Scene Track Results

Tab. 2 reports Camera Control and Scene Revisit together with the task-agnostic general metrics. The task scores diagnose Control Adherence and Spatial Consistency, whereas the general metrics characterize Visual Quality; reporting them separately reveals that these capabilities do not necessarily improve together.

Camera Control. Among camera-driven models, the strongest results come from methods that reconstruct 3D priors and reproject them to target views: NeoVerse [56] scores 97.33 and InSpatio-World [39] scores 85.94. ReCamMaster [3] and FantasyWorld [9], which encode camera poses as learned tokens or embeddings, obtain much lower Camera Control scores of 38.64 and 18.46 despite competitive General averages of 80.97 and 80.23. WorldPlay [38] is the strongest action-driven model at 92.74. Language-driven models are less precise, with Hailuo 2.3 [32] achieving the strongest score of 63.29, consistent with the dificulty of expressing ordered, complex viewpoint changes through natural-language instructions.

Scene Revisit. NeoVerse and InSpatio-World both achieve 1.000 Revisit Success and lead the camera-driven group with Scene Revisit scores of 89.25 and 85.90, respectively. Among action-driven models, WorldPlay performs best with 0.790 Revisit Success and a score of 72.51, while Hailuo 2.3 leads the language-driven group with 0.505 and 48.70. The remaining gap reflects failures either to return to the initial viewpoint or to recover its geometry, appearance, and content after the round trip.

## 5.3 Effect of the Input Translation Multiplier

![](images/fc03584342822322bc0d6a1cb79d5d223593c77f6ee9ed25468bc8c377951d7f.jpg)  
Figure 7 Effect of the input translation multiplier on NeoVerse.

We evaluate NeoVerse [56] under the same camera controls while varying only the translation multiplier of its input SE(3) trajectory. As the multiplier increases from 0.10× to 2.00×, the Camera Control score decreases from 98.25 to 95.32, the Scene Revisit score from 90.98 to 88.37, and the General average from 80.42 to 75.05; all five general metrics decline, with Photometric Consistency falling from 80.17 to 62.45. The results confirm that larger image-space displacements make both controlled generation and scene preservation more dificult. This ablation therefore motivates the pre-generation image-space displacement alignment in Sec. 4.2, which is applied to all reported camera-driven results.

## 5.4 Dynamic-Interaction Track Results

Tab. 4 reveals whether a model follows subject-motion control and whether it can react correctly.

Subject Control. The direct action interfaces provide more precise subject control: LingBot-World [40] scores 55.47 and WorldPlay [38] scores 49.75, compared with the best language-driven score of 37.28 from Veo 3.1 [12]. Even the action-driven results remain far from saturated, with failures often converting the requested subject motion into camera motion or leaving the scene static.

Terrain Interaction. Vidu Q3 [5] and Hailuo 2.3 [32] lead with 64.39 and 61.57, whereas the best action-driven score is 27.49. For action-driven models, the large drop from Subject Control to Terrain Interaction shows that horizontal control adherence does not guarantee vertical terrain adaptation.

Object Interaction. Veo 3.1 and Vidu Q3 lead with 75.96 and 71.59, while the best action-driven score is 33.75.   
Common failures leave the contacted object unchanged or allow the subject to pass through it.

Table 4 Dynamic-interaction track evaluation. Subject Control and five World Reactivity tasks are reported for compatible action- and language-driven models. Task averages the supported task scores, General averages the four general metrics, and Overall averages Task and General scores. Down ↓ and up ↑ arrows indicate that lower and higher values are better, respectively. Dashes mark unsupported tasks; the best and second-best results per paradigm are bold and underlined.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Subject Control</td><td rowspan="2">Terrain Inter.</td><td rowspan="2">Object Inter.</td><td rowspan="2">Social Inter.</td><td rowspan="2">Physical Reaction</td><td rowspan="2">Goal Completion</td><td rowspan="2">Subject Cons.</td><td rowspan="2">Motion Smooth.</td><td rowspan="2">Aesth. Quality</td><td rowspan="2">Imag. Quality</td><td colspan="3">Average</td></tr><tr><td>Task</td><td>General</td><td>Overall</td></tr><tr><td>Action-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WorldPlay [38]</td><td>49.75</td><td>27.49</td><td>33.75</td><td>51.40</td><td>26.91</td><td></td><td>88.06</td><td>98.09</td><td>54.19</td><td>68.76</td><td>37.86</td><td>77.28</td><td>57.57</td></tr><tr><td>LingBot-World [40]</td><td>55.47</td><td>24.33</td><td>25.94</td><td>60.37</td><td>33.43</td><td></td><td>94.98</td><td>98.89</td><td>60.86</td><td>71.69</td><td>39.91</td><td>81.61</td><td>60.76</td></tr><tr><td>Language-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kling 2.5 [17]</td><td>28.40</td><td>35.95</td><td>27.70</td><td>66.80</td><td>31.99</td><td>48.25</td><td>96.00</td><td>99.48</td><td>56.86</td><td>71.83</td><td>39.85</td><td>81.04</td><td>60.45</td></tr><tr><td>Veo 3.1 [12]</td><td>37.28</td><td>44.71</td><td>75.96</td><td>85.10</td><td>61.76</td><td>85.30</td><td>92.07</td><td>99.12</td><td>58.22</td><td>72.65</td><td>65.02</td><td>80.52</td><td>72.77</td></tr><tr><td>Hailuo 2.3 [32]</td><td>36.49</td><td>61.57</td><td>67.01</td><td>72.45</td><td>63.84</td><td>78.86</td><td>93.25</td><td>99.31</td><td>57.74</td><td>72.47</td><td>63.37</td><td>80.69</td><td>72.03</td></tr><tr><td>Wan 2.6 I2V [44]</td><td>29.02</td><td>49.21</td><td>44.56</td><td>66.40</td><td>48.15</td><td>76.39</td><td>94.46</td><td>98.15</td><td>56.30</td><td>74.72</td><td>52.29</td><td>80.91</td><td>66.60</td></tr><tr><td>Seedance 1.5 [35]</td><td>32.51</td><td>53.83</td><td>37.91</td><td>72.09</td><td>47.60</td><td>76.24</td><td>92.14</td><td>98.87</td><td>56.72</td><td>70.84</td><td>53.36</td><td>79.64</td><td>66.50</td></tr><tr><td>Vidu Q3 [5]</td><td>27.67</td><td>64.39</td><td>71.59</td><td>81.91</td><td>61.23</td><td>78.26</td><td>92.78</td><td>98.76</td><td>57.24</td><td>73.25</td><td>64.18</td><td>80.51</td><td>72.35</td></tr><tr><td>HappyHorse 1.0 [1]</td><td>33.11</td><td>56.30</td><td>65.70</td><td>76.17</td><td>47.01</td><td>85.33</td><td>92.69</td><td>98.85</td><td>57.37</td><td>73.23</td><td>60.60</td><td>80.54</td><td>70.57</td></tr></table>

Social Interaction. Veo 3.1 achieves the highest score of 85.10, followed by Vidu Q3 at 81.91; the best actiondriven score is 60.37. Failures typically leave nearby agents unresponsive or allow the controlled subject to move through them without avoidance or yielding.

Physical Reaction. Hailuo 2.3 leads with 63.84, followed by Veo 3.1 and Vidu Q3 at 61.76 and 61.23; the best action-driven score is 33.43. Action-driven generations often execute subject control while leaving unstable or contacted objects unchanged, exposing the gap between explicit control and inherent reactivity.

Goal Completion. HappyHorse 1.0 [1] and Veo 3.1 achieve the strongest results at 85.33 and 85.30. Kling 2.5 [17] scores only 48.25 despite having the highest General average among language-driven models, showing that visual quality does not guarantee grounded goal-directed execution.

## 5.5 Cross-Task Diagnostic Analysis

Fig. 8 consolidates the capability split across the three interfaces. Camera-driven models provide the strongest camera control and scene revisiting but do not support dynamic interaction. Action-driven models control designated subjects more precisely, yet this advantage does not consistently transfer to the scene-conditioned reactions induced by those controls. Language-driven models perform better on interaction and goal-directed tasks but follow composed camera and subject controls less faithfully. No model combines broad coverage with consistently strong performance, leaving current interfaces complementary but incomplete.

The split is also obscured by general visual metrics. For example, the language-driven General averages occupy a narrow range of 79.64–81.04, while their Task averages range from 39.85 to 65.02. Likewise, ReCamMaster and FantasyWorld retain strong General averages despite weak Camera Control scores. These gaps show that the four diagnostic levels capture distinct capabilities. In particular, strong Visual Quality or Control Adherence does not guarantee World Reactivity, motivating separate reporting of the four levels.

## 5.6 Human Alignment of Checklist Evaluation

Table 5 Human alignment of checklist evaluation. Spearman’s ρ and PLCC (Pearson correlation) compare human and VLM checklist-satisfaction scores per task and across all 800 instances.

<table><tr><td>Task</td><td>Checklist Items</td><td>Spearman ρ</td><td>PLCC</td></tr><tr><td>Goal Completion</td><td>1,303</td><td>0.8960</td><td>0.9017</td></tr><tr><td>Physical Reaction</td><td>1,425</td><td>0.8838</td><td>0.8750</td></tr><tr><td>Object Interaction</td><td>1,527</td><td>0.8251</td><td>0.8541</td></tr><tr><td>Social Interaction</td><td>1,538</td><td>0.7019</td><td>0.7103</td></tr><tr><td>Overall</td><td>5,793</td><td>0.8614</td><td>0.8583</td></tr></table>

We validate the VLM judge on the four tasks evaluated using checklists. The validation set contains 800 evaluation instances and 5,793 checklist items, with 200 instances per task. Three human annotators independently label each item from the same 10 temporally ordered frames shown to the VLM, and the majority vote defines the binary reference label. For each instance, the human and VLM scores are computed as the respective fractions of satisfied checklist items. Tab. 5 reports Spearman’s ρ and PLCC for each task separately and across all 800 evaluation instances combined. Across all 800 instances, Spearman’s ρ is 0.8614 and PLCC is 0.8583, showing strong agreement between the VLM judge and human evaluation.

![](images/e23878f2cd3276319c9c2cc4650509395a421d6d35dddc021b8c0e9b746903d7.jpg)  
Figure 8 Task-level performance across evaluation tracks. Two models per interface are compared across all eight tasks, ordered from static-scene diagnostics through Subject Control to World Reactivity; crosses mark unsupported tasks.

## 5.7 Backend Stability with DA3 Reconstruction

To assess sensitivity to the reconstruction backend, we rerun the geometry-based metrics with Depth Anything 3 (DA3) [23] while keeping the benchmark inputs and model outputs fixed. Tabs. 6 and 7 report the corresponding results. Scores for the four tasks evaluated using checklists remain unchanged.

On the static-scene track, the mean absolute relative change in Overall score is 3.09% across all 20 models: 0.44% for camera-driven, 3.08% for action-driven, and 5.36% for language-driven models. Most variation is concentrated in Camera Control, while Scene Revisit and the general metrics change little. NeoVerse, WorldPlay, and Hailuo 2.3 remain the leading models in their respective groups; the camera- and action-driven rankings are fully preserved, with only closely matched language-driven models exchanging positions.

On the dynamic-interaction track, DA3 afects only Subject Control, Terrain Interaction, and their geometrydependent aggregates. The mean absolute relative change in Overall score is 0.57%, the maximum change is 1.16%, and all within-paradigm rankings are preserved. Together, these small changes and stable rankings show that the main model comparisons do not depend on a particular reconstruction backend.

## 6 Conclusion

We presented WorldExam, a unified hierarchical benchmark for diagnosing video world models beyond visual quality and explicit instruction fulfillment. It distinguishes direct fulfillment of explicitly specified controls or targets from scene-conditioned reactions and detailed goal-directed execution that must be inferred from the initial scene. This distinction is instantiated through four diagnostic levels, eight tasks, and 1,474 test cases. Interface adaptation presents shared cases in the native formats of camera-, action-, and language-driven models, while the static-scene and dynamic-interaction tracks restrict evaluation to compatible interfaces rather than treating unsupported tasks as failures.

Evaluation of 20 representative models reveals a clear capability split. Camera-driven models provide the most precise camera control and scene revisiting; action-driven models control subjects more precisely but often leave terrain, objects, nearby agents, and physical processes unresponsive; and language-driven models perform better on interaction and goal-directed tasks but follow composed controls less faithfully. No evaluated model combines broad task coverage with consistently strong performance, showing that high visual quality and explicit instruction fulfillment do not guarantee inherent reactivity. Strong agreement between human and VLM checklist scores, together with stable model rankings under an alternative reconstruction backend, supports the reliability of these findings.

Table 6 Static-scene track evaluation with DA3. We recompute the static-scene metrics with DA3 while keeping benchmark inputs and model outputs fixed; aggregation and highlights follow Tab. 2.
<table><tr><td rowspan="2">Model</td><td colspan="3">Camera Control</td><td colspan="5">Scene Revisit</td><td rowspan="2">3D Cons.</td><td rowspan="2">Photo. Cons.</td><td rowspan="2">Temp. Flick.</td><td rowspan="2">Aesth. Quality</td><td rowspan="2">Imag.</td><td colspan="3">Average</td></tr><tr><td>T. Err.↓</td><td>R. Err.↓</td><td>Score↑</td><td>Success↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>Score↑</td><td>Quality Task</td><td>General</td><td>Overall</td></tr><tr><td>Camera-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TrajectoryCrafter [59]</td><td>0.13</td><td>1.10</td><td>81.97</td><td>1.000</td><td>19.06</td><td>0.240</td><td>0.544</td><td>82.99</td><td>95.63</td><td>62.83</td><td>92.56</td><td>52.18</td><td>67.03</td><td>82.48</td><td>74.05</td><td>78.27</td></tr><tr><td>ReCamMaster [3]</td><td>0.47</td><td>4.01</td><td>38.19</td><td>0.815</td><td>16.03</td><td>0.352</td><td>0.415</td><td>68.04</td><td>99.18</td><td>81.97</td><td>95.51</td><td>54.58</td><td>73.45</td><td>53.12</td><td>80.94</td><td>67.03</td></tr><tr><td>Voyager [13]</td><td>0.30</td><td>4.04</td><td>60.03</td><td>0.995</td><td>16.89</td><td>0.386</td><td>0.467</td><td>76.33</td><td>89.81</td><td>27.44</td><td>93.04</td><td>53.59</td><td>63.69</td><td>68.18</td><td>65.51</td><td>66.85</td></tr><tr><td>FantasyWorld [9]</td><td>1.05</td><td>7.76</td><td>18.24</td><td>0.555</td><td>15.39</td><td>0.323</td><td>0.373</td><td>55.51</td><td>98.79</td><td>75.60</td><td>95.90</td><td>57.11</td><td>73.98</td><td>36.87</td><td>80.28</td><td>58.58</td></tr><tr><td>NeoVerse [56]</td><td>0.01</td><td>0.72</td><td>96.93</td><td>1.000</td><td>21.70</td><td>0.141</td><td>0.661</td><td>89.22</td><td>97.53</td><td>70.83</td><td>93.53</td><td>52.72</td><td>72.19</td><td>93.07</td><td>77.36</td><td>85.22</td></tr><tr><td>InSpatio-World (1.3B) [39]</td><td>0.08</td><td>1.21</td><td>86.91</td><td>1.000</td><td>20.77</td><td>0.225</td><td>0.605</td><td>85.84</td><td>97.79</td><td>64.88</td><td>93.68</td><td>53.77</td><td>73.60</td><td>86.38</td><td>76.74</td><td>81.56</td></tr><tr><td>Action-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hunyuan-GameCraft [20]</td><td>0.13</td><td>11.92</td><td>39.17</td><td>0.375</td><td>13.37</td><td>0.542</td><td>0.369</td><td>41.26</td><td>94.00</td><td>53.93</td><td>93.91</td><td>54.98</td><td>71.70</td><td>40.21</td><td>73.70</td><td>56.96</td></tr><tr><td>Astra [65]</td><td>0.71</td><td>7.97</td><td>15.97</td><td>0.360</td><td>12.18</td><td>0.600</td><td>0.309</td><td>37.89</td><td>97.09</td><td>78.42</td><td>96.37</td><td>51.71</td><td>71.78</td><td>26.93</td><td>79.07</td><td>53.00</td></tr><tr><td>WorldPlay [38]</td><td>0.08</td><td>1.06</td><td>88.31</td><td>0.780</td><td>18.69</td><td>0.269</td><td>0.544</td><td>72.52</td><td>98.55</td><td>81.06</td><td>95.90</td><td>53.92</td><td>73.31</td><td>80.41</td><td>80.55</td><td>80.48</td></tr><tr><td>Yume 1.5 [30]</td><td>0.32</td><td>3.14</td><td>59.69</td><td>0.270</td><td>12.37</td><td>0.614</td><td>0.359</td><td>33.40</td><td>97.74</td><td>67.53</td><td>95.18</td><td>53.42</td><td>75.27</td><td>46.55</td><td>77.83</td><td>62.19</td></tr><tr><td>LingBot-World [40] Infinite-World [50]</td><td>0.19</td><td>6.28</td><td>54.43</td><td>0.620</td><td>15.63</td><td>0.340</td><td>0.409</td><td>59.17</td><td>99.42</td><td>81.52</td><td>96.53</td><td>59.87</td><td>75.01</td><td>56.80</td><td>82.47</td><td>69.64</td></tr><tr><td>Matrix-Game 3.0 [46]</td><td>0.25</td><td>2.38</td><td>64.49</td><td>0.655</td><td>14.30</td><td>0.389</td><td>0.345</td><td>57.76</td><td>99.87</td><td>92.09</td><td>95.99</td><td>56.27</td><td>76.40</td><td>61.12</td><td>84.12</td><td>72.62</td></tr><tr><td></td><td>0.17</td><td>4.27</td><td>68.93</td><td>0.860</td><td>13.83</td><td>0.486</td><td>0.372</td><td>64.23</td><td>95.61</td><td>30.14</td><td>93.39</td><td>48.80</td><td>73.39</td><td>66.58</td><td>68.27</td><td>67.43</td></tr><tr><td>Language-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kling 2.5 [17]</td><td>0.31</td><td>6.93</td><td>43.37</td><td>0.315</td><td>13.50</td><td>0.535</td><td>0.380</td><td>38.13</td><td>99.78</td><td>87.87</td><td>97.57</td><td>56.28</td><td>75.70</td><td>40.75</td><td>83.44</td><td>62.10</td></tr><tr><td>Veo 3.1 [12]</td><td>0.89</td><td>8.20</td><td>21.27</td><td>0.447</td><td>12.30</td><td>0.595</td><td>0.345</td><td>43.03</td><td>98.73</td><td>73.61</td><td>95.42</td><td>55.49</td><td>76.57</td><td>32.15</td><td>79.96</td><td>56.06</td></tr><tr><td>Hailuo 2.3 [32] Wan 2.6 I2V [44]</td><td>0.28</td><td>5.18 6.12</td><td>54.61</td><td>0.510</td><td>13.56</td><td>0.518</td><td>0.388</td><td>49.00</td><td>98.94</td><td>79.88</td><td>95.32</td><td>56.30</td><td>75.52</td><td>51.81</td><td>81.19</td><td>66.50</td></tr><tr><td></td><td>0.38 0.78</td><td>7.78</td><td>46.50 28.47</td><td>0.485</td><td>13.29</td><td>0.527</td><td>0.355</td><td>46.88</td><td>99.27</td><td>69.43</td><td>94.13 94.82</td><td>53.93</td><td>77.53</td><td>46.69</td><td>78.86</td><td>62.78</td></tr><tr><td>Seedance 1.5 [35] Vidu Q3 [5]</td><td>0.49</td><td>8.48</td><td>30.69</td><td>0.370 0.360</td><td>12.29 12.67</td><td>0.601 0.582</td><td>0.343 0.352</td><td>39.01 39.14</td><td>95.96 99.13</td><td>56.60 71.27</td><td>95.01</td><td>54.45 55.38</td><td>74.89 76.73</td><td>33.74 34.92</td><td>75.34</td><td>54.54</td></tr><tr><td>HappyHorse 1.0 [1]</td><td>0.40</td><td>5.55</td><td>45.66</td><td>0.415</td><td>12.45</td><td>0.564</td><td>0.351</td><td>42.16</td><td>99.52</td><td>74.17</td><td>95.02</td><td>55.57</td><td>76.99</td><td>43.91</td><td>79.50 80.25</td><td>57.21</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>62.08</td></tr></table>

Table 7 Dynamic-interaction track evaluation with DA3. We recompute Subject Control, Terrain Interaction, and afected aggregates with DA3; tasks evaluated using checklists remain unchanged, and aggregation and highlights follow Tab. 4.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Subject Control</td><td rowspan="2">Terrain Inter.</td><td rowspan="2">Object Inter.</td><td rowspan="2">Social Inter.</td><td rowspan="2">Physical Reaction</td><td rowspan="2">Goal Completion</td><td rowspan="2">Subject Cons.</td><td rowspan="2">Motion Smooth.</td><td rowspan="2">Aesth. Quality</td><td rowspan="2">Imag. Quality</td><td colspan="3">Average</td></tr><tr><td>Task</td><td>General</td><td>Overall</td></tr><tr><td>Action-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WorldPlay [38]</td><td>47.51</td><td>23.05</td><td>33.75</td><td>51.40</td><td>26.91</td><td></td><td>88.06</td><td>98.09</td><td>54.19</td><td>68.76</td><td>36.52</td><td>77.28</td><td>56.90</td></tr><tr><td>LingBot-World [40]</td><td>53.43</td><td>21.63</td><td>25.94</td><td>60.37</td><td>33.43</td><td></td><td>94.98</td><td>98.89</td><td>60.87</td><td>71.70</td><td>38.96</td><td>81.61</td><td>60.29</td></tr><tr><td>Language-driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kling 2.5 [17]</td><td>26.78</td><td>37.50</td><td>27.70</td><td>66.80</td><td>31.99</td><td>48.25</td><td>96.00</td><td>99.49</td><td>56.86</td><td>71.83</td><td>39.84</td><td>81.05</td><td>60.45</td></tr><tr><td>Veo 3.1 [12]</td><td>35.12</td><td>42.19</td><td>75.96</td><td>85.10</td><td>61.76</td><td>85.30</td><td>92.07</td><td>99.12</td><td>58.21</td><td>72.65</td><td>64.24</td><td>80.51</td><td>72.38</td></tr><tr><td>Hailuo 2.3 [32]</td><td>35.16</td><td>59.34</td><td>67.01</td><td>72.45</td><td>63.84</td><td>78.86</td><td>93.26</td><td>99.31</td><td>57.74</td><td>72.47</td><td>62.78</td><td>80.70</td><td>71.74</td></tr><tr><td>Wan 2.6 I2V [44]</td><td>26.95</td><td>48.55</td><td>44.56</td><td>66.40</td><td>48.15</td><td>76.39</td><td>94.46</td><td>98.15</td><td>56.30</td><td>74.71</td><td>51.83</td><td>80.91</td><td>66.37</td></tr><tr><td>Seedance 1.5 [35]</td><td>29.41</td><td>48.28</td><td>37.91</td><td>72.09</td><td>47.60</td><td>76.24</td><td>92.14</td><td>98.87</td><td>56.71</td><td>70.84</td><td>51.92</td><td>79.64</td><td>65.78</td></tr><tr><td>Vidu Q3 [5]</td><td>26.15</td><td>62.56</td><td>71.59</td><td>81.91</td><td>61.23</td><td>78.26</td><td>92.78</td><td>98.76</td><td>57.24</td><td>73.25</td><td>63.62</td><td>80.51</td><td>72.07</td></tr><tr><td>HappyHorse 1.0 [1]</td><td>31.62</td><td>53.81</td><td>65.70</td><td>76.17</td><td>47.01</td><td>85.33</td><td>92.69</td><td>98.86</td><td>57.37</td><td>73.23</td><td>59.94</td><td>80.54</td><td>70.24</td></tr></table>

The current scope is bounded by the capabilities of available model interfaces. Dynamic-interaction evaluation requires reliable third-person subject control, and Goal Completion remains limited to language-driven models. Moreover, the metrics assess observable end-to-end behavior rather than determining where reasoning occurs or establishing that the video generator itself has learned an internal causal representation. For closed-source commercial systems, proprietary prompt enhancement may contribute to scene grounding and execution planning. Future extensions to broader interfaces, longer-horizon interactions, and more intervention-based settings would provide stronger tests of persistent world understanding. Within its current scope, WorldExam identifies where controllable video models succeed, where their generated worlds remain unresponsive, and which capabilities must be developed jointly.

## Appendix

Third-Person Viewpoint  
![](images/71e94b9b4877911dda8fdcc1b041e44e7e192d1a8240d97051bbd4c395421dfc.jpg)  
Figure 9 WorldExam Gallery. Representative test cases span first- and third-person viewpoints; human, animal, vehicle, and robot subjects; diverse indoor and outdoor scene content; and a range of terrain types.

## B Per-Model Inference Settings

To preserve each model’s native performance, we use its default resolution, video length, and other inference settings whenever applicable. For streaming models with flexible generation lengths, we constrain the output to 100–200 frames to prevent quality drift in excessively long generations from biasing the evaluation. Tab. 8 reports the resulting resolution and frame count for each model.

Eligibility for the dynamic-interaction track requires reliable control of a visible third-person subject. Among action-driven models, only WorldPlay [38] and LingBot-World [40] satisfy this requirement; the other five either do not support third-person subject control or cannot provide it reliably. All evaluated language-driven models accept third-person subject-motion prompts, whereas camera-driven interfaces control only the camera. Goal Completion is language-only within the dynamic-interaction track.

Table 8 Per-model inference settings and eligibility for the dynamic-interaction track. Resolution is the center-cropped input size in width×height order, and Frames reports the evaluated video length. TPV and FPV denote third- and first-person viewpoints, respectively. ✓ denotes eligibility for the dynamic-interaction track; ✗ denotes FPV-only control or unreliable TPV subject control. <sup>†</sup> denotes unreliable TPV subject control. A dash denotes that the track is not applicable because the model controls only the camera.
<table><tr><td>Model</td><td>Backend</td><td>Resolution</td><td>Frames</td><td>Dynamic Track</td></tr><tr><td colspan="5">Camera-driven</td></tr><tr><td>TrajectoryCrafter [59]</td><td>Local</td><td>672×384</td><td>49</td><td></td></tr><tr><td>ReCamMaster [3]</td><td>Local</td><td>832×480</td><td>81</td><td></td></tr><tr><td>Voyager [13]</td><td>Local</td><td>768×512</td><td>49</td><td></td></tr><tr><td>FantasyWorld [9] NeoVerse [56]</td><td>Local</td><td>592×336 560×336</td><td>81</td><td></td></tr><tr><td>InSpatio-World (1.3B) [39]</td><td>Local Local</td><td>832×480</td><td>81 81</td><td></td></tr><tr><td colspan="5">Action-driven</td></tr><tr><td>Hunyuan-GameCraft [20] Astra [65]</td><td>Local Local</td><td>1216×704 832×480 832×480</td><td>132 161</td><td>x TPV† X FPV only</td></tr><tr><td>WorldPlay [38] Yume 1.5 [30]</td><td>Local Local</td><td>1280×704</td><td>125 145</td><td>√ TPV X FPV only</td></tr><tr><td>LingBot-World [40] Infinite-World [50]</td><td>Local Local</td><td>832×464 896×448</td><td>161 161</td><td>√ TPV</td></tr><tr><td>Matrix-Game 3.0 [46]</td><td>Local</td><td>1280×704</td><td>177</td><td>X FPV only X FPV only</td></tr><tr><td colspan="5">Language-driven</td></tr><tr><td>Kling 2.5 [17]</td><td>API</td><td>1280×720</td><td>153</td><td>✓ TPV prompt</td></tr><tr><td>Veo 3.1 [12]</td><td>API</td><td>1280×720</td><td>192</td><td>√ TPV prompt</td></tr><tr><td>Hailuo 2.3 [32]</td><td>API</td><td>1024×768</td><td>141</td><td>√ TPV prompt</td></tr><tr><td>Wan 2.6 I2V [44]</td><td>API</td><td>1280×720</td><td>150</td><td>√ TPV prompt</td></tr><tr><td>Seedance 1.5 [35]</td><td>API</td><td>1280×720</td><td>97</td><td>√ TPV prompt</td></tr><tr><td>Vidu Q3 [5]</td><td>API</td><td>1280×720</td><td>121</td><td>√ TPV prompt</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HappyHorse 1.0 [1]</td><td>API</td><td>1280×720</td><td>123</td><td>√ TPV prompt</td></tr></table>

## C Qualitative Examples of the Eight Tasks

The examples below instantiate all eight tasks in a shared visual format. The highlighted image is the initial frame, followed by four temporally ordered generated frames. The bottom panel shows the shared control intent or high-level goal. For tasks evaluated using checklists, the checklist is shown only to explain the evaluation protocol and is withheld from the input.

## C.1 Camera Control

Camera Control measures whether an ordered composition of camera motions follows the prescribed directions and temporal order without unintended drift. In this case, the camera tilts down, pans left, and moves left.

![](images/094491fae70261ced3e11757a42806507effe02848780599e4062c78afc1a2dd.jpg)  
Figure 10 Camera Control. The three atomic camera controls occupy 0.27, 0.42, and 0.31 of the video, respectively.

## C.2 Subject Control

Subject Control applies atomic controls to a designated third-person subject. Here, the subject should move forward, right, and left in order while remaining visually identifiable.

![](images/2c3dcf9f0919452a98bc69bbed50c19131ca9a960d9c7d0ea9e39d10103d7c79.jpg)  
Figure 11 Subject Control. The Forward, Right, and Left controls occupy 0.25, 0.25, and 0.50 of the video.

## C.3 Scene Revisit

Scene Revisit couples a round-trip camera motion with a spatial-memory requirement: the camera should return to the initial viewpoint while preserving the scene.

![](images/21b393204b4cc7e4026acb7193c71f81fd3cd2858a4d036e5c77bb084f9a017d.jpg)  
Figure 12 Scene Revisit. The camera first tilts down and then tilts up; success requires both execution of the motion and recovery of a consistent revisited view.

## C.4 Terrain Interaction

Terrain Interaction specifies only horizontal subject motion. The model must infer the vertical adaptation required by the visible terrain—in this case, traversing the stairs while continuing forward.

![](images/7d5aff931082dbb85938e517e43b21f018e235c1290e299dfd2ad5250d2f0524.jpg)  
Figure 13 Terrain Interaction. The Forward control is active for the full video, while the stair-climbing response is implied by the scene rather than stated in the input.

## C.5 Object Interaction

Object Interaction tests whether contact with a designated target causes a response consistent with the target’s physical type. Here, the worker pushes a bus cart forward into a lightweight sign.

<table><tr><td>Case-Specific Evaluation Checklist</td></tr><tr><td>Evaluation only; not part of the model-facing input. Does the worker keep the bus cart rolling straight forward on the same approach line into the sign without</td></tr><tr><td>steering away before impact? 0 Does the bus cart make the first contact with the sign, specifically low on the sign&#x27;s lower frame, panel edge,</td></tr><tr><td>or legs, rather than the worker&#x27;s body hitting it directly? After contact, does the A-frame sign behave like a light freestanding hinged sign by tipping, folding, sliding,</td></tr><tr><td>or getting shoved aside? 0 Does the sign remain visibly unattached and mobile, rather than acting as if fixed rigidly to the floor or</td></tr><tr><td>doorway? Does the sign&#x27;s motion follow the cart&#x27;s low forward push, with the base and legs reacting first instead of the</td></tr><tr><td>top moving in an implausible independent way? Does the worker stay behind the cart with a continued forward pushing posture through the moment of</td></tr><tr><td>impact? If the cart continues through the doorway area, is that continuation enabled by the sign being displaced or</td></tr><tr><td>collapsing aside rather than unrealistically blocking the cart like a solid barrier?</td></tr><tr><td>Does the continuation keep the cart-sign contact zone and the sign&#x27;s passive response visible enough to judge the impact clearly?</td></tr></table>

![](images/98493d5164108e06192c71b4ab334a85b965223884dd51e654ff1bb5faec77c8.jpg)  
Figure 14 Object Interaction. Only the Forward control is specified; the sign’s contact response is withheld from the model input and assessed with the checklist below.

## C.6 Social Interaction

Social Interaction evaluates whether nearby agents respond plausibly when a controlled subject enters their social or safety space. Here, a sedan enters a storefront crossing with two pedestrians and a shopping cart.

![](images/bc540fb08e32c6197248778f63ca27e222e6566d277cd1e46a71954a273b7dc2.jpg)  
Figure 15 Social Interaction. Only the car’s Forward control is specified; the checklist evaluates the pedestrians’ response.

## Social Interaction Evaluation Checklist

Evaluation only; not part of the model-facing input.

❑ Does the red sedan continue inching forward into the storefront crossing?

❑ Do the cart pusher and companion remain visible enough to judge how they pass the car nose?

❑ Does the cart path or walking pace visibly change near the sedan’s protruding front end?

❑ Is the cart interaction focused on the crossing area directly in front of the car?

❑ Does the continuation stay at ordinary parking-lot speed and behavior?

❑ Does the sedan remain the dominant source of social pressure in the scene?

❑ Do the pedestrians guiding the shopping cart respond plausibly as they move around the vehicle nose?

❑ Does the clip avoid unrelated dominant events such as another car cutting through the scene?

## C.7 Physical Reaction

Physical Reaction evaluates the temporal development of a physical process, not merely whether contact occurs. Here, the woman remains stationary while a towel-loaded laundry basket, whose center of mass overhangs the washer edge, begins to tip and fall.

![](images/06d68d00b1ecd733728c58b64f4289b6aefca7ac59660bf2bb1988de0ed2e682.jpg)  
Figure 16 Physical Reaction. The ∅ (stop) control specifies no subject motion; the basket’s rotation about the support edge and subsequent fall should unfold autonomously.

## Physical Reaction Evaluation Checklist

Evaluation only; not part of the model-facing input.

❑ Does the woman remain stationary throughout the short continuation?

❑ Does the basket continue rotating outward and downward from its already tilted position?

❑ Does the basket fall of the washing machine rather than sliding back to a fully supported position on top?

❑ Does the motion begin as a tip about the washer edge or remaining contact point, consistent with the overhanging weight pulling it over?

❑ Does the overhanging towel load move with the basket and contribute to the same outward fall direction?

❑ Does the basket fall without any new touch, grab, or bump from the woman?

❑ After leaving support, do the basket and laundry move downward into the open space in front of the machine rather than floating or reversing upward?

## C.8 Goal Completion

Goal Completion provides a high-level goal instead of an atomic control sequence. The model must ground the goal in the initial scene and generate coherent execution steps toward the desired target.

![](images/683183595f69ce8184e2ee741e407a8d903a8a2d8b3d2709071e3f9d44a8601e.jpg)

Figure 17 Goal Completion. The goal is to loosen the blue toy car’s battery-compartment screw with the screwdriver.

## Goal Completion Evaluation Checklist

Evaluation only; not part of the model-facing input.

❑ Does the person pick up and use the visible screwdriver?

❑ Does the action target the blue toy car rather than the red toy vehicle?

❑ Does the screwdriver tip align with the blue car’s battery-compartment screw?

❑ Does the continuation show a clear screw-loosening motion on the blue car?

❑ Does the red toy vehicle remain unused throughout the task?

❑ Does the person avoid using the nearby coin as a tool?

❑ Does the blue car’s battery cover end visibly loosened or ready to open?

## References

[1] Alibaba Group. HappyHorse 1.0 I2V, 2026. URL https://www.alibabacloud.com/help/en/model-studio/ happyhorse-image-to-video-api-reference.

[2] Hidehisa Arai, Keishi Ishihara, Tsubasa Takahashi, and Yu Yamaguchi. Act-bench: Towards action controllable world models for autonomous driving. arXiv preprint arXiv:2412.05337, 2024.

[3] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In ICCV, pages 14834–14844. IEEE, 2025.

[4] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In ICLR, volume 2025, pages 102075–102121, 2025.

[5] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with difusion models. arXiv preprint arXiv:2405.04233, 2024.

[6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.

[7] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first international conference on machine learning, 2024.

[8] Yubin Chen, Xuyang Guo, Zhenmei Shi, Zhao Song, and Jiahao Zhang. T2vworldbench: A benchmark for evaluating world knowledge in text-to-video generation. In WACV, pages 6474–6485. IEEE, 2026.

[9] Yixiang Dai, Fan Jiang, Chiyu Wang, Mu Xu, and Yonggang Qi. Fantasyworld: Geometry-consistent world modeling via unified video and 3d prediction. In ICLR, volume 2026, pages 103603–103622, 2026.

[10] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In ICCV, pages 27713–27724, 2025.

[11] Jianjie Fang, Yingshan Lei, Qin Wan, Ziyou Wang, Yuchao Huang, Yongyan Xu, Baining Zhao, Weichen Zhang, Chen Gao, Xinlei Chen, et al. iworld-bench: A benchmark for interactive world models with a unified action generation framework. arXiv preprint arXiv:2605.03941, 2026.

[12] Google DeepMind. Veo 3 technical report. Technical report, Google DeepMind, 2025. URL https://storage. googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf.

[13] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson Lau, Wangmeng Zuo, et al. Voyager: Long-range and world-consistent video difusion for explorable 3d scene generation. ACM TOG, 44(6):1–15, 2025.

[14] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818. IEEE, 2024.

[15] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. TPAMI, 2025.

[16] Feng Jiang, Yang Chen, Kyle Xu, Yuchen Liu, Haifeng Wang, Zhenhao Shen, Jasper Lu, Shengze Huang, Yuanfei Wang, Chen Xie, et al. Robowm-bench: A benchmark for evaluating world models in robotic manipulation. arXiv preprint arXiv:2604.19092, 2026.

[17] Kuaishou Technology. Kling AI. https://klingai.com, 2025. URL https://klingai.com.

[18] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph Gonzalez, et al. Worldmodelbench: Judging video generation models as world models. NeurIPS, 38, 2026.

[19] Huiqiong Li, Jiayu Wang, Zhiting Mei, Anirudha Majumdar, Jingjing Chen, and Bin Zhu. Robotrustbench: Benchmarking the trustworthiness of video world models for robotic manipulation. arXiv preprint arXiv:2606.01600, 2026.

[20] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.

[21] Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. Worldeval: World model as real-world robot policies evaluator. arXiv preprint arXiv:2505.19017, 2025.

[22] Ao Liang, Lingdong Kong, Tianyi Yan, Hongsi Liu, Yu Yang, Ziqi Huang, Wei Yin, Jialong Zuo, Yixuan Hu, Dekai Zhu, et al. Worldlens: Full-spectrum evaluations of driving world models in real world. In CVPR, pages 36385–36399, 2026.

[23] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

[24] Juyi Lin, Arash Akbari, Yumei He, Lin Zhao, Haichao Zhang, Arman Akbari, Xingchen Xu, Zoe Y Lu, Enfu Nan, Hokin Deng, et al. Phyground: Benchmarking physical reasoning in generative world models. arXiv preprint arXiv:2605.10806, 2026.

[25] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, pages 22160–22169. IEEE, 2024.

[26] Mingxin Liu, Shuran Ma, Shibei Meng, Xiangyu Zhao, Zicheng Zhang, Shaofeng Zhang, Zhihang Zhong, Peixian Chen, Haoyu Cao, Xing Sun, et al. Rise-video: Can video generators decode implicit world rules? arXiv preprint arXiv:2602.05986, 2026.

[27] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In CVPR, pages 22139–22149. IEEE, 2024.

[28] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. NeurIPS, 36:62352–62387, 2023.

[29] Zeyu Liu, Zhangzhe Zhu, Yang Zhang, Chenyou Fan, Chenjia Bai, and Xuelong Li. Kinebench: Benchmarking embodied world models via idm-free kinematic grounding. arXiv preprint arXiv:2607.19876, 2026.

[30] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, and Kaipeng Zhang. Yume1. 5: A text-controlled interactive world generation model. In CVPR, pages 7752–7761, 2026.

[31] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.

[32] MiniMax. Hailuo AI Video. https://hailuoai.video/, 2024. URL https://hailuoai.video/.

[33] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072, 2024.

[34] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In ICLR, volume 2025, pages 28085–28128, 2025.

[35] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[36] Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, et al. Worldarena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026.

[37] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In CVPR, pages 8406–8416. IEEE, 2025.

[38] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

[39] InSpatio Team, Donghui Shen, Guofeng Zhang, Haomin Liu, Haoyu Ji, Hujun Bao, Hongjia Zhai, Jialin Liu, Jing Guo, Nan Wang, et al. Inspatio-world: A real-time 4d world simulator via spatiotemporal autoregressive modeling. arXiv preprint arXiv:2604.07209, 2026.

[40] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[41] Radu Timofte, Eirikur Agustsson, Luc Van Gool, Ming-Hsuan Yang, Lei Zhang, Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee, et al. Ntire 2017 challenge on single image super-resolution: Methods and results. In CVPRW, pages 1110–1121. IEEE, 2017.

[42] Charles Truong, Laurent Oudre, and Nicolas Vayatis. Selective review of ofline change point detection methods. Signal processing, 167:107299, 2020.

[43] Rishi Upadhyay, Howard Zhang, Jim Solomon, Ayush Agrawal, Pranay Boreddy, Shruti Satya Narayana, Yunhao Ba, Alex Wong, Celso M de Melo, and Achuta Kadambi. Worldbench: Disambiguating physics for diagnostic evaluation of world models. arXiv preprint arXiv:2601.21282, 2026.

[44] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[45] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. Vggt-ω. arXiv preprint arXiv:2605.15195, 2026.

[46] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

[47] Jiaxin Wu, Yihao Pi, Yinling Zhang, Yuheng Li, and Xueyan Zou. Quantitative video world model evaluation for geometric-consistency. arXiv preprint arXiv:2605.15185, 2026.

[48] Keming Wu, Yijing Cui, Wenhan Xue, Qijie Wang, Xuan Luo, Zhiyuan Feng, Zuhao Yang, Sudong Wang, Sicong Jiang, Haowei Zhu, et al. Worldreasonbench: Human-aligned stress testing of video generators as future world-state predictors. arXiv preprint arXiv:2605.10434, 2026.

[49] Meiqi Wu, Zhixin Cai, Fufangchen Zhao, Xiaokun Feng, Rujing Dang, Bingze Song, Ruitian Tian, Jiashu Zhu, Jiachen Lei, Hao Dou, et al. Omni-worldbench: Towards a comprehensive interaction-centric evaluation for world models. arXiv preprint arXiv:2603.22212, 2026.

[50] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, et al. Infinite-world: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory. arXiv preprint arXiv:2602.02393, 2026.

[51] Ting-Bing Xu, Jiacheng Sui, Zhe Gao, Kewei Shi, Wenjin Yang, Zhicheng Liu, Zhaoxu Sun, Mingchao Sun, Hongyu Pan, Fan Jiang, et al. Worldroambench: An open-world benchmark for long-horizon stability of interactive world models. arXiv preprint arXiv:2606.31672, 2026.

[52] Xiaojie Xu, Zhengyuan Lin, Kang He, Yukang Feng, Xiaofeng Mao, Yuanyang Yin, Kaipeng Zhang, and Yongtao Ge. Worldmark: A unified benchmark suite for interactive video world models. arXiv preprint arXiv:2604.21686, 2026.

[53] Haotian Xue, Yipu Chen, Liqian Ma, Zelin Zhao, Lama Moukheiber, Yuchen Zhu, and Yongxin Chen. Acwmphys: Investigating generalized physical interaction in action-conditioned video world models. arXiv preprint arXiv:2605.08567, 2026.

[54] Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023.

[55] Tianzhuo Yang, Zihan Shen, Zirui Mi, Zhaoyi Zhang, Jiayi Zhou, Jiaming Ji, Juntao Dai, Jiawei Chen, Boyuan Chen, and Yaodong Yang. Mirabench: Evaluating action-conditioned reliability in robotic world models. arXiv preprint arXiv:2605.29360, 2026.

[56] Yuxue Yang, Lue Fan, Ziqi Shi, Junran Peng, Feng Wang, and Zhaoxiang Zhang. Neoverse: Enhancing 4d world model with in-the-wild monocular videos. arXiv preprint arXiv:2601.00393, 2026.

[57] Yixuan Ye, Xuanyu Lu, Yuxin Jiang, Yuchao Gu, Rui Zhao, Qiwei Liang, Jiachun Pan, Fengda Zhang, Weijia Wu, and Alex Jinpeng Wang. Mind: Benchmarking memory consistency and action control in world models. arXiv preprint arXiv:2602.08025, 2026.

[58] Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation. arXiv preprint arXiv:2605.25874, 2026.

[59] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via difusion models. In ICCV, pages 100–111. IEEE, 2025.

[60] Hu Yue, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, and Guanghui Ren. Ewmbench: Evaluating scene, motion, and semantic quality in embodied world models. arXiv preprint arXiv:2505.09694, 2025.

[61] Jiahan Zhang, Muqing Jiang, Nanru Dai, Taiming Lu, Arda Uzunoglu, Shunchi Zhang, Yana Wei, Jiahao Wang, Vishal Patel, Paul Liang, et al. World-in-world: World models in a closed-loop world. In ICLR, volume 2026, pages 55660–55699, 2026.

[62] Yuke Zhao, Wangbo Zhao, Weijie Wang, Zeyu Zhang, Dakai An, Akide Liu, Yinghao Yu, Jiasheng Tang, Fan Wang, Wei Wang, et al. Worldolympiad: Can your world model survive a triathlon? arXiv preprint arXiv:2606.11129, 2026.

[63] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

[64] Yang Zhou, Hao Shao, Letian Wang, Zhuofan Zong, Hongsheng Li, and Steven Waslander. Drivinggen: A comprehensive benchmark for generative video world models in autonomous driving. In ICLR, volume 2026, pages 103502–103524, 2026.

[65] Yixuan Zhu, Jiaqi Feng, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Jiwen Lu, and Jie Zhou. Astra: General interactive world model with autoregressive denoising. In ICLR, volume 2026, pages 79167–79184, 2026.