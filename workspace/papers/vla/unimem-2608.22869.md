# 2608.22869 (from arXiv HTML; MinerU fallback)



# UniMem: Unifying Multimodal Memory and Control for Vision-Language-Action Models

Lars Osterberg1, Maggie Wang1, and Mac Schwager1

Affiliation: Stanford University, Stanford, CA, USA

###### Abstract

While Vision-Language-Action (VLA) models have leveraged internet-scale pretraining and task-focused finetuning to achieve strong performance on long-horizon tasks, they often struggle with non-Markovian tasks that require memory. Existing approaches to memory typically involve additional Vision-Language-Models (VLMs) for long-term memory management, introducing a memory bottleneck and a fractured training pipeline. Conditioning on multiple historical frames can provide the VLA with access to more descriptive features of past scenes, but can degrade performance if frames are chosen at arbitrary, fixed intervals. To address these limitations, we present UniMem, a framework that unifies high-level, multimodal memory and low-level control under one backbone. UniMem employs an event classifier for memory updates, a keyframe encoder for dense spatial memory, and a keyframe caching technique to minimize overhead during policy rollouts. We evaluate UniMem across five simulation and four hardware tasks targeting sequential and spatial memory, demonstrating that our unified, single-model system outperforms fixed-interval image sampling baselines (93.4% vs. 68.2%) in simulation and hierarchical baselines (80.0% vs. 43.5%) in hardware, while offering faster inference and a simple training pipeline for easy adoption.
Project website: [https://losterberg3.github.io/unimem-vla/](https://losterberg3.github.io/unimem-vla/)

## I Introduction

While vision-language-action (VLA) models [7, 26, 6, 4, 15, 25, 16] have achieved remarkable progress in robot manipulation, their success is largely confined to short-duration, Markovian tasks due to an absence of architectural memory. For instance, state-of-the-art policies [27] typically struggle on rudimentary tasks such as picking up an object and returning it to its starting position [10, 20, 30]. We characterize this failure mode as perceptual aliasing—when several moments in a training dataset share similar observations but demand entirely different actions depending on preceding events. At these functionally distinct but apparently similar phases, a VLA requires not only sequential memory to orient itself within the progress of a task, but spatial memory to retain pertinent information about the scene’s past. Motivated by human cognition and recent breakthroughs in vision-language models showing that temporal history is crucial for context-aware reasoning [9], we pose the central question: Can conditioning VLAs on multimodal memory directly unlock robust performance in non-Markovian tasks?

![Refer to caption](drafts/images/unimem-2608.22869/model_final.png)

*Fig. 1: Overview of UniMem. An event classifier ($f_{\phi}$) detects sparse sub-task transition events from the backbone latent space, autoregressively updating textual memory and a cache of precomputed keyframe hidden states. This routes to our keyframe encoder and a tokenizer, providing event-driven, multimodal memory directly to the backbone and action expert while unifying memory and control in one self-sustaining, low-latency VLA.*

Existing methods [30, 31, 33, 8] incorporate memory in VLA systems via an additional Vision-Language Model (VLM) that maintains textual summaries of completed milestones and produces sub-task commands for the VLA to execute. Although effective in simpler tasks, this factorized approach creates artificial silos that isolate the VLA from the rich historical context required for fine-grained reasoning. Furthermore, these approaches discard essential spatiotemporal information inherently encoded in past video frames—data that VLAs are already structured to process and act on.

To bridge this gap, we present UniMem, a unified, multimodal memory framework that equips vision-language-action models with direct access to historical context across both language and vision. UniMem is unified in two key respects: it integrates historical textual events with visual context, and it unifies memory and control under a shared backbone, as shown in Figure 1. Crucially, UniMem positions the memory loop adjacent to the control loop to seamlessly integrate multimodal history, minimizing inference latency in non-Markovian tasks.

Indeed, a longer historical context can also introduce spurious correlations [35, 11, 32]: a model that sees past observations regardless of their relevance will learn to exploit coincidental structure in the training data rather than the desired relationship between memory and action. In contrast to prior work, UniMem introduces an event-driven memory architecture that retains task-relevant textual and visual context to guide future action prediction. By separating meaningful history from task-irrelevant observations, this design mitigates this causal confusion and extends the VLA’s temporal efficiency. Specifically, UniMem employs a lightweight event classifier to decode latent event signatures and save memories. Since memory grows incrementally, context can be efficiently cached—preserving real-time inference speeds by eliminating expensive attention-based computations over raw images. Furthermore, because frames are cached only when internal VLA features trigger the classifier head, our event detector serves a dual purpose as an in-distribution gating mechanism, ensuring the policy conditions exclusively on keyframes from high-confidence task milestones.

Our core contributions are as follows:

- •

Multi-Modal Memory Conditioning: We present a framework that equips VLAs with a unified multimodal memory across both textual and visual modalities for robust performance in challenging non-Markovian tasks. By integrating the VLA’s memory and control loops, UniMem enables a seamless training pipeline and real-time inference.
- •

Event-Driven Architecture: We introduce a lightweight event classifier that dynamically detects task-critical milestones within the VLA’s latent space and progressively constructs task-relevant memory from informative language and vision keyframes.
- •

Real-Time Inference via Keyframe Caching: By caching keyframes across inference steps, UniMem preserves single-frame inference speeds (${\sim}90$ ms) despite conditioning on rich visual histories, achieving a $6\times$ speedup over hierarchical memory baselines.

Through extensive simulation and hardware experiments, we demonstrate the robust performance of UniMem compared to state-of-the-art baselines, e.g., MemER [33]. Across nine benchmark tasks, UniMem achieves approximately $93.4\%$ and $80.0\%$ average success rates in simulation and real-world hardware, respectively, compared to $72.6\%$ and $43.5\%$ success rates achieved by the best-performing baselines.

## II Related Work

### II-A Recurrent & Sequence-Based Robotic Memory

To tackle partial observability and non-Markovian dynamics, early approaches incorporated temporal sequence models over continuous observation histories. Standard methods explored recurrent architectures like LSTMs and RNNs over temporal sequences [28, 23] or state-space belief encoders [21] to maintain implicit hidden states across time. Some methods maintain memory through proprioception [39], although this is limited to tasks where past scene states are not necessary for future task decisions. Other frameworks use transformer-based architectures to ingest temporal sequences of frames [13, 17, 29], although the length of these sequences are limited and do not necessarily capture all task-pertinent moments due to fixed-interval sampling. While effective in low-dimensional or short-horizon domains, blindly ingesting dense observation streams scales poorly with horizon length—leading to severe computational overhead, context degradation, and an inability to selectively retain task-critical events.

### II-B Decoupled & Multimodal Memory in VLA Paradigms

VLA models generalize across diverse robotic tasks by scaling up internet-scale vision-language pretraining to low-level motor control. While foundational works display remarkable semantic reasoning and zero-shot generalization to novel objects and scenes across different embodiments [4, 27, 25, 36], they provide actions based solely on the immediate camera frame and task instruction. While highly effective for short-horizon tasks and fully observable decision processes, this reactive nature is fundamentally limiting in tasks that demand historical context. To equip these models with long-horizon memory, recent works vary in both the representation stored and how memory conditions the policy.

For navigation, hybrid systems leverage a high-level VLM and a low-level VLA to construct topological environment graphs [37]. For tabletop manipulation, frameworks like SAM2Act [12] maintain spatial memory of past scene states via a latent memory bank. MemER [33] saves historically relevant keyframes, but its decoupled architecture prevents the low-level actor from conditioning directly on memory. Some works have managed to unify the hierarchical approach in one VLA for complicated multi-stage manipulation tasks [22], with thinking and acting modes swapped intermittently; other recent methods condition the low-level policy on temporal image sequences [24, 19]. MEM [34] introduces a memory architecture for VLAs that aligns more closely with prevailing paradigms. Specifically, MEM leverages a VLM to generate textual summaries of past events and subsequently assigns sub-tasks to the VLA. However, this approach perpetuates the artificial silos separating VLAs from critical historical context. Furthermore, by combining sub-task commands with fixed-interval historical frames, MEM exposes the VLA to visual distractions.

While these methods touch on many aspects of memory individually, none condition the VLA directly on memory from multiple modalities, under one unified architecture. In contrast to MEM and prior methods, UniMem unifies memory retrieval and action generation under a single backbone, enabling event-driven visual and textual conditioning while maintaining low-latency control.

## III Unified Memory for VLAs

### III-A Problem Formulation and Method Overview

We consider a long-horizon, non-Markovian robotic manipulation task framed as a Partially Observable Markov Decision Process (POMDP). At each timestep $t$, the agent receives an observation of its environment $o_{t}$ and a high-level task instruction $g$. A standard reactive policy, which conditions exclusively on the current observation ($\pi(a_{t:t+H}|o_{t},g)$), fails when tasks require historical context. We characterize this failure mode as perceptual aliasing. Formally, perceptual aliasing occurs when two global steps $i$ and $j$ in a demonstration dataset yield near-identical observations (${o_{i}\approx o_{j}}$), but require different actions (${a_{i}\neq a_{j}}$) depending on task history. Resolving this ambiguity requires conditioning on a history of past states, mapping $o_{t-T:t}$ to the correct low-level motor action $a_{t}$.

UniMem addresses this problem by maintaining two complementary forms of history: textual memory ($\mathcal{M}_{t}$) that records discrete task events and visual keyframe memory ($\mathcal{H}_{t}$) that retains key spatial information. Both forms are updated online by an event classifier integrated directly into the VLA backbone and jointly condition subsequent action prediction.

### III-B Event-Driven Multimodal Memory

We implement UniMem on top of $\pi_{0.5}$ [27], an open source VLA consisting of a PaliGemma backbone and a Gemma action expert [3]. To compress execution history into discrete, task-relevant events, we define the event vocabulary $\mathcal{E}_{\text{all}}=\mathcal{E}\cup\{\text{null}\}$, where $\mathcal{E}$ represents the subset of valid, task-critical milestones (e.g., "grabbed box"). As shown in Figure 1, we append an event classifier head $f_{\phi}$, implemented as an MLP, directly onto the final layer latent representation $z_{t}$ extracted from the VLA backbone. The classifier produces a probability distribution over event classes, and the predicted event $e_{t}\in\mathcal{E}_{\text{all}}$ at step $t$ is chosen via the most likely class index:

|  | $$ e_{t}=\mathcal{E}_{\text{all}}\left[\arg\max f_{\phi}(z_{t})\right]. $$ |  | (1) |
|---|---|---|---|

We use $z_{t}$ for event prediction because it is the final-layer representation that would otherwise be used for autoregressive text generation in the underlying VLM. Rather than using this representation to predict a language token, we pass $z_{t}$ through the MLP event classifier $f_{\phi}$ to predict the event $e_{t}$. Because $f_{\phi}$ is trained jointly with the VLA, the event classification loss in Eq. (2) also updates the shared backbone used for action generation. This couples memory learning with control—rather than separating them across hierarchical modules—and prioritizes latent features that support both objectives.

At each step $t$, if the predicted class ${e_{t}\in\mathcal{E}}$ (i.e., ${e_{t}\neq\text{null}}$), the framework triggers a memory update and $e_{t}$ is dynamically appended in the form of natural language to our textual history $\mathcal{M}_{t}$, modifying the language instruction for the subsequent step: $\mathcal{M}_{t+1}=[\mathcal{M}_{t}\mathbin{\|}e_{t}]$. At the start of every rollout, we initialize $\mathcal{M}_{t}=\emptyset$. This tightly integrated loop provides explicit textual context that cleanly breaks the visual ambiguity of downstream states and anchors the model to its own task progress.

The same detection of a task-critical event also triggers a visual memory update. When ${e_{t}\in\mathcal{E}}$, UniMem stores the corresponding multi-view visual observation ${I=\{I^{\text{wrist}},I^{\text{ext}}\}}$. These observations are appended to our collection of keyframes, defined as ${\mathcal{H}_{t}=\{I_{k}\}_{k\in\mathcal{K}}}$, where $\mathcal{K}$ represents the set of discrete, non-consecutive event timestamps along with the current time step $t$. At the start of every rollout, we initialize $\mathcal{H}_{t}$ with only the initial scene image. Due to compute limitations during training, the size of this history is capped to three past milestones ($|\mathcal{K}|\leq 4$), dropping the oldest frame when exceeding this limit during rollouts. Depending on event sparsity, this can equate to over a minute of visual memory.

Crucially, the event classifier head $f_{\phi}$ operates on the unified latent representation from both textual memory $\mathcal{M}_{t}$ and visual history $\mathcal{H}_{t}$. Because $z_{t}$ is constructed via cross-attention over both the tokens from textual memory $\mathcal{M}_{t}$ and keyframes $\mathcal{H}_{t}$, the event classifier inherently conditions on the entire multimodal history. This creates a recursive memory loop where prior milestones directly inform the detection of subsequent events. The updated policy formulation is thus defined as $\pi(a_{t:t+H},e_{t}|\mathcal{H}_{t},\mathcal{M}_{t},q_{t},g)$, where $q_{t}$ is proprioception.

### III-C Efficient Keyframe Encoding and Caching

To ingest this keyframe bank $\mathcal{H}_{t}$ without degrading execution speeds, we modify $\pi_{0.5}$’s vision encoder (SigLIP [38]). Specifically, we interleave causal temporal self-attention every few layers of the standard spatial attention stack; at these layers, the temporal attention pass reuses that same layer’s own query/key/value and layer-norm weights, a configuration closely adapted from [34, 2]. Unlike prior approaches that sample frames at fixed temporal intervals and rely on separate pretraining, we train our keyframe encoder end-to-end during fine-tuning. Because keyframes explicitly isolate semantically meaningful events rather than potentially uninformative fixed-interval frames, they provide a concentrated signal for task-relevant memory representations without requiring a separate pretraining phase.

To avoid repeatedly re-encoding historical keyframes, we cache their intermediate representations in a feature cache $\mathcal{H}_{\text{cache}}$ for future timesteps to attend to. Specifically, we cache their hidden states prior to the positional embeddings and layer-normalizations applied during each temporal attention step. When a new frame arrives, we simply shift the positional embedding applied to each cached entry without recomputing costly spatial attention at each layer; the current image alone forms the query for temporal attention. This allows the encoder to project dense spatiotemporal representations into the embedding of the VLA backbone while preserving a real-time control loop as temporal context scales.

### III-D Data and Training

![Refer to caption](drafts/images/unimem-2608.22869/method_labeling.png)

*Fig. 2: Overview of UniMem’s automated data labeling procedure. We prompt an agent to generate a script for the given task that labels demonstrations with $e_{t}$ and $\mathcal{M}_{t}$. Multiple frames in the window $W_{i}$ are labeled with the corresponding event, and $\mathcal{M}_{t}$ is only updated once this window has passed. At training time, a keyframe from each past $W_{i}$ along with $I_{t}$ is used to build $\mathcal{H}_{t}$.*

To train our policies end-to-end, we need a robot action dataset (containing $a_{t:t+H},o_{t},q_{t}$, and $g$) labeled with $e_{t}$ and $\mathcal{M}_{t}$. Once demonstrations have been collected for a specific task, we prompt an agent (Claude Sonnet 5.0 [1]) to generate a script that parses our dataset for action signatures with the corresponding $e_{t}\in\mathcal{E}_{\text{all}}$ (i.e., a robot pitching upwards means scooping, the gripper closing means a bottle is being grasped, etc.). Frames outside these event windows get labeled with the null class by our script. For simplicity, we predefine our event vocabulary $\mathcal{E}_{\text{all}}$, although this could be constructed autonomously using methods like RoboInter [18] in future work. To prevent $\mathcal{M}_{t}$ from leaking the target during training, we only update $\mathcal{M}_{t}$ with an event once the entire window has passed. After the script is run on our dataset, a human verifies a subset of the demonstration videos has been labeled correctly and refines the script through more prompting if necessary. We build $\mathcal{H}_{t}$ at training time, sampling a single frame uniformly from each past event window along with $I_{t}$. Refer to Figure 2 for a conceptual overview of this process.

With our dataset curated, we train the policy end-to-end with a two-term objective: the standard action-chunking loss and an auxiliary cross-entropy on the event classifier head.
We keep the flow-matching objective of $\pi_{0.5}$ [27] unchanged. The action expert integrates the chunk $a_{t:t+H}$ along the denoising velocity field conditioned on $o_{t}$; we denote this term $\mathcal{L}_{\text{a}}(\theta)$.
We supervise the event classifier head $f_{\phi}$ with a class-weighted cross-entropy loss against the automatically generated label $e_{t}$:

|  | $$ \mathcal{L}_{\text{e}}(\theta,\phi)=-\,w(e_{t})\,\log p_{\phi}(e_{t}\mid z_{t})~ $$ |  | (2) |
|---|---|---|---|

where $w(e_{t})=1$ if $e_{t}\in\mathcal{E}$ and $w_{\text{null}}$ if $e_{t}=\text{null}$. Null frames dominate the dataset (most timesteps are not events), so we downweight them by setting $w_{\text{null}}=0.02$ rather than masking them out of the loss entirely. The distinction matters at deployment: masking would leave the head unsupervised on precisely the frames where it must decline to fire. Training on null frames teaches the head to withhold a prediction, while the low weight keeps it from collapsing onto the majority class. The asymmetry is deliberate, since memory is append-only: a missed event can still be caught at the next timestep, whereas a single false positive writes a wrong entry into $\mathcal{M}_{t}$ and an incorrect keyframe into $\mathcal{H}_{t}$ that persist for the remainder of the rollout.
The two terms are optimized jointly,

|  | $$ \mathcal{L}=\mathcal{L}_{\text{a}}+\lambda\,\mathcal{L}_{\text{e}},\qquad\lambda=0.1. $$ |  | (3) |
|---|---|---|---|

Since $z_{t}$ is a shared backbone representation rather than a detached
feature, the classification gradient flows back through the language-model
trunk, realizing the auxiliary supervision described in
Sec. III-B.

## IV Experiments and Analyses

### IV-A Experimental Setup

We evaluate UniMem on nine memory tasks (shown in Figure 3), split across simulation and hardware. We use the robosuite environment with 7DoF Franka Panda for simulation [40] and a UFactory xArm6 for hardware. In both domains, we equip our robot with wrist-mounted and external image streams. We use real time chunking for smooth inference on hardware [5], querying UniMem at ${\sim}10$ Hz. We retain up to three event keyframes ($|\mathcal{K}|=3$) for simulation tasks and four ($|\mathcal{K}|=4$) for hardware tasks. During experiments, a human observes each rollout, either from a saved video in simulation or in real time, and decides binary subtask success.

### IV-B Baselines and Ablations

We compare UniMem against VLA memory baselines and ablations on the memory input. In simulation, we compare against $\pi_{0.5}$ augmented with a video encoder, with frames sampled at 6-second intervals and processed using our keyframe encoder architecture ($|\mathcal{K}|=4$). This baseline tests how arbitrary frame sampling performs relative to event-driven memory.

On hardware, we compare against MemER with $\pi_{0.5}$ as the primary baseline, characterizing a hierarchical memory system [33]. To isolate the contributions of textual and keyframe memory, we train ablations with no memory at all (vanilla $\pi_{0.5}$), text memory only ($I_{t}$ images only), and keyframe memory only (no $\mathcal{M}_{t}$). All ablations are trained with auxiliary classifier supervision.

![Refer to caption](drafts/images/unimem-2608.22869/tasks.png)

*Fig. 3: Overview of Tasks. (Top) Manipulation tasks in robosuite. (Bottom) Real-world tasks on the xArm6 setup.*

### IV-C Simulation Experiments and Ablations

We evaluate UniMem on five simulation tasks designed to verify its ability to overcome perceptual aliasing and outperform fixed-interval visual history baselines across sequential and spatial memory settings:

- •

UpDown: Pick up a box and put it back down once.
- •

UpDown3Times: Pick up a box and put it back down three times.
- •

OccludedTap: Pick up a box, place it in one of two bins, retract until the contents of both bins are occluded, and then tap the bin containing the box.
- •

UpDownSpatial: Pick up a box from a table and then place it back in its original location. Performance is measured using a continuous score based on the final placement position error.
- •

PlateRecall: Pick up a box from one of the four plates, place it to the side, and then tap the plate that originally had the box.

As shown in Table I, UniMem achieves a 93.4% simulation average across non-Markovian evaluation environments, substantially outperforming the fixed-window video baseline ($\pi_{0.5}\text{+V.E.}$, $68.2\%$). Basic non-Markovian tasks like UpDown and OccludedTap require minimal historical context and verify that history conditioning does not impede low-level control.

Visual memory alone struggles in counting tasks; textual memory provides critical progress information by compressing history into discrete events. In UpDown3Times, keyframe-only conditioning fails ($23\%$) due to limited temporal reach, while text-only conditioning reaches $93\%$ success.

In spatial memory tasks, visual and textual memory is similarly shown to improve task performance. In OccludedTap, the keyframe ablation performs $12\%$ better than the text ablation; we hypothesize that this is because keyframes provide many memory pathways for the policy to exploit while text is left to just one. In continuous spatial grounding (UpDownSpatial), text-only drops to $30\%$ without geometric awareness of past scenes. While the video encoder and keyframes offer better spatial guidance ($49\%$ and $52\%$ respectively), they still occasionally lose track of task progression; UniMem, which combines keyframes with textual memory, resolves both failure modes, raising success to $79\%$. Finally, in PlateRecall, text-only achieves $20\%$ success—roughly matching the 1-in-4 chance of correct plate selection. Keyframes and the video baseline match the full model ($96\%$), demonstrating that visual memory cleanly resolves discrete spatial ambiguities once progress is more easily inferred.

*TABLE I: Simulation Results & Ablations. Success rates (%) ($N=25$). ($\dagger$) reports mean subtask success due to long-horizon complexity. ($\ddagger$) reports spatial accuracy.*

| Task | $\pi_{0.5}+\text{V.E.}$ | N.M. | T.O. | K.O. | Ours |
|---|---|---|---|---|---|
| UpDown | 84 | 52 | 96 | 92 | 100 |
| UpDown3Times† | 16 | 22 | 93 | 23 | 96 |
| OccludedTap | 96 | 60 | 88 | 100 | 96 |
| UpDownSpatial‡ | 49 | 6 | 30 | 52 | 79 |
| PlateRecall | 96 | 8 | 20 | 96 | 96 |
| Avg. Performance | 68.2 | 29.6 | 65.4 | 72.6 | 93.4 |
| V.E. = Video Encoder, N.M. = No Memory, T.O. = Text Only, K.O. = Keyframe Only |

### IV-D Real-world Experiments and Analyses

Our empirical evaluations of UniMem in real-world experiments explore the following three core research questions:

1. Q1)

Does providing textual and visual memory to the VLA overcome the memory bottleneck of hierarchical systems that only condition the VLA on subtask commands?
2. Q2)

How do textual and visual memories individually contribute to policy execution, and is joint conditioning necessary for task success?
3. Q3)

Does keyframe caching enable UniMem to maintain near single-frame inference latencies and scalability?

We designed the following real-world tasks around these questions, testing both memory modalities and human interaction in long-horizon, non-Markovian settings:

- •

HammerMeasure: Measure the width of a hammer using a tape measure while controlling its retraction to prevent the hook from slipping from the hammer.
- •

BeanScoop: Pick up a spoon, put three scoops of beans into a bowl, and then place the spoon back.
- •

TableClean: Pick and place a bottle off a 60 cm by 80 cm table, pick up a sponge, wipe the bottle’s original location, and then place the sponge back.
- •

TapScoopPour: Wait for a human to tap one of the eight cups, pick up a spoon, scoop the beans, pour the beans into the cup that the human tapped, and then place the spoon back.

We also perform a simulated inference speed benchmark of UniMem to test Q3. Overall success on these tasks is detailed in Table II, while stage-by-stage analysis for select tasks is in Figure 4.

*TABLE II: Hardware Results & Ablations. Success rates (%) ($N=15$).*

| Task | MemER | N.M. | T.O. | K.O. | Ours |
|---|---|---|---|---|---|
| HammerMeasure | 87 | 13 | 53 | 53 | 87 |
| BeanScoop | 67 | 0 | 27 | 20 | 93 |
| TableClean | 13 | 0 | 0 | 47 | 80 |
| TapScoopPour | 7 | 7 | 7 | 27 | 60 |
| Avg. Performance | 43.5 | 5.0 | 21.8 | 36.8 | 80.0 |
| N.M. = No Memory, T.O. = Text Only, K.O. = Keyframe Only |

*Fig. 4: Cumulative Success Rates. Hardware tasks of BeanScoop, TableClean, and TapScoopPour.*

#### IV-D1 Does UniMem overcome the memory bottleneck of hierarchical VLA systems?

We test UniMem and the hierarchical baseline in tasks of graduated difficulty, ranging in the extent direct memory access is needed. In HammerMeasure, both UniMem and MemER achieve matching success rates (87%), as high-level subtask commands are sufficient when task progression is simple and failure modes are purely mechanical due to the precision required to keep the hook on the hammer. However, as tasks grow in horizon and spatial complexity, the limits of hierarchical methods become evident.

In BeanScoop, MemER performs better than the ablations and proceeds to the third scoop in 67% of rollouts. However, after completing two scoop-pour cycles, MemER’s high-level planner may output an incorrect subtask command—instructing the robot to place the spoon when it should command a third scoop. By the time the high-level planner outputs a correction, the low-level policy is in an unrecoverable regime and ignores subsequent commands. In contrast, UniMem conditions the policy on memory directly at every control step. Continuous memory access prevents high-level misclassifications from happening in the first place, while near single-frame inference speeds ensure the robot continuously adjusts its trajectory before physical drift becomes uncorrectable, ultimately achieving 93.0% success.

In TableClean, we test continuous spatial memory. Although MemER conditions the high-level policy on keyframes and augments its commands to the low-level policy with cues like left, right, and center, it only achieves 13% success due to such a large area to select from. Only when keyframes and textual memory combine in full UniMem does the policy gain access to past visual states and wipe the bottle’s exact spot on the table 80% of the time, while missing by no more than 10 cm.

In TapScoopPour, MemER pours into the correct cup 7% of the time—even though MemER’s command vocabulary is augmented, this is still insufficient to disambiguate such a large number of cups. Only when we provide visual history directly to the low-level policy do we see large improvements, with full UniMem achieving 60% success. Unified, multimodal memory provides spatial and sequential awareness directly to our VLA, an ability that hierarchical policies clearly lack.

#### IV-D2 Is joint conditioning on textual and visual memory necessary for success?

We compare our ablations in tasks of varying lengths and memory requirements to evaluate contributions of both memory modalities.
In HammerMeasure, we explicitly characterize perceptual aliasing. Without memory, the policy moves back and forth, struggling to distinguish task progress (13%). Text-only and keyframe-only ablations improve to 53%, but still struggle to distinguish and remember certain events. Only when both modalities combine in UniMem do we achieve robust memory (87%).

In BeanScoop, with no memory, the policy doesn’t stop scooping and pouring, collapsing on the most common signal from the dataset. With textual memory, long-horizon tracking improves to 27%; however, the policy often moves to the next scoop prematurely without adding the previous pour to its memory, leading to undercounting. The keyframe-only ablation ameliorates this, only proceeding to the next scoop once a pour has been committed to memory. However, it cannot count all scoop-pour cycles with a limited context window and achieves 20% success. When both modalities combine, we see robust long- and short-term memory, with UniMem achieving 93% success.

In TableClean, with no memory, the policy fails to even begin wiping as the grabbing and placing actions alias each other. With only text, the policy wipes in random locations and does not perform a full wiping motion. With keyframes, the policy is endowed with spatial memory, completing the correct wipe 53% of the time, but nonetheless suffers from infinite wipe sequences and fails to place the sponge. Only when keyframes and textual memory combine in full UniMem does the policy overcome these specific failure modes by gaining access to sequential and spatial memory.

In TapScoopPour, the policy without memory only proceeds to the grasp after the first tap 27% of the time, requiring multiple taps for the rest. This is because as soon as the human stops tapping, the robot loses its signal to start. In the text ablation, the policy manages to pour into the correct cup 7% of the time, essentially picking at random. Only with keyframes do we see an improvement, with that ablation achieving 27% success. Its errors are also qualitatively different: misses are to an adjacent cup rather than arbitrary, indicating that the keyframe encoder does supply the correct spatial content. Adding textual memory further refines performance, with full UniMem achieving exact cup selection 60% of the time. Across all tasks, neither the text- nor keyframe-only ablations matched the performance of full UniMem, demonstrating that textual and visual memory are complementary and essential for success on multimodal, non-Markovian tasks.

#### IV-D3 Does UniMem maintain near single-frame inference speeds?

![Refer to caption](drafts/images/unimem-2608.22869/2cameras.png)

![Refer to caption](drafts/images/unimem-2608.22869/4cameras.png)

*Fig. 5: Benchmarking inference latency on an RTX 4090. We compare inference speed when varying images per camera stream for 2 streams (left) and a simulated bimanual setup with 4 streams (right).*

To provide long-horizon memory with minimal computational overhead, UniMem leverages a hidden-state caching mechanism for keyframes. In a naive implementation, keyframes are retained as raw pixels and re-encoded through the vision backbone at every control step, leading to poor scaling as history grows. By instead caching the pre-computed keyframe representations prior to temporal self-attention, we eliminate redundant computations.

We benchmark the latency of our caching mechanism against an ablation without caching, single-frame $\pi_{0.5}$, and MemER’s dual-system VLM architecture across varying context window sizes for 2 and 4 camera streams to the left and right, respectively, of Figure 5. Even on a standard workstation GPU, maintaining a 16-keyframe memory context across four camera streams adds only ${\sim 25}$ milliseconds of latency beyond the single-frame, 2-camera stream base policy. Although our experiments utilize contexts of up to 4 keyframes for 2-camera streams, these figures demonstrate that UniMem can scale to significantly longer context windows and different embodiments like a bimanual manipulator with 4-camera streams without sacrificing inference speed. Thus, in tasks significantly longer than the ones we evaluated UniMem on, the bottleneck would be dataset coverage and training compute, which we leave for future work.

## V Conclusion

We present UniMem, a streamlined framework that unifies long-horizon spatial and sequential memory within a single VLA architecture. By using an event classifier to autoregressively update its multimodal history, UniMem avoids the memory bottleneck and high-latency of dual-system architectures. Across nine tasks in simulation and hardware, our system demonstrates superior task success, simpler training pipelines, and low-latency real-world rollouts.

Despite its strong empirical performance, our framework has several limitations that highlight promising directions for future research. First, while UniMem demonstrates robust state tracking over multi-minute execution horizons, it has not yet been evaluated on extended tasks spanning tens of minutes or hours. Exciting future work includes extending this framework with memory editing mechanisms—such as pruning and consolidation—to maintain high-performance.

Second, our current keyframe extraction pipeline relies on automated offline labeling. While this eliminates some human labor when curating the dataset, future work could leverage unsupervised temporal clustering or reinforcement learning to let the policy autonomously determine which events warrant long-term retention.

Finally, UniMem maintains memory with only one temporal context that extends several minutes. Distinguishing between short and long-term memories, to not only remember past environmental states but also improve motor function and learn from mistakes, remains a promising direction for memory conditioning. Memory that lives through not only the input stream, but also in the weights of the policy itself, could make memory more natural and human-like.

## ACKNOWLEDGMENT

Toyota Research Institute provided funds to support this work. M. Wang is supported by the NASA NSTGRO Fellowship.

## References

- [1]
Anthropic (2026)

Claude 5.0 sonnet.

Note: Large language model

External Links: [Link](https://claude.ai)

Cited by: §III-D.
- [2]
G. Bertasius, H. Wang, and L. Torresani (2021)

Is space-time attention all you need for video understanding?.

In Proceedings of the 38th International Conference on Machine Learning,  M. Meila and T. Zhang (Eds.),

Proceedings of Machine Learning Research, Vol. 139, pp. 813–824.

External Links: [Link](https://proceedings.mlr.press/v139/bertasius21a.html)

Cited by: §III-C.
- [3]
L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, et al. (2024)

PaliGemma: a versatile 3b vlm for transfer.

arXiv preprint arXiv:2407.07726.

Cited by: §III-B.
- [4]
K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky (2025)

$\pi_{0}$: A vision-language-action flow model for general robot control.

In International Conference on Learning Representations (ICLR),

Cited by: §I,
§II-B.
- [5]
K. Black, M. Galliker, and S. Levine (2025)

Real-time execution of action chunking flow policies.

In Advances in Neural Information Processing Systems,  D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Eds.),

Vol. 38, Main Conference, pp. 33383–33407.

External Links: [Document](https://dx.doi.org/10.52202/085713-1122),
[Link](https://proceedings.neurips.cc/paper_files/paper/2025/file/300ccb2187dedd4edcc07f7e76d8e553-Paper-Conference.pdf)

Cited by: §IV-A.
- [6]
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich (2023)

RT-2: vision-language-action models transfer web knowledge to robotic control.

In Conference on Robot Learning (CoRL),

Cited by: §I.
- [7]
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich (2023)

RT-1: robotics transformer for real-world control at scale.

In Robotics: Science and Systems (RSS),

Cited by: §I.
- [8]
T. Chen, Y. Wang, M. Li, Y. Qin, H. Shi, Z. Li, Y. Hu, Y. Zhang, K. Wang, Y. Chen, H. Wang, R. Xu, R. Wu, Y. Mu, Y. Yang, H. Dong, and P. Luo (2026)

RMBench: memory-dependent robotic manipulation benchmark with insights into policy design.

arXiv preprint arXiv:2603.01229.

External Links: [Link](https://arxiv.org/abs/2603.01229)

Cited by: §I.
- [9]
Y. Chen, F. Xue, D. Li, Q. Hu, L. Zhu, X. Li, Y. Fang, H. Tang, S. Yang, Z. Liu, et al. (2025)

Longvila: scaling long-context visual language models for long videos.

In International Conference on Learning Representations,

Vol. 2025, pp. 18227–18246.

Cited by: §I.
- [10]
N. Chung, T. Hanyu, T. Nguyen, H. Le, F. Bumgarner, D. M. H. Nguyen, K. Vo, K. Yamazaki, C. Rainwater, T. Kieu, A. Nguyen, and N. Le (2026)

Rethinking progression of memory state in robotic manipulation: an object-centric perspective.

In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI),

Cited by: §I.
- [11]
P. de Haan, D. Jayaraman, and S. Levine (2019)

Causal confusion in imitation learning.

In Advances in Neural Information Processing Systems (NeurIPS),

Cited by: §I.
- [12]
H. Fang, M. Grotz, W. Pumacay, Y. R. Wang, D. Fox, R. Krishna, and J. Duan (2025)

SAM2Act: integrating visual foundation model with a memory architecture for robotic manipulation.

In Proceedings of the International Conference on Machine Learning (ICML),

Cited by: §II-B.
- [13]
K. Fang, A. Toshev, L. Fei-Fei, and S. Savarese (2019)

Scene memory transformer for embodied agents in long-horizon tasks.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 5380–5390.

Cited by: §II-A.
- [14]
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022)

LoRA: low-rank adaptation of large language models.

In International Conference on Learning Representations (ICLR),

Cited by: Appendix B.
- [15]
P. Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, V. Choudhary, F. Collins, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, M. Dhaka, J. DiCarlo, D. Driess, M. Equi, A. Esmail, Y. Fang, C. Finn, C. Glossop, T. Godden, I. Goryachev, L. Groom, H. Habeeb, H. Hancock, K. Hausman, G. Hussein, V. Hwang, B. Ichter, C. Jacobsen, S. Jakubczak, R. Jen, T. Jones, G. Kammerer, B. Katz, L. Ke, M. Khadikov, C. Kuchi, M. Lamb, D. LeBlanc, B. LeCount, S. Levine, X. Li, A. Li-Bell, V. Lialin, Z. Liang, W. Lim, Y. Lu, E. Luo, V. Mano, N. Marwaha, A. Mongush, L. Murphy, S. Nair, T. Patterson, K. Pertsch, A. Z. Ren, G. Schelske, C. Sharma, B. Shi, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, W. Stoeckle, J. Tang, J. Tanner, S. Tekeste, M. Torne, K. Vedder, Q. Vuong, A. Walling, H. Wang, J. Wang, X. Wang, C. Whalen, S. Whitmore, B. Williams, C. Xu, S. Yoo, L. Yu, W. Zhang, Z. Zhang, and U. Zhilinsky (2026)

${\pi}_{0.7}$: A steerable generalist robotic foundation model with emergent capabilities.

arXiv preprint arXiv:2604.15483.

External Links: [Link](https://arxiv.org/abs/2604.15483)

Cited by: §I.
- [16]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn (2024)

OpenVLA: an open-source vision-language-action model.

In Conference on Robot Learning (CoRL),

Cited by: §I.
- [17]
S. Lee, Y. Wang, H. Etukuru, H. J. Kim, N. M. M. Shafiullah, and L. Pinto (2024)

Behavior generation with latent actions.

In Proceedings of the 41st International Conference on Machine Learning,  R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp (Eds.),

Proceedings of Machine Learning Research, Vol. 235, pp. 26991–27008.

External Links: [Link](https://proceedings.mlr.press/v235/lee24y.html)

Cited by: §II-A.
- [18]
H. Li, Z. Wang, Z. Ding, S. Yang, Y. Chen, Y. Tian, X. Hu, T. Wang, D. Lin, F. Zhao, S. Liu, and J. Pang (2026)

RoboInter: a holistic intermediate representation suite towards robotic manipulation.

In The Fourteenth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=PGUC3mmMoi)

Cited by: §III-D.
- [19]
H. Li, S. Yang, Y. Chen, X. Chen, X. Yang, Y. Tian, H. Wang, T. Wang, D. Lin, F. Zhao, and J. Pang (2026)

CronusVLA: towards efficient and robust manipulation via multi-frame vision-language-action modeling.

In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI),

Cited by: §II-B.
- [20]
S. Lian, B. Yu, X. Lin, Z. Shen, L. T. Yang, Y. Jin, H. Liu, C. Wu, H. Yuan, C. Huang, and K. Chen (2026)

IntentVLA: short-horizon intent modeling for aliased robot manipulation.

arXiv preprint arXiv:2605.14712.

External Links: [Link](https://arxiv.org/abs/2605.14712)

Cited by: §I.
- [21]
Y. Liang and E. Noronha (2025)

MEMBOT: memory-based robot in intermittent pomdp.

arXiv preprint arXiv:2509.11225.

External Links: [Link](https://arxiv.org/abs/2509.11225)

Cited by: §II-A.
- [22]
F. Lin, R. Nai, Y. Hu, J. You, J. Zhao, and Y. Gao (2026)

OneTwoVLA: a unified vision-language-action model with adaptive reasoning.

In The Fourteenth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=tWMfhoP3as)

Cited by: §II-B.
- [23]
A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín (2022)

What matters in learning from offline human demonstrations for robot manipulation.

In Proceedings of the 5th Conference on Robot Learning,  A. Faust, D. Hsu, and G. Neumann (Eds.),

Proceedings of Machine Learning Research, Vol. 164, pp. 1678–1690.

External Links: [Link](https://proceedings.mlr.press/v164/mandlekar22a.html)

Cited by: §II-A.
- [24]
M. S. Mark, J. Liang, M. Attarian, C. Fu, D. Dwibedi, D. Shah, and A. Kumar (2025)

BPP: long-context robot imitation learning by focusing on key history frames.

In Conference on Robot Learning,

Vol. 297, pp. 2679–2713.

Cited by: §II-B.
- [25]
NVIDIA GR00T Team, J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. ”. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu (2025)

GR00T N1: an open foundation model for generalist humanoid robots.

arXiv preprint arXiv:2503.14734.

External Links: [Link](https://arxiv.org/abs/2503.14734)

Cited by: §I,
§II-B.
- [26]
Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. L. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine (2024)

Octo: an open-source generalist robot policy.

In Proceedings of Robotics: Science and Systems,

Delft, Netherlands.

Cited by: §I.
- [27]
Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky (2025)

$\pi_{0.5}$: A vision-language-action model with open-world generalization.

In Proceedings of the Conference on Robot Learning (CoRL),

Cited by: §I,
§II-B,
§III-B,
§III-D.
- [28]
R. Rahmatizadeh, P. Abolghasemi, L. Bölöni, and S. Levine (2018)

Vision-based multi-task manipulation for inexpensive robots using end-to-end learning from demonstration.

In 2018 IEEE International Conference on Robotics and Automation (ICRA),

pp. 3758–3765.

Cited by: §II-A.
- [29]
N. M. M. Shafiullah, Z. J. Cui, A. Altanzaya, and L. Pinto (2022)

Behavior transformers: cloning $k$ modes with one stone.

In Advances in Neural Information Processing Systems (NeurIPS),

Cited by: §II-A.
- [30]
H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang (2026)

MemoryVLA: perceptual-cognitive memory in vision-language-action models for robotic manipulation.

In The Fourteenth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=54U3XHf7qq)

Cited by: §I,
§I.
- [31]
L. X. Shi, B. Ichter, M. R. Equi, L. Ke, K. Pertsch, Q. Vuong, J. Tanner, A. Walling, H. Wang, N. Fusai, A. Li-Bell, D. Driess, L. Groom, S. Levine, and C. Finn (2025)

Hi robot: open-ended instruction following with hierarchical vision-language-action models.

In Proceedings of the 42nd International Conference on Machine Learning,  A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu (Eds.),

Proceedings of Machine Learning Research, Vol. 267, pp. 54919–54933.

External Links: [Link](https://proceedings.mlr.press/v267/shi25d.html)

Cited by: §I.
- [32]
J. Spencer, S. Choudhury, A. Venkatraman, B. Ziebart, and J. A. Bagnell (2021)

Feedback in imitation learning: the three regimes of covariate shift.

In Proceedings of the 38th International Conference on Machine Learning (ICML),

Cited by: §I.
- [33]
A. Sridhar, J. Pan, S. Sharma, and C. Finn (2026)

MemER: scaling up memory for robot control via experience retrieval.

In The Fourteenth International Conference on Learning Representations,

External Links: [Link](https://arxiv.org/abs/2510.20328)

Cited by: §I,
§I,
§II-B,
§IV-B.
- [34]
M. Torne, K. Pertsch, H. Walke, K. Vedder, S. Nair, B. Ichter, A. Z. Ren, H. Wang, J. Tang, K. Stachowicz, K. Dhabalia, M. Equi, Q. Vuong, J. T. Springenberg, S. Levine, C. Finn, and D. Driess (2026)

MEM: multi-scale embodied memory for vision language action models.

arXiv preprint arXiv:2603.03596.

External Links: [Link](https://arxiv.org/abs/2603.03596)

Cited by: §II-B,
§III-C.
- [35]
M. T. Villasevil, A. Tang, Y. Liu, and C. Finn (2025)

Learning long-context diffusion policies via past-token prediction.

In Proceedings of The 9th Conference on Robot Learning,  J. Lim, S. Song, and H. Park (Eds.),

Proceedings of Machine Learning Research, Vol. 305, pp. 1744–1755.

External Links: [Link](https://proceedings.mlr.press/v305/villasevil25a.html)

Cited by: §I.
- [36]
S. Wei, H. Jing, B. Li, Z. Zhao, J. Mao, Z. Ni, S. He, J. Liu, X. Liu, K. Kang, S. Zang, W. Yuan, M. Pavone, D. Huang, and Y. Wang (2026)

$\Psi_{0}$: An open foundation model towards universal humanoid loco-manipulation.

In Proceedings of Robotics: Science and Systems (RSS),

Cited by: §II-B.
- [37]
Z. Xu, H. L. Chiang, Z. Fu, M. G. Jacob, T. Zhang, T. E. Lee, W. Yu, C. Schenck, D. Rendleman, D. Shah, F. Xia, J. Hsu, J. Hoech, P. Florence, S. Kirmani, S. Singh, V. Sindhwani, C. Parada, C. Finn, P. Xu, S. Levine, and J. Tan (2025)

Mobility vla: multimodal instruction navigation with long-context vlms and topological graphs.

In Proceedings of The 8th Conference on Robot Learning,  P. Agrawal, O. Kroemer, and W. Burgard (Eds.),

Proceedings of Machine Learning Research, Vol. 270, pp. 3866–3887.

External Links: [Link](https://proceedings.mlr.press/v270/xu25b.html)

Cited by: §II-B.
- [38]
X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer (2023)

Sigmoid loss for language image pre-training.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),

pp. 11975–11986.

Cited by: §III-C.
- [39]
Z. Zhang, H. Xu, Z. Yang, C. Yue, Z. Lin, H. Gao, Z. Wang, and H. Zhao (2025)

TA-vla: elucidating the design space of torque-aware vision-language-action models.

In Proceedings of the 9th Conference on Robot Learning (CoRL),  J. Lim, S. Song, and H. Park (Eds.),

Vol. 305, pp. 4019–4037.

Cited by: §II-A.
- [40]
Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, K. Lin, S. Nasiriany, and Y. Zhu (2020)

Robosuite: a modular simulation framework and benchmark for robot learning.

In arXiv preprint arXiv:2009.12293,

Cited by: §IV-A.

## Appendix A Data Curation

We collect human teleoperated robot demonstrations at 10 Hz in simulation and 20 Hz on hardware. All labeling is done via a VLM-generated script, except for the human tap in TapScoopPour which is flagged whilst collecting the demonstration. An example prompt provided to Claude for the generation of labeling script for TapScoopPour is shown below.

Example Claude Sonnet 5.0 Prompt

Write a labeling function for our xArm demonstration dataset. The task being demonstrated is: a human taps one cup, and the robot then puts a single scoop of beans into that cup.
Input. One LeRobot parquet per episode, recorded at 20 fps. Per frame you have the end-effector pose ($xyz$ in mm, roll/pitch/yaw), a gripper channel, and a human_event column that is 1.0 on the single frame where the operator pressed the tap key during teleoperation.
Output. The script should label each frame in the dataset with both a discrete event id from a vocabulary set and a textual memory string.
Vocabulary. Five events, each occurring exactly once per episode:
{0: "human tap", 1: "grabbed spoon", 2: "scooped beans", 3: "poured beans",
4: "placed spoon"}. Frames belonging to no event get the null target -1.
Detection and labeling.

•

0 — human tap: first frame where human_event == 1.0. Warn if
there is more than one marker and use the first.

•

1 — grabbed spoon: the fully-closed gripper plateau of the first
gripper close occurring after the tap.

•

2 — scooped beans: scan for the first
40-frame window satisfying all of: roll std $<5^{\circ}$, mean roll within
$\pm 20^{\circ}$ of $180^{\circ}$ (either sign), yaw std $<5^{\circ}$, mean
$z<255$ mm, and pitch increasing on at least 60% of frames (sustained
straightening while low in the bowl). Label the end of the 40-frame window with this event.

•

3 — poured beans: the frame of minimum roll.

•

4 — placed spoon: first gripper open after the scoop.

Label the window of frames surrounding the event, starting 5 frames before and ending 20 frames after the detection.
Textual memory. An event’s phrase becomes visible in the memory string only
once the frame is no longer labeled with that event. Render as
"History: human tap, grabbed spoon, …", and "History: none" before
the first event is visible.

## Appendix B Training Details

We fine-tune VLA checkpoints (MemER low-level $\pi_{0.5}$, $\pi_{0.5}+\text{V.E.}$, all ablations, and full UniMem) with LoRA [14]. For tasks such as BeanScoop and TapScoopPour, we note improved performance when upweighting the probability of sampling decisive moments by $\sim 4\times$ in our training dataset. Conceptually, this focuses training effort on moments/frames where the robot’s path diverges, such as deciding whether to scoop again or which cup to pour into.

