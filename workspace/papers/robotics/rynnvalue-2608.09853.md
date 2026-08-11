# RynnValue: Scaling Robotic Value Foundation Models with Temporal Distance

Dongchi Huang<sup>∗</sup>, Hongyin Zhang<sup>∗</sup>, Bohan Hou<sup>∗</sup>, Siteng Huang<sup>†</sup>, Zhian Su, Hang Guo, Tong Lu, Zhaofeng Xu, Jiahao Tang, Jianfei Yang, Donglin Wang, Peixi Peng<sup>†</sup>, Mingxiu Chen, Deli Zhao, Xin Li

<sup>1</sup>DAMO Academy, Alibaba Group <sup>2</sup>Hupan Lab

<sup>∗</sup>Equal contribution, <sup>†</sup>Corresponding authors

General-purpose reward models are increasingly the bottleneck for scaling robot learning, yet the recipe for learning value-related capabilities from large-scale heterogeneous corpora remains underexplored. Existing approaches tie supervision to task-internal anchors such as preferences or normalized progress, none of which transfer cleanly across embodiments and data sources. We introduce RynnValue, an open-source value foundation model for robotic manipulation that replaces these anchors with temporal distance, the directed cost-to-go from an observation to the language-specified goal. Because temporal-distance labels can be derived directly from timestamps, RynnValue scales to over 7,000 hours and roughly 3M instruction-conditioned clips without preference or progress annotations. To make temporal-value learning reliable at scale, we combine random temporal sampling, temporal-order shufling, and value-isolation attention, suppressing shortcuts that would leave predictions insensitive to failures and regressions. Trained without preference labels, RynnValue attains an average Kendall’s $\tau _ { a }$ of 0.675 on RBM-EVAL-OOD, surpassing the fully preference-supervised state of the art (0.655) and more than doubling a progress-only counterpart (0.292), while generalizing zero-shot to unseen tasks, embodiments, and viewpoints. Converted into dense rewards via potential-based shaping, it raises real-world policy success from 52.5% to 72.5% online and from 63.8% to 82.5% ofline. These results establish temporal distance as a scalable supervision target and practical reward interface for generalist robot policies.

https://alibaba-damo-academy.github.io/RynnValue.github.io

https://github.com/alibaba-damo-academy/RynnValue

https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnvalue

https://www.modelscope.cn/collections/DAMO\_Academy/RynnValue

Date: August 11, 2026

DAMO TECH TO THE FUTURE

## 1 Introduction

Generalist robot policies are increasingly trained with reinforcement learning [2, 12], yet scaling this loop is bottlenecked not by policy capacity but by reward supervision. Hand-designed rewards hardly generalize to open-ended tasks, while sparse success signals ofer limited guidance for long-horizon behavior [31]. Existing general-purpose reward models rely on task-internal anchors such as preferences, reference demonstrations, or local state comparisons [31, 34, 18, 26], which tie supervision to particular trajectories or comparison sets and are dificult to reuse across heterogeneous data. The common fallback, normalized [0,1] progress [16, 15], is an intra-trajectory coordinate rather than a goal-conditioned cost-to-go [25, 24], making it poorly aligned with the standard notion of value in control and dificult to maintain consistently across varying durations, embodiments, and task structures.

We argue for a shift in framing: from a reward model scoring trajectory-level anchors to a value foundation model predicting goal-conditioned cost-to-go as a single reusable interface. We instantiate this view in RynnValue, which adopts temporal distance as its scaling target. Rather than normalizing progress within a trajectory, RynnValue estimates the directed temporal cost from the current observation to the languagespecified goal. Under a minimum-time objective, this corresponds to the hitting-time cost-to-go, yielding clear <Value> <Value>directionality and task conditioning. It can further be converted into dense rewards through potential-based shaping [22], serving as a practical value interface rather than a task-specific heuristic.

![](images/6ad4cabb23a271c74f5750286c5ca64ce6bdeb96896c7a502df0d853ba1443fc.jpg)  
Figure 1 Overview of RynnValue. RynnValue is a language-conditioned value model trained on over 7,000 hours of heterogeneous embodied data, comprising roughly 3M instruction-conditioned trajectory clips across diverse embodiments, viewpoints, and task domains. Given a language instruction and a sequence of sampled observations, the model builds an interleaved multimodal sequence of repeated absolute-value and relative-value queries, which RynnBrain encodes in a single forward pass. Two distributional heads then predict the absolute temporal distance to task completion and the signed relative temporal displacement between observations, while the language branch produces video analysis and language-conditioned verification. The resulting temporal values serve as a unified interface for progress estimation, failure detection, and reward specification in robotic reinforcement learning.

Crucially, temporal distance scales naturally to heterogeneous data, since labels can be derived directly from timestamps once a completion cutof is identified. Using this recipe, we construct a large training mixture of real-world, simulated, and egocentric trajectories spanning diverse embodiments, viewpoints, and task families. After preprocessing, the corpus exceeds 7,000 hours and comprises approximately 3M instruction-conditioned clips. Raw episode boundaries often do not reflect semantic task completion; we therefore apply subtask segmentation and cutof relabeling to define coherent completion points. Unlike progress and comparison-based supervision [16], this recipe requires only instructions, timestamps, and relabeled cutofs, without constructing preference pairs or progress annotations.

Scaling data alone, however, does not guarantee reliable temporal-value learning. In a multi-frame setting, the model may exploit regular sampling intervals, the presentation order of observations, or other value-query representations, rather than grounding predictions in task-relevant visual evidence. Such shortcuts can leave learned values insensitive to regressions, failures, and other non-monotonic events [8]. RynnValue therefore incorporates complementary designs at two levels. At the visual level, we sample observations at irregular timestamps and perturb their temporal order during training, breaking the correspondence between sequence position and temporal distance. At the value level, we introduce value-isolation attention, so that each value-query group attends only to its own language-visual context and cannot access the value queries of other observations. Together, these designs suppress both temporal-order and value-extrapolation shortcuts, forcing each prediction to rely on the corresponding visual observation and task semantics. RynnValue further jointly supervises natural-language video description, instruction matching, and success prediction, adding sequence-level semantic cues alongside continuous temporal-value learning.

We evaluate RynnValue both as a standalone value model and as a reward interface for downstream policy learning. Trained without preference labels, RynnValue-8B attains an average Kendall’s $\tau _ { a }$ of 0.675 on RBM-EVAL-OOD, surpassing the fully preference-supervised state of the art (0.655) and more than doubling a progress-only counterpart (0.292), while generalizing zero-shot to unseen tasks, embodiments, and viewpoints. Converted into dense rewards via potential-based shaping, it raises real-world policy success from 52.5% to 72.5% online and from 63.8% to 82.5% ofline over the strongest baseline.

In summary, our work makes the following contributions:

• We identify normalized progress as a supervision bottleneck for general-purpose robotic reward modeling, and reframe the problem as learning a value foundation model whose scaling target is temporal distance (cost-to-go), a scalable, preference-free value target that unifies heterogeneous data under a single interface.

• We present RynnValue, a robotic value foundation model trained with a label-cheap data recipe: labels are derived directly from timestamps, aided by subtask segmentation and cutof relabeling, over 7,000 hours and roughly 3M instruction-conditioned clips spanning diverse embodiments, viewpoints, and task families.

• We introduce complementary shortcut-suppression designs, temporal-order shuffling and value-isolation attention, that keep temporal-value predictions grounded in visual evidence rather than spurious correlations, and pair them with dual distributional heads for absolute and relative temporal values.

• We show that $\mathrm { R y }$ nnValue surpasses preference-supervised state-of-the-art on out-of-distribution trajectory ranking without any preference annotation, and via potential-based shaping, serves as a dense reward interface that improves both online and ofline real-world policy learning.

## 2 Model Architecture

As shown in Figure 1, RynnValue is a language-conditioned visual value model that estimates the temporal distance from robot observations to task completion. RynnValue is built on RynnBrain [5] and further pretrained on large-scale robot data, which provides representations adapted to embodied scenes, robot states, and language-conditioned manipulation tasks.

Given an instruction $\ell ,$ which includes the task instruction and system prompt of absolute/relative temporal questions, embodiment metadata $m ,$ and a sequence of K observations $\mathcal { T } = \{ I _ { t _ { i } } \} _ { i = 1 } ^ { K }$ , RynnValue jointly predicts the absolute temporal distance of each observation and the relative temporal displacement between consecutive observations. In our main setting, we sample $K = 8$ observations from each video. We then describe the notable elements in the architecture design.

Grouped temporal queries. A single query token is an information bottleneck for temporal-distance estimation, which must summarize several visual cues at once: object configuration, robot–object interaction, intermediate task stages, and task-completion evidence. Following [11], we therefore represent each temporal prediction using a group of N repeated query tokens, allowing the model to construct a richer and more robust temporal representation.

For each observation $I _ { t _ { i } }$ , we insert an absolute-value query group $\mathbf { V } _ { i } = [ V _ { i , 1 } , \ldots , V _ { i , N } ]$ , whose tokens share the <value> query type. Starting from the second observation, we additionally insert a relative-value query group $\mathbf { R } _ { i - 1 } = [ R _ { i - 1 , 1 } , \dotsc , R _ { i - 1 , N } ]$ before $\mathbf { V } _ { i }$ , whose tokens share the <relative\_value> query type. In our implementation, we use $N = 8$ repeated tokens for both absolute- and relative-value query groups.

After the final temporal-query group, we append a natural-language verification prompt $\mathbf { p } _ { \mathrm { v e r } }$ . The resulting multimodal sequence is

$$
\mathbf { x } = \left[ m , \ell , I _ { t _ { 1 } } , \mathbf { V } _ { 1 } , I _ { t _ { 2 } } , \mathbf { R } _ { 1 } , \mathbf { V } _ { 2 } , \ldots , I _ { t _ { K } } , \mathbf { R } _ { K - 1 } , \mathbf { V } _ { K } , \mathbf { p } _ { \mathrm { v e r } } \right] .\tag{1}
$$

The verification prompt is placed after all visual observations and temporal queries. It instructs the model to first generate a natural-language analysis of the video and subsequently determine task matching and task completion.

Tokens within the same temporal-query group interact through bidirectional attention and capture complementary aspects of the corresponding observation. Rather than averaging these representations, we concatenate the N query hidden states along the feature dimension:

$$
\mathbf { \widetilde { h } } _ { i } ^ { V } = H _ { \theta } ( \mathbf { x } ) _ { \mathrm { p o s } ( V _ { i , 1 } ) } \| \cdot \cdot \| H _ { \theta } ( \mathbf { x } ) _ { \mathrm { p o s } ( V _ { i , N } ) } , \qquad \mathbf { \widetilde { h } } _ { i } ^ { R } = H _ { \theta } ( \mathbf { x } ) _ { \mathrm { p o s } ( R _ { i , 1 } ) } \| \cdot \cdot \cdot \| H _ { \theta } ( \mathbf { x } ) _ { \mathrm { p o s } ( R _ { i , N } ) } ,\tag{2}
$$

where $H _ { \theta }$ denotes the contextual representations produced by RynnBrain and ∥ denotes feature concatenation. This operation preserves the complementary information captured by diferent query positions and produces an Nd-dimensional representation for each absolute or relative temporal prediction.

Continuous temporal readouts. RynnValue employs two specialized distributional heads to produce continuous absolute and relative temporal estimates:

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { V } = \mathrm { V a l u e H e a d } \left( \widetilde { \mathbf { h } } _ { i } ^ { V } \right) , \qquad \mathbf { z } _ { i } ^ { R } = \mathrm { R e l a t i v e H e a d } \left( \widetilde { \mathbf { h } } _ { i } ^ { R } \right) . } \end{array}\tag{3}
$$

Here, $\mathbf { z } _ { i } ^ { V }$ and $\mathbf { z } _ { i } ^ { R }$ denote the bin logits produced by the absolute and relative distributional heads, respectively. Numerical prediction is performed exclusively by the two heads, while the original LM head is retained only for the subsequent natural-language video analysis and task verification.

Absolute and relative temporal estimates. The absolute target is the observed remaining time from an observation to its relabeled completion cutof, while the relative target is the signed temporal displacement between adjacent observations in the presented sequence. Relative prediction provides a local temporal learning signal that complements the globally anchored absolute temporal distance. We discretize the absolute range ([0,512]) seconds and the relative range ([-256,256]) seconds into 256 symlog-spaced bins, and train distributional heads using two-hot targets over adjacent bins [9]. This casts temporal-distance regression as a stable classification problem: symlog binning spans the wide dynamic range of temporal distances by compressing large-magnitude tails while leaving near-zero targets essentially undistorted, and the two-hot encoding represents each continuous target exactly through its two nearest bins, decoupling gradient magnitude from target scale for stable, precise regression. At inference time, predictions are decoded by taking the expected bin center in symlog space and applying the inverse symlog transform [7]:

$$
v _ { i } = \mathrm { s y m e x p } \left( \sum _ { b = 1 } ^ { \left| { \cal B } _ { \cal V } \right| } c _ { b } ^ { \cal V } \ \mathrm { s o f t m a x } ( { \bf z } _ { i } ^ { \cal V } ) _ { b } \right) , \qquad \Delta _ { i } = \mathrm { s y m e x p } \left( \sum _ { b = 1 } ^ { \left| { \cal B } _ { \cal R } \right| } c _ { b } ^ { \cal R } \ \mathrm { s o f t m a x } ( { \bf z } _ { i } ^ { \cal R } ) _ { b } \right) ,\tag{4}
$$

where $c _ { b } ^ { V }$ and $c _ { b } ^ { R }$ are the absolute and relative bin centers in symlog space, respectively, and symexp denotes the inverse symlog transformation. The decoded $v _ { i }$ is the predicted absolute temporal distance, i.e., the remaining time from observation i to task completion, whereas $\Delta _ { i }$ is the predicted signed temporal displacement between the i-th and the next presented observation; both are continuous estimates even though their heads are trained with a distributional objective.

Value-isolation attention. Without an explicit attention constraint, a temporal query can predict its value by extrapolating from previously exposed value tokens rather than by interpreting the corresponding visual evidence. This shortcut produces smooth value curves while leaving predictions insensitive to regressions, failures, and other non-monotonic events. Inspired by attention masking strategies that explicitly control information flow and prevent target leakage [32, 6], we introduce value-isolation attention as shown in Figure 2(b), which prevents absolute and relative temporal queries associated with diferent observations from attending to one another. Queries belonging to the same observation remain mutually visible, allowing their features to interact before concatenation.

We further prevent context tokens from attending to temporal-query tokens, so that previous value predictions cannot be propagated indirectly through later language or visual representations. As a result, each temporal estimate must be grounded in the task instruction and its available visual context, while natural-language analysis and verification outputs are generated independently of the predicted values.

Natural-language video analysis and task verification. RynnValue additionally retains the autoregressive language capability of RynnBrain for interpretable task analysis. After processing the complete visual sequence, the verification prompt pvel $\scriptstyle \mathbf { p } _ { \mathrm { v e r } }$ first instructs the model to generate a natural-language description of the video and its task-relevant events. The model then generates task-matching and task-success judgments:

$$
\mathbf { y } _ { \mathrm { l a n g } } = \left[ \mathbf { V } \mathbf { i d e o } \ \mathsf { D e s c r i p t i o n } \colon \mathbf { y } ^ { \mathrm { v i d } } , \mathtt { M a t c h } \colon \mathbf { y } ^ { \mathrm { m a t c h } } , \mathtt { S u c c e s s } \colon \mathbf { y } ^ { \mathrm { s u c c } } \right] .\tag{5}
$$

This autoregressive ordering encourages the model to identify the robot behavior, object interactions, taskrelevant events, and completion evidence before producing the final matching and success judgments.

The natural-language outputs are generated from the task instruction and complete visual context through the original LM head. They are auxiliary outputs produced after the temporal-query sequence and are not fed back into the absolute or relative temporal heads, so the LM head does not participate in temporal-value prediction. RynnValue therefore supports interpretable video analysis, task matching, completion recognition, and failure detection without introducing separate classification heads.

Table 1 Composition of the heterogeneous data mixture before subtask expansion. The corpus contains 1.67M original episodes and is further converted into over 3M instruction-conditioned trajectory segments after subtask segmentation and cutof relabeling.
<table><tr><td>Data Source</td><td># Original Episodes</td><td># Segmentations</td><td># Instructions</td><td>Segmentation Source</td></tr><tr><td>AgiBot [3]</td><td>167,535</td><td>1,166,042</td><td>3,741</td><td>coarse task</td></tr><tr><td>EgoDex [10]</td><td>338,234</td><td>338,234</td><td>2,038</td><td>full trajectory</td></tr><tr><td>Galaxea Open-World [13]</td><td>16,979</td><td>95,671</td><td>11,070</td><td>coarse task</td></tr><tr><td>InternData-A1 [27]</td><td>320,905</td><td>320,905</td><td>348</td><td>full trajectory</td></tr><tr><td>Open X-Embodiment [23]</td><td>693,037</td><td>693,037</td><td>180,090</td><td>full trajectory</td></tr><tr><td>RDT [17]</td><td>6,109</td><td>6,109</td><td>272</td><td>per-file coarse task</td></tr><tr><td>RoboCOIN [30]</td><td>67,420</td><td>410,877</td><td>2,124</td><td>coarse task</td></tr><tr><td>RoboMIND [29]</td><td>32,138</td><td>32,138</td><td>184</td><td>full trajectory</td></tr><tr><td>RoboTwin [20]</td><td>27,414</td><td>27,414</td><td>23,527</td><td>full trajectory</td></tr><tr><td>Soft-FOLD [35]</td><td>1,542</td><td>1,542</td><td>1</td><td>per-file coarse task</td></tr><tr><td>Total</td><td>1,671,313</td><td>3,091,969</td><td>223,395</td><td></td></tr></table>

## 3 Learning Temporal Distance as a Reward Interface

This section describes how RynnValue learns temporal distance from heterogeneous data and exposes it as a reward interface. Our approach has three components: a heterogeneous data mixture and temporal-distance relabeling that turn raw trajectories into supervision (Section 3.1), a distributional training recipe (Section 3.2), and a reward interface for downstream policy learning (Section 3.3). We describe each in turn.

## 3.1 Data Preparation

RynnValue is trained from large-scale heterogeneous data. The goal is to convert trajectories with diferent task boundaries, embodiments, viewpoints, and execution speeds into a unified state-level temporal-distance supervision signal.

## 3.1.1 Heterogeneous Data Mixture

To scale temporal-distance learning beyond curated task-specific demonstrations, we train RynnValue on a large heterogeneous mixture of robot, simulation, and egocentric data [3, 10, 13, 23]. The mixture exceeds 7,000 hours of trajectories and spans a broad range of manipulation settings, including single-arm platforms, dual-arm mobile manipulators, bimanual tabletop robots, dexterous-hand systems, simulated embodiments, and first-person human demonstrations. It also covers diverse camera configurations. This diversity exposes the model to large variations in morphology, viewpoint, control frequency, execution speed, task duration, object distribution, and language annotation granularity.

Before subtask expansion, the corpus contains 1.67M original episodes. After subtask segmentation and cutof-based relabeling, these data are further converted into over 3M instruction-conditioned trajectory segments, each associated with a language goal and a temporally localized completion target. Crucially, these sources are not unified by manually defining dataset-specific progress scales. Instead, they are mapped onto the same temporal-distance supervision interface: each trajectory segment provides timestamp-derived state-level labels measuring the remaining temporal distance to its completion cutof.

## 3.1.2 Data Preprocessing and Temporal-Distance Relabeling

Large-scale robot datasets difer substantially in how episodes and subtasks are defined [3, 23, 4]. We therefore convert each data source into instruction-conditioned trajectory segments. Long demonstrations are split using native temporal annotations when available; otherwise, the complete episode is retained as a coarse segment. Because recorded trajectories may contain post-completion motions, we further assign each segment a completion cutof. The segment endpoint is used by default, while dataset-specific ratio- or duration-based trimming is applied when necessary to better approximate the first semantically completed observation [16].

![](images/c229fe1f2150b476cdd4792a6ba5b446c480f6a707d5e1b3ca7c1f35083a8361.jpg)

![](images/3c80587b5198af53e34c838b36a7d56d7942db94b9845f90659c6f2710b782dd.jpg)  
(a) RynnValue Training Strategy  
(b) Value Isolation Attention  
Figure 2 RynnValue training pipeline and value-isolation attention. (a) Training strategy. Random temporal sampling and temporal-order shufling suppress shortcuts tied to sampling intervals and sequence position, while instructionmismatch augmentation strengthens language–visual grounding. RynnValue jointly learns absolute temporal distance, relative temporal displacement, and natural-language supervision. (b) Value-isolation attention. Within each valuequery group, repeated queries attend to one another and to the language–visual context, while remaining isolated from other value-query groups. Colored cells denote visible attention connections.

Temporal-distance labels are then generated directly from timestamps: observations before the cutof are labeled by their remaining time to completion, while observations at or after the cutof receive zero. This provides unified, dense supervision without dataset-specific progress normalization. Separately, we use Qwen3- VL-27B [1] to generate a segment-level description of the observed behavior and task-relevant events. These captions supervise the natural-language video-analysis output (i.e., Video Description in Equation 5) and do not afect the temporal targets.

## 3.2 Training Recipe

## 3.2.1 Multi-Frame Sampling Strategy

Scaling data alone does not guarantee reliable temporal-value learning, because a multi-frame model can exploit regularities in the input rather than the visual evidence. We therefore combine two complementary sampling components: random temporal sampling removes regularities in the temporal intervals, while temporal-order shufling breaks the correspondence between sequence position and task progress.

Random temporal sampling. For each instruction-conditioned trajectory clip, we randomly sample K = 8 observations at irregular timestamps. The resulting non-uniform temporal gaps break the near-arithmetic value patterns induced by uniform sampling and discourage the model from exploiting fixed sampling intervals as a shortcut.

Temporal-ordershuffling. As illustrated in Figure 2(a), we shufle the chronological order of sampled observations. Half of the training sequences are independently sampled without temporal sorting, while the remainder follow a forward-biased temporal walk with occasional backward transitions, whose per-step probability we refer to as the rewind probability. Relative temporal targets are computed between adjacent observations in the presented order and can therefore be either positive or negative.

Temporal-order shufling and value-isolation attention suppress complementary shortcuts. Shufling prevents the model from regressing a stereotypical value curve from sequence positions, while value-isolation attention prevents extrapolation from other value-query representations. Together, they require each prediction to be grounded in the corresponding visual observation and language-conditioned task semantics.

## 3.2.2 Instruction-Mismatch Augmentation

A value model should also recognize when the language instruction does not describe the observed video, rather than always reporting smooth progress toward the stated goal. As shown in Figure 2, we therefore introduce instruction-mismatch augmentation. For 10% of the training samples, we replace the original instruction with one sampled from a diferent trajectory and supervise the language branch to predict Match: No and Success: No. Because the original completion cutof is no longer valid under the substituted instruction, we mask the absolute temporal-distance loss (Equation 6) to avoid introducing incorrect supervision. The relative temporal loss (Equation 7) is retained, as the relative target measures the temporal displacement between observations and is independent of the task instruction. This augmentation teaches RynnValue to detect instruction-video mismatches and remain reliable under substantial language-visual inconsistency.

## 3.2.3 Joint Training Objectives

RynnValue is jointly optimized using three cross-entropy objectives: an absolute temporal-distance loss, a relative temporal-distance loss, and a causal language-modeling loss. Although all three objectives use cross-entropy, the first two operate over distributional temporal bins, whereas the third operates over the language vocabulary.

Temporal targets. Let $t _ { G }$ denote the relabeled completion cutof and $t _ { i }$ the timestamp of observation $I _ { t _ { i } }$ . We define its absolute temporal-distance target as $v _ { i } ^ { \star } = \operatorname* { m a x } ( 0 , t _ { G } - t _ { i } )$ , and the relative target between two consecutively presented observations as $\Delta _ { i } ^ { \star } = t _ { i + 1 } - t _ { i }$ . Accordingly, $\Delta _ { i } ^ { \star } > 0$ indicates forward temporal progress, whereas $\Delta _ { i } ^ { \star } < 0$ indicates temporal regression. Here, “consecutive” refers to adjacency in the presented multimodal sequence, not in the original video.

Both continuous targets are transformed with symlog and encoded as two-hot distributions over 256-bin supports, denoted $B _ { V }$ and $\scriptstyle { B _ { R } }$ for the absolute and relative supports, respectively.

Absolute temporal-distance loss. The absolute temporal-distance objective is

$$
\mathcal { L } _ { \mathrm { a b s } } = - \frac { \omega } { K } \sum _ { i = 1 } ^ { K } \sum _ { b = 1 } ^ { | \mathcal { B } _ { V } | } \left[ \mathrm { T w o H o t } _ { { \mathcal { B } _ { V } } } ( v _ { i } ^ { \star } ) \right] _ { b } \log \left[ \mathrm { s o f t m a x } ( \mathbf { z } _ { i } ^ { V } ) \right] _ { b } ,\tag{6}
$$

where $\mathbf { z } _ { i } ^ { V }$ denotes the logits produced by the absolute-value head, and the decoded continuous prediction is $v _ { i }$ . The sample-level mask ω is set to 1 for instruction-matched examples and 0 for instruction-mismatched examples, for which the original completion cutof is no longer valid.

Relative temporal-distance loss. The relative temporal-distance objective is

$$
\mathcal { L } _ { \mathrm { r e l } } = - \frac { 1 } { K - 1 } \sum _ { i = 1 } ^ { K - 1 } \sum _ { b = 1 } ^ { | \mathcal { B } _ { R } | } \left[ \mathrm { T w o H o t } _ { \mathcal { B } _ { R } } ( \boldsymbol { \Delta } _ { i } ^ { \star } ) \right] _ { b } \log \left[ \mathrm { s o f t m a x } ( \mathbf { z } _ { i } ^ { R } ) \right] _ { b } ,\tag{7}
$$

where $\mathbf { z } _ { i } ^ { R }$ denotes the logits produced by the relative-value head, and the corresponding decoded prediction is denoted by $\Delta _ { i }$ . The absolute objective anchors each observation to task completion, whereas the relative objective captures forward and backward temporal displacement between consecutively presented observations.

Natural-language loss. For natural-language supervision, RynnValue autoregressively predicts the structured output defined in Equation 5 and is optimized using standard token-level cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { l a n g } } = - \frac { 1 } { | \mathcal { V } | } \sum _ { j \in \mathcal { V } } \log p _ { \theta } \left( y _ { j } \mid \mathbf { x } , y _ { < j } \right) ,\tag{8}
$$

where $\mathcal { V }$ denotes the supervised language-token positions. Success prediction is therefore learned through the natural-language objective, without introducing a separate success-classification head.

Joint objective. The final training objective combines the three cross-entropy losses:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { a b s } } + \mathcal { L } _ { \mathrm { r e l } } + \lambda _ { \mathrm { l a n g } } \mathcal { L } _ { \mathrm { l a n g } } , \qquad \lambda _ { \mathrm { l a n g } } = 2 .\tag{9}
$$

The absolute and relative losses update their corresponding distributional heads and the visual-language backbone. The LM output projection is kept frozen, while gradients from $\mathcal { L } _ { \mathrm { l a n g } }$ still propagate through the fixed projection to adapt the backbone representations. No stop-gradient operation is applied between either temporal head and the backbone.

## 3.3 Inference and Reward Interface

Chronological inference. At inference time, the training-time sampling augmentations are disabled and the input observations are arranged in chronological order. The value-isolation attention mask remains active: each absolute or relative query group can attend to its associated visual-language context, while predictions from other query groups remain inaccessible. This prevents previously predicted values from influencing subsequent predictions during inference. RynnValue then decodes the absolute and relative distributional outputs into continuous temporal-distance estimates. The language branch subsequently generates the video description, instruction-video matching judgment, and success judgment, none of which are fed back into the temporal-value predictions.

Temporal-distance potential. RynnValue predicts temporal distance rather than reward directly. Let $v _ { t }$ denote the final temporal-distance estimate for observation $I _ { t }$ , after distributional decoding. The original prediction remains non-negative: a large $v _ { t }$ indicates that more time is required to complete the task, whereas $v _ { t } = 0$ corresponds to the predicted completion boundary. To match the conventional value semantics that larger is better, we convert temporal distance into an observation potential by reversing its sign:

$$
\Phi _ { t } = \Phi _ { \theta } ( I _ { t } , \ell , m ) = - v _ { t } ,\tag{10}
$$

where ℓ and m denote the task instruction and metadata, respectively. Consequently, states before task completion have negative potential, while the potential approaches zero as the agent reaches the goal. This sign reversal preserves the temporal scale rather than normalizing predictions to a task-specific [0, 1] interval

## 4 Experiments

We evaluate RynnValue along four axes. First, we benchmark its intrinsic value quality against standardized reward-model evaluation suites (Section 4.2). Second, we analyze how its accuracy scales with data quantity and task diversity (Section 4.3). Third, we qualitatively compare its temporal-value predictions with a strong baseline on held-out trajectories, examining whether the learned value tracks task progress and reacts properly to regressions and failures (Section 4.4). Finally, we deploy RynnValue as a reward interface in real-world reinforcement learning to assess its utility under physical execution noise and embodiment shift (Section 4.5).

## 4.1 Experimental Setup

Model configuration. In Table 2 we additionally report a RynnValue-4B variant to probe how performance scales with backbone size; all other experiments use the 8B model. Each training sample contains $K = 8$ observations, with eight repeated query tokens for each absolute or relative prediction slot. Both temporal heads are implemented as BroNet [21] residual MLPs with hidden width 4096, depth 8, and ReLU activations. The absolute and relative temporal targets are represented using 256-bin symlog supports over [0, 512] and [−256, 256], respectively. We use value-isolation attention throughout both training and inference.

Optimization. We jointly optimize the absolute temporal-distance, relative temporal-distance, and naturallanguage objectives as shown in Equation 9. RynnValue is trained with AdamW using a learning rate of $1 \times 1 0 ^ { - 6 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , weight decay 0.1, and $\epsilon = 1 0 ^ { - 8 }$ . We use a constant learning-rate schedule without warm-up and clip the gradient norm at 100. Training is conducted in bfloat16 with FSDP hybrid sharding and a per-device batch size of 2.

During training, temporal-order shufling is applied with probability 0.5, and the forward-biased sampling process uses a rewind probability of 0.3. Instruction-mismatch augmentation is applied to 10% of the samples. For mismatched samples, the absolute temporal-distance loss is masked, while the relative and natural-language objectives are retained.

Table 2 Per-dataset trajectory-ranking results on the RBM-EVAL-OOD test suite, measured by Kendall’s $\tau _ { a } \ ( \uparrow )$ . Bold values indicate the best overall results. † denotes the best result among methods trained without explicit trajectory-level preference supervision, i.e., progress/value-only methods. Baseline results are taken from Robometer [16].
<table><tr><td>Method</td><td>USC Franka</td><td>USC Koch</td><td>USC Trossen</td><td>USC xArm</td><td>MIT Franka</td><td>UTD SO101</td><td>Average</td></tr><tr><td>GVL [19]</td><td>0.250</td><td>-0.008</td><td>0.292</td><td>0.056</td><td>0.306</td><td>0.300</td><td>0.199</td></tr><tr><td>VLAC-2B [34]</td><td>0.292</td><td>0.167</td><td>-0.111</td><td>0.167</td><td>-0.017</td><td>-0.033</td><td>0.077</td></tr><tr><td>VLAC-8B [34]</td><td>0.271</td><td>0.064</td><td>-0.417</td><td>0.139</td><td>0.072</td><td>0.167</td><td>0.049</td></tr><tr><td>RoboDopamine [26]</td><td>0.167</td><td>0.175</td><td>0.000</td><td>0.014</td><td>0.220</td><td>0.067</td><td>0.107</td></tr><tr><td>Dopamine-GRM-2.0-8B-Preview [26]</td><td>0.479</td><td>0.442</td><td>0.333</td><td>0.431</td><td>0.431</td><td>0.700</td><td>0.453</td></tr><tr><td>RoboReward-4B [15]</td><td>0.625</td><td>0.332</td><td>0.333</td><td>0.528</td><td>0.494</td><td>0.700</td><td>0.502</td></tr><tr><td>RoboReward-8B [15]</td><td>0.625</td><td>0.264</td><td>0.389</td><td>0.347</td><td>0.396</td><td>0.767</td><td>0.465</td></tr><tr><td>Robometer (RoboReward data) [16]</td><td>0.583</td><td>0.533</td><td>0.646</td><td>0.403</td><td>0.479</td><td>0.667</td><td>0.552</td></tr><tr><td>ReWiND [33]</td><td>-0.125</td><td>0.336</td><td>0.028</td><td>-0.167</td><td>0.080</td><td>-0.067</td><td>0.014</td></tr><tr><td>Robometer (RBM-1M) [16]</td><td>0.646</td><td>0.471</td><td>0.653</td><td>0.694</td><td>0.601</td><td>0.867</td><td>0.655</td></tr><tr><td>Robometer (Progress only) [16]</td><td>0.083</td><td>0.231</td><td>0.333</td><td>0.389</td><td>0.183</td><td>0.533</td><td>0.292</td></tr><tr><td>RynnValue-4B</td><td>0.542</td><td>0.488</td><td>0.917</td><td>0.667†</td><td>0.473</td><td>0.933†</td><td>0.670</td></tr><tr><td>RynnValue-8B</td><td>0.667†</td><td>0.544†</td><td>1.000†</td><td>0.500</td><td>0.503†</td><td>0.833</td><td>0.675†</td></tr></table>

## 4.2 Benchmark Evaluation

We evaluate RynnValue on the trajectory-ranking track of the RBM-EVAL-OOD test suite introduced by Robometer [16]. The benchmark contains 976 trajectories from six out-of-distribution datasets spanning diferent institutions, robot embodiments, camera viewpoints, and task families. For each task, the benchmark provides trajectories annotated with diferent execution-quality levels, including failed, suboptimal, and successful executions. Performance is measured by Kendall’s $\tau _ { a }$ between the ground-truth trajectory-quality ordering and the ordering induced by the predicted scores.

Because RynnValue predicts temporal distance rather than normalized progress, we score each trajectory by its temporal-distance potential (Equation 10) at the final queried observation, i.e., the negative predicted remaining distance $- v _ { \mathrm { e n d } }$ , so that a trajectory closer to task completion receives a higher score. No additional normalization or cross-dataset calibration is required because Kendall’s $\tau _ { a }$ depends only on the relative ordering of these scores. We use the absolute temporal-distance output for this evaluation; the relative temporal prediction and natural-language outputs are not used to compute the benchmark metric.

## 4.2.1 Main Results

As shown in Table 2, RynnValue-8B achieves the highest average Kendall’s $\tau _ { a }$ of 0.675, and even the smaller RynnValue-4B reaches 0.670; both exceed the 0.655 obtained by the full Robometer model trained with both progress and trajectory-preference supervision. In terms of the best overall results, RynnValue-8B ranks first on USC Franka, USC Koch, and USC Trossen, while RynnValue-4B attains the top score on UTD SO101. Among methods trained without explicit trajectory-level preference supervision, RynnValue variants attain the best results on all six datasets, improving the strongest preference-free prior average of 0.502 to 0.670 at 4B and 0.675 at 8B. RynnValue also substantially outperforms the progress-only Robometer ablation, which achieves an average score of 0.292. Notably, performance is already strong at 4B and improves only marginally at 8B, suggesting that the gains stem from the temporal-distance formulation and its accompanying training and architectural designs rather than sheer model scale.

These results support temporal distance as an efective supervision target for scaling robotic value models. Unlike explicitly constructed trajectory preferences, temporal-distance labels can be obtained automatically from heterogeneous trajectories while preserving a consistent cost-to-go interpretation across tasks, execution speeds, and robot embodiments. Together with random temporal sampling, order shufling, and value-isolation attention, this formulation encourages the model to ground its predictions in task-relevant visual evidence rather than dataset-specific progress scales or temporal shortcuts. Most importantly, RynnValue reaches this quality through a substantially simpler recipe: its targets are read directly from trajectory timestamps, without constructing preference pairs or normalizing every task onto a [0, 1] progress scale. That a preference-free, timestamp-derived target can match and even surpass fully preference-supervised training highlights temporal distance as a more scalable route to general-purpose robotic value learning.

![](images/ad3db54b59612e5f49579012330f5195f7b0a442a4cc68579efeecb3c76c44a7.jpg)  
-0.53

![](images/8d12111d17e1c602c14e27b3700253f221996530184ccab7165238b54bf231bf.jpg)  
0.18

![](images/57af50ed1394577fc7374cd2a5cf3295e6a01efa79659d14927be28f5bb13f9e.jpg)  
0.60

![](images/5192591be53704a4ae8cfa8d3583a2e86d24e583c52949faa77f2700917b2bb3.jpg)  
0.67

![](images/1cb7037469a963df5a3e41327d0d9ab7715a5fe7cd050ea185c404312c0ce5e9.jpg)  
0.59

![](images/49ff961fc6d7426f03c4029a0fbcced8050a48e4b94af71d56836b4f5a129196.jpg)  
0.79  
Figure 3 Instruction-trajectory confusion matrices. Each cell shows the predicted reward when an instruction (rows) is paired with a trajectory (columns); a well-grounded model concentrates mass on the diagonal. Values below each matrix report the normalized diagonal margin. All models are re-evaluated under a unified protocol from their publicly released weights.

Table 3 Ablation study on RBM-EVAL-OOD. We report Kendall’s $\tau _ { a }$ across six out-of-distribution robot datasets. Shufle denotes temporal-order shufling, Isolation denotes value-isolation attention, Language denotes the auxiliary natural-language supervision, Random denotes random temporal sampling, and Relative denotes the relative modeling component.
<table><tr><td></td><td colspan="5">Design Components</td><td colspan="7">Kendall&#x27;s  $\tau _ { a }$ </td></tr><tr><td>Variant</td><td>Shuffle</td><td>Isolation</td><td>Language</td><td>Random Relative</td><td></td><td>USC Franka</td><td>USC Koch</td><td>USC Trossen</td><td>USC xArm</td><td>MIT Franka</td><td>UTD SO101</td><td>Average</td></tr><tr><td>w/o Shuffle</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.583</td><td>0.090</td><td>0.055</td><td>0.222</td><td>-0.017</td><td>0.200</td><td>0.189</td></tr><tr><td>w/o Isolation</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>0.583</td><td>0.428</td><td>0.694</td><td>0.389</td><td>0.400</td><td>0.400</td><td>0.482</td></tr><tr><td>w/o Language</td><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>0.250</td><td>0.491</td><td>0.819</td><td>0.361</td><td>0.501</td><td>0.800</td><td>0.537</td></tr><tr><td>Uniform Sampling</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>0.375</td><td>0.400</td><td>0.305</td><td>0.250</td><td>0.310</td><td>0.633</td><td>0.379</td></tr><tr><td>w/o Relative</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>0.667</td><td>0.587</td><td>0.639</td><td>0.639</td><td>0.464</td><td>0.767</td><td>0.627</td></tr><tr><td>Full Model (8B)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.667</td><td>0.544</td><td>1.000</td><td>0.500</td><td>0.503</td><td>0.833</td><td>0.675</td></tr></table>

## 4.2.2 Instruction-Trajectory Alignment

To assess whether the predicted reward tracks the specified language goal rather than generic visual progress, we perform an instruction-trajectory alignment analysis: we score every instruction against every trajectory and inspect the resulting matrix, following the evaluation protocol of Robometer [16]. A well-grounded model should assign higher rewards to matched instruction-trajectory pairs along the diagonal and lower rewards to mismatched pairs. As shown in Figure 3, to ensure that all confusion matrices are produced under an identical protocol, we re-evaluate every baseline with its publicly released model weights, and report the resulting diagonal-margin scores. RynnValue produces the clearest diagonal structure and achieves the highest normalized diagonal margin of 0.79, outperforming the strongest baseline at 0.67. In contrast, several baselines exhibit more difuse of-diagonal responses. These results show that RynnValue more efectively distinguishes matched from mismatched instruction-trajectory pairs, grounding its temporal-distance estimates in the language-specified goal rather than generic visual progress alone.

## 4.2.3 Ablation Study

We ablate the principal training and architectural components of RynnValue in Table 3. The full model achieves the highest average Kendall’s $\tau _ { a }$ of 0.675. Replacing random temporal sampling with uniform sampling reduces the average score to 0.379. The regular intervals it produces let the model exploit a stereotypical value pattern rather than reason from the observed visual content. Similarly, removing temporal-order shufling causes the largest degradation, reducing the average score to 0.189, as sequence position again becomes a strong proxy for task progress.

Beyond sampling, removing value-isolation attention decreases performance to 0.482. Taken together, these results validate our complementary shortcut-suppression designs: shufling prevents progress inference from presentation order, while value isolation prevents prediction queries from extrapolating values across query groups. We further examine the auxiliary training objectives. Removing natural-language supervision reduces the average score to 0.537, demonstrating that video-description, instruction-matching, and success supervision provide useful semantic grounding. Finally, removing relative temporal-distance supervision yields a score of 0.627. By modeling local forward and backward temporal changes, this objective complements the globally anchored absolute prediction and improves the shared visual representation. This benefit is particularly evident on USC Trossen, where its removal reduces the score from 1.000 to 0.639.

## 4.3 Scaling Analysis

Having isolated the contribution of each model component, we next ask whether the heterogeneous data recipe itself drives generalization. To disentangle quantity from diversity, we construct two families of training subsets from the full mixture. In the episode-volume family (orange), we fix the full set of training tasks and randomly subsample {1%, 10%, 25%, 50%, 75%} of episodes within each task. In the task-diversity family (blue), we fix per-task episode counts at the full-scale level and randomly subsample the same fractions of tasks. At each fraction, both protocols train on a comparable total number of episodes, ensuring that performance diferences reflect diversity rather than data volume. Both families converge to the same full training set at 100%. We train each variant from scratch under identical optimization settings and report the mean absolute temporal-distance error on a held-out validation set whose tasks do not overlap with any training subset.

![](images/c0385188a41f501f46c72be3b1e29556aa510b49db1b7b1b0f0b7b6ac637ff7a.jpg)  
Figure 4 Scaling episode volume vs. task diversity. Mean absolute temporal-distance error on a held-out validation set of unseen tasks. We independently scale two aspects of the training set: (orange) episode count with the full task set fixed, and (blue) task count with per-task episode counts fixed. Both curves converge to the same full-scale training set at 100%. Task diversity yields a consistently steeper error reduction than episode volume, which saturates early.

As shown in Figure 4, episode-volume scaling saturates almost immediately: beyond a small fraction of episodes the error plateaus and does not improve further as more within-task data are added. Task-diversity scaling behaves qualitatively diferently: increasing the number of training tasks reduces error monotonically across the entire range, with substantial gains still observable well past the midpoint. This contrast demonstrates that the heterogeneous data recipe contributes not merely additional samples, but the diversity required for general-purpose temporal-value learning: broader task coverage introduces diverse goal structures, visual configurations, and execution patterns that transfer to unseen tasks, whereas repeated observation of the same tasks quickly stops adding signal.

## 4.4 Temporal-Value Curve Case Study

Figure 5 compares the value curves predicted by RynnValue and Robometer along the same real-world trajectory. Because Robometer outputs normalized progress while RynnValue outputs temporal-distance potential $\left( \Phi _ { t } = - v _ { t } \right)$ , the two scales are not directly comparable; we therefore orient both so that higher values indicate closer proximity to task completion. The visualization reveals three notable diferences.

First, RynnValue responds more strongly to trial-and-error behavior and task regression. During the highlighted interval, its potential decreases substantially as the robot moves away from a productive state, whereas Robometer shows a weaker response. This sensitivity is consistent with temporal-order shufling and valueisolation attention, which encourage predictions to rely on visual evidence rather than sequence position or previously exposed values.

Second, after the robot recovers, RynnValue exhibits a steadier increase toward completion, while Robometer contains longer plateaus and abrupt jumps. This behavior benefits from jointly learning absolute and relative temporal values, which capture global remaining cost and local temporal changes, respectively.

Third, RynnValue remains sensitive near task completion: late disturbances cause an immediate potential decrease followed by recovery. This avoids premature reward saturation and better distinguishes intermediate progress from actual completion.

![](images/46215ab9e827d55edbb4486f6f5a10ba731d3b4f2724fed9561a557dd5a560c0.jpg)  
Figure 5 Temporal-value curve comparison on a real-world trajectory. Higher values indicate closer proximity to task completion. The highlighted interval marks a period of task regression where the robot moves away from a productive state; RynnValue responds with a sharp potential drop, whereas Robometer remains relatively flat.

![](images/c588bc72c3a47790967f26fd761d7e3a2ece0bd830b77f9e0bab9bb93a739a0c.jpg)  
Figure 6 Representative demonstrations for real-world evaluation. Each row shows a sequence of observations from one manipulation task: Each row shows a sequence of observations from one manipulation task. These tasks cover object grasping, spatial manipulation, and articulated-object interaction.

## 4.5 Real-World Policy Learning

Beyond benchmarks, we evaluate whether RynnValue can improve policy learning on a diverse set of challenging real-world manipulation tasks. Since none of these tasks, their manipulated objects, or the workspace scenes appear in the reward-model training corpus, and no target-domain fine-tuning is performed, RynnValue serves as a zero-shot reward annotator. This setting assesses its open-world generalization and its efectiveness for value estimation under physical execution noise.

## 4.5.1 Experimental Settings

Robot platform. Our experiments are conducted on a dual-arm Franka robot system with wrist-mounted cameras on both arms and two additional third-person cameras capturing the workspace from the left and right sides.

Tasks and evaluation protocol. We evaluate on the following four real-world robotic manipulation tasks as shown in Figure 6:

Table 4 Real-world reinforcement-learning results. We report success rates and the average number of action chunks over successful episodes. Average denotes the unweighted mean success rate across four tasks.
<table><tr><td rowspan="2">Algorithm</td><td rowspan="2">Baseline</td><td colspan="2">Bread Basket Placement</td><td colspan="2">Steak Serving with a Spatula</td><td colspan="2">Box-in-Drawer Placement Bimanual Box Transfer Average</td><td colspan="2"></td><td rowspan="2">Success</td></tr><tr><td>Success ↑</td><td>Avg. Steps ↓</td><td>Success ↑</td><td>Avg. Steps ↓</td><td>Success ↑</td><td>Avg. Steps ↓</td><td>Success ↑</td><td>Avg. Steps ↓</td></tr><tr><td rowspan="3"></td><td>RynnValue</td><td>45.0%</td><td> $2 5 . 9 \pm 8 . 2$ </td><td>75.0%</td><td> $1 8 . 6 \pm 1 3 . 1$ </td><td>70.0%</td><td> ${ \bf 2 7 . 0 \pm 8 . 1 }$ </td><td>100.0%</td><td> ${ \bf 2 2 . 8 \pm 4 . 7 }$ </td><td>72.5%</td></tr><tr><td>Online RL Robometer</td><td>35.0%</td><td> ${ \bf 2 2 . 7 \pm 5 . 5 }$ </td><td>45.0%</td><td> ${ \bf 1 5 . 2 \pm 2 . 7 }$ </td><td>65.0%</td><td> $2 7 . 7 \pm 5 . 8$ </td><td>65.0%</td><td> $2 5 . 6 \pm 7 . 4$ </td><td>52.5%</td></tr><tr><td>Sparse</td><td>40.0%</td><td> $5 6 . 0 \pm 3 1 . 7$ </td><td>45.0%</td><td> $1 8 . 4 \pm 4 . 9$ </td><td>40.0%</td><td> $2 7 . 4 \pm 6 . 6$ </td><td>70.0%</td><td> $2 3 . 5 \pm 2 . 7$ </td><td>48.8%</td></tr><tr><td rowspan="3"></td><td>RynnValue</td><td>100.0%</td><td> ${ \bf 1 6 . 8 \pm 3 . 1 }$ </td><td>90.0%</td><td> ${ \bf 1 4 . 9 \pm 4 . 0 }$ </td><td>90.0%</td><td> ${ \bf 1 4 . 9 \pm 4 . 0 }$ </td><td>50.0%</td><td> $3 3 . 6 \pm 1 0 . 5$ </td><td>82.5%</td></tr><tr><td>Offline RL Robometer</td><td>80.0%</td><td> $1 8 . 9 \pm 2 . 7$ </td><td>80.0%</td><td> $1 9 . 4 \pm 7 . 2$ </td><td>50.0%</td><td> $2 7 . 3 \pm 6 . 3$ </td><td>45.0%</td><td> ${ \bf 2 8 . 7 \pm 9 . 3 }$ </td><td>63.8%</td></tr><tr><td>Sparse</td><td>70.0%</td><td> $2 6 . 1 \pm 9 . 1$ </td><td>20.0%</td><td> $3 0 . 2 \pm { 3 . 3 }$ </td><td>0.0%</td><td></td><td>0.0%</td><td></td><td>22.5%</td></tr><tr><td>SFT</td><td></td><td>70.0%</td><td> $2 4 . 8 \pm 8 . 0$ </td><td>25.0%</td><td> $1 8 . 6 \pm 6 . 2$ </td><td>0.0%</td><td></td><td>0.0%</td><td></td><td>23.8%</td></tr></table>

• Bread Basket Placement: The robot places two pieces of bread into a basket.

• Steak Serving with a Spatula: The robot uses a spatula to transfer a steak from a pan to a plate.

• Box-in-Drawer Placement: The robot places a box into a drawer and then closes the drawer.

• Bimanual Box Transfer: Two robot arms collaboratively move a box to a target location.

For each task, we evaluate each policy over 20 trials and report both the success rate and the average number of action chunks over successful episodes. All ofline evaluations use fixed initial configurations. During online evaluation, we randomize the initial object configurations for Bread Basket Placement, Steak Serving with a Spatula, and Bimanual Box Transfer to assess robustness under varying initial conditions, while Box-in-Drawer Placement uses a fixed reset because the task requires precise gripper control and box-drawer alignment.

Baselines and reward shaping. We compare RynnValue against Robometer, the strongest baseline in our benchmark evaluations, and Sparse Reward, which assigns −1 before task completion and 0 upon successful completion. To ensure a fair comparison, all reward models are used through the same potential-based reward shaping interface:

$$
r _ { t } ^ { \prime } = \kappa { \left( \gamma \Phi _ { t + 1 } - \Phi _ { t } \right) } + \left\{ { 0 , \mathrm { i f } \ t = T \mathrm { \ a n d \ t h e \ t r a j e c t o r y \ i s \ s u c c e s s f u l { , } } } \right.\tag{11}
$$

where $\Phi _ { t }$ denotes the potential predicted by the reward model at step t, γ is the discount factor, and κ controls the strength of the shaping term. We use $\kappa = 0 . 1$ for RynnValue and $\kappa = 1 . 0$ for Robometer in both the ofline and online experiments. We retain the sparse completion term because reward-model predictions may be noisy, whereas the success label provides a clean and reliable signal for the final task objective. These success labels are manually annotated by human operators during both data collection and online policy training. For Sparse Reward, we set $\Phi _ { t } = 0$ for all $t ,$ so the reward is −1 before completion and 0 on the transition that successfully completes the task.

Offline RL with mixed-expertise datasets. First, we evaluate whether RynnValue can efectively recover highperforming policies from mixed-expertise datasets. Specifically, using $\pi _ { 0 . 5 }$ as the base policy, we apply Implicit Q-Learning (IQL) [14] to train policies on these datasets with reward signals provided by either RynnValue or the baselines.

Online RL with general-purpose reward models. Next, we investigate whether RynnValue can accelerate online policy learning by providing real-time reward specification. We adopt Difusion Steering via Reinforcement Learning (DSRL) [28] with task-specific policy initialization. Bread Basket Placement and Steak Serving with a Spatula start from their SFT checkpoints, whereas Box-in-Drawer Placement and Bimanual Box Transfer start from the corresponding Robometer-based ofline-RL checkpoints. This initialization is held fixed across the online reward variants, whose collected trajectories are specified by either RynnValue or the baselines.

## 4.5.2 Performance Analysis

As shown in Table 4, RynnValue consistently achieves the highest success rate across all four real-world tasks in both online and ofline RL. In online OOD evaluation, it reaches an average success rate of 72.5%, substantially outperforming Robometer at 52.5% and sparse rewards at 48.8%. The improvement is particularly pronounced on Steak Serving with a Spatula and Bimanual Box Transfer, where RynnValue improves upon Robometer by 30 and 35 percentage points, respectively. Under ofline RL, RynnValue achieves an average success rate of 82.5%, compared with 63.8% for Robometer and only 23.8% for the original SFT policy.

RynnValue also provides a strong balance between task success and execution eficiency. On Bread Basket Placement, it achieves 100% ofline success using 16.8 action chunks on average, compared with 80% and 18.9 chunks for Robometer and 70% and 24.8 chunks for SFT. Similar improvements are observed on Steak Serving with a Spatula and Box-in-Drawer Placement, where RynnValue reaches 90% success while requiring only 14.9 chunks. Notably, reinforcement learning with RynnValue successfully solves Box-in-Drawer Placement and Bimanual Box Transfer, for which the SFT policy records no successful executions. These results show that RynnValue enables substantial improvements over the imitation-learning baseline while maintaining eficient task completion.

The online gains are limited for both reward models on Box-in-Drawer Placement. Starting from the shared Robometer-trained ofline-RL checkpoint with a 50% success rate, online RL reaches 65% with Robometer and 70% with RynnValue. This task requires precise coordination among the gripper, the box, and the drawer geometry. Because the reward models observe only third-person RGB images, visually similar configurations may correspond to substantially diferent grasp stability and placement alignment, making it dificult to assign well-calibrated intermediate rewards. Consequently, neither reward model yields a substantial online improvement in this visually ambiguous, precision-sensitive setting, although RynnValue remains slightly more efective than Robometer.

## 5 Conclusion and Future Works

We presented RynnValue, a value foundation model for robotic manipulation that replaces trajectory-internal progress with temporal distance as its supervision target. By treating a state’s directed, goal-conditioned cost-to-go as the learning objective, RynnValue turns heterogeneous data into a single, preference-free value interface: labels are derived directly from timestamps, and disparate embodiments, viewpoints, and task durations are unified without dataset-specific progress normalization. To make temporal-value learning robust at scale, we combined random temporal sampling and temporal-order shufling with value-isolation attention, suppressing the sampling-interval, sequence-order, and value-extrapolation shortcuts that otherwise leave learned values insensitive to failures and regressions.

Trained without a single preference label, RynnValue surpasses the fully preference-supervised state of the art on RBM-EVAL-OOD and nearly doubles a progress-only counterpart, while generalizing zero-shot across unseen tasks, embodiments, and viewpoints. Converted into dense rewards through potential-based shaping, it improves both online and ofline real-world policy learning over strong reward-model and sparse-reward baselines. Together, these results indicate that temporal distance is both a scalable supervision target for value foundation models and a practical reward interface for generalist robot policies.

Looking forward, we aim to extend RynnValue toward broader temporal horizons, richer value semantics, and more diverse embodiments. RynnValue currently estimates temporal distance from a short window of sampled observations, so extending it to longer horizons and streaming inference would broaden its use as an online reward source. Its targets assume an approximately minimum-time objective, and incorporating task-specific costs such as energy, safety, or precision could yield richer value semantics. Finally, we plan to scale RynnValue to a wider range of end-efectors, including dexterous hands, as well as mobile manipulation settings, where long-horizon navigation and interaction must be jointly grounded in the same temporal-value interface.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[2] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y Galliker, et al. π<sub>0.5</sub>: A Vision-Language-Action Model with Open-World Generalization. In Proceedings of the Conference on Robot Learning, 2025.

[3] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Xindong He, Xu Huang, et al. AgiBot World Colosseum: A large-scale manipulation platform for scalable and intelligent embodied systems. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems, 2025.

[4] Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, Steven Palma, Pepijn Kooijmans, Michel Aractingi, Mustafa Shukor, Dana Aubakirova, Martino Russi, Francesco Capuano, Caroline Pascal, Jade Choghari, Jess Moss, and Thomas Wolf. LeRobot: State-of-the-art machine learning for real-world robotics in PyTorch. https://github.com/huggingface/lerobot, 2024.

[5] Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, et al. RynnBrain: Open embodied foundation models. arXiv preprint arXiv:2602.14979, 2026.

[6] Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. Unified language model pre-training for natural language understanding and generation. In Proceedings of the Advances in Neural Information Processing Systems, 2019.

[7] Jesse Farebrother, Jordi Orbay, Quan Vuong, Adrien Ali Taïga, Yevgen Chebotar, Ted Xiao, Alex Irpan, Sergey Levine, Pablo Samuel Castro, Aleksandra Faust, et al. Stop regressing: Training value functions via classification for scalable deep RL. In Proceedings of the International Conference on Machine Learning, pages 13049–13071, 2024.

[8] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

[9] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, pages 1–7, 2025.

[10] Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

[11] Bohan Hou, Haoqiang Lin, Xuemeng Song, Haokun Wen, Meng Liu, Yupeng Hu, and Xiangyu Zhao. FiRE: Enhancing MLLMs with fine-grained context learning for complex image retrieval. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 803–812, 2025.

[12] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. π<sup>∗</sup> : A VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[13] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and G0 dual-system VLA model. arXiv preprint arXiv:2509.00576, 2025.

[14] Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Ofline reinforcement learning with implicit Q-learning. In Proceedings of the International Conference on Learning Representations, 2022.

[15] Tony Lee, Andrew Wagenmaker, Karl Pertsch, Percy Liang, Sergey Levine, and Chelsea Finn. RoboReward: General-purpose vision-language reward models for robotics. arXiv preprint arXiv:2601.00675, 2026.

[16] Anthony Liang, Yigit Korkmaz, Jiahui Zhang, Minyoung Hwang, Abrar Anwar, Sidhant Kaushik, Aditya Shah, Alex S Huang, Luke Zettlemoyer, Dieter Fox, et al. Robometer: Scaling general-purpose robotic reward models via trajectory comparisons. arXiv preprint arXiv:2603.02115, 2026.

[17] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. RDT-1B: a difusion foundation model for bimanual manipulation. In Proceedings of the International Conference on Learning Representations, pages 29982–30009, 2025.

[18] Yuyang Liu, Chuan Wen, Yihang Hu, Dinesh Jayaraman, and Yang Gao. TimeRewarder: Learning dense reward from passive videos via frame-wise temporal distance. arXiv preprint arXiv:2509.26627, 2025.

[19] Yecheng Jason Ma, Joey Hejna, Ayzaan Wahid, Chuyuan Fu, Dhruv Shah, Jacky Liang, Zhuo Xu, Sean Kirmani, Peng Xu, Danny Driess, Ted Xiao, Jonathan Tompson, Osbert Bastani, Dinesh Jayaraman, Wenhao Yu, Tingnan Zhang, Dorsa Sadigh, and Fei Xia. Vision language models are in-context value learners. In Proceedings of the International Conference on Learning Representations, 2025.

[20] Yao Mu, Tianxing Chen, Zanxin Chen, Shijia Peng, Zhiqian Lan, Zeyu Gao, Zhixuan Liang, Qiaojun Yu, Yude Zou, Mingkun Xu, et al. RoboTwin: Dual-arm robot benchmark with generative digital twins. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27649–27660, 2025.

[21] Michal Nauman, Mateusz Ostaszewski, Krzysztof Jankowski, Piotr Mił oś, and Marek Cygan. Bigger, Regularized, Optimistic: scaling for compute and sample eficient continuous control. In Proceedings of the Advances in Neural Information Processing Systems, pages 113038–113071, 2024.

[22] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the International Conference on Machine Learning, pages 278–287, 1999.

[23] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, and et.al. Open X-Embodiment: Robotic learning datasets and RT-X models. In Proceedings of the IEEE International Conference on Robotics and Automation, pages 6892–6903, 2023.

[24] Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

[25] Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In Proceedings of the International Conference on Machine Learning, pages 1312–1320, 2015.

[26] Huajie Tan, Sixiang Chen, Yijie Xu, Zixiao Wang, Yuheng Ji, Cheng Chi, Yaoxu Lyu, Zhongxia Zhao, Xiansheng Chen, Peterson Co, et al. Robo-Dopamine: General process reward modeling for high-precision robotic manipulation. arXiv preprint arXiv:2512.23703, 2025.

[27] Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, et al. InternData-A1: Pioneering high-fidelity synthetic data for pre-training generalist policy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 976–985, 2026.

[28] Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering your difusion policy with latent space reinforcement learning. In Proceedings of the Conference on Robot Learning, 2025.

[29] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

[30] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025.

[31] Pengzhi Yang, Xinyu Wang, Pengyu Jing, Kehan Wen, Yiduo Qu, Zhenhao Huang, Minghao Fu, Xin Liu, Yaheng Shen, and Fan Shi. RARM: Confidence-gated progress reward modeling for RL in manipulation. arXiv preprint arXiv:2606.22027, 2026.

[32] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. XLNet: Generalized autoregressive pretraining for language understanding. In Proceedings of the Advances in Neural Information Processing Systems, 2019.

[33] Jiahui Zhang, Yusen Luo, Abrar Anwar, Sumedh Anand Sontakke, Joseph J Lim, Jesse Thomason, Erdem Biyik, and Jesse Zhang. ReWiND: Language-guided rewards teach robot policies without new demonstrations. In Proceedings of the Conference on Robot Learning, 2025.

[34] Qi Zhang, Shaopeng Zhai, Shengzhe Zhang, Litao Liu, Tianyi Zhang, Fuxian Huang, and Ming Zhou. A generalist pair-wise progress critic model for vision-language-action robots. In Proceedings of the International Conference on Machine Learning, 2026.

[35] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-languageaction model. arXiv preprint arXiv:2510.10274, 2025.

## Appendix

## A Data Curation for Heterogeneous Robot Corpora

Learning temporal distance requires more than a large collection of robot videos: each trajectory segment must be associated with a meaningful language goal and must describe task-relevant evolution toward that goal. Raw robot corpora do not always satisfy these conditions. Their annotations may contain placeholders, data-quality tags, truncated commands, or descriptions of pure robot motion. We therefore apply a sourceaware curation pipeline to Open X-Embodiment (OXE), InternData-A1 (InternA1), Galaxea Open-World (Galaxea), and RoboCOIN before constructing temporal-distance targets. Since OXE and InternA1 are counted in episodes, whereas Galaxea and RoboCOIN are counted in annotated subtask segments, we refer to their entries collectively as trajectory units.

Instruction validation. We first retain only instructions that specify a well-formed and actionable robot goal. Source-specific usable-instruction sets are used when canonical task vocabularies are available, while malformed strings, placeholders, data-quality metadata, and ambiguous noun-only labels are removed. We also discard CJK annotations from sources whose task interface is defined in English. This stage removes annotation artifacts without altering the semantics of valid manipulation tasks.

Action-relevance filtering. Some mobile-manipulation trajectories contain segments that describe only locomotion or approach behavior, without a manipulation objective. This issue is most prominent in Galaxea. We identify such segments when motion predicates such as moving, approaching, turning, or walking occur without any manipulation predicate, and remove them from temporal-distance training. This ensures that a retained segment describes observable progress toward an interaction goal rather than incidental platform motion.

Table 5 gives representative examples of the annotation issues addressed by the pipeline, spanning several qualitatively diferent failure modes: malformed language, dataset metadata, task-irrelevant motion.

Table 5 Representative annotation issues addressed by data curation. Non-English examples are described rather than reproduced verbatim to avoid introducing non-Roman fonts into the manuscript.
<table><tr><td>Issue</td><td>Representative annotation</td><td>Source</td></tr><tr><td>Non-English task annotation</td><td>Chinese-language manipulation instruction</td><td>Galaxea</td></tr><tr><td>Placeholder or truncated label</td><td>P, shirts, undefined</td><td>InternA1</td></tr><tr><td>Data-quality metadata</td><td>no robot motion, skip frame</td><td>OXE</td></tr><tr><td>Pure-motion instruction</td><td>move to the table, approach the cabinet</td><td>Galaxea</td></tr></table>

Table 6 summarizes the quantitative efect of the complete pipeline. Overall, curation retains 1,436,150 of 1,722,966 trajectory units (83.35%) while preserving 192,989 of 194,967 unique instructions (98.99%). This gap between unit retention and instruction retention indicates that the removed volume is concentrated in invalid annotations rather than in rare valid tasks.

The resulting mixture retains broad task coverage while reducing several sources of supervision bias. Under a diagnostic first-verb grouping, the miscellaneous other category decreases from 4.54% to 1.37%, reflecting the removal of malformed and non-actionable annotations. The dominant pick-place category decreases from 43.38% to 33.61% after filtering invalid OXE annotations. The three largest skill groups still account for 68.21% of the curated data, showing that the pipeline does not impose an artificial uniform distribution; instead, it improves task relevance and limits template domination while preserving the natural structure of the robot data.

As an additional post-hoc diagnostic, we normalize the first verb of each instruction into an atomic skill, merging morphological variants such as pickup, picking, and picked. The number of atomic verb categories decreases from 640 in the raw annotations to 549 after curation. The removed categories correspond mainly to empty tokens, termination markers, malformed strings, and other annotation artifacts. This atomic-skill analysis is used only to diagnose the efect of data curation; it does not define the temporal-distance targets or alter the training sampler.

Table 6 Summary of source-specific data curation. OXE and InternA1 are counted in episodes, while Galaxea and RoboCOIN are counted in annotated subtask segments. The total therefore denotes an aggregate number of trajectory units rather than a homogeneous episode count.
<table><tr><td>Source</td><td>Raw units</td><td>Curated units</td><td>Unit retention</td><td>Raw instructions</td><td>Curated instructions</td><td>Instruction retention</td></tr><tr><td>OXE (episode)</td><td>961,253</td><td>693,037</td><td>72.10%</td><td>180,290</td><td>180,090</td><td>99.89%</td></tr><tr><td>InternA1 (episode)</td><td>320,910</td><td>320,905</td><td>99.99%</td><td>350</td><td>348</td><td>99.43%</td></tr><tr><td>Galaxea (segment)</td><td>97,287</td><td>78,692</td><td>80.89%</td><td>12,685</td><td>10,909</td><td>86.00%</td></tr><tr><td>RoboCOIN (segment)</td><td>343,516</td><td>343,516</td><td>100.00%</td><td>1,642</td><td>1,642</td><td>100.00%</td></tr><tr><td>Total</td><td>1,722,966</td><td>1,436,150</td><td>83.35%</td><td>194,967</td><td>192,989</td><td>98.99%</td></tr></table>

## B Real-World Experiment Details

This section provides the implementation details needed to reproduce the robotic policy-learning experiments in Section 4.5. We first define the interfaces among the VLA policy and the auxiliary RL networks, and then describe the tasks and datasets, robotic platform, evaluation protocol, initialization, ofline and online RL procedures, and compute resources.

## B.1 Implementation Inputs and Outputs

The policy-learning pipeline contains three model interfaces. Throughout this section, $o _ { t }$ denotes the multicamera visual observation at policy-decision step t, and $a _ { t }$ denotes the corresponding executable action chunk. No component receives a proprioceptive or robot-state vector.

VLA policy. The VLA consumes camera images resized to $2 2 4 \times 2 2 4$ , image-validity masks, the task instruction, and the flow-matching timestep. It outputs an action chunk with horizon H = 16 and padded action dimension $d _ { a } = 3 2$

IQL critic and value. The ofline RL networks consume the current and next multi-camera observations together with the demonstrated action chunk. They estimate $Q ( o , a )$ and $V ( o )$ , from which we compute the advantage weight used to refine the VLA.

DSRL actor–critic. The online RL networks consume only multi-camera observations at $6 4 \times 6 4$ . The actor predicts a latent-noise chunk $z \in [ - 1 , 1 ] ^ { 1 6 \times 3 2 }$ , the critics estimate its Q-values, and automatic entropy tuning optimizes the temperature α.

## B.2 Tasks and Datasets

We evaluate all methods on 4 robotic manipulation tasks. For each task, we collect approximately 100 successful trajectories and retain every unsuccessful attempt encountered during the same collection process, yielding a mixed-expertise dataset. Table 7 reports the task instructions and dataset statistics.

Table 7 Real-world task instructions and dataset statistics. Success rate is computed over all collected trajectories for each task.
<table><tr><td>Task</td><td>Language instruction</td><td colspan="3">Trajectories</td><td>Success rate</td></tr><tr><td></td><td></td><td>Success</td><td>Failure</td><td>Total</td><td></td></tr><tr><td>Bread Basket Placement</td><td>&quot;Put the two pieces of bread in the basket.&quot;</td><td>99</td><td>4</td><td>103</td><td>96.1%</td></tr><tr><td>Steak Serving with a Spatula</td><td>&quot;Move the steak from the pan to the plate.&quot;</td><td>98</td><td>4</td><td>102</td><td>96.1%</td></tr><tr><td>Box-in-Drawer Placement</td><td>&quot;Put the box in the drawer and close it.&quot;</td><td>101</td><td>3</td><td>104</td><td>97.1%</td></tr><tr><td>Bimanual Box Transfer</td><td>&quot;Move the box from the right side to the left side.&quot;</td><td>100</td><td>1</td><td>101</td><td>99.0%</td></tr><tr><td>Total</td><td></td><td>398</td><td>12</td><td>410</td><td>97.1%</td></tr></table>

## B.3 Robotic Platform and Evaluation Protocol

Camera configuration. The dual-arm Franka platform is instrumented with 4 Intel RealSense cameras: 2 D435 third-person cameras, denoted left\_side and right\_side, and 2 D405 wrist-mounted cameras, denoted left\_wrist and right\_wrist. Images are resized with aspect-preserving padding to 224 × 224 for the VLA and IQL critic; the DSRL visual encoder receives inputs downsampled to $6 4 \times 6 4$ . Table 8 lists the exact streams used by each pathway.

Table 8 Camera streams used by the policy and RL encoders. Entries list the exact streams provided to each pathway.
<table><tr><td>Pathway</td><td>Single-arm tasks</td><td>Bimanual Box Transfer</td></tr><tr><td>VLA</td><td>left_side, left_wrist</td><td>left_side, left_wrist, right_wrist</td></tr><tr><td>IQL critic</td><td>left_side, left_wrist</td><td>left_side, left_wrist, right_wrist</td></tr><tr><td>DSRL actor-critic</td><td>left_side, left_wrist</td><td>left_side, right_side, left_wrist, right_wrist</td></tr></table>

Control and action representation. The low-level controller executes absolute joint-position commands and relative gripper commands at 10 Hz. A single-arm action contains seven absolute joint-position values and one relative gripper value; a bimanual action contains 14 absolute joint-position values and 2 relative gripper values. These task-specific actions are padded to $d _ { a } = 3 2$ . Each VLA prediction contains an action chunk with horizon $H = 1 6 .$ . During online DSRL, the latent action has shape $H \times d _ { z }$ with $d _ { z } = 3 2$ , and the VLA is queried after every $q = 1 0$ executed low-level control steps.

Reset, success, and termination. Environment resets are performed by an operator, who repositions the robot and objects and confirms that the next episode may begin. All ofline-policy evaluations use fixed initial configurations. During online evaluation, Bread Basket Placement, Steak Serving with a Spatula, and Bimanual Box Transfer use randomized initial-object configurations, whereas Box-in-Drawer Placement uses a fixed reset because the task requires precise gripper control and box–drawer alignment. The operator records a binary task-success label. An episode terminates when success or failure is confirmed or when it reaches the maximum horizon of 600 low-level control steps. Reward models provide only the shaping potential; they do not determine success or termination.

Reward-model inference. RynnValue and Robometer are deployed through the same scoring interface. For each collected trajectory, the reward service receives the task instruction and the RGB sequence from the right\_side third-person camera. Both models operate in a causal, history-conditioned scoring mode. Given a trajectory containing T RGB observations, for each step $t = 1 , \dots , T$ , the service scores the observation history $O _ { 1 : t } .$ , uniformly subsampled to 4 frames (rather than the 8 used during training, to match Robometer’s 4-frame inference protocol) before inference. Robometer decodes its native normalized progress $p _ { t } \in [ 0 , 1 ]$ as the expectation over ten progress bins and uses $\Phi _ { t } ^ { \mathrm { R o b o } } = p _ { t }$ . During reward relabeling and online RL, RynnValue uses only its absolute temporal-distance head and does not invoke the language-generation branch. Following the notation in the main text, we convert its predicted remaining time $v _ { t }$ into a higher-is-better potential using $\Phi _ { t } = - v _ { t }$ . Each complete trajectory is scored once after collection, and the resulting potential sequence is sampled at policy-decision boundaries to construct the shaping rewards. Reward relabeling therefore operates at the action-chunk granularity used by the RL algorithm, rather than at every low-level control step.

## B.4 Policy and RL Initialization

Supervised fine-tuning (SFT), ofline IQL, and online DSRL all use the same flow-matching vision-languageaction policy, $\pi _ { 0 . 5 }$ . Sharing this backbone isolates the efect of the learning algorithm and reward specification.

RL networks. The auxiliary RL networks are small relative to the VLA. Ofline IQL uses a ResNet-18 visual encoder with GroupNorm, spatial softmax, and a 50-dimensional bottleneck, followed by MLP heads with hidden dimensions (256, 256). It maintains 2 Q-functions and a separate value function. Online DSRL uses a 4-layer convolutional visual encoder with 32 channels per layer, GroupNorm, spatial softmax, and a 50-dimensional bottleneck. Its actor and critic MLPs have hidden dimensions (128, 128, 128); the critic ensemble contains ten Q-functions, and the actor is a tanh-squashed Gaussian distribution over latent noise.

Initialization and training stages. SFT and IQL are independent, single-stage fine-tuning procedures initialized from the same pretrained $\pi _ { 0 . 5 }$ checkpoint; IQL is not initialized from SFT. SFT optimizes the standard flow-matching behavior-cloning objective, whereas IQL uses advantage-weighted flow matching. Online DSRL uses task-specific initialization: Bread Basket Placement and Steak Serving with a Spatula start from their SFT checkpoints, while Box-in-Drawer Placement and Bimanual Box Transfer start from the corresponding Robometer-based ofline-RL checkpoints. The initial checkpoint for each task is held fixed across all online reward variants. During DSRL, the initialized VLA remains frozen and serves only as a noise-conditioned action decoder; the latent policy, critic ensemble, and entropy temperature are optimized.

## B.5 Offline RL with IQL

Mixed-expertise dataset construction. For each task $j ,$ , we construct a mixed-expertise ofline dataset by combining all successful and unsuccessful trajectories:

$$
\mathcal { D } _ { j } = \mathcal { D } _ { j , \mathrm { s u c c } } \cup \mathcal { D } _ { j , \mathrm { f a i l } } .\tag{12}
$$

Each trajectory is segmented at policy-decision boundaries into transitions $\left( o _ { h } , a _ { h } , o _ { h + 1 } , m _ { h } \right)$ , where $a _ { h }$ is an action chunk and $m _ { h } \in \{ 0 , 1 \}$ is the bootstrap mask. We set $m _ { h } = 0$ for terminal transitions and $m _ { h } = 1$ otherwise.

All reward variants share the same transitions, task-outcome labels, policy initialization, and optimization configuration. We relabel this common dataset with the potential produced by RynnValue or the corresponding baseline reward model; only the potential source and shaping coeficient difer across variants.

Offline reward construction. For a transition observation $O _ { h } .$ , we write $\Phi _ { h } = - v _ { h }$ for its shaping potential, where $v _ { h }$ is RynnValue’s decoded absolute temporal-distance prediction for $O _ { h }$ (Equation 10); the instruction ℓ and metadata m are omitted for brevity. The sparse task reward is

$$
r _ { h } ^ { \mathrm { s p a r s e } } = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ t h e ~ t a s k ~ i s ~ c o m p l e t e d ~ a f t e r ~ e x e c u t i n g ~ } } a _ { h } , } \\ { - 1 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{13}
$$

Thus, every transition before completion receives −1, while the transition that completes the task receives 0.   
All transitions in an unsuccessful trajectory receive −1.

The ofline potential-based shaping reward is

$$
\begin{array} { r } { r _ { h } ^ { \mathrm { s h a p e } } = \gamma _ { \mathrm { o f f } } \Phi _ { h + 1 } - \Phi _ { h } , } \end{array}\tag{14}
$$

where $\gamma _ { \mathrm { o f f } }$ is the ofline RL discount factor. The final reward used to train IQL is

$$
r _ { h } ^ { \mathrm { o f f } } = r _ { h } ^ { \mathrm { s p a r s e } } + \kappa r _ { h } ^ { \mathrm { s h a p e } } ,\tag{15}
$$

Table 9 Offline IQL and SFT hyperparameters. All IQL reward variants share the same mixed-expertise dataset, sparse task reward, policy initialization, and optimization configuration. SFT uses the same policy optimizer and learning-rate schedule and is also trained for 10,000 steps per task.  
Hyperparameter Value   
Base policy π<sub>0.5</sub> with flow matching; action dimension 32   
Action horizon H 16   
Batch size 64   
Policy optimizer AdamW with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 8 }$ , weight decay $1 0 ^ { - 1 0 } :$ and gradient-norm   
clipping at 1.0   
Learning-rate schedule Cosine decay with 2,000 linear warm-up steps   
Peak / final policy learning rate $3 \times 1 0 ^ { - 5 } / \stackrel { . } { 3 } \times 1 0 ^ { - 6 }$   
Policy EMA decay 0.99   
Training steps per task 10,000   
Critic / value optimizer Adam with a learning rate of 3 × $1 0 ^ { - 4 }$   
Ofline discount γ<sub>off</sub> 0.99   
Target update rate ρ<sub>off</sub> 0.005   
Expectile parameter τ 0.8   
Advantage temperature β 10.0   
Maximum advantage weight w<sub>max</sub> 100   
Number of Q-functions $K _ { \mathrm { I Q L } }$ 2; minimum aggregation   
Critic and value encoder ResNet-18 with GroupNorm and spatial softmax; 50-dimensional bottleneck   
Critic and value hidden dimensions (256, 256)   
Number of critic cameras 2 for single-arm tasks and 3 for the bimanual task   
Critic input resolution 224 × 224 using the VLA preprocessing pipeline   
Policy warm-up N<sub>warm</sub> 200 optimization steps with w(o, a) = 1   
Sparse task reward −1 before task completion and 0 upon task completion   
Potential-based shaping reward $r _ { h } ^ { \mathrm { s h a p e } } = \gamma _ { \mathrm { o f f } } \Phi _ { h + 1 } - \Phi _ { h }$   
Shaping coeficient κ 0.1 for RynnValue and 1.0 for Robometer; fixed across tasks   
Sparse-reward baseline $\kappa = 0$   
Image augmentation Random cropping applied to both current and next observations; no color jitter

where $\kappa \geq 0$ is the shaping coeficient used in the main text. We set $\kappa = 0 . 1$ for RynnValue, $\kappa = 1 . 0$ for Robometer, and $\kappa = 0$ for the sparse-reward baseline. These values are fixed across tasks.

IQL optimization and VLA refinement. We use the standard IQL objective [14]: the value function is trained by expectile regression, each Q-function uses a one-step temporal-diference target, and the 2 target Q-functions are aggregated by their minimum. Target Q-networks are updated by Polyak averaging. Rather than introducing a separate IQL policy, we retain the flow-matching VLA and weight its training loss by

$$
w ( o , a ) = \operatorname* { m i n } \left\{ \exp \left( \beta \left[ \operatorname* { m i n } _ { k } Q _ { \theta _ { k } } ( o , a ) - V _ { \psi } ( o ) \right] \right) , w _ { \operatorname* { m a x } } \right\} .\tag{16}
$$

For the first $N _ { \mathrm { w a r m } }$ optimization steps, we set $w ( o , a ) = 1 ;$ thereafter, demonstrated actions with large estimated advantages receive greater weight. Table 9 reports the complete architecture and optimization configuration.

## B.6 Online RL with DSRL

Latent policy optimization. For online RL, we use Difusion Steering via Reinforcement Learning (DSRL) [28]. The task-specific VLA checkpoint described above is frozen and used as a noise-conditioned action decoder:

$$
a _ { t } = \pi _ { 0 . 5 } ( o _ { t } , z _ { t } ) .\tag{17}
$$

A Soft Actor-Critic (SAC) policy $\mu _ { \varphi } ( z _ { t } \mid o _ { t } )$ is trained over the latent noise $z _ { t } ,$ rather than directly over executable robot actions. The latent action space is

$$
z _ { t } \in [ - 1 , 1 ] ^ { H \times d _ { z } } ,\tag{18}
$$

where $H = 1 6$ is the action horizon and $d _ { z } = 3 2$ is the per-step latent dimension.

SAC optimization. We use the standard entropy-regularized SAC objective in the latent action space. Critic targets use the mean of ten target Q-functions, the actor trades of the ensemble-mean Q-value against policy entropy, and the temperature is optimized automatically toward target entropy $\overline { { \mathcal { H } } } = - \dim ( z )$ . Target critics are updated by Polyak averaging. Table 10 reports the complete architecture and optimization configuration.

Online reward construction. Online DSRL uses the same sparse task-reward convention as Equation 13: each transition receives −1 until task completion, and the completing transition receives 0.

Let $\gamma _ { s }$ denote the per-environment-step shaping discount. Because each policy decision executes q low-level control steps, the corresponding chunk-level discount is

$$
\gamma _ { c } = \gamma _ { s } ^ { q } ,\tag{19}
$$

and the online shaping reward is

$$
r _ { t } ^ { \mathrm { s h a p e } } = \left\{ \begin{array} { l l } { \gamma _ { c } \Phi _ { t + 1 } - \Phi _ { t } , } & { \mathrm { i f ~ } o _ { t + 1 } \mathrm { ~ i s ~ n o n t e r m i n a l } , } \\ { - \Phi _ { t } , } & { \mathrm { i f ~ } o _ { t + 1 } \mathrm { ~ i s ~ t e r m i n a l } . } \end{array} \right.\tag{20}
$$

We use the shaping coeficient κ defined in the main text, setting $\kappa = 0 . 1$ for RynnValue, $\kappa = 1 . 0$ for Robometer, and $\kappa = 0$ for the sparse-reward baseline. These values are fixed across tasks, and all variants otherwise share the same reward definition and online optimization configuration. The final reward used by SAC is

$$
r _ { t } ^ { \mathrm { o n } } = r _ { t } ^ { \mathrm { s p a r s e } } + \kappa r _ { t } ^ { \mathrm { s h a p e } } .\tag{21}
$$

Table 10 Online DSRL hyperparameters. SAC operates in the latent space of the frozen VLA, which decodes latent variables into executable action chunks. All reward variants share the same optimization configuration and difer only in the potential source and shaping coeficient.  
Hyperparameter Value   
Base policy Frozen SFT checkpoint for Bread Basket Placement and Steak Serving with a Spatula;   
frozen Robometer ofline-RL checkpoint for Box-in-Drawer Placement and Bimanual Box   
Transfer   
Latent action space $z \in [ - 1 , 1 ] ^ { H \times d _ { z } }$ , with $H = 1 6$ and $d _ { z } = 3 2$   
RL algorithm SAC with automatic entropy tuning and initial temperature $\alpha _ { 0 } = 1 . 0$   
Actor optimizer Adam with a learning rate of $1 \times 1 0 ^ { - 4 }$   
Critic optimizer Adam with a learning rate of $3 \times 1 0 ^ { - 4 }$   
Temperature optimizer Adam with a learning rate of $3 \times 1 0 ^ { - 4 }$   
Gradient clipping None   
Target entropy H − dim(z)   
Online SAC discount $\gamma _ { \mathrm { o n } }$ 0.999   
Target update rate ρ<sub>on</sub> 0.005   
Number of Q-functions $K _ { \mathrm { S A C } }$ 10; mean aggregation   
Actor and critic hidden dimensions (128, 128, 128)   
Image encoder Four-layer CNN with 32 channels per layer, strides $( 2 , 1 , 1 , 1 )$ , VALID padding, GroupNorm,   
spatial softmax, and a 50-dimensional bottleneck   
SAC input resolution 64 × 64   
Batch size 256   
Update-to-data ratio 100   
Training length 6,000 training steps   
Online rollout trajectories 60 per task   
Replay-bufer capacity max(training steps $/ \mathrm { U T D } , 1 0 ^ { 4 } ) = 1 0 ^ { 4 }$   
Update frequency After each episode   
Exploration warm-up $N _ { \mathrm { n o i s e } } = 2$ episodes with Gaussian noise standard deviation $\sigma = 0 . 1$   
Minimum replay size $N _ { \mathrm { s t a r t } }$ 200 transitions   
Maximum episode length 600 environment steps   
Policy-decision frequency Low-level control at 10 Hz, with one policy decision every q = 10 environment steps   
Number of DSRL cameras 2 for single-arm tasks and 4 for the bimanual task   
Sparse task reward −1 before task completion and 0 upon task completion   
Shaping coeficient κ 0.1 for RynnValue, 1.0 for Robometer, and 0 for the sparse-reward baseline; fixed across   
tasks   
Per-step shaping discount $\gamma _ { s }$ 0.999

## B.7 Compute Resources

Ofline IQL and SFT each use 2 GPUs with 80GB of memory each. IQL assigns one FSDP device to each replica, whereas SFT uses two-way model sharding. Training one task requires approximately 16 hours for ofline IQL and 6 hours for SFT. Online DSRL runs on a single x86 server equipped with two Intel Xeon Platinum 8575C processors (48 physical cores per socket with hyper-threading enabled; 192 logical cores in total), 1.5 TiB of system memory, and eight NVIDIA GeForce RTX 5090 GPUs with 32 GB of memory each. The server has a dual-socket NUMA topology and runs NVIDIA driver 570.211.01 with CUDA 12.8. Each task uses 6,000 training steps and 60 online rollout trajectories, requiring approximately 1.5 hours. All online DSRL experiments use single-node execution.