# Native Active Perception as Reasoning for Omni-Modal Understanding

Zhenghao Xing 1 \* † Ruiyang Xu 2 \* † Yuxuan Wang 3 \* Jinzheng He 3 Ziyang Ma 2 4 † Qize Yang 3 Yunfei Chu 3 Jin Xu 3 ‡ Junyang Lin 3 Chi-Wing Fu 1 Pheng-Ann Heng 1 ‡

1The Chinese University of Hong Kong 2Shanghai Jiao Tong University   
3Qwen Team, Alibaba Group 4Nanyang Technological University

Code: https://github.com/harryhsing/OmniAgent

SFT: https://huggingface.co/harryhsing/OmniAgent-SFT-7B

RL: https://huggingface.co/harryhsing/OmniAgent-RL-7B

\*Equal contribution. †Work done during an internship at Qwen Team, Alibaba Group. ‡Corresponding authors.

## Abstract

Passive models for long video understanding typically rely on a “watch-it-all” paradigm, processing frames uniformly regardless of query difficulty, causing computational cost to grow with video duration. Although interactive frameworks have emerged, they often rely on global pre-scanning, and their context cost still scales with video length. We propose OmniAgent, the first native omni-modal agent that formulates video understanding as a POMDP-based iterative Observation-Thought-Action cycle. OmniAgent executes on-demand actions to selectively distill audio-visual cues into a persistent textual memory, effectively decoupling reasoning complexity from raw video duration. To operationalize this, we introduce (1) Agentic Supervised Fine-Tuning to bootstrap native active perception via best-of-N trajectory synthesis with dual-stage quality control, and (2) Agentic Reinforcement Learning with TAURA (Turn-aware Adaptive Uncertainty Rescaled Advantage), which leverages turn-level entropy to steer credit assignment toward pivotal discovery turns. Crucially, Omni-Agent exhibits positive test-time scaling, where performance improves as the number of reasoning turns increases, validating the efficacy of active perception. Empirical results across ten benchmarks (e.g., VideoMME, LVBench) demonstrate that OmniAgent achieves state-of-the-art performance among open-source models. Notably, on LVBench, our 7B agent outperforms the 10× larger Qwen2.5-VL-72B (50.5% vs. 47.3%).

## 1. Introduction

Recent advances in scaling laws have propelled large language models (LLMs) toward general-purpose intelligence, extending capabilities into the visual (Li et al., 2022; Liu et al., 2023; Bai et al., 2025; Lin et al., 2024) and auditory domains (Chu et al., 2024; Xu et al., 2025a). Despite these strides, current paradigms largely treat multimodal perception as the processing of static snapshots or fixed-window streams. This approach clashes with the nature of human perception, which functions as an active, continuous interrogation of intertwined signals. More critically, the high dimensionality of spatiotemporal data imposes a prohibitive constraint: computational cost scales super-linearly with sequence length. This renders passive end-to-end processing computationally intractable for long-form video understanding, creating a central bottleneck in open-world multimodal modeling.

To mitigate this computational burden, prior work has explored agentic adaptations. One branch utilizes LLMs as controllers to invoke modality-specialized tools (Fan et al., 2024; Zhang et al., 2025b; Long et al., 2026). While expedient, this reliance on intermediate modules creates an information bottleneck, severing the gradient flow between reasoning and perception. A second branch pursues “thinking with images” by integrating transformation tools, such as temporal clipping (Zhang et al., 2026; Yang et al., 2026) or spatial zooming (Shen et al., 2025a), directly within the multimodal large language models (MLLMs)’s chain-ofthought. However, these methods often retain a semi-passive nature: they typically require a global pre-scan of the video or maintain a dense visual buffer to decide “where to look,” failing to truly decouple reasoning complexity from video duration. Consequently, they struggle to scale to hour-long videos where raw pixel retention is infeasible.

![](images/bc2660917d560aca73f3420f0cd639e3545e3a8767c207749113f9f35be53a90.jpg)  
Figure 1. The OmniAgent Framework for Native Active Perception. Unlike passive methods that process video frames uniformly, OmniAgent treats perception as an iterative reasoning process via an Observation-Thought-Action (OTA) cycle. Conditioned on a specific query, the agent executes on-demand actions to selectively gather audio-visual cues, distilling high-dimensional transient percepts into a persistent textual memory until sufficient evidence is gathered to produce the final answer.

In response, we propose OmniAgent (Figure 1), a framework that reimagines MLLMs not as passive observers, but as native active perceivers. We formulate audio-visual exploration as a Partially Observable Markov Decision Process (POMDP), where the agent performs a strict information distillation. Through an iterative Observation-Thought-Action (OTA) cycle, the agent actively browses (frames), listens (audio), and watches (audio-visual clips), distilling high-dimensional transient percepts into a persistent textual memory. This architecture ensures that the agent’s internal state depends solely on the complexity of the reasoning trace rather than the raw duration of the video. This formulation gives rise to an emergent test-time scaling property: the model adaptively allocates more computational steps to resolve harder queries, analogous to System-2 reasoning, where inference-time compute is dynamically expended to resolve ambiguity. Unlike prior agentic approaches that delegate multimodal perception to external modules, Omni-Agent is a single native omni model: the environment only performs raw media extraction (returning frames, audio segments, or video clips), and all perception and reasoning are performed by the same model that acts.

To operationalize this, we introduce a two-stage optimization. First, we bootstrap native active perception via Agentic Supervised Fine-Tuning (SFT), a best-of-N trajectory synthesis pipeline with dual-stage quality control (Sec. 3.2). Second, we refine the policy via Agentic Reinforcement Learning (RL) (Sec. 3.3), built around TAURA (Turnaware Adaptive Uncertainty Rescaled Advantage). We identify that standard Group Relative Policy Optimization (GRPO) (Guo et al., 2025), when applied to multi-turn agentic reasoning, suffers from Advantage Homogenization, where a uniform trajectory-level advantage conflates pivotal discovery turns with trivial actions. TAURA resolves this by using turn-level entropy to steer credit assignment toward these critical moments. Our main contributions are summarized as follows:

• First Native Omni-Modal Active Perception Framework: We introduce OmniAgent, which, to our knowledge, is the first end-to-end native agentic framework that unifies perception, reasoning, and action within a single model for omni-modal video tasks. By formulating multimodal exploration as a POMDP with a persistent textual memory, OmniAgent effectively decouples reasoning complexity from video duration, enabling scalable reasoning over hour-long videos.

• Two-Stage Agentic Optimization: We propose a twostage optimization comprising (i) Agentic SFT, which bootstraps native active perception through a best-of-N trajectory synthesis pipeline with dual-stage quality control, and (ii) TAURA, an entropy-steered RL objective that resolves advantage homogenization in GRPO by using turn-level entropy to amplify credit for pivotal discovery turns over trivial actions.

• Comprehensive SoTA Performance: OmniAgent establishes new state-of-the-art results among opensource models across ten benchmarks, improving over the direct Qwen2.5-Omni baseline on all of them. It advances long-video comprehension (LVBench 50.5%, MLVU 71.1%), excels in omni-modal understanding (DailyOmni +4.7%, OmniVideo +7.8%), and achieves an absolute +33.4% gain in temporal grounding on LongVALE. Notably, our 7B agent outperforms the 10× larger Qwen2.5-VL-72B on LVBench (50.5% vs. 47.3%) with 73% fewer frames, and demonstrates positive test-time scaling (+6.2% on VideoMME-Long).

## 2. Related Work

## 2.1. Passive Omni Modality Understanding

The advancement of Omni Large Language Models (OmniLLMs) has been marked by significant proprietary contributions like GPT-4o (OpenAI, 2024) and Project Astra (Google DeepMind, 2024), alongside a surge in opensource models (Fu et al., 2024; Li et al., 2024; Cheng et al., 2024; Xu et al., 2025a; Xing et al., 2025; Tang et al., 2025; Yang et al., 2025b; Xu et al., 2025b; Meituan LongCat Team, 2025). While these initiatives enhance audio-visual understanding, existing passive methods struggle with long contexts due to the prohibitive complexity of continuous signals. Unlike static frame sampling, the natural temporal continuity of audio-visual streams hinders global simultaneous attention. To address this, OmniAgent endows MLLMs with native active perception, leveraging video compositionality to effectively decouple reasoning complexity from the video duration.

## 2.2. Agentic Video Reasoning

Pioneered by VisProg (Gupta & Kembhavi, 2023), agentic video understanding has evolved along two primary trajectories. The first leverages LLMs to orchestrate expert modules, relying on pre-extracted contexts like captions (Wang et al., 2024) and summaries (Wang et al., 2023; Ma et al., 2025), structured planning (Yang et al., 2025c; 2024; Wang et al., 2025d), or diverse tools including tracking (Fan et al., 2024), ASR (Tao et al., 2025), search (Zhang et al., 2025b; Long et al., 2026), and ensembles (Chen et al., 2025a). The second adapts “Think with Image” paradigms to video, utilizing transformations like temporal clipping (Zhang et al., 2026; Yang et al., 2026), spatial zooming (Shen et al., 2025a), or combinatorial cropping (Rasheed et al., 2026) for exhaustive analysis. However, both approaches neglect video’s inherent sequential nature, treating it effectively as a static information container or large image, which hampers scaling. To address this, we formulate video understanding as a Partially Observable Markov Decision Process (POMDP), injecting native agentic capabilities into MLLMs to achieve robust test-time scaling without reliance on external modules.

Algorithm 1 OmniAgent: Active Perception and Memory   
Consolidation   
Require: Query $Q ,$ metadata $V _ { \mathrm { { m e t a } } } ,$ horizon $K .$   
Ensure: Terminal answer y.   
1: Initialize: Persistent memory $\mathcal { M } _ { 0 } \gets \{ Q , V _ { \mathrm { m e t a } } \}$ , tran  
sient multimodal percept ${ \mathcal { E } } _ { 0 } \gets \emptyset .$   
2: for $k = 1$ to K do   
3: PHASE 1: ACTIVE PERCEPTION   
4: $( O _ { k } , T _ { k } , A _ { k } ) \sim \pi _ { \theta } ( \cdot \mid \mathcal { M } _ { k - 1 } , \mathcal { E } _ { k - 1 } )$   
5: ▷ Query-Conditional: Grounds exploration in intent   
Q and current context.   
6: PHASE 2: MEMORY CONSOLIDATION   
7: $\mathcal { M } _ { k }  \mathcal { M } _ { k - 1 } \cup \{ ( O _ { k } , T _ { k } , A _ { k } ) \}$   
8: ▷ Decoupling: Maintains constant-order media over  
head.   
9: if $A _ { k } = a _ { \mathrm { a n s w e r } } ( y )$ then   
10: return y   
11: end if   
12: PHASE 3: PERCEPTUAL TRANSITION   
13: $\mathcal { E } _ { k }  \Omega ( A _ { k } )$   
14: ▷ Environment executes $A _ { k }$ , purging previous $\mathcal { E } _ { k - 1 }$   
15: end for

## 3. OmniAgent

OmniAgent reconceptualizes omni-modal video reasoning by bridging the gap between perception and cognition, treating active perception not as a preprocessing step but as a query-driven reasoning process. Unlike passive models that process inputs uniformly, our framework decouples reasoning from multimodal data by establishing a strict separation between transient multimodal percept and persistent textual memory. This architecture enables the agent to selectively distill critical audio-visual cues into a compact textual memory while discarding high-dimensional raw media (see Figure 1). Consequently, OmniAgent maintains a longhorizon reasoning trace where the contextual cost is largely independent of the video duration. OmniAgent is optimized through a two-stage regime: (i) Agentic SFT (Sec. 3.2) to bootstrap fundamental action execution capabilities, and (ii) Agentic RL (Sec. 3.3) to refine reasoning-driven perception, with TAURA mitigating the credit assignment challenge in multi-turn reasoning.

## 3.1. Overall Agentic Pipeline

We formulate the interaction between OmniAgent and the video environment as a Partially Observable Markov Decision Process (POMDP). Here, the transient percept $\mathcal { E } _ { k }$ represents the raw media returned by the environment $\Omega ,$ while the persistent memory $\mathcal { M } _ { k }$ serves as the agent’s consolidated internal state. Unlike passive models that process video frames uniformly, OmniAgent employs a policy πθ to selectively distill the transient $\mathcal { E } _ { k - 1 }$ into a compact textual observation $O _ { k }$

The process executes through an iterative Observation-Thought-Action (OTA) cycle (Algorithm 1). The state at turn k is formalized as the memory $\begin{array} { r l } { \mathcal { M } _ { k } } & { { } = } \end{array}$ $\left( O _ { 0 } , \mathbf { O T A _ { 1 } } , \dots , \mathbf { O T A } _ { k } \right)$ , where initial state $\begin{array} { r l } { \mathcal { M } _ { 0 } } & { { } = } \end{array}$ $\{ Q , V _ { \mathrm { m e t a } } \}$ contains the query and video metadata (e.g., duration, FPS, audio availability). At each turn $k \in \{ 1 , \ldots , K \}$ where $K$ is the maximum turn limit, the agent generates the OTA triplet autoregressively. The policy conditions on both the persistent memory and the transient percept:

$$
( O _ { k } , T _ { k } , A _ { k } ) \sim \pi _ { \theta } ( \cdot \mid \mathcal { M } _ { k - 1 } , \mathcal { E } _ { k - 1 } )\tag{1}
$$

At $k = 1 ( \mathcal { E } _ { 0 } = \emptyset )$ , the policy conditions solely on $\mathcal { M } _ { 0 }$ to bootstrap the initial exploration (see Appendix A for a complete notation summary).

Observation $( O _ { k } )$ ): A structured textual summary that distills the high-dimensional percept $\mathcal { E } _ { k - 1 }$ into persistent memory. Unlike raw pixels, $O _ { k }$ serves as an information-dense encoding for the reasoning process, explicitly retaining critical visual and auditory details required for future reasoning before the raw media is purged.

Thought $( T _ { k } ) \colon$ : The internal reasoning process that bridges perception and action. $T _ { k }$ analyzes the preceding memory $\mathcal { M } _ { k - 1 }$ and the current observation $O _ { k }$ to reason over the accumulated evidence. It identifies information gaps between the current percept and the query requirements, deriving the rationale for the subsequent action $A _ { k }$

Action $( A _ { k } ) \colon$ The symbolic operator sampled from ${ \mathcal { A } } =$ $\{ a _ { \mathrm { f r a m e s } } , a _ { \mathrm { a u d i o } } , a _ { \mathrm { c l i p } } , a _ { \mathrm { a n s w e r } } \}$ Specifically: $a _ { \mathrm { f r a m e s } } ( s , e , n )$ retrieves n frames uniformly from the time interval $[ s , e ]$ offering flexible temporal resolution; $a _ { \mathrm { a u d i o } } ( s , e )$ extracts the audio segment; $a _ { \mathrm { c l i p } } ( s , e )$ captures a continuous video segment with synchronized audio to preserve temporal continuity and cross-modal alignment; and $a _ { \mathrm { a n s w e r } } ( y )$ emits the final answer $y ,$ terminating the trajectory.

Memory Consolidation. The environment Ω facilitates turn-level transitions by resolving $A _ { k }$ into new percept $\mathcal { E } _ { k }$ This triggers a strict context purging mechanism: the previous raw multimodal percept $\mathcal { E } _ { k - 1 }$ is discarded from the active context, leaving only the distilled text $O _ { k }$ in $\mathcal { M } _ { k }$ This ensures the model’s media overhead remains constant regardless of the video duration or the number of interaction turns (see Appendix B for context management details).

Note that Ω performs only raw media extraction (frame retrieval, audio extraction, clip capture); all semantic perception and reasoning are carried out natively by $\pi _ { \theta }$ without relying on external modules.

## 3.2. Agentic Supervised Fine-Tuning

Directly optimizing OmniAgent via reinforcement learning risks policy collapse, as base models (Xu et al., 2025a) lack prior training for long-horizon agentic reasoning. To bootstrap these capabilities, we curate an Agentic SFT corpus of 58K trajectories across three task categories (MCQ, numerical reasoning, and temporal grounding), derived from the training splits of five datasets: LongVideo-Reason (Chen et al., 2025c), Video-Holmes (Cheng et al., 2025), VSI-Train-10k (Brown et al., 2025), LongVALE (Geng et al., 2025), and MultiHop-EgoQA (Chen et al., 2025b). This corpus is strictly aligned with the iterative $\left( O _ { k } , T _ { k } , A _ { k } \right)$ cycle formalized in Algorithm 1.

Synthesis via Exploration. Instead of relying on static QA annotations, we prompt a teacher model via in-context learning with the instruction template in Appendix B.5 to perform success-driven exploration in the environment Ω. For each query, we execute a best-of-N generation over the action space A to produce a diverse pool of candidate trajectories. This generation process explicitly allows for self-correction, where the model initially executes invalid actions $( e . g .$ , out-of-bounds timestamps) but successfully recovers based on the symbolic environment feedback. Including these error-correction traces prevents the “teacherforcing” bias, training OmniAgent to interpret diagnostic signals as actionable cues rather than fatal failures (see $\mathbf { A p } \mathbf { \cdot }$ pendix B for diagnostic error protocols).

Dual-Stage Quality Control. To distill high-quality training data from the raw candidate pool, we implement a twostep filtration pipeline. (1) Outcome Verification: We first filter for correctness based on the task-specific success criteria defined in Eq. 2. Specifically, we require exact matches for discrete tasks (MCQ, Numerical) and enforce thresholdbased criteria for continuous tasks: Intersection over Union $\mathrm { ( I o U ) \geq 0 . 5 }$ for temporal grounding and Mean Relative Accuracy $( \mathrm { M R A } ) \geq 0 . 5$ for size estimation. (2) Rationality Audit: Since the textual memory decouples reasoning from raw media, we employ GPT-4o (OpenAI, 2024) to audit the internal coherence of the reasoning trace. GPT-4o evaluates whether the current Thought $T _ { k }$ is logically entailed by the accumulated memory $\mathcal { M } _ { k - 1 }$ and the immediate observation $O _ { k }$ on a 5-point Likert scale. This step filters out “lucky guesses”, trajectories that reach the correct answer through hallucinated or heuristic reasoning steps that are not supported by the recorded memory. By enforcing a minimum coherence score of 3/5, we ensure that all SFT actions are rationally grounded in the agent’s explicit context.

## 3.3. Agentic Reinforcement Learning

To incentivize the policy to handle more complex interactions within the proposed environment and facilitate self-

evolution, we further optimize OmniAgent using reinforcement learning. By utilizing verifiable rewards, we advance the model beyond the initial priors established during SFT.

Verifiable Reward Design. OmniAgent is optimized to maximize a task-specific verifiable reward $R ,$ quantifying the alignment between the prediction $\hat { y }$ and ground-truth y:

$$
R ( \hat { y } , y ) = \left\{ \begin{array} { l l } { { \mathbb { 1 } } [ \hat { y } = y ] } & { \mathrm { D i s c r e t e ~ ( M C Q ~ / ~ N u m e r i c a l ) } } \\ { { \mathrm { I o U } } ( \hat { y } , y ) } & { \mathrm { T e m p o r a l ~ G r o u n d i n g } } \\ { { \mathrm { M R A } } ( \hat { y } , y ) } & { \mathrm { C o n t i n u o u s ~ ( S i z e ~ E s t i m a t i o n ) } } \end{array} \right.\tag{2}
$$

where $\begin{array} { r } { \mathrm { I o U } ( \hat { y } , y ) = \frac { | \mathcal { T } _ { \hat { y } } \cap \mathcal { T } _ { y } | } { | \mathcal { T } _ { \hat { y } } \cup \mathcal { T } _ { y } | } } \end{array}$ represents temporal overlap, and MRA assesses quantitative fidelity across relative precision thresholds $\mathcal { T } = \{ 0 . 5 , \hdots , 0 . 9 5 \}$

The Advantage Homogenization Problem. Applying Group Relative Policy Optimization (GRPO) (Guo et al., 2025) to multi-turn agentic reasoning faces a structural limitation: Advantage Homogenization. By broadcasting a single scalar advantage to every turn, vanilla GRPO inherently overlooks the heterogeneous contributions of individual turns, conflating pivotal forks with trivial fillers. This flaw is substantiated by our empirical analysis in $\mathsf { A p - }$ pendix C, which reveals that 79.2% of critical branching turns exhibit significantly higher mean token entropy than the trajectory mean. Consequently, vanilla GRPO’s uniform advantage broadcasting masks the causal significance of high-uncertainty discovery moments, necessitating the entropy-steered credit assignment in TAURA.

TAURA: Entropy-Steered Credit Assignment. To address the advantage homogenization problem, we propose TAURA (Turn-aware Adaptive Uncertainty Rescaled Advantage), which refines trajectory-level advantages into turn-level attributions. Our method is inspired by findings that high-entropy tokens often represent critical “forks” in reasoning (Wang et al., 2025b). However, while prior work suggests masking low-entropy tokens in a Chain-of-Thought, such a binary masking strategy is ill-suited for agentic trajectories. Since the atomic reasoning unit is the structured turn $\left( O _ { k } , T _ { k } , A _ { k } \right)$ , masking individual tokens disrupts the output structure, while masking entire turns severs essential contextual dependencies and semantic continuity.

Formally, for a group of G trajectories sampled from the same query, we first compute the baseline trajectory-level advantage Ai by normalizing the rewards:

$$
A _ { i } = \frac { R _ { i } - \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R _ { j } } { \operatorname { s t d } ( R _ { 1 } , \dots , R _ { G } ) }\tag{3}
$$

where $R _ { i }$ is the reward for the i-th trajectory as defined in Eq. 2.

Building upon this baseline, TAURA implements a turnlevel rescaling. By using mean token entropy as a continuous weighting factor rather than a binary mask, TAURA prioritizes turns with higher information density while preserving gradient flow for all tokens. For each turn k within trajectory $i ,$ let $H _ { i , k }$ denote its mean token entropy. We define the rescaled advantage $\hat { A } _ { i , k }$ by normalizing $H _ { i , k }$ relative to the group mean:

$$
\hat { A } _ { i , k } = A _ { i } \cdot \underbrace { \frac { H _ { i , k } } { \frac { 1 } { N _ { \mathcal { G } } } \sum _ { j = 1 } ^ { G } \sum _ { m = 1 } ^ { K _ { j } } H _ { j , m } } } _ { w _ { i , k } }\tag{4}
$$

where $G$ is the group size, $K _ { j }$ denotes the turn count of trajectory $j ,$ and $\begin{array} { r } { N _ { \mathcal { G } } = \sum _ { j = 1 } ^ { G } K _ { j } } \end{array}$ represents the total turn count across the group. This normalization ensures that the expected weight $\mathbb { E } [ w _ { i , k } ] = 1$ over the group, thereby preserving the original gradient scale while directing updates toward high-uncertainty discovery moments.

The policy is subsequently optimized using the TAURAenhanced GRPO objective. To resolve the symbol ambiguity, we denote the i-th trajectory as $\tau _ { i }$ and its constituent tokens as $\{ o _ { i , t } \} _ { t = 1 } ^ { | \tau _ { i } | }$ . The surrogate objective is formalized as:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } , \{ \tau _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \tau _ { i } | } \sum _ { t = 1 } ^ { | \tau _ { i } | } \mathcal { L } _ { i , t } ( \theta ) \right]\tag{5}
$$

where the per-token loss $\mathcal { L } _ { i , t } ( \boldsymbol { \theta } )$ incorporates the turn-level rescaled advantage:

$$
\begin{array} { r l } & { \mathcal { L } _ { i , t } ( \theta ) = \mathrm { { m i n } } \big ( \rho _ { i , t } \hat { A } _ { i , \mathrm { { t u r n } } ( t ) } , } \\ & { \qquad \mathrm { { c l i p } } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i , \mathrm { { t u r n } } ( t ) } \big ) } \end{array}\tag{6}
$$

Here, $\begin{array} { r } { \rho _ { i , t } ~ = ~ \frac { \pi _ { \theta } \left( o _ { i , t } \vert q , o _ { i , < t } , \mathcal { E } _ { \mathrm { t u r n } \left( t \right) - 1 } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( o _ { i , t } \vert q , o _ { i , < t } , \mathcal { E } _ { \mathrm { t u r n } \left( t \right) - 1 } \right) } } \end{array}$ is the importance sampling ratio. The mapping function turn(t) assigns each token t to its corresponding turn index $k \in \{ 1 , \ldots , K _ { i } \}$ ensuring that the credit for each token is steered by the information density of its constituent turn.

Why Entropy Scaling Works. TAURA scales the signed advantage $A _ { i }$ . For correct trajectories $( A _ { i } > 0 )$ , high entropy $( w _ { i , k } > 1 )$ amplifies the advantage, upweighting turns where the model navigated genuine uncertainty. Conversely, for incorrect trajectories $( A _ { i } < 0 )$ , high entropy results in a larger negative penalty $( \hat { A } _ { i , k } < A _ { i } < 0 )$ . This strictly penalizes confused guessing while reinforcing valid discovery actions.

## 4. Experimental Results

## 4.1. Experimental Settings

Benchmarks and Metrics. We evaluate across ten benchmarks in three categories. (1) Video Understanding:

Table 1. Main results on video understanding and reasoning. We evaluate OmniAgent across a comprehensive suite of benchmarks featuring diverse temporal scales. Bold and underline indicate the best and second-best performance among open-source models, respectively. Methods marked with ∗ incorporate audio signals. VISTA result is based on LongVA. ∆ denotes the performance gain relative to the Qwen2.5-Omni baseline.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Size</td><td colspan="2">VideoMME (w/o sub.)</td><td>VSI-Bench</td><td>MLVU</td><td>Minerva</td><td>LVBench</td></tr><tr><td>Overall</td><td>Long</td><td>AVG</td><td>M-AVG</td><td>AVG</td><td>AVG</td></tr><tr><td>Duration</td><td></td><td>1-60 min</td><td>30-60 min</td><td>97 sec</td><td>3-120 min</td><td>2-90 min</td><td>4101 sec</td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o (OpenAI, 2024)</td><td></td><td>71.9</td><td>65.3</td><td>34.0</td><td>64.6</td><td>45.5</td><td>48.9</td></tr><tr><td>Gemini-1.5-Pro (Gemini Team,2024)</td><td></td><td>75.0</td><td>67.4</td><td>45.4</td><td></td><td></td><td>33.1</td></tr><tr><td>Gemini-2.5-Pro (Comanici et al., 2025)</td><td></td><td>1</td><td>1</td><td>51.5</td><td></td><td>66.2</td><td>67.4</td></tr><tr><td>Open-Source Agentic Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LongVT (Yang et al., 2026)</td><td>7B</td><td>55.9</td><td>44.4</td><td>34.4</td><td>1</td><td>28.5</td><td>41.3</td></tr><tr><td>Zoom-Zero (Shen et al., 2025a)</td><td>7B</td><td>66.0</td><td>54.8</td><td>1</td><td>70.8</td><td>1</td><td>45.7</td></tr><tr><td>VITAL (Zhang et al., 2026)</td><td>7B</td><td>64.1</td><td>54.0</td><td>41.8</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Video-CoM (Rasheed et al., 2026)</td><td>7B</td><td>59.4</td><td>1</td><td>1</td><td>60.9</td><td>31.7</td><td>1</td></tr><tr><td>Open-Source Thinking Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Video-R1 (Feng et al., 2025)</td><td>7B</td><td>61.4</td><td>1</td><td>37.1</td><td>60.9</td><td>29.1</td><td>40.1</td></tr><tr><td>Open-o3 Video (Meng et al., 2026)</td><td>7B</td><td>63.6</td><td>54.9</td><td>1</td><td>1</td><td>1</td><td></td></tr><tr><td> VideoRFT (Wang et al., 2025a)</td><td>7B</td><td>59.8</td><td>1</td><td>36.8</td><td>59.7</td><td>29.2</td><td>1 34.7</td></tr><tr><td>LongVILA-R1 (Chen et al.,2025c)</td><td>7B</td><td>65.1</td><td>55.2</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Open-Source Non-Thinking Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kangaroo (Liu et al.,2024)</td><td>8B</td><td>56.0</td><td>46.7</td><td>1</td><td>61.0</td><td></td><td>39.4</td></tr><tr><td>VideoLLaMA2* (Cheng et al.,2024)</td><td>7B</td><td>47.9</td><td>1</td><td>1</td><td>48.5</td><td></td><td>1</td></tr><tr><td>LongVA (Zhang et al.,2025a)</td><td>7B</td><td>51.8</td><td>46.1</td><td>29.2</td><td>56.3</td><td>一</td><td>35.9</td></tr><tr><td>LLaVA-OneVision (Li et al., 2025)</td><td>7B</td><td>58.2</td><td>46.7</td><td>32.4</td><td>64.7</td><td>1</td><td>1</td></tr><tr><td>VISTA (Ren et al., 2025b)</td><td>7B</td><td>55.5</td><td>47.4</td><td>1</td><td>62.1</td><td>一</td><td>39.0</td></tr><tr><td>VideoChat-T (Zeng et al.,2025)</td><td>7B</td><td>46.3</td><td>41.9</td><td>1</td><td>1</td><td>1</td><td></td></tr><tr><td>LongVILA (Chen et al., 2025d)</td><td>7B</td><td>60.1</td><td>53.0</td><td>21.6</td><td>1</td><td></td><td>1 1</td></tr><tr><td>Vamba (Ren et al., 2025a)</td><td>10B</td><td>57.8</td><td>1</td><td>1</td><td>65.9</td><td>1</td><td>42.1</td></tr><tr><td>LongVU (Shen et al., 2025b)</td><td>7B</td><td>60.6</td><td>59.5</td><td>1</td><td>65.4</td><td>1</td><td></td></tr><tr><td>Qwen2.5-VL (Bai et al., 2025)</td><td>7B</td><td>65.1</td><td>1</td><td>33.5</td><td>1</td><td>33.0</td><td>45.3</td></tr><tr><td>Qwen2.5-Omni* (Xu et al.,2025a)</td><td>7B</td><td>64.8</td><td>54.8</td><td>35.5</td><td>65.2</td><td>33.4</td><td>43.0</td></tr><tr><td>OmniAgent (Ours)*</td><td>7B</td><td>67.8</td><td>59.6</td><td>48.4</td><td>71.1</td><td>41.4</td><td>50.5</td></tr><tr><td>△ over Baseline</td><td></td><td>+3.0</td><td>+4.8</td><td>+12.9</td><td>+5.9</td><td>+8.0</td><td>+7.5</td></tr></table>

VideoMME (generic) (Fu et al., 2025), VSI-Bench (reasoning) (Yang et al., 2025a), MLVU (long) (Zhou et al., 2025a), Minerva (reasoning & long) (Nagrani et al., 2025), and LVBench (long) (Wang et al., 2025c). (2) Audio-Visual: DailyOmni (generic) (Zhou et al., 2025b), WorldSense (generic) (Hong et al., 2025), and OmniVideoBench (reasoning) (Li et al., 2026). (3) Temporal Grounding: Long-VALE (Geng et al., 2025) and VUE-TR (Vidi Team, 2025). Together, these benchmarks cover video durations ranging from tens of seconds to over two hours. The evaluation metrics (MCQ accuracy, Temporal IoU, MRA) are aligned with the verifiable reward functions defined in Sec. 3.3.

Implementation Details. We adopt Qwen2.5-Omni-7B (Xu et al., 2025a) as the base model. To manage context length, the maximum number of visual tokens per image and per video frame is set to 1024 and 768, respectively, with a maximum context window of 64K tokens. Both training stages employ a dynamic maximum turn limit K (Algorithm 1) that scales with video duration $( K \in [ 5 , 3 2 ]$ for Agentic SFT, $K \in [ 5 , 1 0 ]$ for Agentic RL).

The same agent instruction template (Appendix B.5) is used across Agentic SFT, Agentic RL, and inference. The Agentic SFT is conducted on 58K trajectories for 2 epochs, with a learning rate of $1 \times 1 0 ^ { - 5 }$ , a batch size of 64, and the AdamW optimizer on 16 NVIDIA A100 GPUs.

For Agentic RL, we specifically select queries from Sec. 3.2 where best-of-N sampling failed to yield successful trajectories, thereby focusing RL on exploring and resolving these challenging cases. All video durations for RL are restricted to under 300 seconds. Following DAPO (Yu et al., 2025a), we employ a token-level policy loss with the clip-higher mechanism. The model is optimized for 150 steps with a group size of 8, a constant learning rate of $1 \times 1 0 ^ { - 6 }$ , and upper/lower clip ratios of 0.30 and 0.20, respectively. The global batch size is 256 on 64 NVIDIA A100 GPUs, and neither KL nor entropy regularization is applied.

Table 2. Main results on audio-visual understanding and reasoning. OmniAgent is evaluated on video benchmarks requiring joint audio-visual reasoning. Bold and underline indicate the best and second-best performance among open-source models, respectively. ∆ denotes the performance gain relative to the Qwen2.5- Omni baseline.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Size</td><td>DailyOmni</td><td>WorldSense</td><td>OmniVideo</td></tr><tr><td>AVG</td><td>AVG</td><td>AVG</td></tr><tr><td>Duration</td><td></td><td>43 sec</td><td>141 sec</td><td>384 sec</td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o (OpenAI, 2024)</td><td></td><td>56.5</td><td>42.6</td><td></td></tr><tr><td>Gemini-1.5-Pro (Gemini Team,2024)</td><td></td><td>二</td><td>48.0</td><td></td></tr><tr><td>Gemini-2.0-Flash (Gemini Team,2023)</td><td></td><td>56.1</td><td>-</td><td>41.5</td></tr><tr><td>Open-Source Models</td><td></td><td></td><td></td><td></td></tr><tr><td>Unified-IO-2 (Lu et al.,2024)</td><td>8B</td><td>28.2</td><td>25.9</td><td>1</td></tr><tr><td>VideoLLaMA2 (Cheng et al.,2024)</td><td>7B</td><td>35.2</td><td>25.4</td><td>29.2</td></tr><tr><td>MiniCPM-o (Yu et al.,2025b)</td><td>8B</td><td>53.1</td><td>1</td><td>29.7</td></tr><tr><td>Ola (Liu et al., 2025)</td><td>7B</td><td>50.7</td><td>1</td><td>1</td></tr><tr><td>Qwen2.5-Omni (Xu et al.,2025a)</td><td>7B</td><td>60.1</td><td>45.4</td><td>29.3</td></tr><tr><td>OmniAgent (Ours)</td><td>7B</td><td>64.8</td><td>47.2</td><td>37.1</td></tr><tr><td>△over Baseline</td><td></td><td>+4.7</td><td>+1.8</td><td>+7.8</td></tr></table>

## 4.2. Main Results

We evaluate OmniAgent across three task categories. Results are summarized in Tables 1, 2, and 3.

Video Understanding and Reasoning. OmniAgent-7B achieves state-of-the-art performance among open-source models on long-video benchmarks (Table 1). On LVBench, it scores 50.5%, substantially surpassing the Qwen2.5- Omni-7B baseline (43.0%) listed in Table 1. Notably, it even outperforms the 10× larger Qwen2.5-VL-72B (47.3%; see Figure 3) (Bai et al., 2025) and the agentic baseline Zoom-Zero-7B (45.7%) (Shen et al., 2025a), while using far fewer frames. Furthermore, OmniAgent outperforms recent “Thinking Models” like Video-R1 (60.9% on MLVU) (Feng et al., 2025). While thinking models rely on extended Chainof-Thought (CoT) reasoning over static inputs, OmniAgent (71.1% on MLVU) actively queries the environment to retrieve missing evidence. This suggests that for long-form video, the primary bottleneck is often perceptual incompleteness rather than reasoning depth. Compared to the passive baseline LongVU (Shen et al., 2025b) which uses dense sampling (1 FPS), OmniAgent achieves higher accuracy on VideoMME (+7.2%) and MLVU (+5.7%), validating that query-conditional active perception is more sample-efficient than uniform processing.

Audio-Visual Understanding and Reasoning. On benchmarks requiring joint perception (Table 2), OmniAgent outperforms the native omni-modal baseline Qwen2.5-

Table 3. Main results on audio-visual temporal grounding. We report temporal IoU score on LongVALE and VUE-TR. OmniAgent achieves state-of-the-art results among both proprietary and open-source models. Bold and underline indicate the best and second-best performance among open-source models, respectively. ∆ denotes the performance gain relative to the Qwen2.5-Omni baseline.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Size</td><td>LongVALE</td><td colspan="2">VUE-TR</td></tr><tr><td>IoU</td><td>Vision+Audio</td><td>Vision</td></tr><tr><td>Duration</td><td></td><td>233 sec</td><td>1066 sec</td><td>1114 sec</td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o (OpenAI, 2024)</td><td></td><td></td><td>11.1</td><td>22.0</td></tr><tr><td>Gemini-2.0-Flash (Gemini Team,2023)</td><td></td><td></td><td>18.4</td><td>25.4</td></tr><tr><td>Gemini-2.5-Pro (Comanici et al.,2025)</td><td></td><td></td><td>12.1</td><td>21.8</td></tr><tr><td>Open-Source Agentic Models</td><td></td><td></td><td></td><td></td></tr><tr><td>VITAL (Zhang et al.,2026)</td><td>7B</td><td></td><td></td><td>35.3</td></tr><tr><td>Open-Source Models</td><td></td><td></td><td></td><td></td></tr><tr><td>LongVALE-LLM(Geng et al.,2025)</td><td>7B</td><td>11.0</td><td>二</td><td>1</td></tr><tr><td>Vidi (Vidi Team,2025)</td><td>7B</td><td>1</td><td>33.4</td><td>43.9</td></tr><tr><td>Qwen2.5-Omni (Xu et al.,2025a)</td><td>7B</td><td>5.7</td><td>3.5</td><td>8.0</td></tr><tr><td>OmniAgent (Ours)</td><td>7B</td><td>39.1</td><td>36.5</td><td>46.1</td></tr><tr><td>△over Baseline</td><td></td><td>+33.4</td><td>+33.0</td><td>+38.1</td></tr></table>

Omni (Xu et al., 2025a) on DailyOmni (+4.7%) and OmniVideoBench (+7.8%). Passive models process modalities in a single pass, often trading off resolution for context. In contrast, OmniAgent utilizes auditory cues as temporal anchors to guide subsequent visual sampling, converting audio events into targeted visual search queries.

Temporal Grounding. OmniAgent shows substantial improvements in temporal localization (Table 3), achieving absolute gains over Qwen2.5-Omni on LongVALE (+33.4%) and VUE-TR (+33.0%). Notably, OmniAgent-7B surpasses proprietary models including GPT-4o and Gemini-2.5-Pro on VUE-TR. This performance stems from our iterative strategy: rather than regressing coordinates from a compressed global view, the agent employs on-demand sampling to progressively narrow the search space from coarse to fine granularity, ensuring higher precision.

Qualitative Analysis. To illustrate the interpretability of our framework, we visualize reasoning trajectories in Figure 1 and Appendix D. Figure 1 exemplifies queryconditional perception, where the agent leverages the temporal constraint (“22:03”) from the user query to steer its exploration, ignoring irrelevant segments. Extended case studies (Figures 6, 7, and 8) further demonstrate how the agent bridges information gaps through active audio-visual interactions across both reasoning and grounding tasks.

## 4.3. Ablation Study and Analysis

Impact of Training Paradigms. Table 4 presents the impact of shifting from passive to active perception. (1) Standard SFT: Fine-tuning on static QA pairs induces a performance regression on extremely long contexts (LVBench 43.0% → 41.6%). Lacking a selection mechanism, the passive paradigm suffers from information overload, where redundant frames reduce the signal-to-noise ratio as duration increases. (2) Agentic SFT: In contrast, Agentic SFT substantially improves long-video comprehension on LVBench (48.7%) and omni-modal understanding on Daily-Omni (60.1% → 63.3%), demonstrating the effectiveness of the $\left( O _ { k } , T _ { k } , A _ { k } \right)$ formulation.

Table 4. Component Ablation Study. We compare training paradigms and RL algorithms. Agentic SFT establishes the baseline for active perception, while TAURA mitigates the advantage homogenization problem of Vanilla GRPO, improving both understanding and reasoning tasks.
<table><tr><td rowspan="2">Method</td><td colspan="3">Video Understanding</td><td colspan="2">Omni-Modal Reasoning</td></tr><tr><td>VideoMMELong</td><td>LVBench</td><td>MLVU</td><td>OmniVideo</td><td>DailyOmni</td></tr><tr><td>Qwen2.5-Omni</td><td>54.8</td><td>43.0</td><td>65.2</td><td>29.3</td><td>60.1</td></tr><tr><td>+ Standard SFT</td><td>56.0</td><td>41.6↓</td><td>67.1</td><td>33.8</td><td>61.7</td></tr><tr><td>+ Agentic SFT</td><td>57.3</td><td>48.7</td><td>69.9</td><td>35.2</td><td>63.3</td></tr><tr><td>RL Stage (Initialized from Agentic SFT)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Vanilla GRPO</td><td>59.4</td><td>49.8</td><td>69.9</td><td>35.3</td><td>62.2↓</td></tr><tr><td>+ TAURA</td><td>59.6</td><td>50.5</td><td>71.1</td><td>37.1</td><td>64.8</td></tr></table>

![](images/73d2708609ba5fdd81d502f49ff3e7d6789e4077ef4d170266c05cd71802a07c.jpg)  
Figure 2. Test-time scaling on VideoMME-Long. Accuracy improves with the maximum turn limit (K), demonstrating positive scaling. The average number of turns executed saturates at 11.7 even when $K = 5 2$ , indicating that the model adaptively adjusts its reasoning depth based on information need rather than maximizing the turn count.

Efficacy of TAURA vs. Vanilla GRPO. (1) Advantage Homogenization: Vanilla GRPO stagnates on reasoning (MLVU 69.9%) and degrades perception (DailyOmni 63.3% → 62.2%). By broadcasting a uniform trajectorylevel advantage, it fails to incentivize specific perceptual discoveries, as critical “looking” actions receive the same credit as trivial actions. (2) Entropy-Steered Assignment: TAURA rectifies this by using entropy as a proxy for decision criticality. By up-weighting high-entropy turns—thereby identifying decisive “forks” in the search space—TAURA aligns credit assignment with information gain. This fine-grained supervision drives consistent improvements across both perception (DailyOmni 64.8%) and reasoning (MLVU 71.1%).

Table 5. Duration Analysis on LVBench. As video duration increases, the agent maintains stable accuracy despite a substantial drop in sampling density. This demonstrates that computational cost is driven by task complexity, not video duration.
<table><tr><td>Duration (min)</td><td>20-40</td><td>40-60</td><td>60-80</td><td>80-100</td><td>100-120</td><td>120-140</td></tr><tr><td>Count</td><td>309</td><td>420</td><td>278</td><td>302</td><td>173</td><td>67</td></tr><tr><td>Avg. Turns</td><td>8.5</td><td>9.9</td><td>11.3</td><td>11.2</td><td>10.8</td><td>12.5</td></tr><tr><td>Turns /Hour</td><td>16.9</td><td>11.9</td><td>9.7</td><td>7.5</td><td>5.9</td><td>5.7</td></tr><tr><td> Accuracy (%)</td><td>53.7</td><td>52.1</td><td>45.3</td><td>48.0</td><td>53.2</td><td>50.8</td></tr></table>

![](images/a19e15bbdbcaf57799fb2588fa98a6c3d02e4ab0175c9ed5a8b1eb0b02401bce.jpg)  
Figure 3. Accuracy vs. Visual Frame Count on LVBench. OmniAgent-7B (red diamond, 50.5%) outperforms the 10× larger Qwen2.5-VL-72B (47.3%) while using ∼73% fewer frames (203 vs. 768). Marker shape distinguishes agentic (diamond) from passive (circle) models, and marker size represents parameter scale.

Test-time Scaling Analysis. We analyze the scaling properties on VideoMME-Long by extending the maximum turn limit K from 6 to 52. Figure 2 reveals a monotonic performance improvement, where accuracy climbs from 53.4% to 59.6% (+6.2%). This confirms that deeper interaction enables the agent to uncover critical evidence for complex queries. Even as the upper bound expands by nearly 9×, the actual average turns saturate at merely ∼11.7. This indicates that the agent does not blindly exhaust the available budget, but emits its final answer via $a _ { \mathrm { a n s w e r } }$ once sufficient evidence is gathered, terminating the trajectory. This saturation suggests that the reasoning depth is driven by query complexity rather than the maximum turn limit.

Visual Sampling Efficiency. Figure 3 illustrates the tradeoff between accuracy and sampling cost, revealing a distinct Pareto efficiency for OmniAgent. (1) Parameter Efficiency (7B against 72B): OmniAgent-7B achieves peak accuracy (50.5%) while sampling only 203 frames on average. In sharp contrast, the 10× larger Qwen2.5-VL-72B (47.3%) relies on a dense input of 768 frames. This demonstrates that active perception acts as a computation multiplier: by selectively filtering noise, a 7B agent can achieve higher information density than a 72B passive model processing raw streams. (2) Contrast with Agentic Baselines: Unlike baselines like Zoom-Zero-7B (45.7%) that incur a fixed “entry cost” (256 frames) for initial scanning, OmniAgent operates on a strictly on-demand basis. This eliminates redundant processing, showing that iterative, query-conditional search is more efficient than global scanning. We further report inference runtime in Appendix C.4.

Impact of Video Duration. Table 5 validates that OmniAgent effectively decouples reasoning complexity from video length. As video duration grows over 4× (∼30 to ∼130 min), the absolute reasoning turns grow only marginally (8.5 → 12.5 turns), causing the sampling density to drop sharply from 16.9 to 5.7 turns per hour. Crucially, accuracy remains stable (50.8%) even at the lowest density. This confirms that the agent ignores redundant content (“the haystack”) to focus solely on critical evidence (“the needle”), allocating compute based on information density rather than temporal duration.

## 5. Conclusion

We present OmniAgent, which establishes active perception as an intrinsic reasoning process for omni-modal video understanding. Our central insight is that treating perception as iterative, query-driven information distillation, rather than exhaustive preprocessing, allows reasoning complexity to be driven by task difficulty rather than video duration. By bootstrapping active perception via Agentic SFT and refining it via Agentic RL with TAURA, we further show that native agentic capabilities can emerge within a single multimodal model without relying on external perception modules. Empirically, a 7B agent with active perception outperforms a 10× larger passive model while consuming far fewer frames, and exhibits positive test-time scaling where additional reasoning steps yield consistent gains. Beyond accuracy, the explicit Observation-Thought-Action (OTA) cycle provides interpretable reasoning traces, offering a transparent alternative to black-box video reasoning. While the sequential interaction loop introduces latency overhead, future work will investigate parallelized exploration to address this constraint.

## Acknowledgement

The work described in this paper was supported in part by the Research Grants Council of the Hong Kong Special Administrative Region, China, under Project CUHK 14202125 and Project CUHK 14200824.

## References

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

1, 6, 7

Brown, E., Yang, J., Yang, S., Fergus, R., and Xie, S. Benchmark designers should” train on the test set” to expose exploitable non-visual shortcuts. arXiv preprint arXiv:2511.04655, 2025. 4

Chen, B., Yue, Z., Chen, S., Wang, Z., Liu, Y., Li, P., and Wang, Y. LVAgent: Long video understanding by multiround dynamical collaboration of MLLM agents. In International Conference on Computer Vision (ICCV), 2025a. 3

Chen, Q., Di, S., and Xie, W. Grounded multi-hop videoqa in long-form egocentric videos. In AAAI Conference on Artificial Intelligence (AAAI), pp. 2159–2167, 2025b. 4

Chen, Y., Huang, W., Shi, B., Hu, Q., Ye, H., Zhu, L., Liu, Z., Molchanov, P., Kautz, J., Qi, X., et al. Scaling rl to long videos. In Advances in Neural Information Processing Systems (NeurIPS), 2025c. 4, 6

Chen, Y., Xue, F., Li, D., Hu, Q., Zhu, L., Li, X., Fang, Y., Tang, H., Yang, S., Liu, Z., et al. LongVILA: Scaling long-context visual language models for long videos. In International Conference on Learning Representations (ICLR), 2025d. 6

Cheng, J., Ge, Y., Wang, T., Ge, Y., Liao, J., and Shan, Y. Video-Holmes: Can mllm think like holmes for complex video reasoning? arXiv preprint arXiv:2505.21374, 2025. 4

Cheng, Z., Leng, S., Zhang, H., Xin, Y., Li, X., Chen, G., Zhu, Y., Zhang, W., Luo, Z., Zhao, D., and Bing, L. VideoLLaMA 2: Advancing spatial-temporal modeling and audio understanding in video-llms. CoRR, abs/2406.07476, 2024. 3, 6, 7

Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., et al. Qwen2-Audio technical report. arXiv preprint arXiv:2407.10759, 2024. 1

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6, 7

Fan, Y., Ma, X., Wu, R., Du, Y., Li, J., Gao, Z., and Li, Q. VideoAgent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision (ECCV), volume 15080, pp. 75–92, 2024. 1, 3

Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., and Yue, X. Video-R1:

Reinforcing video reasoning in mllms. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 6, 7

Fu, C., Lin, H., Long, Z., Shen, Y., Zhao, M., Zhang, Y., Wang, X., Yin, D., Ma, L., Zheng, X., He, R., Ji, R., Wu, Y., Shan, C., and Sun, X. VITA: Towards open-source interactive omni multimodal llm. ArXiv, abs/2408.05211, 2024. 3

Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24108–24118, 2025. 6

Gemini Team. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. URL https://gemini.google.com/. 7

Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 6, 7

Geng, T., Zhang, J., Wang, Q., Wang, T., Duan, J., and Zheng, F. LongVALE: Vision-audio-language-event benchmark towards time-aware omni-modal perception of long videos. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18959–18969, 2025. 4, 6, 7

Google DeepMind. Project Astra, 2024. URL https: //deepmind.google/technologies/gemin i/project-astra/. 3

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 5

Gupta, T. and Kembhavi, A. Visual programming: Compositional visual reasoning without training. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14953–14962, 2023. 3

Hong, J., Yan, S., Cai, J., Jiang, X., Hu, Y., and Xie, W. WorldSense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2025. 6

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al. LLaVA-OneVision: Easy visual task transfer. In Transactions on Machine Learning Research (TMLR), 2025. 6

Li, C., Chen, Y., Ji, Y., Xu, J., Cui, Z., Li, S., Zhang, Y., Tang, J., Song, Z., Zhang, D., et al. OmniVideoBench: Towards audio-visual understanding evaluation for omni mllms. In International Conference on Learning Representations (ICLR), 2026. 6

Li, J., Li, D., Savarese, S., and Hoi, S. C. H. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning (ICML), 2022. 1

Li, Y., Sun, H., Lin, M., Li, T., Dong, G., Zhang, T., Ding, B., Song, W., Cheng, Z., Huo, Y., Chen, S., Li, X., Pan, D., Zhang, S., Wu, X., Liang, Z., Liu, J., Zhang, T., Lu, K., Zhao, Y., Shen, Y., Yang, F., Yu, K., Lin, T., Xu, J., Zhou, Z., and Chen, W. Ocean-omni: To understand the world with omni-modality. arXiv preprint arXiv:2410.08565, 2024. 3

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-LLaVA: Learning united visual representation by alignment before projection. In Annual Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5971–5984, 2024. 1

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 1

Liu, J., Wang, Y., Ma, H., Wu, X., Ma, X., Wei, X., Jiao, J., Wu, E., and Hu, J. Kangaroo: A powerful videolanguage model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024. 6

Liu, Z., Dong, Y., Wang, J., Liu, Z., Hu, W., Lu, J., and Rao, Y. Ola: Pushing the frontiers of omni-modal language model. arXiv preprint arXiv:2502.04328, 2025. 7

Long, L., He, Y., Ye, W., Pan, Y., Lin, Y., Li, H., Zhao, J., and Li, W. Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory. In International Conference on Learning Representations (ICLR), 2026. 1, 3

Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., and Kembhavi, A. Unified-IO 2: Scaling autoregressive multimodal models with vision language audio and action. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26439–26455, 2024. 7

Ma, Z., Gou, C., Shi, H., Sun, B., Li, S., Rezatofighi, H., and Cai, J. DrVideo: Document retrieval based long video understanding. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18936–18946, 2025. 3

Meituan LongCat Team. LongCat-Flash-Omni technical report. CoRR, abs/2511.00279, 2025. 3

Meng, J., Li, X., Wang, H., Tan, Y., Zhang, T., Kong, L., Tong, Y., Wang, A., Teng, Z., Wang, Y., and Wang, Z. Open-o3 video: Grounded video reasoning with explicit spatio-temporal evidence. In International Conference on Machine Learning (ICML), 2026. 6

Nagrani, A., Menon, S., Iscen, A., Buch, S., Mehran, R., Jha, N., Hauth, A., Zhu, Y., Vondrick, C., Sirotenko, M., et al. Minerva: Evaluating complex video reasoning. In International Conference on Computer Vision (ICCV), 2025. 6

OpenAI. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024. URL https://openai .com/index/hello-gpt-4o/. 3, 4, 6, 7

Rasheed, H., Zumri, M., Maaz, M., Yang, M., Khan, F. S., and Khan, S. H. Video-CoM: Interactive video reasoning via chain of manipulations. In Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3, 6

Ren, W., Ma, W., Yang, H., Wei, C., Zhang, G., and Chen, W. Vamba: Understanding hour-long videos with hybrid mamba-transformers. In International Conference on Computer Vision (ICCV), 2025a. 6

Ren, W., Yang, H., Min, J., Wei, C., and Chen, W. VISTA: Enhancing long-duration and high-resolution video understanding by video spatiotemporal augmentation. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3804–3814, 2025b. 6

Shen, X., Chen, M.-H., Wang, Y.-C. F., Elhoseiny, M., and Hachiuma, R. Zoom-Zero: Reinforced coarse-to-fine video understanding via temporal zoom-in. arXiv preprint arXiv:2512.14273, 2025a. 1, 3, 6, 7

Shen, X., Xiong, Y., Zhao, C., Wu, L., Chen, J., Zhu, C., Liu, Z., Xiao, F., Varadarajan, B., Bordes, F., Liu, Z., Xu, H., J. Kim, H., Soran, B., Krishnamoorthi, R., Elhoseiny, M., and Chandra, V. LongVU: Spatiotemporal adaptive compression for long video-language understanding. In International Conference on Machine Learning (ICML), 2025b. 6, 7

Tang, C., Li, Y., Yang, Y., Zhuang, J., Sun, G., Li, W., Ma, Z., and Zhang, C. video-SALMONN 2: Captioningenhanced audio-visual large language models. CoRR, abs/2506.15220, 2025. 3

Tao, K., Du, W., Yu, B., Wang, W., Liu, J., and Wang, H. Active perception agent for omnimodal audio-video understanding. arXiv preprint arXiv:2512.23646, 2025. 3

Vidi Team. Vidi: Large multimodal models for video understanding and editing. arXiv preprint arXiv:2504.15681, 2025. 6, 7

Wang, Q., Yu, Y., Yuan, Y., Mao, R., and Zhou, T. VideoRFT: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2025a. 6

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X.-H., Yang, J., Zhang, Z., Liu, Y., Yang, A., Zhao, A., Yue, Y., Song, S., Yu, B., Huang, G., and Lin, J. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In Advances in Neural Information Processing Systems (NeurIPS), 2025b. URL https://openreview.n et/forum?id=yfcpdY4gMP. 5

Wang, W., He, Z., Hong, W., Cheng, Y., Zhang, X., Qi, J., Ding, M., Gu, X., Huang, S., Xu, B., et al. LVBench: An extreme long video understanding benchmark. In International Conference on Computer Vision (ICCV), pp. 22958–22967, 2025c. 6

Wang, X., Zhang, Y., Zohar, O., and Yeung-Levy, S. VideoAgent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision (ECCV), volume 15138, pp. 58–76, 2024. 3

Wang, Y., Yang, Y., and Ren, M. LifelongMemory: Leveraging llms for answering queries in egocentric videos. CoRR, abs/2312.05269, 2023. 3

Wang, Z., Yu, S., Stengel-Eskin, E., Yoon, J., Cheng, F., Bertasius, G., and Bansal, M. VideoTree: Adaptive treebased video representation for LLM reasoning on long videos. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3272–3283, 2025d. 3

Xing, Z., Hu, X., Fu, C.-W., Wang, W., Dai, J., and Heng, P.-A. EchoInk-R1: Exploring audio-visual reasoning in multimodal llms via reinforcement learning. arXiv preprint arXiv:2505.04623, 2025. 3

Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., Zhang, B., Wang, X., Chu, Y., and Lin, J. Qwen2.5-Omni technical report. CoRR, abs/2503.20215, 2025a. 1, 3, 4, 6, 7

Xu, J., Guo, Z., Hu, H., Chu, Y., Wang, X., He, J., Wang, Y., Shi, X., He, T., Zhu, X., Lv, Y., Wang, Y., Guo, D., Wang, H., Ma, L., Zhang, P., Zhang, X., Hao, H., Guo, Z., Yang, B., Zhang, B., Ma, Z., Wei, X., Bai, S., Chen, K., Liu, X., Wang, P., Yang, M., Liu, D., Ren, X., Zheng, B., Men, R., Zhou, F., Yu, B., Yang, J., Yu, L., Zhou, J., and Lin, J. Qwen3-Omni technical report. CoRR, abs/2509.17765, 2025b. 3

Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and Xie, S. Thinking in space: How multimodal large

language models see, remember, and recall spaces. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10632–10643, 2025a. 6

Yang, Q., Yao, S., chen, W., Fu, S., Bai, D., Zhao, J., Sun, B., Yin, B., Wei, X., and Zhou, J. HumanOmniV2: From understanding to omni-modal reasoning with context. CoRR, abs/2506.21277, 2025b. 3

Yang, Z., Chen, G., Li, X., Wang, W., and Yang, Y. DoraemonGPT: Toward understanding dynamic scenes with large language models (exemplified as A video agent). In International Conference on Machine Learning (ICML), 2024. 3

Yang, Z., Chen, D., Yu, X., Shen, M., and Gan, C. VCA: video curious agent for long video understanding. In International Conference on Computer Vision (ICCV), 2025c. 3

Yang, Z., Wang, S., Zhang, K., Wu, K., Leng, S., Zhang, Y., Li, B., Qin, C., Lu, S., Li, X., and Bing, L. LongVT: Incentivizing ”thinking with long videos” via native tool calling. In Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 1, 3, 6

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. DAPO: An open-source llm reinforcement learning system at scale. In Advances in Neural Information Processing Systems (NeurIPS), 2025a. 6

Yu, T., Zhang, H., Li, Q., Xu, Q., Yao, Y., Chen, D., Lu, X., Cui, G., Dang, Y., He, T., et al. RLAIF-V:: Open-source ai feedback leads to super gpt-4v trustworthiness. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19985–19995, 2025b. 7

Zeng, X., Li, K., Wang, C., Li, X., Jiang, T., Yan, Z., Li, S., Shi, Y., Yue, Z., Wang, Y., et al. TimeSuite: Improving mllms for long video understanding via grounded tuning. In International Conference on Learning Representations (ICLR), 2025. 6

Zhang, H., Gu, X., Li, J., Ma, C., Bai, S., Zhang, C., Zhang, B., Zhou, Z., He, D., and Tang, Y. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. In Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 1, 3, 6, 7

Zhang, P., Zhang, K., Li, B., Zeng, G., Yang, J., Zhang, Y., Wang, Z., Tan, H., Li, C., and Liu, Z. Long context transfer from language to vision. In Transactions on Machine Learning Research (TMLR), 2025a. 6

Zhang, X., Jia, Z., Guo, Z., Li, J., Li, B., Li, H., and Lu, Y. Deep video discovery: Agentic search with tool use for long-form video understanding. CoRR, abs/2505.18079, 2025b. 1, 3

Zhou, J., Shu, Y., Zhao, B., Wu, B., Liang, Z., Xiao, S., Qin, M., Yang, X., Xiong, Y., Zhang, B., et al. MLVU: Benchmarking multi-task long video understanding. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13691–13701, 2025a. 6

Zhou, Z., Wang, R., and Wu, Z. Daily-Omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2025b. 6

## A. Detailed Mathematical Notation

To ensure clarity across the multi-level hierarchy of the OmniAgent framework, we provide a structured summary of the mathematical notations used throughout the paper. The framework involves three primary scales: (i) the trajectory level (i), (ii) the interaction turn level (k), and (iii) the autoregressive token level (t).

Table 6. Summary of mathematical notations used in the OmniAgent framework.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td colspan="2">Interaction Hierarchy</td></tr><tr><td> $i , j$ </td><td>Indices for trajectories. i usually denotes the specific trajectory being optimized, while j is used as a summation index over the group  $G .$ </td></tr><tr><td> $k , m$ </td><td>Indices for interaction turns.k usually denotes the current turn, while m is used as a summation index over the total  $K _ { j }$ </td></tr><tr><td> $t$ </td><td>horizon Index for discrete tokens within an autoregressive sequence.</td></tr><tr><td> $K$ </td><td>Maximum number of interaction turns permitted (horizon).</td></tr><tr><td> $G$ </td><td>Group size for relative advantage computation (group size in GRPO).</td></tr><tr><td> $N _ { \mathcal { G } }$ </td><td>Total number of turns across all trajectories in a group  $( \Sigma _ { j = 1 } ^ { G } K _ { j } ) .$ </td></tr><tr><td colspan="2">Agent-Environment States</td></tr><tr><td> $\mathcal { E } _ { k }$ </td><td>Transient Percept: Raw multimodal data acquired from the environment Ω at turn  $k .$ </td></tr><tr><td> $\mathcal { M } _ { k }$ </td><td>Persistent Memory: The cumulative textual reasoning trace up to turn k.</td></tr><tr><td> $O _ { 0 }$ </td><td>Initial persistent memory state containing the query  $Q$  and video metadata.</td></tr><tr><td> $O _ { k } , T _ { k } , A _ { k }$ </td><td>The Observation,Thought,and Action triplet generated at turn  $k .$ </td></tr><tr><td> $\Omega$ </td><td>The interactive environment that resolves action  $A _ { k }$  into new percept  $\mathcal { E } _ { k }$ </td></tr><tr><td> $\pi \theta$ </td><td>The agent policy parameterized by 0.</td></tr><tr><td colspan="2">Reinforcement Learning(TAURA)</td></tr><tr><td> $R _ { i }$ </td><td></td></tr><tr><td> $A _ { i }$ </td><td>Final reward assigned to trajectoryi based on outcome verifiability. Trajectory-level Advantage: Normalized relative reward for trajectory i within its group.</td></tr><tr><td> $H _ { i , k }$ </td><td>Mean token entropy for the sequence generated during turn k of trajectory i.</td></tr><tr><td> $w _ { i , k }$ </td><td>Entropy Weight: The rescaled importance factor for turn k based on information density.</td></tr><tr><td> $\hat { A } _ { i , k }$ </td><td>Turn-aware Advantage: The TAURA-rescaled advantage applied to turn k of trajectory i.</td></tr><tr><td> $\rho _ { i , t }$ </td><td></td></tr><tr><td> $\mathrm { t u r n } ( t )$ </td><td>Importance sampling ratio for token t in trajectory i. Mapping function that assigns token t to its corresponding turn index k.</td></tr></table>

Index Mapping and Dependency. The policy $\pi _ { \theta }$ conditions its generation at turn k on the distilled memory $\mathcal { M } _ { k - 1 }$ and the previous transient percept $\mathcal { E } _ { k - 1 }$ . The internal logic of the framework ensures that raw media percept $\mathcal { E } _ { k - 1 }$ is purged from the active context window once it is distilled into the textual observation $O _ { k }$ , maintaining context efficiency while preserving the causal chain via $\mathcal { M } _ { k }$ . In the optimization phase, TAURA ensures that the global success signal $A _ { i }$ is steered toward the specific turns k where discovery (high entropy $H _ { i , k } )$ occurred.

## B. Agentic Audio-Visual Interaction Environment

This section provides the comprehensive implementation details of the environment Ω, focusing on the distributed infrastructure and robust media processing protocols required for reproducibility.

## B.1. Distributed Architecture via Ray and Verl

To ensure computational efficiency and prevent memory bottlenecks (OOM) during large-scale RL training, the environment is implemented using a distributed actor-based architecture powered by Ray and integrated with the Verl framework.

• Global Resource Singleton: We utilize a detached Ray actor (specifically named GlobalProcessor in our codebase) to manage heavy initialization tasks. This actor ensures that tokenizer weights and multimodal processor configurations are loaded into memory exactly once per physical node, significantly reducing the startup latency of concurrent worker processes.

• Parallel Worker Pool: The VideoQAMultiEnv orchestrates a pool of remote workers, each maintaining an independent instance of the SingleVideoQAEnv. This design allows for asynchronous perception and trajectory generation parallelized across CPU cores.

## B.2. Robust Perception Operators (FFmpeg Implementation)

The environment resolves sensing actions $A _ { k }$ into transient percept $\mathcal { E } _ { k }$ using specialized ffmpeg operators. We employ a two-stage seeking strategy (coarse seeking via keyframes followed by accurate decoding) to ensure sub-second precision.

• Visual Sampling $( a _ { \mathbf { f r a m e s } } ) \colon$ Frames are extracted using ffmpeg with -q:v 2 to maintain high visual fidelity. For queries near video boundaries, the environment employs an -sseof fallback mechanism to automatically adjust retrieval windows, preventing failures at terminal timestamps.

• Auditory Retrieval $( a _ { \mathbf { a u d i o } } ) { : }$ Segments are decoded into a standardized PCM S16LE mono-channel WAV format (16kHz). We enforce strict duration validation via ffprobe to ensure the retrieved audio waveform strictly aligns with the agent’s requested interval $[ s , e ]$

• Adaptive Clip Capture $( a _ { \mathbf { c l i p } } ) \colon$ Sub-clips are encoded using the libx264 codec with a Constant Rate Factor (CRF) of 20. To ensure interaction continuity under high system load, the environment implements a progressive fallback strategy: it prioritizes the superfast preset but automatically downgrades to $\mathtt { u l t r a f a s t }$ if encoding latency exceeds the timeout threshold.

## B.3. Exploration Incentives via Randomization

To improve policy robustness during Agentic SFT, we enforce randomized physical constraints sampled from discrete uniform distributions:

• Discrete Parameter Grid: Frame counts are sampled from $n \in [ 3 0 , 6 0 ]$ ] (step size 2); clip durations from $d _ { \mathrm { c l i p } } \in [ 3 0 , 6 0 ]$ seconds (step size 2); and audio durations from $d _ { \mathrm { a u d i o } } \in [ 1 5 0 , 3 0 0 ]$ seconds (step size 10).

• Diagnostic Error Protocols: Ω provides structured symbolic feedback (e.g., Err.TS OOB, Err.INVALID JSON). These signals allow the agent to acknowledge mistakes in the persistent memory $\mathcal { M } _ { k }$ and execute self-correction in subsequent turns.

## B.4. Memory Consolidation and History Purging

As formalized in Algorithm 1, the environment manages the persistent memory $\mathcal { M } _ { k }$ by purging media content from the interaction history. To decouple memory costs from video duration, the environment performs a non-destructive rewrite of the conversation trace:

1. Media Removal: Once a sensing turn k is completed and its textual distillation $O _ { k }$ is recorded, the environment iterates through the previous chat history. For every user message containing multimodal content, the environment deletes the dictionary entries of type image, video, or audio.

2. Metadata-Preserving Rewrite: To maintain the semantic integrity of the reasoning trace, the deleted media objects are replaced with a text-based summary appended with the marker [MEDIA OMITTED - Refer to Observation $O _ { k } ]$ :

• For Visual Sampling $( a _ { \mathrm { f r a m e s } } )$ , the environment recompiles the exact timestamps of all purged frames from the prior message content (e.g., “Frames 10.0s-20.0s. Timestamps: [10.00s, 12.50s, 15.00s] [MEDIA OMITTED...]”).

• For Audio and Video Clips $( a _ { \mathrm { a u d i o } } , a _ { \mathrm { c l i p } } )$ , the original headers containing the requested temporal intervals are preserved (e.g., “Audio 230.00s-280.00s [MEDIA OMITTED...]” or “Clip 45.00s-60.00s [MEDIA OMITTED...]”).

## B.5. Agent Instruction Template

Figure 4 presents the complete instruction template that governs agent-environment interaction across all stages of the OmniAgent pipeline, including trajectory synthesis (Sec. 3.2), reinforcement learning (Sec. 3.3), and inference. Runtime parameters (max steps, max frames len, max audio len, max clip len) are populated according to the configuration in Sec. 4.

Native Active Perception as Reasoning for Omni-Modal Understanding  
![](images/f76ff02b851535c8e5f0614f210ab99872c91147dc0cd62a8fc4e1e34e0df910.jpg)  
Figure 4. The complete agent instruction template used across all stages of the OmniAgent pipeline.

## C. Empirical Analysis: Entropy as a Proxy for Reasoning Criticality

To validate the motivation behind TAURA, we conducted a detailed analysis of the agent’s reasoning traces to quantify the relationship between model uncertainty (entropy) and reasoning criticality.

## C.1. Methodology: Identifying Decision Forks

We hypothesize that in a multi-turn agentic trajectory, steps are not created equal. Some represent Decision Forks—pivotal junctures where the agent dictates a subsequent search strategy or significantly narrows the hypothesis space—while others are routine execution steps.

To identify these forks without human bias, we employed a model-based evaluation. For successful trajectories in the VideoMME benchmark, we provided an expert evaluator (Gemini-2.5-Pro) with the query, the full interaction trace, and the final outcome. The evaluator was prompted to identify the single ”Top-1 Fork Step” $\left( k _ { \mathrm { f o r k } } \right)$ based on the definition: ”A pivotal juncture in a multi-step reasoning process that dictates the subsequent logical trajectory... where the process diverges into multiple potential paths.”

## C.2. Quantitative Analysis: The Entropy Gap

We computed the mean token entropy for each turn k in a trajectory i, denoted as $H _ { i , k }$ . We then calculated the difference between the entropy of the identified Fork Step $( H _ { i , k _ { \mathrm { f o r k } } } )$ and the average entropy of that entire trajectory $( \overline { { H _ { i } } } )$

Figure 5 (a) presents the distribution of the entropy difference $\Delta H = H _ { i , k _ { \mathrm { f o r k } } } - \overline { { H _ { i } } } .$

• Positive Shift (79.2%): The vast majority of identified fork steps exhibit a positive entropy difference. This confirms that when the agent makes critical decisions (e.g., pivoting from scanning to verification), its policy distribution becomes flatter (higher uncertainty), reflecting the active weighing of potential reasoning paths.

• Negative/Neutral (20.8%): The minority of cases where the fork step entropy is lower or equal to the mean typically correspond to short-horizon trajectories or easy queries where the reasoning path is linear and deterministic.

## C.3. Case Study: Entropy at a Fork Step

To illustrate this phenomenon, we analyze a specific trajectory regarding the query: ”Which company is featured in the video but not mentioned in the audio?” (Figure 5 (b)).

1. Routine Scanning: The agent performs standard scanning. The policy is confident in this information gathering, resulting in low entropy $( H \approx 0 . 3 9 7 )$

2. The Fork Step: A distinct entropy spike $( H \approx 0 . 9 2 7 )$ occurs when the agent identifies that options A, B, and C are explicitly mentioned in the audio. This observation acts as a critical filter, ruling out these candidates. The spike reflects the agent’s pivotal decision to switch modalities (‘get frames‘) to visually verify the presence of the remaining option (American Express), rather than concluding immediately.

3. Resolution: Subsequent verification resolves the ambiguity, returning the agent to a lower entropy state $( H \approx 0 . 7 9 0 )$

This confirms that entropy spikes serve as a reliable signal for identifying high-value reasoning steps that warrant amplified reinforcement.

![](images/447dc4caa5ced1914fe174a1e56e275692e1cb12d6f35beb770f664158991e11.jpg)  
(a) Quantitative Analysis: Entropy Shift Distribution

![](images/46fd84f1f6619b2df28cbd395a3d4997a0025144744a3ce9d4deb7419b28ea7e.jpg)  
(b) Qualitative Case Study: Entropy Spike at Fork Step  
Figure 5. Correlation between Entropy and Critical Reasoning Steps. (a) Histogram of the entropy difference $\Delta H = H _ { i , k _ { \mathrm { f o r k } } } - \overline { { H _ { i } } }$ 79.2% of critical fork steps exhibit higher uncertainty than the trajectory mean. (b) A qualitative case study on the ”Company” query. A distinct entropy spike (0.927) occurs at the ”Fork Step” where the agent processes audio evidence (ruling out A, B, C) and decides to pivot to visual verification.

## C.4. Inference Runtime and Action Analysis

Inference Runtime Analysis. Table 7 reports measured wall-clock latency on a 100-sample LVBench subset. OmniAgent achieves lower wall-clock latency than Qwen2.5-VL-72B (66.8 s vs. 75.1 s) while reaching higher accuracy (51.0% vs. 47.0%). Compared to LongVT, an agentic baseline, OmniAgent is both more accurate and slightly faster. Qwen2.5-Omni-7B remains fastest in single-shot inference (34.8 s) but at substantially lower accuracy (41.0%). Notably, Qwen2.5-VL-72B requires 4× A100 GPUs, whereas OmniAgent requires only 1.

Table 7. Inference Runtime on LVBench (100 samples). Wall-clock latency (seconds) and accuracy. OmniAgent achieves a favorable accuracy–latency tradeoff.
<table><tr><td>Method</td><td>Frames</td><td>Model (s)</td><td>Wall-clock (s)</td><td>Acc. (%)</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>201</td><td>34.8</td><td>34.8</td><td>41.0</td></tr><tr><td>Qwen2.5-VL-72B</td><td>768</td><td>75.1</td><td>75.1</td><td>47.0</td></tr><tr><td>LongVT</td><td>793.8</td><td>64.1</td><td>67.6</td><td>42.0</td></tr><tr><td>OmniAgent (Ours)</td><td>201.6</td><td>56.0</td><td>66.8</td><td>51.0</td></tr></table>

## D. Qualitative Analysis

![](images/ed72decd5cf80d84899a631265118bb1f2138b8df4f0f9d7155e191578d24491.jpg)

[Several OTAs omitted: OmniAgent performs broad temporal sampling over the full ∼140-minute video, then narrows around a suspected split-screen studio moment; the initial clips show no green skirt and no readable name tag.]

O: I have identified a male host in a grey sweater (Studio A) and a male host in a grey suit (Studio B). I noticed a female host in a red jacket and another female host with curly hair. I missed checking the visual details of the first 200 seconds which might contain the studio intro. I need to locate a host wearing a green skirt. Potential segments are the intro, a roundtable discussion, or a different studio segment.

T: The studio segments are the most likely place for a host introduction. I will scan the first 200 seconds with higher density to identify the hosts and check their clothing. I will also listen to the audio in these segments to catch any introductions. A: get frames(start=0.0, end=200.0, num=50)

O: At 8.16s and 175.51s, a female host wearing a vibrant green dress/skirt is clearly visible standing next to two male hosts in the exhibition hall. She has dark hair. I need to verify her name by listening to the audio introduction or checking on-screen text if available.

T: I have located the host in the green skirt visually. Now I need to identify her name. The introduction usually occurs at the beginning of the clip. I will listen to the audio from 8.0s to 18.0s to hear if she is introduced.   
A: get audio(start=8.0, end=18.0)

O: Audio 8s-18s contained a teaser montage. I need the main intro audio to catch the name. Visually, the host in the green skirt was seen around 175s.

T: I will retrieve the audio segment immediately following the teaser (18s-48s) to catch the main introduction of the hosts. I am listening for a female voice introducing herself or being introduced.

A: get audio(start=18.0, end=48.0)

[Several OTAs omitted: OmniAgent triangulates around the host’s repeated appearances (e.g., near 170s and around the split-screen at ∼424s), checks for lower-thirds, and then shifts to searching earlier intro-adjacent windows where name graphics commonly appear.]

O: I need to confirm the identity of the female host in the green dress. I have not yet seen a text lower-third introduction in the intro sequence (0-30s). The host appears clearly at 6.21s and 14.48s. I will check the next segment of the video for a lower-third graphic that might have appeared after the intro montage.

T: I will scan frames from 48s to 160s to find any lower-third graphics identifying the hosts. This is a common place for an introduction to pop up.

A: get frames(start=48.0, end=160.0, num=30)

O: At frame 55.72s, there is a clear lower-third graphic with the name “Nish Parkar” under the image of the female host in the green dress. She is wearing a green dress. She is identified as “Cisco Live Host #CiscoLiveEMEA”. This matches option C. T: I have found the definitive evidence. The lower third graphic at 55.72s identifies the female host in the green dress as Nish Parkar. I have also seen her earlier in the video (around 8s, 175s) in a green dress. This matches Option C. I will select Option C. A: answer(content="C")

Figure 6. Qualitative Example of Active Reasoning (MCQ). OmniAgent first narrows the search space by revisiting the early broadcast window where host intros and lower-thirds typically occur, then visually confirms the green dress/skirt host at multiple timestamps. When audio checks yield only montage narration, it switches to a targeted scan of the intro-adjacent segment and locates the decisive lower-third text “Nish Parkar”, grounding the final choice in on-screen evidence and selecting option C.

<table><tr><td rowspan=2 colspan=4><img src="images/729829a9c5a86221b1e901aac022a6a8887e32480493c3b7418ca4cbbcb0023a.jpg"/></td></tr><tr><td rowspan=1 colspan=4>O: Initial scan initiated to locate segments discussng Kyo-yuzen technique.T: The video is long (44 mins). Ineed to find sections discussing the Kyo-yuzen technique.I will perform a broad scan to identifyscene changes and potential text overlays or speakers.A:get_frames(start=0， end=2645.51， num=60)</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>O: Frames 403-582 cover the “Kyo-kanoko shibori” segment.I suspect “Kyo-yuzen” might be introduced in this context or nearby.Frame 762 shows a pattern that might be Kyo-yuzen. I need to verify the“1O stages”claim and the technique details.T: I willinspect the audio in the shibori section to see if they mention “Kyo-yuzen”and its stages. If not found,I will scan theKyo-yuzen section if it exists.A:get_audio(start=400,end=600)</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>O: The audio confirms the segment at 40Os-60Os is about “Kyo-kanoko shibori”, not Kyo-yuzen. They do mention“10o,0o0 beads&quot;(which relates to counting, not stages).I need to find the actual discussion on Kyo-yuzen to test the options.T: I need to locate the specific section mentioning “Kyo-yuzen&quot;. The video seems to progress chronologically. Since 400s wasKyo-kanoko,I willscan forward to find the Kyo-yuzen segment,likely after a break.Iwillsample frames in the 6OOs-12OOs range.A:get_frames(start=600，end=1200，num=50)</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=2>[Several OTAs omitted: OmniAgent systematically scans the video in large chunks (e.g. 1200-2400s, etc.) without findingthe “Kyo-yuzen” segment.]</td></tr><tr><td rowspan=1 colspan=2>O:I have thoroughly scanned the 160Os-18OOs range and found it to be the“Sushi’ segment. Ihave analyzed the“Kyo-kanoko&quot;(400-600s) and “Kirikane”(75O-105Os) segments.Ihave not found a dedicated“Kyo-yuzen” segment. To find the false statementabout Kyo-yuzen,I must identify its features (e.g.,is it painted on cloth or silk? does it involve 1O stages?). I will scan the 5Os-150srange to se if Kyo-yuzen appears there, as Imay have missed it in the initial sparse scan.T:I need to locate Kyo-yuzen.Iwill check the 5Os-15Os range.Iam looking for the specific Kyo-yuzen segment to evaluate theoptions.A:get_frames(start=50,end=150， num=50)</td></tr><tr><td rowspan=4 colspan=1></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>contains 10</td><td></td></tr><tr><td rowspan=1 colspan=2>for the paintingg</td><td></td></tr><tr><td rowspan=1 colspan=2>chetc). th”. K</td><td rowspan=1 colspan=1>Kvo</td></tr></table>

Answer with time ranges and do not output explanation. What are all the time ranges corresponding to the text query: "John Brassard was not happy about the meeting and adjourns the session by striking the gavel"? GT: [1603, 1607] OmniAgent: [1603, 1607]

![](images/92d383adbe402d3753895661303e3d219a49e71157cf29a3977dac9c73ba65d1.jpg)

O: Initial scan initiated to locate the scene involving John Brassard and the gavel strike.

T: The video is long (26 minutes). I need to find the specific scene where a meeting is taking place and a gavel is used. I will start by scanning the entire video with 60 frames to identify the relevant segment.   
A: get frames(start=0, end=1616.1, num=60)

O: The video features John Brassard, a Chair/President, frequently. I see him gesturing and speaking in various segments. I suspect the event of him adjourning the session happens during a heated moment.

T: The prompt describes a specific action: “striking the gavel” because he is “not happy”. I need to find the audio or visual of a gavel strike. Since I can’t search for specific text, I will search for the audio cue of a gavel strike or the context of an altercation. I’ll sample audio in the segments where he is speaking to identify the moment of conflict.

A: get audio(start=400.0, end=700.0)

[Several OTAs omitted: OmniAgent probes multiple short windows around ∼700s with get frames/get clip to verify the suspected gavel cue, but fails to find an on-camera gavel strike.]

O: Frame 756.32s clearly shows John Brassard, Chair (President), seated with the gavel visible on the desk. He appears calm, holding a pen. The clip ends at 760s. The previous clip (705-725s) showed him interacting with another speaker (Parm Bains) but no gavel strike. The audio cue at 700s might have been a false positive or a background sound. I need to find a moment where he is visibly agitated or gesturing emphatically, followed by a strike.

T: The 700s segment was a misinterpretation of the audio or a different scene. I need to search for another instance of Brassard speaking or being on camera. The 400-700 segment was a debate. I’ll scan the later part of the video (1350-1600s) where he might have one last word or a different session ends. He is the Chair, so he calls adjournment.

A: get frames(start=1350.0, end=1616.0, num=30)

O: Frames 1368-1588 show John Brassard gesturing and speaking agitatedly (pointing finger, touching head, hand on desk). This matches the “not happy” description. I need to find the specific moment he strikes the gavel to conclude the session.   
T: The visual evidence from 1368s to 1588s confirms he is “not happy”. The prompt states he “adjourns the session by striking the gavel”. I haven’t seen the strike yet. It likely occurs at the end of this rant, either around 1600s (before the video ends) or right after he stops speaking. I will examine the end of the video.   
A: get clip(start=1600.0, end=1616.0)

O: The clip from 1600s to 1616s captures the exact sequence: John Brassard declares “I am NOT happy with the way this went” (approx 1603s), immediately states “I am adjourning the meeting” (approx 1605s), and hits the gavel (approx 1606s). The clip ends with an outro logo.

T: I have located the exact moment. The sequence begins with his declaration of unhappiness and ends with the gavel strike. The timestamps align with the speech and the physical impact.

Figure 8. Qualitative Example of Temporal Grounding. OmniAgent combines coarse full-video scanning with targeted audio checks to propose a candidate gavel event, actively falsifies the early false lead via localized probing, then pivots to a later visually-agitated segment and confirms the precise adjournment moment by querying the terminal clip. It outputs the exact interval covering “I am NOT happy” → “I am adjourning the meeting” → gavel strike: [[1603.0, 1607.0]].