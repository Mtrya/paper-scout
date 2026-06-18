# Kairos: A Native World Model Stack for Physical AI

Learning, Maintaining, and Running Worlds for Future Self-Evolving Agents

Kairos Team

## Abstract

World models are transitioning from passive visual generators to foundational, operational infrastructure for Physical AI: they must natively acquire world knowledge from heterogeneous experience, maintain persistent states over long horizons, and execute efficiently within real deployment constraints. We introduce Kairos, a native world model stack designed around these requirements. (1) Kairos learns the world by pioneering a Native Pre-training Paradigm governed by a Cross-Embodiment Data Curriculum, which organizes open-world videos, human behavioral data, and robot interactions into a progressive developmental pathway. (2) Kairos maintains the world by unified world understanding, generation, and prediction within a Native Unified Architecture equipped with Hybrid Linear Temporal Attention, where sliding-window attention captures local dynamics, dilated sliding windows capture mid-range dependencies, and gated linear attention maintains persistent global memory. We establish formal theoretical bounds demonstrating that this temporal factorization strictly limits error accumulation, mathematically guaranteeing state propagation across extended horizons. (3) Kairos runs the world by incorporating a Deployment-Aware System Co-Design to support low-latency rollout generation on server and consumer-grade hardware for real-world observation–action–feedback loops. Experiments on embodied world-model, long-horizon, and action-policy benchmarks show that Kairos achieves top level performance while offering a strong efficiency–capability trade-off. Together, these results position Kairos as a cohesive operational foundation for future self-evolving physical intelligence.

Date: June 17, 2026

Code: https://github.com/kairos-agi/kairos-sensenova

Hugging Face: https://huggingface.co/kairos-agi

ModelScope: https://modelscope.cn/collections/kairos-team/kairos30

## Contents

1 Introduction 4   
2 Model 9   
2.1 Native Architecture with Unified Understanding, Generation and Prediction 9   
2.2 Efficient Diffusion Transformer with Hybrid Linear Attention 12   
2.2.1 Long-term: Gated Linear Attention 13   
2.2.2 Short(Mid)-term: (Dilated) Sliding Window Attention 15   
2.3 Theoretical Analysis of Hybrid Multi-Scale Temporal Memory 16   
3 Native Pretraining Paradigm for Physical AI . 18   
3.1 Native Pretraining with Cross-Embodiment Data Curriculum 19   
3.2 Stage I: Physical Pretraining 2 1   
3.3 Stage II: Embodied Pretraining with Human-centric Data 24   
3.4 Stage III: Joint World-Action Training 25   
3.5 Training Infrastructure 27   
4 Data 27   
4.1 Data Collection 27   
4.2 Data Curation 28   
4.3 Tagging and Captioning 31   
4.3.1 Tagging 31   
4.3.2 Captioning 32   
4.3.3 Enhanced Text with Chain of Thought 3 3   
4.4 Data Engineering Infrastructure . 34   
5 Inference 35   
5.1 Self-evolution 35   
5.2 Prompt Self-alignment 35   
5.3 Inference Efficiency 37   
5.3.1 Timestep Distillation 37   
5.3.2 Hardware-aware Inference Optimization 40   
5.3.3 Efficiency Comparison 4 1   
6 Evaluation Results . 42   
6.1 Embodied World Model Benchmarks 43   
6.1.1 Benchmark Results 43   
6.1.2 Ablation Studies 47   
6.2 World Action Model Benchmarks 48   
6.2.1 Benchmark Results . 48   
6.2.2 Ablation Studies 50   
6.3 General World Model Benchmarks 51   
6.4 Long-horizon Generation 55   
Related Work 57   
7.1 Video Generation Models 57   
7.2 World Models . 58   
7.3 World Action Models . 58   
7.4 Efficient Attention Mechanisms 59   
8 Conclusion and Future Works 60   
Appendix 75   
A Contributors 75   
B Theoretical Analysis 76   
B.1 Problem Setup and Theoretical Scope 76   
B.2 Necessity of Persistent Latent States 78   
B.3 Approximate Sufficiency of a Hybrid Multi-Scale Temporal Memory . 82

## 1 Introduction

World models [1] are rapidly emerging as a central substrate for physical AI. Beyond recent progress in video generation, they are increasingly expected to support physical understanding, temporal prediction, embodied interaction, and, ultimately, continual adaptation in real environments. In this broader view, a useful world model is not merely a generator of visually plausible futures, but an internal system for acquiring, organizing, and updating knowledge about how the world evolves under action and time. Recent progress across academia and industry indicates a paradigm shift: from generative world models as demonstrations to world models as infrastructure for embodied and physical intelligence.

To understand this shift, recent advancements in world models can be broadly categorized into three dominant streams. The first stream focuses on generative pixel-level rendering, which primarily encompasses video generation and the synthesis of high-fidelity visual futures. Models in this category aim to produce temporally coherent continuations of the world directly in pixel space. A prominent example is NVIDIA’s Cosmos [2], which leverages generative video foundation models as digital twins and essential infrastructure for physical AI.

A second stream shifts the focus toward predictive latent embedding (representation) learning. Rather than rendering pixels, this approach explicitly frames world models as systems that learn physically meaningful predictive structures entirely within abstract representation spaces. Meta’s JEPA family (e.g., V-JEPA 2 [3], V-JEPA 2.1 [4], and DINO-world [5]) exemplifies this trajectory. By internally anticipating outcomes in a latent form, these models inherently support downstream tasks such as physical understanding, zero-shot planning, and robot control. The core premise here is that a world model’s utility for decision-making relies on its capacity to anticipate the future abstractly, bypassing the immense computational overhead of pixel-level rendering.

The third direction advances interactive environment modelling, emphasizing the creation of persistent simulations and interactive arenas. This encompasses both static spatial spaces and dynamic interactive environments. For instance, models focusing on static spatial intelligence, such as World Labs’ Marble [6] and TeleWorld [7], excel at building explorable 3D worlds that agents can perceive and navigate, emphasizing geometric consistency and "worldness." Extending into dynamic interaction, environment generators like DeepMind’s Genie 3 [8], HY-World 1.5 [9], and LingBot-World [10] instantiate fully manipulable worlds from simple prompts. Furthermore, frameworks like Dreamer 4 [11] utilize these models as internal simulators where agents can recursively optimize long-horizon behaviors through imagination. In this paradigm, models are judged by their capability to act as comprehensive engines for spatial exploration, interaction, and self-evolution.

Taken together (see Figure 1), these advances show that the field is no longer converging on a single definition of world models as “video generators.” Instead, world models are being asked to serve as a foundation model, a customizable substrate that supports simulation, synthetic data generation, downstream adaptation, and deployment for robotics and autonomous systems. This expansion of ambition is precisely what makes the next set of challenges both urgent and difficult.

• The first major challenge is fragmented world learning across heterogeneous experience sources. The knowledge required for a capable world model is distributed unevenly across open-world videos, human behavioral traces, and scarce robot interaction data. Open-world videos offer broad physical and environmental regularities, but typically lacks action grounding and task intent. Human data reveals structured behavior and interaction patterns, but does not directly align with robot embodiment or control spaces. Robot data is the most relevant for embodied operation, yet remains expensive, narrow, and difficult to scale. As a result, current systems often optimize on one form of experience while underutilizing the others, learning partial competencies rather than unified world knowledge. This is difficult not merely because the datasets are large, but because the knowledge forms themselves are mismatched across perception, behavior, and embodiment. The consequence is substantial: models may learn broad physical priors without actionable grounding, or narrow robot skills without generalization, preventing them from serving as transferable substrates for embodied intelligence.

![](images/a1a7944187499f8e715e85057f901e9872c3b9393ead4b5424625839fc3b7afa.jpg)  
Figure 1 Motivation of Kairos. It is not merely a generative model but a deployable infrastructure natively designed for future self-evolutionary learning in Physical AI.

• The second major challenge is maintaining persistent world states over long temporal horizons. Short video continuation can often rely on local visual smoothness, but world modeling requires something harder: preserving the same world over more time. This includes maintaining object permanence, delayed physical effects, multi-stage interactions, latent task progress, and causal consequences that may unfold only after substantial temporal gaps. Many current systems perform well on short-horizon appearance transitions yet degrade as duration grows, because local continuation heuristics do not guarantee global state consistency. As we formally prove in this report, this degradation is an unavoidable structural limitation. Purely local continuation heuristics are mathematically insufficient to guarantee global state consistency and inherently incur an irreducible excess risk. The difficulty is both representational and computational. Representationally, the model must retain the right abstract state variables rather than merely re-render recent observations. Computationally, dense temporal attention becomes increasingly prohibitive as duration and resolution grow, while simple autoregressive rollout accumulates drift and inconsistency. As a result, apparently strong long-video models may still fail as world models because they do not reliably preserve the world they are meant to simulate.

• The third challenge is the gap between world understanding and embodied control. A model may predict or generate plausible futures, but fails to learn how actions from an embodied agent change those futures in a reliable, controllable way. This gap arises because observation, behavior, and control occupy different representational regimes: open-world videos rarely specify action-conditioned transitions, human demonstrations follow patterns that do not directly map to robot actuation, and robot interaction datasets are often too narrow to induce broad transferable priors. Consequently, many world models remain spectators of the world rather than participants in it. They may support visual forecasting or environment generation, yet still fail to provide stable perception-action grounding for downstream embodied decisionmaking. This limitation directly affects robotics and physical AI, where the central question is not just what will happen next, but what will happen if this agent acts now.

• The fourth challenge is deployment and closed-loop operation under real-world constraints. Even a strong world model in offline evaluation may be unusable in practice if it cannot run with sufficiently low latency, low memory overhead, and realistic communication costs. This is not a peripheral engineering issue. For physical AI, and especially for any future form of selfevolution learning, the model must be able to participate in observation-action-feedback loops in real time or near real time. If inference is too slow, too memory-intensive, or too tightly tied to impractical hardware assumptions, the model cannot be embedded into real systems, cannot accumulate online corrective feedback, and cannot support continual embodied adaptation. In other words, without deployment-ready operation, world models remain demonstrations rather than infrastructure. This is why recent infrastructure-oriented efforts, such as Cosmos and Dreamer 4, place unusual emphasis on scalable operation and efficient inference rather than on model quality alone.

These challenges are tightly coupled. Fragmented learning makes it difficult to acquire coherent world knowledge; weak long-horizon state maintenance makes it difficult to preserve that knowledge over time; poor embodiment grounding makes it difficult to connect world knowledge to control; and the absence of deployment-ready execution prevents any of these capabilities from entering real observation–action–feedback loops. Addressing them separately risks producing systems that are strong along one dimension but fundamentally incomplete as substrates for physical intelligence.

Kairos is designed precisely around this bottleneck structure. Rather than treating these issues as separate engineering concerns, we address them jointly through a native world-action model stack that learns the world through progressive cross-embodiment experience, maintains the world through consistent temporal attention mechanism, and runs the world through deployment-aware system co-design. Besides, fundamentally departing from the prevalent yet disjointed practice of post-training or fine-tuning generic open-domain video generators for downstream embodied control, we pioneer a Native Pre-training Paradigm for Physical AI, championing the philosophy that general physical laws, behavioral semantics, and embodied grounding must be natively synthesized within the foundational architecture from the very inception of scaling, establishing a genuinely cohesive, deployment-aware world-action infrastructure. In this sense, Kairos is not only a stronger world model, but a step toward a world-model substrate that is operationally ready for future self-evolution learning in physical AI.

The core contributions of this work are organized around three foundational echelons:

• Our first contribution is the establishment of A Native Pre-training Paradigm via Cross-Embodiment Data Curriculum (CEDC). We reject the decoupled post-training fine-tuning approach and instead propose a native pre-training paradigm that injects physical and behavioral knowledge from scratch. To operationalize this native injection across highly disparate data domains, self-evolution-inspired CEDC serves as the structural backbone of our pre-training phase, and organizes them into a progressive hierarchy. General videos provide broad physical and environmental regularities; human data contributes structured behavioral patterns and task organization; robot data grounds perception-action alignment and embodied control. Kairos moves beyond flat data scaling to propose a developmental world-knowledge curriculum. By organizing heterogeneous experience—from open-world observation to human imitation and robot embodiment—into a progressive hierarchy, the model systematically acquires internal representations that evolve from passive physics to active task-intent.

![](images/e35ac97859bde38de941474d25e7f3b78a4ffdbe7b31436d3736c573d81d27f9.jpg)  
Figure 2 Framework of Kairos.

• Our second contribution is a Native Understanding-Generation-Prediction Architecture with Hybrid Linear Temporal Memory. Rather than framing long-horizon modeling as a pure video continuation problem, Kairos treats it as a persistent world-state task within a unified Mixture-of-Transformers (MoT) stack, where understanding supplies causal interpretation, generation unfolds physically plausible futures, and prediction outputs hardware-deployable action trajectories. To guarantee long-horizon state maintenance under linear computational complexity, we factorize temporal modeling via a hybrid attention mechanism: Sliding-Window Attention (SWA) anchors local dynamics, Dilated Sliding-Window Attention (DSWA) captures mid-range interactions, and Gated Linear Attention (GLA) serves as a contractive global causal memory. Crucially, we establish formal theoretical bounds demonstrating that this architectural factorization strictly limits error accumulation, mathematically guaranteeing persistent state propagation across extended horizons.

• Our third contribution is a Deployment-Aware System Co-Design that treats real-time edge-side execution as a first-order modeling principle. We argue that for world models aspiring to future closed-loop self-evolution, systems optimization is not a post-hoc acceleration luxury, but an operational necessity. If inference latencies or memory footprints block the model from entering real observation–action–feedback loops, continual adaptation remains unattainable. By co-designing hardware-aware compute kernels, quantization protocols, and token streaming, Kairos sustains high-throughput, low-memory inference on consumer-grade hardware. This optimization establishes the definitive operational conditions under which real-world deployment and future self-evolutionary scaling can practically occur.

WorldModelBench Robot  
![](images/055296764eab4a321ef73ebfc9a1833b944aa277114088a97b1c08a4fe1c5ddc.jpg)

![](images/395a39f2b58daa1627645476e317c669bd627866ffba5272ec7760ef10595d5e.jpg)

(a) Performance comparison across world action model benchmarks  
![](images/1626e0be502ba5c9914da74b9ff1fdcb461e46345d5ab672cb1eb242c3937e75.jpg)

![](images/f9a2d5733c03e09b7a3f3e98e6da1af258273b449acdfea2dca41c82d6943aa0.jpg)

(b) Performance comparison across embodied world model benchmarks  
![](images/451d4267bb35d2e5811977b3e69f5262c67765c7dc4871a7f76fba5d7e2496b9.jpg)

![](images/1e5aff3ade318d9bda44b73f0e14e14477e3263f4f8f349feb41c9c13bbc3700.jpg)  
(c) Inference time comparison per DiT step  
Figure 3 (a) (b) Kairos achieves state-of-the-art performance across diverse both embodied world model and world action model benchmarks while delivering significant efficiency superiority over baseline models. (c) Notably, Kairos scales linearly (see the zoom window for the DiT inference time per step), ensuring consistent throughput for long-duration generation.

Crucially, these three pillars should not be read as isolated engineering additions, but as a deeply coupled response to the structural bottlenecks that prevent world models from becoming operational substrates for continuous self-evolution. Future physical agents cannot achieve autonomous selfevolution within static, decoupled frameworks. Kairos systematically resolves this by framing the world-modeling challenge around an evolvable capability chain that answers three fundamental questions: how to learn, maintain, and run worlds.

First, the Cross-Embodiment Data Curriculum (CEDC) defines how the model learns the world, mitigating the mismatch between broad ungrounded observations and narrow robotic actions. This structured data pyramid establishes a development-aligned progression—from passive physics to human behavior and active embodiment—providing a scalable knowledge pipeline for autonomous data collection and exploration. Second, the native understanding-generation-prediction stack equipped with hybrid linear temporal memory governs how the model maintains the world state. By strategically allocating temporal responsibilities across SWA (local dynamics), DSWA (mid-range interactions), and GLA (global contractive causal memory), the network robustly preserves critical world latents like object permanence over long horizons, mitigating the error accumulation that typically derails reinforcement learning rollouts. Third, the deployment-aware system co-design dictates how the model runs the world under real-world edge constraints. By optimizing low-level kernels and weight-only quantization, Kairos turns execution efficiency into a first-order modeling principle, enabling the sub-millisecond inference necessary to maintain the real-time observation–action–feedback loops through which self-evolution and online corrective learning practically occur.

Results. Extensive evaluations demonstrate that Kairos achieves state-of-the-art (SOTA) performance across diverse benchmarks while maintaining unprecedented inference efficiency (Figure 3). In general world-modeling and generative benchmarks as WorldModelBench [12] and DreamGen Bench [13], Kairos matches or surpasses significantly larger counterparts without incurring quadratic computational overhead, courtesy of its linear scalability. More importantly, when evaluated on rigorous embodied control and World-Action Model benchmarks, Kairos exhibits superior manipulation and generalization capabilities. It achieves new SOTA milestones on trajectory-driven task suites including LIBERO-plus and RoboTwin 2.0, firmly validating its efficacy as a deployable world-action execution infrastructure.

Taken together, these architectural paradigms suggest a fundamental shift in how world models must be conceptualized. Rather than treating world modeling merely as a problem of rendering increasingly realistic pixels, Kairos treats it as the challenge of constructing a deployable, actionsensitive substrate that can acquire, preserve, and operationalize physical knowledge over time. By unifying a hierarchical curriculum, a hybrid persistent memory, and edge-side runtime co-design, Kairos moves beyond a static generative showcase, laying down the definitive operational foundation for future self-evolving physical intelligence.

## 2 Model

## 2.1 Native Architecture with Unified Understanding, Generation and Prediction

The core architecture of Kairos (Fig. 2) represents a departure from modular platform-level integration toward a natively unified world-model stack. Rather than interconnecting disparate components,

![](images/823a120a7dcb88b12154d41c37595fd43f6931a439dc013f174632210dbb3ade.jpg)  
Figure 4 Model Architecture of Kairos.

Kairos employs a single endogenous backbone that integrates understanding, generation, and prediction within a unified architecture. This design ensures that semantic intelligence, visual synthesis, and physical anticipation are maintained as a shared world state, stabilized by a hybrid temporal memory mechanism designed for long-horizon coherence. The architecture primarily comprises three pivotal and deeply interactive modules—World Understanding, World Generation, and World Prediction—which operate in concert to form a highly integrated and fully functional intelligent framework.

World Understanding. As the foundational layer, the Understanding component is responsible for extracting deep representations from diverse heterogeneous sources, spanning from physical law descriptors to multi-modal sensor streams. We leverage a robust Vision-Language Model (VLM) as our Understanding module, which meticulously transforms these raw, heterogeneous inputs into high-level semantic representations, thereby enabling a profound comprehension of the world. This profound understanding model establishes a solid foundation for subsequent generation and prediction tasks. In this study, we utilize Qwen series [14, 15] as our foundational VLM. While we anticipate that specialized variants tailored for embodied intelligence, such as ACE-Brain-0 [16], may offer further performance gains, we defer the exploration of such domain-specific models to future work.

World Generation. The World Generation component plays a central creative role within the Kairos system, responsible for dynamically generating new data or scenarios based on the system’s profound understanding and external directives. Its inputs primarily stem from the semantic understanding outputs of the World Understanding module, seamlessly integrated with real-time interactive instructions from users or external systems (e.g., camera control, natural language commands, keyboard/mouse operations, or trajectory directives). Leveraging these diverse inputs, Kairos’s generation component efficiently produces high-fidelity outcomes—including photorealistic environments. Crucially, Kairos emphasizes deployment as a core modeling principle, rather than merely an added platform value. Therefore, the generative architecture is co-designed with hardware constraints: we utilize hybrid linear attention to ensure that generation is not an offline showcase but a real-time capability. This enables active intervention and creation in both virtual and physical worlds.

As illustrated in Fig. 4, Kairos follows a standard conditional diffusion formulation, where the denoising network is parameterized by a Diffusion Transformer (DiT) backbone and is conditioned on multimodal inputs such as text or images through cross-attention. The overall framework consists of three main components: (1) a high-compression video VAE that maps raw videos into a compact latent space, (2) a multimodal conditioning encoder that produces semantic embeddings, and (3) a temporally scalable DiT backbone that performs diffusion modeling in the latent space.

Kairos supports multi-modal conditioning to enable flexible video generation scenarios, including textto-video (T2V), image-to-video (I2V), and text-image-to-video (TI2V). Text inputs are embedded using a VLM-based text encoder, producing semantically rich token representations. For imageconditioned generation, visual features are extracted and aligned into the same embedding space. All conditioning embeddings are projected to the DiT hidden dimension and injected into the backbone through cross-attention layers. This design allows the diffusion model to leverage high-level semantic guidance while retaining the expressive capacity of the latent diffusion framework. The modular embedding interface also facilitates future extensions to additional modalities, such as audio or proprioceptive signals.

Besides, the hallmark of Kairos is its explicit architectural mechanism for state maintenance, which moves beyond simple next-frame prediction. We decompose the temporal structure into a hierarchical hierarchy: Sliding Window Attention (SWA) for local dynamics, Dilated SWA (DSWA) for mediumrange interactions, and Gated Linear Attention (GLA) for global causal memory. Specifically, the GLA mechanism, with its gated delta updates, serves as a persistent memory that tracks the world state over long horizons. In this way, Kairos explicitly models long-horizon world-state maintenance, not solely world generation plus reasoning. See details in Section 2.2.

World Prediction. As a core component of Kairos, the World Prediction module aims to jointly model environmental dynamics and robot action generation within a unified generative framework. Unlike conventional world models that primarily simulate future observations, Kairos treats robot actions as an intrinsic part of future world evolution. This design enables the learned representations to be directly mapped to executable motor behaviors, bridging high-level semantic understanding and low-level embodied control.

To achieve this, we formulate World Prediction as a unified World-Action Model (WAM) based on a Mixture-of-Transformer (MoT) architecture. The framework consists of two components: a Video DiT and an Action DiT. The Video DiT models future visual tokens and is initialized from the pretrained Kairos-World-Generation model, inheriting rich spatiotemporal priors and strong visual generation capabilities. The Action DiT predicts future action tokens and follows the same architectural design as the Video DiT while adopting a substantially reduced model scale (approximately one-fifth of the Video DiT) to improve inference efficiency.

To jointly model visual dynamics and robot actions, the input sequence is organized into three token groups:

• History Video Tokens, representing historical visual observations;

• Future Video Tokens, representing future visual states;

• Future Action Tokens, representing future robot actions.

Inspired by the mixed-attention strategy proposed in [17], we adopt a unified attention masking mechanism for joint video-action modeling. Specifically, history video tokens are restricted to attend only to other historical video tokens, preventing information leakage from future states and preserving stable representations of past observations. Both future video tokens and future action tokens are allowed to attend to all historical video tokens, enabling future prediction to be conditioned on the complete observed visual context.

Building upon this masking formulation, we further design an asymmetric interaction scheme for future tokens. Future video tokens employ sparse spatiotemporal attention to efficiently capture local visual dynamics while maintaining computational scalability. In contrast, future action tokens utilize full attention, allowing them to aggregate information globally across the entire action sequence and thereby facilitating coherent long-horizon planning. Notably, both branches share the same historical visual context, while the action branch does not rely on future video tokens. Consequently, action prediction can be performed independently without explicitly generating future visual observations, enabling efficient action-only inference at deployment time.

Under this unified masking strategy, the model is jointly optimized for video generation and action generation during training. The video objective encourages learning of environmental dynamics and future state evolution, while the action objective promotes the acquisition of executable control policies. Through joint optimization, the model gradually learns a shared world-action representation that implicitly aligns environmental transitions with robot decision-making.

During inference, the framework supports an efficient action-only prediction mode. Specifically, the future video generation branch can be disabled, and only future action tokens are generated. Since action tokens are significantly fewer than video tokens, this strategy substantially reduces both attention and diffusion computation costs while retaining the benefits of the jointly learned world dynamics. As a result, Kairos enables efficient and scalable robot action generation without explicitly synthesizing future visual observations. The World Prediction module provides a practical and scalable solution for unified world-action modeling, combining strong environment understanding, executable action generation, and efficient deployment within a single generative architecture.

Overall, Kairos demonstrates a stronger unification at the native world-state modeling level: it more clearly and comprehensively addresses how to learn about the world, how to continuously maintain its state, and how to operate efficiently under the complex constraints of the real world.

## 2.2 Efficient Diffusion Transformer with Hybrid Linear Attention

Diffusion Transformers (DiT) have emerged as a powerful backbone for image and video diffusion models, as well as action prediction. However, standard DiT architectures rely on full Softmax self-attention, whose quadratic complexity in the sequence length makes them prohibitively expensive for long video generation. In the context of world modeling, where videos can span several seconds with high spatial resolution, the token count grows rapidly with both frame number and spatial dimensions, resulting in severe computational and memory bottlenecks.

In this work, we highlight three core requirements for a modern world model:

• Efficiency. The model must scale gracefully to long video sequences and high resolutions, ideally with linear or near-linear complexity in the temporal dimension.

• Long-Horizon Modeling. The architecture should support information propagation across long temporal horizons to capture object permanence, delayed effects, and multi-stage physical interactions.

• Expandability. The design should be modular and extensible, allowing the integration of additional modalities, longer durations, and higher resolutions without architectural redesign.

![](images/b9399dff867908c47e884ddad2ff9c79d4c7ed10e6335aa406ce2b7d8c8869ba.jpg)  
Figure 5 DiT block architecture of the proposed hybrid linear attention.

To satisfy these requirements, Kairos introduces a LinearDiT backbone with a hybrid attention design, as illustrated in Fig. 5. Instead of relying solely on full Softmax attention, the backbone interleaves multiple attention mechanisms with complementary inductive biases. The model is organized into M groups of hybrid blocks, where each group contains local, dilated, and global attention components. This design enables efficient local motion modeling, mid-range temporal interaction, and global causal reasoning within a unified architecture.

## 2.2.1 Long-term: Gated Linear Attention

To enable global temporal reasoning with linear complexity, Kairos employs Gated Linear Attention (GLA) as the primary mechanism for long-range information propagation. Concretely, GLA is implemented using GatedDeltaNet [18], a gated linear attention variant closely related to structured state space models (SSMs). Unlike Softmax attention, whose complexity scales quadratically with sequence length, GLA scales linearly and thus remains efficient even for long video sequences.

As illustrated in Fig. 6, the core of the GDN lies in the Delta Update Rule, which addresses the “key collision” problem found in vanilla linear transformers. Instead of purely additive updates, GDN learns to remove outdated or less important key-value associations to make room for new information.

The computation at each time step is defined as follows:

1. Feature Extraction: The query , key , and value are projected from the input . Simultaneously, a soft “writing strength” is computed via a sigmoid gate:

$$
\mathbf { q } _ { t } = \mathbf { W } _ { Q } \mathbf { x } _ { t } , \quad \mathbf { k } _ { t } = \mathbf { W } _ { K } \mathbf { x } _ { t } , \quad \mathbf { v } _ { t } = \mathbf { W } _ { V } \mathbf { x } _ { t } , \quad \boldsymbol { \beta } _ { t } = \sigma ( \mathbf { W } _ { \beta } \mathbf { x } _ { t } )\tag{1}
$$

2. Memory Retrieval and Interpolation: The model retrieves the old value $\mathbf { v } _ { t } ^ { \mathrm { o l d } }$ using the current key and interpolates it with the current value to generate $\mathbf { v } _ { t } ^ { \mathrm { n e w } }$ :

$$
{ \mathbf { v } } _ { t } ^ { \mathrm { o l d } } = \mathbf { S } _ { t - 1 } \mathbf { k } _ { t } , \quad { \mathbf { v } } _ { t } ^ { \mathrm { n e w } } = \beta _ { t } { \mathbf { v } } _ { t } + ( 1 - \beta _ { t } ) { \mathbf { v } } _ { t } ^ { \mathrm { o l d } }\tag{2}
$$

![](images/d6b400b79bf71a56e6fd74fe247b4ed61126d0561a068dd4b1efcd09ccda47ec.jpg)  
Figure 6 Architecture of the gated linear attention module GDN.

where $\mathbf { S } _ { t } \in \mathbb { R } ^ { d _ { v } \times d _ { k } }$ denotes a learnable associative memory that stores key–value correlations over time.

3. Delta State Update: The state matrix $\mathbf { S } _ { t }$ is updated by removing the old association and writing the new one, a process equivalent to a single step of SGD on an online regression loss:

$$
\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } - \underbrace { \mathbf { v } _ { t } ^ { \mathrm { o l d } } \mathbf { k } _ { t } ^ { \top } } _ { \mathrm { r e m o v e } } + \underbrace { \mathbf { v } _ { t } ^ { \mathrm { n e w } } \mathbf { k } _ { t } ^ { \top } } _ { \mathrm { w r i t e } }\tag{3}
$$

This update can be interpreted as an online delta-rule update that approximates one step of gradient descent on $\| \mathbf { v } _ { t } - \mathbf { S } \mathbf { k } _ { t } \| ^ { 2 }$

Gated Delta Update. While the above delta rule corrects key–value associations locally, it does not explicitly control global forgetting of past information. To improve memory management, a gating mechanism is introduced to adaptively modulate the contribution of the previous state.

Specifically, a decay gate $\alpha _ { t } \in ( 0 , 1 )$ is computed as:

$$
\alpha _ { t } = \sigma ( \mathbf { W } _ { \alpha } \mathbf { x } _ { t } )\tag{4}
$$

The state update is then modified as:

$$
\mathbf { S } _ { t } = \alpha _ { t } \mathbf { S } _ { t - 1 } - \mathbf { v } _ { t } ^ { \mathrm { o l d } } \mathbf { k } _ { t } ^ { \top } + \mathbf { v } _ { t } ^ { \mathrm { n e w } } \mathbf { k } _ { t } ^ { \top }\tag{5}
$$

Equivalently, this can be written as:

$$
\mathbf { S } _ { t } = \alpha _ { t } \mathbf { S } _ { t - 1 } + \beta _ { t } ( \mathbf { v } _ { t } - \mathbf { v } _ { t } ^ { \mathrm { o l d } } ) \mathbf { k } _ { t } ^ { \top }\tag{6}
$$

Here, $\alpha _ { t }$ acts as a forget gate that globally scales the previous memory state, enabling the model to discard outdated information more efficiently. Combined with the local delta correction

term, this gated update provides both precise associative correction and adaptive long-term memory control.

## 4. Output Generation: The final output is retrieved from the updated memory: $\mathbf { o } _ { t } = \mathbf { S } _ { t } \mathbf { q } _ { t }$

The role of GLA in Kairos is to propagate global state information across the entire temporal extent of the video. This global pathway is critical for modeling temporal causality, object permanence, and delayed physical effects. By using gating mechanisms, GLA selectively controls information flow, allowing the model to retain long-term memory while preventing uncontrolled accumulation of noise. As a result, the model can maintain coherent global dynamics over hundreds of frames.

Importantly, GLA serves as the only global attention mechanism in the backbone. All other selfattention layers are restricted to local temporal neighborhoods. This architectural choice enforces a clear separation of responsibilities: local attention handles fine-grained motion and interactions, while GLA is responsible for global temporal consistency and causal structure.

## 2.2.2 Short(Mid)-term: (Dilated) Sliding Window Attention

Sliding Window Attention (SWA). While global temporal modeling is handled by GLA, spatial and short-range temporal interactions are modeled using Sliding Window Attention (SWA). Each SWA block restricts attention to a fixed temporal window of size proportional to the number of spatial tokens per frame, enabling efficient modeling of local motion patterns and short-term dynamics.

For a sequence of hidden states $\mathbf { x } \in \mathbb { R } ^ { B \times ( F \cdot L ) \times D }$ , where F and L are the number of frames and the number of tokens per frame in the latent, respectively; the SWA computes attention within a local neighborhood. Specifically, for a query token at index $i ,$ the attention is restricted to keys and values within a window $W :$ :

$$
\operatorname { S W A } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) _ { i } = \sum _ { \substack { j \in [ i - \frac { w } { 2 } , i + \frac { w } { 2 } ] } } \operatorname { S o f t m a x } \left( \frac { \mathbf { Q } _ { i } \mathbf { K } _ { j } ^ { T } } { \sqrt { d } } \right) \mathbf { V } _ { j } .\tag{7}
$$

In our implementation, the window size w is defined as L × window\_size, effectively covering a fixed number of adjacent frames to capture short-range spatial-temporal correlations.

Dilated Sliding Window Attention (DSWA). To extend the temporal receptive field without incurring quadratic cost, Kairos further incorporates Dilated Sliding Window Attention (DSWA). DSWA uses the same window size as SWA but introduces a dilation factor along the temporal dimension. This allows the model to capture mid-range temporal dependencies (e.g., on the order of one second) while preserving linear complexity. By interleaving SWA and DSWA blocks, the backbone progressively aggregates information across increasing temporal scales.

The DSWA operation can be formulated as:

$$
\mathrm { D S W A } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathrm { S W A } ( \mathrm { r e a r r a n g e } ( \mathbf { Q } ) , \mathrm { r e a r r a n g e } ( \mathbf { K } ) , \mathrm { r e a r r a n g e } ( \mathbf { V } ) ) .\tag{8}
$$

Specifically, the input is reshaped from $( \boldsymbol { B } , \boldsymbol { F } \cdot \boldsymbol { L } , \boldsymbol { D } )$ to $( B \cdot d , { \frac { F } { d } } \cdot L , D )$ , where attention is performed within the reorganized sequence. This design enables the model to capture mid-range temporal dependencies (e.g., on the order of one second) while maintaining linear complexity with respect to the total sequence length. By interleaving SWA $( d = 1 )$ and DSWA $( d \in \{ 6 , 1 2 \} )$ blocks, the backbone progressively aggregates information across increasing temporal scales.

All sliding-window attention blocks employ RoPE-based relative positional encoding to encode local temporal and spatial relationships. This ensures that local geometry and motion are accurately modeled within each attention window, while global positional reasoning is delegated to the linear attention pathway.

Expandability through Modular Hybrid Attention. The architectural decoupling in Kairos provides a robust foundation for multi-dimensional expansion:

• Interactive World Modeling: The Gated Linear Attention (GLA) state matrix $\mathbf { S } _ { t }$ acts as a compressed latent memory. By injecting action-conditioned tokens (e.g., from robotic commands or game inputs) directly into the GLA gating mechanism or the latent state update, Kairos can be seamlessly transformed into an interactive world model. The linear recurrent nature allows for zero-latency inference in closed-loop control.

• Infinite Horizon Generation: Unlike Softmax-based models constrained by a fixed context window, the SWA and DSWA components maintain a constant memory footprint per step, while the GLA compresses historical context. This enables the model to generate videos of arbitrary length through recurrent state passing, effectively mitigating the "out-of-memory" issues typical of long-video diffusion.

• Cross-Modal Integration: The modular block design allows for the easy integration of additional modalities (e.g., audio, depth, or semantic maps). These can be treated as additional streams in the hybrid blocks, where local spatial-temporal features are fused via SWA, and global cross-modal alignment is maintained by the GLA pathway.

## 2.3 Theoretical Analysis of Hybrid Multi-Scale Temporal Memory

The proposed world model is grounded in a unified understanding-generation-prediction substrate and a hybrid temporal backbone. To formally justify our architectural design, we address two central questions regarding long-horizon consistency, such as object permanence and delayed physical effects: When is a bounded recent attention window fundamentally insufficient? and Under what conditions can a hybrid multi-scale memory recover near-optimal prediction? In this section, we present the core theorems that motivate the necessity and sufficiency of our hybrid multi-scale temporal memory mechanism. For complete statements of the definitions, lemmas, theorems, and proofs, please refer to Section B in the Appendix.

Problem Setup. We model the world as a partially observed controlled process. Let $Y$ be a square-integrable future target we want to predict, such as the spatial location of an entity in the distant future. Let $\mathcal { H } _ { t }$ denote the complete historical context up to time $t ,$ and let $\mathcal { W } _ { t } ^ { ( w ) }$ denote the recent w-step sliding window context. For any square-integrable predictor $Z ,$ , we define the squared prediction risk as $\mathcal { R } _ { t } ( Z ) = \mathbb { E } [ ( Y - Z ) ^ { 2 } ]$ We denote the optimal full-history risk as $\begin{array} { r } { R _ { \mathrm { f u l l } } ^ { \star } = \operatorname* { i n f } _ { Z \in L ^ { 2 } ( \mathcal { H } _ { t } ) } \mathcal { R } _ { t } ( Z ) } \end{array}$ Here $L ^ { 2 } ( \mathcal { H } _ { t } )$ formally denotes the mathematical space of all squareintegrable predictors that depend solely on the full history ${ \mathcal { H } } _ { t }$ , making $R _ { \mathrm { f u l l } } ^ { \star }$ represent the theoretical minimum error when given access to the entire past. Correspondingly, we denote the optimal recentwindow risk as $R _ { w } ^ { \star } = \operatorname* { i n f } _ { Z \in L ^ { 2 } ( \mathcal { W } _ { t } ^ { ( w ) } ) } \mathcal { R } _ { t } \mathopen { } \mathclose \bgroup \left( Z \aftergroup \egroup \right)$ which bounds the performance of any model restricted to a local temporal context.

The Necessity of Persistent Latent States. Purely local temporal models, such as standard sliding-window attention, struggle with long-horizon prediction because they lose track of events that fall outside their context window. We formalize this limitation mathematically.

Theorem 1 (Supra-window dependence implies the necessity of persistent state). Let $m _ { t } = \mathbb { E } [ Y \mid \mathcal { H } _ { t } ]$ and $m _ { t } ^ { ( w ) } = \mathbb { E } [ Y \mid \mathcal { W } _ { t } ^ { ( w ) } ]$ . The excess risk incurred by restricting prediction to the recent window satisfies the exact identity

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } = \mathbb { E } \big [ ( m _ { t } - m _ { t } ^ { ( w ) } ) ^ { 2 } \big ] = \mathbb { E } \big [ \mathrm { V a r } ( m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ) \big ] .\tag{9}
$$

Consequently, we establish the following formal equivalence:

$$
R _ { w } ^ { \star } > R _ { \mathrm { f u l l } } ^ { \star } ~ \Longleftrightarrow ~ m _ { t } ~ i s ~ n o t ~ \mathcal { W } _ { t } ^ { ( w ) } - m e a s u r a b l e .\tag{10}
$$

In other words, the excess risk is strictly positive if and only if the optimal full-history predictor $m _ { t }$ is not perfectly recoverable from the recent window $\mathcal { W } _ { t } ^ { ( w ) }$

Remark 1 (Local Smoothness Is Insufficient for Long-Horizon Consistency). Purely local temporal attention is mathematically insufficient for long-horizon generation. Short video continuation can often rely successfully on local visual smoothness. However, many current systems that perform well on short-horizon appearance transitions inevitably degrade as the generation duration grows because local continuation heuristics do not guarantee global state consistency. For instance, if an object is temporarily occluded and reappears later, a model looking only at the recent w frames cannot recover the object’s identity. Theorem 1 provides the theoretical explanation for this exact degradation phenomenon. It formally proves that such failure is an unavoidable structural limitation rather than merely a parameter capacity issue. To avoid a strictly positive and irreducible error, the model must inherently maintain a persistent latent state that carries historical context forward across the entire generative horizon.

Corollary 1 (Explicit lower bound under recent-window mismatch). Let E denote a specific recentwindow observation event with positive probability, $\mathbb { P } ( E ) > 0$ . If a past event that heavily influences the future target Y remains unobservable within E, resulting in distinct true future expectations µ1 and $\mu _ { 2 }$ with conditional probabilities α and $1 - \alpha$ , the excess risk is bounded below by the expression

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } \geq \mathbb { P } ( E ) \alpha ( 1 - \alpha ) ( \mu _ { 1 } - \mu _ { 2 } ) ^ { 2 } .\tag{11}
$$

Remark 2 (Interpretation of the Lower Bound). This corollary provides a quantitative characterization of the failure mode inherent to purely local attention. Once an influential past event falls outside the current context window, the model can no longer condition on the true historical cause and is instead forced to average over multiple plausible futures. In other words, it must hypothesize which hidden past has occurred, such as the identity or state of a temporarily occluded object that will become relevant later. The resulting prediction error is therefore strictly bounded from below by the conditional variance induced by these unresolved possibilities. This perfectly explains why local temporal smoothness may remain visually plausible over short horizons yet inevitably breaks down over extended periods. When future consistency depends on information no longer visible within the recent window, no local continuation rule can exactly recover the lost state. Crucially, this lower bound is information-theoretic rather than optimization-related. It arises not from insufficient model capacity or inadequate training, but from the fundamental absence of relevant information within the accessible context. Consequently, simply scaling model parameters or training compute cannot eliminate this performance gap. Resolving this issue strictly requires an architectural mechanism that explicitly preserves supra-window information across time.

The Sufficiency of Hybrid Multi-Scale Memory. While Theorem 1 explains why some persistent state is unavoidable, we now show why our specific temporal factorization into local, dilated, and global components is theoretically adequate. Suppose the true Bayes predictor $\mu _ { t } ^ { \star }$ factorizes into a shared predictive state $U _ { t } ^ { \star }$ , a short-range local state $C _ { t } ^ { \star }$ corresponding to SWA, a mid-range dilated state $D _ { t } ^ { \star }$ corresponding to DSWA, and a global recurrent causal memory $G _ { t } ^ { \star }$ corresponding to GLA.

Theorem 2 (Approximate sufficiency of a hybrid multi-scale temporal memory). Assume the learned hybrid predictor approximates these components with maximum error $\varepsilon ,$ and the global memory branch utilizes a gated delta update that is contractive with factor $\rho < 1$ . Let $\bar { \xi }$ represent the maximum one-step perturbation error in the global memory update. Then, the hybrid predictor asymptotically satisfies the long-horizon excess-risk bound:

$$
\mathcal { R } _ { t } ( \hat { \mu } _ { t } ) - \mathcal { R } _ { t } ^ { \star } \leq \left( L \varepsilon + \frac { L _ { G } \bar { \xi } } { 1 - \rho } \right) ^ { 2 } \quad a s t \to \infty ,\tag{12}
$$

where L and $L _ { G }$ are Lipschitz constants of the underlying decoder.

Remark 3 (Why Hybrid Multi-Scale Memory Is Sufficient). Theorem 2 explains why our specific hybrid design is theoretically well-founded. While Theorem 1 establishes that persistent memory is unavoidable for supra-window dependencies, Theorem 2 proves that factorizing this memory into short-range (SWA), mid-range (DSWA), and gated global (GLA) branches is mathematically sufficient. Crucially, by proving that the gated delta update is contractive, we guarantee that the global-memory error does not accumulate arbitrarily. Instead, it strictly satisfies the bound:

$$
e _ { t } \leq \rho ^ { t } e _ { 0 } + \frac { 1 - \rho ^ { t } } { 1 - \rho } \operatorname* { s u p } _ { 1 \leq i \leq t } \xi _ { i } ,
$$

yielding the asymptotic limit: ${ e _ { t } \le \bar { \xi } / ( 1 - \rho ) }$ as $t  \infty$ This geometric damping ensures that one-step perturbations are strictly bounded rather than amplified over time. Consequently, the long-horizon excess risk is controlled by only two factors: the approximation quality of the local pathways and the asymptotic perturbation limit of the global memory.

Architecturally, the local and mid-range branches efficiently capture localized appearance changes and intermediate temporal structures, while GLA selectively propagates the persistent causal state. Because this propagation is bounded against uncontrolled drift, GLA functions as a stable information bottleneck. It preserves essential long-range context without accumulating compounding errors, effectively enabling consistent generation over extended horizons.

Corollary 2 (Exact sufficiency in the realizable case). If the learned hybrid state exactly recovers the Bayes decomposition at every time step, where $\varepsilon = 0$ and $\bar { \xi } = 0$ , then the learned predictor strictly matches the optimal predictor such that $\hat { \mu } _ { t } = \mu _ { t } ^ { \star }$ , and the associated risk becomes $\mathcal { R } _ { t } ( \hat { \mu } _ { t } ) = \mathcal { R } _ { t } ^ { \star }$

Remark 4 (Interpretation of Exact Sufficiency). If the network perfectly learns the underlying temporal dynamics, it recovers the theoretical optimal performance. This provides the foundational justification for separating local attention from global causal memory in our proposed architecture. It is not just computationally efficient due to linear complexity but also theoretically sound for optimal long-horizon world simulation.

## 3 Native Pretraining Paradigm for Physical AI

In contrast to conventional paradigms that treat physical alignment as a downstream fine-tuning afterthought, Kairos champions a native pre-training paradigm for Physical AI, where general physical laws and embodied grounding are natively injected into the foundational world model from the very inception of scaling. However, building such a native world-action foundational infrastructure faces a fundamental tension between data scale and grounding density: low-level, high-fidelity robot trajectories are inherently scarce, whereas web-scale, non-embodied internet videos offer abundance but lack actionable dynamics. To reconcile this multi-modality, multi-scale data heterogeneity, Kairos introduces a Cross-Embodiment Data Curriculum (CEDC). Instead of mixed, unstructured data dumping, CEDC strategically orchestrates the pre-training trajectory through a progressive data pyramid. This curriculum systematically guides the model’s developmental pathway—transitioning from open-world passive observation to task-structured human imitation, and ultimately to action-grounded physical embodiment. In the following subsections, we detail the architectural stages of this progressive alignment.

## 3.1 Native Pretraining with Cross-Embodiment Data Curriculum

The acquisition of world knowledge is fundamentally bottlenecked by the fragmented nature of heterogeneous experience sources. While open-world videos provide broad physical regularities, they often lack action grounding; conversely, robot data is action-rich but remains narrow and difficult to scale. Instead of pursuing a flat data scaling strategy that simply pools these mismatched sources, Kairos implements a Cross-Embodiment Data Curriculum (CEDC) to systematically evolve from passive physical understanding to active embodied control.

Phase I: Physical Knowledge. The base of the pyramid focuses on Observation, leveraging massive million-hours open-world video datasets to internalize broad physical and environmental regularities. This foundational phase is characterized by its hundreds of millions of video clips, absorbing an expansive breadth of visual experience to ensure the model encounters the full spectrum of natural physical phenomena. By processing this high-density observational data, the model develops the statistical robustness necessary to handle diverse real-world scenarios, effectively constructing a comprehensive "physical common sense" before moving to task-specific learning. At this stage, the model functions as a "spectator" learning the implicit laws of the universe: gravity, mass conservation, and fluid dynamics. To move beyond mere pixel-level synthesis, Kairos integrates causal Chain-of-Thought (CoT) reasoning to formalize these physical principles within its latent space. By grounding dynamics in physical logic, this phase establishes the foundational "worldness" and transforms raw visual experience into a robust substrate of universal regularities.

Phase II: Human-centric Behavior. The curriculum then transitions to Imitation, where the model internalizes the "behavioral experience" encoded in human-centric datasets over 100,000 hours. This represents a shift from understanding what happens to understanding how tasks are organized through human intent. The significance of this phase lies in its ability to extract high-level behavioral abstractions and task logic from human demonstrations. Rather than learning isolated motions, the model learns the causal consequences of structured intervention—how specific sequences of actions lead to predictable changes in the world state. This process bridges the gap between passive observation and active participation, populating the model’s internal world with a library of purposeful interaction patterns.

Phase III: Robotic Action. The apex of the curriculum is Embodiment, which resolves the representational mismatch between broad human behavior and specific robot actuation. By anchoring previous physical and behavioral priors into robot-specific interaction data (e.g., AgiBotWorld-Beta [19], Droid [20]), Kairos achieves perception-action grounding. In this final phase, the model learns the mechanical constraints and sensorimotor coupling of the embodied agent. The world model is no longer a spectator; it becomes an operational substrate that can predict how the future unfolds as a

![](images/cb0bd7cac26b8f724b676ec38a4f063b76d87e27371ea39960b6310f850cba6f.jpg)  
Figure 7 Cross-Embodiment Data Curriculum for Native Pretraining.

direct consequence of the agent’s own actions.

By organizing data as a developmental pathway—moving from observing the world, to imitating behavior, and finally to enacting control—Kairos ensures that world knowledge is accumulated into a unified, actionable representation that is architecturally ready for real-world deployment. This progression is not merely an engineering choice but a prerequisite for self-evolution learning, as it establishes the closed-loop observation–action–feedback pathways necessary for continual adaptation in real environments.

Multi-Stage Native Pre-training Pipeline. To operationalize the Cross-Embodiment Data Curriculum (CEDC) within our Mixture-of-Transformers (MoT) architecture, the native pre-training pipeline is systematically structured into three progressive stages, each dominated by its corresponding data layer. This multi-stage optimization decouples the acquisition of passive world knowledge from active policy execution, ensuring both scaling efficiency and control fidelity.

Specifically, Stage I (Physical Pretraining) and Stage II (Human-Centric Embodied Pretraining) focus exclusively on optimizing the VideoDiT component. In these initial phases, the model internalizes open-world physical dynamics and high-level task taxonomies through dense video forecasting tokens, optimizing the unified spatial-temporal representation without action-space interference. Transitioning to Stage III (Joint World-Action Training), the training emphasis shifts toward the joint optimization of the ActionDiT component alongside the pre-trained VideoDiT. By injecting low-level robotic trajectories and tactile/proprioceptive states, Stage III achieves the definitive cross-modality alignment where visual forecasting natively translates into closed-loop physical execution.

Training Objective. We adopt Flow Matching as the primary training objective for Kairos, following recent flow-matching generative modeling work in both images and videos [21–26]. At a high level, flow matching learns a continuous-time conditional velocity field that transports samples from a simple prior distribution (Gaussian noise) to the data distribution in latent space. This formulation provides a clean continuous-time view of generation and is naturally compatible with ODE-based sampling. To align with the latent notation used in Sec. 2, let $\boldsymbol { z } _ { 0 } \in \mathbb { R } ^ { C \times F \times H \times W }$ denote a clean latent video sample produced by the video VAE encoder, and let c denote the conditioning inputs (e.g., text, image, or other multimodal conditions). We sample Gaussian noise $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , I )$

![](images/54d1b76f7958b72da07d035f2986c3c8dade170942a3410bcb7e4d39a4cdef6e.jpg)  
Figure 8 Shape-aware exponential timestep shifting curves in Kairos across training stages. This figure shows the σ-domain remapping of the Kairos scheduler for three representative training stages, which will be introduced in the following sections. As the resolution becomes higher and the video becomes longer, the effective shift strength s increases, resulting in a stronger upward remapping of the timestep schedule.

with the same shape as $z _ { \mathrm { 0 } }$ , and define a continuous interpolation variable $\sigma \in ( 0 , 1 )$ Following the rectified-flow style linear interpolation parameterization [22, 23], the intermediate latent is constructed as

$$
z _ { \sigma } = ( 1 - \sigma ) z _ { 0 } + \sigma \epsilon .\tag{13}
$$

Under this parameterization, the ground-truth velocity along the path is constant:

$$
{ \pmb u } _ { \sigma } = \frac { d z _ { \sigma } } { d \sigma } = { \pmb \epsilon } - z _ { 0 } .\tag{14}
$$

Kairos trains the Diffusion Transformer, as a conditional velocity predictor $\mathcal { V } _ { \theta } ( z _ { \sigma } , \sigma , \pmb { c } )$ , to regress this target. The flow matching (FM) training loss is the mean squared error (MSE) between the predicted velocity and the ground-truth velocity:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { z _ { 0 } , \epsilon , \sigma , c } \left[ \| \mathcal { V } _ { \theta } ( z _ { \sigma } , \sigma , c ) - \boldsymbol { u } _ { \sigma } \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{15}
$$

This objective is shared across different conditioning modes, allowing a unified latent video generation model while varying only the conditioning inputs.

In the following, we delineate the explicit training formulations, objectives, and hyperparameter schedules for each developmental stage.

## 3.2 Stage I: Physical Pretraining

As the foundation of our curriculum, Stage I focuses on injecting fundamental physical priors into the VideoDiT backbone. This is achieved through two core strategies:

Table 1 Stages of progressive physical pretraining
<table><tr><td>Task 1</td><td>Stage</td><td>Resolution 2</td><td>Max Frames 3</td></tr><tr><td>T2I/N2I</td><td>Image Pretraining</td><td>256P</td><td>1</td></tr><tr><td>T2V/TI2V/I2V/N2V</td><td>Pretraining</td><td>256P</td><td>81</td></tr><tr><td>T2V/TI2V/I2V/N2V</td><td>Pretraining</td><td>480P</td><td>81</td></tr><tr><td>T2V/TI2V/I2V/N2V</td><td>Pretraining</td><td>720P</td><td>81</td></tr><tr><td>T2V/TI2V/I2V/N2V</td><td>CT</td><td>720P</td><td>241</td></tr><tr><td>T2V/TI2V/I2V/N2V</td><td>Domain-specific SFT &amp; Merging</td><td>720P</td><td>241</td></tr><tr><td>T2V/TI2V/I2V/N2V</td><td>RL</td><td>720P</td><td>241</td></tr></table>

1 We define the following task abbreviations: T2I (Text-to-Image), T2V (Text-to-Video), TI2V (Text-Image-to-Video), and I2V (Image-to-Video). Additionally, N2I and N2V denote unconditional image and video generation, respectively. Notably, T2I and N2I are treated as special cases of their video counterparts (T2V and N2V) when the video frame count equal to 1.  
2 Our preprocessing pipeline resizes images and video frames based on their shorter dimension. Importantly, both the long and short sides are constrained to be multiples of 32. Thus, for the 720p configuration, we adopt a shorter-side resolution of 704 pixels.  
3 During training, we utilize samples with variable temporal lengths, ranging from a single frame (e.g., for T2I tasks) to the maximum frame count.

• Web-Scale Physical Prior Injection: The model leverages massive, unconditioned internet videos to internalize general spatial-temporal dynamics—such as gravity, object permanence, and collision mechanics—purely from passive observation.

• Progressive Training Strategy: To optimize the efficiency of training from scratch, we employ a progressive training protocol. The model advances from lower visual resolutions and shorter sequence lengths to high-fidelity, long-horizon video continuation forecasting, significantly accelerating scaling dynamics.

This workflow is structured into three key phases: initial image-based pre-training, image-video mixed training, and continuous pre-training, as shown in Tab 1.

• Image Pretraining. Initial T2I pre-training establishes robust spatial-semantic priors, allowing the model to focus exclusively on temporal consistency and motion dynamics in subsequent stages. To optimize computational efficiency, this foundational phase is conducted at a reduced resolution of 256p, facilitating rapid convergence of core visual features.

• Image-video Mixed Pretraining. In this stage, we initiate video training using a progressive resolution strategy, scaling from 256p to 720p to optimize computational efficiency. To maintain spatial fidelity, we incorporate a small fraction of image data (at a 10% sampling ratio) into the pipeline. By jointly training on both static images and video sequences, the model effectively bridges the gap between spatial structures and motion dynamics, fostering temporal coherence while preventing the catastrophic forgetting of high-quality spatial details.

• Continual Pretraining. Finally, the model undergoes continued pre-training on a meticulously curated, high-quality dataset to refine its aesthetic appeal and capture more nuanced, long-range temporal dependencies for superior video generation. A key advancement in this phase was extending the maximum temporal length of training sequences from 81 to 241 frames, enabling the model to directly synthesize high-fidelity videos of up to 15 seconds.

Timestep Scheduler Distribution Shift. However, a practical challenge in physical multi-stage pretraining is that the effective timestep distribution induced by a fixed scheduler changes with latent spatiotemporal shape (e.g., frame count and resolution), which can degrade robustness and quality when the same model is used across configurations. To address this issue, Kairos applies a shape-aware timestep distribution shift [23–26] to the scheduler-defined σ sequence, as visualized in Figure 8.

Formally, let $\{ \sigma _ { i } ^ { ( 0 ) } \} _ { i = 1 } ^ { N }$ denote a base schedule before shifting, where $\sigma _ { i } ^ { ( 0 ) } \in ( 0 , 1 )$ is monotonically ordered. In the default setting, we apply an exponential shift to each base timestep:

$$
\tilde { \sigma } _ { i } = \frac { s \sigma _ { i } ^ { ( 0 ) } } { 1 + ( s - 1 ) \sigma _ { i } ^ { ( 0 ) } } ,\tag{16}
$$

where s is a factor to control the shift strength. An equivalent form is a translation in logit space, $i . e .$ ，

$$
\tilde { \sigma } _ { i } = \mathrm { s i g m o i d } \Big ( \log \mathrm { i t } \Big ( \sigma _ { i } ^ { ( 0 ) } \Big ) + \log s \Big ) ,\tag{17}
$$

which makes the monotonicity of the transformation explicit and preserves the boundary behavior of the schedule.

For a better and more adaptive support of varying video lengths and resolutions in a unified model, we design a dynamic selecting rule of shift strength s based on the latent shape. Reviewing that F and $L = H \times W$ defined previously are the number of frames after video encoding and the number of tokens per frame, respectively, we set s shape-dependently, i.e.,

$$
s = \exp ( f ( L ) ) { \sqrt { F } } ,\tag{18}
$$

where $f ( L ) = m L + b$ linearly maps $L \in [ L _ { \operatorname* { m i n } } , L _ { \operatorname* { m a x } } ]$ to a predefined range $[ r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ]$ , with

$$
m = \frac { r _ { \mathrm { m a x } } - r _ { \mathrm { m i n } } } { L _ { \mathrm { m a x } } - L _ { \mathrm { m i n } } } . \qquad b = r _ { \mathrm { m i n } } - m L _ { \mathrm { m i n } } ,\tag{19}
$$

This design increases the effective shift for larger latent spatial token counts and longer videos, thereby reallocating scheduler steps toward trajectory regions that are more sensitive to prediction error. Throughout the pre-training phase, we consistently employ the AdamW [27] optimizer with phase-specific hyperparameter configurations. Specifically, the learning rate is set to 5e-5 for image pre-training, and then progressively decayed to 4e-5, 3e-5, 2e-5, and 1e-5 across the successive image-video mixed pre-training (256p, 480p, 720p) and continuous pre-training stages. Weight decay is configured at 1e-3 only during the image and 256p mixed pre-training stages, while being deactivated (set to 0) for all subsequent high-resolution and continuous training phases.

In addition, we also implement delicate finetuning techniques to further boost the performance of physical alignment, which consists of three phases: domain-specific supervised fine-tuning (SFT), model merging, and reinforcement learning.

• Domain-specific SFT and Model Merging. Specifically, we partition high-quality datasets into different domains, and models corresponding to each domain are trained independently. To fully leverage the domain-specific strengths of individual models, we employed a model merging strategy to integrate their features after completing domain-specific supervised finetuning [28] [29]. We explored and applied multiple merging approaches, including Model Soup [30], CART [31], TIES [32], DARE [33], and WUDI-Merging [34]. By analyzing the performance of models obtained from different merging strategies on carefully constructed evaluation subsets, we identified a model that achieves the most balanced performance across all domains.

• Reinforcement Learning. Finally, we applied Direct Preference Optimization (DPO) [35– 37] to align the model’s outputs with human preferences and physical standards. The DPO preference pairs were constructed by generating video candidates from multiple generative models conditioned on identical prompts. We then identified the highest and lowest performing samples to form the (chosen, reject) pairs. This curated dataset was subsequently utilized to fine-tune the model following the DPO objective.

For this stage, the AdamW optimizer is retained, with learning rates tailored to 1e-5 for Supervised Fine-Tuning (SFT) and 1e-6 for Reinforcement Learning (RL). To maintain training stability, weight decay is uniformly set to 0 throughout the entire process.

## 3.3 Stage II: Embodied Pretraining with Human-centric Data

Building upon the general physical priors established in Stage I, Stage II advances the Cross-Embodiment Data Curriculum into the layer of "Human-Centric Behavior Alignment." This phase bridges the gap between passive observation and active robotic execution through two primary paradigms:

• Task-Structured Semantic Injection: By leveraging large-scale, human-centric behavioral datasets (e.g., intentional actions, tool manipulation, and everyday chores), the model transitions from unconditioned video generation to task-structured video forecasting, internalizing high-level behavioral semantics and task taxonomies.

• VideoDiT Optimization for Behavioral Causality: The training execution remains focused exclusively on the VideoDiT component. By predicting intentional human movements and scene state transitions, VideoDiT learns to construct a robust causal representation of goal-directed actions and their environmental consequences, without yet binding to a specific robotic action space.

Specifically, we leverage the both the human demonstrations and robot interaction data to fulfill the human-centric pretraining. Human demonstrations consist of human behavior data captured from both first-person and third-person perspectives, facilitating the model’s learning of complex human operational workflows and behavioral paradigms; robot interaction data enable the model to learn visual variations across different robot embodiments, thereby directly enhancing video generation quality in embodied scenarios. Crucially, the textual guidance in this stage progressively evolves from granular, detail-oriented annotations to abstract, instruction-style commands, thereby systematically strengthening the model’s capacity for complex instruction-following and multi-task intent understanding.

The training protocol is divided into four progressive stages, each with distinct focus to gradually enhance the model’s adaptability to the embodied domain and its instruction-following capability:

• Human-centric Pretraining. We train on a mixed dataset dominated by human-centric data using variable-length clips from 3 to 15 seconds. The wide range of clip lengths, together with detailed and diverse captions, helps the model acquire rich task-structured semantic priors from human behavior and build foundational embodied capabilities for the following stage.

• Robot-Centric Training. We continue training on a high-quality, robot-only subset filtered from our existing robot data, consisting of 5–10s clips with stronger task completion. During training, we continuously increase the chance of sampling instruction-style captions to strengthen instruction following capability of our model.

• Target-Embodiment Fine-Tuning. We fine-tune the model solely on data from the target embodiments to adapt it to the downstream robotic platforms. For embodiments with multiple cameras, we additionally train multi-view video generation to jointly model the synchronized observations across viewpoints. In this stage, we use only instruction-style captions, and all clips are uniformly sampled to 81 frames.

• RL Refinement. As in Stage 3.2 (Physical Pretraining), we use DPO as the last stage to refine our model. The difference is that we now judge the candidates not only by human preference and physical plausibility, but also by how well they follow the instruction.

Throughout the entire training process, the AdamW optimizer is uniformly adopted, with the weight decay consistently set to 0 to preserve the general capabilities acquired in the previous training stage. The learning rate is dynamically adjusted across different training stages. Stage 1 employs a learning rate of 1e-5; the learning rate is reduced to 5e-6 in Stage 2; and it is further decreased to 1e-6 in the Fine-Tuning/DPO Stage, which ensures the stability and convergence of the model training process.

Figure 9 illustrates several samples generated by the Kairos model. We can see Kairos exhibits robust cross-embodiment generalization, seamlessly adapting to diverse robotic forms—including single-arm, dual-arm, dexterous hands and humanoids. This architecture enables a "unified brain for multi-embodiment and multi-tasking," facilitating the highly efficient sharing and transfer of world knowledge across different physical configurations.

## 3.4 Stage III: Joint World-Action Training

Following the successful injection of foundational physical priors and human-centric behavioral semantics in the preceding stages, Stage III shifts the focus toward explicit robotic action grounding. While Stages I and II equipped the VideoDiT backbone with a robust causal understanding of the physical world, the foundational model remains detached from direct physical execution. To operationalize this transition, Stage III focuses entirely on training the ActionDiT component to unify perception and actuation into a true, native World-Action Model.

Video/Action DiT Joint Training. While a frozen VideoDiT retains macro-level physical aesthetics and coarse task semantics from internet pre-training, it remains inherently blind to the microscopic, high-frequency spatial-temporal nuances critical for precise physical control—such as millimeter-level contact dynamics, singular slipping boundaries, and instantaneous force-torque reactions. By co-optimizing the multi-stage Mixture-of-Transformers (MoT) stack [17], the lowlevel action trajectories propagated through ActionDiT forces the VideoDiT parameters to shift from passive visual synthesis to active, action-conditioned prediction. Consequently, joint training eliminates representation misalignment between simulation and execution, preventing the catastrophic drift common in decoupled architectures and cementing a truly cohesive world-action representation.

During training, video frame sequences and action chunk sequences are strictly aligned in the temporal dimension, ensuring that both branches observe corresponding segments of the same trajectory. Both the VideoDiT and the ActionDiT are trained using flow matching objectives. The overall training loss is defined as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { v i d e o } + \lambda \mathcal { L } _ { a c t i o n } , } \end{array}\tag{20}
$$

where λ controls the balance between action training and video co-training. To improve training efficiency, the ActionDiT is initialized by interpolating the pretrained weights from VideoDiT. Besides, ActionDiT utilizes a fixed timestamp shift during training, unlike the dynamic exponential shift used

![](images/4a294f411f9a86313ab4b0fd7fb54de2e85d3185acc007f2fb028f3e85d86a6b.jpg)  
Figure 9 Samples generated by Kairos. Kairos exhibits robust cross-embodiment generalization, seamlessly adapting to diverse robotic forms—including single-arm, dual-arm, dexterous hands and humanoids.

in VideoDiT that scales with the latent token count. Since VideoDiT models high-dimensional spatiotemporal latents with varying sequence lengths, whereas ActionDiT operates on a low-dimensional and nearly fixed-length action space, where a fixed shift is sufficient for stable optimization.

## 3.5 Training Infrastructure

Video generation models implementing spatial-temporal attention with Full Attention, such as Wan2.2 [26], Hunyuan1.5 [9], and Cosmos 2.5 [2], commonly adopt parallel partitioning strategies including Ulysses [38] and RingAttention [39] to decompose attention computation. These approaches reduce training memory footprint and maintain high training efficiency in multi-node and multi-GPU parallel settings.

However, the Kairos architecture exhibits more complex computational dependency characteristics:

• Linear Attention mechanisms impose stringent sequential dependencies on computation order;

• Dilated local attention does not depend on the complete set of global tokens.

Direct application of standard parallelization strategies such as Context Parallel, Ulysses, or RingAttention to our architecture introduces significant performance degradation due to unnecessary token broadcasting, redundant computation, and substantial communication overhead. To address these challenges, we design a customized operator-level parallel partitioning strategy:

• Operator-level customization: We apply tailored Tensor Parallel (TP) or Sequence Parallel (SP) configurations based on the dependency structure of each operator, precisely controlling activation scales to reduce per-GPU memory footprint.

• Operator fusion and communication optimization: Strategic operator merging and execution reordering minimize both the frequency and volume of cross-device communication, substantially reducing distributed training overhead.

• Adaptive parallelism: The partitioning strategy dynamically adapts to different attention mechanisms, enabling efficient scaling for both linear and local attention patterns.

This approach enables Kairos to achieve stable and efficient training on 720p, 15-second video sequences, demonstrating practical viability under resource-constrained settings.

## 4 Data

## 4.1 Data Collection

High-quality data constitutes the cornerstone for world models to comprehend physical laws and synthesize high-fidelity scenes. This chapter elaborates on the hierarchical data collection framework we constructed for the Kairos world model, encompassing two primary sources: open-source public datasets and in-house proprietary data. We also introduce the standardized shot segmentation pipeline as the final stage of data collection. Through our multi-dimensional, multi-modal data acquisition strategy, we ensure the comprehensiveness, representativeness, and domain coverage of training data, laying a robust foundation for the model to learn complex physical world dynamics.

We adopt a hybrid data collection strategy that combines open-source data as the foundational backbone with in-house data for targeted supplementation. Open-source data provides large-scale, diverse basic visual samples, while self-developed proprietary data specifically fills the gaps in open-source data in specific domains. The entire data collection process consists of three core phases: integration of open-source datasets, compliant acquisition of public internet data, and first-person (ego-centric) human manipulation data collection, which ultimately form standardized training samples through unified shot segmentation. Public sources include general datasets (e.g., Koala-36M [40], Openhumanvid [41], VidGen [42]) and specialized corpora for robotics (e.g., AgiBotWorld-Beta [19], Droid [20]). To break through the limitations of open-source data, we have built a large-scale in-house proprietary data system, which mainly consists of two parts: internet-crawled data and real-world collected data. The vast majority of proprietary data is collected from the Internet through a hierarchical taxonomy with tens of millions of leaf nodes. This taxonomy covers four core domains: human, robot, general scenes, and physical phenomena. Each domain is further subdivided into hundreds of secondary categories and thousands of tertiary categories, ensuring fine-grained coverage and diversity of data. We obtained a massive corpus of video data from publicly available Internet sources, and performed standardized preprocessing tailored to the characteristics of video content across different platforms. During the data cleaning phase, we systematically filtered out corrupted videos, duplicate videos, and invalid clips shorter than 5 seconds, ultimately accumulating several millions of hours of valid raw video data. To address the critical shortage of fine-grained robotic manipulation scenario data in existing open-source and publicly available Internet datasets, we additionally collected a large volume of high-precision human manipulation data from a first-person (ego-centric) perspective.

As the final step of the data collection process, we perform unified shot segmentation on all raw videos to split long videos into short shot segments suitable for model training. We employ PySceneDetect with multiple scene detectors, achieving over 95% segmentation precision and 80% recall. Raw videos are split into independent segments based on detected boundaries. To balance spatiotemporal integrity and training efficiency, we adopt the following strategies:

• Keep segments between 5–40 seconds.

• Further split long shots (> 40 seconds) into 20-second clips.

• Discard segments shorter than 5 seconds to avoid fragmented information interfering with model learning.

Through the complete data collection and shot segmentation process described above, we have finally obtained hundreds of millions of standardized video clips, forming the basic data pool for training the Kairos world model. This data pool not only has a massive scale but also reaches industry-leading levels in domain distribution, scene diversity, and data quality, providing strong data support for the model to learn complex physical world laws and generate high-fidelity dynamic scenes.

## 4.2 Data Curation

The primary objective of data filtering is to curate high-quality, low-noise, and semantically diverse training samples from preprocessed clips. By employing a hierarchical filtering strategy, we progressively enhance data purity, thereby supplying optimized data for distinct training phases. Figure 10 illustrates our progressive quality filtering pipeline.

Aesthetic Score. Aesthetic quality serves as a critical indicator of visual richness and content diversity. Videos characterized by low aesthetic scores often exhibit simplistic compositions and limited color palettes, which may hinder the model’s ability to learn complex visual patterns. To assess this attribute, we employ an aesthetic predictor built upon a CLIP backbone with an MLP head. Samples falling below a predefined threshold are subsequently filtered out to maintain high visual standards.

Motion Score. Videos characterized by static content or minimal motion lack sufficient temporal dynamics, whereas those exhibiting rapid flickering suffer from poor temporal consistency; both categories are detrimental to effective model training. To quantify temporal quality, we employ

![](images/ddf936d351156be1dea832ff6325c9df627626f76673bd44cef6181c0810c17b.jpg)  
Figure 10 Data Filtering Pipeline

RAFT [43] to compute optical flow fields between consecutive frames. The magnitude of the flow vector at each pixel serves as a proxy for motion intensity. By aggregating these metrics across all frame pairs, we derive a global motion score for each video. Consequently, we filter out samples with excessively high or low scores, retaining only those that balance rich temporal information with strong consistency.

AIGC Score. The proliferation of AI-synthesized videos online presents a significant challenge due to their highly variable quality; incorporating low-quality synthetic data can introduce substantial noise during training. To mitigate this, we train a discriminator based on the ViT-Large architecture using a proprietary synthetic video dataset. Videos exceeding a predefined threshold are subsequently excluded from the training set.

NSFW Score. Pornographic and violent videos severely mislead model training and may appear in crawled Internet data. We use the open-source Falconsai model [44] to filter such content.

Blurriness Score. We use the Laplacian operator to assess sharpness. A higher Laplacian score indicates richer edges and details, corresponding to a clearer image.

Human Motion Score. Human video generation constitutes a pivotal application of world models. However, a predominance of static human data within the training set can impede the model’s ability to synthesize videos with diverse human dynamics. To address this, we employ YOLOX [45] for human detection and ByteTrack [46] for trajectory tracking. Subsequently, we calculate the normalized pixel velocity for each individual to derive a motion score, which serves as a metric for data selection.

OCR Score. The ability to generate generic text is a key metric for assessing video generation models. Since text generation tasks can hinder model convergence during initial training, we exclude videos containing excessive text in the early phases. Our methodology utilizes DBNet [47] for text detection and region area calculation. We design an OCR scoring mechanism related to the proportion of text regions in the whole frame: the smaller the area occupied by text, the higher the OCR score; when no text exists in the video frame, the OCR score is 1.

Data Deduplication. Internet videos contain large amounts of near-duplicate redundant data, which contributes nothing to model training. We use CLIP to extract video embeddings offline . To support large-scale deduplication, we build an embedding pool. For each new clip, we compute pairwise similarity with historical clips in the pool; if similarity exceeds a threshold, only the higher-resolution version is retained.

![](images/f71ba3d97692d2a4527ef999371899e25673fa57acfa8a5d98d6670e8f0b12a0.jpg)

![](images/f0d8f649749dd6a196b3efa6533db8c13d0a594f705ec9821837499e76f36ed3.jpg)

![](images/8222136d8b2154962530f7b08e2857825be29b062c6f94ce2dab397600cb0e60.jpg)  
(b) Distribution of the data

![](images/8ff75fa83b29e9e9cd8a1d9a9e82fbe8ee32dd581413037e43020f963be78047.jpg)  
(c) Diversity of robotic interactive actions and manipulable objects  
Figure 11 Data Statistics and Examples

Table 2 Video Domain Tags
<table><tr><td>Tag</td><td> Sub-tags</td><td>Brief Description</td></tr><tr><td>Human</td><td>scenes/actions/occupation/ gender/age/context/face blur/body motion</td><td>Human behavior videos for learning human behavior patterns</td></tr><tr><td>Robot</td><td> scenes/actions</td><td>Robot interaction/task execution videos for learning robot-environment interaction mechanisms</td></tr><tr><td>Physics</td><td>principles</td><td> Videos of physical laws/natural phenomena for physical rule modeling</td></tr><tr><td>General</td><td>content type/scene/animal</td><td>General-scene videos ensuring complete coverage of the tagging system</td></tr></table>

## 4.3 Tagging and Captioning

## 4.3.1 Tagging

Video data tagging refers to the process of adding structured and semantic annotations to raw videos. As a core component of video data processing for world models, it is particularly suitable for massive video data collected from the Internet. On the one hand, video data obtained from the Internet has a wide range of sources, scattered distribution, and strong heterogeneity, which makes efficient management difficult. Tagging video data enables the structured organization of massive unordered data. It not only facilitates the balance of data sources during model training and effectively alleviates data bias caused by the uneven distribution of data sources, but also allows fast and accurate retrieval of specific data as needed, greatly improving the efficiency of data filtering and utilization in the training process. On the other hand, video tags can provide explicit and fine-grained semantic prior information for subsequent text annotation tasks. Integrating tag-based semantic constraints into the text annotation pipeline reduces the ambiguity of text annotations and enhances the alignment between generated annotations and video content, thereby significantly improving the accuracy and reliability of text annotations and laying a solid foundation for the effective learning of world models.

To achieve the aforementioned objectives of video data tagging, we propose a comprehensive tagging system. It consists of two core categories of tags: video attribute tags and video domain tags, which jointly enable efficient processing of video data and lay a solid foundation for effective learning in world models.

Attribute Tags. Video attribute tags are used to record the global characteristics of video content, covering intrinsic information irrelevant to semantic interpretation. The video attribute classification system we designed includes camera motion detection, static video detection, and others.Camera motion detection determines whether camera movements such as translation, rotation, and shaking exist throughout the video. Static video detection judges whether a large number of static frames are present in the video. All attributes adopt binary (true/false) values.

Domain Tags. Video Domain Tags are developed to categorize video data into semantically distinct domains based on their visual content and scenario characteristics. The domains we define include Human, Robot, Physics, General, and Unsafe, where each video is assigned to exactly one domain to ensure unambiguous data partitioning.Such a structured domain classification not only allows us to conveniently filter out undesired or low-quality samples but also enables accurate and flexible control over the ratio of different data sources during world model training.In addition, we design a set of specialized sub-tags for each top-level domain.These fine-grained sub-tags support efficient and targeted retrieval of required video data, which greatly improves the efficiency of data selection and dataset construction. Table 2 shows the typical video domain tags, their corresponding sub-tag systems, and brief descriptions. Figure 11 (b) illustrates the distribution of sub-tags across each domain.

![](images/ba62873609c9b3b23c01c495fc67b24592046f905df25b6faa4ae2be3d900252.jpg)  
The robot successfully grasps an appe and lifts it from the wooden shelf. In front of the shelf is a shopping cart with a transparent plastic bag inside; around the apple on the shelf,there are also other fruits.  
The robot operates in a wel-litsupermarket environment.Itsuccessully grasps around,red apple with its right arm,lifts itfrom the wooden shelf,and places it into the shopping cart. While the right arm clamps and lifts the apple, the left arm remains stationary. A partial view of the shopping cart witha transparent plastic bag inside can be seen in front of the shelf; around the apple on the shelf, there are also other fruits such as bananas, oranges, starfruits, grapes, and peaches.  
Figure 12 Divide-and-Conquer caption. The content in the green box above is the caption directly generated by VLM, while the content in the blue box below is generated with reference to tags. In comparison, the caption below contains more detailed information and has a more complete expression.

To accomplish the tag annotation task for large-scale video data, we adopt an end-to-end automatic annotation pipeline based on the Qwen3-VL-8B [48] model. The specific implementation process is as follows: First, original videos are uniformly sampled at fixed time steps. On the premise of balancing the representativeness of frame sampling and annotation efficiency, we ensure that the extracted frames can effectively reflect the overall content characteristics of the videos. Subsequently, preset structured tag rules, annotation paradigms, and semantic constraints are integrated into prompt design, and the model is guided to efficiently complete tag inference and recognition of video frames through precise prompt engineering. Finally, the model’s inference results are stored in a structured JSON format, which not only facilitates batch parsing and efficient retrieval of subsequent data but also enables rapid association between tag information and video meta-information, providing convenience for the standardized management of large-scale video data. This automatic annotation scheme not only significantly improves the annotation efficiency of large-scale video data, effectively reducing the cost and error of manual annotation but also ensures the consistency and accuracy of tag annotation across the entire dataset. Furthermore, its output of a unified structured representation can effectively support data filtering, proportioning, and sampling needs, facilitating the optimization of data proportioning and laying a solid data foundation for world models to learn reliable visualsemantic correspondences and enhance training performance.

## 4.3.2 Captioning

For the caption generation task, we propose a dedicated generation method based on the Qwen3-VL-8B model [48], which requires the generated captions to comprehensively cover five core dimensions: (1) core subjects (humans, animals, objects); (2) subject actions; (3) surrounding environment and background details; (4) lighting and atmosphere; (5) camera motion. Example outputs of this baseline method are presented on the left side of Figure 12. To address hallucination issues and enhance the detail richness of captions, we further propose a Divide-and-Conquer (DC) caption generation strategy: instead of generating captions directly, this strategy produces captions by combining the structured tag information generated in the preceding stage with the inherent visual information of the video itself. It is verified that this strategy can significantly improve caption quality, and the output comparison between the original method and the DC caption strategy is displayed in the lower part of Figure 12.

![](images/28ce333bdcf872b13cfefddb74f84876cb2981c0d1ba5fc648c5af5328f4c260.jpg)

![](images/6cc37e49dbd7bc7bf1b1d8cd88a10729ebc1aadaec21b5659a13e2838c4a7977.jpg)

![](images/55601d10b459dedbc3c759fe2e69f7ddebe2357ccc6bfcbb91f9796bd6442d9e.jpg)

![](images/3ebdadcc2f2a45940c2b6cdfcbbf151e171e4157fb61df20f7d6f30da024b8b4.jpg)  
[PHENOMENON] The video shows many fish swimming in different directions around a coral ref. They move continuously and smoothly, speeding up or slowing down to avoid coral structures. Their positions, orientations, and speeds change constantly as they navigate naturally. [EXPLAINATION] The fish's motion belongs to mechanical motion. They generate forces via fin movements to propel themselves and adjust direction. Speed and orientation changes result from thrust production and body adjustments to navigate and avoid obstacles. The smooth movement reflects the balanced forces acting on the fish, such as water drag and fin thrust.  
[CAPTION] This is a first-person video clip that captures the whole process of making cereal with milk in a kitchen environment. On the wooden tabletop,there is a red electric kettle,a blue bowl,and a black microwave. A man dressed in a yellow top is standing by the table and carefully performing the steps to make cereal with milk. [PROCESs]1.Walking to the shelf, reaching out, and picking up the cereal box.2.Lifting the cereal box and pouring an appropriate amount of cereal from the box into an empty cup.3.Taking the milk carton, opening it, and getting ready to pour. 4.Slowly pouring milk into the cup that already contains cereal.5.Stirring the mixture of cereal and milk in the cup with a spoon evenly until well mixed.  
Figure 13 Enhanced Text with Chain of Thought.The left part of the figure shows examples of our physics-centric captions. These captions not only describe physical phenomena but also explain the underlying physical principles behind them. The right part presents long-term task captions, which not only describe the scenes, subjects, and tasks in the video, but also explicitly decompose tasks into concrete steps to enhance the model’s long-term task planning capability.

In addition, to further elevate caption quality, we design a multi-model ensemble annotation pipeline. Specifically, we first leverage multiple medium-parameter multimodal models (including Qwen3-VL-8B [48], InternVL3.5-8B [49], Mimo-7B [50], MiniCPM V4.5 [51], etc.) to independently generate initial video descriptions. Then, we adopt the large-parameter model Qwen3-VL-8B [48] as the core fuser: by referencing the consensus content from the outputs of medium-parameter models, it supplements finer and more complete details with its strong multimodal understanding capability. Due to the high time cost of this pipeline, we only apply it to high-quality continuous training data to balance efficiency and data quality.

## 4.3.3 Enhanced Text with Chain of Thought

Physics-centric. The core of world models lies in learning and reproducing the operational logic of the real world, where physical laws serve as the key element to achieve this goal and represent the fundamental characteristic that distinguishes world models from ordinary generative models—namely, their "world cognition capability." To fully exploit the core value of physical principles and ensure the physical realism of world models, we construct specialized text datasets that explicitly reinforce physical laws from the data source. Specifically, for data tagged as "physics", we explicitly strengthen the expression of physical laws within their captions. Our physics-centric caption data not only describes the surface phenomena observed in videos but also explains the underlying physical principles embedded in these phenomena, including dynamic behavioral constraints such as object motion trajectories, collision interactions, gravitational effects, and frictional forces. Through this approach, we ensure that the content generated by the model conforms to real-world physical constraints, thereby enhancing physical realism. The left part of Figure 13 presents a typical example of Physics-centric captions.

Long-horizon Tasks. Causality and long-horizon sequence consistency constitute another core element that distinguishes world models from ordinary generative models. To ensure the logical consistency and rationality of long-horizon task execution, we focus on strengthening chain-ofthought construction and task decomposition capabilities, creating datasets that target long-horizon sequence scenarios. For selected human manipulation and robot interaction data, we construct a long-horizon task-oriented annotation framework by clearly decomposing task steps and delineating the causal logical chains between events. Specifically, we decompose complex task captions into multiple executable sub-steps, explicitly defining the dependencies and causal relationships between each step, enabling the model to understand and reproduce task execution logic over extended temporal sequences. This design effectively ensures the logical coherence of the model when handling long-horizon sequence scenarios. The right part of Figure 13 illustrates a typical example of task decomposition in Long-horizon tasks data.

## 4.4 Data Engineering Infrastructure

In the data pre-processing phase, we built an efficient data engineering infrastructure that achieves significant performance improvements for three core operators: Shot Detection, Frame Filtering, and Captioning. Our optimization strategy focuses on three dimensions: computational parallelization, I/O optimization, and task scheduling, as shown in Table 3.

Table 3 Engineering optimization for Data Infrastructure.
<table><tr><td>Operator</td><td>Baseline (hrs)</td><td>Optimized (hrs)</td><td>Speedup</td><td>Key Techniques</td></tr><tr><td>Shot Detection</td><td>1,169.6</td><td>8,640.0</td><td>7.4×</td><td>Dist． scheduling, frame skip, load balancing</td></tr><tr><td>Frame Filtering</td><td>612.0</td><td>18,332.7</td><td>29.9×</td><td>FP16,CPU decode, concurrent scheduling</td></tr><tr><td>Captioning</td><td>137.0</td><td>4,665.9</td><td>34.0×</td><td>CPU decode, pipeline, two-level batching</td></tr></table>

Shot Detection Operator Optimization. The shot detection operator segments long videos into semantically coherent clips. We employ a combination of multiple detectors on a single machine with 8×4090 GPUs and 180 vCPUs. Key techniques include: distributed scheduling to replace the original pipeline; resolution downscaling and frame skipping to reduce redundant computation; dynamic worker allocation to increase parallelism; and duration-based task partitioning for load balancing. These optimizations improve throughput from 1,169.6 to 8,640.0 hours/day (7.4×), while maintaining a shot detection recall rate of 77.44%.

Frame Filtering Operator Optimization. The frame filtering operator conducts multi-dimensional quality assessment, including luminance, blur, aesthetics, and pose detection. We improve throughput from 612.0 to 18,332.7 hours/day (29.9×) through: pre/post-processing refactoring to reduce redundancy; adaptive frame sampling; CPU-concurrent decoding to eliminate serial I/O bottlenecks; FP16 mixed precision inference; and automatic resource scheduling with zero-copy video access to

achieve computation-I/O overlap.

Caption Operator Optimization. The captioning operator generates textual descriptions for video clips. We improve throughput from 137.0 to 4,665.9 hours/day (34.0×) by reusing CPUconcurrent decoding, adaptively sampling key segments, and decoupling video loading from model inference into a pipeline architecture with a two-level batching mechanism to balance throughput and memory utilization.

Overall, our data engineering infrastructure achieves over 30× end-to-end throughput improvement on a single machine with 8×4090 GPUs, with key contributions from: (1) Computational Parallelization via distributed scheduling and load-balanced task partitioning; (2) I/O Optimization via CPU-concurrent decoding, zero-copy streaming, and pipeline overlap; and (3) Task Scheduling via two-level batching and dynamic resource allocation across sub-modules.

## 5 Inference

## 5.1 Self-evolution

The vision of the Kairos framework extends far beyond serving as a static inference engine; it is natively designed as a deployable infrastructure capable of automated model upgrades through self-evolutionary learning. This capability is fundamentally rooted in Kairos’s unified understandinggeneration-prediction architecture. By tightly coupling these three pathways, the model is inherently equipped to operate within the observation-action-feedback closed loop, enabling continuous refinement of its internal representations and decision-making strategies.

In practice, this self-evolution is realized through the rollout-evaluation-refinement cycle as shown in Figure 14. Upon receiving an instruction, the generation and prediction modules simulate multiple physically plausible future rollout paths and action trajectories. Leveraging the Chain-of-Thought (CoT) framework, Kairos’s understanding module acts as a built-in reward function and internal reasoning agent, analyzing, scoring, and ranking these diverse paths. This internal reflective process enables the system to identify optimal strategies, systematically correct errors, and iteratively improve its precision over time.

As another core self-evolutionary strategy, this mechanism can be validated by instantiating prompt rewriting agents. Building upon our prompt alignment strategy, the rewriter (i.e., the prompt template) serves as an active evolutionary proxy. The understanding module scores the outputs generated by different prompts, dynamically evaluates and rewrites user instructions, and constructs a localized self-improving loop. We empirically observe that this automated process directly enhances the model’s precision and alignment performance in deployed generation scenarios.

While prompt optimization and trajectory selection represent successful initial validations, this self-evolutionary paradigm is equally applicable to the policy configurations of the World Action Model (WAM). By enabling the model to autonomously simulate, evaluate, and refine its policy parameters within the closed-loop understanding framework, Kairos can continuously improve its decision-making and execution strategies for physical interactions. We will thoroughly investigate this application in our future work.

## 5.2 Prompt Self-alignment

In real-world deployment scenarios, user queries and prompts exhibit significant variations in expression styles, level of detail granularity, and linguistic conventions. Current mainstream prompt alignment solutions, adopted by leading text-to-video models such as Wan[26] and Seeddance2.0[52], universally follow the static template mapping technical approach: they convert user inputs into standardized dense caption formats through manual induction of model-preferred prompt paradigms and best practices. While this method is simple and effective, it fundamentally relies on the accumulation of human expertise, rendering it incapable of adaptively updating alongside iterative improvements in model capabilities and insufficient to cover the infinite diversity of user inputs.

![](images/ca991cbd8189894b11198512dfaa6514b538859f75fa5281f0af919eb733719a.jpg)  
Figure 14 Kairos Self-evolution framework. It follows a closed-loop rollout-evaluation-refinement cycle for continuous improvement.

Built upon Kairos’s unified self-evolutionary infrastructure, we propose a closed-loop Prompt Self-Alignment strategy following the generation-evaluation-iterative refinement paradigm, which fundamentally eliminates the reliance on static templates. This strategy strictly adheres to Kairos’s general "rollout-evaluation-refinement" self-evolutionary paradigm and is co-driven by two core components: a prompt rewriting agent with self-reflective and iterative capabilities, and a specially designed multi-dimensional video quality reward function. This reward function comprehensively quantifies the physical plausibility, task completion rate, and overall visual quality of generated videos, providing objective and consistent evaluation criteria for the self-evolutionary process.

The implementation workflow is as follows: Upon receiving a user instruction, the prompt rewriting agent first generates an initial batch of diversified rewritten candidates. All candidate prompts are fed into the video generation model to obtain corresponding outputs, which are then subjected to fully automated comprehensive evaluation by the aforementioned multi-dimensional reward function. The system feeds back the evaluation results along with their corresponding candidate prompts to the rewriting agent, enabling it to perform self-reflection, defect attribution, and evolutionary refinement based on actual generation outcomes, thereby producing a new round of higher-quality prompt candidates. After multiple rounds of closed-loop iteration, the system gradually converges to the optimal prompt version, ultimately achieving significant improvement in the comprehensive quality of generated videos. The entire process requires no human intervention and can automatically adapt to the evolution of model capabilities and drift in user input distribution over the course of

deployment.

## 5.3 Inference Efficiency

World generation within embodied AI applications imposes two contradictory yet critical operational requirements. First, to power massive data simulation platforms on cloud infrastructure, the world model must achieve hyper-low latency and high-throughput video synthesis to accelerate policy rollouts. Second, to democratize development and facilitate rapid prototyping, the inference stack must remain highly cost-effective, allowing individual researchers to execute the model on resourceconstrained, consumer-grade computing hardware. To reconcile the tension between generation fidelity and operational cost across these distinct environments, Kairos stack introduces a deploymentaware optimization framework. We systematically address these efficiency bottlenecks through two complementary vectors: Timestep Distillation, which structurally compresses the diffusion sampling trajectories to minimize theoretical latency bounds, and Hardware-aware Inference Optimization, which co-designs low-level computational kernels and memory footprints to maximize hardware utilization. The formal mechanisms of these optimizations are detailed below.

## 5.3.1 Timestep Distillation

High-resolution embodied world models can generate realistic environment dynamics and agent interactions, but their diffusion-based iterative sampling often requires dozens or even hundreds of denoising steps, creating a major computational bottleneck for real-time deployment. To address this issue, we distill a pretrained 480P Embodied World Model into an efficient 4-step generator. Following Eq. 13, noisy samples $\mathbf { z } _ { \sigma }$ are constructed by perturbing clean samples ${ \bf z } _ { 0 } ^ { T }$ from the pretrained teacher model. The student model is then trained to approximate the teacher distribution using significantly fewer sampling steps.

Distribution Matching Distillation. We adopt Distribution Matching Distillation (DMD) [53, 54] to distill the teacher distribution into a compact generator. The student is parameterized as a feed-forward generator

$$
\begin{array} { r } { \mathbf { z } _ { 0 } ^ { \theta } = G _ { \theta } ( \pmb { \xi } ) , } \end{array}\tag{21}
$$

which maps Gaussian noise $\boldsymbol { \xi }$ directly to the data space with only a few sampling steps.

Since the student defines an implicit distribution whose score function is unavailable in closed form, we introduce an auxiliary fake-score network $\phi .$ . Under the Rectified Flow formulation, $\phi$ learns the velocity field through

$$
\mathcal { L } _ { \mathrm { f a k e } } = \mathbb { E } _ { \mathbf { z } _ { 0 } ^ { \theta } , \epsilon , \sigma } \left[ \left\| \mathcal { V } _ { \phi } ( \mathbf { z } _ { \sigma } , \sigma ) - ( \epsilon - \mathbf { z } _ { 0 } ^ { \theta } ) \right\| _ { 2 } ^ { 2 } \right] .\tag{22}
$$

The predicted velocity can be re-parameterized as a reconstruction of the clean sample

$$
\hat { \mathbf { z } } _ { 0 } ^ { \theta } = \frac { \mathbf { z } _ { \sigma } - \sigma \mathcal { V } _ { \phi } ( \mathbf { z } _ { \sigma } , \sigma ) } { \alpha _ { \sigma } } ,\tag{23}
$$

which yields the corresponding student score

$$
{ \bf s } _ { \phi } ( { \bf z } _ { \sigma } , \sigma ) = - \frac { { \bf z } _ { \sigma } - \alpha _ { \sigma } \hat { \bf z } _ { 0 } ^ { \theta } } { \sigma ^ { 2 } } .\tag{24}
$$

This estimated score serves as a surrogate for the student’s score field and enables distribution matching against the teacher during distillation. As the supervision signal, we employ the teacher’s

![](images/951c4ab7bf63deeda2de9d7f13a389e26756633f20be7696420be53bda31f6de.jpg)  
Figure 15 Performance of Distilled Kairos Robot Model on PAI-Bench Dataset

classifier-free guidance (CFG) score:

$$
\begin{array} { r } { \mathbf { s } _ { T } = ( 1 + w ) \mathbf { s } _ { T } ( \mathbf { z } _ { \sigma } , \sigma , \mathbf { c } _ { \mathrm { p o s } } ) - w \mathbf { s } _ { T } ( \mathbf { z } _ { \sigma } , \sigma , \mathbf { c } _ { \mathrm { n e g } } ) , } \end{array}\tag{25}
$$

where $\mathbf { c } _ { \mathrm { p o s } }$ and $\mathbf { c } _ { \mathrm { n e g } }$ denote positive and negative prompt embeddings, respectively. Distilling the CFG-enhanced teacher score enables the student to inherit both the teacher’s generation quality and guided sampling behavior. Moreover, the discrepancy between positive and negative conditioning maintains a non-vanishing optimization signal even when the student approaches the unguided teacher distribution, mitigating optimization stagnation.

To align the student distribution $p _ { \theta }$ with the teacher distribution $p _ { T }$ , we minimize the forward KL divergence $D _ { \mathrm { K L } } ( p _ { \boldsymbol { \theta } } \Vert p _ { T } )$ . Since the student distribution is implicit, directly optimizing the KL objective is intractable. Using score-function identities, its gradient can be expressed as

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { D M D } } = \mathbb { E } _ { \mathbf { z } _ { 0 } ^ { \boldsymbol { \theta } } , \boldsymbol { \epsilon } , \boldsymbol { \sigma } } \left[ \omega ( \boldsymbol { \sigma } ) \left( \mathbf { s } _ { T } - \mathbf { s } _ { \boldsymbol { \phi } } ( \mathbf { z } _ { \boldsymbol { \sigma } } , \boldsymbol { \sigma } ) \right) \frac { \partial \mathbf { z } _ { \boldsymbol { \sigma } } } { \partial \mathbf { z } _ { 0 } ^ { \boldsymbol { \theta } } } \frac { \partial \mathbf { z } _ { 0 } ^ { \boldsymbol { \theta } } } { \partial \boldsymbol { \theta } } \right] ,\tag{26}
$$

where $\omega ( \sigma )$ is a noise-dependent weighting factor. Consequently, the DMD gradient estimator encourages the student score field to match the teacher score field, thereby aligning the student distribution with the teacher distribution.

When applying DMD to embodied world models, we observe several characteristic degradation phenomena. First, prolonged training often leads to instability and mode collapse, producing repetitive or distorted generations. Second, the distilled generator exhibits a motion diminution effect, tending to favor conservative trajectory updates and consequently underestimating agent dynamics. Third, we observe visual homogenization, where scene diversity gradually decreases and backgrounds converge toward simplified textures. These observations suggest that distribution matching alone is insufficient to fully preserve the temporal dynamics and geometric fidelity of the teacher’s trajectories, motivating the additional regularization strategies introduced in the following sections.

Consistency Distillation. To further stabilize training and preserve the teacher’s generation trajectory, we introduce a continuous-time consistency objective based on consistency models (CM) [55–57]. While DMD aligns the student with the teacher at the distribution level, CM additionally enforces trajectory consistency by encouraging the student prediction at a noisy state to match the teacher prediction at a neighboring point along the teacher ODE trajectory. This regularization improves training stability and helps preserve fine-grained temporal and structural details.

Given a noisy sample ${ \bf z } _ { \sigma _ { n + 1 } }$ , we first construct its neighboring state on the teacher ODE trajectory using a single Euler step:

$$
\begin{array} { r } { \hat { \mathbf { z } } _ { \sigma _ { n } } = \mathbf { z } _ { \sigma _ { n + 1 } } + ( \sigma _ { n } - \sigma _ { n + 1 } ) \mathcal { V } _ { \mathrm { t e a } } ( \mathbf { z } _ { \sigma _ { n + 1 } } , \sigma _ { n + 1 } ) , } \end{array}\tag{27}
$$

where $\nu _ { \mathrm { t e a } }$ denotes the teacher velocity field.

The consistency objective is then defined as

$$
\mathcal { L } _ { \mathrm { C M } } = \mathbb { E } _ { \mathbf { z } _ { \sigma _ { n + 1 } } , n } \left[ \lambda ( \sigma _ { n } ) \left. \mathbf { f } _ { \boldsymbol { \theta } } ( \mathbf { z } _ { \sigma _ { n + 1 } } , \sigma _ { n + 1 } ) - \mathbf { f } _ { \mathrm { t e a } } ( \hat { \mathbf { z } } _ { \sigma _ { n } } , \sigma _ { n } ) \right. _ { 2 } ^ { 2 } \right] ,\tag{28}
$$

where $\mathbf { f } _ { \theta }$ and $\mathbf { f } _ { \mathrm { t e a } }$ denote the student and frozen teacher models, respectively. Optimizing $\mathcal { L } _ { \mathrm { C M } }$ enables the student to approximate the teacher’s multi-step generation trajectory with significantly fewer sampling steps while maintaining high fidelity.

Hybrid Objective. Our final training objective integrates the strengths of both distributional alignment and trajectory consistency. Specifically, we jointly optimize the consistency objective and the DMD score-matching objective:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { C M } } + \lambda _ { \mathrm { s c o r e } } \mathcal { L } _ { \mathrm { D M D } } ,\tag{29}
$$

where $\lambda _ { \mathrm { s c o r e } }$ balances the two terms. This hybrid formulation creates a synergistic effect: $\mathcal { L } _ { \mathrm { C M } }$ stabilizes the distillation process by anchoring the student to the teacher’s trajectory and preserving structural integrity, while $\mathcal { L } _ { \mathrm { D M D } }$ leverages CFG-guided teacher supervision to improve generation fidelity and perceptual quality.

Qualitative Results on Paibench. Fig. 15 presents qualitative comparisons on the Paibench benchmark under the TI2V setting. Despite requiring only four inference steps, the distilled generator faithfully reproduces the teacher’s spatial structure, motion dynamics, and physical interactions. Fine-grained scene details and coherent object trajectories are well preserved, while dynamic agents exhibit stable and realistic motions.

Compared with the original teacher sampling process, the distilled model achieves comparable visual quality and temporal consistency with substantially reduced sampling cost. These results demonstrate that the proposed distillation framework effectively transfers both distributional knowledge and trajectory dynamics from the teacher model, enabling high-fidelity embodied video generation with efficient 4-step inference.

## 5.3.2 Hardware-aware Inference Optimization

Low-Latency Generation on Cloud Service Platforms. This aims to achieve low-latency and rapid acquisition of generated video data on cloud service platforms to meet user demand for fast interactive experience the video generation function in embodied scenarios.

Mixed-Parallel Inference Optimization. While the Kairos DiT model has a moderate parameter count, but exhibits extremely long input sequence lengths in each attention block,we adopt a parallel strategy centered on sequence parallelism, supplemented by tensor parallelism. We further incorporate design insights from Megatron-LM Sequence Parallelism and DeepSpeed-Ulysses Sequence Parallelism,and propose a customized hybrid parallelism scheme tailored to our model architecture.

• Slide Window Attention Block: Using ulysses sequence parallelism, each GPU maintains the full weights, and the input is split along the sequence dimension. Data synchronization between GPU is achieved via All-To-All communication, with each GPU only responsible for Attention computation for its assigned heads.

• Cross Attention Block: Adopting the basic Sequence Parallel method. Since the context information of the Attention KV is limited, the results can be precomputed and cached in full on each GPU. We only split the query sequence along the sequence dimension, enabling each GPU to compute Attention using its local query fragment and the global full key-value (KV) pairs. The results are then aggregated via All-Gather.

• Gated Delta Net: Adopting a modified Tensor Parallel method. The weights are split by head and distributed across different GPU. Each GPU receives the full input sequence, but processes attention computation in micro-batches to reduce memory usage.

• VAE decoder: The VAE decoder divides the video into multiple segments along the timeline, and executes these segments in parallel across different GPUs.

DiT-Cache Optimization. TeaCache acts as a dedicated optimization accelerator for Kairos DiT. Reuses calculation results correlated with time steps, which significantly reduces inference latency and GPU memory footprint, thus improving overall inference efficiency without altering the model structure.

Compilation and Kernel Fusion Optimization. Torch.compile directly accelerates neural network training and inference by automatically applying graph optimizations, kernel fusion, and hardware-specific optimizations. In addition, we implement a set of dedicated fusion operators to improve performance.

Cost-Effective Computation on Consumer-Grade Devices. This aims to enable cost-effective inference computing on consumer-grade low-memory computing devices to meet the prototype development and usage needs of individual developers.

Low-Precision Computing Optimization (FP8).We apply quantization to the attention layers but not to linear layers. This mainly involves the following technical points:

• Q/K in INT8/INT4: Keep query (Q) and key (K) in INT8 or INT4 for faster QK computation, while ensuring their outputs match the dynamic range of FP8 for the subsequent PV MatMul.

• PV Matmul (P·V) in FP8: Quantize the attention weight matrix P (after softmax) and the value matrix V to FP8, leveraging hardware-accelerated FP8 Tensor Core instructions.

• Smooth Q by computing Q = Q - mean(Q) (channel-wise mean subtraction) to narrow the value distribution, improving the precision of INT4/INT8 quantization.

• Per-thread / per-warp quantization for Q/K: Finer granularity (16× smaller than per-block) boosts quantization precision without extra overhead, critical for FP8 downstream computation.

Hardware-Aware Memory Optimization. Due to the stringent memory capacity constraints inherent to consumer-grade graphics hardware, Kairos implements specialized low-level architectural optimizations to drastically compress the runtime memory footprint:

• Tiled Gated DeltaNet with Streaming Access: Beyond conventional intra-GPU tensor parallelization across attention heads, we introduce an intra-sequence batching and tiling mechanism tailored for long-horizon tokens. By deploying a Tile-Based Computation & Streaming Access paradigm, the network partitions extensive sequence dimensions into discrete, highly localized blocks (tiles). Queries (Q), Keys (K), and Values (V ) are subsequently processed in a synchronized streaming pipeline. This strategy maximally overlaps hardware tensor computations with asynchronous memory transactions, effectively concealing DRAM access latency and avoiding out-of-memory (OOM) triggers during long-horizon generation.

• Weight-Only INT4 Text Encoder Quantization: To mitigate the massive memory overhead imposed by text conditioning, the text encoder utilizes an aggressive INT4 weight-only quantization protocol. By preserving high-precision activations while compressing stationary model weights to 4-bit representations, this scheme maximizes computational throughput and drastically reduces structural memory allocation. Empirically, this design facilitates sub-millisecond keyword grounding with negligible semantic accuracy degradation, making high-fidelity world-action generation highly viable on standard edge-computing hardware.

## 5.3.3 Efficiency Comparison

We tested the performance of the current model on chips with varying architectures and types, as shown in the following figure. Experimental results show that Kairos achieves strong generation performance on both professional server GPUs and consumer-grade GPUs.

Table 4 Latency Comparison on Various Hardware Platforms, and the results are tested on Kairos-4B-robot 480P(5s) distillation model.
<table><tr><td>GPU</td><td>Resolution</td><td>Memory(GB)</td><td>1GPU (s)</td><td>4GPU (s)</td></tr><tr><td>Nvidia A800</td><td>480P</td><td>23.5</td><td>11.7</td><td>3.0</td></tr><tr><td>Nvidia RTX5090</td><td>480P</td><td>13.9</td><td>11.4</td><td>5.7</td></tr></table>

Table 4 represents latency comparison on various hardware platforms. These results confirm that our model maintains robust generation performance on both professional and consumer-grade GPUs, while its efficient memory utilization supports the generation of longer videos and higher-resolution visual content. Notably, 480P video generation on the Nvidia A800 achieves real-time performance.

Table 5 Latency Comparison on Various Models. The evaluation was conducted in TI2V mode, with a video resolution of 720P and a duration of 5 seconds.
<table><tr><td>Model</td><td>Memory (GB)</td><td>Complexity (PFlops)</td><td>1GPU (s)</td><td>4GPU (s)</td></tr><tr><td>Lingbot-28B [10]</td><td>46.1</td><td>347.4</td><td>5525</td><td>1436</td></tr><tr><td>Cosmos-Predict2.5-14B [2]</td><td>70.2</td><td>156.5</td><td>2526</td><td>687</td></tr><tr><td>Wan2.2-5B [26]</td><td>23.4</td><td>16.6</td><td>201</td><td>85</td></tr><tr><td>Kairos-4B</td><td>23.5</td><td>2.3</td><td>43</td><td>9</td></tr></table>

To further contextualize these findings, we conducted performance tests to evaluate performance discrepancies among different models under identical configurations, as presented in the table 5. Table 5 compares the memory usage, computational complexity, and inference latency of four models on an NVIDIA A800 server. Among them, Kairos-4B demonstrates outstanding efficiency: it consumes only 23.5 GB of memory (comparable to the lightweight Wan2.2-5B), boasts the lowest computational complexity (2.3 PFlops), and achieves the fastest inference speeds—43 seconds on 1 GPU and 9 seconds on 4 GPUs—far outperforming Lingbot-28B, Cosmos-Predict2.5-14B, and Wan2.2-5B across all evaluated metrics.

To eliminate confounding effects from other components in these models, we tested the performance differences of the single-step DiT model across multiple resolutions and various video generation durations, as shown in the Figure 3 (b). We can clearly observe the time cost (inference latency) of four models across different resolutions and video durations, which directly reflects the performance advantages of Kairos-4B:

• The Lowest Latency across All Test Scenarios. Under all combinations of 480p/720p resolutions and 5s/10s/15s durations, Kairos-4B consistently achieves the lowest latency, outperforming all competing models.

• Orders-of-Magnitude Speedup over Larger Models. Compared to Cosmos-Predict2.5- 14B, Kairos-4B achieves a 28×–85× latency reduction. Even against the smaller Cosmos-Predict2.5-2B model, it maintains a performance advantage of 6× to 23×, validating the effectiveness of our optimization strategies.

• Superior Efficiency over Similar-Parameter Competitors. Relative to Wan2.2-5B, Kairos-4B delivers a 2.5× to 3.7× speedup, demonstrating higher computational efficiency despite its smaller parameter scale.

• Stable scalability under increasing workloads. As resolution and duration rise (from 480p 5s to 720p 15s), other models show exponential latency growth, while Kairos-4B scales linearly with increased workload, making it highly suitable for long-duration, high-resolution video generation.

## 6 Evaluation Results

Kairos models demonstrate strong performance across a diverse set of embodied AI and world modeling benchmarks. In the following sections, we evaluate our models from two perspectives:

embodied scenarios and general world modeling tasks, highlighting their capabilities in robotic reasoning and policy generation, as well as their competitive performance on broader video generation benchmarks.

## 6.1 Embodied World Model Benchmarks

To evaluate the capabilities of our embodied foundation model Kairos-4B, we conducted comprehensive evaluations across three benchmark datasets: WorldModelBench [12], DreamGen Bench [13] and PAI-Bench [58]. We systematically assessed its performance across multiple core dimensions, including visual quality, physical plausibility, geometric consistency, text-video alignment, and controllability. We benchmark Kairos-4B against a wide range of mainstream world models spanning different parameter scales. Quantitative results demonstrate that Kairos-4B achieves state-of-the-art performance. In addition to quantitative evaluation, we conducted a comprehensive human evaluation. Subjective assessment results show that Kairos-4B outperforms other baseline models in metrics such as video quality, physical plausibility, and task completion rate, further corroborating its superior overall performance. The combined results from both quantitative and human evaluations confirm that Kairos-4B can consistently generate high-quality scenes that conform to physical laws and exhibit strong spatial consistency across diverse object layouts and environmental conditions. These findings establish that the embodied video generation capabilities of Kairos-4B have reached the current state-of-the-art level.

## 6.1.1 Benchmark Results

WorldModelBench-robot. WorldModelBench [12] is designed to assess the world modeling capability of video generation models, particularly their ability to follow instructions and adhere to real-world physics across diverse domains. It primarily evaluates models along two dimensions: (1) Instruction Following, which measures whether the generated videos accurately follow the given text prompts (and image), and (2) Future Frame Generation, which assesses whether the generated videos represent plausible future states of the world, including adherence to physical laws and common sense reasoning. As these capabilities are essential for embodied reasoning, we conduct our evaluation on the robotics subset of WorldModelBench.

As shown in Table 6, Kairos-4B achieves the highest total score of 9.30 on the WorldModelBench robot subset, outperforming all baselines. It obtains the best Instruction Following score (2.36), matching the 16B Cosmos3-Nano, which indicates its strong language grounding capability. In Physics Adherence, Kairos-4B reaches perfect scores in Newtonian mechanics, fluid dynamics, and gravity (1.00 each), achieving a high overall Physics Adherence score of 4.96. Additionally, it demonstrates robust Common Sense reasoning, highlighted by perfect temporal quality (1.00). Overall, these results show that Kairos-4B delivers leading physical modeling and reasoning capabilities while remaining highly parameter-efficient. Figure 16 presents qualitative results of selected samples from the WorldModelBench robot subset.

DreamGen Bench. DreamGen Bench [13] is a video generation benchmark specifically designed for robotics, aiming to systematically measure the generalization capability of Video World Models on specific robotic embodiments. The benchmark primarily evaluates two core metrics: Instruction Following, which assesses whether generated videos strictly adhere to given task instructions; and Physics Adherence, which evaluates whether generated videos conform to real-world physical laws. DreamGen Bench quantifies the generalization ability of video generation models by testing their performance across three key dimensions: novel object manipulation, novel behavior execution, and novel environment adaptation. Research results indicate that models achieving higher scores on this benchmark also demonstrate better performance in downstream robot policy training when using their generated synthetic data, showing a significant positive correlation between the two.

Table 6 Evaluation on WorldModelBench Robot Set. For each column, the highest score is bolded. Models marked with \* indicate results reproduced by our team.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Param</td><td rowspan="2">Instruction Following</td><td colspan="6">Physics Adherence</td><td rowspan="2">Common Sense</td><td rowspan="2"></td><td rowspan="2">Total Score</td></tr><tr><td>Overall</td><td>Newton</td><td>Deform. Fluid</td><td>Penetr.</td><td>Grav.</td><td>Overall Frame</td></tr><tr><td>Lingbot* [10]</td><td>28B</td><td>2.14</td><td>1.00</td><td>0.96</td><td>1.00</td><td>0.96</td><td>1.00</td><td>4.92</td><td>1.00</td><td>Temp. 0.98</td><td>9.04</td></tr><tr><td>Cosmos3-Nano* [59]</td><td>16B</td><td>2.36</td><td>1.00</td><td>0.98</td><td>1.00</td><td>0.98</td><td>1.00</td><td>4.96</td><td>0.98</td><td>0.96</td><td>9.26</td></tr><tr><td>Abot-Physworld* [60]</td><td>14B</td><td>2.10</td><td>1.00</td><td>0.92</td><td>1.00</td><td>0.96</td><td>1.00</td><td>4.88</td><td>1.00</td><td>0.98</td><td>8.96</td></tr><tr><td>Cosmos-Predict2.5* [2]</td><td>14B</td><td>2.14</td><td>1.00</td><td>0.92</td><td>1.00</td><td>0.94</td><td>1.00</td><td>4.86</td><td>1.00</td><td>0.94</td><td>8.94</td></tr><tr><td>Wan2.2* [26]</td><td>5B</td><td>2.04</td><td>1.00</td><td>0.78</td><td>1.00</td><td>0.86</td><td>0.98</td><td>4.62</td><td>0.96</td><td>0.90</td><td>8.52</td></tr><tr><td>Cosmos-Predict2.5* [2]</td><td>2B</td><td>2.14</td><td>1.00</td><td>0.98</td><td>1.00</td><td>0.96</td><td>1.00</td><td>4.94</td><td>0.98</td><td>0.98</td><td>9.04</td></tr><tr><td>GigaWorld-0* [61]</td><td>2B</td><td>1.50</td><td>1.00</td><td>0.98</td><td>1.00</td><td>1.00</td><td>1.00</td><td>4.98</td><td>0.98</td><td>1.00</td><td>8.46</td></tr><tr><td> Kairos</td><td>4B</td><td>2.36</td><td>1.00</td><td>0.98</td><td>1.00</td><td>0.98</td><td>1.00</td><td>4.96</td><td>0.98</td><td>1.00</td><td>9.30</td></tr></table>

Input frame  
Predicted frames  
![](images/c07d1e89292b957eefc940c9eb4a72f6fbb9a1096213f0531f64c973ad0d5c81.jpg)  
Figure 16 Kairos samples on the WordelModelBench robot subset.

Table 7 presents the evaluation results of Kairos-4B on the DreamGen Bench. Our model ranks first in both Average Physical Adherence (AVG\_PA, 0.538) and the overall Average Score (AVG\_- Score, 0.618), while achieving a highly competitive Average Instruction Following (AVG\_IF, 0.698), second only to the 14B Wan2.2 (0.703). The leading AVG\_PA score demonstrates the model’s precise physical world modeling, and its strong AVG\_IF reflects reliable instruction-following ability. Despite having only 4B parameters, Kairos-4B attains the best overall performance, outperforming substantially larger competitors, which highlights its strong performance and generalization capability. Figure 17 displays selected visualization samples from our DreamGen test set.

PAI-Bench-robot. PAI-Bench [58] is a benchmark designed to evaluate the visual quality and physical plausibility of generated videos in real-world physical AI scenarios. The benchmark reports two primary metrics: Domain Score and Quality Score. The Domain Score evaluates the model’s capability on domain-specific physical AI tasks, while the Quality Score measures the perceptual quality of generated videos. The Quality Score is computed using eight evaluation metrics adapted from VBench [62]. Meanwhile, the Domain Score is obtained through a VQA-based evaluation protocol spanning seven domains, including av, common, human, industry, misc, physics, and robotics. The final Overall Score is defined as the average of the Domain Score and the Quality Score. To specifically assess the model’s capability in embodied scenarios, we conduct quantitative evaluation on the Robotics subset of PAI-Bench.

Table 7 Evaluation on DreamGen Bench. For each column, the highest score is bolded. Models marked with \* indicate results reproduced by our team.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Param</td><td colspan="2">GR1-Object</td><td colspan="2">GR1-Behavior</td><td colspan="2">GR1-Env</td><td rowspan="2">AVG_PA</td><td rowspan="2">AVG_IF</td><td rowspan="2">AVG_Score</td></tr><tr><td>Qwen-IF</td><td>PA</td><td>Qwen-IF</td><td>PA</td><td>Qwen-IF</td><td>PA</td></tr><tr><td>Cosmos-Predict2.5*[2]</td><td>14B</td><td>0.260</td><td>0.515</td><td>0.553</td><td>0.418</td><td>0.621</td><td>0.553</td><td>0.495</td><td>0.478</td><td>0.487</td></tr><tr><td>Cosmos-Predict2.5* [2]</td><td>2B</td><td>0.840</td><td>0.374</td><td>0.277</td><td>0.375</td><td>0.586</td><td>0.507</td><td>0.419</td><td>0.568</td><td>0.494</td></tr><tr><td>GigaWorld-0 [61]</td><td>2B</td><td>0.540</td><td>0.481</td><td>0.638</td><td>0.446</td><td>0.586</td><td>0.529</td><td>0.485</td><td>0.588</td><td>0.537</td></tr><tr><td>Wan2.2* [26]</td><td>5B</td><td>0.420</td><td>0.458</td><td>0.553</td><td>0.180</td><td>0.690</td><td>0.303</td><td>0.314</td><td>0.554</td><td>0.434</td></tr><tr><td>Wan2.2 [26]</td><td>14B</td><td>0.780</td><td>0.531</td><td>0.570</td><td>0.477</td><td>0.760</td><td>0.549</td><td>0.519</td><td>0.703</td><td>0.611</td></tr><tr><td>Cosmos3-Nano* [59]</td><td>16B</td><td>0.460</td><td>0.509</td><td>0.468</td><td>0.455</td><td>0.793</td><td>0.552</td><td>0.505</td><td>0.574</td><td>0.540</td></tr><tr><td>ABot-PhysWorld* [60]</td><td>14B</td><td>0.400</td><td>0.493</td><td>0.404</td><td>0.431</td><td>0.414</td><td>0.528</td><td>0.484</td><td>0.434</td><td>0.459</td></tr><tr><td>Lingbot* [10]</td><td>28B</td><td>0.400</td><td>0.545</td><td>0.617</td><td>0.340</td><td>0.690</td><td>0.513</td><td>0.466</td><td>0.569</td><td>0.518</td></tr><tr><td> Kairos</td><td>4B</td><td>0.660</td><td>0.544</td><td>0.745</td><td>0.489</td><td>0.690</td><td>0.581</td><td>0.538</td><td>0.698</td><td>0.618</td></tr></table>

Input frame  
Predicted frames  
![](images/334c5f4ad73619483bf8da4bf20bd6f12dee574c8006a6e7320c2e37f5a457b6.jpg)  
Figure 17 Kairos samples on the DreamGen dataset.

Table 8 presents the benchmark results between Kairos-4B and baseline models under the PAIBench TI2V evaluation mode. Among small-scale models (<10B), our model achieves the best overall performance, ranking first in both Domain Score (88.59) and Overall Score (82.57). Notably, using only 4B parameters, Kairos-4B matches or surpasses many large-scale (≥10B) baseline models in overall score, outperforming the 16B Cosmos3-Nano and several 14B models such as Cosmos-Predict2.5 and Wan2.1. These results indicate that our model not only generates high-quality videos but also delivers strong physical modeling and instruction-following capability with far fewer parameters. Beyond quantitative analysis, qualitative evaluation offers more intuitive insights into the model’s generation behavior. Figure 18 showcases the visualization samples from the PAIBench-robot test set, illustrating the model’s generation quality across diverse scenarios.

Human Evaluation Video generation involves complex physical causal logic and semantic consistency, and existing objective metrics cannot fully and accurately capture human perceptions of visual quality and semantic coherence. Human evaluation directly assesses a model’s performance in instruction following, adherence to physical laws, and content coherence, effectively compensating for the limitations of automated evaluation. Therefore, introducing human evaluation is a critical step in ensuring that generated outputs align with human preferences and possess practical application value. To this end, we recruited 10 volunteers to conduct subjective evaluations on the complete test sets of PAI-Bench, WorldModelBench, and DreamGen for five models: Kairos-4B, Cosmos-Predict2.5 (2B/14B), Wan 2.2-5B, and Lingbot-28B. To ensure fairness, all models were anonymized with random identifiers, and volunteers ranked the outputs without knowing the model identities. The final subjective evaluation results were obtained by averaging the rankings across different volunteers.

Table 8 Evaluation on PAIbench Robot Set. For each model group, the highest score in each column is bolded. Models marked with \* indicate results reproduced by our team.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Param</td><td colspan="7">Quality Score</td><td rowspan="2">Domain Score</td><td rowspan="2">Overall Score</td></tr><tr><td>i2v-bg i2v-s</td><td>aes</td><td>img</td><td>bg-con</td><td>mot</td><td>sub-cono-con</td><td></td></tr><tr><td colspan="10">Large-scale Models (≥10B)</td></tr><tr><td>Lingbot* [10]</td><td>28B</td><td>97.7</td><td>97.6</td><td>50.2 67.1</td><td></td><td>93.1</td><td>98.9</td><td>91.2</td><td>82.98</td><td>79.97</td></tr><tr><td>Cosmos3-Nano* [59]</td><td>16B</td><td>98.2</td><td>95.3</td><td>48.4 73.0</td><td>92.1</td><td>99.3</td><td>91.7</td><td>19.8 19.7</td><td>88.04</td><td>82.62</td></tr><tr><td>Abot-Physworld [60]</td><td>14B</td><td>97.8</td><td>95.0</td><td>46.2</td><td>69.1</td><td>93.7 99.2</td><td>94.1</td><td>19.3</td><td>87.85</td><td>82.32</td></tr><tr><td>Abot-Physworld + DPO [60]</td><td>14B</td><td>97.7</td><td>94.8</td><td>46.7</td><td>69.2</td><td>93.7</td><td>99.1 93.6</td><td>19.4</td><td>93.06</td><td>84.91</td></tr><tr><td>Cosmos-Predict2.5*[2]</td><td>14B</td><td>94.3</td><td>92.2</td><td>48.0 72.0</td><td>93.1</td><td>99.1</td><td>91.3</td><td>19.2</td><td>82.60</td><td>79.40</td></tr><tr><td>Wan2.5 [63]</td><td>/</td><td>94.4</td><td>94.3</td><td>54.8</td><td>64.6</td><td>89.9 96.2</td><td>87.8</td><td>21.9</td><td>86.44</td><td>80.96</td></tr><tr><td>Veo 3.1 [64]</td><td>/</td><td>93.2</td><td>96.1</td><td>54.6 72.4</td><td>92.2</td><td>97.1</td><td>91.5</td><td>22.1</td><td>83.50</td><td>80.45</td></tr><tr><td>Wan2.1 [26]</td><td>14B</td><td>97.5</td><td>94.5</td><td>47.2 71.2</td><td>93.2</td><td>99.2</td><td>91.9</td><td>19.2</td><td>83.91</td><td>80.32</td></tr><tr><td>WoW-wan [65]</td><td>14B</td><td>96.1</td><td>92.9</td><td>46.6</td><td>70.3</td><td>93.0 98.6</td><td>91.5</td><td>19.4</td><td>83.01</td><td>79.53</td></tr><tr><td>Sora v2 Pro [66]</td><td>/</td><td>95.1</td><td>92.9</td><td>53.2 69.6</td><td>92.9</td><td>97.0</td><td>91.6</td><td>22.0</td><td>76.26</td><td>76.52</td></tr><tr><td colspan="10">Small-scale Models (&lt;10B)</td></tr><tr><td>Wan2.2* [26]</td><td>5B</td><td>97.8</td><td>97.4</td><td>49.3</td><td>70.6</td><td>92.3</td><td>99.1</td><td>90.8 19.4</td><td>80.17</td><td>78.63</td></tr><tr><td>UnifoLM-WMA-0 [67]</td><td>3B</td><td>96.4</td><td>94.0</td><td>45.5</td><td>65.6 94.2</td><td>98.8</td><td>94.1</td><td>18.8</td><td>66.93</td><td>71.43</td></tr><tr><td>Cosmos-Predict2.5* [2]</td><td>2B</td><td>93.7</td><td>91.2</td><td>49.3</td><td>74.1</td><td>92.0</td><td>99.1 90.2</td><td>19.1</td><td>80.44</td><td>78.26</td></tr><tr><td>GigaWorld-0 [61]</td><td>2B</td><td>96.7</td><td>96.1</td><td>47.6</td><td>65.1</td><td>92.2</td><td>99.1 91.1</td><td>19.4</td><td>85.83</td><td>80.87</td></tr><tr><td>Kairos</td><td>4B</td><td>97.8</td><td>94.8</td><td>46.8</td><td>69.0</td><td>93.2</td><td>99.1 93.6</td><td>18.1</td><td>88.59</td><td>82.57</td></tr></table>

We conducted extensive human evaluations of Kairos-4B across three benchmarks: PAI-Bench, WorldModelBench, and DreamGen. Figure 19 presents the detailed results of the human evaluation. The results are as follows: On PAI-Bench robot subset, our model demonstrated exceptional physical understanding capabilities. Facing Cosmos-Predict2.5 14B and Lingbot-28B, which have significantly larger parameter counts, Kairos-4B still achieved win rates of 60.2% and 49.1% respectively, successfully achieving the "more with less" effect. Meanwhile, we attained a dominant 74.1% win rate against Wan 2.2-5B. On WorldModelBench robot subset, the advantage further widened. We achieved a remarkable 86.7% win rate against Wan 2.2-5B and 74.7% against Lingbot-28B. Even when compared to Cosmos-Predict2.5-14B, we demonstrated our leading position in world model consistency with a 65.0% win rate. Finally, on the DreamGen benchmark, Kairos-4B delivered the most outstanding performance. We achieved a strong 88.8% win record against Wan 2.2-5B, and maintained a 47.6% win rate against Cosmos-Predict2.5 14B. This fully validates the model’s strong competitiveness in generation quality and diversity.

Input frame

![](images/f3f9192166583a91439218d1ba71ea907c53ff8cbf5f205f30e19818f352b32c.jpg)  
Figure 18 Kairos samples on the Paibench robot dataset.

## 6.1.2 Ablation Studies

Effect of Human-Centric Data Scaling. To validate the effectiveness of human-centric data scaling on our embodied world model, we perform ablations on WorldModelBench-Robot, with results summarized in Table 9. Starting from a baseline trained without scaled-up human-centric data and adopting Qwen2.5-VL-7B-Instruct [68] as the VLM encoder, we inject large-scale human-centric data into pre-training. This substantially improves instruction following (2.10 → 2.33), leading to a clear gain in the overall benchmark score (9.08 → 9.25). This improvement reflects a stronger world-modeling capability: it follows instructions more accurately and generates dynamics that are better aligned with real-world interaction. We attribute this to the richer behavioral priors carried by human-centric data, which enhance the world model’s generalization across diverse behaviors, resulting in more correct instruction following.

Effect of Understanding. Building on the large-scale human-centric pre-training data, we further upgrade the VLM encoder from Qwen2.5-VL-7B-Instruct to Qwen3.5-2B [15]. Notably, although Qwen3.5-2B has fewer parameters, it exhibits stronger multimodal understanding capability (as shown in Table 10). This upgrade yields further gains in both instruction following (2.33 → 2.36)

![](images/0e3520a9527569fdf03f51070c3c4cafbcefa829f4e4e7abbb43ef0776ac64b3.jpg)  
Figure 19 Human evaluation results.

and total score (9.25 → 9.30). A stronger VLM enables the world model to better interpret and align the language instruction and the visual context of the initial frame, leading to more accurate instruction grounding and higher-quality future predictions.

## 6.2 World Action Model Benchmarks

We finetuned our WAM model and evaluated it in three main-stream benchmarks: LIBERO, LIBERO-Plus, and RoboTwin 2.0. Results show that our WAM model achieved highly competitive performance with only minimal downstream finetuning.

## 6.2.1 Benchmark Results

RoboTwin-2.0. RoboTwin 2.0 [75] is a challenging benchmark for bimanual robotic manipulation, comprising over 50 tasks that require precise coordination between two robotic arms. To comprehensively evaluate our method, we compare against two representative paradigms for embodied control. The first category consists of Vision-Language-Action (VLA) models, including π0, X-VLA, π0.5, StarVLA, Abot-M0, LingBot-VLA and G0.5. These methods typically learn a direct mapping from visual observations and language instructions to robot actions. The second category comprises World-Action-Model (WAM) approaches, including JEPA-VLA, GigaWorld-Policy, Motus, LingBot-VA, Fast-WAM, Being-H0.7, AIM, SANTS, and MotuBrain. Unlike VLA-based methods, WAM approaches explicitly model both environment dynamics and action evolution, enabling joint prediction of future states and executable actions for long-horizon reasoning and planning. As shown in Tab. 11, Kairos achieves strong and consistent performance across the RoboTwin 2.0 benchmark, outperforming the majority of existing baselines and achieving the second-highest average success rate among all evaluated methods. These results demonstrate that jointly modeling world dynamics and action evolution can substantially improve planning and execution capabilities for complex bimanual manipulation tasks.

Table 9 Ablation study on human-centric data scaling and VLM selection
<table><tr><td>Human-Centric Scaling</td><td>Stronger VLM Encoder</td><td>Instruction Following ↑</td><td>Total Score ↑</td></tr><tr><td></td><td></td><td>2.10</td><td>9.08</td></tr><tr><td></td><td></td><td>2.33</td><td>9.25</td></tr><tr><td></td><td>√</td><td>2.36</td><td>9.30</td></tr></table>

Table 10 Vision-language capability comparison referred from [68] [69].
<table><tr><td>Benchmark</td><td>Qwen2.5-VL-7B</td><td>Qwen3.5-2B</td></tr><tr><td>MMMU [70]</td><td>58.6</td><td>64.2/ 64.2</td></tr><tr><td>MMMU-Pro [71]</td><td>38.3</td><td>50.3/47.7</td></tr><tr><td>MathVistamini [72]</td><td>68.2</td><td>76.7/73.9</td></tr><tr><td>RealWorldQA [73]</td><td>68.5</td><td>74.5 /71.2</td></tr><tr><td>MMStar [74]</td><td>63.9</td><td>71.7 / 68.0</td></tr></table>

Table 11 Results on RoboTwin 2.0 benchmark.
<table><tr><td>Model</td><td>Clean</td><td>Randomized</td><td>Average</td></tr><tr><td>#VLA</td><td></td><td></td><td></td></tr><tr><td>π0_[76]</td><td>65.9</td><td>58.4</td><td>62.2</td></tr><tr><td>X-VLA [77]</td><td>72.9</td><td>72.8</td><td>72.9</td></tr><tr><td>π0.5 [78]</td><td>82.7</td><td>76.8</td><td>79.8</td></tr><tr><td>starVLA [79]</td><td>88.2</td><td>88.3</td><td>88.3</td></tr><tr><td>ABot-M0 [80]</td><td>81.2</td><td>80.4</td><td>80.8</td></tr><tr><td>LingBot-VLA [81]</td><td>86.5</td><td>85.3</td><td>85.9</td></tr><tr><td>G0.5 [82]</td><td>93.7</td><td>92.8</td><td>93.2</td></tr><tr><td>#WAM</td><td></td><td></td><td></td></tr><tr><td>JEPA-VLA [83]</td><td>73.5</td><td></td><td></td></tr><tr><td>GigaWorld-Policy [84]</td><td>86.0</td><td>85.0</td><td>85.5</td></tr><tr><td>Motus [85]</td><td>88.7</td><td>87.0</td><td>87.8</td></tr><tr><td>LingBot-VA_[86]</td><td>92.9</td><td>91.6</td><td>92.2</td></tr><tr><td>Fast-WAM [17]</td><td>91.9</td><td>91.8</td><td>91.8</td></tr><tr><td>Being-H0.7 [87]</td><td>90.2</td><td>89.6</td><td>89.8</td></tr><tr><td>AIM [88]</td><td>94.0</td><td>92.1</td><td>93.1</td></tr><tr><td>SANTS [89]</td><td>94.6</td><td>94.2</td><td>94.4</td></tr><tr><td>MotuBrain [90]</td><td>95.8</td><td>96.1</td><td>96.0</td></tr><tr><td> Kairos</td><td>96.9</td><td>95.2</td><td>96.1</td></tr></table>

LIBERO-Plus. LIBERO-Plus [91] is an extended version of LIBERO. Compared with the original benchmark, LIBERO-Plus places substantially stronger emphasis on scene-level generalization, robustness to visual distribution shift, compositional manipulation reasoning, and long-horizon policy stability. As shown in Tab. 12, Our WAM achieves state-of-the-art performance after fine-tuning on LIBERO-Plus. The results indicate that the model generalizes effectively to perturbed evaluation settings and exhibits strong robustness across diverse environmental variations.

Table 12 Results on LIBERO-plus benchmark.
<table><tr><td>Method</td><td>Camera</td><td>Robot</td><td> Language</td><td>Light</td><td>Background</td><td>Noise</td><td>Layout</td><td>Average</td></tr><tr><td>#VLA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ACoT-VLA [92]</td><td>96.6</td><td>70.4</td><td>79.7</td><td>95.1</td><td>97.1</td><td>95.9</td><td>85.0</td><td>88.0</td></tr><tr><td>T0 [76]</td><td>61.0</td><td>40.8</td><td>63.5</td><td>89.3</td><td>84.1</td><td>80.1</td><td>76.4</td><td>69.4</td></tr><tr><td>T0.5 [78]</td><td>75.8</td><td>79.4</td><td>83.3</td><td>95.5</td><td>95.0</td><td>89.6</td><td>87.0</td><td>85.7</td></tr><tr><td>Being-H0.5 [93]</td><td>1</td><td>1</td><td></td><td>1</td><td></td><td>1</td><td>=</td><td>83.1</td></tr><tr><td>MINT-4B [94]</td><td>1</td><td>1</td><td>1</td><td>1</td><td>-</td><td>1</td><td>1</td><td>84.1</td></tr><tr><td>VLANeXt [95]</td><td>90.4</td><td>65.7</td><td>81.8</td><td>95.9</td><td>82.5</td><td>94.1</td><td>80.8</td><td>83.9</td></tr><tr><td>ProGAL-VLA [96]</td><td>93.2</td><td>71.5</td><td>93.6</td><td>86.8</td><td>92.3</td><td>74.8</td><td>86.7</td><td>85.5</td></tr><tr><td>RoVLA [97]</td><td>96.6</td><td>32.0</td><td>91.5</td><td>95.9</td><td>96.1</td><td>95.1</td><td>74.1</td><td>82.0</td></tr><tr><td>OpenVLA-OFT [98]</td><td>92.8</td><td>30.3</td><td>85.8</td><td>94.9</td><td>93.9</td><td>89.3</td><td>77.6</td><td>79.6</td></tr><tr><td>Gr00t-N1.6 [99]</td><td>92.6</td><td>33.5</td><td>80.1</td><td>93.6</td><td>95.4</td><td>93.6</td><td>75.0</td><td>79.4</td></tr><tr><td>ABot-M0 [100]</td><td>60.4</td><td>67.9</td><td>86.4</td><td>96.2</td><td>91.6</td><td>86.4</td><td>82.6</td><td>80.5</td></tr><tr><td>#WAM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Being-H0.7 [87]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>84.8</td></tr><tr><td> Kairos</td><td>95.5</td><td>72.6</td><td>86.8</td><td>97.7</td><td>95.8</td><td>96.8</td><td>81.5</td><td>89.0</td></tr><tr><td> Kairos-joint</td><td>95.9</td><td>74.6</td><td>95.3</td><td>97.1</td><td>97.1</td><td>95.4</td><td>83.8</td><td>90.8</td></tr></table>

## 6.2.2 Ablation Studies

Effect of Embodied Human-centric Pretraining. To investigate the impact of embodied human-centric pretraining, we compare Kairos with VideoDiT pretrained either with or without large-scale human-centric data, while keeping all other settings unchanged. As results shown in Table 13, incorporating human-centric data leads to a significant gain in LIBERO-Plus benchmark, which suggests the effectiveness of native human-centric pretraining. Kairos can effectively leverage transferable action-relevant knowledge learned from human-centric data to improve performance on unseen tasks. More broadly, these findings highlight the potential of large-scale human-centric data as a scalable supervision source complementary to robot trajectories, offering a promising path toward reducing real-robot data requirements and enabling more general-purpose world action models.

Effect of Joint Training of Generation and Prediction. Building upon the human-centric pretrained WAM, we further investigate the impact of Generation-Prediction joint training on action prediction. Specifically, an ablated WAM variant optimizing only the ActionDiT is constructed, with results shown in Table 14. Training only the ActionDiT leads to a consistent performance degradation across LIBERO-Plus benchmarks. We attribute this degradation to the loss of worldmodeling supervision provided by the generation objective. By jointly optimizing generation and prediction, VideoDiT learns control-relevant interaction dynamics and produces more informative visual representations, resulting in stronger conditioning signals and improved action prediction performance.

Effect of Joint Denoising of Generation and Prediction. Rather than omitting video generation during inference, we explore a variant of Kairos, denoted as Kairos-joint (see Table 12). In this configuration, future video tokens and action tokens are jointly denoised, allowing action prediction to actively attend to video generation at inference time. Experimental results demonstrate that this explicit future imagination further elevates performance from 89.0 to 90.8. This improvement highlights the distinct advantages of the joint attention mechanism in coupling

Table 13 Effect of embodied human-centric pretraining
<table><tr><td>Model</td><td>Avg.</td><td>Gain</td></tr><tr><td> w/o human-centric data</td><td>83.0</td><td>1</td></tr><tr><td>w/human-centric data</td><td>89.0</td><td>+6.0</td></tr></table>

Table 14 Effect of Joint Training of Generation and Prediction
<table><tr><td>Model</td><td>Avg.</td><td>Gain</td></tr><tr><td>Action Prediction Only</td><td>65.8</td><td>1</td></tr><tr><td>Video Generation &amp; Action Prediction</td><td>89.0</td><td>+23.2</td></tr></table>

generation and prediction.

## 6.3 General World Model Benchmarks

Beyond the strong embodied capabilities of our model, we further evaluate its world modeling and physical reasoning abilities on general video generation benchmarks. In addition to the previously introduced PAI-Bench and WorldModelBench, we also include VideoPhy [101], a benchmark designed to evaluate the physical validity and semantic consistency of generated videos, with a particular focus on realistic entity interactions and motion dynamics. To examine temporal reasoning and consistency, we further conduct evaluations under a long-term (15-second) generation setting, enabling assessment across different temporal scales. For the evaluation, we select Cosmos-predict2.5-2B/14B [2], and Wan2.2-5B [26] as representative baseline models.

PAI-Bench. Following the same evaluation protocol as in the embodied setting, we extend the evaluation to the full set of PAI-Bench domains, including av, physics, and other real-world scenarios. In the TI2V setting, we compare our method with aforementioned open-source models. Quantitative results are summarized in Table 15. The results show that our model achieves the best performance across multiple sub-domains. In particular, the highest scores on background consistency and i2v-background indicate that Kairos maintains strong stability in background scenes during video generation.Moreover, the consistently strong performance across multiple domains, including robotics, further demonstrates the robustness of Kairos in diverse real-world scenarios.

Table 15 Evaluation on PAI-bench. For each column, the highest score is bolded.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Param</td><td colspan="8">Quality Score</td><td colspan="6">Domain Score</td><td rowspan="2">Overall Score</td></tr><tr><td>i2v-bg</td><td>i2v-s</td><td>aes</td><td>img</td><td>bg-con</td><td>mot</td><td>sub-con</td><td>o-con</td><td>av</td><td>CS</td><td>r0</td><td>in</td><td>hu</td><td>ph</td></tr><tr><td>Cosmos-Predict2.5</td><td>2B</td><td>97.4</td><td>96.6</td><td>52.4</td><td>70.8</td><td>94.2</td><td>99.1</td><td>92.5</td><td>20.1</td><td>66.1</td><td>94.1</td><td>80.8</td><td>87.8</td><td>81.4</td><td>93.9</td><td>81.0</td></tr><tr><td>Cosmos-Predict2.5</td><td>14B</td><td>97.9</td><td>97.2</td><td>52.5</td><td>70.0</td><td>94.8</td><td>99.1</td><td>93.4</td><td>20.1</td><td>67.8</td><td>94.2</td><td>79.9</td><td>87.7</td><td>80.0</td><td>93.5</td><td>81.0</td></tr><tr><td>Wan2.2</td><td>5B</td><td>96.7</td><td>95.9</td><td>51.9</td><td>69.9</td><td>93.7</td><td>98.8</td><td>91.8</td><td>20.3</td><td>65.2</td><td>93.1</td><td>79.3</td><td>88.4</td><td>83.0</td><td>91.5</td><td>80.4</td></tr><tr><td>Kairos</td><td>4B</td><td>97.9</td><td>96.5</td><td>51.9</td><td>68.8</td><td>94.5</td><td>98.7</td><td>92.0</td><td>21.3</td><td>64.4</td><td>94.3</td><td>84.0</td><td>84.5</td><td>84.1</td><td>92.8</td><td>80.8</td></tr></table>

While quantitative metrics provide an objective measure of model performance, qualitative evaluation is also important for a comprehensive assessment. Objective scores may not always fully capture the perceptual quality of generated videos, and qualitative analysis can help complement these metrics by providing a more direct examination of visual fidelity, prompt alignment, and temporal consistency.For qualitative evaluation, we select representative high-quality video samples generated by Kairos from each sub-domain of Paibench, as shown in Fig. 20, covering diverse scenarios such as autonomous driving, industrial manufacturing, indoor human activities, and robotic environments.The results show that Kairos is capable of generating realistic and high-quality videos across different domains. The generated videos demonstrate strong prompt adherence and accurate first-frame conditioning, while maintaining good physical consistency throughout the video sequence.

WorldModelBench. Following the same benchmark introduced in the embodied evaluation, we further evaluate Kairos on the full WorldModelBench dataset. In the embodied evaluation we focus on the TI2V setting, as embodied scenarios are more sensitive to the initial state of the robot or manipulator. Here we follow the official evaluation protocol and conduct experiments under both TI2V and T2V settings.

![](images/b583d5bfada2387becf439da54f294cafd489dec20e6fd7a87a7b8526561b23e.jpg)  
Figure 20 Kairos samples on the PAI-Bench dataset.

Tables 16 report detailed quantitative comparisons between Kairos and other baseline models on this benchmark. Compared with mainstream open-source models of similar scale, Kairos achieves higher scores under both settings, demonstrating competitive performance even compared with models at the 14B parameter scale despite its compact 4B size. In the TI2V setting, where first-frame conditioning introduces stricter constraints on video generation, Kairos consistently outperforms other models on multiple metrics. These results indicate stronger capability in maintaining object structural consistency, preventing physically implausible interactions, and producing higher-quality videos.

We further conduct qualitative analysis of Kairos on WorldModelBench, focusing on its instructionfollowing capability and physics adherence. As illustrated in Fig. 21, Kairos successfully accomplishes the specified tasks across a variety of complex scenarios. For example, in scenes such as the rotation of bottled water in a spiral filling machine and a vehicle being lifted by a hydraulic elevator, the model accurately generates dynamic processes that align with the given instructions, demonstrating strong instruction-following ability. Meanwhile, the generated videos also exhibit plausible physical behaviors. For instance, flower petals tremble in response to the motion of a butterfly, and the motion trajectories and accumulation patterns of falling garbage appear physically reasonable. Similar behaviors can also be observed in the T2V setting, as shown in the Fig. 22

Table 16 Evaluation on WorldModelBench. For each column, the highest score is bolded.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Mode</td><td rowspan="2">Param</td><td rowspan="2">Instruction</td><td colspan="5">Physics Adherence</td><td colspan="2">Common Sense</td><td rowspan="2">Total Score</td></tr><tr><td>Newton</td><td>Deform.</td><td>Fluid</td><td>Penetr.</td><td>Grav.</td><td>Frame</td><td>Temporal</td></tr><tr><td>Cosmos-Predict2.5</td><td>TI2V</td><td>2B</td><td>2.37</td><td>1.00</td><td>0.8</td><td>0.99</td><td>0.85</td><td>1.00</td><td>0.87</td><td>0.82</td><td>8.71</td></tr><tr><td>Cosmos-Predict2.5</td><td>TI2V</td><td>14B</td><td>2.45</td><td>1.00</td><td>0.84</td><td>0.99</td><td>0.88</td><td>1.00</td><td>0.93</td><td>0.87</td><td>8.95</td></tr><tr><td>Wan2.2</td><td>TI2V</td><td>5B</td><td>2.30</td><td>1.00</td><td>0.78</td><td>0.99</td><td>0.82</td><td>0.99</td><td>0.87</td><td>0.79</td><td>8.53</td></tr><tr><td>Kairos</td><td>TI2V</td><td>4B</td><td>2.36</td><td>1.00</td><td>0.85</td><td>0.99</td><td>0.89</td><td>0.99</td><td>0.92</td><td>0.89</td><td>8.89</td></tr><tr><td>Cosmos-Predict2.5</td><td>T2V</td><td>2B</td><td>2.30</td><td>1.00</td><td>0.91</td><td>0.99</td><td>0.90</td><td>1.00</td><td>1.00</td><td>0.95</td><td>9.01</td></tr><tr><td>Cosmos-Predict2.5</td><td>T2V</td><td>14B</td><td>2.30</td><td>1.00</td><td>0.91</td><td>0.99</td><td>0.92</td><td>1.00</td><td>1.00</td><td>0.97</td><td>9.09</td></tr><tr><td>Wan2.2</td><td>T2V</td><td>5B</td><td>2.18</td><td>1.00</td><td>0.87</td><td>1.00</td><td>0.88</td><td>0.99</td><td>0.99</td><td>0.95</td><td>8.87</td></tr><tr><td>Kairos</td><td>T2V</td><td>4B</td><td>2.33</td><td>1.00</td><td>0.83</td><td>1.00</td><td>0.90</td><td>1.00</td><td>0.99</td><td>0.93</td><td>8.99</td></tr></table>

input frame  
Predicted frames  
![](images/f3dcf830c02dd1d700458a541459a8eff0911c2415705c2c821fc48a8daebea5.jpg)  
Figure 21 Kairos samples(TI2V) on the WorldModelBench dataset.

Generated frames  
![](images/70d38a3fbb75478fc638d2c6c2207f1c87dd518012ec917abfcc4a836042c854.jpg)  
Figure 22 Kairos samples(T2V) on the WorldModelBench dataset.

![](images/6af4a2d5bc84f260afd28a4d526b88acb5021699a4c118a752f0ee75e0bff008.jpg)  
Figure 23 Kairos samples on the VideoPhy dataset.

VideoPHY. To evaluate the physical reasoning capability of our model, we benchmark it on VideoPhy, which contains 688 human-verified prompts. These prompts describe interactions between entities with different physical properties, including solid–solid, solid–fluid and fluid–fluid. Following the benchmark’s evaluation protocol, we generate videos using 344 test prompts. Each generated video is evaluated using two metrics: Semantic Adherence (SA) and Physical Commonsense (PC). SA measures whether the generated video correctly reflects the entities and actions described in the carefully designed prompts that simulate diverse physical interactions, while PC evaluates whether the generated scene is consistent with real-world physical laws. Both metrics are defined as binary judgments (0/1) and are computed using an auto evaluator VIDEOCON-PHYSICS provided by the benchmark. We report the Average Score (SA=1, PC=1) as the final results. Prompt enhancement can effectively enrich visual details in generated videos, thereby improving overall generation quality. Since the prompts in VideoPhy are relatively short, we apply model-specific prompt enhancement strategies to ensure a fair comparison across different models. Specifically, for Wan2.2-5B, we follow the official recommendation and employ Qwen/Qwen2.5-7B-Instruct for prompt enhancement. For Cosmos-predict2.5-2B/14B, since the prompt enhancement module has been removed in its NVIDIA official implementation, we adopt the same prompt enhancement strategy used for Kairos, namely leveraging Qwen3-8B. To ensure a fair comparison, we report the best performance for each model with or without prompt enhancement.

Table 17 Evaluation on VideoPhy. The highest score is bolded.
<table><tr><td>Model</td><td>Cosmos-Predict2.5-2B Cosmos-Predict2.5-14B Wan2.2-5B</td><td></td><td></td><td>Kairos</td></tr><tr><td>Average Score</td><td>44.64</td><td>45.16</td><td>38.85</td><td>45.55</td></tr></table>

As shown in Table 17, our model achieves the highest average score on VideoPhy with 45.55, outperforming Wan2.2-5B and Cosmos-Predict2.5-2B/14B. Remarkably, with only 4B parameters, Kairos outperforms Cosmos-Predict2.5-14B, demonstrating both high parameter efficiency and the ability to generate videos that adhere to real-world physical laws. Figure 23 presents qualitative results of Kairos on VideoPhy, covering a range of scenarios that involve physical interactions. For the prompt “A hand mixer stirs through thick cake batter”, the generated video shows a mixer that continuously rotates within the dense cake batter, creating clearly visible swirling and folding patterns that reflect the behavior of viscous fluids under mechanical agitation. For “Milk colliding with piping hot black coffee”, the poured milk gradually disperses into the coffee, forming a natural mixing process with smooth and coherent fluid dynamics. Our model also captures diverse physical interactions in other scenarios, producing realistic scenes such as rocks breaking under a pickaxe, rainwater flowing through a gutter, and a waterfall cascading over jagged rocks. These examples indicate that Kairos correctly reflects the entities and actions described in the prompts, demonstrating strong semantic adherence while maintaining physically consistent motion patterns and interactions.

## 6.4 Long-horizon Generation

To further evaluate long-horizon video generation, we conduct experiments on PAI-Bench using a 15-second generation setting. PAI-Bench offers a diverse set of real-world scenarios and measures models’ ability to capture realistic physical dynamics while maintaining physically plausible behaviors. The extended 15-second generation setting enables a more comprehensive examination.

Table 18 Evaluation on PAI-bench-15s. For each column, the highest score is bolded.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Param</td><td colspan="8">Quality Score</td><td colspan="6">Domain Score</td><td rowspan="2">Overall Score</td></tr><tr><td>i2v-bg</td><td>i2v-s</td><td>aes</td><td>img</td><td>bg-con</td><td>mot</td><td>sub-con</td><td>o-con</td><td>av</td><td>CS</td><td>ro</td><td>in</td><td>hu</td><td>ph</td></tr><tr><td>Cosmos-Predict2.5</td><td>2B</td><td>93.6</td><td>92.0</td><td>55.0</td><td>71.9</td><td>91.0</td><td>99.3</td><td>87.6</td><td>20.6</td><td>59.7</td><td>82.6</td><td>72.8</td><td>84.9</td><td>77.3</td><td>90.6</td><td>77.2</td></tr><tr><td>Cosmos-Predict2.5</td><td>14B</td><td>93.7</td><td>91.8</td><td>52.9</td><td>68.4</td><td>92.4</td><td>99.2</td><td>89.0</td><td>20.4</td><td>61.9</td><td>82.6</td><td>68.1</td><td>81.2</td><td>75.4</td><td>89.2</td><td>76.2</td></tr><tr><td>Wan2.2</td><td>5B</td><td>97.9</td><td>97.1</td><td>51.9</td><td>67.6</td><td>91.5</td><td>99.2</td><td>90.7</td><td>20.7</td><td>54.3</td><td>90.9</td><td>70.1</td><td>84.1</td><td>79.6</td><td>91.4</td><td>77.8</td></tr><tr><td>Kairos</td><td>4B</td><td>97.1</td><td>95.6</td><td>51.9</td><td>68.8</td><td>93.4</td><td>98.5</td><td>89.8</td><td>21.5</td><td>66.7</td><td>89.2</td><td>80.4</td><td>86.8</td><td>83.2</td><td>90.0</td><td>79.9</td></tr></table>

Tables 15 and 18 present the quantitative results of PAI-Bench under 5 and 15 second settings. When the generation horizon is extended to 15 seconds, the baseline models exhibit noticeable degradations in both quality metrics and domain-specific metrics. For example, Cosmos-Predict2.5-2B/14B shows clear drops in image-to-video consistency metrics, including i2v-bg (97.4/97.9→93.6/93.7) and i2v-s (96.6/97.2→92.0/91.8), indicating the increasing difficulty in preserving subject and background consistency in longer sequences. In addition, several domain-level scores decrease substantially, such as autonomous driving (66.1/67.8→59.7/61.9), common sense (94.1/94.2→82.6/82.6) and robot (80.8/79.9→72.8/68.1). Wan2.2-5B exhibits similar trends, with noticeable decreases in autonomous driving (65.2→54.3) and robot (79.3→70.1).

Under the 15-second setting, our model achieves the best overall score of 79.9, outperforming Cosmos-Predict2.5-2B/14B (77.2/76.2) and Wan2.2-5B (77.8). In particular, Kairos maintains strong performance in several domain-level metrics, including autonomous driving (66.7), robot (80.4), industry (86.8) and human (83.2), while also preserving high image-to-video consistency with i2v-bg (97.1) and i2v-s (95.6). These results indicate that Kairos effectively preserves scene consistency and physical interactions in long-horizon video generation.

![](images/e1a0b2aea87501e754736e8bd6c6f893a025e7b9e8ac709361c3f479f125a402.jpg)  
Figure 24 Kairos samples on the PAI-Bench-15s dataset.

Figure 24 presents the qualitative results of our model in the long-horizon generation. The visual results show that Kairos maintains consistent object appearance and scene structure over long temporal durations while producing natural and coherent motion. In the example of a dog interacting with floating bubbles, the dog’s head pose and attention gradually change following the bubble’s trajectory with smooth transitions. In the forest scene, the morning mist and sunlight evolve smoothly over time. As the sun gets stronger, the mist fades and the trees in the background become more visible. These results indicate that Kairos produces temporally coherent and visually consistent long-horizon videos.

## 7 Related Work

## 7.1 Video Generation Models

Diffusion-based Video Generation. The success of Diffusion Models (DMs) [102] in image synthesis has catalyzed their extension to the video domain. Early pioneers like Video Diffusion Models (VDM) [103] first extended the standard 2D U-Net to a 3D structure [104] by replacing 2D convolutions with space-time factorized convolutions. To alleviate the heavy computational burden of 3D operators, many subsequent works [105–109] adopted a "spatial-then-temporal" paradigm, inserting 1D temporal attention layers after 2D spatial blocks to capture dynamic dependencies. A significant architectural shift occurred with the introduction of Diffusion Transformers (DiT) [110], which demonstrated superior scalability over U-Net. This has led to the emergence of highperformance open-source video models such as LTX-Video [111], which refines the VAE decoder for high-frequency detail reconstruction, and HunyuanVideo [112], which integrates Multimodal Large Language Models (MLLMs) as text encoders to enhance text-video alignment. Building upon these advancements, Wan [26] meticulously optimizes each critical module—from the autoencoder to the text-video alignment—and provides comprehensive ablation studies to facilitate future video generation research. Furthermore, recent advancements in Flow Matching [21] have further optimized the training efficiency and generation quality of these diffusion-based frameworks.

Autoregressive Video Generation. Another prominent paradigm treats video generation as a sequence modeling task, analogous to Large Language Models (LLMs). Early works like VideoGPT [113] combined VQ-VAE [114] with GPT-like architectures [115] to autoregressively model discrete latent tokens in a spatio-temporal grid. To extend the duration of generated content, TATS [116] introduced a time-agnostic VQGAN and a time-sensitive transformer to synthesize thousands of frames. While diffusion models have recently dominated the field, autoregressive frameworks remain highly competitive for long-term consistency and streaming generation. For instance, VideoPoet [117] utilizes a large-scale transformer to unify multiple video-related tasks within a single LLM-style framework. More recently, MAGI-1 [118] incorporates causal constraints and KV caching to achieve real-time, high-fidelity video synthesis, demonstrating the enduring potential of autoregressive modeling in simulating physical dynamics and causal sequences.

Large Video Foundation Models. The landscape of video generation has been revolutionized by the emergence of large-scale foundation models that act as general-purpose world simulators. A landmark moment was the introduction of Sora [119], which demonstrated that Scaling Laws [120] previously observed in LLMs also apply to video: increasing parameters and training data significantly enhances the model’s understanding of 3D geometry and world dynamics. Following this, several powerful industrial models have emerged to push the boundaries of cinematic synthesis. Kling [121] (now updated to the Kling O3 architecture in 2026) utilizes a 3D-VAE and a computationally efficient full-attention mechanism to support ultra-long, complex human motion synthesis. Similarly, Luma Dream Machine [122] and Runway Gen-3 Alpha [123] focus on high-fidelity motion and temporal smoothness through massive multi-modal pre-training. Furthermore, ByteDance’s Seedance series [124] has introduced a unified multimodal joint generation architecture. The latest Seedance 2.0 natively supports text, image, audio, and video inputs, leveraging a Mixture-of-Transformer-Experts (MoT) design to balance spatiotemporal consistency with high-fidelity cinematic aesthetics.

## 7.2 World Models

World models aim to learn compact representations of environments that enable agents to predict future states and simulate interactions. Early work such as World Models [125] demonstrated that agents could be trained entirely within a latent “dream” environment produced by a VAE and recurrent dynamics model. This paradigm was further extended by latent dynamics approaches including PlaNet [126] and the Dreamer series [127–129], which perform reinforcement learning directly within imagined trajectories generated by learned world models.

With the rapid progress of large-scale generative models, the concept of world models has expanded beyond reinforcement learning toward video-based simulation of real-world dynamics. Recent approaches leverage transformer and diffusion architectures to learn rich spatiotemporal representations from large video corpora. For example, UniSim [130] proposes a neural closed-loop simulator for autonomous driving that generates consistent sensor observations under different actions. Similarly, large-scale video generation models such as Sora [119] have demonstrated the capability to simulate complex physical interactions and long-term temporal consistency, suggesting that generative video models may serve as general-purpose world simulators.

More recent work explores interactive world modeling by conditioning generation on actions [131] or control signals. GAIA-1 [132] learns a generative world model for autonomous driving that predicts future video conditioned on vehicle controls. Genie [133] further introduces generative interactive environments capable of synthesizing playable worlds from video data. Recently, Cosmos [2] proposes a large-scale world foundation model designed for robotics and physical AI, enabling realistic simulation of environments conditioned on agent actions. These advances indicate a promising direction toward unified world simulators that combine perception, dynamics modeling, and controllable environment generation.

## 7.3 World Action Models

From VLA to WAMs. Generalist embodied agents have long relied on Vision-Language-Action (VLA) models that learn reactive, direct observation-to-action mappings without modeling how the physical world evolves under intervention. To overcome the short-sightedness and weak physical grounding of such purely reactive policies, the World Action Model (WAM) paradigm unifies predictive environment dynamics with motor control by jointly targeting the distribution over future states and actions. By internalizing physical laws through future simulation, WAMs enable stronger long-horizon reasoning and spatial awareness than standard VLAs—though this unified objective introduces sharp architectural trade-offs between generation fidelity and inference latency.

Cascaded WAMs. Cascaded WAMs synthesize future visual representations conditioned on task goals before extracting control actions. Explicit methods first forecast raw 2D RGB frames using inverse dynamics or multimodal conditioning [134–136]. To mitigate spatial hallucinations inherent in pixel-level generation, subsequent explicit architectures extract intermediate 2D geometric representations like optical flow and point tracks [137–139], or extend into 3D and 4D spaces for rigorous spatial reasoning [140, 141]. Despite providing highly interpretable plans, explicit decoding suffers from severe computational latency, rendering high-frequency closed-loop control intractable [142]. To circumvent this bottleneck, implicit cascaded WAMs bypass high-dimensional image rendering by encoding anticipated futures strictly within continuous or discrete latent spaces. This latent planning involves learning quantized semantic codebooks [143], imposing geometric mask bottlenecks [144], or operating entirely within diffusion representations [145]. To further accelerate reactive control, frameworks extract intermediate network features [146] or self-distill multi-step generation into a single feed-forward pass [142]. Ultimately, while explicit WAMs excel in modularity with off-the-shelf simulators, implicit approaches prioritize the execution scalability and efficiency required for real-time deployment.

Joint World-Action Models. Departing from the two-stage pipeline, joint WAMs co-model decision-making and physical dynamics within a shared space, splitting into autoregressive and diffusion-based families. The autoregressive family casts states and actions as a unified sequence over a shared vocabulary: causal transformers first predict pixel-space trajectories as implicit planners [147, 148], then structure a visual chain-of-thought via intermediate subgoal images [149], and ultimately fuse text, images, and discrete actions into one vocabulary [150]. Since strict autoregressive action decoding causes error propagation and trajectory drift, remedies include action-attention masking [150] and a hybrid design with a parallel continuous action head to avoid quantization errors [151]; still, sequentially decoding high-dimensional visual states incurs heavy latency that hampers reactive deployment. Conversely, diffusion-based WAMs frame prediction as denoising or flow-matching, jointly optimizing states and actions to expose diverse marginal and conditional distributions for deeper causal understanding and stronger sample efficiency. Early designs fuse all modalities in a single diffusion transformer [152–154], while later works resolve action-image modality conflicts through separate or bridged token streams [155–157]. To avoid costly pixel-level reconstruction, several adopt implicit latent world modeling [158–161], and others decouple diffusion timesteps or scale via mixture-of-transformers [162–165]. For inference efficiency, recent frameworks adopt lightweight heads, asynchronous sampling, or speculative decoding [84, 166–169]. Across both families, the central challenge remains balancing high-fidelity joint simulation in training against lightweight, asymmetric modality decoupling at inference.

## 7.4 Efficient Attention Mechanisms

The quadratic complexity of the standard self-attention mechanism in Transformers poses a major challenge for modeling long sequences such as high-resolution videos. Given an input sequence of length N , vanilla attention requires $\mathcal { O } ( N ^ { 2 } )$ time and memory, which quickly becomes prohibitive as the spatial and temporal dimensions increase. To address this limitation, a large body of work has explored efficient attention mechanisms that reduce computational complexity while preserving the modeling capacity of Transformers.

Early approaches focused on approximating the softmax attention through kernelization or low-rank decomposition. Linear Transformers [170] reformulated softmax attention using kernel feature maps, enabling attention computation in linear time with respect to sequence length. Similarly, Performer [171] introduced FAVOR+ random feature approximations to achieve scalable attention with theoretical guarantees. Other works explored sparse attention patterns, restricting interactions to local or predefined structures in order to reduce computational cost [172–174].

More recent research has shifted toward designing sequence models with recurrent or state-space style updates that scale linearly with sequence length. Retentive Networks (RetNet) [175] replace the softmax attention with a retention mechanism that supports both parallel training and recurrent inference. The Mamba architecture [176–180] further demonstrates that selective state space models can achieve strong sequence modeling performance while maintaining linear complexity. Building on these developments, recent works further explore gated update mechanisms to improve the stability of long-context modeling. Delta-based architectures such as Gated Delta Networks [18] introduce gated update rules that mitigate key-value interference during sequence updates. In a similar spirit, Gated Linear Attention (GLA) incorporates gating mechanisms into linear attention to regulate information flow[181, 182]. By introducing learnable gates into the attention update, GLA selectively integrates new key-value information while suppressing outdated context, enabling stable long-range dependency modeling with O(N ) complexity [183–186]. Such efficiency is particularly beneficial for tasks involving extremely long token sequences.

## 8 Conclusion and Future Works

In this paper, we introduced Kairos, a native, deployment-aware world model stack engineered to pioneer a foundational and operational infrastructure for Physical AI. Rejecting the fractured paradigm of downstream policy fine-tuning, Kairos unifies world representation and robotic control into a cohesive, non-disjointed framework. This is achieved by anchoring a multi-layer Cross-Embodiment Data Curriculum (CEDC) for native knowledge injection, constructing a native understandinggeneration-prediction Mixture-of-Transformers (MoT) stack backed by Hybrid Linear Temporal Attention to guarantee long-horizon state maintenance, and implementing a deployment-aware system co-design that treats execution efficiency as a first-order modeling condition. Extensive evaluations demonstrate that Kairos delivers unprecedented execution fidelity and linear scalability, establishing new state-of-the-art milestones across rigorous trajectory-driven and embodied benchmarks, including LIBERO-plus and RoboTwin 2.0, while sustaining true real-time, closed-loop inference on consumer-grade edge hardware.

Moving forward, our research within the Kairos ecosystem will aggressively converge toward two transformative frontiers:

• Autonomous Self-Evolution via Recursive Imagination: We aim to break the constraints of static training by engineering a fully closed-loop self-evolution framework. By empowering Kairos to continuously interact with real-world environments, autonomously evaluate policy execution drift, and refine its internal spatial-temporal physics simulator through recursive multi-stage imagination, the model will transition into a self-improving cognitive agent capable of open-ended physical adaptation.

• The Generalist Embodied Substrate: We will scale Kairos beyond isolated, task-specific environments toward a truly generalist Physical AI platform. This frontier entails expanding the unified action space to accommodate highly diverse hardware cross-embodiments—spanning from complex humanoid platforms to dexterous multi-finger manipulation setups. By scaling heterogeneous pre-training across open-world environments, we seek to evolutionize Kairos into a universally deployable foundation capable of zero-shot complex intent recognition and high-success-rate execution across unconstrained physical domains.

Through these unified vectors, we aim to transform Kairos from a robust predictive baseline into an evolvable operational substrate that continuously learns, adapts, and masters the boundless complexities of the physical world.

## References

[1] Jiahua Dong, Qi Lyu, Baichen Liu, Xudong Wang, Wenqi Liang, Duzhen Zhang, Jiahang Tu, Hongliu Li, Hanbin Zhao, Henghui Ding, Yulun Zhang, Zhi Han, Nicu Sebe, Fahad Shahbaz Khan, Salman Khan, Mubarak Shah, Philip Torr, Ming-Hsuan Yang, and Dacheng Tao. Learning to model the world: A survey of world models in artificial intelligence. TechRxiv, 2026.

[2] NVIDIA, Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, Prithvijit Chattopadhyay, Mike Chen, Yongxin Chen, Yu Chen, Shuai Cheng, Yin Cui, Jenna Diamond, Yifan Ding, Jiaojiao Fan, Linxi Fan, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Ruiyuan Gao, Yunhao Ge, Jinwei Gu, Aryaman Gupta, Siddharth Gururani, Imad El Hanafi, Ali Hassani, Zekun Hao, Jacob Huffman, Joel Jang, Pooya Jannaty, Jan Kautz, Grace Lam, Xuan Li, Zhaoshuo Li, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Yen-Chen Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Yifan Lu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Seungjun Nah, Yashraj Narang, Abhijeet Panaskar, Lindsey Pavao, Trung Pham, Morteza Ramezanali, Fitsum Reda, Scott Reed, Xuanchi Ren, Haonan Shao, Yue Shen, Stella Shi, Shuran Song, Bartosz Stefaniak, Shangkun Sun, Shitao Tang, Sameena Tasmeen, Lyne Tchapmi, Wei-Cheng Tseng, Jibin Varghese, Andrew Z. Wang, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Jiashu Xu, Dinghao Yang, Xiaodong Yang, Haotian Ye, Seonghyeon Ye, Xiaohui Zeng, Jing Zhang, Qinsheng Zhang, Kaiwen Zheng, Andrew Zhu, and Yuke Zhu. World simulation with video foundation models for physical ai, 2025.

[3] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[4] Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mahmoud Assran, Koustuv Sinha, Michael Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-jepa 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482, 2026.

[5] Federico Baldassarre, Marc Szafraniec, Basile Terver, Vasil Khalidov, Francisco Massa, Yann LeCun, Patrick Labatut, Maximilian Seitzer, and Piotr Bojanowski. Back to the features: Dino as a foundation for video world models, 2025.

[6] World Labs. Marble: A multimodal world model. https://marble.worldlabs.ai/, 2025.

[7] Yabo Chen, Yuanzhi Liang, Jiepeng Wang, Tingxi Chen, Junfei Cheng, Zixiao Gu, Yuyang Huang, Zicheng Jiang, Wei Li, Tian Li, Weichen Li, Zuoxin Li, Guangce Liu, Jialun Liu, Junqi Liu, Haoyuan Wang, Qizhen Weng, Xuan’er Wu, Xunzhi Xiang, Xiaoyan Yang, Xin Zhang, Shiwen Zhang, Junyu Zhou, Chengcheng Zhou, Haibin Huang, Chi Zhang, and Xuelong Li. Teleworld: Towards dynamic multimodal synthesis with a 4d world model, 2025.

[8] Google DeepMind. Genie 3: A new frontier for world models. https://deepmind.google/blog/ genie-3-a-new-frontier-for-world-models/, 2025.

[9] Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025.

[10] Robbyant Team. Lingbot-world: An interactive world model for embodied intelligence. arXiv preprint, January 2026. Ant Group Robbyant Technology.

[11] Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training agents inside of scalable world models, 2025.

[12] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong

Wang, Hongxu Yin, Joseph E Gonzalez, et al. Worldmodelbench: Judging video generation models as world models. arXiv preprint arXiv:2502.20694, 2025.

[13] Jim Fan, Yoel Jang, Ireayo Akinola, et al. Dreamgen: Unlocking generalization in robot learning through neural trajectories. arXiv preprint arXiv:2505.12705, 2025. Introduces DreamGen Bench, a video generation benchmark for robot learning.

[14] Qwen Team. Qwen2.5-vl, January 2025.

[15] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.

[16] Ziyang Gong, Zehang Luo, Anke Tang, Zhe Liu, Shi Fu, Zhi Hou, Ganlin Yang, Weiyun Wang, Xiaofeng Wang, Jianbo Liu, Gen Luo, Haolan Kang, Shuang Luo, Yue Zhou, Yong Luo, Li Shen, Xiaosong Jia, Yao Mu, Xue Yang, Chunxiao Liu, Junchi Yan, Hengshuang Zhao, Dacheng Tao, and Xiaogang Wang. Ace-brain-0: Spatial intelligence as a shared scaffold for universal embodiments, 2026.

[17] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[18] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations.

[19] AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, Li Chen, Xi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, Yuxin Jiang, Cheng Jing, Hongyang Li, Jialun Li, Chiming Liu, Yi Liu, Yuxiang Lu, Jianlan Luo, Ping Luo, Yao Mu, Yue Niu, Yixuan Pan, Jiangmiao Pang, Yu Qiao, Guanghui Ren, Cheng-Xing Ruan, Jiaqi Shan, Yongjian Shen, Cheng Shi, Mi Shi, Modi Shi, Chonghao Sima, Jia-Yi Song, Huijie Wang, Wenhao Wang, Dafeng Wei, Chengen Xie, Guofeng Xu, Junchi Yan, Cunbiao Yang, Lei Yang, Shukai Yang, Maoqing Yao, Jiansheng Zeng, Chi Zhang, Qingli Zhang, Bin Zhao, Chengyu Zhao, Jiaqi Zhao, and Jianchao Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3549–3556, 2025.

[20] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems (RSS), 2024.

[21] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[22] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations (ICLR), 2023.

[23] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

[24] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[25] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in \$200k. arXiv preprint arXiv:2503.09642, 2025.

[26] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.

[28] Anke Tang, Li Shen, Yong Luo, Enneng Yang, Han Hu, Lefei Zhang, Bo Du, and Dacheng Tao. Fusionbench: A comprehensive benchmark of deep model fusion. Journal of Machine Learning Research, 2025.

[29] Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications, and opportunities. ACM Comput. Surv., 58(8), February 2026.

[30] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time, 2022.

[31] Jiho Choi, Donggyun Kim, Chanhyuk Lee, and Seunghoon Hong. Revisiting weight averaging for model merging. arXiv preprint arXiv:2412.12153, 2024.

[32] Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models, 2023.

[33] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch, 2024.

[34] Runxi Cheng, Feng Xiong, Yongxian Wei, Wanyun Zhu, and Chun Yuan. Whoever started the interference should end it: Guiding data-free model merging via task vectors, 2025.

[35] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization, 2023.

[36] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024.

[37] Runtao Liu, Haoyu Wu, Ziqiang Zheng, Chen Wei, Yingqing He, Renjie Pi, and Qifeng Chen. Videodpo: Omni-preference alignment for video diffusion generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8009–8019, 2025.

[38] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models, 2023.

[39] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context, 2023.

[40] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, Fei Yang, Pengfei Wan, and Di Zhang. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[41] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, and Siyu Zhu. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9365–9374, 2025.

[42] Zirui Tan, Yandong Li, Yaliang Li, and Jingren Zhou. Vidgen-1m: A large-scale dataset for text-to-video generation. arXiv preprint arXiv:2408.02629, 2024.

[43] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European Conference on Computer Vision (ECCV), pages 402–419. Springer, 2020.

[44] FalconsAI Team. Fine-tuned vision transformer for nsfw image classification. Hugging Face Model Hub, 2023. Initial commit 2023-10-14, Last updated 2025-04-06, Apache-2.0 License, 80k training images, 98.04% accuracy, 85.8M params.

[45] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021.

[46] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. arXiv preprint arXiv:2110.06864, 2021.

[47] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, and Xiang Bai. Real-time scene text detection with differentiable binarization. arXiv preprint arXiv:1911.08947, 2019.

[48] Qwen Team. Qwen3 technical report, 2025.

[49] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[50] Xiaomi LLM-Core Team. Mimo: Unlocking the reasoning potential of language model – from pretraining to posttraining, 2025.

[51] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025.

[52] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, apr 2026.

[53] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation, 2024.

[54] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis, 2024.

[55] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023.

[56] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models, 2025.

[57] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency, 2026.

[58] Fengzhe Zhou, Jiannan Huang, Jialuo Li, Deva Ramanan, and Humphrey Shi. Pai-bench: A comprehensive benchmark for physical ai. arXiv preprint arXiv:2512.01989, 2025.

[59] Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, Aarti Basant, Mukesh Beladiya, Mohammad Qazim Bhat, Zaid Pervaiz Bhat, Dan Blick, Vanni Brighella, Han Cai, Tiffany Cai, Eric Cameracci, Jiaxin Cao, Yulong Cao, Mark Carlson, Carlos Casanova, Ting-Yun Chang, Yan Chang, Yu-Wei Chao, Prithvijit Chattopadhyay, Roshan Chaudhari, Chieh-Yun Chen, Junyu Chen, Ke Chen, Qizhi Chen, Wenkai Chen, Xiaotong Chen, Yu Chen, An-Chieh Cheng, Click Cheng, Xiu Chia, Jeana Choi, Chaeyeon Chung, Wenyan Cong, Yin Cui, Magdalena Dadela, Nalin Dadhich, Wenliang Dai, Joyjit Daw, Alperen Degirmenci, Rodrigo Vieira Del Monte, Robert Denomme, Sameer Dharur, Marco Di Lucca, Ke Ding, Wenhao Ding, Yifan Ding, Yuzhu Dong, Nicole Drumheller, Yilun Du, Aigul Dzhumamuratova, Aleksandr Efitorov, Hamid Eghbalzadeh, Naomi Eigbe, Imad El Hanafi, Hassan Eslami, Benedikt Falk, Jiaojiao Fan, Jim Fan, Amol Fasale, Sergiy Fefilatyev, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Vikram Fugro, Prashant Gaikwad, TJ Galda, Katelyn Gao, Yihuai Gao, Wenhang Ge, Sreyan Ghosh, Arushi Goel, Vivek Goel, Akash Gokul, Rama Govindaraju, Jinwei Gu, Miguel Guerrero, Elfie Guo, Aryaman Gupta, Siddharth Gururani, Hugo Hadfield, Song Han, Ankur Handa, Zekun Hao, Mohammad Harrim, Ali Hassani, Nathan Hayes-Roth, Yufan He, Chris Helvig, Cyrus Hogg, Madison Huang, Michael Huang, Sophia Huang, Yufan Huang, Jacob Huffman, DeLesley Hutchins, Suneel Indupuru, Boris Ivanovic, Arihant Jain, Joel Jang, Ryan Ji, Yanan Jian, Dongfu Jiang, Jingyi Jin, Atharva Joshi, Nikhilesh Joshi, Pranjali Joshi, Jaehun Jung, Weiwei Kang, Scott Kassekert, Jan Kautz, Ashna Khetan, Julia Kiczka, Slawek Kierat, Gwanghyun Kim, Kuno Kim, Sunny Kim, Kezhi Kong, Xin Kong, Zhifeng Kong, Tomasz Kornuta, Egor Krivov, Hui Kuang, Saurav Kumar, Chia-Wen Kuo, George Kurian, Wojciech Kutak, JF Lafleche, Himangshu Lahkar, Omar Laymoun, Jayjun Lee, Sanggil Lee, Gabriele Leone, Boyi Li, Freya Li, Jiajun Li, Jinfeng Li, Ling Li, Pengcheng Li, Shangru Li, Tingle Li, Xiaolong Li, Xuan Li, Zhaoshuo Li, Zhiqi Li, Hao Liang, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Ming-Yu Liu, Sifei Liu, Zihan Liu, Hai Loc Lu, Xiangyu Lu, Alice Luo, Ruipu Luo, Wenjie Luo, Jiangran Lyu, Martin Ding Ma, Nic Ma, Qianli Ma, Dawid Majchrowski, Louis Marcoux, Miguel Martin, Qing Miao, Ashkan Mirzaei, Shreyas Misra, Kaichun Mo, Durra Mohsin, Hyejin Moon, Pawel Morkisz, Saeid Motiian, Kirill Motkov, Seungjun Nah, Yashraj Narang, Deepak Narayanan, Thabang Ngazimbi, Julian Ouyang, David Page, Yatian Pang, Sehwi Park, Mahesh Patekar, Mostofa Patwary, Marco Pavone, Trung Pham, Wei Ping, Soha Pouya, Shrimai Prabhumoye, Varun Praveen, Delin Qu, Hesam Rabeti, Morteza Ramezanali, Marilyn Reeb, Xuanchi Ren, Kristen Rumley, Wojciech Rymer, Jun Saito, Yeongho Seol, John Shao, Piyush Shekdar, Tianwei Shen, Humphrey Shi, Min Shi, Stella Shi, Kevin Shih, Mohammad Shoeybi, Mateusz Sieniawski, Shuran Song, Alexander Sotelo, Amir Sotoodeh, Sunil Srinivasa, Vignesh Srinivasakumar, Bartosz Stefaniak, Rahul Heinrich Steiger, Shangkun Sun, Jiaxiang Tang, Shitao Tang, Yangyang Tang, Yue Tang, Tolou Tavakkoli, Kayley Ting, Krzysztof Tomala, Wei-Cheng Tseng, Jibin Varghese, Sergei Vasilev, Thomas Volk, Raju Wagwani, Roger Waleffe, Andrew Z. Wang, Boxiang Wang, Haoxiang Wang, Qiao Wang, Shihao Wang, Shijie Wang, Ting-Chun Wang, Yan Wang, Yu Wang, David Wehr, Fangyin Wei, Xinshuo Weng, Jay Zhangjie Wu, Kedi Wu, Hongchi Xia, Summer Xiao, Tianjun Xiao, Kevin Xie, Daguang Xu, Jiashu Xu, Mengyao Xu, Ruqing Xu, Xingqian Xu, Yao Xu, Dinghao Yang, Dong Yang, Hans Yang, Xiaodong Yang, Xuning Yang, Yichu Yang, Yurong You, Zhiding Yu, Hao Yuan, Simon Yuen, Xiaohui Zeng, Pengcuo Zeren, Cindy Zha, Haotian Zhang, Jenny Zhang, Jing Zhang, Liangkai Zhang, Paris Zhang, Shun Zhang, Xuanmeng Zhang, Zhizheng Zhang, Ann Zhao, Yilin Zhao, Yuliya Zhautouskaya, Charles Zhou, Fengzhe Zhou, Shilin Zhu, Yuke Zhu, Dima Zhylko, and Artur Zolkowski. Cosmos 3: Omnimodal world models for physical ai, 2026.

[60] Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Dekang Qi, Haoyun Liu, Tong Lin, Shuang

Zeng, Junjin Xiao, Xinyuan Chang, et al. Abot-physworld: Interactive world foundation model for robotic manipulation with physics alignment. arXiv preprint arXiv:2603.23376, 2026.

[61] GigaWorld Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jiagang Zhu, Kerui Li, Mengyuan Xu, Qiuping Deng, Siting Wang, Wenkang Qin, Xinze Chen, Xiaofeng Wang, Yankai Wang, Yu Cao, Yifan Chang, Yuan Xu, Yun Ye, Yang Wang, Yukun Zhou, Zhengyuan Zhang, Zhehao Dong, and Zheng Zhu. Gigaworld-0: World models as data engine to empower embodied ai, 2025.

[62] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818. IEEE, 2024.

[63] Alibaba Tongyi Lab. Wan 2.5: Open-source ai video generation with audio. https://wan.video, 2025. Official product page.

[64] Google DeepMind. Veo 3.1. https://deepmind.google/models/veo/, 2025. Official model page.

[65] Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, and Jian Tang. Wow: Towards a world omniscient world model through embodied interaction, 2025.

[66] OpenAI. Sora 2 pro. https://platform.openai.com/docs/models/sora-2-pro, 2025. Official model documentation, accessed 2026-06-08.

[67] Unitree Robotics. Unifolm-wma-0: A world-model-action (wma) framework under the unifolm family. https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base, 2025. Hugging Face model card, accessed June 2026.

[68] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025.

[69] Qwen Team. Qwen3.5-2B. https://huggingface.co/Qwen/Qwen3.5-2B, 2025. Model card and benchmark results. Accessed: 2026-06-10.

[70] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2024.

[71] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark, 2025.

[72] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024.

[73] xAI. Grok-1.5 Vision Preview. https://x.ai/blog/grok-1.5v, 2024. Accessed: 2026-06-10.

[74] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models?, 2024.

[75] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[76] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A vision-language-action flow model for general robot control, 2026.

[77] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model, 2025.

[78] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization, 2025.

[79] StarVLA Community. Starvla: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

[80] Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, et al. Abot-m0: Vla foundation model for robotic manipulation with action manifold learning. arXiv preprint arXiv:2602.11236, 2026.

[81] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Shuailei Ma, He Sun, Yong Wang, Zhenqi Qiu, Houlong Xiong, Ziyu Wang, Shuai Zhou, Yiyu Ren, Kejia Zhang, Hui Yu, Jingmei Zhao, Qian Zhu, Ran Cheng, Yong-Lu Li, Yongtao Huang, Xing Zhu, Yujun Shen, and Kecheng Zheng. A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692v1, 2026.

[82] Galaxea Team. Galaxea g0.5 technical report. 2026.

[83] Shangchen Miao, Ningya Feng, Jialong Wu, Ye Lin, Xu He, Dong Li, and Mingsheng Long. Jepa-vla: Video predictive embedding is needed for vla models, 2026.

[84] Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. Gigaworld-policy: An efficient action-centered world–action model. arXiv preprint arXiv:2603.17240, 2026.

[85] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model, 2025.

[86] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[87] Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, and Zongqing Lu. Being-h0. 7: A latent world-action model from egocentric videos. arXiv preprint arXiv:2605.00078, 2026.

[88] Liaoyuan Fan, Zetian Xu, Chen Cao, Wenyao Zhang, Mingqi Yuan, and Jiayu Chen. Aim: Intent-aware unified world action modeling with spatial value maps, 2026.

[89] Yirui Sun, Guangyu Zhuge, Keliang Liu, Jie Gu, Xinyu Bing, Zhongxue Gan, and Chunxu Tian. Sants: A state-adaptive scheduler for world action models, 2026.

[90] MotuBrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, Louis Liu, Mengchen Cai, Rongxu Cui, Ruowen Zhao, Runqing Wang, Shuhe Huang, Yao Feng, Yinze Rong, Zeyuan Wang, and Jun Zhu. Motubrain: An advanced world action model for robot control, 2026.

[91] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[92] Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, and Guanghui Ren. Acot-vla: Action chain-of-thought for vision-language-action models, 2026.

[93] Hao Luo, Ye Wang, Wanpeng Zhang, Sipeng Zheng, Ziheng Xi, Chaoyi Xu, Haiweng Xu, Haoqi Yuan, Chi Zhang, Yiqing Wang, Yicheng Feng, and Zongqing Lu. Being-h0.5: Scaling human-centric robot learning for cross-embodiment generalization. arXiv preprint arXiv:2601.12993, 2026.

[94] Renming Huang, Chendong Zeng, Wenjing Tang, Jintian Cai, Cewu Lu, and Panpan Cai. Mimic intent, not just trajectories, 2026.

[95] Xiao-Ming Wu, Bin Fan, Kang Liao, Jian-Jian Jiang, Runze Yang, Yihang Luo, Zhonghua Wu, Wei-Shi Zheng, and Chen Change Loy. Vlanext: Recipes for building strong vla models, 2026.

[96] Nastaran Darabi and Amit Ranjan Trivedi. Progal-vla: Grounded alignment through prospective reasoning in vision-language-action models, 2026.

[97] Jingzhou Luo, Yifan Wen, Yongjie Bai, Xinshuai Song, Yang Liu, and Liang Lin. Rovla: Multiconsistency constraints for robust vision-language-action models, 2026.

[98] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success, 2025.

[99] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025.

[100] Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, Feng Xiong, Xing Wei, Zhiheng Ma, and Mu Xu. Abot-m0: Vla foundation model for robotic manipulation with action manifold learning, 2026.

[101] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024.

[102] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[103] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

[104] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. In The Fourteenth International Conference on Learning Representations, 2026.

[105] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

[106] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

[107] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7623–7633, 2023.

[108] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.

[109] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, 133(5):3059–3078, 2025.

[110] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[111] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

[112] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[113] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.

[114] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

[115] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.

[116] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and time-sensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022.

[117] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.

[118] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

[119] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Vickrey, Linas Fedarevičius, Huiwen Chang, Rui Zhang, Zheng Yan, Wei Wei, and Yang Song. Video generation models as world simulators. https://openai.com/index/ video-generation-models-as-world-simulators/, 2024. OpenAI Blog.

[120] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[121] Kuaishou Technology. Kling ai: A next-generation video generation model. https://klingai.com/, 2024.

[122] Luma AI. Dream machine: High-quality video generation from text and images. https://lumalabs. ai/dream-machine, 2024. Luma AI Official Website.

[123] Runway. Introducing gen-3 alpha: A new frontier for video generation. https://runwayml.com/blog/ introducing-gen-3-alpha/, 2024. Runway Blog.

[124] ByteDance Seed Team. Seedance (danceseed): Unified multimodal foundation model for video synthesis. https://seed.bytedance.com/seedance, 2025. ByteDance Technical Report.

[125] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3):440, 2018.

[126] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Learning latent dynamics for planning from pixels. In ICML, 2019.

[127] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.

[128] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020.

[129] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

[130] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, and Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In CVPR, 2023.

[131] Haoyu Wu, Jiwen Yu, Yingtian Zou, and Xihui Liu. Multiworld: Scalable multi-agent multi-view video world models. arXiv preprint arXiv:2604.18564, 2026.

[132] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

[133] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

[134] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023.

[135] Yilun Du, Sherry Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, Leslie Kaelbling, et al. Video language planning. In International Conference on Learning Representations, volume 2024, pages 31138–31155, 2024.

[136] Physical Intelligence, Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, Vedant Choudhary, Foster Collins, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Maitrayee Dhaka, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachlan Groom, Haroun Habeeb, Hunter Hancock, Karol Hausman, Gashon Hussein, Victor Hwang, Brian Ichter, Connor Jacobsen, Szymon Jakubczak, Rowan Jen, Tim Jones, Gregg Kammerer, Ben Katz, Liyiming Ke, Mairbek Khadikov, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Brendon LeCount, Sergey Levine, Xinyu Li, Adrian Li-Bell, Vladislav Lialin, Zhonglin Liang, Wallace Lim, Yao Lu, Enyu Luo, Vishnu Mano, Nandan Marwaha, Aikys Mongush, Liam Murphy, Suraj Nair, Tyler Patterson, Karl Pertsch, Allen Z. Ren, Gavin Schelske, Charvi Sharma, Baifeng Shi, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Jiaming Tang, Jimmy Tanner, Shalom Tekeste, Marcel Torne, Kyle Vedder, Quan Vuong, Anna Walling, Haohuan Wang,

Jason Wang, XuDong Wang, Chris Whalen, Samuel Whitmore, Blake Williams, Charles Xu, Sukwon Yoo, Lili Yu, Wuming Zhang, Zhuoyang Zhang, and Ury Zhilinsky. π0.7: a steerable generalist robotic foundation model with emergent capabilities, 2026.

[137] Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, and Joshua B Tenenbaum. Learning to act from actionless videos through dense correspondences. In International Conference on Learning Representations, volume 2024, pages 40938–40958, 2024.

[138] Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In European Conference on Computer Vision, pages 306–324. Springer, 2024.

[139] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. arXiv preprint arXiv:2407.15208, 2024.

[140] Hongyan Zhi, Peihao Chen, Siyuan Zhou, Yubo Dong, Quanxi Wu, Lei Han, and Mingkui Tan. 3dflowaction: Learning cross-embodiment manipulation from 3d flow world model. arXiv preprint arXiv:2506.06199, 2025.

[141] Jiaxu Wang, Yicheng Jiang, Tianlun He, Jingkai Sun, Qiang Zhang, Junhao He, Jiahang Cao, Zesen Gan, Mingyuan Sun, Qiming Shao, et al. Mvista-4d: View-consistent 4d world model with test-time action inference for robotic manipulation. arXiv preprint arXiv:2602.09878, 2026.

[142] Haodong Yan, Zhide Zhong, Jiaguan Zhu, Junjie He, Weilin Yuan, Wenxuan Song, Xin Gong, Yingjie Cai, Guanyi Zhao, Xu Yan, et al. S-vam: Shortcut video-action model by self-distilling geometric and semantic foresight. arXiv preprint arXiv:2603.16195, 2026.

[143] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Se June Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. In International Conference on Learning Representations, volume 2025, pages 28213–28239, 2025.

[144] Yunfan Lou, Xiaowei Chi, Xiaojie Zhang, Zezhong Qian, Chengxuan Li, Rongyu Zhang, Yaoxu Lyu, Guoyu Song, Chuyao Fu, Haoxuan Xu, et al. Mask world model: Predicting what matters for robust robot policy learning. arXiv preprint arXiv:2604.19683, 2026.

[145] Shuaiyi Huang, Mara Levy, Zhenyu Jiang, Anima Anandkumar, Yuke Zhu, Linxi Fan, De-An Huang, and Abhinav Shrivastava. Ardup: Active region video diffusion for universal policies. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8465–8472. IEEE, 2024.

[146] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.

[147] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In International Conference on Learning Representations, volume 2024, pages 10641–10662, 2024.

[148] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

[149] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.

[150] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.

[151] Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Xin Li, Hao Luo, et al. Rynnvla-002: A unified vision-language-action and world model. arXiv preprint arXiv:2511.17502, 2025.

[152] Yanjiang Guo, Yucheng Hu, Jianke Zhang, Yen-Jen Wang, Xiaoyu Chen, Chaochao Lu, and Jianyu Chen. Prediction with action: Visual policy learning via joint denoising process. Advances in Neural Information Processing Systems, 37:112386–112410, 2024.

[153] Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators. Advances in neural information processing systems, 38:95597–95621, 2026.

[154] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

[155] John Won, Kyungmin Lee, Huiwon Jang, Dongyoung Kim, and Jinwoo Shin. Dual-stream diffusion for world-model augmented vision-language-action model. arXiv preprint arXiv:2510.27607, 2025.

[156] Liudi Yang, Yang Bai, George Eskandar, Fengyi Shen, Mohammad Altillawi, Dong Chen, Ziyuan Liu, and Abhinav Valada. Covar: Co-generation of video and action for robotic manipulation via multi-modal diffusion. arXiv preprint arXiv:2512.16023, 2025.

[157] Jiayi Chen, Wenxuan Song, Pengxiang Ding, Ziyang Zhou, Han Zhao, Feilong Tang, Donglin Wang, and Haoang Li. Unified diffusion vla: Vision-language-action model via joint discrete denoising diffusion process. arXiv preprint arXiv:2511.01718, 2025.

[158] Ruijie Zheng, Jing Wang, Scott Reed, Johan Bjorck, Yu Fang, Fengyuan Hu, Joel Jang, Kaushil Kundalia, Zongyu Lin, Loic Magne, et al. Flare: Robot learning with implicit world modeling. arXiv preprint arXiv:2505.15659, 2025.

[159] Han Zhao, Jingbo Wang, Wenxuan Song, Shuai Chen, Yang Liu, Yan Wang, Haoang Li, and Donglin Wang. Frappe: Infusing world modeling into generalist policies via multiple future representation alignment. arXiv preprint arXiv:2602.17259, 2026.

[160] Runze Li, Hongyin Zhang, Junxi Jin, Qixin Zeng, Zifeng Zhuang, Yiqi Tang, Shangke Lyu, and Donglin Wang. World-value-action model: Implicit planning for vision-language-action systems. arXiv preprint arXiv:2604.14732, 2026.

[161] Ge Yuan, Qiyuan Qiao, Jing Zhang, and Dong Xu. Adaworldpolicy: World-model-driven diffusion policy with online adaptive learning for robotic manipulation. arXiv preprint arXiv:2602.20057, 2026.

[162] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. arXiv preprint arXiv:2504.02792, 2025.

[163] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025.

[164] MotuBrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, et al. Motubrain: An advanced world action model for robot control. arXiv preprint arXiv:2604.27792, 2026.

[165] Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, et al. Lda-1b: Scaling latent dynamics action model via universal embodied data ingestion. arXiv preprint arXiv:2602.12215, 2026.

[166] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025.

[167] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[168] Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4d world action modeling from video priors with asynchronous denoising. arXiv preprint arXiv:2604.26694, 2026.

[169] Yueci Deng, Guiliang Liu, and Kui Jia. Dexworldmodel: Causal latent world modeling towards automated learning of embodied tasks. arXiv preprint arXiv:2604.16484, 2026.

[170] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.

[171] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

[172] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[173] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

[174] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.

[175] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

[176] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on language modeling, 2024.

[177] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

[178] Aakash Lahoti, Kevin Y Li, Berlin Chen, Caitlin Wang, Aviv Bick, J Zico Kolter, Tri Dao, and Albert Gu. Mamba-3: Improved sequence modeling using state space principles. arXiv preprint arXiv:2603.15569, 2026.

[179] Tao Huang, Xiaohuan Pei, Shan You, Fei Wang, Chen Qian, and Chang Xu. Localmamba: Visual state space model with windowed selective scan. In European conference on computer vision, pages 12–22. Springer, 2024.

[180] Xiaohuan Pei, Tao Huang, and Chang Xu. Efficientvmamba: Atrous selective scan for light weight visual mamba. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pages 6443–6451, 2025.

[181] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.

[182] Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, et al. Minimax-01: Scaling foundation models with lightning attention. arXiv preprint arXiv:2501.08313, 2025.

[183] Zhen Qin, Songlin Yang, and Yiran Zhong. Hierarchically gated recurrent neural network for sequence modeling. Advances in Neural Information Processing Systems, 36:33202–33221, 2023.

[184] Zhen Qin, Songlin Yang, Weixuan Sun, Xuyang Shen, Dong Li, Weigao Sun, and Yiran Zhong. Hgrn2: Gated linear rnns with state expansion. arXiv preprint arXiv:2404.07904, 2024.

[185] Tobias Katsch. Gateloop: Fully data-controlled linear recurrence for sequence modeling. arXiv preprint arXiv:2311.01927, 2023.

[186] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended long shortterm memory. Advances in Neural Information Processing Systems, 37:107547–107603, 2024.

## Appendix

## A Contributors

Kairos is contributed to by the following people.

Project Lead: Fei Wang, Shan You, Qiming Zhang

Core Contributor:

Tao Huang, Zuoyi Fu

## Contributor:

Zhisheng Zheng, Yunlong Xi, Feng Lv, Xiaoming Wu, Zeyu Liu, Cong Wan, Pu Li, Ruiqing Yang, Xiaoou Li, Wei Wang, Kangkang Zhu, Yuwei Zhang, Shi Fu, Zheng Zhang, Xiaoning Wu, Xuzeng Fan

## Acknowledgement:

We would like to thank Anke Tang, Changhui Du, Huiwen Xue, Jiakai Huang, Junxi Jia, Lichen Man, Menglin Geng, Ruixuan Zhang, Shuaiqi Cheng, Shuo Huang, Weijie Sun, Yu Li, Zhongbo Wu, Zihao Gao for their valuable support and contributions to this project, including data preparation, model evaluation, infrastructure support, architecture analysis, and helpful discussions.

## B Theoretical Analysis

The proposed world model is grounded in a unified understanding-generation-prediction substrate and a hybrid temporal backbone. To formally analyze its long-horizon consistency, this section investigates future targets requiring extended world-state information, such as object permanence, delayed physical effects, and multi-stage task variables. We address two central questions: when is a bounded recent window fundamentally insufficient, and under what conditions can a hybrid multi-scale memory recover near-Bayes-optimal prediction?

Our theoretical contribution is twofold. First, we establish an information-theoretic necessity result: whenever the Bayes-optimal predictor of a long-horizon target relies on history outside a finite recent window, any window-restricted predictor incurs a strictly positive, irreducible excess risk. This proves the fundamental necessity of a persistent latent state. Second, we establish the approximate sufficiency of a hybrid multi-scale temporal memory. We prove that if the Bayes predictor factorizes into a shared predictive state alongside short-range, mid-range, and contractive global-memory branches, the resulting predictor yields an explicit excess-risk bound controlled by branch-wise approximation errors and a geometrically discounted global-memory perturbation term.

This formal analysis perfectly mirrors the model’s architectural design. The necessity result explains the inevitable degradation of purely local temporal mechanisms in tasks involving delayed effects or multi-stage structures. Conversely, the sufficiency result mathematically validates the proposed hybrid design: short- and mid-range pathways efficiently capture localized and intermediate motion, while the persistent global memory propagates supra-window context with controlled drift. Together, these theorems provide a rigorous mathematical justification for the architectural logic underlying the model’s long-horizon capabilities.

## B.1 Problem Setup and Theoretical Scope

Standing probabilistic setup. Let $( \Omega , \mathcal { F } , \mathbb { P } )$ be a probability space. We model the world as a discrete-time partially observed controlled process

$$
\{ ( X _ { t } , O _ { t } , A _ { t } ) \} _ { t \ge 0 } ,
$$

where

$$
X _ { t } \in \mathcal { X } , \qquad O _ { t } \in \mathcal { O } , \qquad A _ { t } \in \mathcal { A } .
$$

Here, $X _ { t }$ denotes the latent world state, $O _ { t }$ denotes the observable input available to the model, and $A _ { t }$ denotes the control or action signal that can influence future evolution. Fix a time index $t \geq 0$ , a prediction horizon $\tau \geq 1$ , and a window length $w \geq 0$ . Let $Y _ { t } ^ { ( \tau ) } \in L ^ { 2 } ( \Omega , \mathcal { F } , \mathbb { P } )$ be any square-integrable future target, and write

$$
Y : = Y _ { t } ^ { ( \tau ) } .
$$

Throughout, $Y$ may represent a future latent-frame coordinate, an object-permanence indicator, a delayed physical-effect event, a task-progress variable, or any other long-horizon functional of the future world state. All scalar statements below extend coordinatewise to vector-valued targets. For any square-integrable scalar-, vector-, or matrix-valued random variable $Z ,$ define

$$
\| Z \| _ { L ^ { 2 } } : = \big ( \mathbb { E } [ \| Z \| ^ { 2 } ] \big ) ^ { 1 / 2 } ,
$$

where ∥·∥ denotes the absolute value, the Euclidean norm, or the Frobenius norm according to context.

Definition 1 (History and recent window). The complete history up to time t is defined as

$$
H _ { t } : = ( O _ { 0 } , \ldots , O _ { t } , \ A _ { 0 } , \ldots , A _ { t - 1 } ) ,\tag{30}
$$

generating the associated σ-field $\mathcal { H } _ { t } : = \sigma ( H _ { t } )$ . For a window length w $\geq 0$ , the recent w-step window is given by

$$
W _ { t } ^ { ( w ) } : = \bigl ( O _ { ( t - w + 1 ) \vee 0 } , \ldots , O _ { t } , A _ { ( t - w + 1 ) \vee 0 } , \ldots , A _ { t - 1 } \bigr ) ,\tag{31}
$$

with its corresponding $\sigma { - } f i e l d$ denoted by $\mathcal { W } _ { t } ^ { ( w ) } : = \sigma ( W _ { t } ^ { ( w ) } )$ . By construction, it naturally follows that

$$
\mathcal { W } _ { t } ^ { ( w ) } \subseteq \mathcal { H } _ { t } .\tag{32}
$$

Definition 2 (Predictors and optimal risks). For any sub-σ-field ${ \mathcal { G } } \subseteq { \mathcal { F } }$ , define

$$
L ^ { 2 } ( { \mathcal { G } } ) : = \{ Z \in L ^ { 2 } ( \Omega , { \mathcal { F } } , \mathbb { P } ) : Z \ i s \ { \mathcal { G } } { \mathrm { - } } m e a s u r a b l e \} .
$$

For any $Z \in L ^ { 2 } ( \Omega , { \mathcal { F } } , \mathbb { P } )$ , define the squared prediction risk

$$
\mathscr { R } _ { t } ( Z ) : = \mathbb { E } [ ( Y - Z ) ^ { 2 } ] .\tag{33}
$$

The optimal full-history risk is

$$
R _ { \mathrm { f u l l } } ^ { \star } : = \operatorname* { i n f } _ { Z \in L ^ { 2 } ( \mathcal { H } _ { t } ) } \mathcal { R } _ { t } ( Z ) ,\tag{34}
$$

and the optimal recent-window risk is

$$
R _ { w } ^ { \star } : = \operatorname* { i n f } _ { Z \in L ^ { 2 } ( \mathcal { W } _ { t } ^ { ( w ) } ) } \mathcal { R } _ { t } ( Z ) .\tag{35}
$$

Definition 3 (Persistent latent state). A recursively updated internal state $M _ { t } \in \mathbb { R } ^ { d }$ is called a persistent latent state if there exist measurable update maps $\Phi _ { t }$ such that

$$
M _ { t } = \Phi _ { t } ( M _ { t - 1 } , Z _ { t } ) , \qquad Z _ { t } : = ( O _ { t } , A _ { t - 1 } ) , \qquad t \geq 1 ,\tag{36}
$$

with $M _ { 0 }$ given. Thus the model compresses historical information into a state that is propagated through time, rather than recomputing prediction from scratch from a bounded local context at every step.

Definition 4 (Exact sufficient state). A recursively updated state $S _ { t } ^ { \star } \in \mathbb { R } ^ { d }$ is called an exact sufficient state for the target Y if:

1. $S _ { t } ^ { \star }$ is $\mathcal { H } _ { t }$ -measurable;

2. there exist measurable maps $T _ { t }$ such that

$$
S _ { t } ^ { \star } = T _ { t } ( S _ { t - 1 } ^ { \star } , Z _ { t } ) , \qquad Z _ { t } : = ( O _ { t } , A _ { t - 1 } ) ;\tag{37}
$$

3. there exists a measurable decoder $g _ { t } : \mathbb { R } ^ { d }  \mathbb { R }$ such that

$$
\operatorname { \mathbb { E } } [ Y \mid { \mathcal { H } } _ { t } ] = g _ { t } ( S _ { t } ^ { \star } ) \qquad a . s .\tag{38}
$$

In other words, $S _ { t } ^ { \star }$ retains exactly the historical information that is relevant for predicting $Y$

Remark 5 (Scope of the two results). The subsequent necessity theorem relies solely on the filtration generated by the process history. It imposes no architectural assumptions and does not require the existence of an exact sufficient state. Furthermore, the ensuing sufficiency theorem introduces an architecture-motivated factorization of this state into four components: a shared predictive representation, a short-range state, a mid-range state, and a global recurrent memory. This conceptualization serves as the theoretical counterpart to our unified understanding–generation– prediction substrate, alongside its local, dilated, and global temporal pathways.

Lemma 1 (Conditional expectation is the L2-optimal predictor). Let ${ \mathcal { G } } \subseteq { \mathcal { F } }$ be any sub-σ-field and let $Y \in L ^ { 2 } ( \Omega , \mathcal { F } , \mathbb { P } )$ . Define $\eta _ { \mathcal { G } } : = \mathbb { E } [ Y \mid { \mathcal { G } } ]$ . Then ηG is the unique minimizer of squared risk over $L ^ { 2 } ( { \mathcal { G } } )$ , i.e.,

$$
\operatorname* { i n f } _ { Z \in L ^ { 2 } ( { \mathcal { G } } ) } \mathbb { E } [ ( Y - Z ) ^ { 2 } ] = \mathbb { E } [ ( Y - \eta _ { { \mathcal { G } } } ) ^ { 2 } ] .\tag{39}
$$

Moreover, for every $Z \in L ^ { 2 } ( \mathcal { G } )$

$$
\begin{array} { r } { \mathbb { E } [ ( Y - Z ) ^ { 2 } ] = \mathbb { E } [ ( Y - \eta _ { \mathcal { G } } ) ^ { 2 } ] + \mathbb { E } [ ( \eta _ { \mathcal { G } } - Z ) ^ { 2 } ] . } \end{array}\tag{40}
$$

Proof of Lemma 1. Fix any $Z \in L ^ { 2 } ( \mathcal { G } )$ and write

$$
\eta : = \eta _ { \mathcal { G } } = \mathbb { E } [ Y \mid \mathcal { G } ] .
$$

Then

$$
Y - Z = ( Y - \eta ) + ( \eta - Z ) .
$$

Squaring both sides and taking expectations gives

$$
\operatorname { \mathbb { E } } [ ( Y - Z ) ^ { 2 } ] = \operatorname { \mathbb { E } } [ ( Y - \eta ) ^ { 2 } ] + \operatorname { \mathbb { E } } [ ( \eta - Z ) ^ { 2 } ] + 2 \mathbb { E } [ ( Y - \eta ) ( \eta - Z ) ] .\tag{41}
$$

Since $\eta - Z$ is G-measurable,

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( Y - \eta ) ( \eta - Z ) \big ] = \mathbb { E } \Big [ \mathbb { E } \big [ ( Y - \eta ) ( \eta - Z ) \mid \mathcal { G } \big ] \Big ] } \\ & { \qquad = \mathbb { E } \Big [ ( \eta - Z ) \mathbb { E } [ Y - \eta \mid \mathcal { G } ] \Big ] = 0 , } \end{array}\tag{42}
$$

because $\operatorname { \mathbb { E } } [ Y - \eta \mid { \mathcal { G } } ] = \operatorname { \mathbb { E } } [ Y \mid { \mathcal { G } } ] - \eta = 0$ . Substituting back into Eq. (41) proves Eq. (40), and Eq. (39) follows immediately. Uniqueness holds because equality requires $\mathbb { E } [ ( \eta - Z ) ^ { 2 } ] = 0 , { \mathrm { i . e . , ~ } } Z = \eta$ almost surely. □

## B.2 Necessity of Persistent Latent States

This subsection formalizes the obstruction faced by purely local temporal models. In long-horizon prediction, two historical trajectories might share identical recent observations and actions yet differ in earlier latent events that dictate the future. Such events include instances where an object remains present despite temporary occlusion, the prior triggering of a delayed physical effect, or the completion of a specific stage in a task involving multiple steps. Consequently, predictors relying solely on recent windows are not merely more difficult to train but are fundamentally insufficient from a statistical perspective.

Theorem 3 (Supra-window dependence implies the necessity of persistent state). Fix $t \geq 0 , \tau \geq 1$ and $w \geq 0$ , and define

$$
m _ { t } : = \mathbb { E } [ Y \mid \mathcal { H } _ { t } ] , \qquad m _ { t } ^ { ( w ) } : = \mathbb { E } [ Y \mid \mathcal { W } _ { t } ^ { ( w ) } ] .\tag{43}
$$

Then the following statements hold.

(i) The optimal full-history and recent-window risks are

$$
R _ { \mathrm { f u l l } } ^ { \star } = \mathbb { E } [ ( Y - m _ { t } ) ^ { 2 } ] = \mathbb { E } [ \operatorname { V a r } ( Y \mid \mathcal { H } _ { t } ) ] ,\tag{44}
$$

$$
R _ { w } ^ { \star } = \mathbb { E } [ ( Y - m _ { t } ^ { ( w ) } ) ^ { 2 } ] = \mathbb { E } [ \operatorname { V a r } ( Y \mid \mathcal { W } _ { t } ^ { ( w ) } ) ] .\tag{45}
$$

(ii) The excess risk incurred by restricting prediction to the recent window satisfies the exact identity

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } = \mathbb { E } [ ( m _ { t } - m _ { t } ^ { ( w ) } ) ^ { 2 } ] = \mathbb { E } \big [ \mathrm { V a r } ( m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ) \big ] .\tag{46}
$$

(iii) Consequently,

$$
R _ { w } ^ { \star } > R _ { \mathrm { f u l l } } ^ { \star } ~ \Longleftrightarrow ~ m _ { t } ~ i s ~ n o t ~ \mathcal { W } _ { t } ^ { ( w ) } - m e a s u r a b l e .\tag{47}
$$

Specifically, if the Bayes predictor cannot be fully recovered from the most recent w steps, every window-restricted predictor inherently incurs a strictly positive, irreducible excess risk. Consequently, any Bayes-optimal recursive architecture must retain supra-window information, and implementing such retention via recursive state propagation necessitates a persistent latent state.

Proof of Theorem 3. Part (i) follows directly from Lemma 1 with $\mathcal { G } = \mathcal { H } _ { t }$ and $\mathcal { G } = \mathcal { W } _ { t } ^ { ( w ) }$ , respectively. Indeed,

$$
R _ { \mathrm { f u l l } } ^ { \star } = \mathbb { E } [ ( Y - m _ { t } ) ^ { 2 } ] , \qquad R _ { w } ^ { \star } = \mathbb { E } [ ( Y - m _ { t } ^ { ( w ) } ) ^ { 2 } ] ,
$$

and these equalities are equivalent to Eq. (44) and Eq. (45) by the definition of conditional variance. For part (ii), start from the decomposition

$$
Y - m _ { t } ^ { ( w ) } = ( Y - m _ { t } ) + ( m _ { t } - m _ { t } ^ { ( w ) } ) .\tag{48}
$$

After squaring and taking expectations,

$$
\begin{array} { r } { \mathbb { E } [ ( Y - m _ { t } ^ { ( w ) } ) ^ { 2 } ] = \mathbb { E } [ ( Y - m _ { t } ) ^ { 2 } ] + \mathbb { E } [ ( m _ { t } - m _ { t } ^ { ( w ) } ) ^ { 2 } ] + 2 \mathbb { E } [ ( Y - m _ { t } ) ( m _ { t } - m _ { t } ^ { ( w ) } ) ] . } \end{array}\tag{49}
$$

Because both $m _ { t }$ and ${ m } _ { t } ^ { ( w ) }$ are $\mathcal { H } _ { t } .$ -measurable, so is $m _ { t } - m _ { t } ^ { ( w ) }$ . Hence

$$
\begin{array} { r l } & { \mathbb { E } [ ( Y - m _ { t } ) ( m _ { t } - m _ { t } ^ { ( w ) } ) ] = \mathbb { E } \Big [ \mathbb { E } [ ( Y - m _ { t } ) ( m _ { t } - m _ { t } ^ { ( w ) } ) \mid \mathcal { H } _ { t } ] \Big ] } \\ & { \qquad = \mathbb { E } \Big [ ( m _ { t } - m _ { t } ^ { ( w ) } ) \mathbb { E } [ Y - m _ { t } \mid \mathcal { H } _ { t } ] \Big ] = 0 . } \end{array}\tag{50}
$$

Therefore,

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } = \mathbb { E } [ ( m _ { t } - m _ { t } ^ { ( w ) } ) ^ { 2 } ] .\tag{51}
$$

To derive the second expression in Eq. (46), note that by the tower property,

$$
m _ { t } ^ { ( w ) } = \mathbb { E } [ m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ] .\tag{52}
$$

Thus

$$
m _ { t } - m _ { t } ^ { ( w ) } = m _ { t } - \mathbb { E } [ m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ] ,
$$

and applying the conditional-variance identity to the random variable $m _ { t }$ conditioned on $\mathcal { W } _ { t } ^ { ( w ) }$ gives

$$
{ \mathbb E } [ ( m _ { t } - m _ { t } ^ { ( w ) } ) ^ { 2 } ] = { \mathbb E } \big [ \operatorname { V a r } ( m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ) \big ] .\tag{53}
$$

Combining Eq. (51) and Eq. (53) proves part (ii).

For part (iii), Eq. (51) implies

$$
R _ { w } ^ { \star } = R _ { \mathrm { f u l l } } ^ { \star } \quad \Longleftrightarrow \quad \mathbb { E } [ ( m _ { t } - m _ { t } ^ { ( w ) } ) ^ { 2 } ] = 0 \quad \Longleftrightarrow \quad m _ { t } = m _ { t } ^ { ( w ) } \mathrm { a . s . }
$$

Using Eq. (52), this is equivalent to

$$
m _ { t } = \mathbb { E } [ m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ] \ \mathrm { a . s . }
$$

A square-integrable random variable equals its conditional expectation with respect to a σ-field if and only if it is measurable with respect to that σ-field. Hence

$$
R _ { w } ^ { \star } = R _ { \mathrm { { f u l l } } } ^ { \star } { \quad \Longleftrightarrow \quad } m _ { t } { \mathrm { i s } } \mathcal { W } _ { t } ^ { ( w ) } { \mathrm { - m e a s u r a b l e } } ,
$$

and Eq. (47) follows by negation.

Corollary 3 (Explicit lower bound under an atomic recent-window mismatch). Suppose there is a window value s for which the event $E : = \{ W _ { t } ^ { ( w ) } = s \}$ occurs with positive probability, $\mathbb { P } ( E ) > 0$ Furthermore, suppose there exist disjoint events $E _ { 1 } , E _ { 2 } \in \mathcal { H } _ { t }$ that partition E into $E _ { 1 } \cup E _ { 2 } = E$ along with distinct constants $\mu _ { 1 } , \mu _ { 2 } \in \mathbb { R }$ , such that $m _ { t } = \mu _ { 1 }$ almost surely on $E _ { 1 }$ and $m _ { t } = \mu _ { 2 }$ almost surely on $E _ { 2 }$ . If we define the conditional probability $\alpha : = \mathbb { P } ( E _ { 1 } \mid E ) \in ( 0 , 1 )$ , then the excess risk is bounded below by

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } \geq \mathbb { P } ( E ) \alpha ( 1 - \alpha ) ( \mu _ { 1 } - \mu _ { 2 } ) ^ { 2 } .\tag{54}
$$

Proof of Corollary 3. By Theorem $^ { 3 , }$

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } = \mathbb { E } \big [ \mathrm { V a r } ( m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ) \big ] .\tag{55}
$$

Since conditional variance is nonnegative almost surely, we may restrict the expectation to the event E and obtain

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } \geq \mathbb { E } \big [ \mathrm { V a r } ( m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ) \mathbb { 1 } _ { E } \big ] .\tag{56}
$$

Now $E = \{ W _ { t } ^ { ( w ) } = s \} \in \mathcal { W } _ { t } ^ { ( w ) }$ and, by assumption, it is an atom of the σ-field $\mathcal { W } _ { t } ^ { ( w ) }$ . Hence every $\mathcal { W } _ { t } ^ { ( w ) }$ -measurable random variable is almost surely constant on $E .$ In particular, there exists a constant $c \in \mathbb { R }$ such that

$$
\operatorname { V a r } ( m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ) = c \qquad \mathrm { a . s . ~ o n } \ E .
$$

Moreover, because $W _ { t } ^ { ( w ) } = s$ almost surely on $E ,$ conditioning on $\mathcal { W } _ { t } ^ { ( w ) }$ and restricting to $E$ is equivalent to conditioning on the event E itself. Thus

$$
c = { \mathrm { V a r } } ( m _ { t } \mid E ) ,
$$

and therefore

$$
\mathbb { E } \big [ \mathrm { V a r } ( m _ { t } \mid \mathcal { W } _ { t } ^ { ( w ) } ) \mathbb { 1 } _ { E } \big ] = \mathbb { P } ( E ) \mathrm { V a r } ( m _ { t } \mid E ) .\tag{57}
$$

Next, since $E _ { 1 }$ and $E _ { 2 }$ form a partition of $E .$ , and

$$
m _ { t } = \mu _ { 1 } \quad \mathrm { a . s . ~ o n ~ } E _ { 1 } , \qquad m _ { t } = \mu _ { 2 } \quad \mathrm { a . s . ~ o n ~ } E _ { 2 } ,
$$

it follows that under the conditional law given $E ,$ , the random variable $m _ { t }$ takes the value $\mu _ { 1 }$ with probability

$$
\mathbb { P } ( E _ { 1 } \mid E ) = \alpha ,
$$

and the value $\mu _ { 2 }$ with probability

$$
\mathbb { P } ( E _ { 2 } \mid E ) = 1 - \alpha .
$$

Hence

$$
\mathbb { E } [ m _ { t } \mid E ] = \alpha \mu _ { 1 } + ( 1 - \alpha ) \mu _ { 2 } ,
$$

and

$$
\mathbb { E } [ m _ { t } ^ { 2 } \mid E ] = \alpha \mu _ { 1 } ^ { 2 } + ( 1 - \alpha ) \mu _ { 2 } ^ { 2 } .
$$

Therefore

$$
\begin{array} { r l } & { \operatorname { V a r } ( m _ { t } \mid E ) = \operatorname { \mathbb { E } } [ m _ { t } ^ { 2 } \mid E ] - \left( \operatorname { \mathbb { E } } [ m _ { t } \mid E ] \right) ^ { 2 } } \\ & { \quad \quad = \alpha \mu _ { 1 } ^ { 2 } + ( 1 - \alpha ) \mu _ { 2 } ^ { 2 } - \left( \alpha \mu _ { 1 } + ( 1 - \alpha ) \mu _ { 2 } \right) ^ { 2 } } \\ & { \quad \quad = \alpha ( 1 - \alpha ) ( \mu _ { 1 } - \mu _ { 2 } ) ^ { 2 } . } \end{array}\tag{58}
$$

Combining Eq. (56), Eq. (57), and Eq. (58) yields

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } \geq \mathbb { P } ( E ) \alpha ( 1 - \alpha ) ( \mu _ { 1 } - \mu _ { 2 } ) ^ { 2 } ,
$$

which is exactly Eq. (54).

Remark 6 (Information-theoretic interpretation). Theorem 3 yields the exact identity

$$
R _ { w } ^ { \star } - R _ { \mathrm { f u l l } } ^ { \star } = \left\| \mathbb { E } [ Y \mid \mathcal { H } _ { t } ] - \mathbb { E } [ Y \mid \mathcal { W } _ { t } ^ { ( w ) } ] \right\| _ { L ^ { 2 } } ^ { 2 } .\tag{59}
$$

Thus, the necessity claim is quantitative rather than merely qualitative: whenever the full-history Bayes predictor depends on information outside the recent window, every bounded-window predictor incurs an exactly measurable irreducible excess risk. In long-horizon world modeling, this formalizes the intuition that purely local temporal context is insufficient whenever future consistency depends on persistent world state beyond the visible short-range past.

Remark 7 (Architectural interpretation). Theorem 3 is a statement about information rather than about a particular neural implementation. What it proves is that supra-window information must be preserved if one seeks Bayes-optimal long-horizon prediction. In a recursive predictor, the natural implementation of such preservation is a propagated latent memory in the sense of Definition 3. This is the theoretical reason why a long-horizon world model requires a dedicated persistent memory pathway rather than only local attention.

Remark 8 (Connection to long-horizon evaluation). To connect the theorem with practical scenarios, let Y denote a future target directly relevant to long-horizon world-model evaluation. This target could be a future latent-frame coordinate, an object-identity consistency variable, a collision outcome, or a task-progress indicator situated several seconds ahead. Consequently, the gap presented in Eq. (46) provides a formal expression for the identical difficulty probed by our long-horizon evaluations. Specifically, whenever the historical cause governing the future target remains unrecoverable from the recent local window alone, any purely local temporal model inevitably incurs a nonzero prediction gap regardless of its optimization quality.

## B.3 Approximate Sufficiency of a Hybrid Multi-Scale Temporal Memory

The necessity result explains why some persistent state is unavoidable, but it does not yet explain why the particular temporal factorization used in the architecture is adequate. We now show that a hybrid multi-scale temporal memory is approximately sufficient whenever the Bayes predictor admits the corresponding decomposition and the global recurrent memory evolves stably. The result formalizes the complementary roles of a shared predictive representation, a short-range local branch, a mid-range branch, and a global memory branch.

Definition 5 (Exact hybrid multi-scale predictive decomposition). We say that the Bayes predictor admits an exact hybrid multi-scale predictive decomposition $i f$ there exist square-integrable, $\mathcal { H } _ { t } .$ measurable random variables

$$
U _ { t } ^ { \star } \in \mathbb R ^ { d _ { U } } , \qquad C _ { t } ^ { \star } \in \mathbb R ^ { d _ { C } } , \qquad D _ { t } ^ { \star } \in \mathbb R ^ { d _ { D } } , \qquad G _ { t } ^ { \star } \in \mathbb R ^ { d _ { v } \times d _ { k } } ,
$$

together with a measurable decoder

$$
h _ { t } : \mathbb { R } ^ { d _ { U } } \times \mathbb { R } ^ { d _ { C } } \times \mathbb { R } ^ { d _ { D } } \times \mathbb { R } ^ { d _ { v } \times d _ { k } } \to \mathbb { R } ,
$$

such that

$$
\mu _ { t } ^ { \star } : = \mathbb { E } [ Y \mid \mathcal { H } _ { t } ] = h _ { t } ( U _ { t } ^ { \star } , C _ { t } ^ { \star } , D _ { t } ^ { \star } , G _ { t } ^ { \star } ) \qquad a . s .\tag{60}
$$

The four components are interpreted as follows:

$U _ { t } ^ { \star }$ : a shared physical predictive state;

$C _ { t } ^ { \star }$ : a short-range local state;

$D _ { t } ^ { \star }$ : a mid-range dilated state;

$G _ { t } ^ { \star }$ : a global recurrent causal memory.

Remark 9 (Architecture correspondence). Although stated abstractly, Definition 5 is specifically designed to parallel our concrete temporal factorization. Within the implemented architecture, $C _ { t } ^ { \star }$ and D⋆t map to the short- and mid-range temporal pathways instantiated $b y$ SWA and DSWA. Furthermore, $G _ { t } ^ { \star }$ represents the global gated memory pathway instantiated by GLA. Finally, the variable $U _ { t } ^ { \star }$ embodies the shared predictive substrate that is continuously reused across understanding, generation, and prediction tasks.

Definition 6 (Hybrid multi-scale predictor and gated global-memory update). A hybrid multi-scale predictor is of the form

$$
\hat { \mu } _ { t } = h _ { t } ( \hat { U } _ { t } , \hat { C } _ { t } , \hat { D } _ { t } , \hat { G } _ { t } ) ,\tag{61}
$$

where $\hat { U } _ { t } , \hat { C } _ { t } , \hat { D } _ { t } , \hat { G } _ { t }$ are square-integrable, Ht-measurable estimators of $U _ { t } ^ { \star } , C _ { t } ^ { \star } , D _ { t } ^ { \star } , G _ { t } ^ { \star }$ , respectively. The global-memory branch is modeled by a gated delta update. Given a decay gate $\alpha \in ( 0 , 1 )$ , a writing strength $\beta \in ( 0 , 1 )$ , a current value $v \in \mathbb { R } ^ { d _ { v } }$ , and a key $\boldsymbol { k } \in \mathbb { R } ^ { d _ { k } }$ , we define the update map for a state $S \in \mathbb { R } ^ { d _ { v } \times d _ { k } }$ as follows:

$$
F ( S ; \alpha , \beta , v , k ) : = \alpha S + \beta ( v - S k ) k ^ { \top } , \qquad S \in \mathbb { R } ^ { d _ { v } \times d _ { k } } .\tag{62}
$$

Then the exact and learned global memories satisfy

$$
G _ { t } ^ { \star } = F ( G _ { t - 1 } ^ { \star } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) ,\tag{63}
$$

$$
\hat { G } _ { t } = F ( \hat { G } _ { t - 1 } ; \hat { \alpha } _ { t } , \hat { \beta } _ { t } , \hat { v } _ { t } , \hat { k } _ { t } ) .\tag{64}
$$

Definition 7 (Non-global branch approximation quality). For the hybrid predictor in Definition $\delta ,$ define the non-global branch approximation errors at time t by

$$
\begin{array} { r } { \varepsilon _ { t } ^ { U } : = \| \hat { U } _ { t } - U _ { t } ^ { \star } \| _ { L ^ { 2 } } , \qquad \varepsilon _ { t } ^ { \mathrm { { S W A } } } : = \| \hat { C } _ { t } - C _ { t } ^ { \star } \| _ { L ^ { 2 } } , \qquad \varepsilon _ { t } ^ { \mathrm { { D S W A } } } : = \| \hat { D } _ { t } - D _ { t } ^ { \star } \| _ { L ^ { 2 } } . } \end{array}\tag{65}
$$

Here, $\varepsilon _ { t } ^ { U }$ measures the approximation quality of the shared predictive substrate, $\varepsilon _ { t } ^ { \mathrm { S W A } }$ measures the short-range branch error, and $\varepsilon _ { t } ^ { \mathrm { D S W A } }$ measures the mid-range branch error.

Definition 8 (One-step gate approximation errors). Define

$$
\begin{array} { r l r l } & { \varepsilon _ { t } ^ { \alpha } : = \| \hat { \alpha } _ { t } - \alpha _ { t } ^ { \star } \| _ { L ^ { 2 } } , } & & { \varepsilon _ { t } ^ { \beta } : = \left\| \hat { \beta } _ { t } - \beta _ { t } ^ { \star } \right\| _ { L ^ { 2 } } , } \end{array}\tag{66}
$$

$$
\begin{array} { r } { \varepsilon _ { t } ^ { v } : = \| \hat { v } _ { t } - v _ { t } ^ { \star } \| _ { L ^ { 2 } } , \qquad \varepsilon _ { t } ^ { k } : = \left\| \hat { k } _ { t } - k _ { t } ^ { \star } \right\| _ { L ^ { 2 } } . } \end{array}\tag{67}
$$

Also define the initial global-memory discrepancy

$$
e _ { 0 } : = \Big | \Big | \hat { G } _ { 0 } - G _ { 0 } ^ { \star } \Big | \Big | _ { L ^ { 2 } } .\tag{68}
$$

For convenience, define the Bayes risk

$$
\mathcal { R } _ { t } ^ { \star } : = \mathcal { R } _ { t } ( \mu _ { t } ^ { \star } ) = \mathbb { E } [ ( Y - \mu _ { t } ^ { \star } ) ^ { 2 } ] .\tag{69}
$$

Lemma 2 (Bayes excess identity). For any square-integrable, Ht-measurable predictor $\hat { \mu } _ { t }$

$$
\mathcal { R } _ { t } ( \hat { \mu } _ { t } ) - \mathcal { R } _ { t } ^ { \star } = \| \hat { \mu } _ { t } - \mu _ { t } ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } .\tag{70}
$$

Proof of Lemma 2. Recall that

$$
\mathcal { R } _ { t } ( { \hat { \mu } } _ { t } ) = \mathbb { E } \big [ ( Y - { \hat { \mu } } _ { t } ) ^ { 2 } \big ] , \qquad \mathcal { R } _ { t } ^ { \star } = \mathbb { E } \big [ ( Y - \mu _ { t } ^ { \star } ) ^ { 2 } \big ] ,
$$

where $\mu _ { t } ^ { \star } = \mathbb { E } [ Y \mid { \mathcal { H } } _ { t } ]$ is the Bayes predictor. Hence,

$$
\mathscr { R } _ { t } ( { \hat { \mu } } _ { t } ) - \mathscr { R } _ { t } ^ { \star } = \mathbb { E } \big [ ( Y - { \hat { \mu } } _ { t } ) ^ { 2 } - ( Y - \mu _ { t } ^ { \star } ) ^ { 2 } \big ] .\tag{71}
$$

Expanding the difference of squares gives

$$
\begin{array} { c } { { ( Y - \hat { \mu } _ { t } ) ^ { 2 } - ( Y - \mu _ { t } ^ { \star } ) ^ { 2 } = \left( ( Y - \mu _ { t } ^ { \star } ) + ( \mu _ { t } ^ { \star } - \hat { \mu } _ { t } ) \right) ^ { 2 } - ( Y - \mu _ { t } ^ { \star } ) ^ { 2 } } } \\ { { = ( \mu _ { t } ^ { \star } - \hat { \mu } _ { t } ) ^ { 2 } + 2 ( Y - \mu _ { t } ^ { \star } ) ( \mu _ { t } ^ { \star } - \hat { \mu } _ { t } ) . } } \end{array}\tag{72}
$$

Substituting Eq. (72) into Eq. (71), we obtain

$$
\mathcal { R } _ { t } ( { \hat { \mu } } _ { t } ) - \mathcal { R } _ { t } ^ { \star } = \mathbb { E } \big [ ( \mu _ { t } ^ { \star } - { \hat { \mu } } _ { t } ) ^ { 2 } \big ] + 2 \mathbb { E } \big [ ( Y - \mu _ { t } ^ { \star } ) ( \mu _ { t } ^ { \star } - { \hat { \mu } } _ { t } ) \big ] .\tag{73}
$$

It therefore remains to show that the cross term vanishes.

Since both $\hat { \mu } _ { t }$ and $\mu _ { t } ^ { \star }$ are $\mathcal { H } _ { t }$ -measurable, the difference $\mu _ { t } ^ { \star } - \hat { \mu } _ { t }$ is also $\mathcal { H } _ { t }$ -measurable and squareintegrable. By the tower property,

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( Y - \mu _ { t } ^ { \star } ) ( \mu _ { t } ^ { \star } - \hat { \mu } _ { t } ) \big ] = \mathbb { E } \big [ \mathbb { E } \big [ ( Y - \mu _ { t } ^ { \star } ) ( \mu _ { t } ^ { \star } - \hat { \mu } _ { t } ) \mid \mathcal { H } _ { t } \big ] \big ] } \\ & { \qquad = \mathbb { E } [ ( \mu _ { t } ^ { \star } - \hat { \mu } _ { t } ) \mathbb { E } [ Y - \mu _ { t } ^ { \star } \mid \mathcal { H } _ { t } ] ] . } \end{array}\tag{74}
$$

Because $\mu _ { t } ^ { \star } = \mathbb { E } [ Y \mid { \mathcal { H } } _ { t } ]$ , we have

$$
\operatorname { \mathbb { E } } [ Y - \mu _ { t } ^ { \star } \mid \mathcal { H } _ { t } ] = \operatorname { \mathbb { E } } [ Y \mid \mathcal { H } _ { t } ] - \mu _ { t } ^ { \star } = 0 .
$$

Thus the right-hand side of Eq. (74) is zero, and therefore

$$
\mathbb { E } \big [ ( Y - \mu _ { t } ^ { \star } ) ( \mu _ { t } ^ { \star } - \hat { \mu } _ { t } ) \big ] = 0 .
$$

Returning to Eq. (73), we conclude that

$$
\mathcal { R } _ { t } ( { \hat { \mu } } _ { t } ) - \mathcal { R } _ { t } ^ { \star } = \mathbb { E } \big [ ( \mu _ { t } ^ { \star } - { \hat { \mu } } _ { t } ) ^ { 2 } \big ] = \| { \hat { \mu } } _ { t } - \mu _ { t } ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } ,\tag{75}
$$

which proves Eq. (70).

Lemma 3 (Contraction of the exact gated delta update). Fix $( \alpha , \beta , v , k )$ and define F by $E q .$ . (62). Let $\rho : = \alpha + \beta \left\| k \right\| _ { 2 } ^ { 2 }$ . Suppose $\| \boldsymbol { k } \| _ { 2 } ^ { 2 } < 1 - \alpha$ . Therefore, given $\alpha , \beta \in ( 0 , 1 )$ , the update is strictly contractive with a factor $\rho < 1$ , satisfying for all $S , T \in \mathbb { R } ^ { d _ { v } \times d _ { k } }$ :

$$
\begin{array} { r } { \| F ( S ; \alpha , \beta , v , k ) - F ( T ; \alpha , \beta , v , k ) \| _ { F } \leq \rho \| S - T \| _ { F } . } \end{array}\tag{76}
$$

Proof of Lemma 3. We prove that the gated delta update is contractive with respect to its memorystate argument. Fix two memory states $S , T \in \mathbb { R } ^ { d _ { v } \times d _ { k } }$ and let

$$
\Delta : = S - T .
$$

Since the value term v appears identically in both updates, it cancels when taking the difference. Indeed, by the definition of F in Eq. (62),

$$
\begin{array} { r l } & { F ( S ; \alpha , \beta , v , k ) - F ( T ; \alpha , \beta , v , k ) = \alpha S + \beta ( v - S k ) k ^ { \top } - \alpha T - \beta ( v - T k ) k ^ { \top } } \\ & { \qquad = \alpha ( S - T ) + \beta \big [ ( v - S k ) - ( v - T k ) \big ] k ^ { \top } } \\ & { \qquad = \alpha \Delta - \beta \Delta k k ^ { \top } . } \end{array}\tag{77}
$$

Thus, the difference between the two updated memory states depends only on the previous discrepancy $\Delta$ and on the rank-one correction induced by the key k.

We now bound the Frobenius norm of the right-hand side. By the triangle inequality,

$$
\| F ( S ; \alpha , \beta , v , k ) - F ( T ; \alpha , \beta , v , k ) \| _ { F } \leq \alpha \| \Delta \| _ { F } + \beta \| \Delta k k ^ { \top } \| _ { F } .\tag{78}
$$

It remains to control the second term. Observe first that

$$
\Delta \boldsymbol { k } \boldsymbol { k } ^ { \top } = ( \Delta \boldsymbol { k } ) \boldsymbol { k } ^ { \top } .
$$

This is a rank-one matrix. For any vectors a and b, the Frobenius norm of the outer product factorizes as

$$
\| a b ^ { \top } \| _ { F } = \| a \| _ { 2 } \| b \| _ { 2 } .
$$

Applying this identity with $a = \Delta k$ and $b = k \mathrm { ~ y ~ }$ ields

$$
\| \Delta \boldsymbol { k } \boldsymbol { k } ^ { \top } \| _ { F } = \| ( \Delta \boldsymbol { k } ) \boldsymbol { k } ^ { \top } \| _ { F } = \| \Delta \boldsymbol { k } \| _ { 2 } \| \boldsymbol { k } \| _ { 2 } .\tag{79}
$$

Next, using the standard operator-norm inequality together with $\| \Delta \| _ { 2 } \le \| \Delta \| _ { F }$ , we obtain

$$
\begin{array} { r } { \| \Delta k \| _ { 2 } \leq \| \Delta \| _ { 2 } \| k \| _ { 2 } \leq \| \Delta \| _ { F } \| k \| _ { 2 } . } \end{array}\tag{80}
$$

Combining Eq. (79) and Eq. (80), we conclude that

$$
\| \Delta k k ^ { \top } \| _ { F } \leq \| \Delta \| _ { F } \| k \| _ { 2 } ^ { 2 } .\tag{81}
$$

Substituting Eq. (81) into Eq. (78) gives

$$
\begin{array} { r l } & { \| F ( S ; \alpha , \beta , v , k ) - F ( T ; \alpha , \beta , v , k ) \| _ { F } \leq \alpha \| \Delta \| _ { F } + \beta \| \Delta \| _ { F } \| k \| _ { 2 } ^ { 2 } } \\ & { \qquad = \left( \alpha + \beta \| k \| _ { 2 } ^ { 2 } \right) \| \Delta \| _ { F } . } \end{array}\tag{82}
$$

Then we obtain

$$
\| F ( S ; \alpha , \beta , v , k ) - F ( T ; \alpha , \beta , v , k ) \| _ { F } \leq \rho \| \Delta \| _ { F } = \rho \| S - T \| _ { F } .\tag{83}
$$

This proves Eq. (76).

Lemma 4 (One-step perturbation bound for the global-memory update). Fix $t \geq 1$ and define the one-step update discrepancies by

$$
\begin{array} { r } { \varepsilon _ { t } ^ { \alpha } : = \| \hat { \alpha } _ { t } - \alpha _ { t } ^ { \star } \| _ { L ^ { 2 } } , \qquad \varepsilon _ { t } ^ { \beta } : = \| \hat { \beta } _ { t } - \beta _ { t } ^ { \star } \| _ { L ^ { 2 } } , \qquad \varepsilon _ { t } ^ { v } : = \| \hat { v } _ { t } - v _ { t } ^ { \star } \| _ { L ^ { 2 } } , \qquad \varepsilon _ { t } ^ { k } : = \| \hat { k } _ { t } - k _ { t } ^ { \star } \| _ { L ^ { 2 } } . } \end{array}\tag{84}
$$

For notational convenience, let $B _ { G } , B _ { k } , B _ { v } > 0$ denote envelope constants satisfying $\| \hat { G } _ { t - 1 } \| _ { F } \leq B _ { G }$ $\| \hat { k } _ { t } \| _ { 2 } , \| k _ { t } ^ { \star } \| _ { 2 } \leq B _ { k }$ , and $\| \hat { v } _ { t } \| _ { 2 } , \| v _ { t } ^ { \star } \| _ { 2 } \leq B _ { v }$ almost surely. Define

$$
\xi _ { t } : = B _ { G } \varepsilon _ { t } ^ { \alpha } + B _ { k } ( B _ { v } + B _ { G } B _ { k } ) \varepsilon _ { t } ^ { \beta } + B _ { k } \varepsilon _ { t } ^ { v } + ( B _ { v } + 2 B _ { G } B _ { k } ) \varepsilon _ { t } ^ { k } .\tag{85}
$$

Then

$$
\begin{array} { r } { \left\| F ( \hat { G } _ { t - 1 } ; \hat { \alpha } _ { t } , \hat { \beta } _ { t } , \hat { v } _ { t } , \hat { k } _ { t } ) - F ( \hat { G } _ { t - 1 } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) \right\| _ { L ^ { 2 } } \leq \xi _ { t } . } \end{array}\tag{86}
$$

Proof of Lemma 4. For notational brevity, write

$$
\hat { G } : = \hat { G } _ { t - 1 } , \quad \hat { \alpha } : = \hat { \alpha } _ { t } , \quad \alpha ^ { * } : = \alpha _ { t } ^ { * } , \quad \hat { \beta } : = \hat { \beta } _ { t } , \quad \beta ^ { * } : = \beta _ { t } ^ { \star } , \quad \hat { \nu } : = \hat { v } _ { t } , \quad v ^ { \star } : = v _ { t } ^ { \star } , \quad \hat { k } : = \hat { k } _ { t } , \quad k ^ { \star } : = k _ { t } ^ { \star } .
$$

By the definition of $F .$

$$
\begin{array} { r l } & { F ( \hat { G } ; \hat { \alpha } , \hat { \beta } , \hat { v } , \hat { k } ) - F ( \hat { G } ; \alpha ^ { \star } , \beta ^ { \star } , v ^ { \star } , k ^ { \star } ) = ( \hat { \alpha } - \alpha ^ { \star } ) \hat { G } + ( \hat { \beta } - \beta ^ { \star } ) ( \hat { v } - \hat { G } \hat { k } ) \hat { k } ^ { \top } } \\ & { \qquad + \beta ^ { \star } \Big [ ( \hat { v } - \hat { G } \hat { k } ) \hat { k } ^ { \top } - ( v ^ { \star } - \hat { G } k ^ { \star } ) ( k ^ { \star } ) ^ { \top } \Big ] . } \end{array}\tag{87}
$$

Hence

$$
\begin{array} { r } { \left\| F ( \hat { G } ; \hat { \alpha } , \hat { \beta } , \hat { \upsilon } , \hat { k } ) - F ( \hat { G } ; \alpha ^ { \star } , \beta ^ { \star } , \upsilon ^ { \star } , k ^ { \star } ) \right\| _ { F } \leq T _ { 1 } + T _ { 2 } + T _ { 3 } , } \end{array}\tag{88}
$$

where

$$
T _ { 1 } : = \left| \hat { \alpha } - \alpha ^ { \star } \right| \left\| \hat { G } \right\| _ { F } , \qquad T _ { 2 } : = \left| \hat { \beta } - \beta ^ { \star } \right| \left\| ( \hat { v } - \hat { G } \hat { k } ) \hat { k } ^ { \top } \right\| _ { F } ,
$$

and

$$
T _ { 3 } : = \beta ^ { \star } \left\| ( \hat { v } - \hat { G } \hat { k } ) \hat { k } ^ { \top } - ( v ^ { \star } - \hat { G } k ^ { \star } ) ( k ^ { \star } ) ^ { \top } \right\| _ { F } .
$$

Then we obtain:

$$
T _ { 1 } \leq B _ { G } \left| \hat { \alpha } - \alpha ^ { \star } \right| .\tag{89}
$$

Using the fact that $\left\| a b ^ { \top } \right\| _ { F } = \left\| a \right\| _ { 2 } \left\| b \right\| _ { 2 }$ , we obtain

$$
\begin{array} { r } { T _ { 2 } = \Big | \hat { \beta } - \beta ^ { \star } \Big | \left\| \hat { v } - \hat { G } \hat { k } \right\| _ { 2 } \left\| \hat { k } \right\| _ { 2 } } \\ { \leq \Big | \hat { \beta } - \beta ^ { \star } \Big | \left( B _ { v } + B _ { G } B _ { k } \right) B _ { k } . } \end{array}\tag{90}
$$

For $T _ { 3 } .$ , since $\beta ^ { \star } \in ( 0 , 1 )$ , it suffices to bound the norm inside. First,

$$
\begin{array} { r } { ( \hat { v } - \hat { G } \hat { k } ) \hat { k } ^ { \top } - ( v ^ { \star } - \hat { G } k ^ { \star } ) ( k ^ { \star } ) ^ { \top } = ( \hat { v } - v ^ { \star } ) \hat { k } ^ { \top } + v ^ { \star } ( \hat { k } - k ^ { \star } ) ^ { \top } } \\ { - \hat { G } ( \hat { k } \hat { k } ^ { \top } - k ^ { \star } ( k ^ { \star } ) ^ { \top } ) . } \end{array}\tag{91}
$$

Therefore,

$$
\begin{array} { r l } & { T _ { 3 } \leq \left\| ( \hat { v } - v ^ { \star } ) \hat { k } ^ { \top } \right\| _ { F } + \left\| v ^ { \star } ( \hat { k } - k ^ { \star } ) ^ { \top } \right\| _ { F } + \left\| \hat { G } ( \hat { k } \hat { k } ^ { \top } - k ^ { \star } ( k ^ { \star } ) ^ { \top } ) \right\| _ { F } } \\ & { \quad \leq B _ { k } \left\| \hat { v } - v ^ { \star } \right\| _ { 2 } + B _ { v } \left\| \hat { k } - k ^ { \star } \right\| _ { 2 } + \left\| \hat { G } \right\| _ { F } \left\| \hat { k } \hat { k } ^ { \top } - k ^ { \star } ( k ^ { \star } ) ^ { \top } \right\| _ { F } . } \end{array}\tag{92}
$$

Now write

$$
\hat { k } \hat { k } ^ { \top } - k ^ { \star } ( k ^ { \star } ) ^ { \top } = \hat { k } ( \hat { k } - k ^ { \star } ) ^ { \top } + ( \hat { k } - k ^ { \star } ) ( k ^ { \star } ) ^ { \top } .
$$

Hence

$$
\left\| \hat { k } \hat { k } ^ { \top } - k ^ { \star } ( k ^ { \star } ) ^ { \top } \right\| _ { F } \leq 2 B _ { k } \left\| \hat { k } - k ^ { \star } \right\| _ { 2 } ,\tag{93}
$$

and therefore,

$$
T _ { 3 } \le B _ { k } \left\| \hat { v } - v ^ { \star } \right\| _ { 2 } + \left( B _ { v } + 2 B _ { G } B _ { k } \right) \left\| \hat { k } - k ^ { \star } \right\| _ { 2 } .\tag{94}
$$

Combining Eq. (89), Eq. (90), and Eq. (94), then taking the $L ^ { 2 }$ norm and applying Minkowski’s inequality, yields

$$
\begin{array} { r l } & { \left\| F ( \hat { G } _ { t - 1 } ; \hat { \alpha } _ { t } , \hat { \beta } _ { t } , \hat { v } _ { t } , \hat { k } _ { t } ) - F ( \hat { G } _ { t - 1 } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) \right\| _ { L ^ { 2 } } } \\ & { \qquad \leq B _ { G } \varepsilon _ { t } ^ { \alpha } + B _ { k } ( B _ { v } + B _ { G } B _ { k } ) \varepsilon _ { t } ^ { \beta } + B _ { k } \varepsilon _ { t } ^ { v } + \left( B _ { v } + 2 B _ { G } B _ { k } \right) \varepsilon _ { t } ^ { k } = \xi _ { t } , } \end{array}\tag{95}
$$

which proves Eq. (86).

Theorem 4 (Approximate sufficiency of a hybrid multi-scale temporal memory). Suppose that the Bayes predictor admits the decomposition in Definition 5, the decoder $h _ { t }$ is coordinate-wise Lipschitz with constants $L _ { U } , L _ { C } , L _ { D }$ , and $L _ { G }$ , and the conditions in Lemma 3 hold. Define

$$
\begin{array} { r } { e _ { t } : = \| \hat { G } _ { t } - G _ { t } ^ { \star } \| _ { L ^ { 2 } } , \qquad \varepsilon _ { t } : = \operatorname* { m a x } \{ \varepsilon _ { t } ^ { U } , \varepsilon _ { t } ^ { \mathrm { S W A } } , \varepsilon _ { t } ^ { \mathrm { D S W A } } \} , \qquad L : = L _ { U } + L _ { C } + L _ { D } . } \end{array}\tag{96}
$$

Then, for every $t \geq 0$ , the following hold:

(i) Global-memory error bound. The global-memory branch satisfies

$$
e _ { t } \leq \rho ^ { t } e _ { 0 } + \frac { 1 - \rho ^ { t } } { 1 - \rho } \operatorname* { s u p } _ { 1 \leq i \leq t } \xi _ { i } ,\tag{97}
$$

with $\xi _ { i }$ defined in Eq. (85), and $\rho < 1$ . In particular, define $\bar { \xi } : = \operatorname* { s u p } _ { i \geq 1 } \xi _ { i }$ , we have

$$
e _ { t } \leq \frac { \bar { \xi } } { 1 - \rho } \quad a s t \to \infty .\tag{98}
$$

(ii) Long-horizon excess-risk bound. Define $\varepsilon : = \operatorname* { l i m } \operatorname* { s u p } _ { t \to \infty } \varepsilon _ { t }$ . Then, the hybrid predictor asymptotically satisfies

$$
\mathcal { R } _ { t } ( \hat { \mu } _ { t } ) - \mathcal { R } _ { t } ^ { \star } \leq \left( L \varepsilon + \frac { L _ { G } \bar { \xi } } { 1 - \rho } \right) ^ { 2 } \quad a s t \to \infty .\tag{99}
$$

Proof of Theorem $\it 4 .$ We prove parts (i) and (ii) in sequence.

Part (i): global-memory error bound. Recall from Definition 6 that the exact and learned globalmemory states satisfy

$$
\begin{array} { r } { G _ { t } ^ { \star } = F ( G _ { t - 1 } ^ { \star } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) , \qquad \hat { G } _ { t } = F ( \hat { G } _ { t - 1 } ; \hat { \alpha } _ { t } , \hat { \beta } _ { t } , \hat { v } _ { t } , \hat { k } _ { t } ) . } \end{array}
$$

Therefore, by the definition of $e _ { t }$ ,

$$
\begin{array} { r l } & { e _ { t } = \Vert \hat { G } _ { t } - G _ { t } ^ { \star } \Vert _ { L ^ { 2 } } } \\ & { \quad = \Big \Vert F ( \hat { G } _ { t - 1 } ; \hat { \alpha } _ { t } , \hat { \beta } _ { t } , \hat { v } _ { t } , \hat { k } _ { t } ) - F ( G _ { t - 1 } ^ { \star } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) \Big \Vert _ { L ^ { 2 } } . } \end{array}\tag{100}
$$

To separate the perturbation of the current update from the propagation of the previous memory error, we add and subtract the intermediate term

$$
F ( \hat { G } _ { t - 1 } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } )
$$

inside the norm. By the triangle inequality,

$$
\begin{array} { r l } & { e _ { t } \leq \left\| F ( \hat { G } _ { t - 1 } ; \hat { \alpha } _ { t } , \hat { \beta } _ { t } , \hat { v } _ { t } , \hat { k } _ { t } ) - F ( \hat { G } _ { t - 1 } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) \right\| _ { L ^ { 2 } } } \\ & { \qquad + \left\| F ( \hat { G } _ { t - 1 } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) - F ( G _ { t - 1 } ^ { \star } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) \right\| _ { L ^ { 2 } } . } \end{array}\tag{101}
$$

We now bound the two terms on the right-hand side separately.

For the first term, Lemma 4 gives the one-step perturbation bound

$$
\begin{array} { r } { \left\| F ( \hat { G } _ { t - 1 } ; \hat { \alpha } _ { t } , \hat { \beta } _ { t } , \hat { v } _ { t } , \hat { k } _ { t } ) - F ( \hat { G } _ { t - 1 } ; \alpha _ { t } ^ { \star } , \beta _ { t } ^ { \star } , v _ { t } ^ { \star } , k _ { t } ^ { \star } ) \right\| _ { L ^ { 2 } } \leq \xi _ { t } . } \end{array}\tag{102}
$$

For the second term, applying the contraction property from Lemma 3 guarantees that

$$
\begin{array} { r } { \Big \| F ( \hat { G } _ { t - 1 } ; \boldsymbol { \alpha } _ { t } ^ { \star } , \boldsymbol { \beta } _ { t } ^ { \star } , \boldsymbol { v } _ { t } ^ { \star } , \boldsymbol { k } _ { t } ^ { \star } ) - F ( G _ { t - 1 } ^ { \star } ; \boldsymbol { \alpha } _ { t } ^ { \star } , \boldsymbol { \beta } _ { t } ^ { \star } , \boldsymbol { v } _ { t } ^ { \star } , \boldsymbol { k } _ { t } ^ { \star } ) \Big \| _ { L ^ { 2 } } \leq \rho _ { t } \| \hat { G } _ { t - 1 } - G _ { t - 1 } ^ { \star } \| _ { L ^ { 2 } } = \rho _ { t } e _ { t - 1 } , } \end{array}\tag{103}
$$

where the contraction factor satisfies $\rho _ { t } < 1$ for all $t \geq 1$ . We then suppose that $\rho : = \operatorname* { s u p } _ { t \geq 1 } \rho _ { t } < 1$ Substituting Eq. (102) and Eq. (103) into Eq. (101), we obtain the one-step recursion

$$
e _ { t } \leq \xi _ { t } + \rho e _ { t - 1 } .\tag{104}
$$

We next unroll this recursion. Repeated application of Eq. (104) yields

$$
\begin{array} { r l } & { e _ { t } \leq \xi _ { t } + \rho e _ { t - 1 } } \\ & { \quad \leq \xi _ { t } + \rho \xi _ { t - 1 } + \rho ^ { 2 } e _ { t - 2 } } \\ & { \quad \leq \cdots \leq \rho ^ { t } e _ { 0 } + \displaystyle \sum _ { i = 1 } ^ { t } \rho ^ { t - i } \xi _ { i } . } \end{array}\tag{105}
$$

Since $\xi _ { i } \leq \operatorname* { s u p } _ { 1 \leq j \leq t } \xi _ { j }$ for every $1 \leq i \leq t .$ , we further obtain

$$
\begin{array} { c } { { e _ { t } \leq \rho ^ { t } e _ { 0 } + \Big ( \displaystyle \operatorname* { s u p } _ { 1 \leq j \leq t } \xi _ { j } \Big ) \sum _ { i = 1 } ^ { t } \rho ^ { t - i } } } \\ { { = \rho ^ { t } e _ { 0 } + \Big ( \displaystyle \operatorname* { s u p } _ { 1 \leq j \leq t } \xi _ { j } \Big ) \sum _ { m = 0 } ^ { t - 1 } \rho ^ { m } } } \\ { { = \rho ^ { t } e _ { 0 } + \displaystyle \frac { 1 - \rho ^ { t } } { 1 - \rho } \operatorname* { s u p } _ { 1 \leq i < t } \xi _ { i } , } } \end{array}\tag{106}
$$

which proves Eq. (97). Now let

$$
\bar { \xi } : = \operatorname* { s u p } _ { i \geq 1 } \xi _ { i } .
$$

Then Eq. (106) implies

$$
e _ { t } \leq \rho ^ { t } e _ { 0 } + \frac { 1 - \rho ^ { t } } { 1 - \rho } \bar { \xi } .\tag{107}
$$

Since $\rho < 1$ , we have $\rho ^ { t } \to 0$ and $( 1 - \rho ^ { t } ) / ( 1 - \rho ) \to 1 / ( 1 - \rho )$ as $t \to \infty$ . Hence

$$
e _ { t } \leq \frac { \bar { \xi } } { 1 - \rho } \qquad \mathrm { a s ~ } t \to \infty ,\tag{108}
$$

which proves the asymptotic statement in part (i).

Part (ii): long-horizon excess-risk bound. By Definition 5, the Bayes predictor has the form

$$
\mu _ { t } ^ { \star } = h _ { t } ( U _ { t } ^ { \star } , C _ { t } ^ { \star } , D _ { t } ^ { \star } , G _ { t } ^ { \star } ) ,
$$

whereas the learned predictor is given by

$$
\hat { \mu } _ { t } = h _ { t } ( \hat { U } _ { t } , \hat { C } _ { t } , \hat { D } _ { t } , \hat { G } _ { t } ) .
$$

Since $h _ { t }$ is coordinate-wise Lipschitz, we have the pointwise estimate

$$
\begin{array} { r } { \lvert \hat { \mu } _ { t } - \mu _ { t } ^ { \star } \rvert \leq L _ { U } \lVert \hat { U } _ { t } - U _ { t } ^ { \star } \rVert + L _ { C } \lVert \hat { C } _ { t } - C _ { t } ^ { \star } \rVert + L _ { D } \lVert \hat { D } _ { t } - D _ { t } ^ { \star } \rVert + L _ { G } \lVert \hat { G } _ { t } - G _ { t } ^ { \star } \rVert _ { F } . } \end{array}\tag{109}
$$

Taking $L ^ { 2 }$ norms on both sides and applying Minkowski’s inequality yields

$$
\begin{array} { r l } & { \| \hat { \mu } _ { t } - \mu _ { t } ^ { \star } \| _ { L ^ { 2 } } \leq L _ { U } \| \hat { U } _ { t } - U _ { t } ^ { \star } \| _ { L ^ { 2 } } + L _ { C } \| \hat { C } _ { t } - C _ { t } ^ { \star } \| _ { L ^ { 2 } } + L _ { D } \| \hat { D } _ { t } - D _ { t } ^ { \star } \| _ { L ^ { 2 } } + L _ { G } \| \hat { G } _ { t } - G _ { t } ^ { \star } \| _ { L ^ { 2 } } } \\ & { \qquad = L _ { U } \varepsilon _ { t } ^ { U } + L _ { C } \varepsilon _ { t } ^ { \mathrm { S W A } } + L _ { D } \varepsilon _ { t } ^ { \mathrm { D S W A } } + L _ { G } e _ { t } . } \end{array}\tag{110}
$$

By the definitions

$$
\varepsilon _ { t } = \operatorname* { m a x } \bigl \{ \varepsilon _ { t } ^ { U } , \varepsilon _ { t } ^ { \mathrm { { S W A } } } , \varepsilon _ { t } ^ { \mathrm { { D S W A } } } \bigr \} , \qquad L : = L _ { U } + L _ { C } + L _ { D } ,
$$

the first three terms can be grouped as

$$
L _ { U } \varepsilon _ { t } ^ { U } + L _ { C } \varepsilon _ { t } ^ { \mathrm { { S W A } } } + L _ { D } \varepsilon _ { t } ^ { \mathrm { { D S W A } } } \leq \left( L _ { U } + L _ { C } + L _ { D } \right) \varepsilon _ { t } = L \varepsilon _ { t } .\tag{111}
$$

Substituting Eq. (111) into $\operatorname { E q } .$ (110), we obtain

$$
\| \hat { \mu } _ { t } - \mu _ { t } ^ { \star } \| _ { L ^ { 2 } } \leq L \varepsilon _ { t } + L _ { G } e _ { t } .\tag{112}
$$

Using the asymptotic estimate from part (i), we further obtain

$$
\| \hat { \mu } _ { t } - \mu _ { t } ^ { \star } \| _ { L ^ { 2 } } \leq L \varepsilon _ { t } + \frac { L _ { G } \bar { \xi } } { 1 - \rho } \qquad \mathrm { a s \ } t \to \infty .\tag{113}
$$

Finally, Lemma 2 gives

$$
\mathcal { R } _ { t } ( \hat { \mu } _ { t } ) - \mathcal { R } _ { t } ^ { \star } = \| \hat { \mu } _ { t } - \mu _ { t } ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } .
$$

Substituting Eq. (113) into this identity yields

$$
\mathcal { R } _ { t } ( { \hat { \mu } } _ { t } ) - \mathcal { R } _ { t } ^ { \star } \leq \left( L \varepsilon + \frac { L _ { G } \bar { \xi } } { 1 - \rho } \right) ^ { 2 } \qquad \mathrm { a s ~ } t \to \infty ,\tag{114}
$$

which proves Eq. (99).

Corollary 4 (Exact sufficiency in the realizable case). Suppose that the hypotheses of Theorem $\it 4$ hold. Assume further that the learned hybrid state exactly recovers the Bayes decomposition at every time step, in the sense that $\varepsilon _ { t } ^ { U } = \varepsilon _ { t } ^ { \mathrm { { S W A } } } = \varepsilon _ { t } ^ { \mathrm { { D S W A } } } = 0 , \varepsilon _ { t } ^ { \mathrm { { \alpha } } } = \varepsilon _ { t } ^ { \beta } = \varepsilon _ { t } ^ { v } = \varepsilon _ { t } ^ { k } = 0$ for all $t ,$ and that the initial global-memory state is exactly aligned, $e _ { 0 } = 0$ . Then, for every t,

$$
\hat { G } _ { t } = G _ { t } ^ { \star } \qquad a n d \qquad \hat { \mu } _ { t } = \mu _ { t } ^ { \star } \qquad a . s .\tag{115}
$$

Consequently,

$$
\mathcal { R } _ { t } ( \hat { \mu } _ { t } ) = \mathcal { R } _ { t } ^ { \star } .\tag{116}
$$

Proof of Corollary $\it 4 .$ Under conditions in Corollary 4, the quantity $\xi _ { t }$ in $\operatorname { E q }$ . (85) satisfies $\xi _ { t } = 0$ for all t. Since $e _ { 0 } = 0$ , the memory recursion Eq. (97) gives $e _ { t } = 0$ for all t. Substituting these equalities into Eq. (106) and Eq. (112) yields

$$
\| \hat { \mu } _ { t } - \mu _ { t } ^ { \star } \| _ { L ^ { 2 } } = 0 ,
$$

By Lemma $2 ,$ this is equivalent to

$$
\mathcal { R } _ { t } ( \hat { \mu } _ { t } ) - \mathcal { R } _ { t } ^ { \star } = 0 .
$$

which implies $\hat { \mu } _ { t } = \mu _ { t } ^ { \star }$ almost surely. This proves both Eq. (115) and Eq. (116).

Remark 10 (Interpretation of the bound). Theorem 4 shows that the long-horizon prediction error is controlled by two quantities. The first is the aggregated non-global approximation term $L \varepsilon ,$ which summarizes the quality of the shared, short-range, and mid-range branches. The second is the global-memory term, whose asymptotic contribution is bounded by $L _ { G } \bar { \xi } / ( 1 - \rho )$ , where $\bar { \xi }$ measures the worst-case one-step perturbation and $\rho < 1$ ensures geometric damping of memory drift. In this sense, a hybrid multi-scale temporal memory is approximately sufficient: once the Bayes predictor factorizes across the four architectural roles, the remaining long-horizon degradation is fully accounted for by branchwise approximation quality together with a contractively controlled accumulation of global-memory perturbations.

Remark 11 (Why the global-memory branch is nontrivial). Only the global branch can accumulate error across time. The contraction factor $\rho \ : < \ : 1$ turns this accumulation into a geometrically discounted sum, preventing uncontrolled long-horizon drift. Without such a property, even small one-step errors in the global memory could grow super-linearly and destroy long-horizon consistency.

Remark 12 (Architectural and experimental relevance). Theorem 4 establishes architectural sufficiency rather than providing a performance guarantee for any specific model checkpoint. It serves to elucidate why the multi-scale temporal factorization is a principled design for long-horizon world modeling. In the concrete architecture, the short- and mid-range approximation terms reflect the representation quality of the SWA and DSWA pathways, whereas the discounted memory term characterizes the stability of the GLA pathway. This theoretical mechanism strongly complements the empirical long-horizon results presented in this work.

Remark 13 (Necessity–sufficiency template). Taken together, Theorem 3 and Theorem 4 provide a compact necessity–sufficiency template for long-horizon world modeling: persistent memory is necessary whenever the Bayes predictor is not recent-window measurable, and a hybrid multi-scale temporal memory is approximately sufficient whenever the Bayes predictor admits the corresponding factorization and the global memory evolves contractively.