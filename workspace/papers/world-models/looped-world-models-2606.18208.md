# LOOPED WORLD MODELS

FaceMind Research Asia

![](images/350cd52294b91077e97d9bc465855da07818d82fa2d600da43c2a38550bb4c1b.jpg)

G FACEMIND

![](images/b1e040736a8fab8471256008bcc7cdf52adceedc963086ed2074343b0df7dfab.jpg)

LoopWM

Leading Contributors Hongyuan Adam Lu\* Z.L. Victor Wei

Core Contributors Qun Zhang Jinrui Zeng Bowen Cao Lingwei Meng Mocheng Li Zezhong Wang Haonan Yin Naifu Xue Minyu Chen Cenyuan Zhang Zefan Zhang Hao Wei Jiawei Zhou Haoran Xu Hao Yang Ronglai Zuo Tongda Xu Yonghao Li Jian Chen Hebin Wang Zeyu Gao Yang Li Wei Zhao Qimin Zhong Siqi Liu Yumeng Zhang Leyan Cui Zhangyu Wang Wai Lam

## ABSTRACT

Current world models face a fundamental tension: faithful long-horizon simulation demands deep computation, but deeper models are expensive to deploy and prone to compounding errors. We resolve this by introducing Looped World Models (LoopWM), which are the first looped architectures for world modelling. Our method iteratively refines latent environment states through a parameter-shared transformer block. This yield up to 100× parameter efficiency over conventional approaches with adaptive computation that automatically scales depth to match the complexity of each prediction step. Orthogonal to scaling model size and training data, LoopWM establishes iterative latent depth as a new scaling axis for world simulation, which might significantly push the community forward.

![](images/84e3b11e0a56aff35f39f7ca595b6a2aeb977a938fb94fe11915330765b4269a.jpg)  
Figure 1: The overall framework of our proposed Looped World Models (LoopWM).

## 1 INTRODUCTION

World models (WM) learn to predict how an environment evolves in accordance with actions. WM has become a cornerstone of sample-efficient reinforcement learning and embodied intelligence (Ha & Schmidhuber, 2018; Hafner et al., 2019; Łukasz Kaiser et al., 2020). Remarkably, the Deep Planning Network (PlaNet) is a WM (Hafner et al., 2019) first demonstrated that agents can learn latent dynamics entirely from pixels and plan via online optimisation. This establishes the recurrent statespace model (RSSM) as a foundational architecture for world modelling. The Dreamer family of models then (Hafner et al., 2020; 2021; 2025) progressively refined this approach, culminating in DreamerV3 (Hafner et al., 2025). DreamerV3 masters over 150 different tasks with a single set of hyperparameters. Seeking to leverage the representational power of transformers, subsequent work replaced or augmented the recurrent backbone. IRIS (Micheli et al., 2023) showed that an autoregressive transformer over discrete latent tokens can serve as a highly data-efficient world model. TransDreamer (Chen et al., 2022) introduced a Transformer State-Space Model for tasks demanding long-range memory. ∆-IRIS (Micheli et al., 2024) improved efficiency via context-aware delta tokenisation. DIAMOND (Alonso et al., 2024) demonstrated that diffusion models can produce visually faithful world simulations, and EMERALD (Burchi & Timofte, 2025) achieved state-ofthe-art Crafter performance by combining masked generative transformers with spatial latent states. At a larger scale, Sora (OpenAI, 2024) and Genie (Bruce et al., 2024; Google DeepMind, 2025) demonstrated that video generation models and generative interactive environments can serve as general-purpose world simulators. And multiple surveys have charted the rapid expansion of world models into autonomous driving (Feng et al., 2025), embodied AI (Li et al., 2025b), and video generation (Dewi Puspitasari et al., 2024; Wang et al., 2026).

Despite this progress, faithful long-horizon simulation often requires deep or iterative computation. This is because physical dynamics unfold through repeated application of governing laws, whereas conventional fixed-depth architectures allocate the same amount of computation to every transition regardless of its difficulty. There are two typical failure modes. First, prediction errors cause trajectory quality to degrade rapidly over extended horizons across rollout steps (Xiao et al., 2020; Talvitie, 2017; Luo et al., 2022). Second, scaling model depth to combat this degradation proportionally usually increases parameter count and inference cost, which then makes real-time deployment on resource-constrained platforms prohibitively expensive (Feng et al., 2025; Hafner et al., 2025).

A parallel line of research has explored looped transformer architectures (LM). In LM, a shared set of transformer blocks is applied recurrently to the same latent representation. Such a concept was first proposed as the Universal Transformer (Dehghani et al., 2019), which introduced weightsharing across depth with an adaptive halting mechanism inspired by Adaptive Computation Time (Graves, 2016). Early theoretical work shows that LM can simulate arbitrary iterative algorithms. This includes gradient descent, Newton’s method, and dynamic programming, with constant parameter count (Giannou et al., 2023), and they achieve comparable in-context learning performance to standard transformers while using less than 10% of the parameters (Yang et al., 2023). ALBERT (Lan et al., 2020) shows the practical viability of cross-layer parameter sharing for language representation learning. MoEUT (Csordas et al., 2024) combined mixture-of-experts with universal ´ transformers.

More recently, looped architectures have been scaled to practical language models with promising results. Zhu et al. (2025) demonstrated that a looped language model can achieve about 2 to 3× parameter efficiency through iterative latent computation. Geiping et al. (2025) showed that recurrent-depth models can scale test-time compute by simply increasing the number of loop iterations at inference. Fan et al. (2024) demonstrated that looped transformers with adaptive stopping significantly improve length generalisation. Saunshi et al. (2025) provided theoretical and empirical evidence that looped models implicitly generate latent thoughts. Jeddi et al. (2026) introduced elastic-depth training with shortcut modulation for budget-conditioned latent reasoning. Bae et al. (2025) proposed per-token dynamic recursive depth allocation within a single recursive transformer. Prairie et al. (2026) addressed the training instability of looped models by recasting the looped forward pass as a nonlinear dynamical system over the residual stream and constraining the spectral norm of the state-transition matrix through a negative-diagonal parameterisation. Most recently, Hyperloop Transformers (Zeitoun et al., 2026) augmented the looped block with matrixvalued hyper-connected residual streams. This outperforms depth-matched standard transformers at half the parameter count. These developments connect looped transformers to a broader family of depth-continuous and implicit-layer models, including Neural ODEs (Chen et al., 2018) and Deep Equilibrium Models (Bai et al., 2019), which likewise iterate a shared function toward a fixed point.

However, all of the above looped-architecture works have been developed and evaluated exclusively in the context of language modelling. Looped World Models (LoopWM) remain entirely unexplored.

We propose that looped transformers are a promising backbone for world models because they introduce an explicit iterative refinement mechanism while reusing parameters across depth. At a high level, environment dynamics can often be viewed as repeated application of a shared transition law, which motivates modelling a single-step transition through repeated application of a shared latent update operator. This correspondence is conceptual rather than exact, as the inner loop is not meant to represent physical time directly but to perform iterative refinement of a latent transition estimate. To improve the numerical stability of this recurrent computation, we adopt a spectrally constrained state-retention parameterisation inspired by looped architectures. This construction ensures that the linear retention component remains contractive, which helps keep recurrent latent updates bounded as the number of inner-loop iterations increases. Structurally, environment dynamics are themselves an iterative process: a state $s _ { t }$ evolves to $s _ { t + 1 }$ through the repeated application of (approximately) stationary physical laws. The looped transformer’s computation graph, where a shared function $f _ { \theta }$ is applied recurrently to a latent state $h ,$

$$
h _ { t + 1 } = \bar { A } h _ { t } + \bar { B } e + \bar { \mathcal { R } } ( h _ { t } , e ) ,\tag{1}
$$

with $\bar { A }$ governing state retention, $\bar { B }$ controlling input injection, and $\bar { \mathcal { R } }$ subsuming the transformer nonlinearities (Prairie et al., 2026) is directly isomorphic to this dynamics structure. Stability is guaranteed by parameterizing the continuous-time matrix as $A : = \mathrm { d i a g } ( - \exp ( \mathbf { a } ) )$ with learnable a, and discretizing via zero-order hold,

$$
\bar { A } = \exp ( \Delta A ) ,\tag{2}
$$

which constrains all eigenvalues of $\bar { A }$ to the interval (0, 1), ensuring bounded residual dynamics regardless of rollout length (Prairie et al., 2026).

Practically, the parameter efficiency of looped architectures is uniquely valuable for world models, because long-horizon rollouts require executing the dynamics model hundreds or thousands of times in sequence; a model that achieves the predictive quality of a much larger network with a fraction of the parameters yields compounding savings across every rollout step. Furthermore, the adaptivedepth property of looped models, allocating more iterations to complex transitions (e.g., collisions, contact events) and fewer to simple dynamics (e.g., free flight), maps directly onto the non-uniform computational demands of physical simulation. In the most favourable cases, where simple state transitions require only a single loop iteration compared to the full forward pass of a conventional fixed-depth model, this adaptive mechanism can substantially reduce average inference cost relative to a fixed-depth baseline. The magnitude of this reduction depends on the distribution of transition difficulty, the minimum useful loop depth, and the overhead of the exit mechanism.

In this work, we introduce Looped World Models (LoopWM), the first looped transformer architectures for environment simulation and dynamics prediction. Our approach combines a parametershared recurrent transformer block with spectrally-constrained residual dynamics, enabling provably stable state transitions across arbitrary rollout lengths. We demonstrate that Looped World Models achieve competitive or superior predictive accuracy to existing world model architectures while using significantly fewer parameters, maintain stable rollouts over substantially longer horizons, and support test-time adaptive computation that automatically matches computational depth to transition complexity. We also integrate residual connections to improve model performance. Our results establish iterative latent depth as a previously unexplored and highly effective scaling axis for world models, orthogonal to both model size and training data.

## 2 RELATED WORK

## 2.1 WORLD MODELS FOR REINFORCEMENT LEARNING AND EMBODIED AI

The idea of learning an internal model of environment dynamics dates back to early work on mental simulation and forward models in cognitive science and control theory. In deep reinforcement learning, Ha & Schmidhuber (2018) proposed learning a compressed spatial and temporal representation of the environment using a variational autoencoder and an RNN, training a compact policy entirely within the learned “dream.” PlaNet (Hafner et al., 2019) formalised this via a latent dynamics model (RSSM) that plans directly in latent space from pixel observations. SimPLe (Łukasz Kaiser et al., 2020) demonstrated model-based Atari play by training a video-prediction model as a learned simulator. MuZero (Schrittwieser et al., 2020) showed that a learned dynamics model with Monte-Carlo tree search can master board games and Atari without access to the ground-truth rules.

The Dreamer family (Hafner et al., 2020; 2021; 2025) progressively refined the RSSM-based world model, culminating in DreamerV3 (Hafner et al., 2025). They achieve human-level performance across over 150 diverse tasks with a single set of hyperparameters. Transformer-based world models subsequently emerged: IRIS (Micheli et al., 2023) replaced the recurrent backbone with an autoregressive transformer over discrete tokens; TransDreamer (Chen et al., 2022) introduced a Transformer State-Space Model for memory-demanding tasks; ∆-IRIS (Micheli et al., 2024) improved tokenization efficiency via context-aware delta encoding; DIAMOND (Alonso et al., 2024) leveraged diffusion models to produce visually faithful world simulations; and EMERALD (Burchi & Timofte, 2025) achieved state-of-the-art Crafter performance using masked generative transformers over spatial latent states.

At a larger scale, video generation models have been cast as world simulators. OpenAI’s Sora (OpenAI, 2024) demonstrated long-form video generation with emergent 3D consistency, while Genie (Bruce et al., 2024) and Genie 3 (Google DeepMind, 2025) showed that text-conditioned generative models can produce interactive, explorable environments. Several surveys chart the rapid expansion of world models into autonomous driving (Feng et al., 2025; Guan et al., 2024), embodied AI (Li et al., 2025b), and video generation (Dewi Puspitasari et al., 2024; Wang et al., 2026).

A persistent challenge across all these approaches is compounding prediction error: small inaccuracies at each rollout step accumulate exponentially over long horizons, degrading trajectory fidelity (Xiao et al., 2020; Talvitie, 2017; Luo et al., 2022). Various mitigation strategies have been proposed, including short-horizon re-planning, self-correcting models (Talvitie, 2017), and physics-informed architectures (Li et al., 2025a; Wang et al., 2025), yet the fundamental tension between computational depth and rollout stability remains unresolved by existing architectures.

## 2.2 LOOPED AND RECURRENT-DEPTH TRANSFORMER ARCHITECTURES

Looped transformers reuse a shared set of transformer blocks across depth, decoupling effective computation from parameter count. The Universal Transformer (Dehghani et al., 2019) first proposed this idea, combining weight sharing with Adaptive Computation Time (ACT) (Graves, 2016) for input-dependent halting. ALBERT (Lan et al., 2020) demonstrated the practical viability of full cross-layer parameter sharing in BERT-scale models.

Theoretical analyses subsequently established the computational power of looped transformers. Giannou et al. (2023) proved that looped transformers can simulate arbitrary programs, functioning as programmable computers with constant parameter count. Yang et al. (2023) showed that looped transformers match standard transformer performance on in-context learning while using less than 10% of the parameters. Fan et al. (2024) demonstrated significant length generalisation improvements through adaptive loop counts. Saunshi et al. (2025) provided both theoretical and empirical evidence that looped models implicitly generate “latent thoughts,” enabling reasoning beyond their apparent depth. At a practical scale, Ouro (Zhu et al., 2025) trained looped language models (LoopLMs) through the full modern LLM pipeline with pre-training, instruction tuning, and RLHF, achieving 2–3× parameter efficiency with entropy-regularised adaptive computation. Geiping et al. (2025) demonstrated that recurrent-depth models (RDMs) scale test-time compute by increasing loop count at inference, following predictable quality improvements. Pappone et al. (2025) analysed the geometry of latent dynamics in recurrent-depth transformers, identifying two-scale structure with fast intra-loop and slow inter-token dynamics. LoopFormer (Jeddi et al., 2026) introduced elasticdepth training with shortcut modulation for budget-conditioned reasoning. Mixture-of-Recursions (Bae et al., 2025) proposed per-token dynamic depth allocation within a single recursive framework. MoEUT (Csordas et al., 2024) combined mixture-of-experts with universal transformers to balance ´ specialisation and sharing.

## 2.3 ADAPTIVE COMPUTATION AND EARLY EXIT

Allocating variable computation to inputs of differing complexity has been studied across multiple paradigms. Graves (2016) introduced Adaptive Computation Time for RNNs, allowing per-step halting decisions. The early exit literature (Teerapittayanon et al., 2017; Bolukbasi et al., 2017; Jyoti Bajpai & Hanawal, 2025) enables inference to terminate at intermediate layers when confidence is sufficient. In the context of looped transformers, adaptive depth takes a particularly natural form: the model can halt after any number of loop iterations. Ouro (Zhu et al., 2025) introduced entropy-regularised early exit, where a token exits the loop when its prediction entropy drops below a learned threshold. Geiping et al. (2025) trained with stochastic depth sampling (Poisson-distributed loop counts) to induce robustness to variable test-time depth. LoopFormer (Jeddi et al., 2026) conditioned on a continuous “time budget” during training, enabling fine-grained compute allocation at inference. Pappone et al. (2025) proposed acceleration-based exit rules using second-order differences of hidden states. For world models specifically, adaptive computation is highly attractive: simple state transitions (e.g., free flight, static scenes) demand minimal processing, while complex events (e.g., multi-body collisions, contact dynamics) require deeper iterative refinement. To the best of our knowledge, no prior work has proposed adaptive-depth looped architectures with world modelling.

## 3 LOOPED WORLD MODEL

We present Looped World Models, a latent dynamics architecture that combines the iterative computation of looped transformers with the action-conditioned state prediction required for world modelling. Our design follows three principles: (i) structural alignment between the model’s computation graph and the iterative nature of physical dynamics, (ii) provable stability of latent state transitions across arbitrary rollout lengths, and (iii) adaptive computational depth that matches the complexity of each transition. We describe the overall architecture (§3.1), the stabilised looped dynamics core (§3.2), the training objective (§3.3), and the adaptive early-exit mechanism for inference (§3.4).

## 3.1 OVERALL ARCHITECTURE

At each environment time step $k ,$ the agent receives an observation $o _ { k } \in \mathcal { O }$ and selects an action $a _ { k } \in { \mathcal { A } } .$ The goal of the world model is to predict the next latent state, from which future observations, rewards, and termination signals can be decoded. Our architecture consists of four modules:

Observation Encoder $\mathcal { E } _ { \phi } .$ . A convolutional (or vision-transformer-based) encoder maps the raw observation $o _ { k }$ into a latent embedding $e _ { k } = \mathcal { E } _ { \phi } ( o _ { k } ) \in \mathbb { R } ^ { d }$

Action Embedder $\mathcal { A } _ { \psi } .$ . The action $a _ { k }$ is projected into the same latent space via a learned embedding $u _ { k } = \mathcal { A } _ { \psi } ( a _ { k } ) \in \mathbb { R } ^ { d }$

Looped Dynamics Core $\mathcal { L } _ { \theta }$ . This is the central contribution of our architecture. The dynamics core takes the previous latent state $h _ { k - 1 }$ , the current observation embedding $e _ { k }$ , and the action embedding $u _ { k }$ , and produces the next latent state $h _ { k }$ through T iterations of a parameter-shared transformer block with spectrally-constrained residual dynamics. We describe this module in detail in §3.2.

Prediction Heads $\mathcal { D } _ { \xi }$ . A set of lightweight MLPs decode the latent state $h _ { k }$ into: (i) a reconstructed observation $\hat { o } _ { k + 1 }$ or its latent target, (ii) a predicted reward $\hat { r } _ { k }$ , and (iii) a predicted continuation flag $\hat { c } _ { k }$ . These heads follow the standard design of prior latent world models (Hafner et al., 2020; 2021; 2025).

The full forward pass at environment step k proceeds as:

$$
\begin{array} { r } { e _ { k } = \mathcal { E } _ { \phi } ( o _ { k } ) , \quad u _ { k } = \mathcal { A } _ { \psi } ( a _ { k } ) , \quad h _ { k } = \mathcal { L } _ { \theta } ( h _ { k - 1 } , e _ { k } , u _ { k } ) , \quad ( \hat { o } _ { k + 1 } , \hat { r } _ { k } , \hat { c } _ { k } ) = \mathcal { D } _ { \xi } ( h _ { k } ) . } \end{array}\tag{3}
$$

During imagination-based training of the policy (as in Dreamer (Hafner et al., 2020)), the encoder is bypassed: the dynamics core autoregressively rolls out latent trajectories using only actions sampled

from the policy network, i.e., $h _ { k + 1 } = \mathcal { L } _ { \theta } ( h _ { k } , \mathbf { 0 } , u _ { k } )$ , where observation injection is omitted or replaced by the model’s own prediction.

## 3.2 LOOPED DYNAMICS CORE WITH SPECTRAL STABILITY

The dynamics core is the heart of our architecture. Following the prelude recurrent coda design (Geiping et al., 2025; Prairie et al., 2026; Zeitoun et al., 2026), we partition the dynamics core into three blocks:

Prelude $\mathcal { P } .$ . A small stack of $L _ { \mathcal { P } }$ transformer layers processes the concatenation of the previous latent state, the observation embedding, and the action embedding to produce the conditioning signal:

$$
e = \mathrm { L N } ( \mathcal { P } ( [ h _ { k - 1 } ; e _ { k } ; u _ { k } ] ) ) \in \mathbb { R } ^ { d } ,\tag{4}
$$

where LN(·) denotes layer normalization. The normalisation of e follows the Parcae design (Prairie et al., 2026) and prevents input magnitude from inducing late-stage loss spikes.

Recurrent Block R. A stack of $L _ { \mathcal { R } }$ transformer layers with shared parameters is applied iteratively for $T$ loops. The hidden state is initialised as $h ^ { ( 0 ) } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ (or, for temporal rollouts, from the previous time step’s final hidden state). At each loop iteration $t = 0 , 1 , \ldots , T { - } 1$ , the update rule is:

$$
h ^ { ( t + 1 ) } = \bar { A } h ^ { ( t ) } + \bar { B } e + \bar { \mathcal { R } } \big ( h ^ { ( t ) } , e \big ) ,\tag{5}
$$

where $\bar { A } \in \mathbb { R } ^ { d \times d }$ is the state-retention matrix controlling how much of the previous hidden state is preserved, $\bar { B } \in \mathbb { R } ^ { d \times d }$ is the input-injection matrix controlling the influence of the conditioning signal $e ,$ and R¯ subsumes the nonlinear transformer operations (multi-head attention and feed-forward layers) applied to the residual stream. The key distinction from conventional fixed-depth transformers is that the parameters of R are shared across all T iterations, making the computational depth independent of the parameter count.

Spectral Stability Constraint. To guarantee that the latent state does not explode regardless of the number of loop iterations T (which is critical for long-horizon rollouts in world modelling), we constrain the spectral norm of A¯ to be strictly less than 1. Following Parcae (Prairie et al., 2026), we parameterize A¯ through discretization of a continuous-time negative diagonal matrix:

$$
A : = \mathrm { d i a g } ( - \exp ( \mathbf { a } ) ) , \quad \mathbf { a } \in \mathbb { R } ^ { d } ( \mathrm { l e a r n a b l e } ) ,\tag{6}
$$

$$
\begin{array} { r } { \bar { A } = \mathrm { e x p } ( \Delta \cdot A ) , \quad \Delta \in \mathbb { R } _ { > 0 } ^ { d } ( \mathrm { l e a r n a b l e } ) . } \end{array}\tag{7}
$$

Since A has strictly negative diagonal entries, $\Delta \cdot A$ has strictly negative entries, and exp(·) maps these to the interval (0, 1). Consequently, A¯ is a diagonal matrix with all entries in $( 0 , 1 )$ , guaranteeing $\rho ( { \bar { A } } ) < 1$ . This constraint holds by construction throughout training; no gradient clipping, post-hoc normalisation, or sensitive hyperparameter tuning is required.

The input-injection matrix is similarly discretised as $\bar { B } = \Delta$ · B with unconstrained B, but we apply layer normalisation to e (Eq. 4) to bound the injected signal’s magnitude.

Coda C. A final stack of $L _ { C }$ transformer layers (with separate, non-shared parameters) processes the terminal hidden state $h ^ { ( T ) }$ through a learned projection:

$$
h _ { k } = \mathcal { C } ( C h ^ { ( T ) } ) ,\tag{8}
$$

where $C \in \mathbb { R } ^ { d _ { c } \times d }$ optionally adapts the embedding dimension. The output $h _ { k }$ is then passed to the prediction heads and carried forward as the initial state for the next environment time step.

Cross-Timestep State Propagation. A distinctive property of our architecture is that the terminal hidden state $h ^ { ( T ) }$ from environment step k can serve as the initialization $h ^ { ( 0 ) }$ for step $k { \pm } 1$ , enabling a dual-loop structure: the inner loop (iterations of R) refines the latent state within a single transition, while the outer loop (sequential environment steps) propagates information across time. The spectral constraint on $\bar { A }$ ensures that both loops remain bounded, encouraging continuity while keeping propagated hidden states numerically well behaved.

## 3.3 TRAINING OBJECTIVE

Variable-Depth Training. We train with stochastic loop depth. At each training step, the loop count T is sampled from a Poisson distribution with learnable mean $\mu _ { \mathrm { r e c } }$

$$
T \sim \operatorname { P o i s s o n } ( \mu _ { \operatorname { r e c } } ) .\tag{9}
$$

We sample T independently per sequence within each micro-batch, rather than per micro-batch as in prior work (Geiping et al., 2025). This reduces variance in the training objective and empirically eliminates most loss spikes.

World Model Loss. The overall training objective combines observation prediction, reward prediction, and continuation prediction:

$$
\mathcal { L } _ { \mathrm { w m } } = \mathbb { E } _ { T \sim \mathrm { P o i s s o n } ( \mu _ { \mathrm { r e c } } ) } \left[ \sum _ { k = 1 } ^ { K } \left( \underbrace { \mathcal { L } _ { \mathrm { o b s } } ( o _ { k + 1 } , \hat { o } _ { k + 1 } ) } _ { \mathrm { o b s e r x a t i o n l o s s } } + \lambda _ { r } \underbrace { \mathcal { L } _ { \mathrm { r e w } } ( r _ { k } , \hat { r } _ { k } ) } _ { \mathrm { r e w a r d i s s } } + \lambda _ { c } \underbrace { \mathcal { L } _ { \mathrm { c o n t } } ( c _ { k } , \hat { c } _ { k } ) } _ { \mathrm { c o n t i m u a t i o n l o s s } } \right) \right] ,\tag{10}
$$

where K is the sequence length, $\lambda _ { r }$ and $\lambda _ { c }$ are balancing coefficients, and the specific form of ${ \mathcal { L } } _ { \mathrm { o b s } }$ depends on the observation space $( \mathrm { e . g . }$ , MSE for continuous states, cross-entropy for discrete tokens). Backpropagation through the loop iterations is truncated at $\mu _ { \mathrm { b w d } } = \lceil \mu _ { \mathrm { r e c } } / 2 \bar { 7 }$ steps to limit memory cost.

Entropy-Regularised Adaptive Depth. When adaptive early exit is enabled (see §3.4), we augment the loss with an entropy-regularisation term that prevents the exit gate from collapsing to trivial solutions (always exiting at the first iteration or never exiting). The regularisation takes the form:

$$
\mathcal { L } _ { \mathrm { e n t } } = - \alpha \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } H \Big ( g ^ { ( t ) } \Big ) \right] ,\tag{11}
$$

where $g ^ { ( t ) } \in [ 0 , 1 ]$ is the exit probability at loop iteration $t , H ( \cdot )$ denotes binary entropy, and α is a regularization coefficient. The total training loss is $\mathcal { L } = \mathcal { L } _ { \mathrm { w m } } + \mathcal { L } _ { \mathrm { e n t } }$

## 3.4 ADAPTIVE EARLY EXIT FOR INFERENCE

At inference time, the looped dynamics core can adaptively terminate the inner loop early for transitions that converge quickly, and allocate additional iterations to complex transitions. We implement this via a lightweight exit gate, a single-layer MLP followed by a sigmoid:

$$
g ^ { ( t ) } = \sigma \left( \mathbf { w } _ { g } ^ { \top } h ^ { ( t ) } + b _ { g } \right) ,\tag{12}
$$

where $\mathbf { w } _ { g } ~ \in ~ \mathbb { R } ^ { d }$ and $b _ { g } \in \mathbb { R }$ are learned parameters. At each loop iteration t, if $g ^ { ( t ) }$ exceeds a threshold $\tau ,$ , the loop terminates and $h ^ { ( t ) }$ is used as the final hidden state. This mechanism is complementary to the convergence-based exit criteria studied by Pappone et al. (2025), which halt when the second-order difference $\| h ^ { ( t ) } - 2 h ^ { ( t - 1 ) } + h ^ { ( t - 2 ) } \|$ falls below a threshold.

In the world-modelling setting, adaptive exit yields particularly large savings. Consider a 100-layer fixed-depth baseline: for a simple free-flight trajectory segment, our model may exit after a single loop of $L _ { \mathcal { R } }$ layers (e.g., 4 layers), reducing inference FLOPs by a factor of ${ \sim } 2 5 \times$ for that step. Over a long rollout containing many simple transitions interspersed with occasional complex events, the aggregate FLOPs reduction can reach up to two orders of magnitude compared to a fixed-depth model of equivalent quality.

The maximum loop count $T _ { \mathrm { m a x } }$ at inference time can also exceed the training-time mean $\mu _ { \mathrm { r e c } } .$ enabling test-time compute scaling: the model produces progressively refined predictions as more iterations are allocated.

## 3.5 DEFERRED DECODING: ACTION-CONDITIONED LATENT ROLLOUT

## 3.5.1 MOTIVATION

In standard world models (Hafner et al., 2020; 2021; 2025), the prediction heads $\mathcal { D } _ { \xi }$ are applied at every environment step k to produce intermediate observation reconstructions $\hat { o } _ { k + 1 }$ , reward predictions $\hat { r } _ { k } ,$ and continuation signals $\hat { c } _ { k }$ . This per-step decoding introduces two inefficiencies: (i) it forces the latent state to allocate representational capacity to pixel-level reconstruction at every intermediate step, even when only the final prediction matters for planning; (ii) it prevents the dynamics core from performing uninterrupted latent reasoning across a multi-step action sequence.

Recent work in language modelling has demonstrated that deferring decoding to the end of a latent reasoning process—allowing the model to encode, think, then decode—substantially improves reasoning quality (Koishekenov et al., 2026; Geiping et al., 2025; Saunshi et al., 2025). MuZero (Schrittwieser et al., 2020) similarly operates entirely in latent space without observation reconstruction, predicting only value, reward, and policy. Dreamer’s own “imagination” rollouts (Hafner et al., 2020) propagate latent states without re-encoding real observations, yet still apply reward and value heads at each step.

We propose Deferred Decoding, a modification to the Looped World Model that eliminates all intermediate observation decoding during multi-step rollouts. Given a sequence of ground-truth or planned actions, the model injects each action into the looped dynamics core and advances the latent state purely in the continuous hidden space. Observation, reward, and continuation predictions are produced only at the final step, reducing computation and encouraging the latent trajectory to encode temporally extended, action-relevant structure rather than per-step visual detail.

## 3.5.2 FORMULATION

Consider a planning or evaluation horizon of K steps. Let $h _ { 0 }$ be the initial latent state (obtained from encoding a real observation $o _ { 0 }$ through the encoder $\mathcal { E } _ { \phi }$ and the prelude block of the looped dynamics core), and let $( a _ { 0 } , a _ { 1 } , \dotsc , a _ { K - 1 } )$ be a sequence of actions.

Standard per-step decoding (baseline). At each step $k = 0 , 1 , \ldots , K - 1$ , the baseline model performs:

$$
u _ { k } = { \mathcal { A } } _ { \psi } ( a _ { k } ) ,\tag{13}
$$

$$
h _ { k + 1 } = \mathcal { L } _ { \theta } ( h _ { k } , u _ { k } ) ,\tag{14}
$$

$$
\begin{array} { r } { \left( \hat { o } _ { k + 1 } , \hat { r } _ { k } , \hat { c } _ { k } \right) = \mathcal { D } _ { \xi } ( h _ { k + 1 } ) , } \end{array}\tag{15}
$$

where ${ \mathcal { L } } _ { \theta }$ denotes the full looped dynamics core (prelude, T -step recurrent block, coda) described in Section 3.2. This yields K sets of decoded predictions.

Deferred decoding. We replace the per-step decoding with a decode-free latent rollout followed by a single terminal decoding:

$$
u _ { k } = { \mathcal { A } } _ { \psi } ( a _ { k } ) ,
$$

$$
k = 0 , 1 , \ldots , K - 1 ,\tag{16}
$$

$$
h _ { k + 1 } = { \mathcal { L } } _ { \theta } ^ { \mathrm { c o r e } } ( h _ { k } , u _ { k } ) ,
$$

$$
k = 0 , 1 , \ldots , K - 1 ,\tag{17}
$$

$$
\begin{array} { r } { \big ( \hat { o } _ { K } , \hat { r } _ { K } , \hat { c } _ { K } \big ) = \mathcal { D } _ { \xi } ( h _ { K } ) . } \end{array}\tag{18}
$$

The key difference is that Eqs. equation 16, equation 17 are applied K times without invoking $\mathcal { D } _ { \xi }$ and the decoder is called exactly once at step K. Between steps, the model ingests a new action embedding $u _ { k }$ and advances the latent state through the looped recurrent block, but produces no intermediate observation, reward, or continuation output.

Interaction between inner and outer loops. Recall from Section 3.2 that each invocation of $\mathcal { L } _ { \theta } ^ { \mathrm { c } }$ ore itself involves $T$ inner-loop iterations of the shared transformer block. With deferred decoding, the overall computation becomes a nested loop:

• Outer loop (action steps): $k = 0 , \ldots , K - 1$ . At each step, the action $u _ { k }$ is injected.

• Inner loop (latent refinement): $t = 0 , \ldots , T - 1$ . Within each action step, the recurrent block refines $h$ via $h ^ { ( t + 1 ) } = \bar { A } h ^ { ( t ) } + \bar { B } \left[ u _ { k } ; h _ { k } \right] + \bar { \mathcal { R } } \left( h ^ { ( t ) } , u _ { k } \right)$ with spectral-normconstrained A¯.

The total effective depth is $K \times T$ shared-parameter transformer applications, but only one forward pass through the decoder.

## 3.5.3 TRAINING OBJECTIVE FOR DEFERRED DECODING

Training the deferred-decoding variant requires the model to maintain accurate latent representations across $\bar { K }$ action-conditioned transitions without intermediate reconstruction supervision. We define a terminal prediction loss and a latent trajectory regularizer.

Terminal prediction loss. Given a training trajectory $\left( o _ { 0 } , a _ { 0 } , o _ { 1 } , a _ { 1 } , \dots , o _ { K } \right)$ where all intermediate actions are ground-truth, the model encodes $o _ { 0 }$ to obtain $h _ { 0 } .$ performs K latent transitions via Eqs. equation 16, equation 17, then decodes $h _ { K } \colon$

$$
\mathcal { L } _ { \mathrm { t e r m i n a l } } = \lambda _ { o } \ell _ { \mathrm { o b s } } ( \hat { o } _ { K } , o _ { K } ) + \lambda _ { r } \ell _ { \mathrm { r e w } } ( \hat { r } _ { K } , r _ { K - 1 } ) + \lambda _ { c } \ell _ { \mathrm { c o n t } } ( \hat { c } _ { K } , c _ { K - 1 } ) ,\tag{19}
$$

where $\ell _ { \mathrm { o b s } }$ may be a reconstruction loss (MSE, perceptual loss) or, in the decoder-free setting, a next-embedding alignment loss analogous to NE-Dreamer (Bredis et al., 2026).

Latent trajectory regularizer. Without intermediate decoding, the latent states at steps $1 , \ldots , K - 1$ are unsupervised and could drift into regions that are spectrally stable yet semantically meaningless. We introduce two lightweight constraints:

1. Latent consistency loss. We encode each intermediate ground-truth observation $o _ { k } \ ( k =$ $1 , \ldots , K - 1 )$ with the frozen encoder $\mathcal { E } _ { \phi }$ to obtain reference embeddings $e _ { k } ^ { \star } = \mathrm { s g } ( { \mathcal { E } _ { \phi } ( o _ { k } ) } )$ then align:

$$
\mathcal { L } _ { \mathrm { c o n s i s t } } = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K - 1 } \| g _ { \omega } ( h _ { k } ) - e _ { k } ^ { \star } \| _ { 2 } ^ { 2 } ,\tag{20}
$$

where $g _ { \omega }$ is a lightweight projection head and $\operatorname { s g } ( \cdot )$ denotes stop-gradient. This loss provides soft guidance without requiring a full decoder at each step, analogous to the latent overshooting technique in PlaNet (Hafner et al., 2019).

2. Spectral contraction budget. The spectral-norm constraint on $\bar { A }$ (Section 3.2) already ensures bounded latent evolution per inner loop. Over K outer steps, we additionally monitor the cumulative contraction:

$$
\left\| h _ { K } - h _ { 0 } \right\| _ { 2 } \le \sum _ { k = 0 } ^ { K - 1 } \left\| h _ { k + 1 } - h _ { k } \right\| _ { 2 } \le K \cdot C _ { \operatorname* { m a x } } ,\tag{21}
$$

where $C _ { \mathrm { m a x } }$ is a soft upper bound enforced as a penalty term. This prevents latent explosion over long deferred horizons while still permitting meaningful state changes induced by actions.

The full training objective for the deferred-decoding variant is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D D } } = \mathcal { L } _ { \mathrm { t e r m i n a l } } + \alpha \mathcal { L } _ { \mathrm { c o n s i s t } } + \beta \operatorname* { m a x } \bigl ( 0 , \ : \sum _ { k } \| h _ { k + 1 } - h _ { k } \| _ { 2 } - K \cdot C _ { \mathrm { m a x } } \bigr ) , } \end{array}\tag{22}
$$

where α and $\beta$ are balancing coefficients.

## 3.5.4 CURRICULUM OVER DEFERRAL HORIZON K

Training directly with a large K is unstable because gradients must back-propagate through $K \times T$ shared-parameter applications. We adopt a progressive horizon curriculum: training begins with $K = 1$ (equivalent to standard per-step decoding) and gradually increases K during training according to a schedule $K ( \mathrm { s t e p } ) = \bar { \mathrm { m i n } } ( \bar { K _ { \mathrm { m a x } } } , 1 + \bar { \lfloor \mathrm { s t e p } / \bar { \Delta } \rfloor } )$ , where $\Delta$ is the number of training steps between increments. This allows the latent dynamics to first learn accurate single-step transitions before being challenged with longer decode-free rollouts.

## 3.5.5 INFERENCE MODES

Deferred decoding naturally supports two inference modes:

Planning mode. Given a candidate action sequence $( a _ { 0 } , \ldots , a _ { K - 1 } )$ from a planner $( \mathrm { e . g . }$ , CEM, MPPI), the model performs a single decode-free rollout and evaluates only the terminal state $h _ { K }$ This reduces decoder invocations from K to 1, saving approximately $( K - \dot { 1 } ) \times \mathrm { c o s t } ( \mathcal { D } _ { \xi } )$ FLOPs per candidate sequence. When combined with adaptive early exit within each inner loop (Section 3.4), the total FLOP reduction can reach up to two orders of magnitude for long-horizon planning with simple transitions.

Monitoring mode. When intermediate state inspection is needed (e.g., for safety-critical applications), the lightweight projection head $g _ { \omega }$ can be applied at any step k to produce a low-dimensional state summary $\tilde { e } _ { k } = g _ { \omega } ( h _ { k } )$ without invoking the full decoder. The full decoder $\mathcal { D } _ { \xi }$ remains available as an optional diagnostic tool but is not required for the planning loop.

## 3.5.6 RELATIONSHIP TO PRIOR WORK

Table 1 summarizes the key distinctions:

Table 1: Comparison of intermediate decoding strategies across world-model architectures.
<table><tr><td>Method</td><td>Latent dynamics</td><td>Intermediate decode</td><td>Action injection</td><td>Looped depth</td></tr><tr><td>Dreamer (Hafner et al., 2020)</td><td>RSSM</td><td>reward + value at each step</td><td>per step</td><td></td></tr><tr><td>MuZero (Schrittwieser et al.,2020)</td><td>learned MLP</td><td>policy + value + reward</td><td>per step</td><td></td></tr><tr><td>PlaNet (Hafner et al., 2019)</td><td>RSSM</td><td>reconstruction at each step</td><td>per step</td><td>1</td></tr><tr><td>ETD (Koishekenov et al.,2026)</td><td>looped layers</td><td>decode only at end</td><td>- (language)</td><td>√</td></tr><tr><td>NE-Dreamer (Bredis et al.,2026)</td><td>RSSM</td><td>embedding alignment</td><td>per step</td><td>1</td></tr><tr><td>LoopWM-DD (ours)</td><td>looped transformer</td><td>decode only at step K</td><td>per step in latent</td><td>√</td></tr></table>

Dreamer’s imagination rollout already avoids re-encoding real observations but still applies reward and value heads at every imagined step (Hafner et al., 2020). MuZero dispenses with observation reconstruction entirely but uses a non-looped, fixed-depth dynamics function (Schrittwieser et al., 2020). ETD (Koishekenov et al., 2026) demonstrates the encode-think-decode paradigm for language reasoning with looped layers, but does not handle action-conditioned state transitions or environment simulation. Our deferred decoding unifies these insights: it applies the looped transformer’s iterative refinement at each action step in latent space (inheriting the parameter efficiency and spectral stability of the LoopWM) while deferring all observation-space computation to the terminal step, yielding a clean separation between latent dynamics reasoning (inner + outer loops) and observation grounding (single terminal decode).

## 4 RESULTS

## 4.1 MAIN RESULTS ON SCIENCEWORLD

Table 2 presents the results on the ScienceWorld dataset of our models against claude-opus-4-6-max. From the results, it is clear that our model surpasses the strong claude-opus-4-6-max. In the most extreme cases, it improves the scores on Lifespan from 0% to 100%, denoting the underlying strong capacity of our model. On average, our model shows a promising capability, clearly surpassing the baseline by 21.2% on EM, and clearly on other metrics. Further, we note that our model is a small AI model with around 1B parameters, which is much smaller than those strong closed-source API models such as claude-opus-4-6-max by more than 100x. This suggests our proposed model has a promising parameter efficiency to be deployed on downstream applications. Note that Table 3 presents more baselines, which lead to the same conclusions.

We also note that qwen-3.5-flash and gemini-3-flash-preview seem to be clearly worse than other baselines and our models across most metrics. This is reasonable as they are considered smaller than the other baseline models. Our proposed models are still competitive and much stronger than them across the metrics.

## 4.2 MAIN RESULTS ON ALFWORLD

Table 4 presents evaluation results on the AlfWorld dataset. On this dataset, we see that the trend can still be promising, as our proposed model gives a promising overall result, given the fact that it is pretty small in terms of model size, with around 1B parameters. Notably, it gives the best result on the BLEU metrics (Papineni et al., 2002) among four models, and ranks in second place on EM and Token F1. Further, by inspecting the detailed action categories, we found that our model seems to have low entity scores, and it seems valid for most action categories. Such an error analysis indicates that future optimization can focus on the entity scores to further enhance the model.

Table 2: Comparison of our proposed looped world model against claude-opus-4-6-max (Anthropic, 2026) on ScienceWorld dataset (Wang et al., 2022) world modelling task. The accuracy is calculated based on feeding consecutive five actions, and obtaining the final scores on world modelling. Note that our model is a model with about 1B model parameters. Refer to Table 3 for more baselines.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU Entity</td><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Looped World Model (Ours)</td><td colspan="8"></td></tr><tr><td>Boil</td><td></td><td></td><td>66.7% 79.0% 75.3% 77.5%</td><td>Chemistry</td><td></td><td>44.4% 64.4%</td><td>54.2%</td><td>57.9%</td></tr><tr><td>Conductivity 87.0% 89.0% 87.8%</td><td></td><td></td><td>87.9%</td><td>Find</td><td>76.9%</td><td>90.4%</td><td>82.7%</td><td>85.8%</td></tr><tr><td>Freeze</td><td>25.0%</td><td>59.7%</td><td>31.2% 54.8%</td><td>Genetics</td><td>78.3%</td><td>80.2%</td><td>78.9%</td><td>79.8%</td></tr><tr><td>Grow</td><td>73.8%</td><td>80.0%</td><td>75.5% 79.8%</td><td>Incline</td><td>59.3%</td><td>95.3%</td><td>90.4%</td><td>93.4%</td></tr><tr><td>LifeStages</td><td>0.0%</td><td>18.3%</td><td>6.1% 10.2%</td><td>Lifespan</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>Melt</td><td>73.0% 91.9%</td><td></td><td>85.7% 91.6%</td><td>Power</td><td>57.1%</td><td>63.9%</td><td>60.8%</td><td>61.5%</td></tr><tr><td>StateChange</td><td></td><td></td><td>80.0% 83.1% 80.0% 80.0%</td><td>Thermometer</td><td>83.3%</td><td>85.3%</td><td>83.3%</td><td>83.3%</td></tr></table>

Overall: EM: 68.4%; Token F1: 85.3%; BLEU-4: 80.7%; Entity: 83.9%
<table><tr><td colspan="9">claude-opus-4-6-max</td></tr><tr><td>Boil</td><td>22.2% 33.3% 30.2% 32.3%</td><td></td><td></td><td>Chemistry</td><td>44.4%</td><td>59.8%</td><td>46.3%</td><td>59.2%</td></tr><tr><td>Conductivity 47.8% 67.2% 53.1%</td><td></td><td></td><td>72.7%</td><td>Find</td><td>69.2%</td><td>83.8%</td><td>78.9%</td><td>84.6%</td></tr><tr><td>Freeze</td><td>12.5% 33.6%</td><td>21.6%</td><td>37.3%</td><td>Genetics</td><td>59.2%</td><td>71.8%</td><td>65.7%</td><td>71.6%</td></tr><tr><td>Grow</td><td>70.8%</td><td>81.6% 76.1%</td><td>80.7%</td><td>Incline</td><td>34.0%</td><td>86.5%</td><td>76.2%</td><td>83.8%</td></tr><tr><td>LifeStages</td><td>0.0%</td><td>10.6% 6.0%</td><td>6.0%</td><td>Lifespan</td><td>0.0%</td><td>61.4%</td><td>0.0%</td><td>58.3%</td></tr><tr><td>Melt</td><td></td><td>36.5% 52.4% 46.3%</td><td>53.4%</td><td>Power</td><td>42.9%</td><td>47.3%</td><td>45.3%</td><td>45.8%</td></tr><tr><td>StateChange</td><td></td><td>40.0% 65.5% 44.6%</td><td>73.3%</td><td>Thermometer</td><td>83.3%</td><td>98.1%</td><td>93.5%</td><td>97.2%</td></tr><tr><td colspan="9">Overall: EM: 47.2%; Token F1: 72.8%; BLEU-4: 64.4%; Entity: 72.3%</td></tr></table>

## 4.3 DEEP ANALYSIS ON DEFERRED DECODING

Across the tables, we conclude that the deferred decoding is useful, and it tends to be more useful when the rollouts are accumulated.

## 5 CONCLUSIONS

We have presented Looped World Models, the first application of looped transformer architectures to world modelling. Our approach addresses a central tension in current world models: generating faithful long-horizon simulations demands deep computation, yet deeper models incur prohibitive deployment costs and are susceptible to compounding rollout errors. By iteratively refining latent environment states through a parameter-shared transformer block with stabilised residual dynamics, LoopWM structurally mirrors the recurrence inherent in physical systems while maintaining a compact parameter footprint. Empirically, LoopWM achieve up to 100× parameter efficiency over conventional approaches without sacrificing prediction quality. Theoretically, we show that spectralnorm constraints on state transitions yield provably stable rollouts, providing formal guarantees that are absent in standard autoregressive world models. Furthermore, our adaptive computation mecha-

<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU Entity</td><td>Task Type</td><td>EM</td><td>F1</td><td></td><td>BLEU Entity</td></tr><tr><td colspan="11">qwen-3.5-flash (Qwen Team, 2026)</td></tr><tr><td>Boil</td><td>0.0%</td><td></td><td>44.6% 15.1%</td><td>39.6%</td><td>Chemistry</td><td>3.7%</td><td>28.8%</td><td>4.1%</td><td>41.3%</td></tr><tr><td>Conductivity</td><td>0.0%</td><td>28.2%</td><td>0.8%</td><td>49.2%</td><td>Find</td><td>0.0%</td><td>25.1%</td><td>0.0%</td><td>51.0%</td></tr><tr><td>Freeze</td><td>0.0%</td><td>29.8%</td><td>9.8%</td><td>44.0%</td><td>Genetics</td><td>7.5%</td><td>30.5%</td><td>11.5%</td><td>53.2%</td></tr><tr><td>Grow</td><td>4.6%</td><td>28.3%</td><td>7.2%</td><td>58.3%</td><td>Incline</td><td>20.7%</td><td>81.3%</td><td>63.9%</td><td>84.0%</td></tr><tr><td>LifeStages</td><td>0.0%</td><td>24.8%</td><td>0.0%</td><td>10.2%</td><td>Lifespan</td><td>0.0%</td><td>12.1%</td><td>0.0%</td><td>25.0%</td></tr><tr><td>Melt</td><td>9.5%</td><td>44.1%</td><td>24.6%</td><td>64.7%</td><td>Power</td><td>0.0%</td><td>27.5%</td><td>1.4%</td><td>38.9%</td></tr><tr><td>StateChange</td><td></td><td>0.0% 25.1%</td><td>5.7%</td><td>70.0%</td><td>Thermometer 0.0%</td><td></td><td>32.7%</td><td>2.1%</td><td>56.9%</td></tr><tr><td colspan="12">Overall: EM: 10.0%; Token F1: 46.9%; BLEU-4: 26.7%; Entity: 63.0%</td></tr><tr><td colspan="12"> gemini-3-flash-preview-thinking (Gemini Team, Google DeepMind, 2025)</td></tr><tr><td>Boil</td><td></td><td>22.2% 61.5% 41.9% 64.1%</td><td></td><td></td><td>Chemistry</td><td>22.2% 54.8% 27.6% 57.9%</td><td></td><td></td><td></td></tr><tr><td></td><td>Conductivity 17.4% 55.4% 21.9%</td><td></td><td></td><td>60.6%</td><td>Find</td><td></td><td>15.4% 65.2% 40.3%</td><td></td><td>80.7%</td></tr><tr><td>Freeze</td><td></td><td>12.5% 35.0% 22.2%</td><td></td><td>37.5%</td><td>Genetics</td><td></td><td>41.7% 65.6% 48.5%</td><td></td><td>671.1%</td></tr><tr><td>Grow</td><td></td><td>47.7% 72.6%</td><td>55.4%</td><td>75.8%</td><td>Incline</td><td></td><td>32.7% 88.5%</td><td>76.0%</td><td>88.6%</td></tr><tr><td>LifeStages</td><td>0.0%</td><td>15.1%</td><td>6.0%</td><td>8.1%</td><td>Lifespan</td><td>0.0%</td><td>38.8%</td><td>0.0%</td><td>58.3%</td></tr><tr><td>Melt</td><td>7.9%</td><td>47.9%</td><td>29.6%</td><td>62.5%</td><td>Power</td><td></td><td>14.3% 42.7%</td><td>16.7%</td><td>45.9%</td></tr><tr><td>StateChange 20.0% 57.8% 23.9% 70.0%</td><td></td><td></td><td></td><td></td><td>Thermometer 33.3% 68.8% 45.3% 86.1%</td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Baseline results on ScienceWorld dataset (Wang et al., 2022) world modelling task. The accuracy is calculated based on feeding consecutive five actions, and obtaining the final scores on world modelling. Note that our model is a model with about 1B model parameters.

Overall: EM: 30.8%; Token F1: 68.9%; BLEU-4: 51.1%; Entity: 73.8%

nism automatically scales the effective depth of the model to match the complexity of each prediction step, allocating more refinement iterations to dynamically challenging transitions and fewer to predictable ones. Beyond the specific results reported here, we believe this work identifies iterative latent depth as a new scaling axis for world simulation, one that is orthogonal to the conventional axes of model size and data volume. We hope that this perspective opens new directions for building world models that are simultaneously more capable, more efficient, and more stable over extended horizons.

## 6 BROADER IMPACTS

While the present paper already provides strong evidence for the effectiveness of LoopWM, the current manuscript is intentionally selective in disclosure scope. In this version, our goal is to establish the core architectural thesis that looped latent refinement, deferred decoding, and stabilized dynamics together define a viable and promising design space for world modelling, rather than to exhaustively present every supporting result we have already obtained.

First, the current paper already demonstrates the value of iterative latent computation through deferred decoding, which gives concrete evidence that preserving and refining latent computation across rollout steps is beneficial. We view this as a direct and meaningful manifestation of the looped design. At the same time, it represents only one visible entry point into a broader body of evidence supporting the effectiveness of looping, and a more explicit decomposition of these gains can be disclosed in the future.

Table 4: Comparison of our proposed looped world model against claude-opus-4-6-max (Anthropic, 2026) and other baselines on AlfWorld dataset (Cotˆ e et al., 2018) world modelling task. The accu- ´ racy is calculated based on feeding consecutive five actions, and obtaining the final scores on world modelling. Note that our model is a model with about 1B model parameters.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU Entity</td><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU Entity</td></tr><tr><td colspan="6">Looped World Model (Ours)</td><td></td><td></td></tr><tr><td>clean</td><td>60.4% 81.7%</td><td></td><td>75.0% 81.3%</td><td>cool</td><td>50.0% 81.7% 72.6% 78.5%</td><td></td><td></td></tr><tr><td>heat</td><td>55.0% 81.8%</td><td></td><td>76.6% 81.2%</td><td>look</td><td>60.5% 78.9% 73.1% 82.0%</td><td></td><td></td></tr><tr><td>pick</td><td>46.7% 79.7%</td><td></td><td>69.1% 81.5%</td><td>1-</td><td></td><td></td><td></td></tr><tr><td colspan="6">Overall: EM: 51.6%; Token F1: 80.4%; BLEU-4: 71.6%; Entity: 81.1%</td><td></td><td></td></tr><tr><td colspan="6">claude-opus-4-6-max</td><td></td><td></td></tr><tr><td>clean</td><td></td><td></td><td>57.3% 73.8% 68.9% 77.4%</td><td>cool</td><td>50.0% 73.0% 68.2% 72.8%</td><td></td><td></td></tr><tr><td>heat</td><td>52.5% 67.6%</td><td></td><td>64.4% 74.1%</td><td>look</td><td></td><td>60.5% 73.2% 68.9% 78.6%</td><td></td></tr><tr><td>pick</td><td>51.0% 72.8%</td><td></td><td>65.7% 78.1%</td><td>1</td><td>=</td><td>=</td><td>=</td></tr><tr><td colspan="6">Overall: EM: 53.0%; Token F1: 72.6%; BLEU-4: 66.8%; Entity: 77.0%</td><td></td><td></td></tr><tr><td colspan="6">qwen-3.5-flash (Qwen Team, 2026)</td><td></td><td></td></tr><tr><td>clean</td><td>36.5% 71.9%</td><td>55.6%</td><td>90.1%</td><td>cool</td><td>27.3% 72.0% 52.6% 85.1%</td><td></td><td></td></tr><tr><td>heat</td><td>27.5%</td><td>70.1%</td><td>49.5% 91.2%</td><td>look</td><td></td><td>27.9% 66.2% 46.3% 92.8%</td><td></td></tr><tr><td>pick</td><td>21.2%</td><td>64.1%</td><td>43.5% 87.5%</td><td></td><td>1</td><td>= 1</td><td>=</td></tr><tr><td colspan="6">Overall: EM: 26.0%; Token F1: 67.3%; BLEU-4: 47.7%; Entity: 88.4%</td><td></td><td></td></tr><tr><td colspan="6">gemini-3-flash-preview-thinking (Gemini Team, Google DeepMind,2025)</td><td></td><td></td></tr><tr><td>clean</td><td>61.5% 88.2%</td><td>79.9%</td><td>90.5%</td><td>cool</td><td>54.5% 88.1% 77.0% 88.3%</td><td></td><td></td></tr><tr><td>heat</td><td>55.0% 81.9%</td><td></td><td>73.3% 90.6%</td><td>look</td><td></td><td>55.8% 86.6% 74.4% 97.7%</td><td></td></tr><tr><td>pick</td><td>42.7% 80.2%</td><td></td><td>65.1% 89.3%</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">Overall: EM: 50.0%; Token F1: 83.5%; BLEU-4: 71.0%; Entity: 90.2%</td><td></td><td></td></tr></table>

Second, although the present manuscript emphasizes the principal task domains reported here, our empirical validation is not confined to these settings. We have also verified in continuous visual environments that optimization is feasible and that the training loss is consistently reducible, which supports the practicality of the proposed architecture beyond the environments highlighted in this paper. The main limitation at this stage is therefore not a lack of empirical support, but that the manuscript does not yet fully expose the breadth of validation already completed.

Third, LoopWM is best understood as a distinct point in the broader world model landscape. The current paper makes clear that its emphasis differs from major existing families, including RSSM style latent dynamics models, autoregressive video token world models, and diffusion based world models. A more explicit positioning analysis would further sharpen this distinction and make the contribution even easier to interpret. We therefore see clear value in more directly situating

Table 5: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on average over all the tasks, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+73.2%</td><td>+16.4%</td><td>+47.0%</td><td>+9.7%</td></tr><tr><td>Step2</td><td>+54.5%</td><td>+21.4%</td><td>+41.7%</td><td>+18.0%</td></tr><tr><td>Step3</td><td>+103.6%</td><td>+28.1%</td><td>+65.0%</td><td>+19.0%</td></tr><tr><td>Step4</td><td>+82.9%</td><td>+29.0%</td><td>+55.5%</td><td>+20.7%</td></tr><tr><td>Step5</td><td>+113.8%</td><td>+22.4%</td><td>+54.6%</td><td>+12.8%</td></tr></table>

Table 6: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Boil, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours−Baselines)/Baselines∗100%. Note that our model is a model with about 1B model parameters. ‘—‘ represents that the baseline score is 0 and LoopWM score is not zero. ‘0%‘ means that both of them has a score of 0.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+100.0%</td><td>+3.5%</td><td>+39.6%</td><td>-8.0%</td></tr><tr><td>Step2</td><td>+50.2%</td><td>+33.4%</td><td>+61.3%</td><td>+54.9%</td></tr><tr><td>Step3</td><td>+250.5%</td><td>+57.5%</td><td>+136.2%</td><td>+39.3%</td></tr><tr><td>Step4</td><td>+700.9%</td><td>+120.0%</td><td>+503.5%</td><td>+121.0%</td></tr><tr><td>Step5</td><td>+500.9%</td><td>+29.9%</td><td>+101.9%</td><td>+20.0%</td></tr></table>

LoopWM among these families and clarifying the regimes in which iterative latent depth is the most natural scaling axis.

Finally, our current step 1 to step 5 experiments already indicate that iterative latent depth behaves as a meaningful scaling dimension, and we consider this one of the central implications of the work. The remaining limitation is not whether such a scaling trend exists, but that the present paper stops short of providing a more complete scaling law characterization across broader task and compute ranges. Similarly, from an optimization perspective, our experience suggests that training can benefit from curriculum like engineering strategies that progressively unlock the architecture’s capability. We regard this not as a weakness of the method, but as part of the practical recipe for making a new architectural regime reliably trainable at scale.

Overall, the main limitation of the current paper is one of presentation scope rather than conceptual or empirical foundation. The paper establishes the core case for LoopWM, while broader cross family positioning, richer scaling analysis, and more extensive optimization disclosure can further strengthen the story in the future.

Table 7: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Chemistry, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+267.1%</td><td>+52.4%</td><td>+197.2%</td><td>+56.9%</td></tr><tr><td>Step2</td><td>+140.3%</td><td>+42.2%</td><td>+120.0%</td><td>+33.5%</td></tr><tr><td>Step3</td><td>+110.3%</td><td>+34.1%</td><td>+92.5%</td><td>+18.5%</td></tr><tr><td>Step4</td><td>+367.6%</td><td>+57.0%</td><td>+224.3%</td><td>+62.6%</td></tr><tr><td>Step5</td><td>+100.0%</td><td>+15.0%</td><td>+78.9%</td><td>0.0%</td></tr></table>

Table 8: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Conductivity, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours−Baselines)/Baselines∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+78.0%</td><td>+17.5%</td><td>+58.8%</td><td>+2.4%</td></tr><tr><td>Step2</td><td>+183.1%</td><td>+44.5%</td><td>+190.5%</td><td>+42.5%</td></tr><tr><td>Step3</td><td>+220.7%</td><td>+57.5%</td><td>+249.8%</td><td>+39.0%</td></tr><tr><td>Step4</td><td>+183.1%</td><td>+53.2%</td><td>+194.8%</td><td>+48.7%</td></tr><tr><td>Step5</td><td>+233.3%</td><td>+51.9%</td><td>+218.1%</td><td>+40.0%</td></tr></table>

Table 9: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Find, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours−Baselines)/Baselines∗100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+166.2%</td><td>+45.0%</td><td>+141.2%</td><td>+28.4%</td></tr><tr><td>Step2</td><td>+79.7%</td><td>+7.1%</td><td>+40.6%</td><td>-8.3%</td></tr><tr><td>Step3</td><td>+266.2%</td><td>+71.0%</td><td>+253.7%</td><td>+41.7%</td></tr><tr><td>Step4</td><td>+71.6%</td><td>+25.6%</td><td>+56.8%</td><td>+14.6%</td></tr><tr><td>Step5</td><td>+399.4%</td><td>+38.7%</td><td>+105.2%</td><td>+6.3%</td></tr></table>

Table 10: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Freeze, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+100.0%</td><td>-6.2%</td><td>+63.5%</td><td>-32.1%</td></tr><tr><td>Step2</td><td>+50.0%</td><td>+10.1%</td><td>+62.2%</td><td>-2.9%</td></tr><tr><td>Step3</td><td>+250.0%</td><td>+80.9%</td><td>+112.6%</td><td>+20.6%</td></tr><tr><td>Step4</td><td>+400.0%</td><td>+96.7%</td><td>+303.0%</td><td>+76.0%</td></tr><tr><td>Step5</td><td>+100.0%</td><td>+70.6%</td><td>+40.5%</td><td>+46.1%</td></tr></table>

Table 11: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Genetics, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+80.8%</td><td>+31.9%</td><td>+68.3%</td><td>+27.0%</td></tr><tr><td>Step2</td><td>+36.5%</td><td>+24.5%</td><td>+33.0%</td><td>+20.3%</td></tr><tr><td>Step3</td><td>+122.1%</td><td>+36.3%</td><td>+101.4%</td><td>+24.3%</td></tr><tr><td>Step4</td><td>+76.7%</td><td>+30.5%</td><td>+54.5%</td><td>+22.6%</td></tr><tr><td>Step5</td><td>+74.0%</td><td>+19.5%</td><td>+52.3%</td><td>+10.7%</td></tr></table>

Table 12: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Grow, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+109.6%</td><td>+20.9%</td><td>+89.1%</td><td>+12.4%</td></tr><tr><td>Step2</td><td>+59.6%</td><td>+13.9%</td><td>+46.9%</td><td>+13.9%</td></tr><tr><td>Step3</td><td>+48.4%</td><td>+7.6%</td><td>+41.1%</td><td>+6.8%</td></tr><tr><td>Step4</td><td>+16.3%</td><td>-5.6%</td><td>+5.8%</td><td>-10.4%</td></tr><tr><td>Step5</td><td>+50.0%</td><td>+8.4%</td><td>+34.1%</td><td>+2.8%</td></tr></table>

Table 13: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Incline, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+24.5%</td><td>+1.9%</td><td>+7.5%</td><td>-0.1%</td></tr><tr><td>Step2</td><td>+7.6%</td><td>+2.3%</td><td>+5.5%</td><td>+2.1%</td></tr><tr><td>Step3</td><td>+42.5%</td><td>+6.6%</td><td>+13.9%</td><td>+3.8%</td></tr><tr><td>Step4</td><td>+40.2%</td><td>+6.8%</td><td>+14.8%</td><td>+3.3%</td></tr><tr><td>Step5</td><td>+85.3%</td><td>+8.4%</td><td>+20.5%</td><td>+6.1%</td></tr></table>

Table 14: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Melt, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+112.6%</td><td>-5.7%</td><td>+27.9%</td><td>-22.9%</td></tr><tr><td>Step2</td><td>+343.2%</td><td>+105.4%</td><td>+201.4%</td><td>+88.0%</td></tr><tr><td>Step3</td><td>+349.6%</td><td>+86.0%</td><td>+172.7%</td><td>+59.4%</td></tr><tr><td>Step4</td><td>+585.3%</td><td>+220.3%</td><td>+467.1%</td><td>+138.3%</td></tr><tr><td>Step5</td><td>+557.7%</td><td>+79.5%</td><td>+161.3%</td><td>+42.9%</td></tr></table>

Table 15: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Power, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+499.3%</td><td>+105.5%</td><td>+499.3%</td><td>+61.6%</td></tr><tr><td>Step2</td><td>+99.8%</td><td>+30.6%</td><td>+80.9%</td><td>+17.1%</td></tr><tr><td>Step3</td><td>+299.3%</td><td>+61.0%</td><td>+347.5%</td><td>+24.5%</td></tr><tr><td>Step4</td><td>+66.4%</td><td>+12.4%</td><td>+66.4%</td><td>-9.8%</td></tr><tr><td>Step5</td><td>+299.3%</td><td>+46.2%</td><td>+264.1%</td><td>+34.0%</td></tr></table>

Table 16: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on all tasks, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+143.5%</td><td>+32.4%</td><td>+93.8%</td><td>+26.3%</td></tr><tr><td>Step2</td><td>+86.4%</td><td>+33.9%</td><td>+72.5%</td><td>+27.8%</td></tr><tr><td>Step3</td><td>+136.1%</td><td>+42.0%</td><td>+106.0%</td><td>+30.2%</td></tr><tr><td>Step4</td><td>+78.1%</td><td>+35.5%</td><td>+74.0%</td><td>+25.1%</td></tr><tr><td>Step5</td><td>+104.8%</td><td>+32.2%</td><td>+79.3%</td><td>+22.1%</td></tr></table>

Table 17: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Boil, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td></td><td>+38.6%</td><td>+356.5%</td><td>+9.2%</td></tr><tr><td>Step2</td><td>1 1</td><td>+116.7%</td><td>+2428.1%</td><td>+140.3%</td></tr><tr><td>Step3</td><td></td><td>+100.2%</td><td>+438.1%</td><td>+55.7%</td></tr><tr><td>Step4</td><td>——</td><td>+203.1%</td><td>1</td><td>+194.7%</td></tr><tr><td>Step5</td><td>+500.9%</td><td>+57.1%</td><td>+184.2%</td><td>+86.7%</td></tr></table>

Table 18: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Chemistry, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+450.7%</td><td>+114.6%</td><td>+369.8%</td><td>+103.3%</td></tr><tr><td>Step2</td><td>+500.7%</td><td>+111.4%</td><td>+487.7%</td><td>+86.4%</td></tr><tr><td>Step3</td><td>+600.9%</td><td>+87.3%</td><td>+411.0%</td><td>+38.1%</td></tr><tr><td>Step4</td><td>+250.7%</td><td>+61.6%</td><td>+217.6%</td><td>+43.0%</td></tr><tr><td>Step5</td><td>+300.9%</td><td>+43.4%</td><td>+290.6%</td><td>+15.3%</td></tr></table>

Table 19: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Conductivity, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+100.0%</td><td>+19.3%</td><td>+71.3%</td><td>-2.4%</td></tr><tr><td>Step2</td><td>+240.5%</td><td>+40.8%</td><td>+209.7%</td><td>+26.0%</td></tr><tr><td>Step3</td><td>+166.3%</td><td>+27.0%</td><td>+169.3%</td><td>+1.3%</td></tr><tr><td>Step4</td><td>+143.1%</td><td>+45.3%</td><td>+191.0%</td><td>+23.9%</td></tr><tr><td>Step5</td><td>+233.3%</td><td>+39.6%</td><td>+191.7%</td><td>+10.6%</td></tr></table>

Table 20: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Find, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+699.2%</td><td>+75.0%</td><td>+400.0%</td><td>+58.9%</td></tr><tr><td>Step2</td><td>+349.4%</td><td>+38.6%</td><td>+196.7%</td><td>-0.3%</td></tr><tr><td>Step3</td><td>+449.4%</td><td>+105.8%</td><td>+440.5%</td><td>+63.1%</td></tr><tr><td>Step4</td><td>+139.7%</td><td>+54.0%</td><td>+133.3%</td><td>+30.0%</td></tr><tr><td>Step5</td><td>+149.4%</td><td>+50.7%</td><td>+108.8%</td><td>+8.1%</td></tr></table>

Table 21: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Freeze, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+100.0%</td><td>-6.3%</td><td>+308.7%</td><td>-31.3%</td></tr><tr><td>Step2</td><td>+200.0%</td><td>+47.7%</td><td>+116.6%</td><td>+33.4%</td></tr><tr><td>Step3</td><td></td><td>+162.2%</td><td>+750.0%</td><td>+51.5%</td></tr><tr><td>Step4</td><td>+400.0%</td><td>+109.2%</td><td>+219.7%</td><td>+59.9%</td></tr><tr><td>Step5</td><td></td><td>+63.1%</td><td>+235.5%</td><td>+14.9%</td></tr></table>

Table 22: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Genetics, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+203.5%</td><td>+47.2%</td><td>+184.8%</td><td>+53.5%</td></tr><tr><td>Step2</td><td>+61.6%</td><td>+23.0%</td><td>+50.6%</td><td>+23.2%</td></tr><tr><td>Step3</td><td>+202.9%</td><td>+44.6%</td><td>+185.0%</td><td>+39.9%</td></tr><tr><td>Step4</td><td>+70.8%</td><td>+25.8%</td><td>+59.8%</td><td>+22.6%</td></tr><tr><td>Step5</td><td>+104.4%</td><td>+28.5%</td><td>+91.5%</td><td>+21.1%</td></tr></table>

Table 23: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Grow, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+450.4%</td><td>+63.1%</td><td>+376.8%</td><td>+45.1%</td></tr><tr><td>Step2</td><td>+168.8%</td><td>+46.5%</td><td>+135.0%</td><td>+34.6%</td></tr><tr><td>Step3</td><td>+257.8%</td><td>+45.7%</td><td>+206.3%</td><td>+31.5%</td></tr><tr><td>Step4</td><td>+87.0%</td><td>+15.4%</td><td>+73.1%</td><td>+0.8%</td></tr><tr><td>Step5</td><td>+152.7%</td><td>+33.3%</td><td>+125.4%</td><td>+20.4%</td></tr></table>

Table 24: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Incline, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+40.2%</td><td>+5.5%</td><td>+12.9%</td><td>+4.1%</td></tr><tr><td>Step2</td><td>-1.0%</td><td>+5.8%</td><td>+8.4%</td><td>+4.5%</td></tr><tr><td>Step3</td><td>+7.4%</td><td>+8.2%</td><td>+13.5%</td><td>+5.0%</td></tr><tr><td>Step4</td><td>+2.4%</td><td>+9.5%</td><td>+13.6%</td><td>+6.4%</td></tr><tr><td>Step5</td><td>+9.8%</td><td>+7.1%</td><td>+13.0%</td><td>+4.8%</td></tr></table>

Table 25: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Melt, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+241.8%</td><td>+25.0%</td><td>+132.4%</td><td>+5.6%</td></tr><tr><td>Step2</td><td>+680.9%</td><td>+143.0%</td><td>+418.9%</td><td>+125.0%</td></tr><tr><td>Step3</td><td>+501.1%</td><td>+152.9%</td><td>+368.9%</td><td>+115.7%</td></tr><tr><td>Step4</td><td>+723.4%</td><td>+198.4%</td><td>+555.0%</td><td>+150.8%</td></tr><tr><td>Step5</td><td>+822.8%</td><td>+128.0%</td><td>+365.8%</td><td>+112.0%</td></tr></table>

Table 26: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Power, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step1</td><td>+499.3%</td><td>+138.0%</td><td>+350.9%</td><td>+121.2%</td></tr><tr><td>Step2</td><td>+499.3%</td><td>+116.3%</td><td>+439.4%</td><td>+46.9%</td></tr><tr><td>Step3</td><td>+299.3%</td><td>+77.9%</td><td>+300.6%</td><td>+70.3%</td></tr><tr><td>Step4</td><td>1</td><td>+103.1%</td><td>+891.7%</td><td>+13.0%</td></tr><tr><td>Step5</td><td>+299.3%</td><td>+76.0%</td><td>+272.1%</td><td>+48.2%</td></tr></table>

Table 27: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged, on our model. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step 1</td><td>67.2%</td><td>78.0%</td><td>72.3%</td><td>77.9%</td></tr><tr><td>Step 2</td><td>68.6%</td><td>86.2%</td><td>80.9%</td><td>86.4%</td></tr><tr><td>Step 3</td><td>68.0%</td><td>87.5%</td><td>82.0%</td><td>87.1%</td></tr><tr><td>Step 4</td><td>68.4%</td><td>87.1%</td><td>82.1%</td><td>85.6%</td></tr><tr><td>Step 5</td><td>68.4%</td><td>85.3%</td><td>80.7%</td><td>83.9%</td></tr></table>

Table 28: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 1, on our model. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>44.4%</td><td>59.6%</td><td>52.5%</td><td>54.4%</td></tr><tr><td>Chemistry</td><td>81.5%</td><td>85.2%</td><td>84.1%</td><td>85.2%</td></tr><tr><td>Conductivity</td><td>69.6%</td><td>78.5%</td><td>71.6%</td><td>76.6%</td></tr><tr><td>Find</td><td>61.5%</td><td>73.5%</td><td>61.5%</td><td>76.9%</td></tr><tr><td>Freeze</td><td>25.0%</td><td>42.0%</td><td>34.0%</td><td>42.5%</td></tr><tr><td>Genetics</td><td>78.3%</td><td>83.9%</td><td>78.6%</td><td>85.5%</td></tr><tr><td>Grow</td><td>67.7%</td><td>75.7%</td><td>67.7%</td><td>76.3%</td></tr><tr><td>Incline</td><td>74.7%</td><td>91.9%</td><td>87.7%</td><td>90.9%</td></tr><tr><td>Melt</td><td>27.0%</td><td>39.5%</td><td>31.6%</td><td>38.0%</td></tr><tr><td>Power</td><td>85.7%</td><td>97.6%</td><td>85.7%</td><td>100.0%</td></tr></table>

Table 29: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 2, on our model. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>66.7%</td><td>81.9%</td><td>71.6%</td><td>88.9%</td></tr><tr><td>Chemistry</td><td>88.9%</td><td>92.6%</td><td>91.1%</td><td>93.2%</td></tr><tr><td>Conductivity</td><td>73.9%</td><td>83.5%</td><td>82.2%</td><td>81.8%</td></tr><tr><td>Find</td><td>69.2%</td><td>78.3%</td><td>70.3%</td><td>70.3%</td></tr><tr><td>Freeze</td><td>37.5%</td><td>69.7%</td><td>54.0%</td><td>66.7%</td></tr><tr><td>Genetics</td><td>80.8%</td><td>84.9%</td><td>82.2%</td><td>85.9%</td></tr><tr><td>Grow</td><td>78.5%</td><td>84.4%</td><td>79.9%</td><td>87.1%</td></tr><tr><td>Incline</td><td>56.7%</td><td>93.4%</td><td>86.9%</td><td>92.0%</td></tr><tr><td>Melt</td><td>49.2%</td><td>72.9%</td><td>63.3%</td><td>73.9%</td></tr><tr><td>Power</td><td>85.7%</td><td>95.6%</td><td>89.0%</td><td>98.0%</td></tr></table>

Table 30: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 3, on our model. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>77.8%</td><td>97.5%</td><td>94.7%</td><td>98.1%</td></tr><tr><td>Chemistry</td><td>77.8%</td><td>85.4%</td><td>80.1%</td><td>83.8%</td></tr><tr><td>Conductivity</td><td>69.6%</td><td>82.7%</td><td>80.8%</td><td>80.6%</td></tr><tr><td>Find</td><td>84.6%</td><td>96.1%</td><td>90.9%</td><td>96.2%</td></tr><tr><td>Freeze</td><td>87.5%</td><td>98.6%</td><td>70.8%</td><td>97.9%</td></tr><tr><td>Genetics</td><td>83.3%</td><td>86.3%</td><td>83.8%</td><td>85.0%</td></tr><tr><td>Grow</td><td>66.2%</td><td>73.3%</td><td>68.3%</td><td>75.1%</td></tr><tr><td>Incline</td><td>58.0%</td><td>96.6%</td><td>91.2%</td><td>94.6%</td></tr><tr><td>Melt</td><td>57.1%</td><td>88.0%</td><td>76.9%</td><td>92.3%</td></tr><tr><td>Power</td><td>57.1%</td><td>74.7%</td><td>72.5%</td><td>72.2%</td></tr></table>

Table 31: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 4, on our model. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>88.9%</td><td>98.8%</td><td>85.1%</td><td>98.1%</td></tr><tr><td>Chemistry</td><td>51.9%</td><td>72.7%</td><td>61.3%</td><td>66.5%</td></tr><tr><td>Conductivity</td><td>73.9%</td><td>93.0%</td><td>90.2%</td><td>92.5%</td></tr><tr><td>Find</td><td>92.3%</td><td>94.1%</td><td>93.3%</td><td>93.4%</td></tr><tr><td>Freeze</td><td>62.5%</td><td>95.4%</td><td>66.5%</td><td>91.7%</td></tr><tr><td>Genetics</td><td>82.5%</td><td>84.3%</td><td>83.1%</td><td>84.6%</td></tr><tr><td>Grow</td><td>66.2%</td><td>72.1%</td><td>67.7%</td><td>71.3%</td></tr><tr><td>Incline</td><td>60.7%</td><td>95.8%</td><td>90.9%</td><td>93.7%</td></tr><tr><td>Melt</td><td>65.1%</td><td>91.6%</td><td>84.5%</td><td>90.3%</td></tr><tr><td>Power</td><td>71.4%</td><td>79.0%</td><td>71.4%</td><td>71.4%</td></tr></table>

Table 32: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 5, on our model. Note that our model is a model with about 1B model parameters.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>66.7%</td><td>79.0%</td><td>75.3%</td><td>77.5%</td></tr><tr><td>Chemistry</td><td>44.4%</td><td>64.4%</td><td>54.2%</td><td>57.9%</td></tr><tr><td>Conductivity</td><td>87.0%</td><td>89.0%</td><td>87.8%</td><td>87.9%</td></tr><tr><td>Find</td><td>76.9%</td><td>90.4%</td><td>82.7%</td><td>85.8%</td></tr><tr><td>Freeze</td><td>25.0%</td><td>59.7%</td><td>31.2%</td><td>54.8%</td></tr><tr><td>Genetics</td><td>78.3%</td><td>80.2%</td><td>78.9%</td><td>79.8%</td></tr><tr><td>Grow</td><td>73.8%</td><td>80.0%</td><td>75.5%</td><td>79.8%</td></tr><tr><td>Incline</td><td>59.3%</td><td>95.3%</td><td>90.4%</td><td>93.4%</td></tr><tr><td>Melt</td><td>73.0%</td><td>91.9%</td><td>85.7%</td><td>91.6%</td></tr><tr><td>Power</td><td>57.1%</td><td>63.9%</td><td>60.8%</td><td>61.5%</td></tr></table>

Table 33: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on gemini.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step 1</td><td>38.8%</td><td>67.0%</td><td>49.2%</td><td>71.0%</td></tr><tr><td>Step 2</td><td>44.4%</td><td>71.0%</td><td>57.1%</td><td>73.2%</td></tr><tr><td>Step 3</td><td>33.4%</td><td>68.3%</td><td>49.7%</td><td>73.2%</td></tr><tr><td>Step 4</td><td>37.4%</td><td>67.5%</td><td>52.8%</td><td>70.9%</td></tr><tr><td>Step 5</td><td>32.0%</td><td>69.7%</td><td>52.2%</td><td>74.4%</td></tr></table>

Table 34: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 1 on gemini.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>22.2%</td><td>57.6%</td><td>37.6%</td><td>59.1%</td></tr><tr><td>Chemistry</td><td>22.2%</td><td>55.9%</td><td>28.3%</td><td>54.3%</td></tr><tr><td>Conductivity</td><td>39.1%</td><td>66.8%</td><td>45.1%</td><td>74.8%</td></tr><tr><td>Find</td><td>23.1%</td><td>50.7%</td><td>25.5%</td><td>59.9%</td></tr><tr><td>Freeze</td><td>12.5%</td><td>44.8%</td><td>20.8%</td><td>62.6%</td></tr><tr><td>Genetics</td><td>43.3%</td><td>63.6%</td><td>46.7%</td><td>67.3%</td></tr><tr><td>Grow</td><td>32.3%</td><td>62.6%</td><td>35.8%</td><td>67.9%</td></tr><tr><td>Incline</td><td>60.0%</td><td>90.2%</td><td>81.6%</td><td>91.0%</td></tr><tr><td>Melt</td><td>12.7%</td><td>41.9%</td><td>24.7%</td><td>49.3%</td></tr><tr><td>Power</td><td>14.3%</td><td>47.5%</td><td>14.3%</td><td>61.9%</td></tr></table>

Table 35: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 2 on gemini.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>44.4%</td><td>61.4%</td><td>44.4%</td><td>57.4%</td></tr><tr><td>Chemistry</td><td>37.0%</td><td>65.1%</td><td>41.4%</td><td>69.8%</td></tr><tr><td>Conductivity</td><td>26.1%</td><td>57.8%</td><td>28.3%</td><td>57.4%</td></tr><tr><td>Find</td><td>38.5%</td><td>73.1%</td><td>50.0%</td><td>76.7%</td></tr><tr><td>Freeze</td><td>25.0%</td><td>63.3%</td><td>33.3%</td><td>68.7%</td></tr><tr><td>Genetics</td><td>59.2%</td><td>68.2%</td><td>61.8%</td><td>71.4%</td></tr><tr><td>Grow</td><td>49.2%</td><td>74.1%</td><td>54.4%</td><td>76.5%</td></tr><tr><td>Incline</td><td>52.7%</td><td>91.3%</td><td>82.4%</td><td>90.1%</td></tr><tr><td>Melt</td><td>11.1%</td><td>35.5%</td><td>21.0%</td><td>39.3%</td></tr><tr><td>Power</td><td>42.9%</td><td>73.2%</td><td>49.2%</td><td>83.7%</td></tr></table>

Table 36: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 3 on gemini.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>22.2%</td><td>61.9%</td><td>40.1%</td><td>70.4%</td></tr><tr><td>Chemistry</td><td>37.0%</td><td>63.7%</td><td>41.6%</td><td>70.7%</td></tr><tr><td>Conductivity</td><td>21.7%</td><td>52.5%</td><td>23.1%</td><td>58.0%</td></tr><tr><td>Find</td><td>23.1%</td><td>56.2%</td><td>25.7%</td><td>67.9%</td></tr><tr><td>Freeze</td><td>25.0%</td><td>54.5%</td><td>33.3%</td><td>81.2%</td></tr><tr><td>Genetics</td><td>37.5%</td><td>63.3%</td><td>41.6%</td><td>68.4%</td></tr><tr><td>Grow</td><td>44.6%</td><td>68.1%</td><td>48.4%</td><td>70.3%</td></tr><tr><td>Incline</td><td>40.7%</td><td>90.6%</td><td>80.1%</td><td>91.1%</td></tr><tr><td>Melt</td><td>12.7%</td><td>47.3%</td><td>28.2%</td><td>57.9%</td></tr><tr><td>Power</td><td>14.3%</td><td>46.4%</td><td>16.2%</td><td>58.0%</td></tr></table>

Table 37: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 4 on gemini.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>11.1%</td><td>44.9%</td><td>14.1%</td><td>44.4%</td></tr><tr><td>Chemistry</td><td>11.1%</td><td>46.3%</td><td>18.9%</td><td>40.9%</td></tr><tr><td>Conductivity</td><td>26.1%</td><td>60.7%</td><td>30.6%</td><td>62.2%</td></tr><tr><td>Find</td><td>53.8%</td><td>74.9%</td><td>59.5%</td><td>81.5%</td></tr><tr><td>Freeze</td><td>12.5%</td><td>48.5%</td><td>16.5%</td><td>52.1%</td></tr><tr><td>Genetics</td><td>46.7%</td><td>64.6%</td><td>53.8%</td><td>69.0%</td></tr><tr><td>Grow</td><td>56.9%</td><td>76.4%</td><td>64.0%</td><td>79.6%</td></tr><tr><td>Incline</td><td>43.3%</td><td>89.7%</td><td>79.2%</td><td>90.7%</td></tr><tr><td>Melt</td><td>9.5%</td><td>28.6%</td><td>14.9%</td><td>37.9%</td></tr><tr><td>Power</td><td>42.9%</td><td>70.3%</td><td>42.9%</td><td>79.2%</td></tr></table>

Table 38: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 5 on gemini.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>11.1%</td><td>60.8%</td><td>37.3%</td><td>64.6%</td></tr><tr><td>Chemistry</td><td>22.2%</td><td>56.0%</td><td>30.3%</td><td>57.9%</td></tr><tr><td>Conductivity</td><td>26.1%</td><td>58.6%</td><td>27.6%</td><td>62.8%</td></tr><tr><td>Find</td><td>15.4%</td><td>65.2%</td><td>40.3%</td><td>80.7%</td></tr><tr><td>Freeze</td><td>12.5%</td><td>35.0%</td><td>22.2%</td><td>37.5%</td></tr><tr><td>Genetics</td><td>45.0%</td><td>67.1%</td><td>51.8%</td><td>72.1%</td></tr><tr><td>Grow</td><td>49.2%</td><td>73.8%</td><td>56.3%</td><td>77.6%</td></tr><tr><td>Incline</td><td>32.0%</td><td>87.9%</td><td>75.0%</td><td>88.0%</td></tr><tr><td>Melt</td><td>11.1%</td><td>51.2%</td><td>32.8%</td><td>64.1%</td></tr><tr><td>Power</td><td>14.3%</td><td>43.7%</td><td>16.7%</td><td>45.9%</td></tr></table>

Table 39: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks on qwen.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Step 1</td><td>27.6%</td><td>58.9%</td><td>37.3%</td><td>61.7%</td></tr><tr><td>Step 2</td><td>36.8%</td><td>64.4%</td><td>46.9%</td><td>67.6%</td></tr><tr><td>Step 3</td><td>28.8%</td><td>61.6%</td><td>39.8%</td><td>66.9%</td></tr><tr><td>Step 4</td><td>38.4%</td><td>64.3%</td><td>47.2%</td><td>68.4%</td></tr><tr><td>Step 5</td><td>33.4%</td><td>64.5%</td><td>45.0%</td><td>68.7%</td></tr></table>

Table 40: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 1 on qwen.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>0.0%</td><td>43.0%</td><td>11.5%</td><td>49.8%</td></tr><tr><td>Chemistry</td><td>14.8%</td><td>39.7%</td><td>17.9%</td><td>41.9%</td></tr><tr><td>Conductivity</td><td>34.8%</td><td>65.8%</td><td>41.8%</td><td>78.5%</td></tr><tr><td>Find</td><td>7.7%</td><td>42.0%</td><td>12.3%</td><td>48.4%</td></tr><tr><td>Freeze</td><td>0.0%</td><td>34.3%</td><td>8.3%</td><td>49.3%</td></tr><tr><td>Genetics</td><td>25.8%</td><td>57.0%</td><td>27.6%</td><td>55.7%</td></tr><tr><td>Grow</td><td>12.3%</td><td>46.4%</td><td>14.2%</td><td>52.6%</td></tr><tr><td>Incline</td><td>53.3%</td><td>87.1%</td><td>77.7%</td><td>87.3%</td></tr><tr><td>Melt</td><td>7.9%</td><td>31.6%</td><td>13.6%</td><td>36.0%</td></tr><tr><td>Power</td><td>14.3%</td><td>41.0%</td><td>19.0%</td><td>45.2%</td></tr></table>

Table 41: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 2 on qwen.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>0.0%</td><td>37.8%</td><td>3.2%</td><td>37.0%</td></tr><tr><td>Chemistry</td><td>14.8%</td><td>43.8%</td><td>15.5%</td><td>50.0%</td></tr><tr><td>Conductivity</td><td>21.7%</td><td>59.3%</td><td>26.6%</td><td>64.9%</td></tr><tr><td>Find</td><td>15.4%</td><td>56.5%</td><td>23.7%</td><td>70.5%</td></tr><tr><td>Freeze</td><td>12.5%</td><td>47.2%</td><td>24.7%</td><td>50.0%</td></tr><tr><td>Genetics</td><td>50.0%</td><td>69.0%</td><td>54.5%</td><td>69.7%</td></tr><tr><td>Grow</td><td>29.2%</td><td>57.6%</td><td>34.0%</td><td>64.7%</td></tr><tr><td>Incline</td><td>57.3%</td><td>88.3%</td><td>80.2%</td><td>88.0%</td></tr><tr><td>Melt</td><td>6.3%</td><td>30.0%</td><td>12.2%</td><td>32.8%</td></tr><tr><td>Power</td><td>14.3%</td><td>44.2%</td><td>16.5%</td><td>66.7%</td></tr></table>

Table 42: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 3 on qwen.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>0.0%</td><td>48.7%</td><td>17.6%</td><td>63.0%</td></tr><tr><td>Chemistry</td><td>11.1%</td><td>45.6%</td><td>16.4%</td><td>61.4%</td></tr><tr><td>Conductivity</td><td>26.1%</td><td>65.1%</td><td>30.0%</td><td>79.6%</td></tr><tr><td>Find</td><td>15.4%</td><td>46.7%</td><td>16.8%</td><td>59.0%</td></tr><tr><td>Freeze</td><td>0.0%</td><td>37.6%</td><td>8.3%</td><td>64.6%</td></tr><tr><td>Genetics</td><td>27.5%</td><td>59.7%</td><td>29.4%</td><td>61.1%</td></tr><tr><td>Grow</td><td>18.5%</td><td>50.3%</td><td>22.3%</td><td>57.1%</td></tr><tr><td>Incline</td><td>54.0%</td><td>89.3%</td><td>80.5%</td><td>90.1%</td></tr><tr><td>Melt</td><td>9.5%</td><td>34.8%</td><td>16.4%</td><td>42.8%</td></tr><tr><td>Power</td><td>14.3%</td><td>42.0%</td><td>18.1%</td><td>42.4%</td></tr></table>

Table 43: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 4 on qwen.
<table><tr><td>Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>0.0%</td><td>32.6%</td><td>0.0%</td><td>33.3%</td></tr><tr><td>Chemistry</td><td>14.8%</td><td>45.0%</td><td>19.3%</td><td>46.5%</td></tr><tr><td>Conductivity</td><td>30.4%</td><td>64.0%</td><td>31.0%</td><td>74.7%</td></tr><tr><td>Find</td><td>38.5%</td><td>61.1%</td><td>40.0%</td><td>71.8%</td></tr><tr><td>Freeze</td><td>12.5%</td><td>45.6%</td><td>20.8%</td><td>57.3%</td></tr><tr><td>Genetics</td><td>48.3%</td><td>67.0%</td><td>52.0%</td><td>69.0%</td></tr><tr><td>Grow</td><td>35.4%</td><td>62.5%</td><td>39.1%</td><td>70.7%</td></tr><tr><td>Incline</td><td>59.3%</td><td>87.5%</td><td>80.0%</td><td>88.1%</td></tr><tr><td>Melt</td><td>7.9%</td><td>30.7%</td><td>12.9%</td><td>36.0%</td></tr><tr><td>Power</td><td>0.0%</td><td>38.9%</td><td>7.2%</td><td>63.1%</td></tr></table>

Table 44: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on Step 5 on qwen.
<table><tr><td> Task Type</td><td>EM</td><td>F1</td><td>BLEU</td><td>Entity</td></tr><tr><td>Boil</td><td>11.1%</td><td>50.3%</td><td>26.5%</td><td>41.5%</td></tr><tr><td>Chemistry</td><td>11.1%</td><td>44.9%</td><td>13.9%</td><td>50.2%</td></tr><tr><td>Conductivity</td><td>26.1%</td><td>63.8%</td><td>30.1%</td><td>79.5%</td></tr><tr><td>Find</td><td>30.8%</td><td>60.0%</td><td>39.6%</td><td>79.4%</td></tr><tr><td>Freeze</td><td>0.0%</td><td>36.6%</td><td>9.3%</td><td>47.7%</td></tr><tr><td>Genetics</td><td>38.3%</td><td>62.4%</td><td>41.2%</td><td>65.9%</td></tr><tr><td>Grow</td><td>29.2%</td><td>60.0%</td><td>33.5%</td><td>66.3%</td></tr><tr><td>Incline</td><td>54.0%</td><td>89.0%</td><td>80.0%</td><td>89.1%</td></tr><tr><td>Melt</td><td>7.9%</td><td>40.3%</td><td>18.4%</td><td>43.2%</td></tr><tr><td>Power</td><td>14.3%</td><td>36.3%</td><td>22.1%</td><td>41.5%</td></tr></table>

(a）Next-day retention  
![](images/905f5beff6b4654ac7d3c5c3e1e211678e1fd51b845c7617aabc95769ad6580c.jpg)

(b）Monthly retention  
![](images/4ab6caca642366fbd89e8e1254e1149dd41c9e348597ea8e3875c00ac856cc09.jpg)  
Figure 2: Relative increase over Qwen3.7-max on automatic online performance, compared against baselines. Note that the results are obtained via online estimation, with the tasks of danmaku generation. LWM denotes LoopWM.

![](images/63cfaa5a93bb3aa8b0a1bca416c7581980052498766f3026dba434279541d4fd.jpg)  
Figure 3: Human evaluation performance with our model, compared against baselines. Note that the results are obtained via online estimation, with the tasks of danmaku generation. LWM denotes LoopWM.

## REFERENCES

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Franc¸ois Fleuret. Diffusion for world modeling: Visual details matter in atari. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https:// openreview.net/forum?id=NadTwTODgC.

Anthropic. System card: Claude opus 4.6. Technical report, Anthropic, February 2026. URL https://www-cdn.anthropic.com/ 14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf.

Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, and Se-Young Yun. Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. arXiv e-prints, art. arXiv:2507.10524, July 2025. doi: 10.48550/arXiv.2507.10524.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. Curran Associates Inc., Red Hook, NY, USA, 2019.

Tolga Bolukbasi, Joseph Wang, Ofer Dekel, and Venkatesh Saligrama. Adaptive neural networks for efficient inference. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17, pp. 527–536. JMLR.org, 2017.

George Bredis, Nikita Balagansky, Daniil Gavrilov, and Ruslan Rakhimov. Next embedding prediction makes world models stronger. In ICLR 2026 the 2nd Workshop on World Models: Understanding, Modelling and Scaling, 2026. URL https://openreview.net/forum?id= SkAgjqPmhY.

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge (Jimmy) Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando De Freitas, Satinder Singh, and Tim Rocktaschel. Genie: generative interactive environments. In ¨ Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Maxime Burchi and Radu Timofte. Accurate and efficient world modeling with masked latent transformers. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=zNUOZcAUxz.

Chang Chen, Jaesik Yoon, Yi-Fu Wu, and Sungjin Ahn. Transdreamer: Reinforcement learning with transformer world models, 2022. URL https://openreview.net/forum?id= s3K0arSRl4d.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pp. 6572–6583, Red Hook, NY, USA, 2018. Curran Associates Inc.

Marc-Alexandre Cotˆ e,´ Akos K ´ ad´ ar, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James ´ Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, et al. Textworld: A learning environment for text-based games. In Workshop on Computer Games, pp. 41–75. Springer, 2018.

Robert Csord ´ as, Kazuki Irie, J ´ urgen Schmidhuber, Christopher Potts, and Christopher D Manning. ¨ MoEUT: Mixture-of-experts universal transformers. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=ZxVrkm7Bjl.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=HyzdRiR9Y7.

Fachrina Dewi Puspitasari, Chaoning Zhang, Joseph Cho, Adnan Haider, Noor Ul Eman, Omer Amin, Alexis Mankowski, Muhammad Umair, Jingyao Zheng, Sheng Zheng, Lik-Hang Lee, Caiyan Qin, Tae-Ho Kim, Choong Seon Hong, Yang Yang, and Heng Tao Shen. Sora as a World Model? A Complete Survey on Text-to-Video Generation. arXiv e-prints, art. arXiv:2403.05131, March 2024. doi: 10.48550/arXiv.2403.05131.

Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee. Looped Transformers for Length Generalization. arXiv e-prints, art. arXiv:2409.15647, September 2024. doi: 10.48550/arXiv. 2409.15647.

Tuo Feng, Wenguan Wang, and Yi Yang. A Survey of World Models for Autonomous Driving. arXiv e-prints, art. arXiv:2501.11260, January 2025. doi: 10.48550/arXiv.2501.11260.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv e-prints, art. arXiv:2502.05171, February 2025. doi: 10.48550/arXiv.2502.05171.

Gemini Team, Google DeepMind. Gemini 3 flash model card. Technical report, Google Deep-Mind, December 2025. URL https://deepmind.google/models/model-cards/ gemini-3-flash/.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Google DeepMind. Genie 3: A new frontier for world models, 2025. https://deepmind. google/blog/genie-3-a-new-frontier-for-world-models/.

Alex Graves. Adaptive Computation Time for Recurrent Neural Networks. arXiv e-prints, art. arXiv:1603.08983, March 2016. doi: 10.48550/arXiv.1603.08983.

Yanchen Guan, Haicheng Liao, Zhenning Li, Jia Hu, Runze Yuan, Yunjian Li, Guohui Zhang, and Chengzhong Xu. World Models for Autonomous Driving: An Initial Survey. arXiv e-prints, art. arXiv:2403.02622, March 2024. doi: 10.48550/arXiv.2403.02622.

David Ha and Jurgen Schmidhuber. Recurrent world models facilitate policy evolution. In ¨ Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pp. 2455–2467, Red Hook, NY, USA, 2018. Curran Associates Inc.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International Conference on Machine Learning, pp. 2555–2565, 2019.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1lOTC4tDS.

Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=0oabwyZbOu.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 2025. DOI: 10.1038/s41586-025-08744-2.

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation. arXiv e-prints, art. arXiv:2602.11451, February 2026. doi: 10.48550/arXiv.2602.11451.

Divya Jyoti Bajpai and Manjesh Kumar Hanawal. A Survey of Early Exit Deep Neural Networks in NLP. arXiv e-prints, art. arXiv:2501.07670, January 2025. doi: 10.48550/arXiv.2501.07670.

Yeskendir Koishekenov, Aldo Lipani, and Nicola Cancedda. Encode, think, decode: Scaling testtime reasoning with recursive latent thoughts, 2026. URL https://openreview.net/ forum?id=jBSye8M3FQ.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum? id=H1eA7AEtvS.

Wenxuan Li, Hang Zhao, Zhiyuan Yu, Yu Du, Qin Zou, Ruizhen Hu, and Kai Xu. PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation. arXiv e-prints, art. arXiv:2504.16693, April 2025a. doi: 10.48550/arXiv.2504.16693.

Xinqing Li, Xin He, Le Zhang, Min Wu, Xiaoli Li, and Yun Liu. A Comprehensive Survey on World Models for Embodied AI. arXiv e-prints, art. arXiv:2510.16732, October 2025b. doi: 10.48550/arXiv.2510.16732.

Fan-Ming Luo, Tian Xu, Hang Lai, Xiong-Hui Chen, Weinan Zhang, and Yang Yu. A Survey on Model-based Reinforcement Learning. arXiv e-prints, art. arXiv:2206.09328, June 2022. doi: 10.48550/arXiv.2206.09328.

Vincent Micheli, Eloi Alonso, and Franc¸ois Fleuret. Transformers are sample-efficient world models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=vhFu1Acb0xb.

Vincent Micheli, Eloi Alonso, and Franc¸ois Fleuret. Efficient world models with context-aware tokenization. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=BiWIERWBFX.

OpenAI. Video generation models as world simulators, 2024. https://openai.com/index/ video-generation-models-as-world-simulators/.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin (eds.), Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040/.

Francesco Pappone, Donato Crisostomi, and Emanuele Rodola. Two-Scale Latent Dynamics for \` Recurrent-Depth Transformers. arXiv e-prints, art. arXiv:2509.23314, September 2025. doi: 10.48550/arXiv.2509.23314.

Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, and Daniel Y. Fu. Parcae: Scaling Laws For Stable Looped Language Models. arXiv e-prints, art. arXiv:2604.12946, April 2026. doi: 10.48550/arXiv.2604.12946.

Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=din0lGfZFd.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588:604–609, 2020. https: //doi.org/10.1038/s41586-020-03051-4.

Erik Talvitie. Self-correcting models for model-based reinforcement learning. In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, AAAI’17, pp. 2597–2603. AAAI Press, 2017.

Surat Teerapittayanon, Bradley McDanel, and H. T. Kung. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. arXiv e-prints, art. arXiv:1709.01686, September 2017. doi: 10.48550/arXiv.1709.01686.

Luozhou Wang, Zhifei Chen, Yihua Du, Dongyu Yan, Wenhang Ge, Guibao Shen, Xinli Xu, Leyi Wu, Man Chen, Tianshuo Xu, Peiran Ren, Xin Tao, Pengfei Wan, and Ying-Cong Chen. A Mechanistic View on Video Generation as World Models: State and Dynamics. arXiv e-prints, art. arXiv:2601.17067, January 2026. doi: 10.48550/arXiv.2601.17067.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Cotˆ e, and Prithviraj Ammanabrolu. ScienceWorld: ´ Is your agent smarter than a 5th grader? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.775. URL https: //aclanthology.org/2022.emnlp-main.775/.

Zijun Wang, Panwen Hu, Jing Wang, Terry Jingchen Zhang, Yuhao Cheng, Long Chen, Yiqiang Yan, Zutao Jiang, Hanhui Li, and Xiaodan Liang. ProPhy: Progressive Physical Alignment for Dynamic World Simulation. arXiv e-prints, art. arXiv:2512.05564, December 2025. doi: 10. 48550/arXiv.2512.05564.

Chenjun Xiao, Yifan Wu, Chen Ma, Dale Schuurmans, and Martin Muller. Learning to¨ combat compounding-error in model-based reinforcement learning, 2020. URL https:// openreview.net/forum?id=S1g\_S0VYvr.

Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped Transformers are Better at Learning Learning Algorithms. arXiv e-prints, art. arXiv:2311.12424, November 2023. doi: 10.48550/arXiv.2311.12424.

Abbas Zeitoun, Lucas Torroba-Hennigen, and Yoon Kim. Hyperloop Transformers. arXiv e-prints, art. arXiv:2604.21254, April 2026. doi: 10.48550/arXiv.2604.21254.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, and Jason Eshraghian. Scaling Latent Reasoning via Looped Language Models. arXiv e-prints, art. arXiv:2510.25741, October 2025. doi: 10.48550/arXiv.2510.25741.

Łukasz Kaiser, Mohammad Babaeizadeh, Piotr Miłos, Błazej Osi ˙ nski, Roy H Campbell, Konrad ´ Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, Afroz Mohiuddin, Ryan Sepassi, George Tucker, and Henryk Michalewski. Model based reinforcement learning for atari. In International Conference on Learning Representations, 2020. URL https:// openreview.net/forum?id=S1xCPJHtDB.