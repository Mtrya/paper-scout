# FORESIGHT WITHOUT SEEING: LATENT FUTURES FOR WORLD ACTION MODELS

Jiakai Huang<sup>1,2</sup> Zhongbo Wu<sup>1,2</sup> Zheng Zhang<sup>2,3</sup> Zihan Wang<sup>1∗</sup> Shan You<sup>2</sup> Tao Huang<sup>1†</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>ACE Robotics <sup>3</sup>Nanyang Technological University

## ABSTRACT

World Action Models (WAMs) couple future visual prediction with robot action generation, enabling policies to model how the physical world evolves during interaction. Existing WAMs differ primarily in how such predictive dynamics are exposed to the action pathway. Explicit-future WAMs provide direct access to predicted scene evolution through future generation, but incur substantial inference costs from iterative video denoising. In contrast, direct-policy WAMs skip future generation and efficiently predict actions from the current observation, but lack an explicit inference-time interface for exposing predictive dynamics to the Action DiT. To bridge this gap, we propose ForeWAM, a dynamics-conditioned directpolicy WAM that provides predictive context for action generation without decoding future videos. At its core, Future-KV performs a single Video DiT prefill over the clean current visual latent and stochastic future slots, and reuses the resulting layer-wise key-value states throughout action denoising. This allows the Action DiT to access predictive context formed by the video backbone without iterative future generation. We further introduce dynamics registers supervised by a frozen latent action teacher, encouraging the implicit future states to capture interactioninduced transitions, including object motion, contact changes, and task progress. Ground-truth future observations and the teacher are used only during training; deployment requires neither future observations nor the teacher and performs no future video generation. Without embodied robot data pretraining, the standard and accelerated variants of ForeWAM achieve average success rates of 96.7% and 96.9% on LIBERO, respectively. The standard variant further achieves 61.6% success on LIBERO-Plus. These results demonstrate that direct-policy WAMs can retain efficient action prediction while exposing predictive dynamics to the action pathway, without explicitly generating future observations.

## 1 INTRODUCTION

Vision-Language-Action (VLA) models offer a promising approach to Physical AI by predicting robot actions from visual observations and language instructions. However, they primarily learn reactive observation-to-action mappings without explicitly modeling how the physical world evolves through interaction. World Action Models (WAMs) have emerged as a new paradigm that couples future visual prediction with action generation, enabling policies to capture interaction-induced scene dynamics (Du et al., 2023; Hu et al., 2024; Sadigh & Song; Ye et al., 2026b; Zhu et al., 2025).

WAM designs differ primarily in how predictive visual context reaches the action pathway, as summarized in Figure 1. Figure 1(a) first generate future observations and then condition action prediction on them, whereas Figure 1(b) denoise future video and actions together (Du et al., 2023; Ye et al., 2026b; Bi et al., 2026). Both expose predicted scene changes to the action pathway, but iterative video denoising adds inference cost and generation errors may propagate into action prediction. Figure 1(c), represented by Fast-WAM, avoid future-video generation at inference while retaining future-video modeling during training (Yuan et al., 2026b). This improves efficiency, but leaves open how the Action DiT can access predictive, action-relevant context without a future rollout. Together, these designs expose a trade-off between predictive context and inference efficiency, raising a central question:

![](images/c9c7608b06866cdd044f3fa9d8aea1c44eecc70a9eaa7dc658c154d0149dc731.jpg)  
Figure 1: World Action Model paradigms. (a) Cascaded WAMs first generate future observations and then condition action prediction on them. (b) Joint WAMs generate future observations and actions within a unified generative process. (c) Direct-policy WAMs skip future rollout at inference and condition action prediction on a latent world representation extracted from the current observation. (d) Our ForeWAM retains direct action prediction while additionally exposing action-relevant predictive dynamics through hidden future-slot K/V states and dynamics registers. Hatched tokens denote noisy variables; future slots are stochastic internal states rather than observed future frames.

## How can a direct-policy WAM enable its Action DiT to access predictive dynamics without explicitly generatingfuture observations?

We address this question with ForeWAM, a Foresight-without-Seeing World Action Model that learns to act from latent futures without video rollouts. As shown in Figure 1(d), ForeWAM preserves the direct-policy inference structure while replacing explicit future-observation generation with a latent future interface exposed to the Action DiT. At its core is Future-KV, an implicit interface that transfers predictive context from the Video DiT to the Action DiT. Future-KV preserves the clean visual latent of the current observation, initializes unobserved future slots with noise, and processes them through a single Video DiT prefill. The resulting layer-wise key–value states are cached and reused throughout action denoising, allowing the action pathway to access predictive context over both the current observation and latent future slots without iteratively generating or decoding future video.

To further encourage these implicit future states to focus on scene transitions induced by robot interaction, we introduce dynamics registers supervised by a frozen LaWM latent-action teacher (Chen et al., 2026a). During training, the teacher extracts compact, non-executable latent-action representations from pairs of real visual observations before and after a transition. These representations supervise the dynamics registers to encode state-transition information, including object motion, contact changes, and task progress. Future-KV thus establishes a predictive information pathway from the Video DiT to the Action DiT, while latent-action supervision further strengthens the interaction-relevant dynamics represented along this pathway. Ground-truth future observations and the latent-action teacher are used only during training. At deployment, ForeWAM requires neither future observations nor the teacher and performs no future-video generation.

As a result, ForeWAM achieves competitive performance while substantially improving both training and inference efficiency, using only a compact Wan2.1-T2V-1.3B Video DiT and eliminating the need for embodied robot-data pretraining. To further accelerate inference, we apply OneDP (Wang et al., 2024) to distill the action-denoising process into a reduced-step schedule, yielding an accelerated variant termed ForeWAM-Flash. On our observed LIBERO-Plus subset, ForeWAM and ForeWAM-Flash achieve success rates of 61.6% and 58.2%, respectively, surpassing the reported Fast-WAM result of 51.5% by 10.1 and 6.7 percentage points. ForeWAM reduces the mean action-generation latency from 667 ms to 568 ms, a 14.8% reduction relative to Fast-WAM, while ForeWAM-Flash further lowers it to 220 ms, corresponding to a 67.0% reduction. Moreover, Fore-WAM uses approximately one-third of the policy parameters of Fast-WAM (2B versus 6B).

Our main contributions are summarized as follows:

• We identify a key interface problem in direct-policy WAMs: removing future-video generation improves efficiency but eliminates the explicit pathway through which predictive dynamics reach the Action DiT.

• We propose ForeWAM, combining Future-KV with latent-action-supervised dynamics registers. A single Video DiT prefill produces layer-wise K/V states for action denoising, while a frozen LaWM teacher encourages the registers to capture interaction-induced scene transitions.

• Without embodied robot-data pretraining, ForeWAM achieves up to 10.1 percentage points higher LIBERO-Plus success and 67.0% lower action-generation latency than the reported Fast-WAM configuration, while using approximately one-third of its policy parameters. Matched component comparisons further validate the proposed design.

## 2 RELATED WORK

Vision-language-action policies. VLA models map visual observations and language instructions to executable robot actions (Brohan et al., 2022; 2023; Kim et al., 2024; Team et al., 2024; Liu et al., 2025; Huang & Zheng, 2025; Yang et al., 2026), commonly by attaching an action decoder to a pretrained vision-language backbone (Intelligence et al., 2026; 2025; Zhao et al., 2025). Diffusion and flow objectives support multimodal continuous action generation (Chi et al., 2025; Lipman et al., 2022; Black et al., 2024), while large-scale robot pretraining can improve transfer across tasks and embodiments (Bjorck et al., 2025; Bu et al., 2025; Zheng et al., 2026). These methods establish strong direct policies, but do not by themselves provide an explicit action-facing interface through which predictive visual dynamics can be accessed during control.

World-action models. World Action Models (WAMs) augment direct action prediction with predictive world dynamics. Existing future-modeling WAMs broadly follow cascaded and joint paradigms. Cascaded approaches follow an imagine-then-act structure, predicting future observations or intermediate representations before extracting actions. Some methods explicitly generate future visual observations as intermediate plans (Du et al., 2023; 2024; Hu et al., 2024; Huang et al., 2024), whereas others use structured or compressed predictive representations, such as correspondences, point tracks, motion fields, masks, or distilled foresight (Bharadhwaj et al., 2024; Ko et al., 2024; Xu et al., 2024; Zhi et al., 2025; Lou et al., 2026; Yan et al., 2026). Joint WAMs instead co-model future states and actions within a shared architecture, allowing world and action representations to interact during generation. Autoregressive variants organize visual states and actions within a unified generative sequence (Cen et al., 2025b;a; Cheang et al., 2024; Wu et al., 2024), whereas diffusion- and flow-based variants jointly model world dynamics and action trajectories, with some recent approaches using latent or implicit representations for greater efficiency (Bi et al., 2026; Ye et al., 2026b; Zhu et al., 2025; Guo et al., 2024; Shen et al., 2026; Kim et al., 2026; Won et al., 2025; Yang et al., 2025; Chen et al., 2026b; Li et al., 2026; Yuan et al., 2026a; Team et al., 2026; Lyu et al., 2026). Although these approaches expose future scene evolution to action prediction, iterative future generation or tightly coupled world–action computation introduces substantial inference overhead. Direct-policy WAMs such as Fast-WAM avoid future generation by predicting actions from the current observation representation (Yuan et al., 2026b; Ye et al., 2026a). However, future dynamics are not explicitly exposed to the Action DiT under this direct-policy interface.

In contrast, our method retains direct-policy inference while exposing predictive dynamics to the Action DiT through a hidden future-slot K/V interface and dynamics registers supervised by a LaWM latent-action target (Chen et al., 2026a). The intended contribution is therefore the complemen tary composition of these two conditioning paths, rather than no-rollout inference or future-aware representation learning in isolation.

![](images/d4c9517fdca99be3d0271b4324b72f2266aa14db73f9dd6ccabe73713a816120.jpg)  
Figure 2: Dynamics-conditioned Action DiT. During training, demonstrated future frames supervise the video flow objective and a frozen latent-action encoder supplies the LaWM target. At inference, the future frames and teacher path are absent: the current latent is retained, future slots are initialized with noise, and one video prefill produces the per-layer K/V cache read during action denoising.

## 3 METHOD

Our goal is to expose predictive visual context to a direct action policy without decoding a future video at deployment. The proposed model combines a video diffusion transformer, a dedicated Action DiT, a hidden future-slot K/V cache, and latent-action-supervised dynamics registers (Figure 2). The cache preserves distributed visual context, whereas the registers provide a compact transitionoriented pathway. We first formalize the deployment interface, then describe token routing and the two conditioning paths, and finally specify the joint training objective.

## 3.1 PROBLEM FORMULATION

We consider language-conditioned chunk-level control. At control time, the policy receives a synchronized multi-camera observation $^ { O , }$ an instruction l, and a proprioceptive state $p .$ It predicts an executable action chunk $a _ { 1 : H } \in \mathbb { R } ^ { H \times d _ { \mathrm { a c t } } }$ of horizon H. A direct policy models

$$
p _ { \theta } ( a _ { 1 : H } \mid o , l , p ) .\tag{1}
$$

At inference, future observations, privileged simulator state, and teacher outputs are unavailable.

Let $u _ { 1 : T }$ denote a future visual trajectory or its latent representation. An explicit-future WAM may factorize action prediction conceptually as

$$
p ( a _ { 1 : H } \mid o , l , p ) = \int p _ { \phi } ( u _ { 1 : T } \mid o , l , p ) p _ { \theta } ( a _ { 1 : H } \mid o , l , p , u _ { 1 : T } ) \mathrm { d } u _ { 1 : T } .\tag{2}
$$

This factorization is commonly approximated by generating a future representation before or together with the action. It exposes temporal context, but couples control latency to future generation. A direct-policy WAM can instead retain a future-video training objective while omitting future rollout at inference (Yuan et al., 2026b). Our problem is to retain this direct policy while giving its Action DiT an explicit route to predictive visual context.

We distinguish the teacher-forced training target from the deployment-time interface. Let $z _ { 1 : T }$ denote the VAE encoding of the demonstrated video segment used by the video flow-matching loss.

During training, the video branch uses this target; at deployment, we construct a stochastic substrate $\widetilde { \underline { z } } ^ { \mathrm { F s u b } }$ without observing the future segment, and expose its hidden per-layer K/V state $\mathcal { H } _ { \mathrm { K V } }$ together with its dynamics-register slice $D _ { \theta }$ to the Action DiT. Given a current-frame latent $z _ { \mathrm { c u r } } ( o )$ the substrate is

$$
\widetilde { \boldsymbol z } _ { 1 : T } ^ { \mathrm { F s u b } } = \mathrm { c o n c a t } ( \boldsymbol { z } _ { \mathrm { c u r } } ( o ) , \boldsymbol { \epsilon } _ { F } ) , \qquad \boldsymbol { \epsilon } _ { F } \sim \mathcal { N } ( 0 , I )\tag{3}
$$

where the current latent occupies the first position and $\epsilon _ { F }$ fills the future slots. A single video prefill produces the dynamics-register states and their per-layer cache:

$$
\left( D _ { \theta } , \mathcal { H } _ { \mathrm { K V } } \right) = \mathrm { K V P r e f i l l } _ { \phi } \left( \tilde { z } _ { 1 : T } ^ { \mathrm { F s u b } } , l , p \right)\tag{4}
$$

where $D _ { \theta }$ denotes the dynamics-register slice of the prefetched video state. The resulting deployment-time policy is

$$
p _ { \theta } ( a _ { 1 : H } \mid o , l , p , D _ { \theta } ( o , l , p , \epsilon _ { F } ) , \mathcal { H } _ { \mathrm { K V } } ( o , l , p , \epsilon _ { F } ) )\tag{5}
$$

Equation 5 remains a direct action policy: it conditions on neither a ground-truth future nor a decoded video. The stochastic future slots are an internal conditioning substrate, and their usefulness is learned from the joint video–action objective rather than from future observations at deployment.

## 3.2 MODEL ARCHITECTURE

Design rationale. Direct-policy WAMs eliminate the iterative cost of generating future video, but this efficiency also leaves the action expert without an explicit, action-facing representation of how the scene may evolve. When the Action DiT is conditioned primarily on features of the current observation, it must infer both the present scene configuration and the consequences of candidate actions from the same visual context. This is particularly challenging for interaction-dependent behaviors, such as grasping, pushing, and placing, in which the appropriate action depends on the state transition induced by physical contact. We therefore seek to retain direct action prediction while providing the action expert with hidden features that encode task-relevant temporal structure, without access to future observations or decoded future video at inference time.

Our model addresses this challenge through two complementary context pathways. First, Future-KV provides distributed visual context over the current frame and future latent slots. The video backbone preserves the clean latent of the current frame, initializes the future slots with noise, and performs a single prefill. The resulting layer-wise keys and values are cached and made available to the Action DiT throughout action denoising. Because the cache is maintained in feature space, Future-KV exposes spatiotemporal context without requiring an iterative future-video rollout or pixel-space reconstruction.

Second, we apply latent-action (LA) supervision to a compact set of dynamics registers. A frozen LaWM teacher maps the demonstrated visual transition to a latent-action target, and a trainable projection head encourages the dynamics registers to match this target. This supervision biases the registers towards interaction-relevant changes, rather than requiring the action expert to recover such information solely from a generic future-video objective. The latent-action target serves as a non-executable transition cue and is used only during training.

The two pathways impose different inductive biases. Future-KV preserves rich, distributed visual information, whereas the LA-supervised dynamics registers provide a compact, action-oriented summary of transition structure. The Action DiT reads both pathways through the structured attention routing described below. At inference, actions are predicted directly from the current observation and these hidden representations; neither future observations nor the LaWM teacher is available, and no future video is decoded. Sec. 4.4 evaluates the corresponding component configurations, including a coverage-distinct base-policy reference without Future-KV or LA supervision. The observed complementarity is therefore a configuration-level result rather than a fully matched causal conclusion.

Token groups and routing. The reported configuration uses four token groups: current-frame tokens C, dynamics registers $D = \{ D _ { i } \} _ { i = 1 } ^ { N _ { D } }$ , future-slot tokens F, and action tokens A. Readability registers are disabled. The current observation is encoded into C, and $F$ occupies the latent positions initialized in Eq. 3. The Action DiT receives a noisy action chunk and predicts its flow. Both branches use the Wan2.1 text condition; the proprioceptive state is projected into the conditioning space as an additional context token.

![](images/3f10305d3f5e539389d54e40b23811719225fa7615ef313d5574197668826513.jpg)  
Figure 3: The structured mask routes current tokens C, dynamics registers D, future-slot tokens $F _ { \mathrm { { ; } } }$ and action tokens A.

The structured attention mask routes information as Figure 3.

Thus, future-slot tokens can integrate the current frame and dynamics registers, and action tokens can read the complete video sequence together with the registers. In the implementation, each action query concatenates the cached video keys and values with the keys and values computed from the current action tokens at that denoising step. The mask defines architectural routing; it is not by itself evidence of disentanglement or causal sufficiency.

Future-KV prefill. During training, the video branch receives demonstrated future latents and learns a future-latent flow objective, so its intermediate states receive a temporal learning signal. At inference, we preserve the clean current latent, place pure noise in future slots, and run the video branch once at the prefill level $\sigma = 1 . 0$ We cache the resulting key and value tensors at every layer and reuse them throughout action denoising. Future-KV therefore incurs one video prefill per action query instead of an iterative future-video rollout. The cached states are hidden conditioning features, not realized future frames; no future observation is decoded or fed back into the control loop. In the end-to-end configuration, gradients from the action loss remain connected to this prefill during training.

Latent-action-supervised dynamics registers. Generic video supervision need not preferentially retain interaction-relevant change. We therefore use a frozen LaWM latent-action encoder, trained as an inverse-dynamics component, to encode the demonstrated visual transition as a quantized latentaction target $z _ { \mathrm { L A } }$ during training. The mean-pooled dynamics registers pass through a trainable projection $g _ { \psi }$ into the teacher space. This target describes a visual transition; it is neither passed to the policy at deployment nor interpreted as a motor command. Executable actions remain the output of the Action DiT. The LA path is thus a training-time shaping signal for a compact register interface, not a second action decoder.

At inference, the policy encodes the current observation, builds the stochastic future substrate, prefills the cache once, and denoises the action chunk while reading $C , D ,$ , and $\mathcal { H } _ { \mathrm { K V } }$ . The teacher and observed future transition are absent from this computation.

## 3.3 TRAINING OBJECTIVE

We train the video and action branches with continuous flow matching (Lipman et al., 2022). For a target y, either a future video latent or an action chunk, we draw noise ϵ and a time variable t, and form

$$
y _ { t } = ( 1 - t ) y + t \epsilon\tag{6}
$$

The target velocity is $\epsilon - y ,$ , giving

$$
\mathcal { L } _ { \mathrm { F M } } ( y ) = \mathbb { E } _ { y , \epsilon , t } \left[ | | f _ { \theta } ( y _ { t } , t , o , l , p ) - ( \epsilon - y ) | | _ { 2 } ^ { 2 } \right]\tag{7}
$$

The video and action losses are

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v i d e o } } = \mathcal { L } _ { \mathrm { F M } } ( z _ { 1 : T } ) , \qquad \mathcal { L } _ { \mathrm { a c t i o n } } = \mathcal { L } _ { \mathrm { F M } } ( a _ { 1 : H } ) , } \end{array}\tag{8}
$$

where $z _ { 1 : T }$ is the demonstrated video-latent target and $a _ { 1 : H }$ is the demonstrated executable action chunk. The frozen teacher supplies a detached target $z _ { \mathrm { L A } }$ . With mean-pooled dynamics registers,

Table 1: Success rate (%) on the standard LIBERO suites, evaluated with 50 rollouts per task.
<table><tr><td>Method</td><td>Params</td><td>Embodied PT</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Overall</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>7B</td><td>Yes</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>π0 (Black et al., 2024)</td><td>3.3B</td><td>Yes</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.1</td></tr><tr><td>π0.5 (Intelligence et al., 2025)</td><td>3.3B</td><td>Yes</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>π0-Fast (Pertsch et al., 2025)</td><td>3.3B</td><td>Yes</td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>85.5</td></tr><tr><td>UniVLA (Bu et al., 2025)</td><td>7B</td><td>Yes</td><td>96.5</td><td>96.8</td><td>95.6</td><td>92.0</td><td>95.2</td></tr><tr><td>WorldVLA (Cen et al., 2025b)</td><td>7B</td><td>Yes</td><td>87.6</td><td>96.2</td><td>83.4</td><td>60.0</td><td>81.8</td></tr><tr><td>Fast-WAM (Yuan et al., 2026b)</td><td>6B</td><td>No</td><td>98.2</td><td>100.0</td><td>97.0</td><td>95.2</td><td>97.6</td></tr><tr><td>ForeWAM</td><td>2B</td><td>No</td><td>97.0</td><td>99.6</td><td>97.2</td><td>92.8</td><td>96.7</td></tr><tr><td>ForeWAM-Flash</td><td>2B</td><td>No</td><td>97.8</td><td>99.2</td><td>97.4</td><td>93.0</td><td>96.9</td></tr></table>

the distillation loss is

$$
\mathcal { L } _ { \mathrm { L A } } = \left. \boldsymbol { g } _ { \psi } \left( \frac { 1 } { N _ { D } } \sum _ { i = 1 } ^ { N _ { D } } D _ { i } \right) - \mathrm { s g } ( z _ { \mathrm { L A } } ) \right. _ { 2 } ^ { 2 }\tag{9}
$$

The stop-gradient applies to the teacher target only. In the reported end-to-end configuration, gradients from the action objective can flow through the video-to-action K/V interface.

The total objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { a c t i o n } } + \lambda _ { \mathrm { L A } } \mathcal { L } _ { \mathrm { L A } }\tag{10}
$$

The three terms train future latent prediction, executable action generation, and the transitionoriented register bottleneck, respectively. The objective does not establish that the registers are causally necessary or that the latent action is executable; those properties require targeted interventions.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmarks and evaluation protocol. We evaluate in-distribution control on the four standard LIBERO suites: Spatial, Object, Goal, and Long (Liu et al., 2023). We report task success rate over 50 rollouts per task. We evaluate out-of-distribution robustness on LIBERO-Plus (Fei et al., 2025), which perturbs the original tasks along seven dimensions: camera viewpoint, robot initial state, language instruction, lighting, background texture, sensor noise, and object layout.

## 4.2 MAIN RESULTS

Results on LIBERO. Both variants retain strong in-distribution performance without embodied pretraining (Table 1). Ours achieves 96.7% overall, ranging from 92.8% on Long to 99.6% on Object. Ours-Flash reaches 96.9% overall and differs from Ours by at most 0.8 percentage points on any suite. The two variants are 0.9 and 0.7 points below Fast-WAM, respectively. Thus, the accelerated variant preserves the standard-LIBERO performance of the full inference configuration.

Robustness on LIBERO-Plus. On the observed LIBERO-Plus subset (Table 2), Ours reaches 61.6% overall, with the highest rates under lighting and language perturbations and the lowest rate under robot-initial-state shifts. Compared with Fast-WAM, Ours is 10.1 points higher overall; the largest gains are on camera viewpoint (+46.1 points) and sensor noise (+21.1 points), with smaller gains on object layout, language, and background texture but lower success on robot-initial-state shifts and lighting. Ours-Flash reaches 58.2%, 3.4 points below Ours and 6.7 points above Fast-WAM overall, while remaining lower than Fast-WAM on robot-initial-state, language, lighting, and background shifts. Because external results come from different sources, these cross-method differ ences are descriptive rather than coverage-matched causal estimates.

## 4.3 INFERENCE EFFICIENCY

Action-denoising latency. Table 3 reports standalone action-generation inference latency measured on a single NVIDIA A800 GPU with 80 GB of memory. Ours lowers the 10-step latency from 667 ms for FastWAM to 568 ms. Distilling the Ours action branch from 10 to 2 denoising steps

Table 2: Observed success rate (%) on seven LIBERO-Plus perturbation categories.
<table><tr><td>Method</td><td>Camera Robot Language Light Background Noise Layout Overall</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>0.8</td><td>3.5</td><td>23.0</td><td>8.1</td><td>34.8</td><td>15.2</td><td>28.5</td><td>15.6</td></tr><tr><td>π0 (Black et al., 2024)</td><td>13.8</td><td>6.0</td><td>58.8</td><td>85.0</td><td>81.4</td><td>79.0</td><td>68.9</td><td>53.6</td></tr><tr><td>π0.5 (Intelligence et al., 2025)</td><td>75.4</td><td>77.5</td><td>85.6</td><td>96.9</td><td>94.6</td><td>89.7</td><td>85.7</td><td>85.7</td></tr><tr><td>π0-Fast (Pertsch et al., 2025)</td><td>65.1</td><td>21.6</td><td>61.0</td><td>73.2</td><td>73.2</td><td>74.4</td><td>68.8</td><td>61.6</td></tr><tr><td>UniVLA (Bu et al., 2025)</td><td>1.8</td><td>46.2</td><td>69.6</td><td>69.0</td><td>81.0</td><td>21.2</td><td>31.9</td><td>42.9</td></tr><tr><td>WorldVLA (Cen et al., 2025b)</td><td>0.1</td><td>27.9</td><td>41.6</td><td>43.7</td><td>17.1</td><td>10.9</td><td>38.0</td><td>25.0</td></tr><tr><td>Fast-WAM (Yuan et al., 2026b)</td><td>16.4</td><td>44.5</td><td>68.9</td><td>78.2</td><td>53.7</td><td>37.7</td><td>60.7</td><td>51.5</td></tr><tr><td>Ours</td><td>62.5</td><td>37.4</td><td>73.0</td><td>74.1</td><td>55.1</td><td>58.8</td><td>70.4</td><td>61.6</td></tr><tr><td>Ours-Flash</td><td>57.9</td><td>40.4</td><td>67.2</td><td>71.0</td><td>53.0</td><td>53.7</td><td>65.3</td><td>58.2</td></tr></table>

yields Ours-Flash, which reaches 220 ms while retaining the Future-KV and dynamics-register interface, a 61% reduction relative to Ours. These are standalone inference measurements, not average task-completion times or LIBERO-Plus rollout statistics.

## 4.4 ABLATION STUDY

Component comparison on LIBERO-Plus. Among the three coverage-matched configurations, each with 1,482 observed evaluations, Ours achieves the strongest overall result at 61.6% (Table 4). It exceeds Future-KV only (58.5%) and LA supervision only (58.0%) by 3.1 and 3.6 percentage points, respectively. The Base policy uses neither Future-KV nor LA supervision and reaches 53.6% over 10,027 observed evaluations under a different coverage profile. We therefore include it as a contextual reference rather than a matched estimate of the gain from adding both components. All configurations use no embodied pretraining, and the aggregate comparison does not by itself establish the causal contribution of either pathway.

Table 3: Standalone action-generation inference latency.
<table><tr><td>Method</td><td>Inference latency (ms) ↓</td></tr><tr><td>Fast-WAM</td><td>667</td></tr><tr><td>Ours</td><td>568</td></tr><tr><td>Ours-Flash</td><td>220</td></tr></table>

Table 4: Overall observed success rate (%) for the LIBERO-Plus ablation.
<table><tr><td>Configuration</td><td>Overall</td></tr><tr><td>Base policy</td><td>53.6</td></tr><tr><td>Future-KV only LA supervision only</td><td>58.5</td></tr><tr><td>Ours (both)</td><td>58.0 61.6</td></tr></table>

## 5 LIMITATIONS AND DISCUSSION

Our evaluation is currently limited to the standard LIBERO suites and LIBERO-Plus. Although these benchmarks cover a range of manipulation tasks and robustness perturbations, they do not fully capture the diversity of embodiments, interaction dynamics, visual conditions, and long-horizon behaviors encountered in broader robotic settings. It therefore remains unclear how well ForeWAM generalizes to different robot morphologies, task distributions, or real-world deployment scenarios. In particular, the robustness gains observed on LIBERO-Plus should be interpreted within the evaluated subset rather than as evidence of universal out-of-distribution generalization.

## 6 CONCLUSION

We introduced ForeWAM, a dynamics-conditioned direct-policy World Action Model that provides predictive context for action generation without explicit future-video rollout. ForeWAM combines Future-KV with latent-action-supervised dynamics registers, enabling the Action DiT to access distributed future context and compact transition cues. Future observations and the latent-action teacher are used only during training.

ForeWAM achieves 96.7% average success on LIBERO and 61.6% on LIBERO-Plus, while ForeWAM-Flash reaches 96.9% on LIBERO with substantially lower action-generation latency. Component comparisons further show that combining the two pathways performs better than ei ther alone. These results suggest that predictive dynamics can benefit direct action policies without being explicitly materialized as future observations.

## REFERENCES

Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In European Conference on Computer Vision, pp. 306–324. Springer, 2024.

Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 35101–35113, 2026.

Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan,˜ Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions, 2025. URL https://arxiv. org/abs/2505.06111, 2025.

Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Bohan Hou, Yuming Jiang, Jiayan Guo, Xin Li, et al. Rynnvla-002: A unified vision-language-action and world model. arXiv preprint arXiv:2511.17502, 2025a.

Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025b.

Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with webscale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, et al. Lawam: Latent world action models for efficient dynamicsaware robot policies. arXiv preprint arXiv:2606.15768, 2026a.

Jiayi Chen, Wenxuan Song, Pengxiang Ding, Ziyang Zhou, Han Zhao, Barrett Tang, Donglin Wang, and Haoang Li. Unified diffusion vla: Vision-language-action model via joint discrete denosing diffusion process. In International Conference on Learning Representations, volume 2026, pp. 139291–139311, 2026b.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal ofRobotics Research, 44(10-11):1684–1704, 2025.

Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023.

Yilun Du, Sherry Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, Leslie Kaelbling, et al. Video language planning. In International Conference on Learning Representations, volume 2024, pp. 31138–31155, 2024.

Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

Yanjiang Guo, Yucheng Hu, Jianke Zhang, Yen-Jen Wang, Xiaoyu Chen, Chaochao Lu, and Jianyu Chen. Prediction with action: Visual policy learning via joint denoising process. Advances in Neural Information Processing Systems, 37:112386–112410, 2024.

Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.

Jiakai Huang and Weiping Zheng. Size-aware contrastive imitation learning for languageconditioned multi-task robotic manipulation. 2025.

Shuaiyi Huang, Mara Levy, Zhenyu Jiang, Anima Anandkumar, Yuke Zhu, Linxi Fan, De-An Huang, and Abhinav Shrivastava. Ardup: Active region video diffusion for universal policies. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 8465– 8472. IEEE, 2024.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0. 5: a vision-languageaction model with open-world generalization, 2025. URL https://arxiv. org/abs/2504.16054, 1(2): 3, 2025.

Physical Intelligence, Bo Ai, Ali Amin, R Aniceto, A Balakrishna, G Balke, K Black, G Bokinsky, S Cao, T Charbonnier, et al. π0. 7: a steerable generalist robotic foundation model with emergent capabilities, 2026. URL https://arxiv. org/abs/2604.15483, 2026.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, and Joshua B Tenenbaum. Learning to act from actionless videos through dense correspondences. In International Conference on Learning Representations, volume 2024, pp. 40938–40958, 2024.

Runze Li, Hongyin Zhang, Junxi Jin, Qixin Zeng, Zifeng Zhuang, Yiqi Tang, Shangke Lyu, and Donglin Wang. World-value-action model: Implicit planning for vision-language-action systems. arXiv preprint arXiv:2604.14732, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pp. 29982–30009, 2025.

Yunfan Lou, Xiaowei Chi, Xiaojie Zhang, Zezhong Qian, Chengxuan Li, Rongyu Zhang, Yaoxu Lyu, Guoyu Song, Chuyao Fu, Haoxuan Xu, et al. Mask world model: Predicting what matters for robust robot policy learning. arXiv preprint arXiv:2604.19683, 2026.

Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, et al. Lda-1b: Scaling latent dynamics action model via universal embodied data ingestion. arXiv preprint arXiv:2602.12215, 2026.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Shuang Li Yihuai Gao Dorsa Sadigh and Shuran Song. Unified video action model.

Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators. Advances in neural information processing systems, 38:95597–95621, 2026.

Kairos Team, Fei Wang, Shan You, Qiming Zhang, Tao Huang, Zuoyi Fu, Zhisheng Zheng, Yunlong Xi, Feng Lv, Xiaoming Wu, Zeyu Liu, Cong Wan, Pu Li, Ruiqing Yang, Xiaoou Li, Wei Wang, Kangkang Zhu, Yuwei Zhang, Shi Fu, Zheng Zhang, Xiaoning Wu, Xuzeng Fan, Dacheng Tao, and Xiaogang Wang. Kairos: A regret-aware native world-action model stack for physical ai, 2026. URL https://arxiv.org/abs/2606.16533.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Zhendong Wang, Zhaoshuo Li, Ajay Mandlekar, Zhenjia Xu, Jiaojiao Fan, Yashraj Narang, Linxi Fan, Yuke Zhu, Yogesh Balaji, Mingyuan Zhou, et al. One-step diffusion policy: Fast visuomotor policies via diffusion distillation. arXiv preprint arXiv:2410.21257, 2024.

John Won, Kyungmin Lee, Huiwon Jang, Dongyoung Kim, and Jinwoo Shin. Dual-stream diffusion for world-model augmented vision-language-action model. arXiv preprint arXiv:2510.27607, 2025.

Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In International Conference on Learning Representations, volume 2024, pp. 10641–10662, 2024.

Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. arXiv preprint arXiv:2407.15208, 2024.

Haodong Yan, Zhide Zhong, Jiaguan Zhu, Junjie He, Weilin Yuan, Wenxuan Song, Xin Gong, Yingjie Cai, Guanyi Zhao, Xu Yan, et al. S-vam: Shortcut video-action model by self-distilling geometric and semantic foresight. arXiv preprint arXiv:2603.16195, 2026.

Liudi Yang, Yang Bai, George Eskandar, Fengyi Shen, Mohammad Altillawi, Dong Chen, Ziyuan Liu, and Abhinav Valada. Covar: Co-generation of video and action for robotic manipulation via multi-modal diffusion. arXiv preprint arXiv:2512.16023, 2025.

Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, et al. Abot-m0: Vla foundation model for robotic manipulation with action manifold learning. arXiv preprint arXiv:2602.11236, 2026.

Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. Gigaworld-policy: An efficient action-centered world–action model. arXiv preprint arXiv:2603.17240, 2026a.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026b.

Ge Yuan, Qiyuan Qiao, Jing Zhang, and Dong Xu. Adaworldpolicy: World-model-driven diffusion policy with online adaptive learning for robotic manipulation. arXiv preprint arXiv:2602.20057, 2026a.

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026b.

Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for visionlanguage-action models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1702–1713. IEEE, 2025.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-vla: Soft-prompted transformer as scalable crossembodiment vision-language-action model. In International Conference on Learning Representations, volume 2026, pp. 60580–60606, 2026.

Hongyan Zhi, Peihao Chen, Siyuan Zhou, Yubo Dong, Quanxi Wu, Lei Han, and Mingkui Tan. 3dflowaction: Learning cross-embodiment manipulation from 3d flow world model. arXiv preprint arXiv:2506.06199, 2025.

Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. arXiv preprint arXiv:2504.02792, 2025.

## A IMPLEMENTATION DETAILS

Architecture and inputs. We initialize the visual branch from Wan2.1-T2V-1.3B, retaining its video DiT, text encoder, and video VAE (Wan Team, 2025). We precompute instruction embeddings with the corresponding Wan2.1 text encoder. Both the video DiT and the action expert comprise 30 transformer blocks. The video branch uses hidden dimension $d _ { v } = 1 5 3 6$ , whereas the action expert uses $d _ { a } = 1 0 2 4$ and is initialized from a linearly interpolated Wan2.1 ActionDiT checkpoint. The action horizon is $H = 3 2$

Each training example contains 33 observation frames. A temporal ratio of 4 between the action and video streams maps each 32-step action chunk to 9 video frames. We concatenate the two synchronized camera views along the image width before VAE encoding, producing a $2 2 4 \times 4 4 8$ image composed of two $2 2 4 \times 2 2 4$ views. The policy additionally receives an 8-dimensional proprioceptive state. Each action is seven-dimensional, comprising a 6-DoF end-effector pose and one gripper-control dimension.

Optimization and inference. We train the video and action branches with continuous flow matching using a 1,000-timestep schedule and a shift of 5.0. The standard policy uses 10 action denoising steps at inference; Ours-Flash applies the accelerated variant of the same interface. We disable readability registers and use $N _ { D } ~ = ~ 1 6$ dynamics registers. A frozen LaWAM teacher supplies a 32-dimensional latent-action target. Gradients from the action objective propagate through the video-to-action interface without stop-gradient. At inference, we retain the current latent, initialize future slots with noise, and prefill the video K/V cache once at $\sigma = 1 . 0$ . This cache is reused across all action-denoising steps.

We optimize the joint objective with AdamW using a learning rate of $1 \times 1 0 ^ { - 4 }$ , weight decay of 0.01, cosine annealing, and gradient clipping at 1.0. None of the reported variants receives embodied pretraining before LIBERO training.