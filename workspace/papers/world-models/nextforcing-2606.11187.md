# Next Forcing: Causal World Modeling with Multi-Chunk Prediction

Gangwei Xu1,2 Qihang Zhang1,† Jiaming Zhou1,4 Xing Zhu1

Yujun Shen1 Xin Yang2,‡ Yinghao Xu3,1,‡

1Robbyant 2HUST 3HKUST 4HKUST (GZ)

†Project Lead ‡Corresponding Author

## Abstract

Autoregressive video generation has emerged as a powerful paradigm for World Action Models (WAMs). However, existing approaches suffer from slow training convergence and limited converged accuracy, particularly at high frame rates, as the training supervision is confined to the current chunk without explicit signals about future dynamics; they also suffer from slow inference due to iterative video denoising. In this paper, we present Next Forcing, a multi-chunk prediction (MCP) framework for causal world modeling that enables faster training, higher accuracy, and accelerated inference. Inspired by multi-token prediction in large language models, Next Forcing introduces an MCP training objective that augments the main model with lightweight auxiliary MCP modules to simultaneously denoise video chunks at multiple future temporal horizons (next1, next2, next3 chunks). These MCP modules form a causal chain across prediction depths, where intermediate features fused from multiple layers of the main model are leveraged to predict future dynamics, allowing near-future predictions to inform farther-future ones and providing dense multi-scale temporal supervision back to the main model. During training, the MCP modules significantly accelerate convergence and improve converged accuracy, especially at high frame rates: at 50 fps, Next Forcing achieves a 93.1% relative improvement over LingBot-VA at 5k training steps and 2.3× faster convergence, and establishes new state-of-the-art results on the RoboTwin benchmark (94.1/93.5% on Clean/Random). At inference, the MCP modules can be retained to predict the next video chunk in parallel with the current one, achieving 2× inference acceleration. Next Forcing also demonstrates significant improvements on PhyWorld, a benchmark evaluating adherence to physical laws in video generation, and over 50% FVD reduction on general video pretraining. Project website: https://gangweix.github.io/next-forcing/.

## 1 Introduction

Videos capture how the physical world evolves and how agents act within it, recording the dynamics of physical interactions at scale [24, 25, 26, 50, 9, 1, 2]. Building on this, World Action Models (WAMs) [36, 62] have recently emerged as a new paradigm for embodied AI, learning manipulation policies by jointly modeling future video and actions. The dominant training objective for WAMs is teacher-forced next-chunk denoising, where the model denoises the noisy current chunk conditioned on clean past chunks from ground-truth data. Despite the recent achievements of this paradigm [36, 62], teacher-forced next-chunk denoising remains an inefficient and shortcut-prone training signal for video world models. More precisely, predicting the next chunk is a fundamentally local task, which admits an appearance shortcut [21]: since adjacent chunks are visually highly similar, much of the denoising loss can be driven down by learning a near-identity map from the clean past chunk to the current one, with only small residual corrections. Such a shortcut is much easier to fit than the underlying dynamics and can absorb a large share of the gradient signal, leaving weak pressure on the model to capture the long-range temporal evolution that governs how the scene actually changes. We refer to this as myopic supervision, and its cost becomes particularly acute at high frame rates: at 50 fps, the appearance gap between adjacent chunks narrows to the point where the shortcut is nearly lossless, and standard teacher forcing converges significantly slower with lower final accuracy (Figure 1).

Training Steps  
![](images/6b46fe9d860a96b100b154522b5fc54d2dcba1815eb94bde336069715f29b145.jpg)

![](images/8d3ca4f0ebf72105436d9555cafff1124a0ed9c12747b46f86133843a4919775.jpg)

![](images/686d35ed250fdc68531e294a63eeeaa8f5dd841d26e467880af88ebf6e40eaa7.jpg)

![](images/447f4d53f881bd671584efbe39ec4f4ce68e3639176885216c124df7ea72d624.jpg)  
Figure 1: Task success rate (%) on RoboTwin across training steps. Next Forcing converges faster and reaches higher final accuracy than LingBot-VA at both 12 and 50 fps. The advantage is most pronounced at 50 fps: at 5k steps Next Forcing already outperforms LingBot-VA by 29.7 points on Random, and matches its 45k-step accuracy at only 20k steps, a 2.3× training speedup.

Our central insight is that turning the local single-chunk objective into a long-range multi-chunk objective forces the model to learn the latent dynamics governing temporal evolution rather than relying on appearance shortcuts. This idea has been validated in language modeling, where multitoken prediction (MTP) [18, 23] trains auxiliary modules to predict multiple future tokens, improving sample efficiency and enabling inference acceleration. Adapting it to video world models, however, is non-trivial: prediction targets are continuous video latents rather than discrete language tokens, generation proceeds through iterative denoising rather than single-step sampling, and temporal dependencies span multiple horizons of varying scale.

We present Next Forcing, a multi-chunk prediction (MCP) framework that cures myopic supervision for better & faster causal world modeling. Next Forcing augments the main model with a small set of auxiliary MCP modules that simultaneously predict video chunks at multiple future horizons (next1, next2, next3) and form a causal chain across prediction depths. To let the temporal supervision propagate deep into the main model’s representations, the MCP modules fuse intermediate features from multiple layers of the main model rather than the final layer alone. The MCP modules are trained with a higher timestep shift than the main model, forcing them to rely more heavily on the main model’s representations and tightening the coupling between the two.

Our work joins a growing family of “forcing” methods for autoregressive video generation. Existing methods vary either what context the model sees (as in teacher forcing [36, 62] and self forcing [28]) or how noise is scheduled (as in diffusion forcing [13]). Next Forcing is orthogonal to both: it varies what the model is asked to predict, and is therefore composable with all of the above.

We evaluate Next Forcing on RoboTwin [15] at multiple frame rates and on PhyWorld for physical law understanding. Next Forcing sets new state-of-the-art on RoboTwin (94.1/93.5% on Clean/Random), achieves 2.3× faster convergence at 50 fps (Figure 1), 2× inference acceleration, substantial gains on PhyWorld benchmark [31], and over 50% FVD reduction on general video pretraining with 3.5M in-house video clips, confirming its effectiveness beyond robot-specific data. Our main contributions are as follows:

• We propose Next Forcing, a multi-chunk prediction framework that overcomes the myopic supervision problem of autoregressive video world models.

• Next Forcing establishes new state-of-the-art results on the RoboTwin benchmark (94.1/93.5% on Clean/Random) with significantly faster training convergence. At 50 fps, Next Forcing achieves 2.3× faster convergence and higher converged accuracy.

• Beyond improving training, the MCP modules can be retained at inference to predict the next video chunk in parallel with the current one, accelerating generation.

![](images/6d005c47b5407ff32d9775d7cff619210af30e2d7a1325dfa2ced42b941108f3.jpg)  
Figure 2: Overview of Next Forcing. The main model denoises the current chunk, while chained MCP modules predict future chunks (next1, next2, . . .) using features from the main model, providing dense temporal supervision during training and enabling parallel chunk prediction at inference.

• We provide systematic ablations on the design choices that enable multi-chunk prediction to transfer from discrete tokens to continuous video latents (timestep shift, multi-layer feature fusion, and causal MCP chain), offering practical guidance for future work (Section 5).

We hope Next Forcing motivates further investigation of training objectives (beyond context construction and noise scheduling) in autoregressive video generation.

## 2 Related Work

## 2.1 World Action Models

Unlike Vision-Language-Action (VLA) models [8, 33, 69, 7, 46, 45, 22, 41, 42, 17, 67, 70, 66, 47, 58, 43, 61, 57, 71] that directly map visual observations and language instructions to actions, World Action Models (WAMs) [62, 36, 29, 52, 11, 60, 12, 68, 3] incorporate video prediction into robot policy learning, first predicting future visual dynamics and then decoding robot actions from the predicted frames. Several recent works have advanced the WAM paradigm along different axes. DreamZero [62] demonstrates that a 14B-parameter autoregressive video diffusion model, trained on diverse non-repetitive robot data, achieves zero-shot generalization to novel tasks and crossembodiment transfer. LingBot-VA [36] proposes a unified autoregressive framework with teacher forcing that jointly learns video prediction and action execution, achieving state-of-the-art bimanual manipulation through closed-loop control with persistent KV cache memory. Concurrently, Fast-WAM [65] finds that the primary benefit of video modeling lies in improving world representations during training. Other works explore joint video-action generation from complementary perspectives, including latent action representations[6, 16, 63, 51], joint video-action diffusion [72, 59], videoconditioned policy learning [19, 27, 5, 37, 54, 34], and video generation as policy [32, 44]. Our work is complementary to these efforts: rather than proposing a new WAM architecture or exploring data scaling, we focus on improving the training paradigm of autoregressive WAMs through multi-chunk prediction, which is applicable to existing WAM frameworks and also enables inference acceleration.

## 2.2 Autoregressive Video Generation

Autoregressive video generation [30, 64, 40, 14, 49] has become the core generation paradigm for action-conditioned world models and world action models [36, 62, 32]. The standard training approach is teacher forcing [36, 62], where the model learns to denoise the current chunk conditioned on clean ground-truth context. While stable, it creates a distribution gap between training (clean context) and inference (self-generated context), known as exposure bias [48, 4, 28]. Diffusion Forcing [13] mitigates this by training with independently sampled noise levels per frame, so that the model encounters noisy context during training. Self Forcing [28] takes a more direct approach by conditioning on self-generated histories during training, explicitly bridging the train-test distribution gap through a distribution-matching loss. These methods primarily address how context is constructed during training or inference. In contrast, our Next Forcing addresses a different and complementary aspect: what the model is trained to predict. By extending the prediction target from the current chunk to multiple next chunks, Next Forcing provides dense multi-scale temporal supervision that encourages trajectory-level temporal reasoning and avoids myopic supervision.

## 3 Preliminaries

## 3.1 Flow Matching

Flow matching [39] is a generative modeling framework that learns a velocity field to transport samples from a noise distribution to the data distribution. Given a clean sample $\mathbf { x } _ { \mathrm { 0 } }$ and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ , the noisy sample at time $t \in [ 0 , 1 ]$ is constructed via linear interpolation:

$$
\mathbf { x } _ { t } = ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { \epsilon } .\tag{1}
$$

A neural network $v _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } )$ is trained to predict the velocity $\mathbf { v } ^ { * } = \epsilon - \mathbf { x } _ { 0 }$ conditioned on context c, with the training objective:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , { \mathbf { x } _ { 0 } } , \epsilon } \left[ \left\| v _ { \theta } ( { \mathbf { x } } _ { t } , t , { \mathbf { c } } ) - ( \epsilon - { \mathbf { x } } _ { 0 } ) \right\| ^ { 2 } \right] .\tag{2}
$$

The timestep t is sampled with a timestep shift parameter s (see Appendix C for the detailed formulation). At inference, clean samples are generated by integrating the velocity field from $t = 1$ (pure noise) to $t = 0$ (clean data) using an ODE solver.

## 3.2 Autoregressive Video Generation with Teacher Forcing

We build upon the autoregressive video-action framework of LingBot-VA [36]. The model operates on video latents encoded by a pre-trained VAE and generates video frames in chunks of M frames. At each autoregressive step i, the model denoises the noisy current chunk $\mathbf { x } _ { t } ^ { ( i ) }$ conditioned on the clean previous chunks $\mathbf { x } _ { 0 } ^ { ( 1 : \bar { i } - 1 ) }$ from ground-truth data and a language instruction ℓ:

$$
\begin{array} { r } { v _ { \theta } \left( \mathbf { x } _ { t } ^ { ( i ) } , t , \left[ \mathbf { x } _ { 0 } ^ { ( 1 : i - 1 ) } , \ell \right] \right) \approx \epsilon ^ { ( i ) } - \mathbf { x } _ { 0 } ^ { ( i ) } . } \end{array}\tag{3}
$$

This teacher forcing formulation naturally aligns with closed-loop deployment, where ground-truth observations replace generated frames after each action execution. However, as discussed in Section 1, the model’s supervision is confined to the current chunk, leading to the myopic supervision problem.

## 4 Next Forcing

## 4.1 Overview

Next Forcing augments the standard teacher-forcing training objective with multi-chunk prediction (MCP), where auxiliary modules predict video chunks at multiple next temporal horizons in addition to the current chunk. Figure 2 illustrates the overall framework. During training, the main model denoises the current chunk as in standard teacher forcing, while three auxiliary MCP modules denoise the next1, next2, and next3 chunks through a causal chain, where each depth builds on the output of the previous one. At inference time, the MCP modules can either be discarded for zero-overhead deployment or retained for parallel chunk generation.

## 4.2 Multi-Chunk Prediction Objective

The key idea of Next Forcing is to extend the prediction target from the current chunk to multiple next chunks, providing explicit supervision about future dynamics. This encourages trajectory-level temporal reasoning during training and directly addresses the myopic supervision problem.

Temporal Chunk Shifting. Given a training video latent $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { C \times F \times H \times W }$ , where C , F , H , W denote the channel, number of chunks, height, and width dimensions respectively. Each chunk contains M frames, where M is randomly sampled from $\{ 1 , \ldots , M _ { \mathrm { m a x } } \}$ at each training step for robustness across temporal scales. Let i denote the current chunk index. For each MCP depth $k \in \{ 1 , 2 , 3 \}$ , we construct a temporally shifted target $\mathbf { x } _ { 0 } ^ { [ k ] }$ by advancing the video latent by k chunks:

$$
{ \bf x } _ { 0 } ^ { [ k ] } [ i ] = { \bf x } _ { 0 } [ \operatorname* { m i n } ( i + k , F ) ] ,\tag{4}
$$

where chunks beyond the sequence boundary are padded by replicating the last chunk. Each $\mathbf { x } _ { 0 } ^ { [ k ] }$ is the video shifted k chunks into the future.

Independent Noise Injection. Each shifted target is independently noised using the flow matching formulation with its own timestep and noise sample:

$$
\mathbf { x } _ { t _ { k } } ^ { [ k ] } = \left( 1 - t _ { k } \right) \mathbf { x } _ { 0 } ^ { [ k ] } + t _ { k } \epsilon _ { k } , \quad \epsilon _ { k } \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{5}
$$

where $t _ { k }$ is sampled with a dedicated timestep shift parameter $s _ { \mathrm { m c p } } .$ We set $s _ { \mathrm { m c p } } > s _ { \mathrm { m a i n } } ,$ which biases the MCP modules toward higher noise levels. The motivation is to strengthen the coupling between the MCP modules and the main model: at higher noise levels the MCP input carries less information about its own target, so the modules are forced to rely more heavily on the main model’s representations to denoise. This pushes the MCP loss gradients into the main model rather than letting the lightweight MCP modules absorb the supervision themselves.

MCP Position Encoding. To inform the MCP modules of their temporal offset, the chunk shift is incorporated into the rotary position embeddings (RoPE) [53]:

$$
\mathrm { R o P E } ( \mathbf { x } _ { 0 } ^ { [ k ] } [ i ] ) = \mathrm { R o P E } ( i + k ) ,\tag{6}
$$

so that each MCP module receives positional information for the future chunk it predicts.

## 4.3 Chained MCP Modules

The MCP modules are designed to encourage the main model to develop trajectory-level temporal representations. To achieve this, we fuse intermediate features from multiple layers of the main model rather than only the final output, since early layers tend to capture coarse structural patterns while later layers refine fine-grained details. This multi-layer fusion allows the MCP supervision signal to influence the main model’s representations at various depths through gradient backpropagation.

Multi-Layer Feature Fusion. During the main model’s forward pass through its 30 transformer layers, we collect hidden states at 4 intermediate layers {4, 12, 20, 30}, capturing representations at various depths of the backbone. Notably, the collected hidden states include both the noisy current latent and the clean history latent, allowing the fusion to capture both the denoising state and the ground-truth context. These multi-scale features are concatenated along the feature dimension and compressed through a two-layer MLP:

$$
\mathbf { h } _ { \mathrm { f u s e } } = \mathbf { M } \mathbf { L } \mathbf { P } ( [ \mathbf { h } _ { 4 } ; \mathbf { h } _ { 1 2 } ; \mathbf { h } _ { 2 0 } ; \mathbf { h } _ { 3 0 } ] ) \in \mathbb { R } ^ { B \times N \times d } ,\tag{7}
$$

where N is the number of latent tokens and d is the hidden dimension. During backpropagation, the MCP loss gradients flow through $\mathbf { h } _ { \mathrm { f u s e } }$ back into these intermediate layers, providing temporally-aware supervision to both the early and late stages of the main model.

Causal Chain Across Depths. The three MCP modules form a causal chain. For each depth k, the noisy shifted target $\mathbf { x } _ { t _ { k } } ^ { [ k ] }$ is embedded through the shared patch embedding layer and fused with the output from the previous depth:

$$
\begin{array} { r } { \mathbf { z } ^ { [ k ] } = W _ { k } \left[ \mathbf { h } _ { \mathrm { p r e v } } ^ { [ k - 1 ] } ; \mathrm { E m b e d } ( \mathbf { x } _ { t _ { k } } ^ { [ k ] } ) \right] , \quad W _ { k } \in \mathbb { R } ^ { d \times 2 d } , } \end{array}\tag{8}
$$

where $\mathbf { h } _ { \mathrm { p r e v } } ^ { [ 0 ] } = \mathbf { h } _ { \mathrm { f u s e } }$ . The fused representation is processed through 3 lightweight transformer blocks to predict the flow matching velocity $\hat { \mathbf { v } } ^ { [ k ] }$ . The output simultaneously serves as $\mathbf { h } _ { \mathrm { p r e v } } ^ { [ k ] }$ for the next depth, allowing depth-2 predictions to build on depth-1 features, and depth-3 on depth-2. The MCP modules share the same attention mask as the main model, requiring only a single mask construction per training step for efficient training (see Appendix A for details).

## 4.4 Joint Video-Action Architecture

Following LingBot-VA [36], Next Forcing jointly models video prediction and action decoding. The joint prediction is decomposed into two stages: (1) predicting future visual dynamics, and (2) decoding actions via inverse dynamics:

$$
\begin{array} { r } { \mathbf { x } _ { i + 1 } \sim p _ { \theta } ( \cdot \mid \mathbf { x } _ { \le i } , \mathbf { a } _ { < i } , \ell ) , \quad \mathbf { a } _ { i } \sim g _ { \psi } ( \cdot \mid \mathbf { x } _ { \le i + 1 } , \mathbf { a } _ { < i } , \ell ) , } \end{array}\tag{9}
$$

where $\mathbf { x } _ { \leq i }$ denotes the observation history up to chunk $i , \mathbf { a } _ { < i }$ denotes the action history, and ℓ is the language instruction. The video stream first predicts the next visual state $\mathbf x _ { i + 1 }$ , then the action stream decodes ${ \bf a } _ { i }$ by conditioning on the observations including the predicted future chunk $\mathbf x _ { i + 1 }$

Both streams are implemented within a unified Mixture-of-Transformers (MoT) [38] architecture, where a video stream and an action stream fuse through cross-modal attention at each transformer layer. The MCP modules introduced in Section 4.3 operate on the video stream. The improved video representations propagate to the action stream through the shared cross-modal attention, benefiting action decoding indirectly.

## 4.5 Training Objective

Main Loss. The main model is trained with two flow matching losses for the video and action streams. The video dynamics loss supervises the video stream to predict future visual states:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v i d e o } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \epsilon } \left[ \left| \left| v _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } ) - ( \epsilon - \mathbf { x } _ { 0 } ) \right| \right| ^ { 2 } \right] , } \end{array}\tag{10}
$$

and the action loss supervises the action stream via inverse dynamics:

$$
\mathcal { L } _ { \mathrm { a c t i o n } } = \mathbb { E } _ { t , { \mathbf { a } _ { 0 } } , \epsilon } \left[ \left\| v _ { \psi } ( { \mathbf { a } } _ { t } , t , { \mathbf { c } } _ { a } ) - ( \epsilon - { \mathbf { a } } _ { 0 } ) \right\| ^ { 2 } \right] ,\tag{11}
$$

where ${ \bf a } _ { 0 }$ denotes the ground-truth action sequence and $\mathbf { c } _ { a }$ includes the visual context from both current and future observations.

MCP Loss. Each MCP module is supervised with a flow matching loss on its temporally shifted target $\mathbf { x } _ { 0 } ^ { [ k ] } \left( \mathrm { E q . ~ } 4 \right)$ . For depth k:

$$
\begin{array} { r } { \mathcal { L } _ { k } ^ { \mathrm { M C P } } = \mathbb { E } _ { t _ { k } , \mathbf { x } _ { 0 } ^ { [ k ] } , \epsilon _ { k } } \left[ \left\| v _ { \theta } ^ { [ k ] } ( \mathbf { x } _ { t _ { k } } ^ { [ k ] } , t _ { k } , \mathbf { c } ) - ( \epsilon _ { k } - \mathbf { x } _ { 0 } ^ { [ k ] } ) \right\| ^ { 2 } \right] , } \end{array}\tag{12}
$$

where the last k padded chunks are excluded from the loss computation.

Total Loss. The complete training objective combines the main losses and the MCP losses:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { a c t i o n } } + \sum _ { k = 1 } ^ { 3 } w _ { k } \cdot \mathcal { L } _ { k } ^ { \mathrm { M C P } } .\tag{13}
$$

## 4.6 Inference

The MCP modules introduced in Section 4.3 serve a dual role: they are trained as a supervision mechanism but can also be repurposed at inference. We support two inference modes that share the same trained checkpoint and trade off cost against throughput.

Zero-Overhead Mode. All MCP modules (the fusion MLP, projection layers, and lightweight transformer blocks) are discarded. The main model operates exactly as in the standard autoregressive pipeline: at each step, it denoises the current chunk, with the same architecture, latency, and memory footprint as the baseline. All quality gains in this mode come from the enriched training signal that the MCP objective propagates back into the main model during training, with no test-time cost.

Parallel Chunk Generation Mode. The MCP modules can also be retained at inference to amortize the cost of one main-model forward pass over multiple chunks, in spirit similar to speculative and parallel decoding for LLMs [35, 10]. Concretely, in a single denoising trajectory the main model produces the current chunk while the depth-1 MCP module simultaneously produces the next chunk, and the MCP transformer blocks are an order of magnitude lighter than the main model, so adding them to the forward pass is nearly free. Each autoregressive step therefore advances the video by two chunks instead of one, yielding $2 \times$ inference acceleration. The depth-2 and depth-3 MCP modules are not used in this mode, as their predictions are superseded by the main model in the next step, but the same mechanism could be extended for higher speedups at the cost of accumulated drift; we leave this exploration to future work.

Table 1: Evaluation on the RoboTwin benchmark (average success rate % over 50 tasks). Clean: fixed initial configurations. Random: randomized object poses and scene layouts. Best results are bolded.
<table><tr><td></td><td>X-VLA</td><td>TO</td><td>TT0.5</td><td>Motus</td><td>Being-H0.7</td><td>Fast-WAM</td><td>LingBot-VA</td><td>Next Forcing</td></tr><tr><td>Clean</td><td>72.9</td><td>65.9</td><td>82.7</td><td>88.7</td><td>90.2</td><td>91.9</td><td>92.9</td><td>94.1</td></tr><tr><td>Random</td><td>72.8</td><td>58.4</td><td>76.8</td><td>87.0</td><td>89.6</td><td>91.8</td><td>91.5</td><td>93.5</td></tr></table>

Table 2: Ablation studies on RoboTwin Clean subset, trained for 20k steps on 16 GPUs. Default baseline: $s _ { \mathrm { m a i n } } { = } 5$ with noisy history augmentation. Default MCP: $s _ { \mathrm { m c p } } { = } 1 0$ , multi-layer fusion, weight init from main model, 3 transformer blocks per depth.
<table><tr><td colspan="2">Baseline (LingBot-VA [36]) Ablation</td><td colspan="2">MCP Module Ablation</td></tr><tr><td>Configuration</td><td>SR (%)</td><td>Configuration</td><td>SR(%)</td></tr><tr><td>Baseline (default)</td><td>75.6</td><td>Baseline + MCP (default)</td><td>85.8</td></tr><tr><td> $s _ { \mathrm { m a i n } } { = } 1$ </td><td>65.3</td><td> $s _ { \mathrm { m c p } } { = } 5$ </td><td>83.2</td></tr><tr><td> $s _ { \mathrm { m a i n } } { = } 1 0$ </td><td>78.4</td><td>w/o multi-layer fusion</td><td>83.6</td></tr><tr><td> $s _ { \mathrm { m a i n } } { = } 2 0$ </td><td>77.6</td><td>w/o weight init</td><td>83.8</td></tr><tr><td> $s _ { \mathrm { m a i n } } { = } 2 5$ </td><td>77.2</td><td>transformer blocks=1</td><td>86.5</td></tr><tr><td>w/o noisy history aug.</td><td>69.8</td><td>transformer blocks=5</td><td>85.0</td></tr></table>

The two modes use the same trained model, allowing deployments to choose freely between strict baseline parity and $2 \times$ throughput without retraining.

## 5 Experiments

## 5.1 Experimental Setup

Benchmarks. We evaluate Next Forcing on two benchmarks. RoboTwin [15] is a simulation benchmark containing 50 bimanual manipulation tasks that require coordinated dual-arm control. Each task is evaluated under two settings: Clean (fixed initial configurations) and Random (randomized object poses and scene layouts). We report the average task success rate over all 50 tasks. PhyWorld [31] evaluates whether video generation models can discover and adhere to physical laws (e.g., uniform linear motion, elastic collision, parabolic motion) from visual data alone.

Implementation Details. Next Forcing is built on the LingBot-VA framework with a Wan2.2 [56] Transformer backbone (30 layers). We keep all baseline settings identical to LingBot-VA. The main model uses a timestep shift of $s _ { \mathrm { m a i n } } = 5$ with noisy history augmentation applied at probability 0.5. For the MCP modules, we use 3 prediction depths (next1, next2, next3), each with 3 transformer blocks. The MCP timestep shift is set to $s _ { \mathrm { m c p } } = 1 0$ . MCP loss weights are $w _ { 1 } = 0 . 5 , w _ { 2 } = 0 . 2$ $w _ { 3 } = 0 . 1$ The chunk size M is randomly sampled from $\{ 1 , \ldots , \check { M } _ { \mathrm { m a x } } \}$ with $M _ { \mathrm { m a x } } = 4$ MCP module weights are initialized from the last few layers of the main model. Following LingBot-VA, we first pre-train on a large-scale multi-embodiment dataset and then post-train on RoboTwin. We adopt the same training setup as LingBot-VA [36], training on 2,500 Clean demonstrations (50 per task) combined with 25,000 Random demonstrations (500 per task) for up to 50k steps. We train on 64 GPUs and evaluate at multiple frame rates to study the effect of frame rate on training convergence and inference acceleration. For ablation studies, we train on 16 GPUs using only the 2,500 Clean demonstrations at 25 fps for 20k steps for efficient comparison.

![](images/baf09ce726e720767581b59664a71006654a8fe341fdf0497533059116d005cb.jpg)  
Figure 3: Qualitative comparison on PhyWorld. We show 5 frames (start, 3 intermediate, end) from ground truth (top), Next Forcing (middle), and Baseline (bottom). Blue boxes highlight regions where the baseline deviates from the ground-truth physical trajectory, while Next Forcing generates more physically consistent dynamics.

## 5.2 Main Results

## 5.2.1 Comparison with State-of-the-Art

Table 1 compares Next Forcing with state-of-the-art methods on the RoboTwin benchmark. Next Forcing achieves the highest success rate on both Clean (94.1%) and Random (93.5%) settings, outperforming all compared methods including both VLA and WAM approaches.

## 5.2.2 Training Convergence Analysis

Figure 1 presents a detailed comparison of training convergence between Next Forcing and LingBot-VA across two frame rates and two difficulty levels (full numerical results in Appendix B).

At 12 fps, Next Forcing converges roughly 2× faster than LingBot-VA, reaching 90.0% at 10k steps versus the baseline’s ∼20k steps, and achieves higher final accuracy (94.1/93.5% vs. 92.8/91.8% on Clean/Random). The advantage becomes dramatically more pronounced at 50 fps: at 5k steps, Next Forcing already achieves 70.2/61.6% versus LingBot-VA’s 45.5/31.9%, and maintains the lead through convergence. At 50k steps, Next Forcing reaches 91.8/90.5% versus 88.6/85.2%. The large gap at high frame rates confirms that multi-chunk prediction is particularly effective when adjacent frames are nearly identical, forcing the model to learn meaningful dynamics rather than relying on appearance shortcuts.

Why Does MCP Work Better at High Frame Rates? The dramatic improvement at 50 fps can be understood through supervision signal density. At high frame rates, adjacent chunks are nearly identical, making next-chunk denoising trivially solvable via appearance copying. Multi-chunk prediction breaks this shortcut: chunks 2 or 3 steps ahead exhibit substantial visual differences that can only be predicted by understanding the underlying physical dynamics, forcing the model to develop temporally-aware representations.

## 5.2.3 PhyWorld Benchmark

To evaluate whether multi-chunk prediction improves the understanding of physical dynamics beyond robot manipulation, we evaluate on PhyWorld [31], which tests video generation models’ ability to adhere to physical laws. Since PhyWorld is a pure video generation benchmark, we remove the action stream from both LingBot-VA and Next Forcing and evaluate only the video generation component. We report Frechet Video Distance (FVD [55], lower is better) and Abnormal Ratio (percentage of generated videos violating physical laws, lower is better).

As shown in Table 3 and Figure 3, Next Forcing improves both FVD and Abnormal Ratio over LingBot-VA, with larger gains on the out-of-template setting, suggesting that multi-chunk prediction encourages generalizable physical dynamics rather than template-specific memorization.

Table 3: Evaluation on PhyWorld benchmark for combinatorial generalization. We report FVD and Abnormal Ratio under out-of-template (OOT) and in-template (IT) settings.
<table><tr><td rowspan="2">Method</td><td colspan="2">FVD (↓)</td><td colspan="2">Abnormal Ratio (↓)</td></tr><tr><td>OOT</td><td>IT</td><td>OOT</td><td>IT</td></tr><tr><td>LingBot-VA</td><td>5.3</td><td>3.5</td><td>12%</td><td>3%</td></tr><tr><td>Next Forcing</td><td>4.7</td><td>3.2</td><td>8%</td><td>2%</td></tr></table>

![](images/8b5f0af9dcd25599b879c3d19a9256452899d37ac6218591280b5d626c5ac143.jpg)  
Figure 4: FVD (↓) on general video pretraining across training steps. Test Set 1 contains human activity videos, while Test Set 2 focuses on camera-driven scene dynamics. Next Forcing consistently achieves substantially lower FVD than LingBot-VA on both test sets throughout training.

## 5.2.4 Generality on Video Pretraining

To further validate that Next Forcing generalizes beyond robot-specific data, we conduct pretraining experiments on an in-house general video dataset comprising approximately 3.5M video clips of 5–10 seconds, predominantly featuring human activities. We remove the action stream, train on 32 GPUs, and evaluate pure video generation quality using FVD [55] (lower is better). We construct two held-out test sets of 1,024 samples each: Test Set 1 contains human activity videos, while Test Set 2 focuses on camera-driven scene dynamics.

As shown in Figure 4, Next Forcing consistently achieves substantially lower FVD than LingBot-VA throughout training on both test sets. At 50k steps, Next Forcing reduces FVD by 58% on Test Set 1 (94 vs.225) and 52% on Test Set 2 (97 vs.204). Notably, Next Forcing at only 10k steps already surpasses LingBot-VA at 50k steps on both sets, demonstrating significantly faster convergence in video quality. These results confirm that multi-chunk prediction provides effective temporal supervision for general video generation, validating the generality of Next Forcing beyond the robot manipulation domain.

## 5.3 Ablation Studies

We conduct systematic ablation studies on both the baseline and MCP module design choices. All ablations are trained and evaluated on the RoboTwin Clean subset for 20k steps (Table 2).

Baseline Ablation. We first ablate key design choices of the baseline LingBot-VA to understand their impact. The default baseline uses $s _ { \operatorname* { m a i n } } = 5$ with noisy history augmentation, achieving 75.6%. Removing noisy history augmentation causes a significant drop to 69.8%, as the model can shortcut the denoising task by directly copying from the clean context rather than learning meaningful dynamics. The timestep shift has a substantial effect: $s _ { \mathrm { m a i n } } = 1 \mathrm { y i e l d s } 6 5 . 3 \% , s _ { \mathrm { m a i n } } = 1 0$ achieves the best at 78.4%, while $s _ { \mathrm { m a i n } } = 2 0 / 2 5$ gives 77.6/77.2%, showing diminishing returns. Adding MCP improves performance from 75.6% to 85.8% (+10.2%).

MCP Module Ablation. From the default MCP configuration (Table 2), we ablate each design choice. Removing multi-layer fusion drops to 83.6%, confirming that fusing intermediate features enables deeper gradient propagation into the main model. Setting $s _ { \mathrm { m c p } } = 5$ (same as main) reduces to 83.2%, as the higher shift forces the MCP modules to rely more on the main model’s representations and strengthens their coupling. Without weight initialization yields 83.8%. For transformer blocks, 1 block achieves 86.5% and 5 blocks gives 85.0%, suggesting that lighter MCP modules result in tighter coupling and more effective supervision. Despite the slightly higher rate with 1 block, we default to 3 blocks as it produces fewer visual artifacts in the MCP-generated chunks, which is important for the parallel chunk generation mode at inference.

Table 4: Inference acceleration. Standard inference performs separate video denoising for each chunk. MCP-accelerated inference predicts the next video chunk in parallel via the MCP module.
<table><tr><td rowspan="2">Inference Mode</td><td colspan="2">12 fps</td><td colspan="2">25fps</td><td colspan="2">50fps</td></tr><tr><td>Clean</td><td>Random</td><td>Clean</td><td>Random</td><td>Clean</td><td>Random</td></tr><tr><td>Standard</td><td>94.1</td><td>93.5</td><td>92.6</td><td>91.4</td><td>91.8</td><td>90.5</td></tr><tr><td>MCP-accelerated (2×)</td><td>93.5</td><td>90.6</td><td>91.0</td><td>89.8</td><td>92.2</td><td>91.3</td></tr></table>

## 5.4 Inference Acceleration

As described in Section 4.6, the MCP modules can be retained at inference to predict the next video chunk in parallel with the current one, significantly reducing the video denoising cost. Table 4 shows that this acceleration maintains comparable accuracy to the standard pipeline across all frame rates.

## 6 Conclusion

We presented Next Forcing, a multi-chunk prediction framework that addresses the myopic supervision problem in autoregressive video world models. By training lightweight MCP modules to predict multiple future chunks alongside the main model, Next Forcing provides dense temporal supervision that forces the model to learn long-range dynamics rather than relying on appearance shortcuts. Next Forcing establishes new state-of-the-art results on RoboTwin (94.1/93.5% on Clean/Random), achieves 2.3× faster training convergence at 50 fps, enables inference acceleration, and demonstrates over 50% FVD reduction on general video pretraining, validating its generality beyond robot-specific data. The main limitation is that the MCP modules introduce extra training cost. We hope Next Forcing motivates further investigation of training objectives, beyond context construction and noise scheduling, as a key axis for improving autoregressive video generation.

## References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[2] Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video PreTraining (VPT): Learning to act by watching unlabeled online videos. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[3] BeingBeyond Team. Being-h0.7: A latent world-action model from egocentric videos. arXiv preprint arXiv:2605.00078, 2026.

[4] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2015.

[5] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. arXiv preprint arXiv:2409.16283, 2024.

[6] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025.

[7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023.

[9] Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments. In International Conference on Machine Learning (ICML), 2024.

[10] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. In International Conference on Machine Learning (ICML), 2024.

[11] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.

[12] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. GR-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

[13] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[14] Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, and Xiang Bai. Out of sight but not out of mind: Hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716, 2026.

[15] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[16] Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. In Proceedings of the IEEE/CVF international conference on computer vision, 2025.

[17] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robotics: Science and Systems (RSS), 2023.

[18] DeepSeek-AI. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[19] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML), 2024.

[21] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

[22] Gemini Robotics Team. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[23] Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In International Conference on Machine Learning (ICML), 2024.

[24] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[25] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

[26] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

[27] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. In International Conference on Machine Learning (ICML), 2025.

[28] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[29] Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, et al. Dreamgen: Unlocking generalization in robot learning through video world models. arXiv preprint arXiv:2505.12705, 2025.

[30] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.

[31] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024.

[32] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

[33] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024.

[34] Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, and Joshua B. Tenenbaum. Learning to act from actionless videos through dense correspondences. In International Conference on Learning Representations (ICLR), 2024.

[35] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning (ICML), 2023.

[36] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[37] Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, and Carl Vondrick. Dreamitate: Real-world visuomotor policy learning via video generation. arXiv preprint arXiv:2406.16862, 2024.

[38] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. Mixture-oftransformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

[39] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[40] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.

[41] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. RDT-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations (ICLR), 2025.

[42] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: Vision-language-action pretraining from large-scale human videos. arXiv preprint arXiv:2507.15597, 2025.

[43] NVIDIA, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[44] Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond VLAs. arXiv preprint arXiv:2512.15692, 2025.

[45] Physical Intelligence. π0.7: a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

[46] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[47] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. SpatialVLA: Exploring spatial representations for visual-language-action model. In Robotics: Science and Systems (RSS), 2025.

[48] Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. In International Conference on Learning Representations (ICLR), 2016.

[49] Shuhuai Ren, Shuming Ma, Xu Sun, and Furu Wei. Next block prediction: Video generation via semi-autoregressive modeling. arXiv preprint arXiv:2502.07737, 2025.

[50] Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. GAIA-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.

[51] Dominik Schmidt and Minqi Jiang. Learning to act without actions. In International Conference on Learning Representations (ICLR), 2024.

[52] Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators. arXiv preprint arXiv:2512.06963, 2025.

[53] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[54] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109, 2024.

[55] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

[56] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[57] Lirui Wang, Xinlei Chen, Jialiang Zhao, and Kaiming He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[58] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, and Jian Tang. TinyVLA: Towards fast, data-efficient vision-language-action models for robotic manipulation. arXiv preprint arXiv:2409.12514, 2024.

[59] John Won, Kyungmin Lee, Huiwon Jang, Dongyoung Kim, and Jinwoo Shin. Dualstream diffusion for world-model augmented vision-language-action model. arXiv preprint arXiv:2510.27607, 2025.

[60] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In International Conference on Learning Representations (ICLR), 2024.

[61] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, and Jianfeng Gao. Magma: A foundation model for multimodal AI agents. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[62] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Fan, and Joel Jang. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[63] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos. In International Conference on Learning Representations (ICLR), 2025.

[64] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025.

[65] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[66] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Tsung-Yi Lin, Gordon Wetzstein, Ming-Yu Liu, and Donglai Xiang. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[67] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023.

[68] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3D-VLA: A 3D vision-language-action generative world model. In International Conference on Machine Learning (ICML), 2024.

[69] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025.

[70] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024.

[71] Jiaming Zhou, Ke Ye, Jiayi Liu, Teli Ma, Zifan Wang, Ronghe Qiu, Kun-Yu Lin, Zhilin Zhao, and Junwei Liang. Exploring the limits of vision-language-action manipulation in cross-task generalization. Advances in Neural Information Processing Systems (NeurIPS), 38:139899– 139927, 2026.

[72] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. In Robotics: Science and Systems (RSS), 2025.

## Supplementary Material

This appendix provides additional details that complement the main paper. Appendix A describes the attention mask design shared by the main model and MCP modules. Appendix B presents the full numerical results for the training convergence experiments. Appendix C details the timestep shift formulation used for noise scheduling.

## A Attention Mask Details

![](images/65f0a8f3f60b82738ebf4915d5a15da5fa370028e22dc196d697e3c4547d5d9a.jpg)  
(a) Attention mask for main model  
w/o Attention

![](images/cad25a073aef7cadf07775e38e6420fc82c17e1d40978220db2dab32bb7c2292.jpg)  
(b) Attention mask for MCP modules

Figure 5: Attention mask for main model and MCP modules. Only video tokens are shown for clarity (action tokens omitted). Under teacher forcing, the sequence consists of noisy tokens (current chunk being denoised) and clean tokens (ground-truth context). Noisy tokens attend to all causally preceding clean tokens and to noisy tokens within the same chunk; clean tokens follow a standard causal pattern; clean tokens cannot attend to noisy tokens. The MCP modules share the same attention mask structure as the main model, requiring only a single mask construction per training step.

Figure 5 illustrates the attention mask shared by both the main model and MCP modules. For clarity, only video tokens are shown (action tokens omitted). The sequence consists of two groups: noisy tokens (the current chunk being denoised) and clean tokens (ground-truth context from previous chunks). The attention rules under teacher forcing are:

• Noisy → Clean: Each noisy token attends to all causally preceding clean tokens, excluding the clean token at the same chunk index. This prevents information leakage from the current chunk’s ground truth.

• Noisy → Noisy: Noisy tokens only attend to other noisy tokens within the same chunk (self-attention within the chunk being denoised).

• Clean → Clean: Clean tokens follow a standard causal pattern, attending to all clean tokens at the same or earlier chunk indices.

• Clean → Noisy: Not permitted. Clean context tokens cannot attend to noisy tokens.

A key design choice of Next Forcing is that the MCP modules reuse the same attention mask as the main model. Since both operate on sequences with the same structure (noisy target tokens + clean context tokens), the mask can be constructed once per training step and shared across the main model and all MCP depths, reducing training overhead.

## B Detailed Training Convergence Results

Table 5 provides the full training convergence data at 12 fps and 50 fps, complementing the convergence curves in Figure 1. Success rates are reported every 5k training steps from 5k to 50k.

Table 5: Task success rate (%) on RoboTwin across training steps at 12 fps and 50 fps. Next Forcing consistently outperforms LingBot-VA. At 12 fps, Next Forcing leads by ∼10 points at 5k steps and maintains a 1–2 point advantage at convergence. At 50 fps, the gap is substantially larger: Next Forcing leads by 24.7/29.7 points (Clean/Random) at 5k steps, and the advantage persists through convergence.
<table><tr><td>FPS</td><td>Setting</td><td>Method</td><td>5k</td><td>10k</td><td>15k</td><td>20k</td><td>25k</td><td>30k</td><td>35k</td><td>40k</td><td>45k</td><td>50k</td></tr><tr><td rowspan="3">12</td><td rowspan="2">Clean</td><td>LingBot-VA</td><td>74.0</td><td>85.2</td><td>87.8</td><td>90.8</td><td>92.3</td><td>91.3</td><td>92.8</td><td>92.9</td><td>93.1</td><td>92.8</td></tr><tr><td>Next Forcing</td><td>84.9</td><td>90.0</td><td>91.5</td><td>92.3</td><td>93.3</td><td>94.3</td><td>93.1</td><td>93.4</td><td>94.2</td><td>94.1</td></tr><tr><td rowspan="2">Random</td><td>LingBot-VA Next Forcing</td><td>73.5 80.6</td><td>82.2 85.4</td><td>85.0 85.8</td><td>88.3 90.5</td><td>88.9 89.8</td><td>89.4</td><td>89.3</td><td>91.2</td><td>91.4</td><td>91.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>91.5</td><td>89.6</td><td>91.5</td><td>91.6</td><td>93.5</td></tr><tr><td rowspan="3">50</td><td rowspan="2">Clean</td><td>LingBot-VA</td><td>45.5</td><td>64.8</td><td>69.6</td><td>78.5</td><td>79.0</td><td>81.2</td><td>82.4</td><td>83.8</td><td>87.4</td><td>88.6</td></tr><tr><td>Next Forcing</td><td>70.2</td><td>80.5</td><td>85.2</td><td>87.4</td><td>87.6</td><td>90.0</td><td>90.9</td><td>91.5</td><td>91.7</td><td>91.8</td></tr><tr><td rowspan="2">Random</td><td>LingBot-VA</td><td>31.9</td><td>54.7</td><td>59.8</td><td>69.4</td><td>70.7</td><td>75.6</td><td>79.2</td><td>80.4</td><td>84.5</td><td>85.2</td></tr><tr><td></td><td>Next Forcing</td><td>61.6</td><td>77.6</td><td>80.2</td><td>85.0</td><td>85.4</td><td>86.8</td><td>89.9</td><td>88.4</td><td>89.6</td><td>90.5</td></tr></table>

Several observations emerge from the detailed results. First, Next Forcing shows the largest absolute improvement in the early training stages (5k–10k steps), indicating that multi-chunk prediction provides a stronger learning signal from the start by preventing appearance shortcuts. Second, the improvement from MCP is notably frame-rate dependent. At 12 fps, adjacent chunks differ substantially in visual content, so the appearance shortcut is less severe and the baseline can still learn meaningful dynamics, resulting in a moderate improvement from MCP. At 50 fps, however, adjacent chunks are nearly identical, making it easy for the model to shortcut by simply copying the previous chunk. In this regime, the baseline struggles to learn beyond local copying, while MCP forces the model to capture long-range dynamics, leading to dramatically larger gains.

## C Timestep Shift Formulation

Here we provide the detailed formulation of the timestep shift mechanism used in both the main model and MCP modules.

Shifted Timestep Schedule. We first construct a schedule of T = 1000 uniformly spaced base values $\{ \sigma _ { i } \} _ { i = 0 } ^ { T - 1 }$ over $[ \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ] = [ 0 , 1 ]$ . These are then transformed by the shift parameter s [20]:

$$
\tilde { \sigma } _ { i } = \frac { s \cdot \sigma _ { i } } { 1 + ( s - 1 ) \cdot \sigma _ { i } } .\tag{14}
$$

This transformation is monotonic and maps $[ 0 , 1 ]  [ 0 , 1 ]$ , but redistributes the noise levels across the interval: a larger s pushes σ˜ toward higher values, concentrating training on noisier regimes. For the main model we use $s _ { \operatorname* { m a i n } } = 5 ;$ ; for the MCP modules we use $s _ { \mathrm { m c p } } = 1 0$

Timestep Sampling. During training, a timestep index is uniformly sampled: id ∼ Uniform $\{ 0 , 1 , \ldots , T - 1 \}$ and the corresponding shifted noise level $\tilde { \sigma } _ { \mathrm { i d } }$ is looked up from the pre-computed schedule. The noisy training sample is then constructed as:

$$
\begin{array} { r } { \mathbf { x } _ { \tilde { \sigma } } = \left( 1 - \tilde { \sigma } _ { \mathrm { i d } } \right) \mathbf { x } _ { 0 } + \tilde { \sigma } _ { \mathrm { i d } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) . } \end{array}\tag{15}
$$

Since the shift is already encoded in the schedule, uniform sampling over the shifted schedule effectively produces a non-uniform distribution over noise levels, biased toward higher noise for larger shift s.