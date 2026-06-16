# Geometric Action Model for Robot Policy Learning

Jisang Han∗ 1 Seonghu Jeon∗ 1 Jaewoo Jung1 Rene Zurbr ´ ugg ¨ 2,3 Honggyu An1 Tifanny Portela2,3 Marco Hutter2 Marc Pollefeys2 Seungryong Kim† 1 Sunghwan Hong† 2,3

1KAIST AI 2ETH Zurich 3ETH AI Center https://cvlab-kaist.github.io/Geometric-Action-Model

![](images/93cf7766c935f76d249b00ba8bc1021d94924969e8b33a593d0fe401fcaddb1a.jpg)  
(a) Overall Pipeline of Geomeric Action Model

![](images/bae8f111cb88a74c87d364055565d48e51337e6e3409af9782a90c8068b048ba.jpg)

![](images/569e59c61f1749ece88dc5cb74234b1076fe415f9e4dd6987fcab88deccfed71.jpg)  
(b) Quantitative Results  
Figure 1: GAM repurposes geometric foundation models into fast and robust robot policies. (a) GAM jointly predicts future 3D geometry and action chunks within a shared geometric backbone. (b) By leveraging explicit 3D geometric priors, GAM improves robustness and real-world performance while reducing latency and model size compared to existing baselines.

Abstract: Generalist robot policies must follow user instructions while reasoning about how objects, cameras, and robot actions interact in the 3D physical world. Recent vision-language-action models (VLAs) and video world-action models (WAMs) inherit strong semantic or temporal priors from large-scale foundation models, but they still operate primarily on 2D image frames or 2D-derived latent spaces, leaving implicit the 3D geometry required for contact-rich manipulation. We propose the Geometric Action Model (GAM), a language-conditioned manipulation policy that directly repurposes a pretrained geometric foundation model (GFM) as a shared substrate for perception, temporal prediction, and action decoding. GAM splits the GFM at an intermediate layer: the shallow layers serve as an observation encoder, and a causal future predictor inserted at the split layer forecasts future latent tokens conditioned on language, proprioception, and action history. The predicted future tokens are then routed through the remaining GFM blocks for feature propagation and decoding, allowing a single backbone to produce both future geometry and actions. This design equips the GFM with language-conditioned temporal world modeling through minimal architectural modification while preserving its rich geometric priors. Across a broad suite of simulation and real-robot manipulation benchmarks, GAM is more accurate, more robust, faster, and lighter than current foundation-model-scale baselines.

## 1 Introduction

A long-standing goal in robotics is to build generalist manipulation policies that can follow naturallanguage instructions and manipulate arbitrary objects across diverse scenes [1, 2, 3, 4]. To achieve this, a general manipulation model must not only recognize objects and parse instructions, but also reason about how the physical world will evolve under its own actions. This requires a unified understanding of language, visual appearance, scene geometry, robot state, and physical dynamics.

Recent progress has therefore increasingly relied on large-scale foundation models as pretrained substrates for robot policies. Vision-language-action models (VLAs) build on vision-language models whose representations are aligned with natural language, and learn to map visual and linguistic tokens to robot actions [5, 1, 2, 4, 6, 7]. Video world-action models (WAMs) instead leverage pretrained video generation models, using their world prediction priors to jointly model future frames and actions [3, 8]. While these approaches have shown impressive language-conditioned manipulation ability, they are fundamentally in 2D: 3D cues such as depth, scale, and occlusion are left implicit in monocular cues that the action decoder must disentangle on its own, leading to limited generalization across environment changes, especially in robot initial state and camera viewpoint [9].

To overcome this limitation, recent work incorporates 3D geometric information into robot policies. One line learns policies directly on explicit 3D observations such as raw point clouds [10, 11], demonstrating the value of geometry for generalization but typically requiring task-specific encoders trained from scratch. With the emergence of Geometric Foundation Models (GFMs) [12, 13, 14], some works transfer pretrained geometric priors into VLA policies, either distilling selected GFM features into the VLA backbone through representation alignment [15, 16, 17] or attaching a lightweight action head on top of a GFM’s final features [18, 19]. These improve spatial awareness, but use the GFM only as a static feature extractor: its multi-layer geometric structure is never repurposed as the policy’s own temporal and action-generating substrate.

In this work, we propose Geometric Action Model (GAM), which directly repurposes a GFM as a manipulation policy by using it as a shared medium for perception, future-state prediction, and action decoding. We show that by jointly predicting future action and geometry, geometric world dynamics can be inherently incorporated into robot policies.

Specifically, we split the pretrained GFM at an intermediate layer: the shallow layers serve as an observation encoder, while the remaining layers serve as a decoder block. Given the current visual observation, the observation encoder extracts spatially meaningful scene representations. To model how the world evolves over time, we insert a causal transformer at the intermediate layer that predicts future feature representations. This predictor is conditioned on task language, proprioception, and action history by introducing them as additional tokens at each timestep. The predicted future-state tokens are then processed by the remaining GFM decoder together with an action token, allowing the backbone to produce both future geometry and robot actions. An intuitive comparison between existing paradigms and our proposed framework is illustrated in Figure 2.

Across diverse simulation [20, 9, 21] and real-world benchmarks, GAM matches or exceeds the success rate of current foundation-model-scale baselines such as VLAs and WAMs while using substantially fewer trainable parameters and substantially faster inference (55× faster), and shows improved generalization to unseen scenarios. GAM especially achieves outstanding performance in camera perturbation settings (↑9.7%p), which requires geometric understanding priors.

Our contributions are as follows:

• We introduce Geometric Action Model (GAM), a manipulation policy that combines temporal world modeling, latent feature-space prediction, and a geometric foundation-model substrate in a single shared-backbone architecture.

• We show that action and geometry can be predicted in a shared token space: a single autoregressive sequence and a single backbone forward pass produce both action tokens and future-scene tokens, decoded by a lightweight action regression head and a depth head.

![](images/30996a02fe4f7bf9a78b5dffe01f7167010bb7b3b9d57445fdec3ed66c82de01.jpg)  
Figure 2: (a) Video WAMs [3, 8, 22] operate in 2D pixel space, predicting future latents and actions via video diffusion. (b) Geometry-aware VLAs [16, 17] predict actions using a VLA with passive feature distillation from an external GFM. (c) GAM (ours) unifies perception, geometry prediction, and action decoding by inserting a geometric world model inside a single GFM.

• We demonstrate that across diverse simulation and real-world manipulation benchmarks, GAM is simultaneously more accurate, more robust, faster, and lighter than current foundation-model-scale alternatives.

## 2 Related Work

Vision-Language-Action Models. Vision-language-action models (VLAs) adapt a pretrained vision-language model (VLM) into a robot policy. Early works [5, 1] autoregressively decode discrete action tokens from a finetuned VLM. This paradigm is extended through parallel decoding [2], flow-matching action experts [4, 6, 23], diffusion-based action heads [24], frequency-space action tokenizations [7], and compact open-source VLMs [25, 26]. This line of work extends large generalist foundation models for humanoid and embodied control [27, 28, 29, 30], to spatial-representation variants [31], and to refinements in training and post-training [32, 33, 34]. While these VLAs leverage strong open-vocabulary recognition to produce policies, they rely solely on 2D image priors and therefore lack 3D understanding. To address this, recent works [16, 17, 35] attempt to align intermediate VLA features with geometric foundation model. However, they do not fully exploit the geometric understanding of geometric foundation model backbone.

World Action Models for Robot Manipulation. World action models are trained to predict future states in order to learn a policy. One branch builds on large pretrained video generation models [36], fine-tuning them to jointly predict future frames and actions [3, 8, 22], arguing that the benefit of such co-training stems primarily from training-time supervision rather than test-time future imagination [22, 7]. A second branch keeps the visual backbone frozen and trains a separate temporal predictor in its feature space for planning via model-predictive control [37, 38], with related work using implicit future-latent alignment as an auxiliary training signal [39]. However, these video backbones encode only 2D image-space priors, which do not explicitly resolve depth, scale, or occlusion. GAM instead predicts in the latent space of a geometric foundation model that encodes rich 3D priors, while inheriting from both branches the paradigm of using future prediction as a training signal.

Geometric Foundation Models for Manipulation. Geometric foundation models (GFMs) [13, 12, 40, 41] infer dense 3D structures from multi-view images and have recently served as perceptual substrates for robot policies. Early approaches integrate GFMs with VLAs as frozen feature extractors via representation alignment [16, 17], direct encoder replacement [35, 18], or point-cloud fusion [42]. Moving beyond static extraction, recent concurrent works adopt GFMs for predictive control: Song et al. [43] utilizes the GFM to jointly predict actions and current-frame 3D properties, while Xu et al. [44] employs the GFM with a diffusion policy to co-denoise future action chunks and

![](images/9215b385172b466f6244bb790902f5baa63b7b4fa4c9dfc44a253b472a66184a.jpg)  
(a) Geometric Action Model Pipeline

![](images/816181747fb861fb1f848a731daa737f3ffcabc4def0e106c7db937ed279dff6.jpg)  
(b) Block-causal Attention Mask  
Figure 3: Main architecture of GAM.

3D latents. GAM departs from these paradigms in two ways: (1) action and future-scene predictions are jointly modeled within a single autoregressive token sequence rather than separated heads or diffusion processes, and (2) the GFM’s deep blocks are explicitly repurposed to decode predicted future tokens rather than merely processing observed ones.

## 3 Preliminaries: Geometric Foundation Models

A geometric foundation model (GFM) such as VGGT [13] or DA3 [12] is a feed-forward transformer that maps one or more RGB images to dense 3D geometry. Given a sequence of V views $\mathcal { Z } =$ $\{ I _ { v } \} _ { v = 1 } ^ { V }$ with $I _ { v } ~ \in ~ \mathbb { R } ^ { 3 \times h \times w }$ , a GFM produces per-pixel depth $D _ { v } \ \in \ \mathbb { R } ^ { h \times w }$ or 3D point maps $P _ { v } \in \mathbb { R } ^ { 3 \times h \times w }$ in a shared world frame, and per-view camera intrinsics and extrinsics $( K _ { v } , \xi _ { v } ) \in$ $\mathbb { R } ^ { 3 \times 3 } \times S E ( 3 )$ via auxiliary heads. Specifically, each view $I _ { v }$ is partitioned into P non-overlapping patches of size $p \times p$ and projected by a patch embedding into a per-view token sequence:

$$
\begin{array} { r } { \mathbf { z } _ { v } ^ { ( 0 ) } = \left[ \mathbf { c } _ { v } , \mathbf { x } _ { v } ^ { 1 } , \ldots , \mathbf { x } _ { v } ^ { P } \right] \in \mathbb { R } ^ { ( 1 + P ) \times d } , } \end{array}\tag{1}
$$

where $\mathbf { c } _ { v }$ is a per-view camera token, $\{ \mathbf { x } _ { v } ^ { j } \} _ { j = 1 } ^ { P }$ are patch tokens, and d is the hidden dimension. The full input sequence concatenated across views is ${ \bf Z } ^ { ( 0 ) } = [ { \bf z } _ { 1 } ^ { ( 0 ) } , \ldots , { \bf z } _ { V } ^ { ( 0 ) } ] \in \mathbb { R } ^ { V ( 1 + P ) \times d }$

These tokens are processed by a stack of M transformer blocks $\{ f ^ { ( m ) } \} _ { m = 1 } ^ { M }$ employing one of two attention modes. Frame-wise attention $f _ { \mathrm { f r a m e } } ^ { ( m ) }$ operates within each view tokens independently, attending over the $( 1 + P )$ tokens of a single image. Global attention $f _ { \mathrm { g l o b a l } } ^ { ( m ) }$ operates jointly over all $V ( 1 + P )$ tokens, fusing information across viewpoints. The hidden state at the m-th transformer block evolves as follows:

$$
\begin{array} { r } { \mathbf { Z } ^ { ( m ) } = f ^ { ( m ) } \big ( \mathbf { Z } ^ { ( m - 1 ) } \big ) , \quad f ^ { ( m ) } \in \{ f _ { \mathrm { f r a m e } } ^ { ( m ) } , \ f _ { \mathrm { g l o b a l } } ^ { ( m ) } \} . } \end{array}\tag{2}
$$

After the transformer forward pass, dense geometry is decoded from multiple intermediate hidden states $\mathbf { Z } ^ { ( m ^ { * } ) }$ , where $m ^ { * }$ is one of the selected layers in $\mathcal { S } = \{ m _ { 1 } , m _ { 2 } , m _ { 3 } , m _ { 4 } \}$ The extracted multi-layer hidden states are then fed into the DPT [45] head to estimate per-pixel geometry.

## 4 GAM: Geometric Action Model

Problem Formulation. We consider language-conditioned robot manipulation. At each timestep t, the robot receives a multi-view RGB observation $o _ { t } = \{ I _ { v , t } \} _ { v = 1 } ^ { V }$ from V fixed cameras, a proprioceptive state $s _ { t } \in \mathbb { R } ^ { d _ { s } }$ describing the robot’s joint configuration and end-effector pose, and a natural-language task instruction ℓ that is held constant throughout an episode. The policy πθ must produce an action chunk $\hat { \boldsymbol { a } } _ { t } \in \mathbb { R } ^ { C \times d _ { \boldsymbol { a } } }$ of length C, encoding the next C delta-pose or joint commands to be executed open-loop before the next observation is acquired. We learn a policy:

$$
\pi _ { \theta } : \ \left( \{ o _ { t - H + 1 } , \ldots , o _ { t } \} , \ \{ s _ { t - H + 1 } , \ldots , s _ { t } \} , \ \{ a _ { t - H } , \ldots , a _ { t - 1 } \} , \ \ell \right) \ \mapsto \ { \hat { a } } _ { t }\tag{3}
$$

from a dataset of N expert demonstrations $\boldsymbol { \mathcal { D } } \ = \ \{ ( \tau _ { i } , \ell _ { i } ) \} _ { i = 1 } ^ { N }$ , where each trajectory $\begin{array} { r l } { \tau _ { i } } & { { } = } \end{array}$ $( o _ { t } , s _ { t } , a _ { t } ) _ { t = 1 } ^ { T _ { i } }$ pairs a sequence of observations, states, and executed action chunks with a fixed instruction $\ell _ { i }$ . The policy conditions on a context window of H recent timesteps.

Overview. In the following sections, we explain how we transform a pretrained GFM into a language-conditioned world-action model. Our key idea is to split the GFM into two parts and insert a causal temporal predictor between them. This design lets GAM formulate future prediction directly inside the GFM latent space, enabling all predictive computation to be performed in the GFM’s geometric representation space.

Concretely, our framework operates in three sequential stages inside the GFM. First, the observation encoder (§4.1) repurposes the shallow layers of the GFM to extract latent geometric features from multi-view observations. Next, the causal future predictor (§4.2) operates at the split layer, where it combines these geometric features with language, proprioception, and action history to predict future latent tokens. Finally, during feature propagation and decoding (§4.3), the predicted future tokens are routed through the remaining deep GFM blocks to simultaneously decode future geometry and the final action chunk $\hat { a } _ { t }$ . Figure 3 (a) shows the overall architecture of our model.

## 4.1 Observation Encoder

We first reuse the shallow layers of the pretrained GFM as the observation encoder. Let $L _ { s }$ denote the split layer where the causal future predictor is inserted. The original GFM transformer stack is then decomposed into an encoder and a decoder:

$$
E _ { \le L _ { s } } = f ^ { ( L _ { s } ) } \circ \cdot \cdot \cdot \circ f ^ { ( 1 ) } , \qquad D _ { > L _ { s } } = f ^ { ( M ) } \circ \cdot \cdot \cdot \circ f ^ { ( L _ { s } + 1 ) } .\tag{4}
$$

Here, the choice of $L _ { s }$ is important because $L _ { s }$ must be deep enough to extract sufficiently rich visual features from the raw observations, yet shallower than the earliest layer used in the DPT head $L _ { s } < m _ { 1 }$ , so that predicted future states can be decoded into future geometries by the DPT heads.

After defining this split layer $L _ { s }$ , for each timestep $t ^ { \prime }$ in the context window, we tokenize the multiview RGB observation $o _ { t ^ { \prime } } = \{ I _ { v , t ^ { \prime } } \} _ { v = 1 } ^ { V }$ using the original GFM patch embedding. This produces the initial multi-view token sequence:

$$
\mathbf { Z } _ { t ^ { \prime } } ^ { ( 0 ) } = \left[ \mathbf { z } _ { 1 , t ^ { \prime } } ^ { ( 0 ) } , \ldots , \mathbf { z } _ { V , t ^ { \prime } } ^ { ( 0 ) } \right] \in \mathbb { R } ^ { V ( 1 + P ) \times d } ,\tag{5}
$$

where each view contributes one camera token and $P$ patch tokens. The observation encoder maps these tokens to the split-layer representation $\mathbf { Z } _ { t ^ { \prime } } ^ { ( L _ { s } ) }$ . By applying this encoding independently to each timestep in the context window, the output of this stage is a sequence of per-timestep geometric latent states $\{ \mathbf { \bar { Z } } _ { t - H + 1 } ^ { ( L _ { s } ) } , \dots , \mathbf { Z } _ { t } ^ { ( L _ { s } ) } \}$

## 4.2 Causal Future Predictor

After the observation encoder, GAM performs temporal prediction directly at the split layer $L _ { s }$ forecasting the next latent geometric state from current and past observations while conditioning on the task instruction, proprioception, and action history. To this end, we insert a causal future predictor $g _ { \phi }$ between the shallow encoder $E _ { \leq L _ { s } }$ and the deep decoder $D _ { > L _ { s } }$ . For each timestep $t ^ { \prime }$ in the context window, the encoder provides latent tokens $\mathbf { Z } _ { t ^ { \prime } } ^ { ( L _ { s } ) }$ , and we embed the proprioceptive state $s _ { t ^ { \prime } }$ and previous action $a _ { t ^ { \prime } - 1 }$ as tokens:

$$
\mathbf { p } _ { t ^ { \prime } } = \psi _ { s } ( s _ { t ^ { \prime } } ) , \qquad \mathbf { q } _ { t ^ { \prime } } = \psi _ { a } ( a _ { t ^ { \prime } - 1 } ) ,\tag{6}
$$

with $\psi _ { s } , \psi _ { a }$ lightweight projection layers, and the instruction ℓ into language tokens $\mathbf { L } _ { \ell }$ with a pretrained text encoder. We then form a per-timestep token block by concatenating the encoded GFM tokens with the proprioception and action-history tokens $\mathbf { U } _ { t ^ { \prime } } = [ \mathbf { p } _ { t ^ { \prime } } ; \mathbf { q } _ { t ^ { \prime } } ; \mathbf { Z } _ { t ^ { \prime } } ^ { ( L _ { s } ) } ]$ ]. The full input to the causal future predictor is $\mathbf { X } = [ \mathbf { L } _ { \ell } ; \mathbf { U } _ { t ^ { \prime } - H + 1 } ; \ldots ; \mathbf { U } _ { t ^ { \prime } } ]$

The combined sequence X is then processed through block-causal self-attention [40], ensuring the model incorporates past and present contexts without future leakage, as illustrated in Figure 3 (b). At the final layer of the predictor $g _ { \phi }$ , we read off the predictions from their respective sequence slots. Specifically, the hidden states corresponding to the geometry slots forecast the latent geometric tokens of the future frame, denoted as $\tilde { \mathbf { Z } } _ { t ^ { \prime } + 1 } ^ { ( L _ { s } ) }$ Concurrently, the hidden state of the designated previous-action slot is projected to produce a predicted next action token $\tilde { \mathbf { a } } _ { t ^ { \prime } } \in \mathbb { R } ^ { d }$ , in direct analogy to next-token prediction in a causal language model. By jointly forecasting action and geometric latents in this layer, we ensure that action tightly interacts with spatial representations.

This design of introducing a causal transformer predictor $g _ { \phi }$ allows the pretrained GFM to acquire language-conditioned temporal world modeling with minimal architectural modification. Only the inserted $g _ { \phi }$ needs to learn how to fuse language, proprioception, and action history with GFM latent features. The resulting predictions, $\widetilde { \mathbf { Z } } _ { t ^ { \prime } + 1 } ^ { ( L _ { s } ) }$ and $\tilde { \mathbf { a } } _ { t ^ { \prime } }$ , are then passed to the remaining GFM blocks for joint geometry and action decoding.

## 4.3 Feature Propagation and Action Decoding

Following the causal future predictor, the single action token $\tilde { \mathbf { a } } _ { t ^ { \prime } }$ is replicated V times to form a set of per-view action tokens $\left\{ \tilde { \mathbf { a } } _ { v , t ^ { \prime } } \right\} _ { v = 1 } ^ { V }$ , where $\tilde { \mathbf { a } } _ { v , t ^ { \prime } } = \tilde { \mathbf { a } } _ { t ^ { \prime } }$ . Concatenated with the geometry tokens, they are fed through the remaining GFM blocks $D _ { > L _ { s } }$ . We perform this feature propagation by appending each view’s corresponding action token $\tilde { \mathbf { a } } _ { v , t ^ { \prime } }$ directly to its geometry token sequence for each timestep:

$$
\tilde { \mathbf { Z } } _ { t ^ { \prime } + 1 } ^ { ( M ) } = \left( f ^ { ( M ) } \circ \cdot \cdot \cdot \circ f ^ { ( L _ { s } + 1 ) } \right) \left( \left[ \left[ \tilde { \mathbf { Z } } _ { 1 , t ^ { \prime } + 1 } ^ { ( L _ { s } ) } ; \tilde { \mathbf { a } } _ { 1 , t ^ { \prime } } \right] , \dots , \left[ \tilde { \mathbf { Z } } _ { V , t ^ { \prime } + 1 } ^ { ( L _ { s } ) } ; \tilde { \mathbf { a } } _ { V , t ^ { \prime } } \right] \right] \right) .\tag{7}
$$

To prevent future leakage, we extend the predictor’s causal mask strategy to the GFM’s remaining global attention layers $( f _ { \mathrm { g l o b a l } } ^ { ( m ) } )$

Finally, the propagated features are decoded by two heads. The lightweight action head $h _ { \mathrm { a c t } }$ aggregates action tokens over the context window to regress the executable action chunk $\hat { a } _ { t ^ { \prime } }$ , while the original GFM depth head $h _ { \mathrm { d e p t h } }$ decodes geometry tokens into action-aligned future depth maps. The GFM’s deep blocks, originally pretrained to decode shallow features into 3D geometry, are thus repurposed here as the decoder of the world model’s predicted future.

## 4.4 Training and Inference

The policy is trained end-to-end by minimizing a multi-task objective over action execution, world modeling, and geometric decoding:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { a c t } } \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { f e a t } } \mathcal { L } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } } ,\tag{8}
$$

where the λ factors balance each term and $\mathcal { H } = \{ t { - } H { + } 1 , \ldots , t \}$ is the context window. The action loss $\mathcal { L } _ { \mathrm { a c t } }$ is an $\ell _ { 1 }$ regression between the decoded action chunk $\hat { a } _ { t ^ { \prime } }$ and the expert action $\boldsymbol { a } _ { t ^ { \prime } }$ over all $t ^ { \prime } \in \mathcal { H }$ . The future-feature loss ${ \mathcal { L } } _ { \mathrm { f e a t } }$ anchors the predictor $g _ { \phi }$ to temporal geometric transitions by aligning predicted future tokens $\tilde { \mathbf { Z } } _ { t ^ { \prime } + 1 } ^ { ( L _ { s } ) }$ with the actual next frame $\mathbf { Z } _ { t ^ { \prime } + 1 } ^ { ( L _ { s } ) }$ extracted from frozen GFM:

$$
\mathcal { L } _ { \mathrm { f e a t } } = \sum _ { t ^ { \prime } \in \mathcal { H } } \left\| \tilde { \mathbf { Z } } _ { t ^ { \prime } + 1 } ^ { ( L _ { s } ) } - \mathbf { Z } _ { t ^ { \prime } + 1 } ^ { ( L _ { s } ) } \right\| _ { 1 } .\tag{9}
$$

The future-depth loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ grounds the predicted future in valid 3D structure by supervising the decoded depth $\tilde { D } _ { t ^ { \prime } + 1 } = h _ { \mathrm { d e p t h } } ( \tilde { \mathbf { Z } } _ { t ^ { \prime } + 1 } ^ { ( m ^ { * } ) } )$ using depth head $h _ { \mathrm { d e p t h } }$ against ground-truth future depth $D _ { t ^ { \prime } + 1 }$ , adopting the scale-invariant and gradient-matching penalties of the GFM [12, 14].

At inference, we maintain the historical context online with key-value caching, so each step processes only the new observation $o _ { t }$ and previous action $a _ { t - 1 }$ in a single feed-forward pass.

## 5 Experiments

## 5.1 Implementation Details

We use DA3-Giant [12] fine-tuned on Track4World [46] as the backbone. We insert a 12-layer causal predictor with width $d _ { g } = 1 0 2 4$ at layer $L _ { s } = 1 2$ , where alternating attention begins. For the task instruction, we extract language tokens using a frozen T5 encoder [47]. The policy uses a context horizon of H = 4 for pre-training and $H = 1$ for post-training and predicts $C = 8$ step action chunks in a $d _ { a } = 7$ end-effector action space from $d _ { s } = 7$ proprioceptive states. We pretrain GAM on 784K single-arm robot trajectories from RoboCasa365 [48], MimicGen [49], and OpenX-Embodiment [50], then post-train it on each benchmark. We optimize with AdamW using a constant learning rate, freeze layers before $L _ { s }$ and the depth head, and supervise depth with simulator ground truth. We set $\lambda _ { \mathrm { a c t } } = 3 , \lambda _ { \mathrm { f e a t } } = 1$ , and $\lambda _ { \mathrm { d e p t h } } = 3$ . Further details are provided in the appendix.

Table 1: Evaluation results on LIBERO and LIBERO-Plus. Success rates are reported in % with absolute performance drops from LIBERO to LIBERO-Plus shown in parentheses. Color highlights denote the top three performing methods within each column: first , second , and third .
<table><tr><td>Method</td><td>Size</td><td> ${ \mathrm { O r i g . } }$ </td><td>Plus</td><td>Cam.</td><td>Robot</td><td>Lang.</td><td>Light</td><td>BG</td><td>Noise</td><td>Layout</td></tr><tr><td colspan="9">VLAs</td><td></td></tr><tr><td>π0.5 [6]</td><td>3.3B</td><td>96.9</td><td>84.6(↓12.3)</td><td>72.0</td><td>76.6</td><td>86.5</td><td>96.1</td><td>95.2</td><td>86.7</td><td>86.0</td></tr><tr><td>OpenVLA-OFT [2]</td><td>7B</td><td>97.1</td><td>69.6 (↓27.5)</td><td>56.4</td><td>31.9</td><td>79.5</td><td>88.7</td><td>93.3</td><td>75.8</td><td>74.3</td></tr><tr><td>RIPT-VLA [33]</td><td>7B</td><td>97.5 91.3</td><td>68.4(129.1) 69.3 (122.0)</td><td>55.2</td><td>31.2</td><td>77.6</td><td>88.4</td><td>91.6</td><td>73.5</td><td>74.2 75.9</td></tr><tr><td>7To [40] π0-FAST[7]</td><td>3.3B</td><td>85.5</td><td>61.6(123.9)</td><td>61.0</td><td>40.8</td><td>63.7</td><td>89.3</td><td>84.1</td><td>80.1</td><td></td></tr><tr><td>UniVLA [32]</td><td>3.3B</td><td>95.2</td><td>42.9(152.3)</td><td>65.1</td><td>21.6</td><td>61.0</td><td>73.2</td><td>73.3</td><td>74.4</td><td>68.8</td></tr><tr><td>NORA [25]</td><td>8.5B 3B</td><td>87.9</td><td>39.0(148.9)</td><td>1.8</td><td>46.2</td><td>69.5</td><td>69.0</td><td>81.0</td><td>21.2</td><td>31.9</td></tr><tr><td>OpenVLA[1]</td><td></td><td>76.5</td><td>15.6(160.9)</td><td>2.2</td><td>37.0</td><td>65.1</td><td>45.7</td><td>58.6</td><td>12.8</td><td>62.1</td></tr><tr><td></td><td>7B</td><td></td><td></td><td>0.8</td><td>3.5</td><td>23.0</td><td>8.1</td><td>34.8</td><td>15.2</td><td>28.5</td></tr><tr><td colspan="9">WAMs</td><td></td></tr><tr><td>Cosmos-Policy [3]</td><td>2B</td><td>98.5</td><td>82.4(↓16.1)</td><td>73.4</td><td>63.3</td><td>89.3</td><td>98.9</td><td>83.5</td><td>89.3</td><td>84.0</td></tr><tr><td>Fast-WAM[51] WorldVLA [34]</td><td>6B</td><td>97.6</td><td>50.0(147.5)</td><td>16.4</td><td>44.5</td><td>68.9</td><td>78.2</td><td>53.7</td><td>37.7</td><td>60.7</td></tr><tr><td></td><td>7B</td><td>79.1</td><td>25.0(5.1)</td><td>0.1</td><td>27.9</td><td>41.6</td><td>43.7</td><td>17.1</td><td>11.0</td><td>38.0</td></tr><tr><td colspan="9"> Geometry-aware VLAs</td><td></td><td></td></tr><tr><td>π0.5 + Spatial Forcing [16]</td><td>3.3B</td><td>94.0</td><td>25.7(158.3)</td><td>0.1</td><td>0.3</td><td>26.8</td><td>66.0</td><td>45.9</td><td>0.1</td><td>59.8</td></tr><tr><td>π0.5 + ROCKET[17]</td><td>3.3B</td><td>95.3</td><td>47.5 (146.6)</td><td>30.9</td><td>75.6</td><td>29.3</td><td>69.2</td><td>47.0</td><td>25.4</td><td>62.0</td></tr><tr><td colspan="9">GAM</td><td></td></tr><tr><td>GAM (Ours)</td><td>1.4B</td><td>97.6</td><td>85.5 (↓12.1)</td><td>83.1</td><td>70.0</td><td>84.8</td><td>97.2</td><td>94.3</td><td>95.3</td><td>79.1</td></tr></table>

## 5.2 Experimental Setup

Simulation Benchmarks. We evaluate generalization across distinct axes using two simulation benchmarks. Specifically, we train our policy on LIBERO [20], a lifelong single-arm manipulation benchmark spanning diverse spatial layouts, object identities, and task goals. To rigorously assess out-of-distribution robustness, we then evaluate the trained models in a zero-shot manner on LIBERO-Plus [9], which introduces controlled environmental perturbations across dimensions such as camera viewpoint, lighting, and backgrounds. We report additional results in the appendix.

Real-Robot Setup. We train on four manipulation tasks (∼200 demonstrations each) using wristmounted and third-person cameras, adhering to the simulation protocol. Since ground-truth geometry is unavailable in the real world, target future depth maps are obtained as pseudo-labels directly from the pretrained backbone GFM. We evaluate robustness via 20 trials per task, divided equally between nominal setups and perturbed environments, specifically varying external camera positions. See the appendix for robot environment with full task and evaluation details.

Baselines. We compare GAM against representative baselines from three families discussed in §2: VLAs [1, 2, 4, 6, 7, 25, 32, 33], WAMs [3, 34, 51], and geometry-aware VLAs [16, 17]. For the real-robot setup, we compare against $\pi _ { 0 . 5 }$ [6] and Spatial Forcing [16]. For fairness, comparisons utilize a matched evaluation protocol, with performance numbers either re-evaluated using available checkpoints or taken directly from their respective published benchmarks.

![](images/9952b1b177b75df42b23690143714685b01c06003309292dfbe9a34cbfc30d70.jpg)  
Figure 4: Real-world robot tasks and results. Each task is evaluated under both in-domain (Light bar) and out-of-domain (Dark bar) settings. The illustration of each task is shown on the right.

## 5.3 Main Results

Simulation Results. As shown in Table 1, GAM achieves highly competitive success rates on the standard LIBERO benchmark, where performance is heavily saturated. Crucially, on the more challenging LIBERO-Plus benchmark, our model consistently outperforms competing baselines, demonstrating a remarkable improvement in the camera-perturbation setting (↑9.7%p). This gain highlights the advantage of our end-to-end integration of the GFM. While existing geometry-aware VLAs only partially exploit GFM representations, GAM embeds the GFM throughout its entire predictive pathway to yield a deeply geometry-aware policy.

Real-world Results. To examine whether the gains observed in simulation transfer to physical execution, we additionally evaluate GAM in a real-world setting. Figure 4 shows that GAM substantially outperforms all baselines. In particular, our model remains robust under out-of-domain conditions (the camera-perturbation setting) where other baselines struggle. These results demonstrate that GAM generalizes to the real-world domain and is robust under perturbations, owing to its thorough exploitation of the GFM when training the policy.

## 5.4 Ablation Study

Post-training Component Analysis. Table 2 summarizes ablation study of key post-training components on Object suite of LIBERO and LIBERO-Plus. Pretraining is crucial for robustness: omitting it mildly affects nominal LIBERO but severely degrades LIBERO-Plus. With a pretrained backbone, removing $L _ { \mathrm { d e p t h } }$ or $L _ { \mathrm { f e a t } }$ has minimal impact, suggesting geometric dynamics are already encoded. Notably, even without pretraining, these future-prediction losses provide strong geometric supervision and substantially improve robustness on LIBERO-Plus. Finally, the horizon ablation shows that H = 1 is sufficient and more robust than longer histories, consistent with prior observations that extended context can introduce spurious correlations [52, 53].

Split Layer $L _ { s }$ Selection. Table 3 evaluates the depth of future predictor by shifting the split layer $L _ { s }$ and re-initializing the predictor. We exclude future-depth loss in this experiment because it is not equally applicable to all split layers and could isolate the effect of ${ \mathcal { L } } _ { \mathrm { f e a t } }$ itself. Our default choice of $L _ { s } = 1 2$ achieves peak performance, validating it as the opti-

Table 2: Component ablation.
<table><tr><td>Pretrain</td><td> ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ </td><td> $\mathcal { L } _ { \mathrm { f e a t } }$ </td><td>H</td><td>Orig SR(%)</td><td>Plus SR (%)</td></tr><tr><td>【</td><td>【</td><td>√</td><td>1</td><td>99.6</td><td>89.7</td></tr><tr><td>v</td><td>：</td><td>√</td><td>2</td><td>97.2</td><td>84.4</td></tr><tr><td></td><td></td><td>√</td><td>4</td><td>98.2</td><td>85.1</td></tr><tr><td></td><td>×</td><td>√</td><td>1</td><td>98.4</td><td>89.0</td></tr><tr><td></td><td>×</td><td>X</td><td>1</td><td>98.6</td><td>89.5</td></tr><tr><td>√</td><td></td><td>×</td><td>1</td><td>99.6</td><td>89.7</td></tr><tr><td>xx</td><td></td><td>v√</td><td>1</td><td>98.4</td><td>73.4</td></tr><tr><td></td><td>X</td><td></td><td>1</td><td>95.2</td><td>66.5</td></tr><tr><td>xx</td><td></td><td>×</td><td>1</td><td>96.4</td><td>80.0</td></tr><tr><td></td><td>X</td><td>X</td><td>1</td><td>93.6</td><td>50.0</td></tr></table>

Table 3: Layer ablation.
<table><tr><td colspan="3">Split layer LgOrig. (%)Plus (%)</td></tr><tr><td>0</td><td>5.4</td><td>1.8</td></tr><tr><td>12</td><td>99.6</td><td>70.1</td></tr><tr><td>19</td><td>95.6</td><td>63.4</td></tr><tr><td>27</td><td>1.2</td><td>1.6</td></tr><tr><td>33</td><td>0.0</td><td>0.0</td></tr><tr><td>39</td><td>0.0</td><td>0.0</td></tr></table>

Table 4: Inference cost.
<table><tr><td>Method</td><td>Size</td><td>Time</td></tr><tr><td>OpenVLA-OFT[1]</td><td>7B</td><td>77.8ms</td></tr><tr><td>π0.5[6]</td><td>3.3B</td><td>29.2ms</td></tr><tr><td>Cosmos-Policy [3]</td><td>2B</td><td>382.4ms</td></tr><tr><td>GAM(Ours)</td><td>1.4B</td><td>6.9ms</td></tr></table>

mal seam between frame-wise and cross-view attention. While layer 19 remains competitive, inserting the predictor too early $( L _ { s } = 0 )$ or late $( L _ { s } \in \{ 2 7 , 3 3 , 3 9 \} )$ causes total performance collapse. This confirms that forecasted tokens require sufficient interaction through deep layers to properly integrate into the pretrained 3D geometric prior.

## 5.5 Analysis

Inference Speed and Model Size. As shown in Table 4, GAM achieves the lowest latency among all baselines, requiring only 6.9 ms (≈145 Hz) for a single feed-forward pass and running up to 55× faster than the diffusion-based Cosmos Policy.

All methods are benchmarked under the same setup, with further details provided in the appendix. By utilizing single-pass prediction, GAM avoids the multi-step denoising of diffusion policies, achieving low latency while matching prior accuracy and robustness with only 1.4B parameters.

![](images/4b0db3e83c039f699309dbadf1f6b0f1bfc80e8161c293d1cff5146d748ee0a8.jpg)

Robustness to Viewpoint and Scene Variation. Figure 5 further breaks down the camera-perturbation results by difficulty level of

Figure 5: Success rate vs. camera perturbation difficulty.

LIBERO-Plus [9]. GAM achieves consistently higher success rates than all baselines at every level, and the advantage remains clear even under the strongest perturbations.

## 6 Conclusion and Limitation

We introduced Geometric Action Model, which unifies geometry and action prediction with temporal world modeling inside a single shared GFM. By inserting a causal transformer between the GFM’s shallow and deep layers, GAM autoregressively decodes actions and future geometries, resolving the spatial ambiguities of traditional foundation-model substrates. Across extensive simulation and real-world benchmarks, GAM achieves superior accuracy, faster inference, and strong out-of-distribution robustness to environmental perturbations. The framework also has limitations. Its language reasoning and commonsense capabilities are bounded by the frozen text encoder; integrating a large language model or an external reasoning module is a natural next step.

## Acknowledgments

This work was supported under project ID a144 as part of the Swiss AI Initiative, through a grant from the ETH Domain and computational resources provided by the Swiss National Supercomputing Centre (CSCS) under the Alps infrastructure.

## Appendix

This appendix provides experimental details, results, and analyses that complement the main paper.

• Section A describes the training data, implementation details, simulation and real-world evaluation settings, baseline settings, and inference benchmark protocol.

• Section B presents additional simulation benchmark results on LIBERO, LIBERO-Plus, and RoboCasa.

• Section C provides additional ablations and analyses, including backbone variants, pretraining ablations, split-layer analysis, action-token attention, and robustness trends.

## A Experimental Settings and Reproducibility Details

GAM is trained in two stages, following standard practice for generalist robot policies [2, 4]. The first stage jointly trains the predictor, action head, and the GFM backbone end-to-end on a large mixture of single-arm robot data. The model was trained using 64 NVIDIA GH200 GPUs witch batch size of 1024, which takes approximately ˜96 hours. The second stage fine-tunes the entire model on each benchmark’s official training set before evaluation. The second stage on simulation benchmark was trained using 16 NVIDIA GH200 GPUs with batch size of 160, which takes ˜48 hours.

## A.1 Pre-training Details

We pre-train GAM on a weighted mixture of Open-X Embodiment [50] (OXE), MimicGen [49], and RoboCasa365 [48]. OXE provides broad real-robot coverage across multiple embodiments and manipulation domains, while MimicGen and RoboCasa365 provide simulation demonstrations with clean geometric supervision. The sampling ratios are 72%, 18%, and 10% for OXE, MimicGen, and RoboCasa365, respectively. For future-depth supervision, we use teacher pseudo-depth for OXE and re-rendered simulator depth for MimicGen and RoboCasa365. Figure 6 shows the dataset mixture used for pre-training.

For OXE, we use the subset whose actions can be mapped to our common control interface and exclude datasets that are incompatible with our action space.. For RoboCasa365, we use only the manipulation-task subset. Across all three sources, we keep the original task language provided by the datasets and do not synthesize additional instructions. The language encoder is kept frozen during pre-training.

All datasets are converted to a common observation and action format before training. Images are resized to 224×224, and the model uses two RGB views when available: an external view and a wrist view. Following Cosmos-policy [3] and $\pi _ { 0 . 5 }$ [6], standard image augmentations such as random cropping, rotation, and color jitter are applied during training and disabled during evaluation.

## A.2 Simulation Experiments Details

We adopt the LIBERO evaluation protocol established by OpenVLA [1] and OpenVLA-OFT [2]. Specifically, we evaluate on the four standard LIBERO task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long, each consisting of 10 tasks. Following OpenVLA-OFT, we train on filtered LIBERO demonstrations by removing unsuccessful episodes and filtering idle/no-op frames, i.e., training samples with near-zero actions. We fine-tune a separate policy for each LIBERO suite.

We report task execution success rate (SR, %) as our primary evaluation metric. For the original LIBERO benchmark, each task is evaluated over 50 randomized trials, resulting in 500 rollouts per suite. For LIBERO-Plus, we follow the official evaluation setting and use one rollout per perturbed task instance. All models are trained with a global batch size of 160 for up to 110k training steps, until convergence is achieved.

![](images/6367ac0ba4fab8f0db075c0d9a206d45b812dc5098432509dd97012438988e8d.jpg)  
Figure 6: Training Dataset Mixture. We illustrate the dataset mixture utilized during pretraining, detailing the relative proportions of each constituent dataset. The pie chart shows the high-level source mixture, and the bar chart shows the percentage of each constituent dataset relative to the entire training corpus.

Table 5: Hyperparameters for LIBERO, LIBERO-Plus and real-world experiments.
<table><tr><td>hyperparameter</td><td>value</td></tr><tr><td># GPUs learning rate (LR) total batch size input images input image size</td><td>8 ×NVIDIA GH200 5.16e-5 backbone; 5.16e-4 action head and predictor 160 1 external camera image,1 wrist-mounted camera image 224 x 224 px no (use single-step inputs)</td></tr></table>

For baseline comparison, we re-evaluate $\pi _ { 0 . 5 }$ and Cosmos-Policy under our evaluation setting using publicly available checkpoints. We also re-evaluate $\pi _ { 0 . 5 } +$ Spatial Forcing and $\pi _ { 0 . 5 } + \mathrm { R O C K E T }$ using reproduced checkpoints from ROCKET [17]. Results for the remaining baselines are taken from Fei et al. [9] and Zheng et al. [54].

![](images/f0a9a6fc42129f4888f59f952be040ee6ee9ad24e0db1bde5b0aa50f93ba1474.jpg)  
Figure 7: Real-world experiments environment setup and ID vs. OOD Camera setup.

![](images/8809859ecde1c50117adea564c6309440d533748629ad409635cd2135354a66e.jpg)  
Figure 8: Illustration of four real-world manipulation tasks.

## A.3 Real-World Experiments Details

Figure 7 illustrates the experimental environment used for our real-world evaluations. Our hardware setup includes a wrist-mounted ZED Camera and an external RealSense camera providing a thirdperson perspective.

For training and evaluation, we defined four distinct tasks: Pick and place, Stack milk and cube, Place pot and pan on cooktop, and Insert cube into covered pot. We collected teleoperated demonstrations for each task: 284, 202, 184, and 169 demonstrations, respectively. Figure 8 shows the text instructions and corresponding visual illustrations for each task. All tasks were jointly trained within a unified dataset. The hardware specifications and hyperparameters used to train GAM, alongside the baselines (π0.5 [6] and Spatial Forcing [16]), are detailed in Tables 5, 6, and 7, respectively. All baseline models were trained until convergence using the default hyperparameters from the paper.

During evaluation, we measured the success rate of each task across 20 trials. Specifically, 10 trials were conducted under a normal setup (ID), while the remaining 10 trials evaluated robustness under an out-of-distribution (OOD) setup with camera perturbation. The camera perturbation was introduced by applying a translation of 85 cm and a rotation of 45◦ to the external camera shown in Figure 7. The left side of Figure 7, we provide a visualization comparing ID and OOD environment settings. This perturbation setup was kept consistent across all evaluations.

## A.4 Real-World Experiments Baseline Training Details

For the baseline methods of real-world experiments, we follow the training recipes and default hyperparameters from the corresponding papers [16, 6], changing only the task data and camera streams to match our evaluation setup (Table 6 and 7). Both $\pi _ { 0 . 5 }$ [6] and Spatial Forcing [16] use the same two RGB inputs as GAM, one external camera and one wrist-mounted camera, resized to 224 × 224. We keep each baseline’s original inference protocol to preserve its intended deployment behavior. The main baseline-specific settings are summarized below; $\pi _ { 0 . 5 }$ uses flow-matching action decoding with 10 integration steps, while Spatial Forcing adopts its feature alignment loss recipe.

<table><tr><td>hyperparameter</td><td>value</td></tr><tr><td>#GPUs</td><td>8 x NVIDIA H100 GPU</td></tr><tr><td>learning rate (LR)</td><td>2.5e-5</td></tr><tr><td>total batch size</td><td>16</td></tr><tr><td>input images</td><td>1 external camera image, 1 wrist-mounted camera image</td></tr><tr><td>input image size</td><td>224 x 224 px</td></tr><tr><td>use observation history</td><td>no (use single-step inputs)</td></tr><tr><td>action chunk size</td><td>10 steps (predict i0, execute all 10 open-loop at test time)</td></tr><tr><td>use proprio (robot state)</td><td>yes</td></tr><tr><td># trainable parameters</td><td>853M total</td></tr><tr><td>image augmentations</td><td>90% random crops,color jitter: random_resized_crop=dict(scale=[0.9,0.9],</td></tr></table>

Table 6: $\pi _ { 0 . 5 }$ hyperparameters.
<table><tr><td>hyperparameter</td><td>value</td></tr><tr><td>#GPUs</td><td>8 x NVIDIA H100 GPU</td></tr><tr><td>learning rate (LR)</td><td>5e-5 16</td></tr><tr><td>total batch size</td><td></td></tr><tr><td>input images</td><td>1 external camera image, 1 wrist-mounted camera image</td></tr><tr><td>input image size</td><td>224 x 224 px</td></tr><tr><td>use observation history</td><td>no (use single-step inputs)</td></tr><tr><td>action chunk size</td><td>10 steps (predict i0,execute all 10 open-loop at test time)</td></tr><tr><td>use proprio (robot state)</td><td>yes</td></tr><tr><td># trainable parameters</td><td>3.3B total</td></tr><tr><td>diffusion sampling algorithm</td><td>flow matching</td></tr><tr><td>number of integration steps</td><td>10</td></tr><tr><td>image augmentations</td><td>90% random crops, color jitter: random_resized_crop=dict(scale=[0.9,0.9], ratio=[1.0,1.0])</td></tr></table>

Table 7: Spatial Forcing hyperparameters.

Table 8: Model-only inference latency on a single GH200 GPU.
<table><tr><td>Policy</td><td>Official PyTorch</td><td>bf16 precision</td><td>Torch Compile</td><td>CUDA Graphs</td><td>Model-only latency</td></tr><tr><td>GAM (Ours)</td><td></td><td></td><td></td><td></td><td>17.5ms</td></tr><tr><td>GAM (Ours)</td><td>√</td><td></td><td>厂</td><td></td><td>6.9 ms</td></tr><tr><td>pi0.5</td><td></td><td>√</td><td></td><td></td><td>29.2 ms</td></tr><tr><td>OpenVLA-OFT</td><td></td><td></td><td></td><td>xx</td><td>70.1ms</td></tr><tr><td>Cosmos Policy</td><td></td><td></td><td></td><td>×</td><td>382.4 ms</td></tr></table>

## A.5 Inference Latency Comparison Details

Table 8 reports the runtime configuration used for the inference-speed comparison in the main paper. All policies are evaluated on a single GH200 GPU with the same canonical observation input, bf16 precision, warmup and measurement protocol, and model-only latency metric, excluding model loading and input preprocessing.

In main paper, We use the official PyTorch inference path for each baseline. For $\pi _ { 0 . 5 }$ , whose original implementation is based on JAX, we use the official PyTorch implementation. To separate common compiler and runtime effects from deployment-specific execution, we report a matched setting in which all policies use Torch Compile and none uses CUDA Graphs. Under this setting, GAM requires 17.5 ms for a single feed-forward action prediction, compared to 29.2 ms for $\pi _ { 0 . 5 }$ , 70.1 ms for OpenVLA-OFT, and 382.4 ms for Cosmos Policy. GAM’s deployment setting further uses CUDA Graphs over its static single-pass inference path, reducing latency to 6.9 ms. This corresponds to approximately 145 Hz control and up to a 55.4× speedup over the diffusion-based Cosmos Policy.

## A.6 Model Size Breakdown

Table 9 details the parameter breakdown of the DA3-based GAM architecture. As reported, GAM uses a 1.4B-parameter model, making it substantially smaller than VLM-based and video-diffusionbased baselines such as π0.5, OpenVLA-OFT, and Cosmos-Policy. This compact size comes from repurposing a pretrained geometric backbone [12] as the shared substrate for perception, futuregeometry prediction, and action decoding, rather than attaching a large language model or videogeneration model as the policy backbone.

Of the full model, approximately 983.2M parameters are trainable. Most of these trainable parameters come from the later blocks of the ViT-Giant backbone, while the initial geometric layers and the DPT depth head remain frozen to preserve pretrained geometric structure. The Causal Future Predictor and lightweight action head are fully trainable and account for the remaining trainable parameters. This design allows GAM to adapt the geometric representation for control while keeping the overall model size below the larger foundation-model baselines in the main comparison.

Table 9: Parameter breakdown of the DA3-based GAM model.
<table><tr><td>Module</td><td>Parameters</td><td>Trainable?</td><td>Trainable parameters</td></tr><tr><td>backbone (ViT-Giant,40 blocks)</td><td>1136.5M</td><td>blocks 13-39 trainable; blocks 0-12 frozen</td><td>~765M</td></tr><tr><td>DPT head</td><td>50.1M</td><td>frozen</td><td>0</td></tr><tr><td>Causal Future Predictor</td><td>210.2M</td><td>trainable</td><td>210.2M</td></tr><tr><td>action head</td><td>8.0M</td><td>trainable</td><td>8.0M</td></tr><tr><td>total</td><td>~1404.8M</td><td>一</td><td>~983.2M</td></tr></table>

Table 11: Task-wise success rates on RoboCasa-Kitchen. We report per-task success rates and the overall average.
<table><tr><td>Task</td><td>SR</td><td>Task</td><td>SR</td><td>Task</td><td>SR</td><td>Task</td><td>SR</td></tr><tr><td>PnPCabToCounter</td><td>40.0%</td><td>PnPSinkToCounter</td><td>71.3%</td><td>CloseDoubleDoor</td><td>90.0%</td><td>TurnOffSinkFaucet</td><td>88.3%</td></tr><tr><td>PnPCounterToCab</td><td>50.3%</td><td>PnPStoveToCounter</td><td>56.3%</td><td>CloseSingleDoor</td><td>100.0%</td><td>TurnSinkSpout</td><td>84.7%</td></tr><tr><td>PnPCounterToMicrowave</td><td>29.3%</td><td>OpenSingleDoor</td><td>70.3%</td><td>OpenDrawer</td><td>92.3%</td><td>CoffeePressButton</td><td>79.0%</td></tr><tr><td>PnPCounterToSink</td><td>75.7%</td><td>OpenDoubleDoor</td><td>98.3%</td><td>CloseDrawer</td><td>99.3%</td><td>TurnOnMicrowave</td><td>90.7%</td></tr><tr><td>PnPCounterToStove</td><td>44.0%</td><td>TurnOnStove</td><td>74.3%</td><td>TurnOnSinkFaucet</td><td>89.7%</td><td>TurnOffMicrowave</td><td>94.3%</td></tr><tr><td>PnPMicrowaveToCounter</td><td>11.3%</td><td>TurnOffStove</td><td>30.0%</td><td>CoffeeServeMug</td><td>71.3%</td><td>CoffeeSetupMug</td><td>33.7%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Overall</td><td>69.4%</td></tr></table>

## B Additional Benchmark Results

## B.1 Additional Results on RoboCasa-kitchen

RoboCasa-Kitchen is a simulation benchmark derived from RoboCasa [21], which focuses on everyday manipulation in realistic and diverse kitchen environments. We evaluate on 24 kitchen manipulation tasks that cover pick-andplace, articulated-object interaction, appliance control, and coffee-related manipulation skills.

Because our base pre-training setup uses two camera views, whereas RoboCasa-Kitchen adopts a 3-view observation, we further train GAM from the base pre-training checkpoint for 3-view format. We also increase the action chunk size from 8 to 16 steps to better accommodate the longer-horizon nature of RoboCasa-Kitchen tasks. For the benchmark demonstrations, we re-extract depth for the 300 demonstrations per task and use only successful trajectories for training. As summarized in Table 10, We find that GAM

Table 10: Average success rates on RoboCasa Kitchen.
<table><tr><td>Method</td><td>Avg. SR (%)</td></tr><tr><td>GROOT-N1</td><td>49.6</td></tr><tr><td>+DreamGen</td><td>57.6</td></tr><tr><td>+DUST</td><td>58.5</td></tr><tr><td>UWM</td><td>60.8</td></tr><tr><td>TTO</td><td>62.5</td></tr><tr><td>GROOT-N1.5</td><td>64.1</td></tr><tr><td>+HAMLET</td><td>66.4</td></tr><tr><td>Video Policy</td><td>66.0</td></tr><tr><td>FLARE</td><td>66.4</td></tr><tr><td>Cosmos Policy</td><td>67.1</td></tr><tr><td>GAM (Ours)</td><td>69.4</td></tr></table>

outperforms existing baselines, including Cosmos Policy [3], on RoboCasa-Kitchen, We additionally provide the per-task breakdown of RoboCasa-Kitchen in Table 11.

## B.2 Additional LIBERO and LIBERO-Plus Results

Table 12 provides the suite-by-perturbation breakdown for LIBERO-Plus [9], expanding the aggregate LIBERO and LIBERO-Plus results reported in the main paper. Figure 9 further breaks down LIBERO-Plus performance by perturbation difficulty level, showing how GAM behaves as each perturbation becomes more severe. Table 13 reports task-wise success rates on the original LIBERO [20] benchmark.

(d) Long  
Table 12: LIBERO-Plus robustness results across four task suites. Success rates are reported in %. ”Orig.” denotes the results of LIBERO benchmark without any perturbation.
<table><tr><td colspan="10">(a)Spatar</td></tr><tr><td>Method</td><td>Orig.Cam. Robot Lang.Light</td><td></td><td></td><td></td><td></td><td></td><td></td><td>BG Noise Layout Total</td><td></td></tr><tr><td>GAM(Ours)</td><td>98.6</td><td>91.5</td><td>79.4</td><td></td><td>95.6100.0</td><td>99.6</td><td>96.6</td><td>94.3</td><td>93.4</td></tr><tr><td>π0.5</td><td>98.8</td><td>76.6</td><td>86.3</td><td>96.4</td><td>97.9</td><td>99.6</td><td>92.0</td><td>98.2</td><td>92.0</td></tr><tr><td>Cosmos-policy</td><td>98.1</td><td>83.5</td><td>59.7</td><td>96.4</td><td>99.7</td><td>85.3</td><td>91.7</td><td>95.1</td><td>87.3</td></tr><tr><td>Fast-WAM</td><td>98.2</td><td>14.4</td><td>44.0</td><td>69.5</td><td>87.3</td><td>69.8</td><td>35.0</td><td>60.5</td><td>54.4</td></tr><tr><td>OpenVLA-OFT</td><td>97.6</td><td>88.3</td><td>40.0</td><td>80.5</td><td>98.3</td><td>97.3</td><td>96.3</td><td>93.9</td><td>84.0</td></tr><tr><td>RIPT-VLA</td><td>97.5</td><td>85.4</td><td>38.0</td><td>99.7</td><td>99.7</td><td>100.0</td><td>92.0</td><td>92.3</td><td>85.8</td></tr><tr><td></td><td>96.8</td><td>70.7</td><td>49.1</td><td>67.9</td><td>92.8</td><td>95.0</td><td>87.7</td><td>94.0</td><td>78.6</td></tr><tr><td></td><td>96.8</td><td>17.8</td><td>6.6</td><td>58.8</td><td>89.7</td><td>90.7</td><td>90.9</td><td>89.1</td><td>60.7</td></tr><tr><td>πo-Fast</td><td>96.4</td><td>87.2</td><td>26.9</td><td>84.2</td><td>37.0</td><td>97.7</td><td>93.2</td><td>95.5</td><td>74.4</td></tr><tr><td>UniVLA</td><td>96.5</td><td>1.1</td><td>52.6</td><td>83.9</td><td>96.6</td><td>90.7</td><td>15.7</td><td>69.5</td><td>55.5</td></tr><tr><td>NORA</td><td>92.2</td><td>4.3</td><td>50.9</td><td>63.8</td><td>66.8</td><td>65.5</td><td>12.5</td><td>84.6</td><td>47.6</td></tr><tr><td>WorldVLA</td><td>85.6</td><td>0.0</td><td>44.3</td><td>46.3</td><td>65.1</td><td>19.8</td><td>11.7</td><td>46.1</td><td>32.5</td></tr><tr><td>OpenVLA π0.5+ROCKET</td><td>84.7</td><td>.0.0</td><td>3.7</td><td>27.7</td><td>12.3</td><td>50.4</td><td>12.0</td><td>40.7</td><td>19.4</td></tr><tr><td></td><td>96.4</td><td>14.9</td><td>18.0</td><td>35.4</td><td>50.7</td><td>45.0</td><td>13.7</td><td></td><td>48.631.5</td></tr></table>

<table><tr><td>Method</td><td>Orig.Cam.Robot Lang.Light BG Noise Layout Total</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GAM (Ours)</td><td>99.691.4</td><td></td><td></td><td></td><td></td><td>76.9100.0100.0 99.7</td><td>99.2</td><td>79.7</td><td>90.6</td></tr><tr><td>π0.5</td><td>98.2</td><td>86.4</td><td>71.9</td><td>91.0</td><td></td><td>99.099.2</td><td>96.2</td><td>91.1</td><td>89.9</td></tr><tr><td>Cosmos-policy</td><td>100.0</td><td>88.6</td><td>61.6</td><td>94.1</td><td>99.7</td><td>96.8</td><td>97.4</td><td>86.4</td><td>88.3</td></tr><tr><td>Fast-WAM</td><td>100.0</td><td>25.3</td><td>64.1</td><td>96.3</td><td>97.9</td><td>77.8</td><td>63.7</td><td>73.0</td><td>71.2</td></tr><tr><td>OpenVLA-OFT</td><td>98.4</td><td>38.9</td><td>25.4</td><td>99.0</td><td>73.7</td><td>97.6</td><td>72.3</td><td>71.8</td><td>66.5</td></tr><tr><td>RIPT-VLA</td><td>97.5</td><td>37.9</td><td>26.4</td><td>80.8</td><td>85.9</td><td>99.2</td><td>68.0</td><td>70.1</td><td>64.3</td></tr><tr><td>元</td><td>98.8</td><td>80.1</td><td>31.9</td><td>75.4</td><td>94.3</td><td>85.9</td><td>87.9</td><td>76.2</td><td>74.7</td></tr><tr><td></td><td>98.8</td><td>22.2</td><td>8.3</td><td>70.0</td><td></td><td>90.9 91.1</td><td>87.0</td><td>76.2</td><td>61.4</td></tr><tr><td>π-Fast</td><td>96.8</td><td>72.0</td><td>27.6</td><td>71.5</td><td></td><td>71.0 95.2</td><td>93.1</td><td>84.5</td><td>72.7</td></tr><tr><td>UniVLA</td><td>96.8</td><td>0.0</td><td>42.2</td><td>86.9</td><td></td><td>25.6 81.5</td><td>10.4</td><td>27.3</td><td>36.7</td></tr><tr><td>NORA</td><td>95.4</td><td>0.5</td><td>28.4</td><td>76.4</td><td></td><td>25.354.8</td><td>5.7</td><td>55.8</td><td>34.4</td></tr><tr><td>WorldVLA</td><td>89.0</td><td>0.0</td><td>26.4</td><td>57.2</td><td></td><td>20.517.3</td><td>18.0</td><td>53.6</td><td>28.6</td></tr><tr><td>OpenVLA</td><td>88.4</td><td>0.5</td><td>4.5</td><td>21.0</td><td></td><td>1.045.2</td><td>11.4</td><td>22.4</td><td>14.0</td></tr><tr><td>π0.5+ROCKET</td><td>98.8</td><td>41.4</td><td>27.1</td><td>44.6</td><td></td><td>91.6 69.8</td><td>34.1</td><td></td><td>83.653.9</td></tr></table>

<table><tr><td>Method</td><td>Orig.Cam.Robot Lang.Light BG Noise Layout Total</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GAM(Ours)</td><td>67.8100.0 91.8</td><td>94.9</td><td>67.5</td><td></td><td></td><td>97.1</td><td>64.5</td><td>80.4</td></tr><tr><td>π0.5</td><td>97.4 98.0</td><td>77.2</td><td>73.6</td><td>70.5</td><td>93.2</td><td>92.5 87.3</td><td></td><td>70.479.3</td></tr><tr><td>Cosmos-policy</td><td>98.2</td><td>64.0</td><td>64.1</td><td></td><td>73.4100.085.1</td><td>75.2</td><td></td><td>65.2 73.5</td></tr><tr><td>Fast-WAM</td><td>97.0</td><td>8.1</td><td>24.7</td><td>49.8</td><td>75.3 44.5</td><td>23.7</td><td>48.0</td><td>39.2</td></tr><tr><td>OpenVLA-OFT</td><td>97.9</td><td>62.0</td><td>25.2</td><td>53.2</td><td>93.9 92.5</td><td>75.2</td><td>59.1</td><td>63.0</td></tr><tr><td>RIPT-VLA</td><td>97.5</td><td>65.7</td><td>23.2</td><td>45.4</td><td>74.2 79.7</td><td>71.0</td><td>59.8</td><td>58.0</td></tr><tr><td></td><td>95.8</td><td>56.6</td><td>43.3</td><td>43.2</td><td>90.384.7</td><td>82.8</td><td>59.8</td><td>63.4</td></tr><tr><td></td><td>95.8</td><td>12.3</td><td>5.6</td><td>39.3</td><td>84.2 76.5</td><td>76.5</td><td>44.7</td><td>44.9</td></tr><tr><td>πo-Fast</td><td>88.6</td><td>70.8</td><td>20.5</td><td>47.3</td><td>95.3</td><td>60.9 69.7</td><td>51.6</td><td>57.5</td></tr><tr><td>UniVLA</td><td>95.6</td><td>3.9</td><td>37.9</td><td>45.6</td><td>89.678.3</td><td>33.5</td><td>22.6</td><td>40.7</td></tr><tr><td>NORA</td><td>89.4</td><td>2.9</td><td>31.1</td><td>56.6</td><td>60.6 60.5</td><td>18.2</td><td>53.9</td><td>38.8</td></tr><tr><td>WorldVLA</td><td>82.6</td><td>0.3</td><td>30.6</td><td>42.2</td><td>68.830.3</td><td>13.5</td><td>47.4</td><td>31.8</td></tr><tr><td>OpenVLA</td><td>79.2</td><td>2.5</td><td>2.7</td><td>21.5</td><td>9.027.1</td><td>19.5</td><td>25.6</td><td>15.1</td></tr><tr><td>π0.5+ROCKET</td><td>96.6</td><td>41.2</td><td>36.9</td><td>27.6</td><td>57.047.0</td><td>28.2</td><td>46.8</td><td>39.7</td></tr></table>

<table><tr><td>Method</td><td>Orig. Cam. Robot Lang.Light BG Noise Layout Total</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GAM (Ours)</td><td>94.6</td><td>62.5</td><td>62.8</td><td>95.3</td><td>91.6</td><td>88.2</td><td>89.5</td><td>79.5</td><td>578.0</td></tr><tr><td>π0.5</td><td>92.4</td><td>49.2</td><td>76.1</td><td>89.6</td><td>93.8 90.3</td><td></td><td>73.1</td><td>85.9</td><td>77.9</td></tr><tr><td>Cosmos-policy</td><td>97.6</td><td>58.9</td><td>67.4</td><td>94.8</td><td></td><td>96.468.9</td><td>91.8</td><td>92.9</td><td>81.0</td></tr><tr><td>Fast-WAM</td><td>95.2</td><td>17.7</td><td>45.3</td><td>60.1</td><td></td><td>52.2 22.8</td><td>28.5</td><td>61.2</td><td>41.1</td></tr><tr><td>OpenVLA-OFT</td><td>94.5</td><td>38.7</td><td>38.2</td><td>87.0</td><td></td><td>89.4 86.8</td><td>63.5</td><td>76.9</td><td>66.4</td></tr><tr><td>RIPT-VLA</td><td>97.5</td><td>34.1</td><td>38.4</td><td>88.3</td><td></td><td>93.4 89.3</td><td>66.4</td><td>79.2</td><td>67.5</td></tr><tr><td></td><td>73.8</td><td>38.7</td><td>39.9</td><td>69.7</td><td></td><td>79.2 72.3</td><td>64.6</td><td>77.6</td><td>61.3</td></tr><tr><td></td><td>85.2</td><td>3.8</td><td>3.6</td><td>68.4</td><td></td><td>74.5 69.5</td><td>64.4</td><td>69.6</td><td>48.4</td></tr><tr><td>πo-Fast</td><td>60.2</td><td>33.2</td><td>12.0</td><td>43.6</td><td></td><td>91.644.6</td><td>46.1</td><td>47.8</td><td>43.4</td></tr><tr><td>UniVLA</td><td>92.0</td><td>1.9</td><td>53.2</td><td>64.2</td><td></td><td>65.7 74.4</td><td>25.4</td><td>16.4</td><td>39.9</td></tr><tr><td>NORA</td><td>74.6</td><td>1.2</td><td>39.4</td><td>64.0</td><td></td><td>30.354.0</td><td>15.1</td><td>59.5</td><td>36.3</td></tr><tr><td>WorldVLA</td><td>59.0</td><td>0.0</td><td>12.2</td><td>20.6</td><td>20.4</td><td>1.7</td><td>1.6</td><td>4.4</td><td>8.2</td></tr><tr><td>OpenVLA</td><td>53.7</td><td>0.0</td><td>3.0</td><td>22.2</td><td>10.6</td><td>519.4</td><td>17.6</td><td>28.3</td><td>14.3</td></tr><tr><td>π0.5+ROCKET</td><td>89.2</td><td>25.3</td><td>38.4</td><td>11.0</td><td></td><td>77.0 29.4</td><td>23.8</td><td>71.5</td><td>36.7</td></tr></table>

![](images/edbd2a724aa2049e33c5c280408c8df1d97fac15f4a07d6f3f2e3309dadc7838.jpg)  
Figure 9: Detailed zero-shot robustness on LIBERO-Plus. We report success rates across difficulty levels L1–L5 for each perturbation category in the LIBERO-PLUS benchmark. The Average panel summarizes performance across all perturbation categories.

Overall, these breakdowns show that GAM preserves strong performance on the original LIBERO tasks while improving robustness on LIBERO-Plus, especially under perturbations that require stable geometric understanding such as camera-viewpoint changes.

## B.3 Generated Future Depth Maps

In Figure 10, we visualize the future depth predictions generated by GAM across each task suite in the LIBERO benchmark. Given a current RGB observation, GAM predicts the and future depth

Table 13: Per-task success rates on LIBERO Original. Task names are abbreviated by removing suite-level repeated context.
<table><tr><td colspan="2">Spatial 98.6%</td><td colspan="2">Object 99.0%</td><td colspan="2">Goal 97.4%</td><td colspan="2">Long 94.6%</td></tr><tr><td>Task</td><td>SR</td><td>Task</td><td>SR</td><td>Task</td><td>SR</td><td>Task</td><td>SR</td></tr><tr><td>Between plate &amp; ramekin</td><td>100%</td><td>Alphabet soup</td><td>100%</td><td>Open middle drawer</td><td>100%</td><td>Soup + tomato sauce to basket</td><td>96%</td></tr><tr><td>Next to ramekin</td><td>100%</td><td>Cream cheese</td><td>98%</td><td>Bowl on stove</td><td>100%</td><td>Cream cheese + butter to basket</td><td>100%</td></tr><tr><td>From table center</td><td>98%</td><td>Salad dressing</td><td>100%</td><td>Wine bottle on cabinet</td><td>96%</td><td>Stove on + moka pot on stove</td><td>98%</td></tr><tr><td>On cookie box</td><td>100%</td><td>BBQ sauce</td><td>100%</td><td>Top drawer open + bowl inside</td><td>90%</td><td>Bowl in bottom drawer +close</td><td>100%</td></tr><tr><td>In top drawer</td><td>98%</td><td>Ketchup</td><td>98%</td><td>Bowl on cabinet</td><td>98%</td><td>Mugs to left/right plates</td><td>80%</td></tr><tr><td>On ramekin</td><td>96%</td><td>Tomato sauce</td><td>94%</td><td>Plate to front of stove</td><td>98%</td><td>Book to caddy back compartment</td><td>100%</td></tr><tr><td>Next to cookie box</td><td>100%</td><td>Butter</td><td>100%</td><td>Cream cheese in bowl</td><td>100%</td><td>Mug to plate + pudding to plate</td><td>96%</td></tr><tr><td>On stove</td><td>100%</td><td>Milk</td><td>100%</td><td>Turn on stove</td><td>100%</td><td>Soup + cream cheese box to basket</td><td>96%</td></tr><tr><td>Next to plate</td><td>94%</td><td>Chocolate pudding</td><td>100%</td><td>Bowl on plate</td><td>92%</td><td>Both moka pots on stove</td><td>88%</td></tr><tr><td>On wooden cabinet</td><td>100%</td><td>Orange juice</td><td>100%</td><td>Wine bottle on rack</td><td>100%</td><td>Mug to microwave + close</td><td>92%</td></tr></table>

maps while simultaneously generating actions that align spatially with the anticipated future geometry. As demonstrated in the visualizations, GAM accurately forecasts the future depth alongside its corresponding action sequence.

Progress  
![](images/b20a2a825212abed68e5d6661aaf5a891e95b34c5335878ba5875f45c798a6c7.jpg)

(a) LIBERO-Spatial: bowl from table center to plate. Progress  
![](images/4ede8c9b34c8aafe1f9efa518c888fef47ede3200a49b06c82cb976bfb268993.jpg)

(b) LIBERO-Object: tomato sauce to basket. Progress  
![](images/a9f55b9efe7aa9ff84413de9a5bbd91e50ee4eda00aee1ceb876b05ec571d26d.jpg)

(c) LIBERO-Long: cream cheese and butter to basket. Progress  
![](images/13dbc27dd82fe2c4bbe1957a8de571f234997c432a542e2d936aa36b46a9e536.jpg)  
(d) LIBERO-Goal: wine bottle on cabinet.  
Figure 10: Future depth visualizations predicted by our model on representative LIBERO tasks.

![](images/9bf3dff58ee281a085b0acb348a5c76acc47a08826931d5d534525d993b84353.jpg)  
Figure 11: Attention visualizations of action tokens.

## C Ablation and Diagnostic Analyses

## C.1 When to Predict Actions?

We additionally evaluate a direct-action supervision variant that applies the action loss directly to the output action token of the causal future predictor, without passing the action token through the remaining DA3 blocks. This ablation uses the same setting as the main component ablation on LIBERO-Object.

Table 14: Direct action-token ablation.
<table><tr><td>Variant</td><td>Orig.</td><td>Plus</td></tr><tr><td>Direct-action supervision</td><td>98.4</td><td>84.1</td></tr><tr><td>GAM (Ours)</td><td>99.6</td><td>89.7</td></tr></table>

Passing the action token through the deep geometric decoder provides an additional improvement, particularly on LIBERO-Plus Object, suggesting that the remaining GFM layers contribute to refine the action representation, especially under camera perturbations.

## C.2 Attention Analysis

In Figure 11, we visualize the attention maps of action tokens across GFM layers to inspect which visual regions contribute to action decoding. As shown in the attention maps, several intermediate layers attend to task-relevant regions, with clear saliency around manipulated objects and nearby contact regions. This qualitative trend is consistent with the layer ablation: mid-level representations retain object-level structure while still leaving enough depth in the GFM decoder for action-token refinement.

## References

[1] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[2] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[3] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

[4] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[5] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[6] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[7] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[8] J. Pai, L. Achenbach, V. Montesinos, B. Forrai, O. Mees, and E. Nava. mimic-video: Videoaction models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692, 2025.

[9] S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, et al. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[10] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv preprint arXiv:2403.03954, 2024.

[11] W. Huang, Y.-W. Chao, A. Mousavian, M.-Y. Liu, D. Fox, K. Mo, and L. Fei-Fei. Pointworld: Scaling 3d world models for in-the-wild robotic manipulation. arXiv preprint arXiv:2601.03782, 2026.

[12] H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

[13] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

[14] J. Wang, M. Chen, S. Zhang, N. Karaev, J. Schonberger, P. Labatut, P. Bojanowski, D. Novotny, ¨ A. Vedaldi, and C. Rupprecht. Vggt-ω. arXiv preprint arXiv:2605.15195, 2026.

[15] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

[16] F. Li, W. Song, H. Zhao, J. Wang, P. Ding, D. Wang, L. Zeng, and H. Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model. arXiv preprint arXiv:2510.12276, 2025.

[17] G. Sun, T. Du, K. Feng, C. Luo, X. Ding, Z. Shen, Z. Wang, Y. He, and A. Li. Rocket: Residual-oriented multi-layer alignment for spatially-aware vision-language-action models. arXiv preprint arXiv:2602.17951, 2026.

[18] Q. Qian, G. Zhao, G. Zhang, J. Wang, R. Xu, J. Gao, and D. Zhao. Gp3: A 3d geometry-aware policy with multi-view images for robotic manipulation. arXiv preprint arXiv:2509.15733, 2025.

[19] S. Ge, Y. Zhang, S. Xie, W. Zhang, M. Zhou, and Z. Wang. Vggt-dp: Generalizable robot control via vision foundation models. arXiv preprint arXiv:2509.18778, 2025.

[20] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

[21] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

[22] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[23] L. X. Shi, B. Ichter, M. Equi, L. Ke, K. Pertsch, Q. Vuong, J. Tanner, A. Walling, H. Wang, N. Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-languageaction models. arXiv preprint arXiv:2502.19417, 2025.

[24] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[25] C.-Y. Hung, Q. Sun, P. Hong, A. Zadeh, C. Li, U. Tan, N. Majumder, S. Poria, et al. Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854, 2025.

[26] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. others. 2025. qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 4(5), 1.

[27] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, ˜ S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[28] G. R. Team, A. Abdolmaleki, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, A. Balakrishna, N. Batchelor, A. Bewley, J. Bingham, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025.

[29] C. Cheang, S. Chen, Z. Cui, Y. Hu, L. Huang, T. Kong, H. Li, Y. Li, Y. Liu, X. Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025.

[30] J. Yang, R. Tan, Q. Wu, R. Zheng, B. Peng, Y. Liang, Y. Gu, M. Cai, S. Ye, J. Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the computer vision and pattern recognition conference, pages 14203–14214, 2025.

[31] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

[32] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025.

[33] S. Tan, K. Dou, Y. Zhao, and P. Krahenb ¨ uhl. Interactive post-training for vision-language- ¨ action models. arXiv preprint arXiv:2505.17016, 2025.

[34] J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.

[35] A. Abouzeid, M. Mansour, Q. Sun, Z. Sun, and D. Song. Geoaware-vla: Implicit geometry aware vision-language-action model. arXiv preprint arXiv:2509.14117, 2025.

[36] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

[37] G. Zhou, H. Pan, Y. LeCun, and L. Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024.

[38] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[39] R. Zheng, J. Wang, S. Reed, J. Bjorck, Y. Fang, F. Hu, J. Jang, K. Kundalia, Z. Lin, L. Magne, et al. Flare: Robot learning with implicit world modeling. arXiv preprint arXiv:2505.15659, 2025.

[40] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He. π3: Scalable permutation-equivariant visual geometry learning. arXiv e-prints, pages arXiv–2507, 2025.

[41] N. Keetha, N. Muller, J. Sch ¨ onberger, L. Porzi, Y. Zhang, T. Fischer, A. Knapitsch, D. Zauss, ¨ E. Weber, N. Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414, 2025.

[42] L. Sun, B. Xie, Y. Liu, H. Shi, T. Wang, and J. Cao. Geovla: Empowering 3d representations in vision-language-action models. arXiv preprint arXiv:2508.09071, 2025.

[43] Z. Song, Q. Li, J. Zhou, Z. Yuan, T. Chen, L. Lin, and G. Wang. Robotic manipulation is vision-to-geometry mapping (f(v) → g): Vision-geometry backbones over language and video models. arXiv preprint arXiv:2604.12908, 2026.

[44] C. Xu, H. Li, S. Cheng, J. Hu, H. Fan, Z. Feng, and S. Liu. Action-geometry prediction with 3d geometric prior for bimanual manipulation. arXiv preprint arXiv:2602.23814, 2026.

[45] R. Ranftl, A. Bochkovskiy, and V. Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.

[46] J. Lu, J. Xu, W. Hu, R. Zhu, C. Zhao, S.-K. Yeung, Y. Shan, and Y. Liu. Track4world: Feedforward world-centric dense 3d tracking of all pixels. arXiv preprint arXiv:2603.02573, 2026.

[47] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

[48] S. Nasiriany, S. Nasiriany, A. Maddukuri, and Y. Zhu. Robocasa365: A large-scale simulation framework for training and benchmarking generalist robots. arXiv preprint arXiv:2603.04356, 2026.

[49] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. arXiv preprint arXiv:2310.17596, 2023.

[50] O.-E. Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, et al. Open x-embodiment: Robotic learning datasets and rt-x models. arXiv preprint arXiv:2310.08864, 1(2), 2023.

[51] T. Yuan, Z. Dong, Y. Liu, and H. Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[52] C. Wen, J. Lin, T. Darrell, D. Jayaraman, and Y. Gao. Fighting copycat agents in behavioral cloning from observation histories. Advances in Neural Information Processing Systems, 33: 2564–2575, 2020.

[53] P. De Haan, D. Jayaraman, and S. Levine. Causal confusion in imitation learning. Advances in neural information processing systems, 32, 2019.

[54] Y. Zheng, X. Li, S. Gu, Y. Zheng, S. Tian, W. Li, L. Wang, S. Fei, P. Li, Y. Gao, et al. Pokevla: Empowering pocket-sized vision-language-action model with comprehensive world knowledge guidance. arXiv preprint arXiv:2604.20834, 2026.