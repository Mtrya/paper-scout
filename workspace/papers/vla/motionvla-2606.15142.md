# MotionVLA: Vision-Language-Action Model for Humanoid Motion

Nonghai Zhang1∗ Siyu Zhai1∗ Yanjun Li1∗ Zeyu Zhang1∗† Zhihan Yin1 Yandong Guo2 Boxin Shi1 Hao Tang1‡

1 School of Computer Science, Peking University 2 AI2 Robotics ∗Equal contribution. †Project lead. ‡Corresponding author: bjdxtanghao@gmail.com.

## Abstract

Generating realistic humanoid motion from scene images and text involves both low-frequency pose semantics and high-frequency physical dynamics. However, many existing methods tokenize motion with a single shared codebook, forcing heterogeneous motion signals into the same quantization space. Our frequencydomain analysis of human motion data reveals a clear mismatch between singlecodebook quantization and motion statistics: five DCT coefficients capture 93% of joint-position energy but only 37% of joint-velocity energy, which can bias quantization toward pose statistics and under-represent high-frequency velocity components. A second challenge lies in adapting a standard autoregressive model to effectively model high-frequency physical signals in motion sequences. Therefore, we propose DSFT, a dual-stream frequency tokenizer that separates motion into Base and physical streams and compresses them independently with DCT truncation and BPE. Furthermore, we present MotionVLA, a Qwen3.5- based model that arranges Base and physical tokens in a unified sequence, where Phys tokens are predicted after Base tokens. Experiments on HumanML3D and MBench show that, despite using a lightweight 2B backbone, MotionVLA reduces the Diversity gap to real data by over 50% on HumanML3D and improves Motion-Condition Consistency by 3.8% on MBench, supporting frequency-aware dual-stream decoupling as an effective formulation for autoregressive motion generation. Code: https://github.com/AIGeeksGroup/MotionVLA. Website: https://aigeeksgroup.github.io/MotionVLA.

## 1 Introduction

Fine-grained humanoid motion generation is a core capability for embodied intelligence, character animation, and scene-aware action synthesis. In recent years, progress in this area has largely followed an autoregressive paradigm, in which motion is discretized into token sequences and generated step by step with Transformer-based models [35, 10, 7, 32, 27, 15, 31]. At the same time, recent advances in vision-language-action (VLA) modeling [11, 2] highlight the value of grounding action generation in scene observations and language instructions, thereby motivating scene-aware vision-language-to-motion generation [13].

However, although recent studies have shown that applying DCT before quantization can bring clear advantages to long-horizon autoregressive generation [22, 34, 6], these tokenizers still encode motion within a unified tokenization space, implicitly treating heterogeneous motion components as if they followed similar statistics. Our analysis shows that this assumption does not hold well for human motion: joint positions are strongly low-frequency, with the first five DCT coefficients covering 93% of their energy, whereas joint velocities are markedly high-frequency, with the same five coefficients covering only 37%. Consequently, such a shared tokenization mechanism is naturally biased toward low-frequency pose structure, while high-frequency physical signals are more easily under-represented.

![](images/78c790025abe5a0eb5ecd88244b7ef933a7d7058eca569dc779b1141fa6edcbc.jpg)  
Figure 1: Given a text description and a scene video as input, MotionVLA generates motions that closely track the ground truth (GT) across the full sequence (frames at 30%, 60%, and 90% shown). ViMoGen [13], which relies on a single-stream tokenizer, exhibits temporal drift and joint instability that accumulate over time (highlighted in white circles).

At the same time, this representation issue directly creates a second challenge for autoregressive generation. Because a standard autoregressive model predicts motion tokens from a unified codebook, it is naturally encouraged to model the dominant low-frequency pose structure first, while the weaker high-frequency physical signals are less reliably preserved. As generation proceeds over time, this imbalance makes it difficult for the model to faithfully maintain fine-grained physical dynamics, causing errors in contact and motion stability to accumulate. In practice, this often manifests as artifacts such as temporal drift, foot sliding, and contact distortion [4], especially in long-horizon generation [4, 25].

However, all existing tokenization methods, including frequency-domain ones, share an unresolved structural limitation: each frame of motion is represented by a single discrete token, forcing signals with fundamentally different frequency profiles into a single codebook. We quantify this imbalance directly on HumanML3D [8]. Joint positions are dominated by low frequencies: five DCT coefficients reconstruct 93% of position energy. Joint velocities, as first derivatives of positions, obey the differentiation theorem, which scales each DCT coefficient by its frequency index and inherently amplifies high-frequency components. The same five coefficients capture only 37% of velocity energy, a gap exceeding fifty percentage points. When BPE is applied to a single concatenated feature, low-frequency position statistics dominate the vocabulary, causing the codebook to effectively discard the high-frequency velocity signal as noise. The practical consequence is visible in Figure 1: motions generated with a single-stream tokenizer accumulate drift and joint instability over time, whereas our method tracks the ground truth throughout the sequence. Prior work has documented these artifacts, including foot sliding and contact distortion [4], and proposed post-hoc corrections at the decoding stage [4, 25], but no prior method removes the structural cause.

To address these challenges, we propose MotionVLA, a vision-language-to-motion framework built on a dual-stream representation of human motion. Its core component, DSFT, separates motion into a Base stream that captures joint-position semantics and a physical stream that captures joint-velocity dynamics, and tokenizes them independently in the frequency domain. Building on this representation, MotionVLA arranges the two streams in a unified autoregressive sequence, where Phys tokens are generated after Base tokens so that physical-signal prediction can leverage the preceding pose context through causal attention. As show in Figure 1 our method explicitly decouples low-frequency semantic structure from high-frequency physical dynamics while preserving a simple autoregressive formulation. Our contributions are threefold.

![](images/b3f3cd76d8e296427c841fb090fbd653f2cddf458396562e73863c09017f6896.jpg)  
Figure 2: Overview of MotionVLA. (a) DSFT performs dual-stream frequency tokenization by decomposing motion into Base and Phys components and converting them into discrete tokens. (b) During training, MotionVLA learns to autoregressively predict the unified motion token sequence under text and scene-image conditioning, supervised by DSFT tokens derived from ground-truth motion. (c) At inference time, the model generates Base and Phys tokens conditioned on multimodal inputs, which are then decoded and recombined to reconstruct the final motion sequence.

• we propose DSFT, a dual-stream tokenizer that separates motion into Base and Phys streams and compresses them independently in the frequency domain, addressing the mismatch between unified tokenization and heterogeneous motion statistics.

• We present MotionVLA, a vision-language-to-motion framework that models human motion generation as a unified autoregressive process over decoupled semantic and physical token streams.

• Experiments on HumanML3D and MBench show that our method reduces the Diversity gap to real data by over 50% on HumanML3D, while improving Motion-Condition Consistency from 0.53 to 0.55 and reducing Foot Sliding from 0.0051 to 0.0049 on MBench.

## 2 The Proposed Method

## 2.1 Overview

As illustrated in Figure 2, MotionVLA comprises two key components: DSFT, a dual-stream frequency tokenizer that converts motion sequences into discrete Base and Phys token streams, and a Qwen3.5-based autoregressive backbone that generates these tokens conditioned on scene images and text instructions. Given a scene image I and a text description t, the model encodes the multimodal context and autoregressively predicts a unified motion token sequence $[ M _ { \mathrm { B O S } } , b _ { 1 } , \dots , b _ { N } , M _ { \mathrm { S E P } } , p _ { 1 } , \dots , p _ { M } , M _ { \mathrm { E O S } } ]$ , where Base tokens $b _ { i }$ capture low-frequency pose semantics and Phys tokens $p _ { j }$ encode high-frequency physical dynamics. After generation, the two streams are independently decoded through BPE inversion and inverse DCT, and then recombined to reconstruct the full motion sequence. This design preserves a unified autoregressive formulation while explicitly disentangling semantic structure from physical dynamics.

## 2.2 DSFT: Dual-Stream Frequency-Domain Tokenizer

DSFT is motivated by a simple observation: human motion is not spectrally homogeneous. Joint positions and rotations evolve relatively smoothly over time and are therefore dominated by lowfrequency components, whereas joint velocities exhibit much stronger high-frequency behavior. This distinction follows naturally from the differentiation theorem, since temporal differentiation amplifies higher-frequency coefficients. To verify this property directly from data, we analyze the DCT energy distribution of each motion dimension on both HumanML3D (263 dimensions) and ViMoGen (276 dimensions), and characterize each dimension by its low-frequency ratio, defined as the fraction of energy covered by the first five DCT coefficients. As shown in Figure 3, the resulting distributions are strongly bimodal on both datasets, revealing a consistent separation between low-frequency semantic dimensions and high-frequency physical dimensions.

![](images/abfa3d2f04588f71685077b660c2eb85f9d65de13d296b2a91920ffd3abf317d.jpg)

![](images/9c303e56186d678a4eb821c0a72048509cdabf7b736a7ff7d33940f28b1833d1.jpg)

![](images/932a43419d29cafb76978154ff0b51e2021d72edcac62ac365943d58203fcc1f.jpg)

![](images/58061ab8642adc758723a7f3c9007c6ecf2a65e819966306a389f2aa644dc0c6.jpg)  
Figure 3: Frequency-domain clustering of motion dimensions. (a) Per-dimension low-frequency ratio on HumanML3D. (b) Corresponding histogram on HumanML3D. (c/d) Corresponding plots on ViMoGen. Both datasets exhibit a consistent bimodal separation between low-frequency Base dimensions and high-frequency Phys dimensions.

Based on this observation, we partition motion into two streams according to physical semantics. The Base stream contains position- and rotation-related dimensions that primarily encode pose semantics, whereas the Phys stream contains velocity-related dimensions that primarily encode physical dynamics. Concretely, this yields $( D _ { b } , D _ { p } ) = \mathsf { \bar { ( } 1 9 0 , 7 3 ) }$ for HumanML3D and (201, 75) for ViMoGen (see Appendix D for the exact per-field index mapping). We therefore represent a motion sequence $\dot { \mathbf { M } } \in \mathbb { R } ^ { \dot { T } \times D }$ as

$$
\mathbf { M } _ { \mathrm { b a s e } } \in \mathbb { R } ^ { T \times D _ { b } } , \qquad \mathbf { M } _ { \mathrm { p h y s } } \in \mathbb { R } ^ { T \times D _ { p } } .\tag{1}
$$

This distinction is further quantified in Figure 4, which compares the cumulative DCT energy retained by the Base and Phys streams as the number of preserved coefficients increases. As shown in the figure, the Base stream is highly compressible: retaining only $K = 5$ coefficients already covers about 86% to 93% of its energy across datasets. In contrast, the Phys stream is substantially more broadband, with the same $K = 5$ covering only about 37% of its energy. Therefore, compressing both streams under a shared frequency budget would inevitably favor the low-frequency Base stream while discarding a large portion of high-frequency physical information. This also explains why single-stream tokenization tends to preserve pose structure more easily than physical dynamics, leading to systematic information loss in the latter.

Accordingly, we retain different numbers of DCT coefficients for the two streams, using a small truncation length for the Base stream and a larger one for the Phys stream. Specifically, we set $K _ { b } = 5$ and $K _ { p } = 2 5$ , and apply DCT independently to obtain

$$
\begin{array} { r } { \mathbf { C } _ { \mathrm { b a s e } } = \mathrm { D C T } ( \mathbf { M } _ { \mathrm { b a s e } } ) _ { [ : K _ { b } ] } , \qquad \mathbf { C } _ { \mathrm { p h y s } } = \mathrm { D C T } ( \mathbf { M } _ { \mathrm { p h y s } } ) _ { [ : K _ { p } ] } . } \end{array}\tag{2}
$$

After truncation, each stream is flattened and encoded by an independently trained BPE tokenizer, yielding a Base token sequence b and a Phys token sequence p. During decoding, we first recover the truncated coefficients by inverse BPE mapping, and then reconstruct the two time-domain streams by inverse DCT. Finally, the reconstructed Base and Phys streams are concatenated along the feature dimension to recover the complete motion sequence.

In this way, DSFT converts continuous motion into two complementary token streams, which provide the foundation for the unified autoregressive generation framework described next.

## 2.3 MotionVLA: Unified Sequence Formulation, Objective, and Inference

Given a scene image I and a text description t, MotionVLA formulates motion generation as a unified autoregressive sequence modeling problem. Specifically, each motion sample is represented as

$$
\mathbf { s } = [ M _ { \mathrm { B O S } } , b _ { 1 } , \dotsc , b _ { N } , M _ { \mathrm { S E P } } , p _ { 1 } , \dotsc , p _ { M } , M _ { \mathrm { E O S } } ] ,\tag{3}
$$

where $b _ { i }$ denotes Base tokens and $p _ { j }$ denotes Phys tokens. In this way, MotionVLA preserves a simple autoregressive formulation while imposing a structured semantic-to-physical generation order.

![](images/5a7a83a30fdc5968a4dea2cb2a2b9089bbccce692f9ccc27a328d53ff72720dd.jpg)

![](images/4ee8ac3eb555f7c3013a09ed20187dfe15d7178e890feb1a0d8d05ff794241a8.jpg)

![](images/729ba04b6252483ea3f6c80e31e41601344a2c1ba17450408036b3cf5b8c6454.jpg)

![](images/89ad0cb436726e389966bd4c87a3512e12a4cb530af4038e0beb3e8bdb15eff1.jpg)  
Figure 4: Energy coverage of the Base and Phys streams under different DCT truncation lengths. (a,b) Results on HumanML3D. (c,d) Corresponding results on ViMoGen. The Base stream is highly compressible with small K, whereas the Phys stream requires substantially larger K to preserve its energy.

This ordering is important because physical dynamics are typically conditioned on the underlying pose structure. By placing Phys tokens after all Base tokens, MotionVLA allows each Phys prediction to attend to the complete preceding Base context through causal attention. As a result, the model can generate physical dynamics with access to the full semantic pose information, rather than predicting both streams in an entangled manner. This design enables a hierarchical semantic-to-physical generation process within a standard autoregressive transformer.

To instantiate this sequence within the backbone token space, we extend the original vocabulary with motion tokens and three structural markers, yielding

$$
V = V _ { \mathrm { L M } } + V _ { \mathrm { m o t i o n } } + 3 , \qquad V _ { \mathrm { m o t i o n } } = V _ { \mathrm { b a s e } } + V _ { \mathrm { p h y s } } ,\tag{4}
$$

where $V _ { \mathrm { L M } }$ denotes the original backbone vocabulary, and $V _ { \mathrm { m o t i o n } }$ denotes the motion vocabulary induced by DSFT.

During training, the scene image and text description serve as conditioning context, while the model is optimized to predict the motion portion of the sequence with teacher forcing. Let y denote the training targets, where non-motion positions are masked out. We optimize MotionVLA with a masked next-token prediction objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a i n } } = \mathrm { C E } ( \mathbf { z } + \mathbf { m } , \mathbf { y } ) , } \end{array}\tag{5}
$$

where z denotes the output logits, and m is a logit mask that restricts prediction to valid motion tokens and structural markers. This objective prevents probability mass from being assigned to irrelevant vocabulary entries and focuses learning on the motion token space.

During inference, we further impose a phase-aware generation constraint to preserve the intended Base-to-Phys order. Before generating $M _ { \mathrm { S E P } }$ , only Base tokens and $M _ { \mathrm { S E P } }$ are allowed; after $M _ { \mathrm { S E P } }$ is produced, only Phys tokens and $M _ { \mathrm { E O S } }$ are allowed. This phase-aware mask ensures that semantic structure is generated before physical dynamics, while preventing the model from mixing the two streams during decoding. The complete inference procedure is summarized in Algorithm 1.

## 3 Experiments

## 3.1 Datasets and Evaluation Metrics

We conduct experiments on two settings. In the first setting, we train on ViMoGen-228K [13], a large-scale multimodal motion dataset, and evaluate on MBench [13], the associated fine-grained physical quality benchmark. In the second setting, we train and evaluate on HumanML3D [8] under the standard text-to-motion protocol. Table 1 summarizes dataset statistics; detailed metric definitions and evaluation protocols are provided in Appendix B.

## 3.2 Baselines

We compare MotionVLA with representative baselines spanning three paradigms: discrete autoregressive generation, diffusion-based methods, and approaches with improved motion tokenization, as summarized in Table 8. On ViMoGen-228K and MBench, prior methods are evaluated under their original text-driven setting, whereas MotionVLA additionally conditions on the scene image. On HumanML3D, all methods follow the same standard text-to-motion protocol for a fair comparison.

Algorithm 1 Phase-Aware Autoregressive Generation in MotionVLA   
Require: Scene image I, text instruction t, trained MotionVLA, phase-aware logit mask m   
Ensure: Reconstructed motion sequence $\dot { \mathbf { M } } \in \mathbb { R } ^ { T \times D }$   
1: Encode I, t through Qwen3.5 → context representations   
2: Initialize token buffer s $ [ M _ { \mathrm { B O S } } ]$ , phase $ \mathrm { B A S E }$   
3: while last token ̸= $M _ { \mathrm { E O S } }$ do   
4: $\mathbf { i f } p h a s e = \mathbf { B } \mathbf { \bar { A } S E }$ then   
5: Apply mask: allow $\{ b _ { i } \} \cup \{ M _ { \mathrm { S E P } } \}$ only   
6: else   
7: Apply mask: allow $\{ p _ { j } \} \cup \{ M _ { \mathrm { E O S } } \}$ only   
8: end if   
9: Sample next token τ ∼ softmax(logits + m)   
10: Append τ to s   
11: if $\dot { \cdot } \dot { \tau } = M _ { \mathrm { S E P } }$ then   
12: phase ← PHYS   
13: end if   
14: end while   
15: Extract Base tokens b and Phys tokens p from s   
16: $\mathbf { M } _ { \mathrm { b a s e } }  \mathrm { I D C T } \big ( \mathrm { B P E } ^ { - 1 } ( \mathbf { b } ) \big )$   
17: $\mathbf { M } _ { \mathrm { p h y s } }  \mathrm { I D C T } \big ( \mathrm { B P E } ^ { - 1 } ( \mathbf { p } ) \big )$   
18: return $\mathbf { M }  [ \mathbf { M } _ { \mathrm { b a s e } } \parallel \mathbf { M } _ { \mathrm { p h y s } } ]$ ▷ concatenate along spatial dimension

Table 1: Dataset statistics. ViMoGen-228K is the training dataset; MBench is its associated evaluation benchmark (not a dataset subset). †: optical motion capture with marker-based GT. ‡: inthe-wild video with pseudo-GT from SMPL estimation. #: generative synthesis with physics-based renderer.
<table><tr><td>Dataset/Benchmark</td><td>#Clip</td><td>#Hour</td><td>Scene</td><td>Motion</td><td>Video</td><td>Role</td></tr><tr><td>HumanML3D [8]</td><td>14,616</td><td>28.6</td><td>Indoor</td><td>GT</td><td>1</td><td>Train/Eval</td></tr><tr><td>ViMoGen-228K[13]</td><td>228,236</td><td>369.4</td><td>Mixed</td><td>Mixed</td><td>1</td><td>Train</td></tr><tr><td>· Optical MoCap+</td><td>171,542</td><td>292.7</td><td>Indoor</td><td>GT</td><td></td><td>Train</td></tr><tr><td>· In-the-Wild Videot</td><td>41,971</td><td>61.4</td><td>In-the-Wild</td><td>Pseudo GT</td><td>√</td><td>Train</td></tr><tr><td>· Synthetic Video#</td><td>14,723</td><td>16.6</td><td>In-the-Wild</td><td>Pseudo GT</td><td>√</td><td>Not used</td></tr><tr><td>MBench [13]</td><td>450</td><td>1</td><td>In-the-Wild</td><td>GT</td><td>√</td><td>Eval only</td></tr></table>

## 3.3 Experimental Setup

Prior to model training, we train the DSFT tokenizer independently on each benchmark’s training split, ensuring that the discrete motion representation is adapted to the motion statistics of each dataset. Full training details and feature partition specifications are provided in Appendix C and D.

We then conduct four groups of experiments to evaluate MotionVLA comprehensively: a main benchmark evaluation on MBench for scene-conditioned motion generation, a text-to-motion generalization evaluation on HumanML3D, a DSFT tokenizer reconstruction analysis to assess representation quality, and an ablation study to examine the impact of key design choices. Detailed hyperparameter settings are provided in Appendix C.

## 3.4 Main Results on MBench

Table 2 reports the quantitative comparison on MBench, which evaluates models trained on ViMoGen-228K across eight fine-grained quality dimensions. Despite using a lightweight 2B backbone, MotionVLA achieves the best results on Motion-Condition Consistency and Foot Sliding, while ranking second on Motion Generalizability and Jitter Degree. These gains indicate that the proposed framework is particularly effective at improving multimodal condition alignment and suppressing local temporal artifacts.

At inference time, the target motion length T is provided externally, and MotionVLA generates DSFT tokens conditioned on this target horizon. This pattern is consistent with our design. Scene-aware

Table 2: MotionVLA [13] evaluation on MBench. ↑: higher is better; ↓: lower is better. †: uses additional visual (scene) input. Best in bold, second best underlined.
<table><tr><td>Method</td><td>Motion-Cond. Consistency</td><td>Motion Generaliz.个</td><td>Jitter Degree↓</td><td>Dynamic Degree 个</td><td>Foot Floating↓</td><td>Foot Sliding</td><td>Body Penetration ↓</td><td>Pose Quality↓</td></tr><tr><td>MDM [29](ICLR&#x27;23)</td><td>0.42</td><td>0.51</td><td>0.0136</td><td>0.0376</td><td>0.156</td><td>0.0136</td><td>1.68</td><td>2.67</td></tr><tr><td>T2M-GPT [35](CVPR&#x27;23)</td><td>0.39</td><td>0.38</td><td>0.0156</td><td>0.0349</td><td>0.209</td><td>0.0156</td><td>1.33</td><td>2.43</td></tr><tr><td>FineMoGen [37] (NeurIPS&#x27;24)</td><td>0.37</td><td>0.42</td><td>0.0118</td><td>0.0386</td><td>0.281</td><td>0.0091</td><td>1.18</td><td>2.28</td></tr><tr><td>MotionLCM[5](ECCV&#x27;24)</td><td>0.48</td><td>0.55</td><td>0.0218</td><td>0.0439</td><td>0.193</td><td>0.0202</td><td>1.73</td><td>2.40</td></tr><tr><td>MoMask [7] (CVPR&#x27;24)</td><td>0.38</td><td>0.44</td><td>0.0147</td><td>0.0396</td><td>0.178</td><td>0.0147</td><td>1.48</td><td>2.67</td></tr><tr><td>MotionDiffuse [36] (TPAMI&#x27;24)</td><td>0.44</td><td>0.42</td><td>0.0111</td><td>0.0289</td><td>0.126</td><td>0.0063</td><td>1.35</td><td>2.21</td></tr><tr><td>MotionCraft [1](CVPR&#x27;25)</td><td>0.42</td><td>0.45</td><td>0.0132</td><td>0.0420</td><td>0.402</td><td>0.0090</td><td>1.15</td><td>2.12</td></tr><tr><td>ViMoGen [13](ICLR&#x27;26)</td><td>0.53</td><td>0.68</td><td>0.0108</td><td>0.0251</td><td>0.204</td><td>0.0064</td><td>1.78</td><td>2.38</td></tr><tr><td>ViMoGen-light [13](ICLR&#x27;26)</td><td>0.47</td><td>0.55</td><td>0.0129</td><td>0.0294</td><td>0.155</td><td>0.0051</td><td>1.43</td><td>2.10</td></tr><tr><td>MotionVLA (Ours)†</td><td>0.55</td><td>0.66</td><td>0.0110</td><td>0.0419</td><td>0.149</td><td>0.0049</td><td>1.34</td><td>2.14</td></tr></table>

Table 3: Text-to-motion results on HumanML3D. ↑: higher is better; ↓: lower is better; →: closer to real is better. For Diversity, best and second best are determined by the distance to the Real score. ‡: GenM3 uses a retrained evaluator on 30 FPS data; GenM3∗ uses only HumanML3D text pairs. §: DisCoRD is applied on top of MoMask; Diversity is not reported in the original paper.
<table><tr><td>Method</td><td>R-P Top-1个</td><td> $\mathbf { R } { \cdot } \mathbf { P } \ \mathbf { T o p } { \cdot } 2 \uparrow$ </td><td> $\mathbf { R } { \cdot } \mathbf { P } \mathbf { \ T o p } { \cdot } 3 \uparrow$ </td><td>FID↓</td><td></td><td></td><td>MM-Dist ↓Diversity→MModality↑</td></tr><tr><td>Real</td><td> $0 . 5 1 1 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 0 3 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 9 7 ^ { \pm . 0 0 2 }$ </td><td> $0 . 0 0 2 ^ { \pm . 0 0 0 }$ </td><td> $2 . 9 7 4 ^ { \pm . 0 0 8 }$ </td><td> $9 . 5 0 3 ^ { \pm . 0 6 5 }$ </td><td></td></tr><tr><td>Motion generation only</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TEMOS [23](ECCV&#x27;22)</td><td> $0 . 4 2 4 ^ { \pm . 0 0 2 }$ </td><td> $0 . 6 1 2 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 2 2 ^ { \pm . 0 0 2 }$ </td><td> $3 . 7 3 4 ^ { \pm . 0 2 8 }$ </td><td> $3 . 7 0 3 ^ { \pm . 0 0 8 }$ </td><td> $8 . 9 7 3 ^ { \pm . 0 7 1 }$ </td><td> $0 . 3 6 8 ^ { \pm . 0 1 8 }$ </td></tr><tr><td>TM2T [9] (ECCV&#x27;22)</td><td> $0 . 4 2 4 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 1 8 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 2 9 ^ { \pm . 0 0 2 }$ </td><td> $1 . 5 0 1 ^ { \pm . 0 1 7 }$ </td><td> $3 . 4 6 7 ^ { \pm . 0 1 1 }$ </td><td> $8 . 5 8 9 ^ { \pm . 0 7 6 }$ </td><td> $2 . 4 2 4 ^ { \pm . 0 9 3 }$ </td></tr><tr><td>Guo et al.[8](CVPR&#x27;22)</td><td> $0 . 4 5 5 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 3 6 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 3 6 ^ { \pm . 0 0 2 }$ </td><td> $1 . 0 8 7 ^ { \pm . 0 2 1 }$ </td><td> $3 . 3 4 7 ^ { \pm . 0 0 8 }$ </td><td> $9 . 1 7 5 ^ { \pm . 0 8 3 }$ </td><td> $2 . 2 1 9 ^ { \pm . 0 7 4 }$ </td></tr><tr><td>MDM [29](ICLR&#x27;23)</td><td></td><td></td><td> $0 . 6 1 1 ^ { \pm . 0 0 7 }$ </td><td> $0 . 5 4 4 ^ { \pm . 0 4 4 }$ </td><td> $5 . 5 6 6 ^ { \pm . 0 2 7 }$ </td><td></td><td> $2 . 7 9 9 ^ { \pm . 0 7 2 }$ </td></tr><tr><td>MotionDiffuse [36] (TPAMI&#x27;24)</td><td> $0 . 4 9 1 ^ { \pm . 0 0 1 }$ </td><td> $0 . 6 8 1 ^ { \pm . 0 0 1 }$ </td><td> $0 . 7 8 2 ^ { \pm . 0 0 1 }$ </td><td> $0 . 6 3 0 ^ { \pm . 0 0 1 }$ </td><td> $3 . 1 1 3 ^ { \pm . 0 0 1 }$ </td><td> $9 . 4 1 0 ^ { \pm . 0 4 9 }$ </td><td> $\overline { { 1 . 5 5 3 ^ { \pm . 0 4 2 } } }$ </td></tr><tr><td>T2M-GPT [35](CVPR&#x27;23)</td><td> $0 . 4 9 2 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 7 9 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 7 5 ^ { \pm . 0 0 2 }$ </td><td> $0 . 1 4 1 ^ { \pm . 0 0 4 }$ </td><td> $3 . 1 2 1 ^ { \pm . 0 0 9 }$ </td><td> $9 . 7 2 2 ^ { \pm . 0 8 2 }$ </td><td> $1 . 8 3 1 ^ { \pm . 0 4 8 }$ </td></tr><tr><td>FineMoGen [37](NeurIPS&#x27;24)</td><td> $0 . 5 0 4 ^ { \pm . 0 0 2 }$ </td><td> $0 . 6 9 0 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 8 4 ^ { \pm . 0 0 2 }$ </td><td> $0 . 1 5 1 ^ { \pm . 0 0 8 }$ </td><td> $2 . 9 9 8 ^ { \pm . 0 0 8 }$ </td><td> $9 . 2 6 3 ^ { \pm . 0 9 4 }$ </td><td> $2 . 6 9 6 ^ { \pm . 0 7 9 }$ </td></tr><tr><td>MoMask [7] (CVPR&#x27;24)</td><td> $0 . 5 2 1 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 1 3 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 0 7 ^ { \pm . 0 0 2 }$ </td><td> $0 . 0 4 5 ^ { \pm . 0 0 2 }$ </td><td> $2 . 9 5 8 ^ { \pm . 0 0 8 }$ </td><td></td><td> $1 . 2 4 1 ^ { \pm . 0 4 0 }$ </td></tr><tr><td>MotionGPT [38] (AAAI&#x27;24)</td><td></td><td></td><td></td><td>0.567</td><td>3.775</td><td>9.006</td><td></td></tr><tr><td>DisCoRD[4]  $( \mathrm { I C C V } ^ { , } 2 5 ) ^ { 8 }$ </td><td> $0 . 5 2 4 ^ { \pm . 0 0 3 }$ </td><td> $\cdot$ </td><td> $\cdot$ </td><td> $0 . 0 3 2 ^ { \pm . 0 0 2 }$ </td><td> $2 . 9 3 8 ^ { \pm . 0 1 0 }$ </td><td></td><td> $1 . 2 8 8 ^ { \pm . 0 4 3 }$ </td></tr><tr><td>GenM3* [27](ICCV&#x27;25)*</td><td> $0 . 5 1 0 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 0 2 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 0 2 ^ { \pm . 0 0 2 }$ </td><td> $0 . 0 5 3 ^ { \pm . 0 0 2 }$ </td><td> $2 . 8 6 0 ^ { \pm . 0 0 9 }$ </td><td> $9 . 6 2 9 ^ { \pm . 0 7 7 }$ </td><td></td></tr><tr><td> $\mathrm { G e n M 3 } \ [ 2 7 ] \ \mathrm { ( I C C V ^ { \prime } } 2 5 ) ^ { \ddagger }$ </td><td> $0 . 5 1 1 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 0 5 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 0 4 ^ { \pm . 0 0 2 }$ </td><td> $0 . 0 4 6 ^ { \pm . 0 0 2 }$ </td><td> $\overline { { 2 . 8 5 2 ^ { \pm . 0 0 9 } } }$ </td><td> $9 . 6 7 5 ^ { \pm . 0 8 7 }$ </td><td></td></tr><tr><td>MG-MotionLLM [32] (CVPR&#x27;25)</td><td> $0 . 5 1 6 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 0 6 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 0 2 ^ { \pm . 0 0 3 }$ </td><td> $0 . 3 0 3 ^ { \pm . 0 1 0 }$ </td><td> $2 . 9 5 2 ^ { \pm . 0 0 9 }$ </td><td> $9 . 9 6 0 ^ { \pm . 0 7 3 }$ </td><td> $2 . 1 2 5 ^ { \pm . 1 5 9 }$ </td></tr><tr><td>MotionVLA (Ours)</td><td> $0 . 5 0 7 ^ { \pm . 0 0 2 }$ </td><td> $0 . 6 9 9 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 9 8 ^ { \pm . 0 0 2 }$ </td><td> $0 . 0 7 1 ^ { \pm . 0 0 3 }$ </td><td> $2 . 9 0 6 ^ { \pm . 0 0 9 }$ </td><td> $9 . 5 4 8 ^ { \pm . 0 8 1 }$ </td><td> $2 . 8 2 1 ^ { \pm . 0 9 1 }$ </td></tr></table>

## 3.5 Results on HumanML3D

conditioning mainly improves semantic alignment, while the DSFT dual-stream tokenizer reduces local temporal artifacts such as jitter and foot sliding. Meanwhile, MotionVLA does not dominate all physical metrics, indicating that low-level geometric quality remains challenging under the smaller model scale. Overall, MotionVLA provides a favorable trade-off between multimodal controllability and physical motion quality on MBench.

Table 3 reports text-to-motion generation results on HumanML3D under the standard benchmark setting. Although MotionVLA uses a lightweight 2B backbone and is designed for multimodal motion generation, it remains competitive on this purely text-driven benchmark, achieving the Diversity score closest to the real data distribution and the highest MModality among generated methods. Its R-Precision, FID, and MM-Dist scores also remain competitive with strong recent baselines, indicating that the proposed framework transfers beyond the multimodal training setting.

This pattern is consistent with our design. Since HumanML3D removes visual conditioning, the gains mainly reflect the motion representation itself rather than scene input. By separating low-frequency motion semantics from high-frequency physical dynamics, DSFT preserves richer motion variation while maintaining competitive sample fidelity and text-motion alignment. Overall, these results show that MotionVLA generalizes effectively beyond ViMoGen and provides a strong diversity-quality trade-off even at a relatively small 2B model scale.

## 3.6 DSFT Tokenizer Analysis

We analyze DSFT on HumanML3D through controlled comparisons within the DCT+BPE family. Compared with a single-stream DCT+BPE baseline, DSFT produces a more compact token sequence and a substantially lower reconstruction Fréchet inception distance (rFID), despite having higher reconstruction root mean square error (rRMSE) and MPJPE. This suggests that lower pointwise reconstruction error does not necessarily imply better tokenizer quality.

Table 4: DSFT tokenizer reconstruction analysis on HumanML3D. Smaller Tok./Frame indicates a more compact tokenization; lower rRMSE, MPJPE, and rFID are better.
<table><tr><td>Method</td><td>Tok./Frame</td><td>rRMSE</td><td>MPJPE</td><td>rFID</td></tr><tr><td>Single-Stream DCT+BPE</td><td>15.21</td><td>0.0164</td><td>0.0054</td><td>0.9461</td></tr><tr><td>DSFT  $( K _ { p } { = } 2 5 )$ </td><td>11.24</td><td>0.0226</td><td>0.0093</td><td>0.1868</td></tr><tr><td>DSFT  $( K _ { p } { = } 1 0 )$ </td><td>8.80</td><td>0.0240</td><td>0.0108</td><td>1.3404</td></tr><tr><td>DSFT  $( K _ { p } { = } 1 5 )$ </td><td>9.74</td><td>0.0233</td><td>0.0102</td><td>0.5966</td></tr><tr><td>DSFT  $( K _ { p } = 2 0 )$ </td><td>10.55</td><td>0.0228</td><td>0.0097</td><td>0.2895</td></tr><tr><td>DSFT  $( K _ { p } { = } 2 5 )$ </td><td>11.24</td><td>0.0226</td><td>0.0093</td><td>0.1868</td></tr><tr><td>DSFT  $( K _ { p } { = } 3 0 )$ </td><td>11.81</td><td>0.0224</td><td>0.0091</td><td>0.1380</td></tr></table>

Table 5: Backbone scale ablation on MBench. †: default configuration used in main experiments. ↑/↓: higher/lower is better.
<table><tr><td>Backbone</td><td>Params</td><td>Motion-Cond. Consistency</td><td>Motion 个 Generaliz. 个</td><td>Jitter Degree↓</td><td>Dynamic Degree</td><td>Foot 个 Floating↓</td><td>Foot Sliding↓</td><td>Body Penetration ↓</td><td>Pose Quality↓</td></tr><tr><td>Qwen3.5-0.8B</td><td>0.8B</td><td>0.51</td><td>0.60</td><td>0.0122</td><td>0.0364</td><td>0.162</td><td>0.0058</td><td>1.46</td><td>2.26</td></tr><tr><td>Qwen3.5-2Bt</td><td>2B</td><td>0.55</td><td>0.66</td><td>0.0110</td><td>0.0419</td><td>0.149</td><td>0.0049</td><td>1.34</td><td>2.14</td></tr><tr><td>Qwen3.5-4B</td><td>4B</td><td>0.55</td><td>0.66</td><td>0.0109</td><td>0.0393</td><td>0.146</td><td>0.0049</td><td>1.31</td><td>2.12</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>0.56</td><td>0.68</td><td>0.0107</td><td>0.0396</td><td>0.144</td><td>0.0047</td><td>1.30</td><td>2.09</td></tr></table>

As the Phys-stream truncation length $K _ { p }$ increases, rRMSE, MPJPE, and rFID all improve consistently. We use $K _ { p } { = } 2 5$ as the default setting because it already offers a strong compactness-fidelity trade-off while matching the main tokenizer configuration.

Table 4 also shows that increasing the Phys-stream truncation length $K _ { p }$ consistently improves reconstruction quality. As $K _ { p }$ increases from 10 to 30, both rRMSE and MPJPE decrease, while rFID drops from 1.340 to 0.138. We therefore use $K _ { p } { = } 2 5$ as the default setting, since it already provides a strong compactness-fidelity trade-off while matching the main tokenizer configuration.

## 3.7 Ablation Studies

We conduct two ablation studies on MBench to examine the effect of backbone scale and the Physstream truncation length $K _ { p } .$ . Since HumanML3D does not provide visual observations, we focus the ablation analysis on the scene-conditioned ViMoGen-228K–MBench setting.

Backbone Scale. Table 5 shows that scaling up the Qwen3.5 backbone yields consistent but diminishing gains on MBench. The largest improvement comes from 0.8B to 2B, while the gains from 2B to 4B and 9B are relatively small. For some metrics, the 2B and 4B models appear unchanged after rounding, since only two decimal places are reported in those columns. One possible explanation is that, under the current data scale and training recipe, the available supervision is not sufficient to fully exploit substantially larger backbones. Moreover, because MotionVLA predicts a fixed DSFT tokenization rather than continuous motion directly, the effective information carried by the token representation may also limit how much additional capacity can be translated into measurable gains. As a result, a 2B model already captures most of the achievable improvement in the current setting, making it a favorable default choice in terms of both performance and efficiency.

DSFT Truncation Parameter $K _ { p }$ . Table 6 studies the effect of the Phys-stream truncation length $K _ { p }$ while fixing $K _ { b } { = } 5$ under the default 2B backbone. As $K _ { p }$ increases, Phys-stream energy coverage improves consistently, indicating that a larger frequency budget preserves more high-frequency physical dynamics. From $K _ { p } { = } 1 0$ to $K _ { p } { = } 2 5$ , this added physical capacity is accompanied by clear improvements on most MBench metrics, showing that richer physical signals benefit overall motion quality. However, a larger $K _ { p }$ also increases the motion sequence length, and the gains do not continue monotonically at $K _ { p } { = } 3 0$ where several metrics slightly degrade. We therefore use $K _ { p } { = } 2 5$ as the default setting, as it provides the best overall balance between physical detail preservation and sequence efficiency under the current 2B model scale.

Table 6: DSFT $K _ { p }$ ablation on MBench. $K _ { b } { = } 5$ fixed. Tok./Sample: average motion token count. ↑/↓: higher/lower is better.
<table><tr><td></td><td>Phys Coverage ↑</td><td>Tok./Sample</td><td>Motion-Cond. Consistency 个</td><td>Motion Generaliz.个</td><td>Jitter Degree↓</td><td>Dynamic Degree</td><td>Foot 个 Floating↓</td><td>Foot Sliding↓</td><td>Body Penetration ↓</td><td>Pose Quality↓</td></tr><tr><td> $K _ { p }$  10</td><td>~50%</td><td>438</td><td>0.49</td><td>0.58</td><td>0.0131</td><td>0.0338</td><td>0.171</td><td>0.0062</td><td>1.48</td><td>2.28</td></tr><tr><td>15</td><td>~63%</td><td>472</td><td>0.52</td><td>0.61</td><td>0.0121</td><td>0.0364</td><td>0.160</td><td>0.0054</td><td>1.41</td><td>2.21</td></tr><tr><td>20</td><td>~73%</td><td>507</td><td>0.54</td><td>0.63</td><td>0.0116</td><td>0.0375</td><td>0.154</td><td>0.0051</td><td>1.37</td><td>2.17</td></tr><tr><td>25</td><td>~80%</td><td>541</td><td>0.55</td><td>0.66</td><td>0.0110</td><td>0.0419</td><td>0.149</td><td>0.0049</td><td>1.34</td><td>2.14</td></tr><tr><td>30</td><td>~85%</td><td>569</td><td>0.53</td><td>0.64</td><td>0.0113</td><td>0.0379</td><td>0.152</td><td>0.0052</td><td>1.38</td><td>2.16</td></tr></table>

Table 7: Human preference study (%) on 100 prompts $\times 5$ experts. Ours: MotionVLA preferred; Tie: comparable; Base: ViMoGen preferred.
<table><tr><td></td><td>Expert 1</td><td>Expert 2</td><td>Expert 3</td><td>Expert 4</td><td>Expert 5</td><td>Avg.</td></tr><tr><td>Ours (MotionVLA preferred)</td><td>72</td><td>58</td><td>69</td><td>55</td><td>66</td><td>64.0</td></tr><tr><td>Tie (Same taste)</td><td>17</td><td>26</td><td>19</td><td>28</td><td>20</td><td>22.0</td></tr><tr><td>Base (Vimogen preferred)</td><td>11</td><td>16</td><td>12</td><td>17</td><td>14</td><td>14.0</td></tr></table>

## 3.8 Simulation, Deployment and Human Preference Study

To complement automatic metrics, we evaluate MotionVLA in MuJoCo simulation, deploy it on a Unitree G1 EDU humanoid under the text-to-motion setting, and conduct a blinded human preference study against ViMoGen [13]. Given a text prompt, MotionVLA generates motion tokens that DSFT decodes into real-time joint-angle trajectories. Five domain experts assess 100 anonymized textconditioned motion pairs, producing 500 pairwise comparisons. Detailed simulation, deployment, and evaluation protocols are provided in Appendix F and E.

As summarized in Table 7, MotionVLA is preferred in 64.0% of comparisons, compared with 14.0% for ViMoGen and 22.0% ties, indicating a clear perceptual advantage in overall motion quality.

## 4 Discussions and Conclusions

In this work, we address humanoid motion generation through coordinated innovations in tokenizer design, autoregressive modeling, and evaluation. (1) We introduce DSFT, a dual-stream frequencydomain tokenizer that decomposes motion into Base and Phys streams, motivated by the observation that pose-related and dynamic signals exhibit different spectral characteristics and therefore should not be forced into a single shared tokenization space. By assigning separate frequency budgets to the two streams, DSFT better preserves both motion semantics and high-frequency physical dynamics. (2) Built on top of this tokenizer, MotionVLA adapts a standard vision-language autoregressive backbone to unified motion generation, showing that multimodal controllability and physical motion quality can be improved within a simple sequence modeling framework. (3) Experiments on HumanML3D and ViMoGen–MBench show strong performance on both automatic metrics and human preference evaluation, supporting the effectiveness of frequency-aware dual-stream tokenization for both multimodal generation and transfer to standard text-to-motion settings. More broadly, these results suggest that effective motion tokenization depends not only on compression or pointwise reconstruction quality, but also on how motion signals are organized before discretization. A detailed discussion of related work is provided in Appendix A.

Limitations and Future Work. Our current study focuses on a lightweight 2B backbone and a limited set of benchmarks, and therefore does not yet support broader conclusions about scaling behavior or cross-dataset generalization. In addition, the current framework uses a fixed stream partition, fixed truncation lengths, and a predefined Base-to-Phys generation order, which may not be optimal for all motion types or sequence lengths. Future work will extend the evaluation to larger backbones, broader datasets, and more adaptive tokenization and dependency modeling schemes.

## References

[1] Yuxuan Bian, Ailing Zeng, Xuan Ju, Xian Liu, Zhaoyang Zhang, Wei Liu, and Qiang Xu. MotionCraft: Crafting whole-body motion with plug-and-play multimodal controls. AAAI Conference on Artificial Intelligence, 2024.

[2] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv.org, 2024.

[3] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, K. Choromanski, Tianli Ding, Danny Driess, Kumar Avinava Dubey, Chelsea Finn, Peter R. Florence, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, 2023.

[4] Jungbin Cho, Junwan Kim, Jisoo Kim, Minseo Kim, Mingu Kang, Sungeun Hong, Tae-Hyun Oh, and Youngjae Yu. DisCoRD: Discrete tokens to continuous motion via rectified flow decoding. In arXiv.org, 2024. Highlight.

[5] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. MotionLCM: Real-time controllable motion generation via latent consistency model. In European Conference on Computer Vision, 2024.

[6] Chenyang Gu, Mingyuan Zhang, Haozhe Xie, Zhongang Cai, Lei Yang, and Ziwei Liu. Bridging semantic and kinematic conditions with diffusion-based discrete motion tokenizer. arXiv preprint arXiv:2603.19227, 2026.

[7] Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. MoMask: Generative masked modeling of 3D human motions. In Computer Vision and Pattern Recognition, 2023.

[8] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3D human motions from texts. In Computer Vision and Pattern Recognition, pages 5142–5151. IEEE, 2022.

[9] Chuan Guo, X. Zuo, Sen Wang, and Li Cheng. TM2T: Stochastic and tokenized modeling for the reciprocal generation of 3D human motions and texts. In European Conference on Computer Vision, pages 580–597. Springer Nature Switzerland, 2022.

[10] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. MotionGPT: Human motion as a foreign language. In Neural Information Processing Systems, pages 20067–20079. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023.

[11] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, A. Balakrishna, Suraj Nair, Rafael Rafailov, E. Foster, Grace Lam, Pannag R. Sanketi, et al. OpenVLA: An open-source vision-language-action model. Conference on Robot Learning, 2024.

[12] Zekun Li, Sizhe An, Chengcheng Tang, Chuan Guo, Ivan Shugurov, Linguang Zhang, Amy Zhao, Srinath Sridhar, Lingling Tao, and Abhay Mittal. LLaMo: Scaling pretrained language models for unified motion understanding and generation with continuous autoregressive tokens. arXiv.org, 2026.

[13] Jing Lin, Ruisi Wang, Junzhe Lu, Ziqi Huang, Guorui Song, Ailing Zeng, Xian Liu, Chen Wei, Wanqi Yin, Qingping Sun, et al. The quest for generalizable motion generation: Data, model, and evaluation. In arXiv.org, 2025. arXiv:2510.26794.

[14] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. International Conference on Learning Representations, 2022.

[15] Mengyuan Liu, Sheng Yan, Yong Wang, Yingjie Li, Gui-Bin Bian, and Hong Liu. MoSa: Motion generation with scalable autoregressive modeling. arXiv.org, 2025.

[16] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. International Conference on Learning Representations, 2022.

[17] Stephane Mallat. A wavelet tour of signal processing. Elsevier, 1998.

[18] Zichong Meng, Yiming Xie, Xiaogang Peng, Zeyu Han, and Huaizu Jiang. Rethinking diffusion for text-driven human motion generation: Redundant representations and evaluation. In Computer Vision and Pattern Recognition, 2024.

[19] Aäron van den Oord, O. Vinyals, and K. Kavukcuoglu. Neural discrete representation learning. In Neural Information Processing Systems, 2017.

[20] Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, et al. Open x-embodiment: Robotic learning datasets and rt-x models. In ICRA, 2024.

[21] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In Computer Vision and Pattern Recognition, pages 10967–10977. IEEE, 2019.

[22] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient action tokenization for vision-language-action models. Robotics, 2025.

[23] Mathis Petrovich, Michael J. Black, and Gül Varol. TEMOS: Generating diverse human motions from textual descriptions. In European Conference on Computer Vision, pages 480– 497. Springer Nature Switzerland, 2022.

[24] R. Pfeifer and F. Iida. Embodied artificial intelligence: Trends and challenges. Embodied Artificial Intelligence, 2003.

[25] Yijie Qian, Juncheng Wang, Yuxiang Feng, Chao Xu, Wang Lu, Yang Liu, Baigui Sun, Yiqiang Chen, Yong Liu, and Shujun Wang. Think before you move: Latent motion reasoning for text-to-motion generation. arXiv.org, 2025.

[26] Junlong Ren, Gangjian Zhang, Honghao Fu, Pengcheng Wu, and Hao Wang. WaMo: Waveletenhanced multi-frequency trajectory analysis for fine-grained text-motion retrieval. arXiv.org, 2025.

[27] Junyu Shi, Lijiang Liu, Yong Sun, Zhiyuan Zhang, Jinni Zhou, and Q. Nie. GenM3: Generative pretrained multi-path motion model for text conditional human motion generation. In arXiv.org, pages 13129–13139. IEEE, 2025.

[28] O. Team, Dibya Ghosh, H. Walke, Karl Pertsch, Kevin Black, Oier Mees, S. Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. In Robotics: Science and Systems. Robotics: Science and Systems Foundation, 2024.

[29] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit Haim Bermano. Human motion diffusion model. In International Conference on Learning Representations, 2022.

[30] Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033. IEEE, 2012.

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Neural Information Processing Systems, 30, 2017.

[32] Bizhu Wu, Jinheng Xie, Keming Shen, Zhe Kong, Jianfeng Ren, Ruibin Bai, Rong Qu, and LinLin Shen. MG-MotionLLM: A unified framework for motion comprehension and generation across multiple granularities. In Computer Vision and Pattern Recognition, pages 27849–27858. IEEE, 2025.

[33] Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yue Zhou, Ziyong Feng, Xiaowei Zhou, Sida Peng, and Jingbo Wang. MotionStreamer: Streaming motion generation via diffusionbased autoregressive model in causal latent space. In arXiv.org, pages 10086–10096. IEEE, 2025.

[34] Sheng Yan, Yong Wang, Xin Du, Junsong Yuan, and Mengyuan Liu. Language-guided transformer tokenizer for human motion generation. arXiv.org, 2026.

[35] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Shaoli Huang, Yong Zhang, Hongwei Zhao, Hongtao Lu, and Xi Shen. T2M-GPT: Generating human motion from textual descriptions with discrete representations. In Computer Vision and Pattern Recognition, pages 14730–14740. IEEE, 2023.

[36] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. MotionDiffuse: Text-driven human motion generation with diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

[37] Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, and Ziwei Liu. FineMoGen: Fine-grained spatio-temporal motion generation and editing. In Neural Information Processing Systems, pages 13981–13992. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023.

[38] Yaqi Zhang, Di Huang, B. Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Q. Chu, Nenghai Yu, and Wanli Ouyang. MotionGPT: Finetuned LLMs are general-purpose motion generators. In AAAI Conference on Artificial Intelligence, 2023.

[39] Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhan Chen, Xiaowei Zhang, Yuanfang Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, et al. A survey on vision-language-action models: An action tokenization perspective. arXiv.org, 2025.

[40] Bingfan Zhu, Biao Jiang, Sunyi Wang, Shixiang Tang, Tao Chen, Linjie Luo, Youyi Zheng, and Xin Chen. MotionGPT3: Human motion as a second modality. arXiv.org, 2025.

## A Related Work

## A.1 Human Motion Generation

Text-driven human motion generation aims to synthesize realistic 3D human motion sequences from natural language descriptions. Mainstream methods build generation frameworks upon VQ-VAE discretization and autoregressive Transformers. T2M-GPT [35] first combines VQ-VAE with GPT next-token prediction, establishing the representative paradigm in this line of work. MotionGPT [10] treats motion as a foreign language and jointly trains multiple motion tasks under a unified language model. MoMask [7] introduces Residual VQ (RVQ) hierarchical codebooks and Masked Transformers, reducing the FID to 0.045, while MG-MotionLLM [32] builds a multi-granularity motion-language framework with T5 as the backbone, extending semantic granularity from the sequence level to the segment level. GenM3 [27] collects 11 datasets and employs a multi-expert VQ-VAE along with a multi-path Transformer to handle data heterogeneity, achieving a state-of-the-art FID of 0.035. MoSa [15] proposes RQ-VAE and a scalable autoregressive framework, outperforming the 10-step inference speed of MoMask by 27%. More recently, MotionGPT3 [40] and LLaMo [12] shift towards continuous latent space autoregression to reduce the motion jitter caused by discrete quantization. Another line of work adopts diffusion models as the backbone. MDM [29] establishes the Transformer-based diffusion framework, and MotionDiffuse [36] extends it to support fine-grained control at the body-part level. FineMoGen [37] achieves fine-grained spatio-temporal synthesis using Spatial-Temporal Mixed Attention (SAMI) and sparse MoE, while MotionStreamer [33] combines diffusion with autoregression in a causal latent space for streaming generation. Recent systematic comparisons [18] indicate that VQ-based autoregressive methods still hold an overall advantage on standard metrics such as FID and R-Precision.

## A.2 Motion Tokenization and Representation

High-quality discrete motion representations form the basis of autoregressive generation methods. VQ-VAE [19] introduces discrete bottlenecks into sequence modeling, enabling efficient compression and generation of continuous data. T2M-GPT [35] adapts this paradigm to the human motion domain, showing that motion can be effectively tokenized for text-driven generation. MoMask [7] enhances hierarchical representation capabilities with Residual Vector Quantization (RVQ) [19], enabling multi-scale semantic abstraction. In the frequency domain, FAST [22] constructs an efficient robot action tokenizer by combining Discrete Cosine Transform (DCT) with Byte-Pair Encoding (BPE), showing that frequency-domain representations capture high-frequency fine-grained control signals more effectively. Following this direction, WaMo [26] applies wavelet multi-scale decomposition [17] to motion trajectory analysis, further validating that multi-frequency decomposition improves finegrained motion-text correspondence. Recent work also explores the integration of semantic guidance into the tokenization process: LG-Tok [34] proposes a language-guided Transformer tokenizer that introduces semantic alignment during the encoding stage, while MoTok [6] uses a diffusion decoder to decouple semantic abstraction from fine-grained reconstruction, maintaining high-fidelity reconstruction quality with single-layer tokens. DisCoRD [4] approaches the problem from the decoding end, mapping discrete tokens back to continuous motion via rectified flow [16] to partially reduce inter-frame jitter, yet leaves the structural cause (single-codebook quantization) intact at the representation level. These methods collectively point to a core open problem: simultaneously capturing semantic structure (e.g., action labels, phase transitions) and physical dynamics (e.g., velocity, acceleration, contact forces) during tokenization. Since these two types of signals occupy overlapping frequency bands, disentangling them within a single quantization space is difficult. Prior works [7, 6, 4] seek this balance within a unified representation; in contrast, DS-FAST resolves it orthogonally by separating the two signal types before quantization, thereby preserving both high-level structure and low-level motion quality.

## A.3 Vision-Language-Action Models

Vision-Language-Action (VLA) models unify visual perception, language understanding, and action generation within a single end-to-end framework, representing a core research direction in embodied AI [39, 24]. Among representative works, OpenVLA [11], trained with 7B parameters on approximately 970,000 multi-robot trajectories from the Open X-Embodiment dataset [20], demonstrates strong generalization across diverse robotic platforms and manipulation tasks. π0 [2] achieves end-toend generation of 50 Hz high-frequency dexterous manipulation using a flow matching paradigm [14], setting a new standard for fine manipulation tasks that demand precise temporal control. Other notable efforts include RT-2 [3], which uses vision-language models (VLMs) for grounded robot control, and Octo [28], a generalist robot policy trained on large-scale multi-embodiment data. A recent survey [39] identifies that the quality of discrete action representations remains one of the core bottlenecks constraining the fine-grained control capabilities of VLAs, particularly in tasks requiring high-frequency feedback and multi-modal conditioning. Unlike the aforementioned VLA works, which primarily focus on low-level robot control with short-horizon actions (typically <5 seconds), MotionVLA in this paper targets vision- and text-conditioned fine-grained human motion generation, a domain characterized by higher semantic complexity, longer temporal durations, and richer multimodal conditioning. Specifically, human motions are more diverse and nuanced than robotic actions, spanning a richer space of activities, emotions, and styles; motion sequences often span 10–30 seconds, requiring long-range temporal coherence; and generation must be grounded simultaneously in both visual context (e.g., scene layout, object affordances) and linguistic descriptions. By addressing these challenges, MotionVLA extends the VLA framework to the domain of human motion synthesis, connecting robotic action generation and human-centric animation.

## B Detailed datasets and metrics

ViMoGen-228K and MBench. ViMoGen-228K [13] is a large-scale multimodal motion dataset containing 228K motion sequences collected from three sources: optical motion capture, in-thewild video annotation, and synthetic generation. In our experiments, MotionVLA is trained on the ViMoGen-228K training split and evaluated on MBench [13], following the official protocol. MBench contains 450 held-out prompts and reports eight fine-grained evaluation dimensions, including Motion-Condition Consistency, Motion Generalizability, Jitter Degree, Dynamic Degree, Foot Floating, Foot Sliding, Body Penetration, and Pose Quality.

HumanML3D. HumanML3D [8] is a standard benchmark for text-driven human motion generation. In our experiments, we train and evaluate MotionVLA on the official HumanML3D split under the standard text-to-motion setting. Because HumanML3D does not provide visual inputs, the model is used in text-only mode. Following prior work [8], we report FID, R-Precision (Top-1/2/3), MM-Dist, Diversity, and MModality using the official pretrained feature extractor.

Table 8: Baselines across three paradigms. † denotes visual conditioning (MotionVLA only). All prior methods are text-driven.
<table><tr><td>Method</td><td>Venue</td><td>Paradigm</td><td>Tokenizer</td></tr><tr><td colspan="4">Discrete Autoregressive</td></tr><tr><td>T2M-GPT[35]</td><td>CVPR 2023</td><td>AR (GPT)</td><td>VQ-VAE</td></tr><tr><td>MoMask [7]</td><td>CVPR 2024</td><td>Masked AR</td><td>RVQ</td></tr><tr><td>MG-MotionLLM[32]</td><td>CVPR 2025</td><td>AR (T5)</td><td>vQ</td></tr><tr><td>GenM3 [27]</td><td>ICCV 2025</td><td>Multi-path AR</td><td>Multi-expert VQ</td></tr><tr><td colspan="4">Diffusion /Flow Matching</td></tr><tr><td>MDM[29]</td><td>ICLR 2023</td><td>Diffusion</td><td></td></tr><tr><td>ViMoGen [13]</td><td>ICLR 2026</td><td>Flow Matching</td><td></td></tr><tr><td colspan="4">Improved Tokenization</td></tr><tr><td>DisCoRD[4]</td><td>ICCV 2025</td><td>AR + Flow</td><td>VQ + Rect. Flow</td></tr><tr><td colspan="4">MotionVLA (Ours)+</td></tr></table>

Evaluation Metrics. On MBench, we follow the official benchmark protocol and report eight dimensions. Motion-Condition Consistency measures whether the generated motion matches the input condition; Motion Generalizability evaluates semantic plausibility and diversity under unseen prompts; Jitter Degree measures local temporal instability; Dynamic Degree evaluates motion expressiveness; Foot Floating and Foot Sliding quantify contact realism; Body Penetration measures self-intersection artifacts; and Pose Quality evaluates overall pose naturalness.

On HumanML3D, we follow the standard text-to-motion evaluation protocol. FID measures the distribution distance between generated and ground-truth motions in the feature space; R-Precision evaluates text-motion retrieval accuracy; MM-Dist measures multimodal alignment distance; Diversity measures sample diversity across generated motions; and MModality evaluates motion variation under the same text condition.

Data Splits and Protocol. For ViMoGen-228K, we use the official training split for model training and report results on the MBench evaluation set. For HumanML3D, we follow the official train/test split and the standard evaluation pipeline. All reported numbers are obtained under the corresponding benchmark protocols, and additional implementation details are provided in Appendix C.

## C Implementation Details

DS-FAST Tokenizer Training. For ViMoGen, the tokenizer is trained on 41,971 in-the-wild video motions and 50,000 randomly sampled optical motion-capture sequences from AMASS. The 276-dim vector is split into Base $( \dot { D _ { b } } \mathrm { = } 2 0 \bar { 1 } )$ and Phys $( D _ { p } { = } 7 5 )$ by index slicing. For HumanML3D, the tokenizer is trained on the 23,384 official training sequences; the 263-dim vector is split into Base $( D _ { b } { = } 1 9 0$ , indices [7:197]) and Phys $( D _ { p } { = } 7 3 $ , indices [0:7] ∪ [197:263]). In both settings, DCT truncation lengths are $\dot { K _ { b } } { = } 5$ and $K _ { p } { = } 2 5$ , and two independent BPE vocabularies of size 4,096 are trained per dataset. The trained tokenizers are then applied to all training samples: 212,913 for ViMoGen (41,971 in-the-wild video + 170,942 optical mocap) and 23,384 for HumanML3D.

Phase 1 — Embedding Cold Start. The 8,195 newly added motion token embeddings are randomly initialized. All Qwen3.5 transformer layers are frozen; only embed\_tokens and lm\_head are trained for 500 steps with learning rate $1 \times 1 0 ^ { - 3 }$ and the Adafactor optimizer to warm up the motion token embedding space.

Phase 2 — LoRA Fine-Tuning. Starting from the Phase-1 checkpoint, LoRA adapters are applied to all linear projections, while embed\_tokens and lm\_head continue to be updated as full saved modules. Training runs for 10 epochs on 8×H100 (80 GB) GPUs. All base Q wenweights remain frozen throughout.

Training Data. ViMoGen mixes In-the-Wild Video (41,971 samples, image + text) and Optical Mo-Cap (170,942 samples, text-only, real GT from AMASS). HumanML3D uses the official train/val/test split (23,384 / 1,460 / 4,384) with text-only inputs.

Inference. The model runs on a single H100 (80 GB) GPU. The phase-aware logit mask constrains autoregressive decoding to Base tokens before SEP and Phys tokens after SEP. Generated tokens are decoded via BPE inverse mapping followed by IDCT to reconstruct the full motion sequence (276-dim for ViMoGen, 263-dim for HumanML3D).

Hyperparameters. Table 9 lists the complete configuration.

## D DS-FAST Feature Partition Details

The Base/Phys partition assigns each dimension to the stream whose frequency profile it matches, determined by the low-frequency energy ratio (LFR) threshold of 0.6. Table 10 gives the complete per-field mapping for both ViMoGen (276-dim, SMPL+X) and HumanML3D (263-dim).

In the ViMoGen representation, velocity fields occupy non-contiguous index ranges: joints\_vel ([192:258]) is interleaved between the Base joints block ([126:192]) and the Base root\_orient\_6d block ([258:264]). DS-FAST extracts both streams by explicit index slicing prior to DCT, ensuring that each stream contains only physically homogeneous features.

In HumanML3D, the LFR boundary falls within two semantic fields rather than between them: the 21-joint local\_pos block is split so that the root joint position ([4:7]) enters the Phys stream due to its high-frequency root dynamics, while the remaining non-root positions ([7:67]) remain in Base; similarly, the first four elements of the 66-dim joint velocity block ([193:197]), which correspond to the root joint and exhibit low LFR, are assigned to the Base stream. These splits reflect the data-driven LFR criterion and do not require any manual field-boundary annotation.

Table 9: Full hyperparameter configuration for MotionVLA training.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>DS-FAST Tokenizer</td><td></td></tr><tr><td>Base stream dims  $D _ { b }$ </td><td>201(ViMoGen)/190 (HumanML3D)</td></tr><tr><td>Phys stream dims  $D _ { p }$ </td><td>75 (ViMoGen) / 73 (HumanML3D)</td></tr><tr><td>DCT truncation  $K _ { b }$ </td><td>5</td></tr><tr><td>DCT truncation  $K _ { p }$ </td><td>25</td></tr><tr><td>BPE vocabulary per stream</td><td>4.096</td></tr><tr><td>Vocabulary Expansion</td><td></td></tr><tr><td>Qwen3.5 original vocab</td><td>248,320</td></tr><tr><td>Total vocab after expansion</td><td>256,515</td></tr><tr><td>New motion token embeddings</td><td>8,195</td></tr><tr><td>Phase 1-Embedding Cold Start</td><td></td></tr><tr><td>Trainable parameters</td><td>embed_tokens,lm_head</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Optimizer</td><td>Adafactor</td></tr><tr><td>Steps</td><td>500</td></tr><tr><td>Phase 2-LoRA Fine-Tuning</td><td></td></tr><tr><td>LoRA rank r</td><td>32</td></tr><tr><td>LoRA scaling α</td><td>64</td></tr><tr><td>Applied to</td><td>All linear projections</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td>Cosine decay</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Epochs</td><td>10</td></tr><tr><td>Per-device batch size</td><td>3</td></tr><tr><td>Gradient accumulation steps</td><td>2</td></tr><tr><td>Effective batch size</td><td>48 (8 GPUs ×3 × 2)</td></tr><tr><td>Max sequence length</td><td>4,700 tokens</td></tr><tr><td>Hardware</td><td>8× H100 (80GB)</td></tr><tr><td>Training Data</td><td></td></tr><tr><td>ViMoGen total samples</td><td>212,913</td></tr><tr><td>In-the-Wild Video</td><td>41,971 (image + text)</td></tr><tr><td>Optical MoCap</td><td>170,942 (text-only)</td></tr><tr><td>HumanML3D training samples</td><td>23,384 (text-only)</td></tr><tr><td></td><td></td></tr></table>

## E Human Preference Analysis

To complement the quantitative benchmarks reported in Section 3, we conducted a human preference study to assess the perceptual quality of motions generated by MotionVLA. Evaluations were carried out through a custom web-based interface (Figure 5) that presented anonymized side-by-side motion pairs to domain experts.

Study Design. We invited five domain experts in human motion analysis and character animation to participate in a blinded pairwise preference evaluation. Each expert assessed 100 text-conditioned motion pairs, where each pair comprised one motion generated by MotionVLA and the corresponding output from ViMoGen [13]. To enable a comprehensive visual assessment, every motion clip was rendered from two camera perspectives—front view and side view—yielding four synchronized video clips per evaluation trial. Each clip was rendered as a 3-second video at 20 fps with the conditioning text prompt displayed above both clips. The left/right assignment of MotionVLA and the baseline was randomized per trial; experts were not informed of which method produced either clip. All participants volunteered without monetary compensation. As the study involved only viewing and comparing AI-generated skeletal motion clips with no collection of personal data, IRB approval was not required under our institutional guidelines.

Table 10: Per-field breakdown of the ViMoGen 276-dim and HumanML3D 263-dim motion vectors into Base $( D _ { b } ,$ , position/rotation) and Phys $( D _ { p }$ , velocity) streams. Index ranges index into the flat per-frame feature vector d.
<table><tr><td>Field</td><td>Semantics</td><td>Dims</td><td>Index range</td><td>Stream</td></tr><tr><td colspan="5">ViMoGen (276dims，] Base  $D _ { b } { = } 2 0 1$  Phys  $D _ { p } { = } 7 5 )$ </td></tr><tr><td colspan="5">Base stream—position and rotation</td></tr><tr><td>body_pose_6d</td><td>21 joints × 6D rotation</td><td>126</td><td>[:,0:126]</td><td>Base</td></tr><tr><td>joints</td><td>22 joints × XYZ position</td><td>66</td><td>[:,126:192]</td><td>Base</td></tr><tr><td>root_orient_6d</td><td>Root global orientation (6D)</td><td>6</td><td>[:,258:264]</td><td>Base</td></tr><tr><td>root_trans</td><td>Root global translation (XYZ)</td><td>3</td><td>[:,270:273]</td><td>Base</td></tr><tr><td colspan="5">Phys stream-velocity</td></tr><tr><td>joints_vel</td><td>22 joints × XYZ velocity</td><td>66</td><td>[:,192:258]</td><td>Phys</td></tr><tr><td>root_vel_6d</td><td>Root rotational velocity (6D)</td><td>6</td><td>[:,264:270]</td><td>Phys</td></tr><tr><td>root_trans_vel</td><td>Root translational velocity (XYZ)</td><td>3</td><td>[:,273:276]</td><td>Phys</td></tr><tr><td colspan="5">HumanML3D (263 dims, Base  $D _ { b } { = } 1 9 0 ,$  Phys  $D _ { p } { = } 7 3 )$ </td></tr><tr><td>Base stream -position and rotation</td><td></td><td>60</td><td>[:,7:67]</td><td></td></tr><tr><td>local_pos local_rot</td><td>Non-root joints XYZ position (20×3) 21 joints × 6D rotation</td><td>126</td><td>[:,67:193]</td><td>Base Base</td></tr><tr><td>root_joint_vel</td><td>Root joint velocity (low LFR)</td><td>4</td><td>[:,193:197]</td><td>Base</td></tr><tr><td>Phys stream-velocity and root dynamics</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5"></td></tr><tr><td>root_ang_vel</td><td>Root angular velocity (Y-axis)</td><td>1</td><td>[:,0:1]</td><td>Phys</td></tr><tr><td>root_lin_vel</td><td>Root linear velocity (X, Z)</td><td>2</td><td>[:,1:3]</td><td>Phys</td></tr><tr><td>root_height</td><td>Root height (Y)</td><td>1</td><td>[:,3:4]</td><td>Phys</td></tr><tr><td>root_pos</td><td>Root joint XYZ position</td><td>3</td><td>[:,4:7]</td><td>Phys</td></tr><tr><td>local_vel</td><td>Non-root joint velocities (62 dims)</td><td>62</td><td>[:,197:259]</td><td>Phys</td></tr><tr><td>foot_contact</td><td>Foot contact binary labels</td><td>4</td><td>[:,259:263]</td><td>Phys</td></tr><tr><td colspan="3">Total—ViMoGen</td><td>276 [:,0:276] 263</td><td></td></tr><tr><td colspan="3">Total-HumanML3D</td><td>[:,0:263]</td><td></td></tr></table>

Evaluation Protocol. For each pair, the expert selected one of three options:

Good (G): The left clip is clearly better overall. Same (S): The two clips are of comparable quality.   
Bad (B): The right clip is clearly better overall.

After de-anonymizing, G indicates a preference for MotionVLA, S indicates no clear preference, and B indicates a preference for the baseline. Preference rates (%) are reported over all 5 experts × 100 prompts = 500 comparisons.

Aggregate Results. Table 11 reports the GSB preference rates of MotionVLA against ViMoGen, aggregated across all 500 comparisons. MotionVLA receives a clear majority preference $\mathrm { ( G } = 6 4 . 0 \% \mathrm { ) }$ ), while only 14.0% of evaluations favor the baseline, demonstrating a substantial and consistent advantage in perceived motion quality across both front and side views.

Table 11: GSB pairwise preference study results (%). G = MotionVLA preferred; S = no preference; B = baseline preferred. Results aggregated over 5 experts × 100 prompts = 500 comparisons.
<table><tr><td>MotionVLA vs.</td><td></td><td></td><td>G(Ours Better)S (Same)B (Baseline Better)</td></tr><tr><td>ViMoGen [13]</td><td>64.0</td><td>22.0</td><td>14.0</td></tr></table>

## F Simulation and Real-Robot Demonstration

## F.1 MuJoCo Simulation

All qualitative motion visualizations in this paper are produced using MuJoCo [30], a physics engine widely used in locomotion and character animation research.

![](images/c97f1e3d7e545d888637d16e3678625da74f916db0a04e7ec4f6c899a35db065.jpg)  
Figure 5: Screenshot of the GSB evaluation interface used in our human preference study. Each trial displays two motion clips—Motion A and Motion B—rendered from front and side viewpoints simultaneously, with the conditioning text prompt shown above. Experts select one of three options: G (left clip better), S (comparable quality), or B (right clip better). The right panel tracks per-question completion progress across five evaluation dimensions.

![](images/77d8510f91a87332a9539fe29ba3f82a4c8e2fcc5abba6ec0c65de3cdb306ae6.jpg)  
Figure 6: Capsule-skeleton rendering produced by MuJoCo. The motion decoded from DS-FAST tokens is converted to SMPL-X joint positions and rendered with per-sequence ground-contact alignment.

Pipeline. The generated motion token sequence is first decoded by DS-FAST into a per-frame motion vector (276-dim for ViMoGen, 263-dim for HumanML3D) through inverse BPE followed by inverse DCT. This vector is then converted to SMPL-X [21] body parameters (global orientation, 22-joint body pose, and root translation). In MuJoCo, each frame is visualized as a capsule-based skeleton, where bone segments connecting adjacent joints are drawn as capsule geometries and joint centers are marked by spheres. The scene is evaluated with mj\_forward in pure kinematic mode (no physical integration), so the rendered motion exactly reflects the model output without any simulation correction.

Rendering Configuration. Frames are rendered with the MuJoCo offscreen renderer (EGL backend) at 1280×1024 resolution and composited at 20 fps under a fixed side-view camera. To ensure plausible ground contact, a per-sequence vertical offset aligns the lowest foot position with the floor plane.

![](images/736a8927db626e21ccee574502149a616058acc9ad6a29438b9b24763d05a91c.jpg)

![](images/47f5faefec952760c306822557aeac7e40dd242c5b4128e72b9858952fff6927.jpg)

![](images/f1c1b1c2c9ce0179454960ccb70300eb0708b350fe33a55078ad0fbb9873cebe.jpg)  
(a) The person walks straight ahead to the other end of the room.

![](images/5251d2fd92379b3da791af988a13eee9e9cde6ebf748821491118334f4bc0045.jpg)

![](images/c1406a63a5b94edef19f3b58400045afd1385e75fa2382581c9a2756e9be6cfc.jpg)

![](images/f9c60c0162a703ee5b7db7f4934d178d0cc8cc3457aa75e022c5753422056671.jpg)  
(b) The person turns and then walks to the end of the room.

![](images/ff2a4bfd0b92f6f4a90aeee49774ac8b00484985433897645d635fbda1d49710.jpg)

![](images/c2c607445ea467e6cd6d373ebdfaf28060bfbee72b2e65ec67e5199a71fce124.jpg)  
(c) The person walks straight ahead and then turns.

![](images/d356b244d6ae320859313d94e7d11aa63f34a80a38e10e96a80d93713c5e87d2.jpg)  
Figure 7: Real-robot deployment of MotionVLA on a Unitree G1 EDU humanoid robot. Each row shows three exocentric frames from one text-conditioned motion execution, captured at different time steps.

## F.2 Real-Robot Deployment

We further deploy MotionVLA on a Unitree G1 EDU humanoid robot to verify that the generated motions can be executed on real hardware. Given a text prompt, the model produces a motion token sequence, which DS-FAST decodes into joint-angle trajectories. These trajectories are retargeted to the G1 joint configuration and executed in real time.

Figure 7 presents three deployment examples. Each row corresponds to one text prompt, showing three exocentric frames captured at successive time steps.

## G Case Study: Scene-Conditioned Motion Generation

## End-to-End Generation Example #1

## Text prompt

“Generate motion for: The person takes off a shirt and puts it on their head, then bends down to pick up something from the ground.”

![](images/8185c265a127c514b8aa6c5fd1fa6e13c7924059bcb1755e484ab61f9abb1f8c.jpg)

Generated token sequence. Structural markers are highlighted: <mot\_bos> opens the sequence, <mot\_sep> separates Base and Phys streams, <mot\_eos> closes it. Blue = Base stream (bi); orange = Phys stream $( p _ { j } )$ .

```xml
<mot_bos> <mot_b_1496> <mot_b_2644> <mot_b_1521> <mot_b_1410>
<mot_b_1600> <mot_b_1506> <mot_b_1477> <mot_b_1457> <mot_b_1515>
<mot_b_1477> <mot_b_1528> <mot_b_1494> <mot_b_1474> <mot_b_1547>
<mot_b_1422> <mot_b_1585> <mot_b_1494> <mot_b_1624> <mot_b_1506>
<mot_sep> <mot_p_0298> <mot_p_0908> <mot_p_0318> <mot_p_0413>
<mot_p_0431> <mot_p_0299> <mot_p_0336> <mot_p_0458> <mot_p_0240>
<mot_p_2556> <mot_p_2761> <mot_p_3571> · · · <mot_eos>
```

Reconstructed motion. Decoded by DS-FAST (inverse BPE → inverse DCT per stream → concatenate) from the token sequence above.

![](images/4aae50a908cfa359272a3ecab9fee9f71c956a2f65c0d7fbfa3ca3a35e4c1c02.jpg)  
Reconstructed motion sequence visualized at uniform time intervals

## End-to-End Generation Example #2

## Text prompt

“Generate motion for: The man walks towards the camera.”

![](images/b165bf1af59225bad5e02471b4d87dc147c61903c80117000f7d9ebe5afed695.jpg)  
Scene image input

Generated token sequence. Structural markers are highlighted: <mot\_bos> opens the sequence, <mot\_sep> separates Base and Phys streams, <mot\_eos> closes it. Blue = Base stream (bi); orange = Phys stream (pj ).

```xml
<mot_bos> <mot_b_1802> <mot_b_1570> <mot_b_1783> <mot_b_2401>
<mot_b_1802> <mot_b_1416> <mot_b_1735> <mot_b_1423> <mot_b_1549>
<mot_b_1802> <mot_b_1530> <mot_b_1796> <mot_b_1413> <mot_b_1483>
<mot_b_1795> <mot_b_2200> <mot_b_1754> <mot_b_1410>
<mot_b_1543> · · · <mot_sep> <mot_p_0243> <mot_p_0247> <mot_p_1751>
<mot_p_1152> <mot_p_0857> <mot_p_1152> <mot_p_0857> <mot_p_0493>
<mot_p_1152> <mot_p_0857> <mot_p_0493> · · · <mot_eos>
```

Reconstructed motion. Decoded by DS-FAST (inverse BPE → inverse DCT per stream → concatenate) from the token sequence above.

![](images/5bed497eae770f4d5af74f152bd2b8bfab80087685734509eece379e007d0422.jpg)  
Reconstructed motion sequence visualized at uniform time intervals