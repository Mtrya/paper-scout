# RepWAM: World Action Modeling with Representation Visual-Action Tokenizers

Junke Wang1, Qihang Zhang2, Shuai Yang2, Yiming Luo2 Yujun Shen2, Zuxuan Wu1∗, Yu-Gang Jiang1∗, Yinghao Xu2,3,∗

1Institute of Trustworthy Embodied AI, Fudan University 2Robbyant, Ant Group, 3Hongkong University of Science and Technology Web page: https://wdrink.github.io/RepWAM Code: https://github.com/wdrink/RepWAM

## Abstract

This work presents RepWAM (Representation World Action Model), a representation-centric world action model (WAM) built on representation visual-action tokenizers. Existing WAMs typically inherit reconstruction-oriented video tokenizers from pretrained video generation models. Although these tokenizers preserve visual fidelity, pixel reconstruction alone provides limited guidance for learning instruction-following dynamics that connect future prediction with robot control. To address this, we explore a semantic visual-action latent space for representation-centric world action modeling. Specifically, we train a representation visual-action tokenizer that maps visual inputs into aligned visual and latent action tokens. We then pretrain our WAM to jointly model future visual states and the latent actions that connect them under language instructions, followed by adaptation to real robot trajectories for closed-loop manipulation. Experiments on real-world manipulation tasks and simulation benchmarks show that RepWAM delivers strong performance across diverse manipulation settings, while ablations highlight the value of semantic visual-action tokenization over reconstruction-oriented alternatives. These results establish representation visual-action tokenization as a promising foundation for world action models and a step toward generalist robot policies.

Keywords: World Action Models, Visual Tokenizers, Latent Action Models

## 1 Introduction

The capability of an agent to model dynamics and derive behaviors is fundamentally determined by the representation through which it perceives the world [5, 12]. This matters most for world action models (WAMs) [20, 31, 36], which extend classical world modeling [1, 2, 16, 17] from passive future prediction to embodied control: a single latent space must both forecast visual dynamics and infer the robot actions that realize them. In current WAMs, however, this representation is often borrowed from pretrained video generators [28] that are optimized for pixel reconstruction rather than visual semantics or robot action.

This work asks: what kind of representation does a world action model actually need? Current WAM representations fall short along two distinct dimensions. On the visual side, reconstruction-driven video tokenizers [28, 29] spend latent capacity on low-level appearance such as background texture, while object identity, spatial relations, and interaction cues that drive manipulation are underrepresented. This leaves instruction-conditioned world modeling poorly grounded: language goals about specific objects or interactions must be inferred from a latent space dominated by appearance variation rather than task-relevant semantics [15]. On the action side, visual latents and motor commands reside in disjoint spaces, forcing an inverse dynamics model (IDM) to bridge this modality gap at every step [35]. Together, these failures leave WAM representations visually shallow and structurally decoupled from action.

Motivated by this, this work revisits the design of the latent space underlying world action models and proposes RepWAM, a representation-centric WAM built upon representation visual-action tokenizers. At its core, RepWAM introduces a semantic video tokenizer by aligning the latent space of a video autoencoder with a frozen visual foundation model [5]. Building on these semantically rich visual latents, we further learn a latent action tokenizer that captures manipulation-centric motions tightly coupled to object-level interactions. Together, the two tokenizers form a unified semantic visual-action tokenizer that narrows the modality gap between visual latents and robot actions. On top of this space, we pretrain a world action model that jointly forecasts future visual states and their corresponding latent actions, and then adapt it to real robot data for downstream manipulation.

We evaluate RepWAM on both real-world manipulation tasks and simulation benchmarks. Compared with vision-language-action (VLA) and WAN-pretrained WAM baselines, RepWAM delivers competitive closed-loop behavior and reaches 89.3 on Easy tasks and 88.4 on Hard tasks in RoboTwin 2.0. These results suggest that grounding visual and action latents in a shared semantic representation is a promising foundation for world action modeling.

## 2 Related Work

## 2.1 World Action Models

World action models [3, 14, 21, 34] extend visual world modeling to robot control by using predicted scene evolution or latent dynamics as an intermediate representation for action generation. Motus [3] introduces a mixture-of-transformer architecture that assigns understanding, video generation, inverse dynamics, and action prediction to specialized experts within a unified latent action world model, while its follow-up MotuBrain [25] further improves multi-view modeling, cross-embodiment action representation, and actiononly deployment for efficient robot control. Lingbot-VA [20] formulates WAM training through causal world modeling, jointly learning frame prediction and policy execution with an autoregressive diffusion model, and DreamZero [36] adapts a pretrained video generation model for joint video-action modeling, showing strong zero-shot generalization to unseen tasks, scenes, and embodiments. These methods demonstrate the benefit of predictive visual dynamics, but they also raise the question of whether explicit future generation is necessary at deployment. Fast-WAM [37] skips future prediction at test time while remaining competitive with imagine-then-execute variants, indicating that video supervision can improve policy learning even without online imagination. Being-H0.7 [22] follows a similar latent modeling philosophy by aligning an inference-time prior branch with a training-only posterior branch informed by future observations, transferring future-aware reasoning into compact latent queries rather than generated frames. In this work, we explore a complementary direction: constructing semantic visual tokens and aligned latent action tokens as the representation space on which world action modeling is trained.

## 2.2 Latent Action Models

Latent action models recover action-like variables from observation sequences, making unlabeled videos usable for downstream control. Genie [6] establishes this direction in generative environments, where latent actions make video dynamics controllable without ground-truth action labels. LAPO [23] moves from controllable generation toward policy learning, recovering latent-action policies and dynamics models from observed transitions before adapting them with limited supervision or reward. LAPA [35] scales this principle to robot manipulation by learning discrete latent actions between image frames and using them as pretraining targets for VLA models. Moto [9] shifts the abstraction from generic action codes to motion tokens, using autoregressive video pretraining to transfer motion regularities into robot policies. This line of work progressively turns visual change into a substitute for missing motor labels. In this work, we instead learn latent actions within a semantic visual latent space, making each action token a transformation between semantic visual states rather than an isolated code extracted from raw pixel changes. This design aligns the action abstraction with the visual tokens used by the world model, making it better suited for joint visual-action modeling in WAMs.

## 3 Method

## 3.1 Overview

Given visual observations $o _ { 1 : T } ,$ a language instruction $y ,$ and robot actions $a _ { 1 : T - 1 } ,$ world action models ?? ?? ?? ?? ??(WAMs) [20, 36] factor manipulation into a two-stage causal process: the world model expert $p _ { \theta _ { 1 } }$ forecasts the next observation $\hat { \sigma } _ { t } \sim p _ { \theta _ { 1 } } ( o _ { t } \mid o _ { < t } , y )$ , and the action expert $p _ { \theta _ { 2 } }$ infers the action $\hat { a } _ { t - 1 } \sim p _ { \theta _ { 2 } } ( a _ { t - 1 } \mid o _ { < t } , \hat { o } _ { t } , y )$ ???? ?? ???? ?? ?? , ?? ?? ???? ?? ???? ?? ?? , ???? , ??that produces this transition, typically realized by an inverse dynamics model (IDM) [13]. In practice both stages operate over latents $z _ { 1 : T }$ produced by a video tokenizer [28] rather than raw pixels.

In this work, we posit that the fidelity of this formulation is bottlenecked by the latent spaces in which $z _ { t }$ and $a _ { t - 1 }$ reside. Specifically, the visual latent $z _ { t }$ is inherited from reconstruction-oriented tokenizers ???? ???? ????tuned to appearance rather than semantics, making it challenging for the world model expert to learn instruction-following dynamics. Meanwhile, $a _ { t - }$ −1 lives in a space decoupled from $z _ { t } ,$ , forcing the action expert to bridge a modality gap at every step.

Our goal is to overcome these limitations by establishing a semantic latent space in which visual states and their action-induced transitions are jointly represented. To this end, we present RepWAM, a representation-centric WAM built around a representation visual-action tokenizer (RepViTok) that aligns visual latents with a visual foundation model and induces latent action tokens as transitions within the same space (Sec. 3.2). Built on these aligned tokens, RepWAM pretrains a causal world action model via flow matching over paired world model and action experts, and adapts it to robot demonstrations for executable control (Sec. 3.3). The overall framework is illustrated in Figure 1.

## 3.2 Representation Visual-Action Tokenizer

To close the latent-space gap above, we design a tokenizer that yields semantically aligned visual tokens together with action tokens that capture transitions between them. Taking the visual observations $o _ { 1 : T }$ as input, our representation visual-action tokenizer produces visual tokens $\boldsymbol { z } \in \mathbb { R } ^ { T ^ { \prime } L \times d _ { v } }$ ?? ?? that compactly encode visual content and latent action tokens $\ell \in \mathbb { R } ^ { ( T ^ { \prime } - 1 ) \times \dot { d } _ { \ell } }$ ??that capture temporal dynamics, where the visual tokens are organized as $\begin{array} { r } { T ^ { \prime } = 1 + \frac { ( T - 1 ) } { 4 } } \end{array}$ latent frames of  spatial tokens each after temporal patchification. The two tokenization steps are performed sequentially, so latent actions are induced from the learned visual latent space rather than directly from pixels.

Visual tokenization. The visual tokenizer is a vision transformer (ViT) [12, 29] autoencoder. The initial frame $o _ { 1 }$ and the subsequent frames $o _ { 2 : T }$ are split into $1 6 \times 1 6$ patches and $4 \times 1 6 \times 1 6$ (temporal×height×width) ?? ?? ??tubelets, respectively. We concatenate the resulting tokens along the sequence dimension and feed them into the encoder $E _ { \theta } ,$ which consists of stacked attention blocks with temporal-causal masking across frames ??and full spatial attention within each frame. The encoder output is then passed through an attention-based projection layer followed by layer normalization to obtain the video latents $z .$ The decoder $D _ { \theta }$ follows ?? ??a symmetric architecture and maps the latents back to pixels through an unpatchify head composed of transposed convolution layers.

![](images/ee3f7f15b669300084a44eb3d0d1139ea8b266998a76859e0ba3abc21fec0d95.jpg)  
Figure 1 Overview of our representation visual-action tokenizer, which aligns visual latents with a frozen visual foundation model to obtain semantically rich visual tokens, and induces latent actions as transitions within this shared semantic space via coupled inverse and forward dynamics models.

We supervise the visual tokenizer with a reconstruction objective that combines multiple losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \lambda _ { 1 } \| o - \hat { o } \| _ { 1 } + \lambda _ { \mathrm { p e r c } } \mathcal { L } _ { \mathrm { p e r c } } ( o , \hat { o } ) + \lambda _ { \mathrm { g a n } } \mathcal { L } _ { \mathrm { g a n } } ( \hat { o } ) , } \end{array}\tag{1}
$$

where $\scriptstyle { \hat { o } } = D _ { \theta } ( z )$ denotes the reconstructed frames, $\mathcal { L } _ { \mathrm { p e r c } }$ and ${ \mathcal { L } } _ { \mathrm { g a n } }$ are the perceptual and adversarial losses, and $\lambda _ { 1 } , \lambda _ { \mathrm { p e r c } } , \lambda _ { \mathrm { g a n } }$ weight the three terms.

Beyond reconstruction, we add a feature-alignment loss that pulls the video latents toward a frozen visual foundation model [5]. Let denote this teacher model and let $W _ { \mathrm { a l i g n } }$ be a linear projection layer that matches the teacher dimension. The alignment objective is

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \left\| \mathrm { a v g } ( W _ { \mathrm { a l i g n } } z ) - \mathrm { a v g } ( G ( o ) ) \right\| _ { 2 } ^ { 2 } .\tag{2}
$$

where avg denotes temporal average pooling following Perception Encoder [5].

The visual tokenizer is jointly supervised by the total objective $\mathcal { L } _ { \mathrm { v i s } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } ,$ where $\lambda _ { \mathrm { a l i g n } }$ balances reconstruction fidelity and semantic alignment.

Latent action tokenization. Building on these semantically aligned visual tokens, we then train action tokens in the same latent space, so that each action token represents a transition between two visual states. We freeze the visual tokenizer and train a latent action tokenizer (LAT) that couples an IDM $q _ { \phi }$ with a forward dynamics model (FDM) $f _ { \psi }$

For consecutive latent frames $\boldsymbol { z } _ { t } , \boldsymbol { z } _ { t + 1 } \in \mathbb { R } ^ { L \times d _ { v } }$ , the IDM compresses the transition into $\ell _ { t } \in \mathbb R ^ { d _ { \ell } } \ ( d _ { \ell } \ll d _ { v } )$ ???? , ????preventing content leakage. The FDM then realizes $\ell _ { t }$ as a transport map $K _ { t } \in \mathbb { R } ^ { L \times L }$ ℓ?? and a residual $\boldsymbol { \delta } _ { t } \in \mathbb { R } ^ { L \times d _ { v } }$

$$
\begin{array} { c } { { \ell _ { t } = q _ { \phi } ( z _ { t } , z _ { t + 1 } ) , } } \\ { { K _ { t } , \delta _ { t } = f _ { \psi } ( z _ { t } , \ell _ { t } ) , } } \\ { { \hat { z } _ { t + 1 } = K _ { t } z _ { t } + \delta _ { t } , } } \end{array}\tag{3}
$$

where $K _ { t } \ z _ { t }$ denotes left-multiplication along the spatial-token dimension of $z _ { t } ,$ and $\hat { z } _ { t + 1 }$ is the reconstruction ???? ????of the next visual latent. Intuitively, $K _ { t }$ ???? ????acts as a soft transport operator inspired by optical flow [26] in the ????semantic token space, routing visual content according to the state change induced by $\ell _ { t } ,$ while $\delta _ { t }$ captures ℓ?? ??residual changes that cannot be explained by transport alone. Since this transformation is defined over visual states rather than embodiment-specific motor coordinates, the resulting latent action describes a transferable task-level transition.

The LAT is trained with a forward next-latent prediction loss and a backward consistency loss, where the backward $\hat { z } _ { t }$ is obtained by running the LAT on the reversed pair $\left( z _ { t + 1 } , z _ { t } \right)$

$$
\mathcal { L } _ { \mathrm { f w d } } = \sum _ { t = 1 } ^ { T ^ { \prime } - 1 } \big \| \hat { z } _ { t + 1 } - z _ { t + 1 } \big \| _ { 2 } ^ { 2 } , \qquad \mathcal { L } _ { \mathrm { c o n s } } = \sum _ { t = 1 } ^ { T ^ { \prime } - 1 } \big \| \hat { z } _ { t } - z _ { t } \big \| _ { 2 } ^ { 2 } .\tag{4}
$$

## 3.3 Causal World Action Models

With aligned visual and action tokens in hand, we now train the RepWAM world action model that generates both streams jointly under language conditioning.

Causal diffusion transformer. We cast world action modeling as causal generation over visual-action chunks. The language instruction  is embedded by a pretrained text encoder into conditioning tokens . Given a chunk size $k ,$ ?? ?? each chunk groups the visual latents and latent actions over a short temporal window:

$$
\begin{array} { r } { u _ { t : t + k } = \left[ z _ { t : t + k } , \ell _ { t : t + k - 1 } \right] , } \\ { s = \left[ c , z _ { 1 } , u _ { t _ { 1 } : t _ { 1 } + k } , \dots , u _ { t _ { N } : t _ { N } + k } \right] , } \end{array}\tag{5}
$$

where $z _ { t : t + k }$ are the flattened visual tokens in window $t : t + k , \ \ell _ { t : t + k - 1 }$ the paired latent actions between ???? ?? ??consecutive visual steps, $t _ { 1 } , \ldots , t _ { N }$ ?? ?? ?? ℓ?? ?? ??the start indices of the  chunks, and  prefixes the chunk sequence with ?? , . . . , ????language tokens and the initial visual context $z _ { 1 }$ ??. For a target chunk $u _ { t : t + k } , s _ { < t }$ denotes this prefix together ?? ???? ?? ??with all preceding chunks. A block-causal mask lets each chunk attend to $s _ { < t }$ ??but not to future chunks [30]. ?? ??Within each block, visual and action tokens share attention weights but use modality-specific feed-forward networks (FFNs).

Flow-matching objective. We train the transformer with teacher forcing under a conditional flow-matching objective applied jointly to the world model and action experts. At each step we sample a Gaussian noise tensor $\epsilon _ { t : t + k } \sim N ( 0 , I )$ with the same shape as the chunk and a time scalar $\alpha \sim \mathcal { U } ( 0 , 1 )$ , and form the linear ?? ?? ??interpolant:

$$
x _ { \alpha } = ( 1 - \alpha ) \epsilon _ { t : t + k } + \alpha u _ { t : t + k } , \qquad \dot { x } _ { \alpha } = u _ { t : t + k } - \epsilon _ { t : t + k } ,\tag{6}
$$

where $\dot { x } _ { \alpha }$ is the target velocity that the network should regress. Using the velocity estimate $F _ { \theta } ( x _ { \alpha } , \alpha , s _ { < t } )$ ??from the network, the training loss is:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \left[ \left\| F _ { \theta } ^ { v } ( x _ { \alpha } , \alpha , s _ { < t } ) - \dot { x } _ { \alpha } ^ { v } \right\| _ { 2 } ^ { 2 } + \lambda _ { a } \left\| F _ { \theta } ^ { a } ( x _ { \alpha } , \alpha , s _ { < t } ) - \dot { x } _ { \alpha } ^ { a } \right\| _ { 2 } ^ { 2 } \right] ,\tag{7}
$$

where the superscripts  and  index the visual and latent-action components of the chunk, respectively, and $\lambda _ { a }$ ?? ??balances their contributions.

## 4 Experiments

## 4.1 Implementation Details

Training data. We pretrain the WAM on AgiBot [7] with roughly 100 G video-action latent tokens whose action components are produced by our latent action tokenizer. For embodiment adaptation, we use a mixed real-robot corpus spanning AgiBot, RoboMIND [32], RoboCOIN [33], and InternA1 [27] with roughly 300 G tokens, where each demonstration provides continuous motor commands $( e . g .$ , end-effector motion and gripper control) aligned with the visual trajectory.

Model and optimization. Our visual autoencoder adopts 12 transformer layers with hidden dimension 768 for both the encoder and decoder, with reconstruction weights $\lambda _ { 1 } { = } 1 , \lambda _ { \mathrm { p e r c } } { = } 1 , \lambda _ { \mathrm { g a n } } { = } 0 . 1$ , and alignment weight $\lambda _ { \mathrm { a l i g n } } { = } 1$ , following previous work [29]. The latent action IDM $q _ { \phi }$ and FDM $f _ { \psi }$ .are both four-layer MLPs with ?? ??hidden size 256. Visual-latent and latent-action token dimensions are set to $d _ { v } = 9 6$ and $d _ { \ell } { = } 4 _ { \ell }$ , respectively.

![](images/57414edaeafa6fd762019de0158637da1395bbfce066ff62543c77c988e7c322.jpg)  
Figure 2 Real-world success rate (10 rollouts per task) on three manipulation tasks, comparing ??0 5 [4] and Lingbot-VA [20] against our 1 3B and 5B WAMs.

We train the causal WAM backbones at two scales, 1 3B and 5B, entirely from scratch. Both follow the same .architecture: the world model expert is a causal diffusion transformer with 30 layers and hidden dimension $h _ { v } \ ( h _ { v } = 1 5 3 6$ for the 1 3B model and $h _ { v } { = } 3 0 7 2$ for the 5B model), while the action expert shares the same ℎ?? ℎ?? . ℎ??depth with a reduced hidden dimension $h _ { a } = 7 6 8$ , contributing roughly 350M additional parameters. We ℎ??share the attention weights for both experts, but adopt independent feed-forward networks for each modality. Language instructions are encoded by a frozen PLM text encoder [10] and injected via cross-attention. We optimize the WAM with the Muon optimizer [18], peak learning rate $1 \times 1 0 ^ { - 2 }$ , weight decay 0 01, and cosine annealing with linear warmup, in bfloat16 mixed precision with gradient clipping at 2 0. The action loss weight is set to $\lambda _ { a } = 1$ ., and a uniform SNR sampler is applied to both experts. Following [20], we sample the chunk size from [1 4] during training. The 1 3B model packs episodes into sequences of up to 200K tokens , .and is trained on 64 H20 GPUs, whereas the 5B model packs up to 160K tokens on 128 H20 GPUs.

During inference, we set chunk size to $^ { 2 , }$ attention window size to 32. Video and action experts adopt 5 and 10 steps, respectively.

## 4.2 Real-World Experiments

We evaluate RepWAM on a Franka dual-arm robot platform across three manipulation tasks: (1) Pick the fruits and put them into the plate requires localizing target objects among clutter and executing reliable top-down grasps under varied poses and object instances, followed by precise placement within a designated container. (2) Push the drawer and put the building block into it poses a long-horizon challenge: the policy must apply the correct handle motion to actuate the drawer, retrieve a small block from the scene, and deposit it inside without collision. (3) Insert the test tube into the test tube rack demands fine-grained spatial alignment, requiring accurate tip localization relative to the receptacle and compliant contact management under tight geometric clearance. For each task, we fine-tune on 50 real-world demos for 500 steps with learning rate 1e-4 and sequence length 150K.

We evaluate each task over 10 physical rollouts and report success rate in Figure 2 for two model sizes, RepWAM-1 3B and RepWAM-5B. On the short-horizon pick-the-fruit task, both sizes reach 60%, improving over $\pi _ { 0 . 5 }$ .(10%) by 50 points and over Lingbot-VA (50%) by 10 points, with the two tied since the bottleneck .here is perception and grasping rather than model capacity. On the long-horizon push-the-drawer task, RepWAM-5B reaches 80% (best overall, +30 over $\pi _ { 0 . 5 } ,$ +10 over Lingbot-VA) and RepWAM-1 3B reaches 50%, showing that multi-step articulated control benefits substantially from added capacity. The fine-grained insert-the-tube task is the most demanding, where RepWAM-5B reaches 60% (+50 over $\pi _ { 0 . 5 } ,$ +20 over .Lingbot-VA) and RepWAM-1 3B reaches 30%, again confirming that performance improves with model size. .Across tasks, RepWAM-5B is best or tied-best on every task, with the largest gains on long-horizon and fine-grained settings where the world model expert must roll out coherent visual dynamics over many steps before the action expert acts.

Figure 3 provides representative successful real-robot executions. From left to right, the visualizations cover picking the fruit, pushing the drawer, and inserting the tube. In successful rollouts, the robot approaches target objects with stable grasp poses, follows the intended transport or articulation motion, and places or inserts objects near the desired goal without large corrective oscillations. We attribute this behavior to the representation visual-action tokenizer: RepViTok preserves object identity, contact-relevant geometry, and task context in the visual tokens, while the latent actions describe action-induced transitions in the same semantic space. As a result, the world action model can predict coherent scene evolution and the accompanying action stream before the adapted action expert maps these transitions to executable motor commands.

![](images/8f043899dd31d0a55ba166e62316f5ede761d4c26d3f5f4bf55943a08ba26a6a.jpg)  
Pick the fruit

![](images/e61e6bf5545769be8fa8b7851ac25efcb4eb6f28427b2bd4f616268f18b3d27b.jpg)  
Push the drawer

![](images/4201b70b6a39ec1e5ea10aafe9d6a278a319b4b0581f2db50924c89d7237455b.jpg)  
Insert the tube  
Figure 3 Successful real-robot executions. From left to right, we show representative rollouts for picking the fruit, pushing the drawer and inserting the tube.

## 4.3 Simulation Experiments

RoboTwin 2.0 [8] is a large-scale dual-arm simulation benchmark with extensive domain randomization over scene composition, lighting conditions, camera viewpoints, and object physics. We evaluate on its standard task suite and report success rate under the official randomization settings.

As shown in Table 1, RepWAM-5B outperforms $\pi _ { 0 . 5 }$ and Motus on the 50-task average across both Easy .and Hard settings, demonstrating that semantic visual-action tokenization provides strong generalization for bimanual manipulation under randomized scenes. We hypothesize that the remaining performance gap to Lingbot-VA mainly comes from its use of WAN video-generation pretraining, whereas our WAM is trained from scratch without using pretrained weights. In the following ablation study, we further show that RepViTok improves over the WAN2.2 VAE used by such WAN-pretrained pipelines. These results suggest that the semantic structure introduced by representation-centric visual-action tokenization is a key factor for world action modeling, and can provide strong performance even without inheriting a pretrained video-generation backbone.

Table 1 Comparison on RoboTwin 2.0. We follow Lingbot-VA [20] to categorize 50 tasks into different horizons.
<table><tr><td rowspan="2">Metric</td><td colspan="2">70.5[4]</td><td colspan="2">Motus [3]</td><td colspan="2">Lingbot-VA [20]</td><td colspan="2">Ours 1.3B</td><td colspan="2">Ours 5B</td></tr><tr><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td></tr><tr><td>Backbone pretrained</td><td>√</td><td></td><td>！</td><td></td><td>！</td><td></td><td>×</td><td></td><td></td><td>×</td></tr><tr><td>Averagehor=2</td><td>79.3</td><td>73.0</td><td>85.2</td><td>80.9</td><td>85.3</td><td>86.9</td><td>85.7</td><td>84.0</td><td>87.4</td><td>87.6</td></tr><tr><td>Averagehor=3</td><td>78.6</td><td>67.4</td><td>85.0</td><td>84.2</td><td>89.6</td><td>90.6</td><td>92.0</td><td>85.4</td><td>88.0</td><td>90.4</td></tr><tr><td>Average50 Tasks</td><td>82.7</td><td>76.8</td><td>88.7</td><td>87.0</td><td>92.9</td><td>91.6</td><td>86.6</td><td>83.1</td><td>89.3</td><td>88.4</td></tr></table>

## 4.4 Ablation Studies

We adopt the 1 3B WAM for ablation experiments, and train the model for 40 000 steps on 32 H20 GPUs. . ,We split AgiBot into a training set, a seen evaluation set covering tasks observed during training, and an unseen evaluation set covering held-out tasks. We report three groups of metrics: (1) visual generation quality measured by gFVD, PSNR, and SSIM, (2) open-loop action accuracy measured by the open loop score (OLS) [19] under threshold 0 03, and (3) closed-loop execution on the real-robot pick-the-fruit task.

Semantic visual tokenizer for world modeling. We first study the effect of visual latent space on downstream world action modeling in Table 2. Our RepViTok achieves the strongest overall performance, reducing gFVD by 9 5% / 13 2% compared to the reconstruction-only WAN2.2 VAE [28] and improving OLS from 13 68 / . . .11 21 to 18 82 / 14 15. The real-robot evaluation shows the same trend in closed-loop control: RepViTok . . .reaches 30% success on PickFruit, compared with 20% for WAN2.2 VAE and 10% for ViTok. ViTok improves OLS over WAN2.2 VAE but does not translate into reliable execution, suggesting that open-loop action accuracy alone is insufficient for robust manipulation. These results support our hypothesis that a latent space optimized solely for appearance deprives the world model expert of the semantic structure required to follow instructions, and prevents the action expert from grounding motor commands in scene content.

Table 2 Ablation study on tokenizer designs. We compare reconstruction-oriented and semantics-aware video tokenizers on AgiBot Eval Seen and Eval Unseen.
<table><tr><td rowspan="2">Model</td><td colspan="4">Eval Seen</td><td colspan="4">Eval Unseen</td><td rowspan="2">PickFruit</td></tr><tr><td>gFVD</td><td>PSNR</td><td>SSIM</td><td>OLS</td><td>gFVD</td><td>PSNR</td><td>SSIM</td><td>OLS</td></tr><tr><td>WAN2.2 VAE [28]</td><td>67.42</td><td>17.34</td><td>0.67</td><td>13.68</td><td>83.98</td><td>16.86</td><td>0.64</td><td>11.21</td><td>20%</td></tr><tr><td>ViTok</td><td>69.23</td><td>17.21</td><td>0.68</td><td>16.29</td><td>80.14</td><td>17.19</td><td>0.67</td><td>13.81</td><td>10%</td></tr><tr><td>RepViTok</td><td>61.01</td><td>18.47</td><td>0.70</td><td>18.82</td><td>72.91</td><td>17.72</td><td>0.67</td><td>14.15</td><td>30%</td></tr></table>

As discussed above, we hypothesize that the remaining gap to Lingbot-VA [20] mainly comes from its WAN video-generation pretraining. To examine this, we isolate the visual tokenizer while keeping the 1 3B WAM .setting fixed. As shown in Table 3, replacing WAN2.2 VAE with RepViTok improves the average success rate from 78 0 to 86 6 on Easy and from 76 0 to 83 1 on Hard, supporting the importance of semantics-aware . . .visual tokenization for world action modeling.

World action modeling with latent actions. We then study how latent actions should be incorporated into WAM pretraining in Table 4. The w/o baseline uses the RepViTok visual tokenizer but omits latent-action pretraining and trains the WAM directly on robot actions. Joint Pred trains the world model expert to predict video latents and action latents simultaneously through an additional prediction head, with the final action-latent slot padded by zero. Two Stages corresponds to our proposed design, in which the model is first pretrained on visual and latent action tokens and subsequently adapted to robot control.

LAPA  
Ours  
Frame T  
Table 3 Comparison between different VAEs for the 1 3B WAM on RoboTwin 2.0 simulation, Easy and Hard, 50 tasks.
<table><tr><td rowspan="2">Metric</td><td colspan="2">WAN2.2 VAE [28]</td><td colspan="2">RepViTok (Ours)</td></tr><tr><td>Easy</td><td>Hard</td><td>Easy</td><td>Hard</td></tr><tr><td>Averagehor=1</td><td>81.1</td><td>78.4</td><td>86.2</td><td>83.1</td></tr><tr><td>Averagehor=2</td><td>75.5</td><td>73.9</td><td>85.7</td><td>84.0</td></tr><tr><td>Averagehor=3</td><td>67.2</td><td>68.0</td><td>92.0</td><td>85.4</td></tr><tr><td>Average50 Tasks</td><td>78.0</td><td>76.0</td><td>86.6</td><td>83.1</td></tr></table>

Table 4 Ablation study on latent actions for world action modeling. We compare w/o latent actions, joint prediction, and two-stage latent action training.
<table><tr><td rowspan="2">Model</td><td colspan="4">Eval Seen</td><td colspan="4">Eval Unseen</td><td rowspan="2">PickFruit</td></tr><tr><td>gFVD</td><td>PSNR</td><td>SSIM</td><td>OLS</td><td>gFVD</td><td>PSNR</td><td>SSIM</td><td>OLS</td></tr><tr><td>w/0</td><td>61.01</td><td>18.47</td><td>0.70</td><td>18.82</td><td>72.91</td><td>17.72</td><td>0.67</td><td>14.15</td><td>30%</td></tr><tr><td>Joint Pred</td><td>94.25</td><td>15.24</td><td>0.61</td><td>18.52</td><td>98.77</td><td>15.09</td><td>0.55</td><td>15.22</td><td>20%</td></tr><tr><td>Two Stages (Ours)</td><td>48.23</td><td>22.86</td><td>0.75</td><td>19.87</td><td>58.83</td><td>19.93</td><td>0.74</td><td>16.98</td><td>50%</td></tr></table>

Joint Pred degrades visual dynamics, increasing gFVD to 94 25 / 98 77 and lowering both PSNR and SSIM, indicating that a single appended head entangles state prediction with action supervision. Two Stages achieves the best result on every metric, reducing gFVD to 48 23 / 58 83, improving PSNR to 22 86 / 19 93, and raising OLS to 19 87 / 16 98. The PickFruit result further confirms that the benefit is not limited to offline prediction: two-stage training improves closed-loop success to 50%, compared with 30% without latent actions and 20% for joint prediction. This gap suggests that pretraining on semantic latent actions provides a more transferable intermediate action representation, while direct robot-action supervision or joint prediction yields weaker executable behavior after adaptation. These results support decoupling latent-action pretraining from robot-action adaptation.

Latent actions and transferring to robot control. We further analyze whether the learned latent actions provide a useful bridge from visual dynamics to executable robot control in Figure 4. The left side compares action-latent visualizations from LAPA [35] and RepViTok. LAPA produces relatively diffuse responses, while our latent actions concentrate more clearly on manipulation-relevant changes, such as object displacement and contact-induced motion, indicating that RepViTok better captures the action-conditioned state transition. The right side evaluates transferability by freezing the learned action latents and training the same inverse dynamics model to decode robot actions from them. RepViTok achieves a lower IDM action loss than LAPA, showing that our latent actions are not only more visually aligned with motion but also easier to map into the robot-action space. Together, these results support the role of semantic latent actions as an effective intermediate representation for downstream control adaptation.

![](images/dbc1e3d46da2d98f1573a34b1ac74fbea66a1c043ef01886b7a26668d180d0d0.jpg)

![](images/996d31536947b055a2057af50a67060def9176286a8afc6bdb86dc8497070f1a.jpg)  
Frame T+1

![](images/84e4cd30d08f71a47ddc34bfc29f927882f91e2afbd42b3ca825367a047a69d5.jpg)

![](images/2a432891dc9f0c0eeb9bbcee6b3e312c615b668f5b7db2d3edd9ee1988ed70b5.jpg)

![](images/22659ca675a25438fba58b23fd202c6a5a787e513200ce52a6e369cad391b6c1.jpg)  
Figure 4 Comparison of latent actions and their transfer to robot control. Left: action-latent visualizations from LAPA [35] and RepViTok. Right: IDM loss with frozen action latents.

The effects of video classifier-free guidance. In WAMs, action prediction generally does not require CFG, but the world model branch naturally inherits the dependence on CFG from video generation models. Given the stronger semantic structure of the RepViTok latent space, we examine whether this inherited dependence can be reduced by running inference with different video CFG scales. Figure 5 compares our RepViTok-based WAM with Lingbot-VA [20] at scales 1 0, 1 25, and 2 0. In our implementation, video CFG only affects the . . .video denoising path, while action prediction remains unchanged.

![](images/78f2c7a740ad36c00c3f1b8c4c3b9e35cd0cea3db73d598d36340f188c906a24.jpg)  
Figure 5 Effect of video classifier-free guidance on RoboTwin 2.0 success rate. We compare our RepViTok-based WAM with Lingbot-VA [20] on Easy, Hard, and averaged success rates over video CFG scales 1 0, 1 25, and 2 0.

For our model, the averaged success rate is highest at video CFG scale 1 0, i.e., without additional CFG .extrapolation. This indicates that RepViTok already provides a language-aligned visual-action space, making stronger video CFG unnecessary and sometimes harmful in harder randomized settings. It further shows that RepViTok weakens the inherited reliance on video CFG, allowing inference to use the conditional video branch alone when a CFG scale of 1 0 is sufficient, which can reduce latency and activation memory in .deployments that would otherwise evaluate an unconditional video branch.

Visual reconstruction results. We evaluate the visual reconstruction results on ImageNet [11] and UCF101 [24] in Table 5. Across 256 and 512 resolutions, RepViTok remains competitive with WAN2.2 VAE in reconstruction quality, achieving higher ImageNet PSNR and SSIM and stronger UCF101 rFVD while delivering the downstream gains reported in Table 2.

Table 5 Reconstruction quality of different visual tokenizers on ImageNet and UCF101.
<table><tr><td rowspan="2">Model</td><td colspan="3">ImageNet 256</td><td colspan="3">ImageNet 512</td><td colspan="3">UCF101 256×17</td><td colspan="3">UCF101 512×17</td></tr><tr><td>rFID</td><td>PSNR</td><td>SSIM</td><td>rFID</td><td>PSNR</td><td>SSIM</td><td>rFVD</td><td>PSNR</td><td>SSIM</td><td>rFVD</td><td>PSNR</td><td>SSIM</td></tr><tr><td>WAN2.2 VAE [28]</td><td>0.50</td><td>28.16</td><td>0.87</td><td>0.20</td><td>30.48</td><td>0.90</td><td>4.28</td><td>36.61</td><td>0.98</td><td>0.68</td><td>41.45</td><td>0.99</td></tr><tr><td>ViTok</td><td>0.96</td><td>28.65</td><td>0.89</td><td>0.24</td><td>30.77</td><td>0.92</td><td>1.23</td><td>35.52</td><td>0.97</td><td>0.16</td><td>40.68</td><td>0.98</td></tr><tr><td>RepViTok</td><td>0.80</td><td>28.90</td><td>0.89</td><td>0.23</td><td>31.00</td><td>0.92</td><td>1.09</td><td>36.03</td><td>0.97</td><td>0.16</td><td>41.12</td><td>0.98</td></tr></table>

The qualitative reconstructions in Figure 6 complement the quantitative results by showing what RepViTok preserves in individual examples from ImageNet and UCF101. On ImageNet images, RepViTok maintains both low-level appearance and semantic identity: faces remain recognizable, object boundaries stay sharp, and text regions preserve coherent strokes and layouts rather than collapsing into blurry textures. On UCF101 videos, RepViTok further preserves temporal consistency across frames, keeping moving actors and objects stable even in clips with large body motion and fast foreground changes. These examples indicate that the semantic alignment objective preserves competitive reconstruction quality despite tradeoffs on some fidelity metrics. More importantly, it helps the tokenizer retain visually and semantically important details in both static images and dynamic videos.

![](images/f93e6ece58508015da0380ffcfd36a819602171fbf5aef6136ffb67234fe2f9f.jpg)  
Figure 6 Qualitative reconstruction examples on ImageNet and UCF101. RepViTok preserves semantic details in images and maintains temporal consistency in videos.

## 5 Conclusion

We presented RepWAM, a representation-centric world action model built on semantic visual-action tokenization. Rather than relying on reconstruction-oriented video latents inherited from pretrained generation backbones, RepWAM aligns visual latents with a frozen visual foundation model and learns latent actions as transitions between visual states. With this semantic representation, we train a causal diffusion transformer to jointly model instruction-conditioned future observations and the latent actions that realize them, and then adapt these dynamics to embodiment-specific robot control with demonstrations. Experiments on Franka dual-arm real-world manipulation and RoboTwin 2.0, together with tokenizer and latent-action ablations, show that semantic visual-action alignment improves visual dynamics, action prediction, and closed-loop execution. These findings highlight the significance of representation design for WAMs, pointing toward more interpretable, transferable, and effective robot manipulation.

In future work, we will scale WAM pretraining beyond the robotics-domain videos by leveraging large-scale internet videos, especially egocentric human videos. This could broaden the range of behaviors and interaction patterns available during pretraining.

## References

[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvĳit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

[2] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

[3] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025.

[4] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y Galliker, et al. pi05: a vision-language-action model with open-world generalization. In CoRL, 2025.

[5] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Bangalath, et al. Perception encoder: The best visual embeddings are not at the output of the network. NeurIPS, 2026.

[6] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024.

[7] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[8] Tianxing Chen, Zanxin Chen, Baĳun Chen, Zĳian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[9] Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. In ICCV, 2025.

[10] Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, et al. Perceptionlm: Open-access data and models for detailed visual understanding. In NeurIPS, 2026.

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[13] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In NeurIPS, 2023.

[14] Qiuxuan Feng, Jiale Yu, Jiaming Liu, Yueru Jia, Zhuangzhe Wu, Hao Chen, Zezhong Qian, Shuo Gu, Peng Jia, Siwei Ma, et al. Harmowam: Harmonizing generalizable and precise manipulation via adaptive world action models. arXiv preprint arXiv:2605.10942, 2026.

[15] Pengbo Guo, Junke Wang, Zhen Xing, Chengxu Liu, Daoguo Dong, Xueming Qian, and Zuxuan Wu. Dera: Decoupled representation alignment for video tokenization. arXiv preprint arXiv:2512.04483, 2025.

[16] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In NeurIPS, 2018.

[17] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

[18] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github.io/posts/muon/.

[19] Hao Li, Ziqin Wang, Zi-han Ding, Shuai Yang, Yilun Chen, Yang Tian, Xiaolin Hu, Tai Wang, Dahua Lin, Feng Zhao, et al. Robointer: A holistic intermediate representation suite towards robotic manipulation. In ICLR, 2026.

[20] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. In RSS, 2026.

[21] Yushan Liu, Peibo Sun, Shoujie Li, Yifan Xie, Lingfeng Zhang, Xintao Chao, Shiyuan Dong, Fang Chen, Xiao-Ping Zhang, and Wenbo Ding. Oa-wam: Object-addressable world action model for robust robot manipulation. arXiv preprint arXiv:2605.06481, 2026.

[22] Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, and Zongqing Lu. Being-h0. 7: A latent world-action model from egocentric videos. arXiv preprint arXiv:2605.00078, 2026.

[23] Dominik Schmidt and Minqi Jiang. Learning to act without actions. In ICLR, 2024.

[24] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

[25] MotuBrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, et al. Motubrain: An advanced world action model for robot control. arXiv preprint arXiv:2604.27792, 2026.

[26] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, 2020.

[27] Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, et al. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy. arXiv preprint arXiv:2511.16651, 2025.

[28] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[29] Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. In NeurIPS, 2024.

[30] Junke Wang, Xun Wang, Qiushan Guo, Peize Sun, Weilin Huang, Zuxuan Wu, and Yu-Gang Jiang. Omnigen-ar: Autoregressive any-to-image generation. In NeurIPS, 2025.

[31] Siyin Wang, Junhao Shi, Zhaoyang Fu, Xinzhe He, Feihong Liu, Chenchen Yang, Yikang Zhou, Zhaoye Fei, Jingjing Gong, Jinlan Fu, et al. World action models: The next frontier in embodied ai. arXiv preprint arXiv:2605.12090, 2026.

[32] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

[33] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025.

[34] Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. Gigaworld-policy: An efficient action-centered world–action model. arXiv preprint arXiv:2603.17240, 2026.

[35] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Se June Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. In ICLR, 2025.

[36] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[37] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.