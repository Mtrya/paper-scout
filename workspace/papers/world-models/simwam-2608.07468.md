# SimWAM: A Simple World Action Model for End-to-End Autonomous Driving

Zongchuang Zhao<sup>1</sup>, Xin Zhou<sup>1</sup>, Tianyang Xu<sup>1</sup>, Zhengyang Sun<sup>1</sup> Kaixuan Zhou<sup>2</sup>, Honglin Li<sup>2</sup>, Dingkang Liang<sup>1†</sup>, Xiang Bai<sup>1</sup>

<sup>1</sup>Huazhong University of Science & Technology, <sup>2</sup> Dongfeng Research & Development Institute. {zcuangzhao, xzhou03, dkliang, xbai}@hust.edu.cn

## Abstract

World-Action Models (WAMs) improve end-to-end autonomous driving by transferring video dynamics priors to action prediction, but existing methods require costly future generation at inference. We present SimWAM, a simple yet effective WAM that uses video generation purely as a training signal. It co-trains a pretrained video expert and a lightweight action expert with joint flow matching. An isolated attention mask keeps action prediction independent of future frames, allowing the video branch to be discarded after training and leaving a self-contained planner that directly predicts trajectories. Since the two experts share no parameters and interact only through a unified attention interface, the video backbone could be replaced and the action expert scaled independently without modifying the learning objective or inference pipeline. We further apply reinforcement learning to optimize a compositional driving reward beyond trajectory imitation. Our SimWAM achieves 91.5 PDMS on NAVSIM, surpasses state-of-the-art WAM-based planners with substantially lower latency, and transfers zero-shot to nuScenes. These results position SimWAM as a simple yet solid baseline that could readily benefit from advances in video generation for efficient autonomous driving. The code and model weights are available at https://github.com/H-EmbodVis/SimWAM/.

## 1 Introduction

End-to-end autonomous driving [9, 55] maps raw sensor observations directly to a planned trajectory with a unified network. Joint optimization removes hand-crafted interfaces and reduces error propagation in the classical perception, prediction, and planning pipeline [5, 39]. Although recent end-to-end planners [18, 19, 28] have steadily improved planning accuracy, they remain primarily imitation policies. They reproduce behavior from logged trajectories while capturing traffic semantics, user intent, and scene dynamics only implicitly.

Vision-Language-Action (VLA) models [4, 12, 20, 22, 27] address the semantic limitation by adapting pretrained vision-language models to driving. Their semantic knowledge and high-level reasoning improve scene understanding and connect trajectory generation with user intent. Many driving VLAs [49, 60, 67] further produce an explicit rationale before predicting a trajectory, which improves interpretability in complex and instruction-conditioned scenarios. Recent methods [38, 44, 56] introduce future-scene generation or latent reasoning to strengthen spatiotemporal understanding. However, these components remain loosely coupled with action prediction and often require additional training stages or sequential inference. Motion and temporal evolution therefore remain modeled only indirectly, which motivates a more explicit treatment of world dynamics.

World models meet this demand by furnishing an explicit prior over how the environment evolves under motion. Building on this principle, recent World-Action Models (WAMs) in embodied intelligence,

such as DreamZero [53] and LingBot-VA [24], jointly predict future observations and actions through pretrained video-generation backbones. This world-action paradigm has recently been adopted in autonomous driving. DriveLaW [50] and DriveWAM [43] jointly train a video predictor and a planner, allowing anticipated scene dynamics to inform trajectory generation. Nevertheless, existing driving WAMs commonly follow an imagine-then-act pipeline in which the planner conditions its output on generated future frames. This design places costly video synthesis inside the real-time planning loop and substantially increases inference latency, see Fig. 1.

Crucially, explicit future synthesis is unnecessary for effective world-action learning. Fast-WAM [54] shows that video co-training benefits action prediction primarily through trainingtime representation learning rather than test-time future imagination. Building on this insight, we

![](images/5470f6300d892047f974bba1d94635e6d32864a14ca87213ac744613d3772ed1.jpg)  
Figure 1: SimWAM achieves the best PDMS with substantially lower latency than world-modelbased planners on NAVSIM.

introduce SimWAM, a plain yet effective World-Action Model that uses video generation as a training signal while retaining direct trajectory prediction at inference. SimWAM jointly trains a pretrained video expert and a lightweight action expert with flow matching. A simple isolated attention mask prevents the action expert from accessing future frames, which allows the entire video branch to be removed after training. The resulting action dit retains the traffic-dynamics prior without auxiliary motion modules or video generation at deployment. This decoupling also makes the video expert replaceable, allowing more advanced video generators to improve the learned prior without changing the action expert or inference pipeline. Furthermore, we reformulate the deterministic flow ODE as a stochastic SDE and reinforce the action expert with GRPO [14, 30], enabling diverse maneuver exploration and direct optimization of a compositional driving reward. Rather than claiming algorithmic superiority, this work establishes a simple and solid WAM baseline for exploring the potential of generic video models in autonomous driving.

The advantages of SimWAM arise from three aspects: 1) SimWAM effectively transfers traffic dynamics priors from a pretrained video generator to the planner without auxiliary motion modules. 2) Thanks to the isolated attention mask, the action expert remains independent of future frames, allowing the video expert to be removed for efficient inference and direct reward optimization. 3) The decoupled architecture seamlessly accommodates more advanced video generators without modifying the action expert or inference pipeline.

Experiments on the NAVSIM benchmark [10] validate the effectiveness of this simple design. SimWAM achieves 91.5 PDMS with substantially lower inference latency than state-of-the-art planners based on world models, as shown in Fig. 1. Furthermore, our method supports different pretrained video generators and transfers zero-shot to nuScenes [6] without fine-tuning, demonstrating architectural scalability and cross-domain generalization. We hope SimWAM will serve as a strong and practical baseline for efficient world-action modeling in autonomous driving.

## 2 Related Work

## 2.1 Vision-Language-Action Models for Autonomous Driving

End-to-end autonomous driving integrates perception, prediction, and planning within a unified framework. Methods such as UniAD [18] and VAD [19] reduce hand-crafted interfaces and mitigate error propagation in modular pipelines. Despite this integration, these methods are largely trained on driving observations with expert trajectory supervision, which provides limited support for explicit semantic reasoning about route intent and complex traffic interactions. Vision-Language-Action (VLA) models [12, 27, 35, 65] introduce pretrained vision-language representations to enhance driving policies with semantic knowledge and reasoning capabilities. AutoVLA [67] unifies chain-ofthought reasoning and action generation within an autoregressive framework. ORION [12] aggregates long-term visual context through a query-based temporal module and employs a large language model for scenario understanding and driving reasoning. Its generative planner further maps the resulting planning representation into multimodal trajectories. FutureSightDrive [56] and ExploreVLA [42] incorporate future image generation to model scene evolution and support trajectory planning. In contrast, our SimWAM directly transfers the motion prior of a pretrained video generator into a lightweight action expert for direct trajectory prediction.

## 2.2 World-Action Models for Autonomous Driving

World-Action Models [1, 3, 47] have recently attracted growing interest in robotics by jointly learning action prediction and image generation to capture object motion, physical interactions, task progress, and future scene evolution. DreamZero [53] adapts pretrained video generation models for generalizable robotic control. LingBot-VA [24] unifies visual prediction and policy execution for closed-loop robotic control. In autonomous driving, earlier world models mainly focused on predicting and generating future driving scenes. DriveDreamer [48] learns structured traffic constraints and future driving states for controllable video generation. HERMES [65] extends this direction by unifying 3D scene understanding and future scene generation through a shared bird’seye-view representation. More recent studies [26, 43] have integrated visual world modeling with trajectory planning. Epona [59] jointly predicts future videos and trajectories through autoregressive diffusion, while DriveLaW [50] conditions a diffusion planner on latent representations produced by its video generator. These methods follow an imagine-then-act paradigm in which trajectory planning remains coupled with future visual generation during inference. In contrast, SimWAM uses the video generator to learn a motion prior during training and retains only the lightweight action expert for trajectory prediction at inference.

## 2.3 Reinforcement Learning for Autonomous Driving

Imitation learning trains autonomous driving policies to reproduce expert trajectories, but this objective confines learning to demonstrated behavior and only indirectly reflects overall driving quality. Reinforcement learning provides a complementary refinement stage that directly optimizes driving policies with task-level rewards. CarPlanner [58] uses expert guided rewards to improve large scale trajectory planning. Raw2Drive [52] refines driving policies with raw sensor inputs and privileged world models. Recent studies [13, 27, 42, 67] have further introduced reinforcement learning into Vision-Language-Action driving models. MindDrive [13] improves online exploration by optimizing high level decisions and continuous action generation with separate LoRA parameterizations. CritiqueDriveVLM [34] applies verifier guided reinforcement learning to improve driving reasoning and distills the learned capability into an efficient policy. These methods mainly reinforce languagemediated reasoning or high-level decisions in VLA planners. Our SimWAM instead reinforces a self-contained action expert for direct continuous trajectory prediction after video-action co-training.

## 3 Preliminary

Flow matching. We model both trajectories and future frames with rectified flow [29, 33]. Given a clean target x and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , the linear interpolation $x _ { \tau } = ( 1 { - } \tau ) x + \tau \epsilon$ $( \tau \in [ 0 , 1 ] )$ has constant velocity $\epsilon - x .$ , which a network $v _ { \theta }$ learns to predict under conditioning c:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { \epsilon } , \boldsymbol { \tau } } \left[ \| \boldsymbol { v } _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { \tau } , \boldsymbol { \tau } , c ) - ( \boldsymbol { \epsilon } - \boldsymbol { x } ) \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{1}
$$

Sampling integrates the probability-flow ODE dx $\rangle _ { \tau } = v _ { \theta } ( x _ { \tau } , \tau , c )$ dτ from noise $( \tau { = } 1 )$ to data $( \tau { = } 0 )$ .

From ODE to SDE. The deterministic ODE generates a single trajectory and lacks a tractable transition density. These limitations restrict exploration over alternative driving trajectories and preclude policy-gradient optimization. Following Flow-GRPO [30], we therefore transform the ODE into an SDE that preserves the same marginal distributions $p _ { \tau } ( x _ { \tau } )$ , defined as:

$$
\begin{array} { r } { \mathrm { d } x _ { \tau } = \Big [ v _ { \theta } ( x _ { \tau } , \tau ) + \frac { \sigma _ { \tau } ^ { 2 } } { 2 \tau } \big ( x _ { \tau } + ( 1 - \tau ) v _ { \theta } ( x _ { \tau } , \tau ) \big ) \Big ] \mathrm { d } \tau + \sigma _ { \tau } \mathrm { d } w , \qquad \sigma _ { \tau } = a \sqrt { \frac { \tau } { 1 - \tau } } , } \end{array}\tag{2}
$$

![](images/cfe28db81045e206303fae76ea1356cae97a3cdb5ed37cca2d16c4a5b4ccd9f9.jpg)  
Figure 2: Overview of SimWAM. During joint training, the video and action DiTs learn future dynamics and trajectory generation via MoT, while the attention mask restricts the action tokens to the current observation. Only the action DiT is retained for inference and reinforcement learning.

where dw is a Wiener increment and a controls the noise scale. Each Euler-Maruyama step yields an isotropic Gaussian transition $\pi _ { \boldsymbol { \theta } } ( x _ { \tau - \Delta \tau } \ | \ x _ { \tau } ) \ = \ { \mathcal { N } } \big ( \mu _ { \boldsymbol { \theta } } ( x _ { \tau } , \tau ) , \sigma _ { \tau } ^ { 2 } \Delta \tau I \big )$ with tractable loglikelihoods for importance sampling.

## 4 Method

We present SimWAM as a plain yet solid world-action model for end-to-end autonomous driving, as illustrated in Fig. 2. A pretrained video expert transfers traffic dynamics knowledge to a lightweight action expert through joint flow matching. An isolated attention mask keeps action prediction independent of future frames, allowing the video branch to be removed after training. The action branch directly predicts trajectories and is further optimized via reinforcement learning.

## 4.1 Model Architecture

Problem formulation. We consider end-to-end trajectory planning from a front-camera observation $o _ { t }$ , the ego state $s _ { t }$ containing velocity, acceleration, and yaw rate, and a navigation command l. The planner predicts an ego trajectory $a _ { t + 1 : t + H } = ( a _ { t + 1 } , \dots , a _ { t + H } )$ in the ego-vehicle coordinate frame, where each waypoint $a _ { i } = \left( x _ { i } , y _ { i } , \theta _ { i } \right)$ specifies the planned position and heading. Existing driving WAMs [43, 50] commonly adopt an imagine-then-act factorization, expressed as:

$$
p _ { \theta } ( a _ { t + 1 : t + H } \mid o _ { t } , s _ { t } , l ) = \int p _ { \theta } ( z _ { t + 1 : t + N } \mid o _ { t } , s _ { t } , l ) p _ { \theta } ( a _ { t + 1 : t + H } \mid o _ { t } , s _ { t } , l , z _ { t + 1 : t + N } ) \mathrm { d } z _ { t + 1 : t + N } ,\tag{3}
$$

which first synthesizes the future driving-scene latents $z _ { t + 1 : t + N }$ and then conditions trajectory generation on them. This factorization places costly video generation inside the real-time planning loop. SimWAM instead retains a simple and direct policy interface, expressed as:

$$
p _ { \theta } ( a _ { t + 1 : t + H } \mid o _ { t } , s _ { t } , l ) = p _ { \theta } \big ( a _ { t + 1 : t + H } \mid z ( o _ { t } ) , s _ { t } , l \big ) ,\tag{4}
$$

where $z ( o _ { t } )$ is the representation produced from the current observation. Traffic dynamics knowledge is acquired entirely during training. Consequently, inference requires neither future-scene latents nor auxiliary motion modules and remains as efficient as direct trajectory prediction.

Video expert. The video expert is a video Diffusion Transformer [37] initialized from Wan2.2- 5B [46], together with its video VAE [21] and T5 [40] text encoder. The VAE maps each driving frame into latent tokens, while the navigation command enters through T5 cross-attention. The current frame serves as a clean condition, and the N future frames are noised and reconstructed with flow matching. This standard video-generation objective supplies the action expert with a traffic-aware motion prior without introducing a driving-specific prediction module.

Action expert. The action expert is a lightweight Diffusion Transformer with hidden size $d _ { a } = 1 0 2 4$ Conditioned on $c = \{ z ( o _ { t } ) , s _ { t } , l \}$ , it predicts the trajectory velocity field $v _ { \theta _ { a } } \left( a _ { t + 1 : t + H } ^ { \tau } , \tau , c \right)$ via flow matching, where a small MLP embeds the ego state. Integrating the ODE maps noise to a planned trajectory. At inference, we retain only the lightweight action expert and discard the video DiT.

Co-training. The two experts interact only through shared attention [54] and retain their original architectures. Joint flow matching over video and trajectory modalities allows future-scene prediction to shape the observation representation used for planning. The joint objective is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { F M } } ^ { \mathrm { a c t } } + \lambda \mathcal { L } _ { \mathrm { F M } } ^ { \mathrm { v i d } } , } \end{array}\tag{5}
$$

where $\mathcal { L } _ { \mathrm { F M } } ^ { \mathrm { a c t } }$ and $\mathcal { L } _ { \mathrm { F M } } ^ { \mathrm { v i d } }$ instantiate Eq. 1 on the action trajectory $a _ { t + 1 : t + H }$ and the future-frame latents $z _ { t + 1 : t + N }$ , and λ balances the two terms.

Reinforcement. The preceding stage trains the action expert through imitation learning. However, imitation learning relies exclusively on expert trajectories, constraining the policy to the behavior and quality of the demonstrations. We therefore introduce reinforcement learning (RL) to optimize trajectory generation directly toward driving quality. The deterministic flow ODE lacks the stochasticity required to explore diverse maneuvers and provides no tractable transition likelihoods for policy optimization. Following Flow-GRPO [30], we replace the ODE with the marginal-preserving SDE in Eq. 2 and sample a group of G candidate trajectories for each scenario. Each candidate is evaluated using the compositional NAVSIM PDM reward [10], from which group-relative advantages are derived for the clipped policy update [14, 41]. During this RL stage, we focus on the hard navtrain scenarios with the lowest PDMS after imitation learning. To preserve the distilled motion prior and maintain a simple planner, we update only the LoRA adapters [16] of the action expert.

## 4.2 Isolated Attention Mask

SimWAM aims to exploit video generation during training while eliminating its computational cost at inference. To this end, we introduce an isolated attention mask that decouples the action expert from the video branch. As shown in Fig. 2, the shared attention stream contains the current observation latents $z ( o _ { t } )$ , the future frame latents $z _ { t + 1 : t + N }$ , and the action tokens. Both future frame tokens and action tokens attend to $z ( o _ { t } )$ , while remaining mutually invisible. The action expert learns from the shared observation representation without depending on future frame tokens. This mask constitutes the only structural modification required to separate the two experts.

Thanks to this separation, future video generation serves purely as a training signal that enriches the observation representation with traffic dynamics. The action expert remains self-contained and directly predicts trajectories from the current inputs. Consequently, the video DiT and future-frame decoder could be discarded after training, avoiding future scene generation and substantially reducing inference latency. SimWAM thus retains the benefits of video-based motion learning without carrying the video expert into deployment. The same property also allows reinforcement learning to optimize the standalone action expert independently (§4.1).

## 4.3 Flexibility of SimWAM

The structural simplicity of SimWAM naturally yields flexibility in both architecture and model scale. The two experts share no weights and exchange information only through the attention stream. Consequently, neither expert depends on the internal parameterization of the other, and each can be replaced or resized independently.

Video generator flexibility. Thanks to this simple interface, SimWAM can seamlessly accommodate different pretrained video generators. The action expert accesses only the shared observation representation and remains independent of the video expert, VAE decoder, and generated frames. The video generator therefore functions purely as a source of training-time dynamics supervision. Replacing it with a newer or more driving domain-relevant model leaves the action expert, trajectory objective, and inference pipeline unchanged. In this sense, SimWAM can readily benefit from rapid advances in video generation without redesigning the planner.

Scale flexibility. The same simplicity also makes model capacity straightforward to scale. The video and action experts provide two independent capacity controls. A larger video generator can supply a richer motion prior during training without increasing deployment cost, since the entire video DiT branch disappears at inference. Conversely, the width and depth of the action DiT can be adjusted to meet a target latency without changing the video expert or training objective. SimWAM can thus exploit a powerful video model during training while deploying a compact planner, and it naturally supports different performance and computation budgets through one unified design.

Table 1: Comparison with state-of-the-art planners on the NAVSIM navtest benchmark. C denotes camera and L denotes LiDAR. The best learned result in each column is shown in bold.
<table><tr><td>Method</td><td>Reference</td><td>Sensors</td><td>NC↑</td><td>DAC↑</td><td>EP↑</td><td>TTC↑</td><td>C↑</td><td>PDMS↑</td></tr><tr><td>Human Agent</td><td>-</td><td>-</td><td>100.0</td><td>100.0</td><td>87.5</td><td>100.0</td><td>99.9</td><td>94.8</td></tr><tr><td colspan="9">Traditional E2E planners</td></tr><tr><td>UniAD [18]</td><td>CVPR&#x27;23</td><td>6×C</td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><td>TransFuser [8]</td><td>TPAMI&#x27;22</td><td>3×C+L</td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><td>ARTEMIS [11]</td><td>RA-L&#x27;25 AAAI&#x27;26</td><td>3×C+L</td><td>98.3</td><td>95.1</td><td>81.4</td><td>94.3</td><td>100.0</td><td>87.0</td></tr><tr><td>WorldRFT [51]</td><td>CVPR&#x27;25</td><td>3×C</td><td>97.8</td><td>96.8</td><td>81.7</td><td>94.0</td><td>100.0</td><td>87.8</td></tr><tr><td>DiffusionDrive [28]</td><td></td><td>3×C+L</td><td>98.2</td><td>96.2</td><td>82.2</td><td>94.7</td><td>100.0</td><td>88.1</td></tr><tr><td>WoTE [25]</td><td>ICCV’25</td><td>3×C+L</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.9</td><td>99.9</td><td>88.3</td></tr><tr><td>SeerDrive [57]</td><td>NeurIPS&#x27;25</td><td>3×C+L</td><td>98.4</td><td>97.0</td><td>83.2</td><td>94.9</td><td>99.9</td><td>88.9</td></tr><tr><td colspan="9">VLM-based planners</td></tr><tr><td>UniWorldVLA [32]</td><td>arXiv&#x27;26</td><td>1×C</td><td>98.7</td><td>96.7</td><td>83.2</td><td>96.1</td><td>100.0</td><td>89.4</td></tr><tr><td>DriveDreamer-Policy [66]</td><td>arXiv&#x27;26</td><td>3×C</td><td>98.4</td><td>97.1</td><td>83.5</td><td>95.1</td><td>100.0</td><td>89.2</td></tr><tr><td>Vega [68]</td><td>arXiv&#x27;26</td><td>1×C</td><td>98.9</td><td>95.3</td><td>81.6</td><td>96.1</td><td>100.0</td><td>87.9</td></tr><tr><td>ImagiDrive [23]</td><td>ICRA&#x27;26</td><td>1×C</td><td>98.6</td><td>96.2</td><td>80.5</td><td>94.5</td><td>100.0</td><td>87.4</td></tr><tr><td>AutoVLA [67]</td><td>NeurIPS&#x27;25</td><td>3×C</td><td>98.4</td><td>95.6</td><td>81.9</td><td>98.0</td><td>99.9</td><td>89.1</td></tr><tr><td>ReCogDrive [27]</td><td>ICLR&#x27;26</td><td>1×C</td><td>97.9</td><td>97.3</td><td>87.3</td><td>94.9</td><td>100.0</td><td>90.8</td></tr><tr><td>ExploreVLA [42]</td><td>ECCV’26</td><td>1×C</td><td>98.8</td><td>98.4</td><td>83.5</td><td>96.5</td><td>99.9</td><td>90.4</td></tr><tr><td>DriveVLA-W0 [26]</td><td>ICLR&#x27;26</td><td>1×C</td><td>98.7</td><td>99.1</td><td>83.3</td><td>95.3</td><td>99.3</td><td>90.2</td></tr><tr><td>SGDrive [22]</td><td>CVPR&#x27;26</td><td>1×C</td><td>98.6</td><td>97.8</td><td>85.8</td><td>96.2</td><td>100.0</td><td>91.1</td></tr><tr><td colspan="9">World-model-based planners</td></tr><tr><td>Epona [59]</td><td>ICCV’25</td><td>1×C</td><td>97.9</td><td>95.1</td><td>80.4</td><td>93.8</td><td>99.9</td><td>86.2</td></tr><tr><td>PWM [61]</td><td>NeurIPS&#x27;25</td><td>1×C</td><td>98.6</td><td>95.9</td><td>81.8</td><td>95.4</td><td>100.0</td><td>88.1</td></tr><tr><td>DriveLaW [50]</td><td>CVPR&#x27;26</td><td>1×C</td><td>99.0</td><td>97.1</td><td>81.3</td><td>96.7</td><td>100.0</td><td>89.1</td></tr><tr><td>DriveWAM [43]</td><td>arXiv&#x27;26</td><td>1×C</td><td>98.3</td><td>98.1</td><td>84.3</td><td>95.2</td><td>100.0</td><td>90.1</td></tr><tr><td>SimWAM (ours)</td><td>一</td><td>1×C</td><td>98.4</td><td>98.7</td><td>86.4</td><td>95.5</td><td>100.0</td><td>91.5</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Dataset and benchmark. We evaluate SimWAM on NAVSIM [10], a non-reactive planning benchmark built from the OpenScene subset of nuPlan [7]. NAVSIM removes trivial stationary and constant-velocity cases while retaining challenging intersections, merges, and turns. We train on navtrain with 103,288 scenes and evaluate on the held-out navtest split with 12,146 scenes. Although each scene provides multi-view cameras, LiDAR, and ego states, SimWAM uses only the front camera. The primary metric is the Predictive Driver Model Score (PDMS), which combines five closed-loop submetrics according to:

$$
\mathrm { P D M S } = \prod _ { m \in \{ \mathrm { N C } , \mathrm { D A C } \} } r _ { m } \times \frac { \sum _ { m \in \{ \mathrm { E P } , \mathrm { T T C } , \mathrm { C } \} } w _ { m } \cdot r _ { m } } { \sum _ { m \in \{ \mathrm { E P } , \mathrm { T T C } , \mathrm { C } \} } w _ { m } } ,\tag{6}
$$

where NC and DAC denote No Collision and Drivable Area Compliance. They serve as binary penalty factors. EP, TTC, and C denote Ego Progress, Time-to-Collision, and Comfort and form the weighted quality term.

Implementation details. The video expert is initialized from Wan2.2-5B [46], together with its VAE and T5 encoder. The action expert is a lightweight DiT with a hidden size of 1024. Unless otherwise specified, all experiments use a single front camera at a resolution of 384×672. The action expert predicts 8 waypoints over 4 s at 2 Hz, while the video expert predicts the corresponding 8 future frames. In the joint training stage, we adopt AdamW [36] and a cosine learning rate schedule with an

Table 2: Component analysis.
<table><tr><td>Configuration</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>PDMS</td></tr><tr><td>Action-only</td><td>97.6</td><td>95.7</td><td>81.7</td><td>92.6</td><td>86.6</td></tr><tr><td>+ Video</td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td></tr><tr><td>+ RL</td><td>98.4</td><td>98.7</td><td>86.4</td><td>95.5</td><td>91.5</td></tr></table>

Table 3: Attention mask analysis.
<table><tr><td>Mask</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>PDMS</td></tr><tr><td>Bidirectional</td><td>98.4</td><td>98.0</td><td>84.7</td><td>95.1</td><td>90.2</td></tr><tr><td>Action→video</td><td>98.5</td><td>97.8</td><td>84.3</td><td>95.5</td><td>90.1</td></tr><tr><td>Isolated</td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td></tr></table>

Table 4: Video backbone flexibility.
<table><tr><td>Video model</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>PDMS</td></tr><tr><td>LTX-Video [15]</td><td>98.1</td><td>97.2</td><td>83.1</td><td>94.3</td><td>88.7</td></tr><tr><td>Wan2.1-1.3B [46]</td><td>98.6</td><td>98.1</td><td>84.0</td><td>95.9</td><td>90.2</td></tr><tr><td>Cosmos2.5 [2]</td><td>98.7</td><td>98.0</td><td>84.2</td><td>96.0</td><td>90.4</td></tr><tr><td>Wan2.2-5B</td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td></tr></table>

Table 5: Action expert scaling.
<table><tr><td>Action DiT</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>PDMS</td></tr><tr><td>0.21B</td><td>98.6</td><td>97.8</td><td>84.0</td><td>95.4</td><td>89.9</td></tr><tr><td>0.45B</td><td>98.6</td><td>97.9</td><td>83.8</td><td>95.9</td><td>90.1</td></tr><tr><td>1.02 B</td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td></tr></table>

initial learning rate of $1 0 ^ { - 4 }$ . We train the model for 100 epochs and set λ=1. During reinforcement learning (RL), we optimize only rank-32 LoRA adapters [16] with a scale of α=16 on the attention projections of the action expert. We sample G=8 trajectories per scenario and use a learning rate of $\bar { 5 } { \times } \bar { 1 } 0 ^ { - 5 }$ . RL focuses on challenging navtrain scenes where the imitation policy obtains a PDMS below 90, while evaluation always covers the full navtest split.

## 5.2 Main Results

As shown in Tab. 1, we compare SimWAM with recent state-of-the-art planners on NAVSIM navtest. Even with only a single front camera, our method achieves 91.5 PDMS and establishes a new state of the art in end-to-end planning. Our SimWAM notably surpasses the strongest VLM-based planner, SGDrive [22], by 0.4 points. ExploreVLA [42] explicitly incorporates future image prediction to enhance VLA planning, yet still trails our method by 1.1 points. Compared with recent imaginethen-act WAM planners, SimWAM effectively internalizes video dynamics priors during training and directly generates trajectories without costly future prediction at inference. Under the same single-camera setting, SimWAM consistently outperforms DriveLaW [50] and DriveWAM [43] by 2.4 and 1.4 points, respectively. Among the world-model-based planners, our method achieves the best DAC and EP while maintaining competitive NC and TTC. Together with the latency results in Fig. 1, these results compellingly demonstrate that training-time world modeling can deliver both superior planning quality and remarkably efficient inference.

## 5.3 Analysis

Component analysis. We analyze the contributions of different training stages, as listed in Tab. 2. The action-only DiT establishes a solid baseline with 86.6 PDMS. Joint training with the video expert consistently improves all metrics and substantially raises PDMS to 90.3. These broad improvements demonstrate that future-video supervision effectively transfers traffic-dynamics priors into the shared observation representation, enabling the action expert to better understand scene evolution without auxiliary motion modules or future generation at inference. RL further improves PDMS to 91.5 by directly optimizing driving quality beyond trajectory imitation. Although minor trade-offs occur in individual metrics, the improvement confirms that RL better balances safety, compliance, and progress. Video co-training and RL thus contribute complementary gains, improving PDMS by 4.9 points while preserving the simplicity and efficient inference of the standalone action expert.

Attention mask. The attention pattern determines how information flows between the two experts, and we compare three alternatives in Tab. 3. Both bidirectional and action→video attention couple action prediction with future video tokens, making the video branch indispensable at deployment. In contrast, our isolated mask cleanly decouples the action expert from future prediction while retaining the benefits of joint learning through the current observation. Despite its simpler dependency structure, the isolated mask achieves the best PDMS of 90.3, along with the strongest NC and TTC. These results suggest that exposing the action branch to the video tokens provides no measurable benefit in our setting, while the isolated design enables efficient inference without future generation.

Video backbone flexibility. SimWAM accommodates diverse pretrained video generators through a unified attention interface, as summarized in Tab. 4. Wan2.1-1.3B and Wan2.2-5B achieve comparable PDMS values of 90.2 and 90.3, confirming that our method is not tied to a particular video backbone. Notably, the newer Cosmos-Predict2.5 [2] has been pretrained on driving videos and therefore provides stronger driving-relevant dynamics priors, achieving the best PDMS of 90.4 together with the strongest EP and TTC. By comparison, the lightweight LTX-Video reaches 88.7 PDMS, suggesting that the quality of the video prior remains important. These results highlight that SimWAM can seamlessly absorb stronger and more domain-relevant priors from advanced video generation models while preserving the action expert and inference pipeline.

Table 6: Zero-shot generalization on the nuScenes open-loop planning benchmark. ∗ represents only using the front camera as input.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Finetune</td><td rowspan="2">Input</td><td rowspan="2">Auxiliary Supervision</td><td colspan="4">L2 (m)↓</td><td colspan="4">Collision Rate (%) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>ST-P3 [17]</td><td></td><td>Camera</td><td>Map&amp;Box&amp;Depth</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><td>UniAD [18]</td><td>√√</td><td>Camera</td><td>Map&amp;Box&amp;Motion</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><td>OccNet [45]</td><td>S</td><td>Camera</td><td>3D-Occ&amp;Map&amp;Box</td><td>1.29</td><td>2.13</td><td>2.99</td><td>2.14</td><td>0.21</td><td>0.59</td><td>1.37</td><td>0.72</td></tr><tr><td>OccWorld [62]</td><td>√</td><td>Camera</td><td>3D-Occ</td><td>0.52</td><td>1.27</td><td>2.41</td><td>1.40</td><td>0.12</td><td>0.40</td><td>2.08</td><td>0.87</td></tr><tr><td>VAD-Tiny [19]</td><td>√</td><td>Camera</td><td>Map&amp;Box&amp;Motion</td><td>0.60</td><td>1.23</td><td>2.06</td><td>1.30</td><td>0.31</td><td>0.53</td><td>1.33</td><td>0.72</td></tr><tr><td>VAD-Base [19]</td><td>√√</td><td>Camera</td><td>Map&amp;Box&amp;Motion</td><td>0.54</td><td>1.15</td><td>1.98</td><td>1.22</td><td>0.04</td><td>0.39</td><td>1.17</td><td>0.53</td></tr><tr><td>GenAD [63]</td><td></td><td>Camera</td><td>Map&amp;Box&amp;Motion</td><td>0.36</td><td>0.83</td><td>1.55</td><td>0.91</td><td>0.06</td><td>0.23</td><td>1.00</td><td>0.43</td></tr><tr><td>Doe-1 [64]</td><td>v√</td><td> ${ \mathrm { C a m e r a } } ^ { * }$ </td><td>QA</td><td>0.50</td><td>1.18</td><td>2.11</td><td>1.26</td><td>0.04</td><td>0.37</td><td>1.19</td><td>0.53</td></tr><tr><td>Epona [59]</td><td></td><td> ${ \mathrm { C a m e r a } } ^ { * }$ </td><td>None</td><td>0.61</td><td>1.17</td><td>1.98</td><td>1.25</td><td>0.01</td><td>0.22</td><td>0.85</td><td>0.36</td></tr><tr><td>DriveVA [31]</td><td>x</td><td> ${ \mathrm { C a m e r a } } ^ { * }$ </td><td>None</td><td>0.33</td><td>0.76</td><td>1.43</td><td>0.84</td><td>0.00</td><td>0.07</td><td>0.12</td><td>0.06</td></tr><tr><td>DriveWAM [43]</td><td>x</td><td> ${ \mathrm { C a m e r a } } ^ { * }$ </td><td>None</td><td>0.28</td><td>0.81</td><td>1.80</td><td>0.96</td><td>0.00</td><td>0.05</td><td>0.14</td><td>0.06</td></tr><tr><td>SimWAM (ours)</td><td>x</td><td> ${ \mathrm { C a m e r a } } ^ { * }$ </td><td>None</td><td>0.29</td><td>0.82</td><td>1.77</td><td>0.96</td><td>0.00</td><td>0.03</td><td>0.11</td><td>0.04</td></tr></table>

Table 7: Exploration sampler analysis.
<table><tr><td>Sampler</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>PDMS</td></tr><tr><td>Random noise</td><td>97.7</td><td>98.4</td><td>88.0</td><td>94.1</td><td>91.3</td></tr><tr><td>SDE</td><td>98.4</td><td>98.7</td><td>86.4</td><td>95.5</td><td>91.5</td></tr></table>

Table 8: Future-video target analysis.
<table><tr><td>Target</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>PDMS</td></tr><tr><td> $4 \mathrm { f } , 2 \mathrm { s } , 2 \mathrm { H z }$ </td><td>98.6</td><td>97.7</td><td>83.9</td><td>95.5</td><td>89.9</td></tr><tr><td> $4 \mathrm { f } , 4 \mathrm { s } , 1 \mathrm { H z }$ </td><td>98.7</td><td>97.9</td><td>84.2</td><td>95.6</td><td>90.2</td></tr><tr><td> $8 \mathrm { f } , 4 \mathrm { s } , 2 \mathrm { H z }$ </td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td></tr></table>

Action expert scalability. The parameterindependent two-expert design further allows the action expert to scale independently, as reported in Tab. 5. Increasing the action DiT from 0.21B to 1.02B steadily improves PDMS from 89.9 to 90.3. Since the experts interact through a unified attention interface, their capacities can be adjusted separately. A larger video expert can strengthen training-time supervision while leaving deployment cost unchanged, whereas the action expert can be resized according to the desired balance between planning quality and efficiency. This decoupling provides SimWAM with two complementary scaling dimensions. We adopt the 1.02B action expert for the remaining experiments.

![](images/1399d8c22a761cd4872fd4887c050e576e765916d584049cc5ed8394e0066d90.jpg)  
Figure 3: RL training dynamics. The star denotes the imitation checkpoint. Training on the hard subset consistently outperforms training on all navtrain scenes.

Cross-dataset generalization. We directly evaluate the NAVSIM-trained SimWAM on the nuScenes [6] open-loop benchmark without fine-tuning. As shown in Tab. 6, SimWAM achieves the lowest average collision rate of 0.04% without nuScenes supervision or auxiliary annotations. Its average L2 error of 0.96 m remains competitive with the strongest zero-shot baselines. L2 emphasizes agreement with dataset-specific expert trajectories, whereas collision rate more directly measures safe interaction with traffic. The strong safety performance under this domain shift shows that the learned dynamics prior transfers beyond the training benchmark.

## 5.4 Ablation Studies

We ablate RL and other key choices. Unless otherwise noted, configuration ablations use the imitation-trained world-action model, and all latency is measured on a single NVIDIA A100 GPU.

![](images/e3ee578bdb5bba378c33d61f431f0571d4eeaeced1fa8dd86d924f33162acf04.jpg)  
Concat View  
Ours-IL

Ours-RL  
Figure 4: Qualitative comparison of Ours-IL and Ours-RL on two navtest scenarios. Red ellipses highlight regions where Ours-RL progresses farther while remaining within the drivable area.  
Table 9: The effect of input resolution.  
Table 10: The effect of sampling steps.
<table><tr><td rowspan="2">Resolution</td><td colspan="5">navtest metric</td><td rowspan="2">Latency(ms)</td><td rowspan="2">Steps</td><td colspan="4">navtest metric</td><td rowspan="2">PDMS</td><td rowspan="2">Latency(ms)</td></tr><tr><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>PDMS</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td></tr><tr><td>192×352</td><td>98.2</td><td>97.1</td><td>83.0</td><td>94.9</td><td>88.9</td><td>509</td><td>1</td><td>97.4</td><td>91.3</td><td>79.1</td><td>83.3</td><td>68.9</td><td>115</td></tr><tr><td>384×672</td><td>98.7</td><td>98.0</td><td>83.9</td><td>95.9</td><td>90.3</td><td>518</td><td>5</td><td>98.6</td><td>97.9</td><td>84.0</td><td>95.6</td><td>90.1 90.3</td><td>297</td></tr><tr><td>768×1344</td><td></td><td>98.1</td><td>84.3</td><td>96.1</td><td>90.6</td><td>573</td><td>10 20</td><td>98.7 98.6</td><td>98.0 98.0</td><td>83.9 83.9</td><td>95.9 95.8</td><td>90.2</td><td>518 968</td></tr><tr><td></td><td>98.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Exploration sampler. RL requires diverse trajectory candidates, whereas the original flow ODE is deterministic. We therefore compare two stochastic sampling strategies in Tab. 7. Native random perturbations encourage exploration and improve EP, but noticeably degrade NC and TTC due to less structured maneuvers. In contrast, the marginal-preserving SDE explores diverse yet plausible trajectories while providing tractable transition likelihoods for policy optimization. It consequently achieves a better overall balance and 91.5 PDMS. We therefore adopt the SDE throughout RL.

RL training dynamics. We then compare RL training on the full navtrain set and a challenging subset with imitation PDMS below 90 in Fig. 3. Training on the challenging subset consistently outperforms training on all scenes and steadily improves PDMS to a peak of 91.5 at 15k steps. These difficult scenarios expose clearer differences among sampled trajectories and consequently provide more informative reward signals for policy optimization. In contrast, many scenes in the full set are already well handled by imitation learning, contributing limited learning signals and diluting the benefit of RL. Both curves decline slightly beyond 15k steps, indicating diminishing returns from prolonged optimization.

Prediction horizon and frame density. We further examine the temporal configuration of futurevideo supervision in Tab. 8. Shortening the prediction horizon from 4 s to 2 s noticeably reduces PDMS, whereas maintaining the 4 s horizon with half as many frames recovers most of the performance. This comparison indicates that broad temporal coverage is more important than dense frame sampling for learning traffic dynamics. The full 4 s target at 2 Hz achieves the strongest performance.

Input resolution. We next study the trade-off between visual detail and inference efficiency in Tab. 9. Increasing the resolution from 192×352 to 384×672 substantially improves PDMS by 1.4 points with only 9 ms of additional latency. Further increasing the resolution to 768×1344 yields merely a 0.3 point gain while adding considerably more computation. These results identify 384×672 as the most favorable balance between planning accuracy and inference efficiency.

Number of sampling steps. Finally, we investigate the convergence of the action flow sampler in Tab. 10. A single sampling step is insufficient to produce well-refined trajectories, whereas five steps already recover most of the performance. Increasing the budget to ten steps achieves the highest PDMS of 90.3. Using twenty steps provides no further improvement while nearly doubling the latency, indicating that the sampler has already converged.

## 5.5 Qualitative Results

As shown in Fig. 4, we compare the imitation-trained and reinforced models in two scenes. The imitation-trained model produces conservative trajectories and advances only a short distance at the intersection and along the narrow street. After reinforcement, the model follows the intended route more decisively and completes a larger portion of each maneuver. Meanwhile, the trajectories remain within the drivable area and maintain safe clearance from surrounding vehicles.

## 6 Conclusion

In this paper, we presented SimWAM, a simple yet effective and flexible world-action model for end-to-end autonomous driving. Through joint flow matching, it transfers traffic-dynamics priors from a pretrained video expert to a lightweight action expert. An isolated attention mask decouples action prediction from future frames, allowing the video branch to be discarded after training for direct trajectory planning. This design also makes the video backbone replaceable and the two experts independently scalable, enabling stronger video priors without additional deployment cost while adapting the action expert to different efficiency requirements. Reinforcement learning further aligns the standalone planner with driving quality beyond imitation. Using only a single front camera, SimWAM achieves 91.5 PDMS on NAVSIM with efficient direct trajectory inference and transfers zero-shot to nuScenes. These results show that training-time world modeling could support strong real-time planning without costly test-time imagination.

## References

[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

[2] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

[3] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. In Proc. ofRobotics: Science and Systems, 2025.

[5] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316, 2016.

[6] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2020.

[7] Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2021.

[8] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

[9] Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, and Alexey Dosovitskiy. End-to-end driving via conditional imitation learning. In Proc. of the IEEE Int. Conf. on Robotics and Automation, 2018.

[10] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In Proc. ofAdvances in Neural Information Processing Systems, 2024.

[11] Renju Feng, Ning Xi, Duanfeng Chu, Rukang Wang, Zejian Deng, Anzheng Wang, Liping Lu, Jinxiang Wang, and Yanjun Huang. Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE Robotics and Automation Letters, 2025.

[12] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proc. of IEEE Intl. Conf. on Computer Vision, 2025.

[13] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Hongwei Xie, Bing Wang, Guang Chen, Dingkang Liang, and Xiang Bai. Minddrive: A vision-language-action model for autonomous driving via online reinforcement learning. In Proc. of European Conference on Computer Vision, 2026.

[14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Nature, 2025.

[15] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

[16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In Proc. of Intl. Conf. on Learning Representations, 2022.

[17] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In Proc. of European Conference on Computer Vision, 2022.

[18] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2023.

[19] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In Proc. ofIEEE Intl. Conf. on Computer Vision, 2023.

[20] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. In Proc. of Conference on Robot Learning, 2024.

[21] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In Proc. ofIntl. Conf. on Learning Representations, 2014.

[22] Jingyu Li, Junjie Wu, Dongnan Hu, Xiangkai Huang, Bin Sun, Zhihui Hao, Xianpeng Lang, Xiatian Zhu, and Li Zhang. Sgdrive: Scene-to-goal hierarchical world cognition for autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2026.

[23] Jingyu Li, Bozhou Zhang, Xin Jin, Jiankang Deng, Xiatian Zhu, and Li Zhang. Imagidrive: A unified imagination-and-planning framework for autonomous driving. In Proc. ofthe IEEE Int. Conf. on Robotics and Automation, 2026.

[24] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. In Proc. of Robotics: Science and Systems, 2026.

[25] Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. In Proc. of IEEE Intl. Conf. on Computer Vision, 2025.

[26] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. In Proc. of Intl. Conf. on Learning Representations, 2026.

[27] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. In Proc. of Intl. Conf. on Learning Representations, 2026.

[28] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2025.

[29] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In Proc. of Intl. Conf. on Learning Representations, 2023.

[30] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. In Proc. ofAdvances in Neural Information Processing Systems, 2025.

[31] Mengmeng Liu, Diankun Zhang, Jiuming Liu, Jianfeng Cui, Hongwei Xie, Guang Chen, Hangjun Ye, Michael Ying Yang, Francesco Nex, and Hao Cheng. Driveva: Video action models are zero-shot drivers. In Proc. of European Conference on Computer Vision, 2026.

[32] Qiqi Liu, Huan Xu, Jingyu Li, Bin Sun, Zhihui Hao, Dangen She, Xiatian Zhu, and Li Zhang. Uni-world vla: Interleaved world modeling and planning for autonomous driving. arXiv preprint arXiv:2603.27287, 2026.

[33] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In Proc. ofIntl. Conf. on Learning Representations, 2023.

[34] Zhaohong Liu, Hao Ye, Xianlin Zhang, and Mengshi Qi. Critiquedrivevlm: From verifier-guided reinforcement learning to latent thought distillation for autonomous driving. arXiv preprint arXiv:2607.04179, 2026.

[35] Zhe Liu, Runhui Huang, Rui Yang, Siming Yan, Zining Wang, Lu Hou, Di Lin, Xiang Bai, and Hengshuang Zhao. Drivepi: Spatial-aware 4d mllm for unified autonomous driving understanding, perception, prediction and planning. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2026.

[36] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proc. of Intl. Conf. on Learning Representations, 2019.

[37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proc. of IEEE Intl. Conf. on Computer Vision, 2023.

[38] Qihang Peng, Xuesong Chen, Chenye Yang, Shaoshuai Shi, and Hongsheng Li. Colavla: Leveraging cognitive latent reasoning for hierarchical parallel trajectory planning in autonomous driving. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2026.

[39] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In Proc. of Advances in Neural Information Processing Systems, 1988.

[40] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 2020.

[41] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[42] Zihao Sheng, Xin Ye, Jingru Luo, Sikai Chen, and Liu Ren. Explorevla: Dense world modeling and exploration for end-to-end autonomous driving. In Proc. of European Conference on Computer Vision, 2026.

[43] Chen Shi, Jinrui Xu, Shaoshuai Shi, Kehua Sheng, Bo Zhang, and Li Jiang. Drivewam: Video generative priors enable scalable world-action modeling for autonomous driving. arXiv preprint arXiv:2605.28544, 2026.

[44] Shuhan Tan, Kashyap Chitta, Yuxiao Chen, Ran Tian, Yurong You, Yan Wang, Wenjie Luo, Yulong Cao, Philipp Krähenbühl, Marco Pavone, and Boris Ivanovic. Latent chain-of-thought world modeling for end-to-end autonomous driving. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2026.

[45] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, et al. Scene as occupancy. In Proc. of IEEE Intl. Conf. on Computer Vision, 2023.

[46] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[47] Hang Wang, Xin Ye, Feng Tao, Chenbin Pan, Abhirup Mallik, Burhan Yaman, Liu Ren, and Junshan Zhang. Adawm: Adaptive world model based planning for autonomous driving. In Proc. of Intl. Conf. on Learning Representations, 2025.

[48] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In Proc. of European Conference on Computer Vision, 2024.

[49] Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, et al. Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025.

[50] Tianze Xia, Yongkang Li, Lijun Zhou, Jingfeng Yao, Kaixin Xiong, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, et al. Drivelaw: Unifying planning and video generation in a latent driving world. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2026.

[51] Pengxuan Yang, Ben Lu, Zhongpu Xia, Chao Han, Yinfeng Gao, Teng Zhang, Kun Zhan, XianPeng Lang, Yupeng Zheng, and Qichao Zhang. Worldrft: Latent world model planning with reinforcement fine-tuning for autonomous driving. In Proc. of the AAAI Conf. on Artificial Intelligence, 2026.

[52] Zhenjie Yang, Xiaosong Jia, Qifeng Li, Xue Yang, Maoqing Yao, and Junchi Yan. Raw2drive: Reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). In Proc. of Advances in Neural Information Processing Systems, 2025.

[53] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[54] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[55] Ekim Yurtsever, Jacob Lambert, Alexander Carballo, and Kazuya Takeda. A survey of autonomous driving: Common practices and emerging technologies. IEEE access, 2020.

[56] Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran Liu, Yifan Bai, Zheng Pan, Mu Xu, and Xing Wei. Futuresightdrive: Thinking visually with spatio-temporal cot for autonomous driving. In Proc. ofAdvances in Neural Information Processing Systems, 2025.

[57] Bozhou Zhang, Nan Song, Xiatian Zhu, Jiankang Deng, Li Zhang, et al. Future-aware endto-end driving: Bidirectional modeling of trajectory planning and scene evolution. In Proc. of Advances in Neural Information Processing Systems, 2025.

[58] Dongkun Zhang, Jiaming Liang, Ke Guo, Sha Lu, Qi Wang, Rong Xiong, Zhenwei Miao, and Yue Wang. Carplanner: Consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2025.

[59] Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, et al. Epona: Autoregressive diffusion world model for autonomous driving. In Proc. ofIEEE Intl. Conf. on Computer Vision, 2025.

[60] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, 2025.

[61] Zhida Zhao, Talas Fu, Yifan Wang, Lijun Wang, and Huchuan Lu. From forecasting to planning: Policy world model for collaborative state-action prediction. In Proc. of Advances in Neural Information Processing Systems, 2025.

[62] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In Proc. ofEuropean Conference on Computer Vision, 2024.

[63] Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. In European Conference on Computer Vision, 2024.

[64] Wenzhao Zheng, Zetian Xia, Yuanhui Huang, Sicheng Zuo, Jie Zhou, and Jiwen Lu. Doe-1: Closed-loop autonomous driving with large world model. arXiv preprint arXiv:2412.09627, 2024.

[65] Xin Zhou, Dingkang Liang, Sifan Tu, Xiwu Chen, Yikang Ding, Dingyuan Zhang, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation. In Proc. ofIEEE Intl. Conf. on Computer Vision, 2025.

[66] Yang Zhou, Xiaofeng Wang, Hao Shao, Letian Wang, Guosheng Zhao, Jiangnan Shao, Jiagang Zhu, Tingdong Yu, Zheng Zhu, Guan Huang, et al. Drivedreamer-policy: A geometry-grounded world-action model for unified generation and planning. arXiv preprint arXiv:2604.01765, 2026.

[67] Zewei Zhou, Tianhui Cai, Seth Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In Proc. ofAdvances in Neural Information Processing Systems, 2025.

[68] Sicheng Zuo, Yuxuan Li, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Vega: Learning to drive with natural language instructions. arXiv preprint arXiv:2603.25741, 2026.