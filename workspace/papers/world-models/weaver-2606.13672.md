# WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation

ArnavKumar Jain\* Mila - Québec AI Institute Université de Montréal

Yilin Wu\* Carnegie Mellon University

Jesse Farebrother Mila - Quebec AI Institute McGill University

Gokul Swamy Carnegie Mellon University

Andrea Bajcsy Carnegie Mellon University

## Abstract

The potential impacts of world models (WMs,i.e., learned simulators) on robotics are far-reaching-policy evaluation, policy improvement, and test-time planningall with limited real-world interaction. To unlock these downstream capabilities, a WM needs to jointly satisfy three desiderata: (i) fidelity (i.e., producing simulated trajectories that correlate with reality),(ii) consistency (i.e., producing simulated trajectories that are coherent over long horizons),and (ii) efficiency (i.e., producing simulated trajectories quickly). We propose WEAVER (World Estimation Across Views for Embodied Reasoning): a WM architecture that simultaneously achieves all three desiderata,providing state-of-the-art results on robotic manipulation tasks.WEAVER is a multi-view WM trained to predict future latents and reward values via a flow-matching loss.We distill the key design decisions across model architecture, memory, and prediction objectives required to unlock the Kinds of longhorizon dynamic manipulation tasks that have confounded prior world modeling approaches.We apply WEAVER in robotic hardware, demonstrating its effectiveness at policy evaluation $( \rho = 0 . 8 7 0$ correlation with real-world success rate), policy improvement (real-world success rate improvement of 38% on top of the $\pi _ { 0 . 5 }$ robot foundation model),and test-time planning (real-world success rate improvement of 14% with a $5 - 1 0 \times$ speedup over prior WMs). WEAVER also demonstrates better performance than prior WMs when evaluated on out-of-distribution scenarios. Code, models,and videos at: https: //arnavkj1995.github.io/WEAVER/.

## 1 Introduction

World models (WMs,[13]),or learned simulators, have atracted intense interest from both academia [12,54,51,33] and industry [6,36]. This is because of the tremendous promise of WMs for robotics: the ability to both evaluate and improve policies without costly and often unsafe real-world interaction. Furthermore,WMs unlock test-time scaling when incorporated into planning algorithms.

To simultaneously deliver on the three promises of evaluation, improvement, and planning,a robot WM must jointly satisfy three core desiderata. The first is (i) fidelity: producing physically accurate predictions that correlate with real-world outcomes. The second is (ii) consistency: producing predictions that remain coherent over long prediction horizons. The third is (ii) effciency: producing predictions quickly. For example, policy evaluation and improvement require high-fidelity predictions (for handling arbitrary, visuomotor robot policies) as well as consistency (to handle multi-stage tasks). Relatedly, planning requires fast inference for dealing with the real-time requirements of robots.

![](images/366dcc55f97fabfb337450141d46a7dfbee2a2aa8b1ec698f3bfd55d78f0ac9e.jpg)  
Figure 1: We present WEAVER,a WM that satisfies three desiderata: (i) high fidelity, (ii) long-horizon consistency and (ii) efficient generation. With these, we unlock the potential for downstream policy evaluation (middle), policy improvement (top right) and Test-time Planning (bottom right).

Despite rapid progress, no existing robot WM satisfies al three desiderata in tandem. For example, video generation models [29] produce high fidelity generations at the cost of low efficiency. Similarly, JEPA-style WMs [2] have latent states that may not be decodable into the images required to evaluate arbitrary visuomotor robot policies. And while Dreamer-v4 [16] appears promising, learning an encoder from scratch rather than using a pretrained model can harm out-of-distribution robustness.

When we focus on robotic manipulation, the world modeling problem becomes even more complex, as we must handle multiple views of the scene,infer occluded objects from history,and ensure relatively high fidelity predicted world states rather than just visual aesthetics.Handling these complexities often comes at the cost of efficiency,with state-of-the-art WMs for manipulation like Ctrl-World [12] operating at far slower speeds than the real world, precluding their use in test-time planning and making policy improvement computationally challenging.

In response,we introduce WEAVER (World Estimation Across Views for Embodied Reasoning): a WM architecture that achieves (i) high fidelity, (ii) long-horizon consistency,and (iii) efficient generation, unlocking state-of-the-art performance across policy evaluation, improvement, and test-time planning on challenging robotic manipulation tasks. To achieve this trifecta of capabilities, WEAVER fuses together key design decisions from prior world modeling approaches.From the video generation community, we adopt diffusion forcing [7] and flow matching [27] (for long-horizon generation at fast inference speeds) and the use of a pretrained encoder [35] (for out-of-distribution robustness).From latent world models [16,37,12],we adopt the use of a reward prediction head to facilitate efficient evaluation without the need for an external judge model like a VLM. From JEPA [3], we adopt future latent prediction (rather than image reconstruction) as our primary training objective.Lastly,to handle the particular complexities of robot manipulation, we adopt the multi-view generation and memory architecture of Ctrl-World [12].

Put together, we end up with a gestalt whole: a WM for robotic manipulation that can be used flexibly across evaluation, improvement,and planning. On a suite of five manipulation tasks (from pick and place to deformable object manipulation) performed on real hardware, WEAVER demonstrates strong correlation (p = O.87O) with real-world success rate when used for evaluation, improves the real-world success rate of the To.5 [21] robot foundation model by 38% without any real-world interaction,and unlocks test-time planning 5-1O× faster than Ctrl-World [12].

## 2Related Work

Robot World Models. While world models have been explored across autonomous driving [36, 45], video games [18],and code generation [8], we focus on their application to robotics [49,12,37,42, 3] - more specifically visual manipulation. While improvements in video generation [47,4] have lead to high (i) fidelity WMs [29,12,37,33,12,10,36], these WMs are often not (ii) efficient enough to use for test-time planning.However, incorporating key ingredients from the broader vision community, like flow matching [27], diffusion forcing [7] allows us to improve the (ii) eficiency of

![](images/fab3ac75ef30b6c2b9ab60c2af782b1376826a20097049cd1889294d18203b70.jpg)  
Figure 2: WEAVER Architecture. Left: The world model encodes memory, history, and action sequences to image future rollouts in latent space. Middle: The latent verifier, equipped with reward and critic heads, selects samples with high advantage to steer the policy distribution. Right: Decoded generation corresponding to different outcomes of action sequences.

WEAVER.Furthermore,the use of pretrained video generation model encoders [35] enhances WEAVER's robustness to out-of-distribution visual inputs,while the use of pretrained decoders allows us to evaluate arbitrary visuomotor robot policies unlike JEPA-style models [3]. Finally,we adopt the latent reward and value heads of Dreamer-v4 [16] to enable (ii) efficient evaluation and planning without the need to pass decoded images to an external and often slow VLM judge like in [11].

Prior WMs [14,15,17,49, 50, 23, 22] struggle to maintain temporal (ii) consistency across long horizons. In response, we adopt the use of multi-view prediction, history,and memory from [12, 36] to ensure generations remain coherent even when gripper-object interactions are under occlusions. This is in contrast to earlier WMs like WorldGym [33],DreamerV4 [16] and DreamDojo [10].

Perhaps the most similar approaches to our own are Ctrl-World [12] and Dreamer-v4 [16]. By using techniques from the video generation community[7,27] for more (ii) efficient inference,we are able to produce higher (i) fidelity generations that are more temporally(ii) coherent in less time,Pareto dominating Ctrl-World [12]. By using a pretrained encoder [35] instead of learning one from scratch as in Dreamer-v4 [16], we likely inherit better robustness to out-of-distribution visual inputs.

Uses of World Models in Robotics. World models promise“downstream” advances in robotic policy evaluation, improvement,and test-time planning. Prior work has shown that sufficiently faithful world models can enable scalable policy evaluation [42,51, 46], while early results suggest that synthetic trajectories may also improve policies [11, 46], though the extent to which this is true remains an open question. More recently, world models have been explored for test-time planning [32, 5O], where the central challenge is generating accurately quickly for online optimization. WEAVER is designed with each of these downstream applications in mind for robotic manipulation.

## 3WEAVER: World Estimation Across Views for Embodied Reasoning

We now describe the key ingredients in WEAVER: a robot world model designed to support policy evaluation, policy improvement, and test-time planning. These downstream applications of the WM on manipulation tasks imply three key desiderata upstream: (i) fidelity across multiple views during physical interaction,(ii) consistent predictions across long-horizon interactions that can introduce occlusions,and (iii) effcient enough generation for use in a real-time planning algorithm.

To jointly satisfy these three desiderata, WEAVER fuses together a variety of ingredients. We first describe the key WM design decisions and training objective (Sec.3.1), followed by inference acceleration (Sec. 3.2) and latent-space value estimation (Sec. 3.3). We then show how, when put together, these components enable evaluation, improvement, and planning (Sec. 3.4).

Setup:Robot & Policy. We consider long-horizon robotic manipulation tasks specified by a natural language instruction $\ell \in { \mathcal { L } } .$ Let the robot's proprioceptive state (e.g., joint angles) be denoted by $q \in \mathbb { R } ^ { 8 }$ . The robot also has n RGB views of the scene (e.g., from wrist and third person cameras); let this set of multi-view images be $\mathbf { I } : = ( I ^ { 1 } , \ldots , I ^ { n } )$ . At timestep $t ,$ the robot observes both the multiview images and proprioceptive state: $o _ { t } : = ( \mathbf { I } _ { t } , q _ { t } ) \in \mathcal { O }$ . Let the robot's action be denoted by $a \in { \mathcal { A } } ( \mathrm { e . g . }$ , joint velocities). Given any $( o _ { t } , \ell )$ ,the robot's base policy $\pi _ { \theta }$ generates $\mathbf { a } _ { t } \sim \pi _ { \theta } ( \cdot \mid o _ { t } , \ell )$ , an h-step future action chunk $( \mathrm { i . e . , } \mathbf { a } _ { t } : = a _ { t : t + h } )$ ,which is then executed in the WM/environment.

World Model Architecture. Our WM maps an observation $o _ { t }$ into a latent state $z _ { t } \in \mathcal { Z }$ via a pretrained encoder $z _ { t } \sim \mathcal { E } _ { \psi } ( o _ { t } )$ . A key design choice is conditioning our world model on both a memory of every kth prior latent, $\mathbf { z } _ { t } ^ { \mathrm { m e m } } : = ( \dots , z _ { t - 2 k } , z _ { t - k } )$ , as well as a m-step history of the m most recent latents, $\mathbf { z } _ { t } ^ { \bar { \mathrm { h i s t } } } : = ( z _ { t - m } , \dots , z _ { t } )$ . Given memory, history,and an h-step action plan $\mathbf { a } _ { t }$ the WM predicts h future latents:

$$
\hat { \mathbf { z } } _ { t } \sim f _ { \phi } ( \cdot \mid \mathbf { z } _ { t } ^ { \mathrm { m e m } } , \mathbf { z } _ { t } ^ { \mathrm { h i s t } } , \mathbf { a } _ { t } ) ,\tag{1}
$$

where $\hat { \mathbf { z } } _ { t } : = \hat { z } _ { t + 1 : t + h + 1 }$ is the h-step future.We also train a reward model that scores the predicted latent's alignment with the language instruction: $\hat { \mathbf { r } } _ { t } \sim R ( \cdot \mid \hat { \mathbf { z } } _ { t } , \ell )$ where $\hat { \mathbf { r } } _ { t } : = \hat { r } _ { t + 1 : t + h + 1 }$ . To enable iterative calls to the visuomotor policy,we use a pretrained decoder to obtain future observations (camera views and proprioceptive state), $\hat { \mathbf { o } } _ { t } \sim \mathcal { D } _ { \eta } ( \bar { \mathbf { z } } _ { t } )$ ,where $\hat { \mathbf { o } } _ { t } : = \hat { o } _ { t + 1 : t + h + 1 }$ are the h-step future observations.The final prediction, $\hat { o } _ { t + h + 1 }$ , is fed back to the policy to generate the next action chunk.

## 3.1Key Design Decisions for High Fidelity, Temporally Consistent World Model Generation

Multi-View Camera Prediction. Although using multiple views (e.g., wrist and external cameras) is common practice when designing visuomotor robot policies for manipulation to handle partial observability and capture finer-grained object-gripper interactions [21], many WMs only predict a single view [33,10]. Following [12,24,44], WEAVER predicts both external and wrist-camera observations. The increased information provided by multiple views helps with (ii consistency by helping handle occlusions during manipulation. Each view $I _ { t } ^ { j }$ is encoded into $H \times W$ patch tokens using the pretrained Stable Diffusion 3 VAE encoder [9]. We project the proprioceptive state $q _ { t }$ to the same token dimension and obtain $z _ { t }$ by concatenating patch tokens and the proprioceptive token.

Proprioceptive State Prediction.In addition to future visual latents, WEAVER also predicts future proprioceptive states. We find that explicitly predicting the robot's configuration (rather than just visual observations like Ctrl-World [12j) is critical to handle contact-rich manipulation of deformable objects, where knowing the precise position of the arm and width of the gripper is often required.

Sparse Memory and Short-Term History. Temporal (ii) consistency across WM generations requires the WM to understand both what changes and what stays the same across an interaction. This is particularly challenging in manipulation, where occlusions and wrist camera viewpoint changes can cause objects and parts of the background scene to leave and enter the robot's FOv. In response, WEAVER builds upon [12] and conditions on two sets of observations when generating futures: a long-term, sparse memory, and a short-term history. In particular, memory $\mathbf { z } _ { t } ^ { \mathrm { m e m } } : = ( \dots , z _ { t - 2 k } , z _ { t - k } )$ includes every kth encoded observation to help capture longer-term context, while history $\mathbf { z } _ { t } ^ { \mathrm { { h i s t } } } : =$ $\left( z _ { t - 1 } , z _ { t } \right)$ includes the last two frames to capture the shorter-term consequences of actions.

Latent Dynamics Model. The latent dynamics model $\hat { \mathbf { z } } _ { t } \sim f _ { \phi } \big ( \cdot \mid \mathbf { z } _ { t } ^ { \mathrm { m e m } } , \mathbf { z } _ { t } ^ { \mathrm { h i s t } } , \mathbf { a } _ { t } \big )$ predicts future latent states conditioned on memory, history, and a candidate action plan. To balance (i) fidelity with (ii) efficiency, WEAVER adopts an efficient 2D transformer architecture following [16, 33], with L dynamics blocks composed of spatial attention and causal temporal attention.At each prediction step, the model conditions on latent tokens,action tokens,and flow timestep embeddings to autoregressively generate an h-step chunk.For stable training, each block uses RMSNorm [52], RoPE [39], QKNorm [19],and SwiGLU feed-forward layers [38] (see A2 for more details).

Training Objective. Similar to [16], we train our latent dynamics model $f _ { \phi }$ with a flow-matching loss [27] to predict future latents. Let $x _ { t } ^ { 1 } : = z _ { t + 1 : t + h + 1 }$ denote the ground-truth next $h$ latents and let $\hat { x _ { t } ^ { 0 } } \sim \mathcal { N } ( 0 , I )$ denote a Gaussian noise vector of the same dimension.Next,we define $x _ { t } ^ { \tau } = \tau \dot { x _ { t } ^ { 1 } } + ( 1 - \tau ) x _ { t } ^ { 0 }$ ，with $\tau \in [ 0 , 1 )$ ．Then,we train $f _ { \phi }$ to predict“velocity” $\dot { x } _ { t } ^ { 1 } - x _ { t } ^ { 0 }$ by minimizing mean squared error: $\begin{array} { r } { \mathcal { L } ^ { \mathsf { u q } } ( \phi ) = \mathbb { E } _ { x _ { t } ^ { 0 } , x _ { t } ^ { 1 } , \tau } \left[ \big \| ( x _ { t } ^ { 1 } - x _ { t } ^ { 0 } ) - f _ { \phi } ( \mathbf { z } _ { t } ^ { \mathrm { h i s t } } , \mathbf { z } _ { t } ^ { \mathrm { m e n } } , \mathbf { a } _ { t } , x _ { t } ^ { \tau } , \tau ) \big \| _ { 2 } ^ { 2 } \right] } \end{array}$ To improve long-horizon (ii) consistency, we adopt Diffusion Forcing [7], which trains the latent dynamics model with independently sampled noise levels across future timesteps. We also use SPRINT blocks [30l], which aggressively drop patch tokens in the latents to improve (ii) efficiency.

## 3.2Accelerating World Model Inference Speed

For diffusion transformer-based WMs [12,16,33] like WEAVER,latency is a product of both (a) the forward pass through the model and (b) iterative denoising. Thus,(iii) effcient generation requires tackling both of these concerns in tandem. We reduce cost (a) via the use of KV caching to memory and history tokens across denoising steps. We reduce cost (b) by adjusting the denoising process. In particular, building on diffusion forcing\_[7], we use a progressive noise schedule. Rather than using a linear schedule like in [12,16], WEAVER adopts a cosine schedule for higher (i) fidelity generation.

To further increase (i) efficiency to the level required for test-time planning, we post-train WEAVER with a rectified flow objective [28] to enable high-quality generation within a few forward passes. In particular, we first generate a high-quality latent trajectory using the denoising process, before using it as a target for secondary distillation step. See Appendix A2.3 for more implementation details.

## 3.3Accurate and Efficient Value Estimation from the World Model

Reward Model. To enable (ii) efficient scoring of a proposed action chunk without needing to (a) decode a latent into an image and (b) feed it to an external VLM judge model [11, 33], we distill the scores produced by an off-the-shelf reward model into a lightweight reward head that operates directly on latent states and language instruction l. The reward head R aggregates latent tokens with AdaPool [5], followed by MLP layers.We train R with a simple mean squared error objective.

Critic.To support truncated-horizon rollouts with the WM, WEAVER learns a critic network V that estimates the value beyond the imagined horizon. The critic shares the same latent-space design as the reward model and is trained with an MSE objective to predict bootstrapped 入-returns [4O]. Given latent rewards from R, the target is defined recursively as $\mathbf { v } _ { t } ^ { \lambda } = R ( z _ { t } , \ell ) + \gamma \Big ( ( 1 - \lambda ) V ( z _ { t + 1 } , \ell ) + \lambda \mathbf { v } _ { t + 1 } ^ { \lambda } \Big )$ $\mathbf { v } _ { t + k } ^ { \lambda } = V ( z _ { t + k } , \ell )$ . The critic is then trained by minimizing $\mathcal { L } ^ { \mathrm { c r i t i c } } ( V ) = \left\| V ( z _ { t } , \ell ) - \mathbf { v } _ { t } ^ { \lambda } \right\| _ { 2 } ^ { 2 } .$

## 3.4Downstream WM Applications: Evaluation, Improvement, Planning

By satisfying the desiderata of (i) fidelity, (ii) consistency,and (ii) effciency simultaneously, WEAVER can support the downstream capabilities of evaluation, improvement, and planning.

Policy Evaluation. For policy evaluation, we take recorded action trajectories from real-world rollouts and execute them open-loop inside WEAVER, recording predicted reward values along the way. We focus on long-horizon tasks that sometimes require 4O+ iterative evaluations of WEAVER's latent dynamics model, underscoring the importance of temporal (ii) consistency and (ii) effciency.

Policy Improvement.For policy improvement, we sample a h-step action chunk from the policy and forward simulate inside the WMK times fora total of $H = K h$ timesteps, leveraging WEAVER's (i) fidelity and (ii) consistency. After doing this B times from the same initial observation $z _ { t } .$ we collect batch of rollouts $\{ ( z _ { t } , \bar { a } _ { t : t + H - 1 } ^ { b } , \hat { z } _ { t + 1 : t + H } ^ { b } ) \} _ { b = 1 } ^ { B }$ . We then compute a Monte-Carlo estimate of the H-step advantage along each rollout: $\begin{array} { r } { \hat { A } _ { t } ^ { b } = \sum _ { \ell = 1 } ^ { H } \gamma ^ { \ell - 1 } R ( \hat { z } _ { t + \ell } ^ { b } , \ell ) + \gamma ^ { H } V ( \hat { z } _ { t + H } ^ { b } , \ell ) - V ( z _ { t } , \ell ) } \end{array}$ If the highest-scoring rollout in the batch (i $\begin{array} { r l } { . \mathrm { e } . , b ^ { \star } = } \end{array}$ argma $\mathfrak { w } _ { b \in \{ 1 , . . . , B \} } \hat { A } _ { t } ^ { b } )$ has an advantage value above some small, positive threshold $( \mathrm { i . e . , ~ } \hat { A } _ { t } ^ { b ^ { \star } } ~ > ~ \epsilon _ { \mathrm { a d v } } )$ ，we distill it into the base policy. This advantage-based filtering prevents the policy from being updated at states where all H-step sampled plans are predicted to be worse than the current expected behavior of the policy [23,1].

Test-time Planning. We adopt a single-chunk, best-of-N [26] approach to test-time scaling that doesn't involve iteratively calling the latent dynamics model. In particular, given the current observa-tion and instruction, we sample B candidate action chunks from the policy, imagine their outcomes with the world model,and execute the one with the highest advantage estimated with latent reward and critic heads.WEAVER's (iii) efficiency (both in terms of the speed of the latent dynamics model and ability to evaluate a candidate action sequence without needing to call an external VLM judge via the the use of the reward head) are critical to unlocking this test-time scaling capability.

## 4Experimental Setup

Base Policy & Hardware. Our base policy is o.5 [21],a state-of-the-art vision-language-action (VLA) policy trained on the DROID dataset [41]. We follow the DROID hardware setup and use a single Franka Emika Panda manipulator, two external Zed 2i cameras mounted on the left and right sides of the workspace, and a wrist-mounted Zed Mini camera (see Figure 10 in Appendix). The πo.5 VLA policy and our WEAVER world model use only the right camera view and the wrist camera².

<table><tr><td></td><td></td><td colspan="2">Exterior</td><td colspan="2">Wrist</td><td>Time</td></tr><tr><td>Method</td><td>NFE</td><td>FID←</td><td>FVD↓</td><td></td><td>FID↓FVD↓</td><td>(s）←</td></tr><tr><td colspan="7">DROID (val)</td></tr><tr><td>Ctrl-World</td><td>16</td><td>26.09</td><td>78.73</td><td>33.83</td><td>195.37</td><td>14.65</td></tr><tr><td></td><td>50</td><td>22.44</td><td>55.05</td><td>25.32</td><td>91.77</td><td>42.33</td></tr><tr><td>WEAVER</td><td>16</td><td>10.20</td><td>27.83</td><td>21.50</td><td>90.72</td><td> 4.78</td></tr><tr><td></td><td>50</td><td>9.51</td><td>26.54</td><td>16.75</td><td>66.89</td><td>14.25</td></tr><tr><td colspan="7">Task data (00D)</td></tr><tr><td>Ctrl-World</td><td>16</td><td>36.16</td><td>139.54</td><td>38.76</td><td>277.13</td><td>14.65</td></tr><tr><td></td><td>50</td><td>31.44</td><td>91.48</td><td>33.47</td><td>145.86</td><td>42.33</td></tr><tr><td>WEAVER</td><td>16</td><td>23.95</td><td>88.27</td><td>30.77</td><td>184.62</td><td> 4.78</td></tr><tr><td></td><td>50</td><td>23.48</td><td>87.03</td><td> 27.37</td><td>145.04</td><td>14.25</td></tr></table>

![](images/27d79c204a16ca3e9d01ae5e6da41a9cea4e32537b11ede4e6c45549b80969be.jpg)

![](images/1dd9aecab8c6a6aaff73719d3a339c1145b897951d7fcb17634813c85b8c7c0e.jpg)

![](images/f7be5a8ef0ec30da6bdf74dd514a9ac5ec5df74f8bb635e4605652bdd89ab877.jpg)  
Table 1: We report FID and FVD on DROID(val) and OOD Task datasets and inference time at different NFEs. WEAVER pareto dominates Ctrl-World on fidelity Vs inference budget (NFE and inference time).

![](images/1a82325d5c931a9002bf74cf8f16b7d74580d3775d2b351ea0dc04a652a65208.jpg)  
Figure 3: We report FID at various hori-Zon lengths and find that WEAVER is consistently better at long-horizon rollouts.

Datasets & Tasks.To align the world model with the data distribution of the base policy,we first pre-train the WEAVER world model on the DROID dataset and then fine-tune it on our real-world setup. We collect data to fine-tune the world model $\mathcal { D } _ { \mathrm { r e a l } } ^ { \mathrm { F T } }$ by running $\pi _ { 0 . 5 }$ for five real-world manipulation tasks,with 5O rollouts per task.We also collect an additional 2O rollouts per task as evaluation data $\mathcal { D } _ { \mathrm { r e a l } } ^ { \mathrm { v a l } }$ . We select tasks such that the base policy achieves at least 2O% success rate while spanning a range of capabilities from rigid object pick-and-place to deformable object manipulation and dynamic manipulation. Specifically,our tasks are: Stack Bowls (stack one bowl on another); PnP Bag (place a deformable chip bag onto a plate); PnP Marker (reorient a marker and insert it into a cup); PnP Towel (place a soft towel into a basket); and Pour Beans (pour a cup full of coffee beans into a bowl). Details on each task can be found in Appendix A1.1.

World Model Training. WEAVER is a 928M parameter model. We pretrain on the DROID dataset [41] for 1M steps with a batch size of 32 and learning rate of $1 e ^ { - 4 }$ on 4 × H100 GPUs for 10 days.For training the reward model and critic on top of WEAVER's latents,we annotate the DROID dataset with progress-rewards obtained from Robometer [25] (reduced by 1 to get negative rewards). During world model finetuning, the model is updated with a lower learning rate of $\cdot \bar { 2 e } ^ { - 5 }$ for 16k steps on our collected task data. The resulting model is used for policy evaluation, policy finetuning,and test-time planning. Like prior work [12], we downsample the steps by 3 to use frequency of 5Hz for world model imagination. We represent actions as the joint position difference between two timesteps to match the action space of the $\pi _ { 0 . 5 }$ policy. We learn an additional joint-velocity-to-position action adapter to convert between the action spaces for data generation and test-time planning (see A1.2).

## 5Results

We first study the performance of the WEAVER world model in isolation (Sec.5.1) and then in the downstream use-cases of policy evaluation, improvement, and test-time planning (Sec. 5.2).

## 5.1WEAVER Pareto-Dominates leading Manipulation World Models

We start by comparing the performance of WEAVER pre-trained only on the DROID dataset to leading multi-view manipulation world model. Ctrl-World [12] is a 1.5B-parameter diffusion model trained on the DROID dataset and is initialized from a pretrained SVD checkpoint [4].

Setup & Metrics.We evaluate both models on a validation split of the DROID dataset (256 trajectories) and an out-of-distribution dataset $\mathcal { D } _ { \mathrm { r e a l } } ^ { \mathrm { v a l } }$ collected using $\pi _ { 0 . 5 }$ VLA (100 trajectories). For each trajectory, the models are rolled out autoregressively to generate 1Os long sequences where each generation predicts the outcome of 15-step action chunks (1s) jointly. Following prior evaluations [12],we measure the visual fidelity of the decoded generations using FID [2O], and FVD [43] computed with the ground-truth videos.More metrics are detailed in the Appendix A3.1.

Test-time Planning with Advantage Filtering

![](images/2beb22b9da88de852782d293e18586b6d3dbdbe91d7a156059b6d7cc05df1720.jpg)

![](images/c392a101aa64e3a510d965b5b84e8ab7f71257c1bd6164c8f4048468ce301ea4.jpg)  
Figure 4: Reward Prediction & Test-time Planning with Advantage Filtering. (Left) Predicted rewards from WEAVER match the Robometer reward over trajectory. (Right) The highlighted action sample is the one with the best advantage value and the best outcome in WEAVER's imagination.

![](images/d99e15dfe9f52e80d9083ed16e720f3821af09595913fb3a573267e228bbfc81.jpg)  
Figure 5:We present FVD vs inference time (in seconds) for WEAVER and Ctrl-World across views on DROID (val) and OOD task datasets.We find that WEAVER pareto-dominates the leading method Ctrl-World at different NFE while using upto 16× less inference time.

Results:Perceptually High Fidelity Generations.Table 1 compares FID and FVD results for WEAVER and Ctrl-World on different evaluation datasets. WEAVER outperforms the Ctrl-World while having lower inference time. As we decrease the number of function evaluations (NFE) to decrease the latency, we find that the quality of Ctrl-World decreases more significantly than WEAVER; both models incur the highest error when predicting wrist camera viewpoints.We provide additional results to compare NFEs (Appendix A3.2), different noise schedules (Appendix A3.4) and inference speedup obtained with KVcaching (Appendix A3.3).

Results: Higher Quality at Long Horizon. We next measure how the perceptual quality of the world model's imaginations are influenced by long-horizon predictions.For both world models, we generate rollouts with long (150-step or 1Os) action sequences and measure the FID for each predicted video of the 15-step interval to estimate the generation quality with time horizon.As shown in Fig. 3,on the DROID dataset, we find that WEAVER maintains consistently lower FID compared to Ctrl-World even as inference budgets are reduced from 50 to 16 NFE. On the OOD dataset, WEAVER maintains the performance gap on exterior-view and has comparable performance on the wrist-view.

Results: WEAVER Pareto-Dominates Inference Speed vs. Quality. Next, we study how the generation quality is influenced by a fixed inference time budget as measured by NFEs and the inference time to generate 1Os chunk on a single H1Oo GPU.In Fig.5,we see that WEAVER significantly outperforms Ctrl-World at NFEs from 8,16,32,50 while enjoying significantly lower inference speeds (e.g.,30-5Os with Ctrl-World vs. 1O-3Os for WEAVER).By pareto-dominating Ctrl-World, WEAVER unlocks faster evaluation and planning as we explore below in Section 5.2.

Results: Latent Reward Prediction Accuracy. Finally, we compare WEAVER's latent reward prediction to the reward labels from RoboMeter [25], evaluated on real held-out trajectories. Fig. 4 shows the predicted reward for a rollout of the PnP Stack task; WEAVER correctly imagines key events such as grasping and stacking and the reward of the imaginations correlates with the ground-truth RoboMeter reward. In the right panel of Fig.4, we see that the advantage computed with the predicted reward is also able to distinguish diferent outcomes of the action samples. This is a promising indicator that WEAVER and it's latent reward are suitable for filtering synthetic data in Sec.5.2.1 and test-time planning in Sec. 5.2.3.

![](images/e0250a226a763ad0527e4b4c2d4a8f376b29f4b0fe87b7cf86998a78999db3f6.jpg)

![](images/e7ebc4ee2d418c9d548753a8d365251bfe6ae1af512a85e9996b451d523dd2a7.jpg)  
Figure 6: Policy Evaluation. We compare performance across different policies and world models. (Left) For $\mathrm { P n P }$ Towel,only WEAVER and WEAVER-FT accurately imagine the towel inside the basket. For Pour Beans, only WEAVER-FT captures the beans scatering across the table. (Right) Evaluation inside WEAVER-FT attains an impressively high correlation of success rate with the real world.

## 5.2WEAVER Enables Effective Evaluation, Improvement and Planning

Thus far, we have validated that WEAVER effectively balances (i) fidelity, (i) long-horizon consistency, and (iii) efficient generation. Next, we turn to the downstream uses of a world model: policy evaluation (Sec. 5.2.1), policy improvement (Sec. 5.2.2),and test-time planning (Sec.5.2.3).

## 5.2.1WEAVER Enables Effective Policy Evaluation that Tightly Correlates with Reality

First, we evaluate whether WEAVER can serve as a learned simulator for ofline policy evaluation, reducing the need for costly real-world rollouts.

Setup. Given an initial real observation, $o _ { 0 } .$ ,and action sequence, $\mathbf { a } _ { t } \sim \pi _ { \theta } ( \cdot \mid o _ { 0 } , \ell )$ ,we autoregres-sively generate imagined observations and estimate policy performance from the resulting rollout. We compare three world models: Ctrl-World pretrained on DROID [12], WEAVER pretrained on DROID, and WEAVER-FT finetuned on $\mathcal { D } _ { \mathrm { r e a l } } ^ { \mathrm { F T } }$ . To test robustness of the world models across base policy quality, we evaluate each model on rollouts from both the base $\pi _ { 0 . 5 }$ policy and a finetuned policy.

Metrics. Following prior work [51], we measure how well performance of generated rollouts correlates with real-world performance by comparing human-labeled binary success rates on imagined rollouts with real success rates on $\mathcal { D } _ { \mathrm { r e a l } } ^ { \mathrm { v a l } }$ ,averaged over 2O trials per task. We report Pearson Correla-tion coefcient [31] and maximum matrix ranking violation (MMRV) [42] (see Appendix A4.1).

Results. Fig. 6 shows that pretrained world models tend to underestimate policy performance, but WEAVER achieves better agreement with real rollouts than Ctrl-World, with higher Pearson correlation and lower MMRV. This setting is challenging because rollouts can last up to 4O seconds and require accurate long-horizon prediction. The pouring task is particularly difficult for pretrained models, likely because granular dynamics are underrepresented in DROID and inherently hard to model. After finetuning, WEAVER-FT substantially improves evaluation accuracy, increasing Pearson correlations to $\rho = 0 . 8 7$ and better matching real outcomes across policies of varying performance. The qualitative example on the left of Fig. 6 further shows that WEAVER-FT predicts the PnP Towel and Pour Beans task outcomes more accurately than the baselines.

## 5.2.2WEAVER Enables Effective Policy Improvement Without Real Interactions

Another desirable use of high-fidelity world models is synthetic data generation for policy improvement.We use the world model to sample and verify candidate action segments, then distill high-value imagined segments back into the policy [23,1].

Setup. To evaluate the utility of WEAVER towards improving policies,we explore various strategies to generate data for finetuning the policy: (1) Base Policy: the original $\pi _ { 0 . 5 } \mathrm { V L A }$ trained on DROID; (2) FT w/ Real Data: we prune segments in real trajectories using advantage estimates, yielding 1,000 segments of 36-step action chunks per task; (3)FT w/ Synthetic Data: we sample multiple segments using the base policy and WEAVER, filter them based on predicted advantage values (Sec.3.4), and retain 1,OoO segments per task,and (4) FT w/Mixed Data: combine filtered real and synthetic datasets (200O segments per task) (more details and results are presented in App. A2.2 & A4.2).

Results. Fig.7 shows that all finetuned policies substantially improve their success rate over the base policy. Notably, finetuning on synthetic data closely matches that on real data,with only a 4% average performance gap. This indicates that out synthetic data is of such a high quality that it unlocks similar policy improvement to costly real world data. Combining real and synthetic data further improves performance, increasing the average success rate by 11% over real-data finetuning alone.These results suggest that imagined rollouts from the world model provide a useful source for distillation, reducing the need for costly real-world collection and manual filtering. Fig.7 also shows improvements on contact-rich and dynamic manipulation tasks,such as more precise marker placement and bean pouring. We further study synthetic data scaling on the Pour Beans task by varying the number of imagined segments from 1,000 to 2,00O and 5,000. Fig.7 (right) shows that policy performance improves consistently with more synthetic data, eventually exceeding the performance obtained from real-data finetuning alone.

![](images/e4ad9114f70c89f325f29a2c0aec6287233662645025169a3ee3e5bc2dfe7591.jpg)

![](images/fbfaa7427f89867e87c1ffa4a6d1d9ded13b280932c20aee38e5270e28c7f5a1.jpg)  
Figure 7: (Left) Policy Improvement with Finetuning. We finetune $\pi _ { 0 . 5 }$ with multiple data sources and see that combining real and synthetic (Syn) obtained with WEAVER outperforms other variants. (Right) Data Scaling for Policy Improvement. We ablate the number of segments in synthetic data for finetuning and report the success rate across 2O trials for the Pour Beans Task.

![](images/f0885200ee26b275590cef7c22da42cd5b3e00d4a699cfc79639ba81b3ea418c.jpg)  
Figure 8: Policy Improvement Results. We present real rolouts from the base policy and the policy finetuned with synthetic data. Finetuning on synthetic data generated by WEAVER leads to improved policy performance and more successful task execution compared to the base policy.

## 5.2.3WEAVER Enables Test-Time Planning by Balancing Inference Speed and Quality

Finally, test-time search requires evaluating multiple action sequences before execution, making inference speed a key bottleneck. In contrast to planning in the the image space using reconstruction and VLM-as-a-judge [11, 37], WEAVER plans in the latent space for greater efciency [15, 23].

Setup. We use $\pi _ { 0 . 5 }$ as the base policy and sample a batch of action chunks.For each chunk,WEAVER imagines latents of future states and evaluates the advantage using the reward and critic heads. This reduces the cost of decoding predicted observa-tions and querying external VLM judges. Following the policy-improvement setup from Sec.5.2.2, we evaluate test-time planning on five tasks and compare against the base policy. We use $B = 4$ parallel samples and an imagination horizon of $\bar { h } = 1 2$ ,balancing planning quality and latency.

![](images/5c35f3ffa9f18e6dab4fced31bb85f1efbe3d119b1e36b9616129d3c5bd99427.jpg)  
Figure 9: We demonstrate test-time steering with WEAVER outperforms the base policy $\pi _ { 0 . 5 }$ by 14% when averaged across all five tasks.

Results. We report task success rate and the inference-time breakdown in the test-time planning pipeline. Fig. 9 shows that advantage-based selection steers the policy toward successful behaviors. Test-time planning improves the average success rate by 15% over the base policy, with maximum gain up to 2O%. The improvement is larger when the base policy is weaker, although it remains smaller than direct finetuning because planning is limited to a single action chunk and must operate under latency constraints. Table 7 in Appendix A4.3 shows dynamics prediction remains the main computational bottleneck. Nevertheless,WEAVER is about 2O× faster than Ctrl-World inference pipeline [12] on an RTX A6ooO Ada GPU,and batched sampling scales sublinearly with the number of candidates,showing that our inference optimizations make world-model-based test-time planning practical for real-time manipulation.

## 6Conclusion

We introduce WEAVER: a World Model for manipulation that achieves (i) high fidelity, (ij is temporally coherent,and (iii) generates efficiently. Across tasks, WEAVER shows strong correlation (p = 0.870) with real-world success rate for evaluation, improves the success rate of the To.5 policy by 38% without any real-world interaction,and unlocks test-time planning 5-1O× faster than Ctrl-World [12].

Limitations. While WEAVER unlocks the potential of large-scale world models for manipulation, several limitations remain. First, visual world models observe only a partial view of the underlying state,and tactile sensing may be necessary to resolve ambiguities. Second, incorporating physics priors could improve performance on tasks involving deformable-object manipulation. Third, gen-eration latency currently limits test-time planning to short-horizon reasoning over a single action chunk.Finally,reward supervision from RoboMeter can be noisy,motivating the development of better reward models for failure prediction. We provide further discussion in Appendix A5.

Broader Impact. This work explores large-scale world models to improve the efficiency, safety, and scalability of robotic manipulation by reducing reliance on costly real-world interaction. Imagined rollouts can support policy evaluation,improvement, and test-time planning before execution, but inaccurate or biased predictions may lead to risky decisions that are particularly important in safetycritical domains like assistive robots.Responsible deployment therefore requires careful validation, uncertainty estimation,and safeguards against exploiting errors in learned world or reward models.

## Acknowledgments

We would like to thank Jesse Zhang for helpful discussions about reward models and ROBOMETER.AJ is supported by Fonds de Recherche du Quebec (FRQ) (DOI assigned: https://doi.org/10.69777/350253), Calcul Quebec,and Canada Excellence Research Chairs (CERC) program. GKS is supported by a STTR grant. The research was enabled in part by computational resources provided by the Digital Research Alliance of Canada (https://alliancecan.ca) and Mila (https://mila.quebec). YW and AB were partially supported by the National Science Foundation (NSF) award [#2246447] and NSF CAREER award [#2441014]]. The views expressed are those of the authors and do not necessarily reflect those of NSF.

## References

[1] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. In Neural Information Processing Systems (NeurIPS), 2017.

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski,Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, (CVPR), 2023.

[3] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts,Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. CoRR,abs/2506.09985,2025.

[4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch,Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti,Adam Letts,et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR,abs/2311.15127,2023.

[5] Greyson Brothers. Robust noise attenuation via adaptive pooling of transformer outputs. In International Conference on Learning Representations (ICLR), 2025.

[6] Jake Bruce,Michael D Dennis,Ashley Edwards,Jack Parker-Holder, Yuge Shi,Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In International Conference on Machine Learning (ICML),2024.

[7] Boyuan Chen, Diego Marti Monsó, Yilun Du, Max Simchowitz, Russ Tedrake,and Vincent Sitzmann.Diffusion forcing:Next-token prediction meets full-sequence diffusion.In Neural Information Processing Systems (NeurIPS),2024.

[8] Jade Copet, Quentin Carbonneaux, Gal Cohen, Jonas Gehring, Jacob Kahn, Jannik Kossen, Felix Kreuk,Emily McMilin,Michel Meyer,Yuxiang Wei, et al. Cwm: An open-weights llm for research on code generation with world models. CoRR,abs/2510.02387, 2025.

[9]Patrick Esser, Sumith Kulal,Andreas Blattmann,Rahim Entezari, Jonas Muller,Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML),2024.

[10] Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, et al. Dreamdojo: A generalist robot world model from large-scale human videos. CoRR,abs/2602.06949,2026.

[11] Yanjiang Guo, Tony Lee, Lucy Xiaoyang Shi, Jianyu Chen, Percy Liang, and Chelsea Finn. Vlaw: Iterative co-improvement of vision-language-action policy and world model. CoRR, abs/2602.12063,2026.

[12] Yanjiang Guo,Lucy Xiaoyang Shi, Jianyu Chen,and Chelsea Finn. Ctrl-world: A controllable generative world model for robot manipulation. In International Conference on Learning Representations (ICLR),2026.

[13] David Ha and Jurgen Schmidhuber.World models. CoRR,abs/1803.10122,2018.

[14] Danijar Hafner, Timothy Lillcrap,Jimmy Ba,and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations (ICLR),2020.

[15] Danijar Hafner, Timothy P Lilicrap,Mohammad Norouzi,and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations (ICLR), 2021.

[16] Danijar Hafner, Wilson Yan, and Timothy Lillicrap.Training agents inside of scalable world models. CoRR,abs/2509.24527,2025.

[17] Nicklas Hansen, Xiaolong Wang,and Hao Su. Temporal difference learning for model predictive control. In International Conference on Machine Learning (ICML),2022.

[18] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang,Mengyin An, Yangyang Ren, et al. Matrix-game 2.O: An open-source real-time and streaming interactive world model. CoRR,abs/2508.13009,2025.

[19] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Conference on Empirical Methods in Natural Language Processing (EMNLP),2020.

[20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nesser, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Neural Information Processing Systems (NeurIPS),2017.

[21] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess,Adnan Esmail,Michael Equi, Chelsea Finn, Niccolo Fusai, et al. piO.5: a visionlanguage-action model with open-world generalization. CoRR,abs/2504.16054, 2025.

[22] Arnav Kumar Jain, Shiva Kanth Sujit, Shruti Joshi, Vincent Michalski, Danijar Hafner, and Samira Ebrahimi Kahou. Learning robust dynamics through variational sparse gating. In Neural Information Processing Systems (NeurIPS),2022.

[23] Arnav Kumar Jain, Vibhakar Mohta, Subin Kim, Atiksh Bhardwaj, Juntao Ren, Yunhai Feng, Sanjiban Choudhury,and Gokul Swamy. A smooth sea never made a skilled SAILOR:Robust imitation via learning to search. In Neural Information Processing Systems (NeurIPS),2026.

[24] Yuxin Jiang, Shengcong Chen, Siyuan Huang,Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, et al. Enerverse-ac: Envisioning embodied environments with action condition. CoRR,abs/2505.09723,2025.

[25] Anthony Liang, Yigit Korkmaz, Jiahui Zhang,Minyoung Hwang,Abrar Anwar, Sidhant Kaushik,Aditya Shah, Alex S. Huang, Luke Zettlemoyer, Dieter Fox, Yu Xiang,Anqi Li, Andreea Bobu,Abhishek Gupta, Stephen Tu, Erdem Biyik,and Jesse Zhang. Robometer: Scaling general-purpose robotic reward models via trajectory comparisons. In Robotics: Science and Systems 2026,2026.

[26] Hunter Lightman, Vineet Kosaraju, Yura Burda,Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In International Conference on Learning Representations (ICLR), 2024.

[27] Yaron Lipman,Ricky TQ Chen,Heli Ben-Hamu,Maximilian Nickel,and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR),2023.

[28] Xingchao Liu, Chengyue Gong,and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations (ICLR),2023.

[29] Zhiting Mei, Tenny Yin, Ola Shorinwa,Apurva Badithela, Zhonghe Zheng,Joseph Bruno, Madison Bland, Lihan Zha, Asher Hancock,Jaime Fernändez Fisac,et al. Video generation models in robotics-applications,research challenges,future directions. CoRR,abs/2601.07823, 2026.

[30] Dogyun Park, Moayed Haji-Ali, Yanyu Li, Wili Menapace, Sergey Tulyakov, Hyunwoo J Kim, Aliaksandr Siarohin,and Anil Kag.Sprint: Sparse-dense residual fusion for efficient diffusion transformers. In International Conference on Learning Representations (ICLR), 2026.

[31] Karl Pearson. Notes on the history of correlation. Biometrika,13(1):25-45,1920.

[32] Han Qi, Haocheng Yin,Aris Zhu, Yilun Du,and Heng Yang. Inference-time enhancement of generative robot policies via predictive world modeling. IEEE Robotics and Automation Letters, 11(5):5534-5541, 2026.

[33] Julian Quevedo,Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang,and Sherry Yang. Worldgym: World model as an environment for policy evaluation. In International Conference on Learning Representations (ICLR), 2026.

[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,Amanda Askell,Pamela Mishkin,Jack Clark,etal. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021.

[35] Robin Rombach,Andreas Blattmann,Dominik Lorenz, Patrick Esser,and Bjorn Ommer. Highresolution image synthesis with latent diffsion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),2022.

[36] Lloyd Russell, Anthony Hu,Lorenzo Bertoni, George Fedoseev, Jamie Shotton,Elahe Arani,and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving (2025). CoRR,abs/2503.20523,2025.

[37] Ansh Kumar Sharma, Yixiang Sun, Ninghao Lu, Yunzhe Zhang, Jiarao Liu, and Sherry Yang. World-gymnast: Training robots with reinforcement learning in a world model. CoRR, abs/2602.02454,2026.

[38] Noam Shazeer. Glu variants improve transformer. CoRR,abs/2002.05202,2020.

[39] Jianlin Su,Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo,and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing,568:127063,2024.

[40] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge,1998.

[41] DROID Team. Droid: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems,2024.

[42] Gemini Robotics Team, Krzysztof Choromanski, Coline Devin, Yilun Du, Debidatta Dwibedi, Ruiqi Gao, Abhishek Jindal, Thomas Kipf, Sean Kirmani, Isabel Leal, et al. Evaluating gemini robotics policies in a veo world simulator. CoRR,abs/2512.10675,2025.

[43] Thomas Unterthiner, Sjoerd Van Steenkiste,Karol Kurach,Raphael Marinier,Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. CoRR,abs/1812.01717,2018.

[44] Jianyuan Wang,Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),2025.

[45] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu,and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In European Conference on Computer Vision (ECCV),2024.

[46] Yixuan Wang, Rhythm Syed, Fangyu Wu, Mengchao Zhang,Aykut Onol, Jose Barreiros, Hooshang Nayyeri, Tony Dear, Huan Zhang, and Yunzhu Li. Interactive World simulator for robot policy training and evaluation. CoRR,abs/2603.08546,2026.

[47] Thaddäus Wiedemer, Yuxuan Li,Paul Vicol, Shixiang Shane Gu, Nick Matarese,Kevin Swersky, Been Kim,Priyank Jaini,and Robert Geirhos.Video models are zero-shot learners and reasoners. CoRR,abs/2509.20328,2025.

[48] Clark Wissler. The spearman correlation formula. Science, 22(558):309-311,1905.

[49] Philipp Wu,Alejandro Escontrela,Danijar Hafner, Pieter Abbeel,and Ken Goldberg. Daydreamer: World models for physical robot learning. In Conference on Robot Learning (CoRL), 2023.

[50] Yilin Wu, Ran Tian, Gokul Swamy,and Andrea Bajcsy. From foresight to forethought: Vlm-in-the-loop policy steering via latent alignment. In Robotics: Science and Systems (RSS), 2025.

[51] Tenny Yin, Zhiting Mei, Zhonghe Zheng,Miyu Yamane,David Wang,Jade Sceats, Samuel M Bateman, Lihan Zha, Apurva Badithela, Ola Shorinwa, et al. Playworld: Learning robot world models from autonomous play. CoRR,abs/2603.09030,2026.

[52] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Neural Information Processing Systems (NeurIPS),2019.

[53] Richard Zhang,Phillip Isola,Alexei A Efros,Eli Shechtman,and Oliver Wang. The unrea-Sonable effectiveness of deep features as a perceptual metric. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),2018.

[54] Gaoyue Zhou,Hengkai Pan,Yann LeCun,and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. In International Conference on Machine Learning (ICML),2025.

## Contents

A1 Robot Setup & Tasks 16   
A1.1 Tasks Details 16   
A1.2 Action Space 16   
A1.2.1 Action Adapter 17   
A2 Implementation Details 17   
A2.1 Architecture Details 17   
A2.2 Training Details 18   
A2.3 Inference 18   
A3 Additional World Model Evaluation Results 19   
A3.1 World Model Evaluation 19   
A3.2 Quantitative Results 20   
A3.3 Reducing inference time with KV Cache 21   
A3.4 Noise schedules during inference 21   
A3.5 Finetuning 22   
A3.6 Posttraining with Rectifed Flow 22   
A4 Additional Downstream Application Results 22   
A4.1 Policy Evaluation Results . 22   
A4.2 Policy Improvement Results 23   
A4.3 Test-Time Planning Results 23   
A5 Limitations 23   
A5.1 Partial Observability 24   
A5.2 Complex Deformable and Dynamic Interactions 24   
A5.3 Limited Planning Horizon at Test Time 25   
A5.4 Data Coverage and Embodiment Diversity 25   
A5.5 Noisy Reward Supervision 25

## A1 Robot Setup & Tasks

## A1.1Tasks Details

We collect real-world finetuning data from $\pi _ { 0 . 5 }$ on five manipulation tasks,with 5O rollouts per task on our DROID setup as shown in Fig.1O. We select tasks for which the base policy achieves at least 2O% success,ensuring that the collected rollouts contain both successful and failed executions while remaining within the policy's competence. The tasks are designed to cover a diverse set of manipulation regimes, including rigid-object pick-and-place,deformable-object manipulation, and dynamic manipulation as shown in Fig.10.

Stack Bowls requires the robot to stack one bowl on top of another. Two bowls are randomly placed on the table,and the robot must place bowl A on bowl B,where A ∈ {blue,green} and B ∈ {blue, green, pink}\A.

PnP Bag requires the robot to pick up a bag of chips and place it on a green plate. We use two types of chip bags and randomly sample one in each episode. The bag is deformable, making the grasp outcome and object motion difficult to predict.

PnP Marker requires the robot to pick up an Expo marker lying horizontally on the table and place it inside a container. The marker color is randomly selected from black and orange,and the target container is randomly selected from a paper cup and a blue mug. This task requires precise grasping and large end-effector reorientation to insert the marker vertically.

PnP Towel requires the robot to pick up a towel and place it into a basket. We use two towel variants, a folded thick red kitchen towel and a thin gray square towel,and two basket variants, orange and blue. The task is challenging because the towel is deformable,and its resulting shape depends strongly on the grasp location and, for the folded towel, the number of layers grasped.

Pour Beans requires the robot to pick up a cup containing coffee beans and pour them into a blue bowl. This task tests dynamic manipulation,as the granular motion of the beans is difcult to predict and successful execution requires accurate control of cup pose, pouring angle, and motion to avoid spilling outside the bowl.

![](images/ff036bca7d7ab3d3a0606a66deab2456ffd82a497fa4a4365bbdb8309a749b29.jpg)  
Figure 1O: Hardware setup and tasks.On the left, it is the robot setup with cameras. On the right, it shows the five tasks with top row as initial state and bottom row as one of the goal configuration.

## A1.2 Action Space

The $\pi _ { 0 . 5 }$ base policy on the DROID setup outputs joint-velocity commands for control. To match this action representation,we define the action space of our world model in joint space,avoiding potential compounding errors from converting actions into alternative representations such as Cartesian space. However, we find that directly conditioning the world model on joint velocities leads to lower generation quality. Therefore, following prior work [12], we use a lightweight action adapter to convert joint velocities into joint positions.

For policy evaluation, because the joint positions at the end of each trajectory are already available, we directly use joint positions as inputs to the world model. During test-time planning, however, predictions must be made from the joint-velocity actions proposed by the policy.We therefore use the trained action adapter to predict the corresponding joint positions,and condition the world model generation on these adapted actions. The following section describes the details of the action adapter.

## A1.2.1 Action Adapter

Overview.The action adapter is a lightweight feedforward module that bridges the world model's action representation (joint velocity commands and binary gripper signals) and the robot's observable state (absolute joint positions and gripper width). Given the robot's current state and a chunk of $T { = } 1 5$ actions produced by the world model, it predicts the resulting sequence of joint-position and gripper-position deltas, which are then integrated to obtain future absolute states.

Input representation. The model receives two groups of inputs:

· State token: the current 7-DOF joint position concatenated with the current scalar gripper position, forming a (7+1)-dimensional vector.

· Action tokens: a chunk of T joint-velocity commands $( T \times 7 )$ concatenated with T gripper actions $( T \times 1 )$ ,flattened to $\mathsf { \bar { T } } ( 7 + 1 )$ dimensions.

Both groups are concatenated into a single input vector of dimension $( 7 + 1 ) ( T + 1 ) = 1 2 8$ (for $T { = } 1 5 )$ ：

Architecture. The adapter is a three-layer MLP with hidden size 512 and SiLU activations:

$$
f _ { \boldsymbol { \theta } } : \mathbb { R } ^ { 1 2 8 } \longrightarrow \mathbb { R } ^ { T \times 8 } .
$$

The output is reshaped to $( T , 8 )$ and split into predicted joint deltas $( T \times 7 )$ and predicted gripper deltas $( \bar { T } \times 1 )$ ：

Normalization. All continuous inputs and targets are min-max normalized to $[ - 1 , 1 ]$ using perdimension 1st/99th-percentile bounds computed from the training set, which is more robust to outliers than global min/max.Gripper action commands are binarized $( \ge 0 . 5 \mapsto 1$ ,otherwise O) prior to input, reflecting their discrete open/close semantics.

Loss function. The model is trained with a weighted MSE loss on the normalized delta targets:

$$
{ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { j o i n t } } + { \lambda } _ { g } { \mathcal { L } } _ { \mathrm { g r i p p e r } } , \qquad { \lambda } _ { g } = 5 . 0 .
$$

The gripper is up-weighted because it has a much smaller dynamic range than the joint dimensions and would otherwise be under-penalized relative to its importance in grasp and place predictions.

Inference. At test time the model predicts $( \hat { \Delta } _ { \mathrm { j o i n t } } , \hat { \Delta } _ { \mathrm { g r i p p e r } } )$ ,denormalizes them, and integrates from the current state:

$$
q _ { t + k } = q _ { t } + \hat { \Delta } _ { \mathrm { j o i n t } , k } , \qquad g _ { t + k } = g _ { t } + \hat { \Delta } _ { \mathrm { g r i p p e r } , k } , \qquad k = 1 , \ldots , T .
$$

Training details. The adapter is trained for 15 epochs using Adam $( \mathrm { l r } { = } 1 0 ^ { - 4 }$ ,batch size 128) on 50 hours of proprioceptive teleoperation data. Each training sample consists of a randomly drawn window of $\bar { T } { + } \bar { 1 }$ .consecutive timesteps from an episode; the first timestep provides the current state and the remaining $T$ timesteps provide the action chunk and delta targets.

## A2Implementation Details

## A2.1 Architecture Details

We use the VAE from Stable Diffusion 3 [9] to encode $1 9 0 \times 3 2$ image frames from camera views into the latent space. Our efficient transformer architecture is a 32-layer transformer with 1536 hidden dimensions and 16 attention heads. Each layer comprises of a spatial layer that attends to all the patches in $z _ { t }$ and a causal temporal layer to attend over patches from prior observations. The actions and proproceptive states are normalized using statistics obtained with the training dataset of DROID.We obtain the reward annotations from Robometer [25] and use the progress rewards to train the reward head and the critic. The reward and critic networks uses an AdaPool [5] layer to compress the tokens to a vector, and has MLP layers with this vector concatenated with the CLIP embedding [34] (provides a representation for the language instruction).

## A2.2Training Details

WEAVER Pretraining. WEAVER has 928M paramters in total and is trained for 1M gradient steps on 4×H10O GPUs for 1O days. The pretraining is done on the DROID dataset [41]. We also maintain an exponential moving average (EMA) of model weights during training with $\beta = 0 . 9 9 9 9$ . We use a learning rate warmup for the initial 1OooO steps and keep a constant learning rate of $1 e ^ { - 4 }$ post warmup.We provide the hyperparameters in Table 2.

WEAVER Finetuning. For complex tasks like pouring that is underrepresented in the pretraining dataset, the world model is inaccurate at predictions. To mitigate covariate shift and improve generations,a potential solution is to finetune the world model with task dataset. In this work,we finetune WEAVER on a smal datasets of 25O trajectories (5O for each task) collected using the $\pi _ { 0 . 5 } \ \mathrm { V L A }$ . We finetune WEAVER for 16K gradient steps using a smaller learning rate of $2 e ^ { - 5 }$ . Other hyperparameters are similar to pretraining (as described in Table 2) and the training takes 6 hours on 4×H1Oo GPUs.

$\pi _ { 0 . 5 }$ Finetuning. We follow the original setup of the droid finetuning configuration in openpi to finetune our policy. We use the open-sourced pre-trained $\pi _ { 0 . 5 } \mathrm { - d r o i d }$ checkpoint as the base policy. Normalization statistics are inherited from the original DROID checkpoint and held fixed throughout fine-tuning to preserve compatibility with the pretrained observation encoder. All runs with dataset size smaller than 5OoO segments use a batch size of 32,a peak learning rate of $2 . 5 \times 1 0 ^ { - 6 }$ with cosine decay and no warmup, and language instructions sourced from task annotations. For datasets of size ≥5,OOO trajectories,we fine-tune for 10,0oO gradient steps with a cosine decay over 10,000 steps and warmup steps of 1000 and peak learning rate of 2.5e-5. For smaller datasets (1,000-2,000 trajectories), we reduce training to mitigate overfitting.

Reward Labeling.We label each trajectory with a per-frame progress reward using the Robometer evaluation server [25]. For each episode, we extract frames from the recorded video and downsampled to 1 fps using,selecting the right-camera view from the DROID setup. The sampled frames and the episode's language instruction are sent in a single forward pass to the Robometer eval server, which returns a per-frame progress prediction $\hat { r } _ { t } \in [ 0 , 1 ]$ representing the estimated fraction of task completion at frame t,along with an optional per-frame success probability $\hat { s } _ { t } \in [ 0 , 1 ]$

Because frames are subsampled at 1 fps, the resulting reward sequence is shorter than the original video. We realign rewards to the full video length by linear interpolation: letting $T _ { \mathrm { o r i g } }$ denote the original frame count and $T _ { \mathrm { s a m p l e d } }$ the number of inferred frames,we place sampled values at positions $\{ \left( T _ { \mathrm { o r i g } } - 1 \right) i / ( T _ { \mathrm { s a m p l e d } } - 1 ) \} _ { i = 0 } ^ { T _ { \mathrm { s a m p l e d } } - 1 }$ and interpolate onto the integer grid $\{ 0 , 1 , \ldots , T _ { \mathrm { o r i g } } - 1 \}$ . We choose reward\_progress (the interpolated progress signal) as our final reward annotation because it is more aigned with actual task outcome.We substract the reward progress by -1 to make the reward fall in $[ - 1 , 0 ]$ as labels for training.

## A2.3Inference

Inference noise schedules.We evaluate several deterministic schedules to map discrete inference index $i \in \{ 0 , \ldots , K \}$ to the noise level $k \in [ 0 , 1 ]$ ,where K is the number of denoising steps.We describe the different noise schedules compared- linear, sigmoid, power and cosine:

$$
\mathbf { L i n e a r } \colon \ k = { \frac { i } { K } } ,
$$

$$
\mathbf { S i g m o i d } { \mathrm { : } } \quad k = \sigma \left( \alpha \left( { \frac { i } { K } } - 0 . 5 \right) \right) ,
$$

$$
\mathbf { P o w e r : } \quad k = \left( \frac { i } { K } \right) ^ { 0 . 5 } ,
$$

$$
\mathbf { C o s i n e } \colon k = 1 - \cos \left( \frac { i \pi } { 2 K } \right) ,
$$

where $\sigma ( \cdot )$ is the logistic sigmoid and α controls the sharpness. For the sigmoid schedule,we normalize endpoints to be $t _ { 0 } = 0$ and $t _ { K } = 1$ .The linear schedule allocates steps uniformly, cosine and power allocate more budget near low-noise regions, and sigmoid concentrates updates around the middle of the trajectory.

<table><tr><td>Name</td><td>Value</td></tr><tr><td>World Models Layers</td><td>32</td></tr><tr><td>Heads Embedding dimension Head dimension SPRINT probability</td><td>16 1536 96 0.5</td></tr><tr><td>Reward Model and Critic MLP layers</td><td>2</td></tr><tr><td>Discount factor  $\mathit { \Delta } \cdot \gamma$  Return lambda X</td><td>.995 .95</td></tr><tr><td>Pretraining Batch size Batch length</td><td>32 8</td></tr><tr><td>Memory frames (p) Memory frame stride (m) Optimizer Proprioceptive State loss scale  $( \mathcal { L } _ { q } )$  LR Warmup steps EMA decay</td><td>5 AdamW 0.1  $1 e ^ { - 4 }$  10000 .9999</td></tr><tr><td>Training Steps Finetuning</td><td>1000000</td></tr><tr><td>Batch size LR Warmup steps EMA decay Training Steps</td><td>32  $2 e ^ { - 5 }$  200 .9999 16000</td></tr></table>

Table 2: Hyperparameters. We present the list of hyperparameters used for training WEAVER.

Rectified-Flow. To further reduce inference time and NFE for downstream tasks like test-tme planning,we used ReFlow [28] to post-train WEAVER-FT model,and call it WEAVER-ReFlow. The teacher and student model are initialised with a WEAVER-FT model where we freeze the teacher model. At each training iteration, we sample noise $x ^ { 0 }$ and predict future latents with the teacher model ${ \hat { x } } ^ { 1 }$ . This student model is updated with the predicted latent as the target using mean squared error loss given by: $\mathcal { L } ^ { \mathrm { R e F 1 o w } } ( \phi ) = \mathbb { E } _ { x _ { t } ^ { 0 } , \hat { x } _ { t } ^ { 1 } , \tau } \left\lceil \left\| \left( \hat { x } _ { t } ^ { 1 } - x _ { t } ^ { 0 } \right) - f _ { \phi } ( \mathbf { z } _ { t } ^ { \mathrm { h i s t } } , \mathbf { z } _ { t } ^ { \mathrm { m e m } } , \mathbf { a } _ { t } , x _ { t } ^ { \tau } , \tau ) \right\| _ { 2 } ^ { 2 } \right\rceil$ . The post-training with rectified flow is performed for 2K gradient steps on 4 × H1OO (6 hours)with a learning rate of $2 e ^ { - 5 }$

## A3Additional World Model Evaluation Results

We provide additional results to evaluate WEAVER at coherent generations, impact of KV Cache on inference time,benefits of noise schedules, finetuning on task data, and post-training with rectified flow.

## A3.1World Model Evaluation

For a trajectory in validation dataset, we generated the rollout from the 2O-th step,and compute the metrics using the generations for next 1Os. We use the first 2O frames to initialize the memory and history for WEAVER and Ctrl-World. We report LPIPS [53],FID[20],and FVD [43]obtained using

the ground truth videos. To obtain LPIPS,we utilize the functionality in torchmetrics³ that uses the per-frame features obtained from vgg layers. To compute the FID,we use the implementation provided in pytorch-fid4 repository. Our results on FVD are computed using the $\mathsf { S t y l e { - } G A N { - } V } ^ { 5 }$ repository. Here,we subsample multiple trajectories of 16 frames with a stride of 8 from each trajectory.
<table><tr><td></td><td></td><td colspan="3">Exterior</td><td colspan="3">Wrist</td><td></td></tr><tr><td>Method</td><td>NFE</td><td>LPIPS↓</td><td>FID←</td><td>FVD↓</td><td>LPIPS↓</td><td>FID←</td><td>FVD↓</td><td>Time (s)↓</td></tr><tr><td colspan="9">DROID</td></tr><tr><td>Ctrl-World</td><td>8</td><td>0.169</td><td>31.63</td><td>116.14</td><td>0.407</td><td>52.40</td><td>347.69</td><td>8.14</td></tr><tr><td></td><td>16</td><td>0.165</td><td>26.09</td><td>78.73</td><td>0.392</td><td>33.83</td><td>195.37</td><td>14.65</td></tr><tr><td></td><td>32</td><td>0.168</td><td>23.63</td><td>63.55</td><td>0.389</td><td>27.14</td><td>114.87</td><td>27.67</td></tr><tr><td></td><td>50</td><td>0.168</td><td>22.44</td><td>55.05</td><td>0.388</td><td>25.32</td><td>91.77</td><td>42.33</td></tr><tr><td>WEAVER</td><td>8</td><td> 0.117</td><td>10.59</td><td>28.97</td><td>0.372</td><td>24.25</td><td>104.53</td><td> 2.53</td></tr><tr><td></td><td>16</td><td>0.117</td><td>10.20</td><td>27.83</td><td>0.371</td><td>21.50</td><td>90.72</td><td>4.78</td></tr><tr><td></td><td>32</td><td>0.120</td><td>9.67</td><td>25.94</td><td>0.378</td><td>17.53</td><td>63.36</td><td>9.22</td></tr><tr><td></td><td>50</td><td>0.122</td><td>9.51</td><td>26.54</td><td>0.378</td><td>16.75</td><td>66.89</td><td>14.25</td></tr><tr><td colspan="9">New dataset</td></tr><tr><td>Ctrl-World</td><td>8</td><td>0.193</td><td>48.90</td><td>226.29</td><td>0.374</td><td></td><td>51.26434.84</td><td>8.14</td></tr><tr><td></td><td>16</td><td>0.182</td><td>36.16</td><td>139.54</td><td>0.366</td><td>38.76</td><td>277.13</td><td>14.65</td></tr><tr><td></td><td>32</td><td>0.183</td><td>32.18</td><td>105.38</td><td>0.365</td><td>33.73</td><td>173.15</td><td>27.67</td></tr><tr><td></td><td>50</td><td>0.184</td><td>31.44</td><td>91.48</td><td>0.367</td><td>33.47</td><td>145.86</td><td>42.33</td></tr><tr><td>WEAVER</td><td>8</td><td>0.154</td><td>23.89</td><td>89.55</td><td>0.364</td><td>31.70</td><td>193.55</td><td> 2.53</td></tr><tr><td></td><td>16</td><td>0.155</td><td>23.95</td><td>88.27</td><td>0.364</td><td>30.77</td><td>184.62</td><td>4.78</td></tr><tr><td></td><td>32</td><td>0.157</td><td>23.45</td><td>92.36</td><td>0.365</td><td>28.24</td><td>148.85</td><td>9.22</td></tr><tr><td></td><td>50</td><td>0.159</td><td>23.48</td><td>87.03</td><td>0.371</td><td>27.37</td><td>145.04</td><td>14.25</td></tr></table>

Table 3: Comparison of WEAVER and Ctrl-World at LPIPS,FID and FVD metrics. WEAVER generates with higher fidelity than Ctrl-World and has significantly better performance at low inference budgets. Here,NFE is Number of Function Evaluations and inference time is the time required to generate a 10s segment on a single H100 GPU.

## A3.2Quantitative Results

Table 3 reports the comparison of WEAVER and Ctrl-World on DROID and OOD datasets across multiple metrics.We observe that performance of Ctrl-World deteriorates with lower NFE whereas WEAVER shows slight drop in performance with decrease in NFE.Moreover, with similar NFE values, our method is 3× faster at generating rollouts than Ctrl-World. In Fig.11,we present the comparison of FID and inference time and observe that WEAVER with lowe NFE of 8 outperforms Ctrl-World with large NFE of 5O.We also include qualitative results of different NFEs with different world models in Fig. 14 and Fig. 15.

![](images/81005ab428e2cd9ecf572550db4cffbdfd809a787b431c50777b95b530f761f0.jpg)

![](images/e8d7ab2b2c1a1881cb551ddd0c82405d339a8306989c328dfd1347f67adf9ace.jpg)

![](images/25494c820dd1670cb3a9f34d057db4d7af8c46df6c0d5affefd61c29b32b4b64.jpg)

![](images/a254416f218b26e47c7400e28fe6ef0e7240385dd0ab90f3ab2120dad23c5446.jpg)  
Figure 11: We compare FID vs inference time for WEAVER and Ctrl-World and find that WEAVER outperforms the baseline with upto 16 × more inference time.

<table><tr><td>/ NFE</td><td>8</td><td>16</td><td>32</td><td>50</td></tr><tr><td>Without KV cache </td><td>3.306</td><td>6.54</td><td>12.98</td><td>20.21</td></tr><tr><td>With KV cache</td><td>2.53</td><td>4.78</td><td>9.22</td><td>14.25</td></tr></table>

Table 4: We report the inference time (in seconds) taken to generate a 1Os trajectory on a single H100 GPU at different NFE,and observe that using KV Cache can reduce inference time by upto 30%.

<table><tr><td colspan="2"></td><td colspan="3">Exterior</td><td colspan="3">Wrist</td></tr><tr><td>Method</td><td>Schedule LPIPS←1</td><td></td><td>FID←</td><td>FVD↓1</td><td>LPIPS↓</td><td></td><td>FID↓FVD↓</td></tr><tr><td colspan="8">DROID</td></tr><tr><td>Ctrl-World WEAVER</td><td>linear linear sigmoid power cosine</td><td>0.165 0.117 0.117 0.117 0.117</td><td>26.09 11.32 10.88 10.57</td><td>78.73 26.38 29.76 27.93</td><td>0.392 0.375 0.375 0.369</td><td>24.43 22.69 21.17</td><td>33.83195.37 98.82 104.89 91.51</td></tr><tr><td colspan="8">10.20 27.83 0.371 21.50</td></tr><tr><td colspan="8">New dataset Ctrl-World linear 0.182 36.16 139.54 0.366</td></tr><tr><td>WEAVER</td><td>linear</td><td>0.157</td><td>25.37</td><td>96.16</td><td>0.367</td><td>33.24</td><td>38.76277.13 217.65</td></tr><tr><td></td><td>sigmoid</td><td>0.156</td><td>24.83</td><td>93.15</td><td>0.367</td><td>32.48</td><td>216.89</td></tr><tr><td></td><td>power</td><td>0.155</td><td>23.82</td><td> 84.91</td><td>0.363</td><td>31.60</td><td></td></tr><tr><td></td><td>cosine</td><td></td><td></td><td></td><td></td><td></td><td>185.30</td></tr><tr><td></td><td></td><td>0.155</td><td>23.95</td><td>88.27</td><td>0.364</td><td> 30.77</td><td>184.62</td></tr></table>

Table 5: We see that cosine and power noise schedules allocates higher budget at low noise scales and perform better than linear and sigmoid schedules. The numbers are reported with NFE=16.

## A3.3Reducing inference time with KV Cache

During the iterative denoising process, the latents of memory and history frames are passed with a constant noise level k = 1. Since it does not change during this process, we compute the cache of key-value vectors of memory and history latents at the first denoising step. In Table 4,we report that KV Cache can reduce inference time by up to 30%.

## A3.4Noise schedules during inference

Table 5 compares different noise schedules where we observe that both power and cosine noise schedules perform better than sigmoid and linear noise schedules. Since the world model needs to generate with higher fidelity, the noise schedules that allocate more bandwidth at low noise regions aids in generating fine-grained details.

<table><tr><td></td><td></td><td colspan="3">Exterior</td><td colspan="3">Wrist</td></tr><tr><td>Method</td><td>NFE1</td><td>LPIPS←1</td><td>FID↓</td><td>FVD↓</td><td>LPIPS↓</td><td>FID←1</td><td>FVD↓</td></tr><tr><td colspan="8">Task Data (00D)</td></tr><tr><td>Ctrl- World</td><td>50</td><td>0.184</td><td>31.44</td><td>91.48</td><td>0.367</td><td>33.47</td><td>145.86</td></tr><tr><td>Ctrl-World-FT</td><td>16</td><td>0.140</td><td>29.89</td><td>81.16</td><td>0.295</td><td>34.77</td><td>283.00</td></tr><tr><td></td><td>50</td><td>0.142</td><td>25.96</td><td>58.27</td><td>0.292</td><td>25.80</td><td>134.69</td></tr><tr><td>WEAVER</td><td>50</td><td>0.159</td><td>23.48</td><td>87.03</td><td>0.371</td><td>27.37</td><td>145.04</td></tr><tr><td>WEAVER-FT</td><td>4</td><td>0.116</td><td>16.62</td><td>50.68</td><td>0.304</td><td>33.32</td><td>219.15</td></tr><tr><td></td><td>16</td><td>0.118</td><td>14.24</td><td>40.50</td><td>0.303</td><td>23.39</td><td>146.35</td></tr><tr><td></td><td>50</td><td>0.121</td><td>13.69</td><td>40.09</td><td>0.308</td><td>18.73</td><td>99.81</td></tr><tr><td>WEAVER-REFLOW</td><td>4</td><td>0.123</td><td>14.95</td><td>44.30</td><td>0.312</td><td>23.56</td><td>138.67</td></tr></table>

Table 6: Compare the finetuned variants of WEAVER and Ctrl-World on Task dataset (OOD) where WEAVER-FT outperforms the baselines.Moreover, the post-training step (WEAVER-REFLOW) further helps to reduce inference budget.

Table 7: Inference time breakdown for test-time planning on A60oO Ada GPU. We report the runtime of each component with different horizons of world model imaginations across 1O function calls.
<table><tr><td>Component</td><td>Notation</td><td></td><td>Batch Size</td><td>Horizon</td><td>Runtime (s) ↓</td></tr><tr><td>Policy sampling</td><td> $\pi _ { \boldsymbol { \theta } } \big ( a _ { t : t + h } \mid o _ { t } , \ell \big )$ </td><td></td><td></td><td>1</td><td> $0 . 1 9 7 9 _ { \pm 0 . 0 0 0 2 }$ </td></tr><tr><td rowspan="4">Dynamics model (WEAVER)</td><td rowspan="4"> $\begin{array} { r } { \overline { { f _ { \phi } \big ( \hat { z } _ { t + 1 : t + h } ~ | ~ { \mathbf { z } } _ { t } ^ { \mathrm { m e m } } , { \mathbf { z } } _ { t } ^ { \mathrm { h i s t } } , a _ { t : t + h } \big ) } } } \end{array}$ </td><td rowspan="4"></td><td>4</td><td>9</td><td> $1 . 0 2 0 3 { \scriptstyle \pm 0 . 0 1 0 0 }$ </td></tr><tr><td>4</td><td>12</td><td> $1 . 2 4 9 3 { \scriptstyle \pm 0 . 0 1 7 0 }$ </td></tr><tr><td>4</td><td>15</td><td> $1 . 4 5 4 7 _ { \pm 0 . 0 1 6 0 }$ </td></tr><tr><td>1</td><td>15</td><td> $0 . 4 4 7 6 { \scriptstyle \pm 0 . 0 0 4 9 }$ </td></tr><tr><td rowspan="2">Dynamics Model (Ctrl-World)</td><td rowspan="2"> $\overline { { f _ { \phi } ( \hat { z } _ { t + 1 : t + h } \mid \mathbf { z } _ { t } ^ { \mathrm { m e m } } , \mathbf { z } _ { t } ^ { \mathrm { h i s t } } , a _ { t : t + h } ) } }$ </td><td rowspan="2"></td><td>4</td><td>15</td><td> $\overline { { 2 9 . 4 2 4 4 _ { \pm 1 . 0 1 6 2 } } }$ </td></tr><tr><td>1</td><td>15</td><td> $7 . 4 2 3 6 _ { \pm 0 . 1 2 0 1 }$ </td></tr><tr><td>Reward inference</td><td></td><td> $\overline { { R ( \hat { z } _ { t + 1 : t + h } , \ell ) } }$ </td><td>4</td><td>二</td><td> $0 . 0 0 0 6 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td></tr><tr><td>Critic inference</td><td></td><td> $\overline { { V ( \hat { z } _ { t + h } , \ell ) } }$ </td><td>4</td><td>1</td><td> $\overline { { 0 . 0 0 0 5 _ { \pm 0 . 0 0 0 1 } } }$ </td></tr></table>

## A3.5Finetuning

In Table 6,we observe that finetuning significantly improves performance and with NFE=16 it performs better than WEAVER with NFE=5O. To provide a fair comparison with Ctrl-World,we finetune the baseline for 2OK gradient steps on 4 × H1OO and see that the finetuned Ctrl-World (called Ctrl-World-FT) performs better than the pretrained model.However, WEAVER-FT ourperforms Ctrl-World-FT across metrics and the performance is still larger with low NFE=16. This further demonstrates that finetuning does not help in reducing inference time for Ctrl-World.We also provide qualitative results of the rollouts generated from Ctrl-World, WEAVER, WEAVER-FT in Fig.16,Fig.17 and Fig. 18.

## A3.6Posttraining with Rectified Flow

In table 6,we present the results of WEAVER-ReFlow with small inference budget and observe that it reduces the performance gap with WEAVER-FT evaluated with a large NFE=16.This makes it suitable for test-time steering as observed in Section 3.2.

## A4Additional Downstream Application Results

## A4.1Policy Evaluation Results

We provide the full policy evaluation rollouts in Fig.12.Both Ctrl-World and WEAVER struggle to accurately predict policy performance,especially on challenging tasks involving dy-namic manipulation, such as pouring beans,and deformable object manipulation, such as bag and towel manipulation. For the PnP Bag task, grasp-ing the bag is particularly challenging because the world model must accurately infer the gripper depth across two camera views while also modeling the contact dynamics between the gripper and the deformable object. These challenges become more pronounced as the prediction horizon increases.In contrast, WEAVER-FT substantially improves evaluation accuracy through finetuning as shown in Fig.19 and Fig.2O. Future work could further improve long-horizon prediction by designing better memory and

![](images/4943a1c178526f1703bff4888a7f6cd8379aa629b24020fa80edd7df6628dfa7.jpg)  
Figure 12: Policy Evaluation Results. We show policy evaluation results for all five tasks across three world models.

Table 8: Comparison of reward prediction quality across different world models. We report RMSE, Spearman correlation, Pearson correlation,and MMRV.
<table><tr><td>Method</td><td>RMSE↓</td><td>Spearman 个</td><td>Pearson ↑</td><td>MMRV↓</td></tr><tr><td>CtrlWorld</td><td>0.410</td><td>0.523</td><td>0.552</td><td>0.215</td></tr><tr><td>Ours</td><td>0.359</td><td>0.594</td><td>0.563</td><td>0.155</td></tr><tr><td>Ours-FT</td><td>0.188</td><td>0.870</td><td>0.863</td><td>0.035</td></tr></table>

history representations,enabling the model to better reason about occlusions and deformable object dynamics.

In addition to Pearson correlation and MMRV, we also report RMSE and Spearman rank correlation [48]. Across these metrics,we observe a consistent trend: WEAVER-FT achieves the strongest correlation and lowest prediction error. In addition, WEAVER outperforms Ctrl-World in zero-shot policy evaluation on out-of-distribution task dataset. The full quantitative results are shown in Table 8.

## A4.2Policy Improvement Results

We provide additional qualitative results of policy improvement in Fig.13. These examples show that the base policy often suffers from imprecise grasping and placement, as well as insufficient adjustment during dynamic manipulation. We also observe that the base policy tends to produce larger per-step motions,resulting in unstable robot control. In contrast, the finetuned policy substantially reduces these large movements and sharpens the action distribution, leading to smoother and more stable execution.

We also note that the RoboMeter reward labels are not perfect. For the PnP Marker task, we observe cases where the reward model fails to distinguish fine-grained placement accuracy, which can introduce noise into the predicted rewards. Future work could improve reward supervision by collecting more diverse failure data to train a more general and precise reward model. To mitigate the effect of noisy reward labels, we set the advantage threshold to O.1, which helps prevent low-quality segments from being selected for finetuning and potentially degrading policy performance. As shown in Fig.4,our filtering procedure is able to select the best action samples among the candidates.

## A4.3Test-Time Planning Results

Inference-Time Latency. Table 7 reports the inference-time breakdown of test-time planning on an A6OoO Ada GPU. Overall, the runtime is dominated by the dynamics model imagination, while reward and critic inference are negligible,taking less than O.OOis each.For WEAVER,the dynamics runtime increases moderately with the imagination horizon: from 1.02O3s at horizon 9,to 1.2493s at horizon 12,and 1.4547s at horizon 15 with batch size 4. Including policy sampling,reward inference, and critic inference, the full planning latency is approximately 1.22s,1.45s,and 1.65s for horizons 9, 12, and 15, respectively.

Compared to Ctrl-World, WEAVER substantially reduces latency during imagination. At horizon 15 and batch size 4, WEAVER takes 1.4547s for dynamics prediction, while Ctrl-World requires 29.4244s, corresponding to a 2O.2× speedup. The same trend holds at batch size 1,where WEAVER takes O.4476s compared to 7.4236s for Ctrl-World, yielding a 16.6× speedup. These results show that WEAVER enables substantially lower-latency test-time planning, making repeated world-model imagination practical during policy execution.

## A5Limitations

While WEAVER demonstrates the promise of large-scale world models for policy evaluation, policy improvement,and test-time planning,several limitations remain.

![](images/9af43b6c7113bc412f4202144821a74e98281caf40a47c96e5c01706dc8009a3.jpg)  
Figure 13:Policy Improvement Results. We demonstrate the rollouts for five tasks among the base policy and policy FT w/ Synthetic Data. With WEAVER generated synthetic data, policy finetuning can improvement upon all tasks.

## A5.1Partial Observability

Our world model relies primarily on visual observations,which provide only partial access to the underlying physical state.During manipulation,task-relevant information such as object contacts, grasp stability, applied forces, or occluded object geometry may be hidden from all available camera views.This limitation is especially pronounced for wrist-camera observations,where the viewpoint changes continuously,and for cluttered scenes where objects may leave the field of view or become occluded by the gripper. Although memory and multi-view conditioning mitigate this issue,purely visual prediction may still fail when the missing state cannot be inferred from image history alone. Incorporating additional sensing modalities, such as tactile feedback, force-torque sensing, or depth, may improve state estimation and long-horizon prediction under occlusion.

## A5.2Complex Deformable and Dynamic Interactions

Deformable-object manipulation and dynamic manipulation remain challenging for learned world models. Objects such as towels, bags,and granular materials exhibit high-dimensional, historydependent dynamics that are difficult to capture from limited robot data. Small errors in predicted contact, grasp location, or object configuration can compound over time and lead to qualitatively incorrect rollouts. This is particularly evident in tasks such as pouring, where the motion of granular material depends sensitively on cup pose, velocity, and contact with the container. Future work may improve prediction fidelity by incorporating physics priors,hybrid neural-physics models,or neural simulators specialized for deformable and granular dynamics.

## A5.3Limited Planning Horizon at Test Time

Although our inference acceleration strategies make test-time planning feasible with a large generative world model, latency still limits online planning to a single action chunk.As a result, the planner can improve near-term action selection but cannot yet perform long-horizon lookahead. This restricts its ability to reason about delayed consequences or multi-stage recovery behaviors. Further improvements in sampling eficiency, model distillation, value estimation, or hierarchical planning could enable longer-horizon online reasoning while maintaining real-time control.

## A5.4Data Coverage and Embodiment Diversity

Our world model is pretrained primarily on DROID,which provides large-scale robot interaction data but is stil tied to a specific robot embodiment and data collection setup.This may limit generalization to substantially different robots,camera configurations, end-effectors.In addition,some task dynamics in our evaluation, such as granular pouring, are underrepresented in the pretraining data. Scaling world-model training to more diverse sources,including cross-embodiment robot datasets, simulation data,and human videos,may improve robustness and broaden the range of behaviors that can be accurately imagined.

## A5.5Noisy Reward Supervision

Our latent reward and critic heads are trained using labels from an off-the-shelf reward model. While this enables efcient latent-space evaluation, the resulting supervision can be noisy or incomplete, especially for subtle failure modes. For example,a reward model may fail to distinguish between visually similar but semantically different outcomes,or may be insensitive to small errors in contact, placement, or task completion. Such noise can affect both policy evaluation and downstream policy improvement. A more reliable reward model trained on large-scale robot success and failure data, potentially with calibrated uncertainty,would likely improve the reliability of imagined rollout evaluation.

Overall, these limitations suggest that future progress will require not only larger and faster world models,but also richer sensing, broader data coverage,stronger physical inductive biases,and more accurate reward supervision.

![](images/75ea5125c05e004a63adc042dbf966ccb6235e4251b192ba45d53e435e2ef0bb.jpg)  
Figure 14:We compare the rollouts on task obtained from Ctrl-World,WEAVER,WEAVER-FT and WEAVER-REFLOW at different NFE values of 4 and 16.We generate rollouts for 2O seconds and present predicted camera views at every 4 second. We observe that WEAVER-REFLOW is better than WEAVER-FT at NFE=4 and has comparable performance with other models using NFE=16.

![](images/acc5ebc72e027bd81ed0014522b6650f97e540c73ec388a14bef250f0006258b.jpg)  
Figure 15:We compare the rollouts on task obtained from Ctrl-World,WEAVER,WEAVER-FT and WEAVER-REFLOW at different NFE values of 4 and 16.We generate rollouts for 2O seconds and present predicted camera views at every 4 second.We observe that WEAVER-REFLOW is better than WEAVER-FT at NFE=4 and is more consistent than Ctrl-World and WEAVER with NFE=16.

![](images/13498d9f86d1d19bfc96033d613f0c98a135c0893cc13276b21311cc7119568d.jpg)  
Figure 16: We compare the rollouts on task obtained from Ctrl-World, WEAVER and WEAVER-FT and WEAVER-REFLOW at different NFE=50. We generate rollouts for 2O seconds and present predicted camera views at every 4 second. We observe that Ctrl-World struggles at retraining information about the towel after 12 seconds and WEAVER-FT is more consistent with the ground truth.

![](images/df0647072a44735a9517ff7131134927438309749407ca51d49b32f35e650577.jpg)  
Figure 17: We compare the rollouts on task obtained from Ctrl-World, WEAVER and WEAVER-FT and WEAVER-REFLOW at different NFE=50.We generate rollouts for 2O seconds and present predicted camera views at every 4 second.We observe that WEAVER and WEAVER-FT is better than Ctrl-World at predictions.

![](images/c8af95bc938b384f5f6a7e8f5b129563e0fbadb332746c2d3ba6378f9bcefd26.jpg)  
Figure 18:We compare the rollouts on task obtained from Ctrl-World, WEAVER and WEAVER-FT and WEAVER-REFLOW at different NFE=50. We generate rollouts for 2O seconds and present predicted camera views at every 4 second. We see that Ctrl-World struggles to predict the object from t=4s compared to WEAVER and WEAVER-FT.

![](images/fdc5801662ecd7ad6b7b0fc8446f001a79329e630e4547ce732038a5e3d1e660.jpg)  
Figure 19:We compare the rollouts on Pour Beans task obtained from Ctrl-World, WEAVER and WEAVER-FT for policy evaluation at NFE=5O.We generate rollouts for 15 seconds and present predicted camera views at every 3 second. We see that Ctrl-World and WEAVER struggles to predict the beans in the bowl at t=12s compared to WEAVER-FT.

![](images/7ef51d3f7277a7ee2c601aa9c6109eb2c4279b43d3e24076d7dd8a7c8bdc0992.jpg)  
Figure 2O:We compare the rollouts on Pour Beans task obtained from Ctrl-World, WEAVER and WEAVER-FT for policy evaluation at NFE=5O.We generate rollouts for 15 seconds and present predicted camera views at every 3 second. We see that Ctrl-World and WEAVER struggles to predict the beans on the table at t=15s compared to WEAVER-FT.