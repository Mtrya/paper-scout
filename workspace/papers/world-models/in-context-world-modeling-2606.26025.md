# In-Context World Modeling for Robotic Control

Siyin Wang<sup>1,2</sup> Junhao Shi<sup>1,2,</sup> Senyu Fei<sup>2,3</sup> Zhaoyang Fu<sup>1</sup> Li Ji<sup>1,2</sup> Jingjing Gong<sup>2,</sup>† Xipeng Qiu<sup>1,2,</sup>†

<sup>1</sup>Fudan University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Tongji University siyinwang20@fudan.edu.cn

## Abstract

Modern Vision-Language-Action (VLA) models often fail to generalize to novel setups, such as altered camera viewpoints or robot morphologies, because they are typically conditioned only on current observations and language instructions. By ignoring the underlying system configuration as a variable, these models implicitly assume a fixed execution context encountered during training, necessitating data-intensive fine-tuning for any new environment. In this work, we introduce In-Context World Modeling (ICWM), a framework that treats system identification as an in-context adaptation problem. ICWM enables robot policies to autonomously infer essential system variables from a short history of self-generated, task-agnostic interactions. Unlike traditional In-Context Learning that uses demonstrations to specify what task to perform, ICWM leverages the context window to understand how the system operates. By processing these interactions before task execution, the model implicitly captures the world dynamics of the current system, enabling adaptation to novel configurations without parameter updates. Extensive experiments in simulation and on real-world robot platforms demonstrate that ICWM significantly outperforms standard VLA baselines on novel camera viewpoints.

## 1 Introduction

Consider a human operator handed a joystick to control a robot, but with no prior knowledge of the control mapping. Does pushing forward move the robot left, right, or forward? The operator’s first instinct is not to attempt the task directly, but to explore: move the joystick randomly, observe the robot’s response, and infer the input-output relationship. Within seconds, this brief calibration phase yields an internal model of the system dynamics, enabling the transition from stochastic exploration to goal-directed control. Crucially, this calibration is entirely self-generated—the operator needs no prior knowledge of the task, no external guidance, and no task-specific experience under the new control mapping.

Modern Vision-Language-Action (VLA) models [1–3], however, lack this calibration ability entirely. The standard formulation $\pi _ { \theta } \left( a _ { t } \mid o _ { t } , l \right)$ conditions only on the current observation and language instruction, treating system configuration ψ (like camera viewpoints, morphology) as a fixed constant absorbed into model parameters during training. When deployment conditions deviate from training, the model has no mechanism to recover the correct action-observation correspondence, and performance degrades [4]. Scenespecific fine-tuning remains the prevailing remedy, yet it requires human intervention for every new setup contradicting the goal of generalist deployment.

![](images/ad3256eae6f52ddb486625e6ebd105f632a57029262bf8ff6d7cafea82e66f0e.jpg)  
Figure 1 In-Context World Modeling (ICWM). Standard VLA models often fail in novel system configurations due to fixed observation-action assumptions. Similar to how humans explore unfamiliar controls to build a mental world model, ICWM enables robots to autonomously infer system dynamics from self-probing interaction context.

We argue this failure is a system identification problem: the policy lacks knowledge of ψ at test time. This naturally motivates repurposing the context window to recover ψ through targeted interaction. However, existing In-Context Learning (ICL) methods [5, 6] treat context as behavior specification (what to do) rather than system identification (how the system operates), and still require human-provided demonstrations at test time.

In this work, we propose In-Context World Modeling (ICWM) (Fig. 1), which repurposes the context window for system identification rather than behavior specification. Before task execution, the robot performs a short sequence of random exploratory movements and records the resulting visual transitions; these self-probed clips are prepended as context, from which the model implicitly recovers the current system configuration and adapts its actions accordingly. Surprisingly, the task-agnostic random movements alone provide suficient context for implicit system identification without any task-specific demonstrations and we show this simple strategy yields consistent improvements over standard VLA training on novel viewpoints across both simulation and real-robot platforms.

In summary, this work makes the following contributions:

• We reframe VLA generalization as a test-time system identification problem, identifying the absence of explicit ψ conditioning as an underexplored failure mode that persists across standard training strate gies.

• We propose In-Context World Modeling (ICWM), which implicitly identifies the system configuration from self-generated, task-agnostic exploratory transitions prepended as context, achieving test-time adaptation without parameter updates or task-specific demonstrations.

• We validate ICWM on both simulation benchmarks and real-robot experiments, demonstrating significant improvements over standard VLA training on novel camera viewpoints, with further generaliza tion to semantic scene variations and robot morphological changes.

## 2 Related Work

## 2.1 In-Context Adaptation for Robotics

The emergence of large-scale pre-trained models [7–9] has established In-Context Learning (ICL) as an emergent capability for on-the-fly skill acquisition [10, 11]. In robotics, ICL has been explored via in-context imitation: models are prompted with expert trajectories, provided by teleoperation or retrieved from ofline bufers, to perform next-token prediction over observation-action sequences [5, 12, 13]; complementary approaches leverage human play videos for cross-embodiment transfer [6, 14]. Despite diferences in data modality, all such methods treat context as behavior specification (“what to $\mathrm { d o ^ { \prime \prime } } )$ , and critically require human-provided demonstrations at test time. A related line of work pursues adaptation through metalearning. Gradient-based methods [15, 16] meta-optimize an initialization for rapid test-time fine-tuning, while recurrent meta-RL [17, 18] encodes interaction history to infer which task to perform—relying on reward signals unavailable prior to task execution. In contrast, ICWM uses task-agnostic self-generated interactions to understand how the system operates, requiring no demonstrations, no reward signal, and no parameter updates at test time–a harder and more practical deployment setting.

## 2.2 World Modeling for Robotic Control

World modeling captures environment dynamics to ground an agent’s actions in causal state transitions [19– 23]. Existing approaches fall into two paradigms: forward models predict future observations either in pixel space [24–27] or latent space [28–32], while inverse dynamics models abduce actions from observed visual changes [33–36]; some works unify both objectives [37, 38]. All of these introduce dedicated parameters and training objectives for world modeling. ICWM instead realizes world modeling implicitly: the model leverages standard sequence modeling to extract the time-invariant causal structure $( \mathrm { e . g . }$ , control mappings and camera viewpoints) directly from task-agnostic interaction histories. This implicit paradigm introduces zero additional parameters, treating world modeling as an emergent inference capability at test time.

## 3 Preliminary and Motivation

## 3.1 Standard VLA Formulation and Its Limitation

We consider a robot manipulation task where a policy π maps multimodal observations and language instructions to actions. Let O be the observation space and I be the space of natural language instructions. A standard VLA policy $\pi _ { \boldsymbol { \theta } } \left( a _ { t } \mid o _ { t } , l \right)$ processes the current observation $o _ { t } \in \mathcal { O }$ and instruction $l \in \mathcal { T }$ to predict an action $a _ { t } \in A .$ The control task is thus formulated as:

$$
\pi _ { \boldsymbol { \theta } } \left( a _ { t } \mid o _ { t } , l \right) .\tag{1}
$$

where the model parameters θ are optimized on a large-scale dataset D collected under a set of specific system setups. In practice, $a _ { t }$ is often structured as an action chunk to ensure temporal smoothness and control stability during execution.

This formulation embeds an implicit assumption: the system configuration ψ—encompassing camera viewpoints, mounting ofsets, and the robot’s morphological properties—is fixed and known. An ideal policy should instead condition on ψ explicitly:

$$
\pi _ { \theta } ^ { * } \left( a _ { t } \mid o _ { t } , l , \psi \right) .\tag{2}
$$

Without $\psi ,$ training forces the model to marginalize over all configurations in the dataset:

$$
\pi _ { \boldsymbol { \theta } } \left( a _ { t } \mid o _ { t } , l \right) \approx \pi _ { \boldsymbol { \theta } } ^ { * } \left( a _ { t } \mid o _ { t } , l , \boldsymbol { \psi } \right) p \left( \boldsymbol { \psi } \right) d \boldsymbol { \psi } .\tag{3}
$$

At deployment, when a specific $\psi ^ { \prime }$ is realized, this averaged policy lacks the context to correctly interpret the observation-action correspondence, leading to degraded performance on novel configurations. This motivates explicit recovery of ψ at test time

## 3.2 Interaction Context Enriches Configuration Information

We model robot-environment interaction as a POMDP where the latent state decomposes as $s _ { k } = \psi , \xi _ { k }$ , with ψ being the time-invariant system configuration and $\xi _ { k }$ the time-varying scene state. The system evolves as:

$$
s _ { 0 } \ { \xrightarrow { { a _ { 1 } } } } \ s _ { 1 } \ { \xrightarrow { { a _ { 2 } } } } \ \cdot \ \cdot \ { \xrightarrow { { a _ { t } } } } \ s _ { t } , \qquad { o _ { k } } \sim p \left( o \mid s _ { k } \right) .\tag{4}
$$

![](images/243b324d1973e21f4beb2dbb8eff6f76a7347d3894a32401ddbba35b387ab2f5.jpg)  
Figure 2 Overview of the ICWM Training and Inference Pipeline. (1) Training: The model is trained on data collected across diverse system configurations, where task-agnostic interaction clips are prepended to each training sample as context. (2) Inference: At test time, the robot first performs stochastic exploration to collect system context, which then guides the policy to generate precise actions via in-context inference.

We define the interaction context as $\mathcal { T } = \left( o _ { 0 : t } , a _ { 1 : t } \right)$ and analyze its information content under two mild assumptions: (A1) partial observability, $H \left( \boldsymbol { s } _ { k } \mid \boldsymbol { o } _ { k } \right) > 0$ , a single image cannot uniquely identify the viewpoint or kinematics; and (A2) information-preserving transitions, $I \left( s _ { 0 } ; s _ { k } \mid a _ { 1 : k } \right) > 0 ,$ , state transitions preserve ψ, which holds since ψ is time-invariant.

Proposition 1. Under A1 and A2, for any action sequence $a _ { 1 : t } ,$ the interaction context T carries strictly more information about ψ than any single observation:

$$
I \left( \psi ; o _ { 0 : t } , a _ { 1 : t } \right) > I \left( \psi ; o _ { 0 } \right) .\tag{5}
$$

The proof is provided in $\operatorname { A p p . } \mathrm { A . }$ Since the result holds for any action distribution, task-agnostic random movements can also enrich the information available about ψ despite carrying no task-specific information.

## 4 In-Context World Modeling

To resolve the information incompleteness identified in Sec. 3, we introduce In-Context World Modeling (ICWM) (Fig. 2). The core philosophy of ICWM is to transform the VLA from a static mapping into an adaptive inference mechanism that recovers the latent configuration ψ through environmental interactions.

Specifically, given a short history of task-agnostic interaction clips $\mathcal { T } = \{ \left( o _ { i } ^ { s } , a _ { i } , o _ { i } ^ { e } \right) \} _ { i = 1 } ^ { N } ,$ , the configuration can be inferred implicitly with a function Ψ <sup></sup>T <sup></sup>. By conditioning the policy on interactions collected under the current configuration ψ, the policy is reformulated as:

$$
a _ { t } \sim \pi _ { \theta } \left( a _ { t } \mid \Psi \left( \mathcal { T } \right) , o _ { t } , l \right) .\tag{6}
$$

The $\Psi \left( \mathcal { T } \right)$ term serves as a representation of the world dynamics under $\psi ,$ enabling the model to interpret $o _ { t }$ in the context. This interaction-centric conditioning allows the policy to adapt its action selection to the specific visual and physical setup encountered at test time, thereby overcoming the generalization collapse of standard VLA models.

## 4.1 In-Context Training

To implement $\Psi \left( \mathcal { T } \right)$ , we adopt a parameter-eficient design where Ψ shares its parameters with the VLA backbone $\pi _ { \theta } ,$ , motivated by the structural symmetry between action prediction and configuration inference both require understanding the correspondence between observations and actions.

Each training sample is constructed by prepending N task-agnostic interaction clips to the task query. These clips are randomly sampled from a pool of interaction segments collected across all training trajectories and viewpoints, ensuring diversity in the interaction context. The model is trained on data collected across diverse system configurations, where the interaction context T naturally varies with the underlying ψ. This variation provides an implicit training signal: to accurately predict task actions across configurations, the model must learn to extract and utilize the dynamics information carried in T , implicitly modeling the actionto-observation mapping under each specific system setup. The model is trained with the following loss:

$$
\mathcal { L } = - \log \pi _ { \boldsymbol { \theta } } \left( a _ { t } \mid \Psi \left( \mathcal { T } \right) , o _ { t } , l \right) ,\tag{7}
$$

where $\Psi \left( \mathcal { T } \right)$ denotes the hidden states induced by the interaction context.

## 4.2 Test-Time Active Probing and Inference

At deployment, ICWM enables task-specific demonstration-free adaptation to novel system configurations through a two-phase inference protocol that requires no gradient updates, prior calibration, or knowledge of the target environment.

Active Probing Phase. Before task execution, the robot collects the interaction context $\mathcal { T } = \{ \left( o _ { i } ^ { s } , a _ { i } , o _ { i } ^ { e } \right) \} _ { i = 1 } ^ { N }$ by performing N task-agnostic probing actions. For each step $i \in \{ 1 , \ldots , N \}$ , a random target pose is sampled within the robot’s safe workspace, and the robot executes an action $a _ { i }$ to reach it. The resulting transition $\left( o _ { i } ^ { s } , a _ { i } , o _ { i } ^ { e } \right)$ is recorded. These probing movements are designed to be spatially diverse, sampling multiple directions relative to the end-efector’s pose to provide suficient coverage of the local dynamics manifold. The probing workspace is defined to avoid contact with task-relevant objects, ensuring that the task initial state remains undisturbed throughout this phase.

In-Context Execution Phase. Once $\tau$ is collected, the configuration inference function Ψ processes the interaction context to implicitly recover the latent system configuration. Conditioned on the inferred hidden representation Ψ <sup></sup>T <sup></sup>, the current observation $o _ { t } ,$ and the language instruction l, the policy generates the task action $a _ { t } \sim \pi _ { \theta } \left( a _ { t } \mid \boldsymbol { o } _ { t } , \boldsymbol { l } , \boldsymbol { \Psi } \left( \mathcal { T } \right) \right)$ . Since Ψ shares parameters with the VLA backbone, this is implemented as a single forward pass where the Transformer first attends to T , building configuration-aware hidden states, before processing the task query $\left( o _ { t } , l \right)$ to produce actions aligned with the deployed physical setup.

## 5 Experiments

In this section, we evaluate the efectiveness of In-Context World Modeling (ICWM) in enabling visionlanguage-action models to adapt to novel system configurations without parameter updates. We focus on answering three questions:

(1) Generalization: Does ICWM improve generalization to unseen viewpoints?

(2) System Identification: Does the model truly perform implicit world modeling rather than pattern matching?

(3) Versatility: Does ICWM generalize beyond geometric shifts to semantic and morphological perturbations?

## 5.1 Experimental Setup

Simulated Benchmark. We evaluate on LIBERO [39], comprising four task suites (Spatial, Object, Goal, Long) that assess spatial reasoning, object understanding, goal conditioning, and long-horizon execution. We adopt a cross-view protocol: training on 8 azimuthal angles $( \psi \in \{ 3 0 ^ { \circ } , \bar { 6 } 0 ^ { \circ } , \dots , 3 \bar { 3 } 0 ^ { \circ } \} )$ and evaluating on both in-domain and 6 unseen OOD viewpoints $( \psi ^ { \prime } \in \{ 4 5 ^ { \circ } , 1 3 5 ^ { \circ } , \dots , 3 1 5 ^ { \circ } \} )$ , yielding $5 0 0 \times 1 5 \times 4$ total episodes. Details are in App. B.

Real-Robot Setup. Our platform is a UR5e manipulator with a 12-camera multi-view system, split into 6 training and 6 held-out test viewpoints. We evaluate four manipulation tasks (stacking, lifting, pick-andplace) with 25 trials per task per novel viewpoint (600 total), reporting the average success rate. Details are in $\operatorname { A p p . C . }$

![](images/3fff608aed6111f4a3d017cac480821f509b9a139320e72f86cd9a944065735e.jpg)  
Figure 3 Real-World Task Suite with four manipulation tasks, including stacking, lifting, and pick-and-place, evaluated across six distinct camera viewpoints.

![](images/3962957749a5a16e2b82bfcfef74398d12702aac41646d29e4b7964e57fc6f72.jpg)  
Figure 4 Success Rates (%) on LIBERO for Seen (In-Domain) and Unseen (OOD) Viewpoints.

Baselines. We evaluate two categories of models: (i) Controlled Comparisons: (a) Multi-View BC (MV) shares identical architecture and multi-view data with ICWM but lacks context; (b) Explicit Configuration (EXP) augments MV with ground-truth camera angles as text inputs. This category isolates the exact contribution of our interaction context. (ii) Contextual References: Pretrained models, including NORA [40],π- FAST [41], and $\pi _ { 0 . 5 }$ [42], are fine-tuned on a single viewpoint. They serve not as direct benchmarks, but to demonstrate that out-of-box generalization remains an unsolved challenge even under large-scale pretrain ing.

Implementation Details. Our model utilizes Qwen2.5-VL-3B [43] as the backbone and FAST [41] as action tokenizer, with action chunk size 5 and N=5 context clips. Implemented in PyTorch and trained on 8 NVIDIA A100 GPUs, the system is optimized via AdamW with a weight decay of $1 0 ^ { \dot { - } 4 }$ and a learning rate peaking at $5 \times 1 0 ^ { - 5 }$ after a 50k-step warmup, followed by cosine decay. The implementation details of self-exploration is detailed in App. D.

## 5.2 Simulation Results

As shown in Fig. 4, ICWM consistently outperforms all baselines in both seen and unseen viewpoints, leading to several key observations: (1) Viewpoint generalization is a fundamental challenge. While a seento-unseen performance drop is observed globally, ICWM demonstrates significantly stronger resilience, im proving the OOD success rate by 13.0% over the Multi-View BC baseline. This confirms that while multi-view training expands spatial data, geometric extrapolation remains an open challenge that standard imitation learning struggles to resolve without test-time adaptation. (2) Implicit identification outperforms explicit specification. ICWM improves OOD success rate by 9.5% over Explicit Configuration. Even with ground truth camera angles, Exp lacks OOD generalization and an understanding of system dynamics, strengths that ICWM gains from interaction context. (3) ICWM yields the greatest gains on long-horizon tasks. On LIBERO-Long, ICWM surpasses MV by 29.9% (seen) and 26.3% (unseen), the largest relative margins across all suites. This is because long-horizon tasks amplify small spatial errors from viewpoint shift, caus ing cascading failures in baselines, while ICWM mitigates this by continuously grounding actions in system dynamics.

![](images/cb548c38a7cd16616de3887137f11bbee1faeece5daaf6598994f39251de03c4.jpg)  
Figure 6 Qualitative Comparison. Without ICWM, standard policies exhibit (a) position ofsets or (b) premature gripper closure due to viewpoint shifts.

## 5.3 Real-world Results

As shown in Fig. 5, standard VLA performance drops sharply from 68% to 17% upon viewpoint shift, confirming that standard mix-training alone cannot bridge the gap to novel configurations at test time. In contrast, by introducing In-Context World Modeling (ICWM), the policy efectively mitigates this degradation without any parameter updates or task-specific demonstrations. As qualitatively shown in Fig. 6, while the base VLA exhibits end-efector drift and repeated grasp failures due to spatial misalignment, our ICWM enhanced robot utilizes the context window as a dy-

![](images/165e64ac0d217604f5d88126a6c38c751bc7aed54650db31ed2f4a87f4bc85e8.jpg)  
Figure 5 Real-world evaluation on the UR5e platform. ICWM substantially outperforms the Multi-View (MV) baseline under novel viewpoints.

namic calibration frame, achieving precise manipulation by grounding its actions in the current environment’s specific geometry.

## 6 Analysis

## 6.1 What Does the Interaction Context Contribute?

Ablation on context components. We ablate interaction context across five settings: (1) full-context (ICWM), (2) w/o actions (action tokens omitted), (3) w/o images (image tokens omitted), (4) w/o context (all interaction tokens excluded), and (5) false context (clips from a 180◦-ofset viewpoint). The asymmetric drops in Tab. 1 reveal several key findings: w/o images causes the largest collapse (avg. −56.4%)—without visual outcomes, the model mimics exploratory actions as task demonstrations, actively hurting performance; w/o actions degrades more moderately, confirming that visual flow provides a coarse spatial anchor but full calibration requires the paired $o _ { i } ^ { s } , a _ { i } , o _ { i } ^ { e }$ tuple. Critically, false context performs worse than no context at all (18.9 vs. 22.0), indicating that misaligned context actively misleads the policy’s world model rather than being passively ignored. This negative transfer, symmetric in magnitude to the gains from correct context (13.6%), confirms that the model genuinely conditions on context content for configuration inference.

Is in-context capability an emergent property of specialized training? To determine whether in-context adaptation arises from standard sequence modeling, we evaluate a BC policy trained without in-context supervision under the same interaction context. Performance collapses to near-zero (< 1%) when interaction tokens are prepended, confirming that this capability must be explicitly incentivized during training rather than emerging naturally from imitation learning.

Table 1 Ablation on Interaction Context. Removing any context component degrades performance, confirming that joint observation of actions and outcomes is essential for system dynamics.
<table><tr><td></td><td>ICWM</td><td> $\mathbf { w } / \mathbf { o } \ a \mathbf { c } \mathbf { t . }$ </td><td> $\mathbf { w } / \mathbf { o } \operatorname { i m g } .$ </td><td> $\mathbf { w } / \mathbf { o c t x . }$ </td><td>false ctx.</td></tr><tr><td> $4 5 ^ { \circ }$ </td><td>36.6</td><td> $3 0 . 8 _ { \downarrow 1 5 . 8 \% }$ </td><td> $1 7 . 6 { \scriptstyle \downarrow } 5 1 . 9 \%$ </td><td> $3 2 . 4 _ { \downarrow 1 1 . 5 \% }$ </td><td> $2 7 . 8 _ { \downarrow 2 4 . 0 \% }$ </td></tr><tr><td> $1 3 5 ^ { \circ }$ </td><td>2.2</td><td> $2 . 0 _ { \downarrow 9 . 1 \% }$ </td><td> $0 . 8 \downarrow 6 3 . 6 \%$ </td><td> $1 . 6 _ { \downarrow 2 7 . 3 \% }$ </td><td> $1 . 2 _ { \downarrow 4 5 . 5 \% }$ </td></tr><tr><td> $2 2 5 ^ { \circ }$ </td><td>8.8</td><td> $8 . 2 \ g _ { \downarrow 6 . 8 \% }$ </td><td> $3 . 0 _ { \downarrow 6 5 . 9 \% }$ </td><td> $7 . 8 _ { \downarrow 1 . 4 \% }$ </td><td> $4 . 2 _ { \downarrow . 5 2 . 3 \% }$ </td></tr><tr><td> $2 5 5 ^ { \circ }$ </td><td>28.4</td><td> $2 8 . 0 _ { \downarrow 1 . 4 \% }$ </td><td> $1 3 . 2 \scriptstyle \downarrow 5 3 . 5 \%$ </td><td> $2 7 . 8 \AA 1 2 . 1 \%$ </td><td> $2 4 . 4 _ { \downarrow , 1 4 . 1 \% }$ </td></tr><tr><td> $2 8 5 ^ { \circ }$ </td><td>36.6</td><td> $2 5 . 6 _ { \downarrow 3 0 . 1 \% }$ </td><td> $1 4 . 6 \AA 6 0 . 1 \%$ </td><td> $2 7 . 4 _ { \downarrow , 2 5 . 1 \% }$ </td><td> $2 3 . 6 _ { \downarrow , 3 5 . 5 \% }$ </td></tr><tr><td> $3 1 5 ^ { \circ }$ </td><td>37.6</td><td> $3 5 . 0 _ { \downarrow 6 . 9 \% }$ </td><td> $1 6 . 2 \scriptstyle \downarrow 5 6 . 9 \%$ </td><td> $3 5 . 0 _ { \downarrow 6 . 9 \% }$ </td><td> $3 2 . 4 _ { \downarrow , 1 3 . 8 \% }$ </td></tr><tr><td>avg</td><td>25.0</td><td> $2 1 . 6 _ { \downarrow 1 3 . 6 \% }$ </td><td> $1 0 . 9 \scriptstyle \downarrow . 5 6 . 4 \%$ </td><td> $2 2 . 0 _ { \downarrow 1 2 . 0 \% }$ </td><td> $1 8 . 9 _ { \downarrow . 2 4 . 4 \% }$ </td></tr></table>

![](images/05524fbe5d748d4eb1baa1fc8b44cc9b93fe97379d3c12caa90e68992365dc1e.jpg)  
Figure 7 t-SNE of Ψ <sup></sup><sub>T</sub> <sup></sup> across OOD viewpoints (perplexity=30).

Are the Learned Implicit Representations Identifiable? We examine whether $\Psi \mathcal { T }$ forms a structured representation with respect to system configuration. We visualize the hidden representations $\Psi \mathcal { T }$ via t-SNE across the six OOD viewpoints (1,024 points per viewpoint). As shown in $\mathrm { F i g . } 7 ,$ the representations exhibit tight within-viewpoint clustering, demonstrating stability, and clear between-viewpoint separation, demon strating identifiability.

## 6.2 What Probing Strategy is Efective?

A practical question for deploying ICWM is whether the choice of probing strategy matters. We evaluate four strategies: (a) Random, which samples target poses uniformly across all spatial directions; (b) XY-only, which restricts movements to the horizontal plane; (c) Z-only, which moves exclusively along the vertical axis; and (d) R-only, which varies only the end-efector orientation. As shown in Tab. 2, all four strategies consistently outperform the multi-view baseline by $1 5 { - } 2 7 \% ,$ confirming that the benefit of ICWM stems from the interaction format itself rather than any partic-

Table 2 Success rates (%) under diferent probing strategies on OOD viewpoints.
<table><tr><td></td><td>MV</td><td>Random</td><td>R-only</td><td>Z-only</td><td>XY-only</td></tr><tr><td>45°</td><td>30.4</td><td>36.6</td><td>35.8</td><td>34.6</td><td>34.6</td></tr><tr><td>135°</td><td>1.2</td><td>2.2</td><td>1.0</td><td>3.0</td><td>1.0</td></tr><tr><td>225°</td><td>2.8</td><td>8.8</td><td>6.0</td><td>13.0</td><td>14.8</td></tr><tr><td>255°</td><td>24.0</td><td>28.4</td><td>30.4</td><td>25.2</td><td>30.6</td></tr><tr><td>285°</td><td>32.8</td><td>36.6</td><td>38.8</td><td>31.2</td><td>33.6</td></tr><tr><td>315°</td><td>27.6</td><td>37.6</td><td>29.2</td><td>30.0</td><td>35.0</td></tr><tr><td> $\mathrm { A v g }$ </td><td>19.8</td><td>25.0</td><td>23.4</td><td>22.8</td><td>24.9</td></tr></table>

ular movement pattern. Performance diferences across strategies suggest that diferent axes expose diferent aspects of the local dynamics manifold, with no single strategy dominating across all viewpoints.

## 6.3 Does ICWM Generalize Beyond Camera Shifts?

To evaluate whether ICWM generalizes to diferent deployment settings, we stress-test the same trained model against two distinct categories of out-of-distribution perturbation: semantic scene variations and physical morphological changes. Experimental details are provided in Sec. C.2.

Semantic perturbations. As shown in (Fig. 8a), ICWM maintains a consistent margin over MV under both distractor objects (35.0 vs. 27.5) and novel table textures (41.2 vs. 37.5). The more moderate gains relative to viewpoint generalization likely reflect the scarcity of diverse scene-configuration data in current datasets rather than a fundamental limitation of the mechanism.

Morphological generalization. Attaching rigid spacers $( \Delta L \in \{ 2 0 , 4 0 , 8 0 \}$ mm) to the gripper flange alters forward kinematics at test time. ICWM consistently outperforms MV across all ofsets with a stable margin of $+ 6 0 \% ( \mathrm { F i g . ~ } 9 ) ;$ ; at $\Delta L = 8 0$ mm, where MV fails largely (5.6), ICWM retains non-trivial success (14.4) by inferring the altered kinematics from probing context.

Distractor  
20  
![](images/c4593d1a5b4e44003667f9a040cebf2e5cd09779f759521c470cc79d812aa878.jpg)

![](images/96d66a63e8cf033174e3b0e9ef8ede2a1c19a5c63cce337f332084897aea0299.jpg)

![](images/be02335688027f8fb20116a128bf2342a107c756e136812bd9531d3a9a34bde2.jpg)

![](images/1fbaa421da859693e0d3ae8e0009e56f0e03f17c59690ed118c23fbfade40622.jpg)  
(a) Semantic perturbations

![](images/9abf6acdb78bcda087b5088df4cbc40db2d3e40e76841b33c12662cf2374cc3e.jpg)  
(b) Morphological generalization  
Figure 8 Robustness to semantic scene variations and robot morphological changes. ICWM maintains a performance margin over the baseline when facing novel out-of-distribution perturbations.

We further validate this on a WindowX platform with systematically varied link lengths ({100%, 90%, 80%, 70%} of the original), training on two boundary configurations (100% and 70% link length) and evaluating zero-shot on two interpolated OOD configurations (90% and 80%). As the link-length ofset increases from 10% to 20%, MV’s average success rate collapses by more than half (57% → 28%), while ICWM degrades far more gracefully $( 7 7 \%  6 2 \% )$ , widening its margin over MV from 20 to 34 points (??). This mirrors the pattern observed with spacer attachments: as kinematic uncertainty grows, ICWM’s advantage over MV grows with it, consistent with the benefit stemming from explicit kinematic inference via the probing context.

![](images/b51959311a08c62f82366276b9f351361d5cf9e4b3bd3024f8d0f2158b2bd12c.jpg)  
Figure 9 Morphological generalization on WindowX.

## 6.4 Inference Latency Analysis

We evaluate the computational overhead of ICWM on a single NVIDIA RTX 4090 GPU. The extra latency is mainly due to the 2N image and N action tokens. While the baseline VLA requires 0.112s per inference step, our ICWM-enhanced model with N = 3 and N = 5 context clips incurs a latency of 0.165s and 0.185s respectively, without compromising control loop stability (Fig. 10). Furthermore, since the interaction context $\mathcal { T } _ { \psi }$ is static

![](images/c53936205b94d5381169056e17c0b35f52eb31925e8e93d82df241aeb935ee89.jpg)  
Figure 10 Comparison of inference time across diferent setting.

under a fixed configuration, its hidden states can be pre-computed and reused via KV caching, efectively reducing per-step inference cost back to near-baseline levels.

## 7 Conclusion

In this work, we identified the absence of explicit system configuration conditioning in modern VLA mod els as an underexplored limitation to generalization. Inspired by human motor adaptation, we proposed In-Context World Modeling (ICWM), which transforms the VLA policy from a static mapping into an adaptive inference engine. By conditioning the policy on self-generated forward interaction clips, ICWM implicitly captures the underlying sensory-motor relationship at test time without any parameter updates. Experiments across simulation benchmarks and real-world platforms demonstrate that ICWM substantially reduces spatial ambiguities under novel viewpoints, and can also extend to semantic scene variations and robot morphological changes.

## References

[1] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong T. Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J. Joshi, Alex Irpan, Brian Ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, and Kehang Han. RT-2: vision-language-action models transfer web knowledge to robotic control. In Jie Tan, Marc Toussaint, and Kourosh Darvish, editors, Conference on Robot Learning, CoRL 2023, 6-9 November 2023, Atlanta, GA, USA, volume 229 of Proceedings of Machine Learning Research, pages 2165–2183. PMLR, 2023. URL https://proceedings.mlr.press/v229/zitkovich23a.html.

[2] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. ArXiv, abs/2406.09246, 2024. URL https://api.semanticscholar.org/CorpusID:270440391.

[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164.

[4] Senyu Fei, Siyin Wang, Junhao Shi, Z. G. Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Libero-plus: In-depth robustness analysis of vision-language-action models. ArXiv, abs/2510.13626, 2025. URL https://api.semanticscholar.org/CorpusID:282102298.

[5] Letian Fu, Huang Huang, Gaurav Datta, Lawrence Yunliang Chen, Will Panitch, Fangchen Liu, Hui Li, and Ken neth Y. Goldberg. Icrt: In-context imitation learning via next-token prediction. 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 5937–5944, 2024. URL https://api.semanticscholar.org/CorpusID: 271974730.

[6] Rutav Shah, Shuĳing Liu, Qi Wang, Zhenyu Jiang, Sateesh Kumar, Mingyo Seo, Roberto Mart’in-Mart’in, and Yuke Zhu. Mimicdroid: In-context learning for humanoid robot manipulation from human play videos. ArXiv, abs/2509.09769, 2025. URL https://api.semanticscholar.org/CorpusID:281309736.

[7] Alec Radford, Jef Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. URL https://api.semanticscholar.org/CorpusID:160025533.

[8] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Bap tiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aur’elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and eficient foundation language models. ArXiv, abs/2302.13971, 2023. URL https://api.semanticscholar.org/CorpusID:257219404

[9] OpenAI. Introducing chatgpt, 2022. URL https://openai.com/blog/chatgpt.

[10] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jef Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Ma teusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. ArXiv, abs/2005.14165, 2020. URL https://api.semanticscholar.org/CorpusID:218971783.

[11] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. A survey on in-context learning. In Conference on Empirical Methods in Natural Language Processing, 2022. URL https://api.semanticscholar.org/CorpusID:255372865.

[12] Vuong Dinh An, Minh Nhat Vu, Dong An, and Ian D. Reid. Action tokenizer matters in in-context imitation learning. 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13490–13496, 2025. URL https://api.semanticscholar.org/CorpusID:276742267.

[13] Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, and Insup Lee. Ricl: Adding in-context adaptability to pre-trained vision-language-action models. ArXiv, abs/2508.02062, 2025. URL https://api.semanticscholar. org/CorpusID:280422322.

[14] Vidhi Jain, Maria Attarian, Nikhil J. Joshi, Ayzaan Wahid, Danny Driess, Quan Vuong, Pannag R. Sanketi, Pierre Sermanet, Stefan Welker, Christine Chan, Igor Gilitschenski, Yonatan Bisk, and Debidatta Dwibedi. Vid2robot: End to-end video-conditioned policy learning with cross-attention transformers. ArXiv, abs/2403.12943, 2024. URL https://api.semanticscholar.org/CorpusID:268532100.

[15] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

[16] Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. Advances in neural information processing systems, 32, 2019.

[17] Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. $\mathbb { R } ^ { 1 ^ { 2 } } \colon$ : Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.

[18] Luisa Zintgraf, Kyriacos Shiarlis, Maximilian Igl, Sebastian Schulze, Yarin Gal, Katja Hofmann, and Shimon Whiteson. Varibad: A very good method for bayes-adaptive deep rl via meta-learning. arXiv preprint arXiv:1910.08348, 2019.

[19] David R Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Neural Information Processing Systems, 2018. URL https://api.semanticscholar.org/CorpusID:52171619.

[20] Yann LeCun and Courant. A path towards autonomous machine intelligence version 0.9.2, 2022-06-27. 2022. URL https://api.semanticscholar.org/CorpusID:251881108.

[21] Jingtao Ding, Yunke Zhang, Yu Shang, Yuheng Zhang, Zefang Zong, J. Feng, Yuan Yuan, Hongyuan Su, Nian Li, Nicholas Sukiennik, Fengli Xu, and Yong Li. Understanding world or predicting future? a comprehensive survey of world models. ACM Computing Surveys, 58:1 – 38, 2024. URL https://api.semanticscholar.org/CorpusID: 274192171.

[22] Siyin Wang, Zhaoye Fei, Qinyuan Cheng, Shiduo Zhang, Panpan Cai, Jinlan $\mathrm { F u , }$ and Xipeng Qiu. World modeling makes a better planner: Dual preference optimization for embodied task planning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21518–21537, 2025.

[23] Siyin Wang, Junhao Shi, Zhaoyang Fu, Xinzhe He, Feihong Liu, Chenchen Yang, Yikang Zhou, Zhaoye Fei, Jingjing Gong, Jinlan Fu, et al. World action models: The next frontier in embodied ai. arXiv preprint arXiv:2605.12090, 2026.

[24] Hongtao Wu, Ya Jing, Chi-Hou Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. ArXiv, abs/2312.13139, 2023. URL https://api.semanticscholar.org/CorpusID:266374724.

[25] Peiyan Li, Hongtao Wu, Yan Huang, Chi-Hou Cheang, Liang Wang, and Tao Kong. Gr-mg: Leveraging partiallyannotated data via multi-modal goal-conditioned policy. IEEE Robotics and Automation Letters, 10:1912–1919, 2024. URL https://api.semanticscholar.org/CorpusID:271957548.

[26] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1702–1713, 2025. URL https://api.semanticscholar. org/CorpusID:277435005.

[27] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. Worldvla: Towards autoregressive action world model. ArXiv, abs/2506.21539 2025. URL https://api.semanticscholar.org/CorpusID:280010695.

[28] Danĳar Hafner, Timothy P. Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. ArXiv, abs/1912.01603, 2019. URL https://api.semanticscholar.org/CorpusID:208547755.

[29] Danĳar Hafner, Timothy P. Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. ArXiv, abs/2010.02193, 2020. URL https://api.semanticscholar.org/CorpusID:222133157.

[30] Philipp Wu, Alejandro Escontrela, Danĳar Hafner, Ken Goldberg, and P. Abbeel. Daydreamer: World models for physical robot learning. In Conference on Robot Learning, 2022. URL https://api.semanticscholar.org/ CorpusID:250088882.

[31] Danĳar Hafner, J. Pašukonis, Jimmy Ba, and Timothy P. Lillicrap. Mastering diverse domains through world models. ArXiv, abs/2301.04104, 2023. URL https://api.semanticscholar.org/CorpusID:255569874.

[32] Ruĳie Zheng, Jing Wang, Scott Reed, Johan Bjorck, Yu Fang, Fengyuan Hu, Joel Jang, Kaushil Kundalia, Zongyu Lin, Loic Magne, Avnish Narayan, You Liang Tan, Guanzhi Wang, Qi Wang, Jiannan Xiang, Yinzhen Xu, Seonghyeon Ye, Jan Kautz, Furong Huang, Yuke Zhu, and Linxi Fan. Flare: Robot learning with implicit world modeling. ArXiv, abs/2505.15659, 2025. URL https://api.semanticscholar.org/CorpusID:278782778.

[33] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and P. Abbeel. Learning universal policies via text-guided video generation. ArXiv, abs/2302.00111, 2023. URL https://api.semanticscholar.org/CorpusID:256459809.

[34] Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, Loic Magne, Ajay Mandlekar, Avnish Narayan, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Yinzhen Xu, Xi Zeng, Kaiyuan Zheng, Ruĳie Zheng, Ming-Yu Liu, Luke S. Zettlemoyer, Dieter Fox, Jan Kautz, Scott Reed, Yuke Zhu, and Linxi Fan. Dreamgen: Unlocking generalization in robot learning through video world models. 2025. URL https://api.semanticscholar.org/CorpusID:279447300.

[35] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. ArXiv, abs/2412.15109, 2024. URL https: //api.semanticscholar.org/CorpusID:274859727.

[36] Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecofet, Brandon Houghton, Raul Sampedro, and Jef Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. ArXiv abs/2206.11795, 2022. URL https://api.semanticscholar.org/CorpusID:249953673.

[37] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action difusion for pretraining on large robotic datasets. ArXiv, abs/2504.02792, 2025. URL https://api.semanticscholar.org/CorpusID:277510147.

[38] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. ArXiv, abs/2503.00200, 2025. URL https://api.semanticscholar.org/CorpusID:276741531.

[39] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: benchmarking knowledge transfer for lifelong robot learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets\_and\_Benchmarks.html.

[40] Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U-Xuan Tan, Navonil Majumder, and Soujanya Poria. Nora: A small open-sourced generalist vision language action model for embodied tasks. ArXiv, abs/2504.19854, 2025. URL https://api.semanticscholar.org/CorpusID:278165428.

[41] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: eficient action tokenization for vision-language-action models. CoRR, abs/2501.09747, 2025. doi: 10.48550/ARXIV.2501.09747. URL https://doi.org/10.48550/arXiv.2501.09747.

[42] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Rich Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization. ArXiv, abs/2504.16054, 2025. URL https://api.semanticscholar.org/CorpusID:277993634.

[43] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shĳie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025. URL https://api.semanticscholar.org/CorpusID: 276449796.

## Appendix

## A Proof of Proposition 1

We prove that the interaction context $\mathcal { T } ~ = ~ \left( o _ { 0 : t } , a _ { 1 : t } \right)$ carries strictly more information about the system configuration $\psi$ than a single observation $o _ { 0 }$ alone.

Proof. Since $\psi \subseteq s _ { 0 } ,$ , it sufices to prove the stronger statement $I \left( s _ { 0 } ; o _ { 0 : t } \mid a _ { 1 : t } \right) > I \left( s _ { 0 } ; o _ { 0 } \right)$ , from which the theorem follows by the data processing inequality.

Applying the chain rule of mutual information:

$$
I \left( s _ { 0 } ; o _ { 0 : t } \mid a _ { 1 : t } \right) = I \left( s _ { 0 } ; o _ { 0 } \mid a _ { 1 : t } \right) I \left( s _ { 0 } ; o _ { 1 : t } \mid o _ { 0 } , a _ { 1 : t } \right) .\tag{8}
$$

First term. In the graphical model, $s _ { 0 }$ is a root node and $a _ { 1 : t }$ are exogenous root nodes. Every path between $s _ { 0 } (  o \mathbf { r } o _ { 0 } )$ and any $a _ { k }$ passes through a collider at $s _ { k }$ via $s _ { k - 1 } \to s _ { k }  a _ { k }$ . Since no collider or its descendants are conditioned upon, these paths are blocked by d-separation, giving $( s _ { 0 } , o _ { 0 } ) \perp a _ { 1 : t }$ . Therefore:

$$
I \left( s _ { 0 } ; o _ { 0 } \mid a _ { 1 : t } \right) = I \left( s _ { 0 } ; o _ { 0 } \right) .\tag{9}
$$

Second term. We show $I \left( s _ { 0 } ; o _ { 1 : t } \mid o _ { 0 } , a _ { 1 : t } \right) > 0 ,$ . Consider the path $s _ { 0 }  s _ { 1 }  \cdot \cdot \cdot  s _ { k }  o _ { k }$ for any $k \geq 1$ Conditioning on $o _ { 0 } - \mathsf { a }$ descendant of s<sub>0</sub>—activates the collider at $s _ { 0 } ,$ , leaving the path $s _ { 0 }  s _ { k }  o _ { k }$ active under d-separation given $\left\{ o _ { 0 } , a _ { 1 : t } \right\}$ . Hence s<sub>0</sub> $\chi \mid o _ { k } \mid o _ { 0 } , a _ { 1 : t }$ . By A2, the state chain preserves information about $s _ { 0 } ,$ , so:

$$
I \left( s _ { 0 } ; o _ { 1 : t } \mid o _ { 0 } , a _ { 1 : t } \right) \ge I \left( s _ { 0 } ; o _ { k } \mid o _ { 0 } , a _ { 1 : t } \right) > 0 .\tag{10}
$$

Substituting (9) and (10) into (8):

$$
\begin{array} { r } { I \left( s _ { 0 } ; o _ { 0 : t } \mid a _ { 1 : t } \right) = I \left( s _ { 0 } ; o _ { 0 } \right) \underbrace { I \left( s _ { 0 } ; o _ { 1 : t } \mid o _ { 0 } , a _ { 1 : t } \right) } _ { > \ 0 } > I \left( s _ { 0 } ; o _ { 0 } \right) . } \end{array}\tag{11}
$$

□

## B Details of Simulated Experiment Setup

## B.1 Experimental Configuration

System Identification Benchmark Design. Unlike the standard LIBERO setup, which typically evaluates policies under a fixed, single camera viewpoint, our experimental protocol is specifically designed to stresstest OOD viewpoint generalization. We distribute 14 discrete azimuthal angles around the workspace center. We designate 8 In-Domain (ID) angles for training: $\psi _ { t r a i n } \in \{ 3 0 ^ { \circ } , 6 0 ^ { \circ } , \bar { 9 0 ^ { \circ } } , 1 2 0 ^ { \circ } , 2 4 0 ^ { \circ } , 2 7 0 ^ { \circ } , 3 0 \bar { 0 } ^ { \circ } , 3 3 0 ^ { \circ } \}$ while withholding 6 Out-of-Domain (OOD) angles: $\psi _ { t e s t } \in \{ 4 5 ^ { \circ } , 1 3 5 ^ { \circ } , 2 2 5 ^ { \circ } , 2 5 5 ^ { \circ } , 2 8 5 ^ { \circ } , 3 1 5 ^ { \circ } \}$ exclusively for evaluation. Fig. 11 illustrates the drastic visual shifts between these angles, which introduce spatial ambiguities that standard VLA policies struggle to resolve without explicit system identification.

LIBERO Suite Specifications. To evaluate the versatile capabilities of our model, we utilize the LIBERO benchmark [39], categorized into four specialized suites:

• LIBERO-Spatial: Benchmarks spatial reasoning by presenting identical objects in randomized initial layouts, requiring the agent to disambiguate relative geometric cues $( \mathrm { e . g . , \mathrm { ' } p i c k }$ up the black bowl next to the plate and place it on the plate”).

![](images/ba3f5253de2fc0416e49fbd2cfa898fbc2b24680780ded5af3bea29d8a3d0817.jpg)

![](images/92183b45a250b052a54f54bf7e187dad2265355c3684bde5e4d834a33aabd2ab.jpg)  
Figure 11 Viewpoint Distribution for Training and Evaluation. We illustrate the 8 in-domain training angles and 6 outof-domain testing angles used to benchmark viewpoint generalization.

• LIBERO-Goal: Evaluates goal-conditioning by assigning multiple distinct objectives to the same scene configuration, assessing whether the policy can map diferent instructions to varied behavioral distributions (e.g., “open the top drawer and put the bowl inside”).

• LIBERO-Object: Focuses on semantic grounding by introducing a diverse set of object categories within fixed layouts, testing the model’s ability to generalize to novel visual appearances (e.g., “pick up the bbq sauce and place it in the basket”).

• LIBERO-Long: Challenges the model with 10 complex, multi-stage manipulation sequences that require sustained temporal consistency and the ability to recover from small execution errors over extended horizons (e.g., “turn on the stove and put the moka pot on it”).

Data Synthesis and Preprocessing. The training data is generated by replaying expert demonstration trajectories from the original benchmark in the simulator and re-rendering them from the 8 ID camera poses. Following the preprocessing protocol of OpenVLA [2], we filter out unsuccessful episodes and remove redundant frames where action norms are near-zero and the gripper state is static, ensuring a high-density learning signal.

## B.2 Detailed Results

Tab. 3 and Tab. 4 provide the exhaustive breakdown of success rates for ID and OOD viewpoints. Notably, ICWM maintains robust performance across the majority of evaluated conditions, outperforming multi-view behavior cloning and explicit configuration approaches in most settings. In particular, for LIBERO-Long, ICWM achieves a 25.0% average OOD success rate, a significant margin over the 19.8% achieved by the Multi-View (MV) baseline, demonstrating the power of in-context calibration for complex tasks.

We also observe that certain OOD viewpoints, particularly 135°, pose challenges for all methods, including ICWM. We attribute this to viewpoint-specific geometric constraints: at 135°, the camera angle may introduce a certain degree of object occlusion and reduce the efective workspace visible to the model, which can cause manipulation targets to occasionally exit the field of view during execution. This suggests a perceptual limitation shared across methods, rather than a failure specific to ICWM alone.

To further investigate the model’s behavior, we visualize the rollout trajectories in Figure 12. ICWM demonstrates remarkable execution stability; even when the initial viewpoint causes a spatial ofset, the model utilizes the in-context world model to “re-align” its end-efector during the first few steps of the task, ensuring

Table 3 Success Rates (%) on In-Domain (Seen) Viewpoints.
<table><tr><td colspan="2"></td><td> $3 0 ^ { \circ }$ </td><td> $6 0 ^ { \circ }$ </td><td> $9 0 ^ { \circ }$ </td><td> $1 2 0 ^ { \circ }$ </td><td> $2 4 0 ^ { \circ }$ </td><td> $2 7 0 ^ { \circ }$ </td><td> $3 0 0 ^ { \circ }$ </td><td> $3 3 0 ^ { \circ }$ </td><td> $\operatorname { A v g }$ </td></tr><tr><td rowspan="5">Spatial</td><td>π-FAST</td><td>19.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>14.0</td><td>4.1</td></tr><tr><td> $\pi _ { 0 . 5 }$ </td><td>39.0</td><td>0.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.6</td><td>20.6</td><td>7.6</td></tr><tr><td>NORA</td><td>11.4</td><td>0.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.2</td><td>0.0</td><td>17.8</td><td>3.8</td></tr><tr><td>MV</td><td>79.4</td><td>77.2</td><td>69.4</td><td>65.4</td><td>72.8</td><td>71.4</td><td>81.8</td><td>78.4</td><td>74.5</td></tr><tr><td>EXP</td><td>79.8</td><td>81.8</td><td>68.2</td><td>69.2</td><td>70.6</td><td>75.6</td><td>79.4</td><td>82.6</td><td>75.9</td></tr><tr><td></td><td>ICWM (ours)</td><td>84.6</td><td>84.0</td><td>75.6</td><td>73.8</td><td>83.0</td><td>77.0</td><td>83.6</td><td>88.0</td><td>81.2</td></tr><tr><td rowspan="5">Goal</td><td>π-FAST</td><td>7.8</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>10.0</td><td>2.2</td></tr><tr><td>π0.5</td><td>40.2</td><td>4.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.4</td><td>28.0</td><td>9.2</td></tr><tr><td>NORA</td><td>5.6</td><td>0.0</td><td>0.4</td><td>0.0</td><td>0.2</td><td>0.0</td><td>0.0</td><td>4.8</td><td>1.4</td></tr><tr><td>MV</td><td>78.4</td><td>77.2</td><td>68.4</td><td>74.0</td><td>72.8</td><td>69.0</td><td>71.4</td><td>75.4</td><td>73.3</td></tr><tr><td>EXP</td><td>78.2</td><td>71.0</td><td>66.4</td><td>68.4</td><td>68.0</td><td>62.8</td><td>72.8</td><td>77.6</td><td>70.7</td></tr><tr><td></td><td>ICWM (ours)</td><td>82.2</td><td>75.4</td><td>71.0</td><td>69.0</td><td>62.2</td><td>64.6</td><td>72.2</td><td>76.0</td><td>71.6</td></tr><tr><td rowspan="5">Object</td><td>π-FAST</td><td>0.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>11.8</td><td>1.5</td></tr><tr><td>π0.5</td><td>13.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>7.6</td><td>43.4</td><td>8.1</td></tr><tr><td>NORA</td><td>0.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>4.6</td><td>0.6</td></tr><tr><td>MV</td><td>66.0</td><td>67.0</td><td>72.4</td><td>60.2</td><td>54.6</td><td>52.6</td><td>73.8</td><td>72.8</td><td>64.9</td></tr><tr><td>EXP</td><td>62.0</td><td>68.2</td><td>72.4</td><td>63.4</td><td>62.4</td><td>55.6</td><td>73.6</td><td>75.6</td><td>66.6</td></tr><tr><td></td><td>ICWM (ours)</td><td>74.0</td><td>70.6</td><td>71.6</td><td>60.4</td><td>69.8</td><td>62.0</td><td>76.6</td><td>79.0</td><td>70.5</td></tr><tr><td rowspan="5">Long</td><td>π-FAST</td><td>2.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>2.6</td><td>0.7</td></tr><tr><td>π0.5</td><td>13.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>10.0</td><td>2.9</td></tr><tr><td>NORA</td><td>1.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.2</td><td>6.2</td><td>1.0</td></tr><tr><td>MV</td><td>36.2</td><td>32.0</td><td>29.6</td><td>24.8</td><td>28.0</td><td>28.2</td><td>33.4</td><td>34.0</td><td>30.8</td></tr><tr><td>EXP</td><td>39.8</td><td>32.6</td><td>26.4</td><td>31.0</td><td>30.4</td><td>26.8</td><td>33.2</td><td>39.0</td><td>32.4</td></tr><tr><td></td><td>ICWM (ours)</td><td>45.2</td><td>41.2</td><td>37.2</td><td>39.4</td><td>38.4</td><td>34.2</td><td>39.8</td><td>44.6</td><td>40.0</td></tr></table>

a successful grasp that baselines frequently miss due to viewpoint-induced depth and coordinate errors.

1 Pick up salad dressing and place in basket  
![](images/d97d654cf4803eda24b5fc30f332cf42c00da888abf28f4a1da304130ea13629.jpg)

2 Pick up chocolate pudding and place in basket  
![](images/a078360d1716b4258162819eaf32b9d844998f54b7cc91eed7d2f4cd59727653.jpg)

3 Turn on stove and put moka on it  
![](images/7581332e07152141a1b0b9356854121f6cc720233c09897b701522cb967aa716.jpg)

![](images/c884be9bfaca1c2641fa6e7a2b1875bdb515bfa727f8aa9dc9cb0c015942c88a.jpg)  
Figure 12 Visualization of Successful Task Rollouts in Simulation. By utilizing the interaction prefix to resolve viewpoint induced ambiguities, the agent achieves precise grasping and multi-stage execution (e.g., turning on a stove) without any environment-specific fine-tuning.

Table 4 Success Rates (%) on Out-of-Domain (Unseen) Viewpoints.
<table><tr><td></td><td></td><td>45°</td><td>135°</td><td>225°</td><td>255°</td><td>285°</td><td>315°</td><td>Avg</td></tr><tr><td rowspan="5">Spatial</td><td>π-FAST</td><td>2.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.4</td><td>3.4</td><td>1.1</td></tr><tr><td>π0.5</td><td>2.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>8.8</td><td>1.8</td></tr><tr><td>NORA</td><td>6.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>3.6</td><td>1.6</td></tr><tr><td>MV</td><td>73.2</td><td>3.2</td><td>18.2</td><td>51.4</td><td>68.2</td><td>75.6</td><td>48.3</td></tr><tr><td>EXP</td><td>71.2</td><td>1.8</td><td>11.0</td><td>53.6</td><td>65.2</td><td>75.2</td><td>46.3</td></tr><tr><td></td><td>ICWM (ours)</td><td>78.2</td><td>2.4</td><td>17.0</td><td>55.4</td><td>71.2</td><td>75.0</td><td>49.9</td></tr><tr><td rowspan="5">Goal</td><td>π-FAST</td><td>0.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1.0</td><td>0.2</td></tr><tr><td>π0.5</td><td>29.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>6.2</td><td>5.9</td></tr><tr><td>NORA</td><td>0.0</td><td>0.0</td><td>0.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MV</td><td>60.4</td><td>13.2</td><td>13.8</td><td>34.2</td><td>53.2</td><td>57.6</td><td>38.7</td></tr><tr><td>EXP</td><td>62.6</td><td>21.8</td><td>18.0</td><td>40.8</td><td>51.2</td><td>54.6</td><td>41.5</td></tr><tr><td></td><td>ICWM (ours)</td><td>74.8</td><td>18.6</td><td>18.0</td><td>41.0</td><td>50.0</td><td>62.8</td><td>44.2</td></tr><tr><td rowspan="5">Object</td><td>π-FAST</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>4.2</td><td>0.7</td></tr><tr><td>π0.5</td><td>0.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>6.6</td><td>1.2</td></tr><tr><td>NORA</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MV</td><td>25.4</td><td>0.2</td><td>0.4</td><td>11.4</td><td>13.8</td><td>24.8</td><td>12.7</td></tr><tr><td>EXP</td><td>23.4</td><td>0.2</td><td>0.0</td><td>15.2</td><td>14.8</td><td>38.2</td><td>15.3</td></tr><tr><td></td><td>ICWM (ours)</td><td>28.6</td><td>0.0</td><td>1.0</td><td>24.2</td><td>20.8</td><td>20.8</td><td>15.9</td></tr><tr><td rowspan="5">Long</td><td>π-FAST</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>π0.5</td><td>0.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.4</td><td>0.1</td></tr><tr><td>NORA</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MV</td><td>30.4</td><td>1.2</td><td>2.8</td><td>24.0</td><td>32.8</td><td>27.6</td><td>19.8</td></tr><tr><td>EXP</td><td>29.8</td><td>1.0</td><td>6.4</td><td>24.0</td><td>27.8</td><td>32.2</td><td>20.2</td></tr><tr><td></td><td>ICWM (ours)</td><td>36.6</td><td>2.2</td><td>8.8</td><td>28.4</td><td>36.6</td><td>37.6</td><td>25.0</td></tr></table>

## C Details of Real-world Experiments

## C.1 Experimental Configuration

Physical Platform and Perception. As illustrated in Fig. 13, our real-robot workstation is centered around a 6-DoF UR5e manipulator equipped with a Robotiq 2F-85 parallel gripper, providing a versatile platform for diverse manipulation tasks. To create a challenging multi-view sensing environment, we deploy an array of 12 cameras strategically positioned at varying elevations and azimuthal angles. This setup ensures comprehensive coverage of the workspace while introducing significant perspective-induced spatial distortions.

Generalization Protocol and Data Collection. To rigorously evaluate zero-shot adaptation, we partition the 12-camera system into two distinct subsets: 6 cameras are designated for training, while the remaining 6 cameras are withheld exclusively for testing. This balanced split ensures that the model is evaluated on a wide range of previously unseen perspectives, requiring success to depend entirely on functional system identification rather than the memorization of specific camera-to-robot geometries. For task-specific knowledge, we collect approximately 100–150 human demonstrations per task via teleoperation, covering four representative manipulation tasks:

1. Spatial Reasoning & Disambiguation: “Put the toy on the box into the basket.” This task requires the agent to understand the vertical spatial relationship between the toy and the box, necessitating precise endefector positioning to take the toy without disturbing the support surface—a process highly sensitive to viewpoint-induced depth errors

2. Fine-grained Alignment: “Stack the yellow cup onto the red cup.” This serves as a benchmark for high-precision motor control, where the agent must align the principal axes of two objects under novel perspective projections.

3. Structural Manipulation: “Lift the basket.” This task focuses on handle-centric grasping of large-scale empty containers, testing the model’s ability to ground actions on specific structural affordances of an object.

4. Multi-Object Semantic Grounding: “Pick up the eggplant and place it onto the red plate.” Conducted in a cluttered scene with multiple distractor objects, this task assesses the model’s ability to maintain correct object-instruction alignment when viewed from unfamiliar angles that may cause occlusion or visual overlap.

For task-specific knowledge, we collect approximately 100–150 high-quality human demonstrations per task via teleoperation. These demonstrations capture the necessary precision and coordination for successful execution across diverse initial object layouts.

![](images/20d1322626e6dd685f65e42ad2a8610c399227aedb8cf3a7635974040a1301f3.jpg)  
Figure 13 Real-World Experimental Platform. Our physical setup comprises a 6-DoF UR5e manipulator equipped with a parallel gripper and a comprehensive 12-camera perception array distributed at diverse elevations and azimuthal angles.

## C.2 Details of Generalization Experiments

Semantic Perturbations. Semantic perturbation experiments are evaluated across 4 tasks × 4 in-domain viewpoints × 10 trials per condition (160 trials per bar). We evaluate two types of scene-level variation: (1) Distractor objects: 10 task-irrelevant objects are placed within the workspace; (2) Novel table textures: the table surface is replaced with 4 unseen textures absent during training.

Morphological Generalization-UR5E. Morphological generalization experiments are evaluated across 4 tasks × 4 in-domain viewpoints × 10 trials x 3 morphological settings, with rigid spacers of ΔL ∈ {20, 40, 80} mm attached to the gripper flange to systematically alter forward kinematics at test time.

Morphological Generalization-WindowX. To further validate ICWM’s generalization capability beyond rigid spacer attachments, we conduct an additional experiment on a WindowX platform with systematically varied link lengths. Specifically, we scale the robot’s link lengths to {100%, 90%, 80%, 70%} of the original, yielding four distinct morphological configurations (Fig. 14). The model is trained on the two boundary configurations (100% and 70% link length) and evaluated on the two interpolated out-of-distribution configurations (90% and 80%), assessing generalization to unseen kinematic structures without any parameter updates. Experiments are conducted across 3 tasks × 20 trials × 2 morphological settings (120 trials total).

## D Details of Self-Exploration Probing

## D.1 Implementation Details

To enable test-time adaptation, we implement a stochastic self-exploration phase that allows the agent to synchronize its internal coordinate system with the current camera viewpoint. This phase is designed to be task-agnostic and prioritizes safety during real-robot deployment.

The exploration process involves moving the robot’s end-efector toward a randomly sampled target coordinate $g _ { p r o b e } \in \mathbb { R } ^ { 3 }$ . The control is executed via a fixed-step movement policy rather than a goal-reaching controller. It is important to note that the robot is not required to reach the target point; the primary objective is to generate diverse visual-motor transitions $o _ { t } , a _ { t } , o _ { t 1 }$ that expose the underlying action-observation mapping under the current configuration ψ. These interaction fragments are then collected and organized as the in-context prefix T .

2 Pick up the eggplant to the red plate  
![](images/9038f8d9d8cc6e2dd31b86712d3f4761225fb79df720b6ad9c921b44ec4d4607.jpg)  
Figure 14 Additional morphological generalization evaluation on the WindowX platform. We shorten the robot’s link lengths to {100%, 90%, 80%, 70%} of the original, yielding four distinct morphological configurations.

![](images/9475944ddd5fbf4008d24db14ec7a79763d41f78a90d8419409506e1ea86d9c4.jpg)

![](images/57cbdd3bc57f558a4f256e0728ae3287d5286729a32294a99a485e64d74f2fa2.jpg)

![](images/6737186d392bda0a826c15ebdc70b232449022a174810d1f3d2068f477828394.jpg)  
Figure 15 The supplementary successful trajectories from real-robot experiments. These trajectories illustrate the ICWM model’s capability in completing lifting and pick-and-place maneuvers, as well as its robustness against object distur bances.

Workspace-Constrained Random Exploration (Real-Robot). For real-world experiments, probing targets g<sub>probe</sub> are sampled uniformly at random within the robot’s reachable workspace. Rather than requiring manual specification, this workspace bounding box is derived automatically from the robot’s forward kinematics model—given the joint limits of the UR5e, the reachable Cartesian space in the robot base frame can be computed analytically, requiring no human annotation or scene-specific calibration. The robot moves toward each sampled target via fixed-step increments, ensuring all motions remain within the operable joint limits. The resulting transitions cover diverse spatial regions of the workspace, forming the in-context prefix T .

![](images/b56d303a93c42a59c67cde2ee4391138bb2dc063a25ad894cedb24b6512519c9.jpg)  
Figure 16 Visualization of start (o<sup>s</sup>) and end frames (o<sup>e</sup>) for random interaction clips in the real world and simulation.

Crucially, since the bounding box is defined entirely in the robot’s base frame, it remains invariant across arbitrary camera viewpoint shifts or semantic scene changes, requiring only a one-time specification for a fixed physical workstation setup. The random movements are task-agnostic and configuration-agnostic, carrying no task-relevant information. Our zero-shot claim refers specifically to the absence of task-specific demonstrations or parameter updates for any novel configuration; in simulation, this step is unnecessary entirely.

Timing Analysis. The probing phase introduces minimal overhead. Before task execution, the robot performs 20 probing actions spanning the workspace in approximately 5–6 seconds total. During task execution, N=5 triplets are randomly sampled from these 20 recorded transitions as the in-context prefix at each inference step. Crucially, this 20-step probing is performed once per deployment: the same context pool is reused throughout the entire task execution with no additional probing required. Given the multi-step nature of VLA manipulation tasks, this one-time overhead constitutes a small fraction of total deployment time, and is further amortized to negligible levels when the same configuration is reused across tasks.

## D.2 Case Visualization

We present the start (o<sup>s</sup>) and end frames (o<sup>e</sup>) of random action interaction clips from both the real world and simulation in Fig. 16.