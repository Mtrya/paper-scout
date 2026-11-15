<!-- page 1 of 35 -->

arXiv:2511.09484v3 [cs.RO] 26 Sep 2026

# SPIDER: Scalable Physics-Informed Dexterous Retargeting

**Chaoyi Pan**<sup>1,2,∗</sup>, **Changhao Wang**<sup>1</sup>, **Haozhi Qi**<sup>1</sup>, **Julen Urain**<sup>1</sup>, **Zixi Liu**<sup>1</sup>, **Homanga Bharadhwaj**<sup>1</sup>, **Akash Sharma**<sup>1</sup>, **Tingfan Wu**<sup>1,†</sup>, **Guanya Shi**<sup>2,†</sup>, **Jitendra Malik**<sup>1,†</sup>, **Francois Hogan**<sup>1,†</sup>

<sup>1</sup>FAIR at Meta, <sup>2</sup>Carnegie Mellon University <sup>∗</sup>Work done at Meta,<sup>†</sup>Joint last author

Learning agile robotic policies requires large-scale demonstrations, but translating abundant human motion data to robots is bottlenecked by the embodiment gap and missing dynamic information. To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (**SPIDER**), a physics based retargeting framework to transform and augment kinematic-only human demonstrations into dynamically feasible robot trajectories at scale. Our key insight is that human demonstrations should provide global task structure and objective, while a sampling-based solver can be used to find the feasible solution given the physics constraints. As a general framework, **SPIDER** is an efficient physicsbased retargeting method that can be applied to both humanoid whole-body loco-manipulation and dexterous manipulation across 9 humanoid/dexterous hand embodiments, 6 datasets and 3 simulators. By bypassing the need for policy optimization, it achieves state-of-the-art performance while being 10x faster than reinforcement learning (RL) baselines. Furthermore, **SPIDER** enables physics-based data augmentation—such as imposing external payloads, perturbations, and new contact patterns—to generate diverse data. We demonstrate that the retargeted motion can be executed on the robot directly open-loop or serve as feasible reference motion for efficient RL policy learning. Our pipeline enables large-scale generation of robot trajectories from human motion, and we release a dataset of 6,520 simulated demonstrations across four hands and 191 dataset-specific object identities to support future research.

**Date:** September 29, 2026

**Correspondence:** [chaoyip@andrew.cmu.edu](mailto:chaoyip@andrew.cmu.edu) **Code:** [https://github.com/facebookresearch/spider](https://github.com/facebookresearch/spider)**Website:** [https://jc-bao.github.io/spider-project](https://jc-bao.github.io/spider-project)**Dataset:** [https://huggingface.co/datasets/retarget/retarget\_full](https://huggingface.co/datasets/retarget/retarget_full)

∞Meta

![Image block](doc:f3190a8/tier:basic/page:1/block:9)

Figure 1 Overview of SPIDER. The sampling-based physics pipeline converts human-object interaction into dynamically feasible robot trajectories across embodiments and task domains, with virtual contact guidance preserving intended contact modes.

1

<!-- page 2 of 35 -->

## 1 Introduction

Whole-body dexterous manipulation has rapidly advanced by learning from massive internet-scale datasets (Kim et al., 2024) and human demonstrations (Team, 2025). Yet, acquiring generalizable whole-body manipulation across embodiments, from dexterous hands (Yin et al., 2025b; Zhong et al., 2025; Qi et al., 2023) to humanoid whole-body control (Li et al., 2025b; Ze et al., 2025), remains prohibitively expensive due to hardware constraints, task complexity, and lack of large-scale embodiment-specific data (Zhang et al., 2025; Xu et al., 2025). On the other hand, abundant human motion data is readily available, including large-scale motioncapture datasets (Zhan et al., 2024; Mahmood et al., 2019), internet-scale video collections (Ren et al., 2025), and VR-based interfaces (Hoque et al., 2025). Recent advances in computer vision have made it possible to reconstruct 3D human body (Goel et al., 2023) and hand motion (Pavlakos et al., 2024), as well as object meshes (Xiang et al., 2025) and trajectories (Wen et al., 2024), directly from videos. These developments create a unique opportunity to leverage human motion as a scalable source of demonstrations for learning humanoid and dexterous robot control. However, a fundamental challenge arises: the embodiment gap - the mismatch in morphology, dynamics, and actuation between humans and robots - which leads to infeasible motion during transfer. This motivates our central research question:

How can we efficiently and reliably transform human motion into feasible robot trajectories that

respect dynamics and contact?

We formulate this as a physics-based retargeting problem (Reda et al., 2023): given kinematic human demonstrations, generate robot motions that (a) align the robot’s poses with human poses, (b) follow physics constraints, and (c) preserve the task objectives of the demonstrated behavior (e.g. object manipulation). The problem arises from the following challenges: (a) Flexibility and generalizability: given diverse data sources and target embodiments, the framework should work for diverse tasks and human-object interactions. (b) Scalability and efficiency: the abundance of internet-scale human data requires approaches that are both computationally efficient and scalable to large datasets. Existing methods struggle to bridge these gaps: inverse-kinematics (IK) approaches (Qin et al., 2022; Yin et al., 2025a) are efficient but dynamically infeasible; reinforcement learning (RL) approaches (Li et al., 2025c; Lum et al., 2025) are general but require expensive trajectory-specific policy optimization; teleoperation (Yin et al., 2025b) is dynamically feasible but often labor-intensive and embodiment-specific.

To achieve scalable, general, and flexible physics-based cross-embodiment retargeting, we propose Scalable Physics-Informed DExterous Retargeting (**SPIDER**), a sampling-based approach with contact guidance. Our key insight is that human demonstrations provide high-level guidance in terms of robot motion and task specification, while a physics simulator grounds trajectories to ensure dynamical feasibility and contact correctness through large-scale sampling. Our method lies in the middle of the trade-off between gradientbased optimization and reinforcement learning: it preserves the efficiency of optimization-based methods as a learning-free method, while achieving the generalizability of reinforcement learning as no gradient is required. The resulting framework is embodiment-agnostic and task-general: it can be applied to any robot-environment interaction as long as the scene can be simulated.

**Contributions.** Our contributions are summarized as follows:

• We introduce **SPIDER**, a general and scalable physics-based retargeting framework that supports both dexterous manipulation and humanoid whole-body loco-manipulation across 9 diverse robot embodiments, 6 datasets, and 3 simulators. It achieves state-of-the-art performance while being 10× faster than RL-based methods.

• We demonstrate our method’s flexibility and applicability to diverse policy learning and data generation applications, including more efficient RL policy learning, data augmentation for novel physics constraints, user intent preservation with contact guidance, and trajectory robustification for direct real-robot deployment.

• Our pipeline enables the generation of large-scale, robot-feasible datasets, with 6532 demonstrations across 4 distinct robotic hands and 191 different objects, derived from human data. We release the full data generation pipeline to assist future research.

2

<!-- page 3 of 35 -->

## 2 Related Work

## 2.1 Learning Manipulation from Human Data

Recent work explores learning robot skills from large-scale human videos by first detecting actions and transferring them to robots. One common strategy is kinematic retargeting, where human poses or keypoints are mapped to robot motions, as in DemoDiffusion (Park et al., 2025), OKAMI (Li et al., 2024), R+X (Papagiannis et al., 2025), and EgoZero (Liu et al., 2025). Another strategy is to train a human-centric policy and then adapt it to robots through fine-tuning with in-domain data, as in MimicPlay (Wang et al., 2023), Track2Act (Bharadhwaj et al., 2024), and VideoDex (Shaw et al., 2022). Our approach is complementary to these pipelines. Specifically, **SPIDER** can serve as a drop-in replacement for the human-to-robot action transfer components, providing more accurate and contact-aware mappings for dexterous and whole-body control.

## 2.2 Retargeting from Human Data

Motion retargeting seeks to convert human data into robot trajectories that are physically consistent and executable.

**Kinematic Retargeting.** These methods map human motion to robot configurations (Qin et al., 2022). While efficient and easy to compute, they often rely on specialized hardware (Xu et al., 2025), handcrafted motion primitives (Wu et al., 2024), and struggle with realism in contact-rich tasks (Qin et al., 2023). Being kinematic only, the generated motions are not compliant with physics constraints.

**Learning-Based Retargeting Networks.** Neural mapping approaches train networks to convert human motions into robot motions (Park et al., 2025; Yin et al., 2025b). Such models can outperform direct kinematic mappings and retain fast inference, but they require extensive pretraining and may fail when facing out-of-distribution motions or novel embodiments.

**Optimization-Based Retargeting.** Optimization-based approaches explicitly incorporate physics and contact constraints to ensure dynamical feasibility (Reda et al., 2023). They can generate high-quality, physically plausible motions, but often depend on detailed contact data (Lakshmipathy et al., 2025), specific data pipelines (Yang et al., 2025), and strong priors (Nakaoka et al., 2005). Due to the non-convex nature of the problem, sampling-based approaches (Yang et al., 2025; Si et al., 2025) have emerged as a promising solution.

**RL-Based Retargeting.** RL has been used to retarget human demonstrations across embodiments (Lum et al., 2025; Li et al., 2025c). When combined with curriculum learning (Mandi et al., 2025; Liu et al., 2024), RL can produce dexterous, physically feasible robot motions. However, it typically requires training on each trajectory and significant computation, which limits scalability to internet-scale data and real-time deployment.

Existing methods trade off between efficiency (kinematics, neural networks) and physical fidelity (optimization, RL). **SPIDER** combines the generality of RL with the efficiency of optimization-based pipelines, making it a scalable and practical drop-in replacement for human-to-robot transfer.

## 2.3 Sampling-based Optimization for Robot Control

Sampling-based optimization methods such as the cross-entropy method (De Boer et al., 2005), evolutionary algorithms (Salimans et al., 2017), and Bayesian optimization (Frazier, 2018) are powerful tools for solving non-convex and non-smooth problems. Due to their parallelizability and flexibility, these methods have been applied to navigation (Williams et al., 2016), legged locomotion (Xue et al., 2025), and dexterous manipulation (Howell et al., 2022; Li et al., 2025a). Despite their success in contact-rich control, they can suffer from instability and solution ambiguity in trajectory sampling (Kim et al., 2022). **SPIDER** addresses these challenges by guiding sampling with contact information, which helps preserve the intended contact sequence. n

3

<!-- page 4 of 35 -->

![Image block](doc:f3190a8/tier:basic/page:4/block:1)

Figure 2 Overview of the SPIDER pipeline. The pipeline takes reconstructed object meshes, reference robot motion, and object motion and converts them into a dynamically feasible robot trajectory with corrected contacts. The generated trajectory is further robustified and augmented before deployment or policy learning.

## 3 Physics-based Retargeting with Sampling

This section introduces our pipeline for converting human data to feasible robot trajectories, as illustrated in Figure 2. We first formulate the physics-based retargeting problem in Section 3.1, followed by its samplingbased solver design in Section 3.2. Then, we introduce different data augmentation methods in Section 3.3 for downstream applications including trajectory robustification for sim-to-real transfer, data augmentation for novel physics constraints and user intent preservation with contact guidance.

## 3.1 Physics-based Retargeting Problem

We formulate physics-based retargeting as a constrained optimization problem where a robot control sequence $u _ { 0 : T - 1 } \; \left( \mathrm { e . g . } \right.$ , joint position or torque commands) is optimized to minimize the distance to the reference trajectory $x _ { 0 : T } ^ { \mathrm { r e f } }$ and the control effort. As input, a kinematic reference state sequence $x _ { t } ^ { \mathrm { r e f } } , \forall t \in \{ 0 , \ldots , T \}$ is first provided from human demonstrations. The state $x _ { t } ^ { \mathrm { r e f } } = \{ q _ { t } ^ { \mathrm { r e f } } , v _ { t } ^ { \mathrm { r e f } } \}$ is composed of reference generalized coordinates $q _ { t } ^ { \mathrm { r e f } }$ and generalized velocities $v _ { t } ^ { \mathrm { r e f } }$ . Specifically, for robots with $n _ { u }$ joints, the coordinates $q _ { t } ^ { \mathrm { r e f } }$ comprise the robot joint positions $q _ { t } ^ { \mathrm { r e f , j o i n t } } \in \mathbb { R } ^ { n _ { u } }$ , the robot base transformation $X _ { t } ^ { \mathrm { r e f , b a s e } } \in \mathrm { S E } ( 3 )$ , and the object pose $X _ { t } ^ { \mathrm { r e f , o b j e c t } } \in \mathrm { S E } ( 3 )$ . Physics-based retargeting is performed offline: the output is a dynamically feasible open-loop control sequence $u _ { 0 : T - 1 } { \mathrm { ~ ( e . g . } }$ , the joint position commands used in this paper), valid under aligned initial conditions between the robot/object and the reference. Real-robot experiments demonstrate feasibility from such aligned starts. The optimization minimizes the distance to the reference trajectory and the control effort:

$$
\operatorname* { m i n } _ { u _ { 0 : T - 1 } } J ( u _ { 0 : T - 1 } ) = \operatorname* { m i n } _ { u _ { 0 : T - 1 } } \| x _ { T } \ominus x _ { T } ^ { \operatorname { r e f } } \| _ { Q _ { T } } ^ { 2 } + \sum _ { t = 0 } ^ { T - 1 } \left( \left\| x _ { t + 1 } \ominus x _ { t + 1 } ^ { \operatorname { r e f } } \right\| _ { Q _ { t } } ^ { 2 } + \| u _ { t } \| _ { R _ { t } } ^ { 2 } \right)\tag{1a}
$$

$$
\begin{aligned} { \operatorname { s . t . } \quad x _ { t + 1 } = f ( x _ { t } , u _ { t } , t ) \quad \forall t \in \{ 0 , \ldots , T - 1 \} } \\ \end{aligned}\tag{1b}
$$

$$
x _ { 0 : T } \in \mathcal { X } , \quad u _ { 0 : T - 1 } \in \mathcal { U }\tag{1c}
$$

where $x _ { 0 : T }$ is the optimized feasible state, $x _ { t + 1 } = f ( x _ { t } , u _ { t } , t )$ is the state transition function, $\mathcal { X }$ is the state space, U is the control input space, and $Q _ { t }$ and $R _ { t }$ are the state and control input weighting matrices. Because the generalized coordinates contain poses on the SE(3) manifold, standard subtraction is undefined. We use the operator ⊖ to denote the difference in generalized coordinates, which evaluates the error between variables in Euclidean space normally, while mapping differences between SE(3) transformations to the local 6D tangent space using the logarithmic map. In practice, we use diagonal weighting matrices for $Q _ { t }$ and $R _ { t }$ where $Q _ { t } = \mathrm { d i a g } ( \{ q _ { \mathrm { r o b o t } } , q _ { \mathrm { o b j e c t } } \} )$ and $R _ { t } = \mathrm { d i a g } ( \{ r _ { \mathrm { r o b o t } } , r _ { \mathrm { o b j e c t } } \} )$ .

## 3.2 Sampling for Physics-based Retargeting

Due to the contact-rich nature of physics-based retargeting, the optimization problem in Equation (1a) is highly non-convex and often discontinuous. Sampling-based optimization (Mannor et al., 2003) provides a natural way to handle such landscapes, as it does not rely on smoothness or convexity assumptions. Intuitively, this resembles RL in that both rely on sampled trajectories from parallelized simulation to guide decision-making, but instead of updating a policy network, we directly use the samples to optimize the control sequence

4

<!-- page 5 of 35 -->

![Image block](doc:f3190a8/tier:basic/page:5/block:1)

Figure 3 Sampling for Physics-based Retargeting on different robot embodiments. This figure shows the open-loop rollout of the retargeted control on different robot embodiments. The retargeting process corrects artifacts such as foot penetration and slipping, enabling the robot to execute highly dynamic motions. The method can be applied to both dexterous manipulation and humanoid whole-body control tasks.

$U = u _ { 0 : T - 1 }$ . To this end, we adopt a sampling-based optimizer with an annealed sampling kernel (Xue et al., 2025; Pan et al., 2024) as shown in Algorithm 1.

$$
U ^ { i + 1 } = U ^ { i } + \frac { \sum _ { j = 1 } ^ { N _ { W } } \operatorname { e x p } \left( - \frac { J ( U ^ { i } + [ W ] _ { j } ) } { \lambda } \right) [ W ] _ { j } } { \sum _ { j = 1 } ^ { N _ { W } } \operatorname { e x p } \left( - \frac { J ( U ^ { i } + [ W ] _ { j } ) } { \lambda } \right) } ,\tag{2}
$$

$$
\Sigma _ { h } ^ { i } = \operatorname { e x p } \left( - \frac { N - i } { \beta _ { 1 } N } - \frac { H - h } { \beta _ { 2 } H } \right) I ,\tag{3}
$$

where H is the planning horizon, λ is the temperature parameter, $U ^ { i }$ is the solution at iteration $i ,   i = 1 , \ldots , N$ is the iteration index with N being the total number of iterations, $[ W ] _ { j } \sim \mathcal { N } ( 0 , \Sigma _ { 0 : H - 1 } ) , j = 1 , \ldots , N _ { W }$ are $N _ { W }$ sampled Gaussian noise sequences with scheduled covariance $\Sigma _ { h } , h \in \{ 0 , \ldots , H - 1 \}$ , and $\beta _ { 1 } , \beta _ { 2 } \in ( 0 , 1 )$ are annealing parameters controlling sampling covariance in line 6 of Algorithm 1.

In contact-rich dexterous manipulation, the feasible solution set is typically narrow, so effective optimization requires a combination of coarse search to discover feasible contact modes and fine refinement to achieve stable contact. The annealed sampling covariance in Equation (2) implements this exploration-exploitation trade-off: $\beta _ { 1 }$ controls the rate across outer iterations (global-to-local search over optimization updates), and $\beta _ { 2 }$ controls the rate along the prediction horizon (allocating more or less perturbation across timesteps). As the schedule progresses, the effective sampling radius transitions from broad exploration to targeted exploitation around promising trajectories. To further improve speed, we adopt an early stopping strategy (Kamat et al., 2022), where optimization halts when improvements from further sampling become small.

## 3.3 Sampling for Physics-based Data Augmentation

Apart from generating feasible robot actions, another application of physics-based retargeting is the ability to systematically augment the retargeted data for trajectory robustness, data diversity and user intent preservation.

**Trajectory Robustification.** To bridge the gap from reconstructed demonstrations to real hardware, trajectories must be robust against unknown or misspecified dynamics (e.g., friction, contact compliance) and reconstruction

5

<!-- page 6 of 35 -->

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Sampling for Physics-based Retargeting
1: Input: $x _ { 0 : T } ^ { \mathrm { r e f } } ,   U ^ { 0 } ,   H ,   N _ { W } ,   \lambda ,   \beta _ { 1 } , \beta _ { 2 } ,   N ,   \epsilon _ { \mathrm { t o l } }$
2: $J _ { \mathrm { b e s t } } \gets \infty ,   i \gets 0$
3: while $i &lt; N$ and not converged do
4: $$i \gets i + 1 ;$ Update $\Sigma _ { h } ^ { i }$$ using Equation (3) ▷ Annealing schedule for sampling covariance
5: for $j \in \{ 1 , \ldots , N _ { W } \}$ do
6: Sample $[ \widetilde { W } ] _ { j } \sim \mathcal { N } ( 0 , \Sigma _ { 0 : H - 1 } ^ { i } ) ; [ U ^ { i } ] _ { j } \leftarrow U ^ { i } + [ W ] _ { j }$
7: Compute $J ( [ U ^ { i } ] _ { j } )$ by simulation rollout (Equation (1a))
8: end for
9: Update $U ^ { i + 1 }$ using weighted average from Equation (2)
10: if $| J _ { \operatorname { b e s t } } - \operatorname { m i n } _ { j } J ( U _ { j } ) | &lt; \epsilon _ { \operatorname { t o l } }$ then
11: break ▷ Early stop if improvement is small
12: end if
13: $\begin{array} { r } { J _ { \mathrm { b e s t } } \leftarrow \operatorname* { m i n } _ { j } J ( U _ { j } ) } \end{array}$
14: end while
15: Return $U ^ { * }$
</div>

noise (e.g., object mesh estimation error, pose estimation error) that could otherwise make a nominal plan infeasible. Our approach optimizes a control sequence with a pessimistic (min–max) objective over a bounded parameter set $\mathcal { D } ,$ similar to the domain randomization (DR) method (Tobin et al., 2017) but replacing expectation with worst-case cost to ensure universal feasibility:

$$
J _ { \mathrm { r o b } } ( U ) = \operatorname* { m a x } _ { d \in \mathcal { D } } J ( U , d ) ,\tag{4}
$$

where $J ( U , d )$ is the cost in Equation (1a) under dynamics parameter d. In practice, D spans variations such as contact margin size, friction coefficients, and object mass. During optimization, each candidate sequence $U _ { j }$ is rolled out over a mini-batch $d _ { 1 : K } \sim \mathcal { D }$ and evaluated by $\operatorname { m a x } _ { k \leq K } J ( U _ { j } , d _ { k } )$ , following robust sampling-based control (Williams et al., 2018).

**Geometric and Physics Augmentation.** To generate diverse data, one human demonstration can be augmented into a diverse set of physically feasible actions that could be used for policy learning (Mandlekar et al., 2023). **SPIDER** supports generating diverse geometric variations of the objects while still grounding them with physics. For dexterous manipulation tasks, we perturb the initial object pose and scale to generate diverse interaction behaviors while the reference motion is kept the same. We can even directly replace the object mesh to generate new interaction behaviors. For instance, in Figure 4, we adapt a grasping motion for a Coke bottle to a cat toy simply by replacing the object mesh. Similarly, the same idea can be applied to human-object interaction for humanoid robot tasks. By modifying the object geometry to make it smaller and lighter, the robot can automatically adapt to the new object by lowering its body and switching to a single-hand pushing motion with less force. One can also change the terrain to transform a flat-ground running motion into a stair-climbing motion, which is challenging for kinematic-only retargeting because climbing stairs requires new contact patterns. A key advantage of physics-based retargeting is its force-awareness. By applying external forces to the robot when it is pulling a heavy object, the sampling-based solver can generate different behaviors that respond to the force. For instance, if a large force (120 N and 240 N, respectively, in Figure 4) is applied to pull the robot back, the generated trajectory will lean forward and move more slowly.

**Human-preferred Contact Preservation.** Due to the multimodal nature of dexterous manipulation, solution ambiguity could lead to unexpected contact behaviors that do not match the reference motion, as illustrated in Figure 5. For gradient-based optimization methods, one can specify the contact mode in dynamics constraints to enforce the desired contact mode. Inspired by this idea, we apply a virtual constraint between intended contact points on the object and robot fingers, “sticking” the object to the target configuration in early stages and gradually relaxing this constraint in a curriculum-like fashion. For the k-th object-hand contact pair (e.g., handle-thumb), we define the current relative position as robot ${ } ^ { t } p _ { k , t } ^ { \mathrm { o b j e c t } } = \left. p _ { k , t } ^ { \mathrm { r o b o t } } \right. ^ { \circ } - \left. p _ { k , t } ^ { \mathrm { o b j e c t } } \right.$ and the desired reference position as robot $\begin{array} { r } { ^ { ! } p _ { k , t } ^ { \mathrm { o b j e c t , r e f } }   =   p _ { k , t } ^ { \mathrm { r o b o t , r e f } } - p _ { k , t } ^ { \mathrm { o b j e c t , r e f } } } \end{array}$ The constraint is activated when the contact indicator $c _ { k , t } = 1$ , which occurs when the reference relative position is within the contact threshold: $c _ { k , t } = \mathbf { 1 } ( \lVert ^ { \mathrm { r o b o t } } p _ { k , t } ^ { \mathrm { o b j e c t , r e f } } \rVert _ { 2 } \leq \epsilon _ { \mathrm { c o n t a c t } } )$ . The constraint strength is controlled by a penalty parameter $\eta _ { i } \to \infty$

6

<!-- page 7 of 35 -->

![Image block](doc:f3190a8/tier:basic/page:7/block:1)

Figure 4 Geometric and physics augmentation. We augment the retargeted data from a single demonstration into a diverse set of physically feasible actions. Here we demonstrate (a) generating motion with a new object mesh for dexterous manipulation, (b) moving a lighter and smaller object for humanoid robot tasks, (c) adding stairs to the scene for humanoid running motion, and (d) applying external forces to the robot when it is pulling a heavy object.

![Image block](doc:f3190a8/tier:basic/page:7/block:3)

Figure 5 Human-preferred contact preservation. Due to a poor initial reference, the solution may converge to a different contact mode than the human demonstration. Contact guidance corrects this behavior by making the target contact mode easier to sample. With a virtual force applied between the object and the hand, the object is more likely to move towards the target contact region, guiding the sampling towards the desired contact mode.

when $i \rightarrow N ;$

$$
c _ { k , t } \| { } ^ { \mathrm { r o b o t } } p _ { k , t } ^ { \mathrm { o b j e c t } } - { } ^ { \mathrm { r o b o t } } p _ { k , t } ^ { \mathrm { o b j e c t , r e f } } \| _ { 2 } ^ { 2 } \leq \eta _ { i }\tag{5}
$$

The implementation details are described in Section C. Compared to previous methods (Mandi et al., 2025), our formulation focuses on relative hand–object relationships instead of their absolute positions. The method requires annotations of the intended contact points and labels, which we detect automatically using the simulator’s collision detection on kinematic data.

## 4 Experiments

This section evaluates **SPIDER** across four applications: retargeting human demonstrations to feasible robot trajectories, downstream policy training, physical-robot deployment, and large-scale dataset generation. We report retargeting performance separately for dexterous manipulation and humanoid whole-body control in Section 4.1, then evaluate policy learning for each embodiment in Section 4.2. To verify the generated trajectory feasibility, we apply **SPIDER** to in-the-wild data and deploy the resulting motions on physical robots in Section 4.3. Finally, we apply **SPIDER** to retarget common hand-object interaction datasets at scale and report its performance in Section 4.4. Common failure cases and limitations are discussed in Section 4.5. Throughout, we adopt unified notation for tracking errors: $e _ { t }$ and $e _ { r }$ denote translation and rotation errors, $e _ { j }$ denotes joint angle error, $e _ { f t }$ denotes fingertip position error, and $e _ { \mathrm { a r t i } }$ denotes articulation joint angle error for articulated objects. Full implementation details including datasets, robot embodiments, simulator settings, and hyperparameters are provided in Section A.

7

<!-- page 8 of 35 -->

Table 1 Dexterous manipulation tracking. Comparisons align each RL or kinematic baseline’s setting. ADD-AUC evaluates object-pose tracking from 0 to 10 cm; “mixed” aggregates five hands. FPS uses one NVIDIA L40S; full results are in Section B.

<table><tr><td>Method</td><td>Simulator    Dataset   Hand</td><td colspan=2>AUC (%) ↑ FPS ↑</td></tr><tr><td rowspan=2>ManipTransTR  SPIDER</td><td>Isaac Gym   Oakink    Inspire</td><td colspan=2>19.4 ± 27.5   0.1</td></tr><tr><td>Isaac Gym   Oakink    Inspire</td><td colspan=2>22.3 ± 28.1  2.5</td></tr><tr><td rowspan=2>S.  DexMachinaSPIDER</td><td>Genesis     ARCTIC    Inspire</td><td colspan=2>41.0 ± 25.3  0.05</td></tr><tr><td>Genesis     ARCTIC    Inspire</td><td colspan=2>43.6 ± 29.8  1.5</td></tr><tr><td rowspan=3>KinematicSPIDERn.  SPIDER+C</td><td>MuJoCo      Oakink    mixed</td><td colspan=2>11.0 ± 11.1  248.9</td></tr><tr><td>MuJoCoWarp Oakink    mixed</td><td>48.8 ± 24.9</td><td>6.1</td></tr><tr><td>MuJoCoWarp Oakink    mixed</td><td> $\mathbf { 5 6 . 9 \pm 2 4 . 7 }$ </td><td>0.4</td></tr><tr><td rowspan=3>S.  KinematicSPIDERSPIDER+C</td><td>MuJoCo      GigaHands mixed</td><td>24.1 ± 22.3</td><td>244.2</td></tr><tr><td>MuJoCo Warp GigaHands mixed</td><td>56.9 ± 20.6</td><td>5.9</td></tr><tr><td>MuJoCoWarp GigaHands mixed</td><td>57.7± 23.9</td><td>0.5</td></tr></table>

Table 2 Humanoid whole-body tracking. Open-loop robot and object tracking. Robot errors are joint (degrees), pelvis translation (cm), and orientation (degrees); object errors apply only to OMOMO (cm/degrees). FPS is trajectory-generation throughput.

| Data Method | Robot Object e<sub>j</sub> ↓ e<sub>t</sub> ↓ e<sub>r</sub> ↓ e<sub>t</sub> ↓ e<sub>r</sub> ↓ FPS ↑ |
| --- | --- |
| GMRLAFAN1SPIDER | 1.08 2.01 2.40 - - 35.20.58 0.11 0.07 - - 23.1 |
| GMR AMASS SPIDER | 6.2 4.1 18.7 - - 37.20.75 0.23 0.08 - - 22.0 |
| GMR OMOMO SPIDER | 0.57 1.43 2.28 1.18 1.68 28.00.83 0.20 0.17 0.18 0.06 19.6 |

## 4.1 Retargeting Performance

## 4.1.1 Dexterous Manipulation

**Baselines.** We compare against two RL-based physics retargeting methods, **ManipTrans** (Li et al., 2025c) and **DexMachina** (Mandi et al., 2025), and a **Kinematic** retargeting baseline (fingertip-based IK, kinematically feasible but dynamically infeasible). Since RL-based methods train per-trajectory policies whereas **SPIDER** directly optimizes control sequences, this comparison serves as a performance reference. To isolate the effect of retargeting from confounds, each comparison aligns the simulator, dataset, hand, and reward setting with the corresponding baseline: we evaluate against **ManipTrans** on Oakink in Isaac Gym and against **DexMachina** on ARCTIC in Genesis, both using the 12-DoF Inspire hand, and additionally align the nominal base controller with **ManipTrans**’s universal base imitator; the kinematic comparison is run in MuJoCo. Full protocol details are in Section B.

**Metrics.** We report two metrics. ADD-AUC measures object tracking quality without committing to a single success threshold: we sample keypoints on the object mesh, transform them by both the achieved and the reference object poses at every frame, and take the area under the per-frame success-rate curve as the mean keypoint distance threshold sweeps from 0 to 10 cm (normalized so the metric lies in [0, 100]%). Because it is computed on transformed mesh keypoints, it captures both translational and rotational error in a single number; the full definition is in Section A. We use ADD-AUC rather than a thresholded success rate because success is defined differently across baseline papers and neither has released a public evaluation protocol. FPS measures retargeting throughput, defined as the total number of retargeted frames divided by the total retargeting wall-clock time (higher is better).

**Comparison Results.** Table 1 aggregates all dexterous-hand comparisons. Against the RL-based baselines, **SPIDER** achieves comparable or better ADD-AUC while generating trajectories over an order of magnitude faster, since it optimizes control sequences with a parallel simulator instead of training a policy per trajectory. Against **Kinematic** retargeting, **SPIDER** more than doubles the ADD-AUC, and contact guidance (**SPIDER+C**, with contacts detected by MuJoCo’s contact detector) improves it further. The high variance is because results are aggregated across tasks of varying difficulty; detailed per-task, per-hand tracking performance is reported in Section B.

## 4.1.2 Humanoid Whole-Body Control

**Baselines and Reference Methods.** For humanoid whole-body loco-manipulation tasks, there is no existing physics-based retargeting method since the kinematic motion is directly used as the reference motion for RL training. Thus, we compare **SPIDER** with the state-of-the-art kinematics-based retargeting method general motion retargeting (**GMR**) (Araujo et al., 2025), where we report the open-loop rollout motion tracking error. Since humanoid robots are unstable systems, a stabilizer policy is required to track the retargeted motion, where reference motion quality is crucial. For LAFAN1, we use representative dance and stand-up sequences;

8

<!-- page 9 of 35 -->

![Image block](doc:f3190a8/tier:basic/page:9/block:1)

Figure 6 Dexterous grasping policy training. With identical hyperparameters over seven seeds, the SPIDER reference yields fast, consistent task completion; the kinematic reference contains physical artifacts and fails. Yellow shows learned motion and green shows the reference.

![Image block](doc:f3190a8/tier:basic/page:9/block:3)

Figure 7 Humanoid loco-manipulation policy training. The original reference misses box contact, whereas SPIDER corrects contact and supplies feasible feedforward control. The resulting policy converges faster and completes more of the target motion.

for AMASS, we randomly sample eight motions; and for OMOMO, we evaluate an object-transport trajectory. The table reports averages over every frame and sequence in each dataset. Robot translation and rotation errors measure pelvis tracking, while the object columns measure pose tracking for the OMOMO interaction.

**Comparison Against Kinematics-Based Baselines.** As summarized in Table 2, **SPIDER** achieves lower open-loop tracking error thanks to physics grounding with minimal speed degradation, including significantly better object tracking on the OMOMO dataset.

## 4.2 Policy Training

## 4.2.1 Dexterous Hand Policy Training

**SPIDER** provides an already-feasible reference motion, making the policy’s tracking problem substantially easier. We evaluate a tabletop cup-grasping task with a fixed-base 7-DoF right arm and a 22-DoF, five-fingered hand. The policy outputs 6D wrist and 22D finger target deltas at 20 Hz and is trained with PPO in 4096 parallel environments using only imitation rewards for object pose, wrist pose, finger joints, and hand landmarks. The reference comes from an in-the-wild human pouring demonstration reconstructed by the pipeline in Section 4.3. Occlusion causes the kinematic reference to let the cup drift out of the hand; **SPIDER** removes this artifact before training. Over seven seeds with identical hyperparameters, the **SPIDER**-reference policy completes the task (0.99 ± 0.01 task progress), whereas the kinematic-reference policy stalls (0.07 ± 0.11), as shown

9

<!-- page 10 of 35 -->

![Image block](doc:f3190a8/tier:basic/page:10/block:1)

Figure 8 Open-loop rollout with robustification. Left: showing single-arm tasks including pouring from a cup, picking a spoon, and rotating a light bulb. Right: showing dual-arm tasks including scooping a bowl, unplugging a charger, and playing a guitar. The retargeted trajectories are executed directly on the physical robot.

![Image block](doc:f3190a8/tier:basic/page:10/block:3)

Figure 9 Retargeting from a Single RGB Camera. Starting from a single RGB observation of a user manipulating an object, we first reconstruct the 3D scene with hand and object meshes, which often contain reconstruction artifacts and penetrations. Our method then retargets this noisy data into a physically plausible robot trajectory with corrected contacts. Finally, the retargeted trajectory is successfully executed on the physical robot, demonstrating the pipeline’s robustness to real-world data quality issues.

in Figure 6. Full action, observation, reward, reset, and PPO settings are given in Section A.6.

## 4.2.2 Humanoid Policy Training

We evaluate a whole-body loco-manipulation policy that must pick up a box, transport it, and place it on the floor using the framework of Weng et al. (2025). We train PPO with body and object tracking rewards plus an action-rate penalty, while removing contact rewards and auxiliary regularizers so the comparison isolates reference-motion quality. The original kinematic motion is offset from the object and misses the grasp contact; the **SPIDER** version corrects the contact and provides a dynamically feasible nominal control. Under identical training settings, the kinematic-reference policy learns body tracking but fails to grasp the object, while the **SPIDER**-reference policy converges faster and achieves substantially better object tracking (Figure 7). Humanoid reward and termination details are provided in Section A.5.

## 4.3 Deployment Results

**Replay Generated Motion on Physical Robots.** To demonstrate the dynamical feasibility of our retargeted trajectories, we deploy **SPIDER** on a physical system consisting of a 7-DoF Franka Emika Panda with an Allegro hand, using the robustification method in Section 3.3. The arm is controlled using operational space control (OSC), while the hand directly executes the optimized joint actions from our method. We evaluate trajectory feasibility on four dexterous manipulation tasks—rotating a light bulb, manipulating a small spoon, playing a guitar, and unplugging a charger—each demanding precise finger coordination and showcasing the practical applicability of our retargeting approach. Representative execution sequences are shown in Figure 8. We note an inherent limitation of this open-loop execution: while the robustification and randomization in Section 3.3 make the trajectory tolerant to moderate perturbations and model mismatch, they cannot compensate for large deviations in the object’s initial position. If the object is placed significantly off from the reference initial pose, the open-loop rollout has no feedback mechanism to recover, and the execution fails;

10

<!-- page 11 of 35 -->

![Image block](doc:f3190a8/tier:basic/page:11/block:1)

Figure 10 Dataset trajectory visualization. We apply SPIDER to human demonstrations and retain 6,520 simulated robot trajectories spanning 191 dataset-specific object identities.

Table 3 Large-scale dataset generation with SPIDER. We retarget human demonstrations to four robotic hands. ADD-AUC (%, higher is better) averages all completed seed-0 rollouts, including failed grasps. Bold shaded cells identify the best hand for each dataset; the aggregated row weights trajectories equally.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Human Data</td><td colspan="3">ADD-AUC (%)↑</td><td rowspan="2">Sharpa</td></tr><tr><td># Demos</td><td># Objects</td><td>Allegro</td><td>XHand</td><td>Inspire</td></tr><tr><td>DexYCB [4]</td><td>499</td><td>20</td><td>55.70</td><td>53.21</td><td>54.47</td><td>45.53</td></tr><tr><td>OakInk-Image [55]</td><td>169</td><td>66</td><td>59.17</td><td>58.07</td><td>59.14</td><td>56.94</td></tr><tr><td>HRDexDB [ [22]</td><td>441</td><td>94</td><td>64.21</td><td>64.44</td><td>63.76</td><td>62.14</td></tr><tr><td>H0T3D [2]</td><td>2,718</td><td>33</td><td>71.16</td><td>70.09</td><td>69.41</td><td>40.74</td></tr><tr><td>Aggregated</td><td>3,827</td><td>213</td><td>67.86</td><td>66.76</td><td>66.41</td><td>44.46</td></tr></table>

closing the loop with a learned tracking policy (e.g., Section A.5) is the natural remedy.

**Using In-the-Wild Data.** To demonstrate the flexibility of **SPIDER** for upstreaming data, we use the pipeline to convert video from a single RGB camera into executable robot trajectories. The pipeline consists of: (1) 3D mesh reconstruction with Trellis (Xiang et al., 2025), (2) hand pose estimation with HAMER (Pavlakos et al., 2024) to obtain MANO parameters, and (3) object pose tracking with FoundationPose (Wen et al., 2024) for 6D trajectories. Due to occlusion and limited resolution, the human and object motions are noisier than those in the dataset, which requires more robust physics grounding. As shown in Figure 9, we validate on single-hand manipulation tasks like pouring with a cup and installing a bolt. Real-world noise, artifacts, and penetrations are corrected through physics grounding in **SPIDER**, and the retargeted trajectories are directly executable on physical robots. One limitation is that, since it is an open-loop rollout, it can only handle a narrow range of initial states and disturbances. We leave the closed-loop control for future work.

## 4.4 Large Scale Data Generation

**Source Data Processing.** We aggregate 3,827 right-hand demonstrations from DexYCB (Chao et al., 2021), OakInk-Image (Yang et al., 2022), HRDexDB (Lim et al., 2026), and HOT3D (Banerjee et al., 2025), covering 213 dataset-specific object identities (Table 3). The key processing step is to identify hand–object interaction intervals and select the contact-establishment segments for retargeting, retaining a short approach to initialize the hand before contact.

11

<!-- page 12 of 35 -->

**Generation Method.** We apply **SPIDER** without contact guidance to four hands: Allegro, XHand, Inspire, and Sharpa, since noisy demonstrations can make contact signals unreliable. We use shared optimization hyperparameters across datasets and hands, with dataset-specific reference processing described in Section A.4.

**Generation Scale and Quality.** With the shared optimization recipe, the completed rollouts achieve 61.48% ADD-AUC, evaluated using the same surface-point metric as Section 4.1.1. Allegro achieves the highest aggregated ADD-AUC, while XHand performs best on HRDexDB; performance varies across hands and datasets. Example trajectories can be found in Figure 10.

## 4.5 Limitations and Common Failure Cases

**SPIDER** has several failure modes. (1) **Perception errors**: inaccurate upstream reconstruction (**HAMER** (Pavlakos et al., 2024), **FoundationPose** (Wen et al., 2024), **Trellis** (Xiang et al., 2025)) can introduce erroneous reference trajectories; large misalignments may cause the optimizer to converge to a semantically incorrect contact mode despite physical consistency. (2) **Severe occlusion**: heavy occlusion produces large temporal gaps in the reference, degrading retargeting quality; temporal consistency filtering is a promising mitigation. (3) **Open-loop sensitivity**: the generated trajectory is valid only near the aligned initial state—any initial pose mismatch leads to growing execution errors; closed-loop policy distillation is left as future work. (4) **Contact mode ambiguity**: without contact guidance, sampling may converge to an alternative valid grasp; **SPIDER+C** resolves most such cases at the cost of contact annotations.

## 5 Conclusion and Future Work

This paper introduces **SPIDER**, a general, flexible and efficient physics-based retargeting framework that enables large-scale data transfer from human data. Our framework achieves competitive performance compared to state-of-the-art methods while being an order of magnitude faster. Despite its generality, **SPIDER**’s performance depends on the quality of reconstructed 3D human-object interaction data; noisy meshes and motion can yield degraded trajectories. Future work includes improving the robustness of the framework to noisy data and distilling the framework output into a policy for closed-loop control.

12

<!-- page 13 of 35 -->

## References

Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, and C. Karen Liu. Retargeting matters: General motion retargeting for humanoid motion tracking. arXiv:2510.02252, 2025.

Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan. HOT3D: Hand and object tracking in 3D from egocentric multi-view videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. URL [https://facebookresearch.github.io/hot3d/](https://facebookresearch.github.io/hot3d/).

Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In ECCV, 2024.

Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, and Dieter Fox. DexYCB: A benchmark for capturing hand grasping of objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. URL [https://dex-ycb.github.io/](https://dex-ycb.github.io/).

Pieter-Tjerk De Boer, Dirk P. Kroese, Shie Mannor, and Reuven Y. Rubinstein. A Tutorial on the Cross-Entropy Method. Annals of Operations Research, 2005.

Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, and Otmar Hilliges. ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation, April 2023.

Peter I. Frazier. A Tutorial on Bayesian Optimization. arXiv:1807.02811, 2018.

Rao Fu, Dingxi Zhang, Alex Jiang, Wanjia Fu, Austin Funk, Daniel Ritchie, and Srinath Sridhar. GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities. In CVPR, 2025.

Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4D: Reconstructing and Tracking Humans with Transformers. In ICCV, 2023.

Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. Robust motion in-betweening. In SIGGRAPH, 2020.

Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video. arXiv:2505.11709, 2025.

Taylor Howell, Nimrod Gileadi, Saran Tunyasuvunakool, Kevin Zakka, Tom Erez, and Yuval Tassa. Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo. arXiv:2212.00541, 2022.

Jay Kamat, Joaquim Ortiz-Haro, Marc Toussaint, Florian T. Pokorny, and Andreas Orthey. BITKOMO: Combining Sampling and Optimization for Fast Convergence in Optimal Motion Planning. In IROS, 2022.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-Language-Action Model. In CoRL, 2024.

Taekyung Kim, Gyuhyun Park, Kiho Kwak, Jihwan Bae, and Wonsuk Lee. Smooth Model Predictive Path Integral Control without Smoothing. RA-L, 2022.

Arjun S. Lakshmipathy, Jessica K. Hodgins, and Nancy S. Pollard. Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations. ACM Transactions on Graphics, 2025.

Albert H. Li, Preston Culbertson, Vince Kurtz, and Aaron D. Ames. DROP: Dexterous Reorientation via Online Planning. In ICRA, 2025a.

Jialong Li, Xuxin Cheng, Tianshu Huang, Shiqi Yang, Ri-Zhao Qiu, and Xiaolong Wang. AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control. In RSS, 2025b.

Jiaman Li, Jiajun Wu, and C Karen Liu. Object motion guided human motion synthesis. ACM Transactions on Graphics, 2023.

Jinhan Li, Yifeng Zhu, Yuqi Xie, Zhenyu Jiang, Mingyo Seo, Georgios Pavlakos, and Yuke Zhu. OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation. In CoRL, 2024.

13

<!-- page 14 of 35 -->

Kailin Li, Puhao Li, Tengyu Liu, Yuyang Li, and Siyuan Huang. ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning. In CVPR, 2025c.

Jongbin Lim, Taeyun Ha, Mingi Choi, Jisoo Kim, Byungjun Kim, Subin Jeon, and Hanbyul Joo. HRDexDB: A paired human-robot dataset for cross-embodiment dexterous grasping. arXiv preprint arXiv:2604.14944, 2026. URL [https://arxiv.org/abs/2604.14944](https://arxiv.org/abs/2604.14944).

Vincent Liu, Ademi Adeniji, Haotian Zhan, Siddhant Haldar, Raunaq Bhirangi, Pieter Abbeel, and Lerrel Pinto. EgoZero: Robot Learning from Smart Glasses. arXiv:2505.20290, 2025.

Xueyi Liu, Kangbo Lyu, Jieqiong Zhang, Tao Du, and Li Yi. QuasiSim: Parameterized Quasi-Physical Simulators for Dexterous Manipulations Transfer. In ECCV, 2024.

Tyler Ga Wei Lum, Olivia Y. Lee, C. Karen Liu, and Jeannette Bohg. Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration. In CoRL, 2025.

Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. AMASS: Archive of Motion Capture as Surface Shapes. In ICCV, 2019.

Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, and Shuran Song. DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation. arXiv:2505.24853, 2025.

Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. arXiv:2310.17596, 2023.

Shie Mannor, Reuven Rubinstein, and Yohai Gat. The Cross Entropy Method for Fast Policy Search. In ICML, 2003.

Shinichiro Nakaoka, Atsushi Nakazawa, Fumio Kanehiro, Kenji Kaneko, Mitsuharu Morisawa, and Katsushi Ikeuchi. Task model of lower body motion for a biped humanoid robot to imitate human dances. In IROS, 2005.

Chaoyi Pan, Zeji Yi, Guanya Shi, and Guannan Qu. Model-Based Diffusion for Trajectory Optimization. In NeurIPS, 2024.

Georgios Papagiannis, Norman Di Palo, Pietro Vitiello, and Edward Johns. R+X: Retrieval and Execution from Everyday Human Videos. In ICRA, 2025.

Sungjae Park, Homanga Bharadhwaj, and Shubham Tulsiani. DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy. arXiv:2506.20668, 2025.

Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing Hands in 3D with Transformers. In CVPR, 2024.

Haozhi Qi, Brent Yi, Sudharshan Suresh, Mike Lambeta, Yi Ma, Roberto Calandra, and Jitendra Malik. General in-hand object rotation with vision and touch. In CoRL, 2023.

Yuzhe Qin, Hao Su, and Xiaolong Wang. From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation. In IROS, 2022.

Yuzhe Qin, Wei Yang, Binghao Huang, Karl Van Wyk, Hao Su, Xiaolong Wang, Yu-Wei Chao, and Dieter Fox. AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System. In RSS, 2023.

Daniele Reda, Jungdam Won, Yuting Ye, Michiel van de Panne, and Alexander Winkler. Physics-based Motion Retargeting from Sparse Inputs. arXiv:2307.01938, 2023.

Juntao Ren, Priya Sundaresan, Dorsa Sadigh, Sanjiban Choudhury, and Jeannette Bohg. Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning. Computer Graphics and Interactive Techniques, 2025.

Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864, 2017.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. 1707.06347, 2017.

Kenneth Shaw, Shikhar Bahl, and Deepak Pathak. VideoDex: Learning Dexterity from Internet Videos. In CoRL, 2022.

14

<!-- page 15 of 35 -->

Zilin Si, Jose Enrique Chen, M. Emre Karagozler, Antonia Bronars, Jonathan Hutchinson, Thomas Lampe, Nimrod Gileadi, Taylor Howell, Stefano Saliceti, Lukasz Barczyk, Ilan Olivarez Correa, Tom Erez, Mohit Shridhar, Murilo Fernandes Martins, Konstantinos Bousmalis, Nicolas Heess, Francesco Nori, and Maria Bauza. Exostart: Efficient learning for dexterous manipulation with sensorized exoskeleton demonstrations. 2025. URL [https://arxiv.org/abs/2506.11775](https://arxiv.org/abs/2506.11775).

RDT Team. Rdt2: Enabling zero-shot cross-embodiment generalization by scaling up umi data, 2025. URL [https://github.com/thu-ml/RDT2](https://github.com/thu-ml/RDT2).

Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. In IROS, 2017.

Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar. MimicPlay: Long-Horizon Imitation Learning by Watching Human Play. In CoRL, 2023.

Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects. In CVPR, 2024.

Haoyang Weng, Yitang Li, Nikhil Sobanbabu, Zihan Wang, Zhengyi Luo, Tairan He, Deva Ramanan, and Guanya Shi. HDMI: Learning interactive humanoid whole-body control from human videos. arXiv:2509.16757, 2025.

Grady Williams, Paul Drews, Brian Goldfain, James M. Rehg, and Evangelos A. Theodorou. Aggressive driving with model predictive path integral control. In ICRA, 2016.

Grady Williams, Brian Goldfain, Paul Drews, Kamil Saigol, James Rehg, and Evangelos Theodorou. Robust Sampling Based Model Predictive Control with Sparse Objective Information. In RSS, 2018.

Albert Wu, Ruocheng Wang, Sirui Chen, Clemens Eppner, and C. Karen Liu. One-Shot Transfer of Long-Horizon Extrinsic Manipulation Through Contact Retargeting. arXiv:2404.07468, 2024.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3D Latents for Scalable and Versatile 3D Generation. In CVPR, 2025.

Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Xu, Linxi Fan, Manuela Veloso, and Shuran Song. DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation. In CoRL, 2025.

Haoru Xue, Chaoyi Pan, Zeji Yi, Guannan Qu, and Guanya Shi. Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing. In ICRA, 2025.

Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, and Cewu Lu. OakInk: A large-scale knowledge repository for understanding hand-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20953–20962, 2022. URL [https://openaccess.thecvf.com/content/CVPR2022/html/Yang\_OakInk\_A\_Large-Scale\_Knowledge\_Repository\_for\_Understanding\_Hand-Object\_Interaction\_CVPR\_2022\_paper.html](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_OakInk_A_Large-Scale_Knowledge_Repository_for_Understanding_Hand-Object_Interaction_CVPR_2022_paper.html).

Lujie Yang, H. J. Terry Suh, Tong Zhao, Bernhard Paus Graesdal, Tarik Kelestemur, Jiuguang Wang, Tao Pang, and Russ Tedrake. Physics-Driven Data Generation for Contact-Rich Manipulation via Trajectory Optimization. In RSS, 2025.

Zhao-Heng Yin, Changhao Wang, Luis Pineda, Krishna Bodduluri, Tingfan Wu, Pieter Abbeel, and Mustafa Mukadam. Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm. In IROS, 2025a.

Zhao-Heng Yin, Changhao Wang, Luis Pineda, Francois Hogan, Krishna Bodduluri, Akash Sharma, Patrick Lancaster, Ishita Prasad, Mrinal Kalakrishnan, Jitendra Malik, Mike Lambeta, Tingfan Wu, Pieter Abbeel, and Mustafa Mukadam. DexterityGen: Foundation Controller for Unprecedented Dexterity. In RSS, 2025b.

Yanjie Ze, Zixuan Chen, João Pedro Araújo, Zi-ang Cao, Xue Bin Peng, Jiajun Wu, and C. Karen Liu. TWIST: Teleoperated Whole-Body Imitation System. In CoRL, 2025.

Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu. OAKINK2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion. In CVPR, 2024.

Han Zhang, Songbo Hu, Zhecheng Yuan, and Huazhe Xu. DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove. In RSS, 2025.

Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Zhang Chen, Tianrui Guan, Fanlian Zeng, Ka Num Lui, Yuyao Ye, Yitao Liang, Yaodong Yang, and Yuanpei Chen. DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping. arXiv:2502.20900, 2025.

15

<!-- page 16 of 35 -->

## Appendix

## A Implementation Details

## A.1 Experimental Setup

**Dataset Selection.** For dexterous manipulation tasks, we evaluate on 3 bimanual manipulation datasets: GigaHands (Fu et al., 2025), Oakink (Zhan et al., 2024) and ARCTIC (Li et al., 2023). For humanoid tasks, we choose the commonly used LAFAN1 (Harvey et al., 2020) and AMASS (Mahmood et al., 2019) datasets for locomotion and the OMOMO (Li et al., 2023) dataset for human-object interaction. To the best of our knowledge, **SPIDER is the first universal retargeting method that can handle both dexterous manipulation and humanoid whole-body control tasks at this scale.**

**Robot Embodiments.** For dexterous manipulation, we evaluate across 5 different robotic hands to demonstrate the generalizability of our method: Allegro, XHand, Inspire, Ability, and Schunk. On humanoid whole-body control tasks, we use Unitree G1, Unitree H1-2, Fourier N1, and Booster T1 as the target embodiments These platforms exhibit significant variations in degrees of freedom, dimensions, and finger configurations (see Table 4), showcasing our method’s cross-embodiment capabilities. Figure 11 shows some example trajectories generated by **SPIDER** on different hands.

![Image block](doc:f3190a8/tier:basic/page:16/block:6)

![Image block](doc:f3190a8/tier:basic/page:16/block:7)

Figure 11 Retargeted dexterous manipulation trajectories on different hands. Starting with an infeasible motion with penetration and infeasible contact, SPIDER grounds it with physics. Different hands exhibit different behaviors with the same sampling parameters.

**Evaluation Metrics.** In our own ablation study where we validate the effect of our annealing and contact guidance design, we report tracking performance on the robot and object motion. For object tracking error, we compute per-step averaged rotation error $\begin{array} { r } { E _ { \mathrm { r o t } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } } \end{array}$ arccos $\left( 2 \langle q _ { \mathrm { o b j } , t } , q _ { \mathrm { o b j } , t } ^ { \mathrm { r e f } } \rangle ^ { 2 } - 1 \right)$ and position error $\begin{array} { r } { E _ { \mathrm { p o s } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| p _ { \mathrm { o b j } , t } - p _ { \mathrm { o b j } , t } ^ { \mathrm { r e f } } \| _ { 2 } } \end{array}$ , where T is the number of timesteps, $\langle \cdot , \cdot \rangle$ is the dot product, and $q _ { \mathrm { o b j } , t }$ and $p _ { \mathrm { o b j } , t }$ denote the object quaternion and position at timestep t, respectively. We define success only on the object motion since the goal is to move the object rather than to precisely track the robot hand. Task success is defined using: (1) the rotation error $E _ { \mathrm { r o t } } < 0 . 5$ rad, and (2) the average per-step translation error $E _ { \mathrm { p o s } } < 0 . 1$ m. Since a single success threshold is sensitive to its choice, we additionally report ADD-AUC10, following the ADD-AUC protocol (Mandi et al., 2025): we sample 512 keypoints on the object mesh surface, transform

16

<!-- page 17 of 35 -->

![Image block](doc:f3190a8/tier:basic/page:17/block:1)

Figure 12 Specifications of robots used in evaluation. SPIDER supports both dexterous hands and humanoid robots. The significant variations in DoF, dimensions, and finger count demonstrate the cross-embodiment generalizability of our approach. We employ a simulated 12-DoF configuration (Li et al., 2025c) of the Inspire and Ability hands, removing the joint constraints compared to their real-world versions.

| Hand | DOF | Width (cm) | Length (cm) | Fingers |
| --- | --- | --- | --- | --- |
| Allegro | 16 | 12.0 | 26.0 | 4 |
| XHAND | 12 | 9.2 | 19.5 | 5 |
| Inspire | 12 | 8.8 | 15.6 | 5 |
| Ability | 12 | 7.5 | 14.6 | 5 |
| Schunk SVH | 20 | 9.2 | 24.2 | 5 |

Table 4 Specifications of robotic hands used in evaluation. The significant variations in DoF, dimensions, and finger count demonstrate the cross-embodiment generalizability of our approach. We employ a simulated 12-DoF configuration (Li et al., 2025c) of the Inspire and Ability hands, removing the joint constraints compared to their real-world versions.

them by both the achieved and the reference object poses at every frame, and compute the mean per-keypoint L2 distance (the classic ADD metric for rigid bodies). The metric (denoted ADD-AUC, or ADD-AUC10 to make the 10 cm range explicit) is the area under the per-frame success-rate curve as the distance threshold sweeps from 0 to 10 cm, with the threshold axis normalized so the metric lies in [0, 100]% (higher is better). This threshold-free metric summarizes tracking quality without committing to a single cutoff and is sensitive to both translational and rotational errors through the mesh keypoints. For humanoid locomotion, the joint tracking error, pelvis position error and pelvis orientation error are computed.

When comparing with baselines, we use the same evaluation metrics in the original papers to make a fair comparison. Specifically, we follow the same evaluation setting as **ManipTrans** (Li et al., 2025c) and **DexMachina** (Mandi et al., 2025) with their published evaluation protocol.

## A.2 Data Preprocessing

We extract a 21D keypoint representation per hand using 3D fingertip positions and 6D wrist pose from MANO. An IK solver maps these keypoints to robot-specific joint positions by minimizing $\begin{array} { r } { \mathcal { L } _ { \mathrm { I K } } \stackrel { \cdot } { = } \sum _ { i = 1 } ^ { n } \| \mathbf { p } _ { i } ^ { \mathrm { r o b o t } } - } \end{array}$ $\| \mathbf { p } _ { i } ^ { \mathrm { h u m a n } } \| ^ { 2 } + 0 . 1 \| \widetilde { \mathbf { R } _ { \mathrm { w r i s t } } ^ { \mathrm { r o b o t } } } - \widetilde { \mathbf { R } _ { \mathrm { w r i s t } } ^ { \mathrm { h u m a n } } } \| _ { F } ^ { 2 }$ . Trajectories are resampled at 50 Hz and low-pass filtered at 10 Hz.

## A.3 Hyperparameters and Setup

The settings below describe the retargeting comparisons; the large-scale generation campaign uses the separate recipe in Section A.4. Simulations for these comparisons run at 100 Hz physics and 50 Hz control in MuJoCo Warp (most datasets) and Genesis (ARCTIC dataset, as in DexMachina (Mandi et al., 2025)). Ablations and speed tests use RTX 4090 GPUs; the large-scale generation campaign uses L40S GPUs. All hyperparameters are listed in Table 5.

17

<!-- page 18 of 35 -->

**Automatic difficulty boosting.** In practice, **SPIDER** requires little per-task hyperparameter tuning because per-step failures are handled by an automatic boosting mechanism. When the optimizer fails to reach the error tolerance at a step, the system retries from the last checkpoint with a boosted budget: each boost level doubles the number of optimization iterations and scales the sampling noise by $\sqrt { 2 }$ (both relative to the original configuration), and the noise-decay rate is re-derived so that the final-iteration noise still matches the configured target. If local retries are exhausted, the system reverts to an earlier checkpoint and re-optimizes that segment with the boosted budget before continuing. This acts as an automatic per-step hyperparameter adaptation: easy segments are solved cheaply with the default budget, while hard segments (e.g., contact-rich transitions) automatically receive more samples and wider exploration, so the iteration count and noise scale do not need to be tuned for the hardest step in advance.

Table 5 SPIDER hyperparameters for Oakink and GigaHands retargeting with MuJoCo Warp Simulator. Values that differ from the default are shown per dataset.

| Parameter | Oakink | Oakink (w/ guidance) | GigaHands | GigaHands (w/ guidance) |
| --- | --- | --- | --- | --- |
| Simulation |  |  |  |  |
| Sim. timestep $\Delta t _ { \mathrm { s i m } }$ | 0.01 s | 0.005 s | 0.01 s | 0.005 s |
| Control timestep $\Delta t _ { \mathrm { c t r l } }$ | 0.4 s | 0.05 s | 0.4 s | 0.2 s |
| Knot spacing $\Delta t _ { \mathrm { k n o t } }$ | 0.4 s | 0.1 s | 0.4 s | 0.4 s |
| MPC / CEM Optimization |  |  |  |  |
| Planning horizon H | 2.0 s | 0.4 s | 1.6 s | 1.6 s |
| Horizon steps | 200 | 80 | 160 | 320 |
| Parallel samples $N _ { w }$ | 1024 | 1024 | 1024 | 1024 |
| Max CEM iterations N | 32 | 32 | 32 | 32 |
| CEM temperature τ | 1.0 | 1.0 | 1.0 | 1.0 |
| Early stop threshold € | 0.01 | 0.0 | 0.01 | 0.0 |
| Early stop patience | 2 | 1 | 1 | 1 |
| Noise Schedule |  |  |  |  |
| Initial noise $\sigma _ { \mathrm { f i r s t } }$ | 0.5 | 1.0 | 0.5 | 1.0 |
| Final noise (horizon end) $\sigma _ { \mathrm { l a s t } }$ | 1.0 | 1.0 | 1.0 | 1.0 |
| Noise decay β | 0.931 | 0.963 | 0.931 | 0.963 |
| Joint noise scale | 0.03 | 0.02 | 0.03 | 0.15 |
| Wrist pos noise scale | 0.01 | 0.005 | 0.03 | 0.01 |
| Wrist rot noise scale | 0.01 | 0.01 | 0.01 | 0.01 |
| Reward Weights |  |  |  |  |
| Base pos $w _ { \mathrm { p o s } } ^ { \mathrm { b a s e } }$ | 0.1 | 0.3 | 0.1 | 0.03 |
| Base rot $w _ { \mathrm { r o t } } ^ { \mathrm { \bar { b } a s e } }$ | 0.03 | 0.3 | 0.03 | 0.03 |
| Fingertip pos $w _ { \mathrm { p o s } }$ | 1.0 | 1.0 | 1.0 | 1.0 |
| Fingertip rot $w _ { \mathrm { r o t } }$ | 0.3 | 1.0 | 0.3 | 1.0 |
| Contact Guidance (w/ guidance only) |  |  |  |  |
| Contact guidance | x | √ | x | √ |
| Guidance decay $\gamma _ { g }$ |  | 0.8 |  | 0.5 |
| Pos actuator gain |  | 10.0 |  | 2.0 |
| Rot actuator gain |  | 0.3 |  | 0.3 |

## A.4 Large-Scale Dataset Generation

**Source Data and Clip Selection.** The central processing objective is to isolate the hand–object interaction interval, particularly contact establishment, and retarget the motion that forms and maintains the grasp. We preserve a short pre-contact approach for initialization; contact-focused selection therefore includes the transition into contact rather than only frames in which contact is already present. The campaign in Section 4.4 uses 499 usable right-hand DexYCB (Chao et al., 2021) demonstrations (out of 501 source folders),

18

<!-- page 19 of 35 -->

169 OakInk-Image (Yang et al., 2022) sequences with the liftup intent, 441 HRDexDB (Lim et al., 2026) sequences, and 2,718 single-hand HOT3D clips. OakInk-Image is distinct from the bimanual OAKINK2 data used in the retargeting comparisons. HOT3D (Banerjee et al., 2025) clips are extracted from 424 recordings (198 Aria and 226 Quest3). Hand and object timestamps are matched within 20 ms, and Quest3 trajectories are rotated from Y-up to Z-up. A grasp event requires a fingertip within 1.5 cm of the object surface for at least five frames; lift onset requires object displacement above 2 mm per frame during contact. Clips begin 30 frames before first touch and end 10 frames after lift onset. Static grasps and clips with the carrying hand untracked in more than 20% of frames are discarded. The 141 additional right-hand HOT3D bimanual clips and all left-hand clips are excluded from the reported results. Object counts are dataset-local identifiers, not globally deduplicated physical objects or semantic classes.

**Representation and Lift Augmentation.** Each source episode contains object poses, 21 hand landmarks in MediaPipe order, MANO vertices, and an object mesh, in metres at 30 Hz. GLB scene transforms are preserved, including HOT3D’s millimetre-to-metre scale, and the initial object is translated so its lowest mesh vertex lies on the ground. The wrist orientation is constructed from the wrist-to-middle-MCP direction and the index-to-ring-MCP direction; the five fingertip targets use landmark indices 4, 8, 12, 16, and 20. DexYCB and OakInk-Image packages include an appended synthetic lift; HOT3D packages are extended to a target height, whereas HRDexDB contains recorded lifts. The common processor further appends a lift if needed to reach 0.30 m above the initial object position, translating the hand and object together at 0.3 m/s while holding their relative configuration fixed. References are resampled to 12.5 Hz with duration-preserving interpolation and a nominal 12–64-frame range before IK. HRDexDB instead uses a fixed 24-frame recipe, yielding 20 saved reference frames after IK trimming. Trimming and resampling can reduce the final reference lift below 0.30 m. No hand-size normalization or fingertip tightening is applied.

**Embodiments and Initialization.** The four floating hands have six base coordinates and 16 (Allegro), 12 (XHand), 12 (Inspire), or 22 (Sharpa) finger control dimensions; Allegro has four fingers and the others have five. We generate convex object collision geometry, detect reference contacts, and construct a MuJoCo scene for each hand. Open-hand IK transitions each finger from an open pose to its reference target at its first detected contact frame. An IK rollout is rejected if it contains non-finite states, has fewer than eight frames, or places the object outside the vertical range [−0.3, 2.0] m. The hands are simulated without attached robot arms.

**MPC Configuration.** All reported runs use seed 0 and the same optimization settings in Table 6, without contact guidance or external force/torque perturbations. The campaign uses NVIDIA L40S GPUs with concurrent optimization jobs; the archived runs report a median task wall time of approximately 220 s, including preprocessing. This runtime measures generation cost, not the duration of the resulting trajectory.

Table 6 Shared large-scale generation configuration. HRDexDB differs only in its reference-length processing.

| Parameter | Value | Parameter | Value |
| --- | --- | --- | --- |
| Simulation timestep | 0.01s | Reference timestep | 0.08s |
| Control interval | 0.25s | Knot spacing | 0.5s |
| Planning horizon | 2.0s | Samples per iteration | 1,024 |
| Maximum iterations | 16 | Temperature | 1.0 |
| Initial control-noise scale | 2.0 | Horizon-end noise scale | 4.0 |
| Joint noise scale | 0.1 | Position / rotation noise scale | 0.01 / 0.01 |
| Final optimization noise scale | 0.01 | Contact reward weight | 0 |

**Tracking Evaluation.** We recompute surface-point ADD-AUC directly from saved object poses and source meshes, following the metric in Section A. For each sequence, we sample K = 512 points $v _ { j }$ uniformly by surface area on the object mesh and use the same points for all robot hands. The per-frame error is

$$
d _ { t } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \left\| R _ { t } v _ { j } + p _ { t } - \left( R _ { t } ^ { \operatorname * { r e f } } v _ { j } + p _ { t } ^ { \operatorname * { r e f } } \right) \right\| _ { 2 } .\tag{6}
$$

19

<!-- page 20 of 35 -->

We evaluate the fraction of frames with $d _ { t } \leq \tau$ at 100 equally spaced thresholds $\tau \in [ 0 , 0 . 1 0 ]$ m and apply trapezoidal integration after normalizing the threshold axis to [0, 1]. Reported values are multiplied by 100. Surface samples use a deterministic per-sequence seed. For temporal alignment, we take every eighth state from the 100 Hz simulation and truncate simulation and reference to their common length at 12.5 Hz, without mean-centering or extending the reference. Each trajectory contributes equally to dataset and overall averages. The main table includes all 15,041 completed single-hand rollouts, including poor reference lifts and failed grasps. The 281 rollouts containing non-finite simulated states are retained; invalid evaluated frames contribute zero threshold successes. A position-only AUC implementation, which ignores rotation, gives 63.25% overall; the reported surface-point ADD-AUC is 61.48%.

**Completion and Retention Criteria.** The 3,827 source demonstrations and four hands define 15,308 planned tasks. There are 15,302 task records: 15,041 completed rollouts, 161 IK-sanity failures, and 100 preparation failures; six further planned tasks have no record. Failures without a completed trajectory have no ADD-AUC and are excluded from its denominator. For dataset retention, we require a reference rise of at least 0.15 m, a final simulated rise of at least 0.10 m, and a final object-position error of at most 0.10 m. For this final-error test, reference positions are linearly interpolated onto the simulation timestamps, with endpoint holding beyond the reference duration. These labels are recomputed from the saved trajectories. Excluding the 437 completed rollouts with insufficient reference lift leaves 14,604 eligible rollouts with ADD-AUC 61.67%; applying the full retention filter yields the 6,520 trajectories in Table 7. Their higher ADD-AUC describes the selected dataset rather than unfiltered generation performance. No perturbation robustness or real-robot execution is evaluated for this campaign.

Table 7 Retained simulated demonstrations, seed 0. Object identities are counted within each source dataset. Hours equal saved frames divided by 100 Hz and 3,600.

| Dataset | Trajectories | Objects | Frames | Hours | ADD-AUC (%) |
| --- | --- | --- | --- | --- | --- |
| DexYCB [4] | 870 | 20 | 151,650 | 0.42 | 73.27 |
| OakInk-Image [55] | 241 | 58 | 36,850 | 0.10 | 76.52 |
| HRDexDB [22] | 871 | 80 | 152,425 | 0.42 | 79.13 |
| HOT3D [2] | 4,538 | 33 | 950,650 | 2.64 | 79.19 |
| Total / overall | 6,520 | 191 | 1,291,575 | 3.59 | 78.30 |

## A.5 Humanoid Policy Training

This section describes the training details for the humanoid loco-manipulation policy experiment in the main paper. We use the PPO algorithm Schulman et al. (2017) and port the implementation from Weng et al. (2025).

**Rewards.** We use the following rewards:

• Body tracking reward: tracking humanoid pelvis, hand, and leg motions.

• Object tracking reward: tracking the object motion.

• Action rate penalty: penalizing the action rate to avoid jittering.

We remove the contact reward, self-collision penalty as well as the auxiliary contact reward proposed in the paper to evaluate the reference motion retargeting quality.

**Termination.** We terminate the training when the average body tracking error is greater than 10 cm and the average object tracking error is greater than 10 cm.

## A.6 Dexterous Hand Policy Training

This section describes the training details for the dexterous grasping policy experiment in the main paper. **SPIDER** is complementary to RL algorithms: it provides an already-feasible reference motion, so the policy only needs to track it rather than discover the behavior from scratch. The task uses a right arm (7-DoF)

20

<!-- page 21 of 35 -->

paired with a 22-DoF dexterous hand (5 fingers, 29 DoF total) grasping a cup on a table. The reference is collected from an in-the-wild human demonstration and physics-grounded with **SPIDER** (Section 4.3). The **SPIDER** nominal feedforward control is used both to initialize the policy targets and as the reference for the tracking reward; we do not use it as a residual control term, though such a variant is also possible.

**Scene and simulation.** The scene consists of the right arm mounted at a fixed base, the object (mesh loaded from the demonstration manifest) as a dynamic rigid body, and a table with its top surface at 0.75 m. The environment runs at 20 Hz control over a 120 Hz PhysX simulation (decimation 6; TGS solver with 16 position iterations, 2 mm contact offset), with episodes of 12 s (≈ 240 policy steps) and 4096 parallel environments. The arm uses per-joint PD gains identified from the physical system.

**Action space (**28**-D).** The policy outputs a 28-D vector in [−1, 1] that is scaled by per-step limits and accumulated onto persistent wrist and finger targets (delta control). The 6-D wrist delta (position ±1 cm per step, rotation ±4.5◦ per step as axis-angle) drives a nullspace differential-IK controller with damped least squares; the 22-D finger delta (±0.2 rad per step) is sent to a joint-position controller clamped to joint limits.

**Observation space (**325**-D).** The observation concatenates: relative joint positions and velocities (29 + 29), wrist pose (3 + 6, using 6-D rotation), accumulated wrist and finger targets (3 + 6 + 22), object pose (3 + 6), the reference wrist pose, finger joints, and object pose at the current reference step $( 3 + 6 + 22 + 3 + 6 )$ , the goal object bounding box (24) and goal hand landmarks in the object frame (51), the current object bounding box (24) and hand landmarks (51), and the last action (28). We use an asymmetric actor-critic: the critic additionally receives privileged information (per-joint PD gains, effort limits and armature, object mass and friction, object velocities, and the object–wrist relative pose). The policy network receives the current frame concatenated with a 4-step history buffer; actor and critic use (512, 256, 128) MLPs.

**Rewards.** We use dense tracking rewards of the form exp(−s · e), where e is the tracking error and s is a per-term scale, listed in Table 8, together with a small action-magnitude penalty. The hand-landmark term tracks 17 landmarks on the hand expressed in the object frame, which couples the hand and object motion and is given the largest weight. We remove the lifting reward and the success bonus, keeping only the imitation tracking terms so that performance reflects the quality of the reference motion, mirroring the humanoid setting.

| Term | Error e | Scale s | Weight |
| --- | --- | --- | --- |
| Object position | ∥∆pobj∥ | 30 | 0.065 |
| Object rotation | geodesic angle | 1.9 | 0.065 |
| Wrist position | ∥∆p<sub>wrist</sub>∥ | 20 | 0.039 |
| Wrist rotation | geodesic angle | 1.9 | 0.039 |
| Finger joints | mean \|∆q\| | 10 | 0.039 |
| Hand landmarks | per-point distance (object frame) | 60 | 0.279 |
| Action penalty | mean(a2t) | - | -0.10 |

Table 8 Reward terms for the dexterous grasping policy. Each tracking term is exp(−s · e) ∈ (0, 1]; weights are normalized. The lifting reward and success bonuses of the original environment are removed.

**Termination and success.** An episode terminates if the object leaves the workspace (height below 0.5 m or above 1.5 m in the environment frame), if the wrist drifts more than 1.2 m from the object, or if any arm joint exceeds its position limit; episodes truncate at 12 s. An episode is counted as successful when the object is lifted at least 4 cm above the table while the hand remains in contact (minimum hand-to-object distance below 2 cm).

**Initial-state generation via sequential IK rollout.** To turn the single **SPIDER**-retargeted demonstration into a diverse initial-state distribution, we augment it into thousands of randomized, reachability-validated placements. For each candidate placement, the object is randomly translated on the table and rotated in yaw (uniform over the table workspace, discarding placements outside the table bounds), and the demonstrated wrist trajectory—stored in an object-centered frame—is re-expressed in the robot frame at that placement. A GPU-batched damped-least-squares IK solver then rolls the trajectory forward waypoint-by-waypoint: the first waypoint is warm-started from the nearest configuration in a precomputed table of healthy arm configurations

21

<!-- page 22 of 35 -->

(with a larger iteration budget of 400 vs. 50 for subsequent waypoints), and every later waypoint is warmstarted from the previous solution. A placement is accepted only if the arm can track the entire trajectory within tolerance (wrist position error < 3 cm, orientation error < 0.15 rad) while keeping the proximal joints within 0.5 rad of the healthy-configuration manifold (rejecting contorted elbow postures). Before the object is lifted, its pose follows the transformed reference; after lift-off, it is treated as rigidly attached to the wrist and its pose is recomputed from the achieved forward kinematics. Accepted placements—each carrying a complete arm, finger, and object trajectory—are further filtered by a physics rollout that verifies grasp stability before being used for training.

**Reset.** On reset, each environment samples a placement and a starting waypoint along its trajectory (biased toward early waypoints, with a fixed fraction starting from the beginning) and initializes the robot and object from the corresponding validated state; for early waypoints, the object position is additionally perturbed by ±4 cm in the horizontal plane with $\pm 3 0 ^ { \circ }$ yaw, and joint targets receive small noise. Episode start times are desynchronized across environments so that resets do not coincide.

**PPO hyperparameters.** We train with PPO (Schulman et al., 2017): adaptive learning rate initialized at $3 \times 1 0 ^ { - 4 }$ (KL-targeted at 0.016), discount $\gamma = 0 . 9 5$ , GAE $\lambda = 0 . 9 .$ , clip 0.2, horizon length 16, minibatch size 16384, 5 mini-epochs, gradient norm clipping at 1.0, and observation/value normalization. Each policy is trained on a single L40 GPU for roughly 3–4 hours.

## B Comparison with Baselines

## B.1 Comparison Against Kinematics-Based Retargeting Baselines

**Setup and Per-Hand Results.** To ensure representative evaluation while maintaining computational feasibility, we conduct our ablation study on a collection of challenging manipulation tasks from established datasets: (1) for Oakink (Zhan et al., 2024), we evaluate on 7 distinct bimanual two-object manipulation tasks that were previously benchmarked in **ManipTrans** (Li et al., 2025c); and (2) for GigaHands (Fu et al., 2025), we utilize their released example dataset comprising 3 manipulation trajectories over five hands. Per-hand tracking results are presented in Tables 9 and 10, where **SPIDER** wins in terms of object tracking precision, especially with contact guidance enabled. The aggregated comparison is summarized in the main paper (Table 1).

We report per-trajectory tracking metrics for Oakink and GigaHands datasets in the following tables.

22

<!-- page 23 of 35 -->

<table><tr><td>Robot</td><td colspan=3>Kinematic       SPIDER        $\mathrm { S P I D E R + C }$ </td></tr><tr><td rowspan=11> $e _ { t } ( \mathrm { c m } ) \downarrow$ Allegro     $e _ { r } ( \deg ) \downarrow$ AUC↑ $e _ { t } ( \mathrm { c m } ) \downarrow$ Ability     $e _ { r } ( \deg ) \downarrow$ AUC↑ $e _ { t } ( \mathrm { c m } ) \downarrow$ Inspire     $e _ { r } ( \deg ) \downarrow$ AUC ↑ $e _ { t } ( \mathrm { c m } ) \downarrow$ Schunk       $e _ { r } ( \deg ) \downarrow$ </td><td> $9 . 4 _ { \pm 6 . 8 }$ </td><td> $7 . 7 _ { \pm 7 . 5 }$ </td><td rowspan=3> $\mathbf { 4 . 6 _ { \pm 2 . 8 } }$  $\mathbf { 1 9 . 9 _ { \pm 1 8 . 9 } }$ 56.6</td></tr><tr><td rowspan=2> $4 4 . 2 _ { \pm 3 0 . 9 }$ 21.5</td><td> $3 7 . 6 _ { \pm 3 3 . 1 }$ </td></tr><tr><td>53.4</td></tr><tr><td> $7 6 . 3 _ { \pm 1 4 3 . 5 }$ </td><td> $\mathbf { 9 . 3 _ { \pm 9 . 1 } }$ </td><td> $1 0 . 9 _ { \pm 8 . 2 }$ </td></tr><tr><td> $7 5 . 2 _ { \pm 2 6 . 5 }$ </td><td> $4 4 . 4 _ { \pm 3 9 . 1 }$ </td><td rowspan=2> $\mathbf { 3 6 . 4 _ { \pm 3 2 . 7 } }$ 57.9</td></tr><tr><td>8.2</td><td>43.7</td></tr><tr><td rowspan=2> $1 9 6 . 6 _ { \pm 5 4 5 . 3 }$  $9 1 . 0 _ { \pm 3 1 . 2 }$ </td><td> $\mathbf { 9 . 1 _ { \pm 8 . 1 } }$ </td><td> $1 1 . 7 _ { \pm 1 1 . 7 }$ </td></tr><tr><td> $\mathbf { 4 1 . 0 _ { \pm 2 9 . 4 } }$ </td><td> $4 3 . 3 _ { \pm 2 9 . 2 }$ </td></tr><tr><td>7.7</td><td>46.4</td><td>63.4</td></tr><tr><td> $1 7 . 8 _ { \pm 9 . 7 }$ </td><td> $9 . 1 _ { \pm 8 . 7 }$ </td><td rowspan=3> $\mathbf { 7 . 6 _ { \pm 4 . 7 } }$  $\mathbf { 2 8 . 8 _ { \pm 1 9 . 6 } }$ 52.4</td></tr><tr><td> $8 6 . 5 _ { \pm 2 4 . 7 }$ </td><td> $3 3 . 1 _ { \pm 2 6 . 2 }$ </td></tr><tr><td rowspan=4> $\mathrm { A U C } \uparrow$  $e _ { t } ( \mathrm { c m } ) \downarrow$ XHand          $e _ { r } ( \deg ) \downarrow$  $\mathrm { A U C } \uparrow$ </td><td>14.6</td><td>46.7</td></tr><tr><td> $3 9 . 5 _ { \pm 2 1 . 4 }$ </td><td> $7 . 1 _ { \pm 7 . 6 }$ </td><td> $\mathbf { 5 . 2 _ { \pm 3 . 3 } }$ </td></tr><tr><td> $1 0 8 . 4 _ { \pm 1 5 . 6 }$ </td><td> $3 2 . 4 _ { \pm 2 9 . 7 }$ </td><td rowspan=2> $\mathbf { 2 4 . 1 _ { \pm 1 9 . 6 } }$ 54.5</td></tr><tr><td>5.7</td><td>53.4</td></tr></table>

Table 9 Comparison with Kinematic retargeting on Oakink. $\mathrm { M e a n } _ { \pm \mathrm { s t d } }$ over 5 seeds. AUC denotes ADD-AUC in percent (Section A).

## B.2 ManipTrans Retargeting Results

To match the exact setting of **ManipTrans** and avoid discrepancies caused by different simulators, we directly port the training environment and evaluation protocol from **ManipTrans** on the Oakink dataset. The following are the details of the training and evaluation protocol when comparing **SPIDER** with the **ManipTrans** RL baseline.

**RL Baseline Training.** We train one residual RL policy per trajectory using the **ManipTrans** framework on 16 randomly picked single-hand picking trajectories from the Oakink dataset. Each policy is trained for 100 epochs on a single trajectory with the Inspire dexterous hand using the same 12-DoF simulation model as in the original **ManipTrans** paper. The action space consists of 18-dimensional residual actions (6 wrist force/torque + 12 joint DOF positions), applied on top of a pre-trained base imitator that provides nominal actions from the current observations. We use an action smoothing factor of 0.4 as suggested in the baseline repository.

**RL Evaluation.** We evaluate each trained policy with 5 random seeds (0-4) to capture stochasticity. For each seed, we run 2048 rollouts across 256 parallel IsaacGym environments without early termination, deterministic initial states from the demo trajectory, and save the 256 longest rollouts to HDF5.

**SPIDER Retargeting.** We run **SPIDER** with 1024 parallel IsaacGym environments. **SPIDER** optimizes only the 18-dim residual actions; the base imitator provides nominal actions at each step. The action smoothing factor is 0.4, matching RL. For the reward, we use the same reward function as the RL baseline but remove the action penalty term for efficiency (since our action is smooth by interpolation). Each trajectory is run

23

<!-- page 24 of 35 -->

<table><tr><td colspan=2>Robot</td><td colspan=3>Kinematic      SPIDER      SPIDER+C</td></tr><tr><td rowspan=15 colspan=2> $e _ { t } ( \mathrm { c m } ) \downarrow$ Allegro     $$e _ { r }$ (deg) $\downarrow$$  $\mathrm { A U C } \uparrow$  $e _ { t } ( \mathrm { c m } ) \downarrow$ Ability     $$e _ { r }$ (deg) $\downarrow$$  $\mathrm { A U C } \uparrow$  $e _ { t } ( \mathrm { c m } ) \downarrow$ Inspire     $$e _ { r }$ (deg) $\downarrow$$ AUC ↑ $e _ { t } ( \mathrm { c m } ) \downarrow$ Schunk $\mathrm { A U C } \uparrow$  $e _ { t } ( \mathrm { c m } ) \downarrow$ XHand           $$e _ { r }$ (deg) $\downarrow$$  $\mathrm { A U C } \uparrow$ </td><td rowspan=2> $2 3 . 9 _ { \pm 1 1 . 2 }$  $4 5 . 0 _ { \pm 2 2 . 5 }$ </td><td> $\mathbf { 4 . 5 _ { \pm 1 . 0 } }$ </td><td> $8 . 7 _ { \pm 3 . 8 }$ </td></tr><tr><td> $\mathbf { 9 . 1 _ { \pm 5 . 7 } }$ </td><td> $9 . 9 _ { \pm 2 . 9 }$ </td></tr><tr><td>41.4</td><td>58.7</td><td>83.1</td></tr><tr><td rowspan=2> $3 7 . 3 _ { \pm 3 0 . 8 }$  $6 7 . 7 _ { \pm 4 4 . 2 }$ </td><td> $\mathbf { 7 . 9 _ { \pm 5 . 5 } }$ </td><td> $1 2 . 5 _ { \pm 4 . 8 }$ </td></tr><tr><td> $1 9 . 7 _ { \pm 1 0 . 1 }$ </td><td> $\mathbf { 1 7 . 8 _ { \pm 6 . 3 } }$ </td></tr><tr><td>17.4</td><td>55.5</td><td>30.8</td></tr><tr><td> $4 4 . 2 _ { \pm 2 7 . 3 }$ </td><td> $\mathbf { 4 . 5 _ { \pm 0 . 3 } }$ </td><td> $1 1 . 4 _ { \pm 7 . 6 }$ </td></tr><tr><td> $7 1 . 3 _ { \pm 2 9 . 6 }$ </td><td> $1 5 . 1 _ { \pm 1 2 . 3 }$ </td><td rowspan=2> $\mathbf { 1 4 . 8 _ { \pm 8 . 1 } }$ 73.4</td></tr><tr><td>12.5</td><td>53.1</td></tr><tr><td> $3 8 . 3 _ { \pm 1 7 . 5 }$ </td><td> $\mathbf { 1 1 . 1 _ { \pm 9 . 4 } }$ </td><td> $1 4 . 0 _ { \pm 2 . 5 }$ </td></tr><tr><td>er (deg) ↓</td><td> $5 9 . 2 _ { \pm 1 6 . 4 }$ </td><td> $1 9 . 9 _ { \pm 2 0 . 6 }$ </td><td> $\mathbf { 1 7 . 1 _ { \pm 7 . 1 } }$ </td></tr><tr><td>18.1</td><td>60.8</td><td>30.9</td></tr><tr><td> $4 6 . 4 _ { \pm 4 2 . 8 }$ </td><td> $\mathbf { 4 . 5 _ { \pm 1 . 1 } }$ </td><td> $8 . 1 _ { \pm 4 . 5 }$ </td></tr><tr><td> $7 3 . 4 _ { \pm 4 5 . 6 }$ </td><td> $1 4 . 3 _ { \pm 6 . 2 }$ </td><td> $\mathbf { 1 3 . 2 _ { \pm 8 . 1 } }$ </td></tr><tr><td>31.1</td><td>56.6</td><td>64.0</td></tr></table>

Table 10 Comparison with Kinematic retargeting on GigaHands. Mean $\mathtt { l } { \pm } \mathtt { s t d }$ over $5$ seeds. AUC denotes ADD-AUC in percent (Section A).

with 5 seeds. All hyperparameters are set as in Table 16.

**Evaluation Metrics.** We compute four metrics against ground-truth demo data from reference motion, following the **ManipTrans** evaluation protocol. Evaluation metrics are:

$e _ { t } ;$ mean $L _ { 2 }$ object translation error (cm)

• $e _ { r } ;$ geodesic object rotation error (degrees), computed as the angle of the axis-angle representation of $R _ { g t } R _ { p r e d } ^ { T }$

$e _ { j } \colon$ mean $L _ { 2 }$ hand joint position error (cm), computed over all body positions including wrist

$e _ { f t } \colon$ mean $L _ { 2 }$ fingertip position error (cm), computed over 5 fingertip joints (thumb, index, middle, ring, pinky)

All metrics are averaged over the trajectory length, then reported as mean ± std across 5 seeds per trajectory.

**Experiment Results.** We randomly select 16 trajectories (8 left-hand, 8 right-hand) from the Oakink dataset. Full results are shown in Table 17. For some trajectories, **SPIDER**’s standard deviation is 0 because the object is not moving and **SPIDER** perfectly tracks the hand motion.

24

<!-- page 25 of 35 -->

| Dataset Task Metric | Kinematic | SPIDER | SPIDER+C |
| --- | --- | --- | --- |
| eft (cm) | 1.6±0.0 | 2.8±0.4 | 4.4±0.9 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.03±0.00 | 0.04±0.00 |
| lift_board |  |  |  |
| et (cm) | 3.8±0.0 | 1.9±0.2 | 1.8±0.2 |
| e<sub>r</sub> (deg) | 12.5±0.0 | 7.6±1.1 | 4.4±1.1 |
| eft (cm) | 0.9±0.0 | 2.4±0.6 | 3.8±1.9 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.03±0.00 | 0.08±0.09 |
| pick_spoon_bowl |  |  |  |
| et (cm) | 6.1±0.0 | 4.5±0.8 | 7.2±1.4 |
| e<sub>r</sub> (deg) | 56.3±0.0 | 20.9±2.0 | 50.7±26.4 |
| eft (cm) | 0.9±0.0 | 0.7±0.2 | 5.7±0.2 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.20±0.15 | 0.04±0.00 |
| pour_tube |  |  |  |
| et (cm) | 23.3±0.0 | 23.6±2.5 | 5.7±0.4 |
| e<sub>r</sub> (deg) | 109.0±0.0 | 110.2±14.2 | 20.6±2.5 |
| eft (cm) | 1.3±0.0 | 1.9±0.5 | 2.9±0.2 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.02±0.00 | 0.02±0.00 |
| stir_beaker |  |  |  |
| et (cm) | 15.4±0.0 | 13.5±1.4 | 7.3±3.1 |
| e<sub>r</sub> (deg) | 46.9±0.0 | 53.1±11.7 | 24.7±5.2 |
| oakink |  |  |  |
| eft (cm) | 1.1±0.0 | 1.4±0.2 | 2.6±0.5 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.01±0.00 | 0.02±0.00 |
| uncap_alcohol_burner |  |  |  |
| et (cm) | 3.3±0.0 | 2.6±0.2 | 3.1±0.4 |
| e<sub>r</sub> (deg) | 22.6±0.0 | 24.4±0.9 | 9.9±2.5 |
| eft (cm) | 1.0±0.0 | 1.2±0.1 | 2.7±0.3 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.02±0.00 | 0.06±0.06 |
| unplug |  |  |  |
| et (cm) | 6.4±0.0 | 3.8±1.6 | 6.0±1.5 |
| e<sub>r</sub> (deg) | 14.9±0.0 | 22.4±4.4 | 25.4±12.2 |
| eft (cm) | 0.9±0.0 | 2.1±0.3 | 3.0±0.2 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.02±0.00 | 0.05±0.00 |
| wipe_board |  |  |  |
| et (cm) | 7.5±0.0 | 3.7±0.6 | 0.9±0.1 |
| e<sub>r</sub> (deg) | 47.0±0.0 | 24.3±4.6 | 3.3±0.4 |
| eft (cm) | 1.1±0.2 | 1.8±0.7 | 3.6±1.0 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.05±0.06 | 0.04±0.02 |
| Avg. |  |  |  |
| et (cm) | 9.4±6.8 | 7.7±7.4 | 4.6±2.4 |
| e<sub>r</sub> (deg) | 44.2±30.9 | 37.6±32.2 | 19.9±15.2 |
| eft (cm) | 1.7±0.4 | 4.9±0.5 | 7.0±1.5 |
| e<sub>j</sub> (rad) | 0.29±0.17 | 0.12±0.13 | 0.11±0.01 |
| p36-tea |  |  |  |
| et (cm) | 11.1±0.0 | 3.6±1.1 | 4.4±2.9 |
| e<sub>r</sub> (deg) | 72.9±0.0 | 16.1±4.5 | 7.9±2.6 |
| eft (cm) | 4.0±0.1 | 5.8±0.6 | 2.0±0.0 |
| e<sub>j</sub> (rad) | 0.20±0.09 | 0.10±0.08 | 0.05±0.00 |
| p44-dog |  |  |  |
| et (cm) | 37.7±0.0 | 4.7±0.7 | 11.3±0.8 |
| e<sub>r</sub> (deg) | 40.0±0.0 | 4.6±0.4 | 13.1±0.8 |
| gigahand |  |  |  |
| eft (cm) | 3.1±0.0 | 4.2±0.1 | 4.5±0.3 |
| e<sub>j</sub> (rad) | 0.08±0.00 | 0.07±0.00 | 0.13±0.12 |
| p52-instrument |  |  |  |
| et (cm) | 22.6±0.0 | 5.2±0.3 | 10.4±2.3 |
| e<sub>r</sub> (deg) | 17.6±0.0 | 6.7±0.1 | 8.7±1.5 |
| eft (cm) | 3.0±0.9 | 5.0±0.6 | 4.5±2.0 |
| e<sub>j</sub> (rad) | 0.19±0.09 | 0.10±0.02 | 0.10±0.04 |
| Avg. |  |  |  |
| et (cm) | 23.8±10.9 | 4.5±0.7 | 8.7±3.1 |
| e<sub>r</sub> (deg) | 43.5±22.7 | 9.1±5.0 | 9.9±2.3 |

Table 11 Per-trajectory tracking metrics for allegro. Mean ± std across 5 seeds.

25

<!-- page 26 of 35 -->

| Dataset Task Metric | Kinematic | SPIDER | SPIDER+C |
| --- | --- | --- | --- |
| eft (cm) | 6.5±0.0 | 3.7±1.1 | 2.0±0.4 |
| e<sub>j</sub> (rad) | 0.19±0.07 | 0.04±0.00 | 0.02±0.00 |
| lift_board |  |  |  |
| et (cm) | 18.0±0.0 | 7.8±2.0 | 5.5±2.6 |
| e<sub>r</sub> (deg) | 19.3±0.0 | 6.6±2.3 | 4.9±1.8 |
| eft (cm) | 1.3±0.0 | 3.3±0.6 | 7.0±0.3 |
| e<sub>j</sub> (rad) | 0.13±0.07 | 0.02±0.00 | 0.12±0.00 |
| pick_spoon_bowl |  |  |  |
| et (cm) | 12.4±0.0 | 6.5±1.8 | 9.0±0.3 |
| e<sub>r</sub> (deg) | 53.7±0.0 | 64.3±12.4 | 41.1±12.3 |
| eft (cm) | 1.8±0.4 | 2.2±0.4 | 8.6±1.3 |
| e<sub>j</sub> (rad) | 0.16±0.10 | 0.06±0.09 | 0.01±0.00 |
| pour_tube |  |  |  |
| et (cm) | 30.9±0.0 | 30.0±2.3 | 26.6±1.0 |
| e<sub>r</sub> (deg) | 93.2±0.0 | 126.7±7.2 | 92.6±8.7 |
| eft (cm) | 3.5±0.1 | 3.5±0.3 | 8.4±0.3 |
| e<sub>j</sub> (rad) | 0.09±0.05 | 0.05±0.04 | 0.08±0.00 |
| stir_beaker |  |  |  |
| et (cm) | 14.2±0.0 | 8.3±3.2 | 19.5±0.7 |
| e<sub>r</sub> (deg) | 72.6±0.0 | 38.8±6.1 | 74.2±5.7 |
| oakink |  |  |  |
| eft (cm) | 3.5±0.4 | 3.8±1.6 | 4.0±0.8 |
| e<sub>j</sub> (rad) | 0.26±0.15 | 0.07±0.09 | 0.05±0.00 |
| uncap_alcohol_burner |  |  |  |
| et (cm) | 10.1±0.0 | 5.1±2.4 | 4.0±2.0 |
| e<sub>r</sub> (deg) | 75.6±0.0 | 43.5±11.7 | 12.4±2.9 |
| eft (cm) | 3.2±0.1 | 2.2±0.5 | 2.6±0.4 |
| e<sub>j</sub> (rad) | 0.09±0.05 | 0.05±0.05 | 0.01±0.00 |
| unplug |  |  |  |
| et (cm) | 422.5±0.0 | 1.6±0.3 | 5.4±2.0 |
| e<sub>r</sub> (deg) | 98.9±0.0 | 12.6±0.8 | 19.9±7.3 |
| eft (cm) | 3.0±0.1 | 3.3±0.3 | 4.3±0.4 |
| e<sub>j</sub> (rad) | 0.15±0.08 | 0.07±0.07 | 0.01±0.00 |
| wipe_board |  |  |  |
| et (cm) | 16.6±0.0 | 4.9±0.2 | 6.2±1.5 |
| e<sub>r</sub> (deg) | 104.0±0.0 | 22.4±0.6 | 9.8±1.0 |
| eft (cm) | 3.3±1.6 | 3.1±0.6 | 5.3±2.5 |
| e<sub>j</sub> (rad) | 0.15±0.05 | 0.05±0.02 | 0.04±0.04 |
| Avg. |  |  |  |
| et (cm) | 74.9±142.0 | 9.2±8.8 | 10.9±8.0 |
| e<sub>r</sub> (deg) | 73.9±27.5 | 45.0±38.0 | 36.4±32.0 |
| eft (cm) | 2.0±0.2 | 4.3±0.4 | 7.2±1.6 |
| e<sub>j</sub> (rad) | 0.11±0.04 | 0.08±0.03 | 0.11±0.02 |
| p36-tea |  |  |  |
| et (cm) | 10.8±0.0 | 4.2±1.1 | 11.3±5.2 |
| e<sub>r</sub> (deg) | 49.0±0.0 | 21.3±0.6 | 21.7±2.7 |
| eft (cm) | 5.8±0.0 | 5.5±0.1 | 3.9±0.0 |
| e<sub>j</sub> (rad) | 0.15±0.04 | 0.11±0.04 | 0.09±0.00 |
| p44-dog |  |  |  |
| et (cm) | 24.7±0.0 | 4.7±0.2 | 10.9±0.5 |
| e<sub>r</sub> (deg) | 30.3±0.0 | 6.7±0.4 | 9.4±0.4 |
| gigahand |  |  |  |
| eft (cm) | 6.7±0.0 | 3.5±0.1 | 3.7±0.3 |
| e<sub>j</sub> (rad) | 0.08±0.00 | 0.10±0.00 | 0.14±0.02 |
| p52-instrument |  |  |  |
| et (cm) | 84.2±0.0 | 14.8±4.0 | 15.4±5.5 |
| e<sub>r</sub> (deg) | 135.1±0.0 | 31.0±1.9 | 22.2±2.3 |
| eft (cm) | 4.8±2.0 | 4.4±0.8 | 4.9±1.6 |
| e<sub>j</sub> (rad) | 0.11±0.03 | 0.10±0.01 | 0.11±0.02 |
| Avg. |  |  |  |
| et (cm) | 39.9±31.8 | 7.9±4.9 | 12.5±2.0 |
| e<sub>r</sub> (deg) | 71.5±45.6 | 19.7±10.0 | 17.8±5.9 |

Table 12 Per-trajectory tracking metrics for ability. Mean ± std across 5 seeds.

26

<!-- page 27 of 35 -->

| Dataset Task Metric | Kinematic | SPIDER | SPIDER+C |
| --- | --- | --- | --- |
| eft (cm) | 7.1±0.0 | 1.8±0.2 | 2.4±0.7 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.03±0.00 | 0.04±0.00 |
| lift_board |  |  |  |
| et (cm) | 66.1±0.0 | 8.1±0.2 | 2.6±1.9 |
| e<sub>r</sub> (deg) | 21.7±0.0 | 6.3±0.8 | 8.3±1.4 |
| eft (cm) | 6.5±0.0 | 1.1±0.3 | 2.0±0.3 |
| e<sub>j</sub> (rad) | 0.14±0.05 | 0.05±0.06 | 0.02±0.01 |
| pick_spoon_bowl |  |  |  |
| et (cm) | 184.1±0.0 | 9.4±2.5 | 12.8±1.7 |
| e<sub>r</sub> (deg) | 87.0±0.0 | 44.5±8.5 | 52.6±12.6 |
| eft (cm) | 7.0±0.0 | 0.4±0.1 | 3.7±1.4 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.01±0.00 | 0.02±0.01 |
| pour_tube |  |  |  |
| et (cm) | 149.4±0.0 | 27.0±1.5 | 31.7±17.8 |
| e<sub>r</sub> (deg) | 124.3±0.0 | 85.2±8.2 | 88.3±17.7 |
| eft (cm) | 6.9±0.0 | 2.2±0.2 | 3.9±0.5 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.03±0.00 | 0.02±0.00 |
| stir_beaker |  |  |  |
| et (cm) | 169.2±0.0 | 4.5±1.4 | 17.5±1.8 |
| e<sub>r</sub> (deg) | 106.7±0.0 | 32.4±10.7 | 65.4±9.2 |
| oakink |  |  |  |
| eft (cm) | 7.0±0.0 | 1.5±0.2 | 3.4±0.5 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.04±0.00 | 0.07±0.00 |
| uncap_alcohol_burner |  |  |  |
| et (cm) | 23.9±0.0 | 3.6±1.0 | 4.6±0.6 |
| e<sub>r</sub> (deg) | 82.2±0.0 | 30.1±5.8 | 36.8±28.1 |
| eft (cm) | 9.2±0.0 | 2.5±0.7 | 2.7±0.9 |
| e<sub>j</sub> (rad) | 0.22±0.07 | 0.07±0.09 | 0.03±0.00 |
| unplug |  |  |  |
| et (cm) | 731.4±1313.0 | 2.8±1.2 | 7.1±1.0 |
| e<sub>r</sub> (deg) | 109.1±1.1 | 13.6±6.5 | 24.7±10.5 |
| eft (cm) | 7.0±0.2 | 1.7±0.3 | 3.7±0.3 |
| e<sub>j</sub> (rad) | 0.19±0.07 | 0.14±0.09 | 0.03±0.00 |
| wipe_board |  |  |  |
| et (cm) | 51.9±18.7 | 5.9±3.7 | 5.9±0.8 |
| e<sub>r</sub> (deg) | 106.1±4.7 | 70.3±21.9 | 27.0±4.6 |
| eft (cm) | 7.2±0.8 | 1.6±0.6 | 3.1±0.7 |
| e<sub>j</sub> (rad) | 0.11±0.07 | 0.05±0.04 | 0.03±0.02 |
| Avg. |  |  |  |
| et (cm) | 196.6±225.8 | 8.8±7.7 | 11.7±9.4 |
| e<sub>r</sub> (deg) | 91.0±31.2 | 40.3±26.6 | 43.3±25.3 |
| eft (cm) | 8.2±0.0 | 4.7±0.2 | 6.5±0.6 |
| e<sub>j</sub> (rad) | 0.07±0.00 | 0.07±0.00 | 0.13±0.01 |
| p36-tea |  |  |  |
| et (cm) | 11.5±0.0 | 4.9±0.3 | 4.2±1.1 |
| e<sub>r</sub> (deg) | 108.4±0.0 | 32.4±1.6 | 13.7±1.3 |
| eft (cm) | 6.3±2.4 | 3.8±0.1 | 3.6±0.0 |
| e<sub>j</sub> (rad) | 0.12±0.09 | 0.07±0.00 | 0.06±0.00 |
| p44-dog |  |  |  |
| et (cm) | 65.0±24.4 | 4.3±0.2 | 10.5±0.3 |
| e<sub>r</sub> (deg) | 60.0±21.4 | 5.9±0.2 | 8.4±0.2 |
| gigahand |  |  |  |
| eft (cm) | 6.5±0.0 | 4.1±0.1 | 1.7±0.3 |
| e<sub>j</sub> (rad) | 0.07±0.00 | 0.06±0.00 | 0.10±0.01 |
| p52-instrument |  |  |  |
| et (cm) | 55.9±0.0 | 4.3±0.2 | 19.6±7.3 |
| e<sub>r</sub> (deg) | 45.4±0.0 | 7.1±0.5 | 22.4±9.8 |
| eft (cm) | 7.0±0.9 | 4.2±0.4 | 3.9±2.0 |
| e<sub>j</sub> (rad) | 0.09±0.02 | 0.06±0.00 | 0.10±0.03 |
| Avg. |  |  |  |
| et (cm) | 44.2±23.4 | 4.5±0.3 | 11.4±6.3 |
| e<sub>r</sub> (deg) | 71.3±26.9 | 15.1±12.2 | 14.8±5.8 |

Table 13 Per-trajectory tracking metrics for inspire. Mean ± std across 5 seeds.

27

<!-- page 28 of 35 -->

| Dataset Task Metric | Kinematic | SPIDER | SPIDER+C |
| --- | --- | --- | --- |
| eft (cm) | 5.2±0.0 | 3.1±0.4 | 3.0±0.3 |
| e<sub>j</sub> (rad) | 0.28±0.00 | 0.03±0.00 | 0.02±0.00 |
| lift_board |  |  |  |
| et (cm) | 17.8±0.0 | 9.2±3.6 | 6.6±8.6 |
| e<sub>r</sub> (deg) | 54.6±0.0 | 10.7±3.7 | 9.9±11.4 |
| eft (cm) | 4.1±0.0 | 3.3±0.3 | 4.2±0.4 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.02±0.00 | 0.03±0.00 |
| pick_spoon_bowl |  |  |  |
| et (cm) | 9.2±0.0 | 4.9±0.4 | 6.6±1.2 |
| e<sub>r</sub> (deg) | 98.2±0.0 | 43.3±11.1 | 54.0±11.8 |
| eft (cm) | 5.1±0.0 | 1.7±0.5 | 5.9±0.3 |
| e<sub>j</sub> (rad) | 0.28±0.00 | 0.02±0.00 | 0.03±0.00 |
| pour_tube |  |  |  |
| et (cm) | 35.8±0.0 | 28.8±0.6 | 14.6±0.4 |
| e<sub>r</sub> (deg) | 103.4±0.0 | 82.5±10.5 | 55.6±5.9 |
| eft (cm) | 4.1±0.0 | 2.8±0.2 | 3.9±0.2 |
| e<sub>j</sub> (rad) | 0.15±0.00 | 0.02±0.00 | 0.02±0.00 |
| stir_beaker |  |  |  |
| et (cm) | 21.2±0.0 | 10.6±1.7 | 9.9±0.5 |
| e<sub>r</sub> (deg) | 112.1±0.0 | 48.3±8.2 | 35.4±6.1 |
| oakink |  |  |  |
| eft (cm) | 3.3±0.0 | 1.4±0.1 | 2.1±0.2 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.02±0.00 | 0.01±0.00 |
| uncap_alcohol_burner |  |  |  |
| et (cm) | 22.3±0.0 | 3.0±0.8 | 4.5±0.9 |
| e<sub>r</sub> (deg) | 106.4±0.0 | 5.8±0.4 | 16.9±9.1 |
| eft (cm) | 8.0±1.9 | 2.2±0.0 | 3.9±0.2 |
| e<sub>j</sub> (rad) | 0.28±0.11 | 0.02±0.00 | 0.02±0.00 |
| unplug |  |  |  |
| et (cm) | 12.3±7.4 | 1.9±0.1 | 3.8±0.2 |
| e<sub>r</sub> (deg) | 58.8±22.5 | 12.5±0.7 | 11.5±5.2 |
| eft (cm) | 4.1±0.1 | 3.4±0.1 | 4.3±0.1 |
| e<sub>j</sub> (rad) | 0.13±0.04 | 0.03±0.00 | 0.02±0.00 |
| wipe_board |  |  |  |
| et (cm) | 6.0±0.9 | 5.1±0.3 | 8.2±1.0 |
| e<sub>r</sub> (deg) | 71.8±16.8 | 28.3±1.8 | 23.6±9.3 |
| eft (cm) | 4.8±1.4 | 2.6±0.7 | 3.9±1.1 |
| e<sub>j</sub> (rad) | 0.18±0.10 | 0.02±0.00 | 0.02±0.01 |
| Avg. |  |  |  |
| et (cm) | 17.8±9.3 | 9.1±8.5 | 7.8±3.4 |
| e<sub>r</sub> (deg) | 86.5±22.3 | 33.1±25.3 | 29.6±17.8 |
| eft (cm) | 4.4±0.0 | 4.8±0.7 | 4.0±0.2 |
| e<sub>j</sub> (rad) | 0.07±0.00 | 0.05±0.00 | 0.08±0.00 |
| p36-tea |  |  |  |
| et (cm) | 15.2±0.0 | 7.1±9.1 | 14.1±0.9 |
| e<sub>r</sub> (deg) | 80.7±0.0 | 36.1±28.7 | 26.9±1.2 |
| eft (cm) | 4.8±0.1 | 5.5±0.3 | 4.3±0.0 |
| e<sub>j</sub> (rad) | 0.12±0.10 | 0.05±0.00 | 0.05±0.00 |
| p44-dog |  |  |  |
| et (cm) | 45.3±8.5 | 4.5±0.2 | 11.0±0.3 |
| e<sub>r</sub> (deg) | 42.0±5.8 | 6.6±0.3 | 10.2±0.2 |
| gigahand |  |  |  |
| eft (cm) | 4.9±0.0 | 5.9±0.6 | 4.1±0.1 |
| e<sub>j</sub> (rad) | 0.07±0.00 | 0.04±0.00 | 0.07±0.00 |
| p52-instrument |  |  |  |
| et (cm) | 54.5±0.0 | 21.8±3.1 | 16.8±1.1 |
| e<sub>r</sub> (deg) | 54.9±0.0 | 16.9±1.9 | 14.3±0.8 |
| eft (cm) | 4.7±0.2 | 5.4±0.5 | 4.1±0.1 |
| e<sub>j</sub> (rad) | 0.09±0.03 | 0.05±0.00 | 0.07±0.01 |
| Avg. |  |  |  |
| et (cm) | 38.3±16.8 | 11.1±7.6 | 14.0±2.4 |
| e<sub>r</sub> (deg) | 59.2±16.1 | 19.9±12.2 | 17.1±7.1 |

Table 14 Per-trajectory tracking metrics for schunk. Mean ± std across 5 seeds.

28

<!-- page 29 of 35 -->

| Dataset Task Metric | Kinematic | SPIDER | SPIDER+C |
| --- | --- | --- | --- |
| eft (cm) | 6.4±0.1 | 2.4±0.1 | 2.4±0.2 |
| e<sub>j</sub> (rad) | 0.08±0.06 | 0.03±0.00 | 0.04±0.00 |
| lift_board |  |  |  |
| et (cm) | 74.6±18.3 | 1.9±0.5 | 1.5±0.1 |
| e<sub>r</sub> (deg) | 74.3±8.0 | 9.4±0.4 | 4.3±0.7 |
| eft (cm) | 6.1±0.4 | 2.8±0.4 | 4.7±1.0 |
| e<sub>j</sub> (rad) | 0.08±0.07 | 0.02±0.00 | 0.01±0.00 |
| pick_spoon_bowl |  |  |  |
| et (cm) | 29.0±0.8 | 5.4±0.2 | 7.5±1.3 |
| e<sub>r</sub> (deg) | 109.1±5.0 | 21.6±2.8 | 33.2±15.4 |
| eft (cm) | 6.0±0.0 | 1.3±0.7 | 5.2±3.9 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.12±0.12 | 0.08±0.14 |
| pour_tube |  |  |  |
| et (cm) | 51.1±0.0 | 22.7±8.5 | 10.5±3.4 |
| e<sub>r</sub> (deg) | 112.5±0.0 | 95.0±27.6 | 58.6±9.7 |
| eft (cm) | 6.8±0.0 | 2.7±0.5 | 5.0±0.3 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.05±0.05 | 0.03±0.00 |
| stir_beaker |  |  |  |
| et (cm) | 48.2±0.0 | 9.8±2.4 | 7.0±0.9 |
| e<sub>r</sub> (deg) | 105.2±0.0 | 45.0±7.3 | 21.7±0.5 |
| oakink |  |  |  |
| eft (cm) | 5.9±0.0 | 1.6±0.2 | 2.4±0.2 |
| e<sub>j</sub> (rad) | 0.06±0.00 | 0.04±0.00 | 0.02±0.00 |
| uncap_alcohol_burner |  |  |  |
| et (cm) | 8.4±0.0 | 5.1±1.5 | 3.4±0.3 |
| e<sub>r</sub> (deg) | 119.2±0.0 | 22.1±4.3 | 10.8±1.6 |
| eft (cm) | 6.4±0.0 | 1.4±0.1 | 3.8±0.3 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.02±0.00 | 0.03±0.00 |
| unplug |  |  |  |
| et (cm) | 43.7±0.0 | 1.3±0.4 | 3.9±0.7 |
| e<sub>r</sub> (deg) | 125.5±0.0 | 12.4±1.1 | 32.0±12.3 |
| eft (cm) | 7.1±0.0 | 2.5±0.1 | 3.1±0.2 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.03±0.00 | 0.02±0.00 |
| wipe_board |  |  |  |
| et (cm) | 21.5±0.0 | 3.9±0.3 | 2.6±1.2 |
| e<sub>r</sub> (deg) | 113.0±0.0 | 21.5±0.2 | 8.5±6.6 |
| eft (cm) | 6.4±0.4 | 2.1±0.6 | 3.8±1.1 |
| e<sub>j</sub> (rad) | 0.06±0.01 | 0.04±0.03 | 0.03±0.02 |
| Avg. |  |  |  |
| et (cm) | 39.5±20.2 | 7.1±6.8 | 5.2±3.0 |
| e<sub>r</sub> (deg) | 108.4±15.2 | 32.4±27.6 | 24.1±17.5 |
| eft (cm) | 6.9±0.0 | 5.8±0.5 | 4.2±1.0 |
| e<sub>j</sub> (rad) | 0.07±0.00 | 0.07±0.00 | 0.12±0.03 |
| p36-tea |  |  |  |
| et (cm) | 30.6±0.0 | 3.6±0.4 | 8.3±4.2 |
| e<sub>r</sub> (deg) | 114.2±0.0 | 19.4±1.1 | 23.3±4.4 |
| eft (cm) | 5.2±1.2 | 5.7±0.3 | 4.1±0.0 |
| e<sub>j</sub> (rad) | 0.16±0.16 | 0.07±0.00 | 0.07±0.00 |
| p44-dog |  |  |  |
| et (cm) | 97.8±36.5 | 4.8±0.4 | 12.5±0.8 |
| e<sub>r</sub> (deg) | 90.0±31.8 | 5.9±0.4 | 11.7±0.9 |
| gigahand |  |  |  |
| eft (cm) | 1.7±0.0 | 4.3±1.2 | 2.9±0.2 |
| e<sub>j</sub> (rad) | 0.05±0.00 | 0.07±0.01 | 0.06±0.00 |
| p52-instrument |  |  |  |
| et (cm) | 10.7±0.0 | 5.1±1.3 | 3.4±0.7 |
| e<sub>r</sub> (deg) | 16.0±0.0 | 17.5±2.7 | 4.7±0.8 |
| eft (cm) | 4.6±2.2 | 5.3±0.7 | 3.7±0.6 |
| e<sub>j</sub> (rad) | 0.10±0.04 | 0.07±0.00 | 0.09±0.03 |
| Avg. |  |  |  |
| et (cm) | 46.4±37.3 | 4.5±0.7 | 8.1±3.7 |
| e<sub>r</sub> (deg) | 73.4±41.8 | 14.3±6.0 | 13.2±7.7 |

Table 15 Per-trajectory tracking metrics for xhand. Mean ± std across 5 seeds.

29

<!-- page 30 of 35 -->

Table 16 SPIDER hyperparameters for the ManipTrans (IsaacGym) experiments.

| Parameter | Symbol | Value |
| --- | --- | --- |
| Simulation |  |  |
| Simulator | - | IsaacGym (GPU PhysX) |
| Simulation timestep | ∆t<sub>sim</sub> | 1/120 s |
| Control timestep | ∆t<sub>ctrl</sub> | 0.2 s |
| Knot point spacing | ∆t<sub>knot</sub> | 0.2 s |
| Action dimensions | n<sub>u</sub> | 18 |
| Sampling-based Optimization |  |  |
| Planning horizon | H | 0.6 s (36 sim steps) |
| Parallel samples | N<sub>w</sub> | 1024 |
| Max sampling iterations | N | 32 |
| Early stopping threshold | ϵ | 0.001 |
| Early stopping patience | - | 4 iterations |
| Noise Schedule |  |  |
| Initial noise scale | σfirst | 0.5 |
| Final noise scale (horizon end) | σlast | 1.0 |
| Noise decay per iteration | β | 0.93 |
| Wrist pos/rot noise scale | - | 0.0 |
| Joint noise scale | - | 1.0 |
| Reward |  |  |
| Terminal reward scale | α<sub>T</sub> | 10.0 |
| Sampling temperature | τ | 1.0 |
| ManipTrans-specific |  |  |
| Actions moving average | γ | 0.4 |

## B.3 DexMachina Retargeting Results

We evaluate **SPIDER** on bimanual dexterous manipulation tasks from the ARCTIC dataset (Fan et al., 2023), using the **DexMachina** framework (Mandi et al., 2025) and the Genesis physics engine. We use the Inspire dexterous hand in a bimanual configuration (two hands manipulating articulated objects simultaneously). To ensure a fair comparison, **SPIDER** directly uses the same RL training environment and reward function as **DexMachina**, and all evaluation metrics are computed identically for both methods. We remove the object virtual joint curriculum from **DexMachina**.

**RL Baseline Training.** We use the **DexMachina** RL policy as our baseline. **DexMachina** trains one RL policy per ARCTIC clip using the Genesis physics engine with the Inspire hand in a bimanual configuration. The action space consists of 36 dimensions in hybrid mode: for each hand, 6 wrist force/torque dimensions (3 position + 3 rotation) and 12 finger joint position targets. The RL policy is deterministic at evaluation time; since the initial state is fixed, the RL policy produces identical rollouts across random seeds.

**RL Evaluation.** We evaluate the PPO policy once per task with 5 seeds and report the mean and std of the metrics.

**Reward Setting.** To ensure **SPIDER** optimizes the same objective as the RL baseline, we delegate reward computation to **DexMachina**’s reward function but remove the action penalty term for efficiency (since our action is smooth by interpolation).

**Curriculum Learning.** In **SPIDER** retargeting, we disable curriculum learning in **DexMachina**, so the object joints are always free (passive) throughout optimization. For **DexMachina**, we use the default curriculum learning settings where object joint stiffness and damping are annealed from high values (providing a guiding force) to zero across training iterations.

**SPIDER Retargeting.** We run **SPIDER** with 1024 parallel Genesis environments. **SPIDER** optimizes the full

30

<!-- page 31 of 35 -->

36-dim action space (both hands’ wrist and finger actions) using sampling-based MPC. Each task is run with 5 random seeds. All hyperparameters are listed in Table 18.

**Evaluation Metrics.** We evaluate object tracking performance using three metrics computed at each simulation timestep in the original **DexMachina** paper (Mandi et al., 2025), comparing the simulated object state against the ground-truth reference trajectory from the ARCTIC dataset:

$d _ { \mathrm { p o s } } \mathrm { : }$ L2 position distance (m) between simulated and reference object positions, $d _ { \mathrm { p o s } } = \| \mathbf { p } - \mathbf { p } ^ { * } \| _ { 2 }$

$d _ { \mathrm { r o t } } \colon$ geodesic rotation distance (rad) on SO(3) between simulated and reference orientations, computed as $d _ { \operatorname { r o t } } = 2 \operatorname { a r c s i n } \bigl ( \operatorname* { m i n } ( \| \operatorname { I m } ( \mathbf { q } \mathop { { \otimes } } \bar { \mathbf { q } } ^ { * } ) \| _ { 2 } , \; 1 ) \bigr )$ , where q and $\mathbf { q } ^ { * }$ are unit quaternions in $( w , x , y , z )$ convention and $\bar { \mathbf { q } } ^ { * }$ denotes the conjugate.

$d _ { \mathrm { a r t i } } \mathrm { : }$ articulation distance (rad), the absolute difference in the object’s articulated joint angle, $d _ { \mathrm { a r t i } } =$ $| a - a ^ { * } |$

All metrics are averaged over the trajectory length, then reported as mean ± std across 5 seeds per task for **SPIDER**. For the RL baseline, we report the single deterministic evaluation, as it produces identical rollouts across seeds ${ \big ( } \mathrm { s t d } = 0 { \big ) }$ . In the original **DexMachina** paper, they also report the success rate by computing Average Distance Area Under the Curve (ADD-AUC) for each object part separately. However, since this metric is not open source, we directly report the raw metrics for comparison.

**Experiment Results.** We evaluate on 7 bimanual dexterous manipulation tasks from the ARCTIC dataset (box, ketchup, ketchup-u02, laptop, mixer, notebook, waffleiron), each defined by a 200-frame clip at 50 Hz. Each task is named in the format of &lt;object-name&gt;-&lt;start-frame&gt;-&lt;end-frame&gt;, similar to the original **DexMachina** paper. Full results are shown in Table 19.

**SPIDER** wins on $6 / 7$ tasks for $d _ { \mathrm { p o s } } ,   5 / 7$ for $d _ { \mathrm { r o t } }$ , and $5 / 7$ for $d _ { \mathrm { a r t i } }$ . On average, **SPIDER** achieves lower position error (0.421 vs. 0.517 m) and rotation error (0.959 vs. 1.099 rad), while the RL baseline achieves slightly lower articulation error (0.622 vs. 0.643 rad). Notably, **SPIDER** achieves these results without any task-specific training, using only test-time optimization with the same reward function as the RL baseline.

## C Virtual Contact Guidance Details

**Solution Ambiguity.** For a retargeting method to be valid, it is also important to preserve the human-preferred contact in the demonstration. Take the stir stick manipulation in Figure 5 as an example: the robot can hold the stick either between the thumb and index finger or between the index and middle finger; both achieve the task, but the former is more natural and aligns with the human demonstration. However, due to the non-convex cost landscape of Equation (1a), there may be multiple solutions that achieve similar object motion with different contact behaviors. As a result, a sampling-based optimizer may converge to alternative contact modes, leading to implausible contact trajectories in a demonstration even when the object trajectory is reproduced.

**Virtual Contact Guidance.** To guide sampling towards the desired contact mode, we introduce virtual contact guidance. This guidance factor expands the feasible solution set and biases the optimization process towards the target configuration. Unlike a soft contact reward or cost - which may still fail when the desired mode is harder to sample - virtual contact guidance explicitly enlarges the basin of attraction to make it easier to sample from, as illustrated in Figure 5. Our method applies a virtual constraint between intended contact points on the object and robot fingers, “sticking” the object to the target configuration in early stages and gradually relaxing this constraint in a curriculum-like fashion. Concretely, we impose a virtual constraint that maintains the relative position between contact pairs. For the k-th object-hand contact pair (e.g., handle-thumb), we define the current relative position as robot ${ } ^ { t } p _ { k , t } ^ { \mathrm { o b j e c t } } = \left. p _ { k , t } ^ { \mathrm { r o b o t } } \right. ^ { \circ } - \left. p _ { k , t } ^ { \mathrm { o b j e c t } } \right.$ and the desired reference position as robot $\begin{array} { r } { ^ { ! } p _ { k , t } ^ { \mathrm { o b j e c t , r e f } }   =   p _ { k , t } ^ { \mathrm { r o b o t , r e f } } - p _ { k , t } ^ { \mathrm { o b j e c t , r e f } } } \end{array}$ The constraint is activated when the contact indicator $c _ { k , t } = 1$ , which occurs when the reference relative position is within the contact threshold: $c _ { k , t } = \mathbf { 1 } ( \lVert ^ { \mathrm { r o b o t } } p _ { k , t } ^ { \mathrm { o b j e c t , r e f } } \rVert _ { 2 } \leq \epsilon _ { \mathrm { c o n t a c t } } )$ . The constraint strength is controlled by a penalty parameter $\eta _ { i } \to \infty$ when $i \rightarrow N$

$$
c _ { k , t } \| { } ^ { \mathrm { r o b o t } } p _ { k , t } ^ { \mathrm { o b j e c t } } - { } ^ { \mathrm { r o b o t } } p _ { k , t } ^ { \mathrm { o b j e c t , r e f } } \| _ { 2 } ^ { 2 } \leq \eta _ { i }\tag{7}
$$

31

<!-- page 32 of 35 -->

This strategy connects to prior ideas: virtual object constraint (Mandi et al., 2025) $\| p _ { k , t } ^ { \mathrm { o b j e c t } } - p _ { k , t } ^ { \mathrm { o b j e c t , r e f } } \| _ { 2 } ^ { 2 } \leq \eta ,$ which expands feasibility around absolute object states in RL, and contact cost (Lakshmipathy et al., 2025), biasing optimization towards desired hand–object relative states. Our formulation reduces sampling complexity while preserving the intended contact sequence by maintaining relative hand–object relationships.

**Robustness against imperfect reference contact.** It is important to note that, in order to maintain robustness against imperfect demonstrations with noisy or unstable contact, virtual constraints should be selectively relaxed rather than enforced indiscriminately. To achieve this, we apply a contact filter that detects unstable interactions: if a contact duration is shorter than $t _ { c , \mathrm { m i n } }$ or if the contact point drifts more than $d _ { c , \operatorname* { m a x } }$ during that period, the contact is classified as unstable and the corresponding virtual constraint is disabled. This implementation ensures that only reliable contacts contribute to guidance, thereby preventing noisy demonstrations from biasing the optimization process.

32

<!-- page 33 of 35 -->

Table 17 Multi-seed comparison between SPIDER and ManipTrans RL baseline on 16 trajectories from Oakink dataset. Each method is evaluated with 5 seeds per trajectory. Metrics: $e _ { t } = \mathrm { o b j e c t }$ translation error (cm), e<sub>r</sub> = object rotation error (deg), e<sub>j</sub> = hand joint position error (cm), eft = fingertip position error (cm). Values reported as mean ± std.

| Data ID Side Method | e<sub>t</sub> (cm) ↓ | e<sub>r</sub> (deg) ↓ | e<sub>j</sub> (cm) ↓ | eft (cm) ↓ |
| --- | --- | --- | --- | --- |
| SPIDER | 34.81 ± 0.30 | 101.76 ± 1.12 | 4.73 ± 0.01 | 4.10 ± 0.01 |
| 01857@10 right |  |  |  |  |
| ManipTrans | 34.50 ± 3.34 | 112.67 ± 19.26 | 25.98 ± 19.13 | 25.59 ± 18.77 |
| SPIDER | 40.74 ± 0.74 | 47.17 ± 22.58 | 5.03 ± 0.01 | 3.95 ± 0.01 |
| 031b7@0 right |  |  |  |  |
| ManipTrans | 40.17 ± 0.07 | 35.20 ± 3.39 | 101.47 ± 2.08 | 98.49 ± 2.02 |
| SPIDER | 44.27 ± 0.00 | 80.72 ± 0.00 | 4.80 ± 0.00 | 3.94 ± 0.00 |
| 03ac9@0 right |  |  |  |  |
| ManipTrans | 44.31 ± 0.00 | 80.54 ± 0.00 | 293.28 ± 20.40 | 294.40 ± 20.62 |
| SPIDER | 25.96 ± 0.02 | 27.95 ± 0.67 | 5.09 ± 0.01 | 4.05 ± 0.00 |
| 07301@0 right |  |  |  |  |
| ManipTrans | 13.40 ± 4.65 | 137.87 ± 2.71 | 10.20 ± 1.02 | 10.03 ± 1.07 |
| SPIDER | 31.09 ± 15.33 | 29.26 ± 3.27 | 4.86 ± 0.06 | 4.03 ± 0.07 |
| 07bb1@1 right |  |  |  |  |
| ManipTrans | 34.43 ± 2.16 | 57.47 ± 40.98 | 11.12 ± 2.91 | 10.04 ± 3.18 |
| SPIDER | 26.88 ± 0.01 | 40.46 ± 0.16 | 4.79 ± 0.00 | 3.96 ± 0.00 |
| 0b3a0@6 left |  |  |  |  |
| ManipTrans | 25.64 ± 0.38 | 73.22 ± 10.33 | 8.05 ± 0.05 | 7.63 ± 0.05 |
| SPIDER | 5.99 ± 2.61 | 56.57 ± 10.51 | 4.57 ± 0.07 | 3.94 ± 0.09 |
| 0f4a7@1 left |  |  |  |  |
| ManipTrans | 21.33 ± 14.55 | 57.51 ± 12.53 | 9.93 ± 2.07 | 9.68 ± 2.29 |
| SPIDER | 20.56 ± 10.22 | 52.75 ± 31.28 | 4.45 ± 0.01 | 3.89 ± 0.04 |
| 0f900@0 left |  |  |  |  |
| ManipTrans | 11.66 ± 10.62 | 58.05 ± 41.20 | 7.19 ± 2.73 | 6.34 ± 3.06 |
| SPIDER | 45.45 ± 0.00 | 51.92 ± 0.00 | 4.89 ± 0.02 | 3.98 ± 0.01 |
| 155cb@0 right |  |  |  |  |
| ManipTrans | 45.41 ± 0.00 | 51.81 ± 0.00 | 54.05 ± 1.19 | 52.83 ± 1.17 |
| SPIDER | 22.06 ± 0.40 | 59.61 ± 2.24 | 4.94 ± 0.00 | 3.91 ± 0.00 |
| 235ac@0 right |  |  |  |  |
| ManipTrans | 17.04 ± 3.14 | 70.61 ± 44.86 | 21.54 ± 7.76 | 23.50 ± 6.88 |
| SPIDER | 14.00 ± 0.00 | 85.78 ± 0.00 | 4.60 ± 0.01 | 3.96 ± 0.01 |
| 2ddc8@2 left |  |  |  |  |
| ManipTrans | 14.03 ± 0.00 | 85.77 ± 0.00 | 6.40 ± 0.03 | 5.36 ± 0.02 |
| SPIDER | 11.36 ± 4.52 | 23.57 ± 12.31 | 5.82 ± 0.57 | 5.87 ± 0.81 |
| 380c7@0 left |  |  |  |  |
| ManipTrans | 16.97 ± 0.17 | 15.78 ± 8.39 | 19.06 ± 0.81 | 20.73 ± 0.51 |
| SPIDER | 12.64 ± 0.01 | 33.06 ± 0.30 | 4.73 ± 0.02 | 4.01 ± 0.02 |
| 39e5e@5 left |  |  |  |  |
| ManipTrans | 18.61 ± 4.90 | 99.13 ± 26.68 | 19.24 ± 2.30 | 18.77 ± 2.90 |
| SPIDER | 21.04 ± 9.41 | 49.52 ± 3.41 | 4.64 ± 0.12 | 4.03 ± 0.18 |
| 49793@0 right |  |  |  |  |
| ManipTrans | 23.99 ± 3.82 | 122.98 ± 20.92 | 1618.07 ± 215.12 | 1617.23 ± 214.88 |
| SPIDER | 6.25 ± 4.36 | 85.93 ± 13.46 | 4.55 ± 0.02 | 3.99 ± 0.04 |
| 70ca2@0 left |  |  |  |  |
| ManipTrans | 3.93 ± 0.12 | 50.81 ± 5.08 | 11.91 ± 3.74 | 10.67 ± 3.93 |
| SPIDER | 6.77 ± 5.14 | 44.97 ± 21.95 | 4.90 ± 0.13 | 3.71 ± 0.04 |
| 81a95@5 left |  |  |  |  |
| ManipTrans | 3.13 ± 2.85 | 38.57 ± 25.20 | 6.61 ± 2.34 | 4.88 ± 2.33 |
| SPIDER | 23.12 ± 12.91 | 54.44 ± 22.43 | 4.84 ± 0.31 | 4.08 ± 0.47 |
| Overall - |  |  |  |  |
| ManipTrans | 23.03 ± 12.93 | 71.75 ± 32.30 | 139.01 ± 388.16 | 138.51 ± 388.12 |

33

<!-- page 34 of 35 -->

Table 18 SPIDER hyperparameters for the DexMachina (Genesis) experiments.

| Parameter | Symbol | Value |
| --- | --- | --- |
| Simulation |  |  |
| Simulator | - | Genesis (GPU) |
| Simulation timestep | ∆t<sub>sim</sub> | 0.02 s (50 Hz) |
| Control timestep | ∆t<sub>ctrl</sub> | 0.2 s (5 Hz) |
| Knot point spacing | ∆t<sub>knot</sub> | 0.2 s |
| Action dimensions | n<sub>u</sub> | 36 |
| Sampling-based Optimization |  |  |
| Planning horizon | H | 0.4 s (20 sim steps) |
| Parallel samples | N<sub>w</sub> | 1024 |
| Max sampling iterations | N | 32 |
| Early stopping threshold | ϵ | 0.01 |
| Early stopping patience | - | 4 iterations |
| Noise Schedule |  |  |
| Initial noise scale | σfirst | 0.5 |
| Final noise scale (horizon end) | σlast | 1.0 |
| Noise decay per iteration | β | 0.87 |
| Wrist pos/rot noise scale | - | 1.0 |
| Joint noise scale | - | 1.0 |
| Reward |  |  |
| Reward mode | - | RL training reward |
| Terminal reward scale | α<sub>T</sub> | 10.0 |
| Sampling temperature | τ | 1.0 |
| DexMachina-specific |  |  |
| Imitation reward weight | wimi | 0.3 |
| Contact reward weight | w<sub>con</sub> | 3.0 |
| BC reward weight | wbc | 0.3 |
| Object joint stiffness | k<sub>p</sub> | 0.0 (no curriculum) |
| Object joint damping | k<sub>v</sub> | 0.0 (no curriculum) |

34

<!-- page 35 of 35 -->

Table 19 Multi-seed comparison between SPIDER and DexMachina RL baseline on 7 bimanual tasks from the ARCTIC dataset. Both methods are evaluated with 5 seeds per task. Metrics: $d _ { \mathrm { p o s } } = \mathrm { o b j e c t }$ position distance (m), $d _ { \mathrm { r o t } } = \mathrm { o b j e c t }$ rotation distance (rad), $d _ { \mathrm { a r t i } } =$ articulation distance (rad). Values reported as mean $\pm \mathrm { s t d }$ . Best per task in bold.

| Task Method | d<sub>pos</sub> (m) ↓ | d<sub>rot</sub> (rad) ↓ | d<sub>arti</sub> (rad) ↓ |
| --- | --- | --- | --- |
| SPIDER | 0.0172 ± 0.0069 | 0.0988 ± 0.0198 | 0.0929 ± 0.0801 |
| box-30-230 |  |  |  |
| DexMachina | 0.0906 ± 0.0373 | 0.3709 ± 0.1155 | 0.2897 ± 0.0522 |
| SPIDER | 0.8917 ± 0.1311 | 1.7862 ± 0.1723 | 0.7836 ± 0.0850 |
| ketchup-30-230 |  |  |  |
| DexMachina | 1.1865 ± 0.3021 | 2.0616 ± 0.1768 | 0.8057 ± 0.0806 |
| SPIDER | 0.7573 ± 0.3397 | 1.5418 ± 0.3500 | 1.1717 ± 0.0227 |
| ketchup-30-230-s01-u02 |  |  |  |
| DexMachina | 1.1569 ± 0.2231 | 2.0379 ± 0.2021 | 1.1863 ± 0.0851 |
| SPIDER | 0.0427 ± 0.0171 | 0.2155 ± 0.0726 | 1.0548 ± 0.5011 |
| laptop-30-230 |  |  |  |
| DexMachina | 0.0573 ± 0.0840 | 0.1980 ± 0.2344 | 0.4094 ± 0.3296 |
| SPIDER | 0.0176 ± 0.0098 | 0.1203 ± 0.0360 | 0.1440 ± 0.0118 |
| mixer-30-230 |  |  |  |
| DexMachina | 0.1211 ± 0.1563 | 0.7360 ± 0.2814 | 0.3654 ± 0.1230 |
| SPIDER | 0.9968 ± 0.0245 | 1.9491 ± 0.2248 | 0.7024 ± 0.0007 |
| notebook-30-230 |  |  |  |
| DexMachina | 0.4969 ± 0.0682 | 1.1062 ± 0.2850 | 0.7468 ± 0.1131 |
| SPIDER | 0.2224 ± 0.1585 | 1.0033 ± 0.2985 | 0.5528 ± 0.3142 |
| waffleiron-30-230 |  |  |  |
| DexMachina | 0.5113 ± 0.0682 | 1.1791 ± 0.0903 | 0.5480 ± 0.0753 |
| SPIDER | 0.4208 ± 0.4096 | 0.9593 ± 0.7562 | 0.6432 ± 0.3840 |
| Overall |  |  |  |
| DexMachina | 0.5172 ± 0.4485 | 1.0985 ± 0.6854 | 0.6216 ± 0.2915 |

35