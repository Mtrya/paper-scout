<!-- page 1 of 9 -->

# MorphMimic: Morphology-Conditioned World Model and Policy Sharing Across Humanoids

Lizhi Yang<sup>1</sup>, Fengze Xie<sup>1,∗</sup>, Yuyang Wang<sup>1,∗</sup>, Lu Gan<sup>2</sup>, Yisong Yue<sup>1</sup>, and Aaron D. Ames<sup>1</sup>

**Abstract—As more humanoid platforms become available, training a separate whole-body motion tracker for each robot becomes costly. We present MorphMimic, which trains one world model and one policy that takes the embodiment as input. A canonical joint and body layout maps the Unitree G1, AgiBot X2 Ultra, and Booster T2 to a common coordinate system. A physical descriptor of actuation, geometry, and mass conditions the world model, policy decoder, and critics based on the robot being controlled. An alignment loss brings the latent commands for corresponding motions closer across robots, while a second encoder learns to reproduce those commands from pelvis, ankle, and wrist targets through the same decoder. Both interfaces use position-free targets with joint-encoder and inertial orientation feedback. Across four sharing configurations evaluated on 128,786 common clips, every shared configuration stays within 0.02 mean return of a single-robot specialist on all six robot– interface pairs. On a physical Unitree G1, the same unified policy performs dance, walking, and real-time teleoperation. Code and video are available at** [**https://lzyang2000.github.io/morphmimic**](https://lzyang2000.github.io/morphmimic)**.**

**Index Terms—Humanoid robot systems, reinforcement learning, transfer learning, whole-body motion planning and control.**

## I. INTRODUCTION

<sup>HOLE-BODY</sup> motion tracking converts human ref-W erence motion into humanoid behavior and supports teleoperation and command interfaces built on a tracking policy [1], [2]. More humanoid platforms are now available, and many have similar sizes and kinematic layouts. Wholebody trackers are usually trained for one robot [1], [2], so a new embodiment requires retraining or adaptation [3]. A controller that takes the embodiment as an input would let a new humanoid use the motion repertoire, training procedure, and command interfaces developed for the others. Experience collected on one robot could then benefit the rest.

Sharing a controller across humanoids is difficult because similar kinematic structures can have different physics. Joint axes, sign conventions, neutral poses, limb lengths, proportional–derivative (PD) gains, and torque limits differ, so the same action produces different motions on different bodies. The problem is harder when the policy learns through a world model. SuperMimic [2] trains a tracker by differentiating through short rollouts of a learned dynamics model. Prediction errors therefore affect both the imagined trajectory and the policy gradient, and a shared model must predict each robot well enough for that gradient to remain useful. Prior cross-embodiment methods condition policies on

![Image block](doc:b4259b6/tier:basic/page:1/block:8)

Figure 1. One world model and one policy shared across humanoids. Fullmotion latents are aligned across robots, and a five-point encoder learns to match them through the same decoder. Morphology conditions the world model and decoder; the dashed arrow marks imagined rollouts. Robots are simulator renders at a common scale; latent cells are schematic.

hardware descriptions [4], share locomotion policies through morphology-aware encoding or semantic joint alignment [5]– [7], condition world models on morphology across robot bodies [8], [9], or transfer and co-train humanoid trackers with robot-specific modules [3], [10]. We address this setting by sharing one differentiable world model and one complete tracking policy across humanoids.

MorphMimic treats the embodiment as an input to one world model and one policy (Fig. 1). A canonical layout maps joints by kinematic function, with joint-specific signs and offsets, to a common joint order and set of tracked bodies. A physical descriptor of actuation, geometry, and mass conditions the world model, critics, and policy decoder. Network weights are shared, while normalization statistics, reference motions, and replay data remain robot-specific. A new humanoid that fits the layout can therefore be added with a joint map, a descriptor, and retargeted motions. An alignment loss pulls the latent commands for corresponding retargeted frames toward their cross-robot mean. A five-point encoder for pelvis, ankle, and wrist targets learns to reproduce the fullmotion latents of its robot and drives every robot through the shared decoder. Other interfaces can target the same latent

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>∗</sup>Equal contribution. <sup>1</sup>California Institute of Technology, CA 91106, USA. <sup>2</sup>Georgia Institute of Technology, Atlanta, GA 30332, USA. Corresponding author email: [lzyang@caltech.edu.](mailto:lzyang@caltech.edu) This research is funded by The Dow Chemical Company project #227027AW.</span></small>

<!-- page 2 of 9 -->

![Image block](doc:b4259b6/tier:basic/page:2/block:1)

Figure 2. Shared policy and command interfaces. Robot-specific normalizers feed a shared motion encoder, whose 32-dimensional latent enters the shared decoder with proprioception and morphology; morphology also modulates decoder features through FiLM. Motion latents of corresponding frames align to their mean, and the five-point encoder matches its robot’s motion latent. Latent bars are schematic.

space. Both interfaces use position-free targets with feedback from joint encoders and the inertial measurement unit (IMU).

We evaluate four sharing configurations on the 128,786 clips common to the three training libraries. Every shared configuration stays within 0.02 mean return of a single-robot specialist on all six robot–interface pairs. The unified run uses 16.3 GPU-hours, compared with 13.9 GPU-hours for the three specialists together, and the resulting policy performs dance, walking, and real-time teleoperation on a physical G1 (Fig. 5).

Our contributions are a common robot representation and morphology-conditioned training for one differentiable world model and one whole-body tracking policy across humanoids with different joint conventions, geometry, and actuation. We also introduce cross-robot alignment of full-motion latent commands and a five-point interface that uses the shared decoder, both with position-free targets. A full-corpus simulation compares the shared policy with single-robot specialists through both command interfaces, and hardware experiments on a physical humanoid evaluate dance, walking, and teleoperation with the shared policy.

## II. RELATED WORK

## A. Cross-Embodiment Control Policies

Sharing control policies across robot bodies has a long history. Chen et al. [4] condition a universal policy on a vector encoding the robot hardware. Shared Modular Policies [11] reuse an actuator-level controller across bodies and coordinate modules through message passing. MetaMorph [12] trains morphology-conditioned Transformers across modular

robot designs and transfers them to unseen morphologies. For dynamic legged locomotion, GenLoco [13] combines motion imitation with procedurally generated quadruped morphologies and transfers to unseen simulated and real robots, using sensor and action histories to infer the body. URMA [5] shares a locomotion policy across embodiments with different joint configurations through attention-based observation encoding and joint-conditioned action decoding. Bohlinger et al. [6] extend this idea with extreme Embodiment Randomization across 50 legged robots, including zero-shot transfer to unseen humanoids and quadrupeds. XHugWBC [7] combines semantic joint alignment with physics-consistent morphological randomization for cross-humanoid locomotion. MS-PPO [14] encodes the permutations and sign flips induced by the morphological symmetry of a single robot in a graph policy, using a symmetry-equivariant actor and symmetry-invariant critic for locomotion on the Unitree Go2 and G1. We apply explicit morphology conditioning to whole-body tracking with a fixed semantic joint layout and a decoder conditioned on geometry and actuation. The same physical descriptor also conditions the learned dynamics.

## B. Cross-Embodiment Whole-Body Tracking

Whole-body tracking across bodies also needs a command representation that relates motions between robots. Any2Any [3] transfers a pretrained humanoid tracker to a new body through kinematic alignment and parameter-efficient adaptation of dynamics-sensitive policy modules. X-WBC [10] jointly trains a shared temporal backbone with robot-specific

<!-- page 3 of 9 -->

command encoders, proprioception encoders, and action decoders. It aligns command tokens from robot, human, and sparse motion inputs, separating the requested motion from its execution by a particular robot. In MorphMimic, the goal encoders and action decoder are shared across robots. Corresponding retargeted frames supervise alignment of motion latents, and the five-point encoder learns to reproduce each robot’s full-motion latent so sparse targets can use the shared decoder.

## C. World Models for Policy Learning

World models offer another way to share information across bodies. PWM [15] learns policies through differentiable multi-task world models. MS-HGNN [16] incorporates kinematic structure and morphological symmetry into a heterogeneous graph network for robot dynamics. QWM [8] conditions a recurrent latent world model on physical morphology, learns quadrupedal locomotion policies through imagined rollouts, and reports zero-shot transfer to held-out quadrupedal morphologies. WestWorld [9] scales a trajectory world model across diverse robotic systems with a system-aware mixture of experts and a morphology-based structural embedding. SuperMimic [2] applies differentiable world-model learning to humanoid whole-body tracking and supports additional command encoders. MorphMimic extends this setting from one body to several humanoids by sharing the state-space dynamics model and tracking policy through a common representation, morphology conditioning, and command alignment. Both command interfaces use one morphology-conditioned action decoder.

## III. METHODOLOGY

MorphMimic follows SuperMimic’s alternating loop of simulator data collection, world-model fitting, and policy updates through imagined rollouts [2, Sections 3.1–3.3]. Fig. 2 shows the shared loop across robots. Robot-specific details and constants are given in Appendix B.

## A. Problem Formulation

Let $b \; \in \; { \mathcal { B } } \; = \; \{ \mathrm { G 1 } , \mathrm { X 2 } , \mathrm { T 2 } \}$ denote the embodiment, $s _ { t } ^ { b }$ its state, and $a _ { t }   \in   \mathbb { R } ^ { N _ { j } }$ the canonical action over $N _ { j }$ body joints. The simulator transition is $s _ { t + 1 } ^ { b } \sim \mathit { P } _ { b } ( \cdot | s _ { t } ^ { b } , \vec { a _ { t } } )$ with control interval $\Delta t$ . A reference motion supplies a target state $\bar { s } _ { t } ^ { b }$ and a goal $g _ { t } ^ { b }$ constructed from a window of $K$ target frames, and the policy uses a history $o _ { t - K + 1 : t } ^ { b }$ of the last K proprioceptive observations. With the discount factor $\gamma$ , we seek shared policy parameters θ that maximize the equally weighted tracking objective

$$
J ( \theta ) = \frac { 1 } { | \mathcal { B } | } \sum _ { b \in \mathcal { B } } \mathbb { E } \Bigg [ \sum _ { t = 0 } ^ { T _ { b } - 1 } \gamma ^ { t } r _ { t + 1 } ^ { b } \Bigg ] \: ,\tag{1}
$$

where $T _ { b }$ is the episode length and $r _ { t + 1 } ^ { b }$ evaluates the state reached by action $a _ { t }$ against the target $\bar { s } _ { t } ^ { b }$ .

We use SuperMimic’s motion-tracking reward and weights [2, Appendix $\mathrm { A } ,$ Table A.2] and evaluate them on the $N _ { b }$ canonical body slots of Section III-B with the

torso-equivalent body as the tracking anchor, and denote the maximum reward per step by $r _ { \mathrm { m a x } }$ . An episode terminates when the anchor-height error $e _ { z }$ exceeds $\bar { e } _ { z }$ or the tilt error $e _ { u }$ exceeds $\bar { e } _ { u } ,$ where the tilt error is the absolute difference between the vertical gravity components in the robot and reference anchor frames. Reaching the end of the reference counts as a time-out, and all three robots share these tolerances. Appendix B summarizes the simulated-body setup.

## B. Common Robot Representation

1) Canonical Joints and Actuator Targets: We map joints by kinematic function. For canonical slot $j ,$ let $p _ { b } ( j )$ be the corresponding native joint index, $\epsilon _ { b , j } \in \{ - 1 , 1 \}$ its sign, and $\delta _ { b , j }$ its angular offset. A native joint position becomes

$$
q _ { j } ^ { c } = \epsilon _ { b , j } \big ( q _ { p _ { b } ( j ) } ^ { b } - \delta _ { b , j } \big ) .\tag{2}
$$

Velocities and other joint differences use the permutation and sign without the offset. This map aligns joint slots across the naming, sign, and neutral-pose conventions of the robots. The learned networks account for the remaining kinematic differences. With the effort limit $\tau _ { b , j } ^ { \mathrm { m a x } }$ , the proportional gain $k _ { p , b , j } ,$ and an action scale $\kappa _ { a } ,$ , the action is a scaled displacement from the robot’s default pose, with native position target

$$
q _ { p _ { b } ( j ) } ^ { b , \mathrm { t a r g e t } } = q _ { p _ { b } ( j ) } ^ { b , \mathrm { d e f a u l t } } + \epsilon _ { b , j } \alpha _ { b , j } a _ { t , j } , \qquad \alpha _ { b , j } = \frac { \kappa _ { a }   \tau _ { b , j } ^ { \mathrm { m a x } } } { k _ { p , b , j } } .\tag{3}
$$

Since $k _ { p , b , j } \alpha _ { b , j } \: = \: \kappa _ { a } \tau _ { b , j } ^ { \operatorname* { m a x } }$ , a unit action changes the proportional torque at a fixed joint state by the fraction $\kappa _ { a }$ of the effort limit. The displacement in (3) is measured from the native default pose, so the offset in (2) affects only the absolute joint configuration.

2) Tracked Bodies and State: The $N _ { b }$ body slots contain a pelvis equivalent, three bodies on each leg, a torso equivalent, and three bodies on each arm. The pelvis and torso occupy slots 0 and 7. Gravity and angular velocity are measured at each robot’s IMU body. The world-frame state contains joint positions and velocities, together with the positions, quaternion orientations, and linear and angular velocities of the tracked bodies. Let $p _ { P , t }$ denote the world-frame position of the pelvis equivalent, and let $F _ { t }$ be the frame with origin $( p _ { P , t , x } , p _ { P , t , y } , 0 )$ and the full orientation of the pelvis equivalent. The transform $\Phi _ { F _ { t } }$ expresses body positions relative to this origin and rotates positions, orientations, and velocities into $F _ { t } ,$ , while leaving canonical joint coordinates unchanged. Each orientation is represented by the first two columns of its rotation matrix. A separate context $\xi _ { b } \in \mathbb { R } ^ { d _ { \xi } }$ contains a footfriction coefficient, joint encoder biases, and a center-of-mass offset.

3) Observations and Position-Free Goals: The policy uses SuperMimic’s proprioceptive observation [2, Appendix $\mathrm { A } ,$ Table $[ \mathrm { A . 1 ] }$ over a K-frame history. Joint positions, measured relative to the default pose and including encoder bias, and joint velocities are expressed in canonical coordinates. Angular velocity and projected gravity are expressed in the IMU-body frame, and the previous canonical action completes the observation. The history is normalized by a robot-specific transform $\mathcal { N } _ { o } ^ { b }$

<!-- page 4 of 9 -->

The full-motion goal concatenates K target frames, each holding the reference joint positions and velocities and the position and six-dimensional orientation of the torso anchor (slot 7). The frames are the current reference frame and the $K - 1$ subsequent frames at the control rate; the final frame is repeated at the clip boundary. Anchor positions are expressed in the reference pelvis heading frame at the first goal frame. Its origin is the horizontal position of the reference pelvis at zero height. This frame describes the path ahead and the absolute target height without depending on the robot’s position. Anchor orientations are expressed relative to the robot’s current torso anchor and provide orientation feedback, including yaw. The goal is normalized by $\mathcal { N } _ { g } ^ { b }$ . With these position-free goals, the policy inputs are the reference targets, joint-encoder measurements, angular velocity, and an IMU-based attitude estimate. All of these inputs are available from onboard sensing and the reference, which simplifies deployment.

4) Morphology Descriptor: The static descriptor $m _ { b } \in$ $\mathbb { R } ^ { d _ { m } }$ of dimension $d _ { m }$ is

$$
m _ { b } = \mathcal { S } ( [ \operatorname { l o g } \mathbf { \alpha } _ { b } , \operatorname { l o g } \mathbf { k } _ { p , b } , \operatorname { l o g } \mathbf { k } _ { d , b } , \operatorname { l o g } \mathbf { \tau } _ { b } ^ { \operatorname* { m a x } } , \eta _ { b } ] ) \thinspace ,\tag{4}
$$

where $k _ { p , b }$ and $k _ { d , b }$ are the proportional and derivative gains, the four actuator blocks have $N _ { j }$ entries in canonical joint order, and $\eta _ { b }$ holds geometric and mass measurements, namely thigh, shank, upper-arm, and forearm lengths, hip width, torso height, total mass, and leg and arm subtree masses. The transform S scales the features to [−1, 1] across the three robots and maps constant features to zero.

## C. Reference Motions

We use BONES-SEED [17]<sup>1</sup> with its provided G1 retargets and retarget the same source motions to X2 and T2 with NVIDIA’s SOMA Retargeter [18]. Because all robots share one human-motion source, every clip has corresponding reference frames on the three bodies, which the latent alignment of Section III-E relies on. The retargeted motions are converted to reference states at the control rate within the soft joint limits of the simulator, as described in Appendix B.

## D. Shared World Model

Shared training uses SuperMimic’s $H _ { m } \mathrm { - s t e p }$ autoregressive model fitting [2, Section 3.2], with one dynamics network and robot-specific replay buffers. The network predicts a residual in the frame $F _ { t }$ . With $u _ { t }   =   \Phi _ { F _ { t } } ( s _ { t } )$ and a diagonal context scaling $S _ { \xi }$ that brings friction, encoder bias, and center-of-mass offset to comparable magnitudes, the update is

$$
\begin{aligned} { x _ { t } ^ { b } } & { { } = [ \mathcal { N } _ { s } ^ { b } ( u _ { t } ) , e _ { t } ^ { b } , a _ { t } , S _ { \xi } \xi _ { b } , m _ { b } ] , } \\ { \hat { s } _ { t + 1 } ^ { b } } & { { } = \Phi _ { F _ { t } } ^ { - 1 } \big ( u _ { t } + \sigma _ { \Delta , b } \odot f _ { \omega } ( x _ { t } ^ { b } ) \big ) \thinspace . } \\ \end{aligned}\tag{5}
$$

Here, $\mathcal { N } _ { s } ^ { b }$ uses robot-specific motion-library statistics, $e _ { t } ^ { b }$ contains gravity in $F _ { t }$ and the normalized world-frame heights of the tracked bodies, and the fixed scale $\sigma _ { \Delta , b }$ is the root-meansquare (RMS) one-step residual in warm-up replay. Because the residual output layer is initialized to zero, the initial model predicts an unchanged state in $F _ { t } .$

The state-fitting term compares predicted and recorded features in the frame of the recorded state. It sums squared errors after division by $\sigma _ { \Delta , b }$ and averages them over the $H _ { m }$ steps and the sampled windows. A termination head is fit with binary cross-entropy on the same windows, which gives $\mathcal { L } _ { \mathrm { w m } } ^ { b } ~ = ~ \mathcal { L } _ { \mathrm { s t a t e } } ^ { b }   +   \mathcal { L } _ { \mathrm { t e r m } } ^ { b } ,$ as in the analytic reward and observation case of SuperMimic’s Eq. (1). With learned positive gains $k _ { z }$ and $k _ { u }$ and learned biases $b _ { z }$ and $b _ { u } ,$ we model termination as a monotone hazard in the anchor-height and tilt errors:

$$
\begin{aligned} { p _ { z } } & { { } = \operatorname { s i g m o i d } ( k _ { z } ( e _ { z } - \bar { e } _ { z } ) + b _ { z } ) \thinspace , } \\ { p _ { u } } & { { } = \operatorname { s i g m o i d } ( k _ { u } ( e _ { u } - \bar { e } _ { u } ) + b _ { u } ) \thinspace , } \\ { \hat { d } } & { { } = 1 - ( 1 - p _ { z } ) ( 1 - p _ { u } ) , } \\ \end{aligned}\tag{6}
$$

with the termination thresholds of the simulator.

## E. Morphology-Conditioned Policy with Aligned Commands

A motion encoder maps the normalized goal to a latent command of dimension $d _ { z } ,$ and a decoder maps this latent, the proprioceptive history, and the morphology descriptor to an action:

$$
\begin{aligned} { z _ { t } ^ { b } = E ( \mathcal { N } _ { g } ^ { b } ( g _ { t } ^ { b } ) ) , \qquad a _ { t } = D ( z _ { t } ^ { b } , \mathcal { N } _ { o } ^ { b } ( o _ { t - K + 1 : t } ^ { b } ) , m _ { b } ) . } \\ \end{aligned}\tag{7}
$$

The encoder and decoder follow SuperMimic’s architecture [2, Table A.4], with $z _ { t } ^ { b } \in ( - 1 , 1 ) ^ { d _ { z } }$ . While the encoder reads only the normalized reference goal, the full morphology descriptor is appended to the decoder and critic inputs. In addition, the decoder uses feature-wise linear modulation (FiLM) [19]. With learned matrices $\Gamma _ { \ell }$ and $B _ { \ell }$ and learned biases $\mu _ { \ell }$ and $\nu _ { \ell } ,$ the pre-activation $h _ { \ell }$ of hidden layer ℓ becomes

$$
\tilde { h } _ { \ell } = h _ { \ell } \odot ( 1 + \Gamma _ { \ell } m _ { b } + \mu _ { \ell } ) + B _ { \ell } m _ { b } + \nu _ { \ell } .\tag{8}
$$

These modulation parameters start at zero, and training therefore begins from the concatenation-only decoder. The world model receives morphology by concatenation.

1) Imagined Policy Learning: Let $\hat { J } _ { b }$ denote SuperMimic’s survival-weighted imagined return [2, Eq. (2)], computed over $H _ { \pi } \mathrm { - s t e p }$ imagined rollouts with the dynamics and critic conditioned on $m _ { b }$ . The policy gradient passes through the frozen model and critic, goals are recomputed from the predicted states and the reference window, and critic bootstrapping is retained at time-outs. Because the reward depends on the displacement of the robot from the reference, both critics also receive the reference anchor positions expressed in the current anchor frame of the robot for all K goal frames, divided by the width $\sigma _ { p }$ of the position-reward kernel. These inputs are used only during training and remain outside the position-free policy inputs. Critics fit detached λ-return targets following SuperMimic’s Eq. (3).

2) Latent Command Alignment: Corresponding reference frames should produce similar latent commands on all robots. At each policy update, the robots draw the same $N _ { a }$ clip– frame pairs from the intersection of their motion subsets, with frame indices restricted to the shortest version of the clip. On each robot, the shared encoder processes that robot’s paired reference goals, computed with the robot placed at

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>Motion Data by Bones Studio - [https://bones.studio/](https://bones.studio/)</span></small>

<!-- page 5 of 9 -->

![Image block](doc:b4259b6/tier:basic/page:5/block:1)

Figure 3. Five-point tracking with the shared policy during the walking (left) and waving (right) phases of one reference motion. Spheres mark the current pelvis, ankle, and wrist targets, solid curves the ten-frame preview, and fading trails the past targets (display only).

![Image block](doc:b4259b6/tier:basic/page:5/block:3)

Figure 4. Broad jump, jogging, and dance (pirouette) with the same shared full-motion policy on G1, X2 Ultra, and T2. Each strip shows five simulator poses in time order from left to right; headings and horizontal spacing are adjusted for display.

the reference state. For pair i, we form the mean latent $\begin{array} { r } { \bar { z } _ { i } = | \mathcal { B } | ^ { - 1 } \sum _ { b } z _ { i } ^ { b } } \end{array}$ and add

$$
\mathcal { L } _ { \mathrm { a l i g n } } ^ { b } = \frac { 1 } { N _ { a } } \sum _ { i = 1 } ^ { N _ { a } } \left\| z _ { i } ^ { b } - \operatorname { s g } ( \bar { z } _ { i } ) \right\| _ { 2 } ^ { 2 }\tag{9}
$$

to the policy loss with weight $\beta _ { a }$ . The operator sg stops gradients through the gathered mean, so each robot receives gradients only through its own encoder output. The complete motion-policy loss is $\mathcal { L } _ { \mathrm { m o t i o n } } ^ { b }   =   - \hat { J } _ { b } + \beta _ { a } \mathcal { L } _ { \mathrm { a l i g n } } ^ { b }$ , optimized over the motion encoder and decoder. Following the separation of command and execution in X-WBC [10], the latent describes the requested motion and the morphology-conditioned decoder realizes it on the robot being controlled.

## F. Five-Point Command Interface

Based on the multi-interface design of SuperMimic [2, Appendix B], a second goal encoder accepts sparse bodypose targets for the pelvis, the left and right ankles, and the left and right wrists, corresponding to canonical body slots (0, 3, 6, 10, 13). For the current reference frame and the next K − 1 frames, each point contributes a position in meters and

a six-dimensional rotation. Target positions use the reference heading frame of the full-motion goal, so that target heights are absolute and target positions are independent of the position of the robot. Target orientations are expressed relative to the full current pelvis rotation of the robot, which provides feedback about roll, pitch, and yaw from onboard sensing.

A running normalizer acts on the target features before masking. A five-bit point mask and a K-bit frame mask identify visible targets. Masked frame–point blocks are zeroed and the mask bits are appended, producing the input $\tilde { g } _ { t } ^ { b , \mathrm { k p } }$ . In the dedicated five-point mode used here, all points and all K frames are visible. The keypoint encoder is a multilayer perceptron with a $d _ { z }$ -dimensional tanh output, and the action path is

$$
z _ { t } ^ { b , \mathrm { k p } } = E _ { \mathrm { k p } } ( \tilde { g } _ { t } ^ { b , \mathrm { k p } } ) , \qquad a _ { t } = D ( z _ { t } ^ { b , \mathrm { k p } } , \mathcal { N } _ { o } ^ { b } ( o _ { t - K + 1 : t } ^ { b } ) , m _ { b } ) .\tag{10}
$$

The decoder D, including its FiLM layers, is the morphologyconditioned decoder of the motion interface, and target coordinates remain in the physical scale of the robot. Fig. 3 shows walking and waving through this interface on all three robots.

<!-- page 6 of 9 -->

1) Keypoint Encoder Training: After the motion-policy update and its critic fit, an additional step trains the keypoint encoder on $N _ { \mathrm { k p } }$ imagined starts. Let $\mathcal { I } _ { b }$ contain the valid rewarded rollout steps, let $z _ { i , t } ^ { b , \mathrm { m o t i o n } }$ be the latent that the motion encoder produces from the clean full goal at the corresponding imagined state, and let $\beta _ { \mathrm { k p } }$ weight the latentmatching term. The keypoint loss is

$$
\begin{aligned} { \mathcal { L } _ { \operatorname { k p } } ^ { b } } & { { } = - \hat { J } _ { b } ^ { \operatorname { k p } } + \beta _ { \operatorname { k p } } \: \mathcal { A } _ { b } , } \\ { \mathcal { A } _ { b } } & { { } = \frac { 1 } { | \mathcal { I } _ { b } | } \sum _ { ( i , t ) \in \mathcal { I } _ { b } } \left\| z _ { i , t } ^ { b , \operatorname { k p } } - \operatorname { s g } ( z _ { i , t } ^ { b , \operatorname { m o t i o n } } ) \right\| _ { 2 } ^ { 2 } . } \\ \end{aligned}\tag{11}
$$

The keypoint return $\hat { J } _ { b } ^ { \mathrm { k p } }$ uses the imagined-return objective with the keypoint encoder and a separate critic, computed with the full motion-tracking reward and the clean reference. The alignment term sums over the latent dimensions and averages uniformly over the valid rollout steps. As in SuperMimic’s downstream-encoder update [2, Appendix B], only the key point encoder is optimized in this step, and derivatives pass through the fixed decoder, model, and critic. The keypoint critic is then fit on these rollouts. It receives state features, the clean full-motion goal, the reference anchor positions relative to the robot, the context, and the morphology descriptor, while the actor receives the five-point targets.

To expose the encoder to imprecise targets, we perturb the keypoint targets of the actor during imagination with bounded position and orientation jitter, sampled once for the targetframe sequence of a rollout so that overlapping windows reuse the same perturbation for a given frame. Rewards, termination, and critic targets use the clean reference, as do collection and evaluation. A fraction $f _ { \mathrm { k p } }$ of the collection environments acts through the keypoint encoder and the rest through the motion encoder, and both feed the replay buffer, while the motion update keeps its original batch size and loss. The keypoint update therefore also changes the collected data and hence the data used to train the world model.

## G. Training Across Robots

Each robot runs in its own process with its own simulator, motion library, replay buffer, and reset sampler. For any shared parameter block ω, the optimizer uses

$$
\overline { { \nabla _ { \omega } \mathcal { L } } } = \frac { 1 } { | \mathcal { B } | } \sum _ { b \in \mathcal { B } } \nabla _ { \omega } \mathcal { L } _ { b } ,\tag{12}
$$

averaged before SuperMimic’s gradient clipping and Adam updates [2, Table A.5]. State, observation, and goal normalizers, residual scales, body-height statistics, joint limits, default poses, and morphology vectors remain local to each robot and are excluded from parameter synchronization. A checkpoint therefore contains the shared weights together with the statistics and constants for its selected robot. Here, a shared policy means shared encoder and decoder weights combined with these robot-specific buffers and the native action map. In MorphMimic, both encoders, the decoder, and both interfacespecific critics share weights across robots, while keypoint normalizers remain robot-specific. Cross-robot alignment in (9) acts on motion latents, and keypoint latents align to the motion latents of their own robot through A<sub>b</sub>. After warm-up,

each training iteration collects simulator transitions into robotspecific replay buffers, fits the shared world model, updates the motion encoder and decoder, and fits the motion critic. It then updates the keypoint encoder and fits the keypoint critic on a fresh set of imagined starts.

## IV. EXPERIMENTS

## A. Experimental Setup

We evaluate four sharing configurations: a specialist with separate world model and policy per robot, a shared-worldmodel configuration, a shared-policy configuration, and a unified configuration that shares the world model, policy, and critics. All configurations use the G1, X2 Ultra, and T2, position-free full-motion and five-point interfaces, 4096 environments per robot, and 40,000 iterations. Architectures and optimizer settings are matched, actuator delay is randomized over 0–40 ms, and critic targets are clipped at the maximum discounted return. The unified run processes three robot streams at once, whereas each specialist processes one.

Evaluation uses the 128,786 clips common to all three training libraries. Each clip runs once from its first frame under the deterministic policy, with nominal simulator parameters and clean observations, resets, and targets. Five-point commands include all five targets and ten preview frames. Survival is the fraction of clips that reach their final frame without early termination. For a clip of $L _ { i }$ frames and $n _ { i } \leq L _ { i } - 1$ executed transitions, the normalized return is

$$
R _ { i } ^ { \mathrm { n o r m } } = \frac { \sum _ { t = 0 } ^ { n _ { i } - 1 } r _ { i , t + 1 } } { r _ { \operatorname* { m a x } } ( L _ { i } - 1 ) } .\tag{13}
$$

Because the denominator covers the complete clip, early termination lowers the return, and perfect tracking without penalties gives a return of one. We report the mean over clips.

## B. Full-Corpus Tracking

Table I reports the specialist and unified configurations on the common set. The single-morphology diagnostic is listed in separate rows and discussed in Section IV-C; the sharedworld-model and shared-policy ablations are reported in the appendix. Every shared configuration stays within 0.02 mean return of its specialist on all six robot–interface pairs. The largest difference is 0.018 on the G1 five-point interface for the unified configuration. The unified policy therefore preserves survival performance: it leads on both X2 interfaces and T2 full-motion tracking and stays within 0.12 percentage points of the specialist on the other three pairs. The unified run takes 5 h 26 min and 16.3 GPU-hours on three H100 NVL GPUs; the three specialists use 13.9 GPU-hours in total.

## C. Cross-Morphology Training

To demonstrate the necessity of cross-morphology training, we also evaluate a 40,000-iteration G1 specialist and the shared-world-model G1 actor without retraining on X2 and T2. The full comparison appears in Appendix A-A. Using the target robot’s action scale and default pose causes the policy to fail on both bodies. Keeping the G1 action scale and default pose gives the specialist 87.7% survival on X2 full-motion

<!-- page 7 of 9 -->

![Image block](doc:b4259b6/tier:basic/page:7/block:1)

Figure 5. MorphMimic on a real Unitree G1. (a) Dance, (b) walking, and (c) teleoperation. Frames run left to right.

Table I

SPECIALIST, UNIFIED, AND SINGLE-G1 TRANSFER POLICIES ON THE 128,786-CLIP COMMON SET. TRANSFER ROWS USE TARGET OR G1 ACTION PARAMETERS; SURVIVAL IS THE FINAL-FRAME FRACTION (↑).

<table><tr><td></td><td></td><td colspan="2">G1</td><td colspan="2">X2 Ultra</td><td colspan="2">T2</td></tr><tr><td>Interface</td><td>Configuration</td><td>Survival (%) ↑</td><td>Mean return ↑</td><td>Survival (%) ↑</td><td>Mean return ↑</td><td>Survival (%) ↑</td><td>Mean return ↑</td></tr><tr><td>Full motion</td><td>Specialist</td><td>99.797</td><td>0.9306</td><td>99.818</td><td>0.9253</td><td>99.860</td><td>0.9448</td></tr><tr><td></td><td>Unified</td><td>99.747</td><td>0.9358</td><td>99.891</td><td>0.9304</td><td>99.886</td><td>0.9398</td></tr><tr><td></td><td>G1 specialist + target params</td><td></td><td></td><td>7.7</td><td>0.213</td><td>1.9</td><td>0.048</td></tr><tr><td></td><td>G1 specialist + G1 params</td><td></td><td></td><td>87.7</td><td>0.724</td><td>1.6</td><td>0.042</td></tr><tr><td>Five-point</td><td>Specialist</td><td>99.849</td><td>0.8807</td><td>99.841</td><td>0.8689</td><td>99.857</td><td>0.9014</td></tr><tr><td></td><td>Unified</td><td>99.729</td><td>0.8626</td><td>99.880</td><td>0.8774</td><td>99.852</td><td>0.8890</td></tr><tr><td></td><td>G1 specialist + target params G1 specialist + G1 params</td><td>一</td><td></td><td>17.2 76.8</td><td>0.264 0.401</td><td>2.6 2.1</td><td>0.047 0.027</td></tr></table>

clips and 76.8% on five-point clips, with mean returns of 0.724 and 0.401. The shared-world-model actor is within 0.032 mean return of the specialist on X2, while T2 remains below 2.1% survival in every setting. The T2 result also reflects its trunk-mounted IMU, a sensing frame the G1-only policy has not encountered. Moving the simulated T2 IMU to the waist-yaw link raises survival for the transferred G1 policies to 81–83% for motion and 74–75% for five-point transfer (Appendix A-A). Training on all three embodiments includes these sensing and morphology differences, and the unified model reaches 99.9% survival on X2 and T2, surpassing the G1-specific policy.

## D. Command Interfaces

Figs. 4 and 3 show full-motion and five-point tracking with the unified policy. Across the four configurations, the five-point interface lowers mean return by 0.04–0.07 on each robot. Its effect on survival depends on the configuration: the specialist loses 67 clips on G1, 29 on X2, and gains 4 on T2, while the unified policy loses 23, 13, and 44 clips,

respectively. The smaller target set therefore preserves the command interface while reducing tracking precision.

## E. Hardware Experiments

We deploy the simulation-trained MorphMimic policy directly on a physical Unitree G1. The deployed weights are those of the 40,000-iteration MorphMimic policy in Table I, exported to ONNX and identical for the three robots. At each control step, the policy reads the orientation and angular velocity of the pelvis IMU and the 29 joint encoders and sends joint position targets to the PD controllers with the training gains. In Fig. 5(a) and (b), the G1 tracks a dance clip and a walking clip from the reference library through the full-motion interface. Each clip is rotated to the heading of the robot at its start, so that the IMU heading aligns the reference with the robot. In Fig. 5(c), an operator wearing a PICO headset, two hand controllers, and two ankle trackers teleoperates the robot through the five-point interface. The operator’s motion is resampled and retargeted to G1 reference states at the control rate, and a nominal buffer of K−1 frames supplies the preview window from the received reference states. The G1 follows the

<!-- page 8 of 9 -->

operator through standing, squatting, and arm motions in real time.

## V. CONCLUSION

MorphMimic trains one differentiable world model and one whole-body tracking policy for the Unitree G1, AgiBot X2 Ultra, and Booster T2. With the shared representation and morphology conditioning, each shared configuration stays within 0.02 mean return of its single-robot specialist on the 128,786-clip common set, while the five-point interface uses the same decoder for sparse targets.

The same policy dances, walks, and follows a teleoperator on a physical G1 using onboard sensing. The singlemorphology test shows why training across embodimentspecific sensing and morphology matters: the unified policy reaches 99.9% survival on X2 and T2. Future work will extend the hardware experiments to the X2 Ultra and T2 and to a wider range of humanoid platforms.

## REFERENCES

[1] Z. Luo, Y. Yuan, T. Wang, C. Li, F. Castañeda, S. Chen, Z.-A. Cao, J. Li, D. Minor, Q. Ben, J. Park, D. Sami, Z. Wang, X. Da, R. Ding, C. Hogg, L. Song, E. Lim, E. Jeong, T. He, H. Xue, W. Xiao, S. Yuen, J. Kautz, Y. Chang, U. Iqbal, L. J. Fan, and Y. Zhu, “SONIC: Supersizing motion tracking for natural humanoid whole-body control,” Science Robotics, vol. 11, no. 117, p. eaed4592, 2026. [Online]. Available: [https://doi.org/10.1126/scirobotics.aed4592](https://doi.org/10.1126/scirobotics.aed4592)

[2] Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, and K. Sreenath, “SuperMimic: Scaling humanoid whole-body control with one differentiable world model,” in Conference on Robot Learning (CoRL), 2026, accepted to CoRL 2026 (oral). [Online]. Available: [https://arxiv.org/abs/2610.28480](https://arxiv.org/abs/2610.28480)

[3] M. Yang, T. Yu, F. Li, and H. Chen, “Any2Any: Efficient cross-embodiment transfer for humanoid whole-body tracking,” arXiv preprint arXiv:2605.23733, 2026. [Online]. Available: [https://arxiv.org/abs/2605.23733v3](https://arxiv.org/abs/2605.23733v3)

[4] T. Chen, A. Murali, and A. Gupta, “Hardware conditioned policies for multi-robot transfer learning,” in Advances in Neural Information Processing Systems, vol. 31, 2018. [Online]. Available: [https://papers.neurips.cc/paper\_files/paper/2018/hash/b8cfbf77a3d250a4523ba67a65a7d031-Abstract.html](https://papers.neurips.cc/paper_files/paper/2018/hash/b8cfbf77a3d250a4523ba67a65a7d031-Abstract.html)

[5] N. Bohlinger, G. Czechmanowski, M. P. Krupka, P. Kicki, K. Walas, J. Peters, and D. Tateo, “One policy to run them all: an end-to-end learning approach to multi-embodiment locomotion,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270, 2025, pp. 3356–3378. [Online]. Available: [https://proceedings.mlr.press/v270/bohlinger25a.html](https://proceedings.mlr.press/v270/bohlinger25a.html)

[6] N. Bohlinger and J. Peters, “Multi-embodiment locomotion at scale with extreme embodiment randomization,” arXiv preprint arXiv:2509.02815, 2025. [Online]. Available: [https://arxiv.org/abs/2509.02815](https://arxiv.org/abs/2509.02815)

[7] Y. Xue, Y. Lin, W. Dong, Y. Tang, J. Wang, J. Pang, M. Zhou, M. Liu, and W. Zhang, “Scalable and general whole-body control for cross-humanoid locomotion,” arXiv preprint arXiv:2602.05791, 2026. [Online]. Available: [https://arxiv.org/abs/2602.05791v3](https://arxiv.org/abs/2602.05791v3)

[8] M. H. Danesh, C. Li, A. Abyaneh, A. Houssaini, K. Ellis, G. Berseth, M. Hutter, and H.-C. Lin, “Morphology-conditioned world model for cross-embodiment quadrupedal locomotion,” arXiv preprint arXiv:2604.08780, 2026. [Online]. Available: [https://arxiv.org/abs/2604.08780v2](https://arxiv.org/abs/2604.08780v2)

[9] Y. Wang, J. Kong, S. Wei, X. Li, H. Lin, H. Zhao, T. Zhou, L. Gan, and H. Shao, “WestWorld: A knowledge-encoded scalable trajectory world model for diverse robotic systems,” in International Conference on Machine Learning (ICML), 2026.

[10] J. Zhang, C. Gu, and L. Zhang, “X-WBC: A cross-embodiment foundation model for humanoid whole-body control,” in Conference on Robot Learning (CoRL), 2026, accepted to CoRL 2026. [Online]. Available: [https://arxiv.org/abs/2609.15213](https://arxiv.org/abs/2609.15213)

[11] W. Huang, I. Mordatch, and D. Pathak, “One policy to control them all: Shared modular policies for agent-agnostic control,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119, 2020, pp. 4455–4464. [Online]. Available: [https://proceedings.mlr.press/v119/huang20d.html](https://proceedings.mlr.press/v119/huang20d.html)

[12] A. Gupta, L. Fan, S. Ganguli, and L. Fei-Fei, “MetaMorph: Learning universal controllers with transformers,” in International Conference on Learning Representations, 2022. [Online]. Available: [https://openreview.net/forum?id=Opmqtk\_GvYL](https://openreview.net/forum?id=Opmqtk_GvYL)

[13] G. Feng, H. Zhang, Z. Li, X. B. Peng, B. Basireddy, L. Yue, Z. Song, L. Yang, Y. Liu, K. Sreenath, and S. Levine, “GenLoco: Generalized locomotion controllers for quadrupedal robots,” in Proceedings of the 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205, 2023, pp. 1893–1903. [Online]. Available: [https://proceedings.mlr.press/v205/feng23a.html](https://proceedings.mlr.press/v205/feng23a.html)

[14] S. Wei, X. Chen, F. Xie, G. E. Katz, Z. Gan, and L. Gan, “Beyond topology: A morphological symmetry graph representation for locomotion policy learning,” arXiv preprint arXiv:2512.00727, 2025. [Online]. Available: [https://arxiv.org/abs/2512.00727](https://arxiv.org/abs/2512.00727)

[15] I. Georgiev, V. Giridhar, N. Hansen, and A. Garg, “PWM: Policy learning with multi-task world models,” arXiv preprint arXiv:2407.02466, 2024, version 3, revised February 2025. [Online]. Available: [https://arxiv.org/abs/2407.02466v3](https://arxiv.org/abs/2407.02466v3)

[16] F. Xie, S. Wei, Y. Song, Y. Yue, and L. Gan, “Morphologicalsymmetry-equivariant heterogeneous graph neural network for robotic dynamics learning,” in Proceedings of the 7th Annual Learning for Dynamics & Control Conference, ser. Proceedings of Machine Learning Research, vol. 283, 2025, pp. 1392–1405. [Online]. Available: [https://proceedings.mlr.press/v283/xie25a.html](https://proceedings.mlr.press/v283/xie25a.html)

[17] Bones Studio, “BONES-SEED: Skeletal everyday embodiment dataset,” Hugging Face dataset, accessed October 22, 2026. [Online]. Available: [https://huggingface.co/datasets/bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)

[18] NVIDIA, “SOMA Retargeter,” Software repository, experiments use commit cbdb2b4; accessed October 22, 2026. [Online]. Available: [https://github.com/NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter)

[19] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2018. [Online]. Available: [https://ojs.aaai.org/index.php/AAAI/article/view/11671](https://ojs.aaai.org/index.php/AAAI/article/view/11671)

<!-- page 9 of 9 -->

## APPENDIX A ADDITIONAL RESULTS

A. Zero-Shot Transfer Comparison

Table II

ZERO-SHOT TRANSFER OF THE SHARED-WORLD-MODEL G1 ACTOR (W) TO THE X2 ULTRA AND T2. EACH ENTRY IS SURVIVAL (%) / MEAN NORMALIZED RETURN; THE SPECIALIST ROWS ARE IN TABLE I.

|  | Motion | Five-point |
| --- | --- | --- |
| Target parameters, X2 | 8.1 / .223 | 24.6 / .278 |
| Target parameters, T2 | 1.5 / .040 | 2.2 / .035 |
| G1 parameters, X2 | 82.8 / .692 | 82.2 / .407 |
| G1 parameters, T2 | 1.6 / .036 | 1.7 / .025 |

Target parameters use the target action scale and default pose, while G1 parameters retain the G1 scale and pose together with the target head joints. For T2, the transfer uses the trunk-mounted IMU and an unseen morphology vector.

With the T2 IMU moved to the waist-yaw link, the specialist/shared-world-model actor reaches 83.4%/81.5% survival on motion and 75.4%/74.2% on five-point commands under G1 parameters. Under target parameters, the corresponding values are 64.0%/79.8% and 67.6%/66.4%.

The transfer uses the unchanged 40,000-iteration checkpoint and deterministic rollouts on the common set. The canonical translator maps each action to the target robot’s native joints, while the target simulator keeps its own body model, PD gains, effort limits, default pose, and IMU body. The shared-worldmodel actor retains the G1 morphology descriptor; substituting a target descriptor is outside its training distribution and is not included here.

## APPENDIX B IMPLEMENTATION DETAILS

The implementation follows the constants in Section III. The policy uses 93-value proprioception and 67-value full-motion goals over a K-frame window; the dynamics model receives 472 features and the five-point encoder 465. Shared networks follow SuperMimic’s Table A.4, with critic targets clamped at the maximum discounted return; normalizers, residual scales, limits, and default poses remain robot-specific. The canonical map is checked with forward kinematics and neutral-pose alignment. T2 uses a waist-yaw pelvis equivalent, a trunk IMU, and robot-specific shoulder and elbow conventions; X2 uses a different waist Euler order. Five-point targets use ankleroll and distal wrist links, with pelvis orientation from the robot IMU and encoders. SOMA retargets at 120 Hz, converts references to 50 Hz, applies feasibility filters and soft joint

limits, and yields 128,981 X2 and 129,048 T2 clips from 129,398 sources. The physical-G1 stand and teleoperation buffer use the settings described in Section IV-E.

The three simulated bodies expose 29 policy joints. X2 Ultra and T2 each add two head joints outside the policy; their compiled-model masses are 33.3, 43.5, and 42.0 kg for G1, X2, and T2, respectively. The thigh/shank lengths are (0.341, 0.300), (0.327, 0.282), and (0.347, 0.336) m, and the upper-arm/forearm lengths are (0.185, 0.184), (0.205, 0.203), and (0.218, 0.212) m. Control runs at 50 Hz with K = 10 frames in both the history and goal windows, action scale 0.25, descriptor and context sizes 125 and 33, latent size 32, modelfitting horizon 8, and imagination horizon 32. Alignment uses 1024 pairs with weight 0.05; the five-point update uses 4096 imagined starts, a 0.5 collection share, latent-matching weight 0.1, and the jitter ranges listed in the method.

Each robot has its own simulator process, replay buffer, motion-library normalizer, and reset sampler. Shared gradients are averaged across the active robot processes before clipping and Adam updates. The shared checkpoint stores these common weights together with the selected robot’s statistics and constants. Retargeted references use robot-specific actuator gains, effort limits, and soft joint limits; the pelvis target uses the IMU attitude on G1 and X2 and the trunk IMU plus waist encoders on T2. Evaluation uses deterministic rollouts with nominal simulator parameters and clean observations and targets.

The control interval is 0.02 s with discount factor 0.99, maximum per-step reward 0.1, and termination thresholds of 0.25 m in anchor height and 0.8 in tilt. Soft joint limits use 90% of each native range. The five-point encoder uses position jitter of ±0.05 m horizontally and ±0.01 m vertically, with roll–pitch and yaw jitter of ±0.1 and ±0.2 rad. These perturbations affect the actor targets during imagined training; the clean reference is retained for rewards, termination, critic targets, and evaluation.

Table III

SHARED-WORLD-MODEL (WM) AND SHARED-POLICY (P) ABLATIONS ON THE COMMON SET. ENTRIES REPORT SURVIVAL (%) AND MEAN RETURN.

<table><tr><td colspan="3">WM</td><td colspan="3">P</td></tr><tr><td>Robot</td><td>Interface</td><td>Survival (%)</td><td>Return</td><td>Survival (%)</td><td>Return</td></tr><tr><td>G1</td><td>Full motion</td><td>99.734</td><td>0.9274</td><td>99.806</td><td>0.9317</td></tr><tr><td>X2</td><td>Full motion</td><td>99.845</td><td>0.9269</td><td>99.861</td><td>0.9278</td></tr><tr><td>T2</td><td>Full motion</td><td>99.861</td><td>0.9401</td><td>99.775</td><td>0.9404</td></tr><tr><td>G1</td><td>Five-point</td><td>99.720</td><td>0.8681</td><td>99.769</td><td>0.8781</td></tr><tr><td>X2</td><td>Five-point</td><td>99.851</td><td>0.8788</td><td>99.847</td><td>0.8855</td></tr><tr><td>T2</td><td>Five-point</td><td>99.875</td><td>0.8979</td><td>99.766</td><td>0.8976</td></tr></table>