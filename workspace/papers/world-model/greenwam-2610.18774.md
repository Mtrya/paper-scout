<!-- page 1 of 43 -->

arXiv:2610.18774v1 [cs.RO] 13 Oct 2026

# GreenWAM: A World Foundation Model for Robotics

C Robotics Center

Gennady Sidorov, Valerii Kornilov, Denis Grankin, Polina Fedotova, Aleksey Postnikov Brain Team, Sber Robotics Center

A robot needs a world model that predicts how its actions will change the scene and a policy that chooses those actions. Models are usually built and evaluated for only one of these roles, and every robot records its actions in its own format. We ask whether one robot world model can serve both roles across different bodies, and what makes this work. We present GreenWAM, a world foundation model for robotics, together with the recipe that produces it: (1) a shared anatomical action space that describes 20 robot families, from bimanual arms to dexterous humanoids, in common joint coordinates; (2) joint training on these families in visual, action-conditioned and world–action modes; and (3) policies that split an action expert off the generator. As a world model, GreenWAM attains top-tier trajectory, depth and 3D accuracy on WorldArena 1.0, ahead of every world–action model and video generator in our comparison. As a policy, it scores among the top entries of the RoboTwin 2.0 leaderboard, reaches the highest zero-shot LIBERO-plus success among the compared methods and is competitive with dedicated vision–language–action and world–action policies on CALVIN and LIBERO. We further adapt GreenWAM to a real humanoid, where it succeeds more often than policies adapted from $\pi _ { 0 . 5 }$ and from Cosmos 3 Edge, and distill it into a single-step policy for real-time control. Controlled experiments yield three findings: robot specialization improves world prediction and control together; an action expert helps when split off at the start of adaptation; and keeping world-model modes during policy adaptation preserves robustness to scene changes.

[**Project page**](https://greenvla.github.io/GreenWAM) ⋅ [**Code**](https://github.com/greenvla/GreenWAM) ⋅ [**Weights**](https://huggingface.co/SberRoboticsCenter/models) ⋅ October 2026

## 1 Introduction

World foundation models learn how scenes evolve from large-scale video, and world–action models reuse this knowledge to predict observations and actions jointly [58, 92]. For a robot, the two uses are linked: the model must predict how the scene changes when the robot moves, and it must produce that motion itself. Both uses therefore depend on how the model represents the robot body. Yet models are usually built and evaluated for one of these roles: world models are judged by their predictions, world–action models by the success of the resulting policy.

Robot data are fragmented across bodies and recording conventions. Joint layouts, grippers and dexterous hands, camera sets, frame rates and task boundaries differ between datasets, and the same tensor dimension can mean different motions on different robots. Current models describe a robot through an action projection for each embodiment domain, through end-effector poses with gripper or hand states, or through joint vectors padded to a common length [58, 2]. In each case the correspondence between the arms, torso, legs and fingers of different bodies remains implicit or absent. We ask whether one robot world model can serve both as a world model and as a policy across different bodies, and what makes this work.

We present GreenWAM, a world foundation model for robotics built on Cosmos 3 Edge [58], together with the recipe that produces it (Figure 1). Cosmos 3 Edge learned robot actions only coarsely: from six embodiments dominated by a single dataset, as end-effector pose deltas and a gripper opening, with a separate projection for each embodiment domain (Figure 2). The recipe replaces this in three steps. First, a shared anatomical action space describes the bodies of 20 robot families, from bimanual arms to humanoids with dexterous hands, in common joint coordinates: a codec for each robot layout maps its joints onto shared slots for the legs, torso, arms and finger joints, so that one slot means the same joint motion on every robot (Section 3.2). Second, one model is trained jointly on these families in visual, action-conditioned and

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">© 2026 Sber Robotics Center.</span></small>

<!-- page 2 of 43 -->

![Image block](doc:f38ca8c/tier:basic/page:2/block:1)

Figure 1 From robot experience to prediction and control. (a–b) Video, measured states and text from 20 robot families form aligned subtask examples. (c–d) A shared 79-slot anatomical action space connects robot motion to a joint video–action model built on Cosmos 3 Edge. (e) Forward prediction of GreenWAM for Unitree G1 and AgiBot G2A on held-out recordings (Section 5.1). (f) As a policy, the foundation adds a coupled action expert and is distilled for humanoid control. (g) GreenWAM against the best of three references on every axis (grey dashed outline). Left: policy, against $\pi _ { 0 . 5 } \; [ 6 2 ]$ , X-VLA [103] and FastWAM [95]; the $\pi _ { 0 . 5 }$ CALVIN value is the supervised fine-tuned checkpoint reported by $\pi _ { \mathrm { R L } }$ [17], RoboDojo scores come from the official leaderboard [67]. Right: WorldArena 1.0 against Motubrain [56], FlowWAM-FiveAges [21] and UNIS [46]: three categories of the motion-capped score (Section 5.2; for the other models an upper bound) and three fidelity metrics. Every axis starts where its bar chart in Figures 9 and 11 starts, but not below half of its best value, and ends at the best value on that axis.

world–action modes, on recordings prepared as whole subtasks with instructions (Sections 3.3 and 4). Third, policies split an action expert off the generator (Section 6.1).

Both roles are obtained from this one foundation, through one action interface. A single model predicts video on held-out recordings of six robot families, with no robot-specific generator or action projection (Section 5.1), and a single-arm Franka, absent from the robotics corpus, is controlled through the same shared interface on CALVIN and LIBERO. As a world model, GreenWAM attains top-tier trajectory, depth and 3D accuracy on WorldArena 1.0, ahead of every world–action model and video generator in our comparison, including Motubrain [56] and FlowWAM-FiveAges [21] (Section 5.2). As a policy, it scores among the top entries of the RoboTwin 2.0 leaderboard, reaches the highest zero-shot LIBERO-plus success among the compared methods and is competitive on CALVIN and LIBERO (Section 6). On a real humanoid, it completes 95.6% of in-distribution instructions, against 88.9% for a policy adapted from its parent and 84.4% for $\pi _ { 0 . 5 }$ , and a distilled single-step student answers 12.3 times faster than the unoptimized policy (Sections 6.4 and 6.5).

Controlled comparisons (Section 7) identify what makes this work. First, robot specialization improves world prediction and control together (Section 7.1). Adapted from GreenWAM rather than from its actiontrained parent, the world model raises trajectory accuracy on WorldArena by 3.0 points and lowers its video prediction error by 23%. The policy predicts held-out RoboTwin frames more accurately and succeeds 5.4% more often, or 8.4% with the action expert. Second, splitting an action expert off the generator at the start

2

<!-- page 3 of 43 -->

of adaptation raises policy success by 3.3% (Section 7.2). Third, keeping world-model modes during policy adaptation raises success in randomized scenes by 8.0% (Section 7.3).

**Contributions.** We make four contributions.

• **Recipe.** A shared anatomical action space and a data pipeline that adapt an action-trained world foundation model to 20 robot families (Sections 3 and 4).

• **One model, two roles.** Joint training in seven visual, action-conditioned and world–action modes yields a world model and, through an action expert split off the generator, a policy (Sections 3.3, 4.3, 6.1 and 6.2).

• **What it brings and why.** Results in both roles on WorldArena 1.0, RoboTwin 2.0, CALVIN, RoboDojo, LIBERO and LIBERO-plus, and controlled comparisons showing that robot specialization benefits both roles (Sections 5 to 7).

• **Real robot.** Adaptation to a real humanoid and distillation into a single-step policy for real-time control (Sections 6.4 and 6.5).

## 2 Related Work

GreenWAM builds on four lines of work: robotics specialization of world foundation models, action representations shared across robot bodies, policies derived from video generators, and temporal context for long rollouts and simulation. Table 1 compares the closest technical reports by the capabilities of their models.

Table 1 Closest technical reports by the capabilities of the model they present. A mark means the report presents the model in that role; it does not rank quality. Actions→video: generation conditioned on robot actions (forward dynamics, action-conditioned simulation); Video→actions: inverse dynamics; Policy: joint prediction of video and actions used for control; Long rollout: autoregressive continuation with memory of past blocks; Real robot: experiments on a physical robot, for GreenWAM on the Green humanoid (Section 6.5).

| Report | Group | Video gen. | Actions→video | Video→actions | Policy | Long rollout | Real robot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cosmos 3 [58] | NVIDIA | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| DreamZero [92] | NVIDIA | ✓ |  |  | ✓ | ✓ | ✓ |
| DreamDojo [25] | NVIDIA | ✓ | ✓ |  |  | ✓ | ✓ |
| DreamGen [34] | NVIDIA | ✓ |  | ✓ |  |  | ✓ |
| GigaWorld-0, GigaWorld-1, GigaWorld-Policy-0.5 [27, 28, 29] | GigaAI | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Motubrain, Motus2 [56, 8] | ShengShu | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GE-Act 2.0, GE-Sim 2.0 [2, 63] | AgiBot | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LingBot-VA, 2.0 [42, 97] | Robbyant | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ABot-PhysWorld [20] | AMAP | ✓ | ✓ |  |  |  |  |
| WALL-WM, WALL-SS [44, 96] | X Square | ✓ | ✓ |  | ✓ | ✓ | ✓ |
| Xiaomi-Robotics-U0 [46] | Xiaomi | ✓ | ✓ |  |  |  | ✓ |
| 1X World Model [1] | 1X | ✓ | ✓ |  |  |  | ✓ |
| GreenWAM | Sber Robotics | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Robotics specialization of world foundations.** Cosmos 3 already combines visual generation, forward and inverse dynamics, and world–action modeling in a reasoner–generator architecture [58]. Its robot data, however, cover six embodiments, with 81% of the hours from one dataset, and represent actions as endeffector pose deltas with a gripper opening. GreenWAM retrains this foundation on measured joint and Cartesian trajectories of 20 robot families, aligned with subtask video and text. Genie Envisioner likewise connects a shared world foundation to prediction, control and simulation applications [47]. Alternative data routes include human video with latent proxy actions in DreamDojo and generated robot experience in GigaWorld-0 [25, 27]. Earlier work established action-conditioned video prediction as an interactive simulator learned from diverse datasets [89], and V-JEPA 2 post-trains a latent action-conditioned world model for robot planning [6].

**Correspondences across robot bodies.** Cross-embodiment policies face the same correspondence problem. Open X-Embodiment pools demonstrations of 22 robots [60]; Octo is fine-tuned to new sensors and action spaces [59], and CrossFormer controls robots for manipulation, navigation, locomotion and aviation without manually aligning their observation or action spaces [22]. HPT maps the proprioception and vision of each embodiment to tokens processed by a shared trunk [80], X-VLA adds embodiment-specific soft prompts [103], UniAct learns a universal space of atomic actions [102], and RDT-1B defines a unified action space whose coordinates keep their physical meaning [49]. Green-VLA uses a unified, embodiment-aware action interface with which a single policy controls humanoids, mobile manipulators and fixed-base arms, and then adapts this policy to the Green humanoid [5]. Among world models, Cosmos 3 represents camera and end-effector motion together with gripper or fingertip state, using an input and output projection for each embodiment

3

<!-- page 4 of 43 -->

domain [58]. GE-Sim 2.0 represents actions as action maps: the end-effector pose of each arm is projected into every camera image with calibrated intrinsics and extrinsics, drawn as a marker whose shading gives the gripper opening, and concatenated with the video latents [63]. Such maps are natural for a video generator and simple to build, and they suit robots whose cameras are calibrated in every frame. For humanoids with moving head and wrist cameras they require exact calibration of every view, and they cannot express finger joints. GreenWAM instead passes the measured joint and Cartesian states through one unified projector (Section 3.2) and still follows the commanded actions closely (Sections 5.1 and 5.2). GreenWAM uses explicit slots and masks: its 79 anatomical slots cover legs, base, torso, head, arm joints and finger joints.

**From world prediction to policy.** Early approaches generate a plan as video and extract the actions from it [23], or pretrain on video and fine-tune for both video and action prediction [16]. UVA learns a joint video–action latent with decoupled decoding heads [43], and UWM couples video and action diffusion so that one model acts as a policy, forward dynamics, inverse dynam ics or video generator depending on the noise levels of the two streams [104]; the modes of GreenWAM follow the same idea across many robot bodies. Video-model priors support control through several prediction interfaces: joint video–action generation in DreamZero, causal prediction and asynchronous execution in LingBot-VA, and visual planning followed by inverse dynamics in GE-Act 2.0 [92, 42, 2]. As a policy, GreenWAM retains joint video–action prediction and specializes the action stream through a dedicated expert.

Robot data, share of hours by source

Policies built on video models also differ in whether actions share the video backbone or receive an expert of their own, and in how that expert is initialized. Motus pretrains an action expert with latent actions, and ME-Dex inherits its video and action experts [7, 99]. GreenWAM copies its generator into an action

![Image block](doc:f38ca8c/tier:basic/page:4/block:5)

Figure 2 Robot data and action types of Cosmos 3 [58] and GreenWAM: shares of robot hours (top; AgiBot World supplies 81% of the robot hours of Cosmos 3, the largest GreenWAM family 15% of ours) and action representations (bottom).

expert at the split, preserving the parent function before further optimization. Section 7.2 compares this split with a shared tower from the same start.

**Temporal context and simulation.** WALL-WM aligns captions, video and actions at the level of semantic events [44], related to the complete-subtask atoms used in our data pipeline. Genie Envisioner and GE-Sim 2.0 use sparse history for long rollouts, with GE-Sim 2.0 perturbing history during training to account for generated-context errors [47, 63]. DreamZero updates its cached predictions with real observations during closed-loop execution [92]. GreenWAM takes four anchors at fixed positions of the history, each encoded separately, and trains continuation with recorded history; Figure 10 shows its rollouts of more than 20 s on WorldArena.

World simulators also support uses beyond video prediction. GigaWorld-1 studies agreement between generated rollouts and real policy executions [28]; GE-Sim 2.0 predicts proprioception and evaluates task completion in simulated rollouts [63]. WALL-SS addresses long action-conditioned simulation with bounded memory and alignment against drift [96], and the 1X World Model supports policy and checkpoint selec tion [1].

## 3 GreenWAM

GreenWAM maps the states of different robots into shared anatomical coordinates and trains one generator to predict video, robot motion or both; a policy is obtained by splitting an action expert off this generator (Section 6.1). Layout-specific codecs absorb the differences between native robot interfaces; Section 4 describes the data and the training.

4

<!-- page 5 of 43 -->

## 3.1 Model Overview

We initialize GreenWAM from the released Cosmos 3 Edge 4B checkpoint [58]. Its mixture-of-transformers backbone (Figure 3) couples a reasoner and a generator, with a causal video VAE and rectified-flow training [50]. Its robot actions, however, come from six embodiments and are represented as end-effector pose deltas with a gripper opening (Figure 2); the robotics stage retrains the action modes on measured joint and Cartesian states of 20 robot families, with the joints in a shared anatomical action space.

During the robotics stage the reasoner and the VAE stay frozen. The generator, the shared action projections and the action-modality embeddings are trained. All modes retain the Cosmos 3 JSON prompt format (Section 4.2).

![Image block](doc:f38ca8c/tier:basic/page:5/block:4)

Figure 3 GreenWAM architecture, shown for the world–action (WAM) mode on one held-out Green humanoid episode. The task instruction, the multi-camera canvas and the robot’s native joint state at 𝑡 = 0 enter a frozen reasoner, a trained generator for the video tokens and, in the adapted policy, an action expert for the state and action tokens, all sharing multimodal attention (the Cosmos 3 Edge mixture of transformers); the layout’s codec maps the native joints to the 79 anatomical slots before the shared input projection (Figure 4). In GreenWAM the state and action tokens pass through the generator; the action expert is split off it for policy adaptation (Section 6.1). Filled tokens are clean conditions and dashed tokens are noised targets; which streams are conditions changes with the mode (Figure 29). Predicted state targets are decoded relative to $q _ { 0 }$ (Equation (2)) and mapped back to native joint states. Video: ground truth (top) and the GreenWAM prediction (bottom). Frozen and trained refer to the robotics stage.

5

<!-- page 6 of 43 -->

## 3.2 Shared Anatomical Action Space

Robot datasets disagree on state layout, joint order, units, sign and zero conventions. Embodiment-specific projectors accommodate these formats but leave the correspondence between their input coordinates implicit. GreenWAM establishes that correspondence before projection: layout-specific codecs map native channels to 79 anatomical coordinates, which then pass through one shared input and output projector (Figure 4). Thus the left elbow occupies the same slot on every platform, while platform-specific conventions are handled by the codec. All 20 families therefore share one action projector, and no robot has action parameters of its own. This is a stricter design than one projection per embodiment domain, as in Cosmos 3 Edge, or end-effector poses rendered into the camera images, as in GE-Sim 2.0 [63]. The vocabulary has four blocks:

• **Legs** (12).

• **Body** (23): base pose (7), base twist (6), torso (7) and head (3). Pose and twist are separate quantities.

• **Arms** (18).

• **Hands** (26): one open–closed coordinate per parallel gripper and per-finger coordinates for dexterous hands, with auxiliary slots for a second gripper finger, encoders and finger abduction.

Layout-specific codecs select, reorder and normalize native channels into anatomical slots. Validity masks exclude missing or unresolved coordinates from the loss. Correspondences are established from documented channels, robot kinematics and measured trajectories; Section A.3 gives the details. New post-training layouts use these same slots and shared projectors. For a layout ℓ with native state $x ,$ the codec assigns slot 𝑗 either a native channel $\pi _ { \ell } ( j )$ with a sign $\rho _ { \ell j } \in \{ - 1 , 1 \}$ and a zero offset $o _ { \ell j } .$ , or nothing:

$$
q _ { j } = \left\{ \begin{aligned} & \rho _ { \ell j } ( x _ { \pi _ { \ell } ( j ) } - o _ { \ell j } ) , & m _ { \ell j } = 1 , \\ & 0 , & m _ { \ell j } = 0 , \end{aligned} \right. \qquad \bar { q } _ { j } = \frac { q _ { j } - \mu _ { \ell j } } { s _ { \ell j } } ,\tag{1}
$$

where $m _ { \ell j }$ is the validity mask and $\mu _ { \ell j } , \; s _ { \ell j }$ are statistics of the layout’s training data. Masked slots are excluded from the loss, so a coordinate that a robot does not have never receives a target.

**Encodings by mode.** Joint states and Cartesian end-effector (EEF) poses are two views of the robot state; absolute and relative values are two encodings. Inverse dynamics predicts the absolute trajectory $q _ { 0 : T } ,$ including the initial state, from video alone. Forward dynamics, forward dynamics with history and WAM express a subset of coordinates relative to the state at the block boundary:

$$
\delta _ { k } = q _ { k } - d _ { \ell } \odot q _ { 0 } , \qquad \hat { q } _ { k } = \hat { \delta } _ { k } + d _ { \ell } \odot q _ { 0 } ,\tag{2}
$$

where 𝑞 is the state at the block boundary, 𝑞 the state at the 𝑘-th target time and $q _ { 0 }$ $q _ { k }$ $d _ { \ell }$ a binary mask of the layout. By default leg, torso, head and arm joints are relative, while grippers, hands and fingers stay absolute; a layout can declare its whole state relative, as the Green humanoid does. The relative values are neither velocities nor accumulated increments, and $d _ { \ell }$ is separate from the validity mask $m _ { \ell }$ , which only removes unavailable coordinates from the loss. Where a source also records a reliable end-effector pose, the joint or EEF view is chosen with equal probability after the window is sampled, so a window never gains sampling mass from having two views. EEF poses occupy separate slots with their own frames; their codec composes poses in SE(3), so the additive rule above applies to joint coordinates only. The projector performs no forward or inverse kinematics.

Families overlap mostly in the arm block, while legs, torso, head and fingers are present only for part of the corpus (Figure 4). Every robot that has a joint contributes to training its slot, and an absent finger or leg joint never becomes a zero target that the model would have to learn as motion.

## 3.3 Joint Video–Action Modeling

A robot clip with its states can answer more than one question: what the scene will look like, what the robot will see if it follows a given trajectory, which trajectory produced the observed video, and what to do next. Instead of training a separate model for each question, we train one set of generator weights on seven modes (Table 2). A mode is defined by which streams are clean conditions and which are noised targets. For a mode $m ,$ let $c _ { m }$ be the clean conditions and $\mathcal { S } _ { m }$ the generated streams, such as video latents 𝑧 or anatomical states 𝑞. All generated streams of an example share one noise level $\sigma \in [ 0 , 1 ]$ . A clean target $x _ { 0 }$ with Gaussian noise 𝜖 is noised along the rectified-flow path, and the generator predicts its velocity:

6

<!-- page 7 of 43 -->

![Image block](doc:f38ca8c/tier:basic/page:7/block:1)

![Image block](doc:f38ca8c/tier:basic/page:7/block:2)

Figure 4 Shared 79-slot anatomical action space. Top: the codec of one layout selects its native channels, puts them in slot order and flips signs and shifts zeros where needed, so that a slot means the same joint motion on every robot, for example the same positive direction of the left shoulder roll; three of the fourteen Worker channels are shown. A shared input projection maps the slot vector to state tokens. Bottom: slots filled by eight source layouts; empty slots are masked in the loss.

$$
x _ { \sigma } = \left( 1 - \sigma \right) x _ { 0 } + \sigma   \epsilon , \qquad v = \epsilon - x _ { 0 } , \qquad \mathcal { L } _ { m } = \mathbb { E } _ { \sigma , \epsilon } \sum _ { s \in \mathcal { B } _ { m } } \lambda _ { s } \frac { \left\| M _ { s } \odot \left( \hat { v } _ { \theta } ( x _ { \sigma } , c _ { m } , \sigma ) _ { s } - v _ { s } \right) \right\| ^ { 2 } } { \left\| M _ { s } \right\| _ { 1 } } ,\tag{3}
$$

where $M _ { s }$ masks the valid elements of stream 𝑠 and $\lambda _ { s }$ weights the stream; for states the average runs over body-part groups (Equation (6)). The WAM mode generates future video and the state offsets of Equation (2) from the first frame $z _ { 0 } ,$ the measured state $q _ { 0 }$ and the instruction $c .$ History modes instead condition the next block on an ordered tuple $A _ { b }$ of four independently encoded RGB anchors:

$$
\begin{align*}p_{\theta}\big(z_{1:H},\delta_{1:H} \mid z_{0},q_{0},c\big), \quad & \\p_{\theta}\big(z_{b+1} \mid A_{b},c_{b+1}\big), \qquad & p_{\theta}\big(z_{b+1} \mid A_{b},U_{b+1},c_{b+1}\big),\end{align*}\tag{4}
$$

where the last two distributions correspond to image-to-video and forward dynamics with history, respectively, and $U _ { b + 1 }$ contains the supplied state-trajectory tokens. The anchors may repeat when history is short; their exact positions are specified in Section B.2.

An eligible example can be used in several modes; the admissible subsets depend on available modalities, text and outcome labels (Sections B.4 and 4.2).

The fixed mixture gives three quarters of the examples to predicting video, with or without the robot’s trajectory, and a quarter to producing robot states through inverse dynamics and WAM. Among the autoregressive examples, forward dynamics with history receives twice the share of image-to-video with history.

The modes share a physical-time grid for video and state targets. Visual and dynamics modes use whole prepared subtasks; autoregressive modes continue across complete subtask blocks, while WAM predicts a short future inside a successful demonstration. The exact horizons, resampling rules and condition–target arrangement are given in Section B.2 and Figure 29.

7

<!-- page 8 of 43 -->

RealMan AIDAL grippers

Table 2 Training modes. Shares are sampling probabilities over examples, not over tokens or compute.

| Mode | Clean conditions | Targets | Share |
| --- | --- | --- | --- |
| Text-to-video | Visual caption | Subtask video | 20% |
| Image-to-video | Visual caption, initial RGB | Subtask video | 10% |
| Image-to-video with history | Block caption, four RGB anchors from history | Next block video | 10% |
| Forward dynamics | Task instruction, initial RGB, future states | Subtask video | 15% |
| Forward dynamics with history | Block instruction, four RGB anchors, future states | Next block video | 20% |
| Inverse dynamics | Observed video, inverse instruction | Absolute state trajectory incl. q0 | 10% |
| World-action (WAM) | Task instruction, initial RGB, measured q0 | Future video and state offsets | 15% |

## 3.4 Multi-View Observations and Temporal Context

Our corpus contains episodes with either one camera or three synchronized cameras. For three-view sources, we compose the head and two wrist views offline into one video per contiguous kept segment (Figure 5). The head view occupies approximately the top two thirds, with the wrist views below in a fixed Left | Right order. Choosing among five 480p canvas buckets preserves the head-camera aspect ratio and keeps the aspect-ratio change of every panel within 0.94–1.15; single-camera sources use the full frame. Camera roles are verified per source by correlating view motion with left- and right-arm motion. Canvas sizes and temporal sampling are specified in Section B.2.

The autoregressive modes condition each new block on sparse visual history, as in Genie Envisioner and GE-Sim 2.0 [47, 63]. As in Genie Envisioner, the generator receives four RGB anchors; unlike its random draws during training, ours sit at the same fixed positions in training and at inference: the start, one third, two thirds and end of the available history on the effective video grid, so the last anchor is always the most recent frame. Each anchor is encoded separately by the VAE; at the start of an episode the initial frame is repeated. Training uses ground-truth history (teacher forcing); at inference the history is the generated video. History may cross neighboring kept subtasks of the same episode and camera, and it is reset at rejected gaps and episode boundaries. Each block carries its own caption or instruction.

![Image block](doc:f38ca8c/tier:basic/page:8/block:7)

![Image block](doc:f38ca8c/tier:basic/page:8/block:8)

![Image block](doc:f38ca8c/tier:basic/page:8/block:9)

![Image block](doc:f38ca8c/tier:basic/page:8/block:10)

![Image block](doc:f38ca8c/tier:basic/page:8/block:11)

AgileX Cobot Magic grippers

Figure 5 Training canvases of five robot families. From left: Unitree G1 and AIRBOT MMK2 with dexterous hands, RealMan AIDAL, AgileX Cobot Magic and AgiBot G2 with grippers; all recorded on real robots except AgiBot G2, which comes from simulation. The head camera occupies the top two thirds of the canvas, and the left and right wrist cameras share the bottom row (dashed lines); each view is resized to its panel without cropping. All five use the 640×640 canvas, the most frequent of the five canvas sizes.

## 4 Robot Experience and Training

Robot datasets differ in frame rate, camera layout, state semantics and task boundaries, yet every training example must align a robot’s motion with the video and language of the same interval. We cut recordings

8

<!-- page 9 of 43 -->

![Image block](doc:f38ca8c/tier:basic/page:9/block:1)

into subtask clips, filter unusable ones and derive separate text for visual prediction and for action learning, and then train the robotics stage. Application-specific adaptations are described with their evaluations in Sections 5.2 and 6.

## 4.1 Cross-Embodiment Corpus

**Families.** The training data span 20 main robot families (Figure 26), all two-armed: bimanual platforms with parallel grippers and humanoids with grippers or dexterous hands, in real and simulated scenes. Section A lists the layouts and the subsets admitted to training.

They combine public and internal robot datasets in LeRobot format [13] with our humanoid data. The training unit is an atom: a subtask clip from one episode, with synchronized views, measured states, text and provenance. Section A gives the admission rules and the counts of each stage.

## 4.2 Data Curation and Training Mixture

Public robot datasets come as whole episodes, each with its own labels, camera set and quality. Sampling fixed-length windows from them would place unrelated subtasks in one example and pair each window with a label written for a different interval. We therefore prepare the data in stages, each adding one decision without rewriting earlier ones (Figure 6). Registration fixes the camera roles, state layout and robot properties of each source, and cutting into atoms yields the admitted corpus of 1.44 million whole-subtask atoms (1 688.5 h); the quality pass keeps 1.43 million atoms (1 674.7 h); captioning and command derivation give 1.19 million atoms an accepted command.

**Admission.** We first resolve semantic subtask boundaries from the source annotations. An episode is used whole only when it has no finer temporal annotation; ambiguous or incomplete boundaries are excluded. A

(b) Selecting subtask clips

1 6XEWDVNVHJPHQWDWLRQ

0FOLSVy.K

• idle, duplicated or cut clips

• undecodable video

0FOLSVy.K

RoboCOIN · Unitree G1

Figure 6 Robot data. (a) The corpus: 20 robot families, 1 688.5 h after admission, 74% real, all two-armed, in three embodiment groups; each group is shown by a head-camera frame whose width is proportional to its hours. (b) Episodes are cut into subtask clips (atoms) at annotated subtask boundaries within length limits (Equation (10)) (1.44 million clips, 1 688.5 h); video-quality filtering removes idle, duplicated, cut and undecodable clips (1.43 million clips, 1 674.7 h). (c) One clip (RoboCOIN, Unitree G1 with dexterous hands, real), shown at the boundaries of its action phases. Its visual description and derived command are quoted in part, verbatim; its recorded state trajectory shows the three joints that move most of the left arm and of the left hand, each scaled to its own range. Arrows name the training modes that use each output, with their shares of examples (Table 2); the state trajectory serves all three action modes, as a condition of forward dynamics and as the target of world–action prediction and inverse dynamics, so the shares overlap.

9

<!-- page 10 of 43 -->

length limit rejects overlong atoms without cutting them into arbitrary windows (Figure 6b). A separate quality pass removes idle intervals, duplicated frame runs, temporal cuts and unreadable clips (Figure 6b). Passing the video-quality check does not by itself make an atom available to every training mode: the text-dependent modes also need an accepted caption or command (Section A.2 and Table 4).

![Image block](doc:f38ca8c/tier:basic/page:10/block:2)

Figure 7 Captioner and command pass. The inputs are the head-camera clip of Figure 6c with its duration, subtask label, embodiment, effector and domain. The captioner (Qwen3.8-27B [64]) writes nine fields in one dialogue, one field per turn; the attempt record and the consistency field (red) are checks and never enter a training prompt. Blue dots mark the fields that form the visual description (fields from subjects through context, action phases and the temporal caption); green dots mark the fields the command pass reads (context, action phases and the temporal caption). Content gates test forbidden phrases, the effector against the registry, and contact and completion claims against the attempt record; a failing field is rewritten at most twice, and the whole dialogue is attempted at most four times. The command pass turns the action phases into one command per phase, keeping all phases in order; shown are the opening words of the three commands derived for the input clip. Bottom: the visual description conditions text-to-video and image-to-video, the command conditions forward dynamics and world–action prediction, and inverse dynamics receives a fixed instruction.

**From captions to instructions.** The text pipeline produces a visual description and a prospective command for each atom (Figure 6c and Figure 7). A video captioner (Qwen3.8-27B [64]) writes a structured record one field per turn in a single dialogue, so each field is conditioned on those already written: subjects (the robot and its effector), background, lighting, aesthetics, context, an attempt record, timed action phases, a temporal caption and a consistency check. The source task label supplies the intended goal, and registry facts fix the embodiment, effector type and real or simulated domain. Before the action phases, the captioner records how far the attempt got, reading it from the final frame rather than from the label; later fields must end where the clip ends, and a caption whose motion does not match the object and kind of action named by the label is rejected. A second, text-only stage converts the action phases into a prospective command for the same fragment. It preserves the observed motions, object references and phase order, including partial or failed attempts, rather than completing an intended task that is not demonstrated. Using the source label as the command would instead pair an unfinished attempt with a goal that the clip never reaches. The conversion does not see native dataset instructions. Visual modes use the visual description; forward dynamics and WAM use the derived command; inverse dynamics uses a fixed trajectory-recovery instruction. Section B.3 details the captioner, the conversion, their checks and the final prompt format.

**Aligned training inputs.** Camera selection and ordering are explicit for each source. Synchronized RGB views are assembled into one canvas while retaining the original episode timestamps for states and text. Frame-rate reduction selects corresponding video and state times; it does not rescale physical joint positions or displacements. Autoregressive examples join complete adjacent atoms and stop at rejected gaps. WAM additionally requires a positive success label; failed, unknown and unlabeled outcomes are excluded from WAM but can enter other eligible modes. Its short prediction window retains the command of the enclosing atom, so the instruction describes the ongoing subtask rather than a task that must finish within that window.

10

<!-- page 11 of 43 -->

**State-derived targets.** Recorded action columns mix joint targets, velocities, teleoperation master states and progress signals. We instead construct a common trajectory from measured states with the codec of Equation (1):

$$
q _ { k } = \mathrm { c o d e c } _ { \ell } \big ( s ( t _ { 0 } + \Delta _ { k } ) \big ) ,\tag{5}
$$

where 𝑠 is the recorded state, $\Delta _ { k }$ is a physical time offset within one admitted interval and codec maps layout ℓ to the shared anatomical slots. Inverse dynamics predicts absolute $q _ { k } ;$ forward dynamics and WAM use the targets $\delta _ { k }$ of Equation (2). The targets therefore describe realized robot motion. Executing a predicted trajectory additionally requires the robot-specific decoder of Section 6.2.

Normalization statistics are computed on the training split only (Section A.3).

**Training mixture.** Sampling chooses a mode, a robot cluster (Equation (12)), a dataset and a window. The Worker/ARX5 and Green humanoid clusters receive fixed shares; other clusters and datasets are weighted by the square root of their admissible subtask counts. Each source is sampled only in the modes its data support. Full sampling targets are given in Section B.4 and Table 8.

## 4.3 GreenWAM Training

**Objective.** The robotics stage trains all seven modes with the rectified-flow objective (Equation (3)). The generator predicts the velocity of each noised target stream. Video loss averages over valid target elements. State loss first averages within each represented body-part group and then across groups, so a group with more coordinates does not automatically receive more weight. Let $e _ { t j } = \hat { v } _ { t j } - v _ { t j }$ be the velocity error of the state targets in Equation (5), $m _ { t j }$ select valid noised targets, and 𝒢 contain the body-part groups with at least one such target. Then

$$
\mathcal { L } _ { \mathrm { s t a t e } } = \frac { 1 } { | \mathcal { G } | } \sum _ { g \in \mathcal { G } } \frac { \sum _ { t } \sum _ { j \in g } m _ { t j }   e _ { t j } ^ { 2 } } { \sum _ { t } \sum _ { j \in g } m _ { t j } } , \qquad \mathcal { L } = \lambda   \big ( \mathcal { L } _ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { s t a t e } } \big ) .\tag{6}
$$

All generated streams share the noise level 𝜎 of Equation (3), drawn from the noise schedule of Cosmos 3 Edge (Section B). The loss carries no noise-level weight, and both terms are scaled by $\lambda = 1 0$ , as in Cosmos 3 Edge, so video and state carry equal weight. Losses are averaged over examples, and a stream without targets contributes zero. The anatomical groups are legs, body, arms and hands; unresolved and absent coordinates are masked. Clean conditions, including the measured $q _ { 0 }$ in WAM, carry no loss; in inverse dynamics the initial state is a target. The RoboTwin policy recipe instead weights valid action coordinates equally (Section B.5). Text conditioning is dropped with probability 0.1 for classifier-free guidance [32].

With the reasoner and VAE frozen, training runs on 64 H100 GPUs with a global batch of 256 examples. Parameters are sharded within each node of 8 GPUs and replicated across nodes and attention is checkpointed selectively. The full optimizer, schedule and sampling recipe are given in Table 5 and Section B.

**Training scale.** The robotics stage produces GreenWAM after 36 500 optimizer steps, corresponding to 9.34 million sampled examples. This checkpoint initializes downstream adaptations. The learning-rate schedule and checkpoint settings are given in Table 5. Sections 5 and 6 evaluate this checkpoint as a world model and as a policy.

## 5 GreenWAM as a World Model

Section 5.1 tests the robotics checkpoint itself, without any benchmark adaptation, on held-out recordings of robots from the training corpus: whether one model predicts video for different bodies and whether the generated motion follows the commanded trajectory. Section 5.2 adapts the checkpoint to WorldArena 1.0, a public benchmark of action-conditioned video prediction, and compares it with the models on its leaderboard. All results in this section use the generator alone, without the action expert of the policies.

11

<!-- page 12 of 43 -->

## 5.1 The Robotics Checkpoint on Held-Out Recordings

We take one held-out recording for each of six robot families: AgiBot G2A, AgileX, Unitree G1, Fourier GR1, Worker/ARX5 and the Green humanoid, the last both in simulation and on the real robot. Given the first frame and the recorded joint trajectory, the model generates the video of the whole recording (Figure 1e).

Unless stated otherwise, sampling uses 35 steps, classifier-free guidance 6 and shift 5, and comparisons between checkpoints keep the sampler fixed. Protocols and evaluation details are given in Sections C and C.5. **Action following.** A world model is useful for control only if its video follows the commanded actions instead of continuing the scene on its own. For each of these recordings we keep the first frame and multiply the commanded displacement from the starting pose by 0, 0.5, 1 and 1.5: zero commands the robot to stand still, and 1 replays the recording (Figure 8). On every robot the generated motion grows with the command. With zero displacement the video keeps less than a fifth of the motion it shows under the recorded command, and at one and a half times the displacement it shows about a quarter more; motion is the mean change between consecutive frames. Trajectory accuracy is measured on WorldArena (Section 5.2).

## One start frame, four commanded actions

Unitree G1, head camera · recorded actions scaled before prediction

![Image block](doc:f38ca8c/tier:basic/page:12/block:6)

Figure 8 Action following. One start frame of Unitree G1 (head camera) predicted forward with the recorded state offsets scaled by 0, 0.5, 1 and 1.5; each column shows the halfway point (top) and the end (bottom) of the prediction. With zero offsets the bread stays on the table; with the other three scales it ends on the plate, and at the halfway point the larger the scale, the further the arm has moved.

## 5.2 WorldArena Adaptation and Results

WorldArena 1.0 is a public benchmark of embodied video generation; its track 1 scores 15 metrics grouped into visual quality, motion quality, content consistency, physics adherence, 3D accuracy and controllability [70]. We adapt GreenWAM on the training data that the benchmark allows and predict its test episodes from their recorded actions. GreenWAM is strongest where the prediction must match the recorded motion and geometry: among the compared world–action models and video generators it has the highest trajectory, depth and 3D accuracy (56.0, 98.2 and 93.4), and among all 127 entries of the board it is second in JEPA similarity (98.41, after BLM at 98.49).

The action variant of track 1 gives the first frame and the robot actions of a held-out episode and asks for the video. Figure 10 shows two such episodes. This variant maps onto forward dynamics with history: every block is conditioned on the recorded actions and on four anchors of the history, which for the first block repeat the recorded first frame (Section B.5). The text prompt is empty, so sampling uses 35 steps without guidance (Section C.5). The benchmark’s embodiment has the ARX5 arms of RoboTwin and reuses its layout extension, so the benchmark’s actions enter through the shared projection of GreenWAM rather than an action projection of their own (Section 3.2).

The overall score averages the board’s 15 metrics after capping the three motion metrics at the ground truth, so that a model cannot gain by moving more than the recorded robot:

$$
\mathrm { E W M S c o r e } = \frac { 1 } { 1 5 } \sum _ { k = 1 } ^ { 1 5 } \tilde { m } _ { k } , \qquad \tilde { m } _ { k } = \begin{cases} \min \big ( m _ { k } ,   m _ { k } ^ { \mathrm { G T } } \big ) , & k \in \mathcal { K } _ { \mathrm { m o t i o n } } , \\ m _ { k } & \mathrm { o t h e r w i s e } , \end{cases}\tag{7}
$$

where $\mathcal { K } _ { \mathrm { m o t i o n } }$ contains dynamic degree, flow score and motion smoothness. The cap is needed because these metrics reward motion beyond the recording; the organizers of WorldArena 2.0 found that they reach

12

<!-- page 13 of 43 -->

Overall and categories

![Chart block](doc:f38ca8c/tier:basic/page:13/block:2)

Figure 9 WorldArena 1.0, track 1: GreenWAM among models of the leaderboard. Fidelity: agreement with the ground-truth video. JEPA similarity compares the distributions of V-JEPA features of real and generated videos at the set level (JEDi [54]; higher is better); trajectory and depth accuracy are computed per video. Overall and categories: EWMScore (Equation (7)), the mean of the 15 board metrics with dynamic degree, flow score and motion smoothness capped at the ground truth, and three of its categories. GreenWAM is capped per video on all 1 000 test episodes; for the other models the caps are applied to their leaderboard means [84], which gives them an upper bound. Controllability, visual quality and motion quality also enter EWMScore and are listed with all categories in Table 11. Cut marks identify axes that do not start at zero.

abnormally high values without better modeling of the dynamics and adopted the same rule [83]. Of the 127 entries of the WorldArena 1.0 board, 112 exceed the recorded-video reference in dynamic degree and 113 in flow score, the leading ones by up to four and eight times. Under the cap the scores of the leading entries drop by 6 to 11, UNIS for example from 73.64 to 63.65, while GreenWAM, whose motion scores stay below the reference, loses 0.05.

Under this aggregate GreenWAM reaches 63.31 on the 1 000 test episodes (Figure 9), above the published world–action models Motubrain [56] (63.16) and FlowWAM-FiveAges [21] (62.99), whose recomputed scores are upper bounds. Scores of other models are recomputed from their leaderboard entries under the same motion cap, so they can differ from the values quoted in their reports.

**The recorded videos as a reference.** We also score the real test videos with the same evaluator. They reach 67.23, the score of a model that would reproduce every recording exactly, and on 13 of the 15 metrics GreenWAM differs from them by less than four. The two categories in which it trails, visual quality and controllability, do not compare a video with its recording: visual quality rates image appearance, and controllability is assigned by a vision–language judge. On these categories many board entries score above the real videos themselves, so they follow image appearance more than agreement with the recorded episode (Section C.1).

13

<!-- page 14 of 43 -->

WorldArena 1.0, long rollouts

WKHRQO\IUDPHWDNHQIURPWKHUHFRUGLQJ

episode 944 RSHQWKHíDSyVy3615G%DERXWWKHPHGLDQHSLVRGH

![Image block](doc:f38ca8c/tier:basic/page:14/block:4)

episode 213 PRYHWKHERWWOHVyVy3615G%DERXWWKHPHGLDQHSLVRGH

![Image block](doc:f38ca8c/tier:basic/page:14/block:6)

Figure 10 Long WorldArena rollouts. Two test episodes of WorldArena 1.0, “open the $\mathrm{gap}^{\prime \prime}$ and “move the bottles”, each over 20 s, with the recording above the prediction of GreenWAM at six moments. Only the first frame, outlined, comes from the recording; the model then follows the recorded actions in blocks of about $\operatorname { 7 s } ,$ each conditioned on four of its own generated frames.

## 6 GreenWAM as a Policy

We adapt GreenWAM to control by predicting a future state trajectory together with video. Sections 6.1 and 6.2 describe how a policy is obtained: the action expert split off the generator and the post-training common to all benchmarks, while each benchmark keeps its own demonstrations, command interface and inference settings. Section 6.3 reports the four simulation benchmarks. Real-time control on the humanoid benefits from a faster policy, so Section 6.4 distills the policy into a one-step student and measures its latency, and Section 6.5 evaluates the teacher and the student on the robot.

## 6.1 Policy Architecture

In the robotics stage of GreenWAM, video and action tokens pass through one generator. For policy adap tation, action tokens move to a dedicated transformer while video tokens remain in the generator. Joint attention connects the two streams at every layer, so action prediction retains access to visual dynamics while the streams acquire separate parameters during adaptation. Related expert architectures differ in how the expert is initialized [7, 99].

Let $h ^ { z }$ and $h ^ { q }$ be the video and action tokens entering a layer. The generator 𝐺 projects the video tokens and the expert $E$ the action tokens, attention runs over both streams together, and each stream keeps its own feed-forward block:

$$
( Q , K , V ) ^ { z } = W _ { G }   h ^ { z } , \quad ( Q , K , V ) ^ { q } = W _ { E }   h ^ { q } , \qquad [ \tilde { h } ^ { z } ; \tilde { h } ^ { q } ] = \mathrm { A t t n } \big ( [ Q ^ { z } ; Q ^ { q } ] , [ K ^ { z } ; K ^ { q } ] , [ V ^ { z } ; V ^ { q } ] \big ) .\tag{8}
$$

We initialize the action expert by copying the generator tower, $\theta _ { E } \leftarrow \theta _ { G }$ . The expert has the same shape as the parent tower, so the split does not change the predictions at initialization. The towers then diverge

14

<!-- page 15 of 43 -->

during post-training. The expert adds as many parameters as the generator tower and about $8 \%$ to the time of a training step. Section 7.2 measures what the split brings.

## 6.2 Policy Post-Training

Policy adaptation connects joint video–state prediction to a target robot’s control loop. Each benchmark specifies its demonstrations, camera layout and command interface; the common learning objective is the WAM mode of GreenWAM (Equation (4); recipes in Section B.5).

**Shared recipe.** The policy receives a task instruction, the current camera canvas and the measured state $q _ { 0 } ,$ and predicts future video together with state offsets over horizon 𝐻. Training uses demonstrations of the target robot, with measured future states as supervision. At inference, the predicted targets are decoded relative to $q _ { 0 }$ (Equation (2)) into the robot’s native command representation. The controller executes a prefix of the prediction, acquires a new observation and replans. The decoder and the execution interval are specific to each application.

**Appearance augmentation.** Clean demonstrations show one appearance of each scene, and benchmarks that randomize scenes test others. Two augmentations address this without touching the geometry. Background replacement: in half of the samples the static background (table, walls, floor), segmented by replaying the demonstration in the simulator, is replaced by a procedural texture modulated by the original shading, so that shadows are preserved; it needs the simulator for segmentation and is used on RoboTwin. Color augmentation perturbs the colors of each camera image independently; it is applied to 80% of the samples and adds no crops or rotations. Section 7.5 measures what background replacement brings.

The per-benchmark recipes, camera layouts, data volumes and execution settings are given in Section B.5.

## 6.3 Simulation Benchmarks

We evaluate the adapted policies on RoboTwin 2.0, CALVIN, RoboDojo and LIBERO. The target benchmark protocols are specified in Table 9 and Section C. No RoboTwin, RoboDojo, CALVIN or LIBERO data enter the corpus of GreenWAM. Policies use benchmark-specific denoising steps, guidance and replanning intervals (Section B.5). Results of other methods are taken from their reports or leaderboards, as noted in each table; Figure 11 summarizes the four benchmarks, and Section D.1 gives the full CALVIN, RoboDojo and LIBERO tables.

## 6.3.1 RoboTwin 2.0

On RoboTwin 2.0, GreenWAM scores among the top entries of the leaderboard (Table 3); Figure 12 shows one task in the clean and the randomized setting. The leaderboard fixes the post-training data, 50 clean demonstrations per task, but not the pretraining.

Our RoboTwin policy reaches 86.98% clean and 48.10% randomized success, averaging 67.54%, over 50 tasks with 100 trials per task in each setting. It is evaluated under the leaderboard protocol on the official seeds. Controlled comparisons at a fixed adaptation budget support both design choices: initializing from GreenWAM and splitting off an action expert at the start of adaptation together raise clean success by 10.8% over a single-tower policy adapted from Cosmos 3 Edge (Sections 7.1 and 7.2). The leaderboard trains on clean demonstrations only; Section 7.5 examines how individual scene changes affect randomized success.

The GreenWAM corpus contains no RoboTwin data, whereas pretraining can expose a policy to randomized RoboTwin demonstrations. ME-Dex-1.0 inherits Motus experts [99], and Motus and X-WAM list 27 500 RoboTwin pretraining episodes, matching the set of 2 500 clean and 25 000 randomized demonstrations [7, 30]; WorldScape Policy 2.0 lists 35.7 thousand without specifying their split [74]. Table 3 marks the two leaderboard entries whose pretraining includes randomized RoboTwin scenes.

## 6.3.2 CALVIN

CALVIN tests whether a policy can chain language-conditioned skills in an unseen environment: each successful instruction advances a chain of five without resetting the scene, so earlier errors carry over to later instructions. We adapt on environments A, B and C, with no data from D, and evaluate on 1 000 chains in

15

<!-- page 16 of 43 -->

Table 3 RoboTwin 2.0, 50 tasks. Average success rate, $\% .$

<table><tr><td>Method</td><td>Clean</td><td>Randomized</td><td>Average</td></tr><tr><td colspan="4">Leaderboard: 50 clean demonstrations per task</td></tr><tr><td>ME-Dex-1.0 [99]†</td><td>89.6</td><td>68.1</td><td>78.8</td></tr><tr><td>OLA-Sem</td><td>75.1</td><td>67.6</td><td>71.3</td></tr><tr><td>GigaBrain-0.7 [26]</td><td>66.8</td><td>67.9</td><td>67.3</td></tr><tr><td>WorldScape Policy 2.0 [74]</td><td>84.9</td><td>38.5</td><td>61.7</td></tr><tr><td>4D-WAM [88]</td><td>81.5</td><td>41.8</td><td>61.6</td></tr><tr><td> $\pi _ { 0 . 5 } [ 6 2 ]$ </td><td>70.7</td><td>46.0</td><td>58.4</td></tr><tr><td>OLA-Geo</td><td>83.0</td><td>33.6</td><td>58.3</td></tr><tr><td>Spatial Forcing [41]</td><td>77.2</td><td>26.7</td><td>52.0</td></tr><tr><td>X-WAM† [30]</td><td>70.0</td><td>25.8</td><td>47.9</td></tr><tr><td>X-VLA [103]</td><td>68.0</td><td>20.9</td><td>44.5</td></tr><tr><td>ABot-M0 [90]</td><td>57.4</td><td>30.4</td><td>43.9</td></tr><tr><td>EventVLA [87]</td><td>65.6</td><td>15.7</td><td>40.6</td></tr><tr><td>Xiaomi-Robotics-0 [15]</td><td>62.9</td><td>18.2</td><td>40.5</td></tr><tr><td>Fast-WAM [95]</td><td>77.8</td><td>1.9</td><td>39.9</td></tr><tr><td>GalaxeaVLA [35]</td><td>62.7</td><td>12.7</td><td>37.7</td></tr><tr><td>AHA-WAM [14]</td><td>64.3</td><td>3.2</td><td>33.8</td></tr><tr><td>starVLA [73]</td><td>46.5</td><td>3.2</td><td>24.8</td></tr><tr><td>GreenWAM, leaderboard protocol</td><td>86.98</td><td>48.10</td><td>67.54</td></tr><tr><td colspan="4">Outside the leaderboard: clean demonstrations only, own evaluation</td></tr><tr><td>Qwen-RobotManip-Context [94]</td><td>84.7</td><td>69.4</td><td>77.05</td></tr><tr><td colspan="4">Published: clean and randomized demonstrations</td></tr><tr><td>π0 [11]</td><td>65.92</td><td>58.40</td><td>62.16</td></tr><tr><td>[62]  $\pi _ { 0 . 5 }$ </td><td>82.74</td><td>76.76</td><td>79.75</td></tr><tr><td>X-VLA [103]</td><td>72.80</td><td>72.84</td><td>72.82</td></tr><tr><td>DeMaVLA [75]</td><td>88.42</td><td>86.78</td><td>87.60</td></tr><tr><td>Motus [7]</td><td>88.66</td><td>87.02</td><td>87.84</td></tr><tr><td>Fast-WAM [95]</td><td>91.88</td><td>91.78</td><td>91.83</td></tr><tr><td>DSWAM [106]</td><td>92.38</td><td>91.90</td><td>92.14</td></tr><tr><td>LingBot-VA [42]</td><td>92.9</td><td>91.6</td><td>92.25</td></tr><tr><td>LingBot-VA 2.0 [97]</td><td>93.8</td><td>93.4</td><td>93.6</td></tr><tr><td>Motubrain [56]</td><td>95.8</td><td>96.1</td><td>95.95</td></tr></table>

Upper block: the official leaderboard, which trains on 50 clean demonstrations per task (co-train track, ordered by the leaderboard average of clean and randomized success; the single-task baselines $\pi _ { 0 } ,$ DP3, RDT, ACT and DP, at most 31.4 on average, are omitted) [68]; †: the checkpoint that post-training starts from was trained on randomized RoboTwin scenes. The reports of GigaBrain-0.7, WorldScape Policy 2.0 and OLA-Geo, and of Qwen-RobotManip outside the leaderboard, do not state whether their pretraining included randomized RoboTwin scenes; for the other entries of these two blocks it was not. Outside the leaderboard: a report that also trains on clean demonstrations only but uses its own evaluation: Qwen-RobotManip-Context with joint control [94] (Table 5, easy and hard settings). Lower block: published results whose training adds 500 randomized demonstrations per task (stated for Motus, Fast-WAM, LingBot-VA, LingBot-VA 2.0 and Motubrain), as reported in DSWAM [106] (Table 1) for $\pi _ { 0 } ,   \pi _ { 0 . 5 }$ , DeMaVLA, Motus, Fast-WAM and DSWAM; in Motus [7] (Table 2) for X-VLA; in the LingBot-VA repository [66] for LingBot-VA; in the reports of LingBot-VA 2.0 [97] (Table 1) and Motubrain [56]. GreenWAM follows the leaderboard protocol and is listed under its block.

D (Table 12; protocol in Section C). GreenWAM completes 4.64 instructions per chain on average, with a standard error of about 0.03, above FLOWER (4.53).

## 6.3.3 RoboDojo

RoboDojo tests generalization, precision, long-horizon execution, memory and open-vocabulary instructions in simulation (Table 13). Its tasks use the same ARX5 arms as RoboTwin. The RoboDojo aggregate gives

16

<!-- page 17 of 43 -->

![Chart block](doc:f38ca8c/tier:basic/page:17/block:1)

RoboTwin 2.0

![Chart block](doc:f38ca8c/tier:basic/page:17/block:3)

CALVIN ABC→D

![Chart block](doc:f38ca8c/tier:basic/page:17/block:5)

![Chart block](doc:f38ca8c/tier:basic/page:17/block:6)

LIBERO

![Chart block](doc:f38ca8c/tier:basic/page:17/block:8)

LIBERO-plus, zero-shot

Figure 11 Manipulation benchmarks. GreenWAM (first bar of each panel) and selected baselines from Tables 3, 12 and 13 and from published LIBERO [48] and LIBERO-plus [24] results. RoboTwin reports mean clean/randomized success over 50 tasks with 100 trials per task; pretraining on randomized RoboTwin scenes is marked in Table 3. LIBERO reports the average success over its four task suites and LIBERO-plus the zero-shot success over all perturbed episodes; GreenWAM values on both come from the same policy (Table 14). LIBERO and LIBERO-plus values and their sources are listed in Table 14. Cut marks identify axes that do not start at zero.

equal weight to its five capability dimensions rather than to every individual task. Trained on the official training set and compared with the official leaderboard, GreenWAM reaches a score of 16.12 and a success rate of 10.49% (Table 13), higher on both aggregates than $\pi _ { 0 . 5 } ,$ X-VLA and X-WAM. The recipe is given in Section B.5.

17

<!-- page 18 of 43 -->

RoboTwin 2.0 · “stack blocks two”

![Image block](doc:f38ca8c/tier:basic/page:18/block:2)

Figure 12 RoboTwin 2.0 rollouts of the GreenWAM policy. The task stack\_blocks\_two in a clean (top) and a randomized scene (bottom). Head-camera frames at six phases: start, red block grasped, red block placed, green block grasped, green block carried, tower built. Both episodes succeed.

## 6.3.4 LIBERO and LIBERO-plus

LIBERO evaluates single-arm manipulation with a Franka arm in simulation over four task suites that vary spatial layout, objects, goals and task length [48]. Recent policies exceed 96% average success on it. LIBERO-plus keeps the LIBERO tasks and perturbs them along seven axes, from camera viewpoint to object layout [24]. In the zero-shot setting a policy trained on standard LIBERO is evaluated on the perturbed episodes, which exposes robustness gaps that the standard suites hide: Fast-WAM [95], for example, reaches 97.6% on LIBERO and 49.1% on LIBERO-plus [100] (Table 14). The single-arm Franka embodiment is absent from the robotics corpus, whose families are all two-armed. With one policy, GreenWAM is close to the dedicated policies on LIBERO and reaches the highest zero-shot LIBERO-plus success in Table 14; Figure 31 shows typical failures.

![Chart block](doc:f38ca8c/tier:basic/page:18/block:6)

Figure 13 LIBERO-plus by axis. Zero-shot success; baselines from [52].

Success is highest under lighting, sensor-noise and language perturbations, where GreenWAM leads both baselines (Figure 13). Under changed viewpoints, success is 81.7%, above both baselines. Changed backgrounds are handled without the background replacement used on RoboTwin, which needs simulator segmentation of the demonstrations.

## 6.4 Distillation for Real-Time Control

We distill the adapted policy into a one-step student with modality-specific consistency objectives, following Flash-WAM [4] (Figure 14). The student keeps the teacher’s observations, state representation and joint video–action outputs; only its sampling changes at deployment. We initialize a frozen teacher, a trainable student and an EMA target student from the adapted policy. Unlike Flash-WAM, which uses separate noise schedules for video and actions, video latents and the action chunk share one noise level. The teacher steps between neighboring levels of a shared schedule of 4 noise levels, from a noisier sample to a less noisy one. The teacher makes two passes, with the instruction and with an empty prompt, and applies guidance to video (scale 3–4) but not to actions; the student learns to produce a consistent clean prediction at both points, with the lower-noise target supplied by the EMA student and detached from the gradient.

Video and action targets use separate consistency maps. The video map uses noise-dependent skip and output coefficients, while the action map directly estimates the clean state target as $x _ { \sigma } - \sigma \hat { v } _ { \theta } .$ An action flow-matching term, which Flash-WAM uses only in a regularized baseline, keeps the student anchored to the demonstration targets:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \lambda _ { v } \mathcal { L } _ { \mathrm { C D } , v } + \lambda _ { a } \mathcal { L } _ { \mathrm { C D } , a } + \lambda _ { \mathrm { F M } } \mathcal { L } _ { \mathrm { F M } , a } .\tag{9}
$$

Loss distances, weights and the EMA decay are given in Section B.5.

18

<!-- page 19 of 43 -->

![Image block](doc:f38ca8c/tier:basic/page:19/block:1)

Figure 14 Consistency distillation of the policy. The losses are those of Equation (9). (a) One step on the shared noise path: the teacher steps from $x _ { s }$ to $\tilde { x } _ { e }$ between neighboring anchors of the schedule, and the student at $x _ { s }$ and the exponential-moving-average (EMA) target at $\tilde { x } _ { e }$ must give the same clean estimate. (b) Training: a frozen teacher, the trained student and its EMA copy as the target; video and action consistency losses and an action flow-matching term, with the gradient to the student only. (c) On the robot the student replaces the teacher’s guided multi-step sampling with one step and one pass. The recipe follows Flash-WAM [4], with one noise level for both streams and an added action flow-matching term.

![Chart block](doc:f38ca8c/tier:basic/page:19/block:3)

Figure 15 Policy inference latency on one NVIDIA RTX 4090. Median latency per policy call over 100 episodes. The optimizations are applied cumulatively to the GreenWAM policy; the light segment of each row is the time removed by that step. Speed-ups are relative to GreenWAM without optimizations; Table 10 also gives the time to the first action.

**Latency.** Caching the instruction and vision key–value states across denoising steps, evaluating both branches of classifier-free guidance in one pass, encoding only the conditioning frame, compiling the generator and running the feed-forward layers with 8-bit weights and activations bring the GreenWAM policy to 134.6 ms (Figure 15). Distillation then replaces the four guided denoising steps with one consistency step: the distilled policy answers in 35.4 ms, 3.8 times faster than the optimized teacher and 12.3 times faster than the unoptimized policy, at 28.3 Hz. Distillation pays off most on the robot’s onboard computer, whose compute budget is well below that of an RTX 4090; running the student there is our next step (Section 8). Section C.3 describes the benchmark and each optimization; Section 6.5 reports the task success of the distilled policy on the humanoid.

19

<!-- page 20 of 43 -->

## 6.5 Real-World Humanoid Evaluation

Green humanoid, one continuous episode

![Image block](doc:f38ca8c/tier:basic/page:20/block:3)

start

![Image block](doc:f38ca8c/tier:basic/page:20/block:5)

grasp cap

cap into bag

bottle into bag

![Image block](doc:f38ca8c/tier:basic/page:20/block:9)

cap into bag

lift and hand over bag

![Image block](doc:f38ca8c/tier:basic/page:20/block:12)

grasp bottle

![Image block](doc:f38ca8c/tier:basic/page:20/block:14)

bottle into bag

![Image block](doc:f38ca8c/tier:basic/page:20/block:16)

grasp handles

![Image block](doc:f38ca8c/tier:basic/page:20/block:18)

lift bag

![Image block](doc:f38ca8c/tier:basic/page:20/block:20)

hand over

Figure 16 GreenWAM policy inference on the Green humanoid. Eight frames of one continuous episode, cropped from a side camera: the robot puts a cap and a bottle into a bag, lifts the bag by its handles and hands it to a person. Colors mark the three subtasks.

We evaluate the policies on a packing task in which the Green humanoid places a bottle and a cap into a bag, then lifts the loaded bag with both hands. For evaluation the task is given as five consecutive instructions, grasping and placing each object and then lifting the bag; each names the object and, for the single-arm actions, the hand to use, and the scene is not reset between them. Variants use different bottles and assign the pick-and-place steps to the left or right hand. We measure success at each instruction. Figure 16 shows a packing rollout followed by a handover to a person.

We adapt $\pi _ { 0 . 5 }$ [62, 61], Cosmos 3 Edge [58] and GreenWAM using the same demonstration split. We also evaluate the distilled GreenWAM policy (Section 6.4) against its teacher.

**Observation and control.** Every policy receives the task instruction, the current camera observations and the measured robot state. Cameras and controlled joints are the same for all policies; preprocessing and action representation follow each model’s adaptation recipe (Section B.5). GreenWAM, its distilled student and the Cosmos 3 Edge policy predict 16 actions per call, and $\pi _ { 0 . 5 }$ predicts 50.

**In-distribution (ID) and out-of-distribution (OOD) evaluation.** ID trials use new resets of the objects and scenes represented in policy adaptation; OOD trials keep the task but change objects or scene conditions excluded from adaptation. Each policy is evaluated on 70 instructions in 14 scene configurations, 9 ID and 5 OOD, with one attempt per instruction; an instruction succeeds if it is completed before its timeout without human help, and success is the fraction of successful instructions in each split (Figure 17; protocol in Section C.2). GreenWAM succeeds in 95.6% of ID and 84.0% of OOD trials, ahead of the Cosmos 3 Edge policy (88.9% and 80.0%) and $\pi _ { 0 . 5 }$ (84.4% and 76.0%).

**Distilled policy.** The teacher and the distilled student run with the same observations, action horizon and replanning schedule, so their success rates are directly comparable (Figure 17): the student reaches 91.1% ID and 80.0% OOD success.

Across the four simulation benchmarks and the humanoid, the same foundation and the same action interface yield policies for two-armed robots with grippers, a single-arm Franka and a humanoid with dexterous hands. Section 7 isolates the choices behind these results.

20

<!-- page 21 of 43 -->

World model adapted from GreenWAM vs Cosmos 3 Edge

![Chart block](doc:f38ca8c/tier:basic/page:21/block:2)

Figure 17 Instruction success on the Green humanoid. In-distribution and out-of-distribution success of the four policies, the fraction of successful instructions within each split (Equation (14)). The teacher and the distilled student run with the same execution settings.

GreenWAM choice better →

<table><tr><td></td><td></td><td colspan="3">Forward PSNR, dB RoboTwin held-out</td><td colspan="3">First-block PSNR, dB WorldArena</td><td colspan="3">EWMScore WorldArena</td></tr><tr><td>Q1</td><td>Specialized start 3,000 adaptation steps</td><td></td><td>+1.26 T●I</td><td></td><td></td><td>+1.14 ●</td><td></td><td></td><td>+0.23 ●</td><td></td></tr><tr><td></td><td></td><td>-2</td><td>+2</td><td></td><td>0</td><td></td><td></td><td></td><td>0</td><td>+1</td></tr></table>

![Chart block](doc:f38ca8c/tier:basic/page:21/block:6)

Figure 18 Controlled comparisons at a fixed budget. Paired differences (Equation (15)) with 95% intervals from a hierarchical bootstrap over tasks and episodes; the sign is the choice made in GreenWAM minus the alternative, and teal marks intervals that exclude zero. World model: forward PSNR on 50 held-out RoboTwin windows, and first-block PSNR and capped EWMScore on the 1 000 WorldArena test episodes, bootstrapped over episodes; each column has its own scale. Policy: RoboTwin 2.0 success over 50 tasks with 20 trials per task and setting. World-model modes: WAM with visual prediction, that is forward dynamics and image-to-video with weights 0.05 each next to WAM at 0.9, compared with WAM only. Q1: robot specialization, GreenWAM against Cosmos 3 Edge; Q2: an action expert against a single tower, both from GreenWAM; Q3: world-model modes kept during policy adaptation. The row Q1 + Q2 compares GreenWAM with an action expert split off at the start against the single-tower Cosmos 3 Edge run; each pair is estimated on the common seeds of its two runs (Equation (15)), so this row need not equal the sum of Q1 and Q2.

## 7 Empirical Analysis

We test three choices in adapting the shared foundation: robot-specialized initialization, a separate action expert and the retention of world-model modes during policy adaptation. Each comparison holds the downstream data and the adaptation budget fixed (Figure 18). We then examine how individual scene changes affect the policy and what appearance augmentation recovers; evaluation details are given in Section C.5.

21

<!-- page 22 of 43 -->

**Comparisons.** The pairs are specified below; a further comparison combines the first two choices (row Q1 + Q2 in Figure 18).

• **Robot specialization.** GreenWAM and its parent Cosmos 3 Edge are adapted with the same recipe for 3 000 steps: to WorldArena 1.0, evaluated on its 1 000 test episodes, and to RoboTwin 2.0 as single-tower policies, evaluated with 20 trials per task and setting; the two RoboTwin policies are also compared by forward prediction on 50 held-out clips.

• **Action expert.** From GreenWAM, the single-tower RoboTwin policy is compared with one trained by the same recipe but with the action expert split off at the start; 3 000 steps each.

• **World-model modes.** Both runs are warm-started from the same RoboTwin policy, trained with the full recipe: WAM at 0.9, forward dynamics and image-to-video at 0.05 each. One run then continues with this mixture, the other with WAM only; 2 500 steps each.

Both runs of a pair are evaluated with the same evaluation code, sampler and seeds.

**Statistics.** Differences average over tasks, as the benchmarks do, and their 95% intervals come from a hierarchical bootstrap over tasks and episodes (Section C.4).

## 7.1 What Robot Specialization Brings to Each Role

Starting from GreenWAM improves both forward prediction and policy success at a fixed adaptation budget. We adapt GreenWAM and its parent Cosmos 3 Edge with the same recipe and budget, once as a world model and once as a policy. The parent learns its joint-space action projections from scratch, as it must for joint-space actions, while the prompt format, the WAM mode and the multi-camera canvas are native to both.

![Chart block](doc:f38ca8c/tier:basic/page:22/block:9)

Figure 19 Forward PSNR per held-out clip of the policies from GreenWAM and from Cosmos 3 Edge: 38 of 50 RoboTwin clips lie above the diagonal.

![Chart block](doc:f38ca8c/tier:basic/page:22/block:11)

Figure 20 Gain over Cosmos 3 Edge during adaptation. Mean RoboTwin 2.0 success (Figure 18).

**World model.** We adapt both checkpoints to WorldArena with the recipe of Section 5.2 for 3 000 steps and compare them on all 1 000 test episodes. The model adapted from the robot-specialized checkpoint is closer to the recording in 87.1% of the episodes. PSNR rises by 1.1 dB on the first block, generated from the recorded frame alone, and by 1.2 dB on the later blocks, which continue from the model’s own frames, so the error falls by about a quarter and the gain persists through autoregressive continuation. Specialization improves the metrics on which GreenWAM is strongest: trajectory accuracy rises by 3.0, from 44.4 to 47.4, and depth accuracy by 0.5, from 96.65 to 97.12. The aggregate score rises by 0.23, mostly from physics adherence (+1.3).

22

<!-- page 23 of 43 -->

The RoboTwin policies compared below show the same effect on held-out frames. On one held-out window per task, the policy adapted from GreenWAM predicts the next 20 frames 1.3 dB more accurately than the one adapted from the parent and is closer to the recording in 38 of the 50 windows (Figure 19).

**Policy.** Adapted to RoboTwin 2.0 with the single-tower recipe for 3 000 steps, the policy from GreenWAM succeeds more often than the policy from the parent: by 7.9% in clean scenes and 5.4% on average (Figure 18).

The advantage emerges early in training and persists: under 1% after 500 steps, 6.2% at 2 500 and 5.4% at 3 000 (Figure 20). The validation losses behave differently: they start far apart, 8.15 for GreenWAM against 19.61 for the parent, and nearly meet by step 3 000 (1.51 and 1.56), while the gap in closed-loop success remains. The advantage is similar against a parent trained with the WAM mode only (5.2%). Adding the action expert to the specialized start (Section 7.2) widens it to 8.4% on average and 10.8% in clean scenes. On the real humanoid, the comparison of policies adapted from $\pi _ { 0 . 5 }$ , from the parent and from GreenWAM poses the same question (Section 6.5).

## 7.2 Coupling the World Model to Action

Splitting the action expert at the start of adaptation raises policy success while preserving the initial pre dictions. During policy adaptation, GreenWAM moves action tokens into an expert (Equation (8)) that starts as an exact copy of the generator tower, so the split changes the trainable parameterization without changing the initial predictions (Section 6.1). Both runs therefore start from the same function.

Trained for 3 000 steps from GreenWAM with the recipe of Section 7.1, the policy with the expert succeeds 3.3% more often on average than the one that keeps the shared generator, 2.8% in clean and 3.7% in randomized scenes. It raises success on 28 tasks and lowers it on 11 in clean scenes, and on 22 and 7 in randomized scenes. The gain comes with almost equal validation losses (1.503 with the expert and 1.510 without).

## 7.3 Retaining World-Model Modes during Policy Adaptation

Keeping world-model modes during policy adaptation improves robustness to scene changes. On RoboTwin, policy adaptation trains the WAM mode and keeps small shares of forward prediction and imageto-video generation (0.9, 0.05 and 0.05). Against WAM alone, trained for 2 500 steps from the same checkpoint and evaluated identically, the additional modes raise randomized success by 8.0%, with gains on 31 of the 50 tasks and losses on 9, and leave clean success without a significant change (Figure 21). Both runs predict video inside the WAM mode, so the gain comes from the additional modes rather than from predicting video at all. Modes that receive a tenth of the training examples thus preserve robustness to scene changes that the WAM objective alone loses.

Together, the three comparisons show what makes the shared foundation work as a policy: robot specialization, an action expert copied from the generator and retained world-model modes each raise success at a fixed adaptation budget, the last mainly in randomized scenes, and specialization improves the world model at the same time.

![Chart block](doc:f38ca8c/tier:basic/page:23/block:10)

Higher with WAM + visual prediction

Higher with WAM only

Equal

## 7.4 What the Policy Imagines

A world–action model predicts the video of each action chunk together with the actions, so, unlike a policy that predicts actions alone, it shows what it expects to happen: decoding its predictions at every replanning

Figure 21 Per-task effect of visual prediction. Randomized RoboTwin success per task, 20 trials, with and without forward dynamics and imageto-video.

step gives a view of the policy’s imagination (Figure 22). We evaluate the final RoboTwin policy on all 50 tasks in two settings: clean scenes, which match the 50 clean demonstrations per task used for adaptation (in distribution), and randomized scenes with new backgrounds, clutter, lighting and table heights, which appear neither in these demonstrations nor in the GreenWAM corpus (out of distribution). At every replanning

23

<!-- page 24 of 43 -->

Clean scenes · in distribution

![Image block](doc:f38ca8c/tier:basic/page:24/block:2)

failure · “blocks ranking size”

LPIPS 0.030

![Image block](doc:f38ca8c/tier:basic/page:24/block:5)

Randomized scenes · out of distribution

task: blocks from large to small, left to right imagined correctly, yet they end medium, large, small

![Image block](doc:f38ca8c/tier:basic/page:24/block:8)

Figure 22 What the RoboTwin policy imagines while it acts. A success and a failure in clean scenes (in distribution) and in randomized scenes (out of distribution). For each episode, the head-camera frame rendered after an action chunk is executed (top strip) and the frame the policy predicted for the same moment in the WAM mode (bottom strip), at four moments from the start to the end of the episode; LPIPS is the mean over the episode. In the clean failure the policy imagines the outcome of its actions correctly, including the wrong order in which the blocks end; in the randomized failure the imagined scene departs from the executed one.

![Image block](doc:f38ca8c/tier:basic/page:24/block:9)

task: put the toy car into the basket and lift it the imagined arm smears, then the bottle vanishes start end of episode

## Per episode

![Chart block](doc:f38ca8c/tier:basic/page:24/block:13)

Success against prediction error

![Chart block](doc:f38ca8c/tier:basic/page:24/block:15)

Figure 23 Prediction error and success on RoboTwin 2.0. LPIPS between the predicted and the executed head-camera frames, averaged over each episode. Left: successful and failed episodes in clean and randomized scenes (box: quartiles; line: median). Right: success rate in this run of the episodes grouped by their mean error (below 0.03, 0.03–0.05, 0.05–0.07 and above 0.07), drawn at the median error of each group.

step we compare the predicted head-camera frames with the frames observed after the chunk is executed, by LPIPS and by PSNR against a baseline that repeats the last observed frame. In distribution the imagination is accurate: in successful clean episodes the predictions exceed this baseline by a median of 9.6 dB of PSNR, and in the clean success of Figure 22 the imagined frames follow the executed ones to the end of the episode.

Success follows imagination in both settings, but depends on it far more out of distribution (Figure 23). In clean scenes, success stays between 86% and 96% for all but the 6% of episodes with the largest error. In randomized scenes, it falls from 85% for the most accurate imagination to 27% for the least accurate, and

24

<!-- page 25 of 43 -->

![Image block](doc:f38ca8c/tier:basic/page:25/block:1)

Figure 24 Randomization axes of RoboTwin 2.0. One scene of stack\_blocks\_two right after reset, without a policy, under each axis: background textures, table height (lowered by up to 3 cm in the official setting, raised or lowered by 3 cm in our test), distractor objects on the table, lighting color and intensity, and all axes together as in the official randomized setting. All tiles use the head camera except table height, which is barely visible there and is shown from the left wrist camera; the dashed line marks the table edge of the clean scene. The scene is generated anew for each setting, so object positions differ between tiles.

a third of these episodes are in that worst group. The link is not only a consequence of failing: judged by the first 100 steps alone, randomized episodes with an accurate imagination succeed twice as often as those with a poor one.

A review of 20 randomly drawn failures in each setting shows how they arise. In clean scenes, 19 of them imagine the executed scene correctly, outcome included: a block that is never released, blocks stacked in the wrong order, an object dropped beside its target (Figure 22). The policy foresees what its actions do and still chooses the wrong ones. In randomized scenes, 13 of the 20 failures come from how the policy sees the changed scene: its imagination departs from what happens, or it grasps a distractor instead of the target and imagines doing so. DreamZero attributes most of its failures to video generation rather than action prediction [92]; our measurements show that this holds out of distribution, while in distribution the policy imagines correctly and fails in its actions.

This accounts for the result of Section 7.3. Retaining forward dynamics and image-to-video during adap tation raised randomized success by 8.0% and left clean success unchanged: the training signal of video prediction pays off where failures come from imagination, not where the policy already imagines correctly. Robustness to new scenes thus goes with imagining them well, and GreenWAM keeps the video prediction of its world-model role while it learns to act.

## 7.5 Robustness to Scene Randomization

Randomization changes appearance and geometry at once, so a single success rate does not show which change matters most. We therefore vary one axis at a time (Figure 24) and then test appearance augmentation; Table 3 lists the clean and randomized success of other methods with their training data.

**Individual scene changes.** The single-axis tests use seeds outside the official set. Background textures and table height cause the largest drops, 12% and 8% below the clean reference; distractors and lighting change success by less than 1%, and unseen instructions do not lower it. A new background changes only the image, which augmentation can target. A table 3 cm higher or lower changes the trajectory the arm must follow, and the augmentations used here change observations while keeping the recorded states and actions, so they add no demonstrations of such motions.

**Trajectory accuracy and replanning.** Execution is sensitive to small errors: replaying expert trajectories with joint noise of 0.05 rad succeeds in 80% of trials, and with 0.1 rad in 30%. Executing 8 instead of 16 predicted steps before replanning lowers clean success by 14.8%.

**Appearance augmentation.** Background replacement targets the costliest appearance change: in a paired comparison on tasks chosen before training, it raises randomized success by 13.1% and leaves clean success unchanged.

## 8 Discussion and Limitations

**One foundation for both roles.** A single robotics checkpoint yields both a world model and policies. The controlled comparisons connect the two roles: robot specialization improves forward prediction and policy

25

<!-- page 26 of 43 -->

success at the same budget, and the policy that predicts held-out frames more accurately also succeeds more often.

**World modeling as part of control.** Keeping forward dynamics and image-to-video next to WAM during adaptation raises randomized RoboTwin success by 8.0%, although both variants predict video. Decoding the policy’s own predictions shows the same link inside one model: out of distribution, success follows the accuracy of the imagined scene, which is where retaining world-model modes pays off (Section 7.4).

**One action space for all bodies.** One projector serves all 20 families, so a new robot needs a codec rather than new action parameters (Section A.3). The single-arm Franka of CALVIN and LIBERO, absent from the robotics corpus, enters through the same interface and reaches 4.64 on CALVIN and 97.4% on LIBERO. **Limitations.** The policies remain sensitive to changes in task geometry, such as table height on RoboTwin or the robot’s initial configuration on LIBERO-plus: the augmentations used here change observations but add no new motions. Long rollouts continue from the model’s own frames, so prediction error accumulates from block to block. Each benchmark is reached through its own post-training, and the real-robot evaluation covers one humanoid and one packing task.

**Outlook.** Natural next steps are running the distilled single-step policy on the robot’s onboard computer, transfer to bodies that neither the parent nor the robotics stage has seen, motion-level augmentation for robustness to task geometry, and the use of the world-model role to evaluate policies in imagined rollouts.

## 9 Conclusion

We asked whether one robot world model can serve both as a world model and as a policy across different bodies, and what makes this work. Adapted from Cosmos 3 Edge to 20 robot families through a shared anatomical action space and joint video–action training, one foundation yields a world model with top-tier trajectory, depth and 3D accuracy on WorldArena 1.0, ahead of all compared world–action models and video generators, and policies that score among the top entries of the RoboTwin 2.0 leaderboard, are competitive on CALVIN and LIBERO and, on a real humanoid, succeed more often than policies adapted from $\pi _ { 0 . 5 }$ and from Cosmos 3 Edge. Controlled comparisons identify three choices that make this work: robot specialization, which improves both roles together, an action expert split off the generator, and world-model modes kept during policy adaptation. A distilled one-step student answers at 28.3 Hz and keeps its lead over $\pi _ { 0 . 5 }$ on the robot. We release the weights of the RoboTwin 2.0, LIBERO and CALVIN policies and of the WorldArena world model, together with the code that evaluates them. Because a new robot needs a codec rather than new action parameters, the next steps are bodies unseen in training and more reliable long-horizon prediction and control under changes in task geometry.

## References

[1] 1X Technologies. 1X World Model: Evaluating bits, not atoms. Technical progress report, 2025. URL [https://www.1x.tech/1x-world-model.pdf](https://www.1x.tech/1x-world-model.pdf).

[2] AgiBot Research Team, R. Liu, W. Zhao, Z. Yang, L. Chen, P. Zhou, S. Chen, G. Ren, Y. Peng, et al. GE-Act 2.0: Pretraining and scaling a world-action model for robotic manipulation, 2026. URL [https://arxiv.org/abs/2609.05588](https://arxiv.org/abs/2609.05588).

[3] AgiBot World Team. AgiBot World 2026. Hugging Face dataset, 2026. URL [https://huggingface.co/datasets/agibot-world/AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026).

[4] A. Akbari, C. Zhang, A. Akbari, L. Zhao, Y. Chen, W. Chen, X. Zhang, G. Yuan, and Y. Wang. Flash-WAM: Modality-aware distillation for world action models, 2026. URL [https://arxiv.org/abs/2606.05254](https://arxiv.org/abs/2606.05254).

[5] I. Apanasevich, M. Artemyev, R. Babakyan, P. Fedotova, D. Grankin, E. Kupryashin, A. Misailidi, D. Nerus, et al. Green-VLA: Staged vision-language-action model for generalist robots, 2026. URL [https://arxiv.org/abs/2602.00919](https://arxiv.org/abs/2602.00919).

[6] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL [https://arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985).

[7] H. Bi, H. Tan, S. Xie, Z. Wang, S. Huang, H. Liu, R. Zhao, Y. Feng, C. Xiang, Y. Rong, H. Zhao, H. Liu, Z. Su, L. Ma, H. Su, and J. Zhu. Motus: A Unified Latent Action World Model, 2025. URL [https://arxiv.org/abs/2512.13030](https://arxiv.org/abs/2512.13030).

[8] H. Bi, Z. Zhou, Y. Tang, J. Pang, S. Huang, H. Liu, R. Wang, S. Huang, et al. Motus2: A self-evolving general world model for dexterous manipulation, 2026. URL [https://arxiv.org/abs/2608.30237](https://arxiv.org/abs/2608.30237).

[9] BitRobot, Unitree, and Hugging Face. HIW-500: Humanoids in-the-wild dataset for robot learning. Dataset, 2026. URL [https://bitrobot-foundation.github.io/humanoids-in-the-wild-500-hours](https://bitrobot-foundation.github.io/humanoids-in-the-wild-500-hours).

[10] BitRobot Foundation. G1 WBT Dex1 building-children-table, 2026. URL [https://huggingface.co/datasets/BitRobot/G1\_WBT\_Dex1\_Building-Children-Table](https://huggingface.co/datasets/BitRobot/G1_WBT_Dex1_Building-Children-Table). Dataset on Hugging Face.

[11] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, et al. 𝜋<sub>0</sub>: A vision-language-action flow model for general robot control, 2024. URL [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164).

26

<!-- page 27 of 43 -->

[12] BWM Team. BWM: A low-cost high-fidelity world simulator for robot learning, 2026. URL [https://arxiv.org/abs/2607.29302](https://arxiv.org/abs/2607.29302).

[13] R. Cadene, S. Alibert, A. Soare, Q. Gallouedec, A. Zouitine, S. Palma, P. Kooijmans, M. Aractingi, M. Shukor, D. Aubakirova, et al. LeRobot: State-of-the-art machine learning for real-world robotics in PyTorch. GitHub repository, 2024. URL [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot).

[14] J. Cai, L. Ling, S. Chu, Z. Liu, J. Kang, Z. Liang, W. Xu, Y. Mao, et al. AHA-WAM: Asynchronous horizon-adaptive world-action modeling with observation-guided context routing, 2026. URL [https://arxiv.org/abs/2606.09811](https://arxiv.org/abs/2606.09811).

[15] R. Cai, J. Guo, X. He, P. Jin, J. Li, B. Lin, F. Liu, W. Liu, et al. Xiaomi-Robotics-0: An open-sourced vision-language-action model with real-time execution, 2026. URL [https://arxiv.org/abs/2602.12684](https://arxiv.org/abs/2602.12684).

[16] C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, et al. GR-2: A generative video-language-action model with web-scale knowledge for robot manipulation, 2024. URL [https://arxiv.org/abs/2410.06158](https://arxiv.org/abs/2410.06158).

[17] K. Chen, Z. Liu, T. Zhang, Z. Guo, S. Xu, H. Lin, H. Zang, X. Li, B. Wei, J. Zhou, Q. Zhang, Z. Yu, G. Fan, T. Huang, Y. Wang, and C. Yu. <sup>𝜋</sup>RL: Online RL fine-tuning for flow-based vision-language-action models, 2025. URL [https://arxiv.org/abs/2510.25889](https://arxiv.org/abs/2510.25889).

[18] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation, 2025. URL [https://arxiv.org/abs/2506.18088](https://arxiv.org/abs/2506.18088).

[19] T. Chen, Y. Chen, Z. Li, J. Tang, K. Su, H. Lu, W. Wan, B. Chen, et al. RoboDojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies, 2026. URL [https://arxiv.org/abs/2607.04434](https://arxiv.org/abs/2607.04434).

[20] Y. Chen, R. Chen, D. Huo, Y. Yang, D. Qi, H. Liu, T. Lin, S. Zeng, et al. ABot-PhysWorld: Interactive world foundation model for robotic manipulation with physics alignment, 2026. URL [https://arxiv.org/abs/2603.23376](https://arxiv.org/abs/2603.23376).

[21] Y. Chen, P. Li, Y. Xu, Q. Ma, J. Yang, K. Wang, J. Yang, D. An, et al. FlowWAM: Optical flow as a unified action representation for world action models, 2026. URL [https://arxiv.org/abs/2607.13017](https://arxiv.org/abs/2607.13017).

[22] R. Doshi, H. Walke, O. Mees, S. Dasari, and S. Levine. Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation, 2024. URL [https://arxiv.org/abs/2408.11812](https://arxiv.org/abs/2408.11812).

[23] Y. Du, M. Yang, B. Dai, H. Dai, O. Nachum, J. B. Tenenbaum, D. Schuurmans, and P. Abbeel. Learning universal policies via text-guided video generation, 2023. URL [https://arxiv.org/abs/2302.00111](https://arxiv.org/abs/2302.00111).

[24] S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, J. Fu, J. Gong, and X. Qiu. LIBERO-Plus: In-depth robustness analysis of vision-language-action models, 2025. URL [https://arxiv.org/abs/2510.13626](https://arxiv.org/abs/2510.13626).

[25] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, et al. DreamDojo: A generalist robot world model from large-scale human videos, 2026. URL [https://arxiv.org/abs/2602.06949](https://arxiv.org/abs/2602.06949).

[26] GigaBrain Team, A. Ye, A. Sun, C. Jin, C. Cheng, C. Shi, D. Shang, D. Zhang, G. Huang, et al. GigaBrain-0.7: Scaling embodied foundation models to emergent capabilities with a three-system architecture, 2026. URL [https://arxiv.org/abs/2608.15875](https://arxiv.org/abs/2608.15875).

[27] GigaWorld Team, A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, J. Zhu, K. Li, et al. GigaWorld-0: World models as data engine to empower embodied AI, 2025. URL [https://arxiv.org/abs/2511.19861](https://arxiv.org/abs/2511.19861).

[28] GigaWorld Team, A. Ma, B. Wang, B. Li, C. Ni, G. Li, G. Huang, G. Zhao, et al. GigaWorld-1: A roadmap to build world models for robot policy evaluation, 2026. URL [https://arxiv.org/abs/2607.02642](https://arxiv.org/abs/2607.02642).

[29] GigaWorld Team, A. Ye, A. Ma, B. Wang, C. Ni, F. Ye, G. Huang, G. Li, G. Zhao, et al. GigaWorld-Policy-0.5: A faster and stronger WAM empowered by autoresearch, 2026. URL [https://arxiv.org/abs/2607.13960](https://arxiv.org/abs/2607.13960).

[30] J. Guo, Q. Li, P. Li, Z. Chen, N. Sun, Y. Su, H. Wang, Y. Zhang, et al. Unified 4D world action modeling from video priors with asynchronous denoising, 2026. URL [https://arxiv.org/abs/2604.26694](https://arxiv.org/abs/2604.26694).

[31] Y. Guo, L. X. Shi, J. Chen, and C. Finn. Ctrl-World: A controllable generative world model for robot manipulation, 2025. URL [https://arxiv.org/abs/2510.10125](https://arxiv.org/abs/2510.10125).

[32] J. Ho and T. Salimans. Classifier-free diffusion guidance, 2022. URL [https://arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598).

[33] Y. Hu, Y. Guo, P. Wang, X. Chen, Y.-J. Wang, J. Zhang, K. Sreenath, C. Lu, et al. Video prediction policy: A generalist robot policy with predictive visual representations, 2024. URL [https://arxiv.org/abs/2412.14803](https://arxiv.org/abs/2412.14803).

[34] J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, et al. DreamGen: Unlocking generalization in robot learning through video world models, 2025. URL [https://arxiv.org/abs/2505.12705](https://arxiv.org/abs/2505.12705).

[35] T. Jiang, T. Yuan, Y. Liu, C. Lu, J. Cui, X. Liu, S. Cheng, J. Gao, et al. Galaxea open-world dataset and G0 dual-system VLA model, 2025. URL [https://arxiv.org/abs/2509.00576](https://arxiv.org/abs/2509.00576).

[36] Y. Jiang, Y. Guo, H. Yang, G. Lei, N. Chen, Y. Zhang, S. Yan, B. Lin, F. Gao, and B. Qi. CKT-WAM: Parameter-efficient context knowledge transfer between world action models, 2026. URL [https://arxiv.org/abs/2605.06247](https://arxiv.org/abs/2605.06247).

[37] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang. MUSIQ: Multi-scale image quality transformer, 2021. URL [https://arxiv.org/abs/2108.05997](https://arxiv.org/abs/2108.05997).

[38] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success, 2025. URL [https://arxiv.org/abs/2502.19645](https://arxiv.org/abs/2502.19645).

[39] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, et al. Cosmos Policy: Fine-tuning video models for visuomotor control and planning, 2026. URL [https://arxiv.org/abs/2601.16163](https://arxiv.org/abs/2601.16163).

[40] C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Martín-Martín, C. Wang, G. Levine, et al. BEHAVIOR-1K: A humancentered, embodied ai benchmark with 1,000 everyday activities and realistic simulation, 2024. URL [https://arxiv.org/abs/2403.09227](https://arxiv.org/abs/2403.09227).

[41] F. Li, W. Song, H. Zhao, J. Wang, P. Ding, D. Wang, L. Zeng, and H. Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model, 2025. URL [https://arxiv.org/abs/2510.12276](https://arxiv.org/abs/2510.12276).

[42] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, et al. Causal world modeling for robot control, 2026. URL [https://arxiv.org/abs/2601.21998](https://arxiv.org/abs/2601.21998).

[43] S. Li, Y. Gao, D. Sadigh, and S. Song. Unified video action model, 2025. URL [https://arxiv.org/abs/2503.00200](https://arxiv.org/abs/2503.00200).

[44] S. Li, V. Yao, C. Yang, T. Qu, R. Cheng, R. Yu, H. Lu, N. Von, et al. WALL-WM: Carving world action modeling at the event joints, 2026. URL [https://arxiv.org/abs/2606.01955](https://arxiv.org/abs/2606.01955).

[45] X. Li, P. Li, L. Qian, M. Liu, D. Wang, J. Liu, B. Kang, X. Ma, et al. What matters in building vision-language-action models for generalist robots, 2024. URL [https://arxiv.org/abs/2412.14058](https://arxiv.org/abs/2412.14058).

27

<!-- page 28 of 43 -->

[46] X. Li, J. Guo, Q. Li, L. Qian, H. Lai, Y. Wang, H. Yan, J. Cao, et al. Xiaomi-Robotics-U0: Unified embodied synthesis with world foundation model, 2026. URL [https://arxiv.org/abs/2607.11643](https://arxiv.org/abs/2607.11643).

[47] Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, J. Cai, S. Liu, J. Luo, L. Chen, S. Yan, M. Yao, and G. Ren. Genie Envisioner: A unified world foundation platform for robotic manipulation, 2025. URL [https://arxiv.org/abs/2508.05635v3](https://arxiv.org/abs/2508.05635v3).

[48] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), 2023.

[49] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, et al. RDT-1B: a diffusion foundation model for bimanual manipulation, 2024. URL [https://arxiv.org/abs/2410.07864](https://arxiv.org/abs/2410.07864).

[50] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022. URL [https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003).

[51] Y. Liu, P. Ding, T. Jiang, X. Wang, M. Lin, W. Song, H. Zhang, Z. Zhuang, H. Zhao, W. Zhao, S. Huang, J. Shi, and D. Wang. MMaDA-VLA: Large diffusion vision-language-action model with unified multi-modal instruction and generation, 2026. URL [https://arxiv.org/abs/2603.25406](https://arxiv.org/abs/2603.25406).

[52] Y. Liu, P. Sun, S. Li, Y. Xie, L. Zhang, X. Chao, S. Dong, F. Chen, X.-P. Zhang, and W. Ding. OA-WAM: Object-addressable world action model for robust robot manipulation, 2026. URL [https://arxiv.org/abs/2605.06481](https://arxiv.org/abs/2605.06481).

[53] I. Loshchilov and F. Hutter. Decoupled weight decay regularization, 2017. URL [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101).

[54] G. Y. Luo, G. M. Favero, Z. H. Luo, A. Jolicoeur-Martineau, and C. Pal. Beyond FVD: Enhanced evaluation metrics for video generation quality, 2024. URL [https://arxiv.org/abs/2410.05203](https://arxiv.org/abs/2410.05203).

[55] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks, 2021. URL [https://arxiv.org/abs/2112.03227](https://arxiv.org/abs/2112.03227).

[56] Motubrain Team, C. Xiang, F. Bao, H. Liu, H. Tan, H. Bi, J. Li, J. Liu, J. Pang, et al. Motubrain: An advanced world action model for robot control, 2026. URL [https://arxiv.org/abs/2604.27792](https://arxiv.org/abs/2604.27792).

[57] NVIDIA, A. Ali, J. Bai, M. Bala, Y. Balaji, A. Blakeman, T. Cai, J. Cao, T. Cao, et al. World simulation with video foundation models for physical ai, 2025. URL [https://arxiv.org/abs/2511.00062](https://arxiv.org/abs/2511.00062).

[58] NVIDIA, Aditi, N. Agarwal, A. Ali, J. Allen, M. Antolini, A. Aubame, A. Azzolini, J. Bai, et al. Cosmos 3: Omnimodal world models for physical AI, 2026. URL [https://arxiv.org/abs/2606.02800](https://arxiv.org/abs/2606.02800).

[59] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, et al. Octo: An open-source generalist robot policy, 2024. URL [https://arxiv.org/abs/2405.12213](https://arxiv.org/abs/2405.12213).

[60] Open X-Embodiment Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, et al. Open X-Embodiment: Robotic learning datasets and RT-X models, 2023. URL [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864).

[61] Physical Intelligence. OpenPI: Open-source robotics models and fine-tuning. Official software repository, 2025. URL [https://github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).

[62] Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, et al. 𝜋<sub>0</sub>.5: a vision-language-action model with open-world generalization, 2025. URL [https://arxiv.org/abs/2504.16054](https://arxiv.org/abs/2504.16054).

[63] B. Qiu, L. Chen, Y. Liao, N. Wang, L. Wang, J. Luo, W. Zhao, S. Chen, et al. GE-Sim 2.0: A roadmap towards comprehensive closed-loop video world simulators for robotic manipulation, 2026. URL [https://arxiv.org/abs/2605.27491](https://arxiv.org/abs/2605.27491).

[64] Qwen Team. Qwen3.8-Max: A new bar for coding and cowork. Blog post, August 2026. URL [https://qwen.ai/blog?id=qwen3.8](https://qwen.ai/blog?id=qwen3.8).Model card: [https://huggingface.co/Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

[65] M. Reuss, H. Zhou, M. Rühle, Ömer Erdinç Yağmurlu, F. Otto, and R. Lioutikov. FLOWER: Democratizing generalist robot policies with efficient vision-language-action flow policies, 2025. URL [https://arxiv.org/abs/2509.04996](https://arxiv.org/abs/2509.04996).

[66] Robbyant. LingBot-VA repository, 2026. URL [https://github.com/Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va).

[67] RoboDojo Team. RoboDojo leaderboard. Project website, 2026. URL [https://robodojo-benchmark.com/leaderboard](https://robodojo-benchmark.com/leaderboard).

[68] RoboTwin Team. RoboTwin 2.0 leaderboard. Project website, 2026. URL [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

[69] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models, 2022. URL [https://arxiv.org/abs/2210.08402](https://arxiv.org/abs/2210.08402).

[70] Y. Shang, Z. Li, Y. Ma, W. Su, X. Jin, Z. Wang, L. Jin, X. Zhang, et al. WorldArena: A unified benchmark for evaluating perception and functional utility of embodied world models, 2026. URL [https://arxiv.org/abs/2602.08971](https://arxiv.org/abs/2602.08971).

[71] Y. Shang, Y. Tang, Y. Ma, Z. Li, L. Jin, W. Su, X. Jin, Z. Wang, et al. WorldArena 2.0: Extending embodied world model benchmarking on modality, functionality and platform, 2026. URL [https://arxiv.org/abs/2605.17912](https://arxiv.org/abs/2605.17912).

[72] Y. Song and P. Dhariwal. Improved techniques for training consistency models, 2023. URL [https://arxiv.org/abs/2310.14189](https://arxiv.org/abs/2310.14189).

[73] StarVLA Community. StarVLA: A Lego-like codebase for vision-language-action model developing, 2026. URL [https://arxiv.org/abs/2604.05014](https://arxiv.org/abs/2604.05014).

[74] H. Su, Z. Liu, X. Jin, H. Dou, C. Hu, B. Li, Z. Liu, R. Xu, J. Fang, X. Zhang, Z. Yang, X. Yang, C. Gao, J. Yan, Y. Li, and W. Wu. WorldScape Policy 2.0: Empowering steerable world action modeling with reasoning-augmented memory and in-context learning, 2026. URL [https://arxiv.org/abs/2607.18840](https://arxiv.org/abs/2607.18840).

[75] T. Su, J. Zhu, T. Wang, Y. He, Z. Huang, J. Zhang, C. Ma, H. Wang, et al. DeMaVLA: A vision-language-action foundation model for generalizable deformable manipulation, 2026. URL [https://arxiv.org/abs/2605.31286](https://arxiv.org/abs/2605.31286).

[76] Y. Tian, S. Yang, J. Zeng, P. Wang, D. Lin, H. Dong, and J. Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation, 2024. URL [https://arxiv.org/abs/2412.15109](https://arxiv.org/abs/2412.15109).

[77] Y. Tian, Y. Yang, Y. Xie, Z. Cai, X. Shi, N. Gao, H. Liu, X. Jiang, et al. InternData-A1: Pioneering high-fidelity synthetic data for pre-training generalist policy, 2025. URL [https://arxiv.org/abs/2511.16651](https://arxiv.org/abs/2511.16651).

[78] Unitree Robotics. Unitree open robot datasets, 2025. URL [https://huggingface.co/unitreerobotics](https://huggingface.co/unitreerobotics). Dataset on Hugging Face.

[79] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, et al. Wan: Open and advanced large-scale video generative models, 2025. URL [https://arxiv.org/abs/2503.20314](https://arxiv.org/abs/2503.20314).

[80] L. Wang, X. Chen, J. Zhao, and K. He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers, 2024. URL [https://arxiv.org/abs/2409.20537](https://arxiv.org/abs/2409.20537).

28

<!-- page 29 of 43 -->

[81] Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang. Unified vision-language-action model, 2025. URL [https://arxiv.org/abs/2506.19850](https://arxiv.org/abs/2506.19850).

[82] Z. Wang, C. Liu, Y. Xiang, R. Zhang, Q. Hao, H. Lu, H. Chen, Z. Feng, et al. The great march 100: 100 detail-oriented tasks for evaluating embodied ai agents, 2026. URL [https://arxiv.org/abs/2601.11421](https://arxiv.org/abs/2601.11421).

[83] WorldArena Team. WorldArena 2.0: Motion quality evaluation update. Project website, 2026. URL [https://v2.world-arena.ai/WorldArena-2.0/motion-quality-evaluation-update.html](https://v2.world-arena.ai/WorldArena-2.0/motion-quality-evaluation-update.html).

[84] WorldArena Team. WorldArena leaderboard. Hugging Face Space, 2026. URL [https://huggingface.co/spaces/WorldArena/WorldArena](https://huggingface.co/spaces/WorldArena/WorldArena).

[85] K. Wu, C. Hou, J. Liu, Z. Che, X. Ju, Z. Yang, M. Li, Y. Zhao, et al. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation, 2024. URL [https://arxiv.org/abs/2412.13877](https://arxiv.org/abs/2412.13877).

[86] S. Wu, X. Liu, S. Xie, P. Wang, X. Li, B. Yang, Z. Li, K. Zhu, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation, 2025. URL [https://arxiv.org/abs/2511.17441](https://arxiv.org/abs/2511.17441).

[87] G. Yang, Z. Tu, Y. Yang, S. Mao, J. Dong, T. Chen, J. Peng, J. Xiong, et al. EventVLA: Event-driven visual evidence memory for long-horizon vision-language-action policies, 2026. URL [https://arxiv.org/abs/2606.20092](https://arxiv.org/abs/2606.20092).

[88] L. Yang, W. Song, X. Wang, P. Sheng, Z. Fang, Z. Zhou, J. He, H. Yan, et al. 4D-WAM: Infusing spatiotemporal awareness into world action models through trajectory fields, 2026. URL [https://arxiv.org/abs/2608.08023](https://arxiv.org/abs/2608.08023).

[89] S. Yang, Y. Du, K. Ghasemipour, J. Tompson, L. Kaelbling, D. Schuurmans, and P. Abbeel. Learning interactive real-world simulators, 2023. URL [https://arxiv.org/abs/2310.06114](https://arxiv.org/abs/2310.06114).

[90] Y. Yang, S. Zeng, T. Lin, X. Chang, D. Qi, J. Xiao, H. Liu, R. Chen, et al. ABot-M0: VLA foundation model for robotic manipulation with action manifold learning, 2026. URL [https://arxiv.org/abs/2602.11236](https://arxiv.org/abs/2602.11236).

[91] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, et al. CogVideoX: Text-to-video diffusion models with an expert transformer, 2024. URL [https://arxiv.org/abs/2408.06072](https://arxiv.org/abs/2408.06072).

[92] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, et al. World action models are zero-shot policies, 2026. URL [https://arxiv.org/abs/2602.15922](https://arxiv.org/abs/2602.15922).

[93] C. Yin, D. Huang, D. Yang, J. Wang, N. Zhao, C. Xu, W. Sun, L. Hou, et al. Genie sim 3.0 : A high-fidelity comprehensive simulation platform for humanoid robot, 2026. URL [https://arxiv.org/abs/2601.02078](https://arxiv.org/abs/2601.02078).

[94] H. Yuan, Z. Liang, A. Chen, Y. Wang, H. Li, P. Lin, Y. Huang, Z. Lei, et al. Qwen-robotmanip technical report: Alignment unlocks scale for robotic manipulation foundation models, 2026. URL [https://arxiv.org/abs/2606.17846](https://arxiv.org/abs/2606.17846).

[95] T. Yuan, Z. Dong, Y. Liu, and H. Zhao. Fast-WAM: Do world action models need test-time future imagination?, 2026. URL [https://arxiv.org/abs/2603.16666](https://arxiv.org/abs/2603.16666).

[96] M. Zhang, R. Sun, X. Wang, C. Zhang, S. Li, M. Cao, H. Lu, E. Chen, et al. WALL-SS: Scaling long-horizon world models via next-scale autoregression, 2026. URL [https://arxiv.org/abs/2608.26239](https://arxiv.org/abs/2608.26239).

[97] Q. Zhang, L. Li, L. Zhang, S. Yang, Y. Luo, S. Li, R. Wang, J. Wang, et al. Native video-action pretraining for generalizable robot control, 2026. URL [https://arxiv.org/abs/2607.08639](https://arxiv.org/abs/2607.08639).

[98] W. Zhang, H. Liu, Z. Qi, Y. Wang, X. Yu, J. Zhang, R. Dong, J. He, F. Lu, H. Wang, Z. Zhang, L. Yi, W. Zeng, and X. Jin. DreamVLA: A vision-language-action model dreamed with comprehensive world knowledge, 2025. URL [https://arxiv.org/abs/2507.04447](https://arxiv.org/abs/2507.04447).

[99] X. Zhang, X. Liu, Q. Tang, J. Wang, R. Wang, Z. Cheng, B. Lin, H. Wen, M. Li, K. Zhan, and Y. Liu. ME-Dex 1.0: Bringing Heterogeneous Tactile Sensing into World Action Modeling, 2026. URL [https://arxiv.org/abs/2609.21449](https://arxiv.org/abs/2609.21449).

[100] W. Zhao, H. Jiang, X. Shi, L. Liu, F. Huang, Z. Su, W. Sui, and X. Wang. Faster-WAM: Efficient inference-time future conditioning for robust world action models, 2026. URL [https://arxiv.org/abs/2608.04404](https://arxiv.org/abs/2608.04404).

[101] Z. Zhao, H. Jing, X. Liu, J. Mao, A. Jha, H. Yang, R. Xue, S. Zakharov, et al. Humanoid everyday: A comprehensive robotic dataset for open-world humanoid manipulation, 2025. URL [https://arxiv.org/abs/2510.08807](https://arxiv.org/abs/2510.08807).

[102] J. Zheng, J. Li, D. Liu, Y. Zheng, Z. Wang, Z. Ou, Y. Liu, J. Liu, et al. Universal actions for enhanced embodied foundation models, 2025. URL [https://arxiv.org/abs/2501.10105](https://arxiv.org/abs/2501.10105).

[103] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model, 2025. URL [https://arxiv.org/abs/2510.10274](https://arxiv.org/abs/2510.10274).

[104] C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets, 2025. URL [https://arxiv.org/abs/2504.02792](https://arxiv.org/abs/2504.02792).

[105] F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong. IRASim: A fine-grained world model for robot manipulation, 2024. URL [https://arxiv.org/abs/2406.14540](https://arxiv.org/abs/2406.14540).

[106] J. Zhu, J. Zhang, T. Su, T. Liu, Z. Wang, K. Xie, Z. Huang, C. Ma, Y. He, T. Wang, H. Wang, W. Ding, and Y. Xu. DSWAM: A dual-system world action foundation model for fine-grained robot manipulation, 2026. URL [https://arxiv.org/abs/2607.04927](https://arxiv.org/abs/2607.04927).

## A Data Inventory and Anatomical Mappings

## A.1 Family and Source Inventory

The corpus groups datasets by the robot body they describe (Figure 25). All twelve Unitree G1 layouts, which differ in hands and cameras, share one family; the two Worker datasets share another, as do the Green humanoid and its simulator. The largest family contributions to the admitted corpus are AgiBot G2A (256.0 h), Piper / Cobot Magic (227.2 h), the Green humanoid (214.8 h), Unitree G1 (204.9 h) and AgiBot G1 / Genie-1 (148.5 h). Figure 26 shows the 20 main families and their real/simulated composition. The inventory covers 1 217 datasets.

29

<!-- page 30 of 43 -->

Unitree H1 dexterous hands

AgiBot G2A grippers

Unitree G1 dexterous hands

Green humanoid dexterous hands

Galaxea R1Pro grippers

Leju dexterous hands

![Image block](doc:f38ca8c/tier:basic/page:30/block:7)

AIRBOT MMK2 dexterous hands

Piper / Cobot Magic grippers

AgiBot G1 / Genie-1 grippers

Galaxea R1Lite grippers

Worker / ARX5 grippers

RealMan AIDAL grippers

ManipArena EX001 grippers

Figure 25 Robot families of the corpus. Training canvases of the robot families, as the model receives them: the head view on top and the other cameras in a strip below. Each tile shows one dataset of its family, so other datasets of the same family may differ in hands or cameras. Humanoids come first, then two-arm platforms; the chip on each tile gives the body type, and the label below gives the effector.

Source-level filtering retains 1.43 million atoms totaling 1 674.7 h. InternData-A1 [77], the Green humanoid, RoboCOIN [86], GM100 [82] and AgiBotWorld-2026 [3] supply the largest retained volumes; further public sources include GR00T-Teleop-GR1 [25], BEHAVIOR-1K [40], Genie Sim 3.0 [93], RoboMIND [85], Humanoid Everyday [101], HIW-500 [9], the G1 IKEA assembly recordings [10] and Unitree datasets [78]. Galaxea Open-World [35] has the largest quality-filter reduction (6.84 h, 9% of its admitted hours).

## A.2 Corpus Admission and Quality Filtering

**Sources.** Each source is converted to one LeRobot layout and registered with its camera roles, state layout and three static facts that the captioner later receives: embodiment, end-effector type and real or simulated domain. Camera roles and the meaning of state and action columns are established from metadata, sampled tables and video rather than from column names. A video stream needs more than 10 FPS and at least 448 pixels on the short side; depth and infrared streams are not used. Failed and recovery demonstrations are kept, since for a world model they are valid physical rollouts.

**Semantic atoms.** An atom is the smallest annotated temporal unit of a recording, recovered from subtask tables, task-index runs, skill annotations or instruction intervals. An entire episode becomes one atom only when no finer annotation exists; missing or overlapping boundaries are marked unresolved rather than replaced by the whole episode. The source label is kept separately from the training instruction.

**Length.** Admission uses the camera in the head or ego role. The maximum admitted length is

$$
e - s \leq \operatorname* { m i n } ( B ( h ) , \lfloor 1 0   f \rfloor )   , \qquad B ( h ) = \begin{cases} { 3 0 0 , } & { h \leq 4 8 0 , } \\ { 2 0 0 , } & { 4 8 0 < h \leq 1 0 0 0 , } \end{cases}\tag{10}
$$

30

<!-- page 31 of 43 -->

![Chart block](doc:f38ca8c/tier:basic/page:31/block:1)

Figure 26 Hours per robot family, real-world and simulated. Combined names such as Piper / Cobot Magic denote datasets of one robot body grouped into one family (Section A). End labels are rounded. Hour shares differ from the sampling probabilities used in training (Section B.4).

where ℎ is the admission-view height and 𝑓 its frame rate. An overlong atom is rejected as a whole; the gate does not shorten it or invent new boundaries.

**Quality filtering.** The quality pass, computed on the admission camera, removes idle, repeated-frame, abruptly cut and unreadable clips (Table 4). An atom is idle when more than 70% of its motion values fall below 0.2 of the episode median; a whole episode is compared with the median motion of episodes from the same dataset instead. Repeated frames require both a repeated share above 0.30 and a run of at least two repeated transitions, which separates sustained freezes from isolated duplicates left by frame-rate conversion. A cut requires a histogram-distance peak together with large color and grayscale changes.

**Splits and text.** Training and holdout are separated at the episode level, giving 1.40 million training and 28.6 thousand held-out atoms before text exclusions. Atoms without an accepted caption or derived command stay in the quality-admitted pool but do not enter the text-dependent modes; native instructions are not used as a fallback.

Table 4 Data-quality pass over the admitted corpus. Atoms in thousands, hours rounded to one decimal. An atom can be removed for several reasons (40 atoms are), so the reason rows do not add up to the total.

|  | Atoms, thousands | Hours |
| --- | --- | --- |
| Admitted subtask atoms | 1437 | 1688.5 |
| Not scored, excluded | 0.1 | 0.2 |
| Removed |  |  |
| idle clip (idle score above 0.7) | 7.3 | 8.5 |
| repeated frames | 2.1 | 2.9 |
| abrupt cut within the clip | 1.8 | 2.3 |
| unreadable video | &lt; 0.1 | &lt; 0.1 |
| total | 11.1 | 13.6 |
| Kept for training | 1426 | 1674.7 |

31

<!-- page 32 of 43 -->

## A.3 Anatomical Mappings

![Chart block](doc:f38ca8c/tier:basic/page:32/block:2)

Figure 27 Coverage of the 79 anatomical slots by family. Rows: the 20 families ordered by admitted hours. Columns: slots grouped into legs (12), body (23: base pose, twist, torso and head), arms (18) and hands (26), with left and right subgroups. A colored cell is filled by a named channel of at least one layout of the family. A gray cell holds only unresolved channels of that body part: they stay in the data, are excluded from the loss and claim no correspondence to the slot. A light cell marks an absent slot. The hands of Ruantong/A2D are unresolved.

Figure 27 shows which slots each family fills. Each of the 43 registered source layouts has a codec that selects its channels, permutes them into slot order and fixes sign and zero for every slot role; scale and offset are absorbed by per-layout statistics. A coordinate enters a shared slot only when its correspondence follows from the URDF at zero pose, the manufacturer documentation or named channels, and for the main families the joint assignment is validated by forward kinematics against measured end-effector poses. Channels without an established correspondence stay outside the shared slots. A validity mask removes absent and unresolved coordinates from the loss, so equal tensor width never implies equal meaning. The Worker platform, for example, stores two 6-DoF arms and two grippers under generic channel names, and its codec assigns each channel an anatomical role (Figure 4). Robots that appear only in post-training enter through the same interface as layout extensions: a new codec maps their channels onto existing slots, and the shared projectors of GreenWAM stay unchanged. RoboTwin uses ARX5 arms, so its layout reuses the Worker family, whose forward kinematics reproduces the recorded RoboTwin gripper paths to within 5.2 and 7.1 mm; the single Franka arm of CALVIN enters through the Cartesian end-effector slots. State rows are aligned to video frames per source. Normalization statistics are computed on the training split only, separately for absolute and offset encodings and per layout, and pooled per slot with a lower bound on the scale in physical units; WAM statistics use successful demonstrations only.

## B Training and Adaptation Recipes

## B.1 Robotics Training Recipe

**Recipe.** Figure 28 shows the training losses. The noise level shared by all generated streams is drawn as in the video training of Cosmos 3 Edge: $\begin{array} { r } { \tilde { \sigma } = 1 - u - s \left( \cos ^ { 2 } \frac { \pi u } { 2 } - 1 + u \right) } \end{array}$ with $u \sim \mathcal { U } ( 0 , 1 )$ and $s = 1 . 2 9$ shifted to $\sigma = k \tilde { \sigma } / \big ( 1 + ( k - 1 ) \tilde { \sigma } \big )$ with $k = 5$ for 480p canvases. Each micro-batch packs 4 examples of one mode, so that examples with history anchors are not mixed with examples without them. The learning rate

32

<!-- page 33 of 43 -->

![Chart block](doc:f38ca8c/tier:basic/page:33/block:1)

![Chart block](doc:f38ca8c/tier:basic/page:33/block:2)

![Chart block](doc:f38ca8c/tier:basic/page:33/block:3)

![Chart block](doc:f38ca8c/tier:basic/page:33/block:4)

![Chart block](doc:f38ca8c/tier:basic/page:33/block:5)

![Chart block](doc:f38ca8c/tier:basic/page:33/block:6)

![Chart block](doc:f38ca8c/tier:basic/page:33/block:7)

Figure 28 Robotics-stage training. Training loss of each mode as reported (light) and its running median over about 500 steps (dark); inverse dynamics and WAM are on a log scale because they start an order of magnitude higher, and their medians fall from 8.3 to 0.29 and from 6.2 to 1.98.

follows a square-root rule from the base rate of $2 \times 1 0 ^ { - 5 }$ at batch 128, i.e. $\eta = 2 \times 1 0 ^ { - 5 } \sqrt { B / 1 2 8 }$ , and the newly initialized action projections and modality embeddings use a rate five times larger.

Table 5 Recipe of the robotics stage.

| Setting | Value |
| --- | --- |
| Initialization | Cosmos 3 Edge 4B |
| Trainable | Generator, action projections, action-modality embeddings |
| Frozen | Reasoner, video VAE |
| Modes, % of examples | text-to-video 20, image-to-video 10, image-to-video with history 10, forward dynam-ics 15, forward dynamics with history 20, inverse dynamics 10, WAM 15 |
| Frame rate | native, capped at 30 FPS |
| Training unit | whole subtask; each autoregressive block is one subtask |
| Rollout | up to 30s of future video in whole-subtask blocks; 4 RGB anchors of history |
| WAM horizon | about 1.07s; successful demonstrations only |
| Joint targets | inverse dynamics: absolute, including 𝑞<sub>0</sub>; other modes: offsets from the block start |
| Canvas sizes | 640×640, 736×544, 544×736, 832×480, 480×832 |
| Global batch | 256 examples (64 GPUs × 4 packed examples) |
| Optimizer | AdamW [53], 𝛽 = (0.9,0.99), weight decay 0.05 |
| Learning rate | 2.83 × 10<sup>-5</sup>; ×5 on action projections and embeddings |
| Schedule | 500 warm-up steps, then linear decay |
| Gradient clipping | 1.0 |
| Noise level | shared by all generated streams; Cosmos 3 Edge video schedule, shift 5 at 480p (Section 4.3) |
| Loss | video and state terms, each scaled by 10; state: mean within valid body-part groups, then across groups |
| Prompt | Cosmos 3 JSON, up to 768 tokens, text dropout 0.1 |
| Validation | every 1200 steps |
| Steps trained | 36500 (9.34 million examples) |

33

<!-- page 34 of 43 -->

## B.2 Temporal Sampling and Prediction Targets

**Physical time.** We retain each source’s physical timing and cap the effective video rate at 30 FPS. A source recorded at $f _ { s }$ is read with stride max(1, $\left[ f _ { s } / 3 0 \right] )$ : 15, 20 and 30 FPS are retained, while 50 FPS becomes 25 FPS. Video frames and measured states are selected on the same time grid, and the effective rate is passed to the model. This preserves motion speed in physical time; joint positions and displacements are not multiplied by the sampling-rate ratio. Three-view sources use the 480p canvas buckets 640×640, 736×544, 544×736, 832×480 and 480×832 according to the head-camera aspect ratio.

**Whole subtasks.** A fixed-length clip can cut an action in the middle and pair an incomplete grasp with the description of the whole grasp. Text-to-video, image-to-video, forward dynamics and inverse dynamics therefore use a complete prepared subtask rather than a fixed-length clip. In the autoregressive modes one block is one whole subtask; a following block is added only if it fits completely, and the last block is never truncated. The autoregressive budget is expressed in physical time, up to 30 s of future video, so the number of predicted frames follows the frame rate of the source.

**History anchors.** Let 𝑎 be the start of the contiguous kept history and 𝑏 the last observed boundary, both in native-frame indices, and let 𝑠 be the stride on the effective video grid. With $N = \lfloor ( b - a ) / s \rfloor$ , the four RGB anchors are

$$
h _ { i } = a + s \left\lfloor \frac { i N } { 3 } \right\rfloor , \qquad i = 0 , 1 , 2 , 3 .\tag{11}
$$

Each frame is encoded separately. At the start of an episode or an admitted rollout, $a   =   b$ and all four anchors are the initial frame; short histories can likewise repeat an anchor. Target frames begin at $b + s ,$ so every anchor precedes the predicted block. History extends across adjacent kept subtasks of the same episode and camera and resets at rejected gaps. A target block must fit within its admitted subtask; decoder padding does not supply missing history or target frames.

**WAM horizon.** WAM predicts a short future inside a subtask with a physical horizon of about 1.07 s. Video and state targets cover this interval; VAE padding creates no additional action targets. The command is bound to the enclosing admitted atom and is retained when sampling the shorter WAM window. It specifies the ongoing task rather than promising its completion within 𝐻. WAM admits only positive success labels; failed, unknown and absent outcomes remain outside its training subset.

![Image block](doc:f38ca8c/tier:basic/page:34/block:8)

7H[WFDSWLRQRULQVWUXFWLRQLVDFOHDQFRQGLWLRQLQHYHU\PRGH $\Delta \mathbf{q} = \mathbf{q} - \mathbf{q}_{0}$

Figure 29 Clean conditions and noised targets in the seven modes, with each mode’s share of training examples. Text (caption or instruction) is a clean condition in every mode. Forward dynamics, with and without history, conditions on $q _ { 0 }$ and the clean future targets $\delta _ { k }$ of Equation (2); inverse dynamics predicts absolute states including $q _ { 0 }$ . WAM predicts future frames and state offsets over about 1.07 s at the effective frame rate of the source. Autoregressive modes predict the next whole-subtask block from four RGB anchors of the history.

34

<!-- page 35 of 43 -->

## B.3 Structured Captions and Commands

A visual description records what is visible, including the scene and the observed outcome; an action instruction specifies what the robot should do during the same fragment. We keep both for every atom instead of treating a retrospective caption as a policy command.

**Captioner.** A vision-language model (Qwen3.8-27B [64] with FP8 weights) captions the admission-camera clip of one atom. It also receives the clip duration, the source subtask label and three registry facts: embodiment (humanoid or bimanual platform), end effector (dexterous hand or two-finger gripper) and domain (real or simulation). The label supplies the goal and the target object, while only the frames establish how far the attempt got. The registry facts are taken as given, so a cropped or occluded effector does not change the described embodiment.

**Field-by-field generation.** The captioner writes one field per turn in a single dialogue, and each answer is appended to the context before the next request (Table 6). Later fields must therefore agree with earlier ones, and one object keeps one name and one color throughout the record. Action phases start at 0:00, stay ordered and contiguous and end at the clip duration. A shared instruction block fixes the vocabulary: actor-relative left and right, the registry’s effector term, no camera language, no hedging and no contact detail that the view cannot resolve. An attempt record, written before the action phases, states the end state, whether the target moved and where the effector ends; a clip that ends mid-motion then receives phases and a narrative that also end mid-motion.

Table 6 Fields of the caption record in generation order. The two checks are generated in the same dialogue but stored outside the caption and never enter a training prompt.

|  | Field | Content | Role |
| --- | --- | --- | --- |
|  | 1 Subjects | Robot, and a person if present: embodiment, effector, lo- Caption cation, orientation, pose, action, state changes, arm and leg counts; person attributes stay empty for the robot |  |
|  | 2 Background | Setting, furniture and objects with colors and materials; Caption real or simulated domain |  |
|  | 3 Lighting | Conditions, direction, shadows, effect on the scene | Caption |
|  | 4 Aesthetics | Composition, color scheme, mood, patterns | Caption |
|  | 5 Context | One sentence: activity, setting, active effector and its Caption; rewriter side |  |
|  | 6 Attempt record | End state in the final frame, whether the target moved, Check effector position at the end, outcome |  |
|  | 7 Action phases | One to five timed phases: effector, object, where it goes Caption; rewriter |  |
|  | 8 Temporal caption | One chronological paragraph of 120–180 words that ends Caption; rewriter on the final state |  |
|  | 9 Consistency | Agreement with the label and the registry facts; mis- Check match clause |  |

**Content gates.** Each answer is checked for forbidden phrases, effector terms that contradict the registry, contact claims and completion words that the attempt record rules out. A violation returns the phrase to the model for a rewrite of the same field, at most twice; a remaining violation restarts the atom, up to 4 attempts. Atoms without an accepted caption stay in the quality-admitted pool but not in the text-dependent modes. **Coverage.** In total, 1.24 million atoms have an accepted visual caption and 187 thousand do not; 1.19 million of them also have an accepted command.

**From caption to command.** A second language-model pass turns the ordered action phases into one imperative command per phase, keeping the phase order, the effector, significant motions, releases and spatial relations. It does not see native operator commands, and partial or failed motions remain attempts rather than completed actions. A command is accepted only if every phase is present, in order and within the token budget; it is never truncated. Table 7 follows one atom from the source label to the derived command.

35

<!-- page 36 of 43 -->

Table 7 One atom from the source label to the derived command: the clip of Figure 6c (RoboCOIN, Unitree G1 with dexterous hands, real). Captioner and rewriter outputs are quoted verbatim.

| Source label | Grasp the bread from the table and place it into the plate. |
| --- | --- |
| Registry facts | humanoid; dexterous hand; real |
| Action phases | 0:00-0:04 The left dexterous hand reaches toward the orange and white bread roll on the left side of the table and grasps it. 0:04-0:07 The left dexterous hand lifts the bread roll, carries it over the purple plate, and lowers it onto the plate. 0:07-0:10 The left dexterous hand releases the bread roll on the plate and retracts away from the workspace. |
| Attempt record | target moved: yes; effector at end: away from the workspace; outcome: reached |
| Consistency | agrees |
| Derived command | Extend the left dexterous hand toward the orange and white bread roll on the left side of the table and grasp it. Lift the bread roll, carry it over the purple plate, and lower it onto the plate. Release the bread roll on the plate and retract the left hand away from the workspace. |

## B.4 Training Mixture

Sampling in proportion to hours would let four families supply more than half of all examples (Table 8); sampling uniformly over datasets would do the opposite and repeat the smallest datasets many times. We use a middle path. Sampling first chooses a mode, then a robot cluster, then a dataset and a window. Mode shares are those of Table 2. Cluster shares are fixed for our two main targets: the Worker/ARX5 cluster receives 12% of examples and the Green humanoid cluster (real and simulation) 18%. In this configured mixture, the remaining 19 clusters share 70% in proportion to $\sqrt { N }$ , where 𝑁 is the number of unique admissible training subtasks of the cluster:

$$
p _ { c } = 0 . 7 0 \cdot \frac { \sqrt { N _ { c } } } { \sum _ { c ^ { \prime } } \sqrt { N _ { c ^ { \prime } } } } .\tag{12}
$$

The sampler first enumerates eligible unique subtasks for each mode, cluster and dataset after text admission, modality checks and temporal-window admission. Let $p _ { m }$ and $p _ { c }$ be the configured mode and cluster shares, and let $\mathcal { E }$ contain the mode–cluster pairs with eligible data. We obtain their joint probabilities by maximum entropy:

$$
\begin{align*}P^{*} &= \arg \max_{P \geq 0} - \sum_{m,c} P_{mc} \log P_{mc}, \\\sum_{c} P_{mc} = p_{m}, \qquad & \sum_{m} P_{mc} = p_{c}, \qquad P_{mc} = 0 \quad  if   (m,c) \notin \mathcal{E}.\end{align*}\tag{13}
$$

The implementation checks support feasibility before solving for the joint table, rather than renormalizing one marginal at t<u>he ex</u>pense of the other. Within a supported pair, a dataset 𝑑 has conditional probability proportional to $\sqrt { N _ { m c d } } ,$ where $N _ { m c d }$ counts its eligible unique subtasks in that mode. Sampling then chooses a subtask uniformly within the dataset before selecting its temporal window and, where available, its joint or EEF view. Thus additional windows or a second action view do not increase a subtask’s sampling mass. Table 8 lists the configured cluster targets next to the admitted-hour shares; all sampling shares refer to examples.

The table makes the trade-off explicit. Our two deployment targets are sampled well above their share of hours, the largest public families are sampled below it, and families with a few hours of data are seen several times more often than their volume alone would warrant, at the cost of repeating their subtasks.

## B.5 Policy Adaptation Recipes

**RoboTwin 2.0.** The policy is trained under the leaderboard protocol on all 2 500 clean demonstrations (50 per task); a split into 40 training and 10 held-out episodes per task is used only for recipe selection. The

36

<!-- page 37 of 43 -->

Table 8 Configured sampling targets by registered family, computed from the training split, next to admitted hours. Targets are set before text and mode admission.

| Family | Admitted, h | Hours, % | Sampling, % |
| --- | --- | --- | --- |
| Green humanoid | 214.8 | 12.7 | 18.0 |
| Worker / ARX5 | 80.3 | 4.8 | 12.0 |
| AgiBot G2A | 256.0 | 15.2 | 9.1 |
| Piper / Cobot Magic | 227.2 | 13.5 | 7.8 |
| Unitree G1 | 204.9 | 12.1 | 7.2 |
| Fourier GR1T1 | 87.9 | 5.2 | 6.9 |
| AgiBot G1 / Genie-1 | 148.5 | 8.8 | 5.2 |
| Galaxea R1Pro | 82.9 | 4.9 | 4.9 |
| Galaxea R1Lite | 92.6 | 5.5 | 4.3 |
| ARX R5a / Lift-2 | 105.8 | 6.3 | 4.2 |
| TienKung | 59.3 | 3.5 | 4.1 |
| RealMan AIDAL | 57.7 | 3.4 | 4.0 |
| Leju | 30.0 | 1.8 | 3.2 |
| AIRBOT MMK2 | 7.8 | 0.5 | 2.4 |
| Unitree H1 | 12.5 | 0.7 | 2.2 |
| RoboCOIN Ruantong/A2D | 10.1 | 0.6 | 1.7 |
| Tianyi | 6.8 | 0.4 | 1.4 |
| AlphaBot2 | 2.2 | 0.1 | 0.7 |
| Fourier GR1T2 | 0.8 | &lt; 0.1 | 0.4 |
| ManipArena EX001 | 0.4 | &lt; 0.1 | 0.2 |
| OpenArm | &lt; 0.01 | &lt; 0.01 | 0.04 |

policy commands both arms and grippers in joint space. RoboTwin records a frame every 15 planner steps of $1 / 2 5 0   \mathrm { s } ,$ so the physical frame rate is 16.67 Hz rather than the 30 Hz in its metadata, and the WAM horizon keeps the 1.07 s of GreenWAM. Episodes are cut into equal contiguous atoms of at most 10 s, and WAM windows start at every frame where a full window fits, which gives about 470 thousand windows instead of 15 thousand tiled ones. The mode mixture is WAM-heavy (0.9 WAM, 0.05 forward, 0.05 image-to-video), each of the 14 action coordinates carries the same loss weight, and each sample draws a random paraphrase from the seen instructions of its episode. The reported policy is trained in stages from GreenWAM with background replacement, color augmentation and the action expert. At evaluation the policy predicts 16 steps, samples them with 8 denoising steps without guidance and executes the whole prediction before replanning.

**WorldArena 1.0.** The world model starts from GreenWAM and is adapted for 10 000 steps with a global batch of 64, a peak learning rate of $1 . 4 1 \times 1 0 ^ { - 5 }$ after 500 warm-up steps and a cosine decay to 0.1 of the peak. It is trained on the clean RoboTwin 2.0 demonstrations, whose ARX5 arms are those of the benchmark, rendered as a single head-camera canvas of 736×544 and cut into blocks of at most 15 s. Only forward dynamics with history is trained, and the text prompt is always empty, as in the action variant of the benchmark. At evaluation every block, including the first, is generated in this mode from four anchors of the history, which for the first block repeat the recorded first frame; sampling uses 35 steps without guidance.

**LIBERO.** The LIBERO policy is adapted from GreenWAM with the action expert in stages, with appearance augmentation, a random perspective distortion of the third-person camera image and paraphrased instructions. No LIBERO-plus episode is used in training, and the same policy is evaluated on both benchmarks.

**RoboDojo.** The policy follows the official protocol: it is trained on the official training set and compared with the entries of the official leaderboard (Section C.1). It uses the generator–expert architecture of GreenWAM with attention restricted by stream: the history attends to the instruction and to itself, the current frame to the instruction, the history and itself, the state to the current frame and itself, and the actions to the generated video, the state and themselves. The history is pinned after the instruction, and the current frame, the generated video and the actions are placed at their time offsets from it. Training runs for

37

<!-- page 38 of 43 -->

60 000 steps with a global batch of 352, a learning rate warmed up over 3 000 steps to $8   \times   1 0 ^ { - 5 }$ and decayed by a cosine to $1 0 ^ { - 6 }$ , weight decay 0.01, an exponential moving average of the weights and quantile-normalized actions; a fifth of the training examples are image-to-video.

**CALVIN.** The CALVIN policy is trained on the ABC training split only (Section C). Its single Franka arm enters GreenWAM through the Cartesian end-effector slots with per-frame SE(3) offsets and the absolute initial pose as the current state. At evaluation the policy predicts 20 steps, samples them with 4 UniPC steps, a timestep shift of 3 and guidance 1.5, and executes the first 10 before replanning.

**Green humanoid.** The comparison uses three initializations: the public $\pi _ { 0 . 5 }$ base checkpoint [61], Cosmos 3 Edge, and GreenWAM. They are adapted on the same Green demonstration split. The distilled policy starts from the reported GreenWAM teacher and uses only the training split for the consistency and demonstration losses. Its consistency losses use a masked pseudo-Huber distance [72], with unavailable action coordinates and clean conditions excluded; the loss weights are (1, 1, 0.01) and the EMA decay is 0.995. Only the online student is optimized; the teacher and the EMA target are used only during training. Validation episodes select checkpoints and inference settings; neither ID test episodes nor OOD trials are used for selection.

Each policy retains its own observation preprocessing, action representation and training recipe, with a common set of camera streams and controlled joints at the robot interface. Normalization statistics are fitted on training data.

## C Evaluation Protocols

![Image block](doc:f38ca8c/tier:basic/page:38/block:6)

![Image block](doc:f38ca8c/tier:basic/page:38/block:7)

![Image block](doc:f38ca8c/tier:basic/page:38/block:8)

![Image block](doc:f38ca8c/tier:basic/page:38/block:9)

![Image block](doc:f38ca8c/tier:basic/page:38/block:10)

![Image block](doc:f38ca8c/tier:basic/page:38/block:11)

shifted camera viewpoint

![Image block](doc:f38ca8c/tier:basic/page:38/block:13)

![Image block](doc:f38ca8c/tier:basic/page:38/block:14)

changed table texture

Figure 30 Benchmark scenes. One frame per benchmark and setting. The RoboTwin 2.0, LIBERO and LIBERO plus frames come from evaluation episodes of our policies: the first head-camera frames of the clean and randomized episodes of Figure 12, a standard LIBERO scene, a LIBERO-plus scene with the camera turned by −32∘ and one with a changed table texture. The other frames are recorded benchmark data: the static camera of an environment-D episode of CALVIN, the head camera of a RoboDojo demonstration and a recorded frame of a WorldArena 1.0 test episode.

## C.1 Public Benchmarks

**RoboTwin 2.0.** The benchmark comprises 50 bimanual tasks on the Aloha-AgileX embodiment, whose arms share the ARX5 kinematics (Section A.3). The leaderboard protocol trains on 50 clean demonstrations per task (2 500 in total) and evaluates 100 trials per task in clean scenes (clean2clean, easy) and in randomized scenes with clutter, random lighting, background textures and table height (clean2random, hard); the leaderboard ranks by the average of the two. A leaderboard entry requires public code, public weights and a technical report [68].

38

<!-- page 39 of 43 -->

Table 9 Evaluation protocols. Figure 30 shows one frame of each benchmark.

| Benchmark | Protocol | Metric |
| --- | --- | --- |
| WorldArena 1.0, track 1 [70] | Action-conditioned prediction on 1 000 test episodes | Perceptual and embodiment metrics; motion-capped aggregate in Figure 9 |
| RoboTwin 2.0 [18] | 50 bimanual tasks, 100 trials per task, clean and randomized scenes | Success rate, clean and randomized |
| LIBERO [48] | Four task suites of a single Franka arm in simulation | Average success over the suites |
| LIBERO-plus [24] | LIBERO tasks under seven perturbation axes, zero-shot after standard LIBERO training | Success over all perturbed episodes |
| RoboDojo [19] | 42 simulation tasks in five capability dimensions, 150 trials per task over 3 seeds | Score and success rate averaged over dimensions |
| CALVIN [55] | ABC→D, chains of five instructions | Average completed length |
| Green humanoid | Shared ID/OOD trial set for π0.5, Cosmos 3 Edge, GreenWAM and its distilled student | Instruction success by split |

**WorldArena.** Evaluation uses track 1 of WorldArena 1.0. The metric breakdown includes image quality, aesthetic quality, JEPA similarity, dynamic degree, flow score, motion smoothness, subject consistency, background consistency, photometric consistency, interaction quality, trajectory accuracy, depth accuracy, perspectivity, instruction following and semantic alignment [70]. GreenWAM is evaluated on all 1 000 test episodes. For GreenWAM, dynamic degree, flow score and motion smoothness are capped separately against each episode’s ground truth before averaging. For competitors, only model-level metric means are available, so the caps are the corresponding ground-truth means 23.05, 12.50 and 69.79 over all 1 000 reference episodes. This follows the motion-capping rule of WorldArena 2.0 [71, 83]; the videos and test episodes remain those of WorldArena 1.0. The recomputed aggregate therefore differs from the official WorldArena 1.0 leaderboard score. The figure shows a selected comparison: BLM, WorldScape v0.2, GigaWorld-1, GE-Sim 2.0 and SisyphusWorld are omitted. Visual quality combines two scores computed without reference to the recording, MUSIQ image quality [37] and LAION aesthetic quality [69], with JEPA similarity; GreenWAM reaches 57.2 against 59.5 for the recordings, while 103 of the 127 board entries score above the recordings in image quality. Instruction following and interaction quality are assigned by a vision–language judge, Qwen3-VL in the WorldArena evaluator [70]. Across the judged GreenWAM videos, both correlate weakly with trajectory accuracy (𝑟 = 0.05) and more strongly with aesthetic quality (𝑟 = 0.29 and 0.32), and 41 and 67 board entries score above the recordings on them.

**RoboDojo.** The benchmark comprises 42 simulation tasks grouped into five capability dimensions (generalization, memory, precision, long-horizon, open-vocabulary instructions) and 18 real-world tasks. Each policy is evaluated with 150 trials per task over three seeds; score and success rate are averaged within dimensions and then across dimensions [19]. The simulated robot is a bimanual ARX5 at 25 Hz with a head and two wrist cameras; training uses 3 500 demonstrations, 100 for each of 34 tasks and 100 auxiliary domain-randomized ones, and the open-vocabulary tasks are not in training. A verified leaderboard entry reports three seeds on the public layouts, passes a hidden-layout consistency check and releases the evaluated checkpoint and code [67].

**LIBERO and LIBERO-plus.** LIBERO provides suites of simulated tasks for a single Franka arm [48]; following common practice [38], evaluation uses LIBERO-Spatial, LIBERO-Object, LIBERO-Goal and LIBERO-Long, the long-horizon subset of LIBERO-100, and reports the success rate averaged over the four suites. LIBERO-plus applies perturbations of camera viewpoint, robot initial state, language, lighting, background, sensor noise and object layout to the LIBERO tasks, with one trial per perturbed episode; in the zero-shot setting the policy is trained on standard LIBERO only [24].

**CALVIN.** Policies train on environments A, B and C and are evaluated in the unseen environment D on chains of five language instructions; the metric is the average number of completed instructions [55]. Training

39

<!-- page 40 of 43 -->

uses the ABC training split only: no episode of environment D, including its validation split, is used for training, and results are reported on 1 000 chains.

## C.2 Real-Humanoid Comparison Protocol

**Tasks and splits.** The packing task is given as five consecutive instructions without resetting the scene (Section 6.5). ID trials use new resets of the objects and scenes represented in policy adaptation; OOD trials change objects or scene conditions excluded from adaptation, validation and distillation data. Every policy is evaluated on the same 14 scene configurations, 9 ID and 5 OOD, each with the five instructions of the task, which gives 45 ID and 25 OOD instructions per policy, each attempted once. All policies receive the same instructions, cameras and controlled joints.

**Success.** An instruction succeeds when it is completed before its timeout without human assistance; failures, timeouts and stops caused by the policy count as failed instructions. For split 𝑑 and policy 𝑝, let $s _ { p d }$ be the number of successful instructions and $n _ { p d }$ the number of evaluated instructions. The split score is

$$
\mathrm { S R } _ { p , d } = 1 0 0   \frac { s _ { p d } } { n _ { p d } } ,\tag{14}
$$

reported separately for ID and OOD.

## C.3 Latency Benchmark

We benchmark inference latency on a single NVIDIA RTX 4090 with FlashAttention-2 (Table 10). The video branch works on three latent frames: one conditioning frame and two generated frames, which decode into an eight-frame prediction at 10 Hz. Each request also conditions on the proprioceptive state and a 128-token instruction. To isolate the cost of the model, the observation and the state stay fixed within an episode, and the instruction changes once, at its start. Median latency is the steady-state latency after the instruction-dependent quantities have been cached; time to first action (TTA) is the latency of the first request after a new instruction and includes building that cache. Each value is computed over 100 episodes, and the rate is 1000 divided by the latency in milliseconds.

Table 10 Policy inference latency on one NVIDIA RTX 4090. Optimizations are applied cumulatively to the GreenWAM policy, and speedups are relative to it. Median latency over 100 episodes; TTA is the time to the first action after a new instruction.

| Configuration | Median, ms | Rate, Hz | TTA, ms | Speedup |
| --- | --- | --- | --- | --- |
| GreenWAM | 435.9 | 2.3 | 435.9 | 1.00× |
| + caching | 411.1 | 2.4 | 430.8 | 1.06× |
| + guidance packing | 314.1 | 3.2 | 334.0 | 1.39× |
| + first-frame VAE | 236.4 | 4.2 | 256.5 | 1.84× |
| + compilation | 184.6 | 5.4 | 195.5 | 2.36× |
| + W8A8 | 134.6 | 7.4 | 148.6 | 3.24× |
| + distillation | 35.4 | 28.3 | 48.8 | 12.32× |

The optimizations are applied in turn. Caching reuses the instruction and vision key–value states across denoising steps, so later steps run only the generative backbone. Guidance packing evaluates the conditional and empty-prompt branches of classifier-free guidance in one forward pass and also caches the empty-prompt key–value states. First-frame VAE encoding passes only the conditioning frame through the VAE encoder, since the other two latent frames are generated by the policy. Compilation traces the generator blocks together with this single-frame encoder, and W8A8 runs the feed-forward layers with 8-bit weights and activations. Distillation replaces the four guided denoising steps with one consistency step and keeps the instruction cache. Once caching is enabled, a new instruction adds 11–20 ms to the first request, and 13.4 ms for the distilled policy.

40

<!-- page 41 of 43 -->

## C.4 Paired Comparisons

Let $y _ { t e } ^ { A } , y _ { t e } ^ { B } \in \{ 0 , 1 \}$ be the outcomes of the two runs on episode 𝑒 of task 𝑡, over the common seeds $E _ { t }$ of the two runs. The reported difference averages over tasks, as the benchmark does:

$$
\Delta = \frac { 1 } { | \mathcal { T } | } \sum _ { t \in \mathcal { T } } \frac { 1 } { | E _ { t } | } \sum _ { e \in E _ { t } } \big ( y _ { t e } ^ { A } - y _ { t e } ^ { B } \big ) .\tag{15}
$$

Its 95% interval comes from a hierarchical bootstrap with 5 000 replicates: tasks are resampled with replacement, then episode pairs within each resampled task, and the interval spans the 2.5th to 97.5th percentiles. The interval therefore describes the uncertainty of evaluating the two checkpoints, not the variation between training runs. Forward prediction uses the same estimator with PSNR per clip in place of 𝑦.

## C.5 Evaluation Details

With an empty text prompt, as in the action variant of WorldArena, classifier-free guidance has no effect, and that variant is sampled without it. CALVIN test instructions are excluded from the language-augmentation pools. The reported RoboTwin result uses 100 trials per task and setting, and the controlled comparisons pair episodes on common seeds (Equation (15)).

## D Additional Results and Qualitative Examples

## D.1 Benchmark Tables

Table 11 gives the WorldArena categories behind Figure 9, and Tables 12 to 14 list the CALVIN, RoboDojo, LIBERO and LIBERO-plus results summarized in Figure 11, with the source of every value.

Table 11 WorldArena 1.0, track 1: overall score, categories and fidelity of the models in Figure 9. EWMScore (Equation (7)) is the mean of the 15 board metrics with dynamic degree, flow score and motion smoothness capped at the ground truth: per video for GreenWAM, at the leaderboard means for the other models (an upper bound for them). The last row scores the recorded test videos themselves with the same evaluator.

<table><tr><td></td><td></td><td colspan="5">Categories</td><td colspan="3">Fidelity to ground truth</td></tr><tr><td>Model</td><td>EWMScore</td><td>Visual</td><td>Content</td><td>Physics</td><td>3D</td><td>Control</td><td>JEPA</td><td>Traj.</td><td>Depth</td></tr><tr><td>GreenWAM</td><td>63.31</td><td>57.2</td><td>65.6</td><td>62.2</td><td>93.4</td><td>84.1</td><td>98.4</td><td>56.0</td><td>98.2</td></tr><tr><td>BWM-Fast [12]</td><td>63.80</td><td>63.1</td><td>58.2</td><td>62.4</td><td>91.8</td><td>89.7</td><td>97.9</td><td>44.9</td><td>86.4</td></tr><tr><td>UNIS [46]</td><td>63.65</td><td>61.8</td><td>55.9</td><td>64.6</td><td>92.0</td><td>91.6</td><td>90.6</td><td>41.9</td><td>85.2</td></tr><tr><td>ABot-PhysWorld (text) [20]</td><td>63.65</td><td>64.4</td><td>63.4</td><td>56.7</td><td>85.5</td><td>90.8</td><td>90.4</td><td>31.5</td><td>72.0</td></tr><tr><td>Motubrain [56]</td><td>63.16</td><td>64.0</td><td>59.5</td><td>58.5</td><td>91.5</td><td>85.8</td><td>96.9</td><td>43.1</td><td>88.7</td></tr><tr><td>FlowWAM-FiveAges [21]</td><td>62.99</td><td>63.5</td><td>57.3</td><td>60.5</td><td>90.9</td><td>87.0</td><td>95.3</td><td>44.4</td><td>85.6</td></tr><tr><td>CogVideoX [91]</td><td>61.30</td><td>55.8</td><td>67.7</td><td>50.9</td><td>88.3</td><td>82.6</td><td>94.8</td><td>34.8</td><td>91.1</td></tr><tr><td>Wan2.6 [79]</td><td>60.96</td><td>61.4</td><td>64.0</td><td>42.7</td><td>84.7</td><td>89.0</td><td>72.6</td><td>12.2</td><td>75.4</td></tr><tr><td>Ctrl-World [31]</td><td>60.56</td><td>57.4</td><td>62.2</td><td>55.4</td><td>88.5</td><td>78.2</td><td>92.8</td><td>48.2</td><td>93.2</td></tr><tr><td>IRASim [105]</td><td>59.63</td><td>56.1</td><td>65.7</td><td>49.3</td><td>88.8</td><td>77.4</td><td>93.7</td><td>35.9</td><td>93.5</td></tr><tr><td>Cosmos-Predict 2.5 [57]</td><td>57.83</td><td>58.1</td><td>61.7</td><td>44.0</td><td>87.5</td><td>75.2</td><td>90.7</td><td>27.5</td><td>90.3</td></tr><tr><td>Recorded test videos</td><td>67.23</td><td>59.5</td><td>66.7</td><td>79.6</td><td>95.4</td><td>87.2</td><td>99.9</td><td>86.9</td><td>100.0</td></tr></table>

## D.2 Failure Cases

Figure 31 shows typical failures of the LIBERO and RoboTwin policies. In both episodes the task is carried out up to its last phase: the object is grasped, carried and brought to its target. The error lies in the final placement, a cream cheese tilted on the rim of the bowl and a stapler released only partly onto the pad, and the policy does not correct it before the step limit. The errors lie in precise placement and recovery.

41

<!-- page 42 of 43 -->

Table 12 CALVIN ABC→D, 1 000 chains. Average number of completed instructions in a chain of five. Values are taken from Xiaomi-Robotics-0 [15] (Table 2), except DreamVLA [98] (Table 1), MMaDA-VLA [51] (Table 1) and $\pi _ { 0 . 5 } ,$ the supervised fine-tuned checkpoint reported in $\pi _ { \mathrm { R L } }$ [17] (Table 5).

| Method | Avg. length |
| --- | --- |
| Published methods |  |
| 𝜋<sub>0.5</sub> (SFT) [62] | 3.84 |
| RoboVLMs [45] | 4.25 |
| Seer-Large [76] | 4.28 |
| VPP [33] | 4.29 |
| UniVLA [81] | 4.41 |
| DreamVLA [98] | 4.44 |
| FLOWER [65] | 4.53 |
| Xiaomi-Robotics-0 [15] | 4.75 |
| MMaDA-VLA [51] | 4.78 |
| GreenWAM | 4.64 |

Table 13 RoboDojo simulation benchmark. Score and success rate averaged over the five capability dimensions, from the official leaderboard [67]: the top five entries and the methods shown in Figure 11. Human teleoperation reaches a score of 80.42 and a success rate of 76.03% [19].

| Method | Rank | Score | Success rate, % |
| --- | --- | --- | --- |
| Leaderboard entries |  |  |  |
| PhysicalRSI | 1 | 36.27 | 31.38 |
| Awomo-0.5 | 2 | 35.34 | 29.64 |
| Simate-beta | 3 | 33.95 | 27.96 |
| Rex-M1 Preview | 4 | 33.79 | 27.44 |
| VPP2-Preview | 5 | 31.40 | 25.62 |
| DM0.5 | 10 | 24.90 | 19.34 |
| Xiaomi-Robotics-1 | 13 | 20.07 | 13.93 |
| 𝜋<sub>0.5</sub> [62] | 20 | 11.44 | 6.93 |
| X-VLA [103] | 24 | 10.13 | 6.52 |
| X-WAM [30] | 28 | 7.69 | 3.83 |
| Our adapted policy |  |  |  |
| GreenWAM |  | 16.12 | 10.49 |

Table 14 LIBERO and LIBERO-plus. Average success over the four LIBERO task suites and zero-shot LIBERO-plus success over all perturbed episodes, %.

| Method | LIBERO | LIBERO-plus, zero-shot |
| --- | --- | --- |
| Faster-WAM [100] | 99.0 | 73.6 |
| CKT-WAM [36] | 98.8 | 86.1 |
| Cosmos Policy [39] | 98.5 | 82.2 |
| OA-WAM [52] | 97.8 | 83.9 |
| Fast-WAM [95] | 97.6 | 49.1 |
| OpenVLA-OFT [38] | 97.1 | 69.6 |
| 𝜋<sub>0.5</sub> [62] | 96.9 | 85.7 |
| GreenWAM | 97.4 | 86.8 |

GreenWAM: the same policy for both benchmarks (recipe in Section B.5); LIBERO-plus over all 10 030 perturbed episodes, one trial each. LIBERO: Cosmos Policy [39], Table 1 $( \pi _ { 0 . 5 } ,$ OpenVLA-OFT, Cosmos Policy), OA-WAM [52], Table 1, the LIBERO table of Faster-WAM [100] (Fast-WAM, Faster-WAM) and CKT-WAM, Table 9. LIBERO-plus: OA-WAM, Table 2, zero-shot after standard LIBERO training $( \pi _ { 0 . 5 } ,$ Cosmos Policy, OpenVLA-OFT, OA-WAM), the LIBERO-plus table of Faster-WAM (Fast-WAM, Faster-WAM) and CKT-WAM [36], Table 1. All LIBERO-plus values are success rates over all perturbed episodes, as in the benchmark.

42

<!-- page 43 of 43 -->

## ,PSUHFLVHìQDOSKDVHZLWKRXWUHFRYHU\

LIBERO-Goal · “put the cream cheese in the bowl”

![Image block](doc:f38ca8c/tier:basic/page:43/block:3)

start

![Image block](doc:f38ca8c/tier:basic/page:43/block:5)

grasp

![Image block](doc:f38ca8c/tier:basic/page:43/block:7)

over the bowl

![Image block](doc:f38ca8c/tier:basic/page:43/block:9)

released tilted

![Image block](doc:f38ca8c/tier:basic/page:43/block:11)

moves away

![Image block](doc:f38ca8c/tier:basic/page:43/block:13)

step limit

RoboTwin 2.0 · “move stapler pad”

![Image block](doc:f38ca8c/tier:basic/page:43/block:16)

start

![Image block](doc:f38ca8c/tier:basic/page:43/block:18)

grasp

![Image block](doc:f38ca8c/tier:basic/page:43/block:20)

carry

![Image block](doc:f38ca8c/tier:basic/page:43/block:22)

partly on the pad

![Image block](doc:f38ca8c/tier:basic/page:43/block:24)

held, no correction

![Image block](doc:f38ca8c/tier:basic/page:43/block:26)

step limit

**Figure 31 Failure cases of the LIBERO and RoboTwin policies** (Tables 3 and 14). Imprecise final phase without recovery. LIBERO-Goal, “put the cream cheese in the bowl”: the cheese reaches the bowl but rests tilted on its rim, which the success check does not accept; the policy then moves toward the plate and does not return before the limit of 310 steps. RoboTwin 2.0, move\_stapler\_pad in a clean scene: the stapler is carried to the green pad but released only partly onto it, and the gripper holds it without correction until the limit of 400 control steps.

43